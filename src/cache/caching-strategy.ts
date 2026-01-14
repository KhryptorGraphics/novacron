/**
 * NovaCron Caching Strategy Implementation
 * Multi-tier caching with Redis L1 and Memcached L2
 */

import Redis from 'ioredis';
import Memcached from 'memcached';
import { createHash } from 'crypto';
import { EventEmitter } from 'events';

// ============================================================================
// INTERFACES AND TYPES
// ============================================================================

interface CacheConfig {
  redis: RedisConfig;
  memcached: MemcachedConfig;
  strategies: CacheStrategies;
}

interface RedisConfig {
  cluster: {
    nodes: string[];
    options: {
      redisOptions: {
        password?: string;
        db?: number;
      };
      maxRetriesPerRequest: number;
      enableReadyCheck: boolean;
      clusterRetryStrategy: (times: number) => number;
    };
  };
  keyPrefix: string;
}

interface MemcachedConfig {
  servers: string[];
  options: {
    poolSize: number;
    timeout: number;
    retries: number;
    retry: number;
    remove: boolean;
  };
}

interface CacheStrategies {
  defaultTTL: number;
  compressionThreshold: number;
  maxKeyLength: number;
  namespaces: {
    [key: string]: {
      ttl: number;
      priority: 'high' | 'medium' | 'low';
      compress: boolean;
    };
  };
}

interface CacheEntry<T = any> {
  data: T;
  metadata: {
    created: number;
    accessed: number;
    hits: number;
    size: number;
    compressed: boolean;
  };
}

// ============================================================================
// CACHE MANAGER
// ============================================================================

export class CacheManager extends EventEmitter {
  private redisClient: Redis.Cluster;
  private memcachedClient: any;
  private config: CacheConfig;
  private stats: CacheStatistics;

  constructor(config: CacheConfig) {
    super();
    this.config = config;
    this.stats = new CacheStatistics();
    this.initializeClients();
  }

  private initializeClients(): void {
    // Initialize Redis Cluster
    this.redisClient = new Redis.Cluster(
      this.config.redis.cluster.nodes.map(node => {
        const [host, port] = node.split(':');
        return { host, port: parseInt(port) };
      }),
      this.config.redis.cluster.options
    );

    // Initialize Memcached
    this.memcachedClient = new Memcached(
      this.config.memcached.servers,
      this.config.memcached.options
    );

    // Set up event handlers
    this.setupEventHandlers();
  }

  private setupEventHandlers(): void {
    this.redisClient.on('error', (err) => {
      this.emit('error', { layer: 'L1', error: err });
    });

    this.redisClient.on('connect', () => {
      this.emit('connected', { layer: 'L1' });
    });
  }

  // ============================================================================
  // CORE CACHE OPERATIONS
  // ============================================================================

  /**
   * Get value from cache with multi-tier fallback
   */
  async get<T>(key: string, options?: GetOptions): Promise<T | null> {
    const startTime = Date.now();
    const hashedKey = this.hashKey(key);

    try {
      // Try L1 Cache (Redis)
      let value = await this.getFromRedis<T>(hashedKey);
      if (value !== null) {
        this.stats.recordHit('L1', Date.now() - startTime);
        this.emit('cache:hit', { layer: 'L1', key });
        return value;
      }

      // Try L2 Cache (Memcached)
      value = await this.getFromMemcached<T>(hashedKey);
      if (value !== null) {
        this.stats.recordHit('L2', Date.now() - startTime);
        this.emit('cache:hit', { layer: 'L2', key });
        
        // Promote to L1 if high-priority
        if (options?.promoteToL1) {
          await this.setInRedis(hashedKey, value, options.ttl);
        }
        
        return value;
      }

      // Cache miss
      this.stats.recordMiss(Date.now() - startTime);
      this.emit('cache:miss', { key });
      
      // Execute loader function if provided
      if (options?.loader) {
        const loadedValue = await options.loader();
        if (loadedValue !== null) {
          await this.set(key, loadedValue, options);
        }
        return loadedValue;
      }

      return null;
    } catch (error) {
      this.emit('error', { operation: 'get', key, error });
      return null;
    }
  }

  /**
   * Set value in cache with intelligent tier placement
   */
  async set<T>(
    key: string,
    value: T,
    options?: SetOptions
  ): Promise<boolean> {
    const hashedKey = this.hashKey(key);
    const namespace = this.getNamespace(key);
    const config = this.config.strategies.namespaces[namespace] || {
      ttl: this.config.strategies.defaultTTL,
      priority: 'medium',
      compress: false,
    };

    const ttl = options?.ttl || config.ttl;
    const priority = options?.priority || config.priority;

    try {
      // Serialize and optionally compress
      const serialized = this.serialize(value, config.compress);
      
      // Store in appropriate tier based on priority
      if (priority === 'high') {
        // Store in both L1 and L2
        await Promise.all([
          this.setInRedis(hashedKey, serialized, ttl),
          this.setInMemcached(hashedKey, serialized, ttl),
        ]);
      } else if (priority === 'medium') {
        // Store in L1 only
        await this.setInRedis(hashedKey, serialized, ttl);
      } else {
        // Store in L2 only
        await this.setInMemcached(hashedKey, serialized, ttl);
      }

      this.emit('cache:set', { key, priority, ttl });
      return true;
    } catch (error) {
      this.emit('error', { operation: 'set', key, error });
      return false;
    }
  }

  /**
   * Delete value from all cache tiers
   */
  async delete(key: string): Promise<boolean> {
    const hashedKey = this.hashKey(key);

    try {
      await Promise.all([
        this.deleteFromRedis(hashedKey),
        this.deleteFromMemcached(hashedKey),
      ]);

      this.emit('cache:delete', { key });
      return true;
    } catch (error) {
      this.emit('error', { operation: 'delete', key, error });
      return false;
    }
  }

  /**
   * Invalidate cache entries by pattern
   */
  async invalidatePattern(pattern: string): Promise<number> {
    let count = 0;

    try {
      // Redis pattern deletion
      const stream = this.redisClient.scanStream({
        match: `${this.config.redis.keyPrefix}:${pattern}`,
        count: 100,
      });

      const keys: string[] = [];
      stream.on('data', (resultKeys) => {
        keys.push(...resultKeys);
      });

      await new Promise((resolve, reject) => {
        stream.on('end', resolve);
        stream.on('error', reject);
      });

      if (keys.length > 0) {
        const pipeline = this.redisClient.pipeline();
        keys.forEach((key) => pipeline.del(key));
        await pipeline.exec();
        count = keys.length;
      }

      this.emit('cache:invalidate', { pattern, count });
      return count;
    } catch (error) {
      this.emit('error', { operation: 'invalidatePattern', pattern, error });
      return 0;
    }
  }

  // ============================================================================
  // TIER-SPECIFIC OPERATIONS
  // ============================================================================

  private async getFromRedis<T>(key: string): Promise<T | null> {
    const value = await this.redisClient.get(key);
    return value ? this.deserialize<T>(value) : null;
  }

  private async setInRedis(
    key: string,
    value: any,
    ttl: number
  ): Promise<void> {
    await this.redisClient.setex(key, ttl, value);
  }

  private async deleteFromRedis(key: string): Promise<void> {
    await this.redisClient.del(key);
  }

  private async getFromMemcached<T>(key: string): Promise<T | null> {
    return new Promise((resolve) => {
      this.memcachedClient.get(key, (err: any, data: any) => {
        if (err || !data) {
          resolve(null);
        } else {
          resolve(this.deserialize<T>(data));
        }
      });
    });
  }

  private async setInMemcached(
    key: string,
    value: any,
    ttl: number
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      this.memcachedClient.set(key, value, ttl, (err: any) => {
        if (err) reject(err);
        else resolve();
      });
    });
  }

  private async deleteFromMemcached(key: string): Promise<void> {
    return new Promise((resolve) => {
      this.memcachedClient.del(key, () => resolve());
    });
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  private hashKey(key: string): string {
    if (key.length <= this.config.strategies.maxKeyLength) {
      return `${this.config.redis.keyPrefix}:${key}`;
    }
    
    const hash = createHash('sha256').update(key).digest('hex');
    return `${this.config.redis.keyPrefix}:${hash}`;
  }

  private getNamespace(key: string): string {
    const parts = key.split(':');
    return parts[0] || 'default';
  }

  private serialize(value: any, compress: boolean): string {
    const json = JSON.stringify(value);
    
    if (compress && json.length > this.config.strategies.compressionThreshold) {
      // Implement compression (e.g., using zlib)
      return json; // TODO: Add compression
    }
    
    return json;
  }

  private deserialize<T>(value: string): T {
    try {
      return JSON.parse(value);
    } catch {
      return value as any;
    }
  }

  // ============================================================================
  // CACHE WARMING
  // ============================================================================

  /**
   * Warm cache with frequently accessed data
   */
  async warmCache(items: WarmCacheItem[]): Promise<void> {
    const promises = items.map(async (item) => {
      const value = await item.loader();
      if (value !== null) {
        await this.set(item.key, value, {
          ttl: item.ttl,
          priority: item.priority || 'high',
        });
      }
    });

    await Promise.all(promises);
    this.emit('cache:warmed', { count: items.length });
  }

  // ============================================================================
  // MONITORING AND STATISTICS
  // ============================================================================

  getStatistics(): CacheStats {
    return this.stats.getStats();
  }

  async getMemoryUsage(): Promise<MemoryUsage> {
    const redisInfo = await this.redisClient.info('memory');
    const memoryUsage: MemoryUsage = {
      redis: this.parseRedisMemory(redisInfo),
      memcached: await this.getMemcachedMemory(),
      total: 0,
    };
    
    memoryUsage.total = memoryUsage.redis + memoryUsage.memcached;
    return memoryUsage;
  }

  private parseRedisMemory(info: string): number {
    const match = info.match(/used_memory:(\d+)/);
    return match ? parseInt(match[1]) : 0;
  }

  private async getMemcachedMemory(): Promise<number> {
    return new Promise((resolve) => {
      this.memcachedClient.stats((err: any, stats: any) => {
        if (err) {
          resolve(0);
        } else {
          const totalBytes = Object.values(stats).reduce(
            (sum: number, serverStats: any) => sum + (serverStats.bytes || 0),
            0
          );
          resolve(totalBytes as number);
        }
      });
    });
  }
}

// ============================================================================
// CACHE STATISTICS
// ============================================================================

class CacheStatistics {
  private stats = {
    hits: { L1: 0, L2: 0 },
    misses: 0,
    sets: 0,
    deletes: 0,
    errors: 0,
    avgLatency: { L1: 0, L2: 0, miss: 0 },
    totalLatency: { L1: 0, L2: 0, miss: 0 },
  };

  recordHit(layer: 'L1' | 'L2', latency: number): void {
    this.stats.hits[layer]++;
    this.stats.totalLatency[layer] += latency;
    this.stats.avgLatency[layer] = 
      this.stats.totalLatency[layer] / this.stats.hits[layer];
  }

  recordMiss(latency: number): void {
    this.stats.misses++;
    this.stats.totalLatency.miss += latency;
    this.stats.avgLatency.miss = 
      this.stats.totalLatency.miss / this.stats.misses;
  }

  getStats(): CacheStats {
    const totalHits = this.stats.hits.L1 + this.stats.hits.L2;
    const totalRequests = totalHits + this.stats.misses;
    
    return {
      ...this.stats,
      hitRate: totalRequests > 0 ? (totalHits / totalRequests) * 100 : 0,
      l1HitRate: totalHits > 0 ? (this.stats.hits.L1 / totalHits) * 100 : 0,
    };
  }
}

// ============================================================================
// CACHE INVALIDATION STRATEGIES
// ============================================================================

export class CacheInvalidator {
  private cacheManager: CacheManager;
  private invalidationRules: InvalidationRule[];

  constructor(cacheManager: CacheManager) {
    this.cacheManager = cacheManager;
    this.invalidationRules = [];
  }

  /**
   * Register invalidation rule
   */
  addRule(rule: InvalidationRule): void {
    this.invalidationRules.push(rule);
  }

  /**
   * Invalidate caches based on entity change
   */
  async invalidateEntity(
    entityType: string,
    entityId: string,
    operation: 'create' | 'update' | 'delete'
  ): Promise<void> {
    const rules = this.invalidationRules.filter(
      (rule) => rule.entityType === entityType && rule.operations.includes(operation)
    );

    for (const rule of rules) {
      const patterns = rule.getPatterns(entityId);
      for (const pattern of patterns) {
        await this.cacheManager.invalidatePattern(pattern);
      }
    }
  }

  /**
   * Cascade invalidation for related entities
   */
  async cascadeInvalidation(
    entityType: string,
    entityId: string,
    relations: string[]
  ): Promise<void> {
    // Invalidate primary entity
    await this.invalidateEntity(entityType, entityId, 'update');

    // Invalidate related entities
    for (const relation of relations) {
      await this.cacheManager.invalidatePattern(`${relation}:*:${entityId}`);
    }
  }
}

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

interface GetOptions {
  loader?: () => Promise<any>;
  promoteToL1?: boolean;
  ttl?: number;
}

interface SetOptions {
  ttl?: number;
  priority?: 'high' | 'medium' | 'low';
  tags?: string[];
}

interface WarmCacheItem {
  key: string;
  loader: () => Promise<any>;
  ttl?: number;
  priority?: 'high' | 'medium' | 'low';
}

interface CacheStats {
  hits: { L1: number; L2: number };
  misses: number;
  sets: number;
  deletes: number;
  errors: number;
  avgLatency: { L1: number; L2: number; miss: number };
  hitRate: number;
  l1HitRate: number;
}

interface MemoryUsage {
  redis: number;
  memcached: number;
  total: number;
}

interface InvalidationRule {
  entityType: string;
  operations: Array<'create' | 'update' | 'delete'>;
  getPatterns: (entityId: string) => string[];
}

// ============================================================================
// DEFAULT CONFIGURATION
// ============================================================================

export const defaultCacheConfig: CacheConfig = {
  redis: {
    cluster: {
      nodes: [
        'redis-1:6379',
        'redis-2:6379',
        'redis-3:6379',
      ],
      options: {
        redisOptions: {
          password: process.env.REDIS_PASSWORD,
          db: 0,
        },
        maxRetriesPerRequest: 3,
        enableReadyCheck: true,
        clusterRetryStrategy: (times) => Math.min(100 * times, 3000),
      },
    },
    keyPrefix: 'novacron',
  },
  memcached: {
    servers: [
      'memcached-1:11211',
      'memcached-2:11211',
      'memcached-3:11211',
    ],
    options: {
      poolSize: 10,
      timeout: 5000,
      retries: 3,
      retry: 3000,
      remove: true,
    },
  },
  strategies: {
    defaultTTL: 300, // 5 minutes
    compressionThreshold: 1024, // 1KB
    maxKeyLength: 250,
    namespaces: {
      vm: {
        ttl: 300,
        priority: 'high',
        compress: true,
      },
      user: {
        ttl: 900,
        priority: 'high',
        compress: false,
      },
      metrics: {
        ttl: 60,
        priority: 'medium',
        compress: true,
      },
      session: {
        ttl: 28800,
        priority: 'high',
        compress: false,
      },
      query: {
        ttl: 600,
        priority: 'low',
        compress: true,
      },
    },
  },
};