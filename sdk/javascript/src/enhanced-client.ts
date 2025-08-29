/**
 * Enhanced NovaCron TypeScript SDK Client with multi-cloud, AI integration, and advanced features
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import WebSocket from 'ws';
import EventEmitter from 'events';
import Redis from 'ioredis';

// Types and Interfaces
export enum CloudProvider {
  LOCAL = 'local',
  AWS = 'aws',
  AZURE = 'azure',
  GCP = 'gcp',
  OPENSTACK = 'openstack',
  VMWARE = 'vmware',
}

export enum AIFeature {
  INTELLIGENT_PLACEMENT = 'intelligent_placement',
  PREDICTIVE_SCALING = 'predictive_scaling',
  ANOMALY_DETECTION = 'anomaly_detection',
  COST_OPTIMIZATION = 'cost_optimization',
}

export interface VMSpec {
  name: string;
  cpu_shares: number;
  memory_mb: number;
  disk_size_gb: number;
  command?: string;
  args?: string[];
  tags?: Record<string, string>;
}

export interface PlacementRecommendation {
  recommended_node: string;
  confidence_score: number;
  reasoning: string;
  alternative_nodes: Array<{
    node_id: string;
    score: number;
    pros: string[];
    cons: string[];
  }>;
}

export interface MigrationSpec {
  vm_id: string;
  target_node_id: string;
  type?: string;
  force?: boolean;
  bandwidth_limit?: number;
  compression?: boolean;
}

export interface ClientConfig {
  baseURL: string;
  apiToken?: string;
  username?: string;
  password?: string;
  timeout?: number;
  maxRetries?: number;
  retryDelay?: number;
  redisUrl?: string;
  cacheTTL?: number;
  enableAIFeatures?: boolean;
  cloudProvider?: CloudProvider;
  region?: string;
  circuitBreakerThreshold?: number;
  circuitBreakerTimeout?: number;
  enableMetrics?: boolean;
}

export interface RequestMetrics {
  count: number;
  avgDuration: number;
  minDuration: number;
  maxDuration: number;
  p95Duration: number;
}

export interface CircuitBreakerStatus {
  failures: number;
  isOpen: boolean;
  lastFailure?: string;
}

export class NovaCronError extends Error {
  constructor(message: string, public code?: string, public status?: number) {
    super(message);
    this.name = 'NovaCronError';
  }
}

export class AuthenticationError extends NovaCronError {
  constructor(message: string) {
    super(message, 'AUTHENTICATION_ERROR', 401);
    this.name = 'AuthenticationError';
  }
}

export class ValidationError extends NovaCronError {
  constructor(message: string) {
    super(message, 'VALIDATION_ERROR', 422);
    this.name = 'ValidationError';
  }
}

export class ResourceNotFoundError extends NovaCronError {
  constructor(message: string) {
    super(message, 'RESOURCE_NOT_FOUND', 404);
    this.name = 'ResourceNotFoundError';
  }
}

export class CircuitBreakerError extends NovaCronError {
  constructor(endpoint: string) {
    super(`Circuit breaker open for ${endpoint}`, 'CIRCUIT_BREAKER_OPEN', 503);
    this.name = 'CircuitBreakerError';
  }
}

/**
 * Enhanced NovaCron API Client with multi-cloud federation, AI integration,
 * caching, and advanced reliability features
 */
export class EnhancedNovaCronClient extends EventEmitter {
  private http: AxiosInstance;
  private redis?: Redis;
  private tokenRefreshTimer?: NodeJS.Timer;
  private tokenExpiresAt?: Date;
  private circuitBreakerFailures: Map<string, number> = new Map();
  private circuitBreakerLastFailure: Map<string, Date> = new Map();
  private requestMetrics: Map<string, number[]> = new Map();

  constructor(private config: ClientConfig) {
    super();

    if (!config.baseURL) {
      throw new Error('baseURL is required');
    }

    // Set defaults
    this.config = {
      timeout: 30000,
      maxRetries: 3,
      retryDelay: 1000,
      cacheTTL: 300,
      enableAIFeatures: false,
      cloudProvider: CloudProvider.LOCAL,
      circuitBreakerThreshold: 5,
      circuitBreakerTimeout: 60000,
      enableMetrics: true,
      ...config,
    };

    // Create axios instance with enhanced configuration
    this.http = axios.create({
      baseURL: this.config.baseURL,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'User-Agent': `NovaCron-TypeScript-SDK/2.0.0 (${this.config.cloudProvider})`,
        'X-Cloud-Provider': this.config.cloudProvider,
      },
    });

    if (this.config.region) {
      this.http.defaults.headers['X-Cloud-Region'] = this.config.region;
    }

    this.setupInterceptors();
    this.initializeRedis();
    this.startTokenRefresh();
  }

  private setupInterceptors(): void {
    // Request interceptor
    this.http.interceptors.request.use(
      (config) => {
        // Add authentication
        if (this.config.apiToken) {
          config.headers.Authorization = `Bearer ${this.config.apiToken}`;
        } else if (this.config.username && this.config.password) {
          config.auth = {
            username: this.config.username,
            password: this.config.password,
          };
        }

        // Check circuit breaker
        const endpoint = `${config.method?.toUpperCase()}:${config.url}`;
        if (this.isCircuitBreakerOpen(endpoint)) {
          throw new CircuitBreakerError(endpoint);
        }

        // Add request start time for metrics
        config.metadata = { startTime: Date.now() };

        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.http.interceptors.response.use(
      (response) => {
        const endpoint = `${response.config.method?.toUpperCase()}:${response.config.url}`;
        
        // Record metrics
        if (this.config.enableMetrics && response.config.metadata?.startTime) {
          const duration = Date.now() - response.config.metadata.startTime;
          this.recordMetrics(endpoint, duration);
        }

        // Record circuit breaker success
        this.recordCircuitBreakerSuccess(endpoint);

        return response;
      },
      async (error) => {
        const endpoint = error.config ? 
          `${error.config.method?.toUpperCase()}:${error.config.url}` : 'unknown';

        // Record metrics
        if (this.config.enableMetrics && error.config?.metadata?.startTime) {
          const duration = Date.now() - error.config.metadata.startTime;
          this.recordMetrics(endpoint, duration);
        }

        // Record circuit breaker failure
        this.recordCircuitBreakerFailure(endpoint);

        // Handle specific error types
        if (error.response?.status === 401) {
          throw new AuthenticationError(error.response.data?.message || 'Authentication failed');
        } else if (error.response?.status === 404) {
          throw new ResourceNotFoundError(error.response.data?.message || 'Resource not found');
        } else if (error.response?.status === 422) {
          throw new ValidationError(error.response.data?.message || 'Validation failed');
        }

        // Retry logic
        const { config } = error;
        if (!config._retryCount) {
          config._retryCount = 0;
        }

        const shouldRetry = config._retryCount < this.config.maxRetries! &&
          (!error.response || error.response.status >= 500);

        if (shouldRetry) {
          config._retryCount += 1;
          const delay = this.config.retryDelay! * Math.pow(2, config._retryCount - 1);
          
          await new Promise(resolve => setTimeout(resolve, delay));
          return this.http.request(config);
        }

        return Promise.reject(error);
      }
    );
  }

  private async initializeRedis(): Promise<void> {
    if (this.config.redisUrl) {
      try {
        this.redis = new Redis(this.config.redisUrl, {
          maxRetriesPerRequest: 3,
          retryDelayOnFailover: 100,
          lazyConnect: true,
        });

        await this.redis.ping();
        console.log('Redis connected successfully');
      } catch (error) {
        console.warn('Redis connection failed:', error);
        this.redis = undefined;
      }
    }
  }

  private startTokenRefresh(): void {
    if (this.config.apiToken && !this.tokenRefreshTimer) {
      this.tokenRefreshTimer = setInterval(async () => {
        try {
          if (this.tokenExpiresAt) {
            const refreshAt = new Date(this.tokenExpiresAt.getTime() - 5 * 60 * 1000); // 5 minutes before
            if (new Date() >= refreshAt) {
              await this.refreshToken();
              console.log('Token refreshed successfully');
            }
          }
        } catch (error) {
          console.error('Token refresh error:', error);
        }
      }, 60000); // Check every minute
    }
  }

  private isCircuitBreakerOpen(endpoint: string): boolean {
    const failures = this.circuitBreakerFailures.get(endpoint) || 0;
    const lastFailure = this.circuitBreakerLastFailure.get(endpoint);

    if (failures >= this.config.circuitBreakerThreshold!) {
      if (lastFailure) {
        const timeSinceFailure = Date.now() - lastFailure.getTime();
        if (timeSinceFailure < this.config.circuitBreakerTimeout!) {
          return true;
        } else {
          // Reset circuit breaker after timeout
          this.circuitBreakerFailures.delete(endpoint);
          this.circuitBreakerLastFailure.delete(endpoint);
        }
      }
    }

    return false;
  }

  private recordCircuitBreakerFailure(endpoint: string): void {
    const failures = this.circuitBreakerFailures.get(endpoint) || 0;
    this.circuitBreakerFailures.set(endpoint, failures + 1);
    this.circuitBreakerLastFailure.set(endpoint, new Date());
  }

  private recordCircuitBreakerSuccess(endpoint: string): void {
    this.circuitBreakerFailures.delete(endpoint);
    this.circuitBreakerLastFailure.delete(endpoint);
  }

  private recordMetrics(endpoint: string, duration: number): void {
    if (!this.config.enableMetrics) return;

    if (!this.requestMetrics.has(endpoint)) {
      this.requestMetrics.set(endpoint, []);
    }

    const metrics = this.requestMetrics.get(endpoint)!;
    metrics.push(duration);

    // Keep only last 100 measurements
    if (metrics.length > 100) {
      metrics.shift();
    }
  }

  private async getFromCache(key: string): Promise<any> {
    if (!this.redis) return null;

    try {
      const cached = await this.redis.get(key);
      return cached ? JSON.parse(cached) : null;
    } catch (error) {
      console.debug('Cache get error:', error);
      return null;
    }
  }

  private async setCache(key: string, data: any, ttl?: number): Promise<void> {
    if (!this.redis) return;

    try {
      const cacheKey = `novacron:${key}`;
      const cacheTTL = ttl || this.config.cacheTTL!;
      await this.redis.setex(cacheKey, cacheTTL, JSON.stringify(data));
    } catch (error) {
      console.debug('Cache set error:', error);
    }
  }

  private getCacheKey(method: string, url: string, params?: any): string {
    const parts = [method, url, this.config.cloudProvider];
    if (this.config.region) parts.push(this.config.region);
    if (params) parts.push(JSON.stringify(params));
    return parts.join(':');
  }

  // AI-Powered Methods

  async getIntelligentPlacementRecommendation(
    vmSpecs: VMSpec,
    constraints?: Record<string, any>
  ): Promise<PlacementRecommendation> {
    if (!this.config.enableAIFeatures) {
      throw new NovaCronError('AI features not enabled');
    }

    const requestData = {
      vm_specs: vmSpecs,
      constraints: constraints || {},
      cloud_provider: this.config.cloudProvider,
      region: this.config.region,
    };

    const response = await this.http.post('/api/ai/placement', requestData);
    return response.data;
  }

  async getPredictiveScalingForecast(
    vmId: string,
    forecastHours: number = 24
  ): Promise<any> {
    if (!this.config.enableAIFeatures) {
      throw new NovaCronError('AI features not enabled');
    }

    const response = await this.http.get(`/api/ai/scaling/${vmId}`, {
      params: { forecast_hours: forecastHours }
    });

    return response.data;
  }

  async detectAnomalies(
    vmId?: string,
    timeWindow: number = 3600
  ): Promise<any[]> {
    if (!this.config.enableAIFeatures) {
      throw new NovaCronError('AI features not enabled');
    }

    const params: any = { time_window: timeWindow };
    if (vmId) params.vm_id = vmId;

    const response = await this.http.get('/api/ai/anomalies', { params });
    return response.data;
  }

  async getCostOptimizationRecommendations(
    tenantId?: string
  ): Promise<any[]> {
    if (!this.config.enableAIFeatures) {
      throw new NovaCronError('AI features not enabled');
    }

    const params: any = {};
    if (tenantId) params.tenant_id = tenantId;

    const response = await this.http.get('/api/ai/cost-optimization', { params });
    return response.data;
  }

  // Multi-Cloud Federation Methods

  async listFederatedClusters(): Promise<any[]> {
    const cacheKey = this.getCacheKey('GET', '/api/federation/clusters');
    
    let cached = await this.getFromCache(cacheKey);
    if (cached) return cached;

    const response = await this.http.get('/api/federation/clusters');
    await this.setCache(cacheKey, response.data, 300); // Cache for 5 minutes
    
    return response.data;
  }

  async createCrossCloudMigration(
    vmId: string,
    targetCluster: string,
    targetProvider: CloudProvider,
    targetRegion: string,
    migrationOptions?: Record<string, any>
  ): Promise<any> {
    const requestData = {
      vm_id: vmId,
      target_cluster: targetCluster,
      target_provider: targetProvider,
      target_region: targetRegion,
      options: migrationOptions || {}
    };

    const response = await this.http.post('/api/federation/migrations', requestData);
    return response.data;
  }

  async getCrossCloudCosts(
    sourceProvider: CloudProvider,
    targetProvider: CloudProvider,
    vmSpecs: VMSpec
  ): Promise<any> {
    const requestData = {
      source_provider: sourceProvider,
      target_provider: targetProvider,
      vm_specs: vmSpecs
    };

    const response = await this.http.post('/api/federation/cost-comparison', requestData);
    return response.data;
  }

  // Enhanced VM Management

  async createVMWithAIPlacement(
    vmRequest: VMSpec,
    useAIPlacement: boolean = true,
    placementConstraints?: Record<string, any>
  ): Promise<any> {
    let requestData = { ...vmRequest };

    if (useAIPlacement && this.config.enableAIFeatures) {
      // Get placement recommendation first
      const placementRec = await this.getIntelligentPlacementRecommendation(
        vmRequest,
        placementConstraints
      );

      if (placementRec.recommended_node) {
        requestData = {
          ...requestData,
          preferred_node: placementRec.recommended_node,
          placement_reasoning: placementRec.reasoning,
        };
      }
    }

    const response = await this.http.post('/api/vms', requestData);
    return response.data;
  }

  // Batch Operations

  async batchCreateVMs(
    requests: VMSpec[],
    concurrency: number = 5,
    useAIPlacement: boolean = false
  ): Promise<Array<any | Error>> {
    const semaphore = new Array(concurrency).fill(null);
    let semaphoreIndex = 0;

    const createSingleVM = async (request: VMSpec): Promise<any | Error> => {
      // Wait for available slot
      await new Promise<void>(resolve => {
        const checkSlot = () => {
          if (semaphore[semaphoreIndex] === null) {
            semaphore[semaphoreIndex] = true;
            resolve();
          } else {
            semaphoreIndex = (semaphoreIndex + 1) % concurrency;
            setTimeout(checkSlot, 10);
          }
        };
        checkSlot();
      });

      try {
        const result = await this.createVMWithAIPlacement(request, useAIPlacement);
        return result;
      } catch (error) {
        return error instanceof Error ? error : new Error(String(error));
      } finally {
        semaphore[semaphoreIndex] = null;
      }
    };

    return await Promise.all(requests.map(createSingleVM));
  }

  async batchMigrateVMs(
    migrations: MigrationSpec[],
    concurrency: number = 3
  ): Promise<Array<any | Error>> {
    const semaphore = new Array(concurrency).fill(null);
    let semaphoreIndex = 0;

    const migrateSingleVM = async (migrationSpec: MigrationSpec): Promise<any | Error> => {
      // Wait for available slot
      await new Promise<void>(resolve => {
        const checkSlot = () => {
          if (semaphore[semaphoreIndex] === null) {
            semaphore[semaphoreIndex] = true;
            resolve();
          } else {
            semaphoreIndex = (semaphoreIndex + 1) % concurrency;
            setTimeout(checkSlot, 10);
          }
        };
        checkSlot();
      });

      try {
        const { vm_id, ...request } = migrationSpec;
        const response = await this.http.post(`/api/vms/${vm_id}/migrate`, request);
        return response.data;
      } catch (error) {
        return error instanceof Error ? error : new Error(String(error));
      } finally {
        semaphore[semaphoreIndex] = null;
      }
    };

    return await Promise.all(migrations.map(migrateSingleVM));
  }

  // Enhanced WebSocket Support

  streamFederatedEvents(
    eventTypes?: string[],
    cloudProviders?: CloudProvider[]
  ): EventEmitter {
    const wsUrl = this.config.baseURL!
      .replace('http://', 'ws://')
      .replace('https://', 'wss://') + '/ws/federation/events';

    const params: string[] = [];
    if (eventTypes) {
      params.push(...eventTypes.map(et => `event_type=${et}`));
    }
    if (cloudProviders) {
      params.push(...cloudProviders.map(cp => `provider=${cp}`));
    }

    const finalUrl = params.length ? `${wsUrl}?${params.join('&')}` : wsUrl;

    const headers: Record<string, string> = {};
    if (this.config.apiToken) {
      headers.Authorization = `Bearer ${this.config.apiToken}`;
    }

    const ws = new WebSocket(finalUrl, { headers });
    const emitter = new EventEmitter();

    ws.on('open', () => {
      emitter.emit('connected');
    });

    ws.on('message', (data) => {
      try {
        const event = JSON.parse(data.toString());
        emitter.emit('event', event);
      } catch (error) {
        emitter.emit('error', error);
      }
    });

    ws.on('error', (error) => {
      emitter.emit('error', error);
    });

    ws.on('close', () => {
      emitter.emit('disconnected');
    });

    // Add close method to emitter
    (emitter as any).close = () => ws.close();

    return emitter;
  }

  // Performance Monitoring

  getRequestMetrics(): Record<string, RequestMetrics> {
    if (!this.config.enableMetrics) return {};

    const metrics: Record<string, RequestMetrics> = {};

    for (const [endpoint, timings] of this.requestMetrics.entries()) {
      if (timings.length > 0) {
        const sorted = [...timings].sort((a, b) => a - b);
        metrics[endpoint] = {
          count: timings.length,
          avgDuration: timings.reduce((a, b) => a + b) / timings.length,
          minDuration: Math.min(...timings),
          maxDuration: Math.max(...timings),
          p95Duration: sorted[Math.floor(sorted.length * 0.95)],
        };
      }
    }

    return metrics;
  }

  getCircuitBreakerStatus(): Record<string, CircuitBreakerStatus> {
    const status: Record<string, CircuitBreakerStatus> = {};

    for (const [endpoint, failures] of this.circuitBreakerFailures.entries()) {
      const isOpen = this.isCircuitBreakerOpen(endpoint);
      const lastFailure = this.circuitBreakerLastFailure.get(endpoint);

      status[endpoint] = {
        failures,
        isOpen,
        lastFailure: lastFailure?.toISOString(),
      };
    }

    return status;
  }

  // Enhanced Authentication

  async authenticate(username: string, password: string): Promise<string> {
    const response = await this.http.post('/api/auth/login', {
      username,
      password,
    });

    const token = response.data.token;
    const expiresIn = response.data.expires_in || 3600; // Default 1 hour

    if (token) {
      this.config.apiToken = token;
      this.tokenExpiresAt = new Date(Date.now() + expiresIn * 1000);
      return token;
    } else {
      throw new AuthenticationError('Failed to get authentication token');
    }
  }

  async refreshToken(): Promise<string> {
    const response = await this.http.post('/api/auth/refresh');
    
    const token = response.data.token;
    const expiresIn = response.data.expires_in || 3600;

    if (token) {
      this.config.apiToken = token;
      this.tokenExpiresAt = new Date(Date.now() + expiresIn * 1000);
      return token;
    } else {
      throw new AuthenticationError('Failed to refresh authentication token');
    }
  }

  // Cleanup

  async close(): Promise<void> {
    if (this.tokenRefreshTimer) {
      clearInterval(this.tokenRefreshTimer);
      this.tokenRefreshTimer = undefined;
    }

    if (this.redis) {
      await this.redis.quit();
      this.redis = undefined;
    }
  }

  // All original methods can be added here with enhancements...
}

export default EnhancedNovaCronClient;