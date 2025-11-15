/**
 * Database Connection Tests
 * Tests PostgreSQL and Redis connectivity with various scenarios
 */

const { describe, it, expect, beforeEach, afterEach } = require('@jest/globals');
const { PlatformInitializer, DatabaseConnectionError } = require('../../../src/init');

// Mock database modules
const mockPostgresPool = {
  connect: jest.fn(),
  end: jest.fn().mockResolvedValue(),
  query: jest.fn()
};

const mockPostgresClient = {
  query: jest.fn().mockResolvedValue({ rows: [{ now: new Date() }] }),
  release: jest.fn()
};

const mockRedisClient = {
  connect: jest.fn().mockResolvedValue(),
  ping: jest.fn().mockResolvedValue('PONG'),
  quit: jest.fn().mockResolvedValue()
};

jest.mock('pg', () => ({
  Pool: jest.fn().mockImplementation(() => mockPostgresPool)
}));

jest.mock('redis', () => ({
  createClient: jest.fn().mockReturnValue(mockRedisClient)
}));

jest.mock('fs', () => ({
  existsSync: jest.fn().mockReturnValue(true),
  readFileSync: jest.fn().mockReturnValue(JSON.stringify({
    database: {
      postgres: {
        host: 'localhost',
        port: 5432,
        database: 'novacron_test',
        user: 'test_user',
        password: 'test_pass',
        poolSize: 10
      },
      redis: {
        host: 'localhost',
        port: 6379,
        password: 'redis_pass',
        database: 0
      }
    },
    services: {},
    logging: { level: 'error' }
  })),
  mkdirSync: jest.fn(),
  accessSync: jest.fn()
}));

describe('Database Connections - PostgreSQL', () => {
  let initializer;
  const { Pool } = require('pg');

  beforeEach(() => {
    jest.clearAllMocks();
    mockPostgresPool.connect.mockResolvedValue(mockPostgresClient);
  });

  afterEach(async () => {
    if (initializer) {
      await initializer.shutdown().catch(() => {});
    }
  });

  it('should connect to PostgreSQL successfully', async () => {
    initializer = new PlatformInitializer({ autoConnect: true });
    await initializer.initialize();

    expect(Pool).toHaveBeenCalledWith(
      expect.objectContaining({
        host: 'localhost',
        port: 5432,
        database: 'novacron_test',
        user: 'test_user',
        password: 'test_pass'
      })
    );

    expect(mockPostgresPool.connect).toHaveBeenCalled();
    expect(mockPostgresClient.query).toHaveBeenCalledWith('SELECT NOW()');
    expect(mockPostgresClient.release).toHaveBeenCalled();
  });

  it('should add postgres pool to services', async () => {
    initializer = new PlatformInitializer({ autoConnect: true });
    await initializer.initialize();

    const postgresService = initializer.getService('postgres');
    expect(postgresService).toBeDefined();
    expect(postgresService).toBe(mockPostgresPool);
  });

  it('should configure connection pool settings', async () => {
    initializer = new PlatformInitializer({ autoConnect: true });
    await initializer.initialize();

    expect(Pool).toHaveBeenCalledWith(
      expect.objectContaining({
        max: 10,
        idleTimeoutMillis: 30000,
        connectionTimeoutMillis: 10000
      })
    );
  });

  it('should throw DatabaseConnectionError on connection failure', async () => {
    mockPostgresPool.connect.mockRejectedValue(new Error('Connection refused'));

    initializer = new PlatformInitializer({ autoConnect: true });

    await expect(initializer.initialize()).rejects.toThrow();
  });

  it('should throw DatabaseConnectionError on query failure', async () => {
    mockPostgresClient.query.mockRejectedValue(new Error('Query failed'));

    initializer = new PlatformInitializer({ autoConnect: true });

    await expect(initializer.initialize()).rejects.toThrow();
  });

  it('should close postgres connection on shutdown', async () => {
    initializer = new PlatformInitializer({ autoConnect: true });
    await initializer.initialize();

    await initializer.shutdown();

    expect(mockPostgresPool.end).toHaveBeenCalled();
  });
});

describe('Database Connections - Redis', () => {
  let initializer;
  const redis = require('redis');

  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(async () => {
    if (initializer) {
      await initializer.shutdown().catch(() => {});
    }
  });

  it('should connect to Redis successfully', async () => {
    initializer = new PlatformInitializer({ autoConnect: true });
    await initializer.initialize();

    expect(redis.createClient).toHaveBeenCalledWith(
      expect.objectContaining({
        host: 'localhost',
        port: 6379,
        password: 'redis_pass',
        db: 0
      })
    );

    expect(mockRedisClient.connect).toHaveBeenCalled();
    expect(mockRedisClient.ping).toHaveBeenCalled();
  });

  it('should add redis client to services', async () => {
    initializer = new PlatformInitializer({ autoConnect: true });
    await initializer.initialize();

    const redisService = initializer.getService('redis');
    expect(redisService).toBeDefined();
    expect(redisService).toBe(mockRedisClient);
  });

  it('should throw DatabaseConnectionError on connection failure', async () => {
    mockRedisClient.connect.mockRejectedValue(new Error('Connection refused'));

    initializer = new PlatformInitializer({ autoConnect: true });

    await expect(initializer.initialize()).rejects.toThrow();
  });

  it('should throw DatabaseConnectionError on ping failure', async () => {
    mockRedisClient.ping.mockRejectedValue(new Error('Ping failed'));

    initializer = new PlatformInitializer({ autoConnect: true });

    await expect(initializer.initialize()).rejects.toThrow();
  });

  it('should close redis connection on shutdown', async () => {
    initializer = new PlatformInitializer({ autoConnect: true });
    await initializer.initialize();

    await initializer.shutdown();

    expect(mockRedisClient.quit).toHaveBeenCalled();
  });
});

describe('Database Connections - Configuration Variations', () => {
  let initializer;

  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(async () => {
    if (initializer) {
      await initializer.shutdown().catch(() => {});
    }
  });

  it('should skip database connections when autoConnect is false', async () => {
    const { Pool } = require('pg');
    const redis = require('redis');

    initializer = new PlatformInitializer({ autoConnect: false });
    await initializer.initialize();

    expect(Pool).not.toHaveBeenCalled();
    expect(redis.createClient).not.toHaveBeenCalled();
  });

  it('should handle missing database configuration', async () => {
    const fs = require('fs');
    fs.readFileSync.mockReturnValue(JSON.stringify({
      database: {},
      services: {},
      logging: {}
    }));

    initializer = new PlatformInitializer({ autoConnect: true });
    await initializer.initialize();

    // Should complete without errors
    expect(initializer.state).toBe('initialized');
  });

  it('should handle postgres-only configuration', async () => {
    const fs = require('fs');
    fs.readFileSync.mockReturnValue(JSON.stringify({
      database: {
        postgres: {
          host: 'localhost',
          port: 5432,
          database: 'test',
          user: 'user',
          password: 'pass'
        }
      },
      services: {},
      logging: {}
    }));

    const redis = require('redis');
    initializer = new PlatformInitializer({ autoConnect: true });
    await initializer.initialize();

    expect(initializer.getService('postgres')).toBeDefined();
    expect(redis.createClient).not.toHaveBeenCalled();
  });

  it('should handle redis-only configuration', async () => {
    const fs = require('fs');
    fs.readFileSync.mockReturnValue(JSON.stringify({
      database: {
        redis: {
          host: 'localhost',
          port: 6379,
          password: 'pass'
        }
      },
      services: {},
      logging: {}
    }));

    const { Pool } = require('pg');
    initializer = new PlatformInitializer({ autoConnect: true });
    await initializer.initialize();

    expect(initializer.getService('redis')).toBeDefined();
    expect(Pool).not.toHaveBeenCalled();
  });
});

describe('Database Connections - Error Recovery', () => {
  let initializer;

  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(async () => {
    if (initializer) {
      await initializer.shutdown().catch(() => {});
    }
  });

  it('should handle database connection timeout', async () => {
    mockPostgresPool.connect.mockImplementation(() =>
      new Promise((resolve) => setTimeout(resolve, 20000))
    );

    initializer = new PlatformInitializer({
      autoConnect: true,
      timeout: 1000
    });

    await expect(initializer.initialize()).rejects.toThrow();
  });

  it('should continue shutdown even if postgres close fails', async () => {
    mockPostgresPool.end.mockRejectedValue(new Error('Close failed'));

    initializer = new PlatformInitializer({ autoConnect: true });
    await initializer.initialize();

    await expect(initializer.shutdown()).resolves.not.toThrow();
  });

  it('should continue shutdown even if redis quit fails', async () => {
    mockRedisClient.quit.mockRejectedValue(new Error('Quit failed'));

    initializer = new PlatformInitializer({ autoConnect: true });
    await initializer.initialize();

    await expect(initializer.shutdown()).resolves.not.toThrow();
  });
});

describe('Database Connections - Concurrent Operations', () => {
  let initializer;

  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(async () => {
    if (initializer) {
      await initializer.shutdown().catch(() => {});
    }
  });

  it('should handle concurrent initialization attempts', async () => {
    initializer = new PlatformInitializer({ autoConnect: true });

    const init1 = initializer.initialize();
    const init2 = initializer.initialize();

    const results = await Promise.allSettled([init1, init2]);

    // At least one should succeed
    const successes = results.filter(r => r.status === 'fulfilled');
    expect(successes.length).toBeGreaterThan(0);
  });

  it('should handle concurrent shutdown calls', async () => {
    initializer = new PlatformInitializer({ autoConnect: true });
    await initializer.initialize();

    const shutdown1 = initializer.shutdown();
    const shutdown2 = initializer.shutdown();

    await expect(Promise.all([shutdown1, shutdown2])).resolves.toBeDefined();
  });
});
