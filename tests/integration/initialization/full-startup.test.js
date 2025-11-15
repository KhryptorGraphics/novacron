/**
 * Integration Tests - Full Startup Sequence
 * Tests complete platform initialization from start to finish
 */

const { describe, it, expect, beforeEach, afterEach } = require('@jest/globals');
const { PlatformInitializer, initializePlatform, InitState } = require('../../../src/init');
const fs = require('fs');
const path = require('path');

// Mock external dependencies
jest.mock('pg', () => ({
  Pool: jest.fn().mockImplementation(() => ({
    connect: jest.fn().mockResolvedValue({
      query: jest.fn().mockResolvedValue({ rows: [{ now: new Date() }] }),
      release: jest.fn()
    }),
    end: jest.fn().mockResolvedValue()
  }))
}));

jest.mock('redis', () => ({
  createClient: jest.fn().mockReturnValue({
    connect: jest.fn().mockResolvedValue(),
    ping: jest.fn().mockResolvedValue('PONG'),
    quit: jest.fn().mockResolvedValue()
  })
}));

describe('Integration - Full Platform Startup', () => {
  let initializer;
  let configPath;

  beforeEach(() => {
    // Create temporary config directory
    configPath = path.join(process.cwd(), 'config');

    // Mock fs to return valid configuration
    jest.spyOn(fs, 'existsSync').mockReturnValue(true);
    jest.spyOn(fs, 'readFileSync').mockReturnValue(JSON.stringify({
      database: {
        postgres: {
          host: 'localhost',
          port: 5432,
          database: 'novacron_integration_test',
          user: 'test_user',
          password: 'test_pass',
          poolSize: 5
        },
        redis: {
          host: 'localhost',
          port: 6379,
          password: 'redis_test',
          database: 1
        }
      },
      services: {
        cache: { enabled: true, ttl: 3600 },
        'workload-monitor': { enabled: true, interval: 5000 },
        'mcp-integration': { enabled: true }
      },
      logging: {
        level: 'info',
        format: 'json',
        destination: 'console'
      }
    }));
    jest.spyOn(fs, 'mkdirSync').mockImplementation(() => {});
    jest.spyOn(fs, 'accessSync').mockImplementation(() => {});
  });

  afterEach(async () => {
    if (initializer) {
      await initializer.shutdown().catch(() => {});
    }
    jest.restoreAllMocks();
  });

  it('should complete full initialization lifecycle', async () => {
    const events = [];

    initializer = new PlatformInitializer({ autoConnect: true });

    // Track all lifecycle events
    initializer.on('init:start', (data) => events.push({ event: 'start', data }));
    initializer.on('init:config-loaded', (data) => events.push({ event: 'config-loaded', data }));
    initializer.on('init:logging-setup', () => events.push({ event: 'logging-setup' }));
    initializer.on('init:environment-validated', () => events.push({ event: 'environment-validated' }));
    initializer.on('init:core-services-ready', () => events.push({ event: 'core-services' }));
    initializer.on('init:databases-connected', () => events.push({ event: 'databases' }));
    initializer.on('init:optional-services-ready', () => events.push({ event: 'optional-services' }));
    initializer.on('init:error-handlers-setup', () => events.push({ event: 'error-handlers' }));
    initializer.on('init:complete', (data) => events.push({ event: 'complete', data }));

    const result = await initializer.initialize();

    expect(result.status).toBe('success');
    expect(result.state).toBe(InitState.INITIALIZED);
    expect(events.length).toBeGreaterThan(0);
    expect(events[0].event).toBe('start');
    expect(events[events.length - 1].event).toBe('complete');
  });

  it('should initialize all components in correct order', async () => {
    const order = [];

    initializer = new PlatformInitializer({ autoConnect: true });

    initializer.on('init:config-loaded', () => order.push('config'));
    initializer.on('init:logging-setup', () => order.push('logging'));
    initializer.on('init:environment-validated', () => order.push('environment'));
    initializer.on('init:core-services-ready', () => order.push('core'));
    initializer.on('init:databases-connected', () => order.push('databases'));
    initializer.on('init:optional-services-ready', () => order.push('optional'));
    initializer.on('init:error-handlers-setup', () => order.push('error-handlers'));

    await initializer.initialize();

    expect(order).toEqual([
      'config',
      'logging',
      'environment',
      'core',
      'databases',
      'optional',
      'error-handlers'
    ]);
  });

  it('should complete initialization within reasonable time', async () => {
    initializer = new PlatformInitializer({ autoConnect: true });

    const start = Date.now();
    await initializer.initialize();
    const duration = Date.now() - start;

    expect(duration).toBeLessThan(5000); // Should complete within 5 seconds
  });

  it('should provide detailed initialization result', async () => {
    initializer = new PlatformInitializer({ autoConnect: true });

    const result = await initializer.initialize();

    expect(result).toMatchObject({
      status: 'success',
      state: InitState.INITIALIZED,
      duration: expect.any(Number),
      services: expect.any(Array),
      config: expect.any(Object),
      timestamp: expect.any(String)
    });
  });

  it('should perform graceful shutdown after initialization', async () => {
    initializer = new PlatformInitializer({ autoConnect: true });
    await initializer.initialize();

    const shutdownEvents = [];
    initializer.on('shutdown:start', () => shutdownEvents.push('start'));
    initializer.on('shutdown:complete', () => shutdownEvents.push('complete'));

    await initializer.shutdown();

    expect(shutdownEvents).toEqual(['start', 'complete']);
  });

  it('should handle SIGTERM gracefully', async () => {
    initializer = new PlatformInitializer({ autoConnect: true });
    await initializer.initialize();

    const shutdownSpy = jest.spyOn(initializer, 'shutdown');

    // Simulate SIGTERM
    process.emit('SIGTERM');

    // Wait for async handling
    await new Promise(resolve => setTimeout(resolve, 100));

    expect(shutdownSpy).toHaveBeenCalled();

    shutdownSpy.mockRestore();
  });
});

describe('Integration - Factory Function', () => {
  beforeEach(() => {
    jest.spyOn(fs, 'existsSync').mockReturnValue(true);
    jest.spyOn(fs, 'readFileSync').mockReturnValue(JSON.stringify({
      database: {
        host: 'localhost',
        port: 5432,
        database: 'test'
      },
      services: {},
      logging: {}
    }));
    jest.spyOn(fs, 'mkdirSync').mockImplementation(() => {});
    jest.spyOn(fs, 'accessSync').mockImplementation(() => {});
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('should create and initialize platform via factory', async () => {
    const platform = await initializePlatform({ autoConnect: false });

    expect(platform).toBeInstanceOf(PlatformInitializer);
    expect(platform.state).toBe(InitState.INITIALIZED);

    await platform.shutdown();
  });

  it('should pass options through factory function', async () => {
    const platform = await initializePlatform({
      environment: 'test',
      enableLogging: false,
      autoConnect: false,
      timeout: 15000
    });

    expect(platform.options.environment).toBe('test');
    expect(platform.options.enableLogging).toBe(false);
    expect(platform.options.timeout).toBe(15000);

    await platform.shutdown();
  });
});

describe('Integration - Error Scenarios', () => {
  beforeEach(() => {
    jest.spyOn(fs, 'mkdirSync').mockImplementation(() => {});
    jest.spyOn(fs, 'accessSync').mockImplementation(() => {});
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('should handle missing configuration files', async () => {
    jest.spyOn(fs, 'existsSync').mockReturnValue(false);

    const initializer = new PlatformInitializer();

    await expect(initializer.initialize()).rejects.toThrow();
  });

  it('should handle corrupted configuration JSON', async () => {
    jest.spyOn(fs, 'existsSync').mockReturnValue(true);
    jest.spyOn(fs, 'readFileSync').mockReturnValue('{ invalid json }');

    const initializer = new PlatformInitializer();

    await expect(initializer.initialize()).rejects.toThrow();
  });

  it('should emit error events on failure', async () => {
    jest.spyOn(fs, 'existsSync').mockReturnValue(false);

    const initializer = new PlatformInitializer();
    const errorEvents = [];

    initializer.on('init:failed', (data) => errorEvents.push(data));

    try {
      await initializer.initialize();
    } catch (error) {
      expect(errorEvents.length).toBeGreaterThan(0);
      expect(errorEvents[0].status).toBe('failed');
    }
  });
});

describe('Integration - Performance Benchmarks', () => {
  beforeEach(() => {
    jest.spyOn(fs, 'existsSync').mockReturnValue(true);
    jest.spyOn(fs, 'readFileSync').mockReturnValue(JSON.stringify({
      database: { host: 'localhost', port: 5432, database: 'test' },
      services: {},
      logging: {}
    }));
    jest.spyOn(fs, 'mkdirSync').mockImplementation(() => {});
    jest.spyOn(fs, 'accessSync').mockImplementation(() => {});
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('should initialize multiple times with consistent performance', async () => {
    const durations = [];

    for (let i = 0; i < 5; i++) {
      const initializer = new PlatformInitializer({ autoConnect: false });

      const start = Date.now();
      await initializer.initialize();
      const duration = Date.now() - start;

      durations.push(duration);
      await initializer.shutdown();
    }

    const avgDuration = durations.reduce((a, b) => a + b, 0) / durations.length;
    const maxDuration = Math.max(...durations);

    expect(avgDuration).toBeLessThan(3000);
    expect(maxDuration).toBeLessThan(5000);
  });

  it('should handle rapid init-shutdown cycles', async () => {
    for (let i = 0; i < 10; i++) {
      const initializer = new PlatformInitializer({ autoConnect: false });
      await initializer.initialize();
      await initializer.shutdown();
    }

    // Should complete without errors or memory leaks
    expect(true).toBe(true);
  });
});
