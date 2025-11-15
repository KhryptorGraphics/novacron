/**
 * Performance Benchmarks for Initialization System
 * Tests startup time, memory usage, and concurrent operations
 */

const { describe, it, expect, beforeEach, afterEach } = require('@jest/globals');
const { PlatformInitializer } = require('../../../src/init');

// Mock dependencies
jest.mock('fs', () => ({
  existsSync: jest.fn().mockReturnValue(true),
  readFileSync: jest.fn().mockReturnValue(JSON.stringify({
    database: { host: 'localhost', port: 5432, database: 'bench' },
    services: {},
    logging: { level: 'error' }
  })),
  mkdirSync: jest.fn(),
  accessSync: jest.fn()
}));

jest.mock('pg', () => ({
  Pool: jest.fn().mockImplementation(() => ({
    connect: jest.fn().mockResolvedValue({
      query: jest.fn().mockResolvedValue({}),
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

describe('Performance Benchmarks', () => {
  describe('Startup Time', () => {
    it('should initialize under 1 second (cold start)', async () => {
      const initializer = new PlatformInitializer({ autoConnect: false });

      const start = performance.now();
      await initializer.initialize();
      const duration = performance.now() - start;

      expect(duration).toBeLessThan(1000);

      await initializer.shutdown();
    }, 5000);

    it('should initialize under 500ms (warm start)', async () => {
      // First initialization (warm up)
      const warmup = new PlatformInitializer({ autoConnect: false });
      await warmup.initialize();
      await warmup.shutdown();

      // Actual benchmark
      const initializer = new PlatformInitializer({ autoConnect: false });

      const start = performance.now();
      await initializer.initialize();
      const duration = performance.now() - start;

      expect(duration).toBeLessThan(500);

      await initializer.shutdown();
    }, 5000);

    it('should complete database connections under 2 seconds', async () => {
      const initializer = new PlatformInitializer({ autoConnect: true });

      const start = performance.now();
      await initializer.initialize();
      const duration = performance.now() - start;

      expect(duration).toBeLessThan(2000);

      await initializer.shutdown();
    }, 5000);
  });

  describe('Memory Usage', () => {
    it('should maintain reasonable memory footprint', async () => {
      if (global.gc) {
        global.gc();
      }

      const before = process.memoryUsage().heapUsed;

      const initializer = new PlatformInitializer({ autoConnect: false });
      await initializer.initialize();

      const after = process.memoryUsage().heapUsed;
      const increase = (after - before) / 1024 / 1024; // MB

      await initializer.shutdown();

      // Should use less than 50MB
      expect(increase).toBeLessThan(50);
    });

    it('should not leak memory over multiple cycles', async () => {
      if (global.gc) {
        global.gc();
      }

      const before = process.memoryUsage().heapUsed;

      // Run 10 init-shutdown cycles
      for (let i = 0; i < 10; i++) {
        const initializer = new PlatformInitializer({ autoConnect: false });
        await initializer.initialize();
        await initializer.shutdown();
      }

      if (global.gc) {
        global.gc();
      }

      const after = process.memoryUsage().heapUsed;
      const increase = (after - before) / 1024 / 1024; // MB

      // Should not accumulate more than 100MB over 10 cycles
      expect(increase).toBeLessThan(100);
    });
  });

  describe('Concurrent Operations', () => {
    it('should handle 10 concurrent initializations', async () => {
      const initializers = Array(10).fill(null).map(() =>
        new PlatformInitializer({ autoConnect: false })
      );

      const start = performance.now();
      await Promise.all(initializers.map(i => i.initialize()));
      const duration = performance.now() - start;

      // Should complete within reasonable time
      expect(duration).toBeLessThan(5000);

      await Promise.all(initializers.map(i => i.shutdown()));
    }, 10000);

    it('should handle rapid sequential operations', async () => {
      const iterations = 50;
      const durations = [];

      for (let i = 0; i < iterations; i++) {
        const initializer = new PlatformInitializer({ autoConnect: false });

        const start = performance.now();
        await initializer.initialize();
        await initializer.shutdown();
        const duration = performance.now() - start;

        durations.push(duration);
      }

      const avgDuration = durations.reduce((a, b) => a + b, 0) / durations.length;
      const maxDuration = Math.max(...durations);

      expect(avgDuration).toBeLessThan(100);
      expect(maxDuration).toBeLessThan(500);
    }, 30000);
  });

  describe('Event Emission Performance', () => {
    it('should emit events without significant overhead', async () => {
      const initializer = new PlatformInitializer({ autoConnect: false });
      const events = [];

      // Attach many listeners
      for (let i = 0; i < 100; i++) {
        initializer.on('init:start', () => events.push('start'));
        initializer.on('init:complete', () => events.push('complete'));
      }

      const start = performance.now();
      await initializer.initialize();
      const duration = performance.now() - start;

      // Should not add significant overhead
      expect(duration).toBeLessThan(1500);
      expect(events.length).toBeGreaterThan(0);

      await initializer.shutdown();
    });
  });

  describe('Configuration Loading Performance', () => {
    it('should load large configuration quickly', async () => {
      const fs = require('fs');

      // Mock large config
      const largeConfig = {
        database: { host: 'localhost', port: 5432, database: 'test' },
        services: {},
        logging: {},
        ...Object.fromEntries(
          Array(1000).fill(null).map((_, i) => [`key${i}`, `value${i}`])
        )
      };

      fs.readFileSync.mockReturnValue(JSON.stringify(largeConfig));

      const initializer = new PlatformInitializer({ autoConnect: false });

      const start = performance.now();
      await initializer.loadConfiguration();
      const duration = performance.now() - start;

      expect(duration).toBeLessThan(100);
      expect(initializer.config).toBeDefined();

      await initializer.shutdown();
    });
  });

  describe('Service Initialization Performance', () => {
    it('should handle missing services without delay', async () => {
      const initializer = new PlatformInitializer({ autoConnect: false });

      const start = performance.now();
      await initializer.initializeCoreServices();
      const duration = performance.now() - start;

      // Should fail fast for missing services
      expect(duration).toBeLessThan(500);

      await initializer.shutdown();
    });
  });

  describe('Shutdown Performance', () => {
    it('should shutdown quickly', async () => {
      const initializer = new PlatformInitializer({ autoConnect: false });
      await initializer.initialize();

      const start = performance.now();
      await initializer.shutdown();
      const duration = performance.now() - start;

      expect(duration).toBeLessThan(1000);
    });

    it('should handle concurrent shutdowns', async () => {
      const initializer = new PlatformInitializer({ autoConnect: false });
      await initializer.initialize();

      const start = performance.now();
      await Promise.all([
        initializer.shutdown(),
        initializer.shutdown(),
        initializer.shutdown()
      ]);
      const duration = performance.now() - start;

      expect(duration).toBeLessThan(1500);
    });
  });
});

describe('Stress Tests', () => {
  it('should handle 100 rapid init-shutdown cycles', async () => {
    const start = performance.now();

    for (let i = 0; i < 100; i++) {
      const initializer = new PlatformInitializer({
        autoConnect: false,
        enableLogging: false
      });
      await initializer.initialize();
      await initializer.shutdown();
    }

    const duration = performance.now() - start;
    const avgPerCycle = duration / 100;

    expect(avgPerCycle).toBeLessThan(100); // Average under 100ms per cycle
  }, 60000);

  it('should handle many service registrations', async () => {
    const initializer = new PlatformInitializer({ autoConnect: false });
    await initializer.initialize();

    const start = performance.now();

    // Add many services
    for (let i = 0; i < 1000; i++) {
      initializer.services.set(`service-${i}`, { name: `service-${i}` });
    }

    const duration = performance.now() - start;

    expect(duration).toBeLessThan(100);
    expect(initializer.services.size).toBe(1000);

    await initializer.shutdown();
  });
});
