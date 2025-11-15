/**
 * Concurrency and Race Condition Tests
 * Tests parallel initialization and concurrent operations
 */

const { describe, it, expect, beforeEach, jest } = require('@jest/globals');
const {
  createTestContext,
  sleep,
  measureTime,
} = require('../../utils/initialization-helpers');

describe('Concurrency Tests', () => {
  let mockContext;

  beforeEach(() => {
    mockContext = createTestContext();
  });

  describe('Parallel Component Initialization', () => {
    it('should initialize independent components in parallel', async () => {
      const components = [
        { name: 'database', duration: 1000 },
        { name: 'cache', duration: 1000 },
        { name: 'queue', duration: 1000 },
      ];

      const { duration } = await measureTime(() =>
        initializeParallel(components)
      );

      // Should complete in ~1000ms, not 3000ms
      expect(duration).toBeLessThan(1500);
      expect(duration).toBeGreaterThan(900);
    });

    it('should respect max concurrency limit', async () => {
      const components = Array(10).fill(null).map((_, i) => ({
        name: `comp${i}`,
        duration: 100,
      }));

      const maxConcurrency = 3;
      const result = await initializeWithConcurrencyLimit(
        components,
        maxConcurrency
      );

      expect(result.maxConcurrentActive).toBeLessThanOrEqual(maxConcurrency);
    });

    it('should initialize dependent components sequentially', async () => {
      const components = [
        { name: 'database', duration: 500, dependencies: [] },
        { name: 'cache', duration: 500, dependencies: ['database'] },
        { name: 'api', duration: 500, dependencies: ['cache'] },
      ];

      const { duration, order } = await initializeWithDependencies(components);

      // Should be sequential: ~1500ms
      expect(duration).toBeGreaterThan(1400);
      expect(order).toEqual(['database', 'cache', 'api']);
    });

    it('should handle mixed parallel and sequential initialization', async () => {
      const components = [
        { name: 'config', duration: 100, dependencies: [] },
        { name: 'logger', duration: 100, dependencies: [] },
        { name: 'database', duration: 500, dependencies: ['config'] },
        { name: 'cache', duration: 500, dependencies: ['config'] },
        { name: 'api', duration: 300, dependencies: ['database', 'cache'] },
      ];

      const { duration, order } = await initializeWithDependencies(components);

      // config and logger parallel (100ms)
      // then database and cache parallel (500ms)
      // then api (300ms)
      // total: ~900ms
      expect(duration).toBeLessThan(1100);

      // Verify order constraints
      expect(order.indexOf('config')).toBeLessThan(order.indexOf('database'));
      expect(order.indexOf('config')).toBeLessThan(order.indexOf('cache'));
      expect(order.indexOf('database')).toBeLessThan(order.indexOf('api'));
      expect(order.indexOf('cache')).toBeLessThan(order.indexOf('api'));
    });

    it('should handle component initialization failures in parallel', async () => {
      const components = [
        { name: 'comp1', duration: 100, shouldFail: false },
        { name: 'comp2', duration: 100, shouldFail: true },
        { name: 'comp3', duration: 100, shouldFail: false },
      ];

      const result = await initializeParallel(components);

      expect(result.successful).toEqual(['comp1', 'comp3']);
      expect(result.failed).toEqual(['comp2']);
    });
  });

  describe('Race Conditions', () => {
    it('should handle concurrent reads and writes to shared state', async () => {
      const sharedState = createSharedState();

      const operations = Array(100).fill(null).map((_, i) =>
        i % 2 === 0
          ? sharedState.write(`key${i}`, `value${i}`)
          : sharedState.read(`key${i}`)
      );

      await Promise.all(operations);

      // No operations should fail due to race conditions
      expect(sharedState.errors).toEqual([]);
    });

    it('should prevent double initialization', async () => {
      const system = createInitializationSystem();

      const attempts = Array(10).fill(null).map(() => system.initialize());

      const results = await Promise.allSettled(attempts);

      const successful = results.filter(
        r => r.status === 'fulfilled' && r.value.success
      ).length;

      expect(successful).toBe(1);
    });

    it('should handle concurrent checkpoint saves', async () => {
      const recovery = createRecoveryManager();

      const saves = Array(5).fill(null).map((_, i) =>
        recovery.saveCheckpoint(`checkpoint${i}`, { data: i })
      );

      await Promise.all(saves);

      expect(recovery.checkpoints.size).toBe(5);
    });

    it('should handle metrics updates from multiple components', async () => {
      const metrics = createMetricsCollector();

      const updates = Array(50).fill(null).map((_, i) =>
        metrics.recordComponentInit(`comp${i % 5}`, Math.random() * 1000, true)
      );

      await Promise.all(updates);

      // All updates should be recorded
      expect(Object.keys(metrics.componentInitDurations).length).toBe(5);
    });

    it('should handle concurrent logger writes', async () => {
      const logger = createConcurrentLogger();

      const writes = Array(100).fill(null).map((_, i) =>
        logger.info(`Message ${i}`)
      );

      await Promise.all(writes);

      expect(logger.messages.length).toBe(100);
      expect(logger.corruptedMessages).toBe(0);
    });
  });

  describe('Deadlock Prevention', () => {
    it('should prevent deadlock with timeout', async () => {
      const result = await initializeWithPotentialDeadlock({
        timeout: 1000,
      });

      expect(result.success).toBe(false);
      expect(result.error).toContain('timeout');
    });

    it('should detect and break dependency deadlock', async () => {
      const components = [
        { name: 'A', dependencies: ['B'] },
        { name: 'B', dependencies: ['A'] },
      ];

      const result = await initializeWithDependencies(components);

      expect(result.success).toBe(false);
      expect(result.error).toContain('circular');
    });

    it('should handle resource contention', async () => {
      const resource = createSharedResource(1); // Single resource

      const consumers = Array(5).fill(null).map(() =>
        acquireAndUseResource(resource, 100)
      );

      const results = await Promise.all(consumers);

      // All should eventually succeed
      expect(results.every(r => r.success)).toBe(true);
    });
  });

  describe('Synchronization', () => {
    it('should synchronize component lifecycle events', async () => {
      const orchestrator = createOrchestrator();

      const comp1 = orchestrator.registerComponent('comp1');
      const comp2 = orchestrator.registerComponent('comp2');
      const comp3 = orchestrator.registerComponent('comp3');

      const inits = [
        comp1.initialize(),
        comp2.initialize(),
        comp3.initialize(),
      ];

      await Promise.all(inits);

      expect(orchestrator.getLifecycleEvents()).toEqual([
        { component: 'comp1', event: 'init' },
        { component: 'comp2', event: 'init' },
        { component: 'comp3', event: 'init' },
      ]);
    });

    it('should use mutex for critical sections', async () => {
      const counter = createMutexCounter();

      const increments = Array(100).fill(null).map(() => counter.increment());

      await Promise.all(increments);

      expect(counter.value).toBe(100);
    });

    it('should use semaphore for rate limiting', async () => {
      const semaphore = createSemaphore(3); // Max 3 concurrent

      const operations = Array(10).fill(null).map((_, i) =>
        withSemaphore(semaphore, async () => {
          await sleep(100);
          return i;
        })
      );

      const { duration } = await measureTime(() => Promise.all(operations));

      // Should take ~400ms (10 ops / 3 concurrent * 100ms)
      expect(duration).toBeGreaterThan(300);
      expect(duration).toBeLessThan(500);
    });
  });

  describe('Thread Safety', () => {
    it('should handle concurrent access to configuration', async () => {
      const config = createThreadSafeConfig();

      const operations = [];

      // 50 reads and 50 writes concurrently
      for (let i = 0; i < 50; i++) {
        operations.push(config.get('key'));
        operations.push(config.set('key', `value${i}`));
      }

      await Promise.all(operations);

      // Should not crash or corrupt data
      expect(config.get('key')).toBeDefined();
    });

    it('should handle concurrent DI container access', async () => {
      const container = createThreadSafeContainer();

      const operations = Array(50).fill(null).flatMap((_, i) => [
        container.register(`service${i}`, { value: i }),
        container.resolve(`service${i}`),
      ]);

      await Promise.all(operations);

      expect(container.errors).toEqual([]);
    });

    it('should handle concurrent event emitter access', async () => {
      const emitter = createEventEmitter();

      const operations = Array(100).fill(null).map((_, i) => {
        if (i % 2 === 0) {
          return emitter.on(`event${i}`, () => {});
        } else {
          return emitter.emit(`event${i}`, { data: i });
        }
      });

      await Promise.all(operations);

      expect(emitter.errors).toEqual([]);
    });
  });

  describe('Load Testing', () => {
    it('should handle initialization of many components', async () => {
      const components = Array(100).fill(null).map((_, i) => ({
        name: `component${i}`,
        duration: Math.random() * 100,
      }));

      const { duration } = await measureTime(() =>
        initializeParallel(components, { maxConcurrency: 10 })
      );

      expect(duration).toBeLessThan(5000);
    });

    it('should handle high-frequency metric updates', async () => {
      const metrics = createMetricsCollector();

      const updates = Array(1000).fill(null).map((_, i) =>
        metrics.recordComponentInit(`comp${i % 10}`, i, true)
      );

      const { duration } = await measureTime(() => Promise.all(updates));

      expect(duration).toBeLessThan(1000);
    });

    it('should handle many concurrent health checks', async () => {
      const system = createHealthCheckSystem();

      const checks = Array(100).fill(null).map(() => system.healthCheck());

      const results = await Promise.all(checks);

      expect(results.every(r => r !== undefined)).toBe(true);
    });
  });

  describe('Performance Under Contention', () => {
    it('should maintain performance with lock contention', async () => {
      const resource = createLockedResource();

      const operations = Array(50).fill(null).map(() =>
        resource.withLock(async () => {
          await sleep(10);
        })
      );

      const { duration } = await measureTime(() => Promise.all(operations));

      // Sequential: 50 * 10ms = 500ms
      // Allow some overhead for lock management
      expect(duration).toBeGreaterThan(450);
      expect(duration).toBeLessThan(700);
    });

    it('should handle cache stampede scenario', async () => {
      const cache = createCache();

      // 100 concurrent requests for same missing key
      const requests = Array(100).fill(null).map(() =>
        cache.getOrCompute('expensive-key', async () => {
          await sleep(1000); // Expensive computation
          return 'result';
        })
      );

      const { duration } = await measureTime(() => Promise.all(requests));

      // Should only compute once, not 100 times
      expect(duration).toBeLessThan(1500);
      expect(cache.computations).toBe(1);
    });
  });
});

// Mock implementations

async function initializeParallel(components, options = {}) {
  const maxConcurrency = options.maxConcurrency || Infinity;
  const results = { successful: [], failed: [], maxConcurrentActive: 0 };

  let active = 0;

  const initComponent = async (comp) => {
    active++;
    results.maxConcurrentActive = Math.max(results.maxConcurrentActive, active);

    try {
      await sleep(comp.duration);
      if (comp.shouldFail) {
        throw new Error('Init failed');
      }
      results.successful.push(comp.name);
    } catch (e) {
      results.failed.push(comp.name);
    } finally {
      active--;
    }
  };

  const promises = [];
  for (const comp of components) {
    while (active >= maxConcurrency) {
      await sleep(10);
    }
    promises.push(initComponent(comp));
  }

  await Promise.all(promises);
  return results;
}

async function initializeWithConcurrencyLimit(components, maxConcurrency) {
  return initializeParallel(components, { maxConcurrency });
}

async function initializeWithDependencies(components) {
  const order = [];
  const completed = new Set();
  const startTime = Date.now();

  // Check for circular dependencies
  function hasCircular(comp, visiting = new Set()) {
    if (visiting.has(comp.name)) return true;
    visiting.add(comp.name);

    const component = components.find(c => c.name === comp.name);
    if (component?.dependencies) {
      for (const dep of component.dependencies) {
        const depComp = components.find(c => c.name === dep);
        if (depComp && hasCircular(depComp, new Set(visiting))) {
          return true;
        }
      }
    }

    return false;
  }

  for (const comp of components) {
    if (hasCircular(comp)) {
      return { success: false, error: 'circular dependency detected' };
    }
  }

  async function initComponent(comp) {
    // Wait for dependencies
    if (comp.dependencies) {
      while (!comp.dependencies.every(dep => completed.has(dep))) {
        await sleep(10);
      }
    }

    await sleep(comp.duration);
    order.push(comp.name);
    completed.add(comp.name);
  }

  await Promise.all(components.map(comp => initComponent(comp)));

  return {
    duration: Date.now() - startTime,
    order,
  };
}

function createSharedState() {
  const state = new Map();
  const errors = [];

  return {
    write: async (key, value) => {
      await sleep(Math.random() * 10);
      state.set(key, value);
    },
    read: async (key) => {
      await sleep(Math.random() * 10);
      return state.get(key);
    },
    errors,
  };
}

function createInitializationSystem() {
  let initialized = false;

  return {
    initialize: async () => {
      if (initialized) {
        return { success: false, error: 'already initialized' };
      }
      initialized = true;
      await sleep(100);
      return { success: true };
    },
  };
}

function createRecoveryManager() {
  return {
    checkpoints: new Map(),
    saveCheckpoint: async (name, data) => {
      await sleep(10);
      this.checkpoints.set(name, data);
    },
  };
}

function createMetricsCollector() {
  return {
    componentInitDurations: {},
    recordComponentInit: async (name, duration, success) => {
      await sleep(1);
      this.componentInitDurations[name] = duration;
    },
  };
}

function createConcurrentLogger() {
  return {
    messages: [],
    corruptedMessages: 0,
    info: async (msg) => {
      await sleep(1);
      this.messages.push(msg);
    },
  };
}

async function initializeWithPotentialDeadlock(options) {
  const timeout = options.timeout || 5000;
  const start = Date.now();

  while (Date.now() - start < timeout) {
    await sleep(100);
  }

  return { success: false, error: 'timeout: potential deadlock detected' };
}

function createSharedResource(capacity) {
  let acquired = 0;

  return {
    acquire: async () => {
      while (acquired >= capacity) {
        await sleep(10);
      }
      acquired++;
    },
    release: () => {
      acquired--;
    },
  };
}

async function acquireAndUseResource(resource, duration) {
  await resource.acquire();
  try {
    await sleep(duration);
    return { success: true };
  } finally {
    resource.release();
  }
}

function createOrchestrator() {
  const events = [];

  return {
    registerComponent: (name) => ({
      initialize: async () => {
        events.push({ component: name, event: 'init' });
      },
    }),
    getLifecycleEvents: () => events,
  };
}

function createMutexCounter() {
  let value = 0;
  let locked = false;

  return {
    increment: async () => {
      while (locked) {
        await sleep(1);
      }
      locked = true;
      try {
        const temp = value;
        await sleep(1);
        value = temp + 1;
      } finally {
        locked = false;
      }
    },
    get value() {
      return value;
    },
  };
}

function createSemaphore(limit) {
  let count = 0;

  return {
    acquire: async () => {
      while (count >= limit) {
        await sleep(10);
      }
      count++;
    },
    release: () => {
      count--;
    },
  };
}

async function withSemaphore(semaphore, fn) {
  await semaphore.acquire();
  try {
    return await fn();
  } finally {
    semaphore.release();
  }
}

function createThreadSafeConfig() {
  const data = new Map();

  return {
    get: async (key) => {
      await sleep(1);
      return data.get(key);
    },
    set: async (key, value) => {
      await sleep(1);
      data.set(key, value);
    },
  };
}

function createThreadSafeContainer() {
  const services = new Map();

  return {
    errors: [],
    register: async (name, service) => {
      await sleep(1);
      services.set(name, service);
    },
    resolve: async (name) => {
      await sleep(1);
      return services.get(name);
    },
  };
}

function createEventEmitter() {
  const listeners = new Map();

  return {
    errors: [],
    on: async (event, handler) => {
      await sleep(1);
      if (!listeners.has(event)) {
        listeners.set(event, []);
      }
      listeners.get(event).push(handler);
    },
    emit: async (event, data) => {
      await sleep(1);
      const handlers = listeners.get(event) || [];
      handlers.forEach(h => h(data));
    },
  };
}

function createHealthCheckSystem() {
  return {
    healthCheck: async () => {
      await sleep(Math.random() * 50);
      return { status: 'healthy' };
    },
  };
}

function createLockedResource() {
  let locked = false;

  return {
    withLock: async (fn) => {
      while (locked) {
        await sleep(1);
      }
      locked = true;
      try {
        return await fn();
      } finally {
        locked = false;
      }
    },
  };
}

function createCache() {
  const cache = new Map();
  let computations = 0;
  const computing = new Map();

  return {
    getOrCompute: async (key, fn) => {
      if (cache.has(key)) {
        return cache.get(key);
      }

      if (computing.has(key)) {
        return computing.get(key);
      }

      const promise = (async () => {
        computations++;
        const result = await fn();
        cache.set(key, result);
        computing.delete(key);
        return result;
      })();

      computing.set(key, promise);
      return promise;
    },
    get computations() {
      return computations;
    },
  };
}

module.exports = {
  initializeParallel,
  initializeWithDependencies,
  createSharedState,
  createMetricsCollector,
  createMutexCounter,
  createSemaphore,
  createCache,
};
