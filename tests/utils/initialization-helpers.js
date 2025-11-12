/**
 * Test Utilities and Helpers for Initialization Tests
 * Provides mocks, fixtures, and utility functions
 */

/**
 * Mock Security Orchestrator
 */
class MockSecurityOrchestrator {
  constructor(config = {}) {
    this.config = config;
    this.components = new Map();
    this.metrics = {};
    this.health = {
      overallHealth: 'healthy',
      components: {},
      healthScore: 100,
      criticalIssues: [],
      warnings: [],
      lastHealthCheck: new Date(),
    };
  }

  GetHealthStatus() {
    return this.health;
  }

  GetSecurityMetrics() {
    return this.metrics;
  }

  GetMiddlewares() {
    return [
      (req, res, next) => next(),
      (req, res, next) => next(),
      (req, res, next) => next(),
    ];
  }

  setHealth(health) {
    this.health = { ...this.health, ...health };
  }

  setMetrics(metrics) {
    this.metrics = { ...this.metrics, ...metrics };
  }
}

/**
 * Mock Module Loader
 */
class MockModuleLoader {
  constructor() {
    this.modules = new Map();
    this.loadOrder = [];
  }

  register(module) {
    this.modules.set(module.name, module);
  }

  async load(name) {
    this.loadOrder.push(name);
    const module = this.modules.get(name);
    if (module && module.initialize) {
      await module.initialize();
    }
    return module;
  }

  getLoadOrder() {
    return this.loadOrder;
  }
}

/**
 * Test Fixtures
 */
const fixtures = {
  validConfig: {
    environment: 'test',
    database: {
      host: 'localhost',
      port: 5432,
      name: 'test_db',
      username: 'test_user',
      password: 'test_pass',
    },
    cache: {
      type: 'redis',
      host: 'localhost',
      port: 6379,
    },
    security: {
      zeroTrust: true,
      mfaEnabled: true,
      encryption: {
        algorithm: 'AES-256-GCM',
        keyRotationDays: 90,
      },
    },
    api: {
      port: 3000,
      cors: true,
      rateLimit: {
        maxRequests: 1000,
        windowMs: 60000,
      },
    },
  },

  invalidConfig: {
    // Missing required fields
    environment: 'test',
  },

  module: {
    name: 'test-module',
    dependencies: [],
    initialize: jest.fn().mockResolvedValue(true),
    shutdown: jest.fn().mockResolvedValue(true),
  },

  moduleWithDeps: {
    name: 'dependent-module',
    dependencies: ['test-module'],
    initialize: jest.fn().mockResolvedValue(true),
    shutdown: jest.fn().mockResolvedValue(true),
  },
};

/**
 * Utility Functions
 */

/**
 * Create a mock database connection
 */
function createMockDatabase() {
  return {
    connected: false,
    connections: [],
    connect: jest.fn().mockImplementation(async function() {
      this.connected = true;
      return true;
    }),
    disconnect: jest.fn().mockImplementation(async function() {
      this.connected = false;
      return true;
    }),
    query: jest.fn().mockResolvedValue({ rows: [] }),
    transaction: jest.fn().mockImplementation(() => ({
      query: jest.fn().mockResolvedValue({ rows: [] }),
      commit: jest.fn().mockResolvedValue(true),
      rollback: jest.fn().mockResolvedValue(true),
    })),
  };
}

/**
 * Create a mock cache client
 */
function createMockCache() {
  const cache = new Map();

  return {
    connected: false,
    connect: jest.fn().mockImplementation(async function() {
      this.connected = true;
      return true;
    }),
    disconnect: jest.fn().mockImplementation(async function() {
      this.connected = false;
      cache.clear();
      return true;
    }),
    get: jest.fn().mockImplementation((key) => {
      return Promise.resolve(cache.get(key));
    }),
    set: jest.fn().mockImplementation((key, value) => {
      cache.set(key, value);
      return Promise.resolve('OK');
    }),
    del: jest.fn().mockImplementation((key) => {
      cache.delete(key);
      return Promise.resolve(1);
    }),
    quit: jest.fn().mockImplementation(async function() {
      this.connected = false;
      cache.clear();
      return true;
    }),
  };
}

/**
 * Create a mock logger
 */
function createMockLogger() {
  return {
    debug: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    log: jest.fn(),
  };
}

/**
 * Create a mock vault client
 */
function createMockVault() {
  const secrets = new Map();

  return {
    connected: false,
    connect: jest.fn().mockImplementation(async function() {
      this.connected = true;
      return true;
    }),
    disconnect: jest.fn().mockImplementation(async function() {
      this.connected = false;
      return true;
    }),
    getSecret: jest.fn().mockImplementation((path) => {
      return Promise.resolve(secrets.get(path));
    }),
    setSecret: jest.fn().mockImplementation((path, value) => {
      secrets.set(path, value);
      return Promise.resolve(true);
    }),
    deleteSecret: jest.fn().mockImplementation((path) => {
      secrets.delete(path);
      return Promise.resolve(true);
    }),
  };
}

/**
 * Wait for condition to be true
 */
async function waitForCondition(condition, timeout = 5000, interval = 100) {
  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    if (await condition()) {
      return true;
    }
    await sleep(interval);
  }

  throw new Error('Condition not met within timeout');
}

/**
 * Sleep utility
 */
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Measure execution time
 */
async function measureTime(fn) {
  const startTime = Date.now();
  const result = await fn();
  const duration = Date.now() - startTime;

  return {
    result,
    duration,
  };
}

/**
 * Measure memory usage
 */
function measureMemory(fn) {
  const startMemory = process.memoryUsage().heapUsed;
  const result = fn();
  const endMemory = process.memoryUsage().heapUsed;

  return {
    result,
    memoryDelta: endMemory - startMemory,
    memoryDeltaMB: (endMemory - startMemory) / 1024 / 1024,
  };
}

/**
 * Generate random string
 */
function randomString(length = 10) {
  return Math.random().toString(36).substring(2, 2 + length);
}

/**
 * Generate random port
 */
function randomPort(min = 3000, max = 9000) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

/**
 * Create test context
 */
function createTestContext(overrides = {}) {
  return {
    environment: 'test',
    testId: randomString(8),
    startTime: Date.now(),
    logger: createMockLogger(),
    ...overrides,
  };
}

/**
 * Assert module loaded
 */
function assertModuleLoaded(modules, moduleName) {
  const module = modules.find(m => m.name === moduleName);
  expect(module).toBeDefined();
  expect(module.loaded).toBe(true);
  return module;
}

/**
 * Assert initialization order
 */
function assertInitializationOrder(actualOrder, expectedOrder) {
  expect(actualOrder).toEqual(expectedOrder);

  // Verify each module was initialized after its dependencies
  actualOrder.forEach((module, index) => {
    const dependencies = module.dependencies || [];
    dependencies.forEach(dep => {
      const depIndex = actualOrder.findIndex(m => m.name === dep);
      expect(depIndex).toBeLessThan(index);
    });
  });
}

/**
 * Assert cleanup occurred
 */
async function assertCleanupOccurred(resources) {
  if (resources.database) {
    expect(resources.database.disconnect).toHaveBeenCalled();
  }

  if (resources.cache) {
    expect(resources.cache.quit).toHaveBeenCalled();
  }

  if (resources.vault) {
    expect(resources.vault.disconnect).toHaveBeenCalled();
  }
}

/**
 * Create retry configuration
 */
function createRetryConfig(overrides = {}) {
  return {
    maxRetries: 3,
    initialDelay: 100,
    maxDelay: 5000,
    backoff: 'exponential',
    ...overrides,
  };
}

/**
 * Mock process signals
 */
function mockProcessSignals() {
  const originalListeners = {
    SIGTERM: process.listeners('SIGTERM'),
    SIGINT: process.listeners('SIGINT'),
  };

  process.removeAllListeners('SIGTERM');
  process.removeAllListeners('SIGINT');

  return {
    restore: () => {
      process.removeAllListeners('SIGTERM');
      process.removeAllListeners('SIGINT');
      originalListeners.SIGTERM.forEach(l => process.on('SIGTERM', l));
      originalListeners.SIGINT.forEach(l => process.on('SIGINT', l));
    },
  };
}

/**
 * Create initialization metrics
 */
function createMetrics() {
  return {
    totalDuration: 0,
    phases: [],
    modules: [],
    errors: [],

    recordPhase(name, duration) {
      this.phases.push({ name, duration, timestamp: Date.now() });
    },

    recordModule(name, duration, success) {
      this.modules.push({ name, duration, success, timestamp: Date.now() });
    },

    recordError(phase, error) {
      this.errors.push({ phase, error: error.message, timestamp: Date.now() });
    },

    getSummary() {
      return {
        totalDuration: this.totalDuration,
        totalPhases: this.phases.length,
        totalModules: this.modules.length,
        successfulModules: this.modules.filter(m => m.success).length,
        failedModules: this.modules.filter(m => !m.success).length,
        totalErrors: this.errors.length,
      };
    },
  };
}

module.exports = {
  MockSecurityOrchestrator,
  MockModuleLoader,
  fixtures,
  createMockDatabase,
  createMockCache,
  createMockLogger,
  createMockVault,
  waitForCondition,
  sleep,
  measureTime,
  measureMemory,
  randomString,
  randomPort,
  createTestContext,
  assertModuleLoaded,
  assertInitializationOrder,
  assertCleanupOccurred,
  createRetryConfig,
  mockProcessSignals,
  createMetrics,
};
