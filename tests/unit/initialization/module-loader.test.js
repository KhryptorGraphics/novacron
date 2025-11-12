/**
 * Unit Tests for Module Loader
 * Tests dynamic module loading, dependency resolution, and initialization order
 */

const { describe, it, expect, beforeEach, afterEach, jest } = require('@jest/globals');

describe('Module Loader', () => {

  let moduleLoader;
  let mockModules;

  beforeEach(() => {
    mockModules = {
      core: {
        name: 'core',
        dependencies: [],
        initialize: jest.fn().mockResolvedValue(true),
      },
      database: {
        name: 'database',
        dependencies: ['core'],
        initialize: jest.fn().mockResolvedValue(true),
      },
      api: {
        name: 'api',
        dependencies: ['core', 'database'],
        initialize: jest.fn().mockResolvedValue(true),
      },
    };

    moduleLoader = new ModuleLoader();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Module Registration', () => {

    it('should register module successfully', () => {
      const result = moduleLoader.register(mockModules.core);

      expect(result.success).toBe(true);
      expect(moduleLoader.getModule('core')).toBeDefined();
    });

    it('should reject duplicate module registration', () => {
      moduleLoader.register(mockModules.core);

      expect(() => moduleLoader.register(mockModules.core))
        .toThrow('Module core already registered');
    });

    it('should validate module structure', () => {
      const invalidModule = {
        name: 'invalid',
        // Missing initialize function
      };

      expect(() => moduleLoader.register(invalidModule))
        .toThrow('Invalid module structure');
    });
  });

  describe('Dependency Resolution', () => {

    it('should resolve dependencies in correct order', () => {
      moduleLoader.register(mockModules.core);
      moduleLoader.register(mockModules.database);
      moduleLoader.register(mockModules.api);

      const order = moduleLoader.resolveDependencies();

      expect(order).toEqual(['core', 'database', 'api']);
    });

    it('should detect circular dependencies', () => {
      const circular1 = {
        name: 'module1',
        dependencies: ['module2'],
        initialize: jest.fn(),
      };

      const circular2 = {
        name: 'module2',
        dependencies: ['module1'],
        initialize: jest.fn(),
      };

      moduleLoader.register(circular1);
      moduleLoader.register(circular2);

      expect(() => moduleLoader.resolveDependencies())
        .toThrow('Circular dependency detected');
    });

    it('should detect missing dependencies', () => {
      moduleLoader.register(mockModules.database);

      expect(() => moduleLoader.resolveDependencies())
        .toThrow('Missing dependency: core');
    });

    it('should handle complex dependency graphs', () => {
      const modules = {
        a: { name: 'a', dependencies: [], initialize: jest.fn() },
        b: { name: 'b', dependencies: ['a'], initialize: jest.fn() },
        c: { name: 'c', dependencies: ['a'], initialize: jest.fn() },
        d: { name: 'd', dependencies: ['b', 'c'], initialize: jest.fn() },
        e: { name: 'e', dependencies: ['d'], initialize: jest.fn() },
      };

      Object.values(modules).forEach(m => moduleLoader.register(m));

      const order = moduleLoader.resolveDependencies();

      expect(order[0]).toBe('a');
      expect(order[order.length - 1]).toBe('e');
      expect(order.indexOf('b')).toBeGreaterThan(order.indexOf('a'));
      expect(order.indexOf('c')).toBeGreaterThan(order.indexOf('a'));
      expect(order.indexOf('d')).toBeGreaterThan(order.indexOf('b'));
      expect(order.indexOf('d')).toBeGreaterThan(order.indexOf('c'));
    });
  });

  describe('Module Initialization', () => {

    it('should initialize modules in dependency order', async () => {
      moduleLoader.register(mockModules.core);
      moduleLoader.register(mockModules.database);
      moduleLoader.register(mockModules.api);

      await moduleLoader.initializeAll();

      // Verify initialization order
      const coreCallOrder = mockModules.core.initialize.mock.invocationCallOrder[0];
      const dbCallOrder = mockModules.database.initialize.mock.invocationCallOrder[0];
      const apiCallOrder = mockModules.api.initialize.mock.invocationCallOrder[0];

      expect(coreCallOrder).toBeLessThan(dbCallOrder);
      expect(dbCallOrder).toBeLessThan(apiCallOrder);
    });

    it('should pass context to module initialization', async () => {
      moduleLoader.register(mockModules.core);

      const context = { environment: 'test', config: {} };
      await moduleLoader.initializeAll(context);

      expect(mockModules.core.initialize).toHaveBeenCalledWith(
        expect.objectContaining(context)
      );
    });

    it('should stop initialization on module failure', async () => {
      mockModules.database.initialize.mockRejectedValue(
        new Error('Database connection failed')
      );

      moduleLoader.register(mockModules.core);
      moduleLoader.register(mockModules.database);
      moduleLoader.register(mockModules.api);

      await expect(moduleLoader.initializeAll())
        .rejects.toThrow('Database connection failed');

      expect(mockModules.core.initialize).toHaveBeenCalled();
      expect(mockModules.database.initialize).toHaveBeenCalled();
      expect(mockModules.api.initialize).not.toHaveBeenCalled();
    });

    it('should track initialization progress', async () => {
      const progressCallback = jest.fn();

      moduleLoader.register(mockModules.core);
      moduleLoader.register(mockModules.database);
      moduleLoader.register(mockModules.api);

      await moduleLoader.initializeAll({}, progressCallback);

      expect(progressCallback).toHaveBeenCalledTimes(3);
      expect(progressCallback).toHaveBeenCalledWith({
        module: 'core',
        status: 'completed',
        progress: 33.33,
      });
    });
  });

  describe('Module Lifecycle', () => {

    it('should support module shutdown', async () => {
      const moduleWithShutdown = {
        name: 'lifecycle',
        dependencies: [],
        initialize: jest.fn().mockResolvedValue(true),
        shutdown: jest.fn().mockResolvedValue(true),
      };

      moduleLoader.register(moduleWithShutdown);
      await moduleLoader.initializeAll();

      await moduleLoader.shutdown('lifecycle');

      expect(moduleWithShutdown.shutdown).toHaveBeenCalled();
    });

    it('should shutdown modules in reverse dependency order', async () => {
      const shutdownOrder = [];

      Object.values(mockModules).forEach(m => {
        m.shutdown = jest.fn().mockImplementation(() => {
          shutdownOrder.push(m.name);
          return Promise.resolve(true);
        });
      });

      moduleLoader.register(mockModules.core);
      moduleLoader.register(mockModules.database);
      moduleLoader.register(mockModules.api);

      await moduleLoader.initializeAll();
      await moduleLoader.shutdownAll();

      expect(shutdownOrder).toEqual(['api', 'database', 'core']);
    });

    it('should continue shutdown even if module fails', async () => {
      mockModules.database.shutdown = jest.fn().mockRejectedValue(
        new Error('Shutdown error')
      );
      mockModules.core.shutdown = jest.fn().mockResolvedValue(true);
      mockModules.api.shutdown = jest.fn().mockResolvedValue(true);

      moduleLoader.register(mockModules.core);
      moduleLoader.register(mockModules.database);
      moduleLoader.register(mockModules.api);

      await moduleLoader.initializeAll();
      const errors = await moduleLoader.shutdownAll();

      expect(errors).toHaveLength(1);
      expect(errors[0].module).toBe('database');
      expect(mockModules.core.shutdown).toHaveBeenCalled();
      expect(mockModules.api.shutdown).toHaveBeenCalled();
    });
  });

  describe('Module Status', () => {

    it('should track module initialization status', async () => {
      moduleLoader.register(mockModules.core);

      expect(moduleLoader.getStatus('core')).toBe('registered');

      await moduleLoader.initialize('core');

      expect(moduleLoader.getStatus('core')).toBe('initialized');
    });

    it('should provide detailed module information', () => {
      moduleLoader.register(mockModules.database);

      const info = moduleLoader.getModuleInfo('database');

      expect(info.name).toBe('database');
      expect(info.dependencies).toEqual(['core']);
      expect(info.status).toBe('registered');
      expect(info.registeredAt).toBeDefined();
    });

    it('should list all registered modules', () => {
      moduleLoader.register(mockModules.core);
      moduleLoader.register(mockModules.database);

      const modules = moduleLoader.listModules();

      expect(modules).toHaveLength(2);
      expect(modules.map(m => m.name)).toEqual(['core', 'database']);
    });
  });

  describe('Error Recovery', () => {

    it('should retry failed module initialization', async () => {
      let attempts = 0;
      const flakeyModule = {
        name: 'flakey',
        dependencies: [],
        initialize: jest.fn().mockImplementation(() => {
          attempts++;
          if (attempts < 3) {
            return Promise.reject(new Error('Temporary failure'));
          }
          return Promise.resolve(true);
        }),
      };

      moduleLoader.register(flakeyModule);

      await moduleLoader.initializeAll({ retries: 3 });

      expect(flakeyModule.initialize).toHaveBeenCalledTimes(3);
      expect(moduleLoader.getStatus('flakey')).toBe('initialized');
    });

    it('should rollback on initialization failure', async () => {
      mockModules.core.rollback = jest.fn().mockResolvedValue(true);
      mockModules.database.rollback = jest.fn().mockResolvedValue(true);

      mockModules.api.initialize.mockRejectedValue(
        new Error('Initialization failed')
      );

      moduleLoader.register(mockModules.core);
      moduleLoader.register(mockModules.database);
      moduleLoader.register(mockModules.api);

      try {
        await moduleLoader.initializeAll({ rollbackOnFailure: true });
      } catch (e) {
        // Expected to fail
      }

      expect(mockModules.database.rollback).toHaveBeenCalled();
      expect(mockModules.core.rollback).toHaveBeenCalled();
    });
  });

  describe('Performance', () => {

    it('should initialize modules within timeout', async () => {
      const slowModule = {
        name: 'slow',
        dependencies: [],
        initialize: jest.fn().mockImplementation(() =>
          new Promise(resolve => setTimeout(resolve, 100))
        ),
      };

      moduleLoader.register(slowModule);

      const startTime = Date.now();
      await moduleLoader.initializeAll({ timeout: 5000 });
      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(5000);
    });

    it('should support parallel initialization of independent modules', async () => {
      const module1 = {
        name: 'parallel1',
        dependencies: [],
        initialize: jest.fn().mockImplementation(() =>
          new Promise(resolve => setTimeout(resolve, 100))
        ),
      };

      const module2 = {
        name: 'parallel2',
        dependencies: [],
        initialize: jest.fn().mockImplementation(() =>
          new Promise(resolve => setTimeout(resolve, 100))
        ),
      };

      moduleLoader.register(module1);
      moduleLoader.register(module2);

      const startTime = Date.now();
      await moduleLoader.initializeAll({ parallel: true });
      const duration = Date.now() - startTime;

      // Should take ~100ms, not 200ms
      expect(duration).toBeLessThan(150);
    });
  });
});

// Mock ModuleLoader implementation
class ModuleLoader {
  constructor() {
    this.modules = new Map();
    this.statuses = new Map();
  }

  register(module) {
    if (this.modules.has(module.name)) {
      throw new Error(`Module ${module.name} already registered`);
    }

    if (!module.initialize || typeof module.initialize !== 'function') {
      throw new Error('Invalid module structure');
    }

    this.modules.set(module.name, module);
    this.statuses.set(module.name, 'registered');

    return { success: true };
  }

  getModule(name) {
    return this.modules.get(name);
  }

  resolveDependencies() {
    const order = [];
    const visited = new Set();
    const visiting = new Set();

    const visit = (name) => {
      if (visited.has(name)) return;
      if (visiting.has(name)) {
        throw new Error('Circular dependency detected');
      }

      const module = this.modules.get(name);
      if (!module) {
        throw new Error(`Missing dependency: ${name}`);
      }

      visiting.add(name);

      module.dependencies.forEach(dep => visit(dep));

      visiting.delete(name);
      visited.add(name);
      order.push(name);
    };

    this.modules.forEach((_, name) => visit(name));

    return order;
  }

  async initializeAll(context = {}, progressCallback) {
    const order = this.resolveDependencies();
    const total = order.length;

    for (let i = 0; i < order.length; i++) {
      const name = order[i];
      const module = this.modules.get(name);

      await module.initialize(context);
      this.statuses.set(name, 'initialized');

      if (progressCallback) {
        progressCallback({
          module: name,
          status: 'completed',
          progress: ((i + 1) / total) * 100,
        });
      }
    }
  }

  async initialize(name, context = {}) {
    const module = this.modules.get(name);
    await module.initialize(context);
    this.statuses.set(name, 'initialized');
  }

  async shutdownAll() {
    const order = this.resolveDependencies().reverse();
    const errors = [];

    for (const name of order) {
      const module = this.modules.get(name);
      if (module.shutdown) {
        try {
          await module.shutdown();
        } catch (e) {
          errors.push({ module: name, error: e });
        }
      }
    }

    return errors;
  }

  async shutdown(name) {
    const module = this.modules.get(name);
    if (module.shutdown) {
      await module.shutdown();
    }
  }

  getStatus(name) {
    return this.statuses.get(name);
  }

  getModuleInfo(name) {
    const module = this.modules.get(name);
    return {
      name: module.name,
      dependencies: module.dependencies,
      status: this.statuses.get(name),
      registeredAt: new Date(),
    };
  }

  listModules() {
    return Array.from(this.modules.values()).map(m => ({
      name: m.name,
      status: this.statuses.get(m.name),
    }));
  }
}

module.exports = { ModuleLoader };
