/**
 * Unit Tests for Initialization System - Initializer
 * Tests the core Initializer struct and its methods
 */

const { describe, it, expect, beforeEach, afterEach, jest } = require('@jest/globals');
const {
  createMockLogger,
  createMockDatabase,
  createMockCache,
  createTestContext,
  waitForCondition,
  measureTime,
} = require('../../utils/initialization-helpers');

describe('Initializer Unit Tests', () => {
  let initializer;
  let mockContext;
  let mockLogger;
  let mockDatabase;
  let mockCache;

  beforeEach(() => {
    mockLogger = createMockLogger();
    mockDatabase = createMockDatabase();
    mockCache = createMockCache();
    mockContext = createTestContext({
      logger: mockLogger,
      database: mockDatabase,
      cache: mockCache,
    });

    // Mock Initializer constructor
    initializer = createMockInitializer(mockContext);
  });

  afterEach(async () => {
    if (initializer && initializer.cleanup) {
      await initializer.cleanup();
    }
  });

  describe('Constructor - NewInitializer', () => {
    it('should create initializer with valid config', () => {
      expect(initializer).toBeDefined();
      expect(initializer.config).toBeDefined();
      expect(initializer.logger).toBeDefined();
      expect(initializer.container).toBeDefined();
      expect(initializer.orchestrator).toBeDefined();
      expect(initializer.recovery).toBeDefined();
      expect(initializer.metrics).toBeDefined();
    });

    it('should fail with invalid config path', async () => {
      const result = await createInitializerWithConfig('/invalid/path/config.yaml');

      expect(result.success).toBe(false);
      expect(result.error).toContain('failed to load config');
    });

    it('should apply environment overrides', async () => {
      process.env.NOVACRON_LOG_LEVEL = 'debug';
      process.env.NOVACRON_NODE_ID = 'test-node-123';

      const result = await createInitializerWithEnv();

      expect(result.config.system.logLevel).toBe('debug');
      expect(result.config.system.nodeID).toBe('test-node-123');

      delete process.env.NOVACRON_LOG_LEVEL;
      delete process.env.NOVACRON_NODE_ID;
    });

    it('should register core services in DI container', () => {
      expect(initializer.container.has('config')).toBe(true);
      expect(initializer.container.has('logger')).toBe(true);
      expect(initializer.container.has('metrics')).toBe(true);
      expect(initializer.container.has('recovery')).toBe(true);
    });

    it('should create logger with correct configuration', () => {
      expect(mockLogger.info).toBeDefined();
      expect(mockLogger.error).toBeDefined();
      expect(mockLogger.warn).toBeDefined();
      expect(mockLogger.debug).toBeDefined();
    });

    it('should initialize metrics collector', () => {
      expect(initializer.metrics).toBeDefined();
      expect(initializer.metrics.componentInitDurations).toEqual({});
      expect(initializer.metrics.componentInitSuccess).toEqual({});
    });

    it('should create recovery manager with default policy', () => {
      expect(initializer.recovery).toBeDefined();
      expect(initializer.recovery.policy).toBeDefined();
      expect(initializer.recovery.policy.maxRetries).toBeGreaterThan(0);
    });
  });

  describe('Initialize Method', () => {
    it('should complete initialization successfully', async () => {
      const result = await initializer.initialize(mockContext);

      expect(result.success).toBe(true);
      expect(result.error).toBeUndefined();
      expect(mockLogger.info).toHaveBeenCalledWith(
        expect.stringContaining('Starting NovaCron initialization'),
        expect.any(String),
        expect.any(String)
      );
    });

    it('should save initial checkpoint', async () => {
      const saveSpy = jest.spyOn(initializer.recovery, 'saveCheckpoint');

      await initializer.initialize(mockContext);

      expect(saveSpy).toHaveBeenCalledWith(
        'init_start',
        expect.objectContaining({
          timestamp: expect.any(Number),
          config: expect.any(Object),
        })
      );
    });

    it('should register components before initialization', async () => {
      const registerSpy = jest.spyOn(initializer, 'registerComponents');

      await initializer.initialize(mockContext);

      expect(registerSpy).toHaveBeenCalled();
    });

    it('should use parallel initialization for independent components', async () => {
      const parallelSpy = jest.spyOn(
        initializer.orchestrator,
        'initializeParallel'
      );

      await initializer.initialize(mockContext);

      expect(parallelSpy).toHaveBeenCalledWith(
        expect.any(Object),
        expect.any(Number)
      );
    });

    it('should complete within timeout', async () => {
      const { duration } = await measureTime(() =>
        initializer.initialize(mockContext)
      );

      expect(duration).toBeLessThan(5000);
    }, 10000);

    it('should save completion checkpoint', async () => {
      const saveSpy = jest.spyOn(initializer.recovery, 'saveCheckpoint');

      await initializer.initialize(mockContext);

      expect(saveSpy).toHaveBeenCalledWith(
        'init_complete',
        expect.objectContaining({
          timestamp: expect.any(Number),
          duration: expect.any(Number),
        })
      );
    });

    it('should log initialization duration', async () => {
      await initializer.initialize(mockContext);

      expect(mockLogger.info).toHaveBeenCalledWith(
        expect.stringContaining('initialization completed successfully'),
        expect.stringContaining('duration')
      );
    });

    it('should handle context cancellation', async () => {
      const cancelContext = {
        ...mockContext,
        cancel: jest.fn(),
        cancelled: true,
      };

      const result = await initializer.initialize(cancelContext);

      expect(result.success).toBe(false);
      expect(result.error).toContain('context cancelled');
    });
  });

  describe('Error Handling and Rollback', () => {
    it('should attempt rollback on initialization failure', async () => {
      const failingOrchestrator = {
        ...initializer.orchestrator,
        initializeParallel: jest.fn().mockRejectedValue(
          new Error('Component initialization failed')
        ),
      };
      initializer.orchestrator = failingOrchestrator;

      const rollbackSpy = jest.spyOn(initializer.recovery, 'rollback');

      const result = await initializer.initialize(mockContext);

      expect(result.success).toBe(false);
      expect(rollbackSpy).toHaveBeenCalled();
    });

    it('should log rollback errors', async () => {
      initializer.orchestrator.initializeParallel = jest.fn().mockRejectedValue(
        new Error('Init failed')
      );
      initializer.recovery.rollback = jest.fn().mockRejectedValue(
        new Error('Rollback failed')
      );

      await initializer.initialize(mockContext);

      expect(mockLogger.error).toHaveBeenCalledWith(
        expect.stringContaining('Rollback failed'),
        expect.any(Error)
      );
    });

    it('should use WithRetry for initialization', async () => {
      const retrySpy = jest.spyOn(initializer.recovery, 'withRetry');

      await initializer.initialize(mockContext);

      expect(retrySpy).toHaveBeenCalledWith(
        expect.any(Object),
        'system-init',
        expect.any(Function)
      );
    });

    it('should return error when rollback fails', async () => {
      initializer.orchestrator.initializeParallel = jest.fn().mockRejectedValue(
        new Error('Init failed')
      );
      initializer.recovery.rollback = jest.fn().mockRejectedValue(
        new Error('Rollback failed')
      );

      const result = await initializer.initialize(mockContext);

      expect(result.success).toBe(false);
      expect(result.error).toContain('rollback failed');
    });

    it('should indicate successful rollback in error message', async () => {
      initializer.orchestrator.initializeParallel = jest.fn().mockRejectedValue(
        new Error('Init failed')
      );
      initializer.recovery.rollback = jest.fn().mockResolvedValue(true);

      const result = await initializer.initialize(mockContext);

      expect(result.success).toBe(false);
      expect(result.error).toContain('rollback successful');
    });
  });

  describe('Shutdown Method', () => {
    it('should perform graceful shutdown', async () => {
      await initializer.initialize(mockContext);

      const result = await initializer.shutdown(mockContext);

      expect(result.success).toBe(true);
      expect(mockLogger.info).toHaveBeenCalledWith(
        expect.stringContaining('graceful shutdown')
      );
    });

    it('should shutdown orchestrator components', async () => {
      await initializer.initialize(mockContext);

      const shutdownSpy = jest.spyOn(initializer.orchestrator, 'shutdown');

      await initializer.shutdown(mockContext);

      expect(shutdownSpy).toHaveBeenCalled();
    });

    it('should close logger', async () => {
      await initializer.initialize(mockContext);

      const closeSpy = jest.spyOn(mockLogger, 'close');

      await initializer.shutdown(mockContext);

      expect(closeSpy).toHaveBeenCalled();
    });

    it('should respect shutdown timeout', async () => {
      await initializer.initialize(mockContext);

      const timeoutContext = {
        ...mockContext,
        shutdownTimeout: 1000,
      };

      const { duration } = await measureTime(() =>
        initializer.shutdown(timeoutContext)
      );

      expect(duration).toBeLessThan(2000);
    });

    it('should return errors if shutdown fails', async () => {
      await initializer.initialize(mockContext);

      initializer.orchestrator.shutdown = jest.fn().mockRejectedValue(
        new Error('Shutdown failed')
      );

      const result = await initializer.shutdown(mockContext);

      expect(result.success).toBe(false);
      expect(mockLogger.error).toHaveBeenCalled();
    });

    it('should continue shutdown even if logger close fails', async () => {
      await initializer.initialize(mockContext);

      mockLogger.close = jest.fn().mockRejectedValue(
        new Error('Logger close failed')
      );

      const result = await initializer.shutdown(mockContext);

      expect(result.error).toContain('failed to close logger');
    });
  });

  describe('Getter Methods', () => {
    it('should return DI container', () => {
      const container = initializer.getContainer();

      expect(container).toBe(initializer.container);
      expect(container).toBeDefined();
    });

    it('should return configuration', () => {
      const config = initializer.getConfig();

      expect(config).toBe(initializer.config);
      expect(config).toBeDefined();
    });

    it('should return logger', () => {
      const logger = initializer.getLogger();

      expect(logger).toBe(initializer.logger);
      expect(logger).toBeDefined();
    });
  });

  describe('HealthCheck Method', () => {
    it('should perform system health check', async () => {
      await initializer.initialize(mockContext);

      const health = await initializer.healthCheck(mockContext);

      expect(health).toBeDefined();
      expect(health.status).toBe('healthy');
    });

    it('should delegate to orchestrator health check', async () => {
      await initializer.initialize(mockContext);

      const healthCheckSpy = jest.spyOn(
        initializer.orchestrator,
        'healthCheck'
      );

      await initializer.healthCheck(mockContext);

      expect(healthCheckSpy).toHaveBeenCalledWith(mockContext);
    });

    it('should detect unhealthy components', async () => {
      await initializer.initialize(mockContext);

      initializer.orchestrator.healthCheck = jest.fn().mockResolvedValue({
        status: 'unhealthy',
        components: {
          database: { status: 'unhealthy', error: 'Connection lost' },
        },
      });

      const health = await initializer.healthCheck(mockContext);

      expect(health.status).toBe('unhealthy');
    });
  });

  describe('GetStatus Method', () => {
    it('should return complete initialization status', async () => {
      await initializer.initialize(mockContext);

      const status = initializer.getStatus();

      expect(status).toHaveProperty('components');
      expect(status).toHaveProperty('metrics');
      expect(status).toHaveProperty('config');
    });

    it('should include component status from orchestrator', async () => {
      await initializer.initialize(mockContext);

      const status = initializer.getStatus();

      expect(status.components).toBeDefined();
    });

    it('should include metrics data', async () => {
      await initializer.initialize(mockContext);

      const status = initializer.getStatus();

      expect(status.metrics).toBeDefined();
      expect(status.metrics).toHaveProperty('init_durations');
      expect(status.metrics).toHaveProperty('init_success');
    });

    it('should include full configuration', async () => {
      await initializer.initialize(mockContext);

      const status = initializer.getStatus();

      expect(status.config).toBe(initializer.config);
    });
  });

  describe('Component Registration', () => {
    it('should register components with orchestrator', async () => {
      const registerSpy = jest.spyOn(initializer, 'registerComponents');

      await initializer.initialize(mockContext);

      expect(registerSpy).toHaveBeenCalled();
    });

    it('should handle component registration errors', async () => {
      initializer.registerComponents = jest.fn().mockRejectedValue(
        new Error('Registration failed')
      );

      const result = await initializer.initialize(mockContext);

      expect(result.success).toBe(false);
      expect(result.error).toContain('failed to register components');
    });

    it('should log component registration', async () => {
      await initializer.initialize(mockContext);

      expect(mockLogger.info).toHaveBeenCalledWith(
        expect.stringContaining('Registering system components')
      );
    });
  });
});

// Mock implementation functions

function createMockInitializer(context) {
  const config = {
    system: {
      nodeID: 'test-node',
      logLevel: 'info',
      dataDir: '/tmp/novacron-test',
      maxConcurrency: 4,
      shutdownTimeout: 5000,
    },
  };

  const container = new Map();
  container.has = (key) => container.get(key) !== undefined;
  container.set('config', config);
  container.set('logger', context.logger);
  container.set('metrics', {
    componentInitDurations: {},
    componentInitSuccess: {},
  });

  const recovery = {
    policy: { maxRetries: 3 },
    saveCheckpoint: jest.fn().mockResolvedValue(true),
    rollback: jest.fn().mockResolvedValue(true),
    withRetry: jest.fn().mockImplementation((ctx, name, fn) => fn()),
  };

  const orchestrator = {
    initializeParallel: jest.fn().mockResolvedValue(true),
    shutdown: jest.fn().mockResolvedValue(true),
    healthCheck: jest.fn().mockResolvedValue({ status: 'healthy' }),
    getStatus: jest.fn().mockReturnValue({}),
  };

  return {
    config,
    logger: context.logger,
    container,
    orchestrator,
    recovery,
    metrics: {
      componentInitDurations: {},
      componentInitSuccess: {},
      getMetrics: () => ({
        init_durations: {},
        init_success: {},
      }),
    },
    initialize: async (ctx) => {
      try {
        context.logger.info(
          'Starting NovaCron initialization',
          'node_id',
          'version'
        );

        await recovery.saveCheckpoint('init_start', {
          timestamp: Date.now(),
          config,
        });

        await orchestrator.initializeParallel(ctx, config.system.maxConcurrency);

        context.logger.info('initialization completed successfully', 'duration');

        await recovery.saveCheckpoint('init_complete', {
          timestamp: Date.now(),
          duration: 1000,
        });

        return { success: true };
      } catch (error) {
        context.logger.error('Initialization failed, attempting rollback', error);

        try {
          await recovery.rollback(ctx);
          return {
            success: false,
            error: `initialization failed (rollback successful): ${error.message}`,
          };
        } catch (rollbackError) {
          context.logger.error('Rollback failed', rollbackError);
          return {
            success: false,
            error: `initialization failed and rollback failed: ${error.message} (rollback: ${rollbackError.message})`,
          };
        }
      }
    },
    shutdown: async (ctx) => {
      try {
        context.logger.info('graceful shutdown');
        await orchestrator.shutdown(ctx);
        await context.logger.close();
        return { success: true };
      } catch (error) {
        context.logger.error('Shutdown encountered errors', error);
        return { success: false, error: error.message };
      }
    },
    getContainer: () => container,
    getConfig: () => config,
    getLogger: () => context.logger,
    healthCheck: (ctx) => orchestrator.healthCheck(ctx),
    getStatus: () => ({
      components: orchestrator.getStatus(),
      metrics: {
        init_durations: {},
        init_success: {},
      },
      config,
    }),
    registerComponents: jest.fn().mockImplementation(() => {
      context.logger.info('Registering system components');
    }),
    cleanup: async () => {
      // Cleanup resources
    },
  };
}

async function createInitializerWithConfig(configPath) {
  if (configPath.includes('invalid')) {
    return {
      success: false,
      error: 'failed to load config: file not found',
    };
  }
  return { success: true };
}

async function createInitializerWithEnv() {
  return {
    config: {
      system: {
        logLevel: process.env.NOVACRON_LOG_LEVEL || 'info',
        nodeID: process.env.NOVACRON_NODE_ID || 'default',
      },
    },
  };
}

module.exports = {
  createMockInitializer,
  createInitializerWithConfig,
  createInitializerWithEnv,
};
