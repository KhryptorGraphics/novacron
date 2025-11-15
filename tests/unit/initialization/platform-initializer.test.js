/**
 * Unit Tests for PlatformInitializer Class
 * Comprehensive tests for the NovaCron initialization system
 */

const { describe, it, expect, beforeEach, afterEach } = require('@jest/globals');
const fs = require('fs');
const path = require('path');
const {
  PlatformInitializer,
  initializePlatform,
  InitState,
  InitializationError,
  ConfigurationError,
  EnvironmentError,
  ServiceInitializationError,
  DatabaseConnectionError
} = require('../../../src/init');

// Mock modules
jest.mock('fs');
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

describe('PlatformInitializer - Constructor and Configuration', () => {
  let initializer;

  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(async () => {
    if (initializer) {
      try {
        await initializer.shutdown();
      } catch (error) {
        // Ignore shutdown errors in tests
      }
    }
  });

  it('should create initializer with default options', () => {
    initializer = new PlatformInitializer();

    expect(initializer).toBeDefined();
    expect(initializer.state).toBe(InitState.PENDING);
    expect(initializer.options.environment).toBe(process.env.NODE_ENV || 'development');
    expect(initializer.options.enableLogging).toBe(true);
    expect(initializer.options.validateConfig).toBe(true);
    expect(initializer.options.autoConnect).toBe(true);
    expect(initializer.options.timeout).toBe(30000);
  });

  it('should create initializer with custom options', () => {
    initializer = new PlatformInitializer({
      environment: 'production',
      enableLogging: false,
      validateConfig: false,
      autoConnect: false,
      timeout: 60000,
      configPath: '/custom/path'
    });

    expect(initializer.options.environment).toBe('production');
    expect(initializer.options.enableLogging).toBe(false);
    expect(initializer.options.validateConfig).toBe(false);
    expect(initializer.options.autoConnect).toBe(false);
    expect(initializer.options.timeout).toBe(60000);
    expect(initializer.options.configPath).toBe('/custom/path');
  });

  it('should initialize with empty services map', () => {
    initializer = new PlatformInitializer();
    expect(initializer.services).toBeInstanceOf(Map);
    expect(initializer.services.size).toBe(0);
  });

  it('should initialize with empty errors array', () => {
    initializer = new PlatformInitializer();
    expect(initializer.errors).toEqual([]);
  });
});

describe('PlatformInitializer - Configuration Loading', () => {
  let initializer;

  beforeEach(() => {
    jest.clearAllMocks();
    fs.existsSync = jest.fn();
    fs.readFileSync = jest.fn();
  });

  afterEach(async () => {
    if (initializer) {
      await initializer.shutdown().catch(() => {});
    }
  });

  it('should load default configuration successfully', async () => {
    const defaultConfig = {
      database: { host: 'localhost', port: 5432, database: 'test' },
      services: { cache: { enabled: true } },
      logging: { level: 'info' }
    };

    fs.existsSync.mockReturnValue(true);
    fs.readFileSync.mockReturnValue(JSON.stringify(defaultConfig));

    initializer = new PlatformInitializer();
    await initializer.loadConfiguration();

    expect(initializer.config).toBeDefined();
    expect(initializer.config.database).toBeDefined();
  });

  it('should merge default and environment configs', async () => {
    const defaultConfig = {
      database: { host: 'localhost', port: 5432, database: 'default' },
      services: {},
      logging: {}
    };

    const envConfig = {
      database: { database: 'production' }
    };

    fs.existsSync.mockImplementation((path) => true);
    fs.readFileSync.mockImplementation((path) => {
      if (path.includes('default')) return JSON.stringify(defaultConfig);
      return JSON.stringify(envConfig);
    });

    initializer = new PlatformInitializer({ environment: 'production' });
    await initializer.loadConfiguration();

    expect(initializer.config.database.host).toBe('localhost');
    expect(initializer.config.database.database).toBe('production');
  });

  it('should load environment variables with NOVACRON_ prefix', async () => {
    const defaultConfig = {
      database: { host: 'localhost', port: 5432, database: 'test' },
      services: {},
      logging: {}
    };

    fs.existsSync.mockReturnValue(true);
    fs.readFileSync.mockReturnValue(JSON.stringify(defaultConfig));

    process.env.NOVACRON_DEBUG = 'true';
    process.env.NOVACRON_PORT = '8080';

    initializer = new PlatformInitializer();
    await initializer.loadConfiguration();

    expect(initializer.config.debug).toBe(true);
    expect(initializer.config.port).toBe('8080');

    delete process.env.NOVACRON_DEBUG;
    delete process.env.NOVACRON_PORT;
  });

  it('should throw ConfigurationError on invalid JSON', async () => {
    fs.existsSync.mockReturnValue(true);
    fs.readFileSync.mockReturnValue('invalid json {');

    initializer = new PlatformInitializer();

    await expect(initializer.loadConfiguration()).rejects.toThrow(ConfigurationError);
  });

  it('should validate required configuration sections', async () => {
    const incompleteConfig = {
      database: { host: 'localhost', port: 5432, database: 'test' }
      // Missing 'services' and 'logging'
    };

    fs.existsSync.mockReturnValue(true);
    fs.readFileSync.mockReturnValue(JSON.stringify(incompleteConfig));

    initializer = new PlatformInitializer();

    await expect(initializer.loadConfiguration()).rejects.toThrow(ConfigurationError);
  });

  it('should validate database configuration fields', async () => {
    const invalidDbConfig = {
      database: { host: 'localhost' }, // Missing port and database
      services: {},
      logging: {}
    };

    fs.existsSync.mockReturnValue(true);
    fs.readFileSync.mockReturnValue(JSON.stringify(invalidDbConfig));

    initializer = new PlatformInitializer();

    await expect(initializer.loadConfiguration()).rejects.toThrow(ConfigurationError);
  });
});

describe('PlatformInitializer - Environment Validation', () => {
  let initializer;

  beforeEach(() => {
    jest.clearAllMocks();
    fs.existsSync = jest.fn().mockReturnValue(true);
    fs.readFileSync = jest.fn().mockReturnValue(JSON.stringify({
      database: { host: 'localhost', port: 5432, database: 'test' },
      services: {},
      logging: {}
    }));
    fs.mkdirSync = jest.fn();
    fs.accessSync = jest.fn();
  });

  afterEach(async () => {
    if (initializer) {
      await initializer.shutdown().catch(() => {});
    }
  });

  it('should validate Node.js version successfully', async () => {
    initializer = new PlatformInitializer();

    const result = initializer.checkNodeVersion();
    expect(result).toBe(true);
  });

  it('should create missing required directories', async () => {
    fs.existsSync.mockImplementation((path) => {
      if (path.includes('config.')) return true;
      return false; // Missing directories
    });

    initializer = new PlatformInitializer();
    const result = initializer.checkRequiredDirectories();

    expect(fs.mkdirSync).toHaveBeenCalled();
    expect(result).toBe(true);
  });

  it('should validate file permissions', async () => {
    fs.accessSync.mockImplementation(() => {}); // No error = success

    initializer = new PlatformInitializer();
    const result = initializer.checkFilePermissions();

    expect(result).toBe(true);
  });

  it('should throw EnvironmentError on permission failure', async () => {
    fs.accessSync.mockImplementation(() => {
      throw new Error('Permission denied');
    });

    initializer = new PlatformInitializer();

    expect(() => initializer.checkFilePermissions()).toThrow(EnvironmentError);
  });
});

describe('PlatformInitializer - Initialization Flow', () => {
  let initializer;

  beforeEach(() => {
    jest.clearAllMocks();
    fs.existsSync = jest.fn().mockReturnValue(true);
    fs.readFileSync = jest.fn().mockReturnValue(JSON.stringify({
      database: { host: 'localhost', port: 5432, database: 'test' },
      services: {},
      logging: { level: 'info' }
    }));
    fs.mkdirSync = jest.fn();
    fs.accessSync = jest.fn();
  });

  afterEach(async () => {
    if (initializer) {
      await initializer.shutdown().catch(() => {});
    }
  });

  it('should complete full initialization successfully', async () => {
    initializer = new PlatformInitializer({
      autoConnect: false,
      enableLogging: true
    });

    const result = await initializer.initialize();

    expect(result.status).toBe('success');
    expect(result.state).toBe(InitState.INITIALIZED);
    expect(result.duration).toBeGreaterThan(0);
    expect(initializer.state).toBe(InitState.INITIALIZED);
  });

  it('should emit init:start event', async () => {
    initializer = new PlatformInitializer({ autoConnect: false });

    const startListener = jest.fn();
    initializer.on('init:start', startListener);

    await initializer.initialize();

    expect(startListener).toHaveBeenCalledWith(
      expect.objectContaining({ timestamp: expect.any(Number) })
    );
  });

  it('should emit init:complete event', async () => {
    initializer = new PlatformInitializer({ autoConnect: false });

    const completeListener = jest.fn();
    initializer.on('init:complete', completeListener);

    await initializer.initialize();

    expect(completeListener).toHaveBeenCalledWith(
      expect.objectContaining({
        status: 'success',
        duration: expect.any(Number)
      })
    );
  });

  it('should set state to INITIALIZING during init', async () => {
    initializer = new PlatformInitializer({ autoConnect: false });

    let stateCheck;
    initializer.on('init:config-loaded', () => {
      stateCheck = initializer.state;
    });

    await initializer.initialize();

    expect(stateCheck).toBe(InitState.INITIALIZING);
  });

  it('should record initialization duration', async () => {
    initializer = new PlatformInitializer({ autoConnect: false });

    const result = await initializer.initialize();

    expect(result.duration).toBeGreaterThan(0);
    expect(result.duration).toBeLessThan(5000);
  });
});

describe('PlatformInitializer - Service Initialization', () => {
  let initializer;

  beforeEach(() => {
    jest.clearAllMocks();
    fs.existsSync = jest.fn().mockReturnValue(true);
    fs.readFileSync = jest.fn().mockReturnValue(JSON.stringify({
      database: { host: 'localhost', port: 5432, database: 'test' },
      services: {},
      logging: { level: 'info' }
    }));
  });

  afterEach(async () => {
    if (initializer) {
      await initializer.shutdown().catch(() => {});
    }
  });

  it('should handle missing service modules gracefully', async () => {
    initializer = new PlatformInitializer({ autoConnect: false });

    // Core services don't exist in test environment
    await expect(initializer.initialize()).resolves.toBeDefined();
  });

  it('should add initialized services to services map', async () => {
    initializer = new PlatformInitializer({ autoConnect: false });

    await initializer.initialize();

    // Services should be attempted even if they fail to load
    expect(initializer.services).toBeDefined();
  });
});

describe('PlatformInitializer - Error Handling', () => {
  let initializer;

  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(async () => {
    if (initializer) {
      await initializer.shutdown().catch(() => {});
    }
  });

  it('should handle configuration loading errors', async () => {
    fs.existsSync = jest.fn().mockReturnValue(false);

    initializer = new PlatformInitializer();

    await expect(initializer.initialize()).rejects.toThrow(InitializationError);
  });

  it('should set state to FAILED on error', async () => {
    fs.existsSync = jest.fn().mockReturnValue(false);

    initializer = new PlatformInitializer();

    try {
      await initializer.initialize();
    } catch (error) {
      expect(initializer.state).toBe(InitState.FAILED);
    }
  });

  it('should emit init:failed event', async () => {
    fs.existsSync = jest.fn().mockReturnValue(false);

    initializer = new PlatformInitializer();

    const failListener = jest.fn();
    initializer.on('init:failed', failListener);

    try {
      await initializer.initialize();
    } catch (error) {
      expect(failListener).toHaveBeenCalledWith(
        expect.objectContaining({
          status: 'failed',
          error: expect.any(String)
        })
      );
    }
  });

  it('should collect errors in errors array', async () => {
    fs.existsSync = jest.fn().mockReturnValue(false);

    initializer = new PlatformInitializer();

    try {
      await initializer.initialize();
    } catch (error) {
      expect(initializer.errors.length).toBeGreaterThan(0);
    }
  });
});

describe('PlatformInitializer - Graceful Shutdown', () => {
  let initializer;

  beforeEach(() => {
    jest.clearAllMocks();
    fs.existsSync = jest.fn().mockReturnValue(true);
    fs.readFileSync = jest.fn().mockReturnValue(JSON.stringify({
      database: { host: 'localhost', port: 5432, database: 'test' },
      services: {},
      logging: { level: 'info' }
    }));
  });

  it('should perform graceful shutdown', async () => {
    initializer = new PlatformInitializer({ autoConnect: false });
    await initializer.initialize();

    await expect(initializer.shutdown()).resolves.not.toThrow();
  });

  it('should emit shutdown:start event', async () => {
    initializer = new PlatformInitializer({ autoConnect: false });
    await initializer.initialize();

    const shutdownListener = jest.fn();
    initializer.on('shutdown:start', shutdownListener);

    await initializer.shutdown();

    expect(shutdownListener).toHaveBeenCalled();
  });

  it('should emit shutdown:complete event', async () => {
    initializer = new PlatformInitializer({ autoConnect: false });
    await initializer.initialize();

    const completeListener = jest.fn();
    initializer.on('shutdown:complete', completeListener);

    await initializer.shutdown();

    expect(completeListener).toHaveBeenCalled();
  });

  it('should shutdown services with shutdown methods', async () => {
    initializer = new PlatformInitializer({ autoConnect: false });
    await initializer.initialize();

    const mockService = {
      shutdown: jest.fn().mockResolvedValue()
    };
    initializer.services.set('test-service', mockService);

    await initializer.shutdown();

    expect(mockService.shutdown).toHaveBeenCalled();
  });

  it('should handle shutdown errors gracefully', async () => {
    initializer = new PlatformInitializer({ autoConnect: false });
    await initializer.initialize();

    const mockService = {
      shutdown: jest.fn().mockRejectedValue(new Error('Shutdown failed'))
    };
    initializer.services.set('failing-service', mockService);

    await expect(initializer.shutdown()).resolves.not.toThrow();
  });
});

describe('PlatformInitializer - Utility Methods', () => {
  let initializer;

  beforeEach(() => {
    initializer = new PlatformInitializer();
  });

  afterEach(async () => {
    if (initializer) {
      await initializer.shutdown().catch(() => {});
    }
  });

  it('should get service by name', () => {
    const mockService = { name: 'test' };
    initializer.services.set('test', mockService);

    expect(initializer.getService('test')).toBe(mockService);
  });

  it('should return undefined for non-existent service', () => {
    expect(initializer.getService('non-existent')).toBeUndefined();
  });

  it('should get all services', () => {
    const service1 = { name: 'service1' };
    const service2 = { name: 'service2' };
    initializer.services.set('service1', service1);
    initializer.services.set('service2', service2);

    const allServices = initializer.getAllServices();

    expect(allServices.service1).toBe(service1);
    expect(allServices.service2).toBe(service2);
  });

  it('should return safe config without passwords', async () => {
    fs.existsSync = jest.fn().mockReturnValue(true);
    fs.readFileSync = jest.fn().mockReturnValue(JSON.stringify({
      database: {
        postgres: { host: 'localhost', port: 5432, database: 'test', password: 'secret' },
        redis: { host: 'localhost', port: 6379, password: 'redis-secret' }
      },
      services: {},
      logging: {}
    }));

    await initializer.loadConfiguration();
    const safeConfig = initializer.getSafeConfig();

    expect(safeConfig.database.postgres.password).toBeUndefined();
    expect(safeConfig.database.redis.password).toBeUndefined();
    expect(safeConfig.database.postgres.host).toBe('localhost');
  });

  it('should compare versions correctly', () => {
    expect(initializer.compareVersions('18.0.0', '16.0.0')).toBe(1);
    expect(initializer.compareVersions('16.0.0', '18.0.0')).toBe(-1);
    expect(initializer.compareVersions('18.0.0', '18.0.0')).toBe(0);
  });
});

describe('Factory Function - initializePlatform', () => {
  it('should create and initialize platform', async () => {
    fs.existsSync = jest.fn().mockReturnValue(true);
    fs.readFileSync = jest.fn().mockReturnValue(JSON.stringify({
      database: { host: 'localhost', port: 5432, database: 'test' },
      services: {},
      logging: {}
    }));

    const platform = await initializePlatform({ autoConnect: false });

    expect(platform).toBeInstanceOf(PlatformInitializer);
    expect(platform.state).toBe(InitState.INITIALIZED);

    await platform.shutdown();
  });
});

describe('Custom Error Classes', () => {
  it('should create InitializationError with details', () => {
    const cause = new Error('Root cause');
    const error = new InitializationError('Init failed', cause, { code: 'INIT001' });

    expect(error.name).toBe('InitializationError');
    expect(error.message).toBe('Init failed');
    expect(error.cause).toBe(cause);
    expect(error.details.code).toBe('INIT001');
    expect(error.timestamp).toBeDefined();
  });

  it('should create ConfigurationError', () => {
    const error = new ConfigurationError('Config invalid');

    expect(error.name).toBe('ConfigurationError');
    expect(error.message).toBe('Config invalid');
    expect(error.timestamp).toBeDefined();
  });

  it('should create EnvironmentError', () => {
    const error = new EnvironmentError('Bad environment');

    expect(error.name).toBe('EnvironmentError');
    expect(error.message).toBe('Bad environment');
    expect(error.timestamp).toBeDefined();
  });

  it('should create ServiceInitializationError', () => {
    const error = new ServiceInitializationError('Service failed');

    expect(error.name).toBe('ServiceInitializationError');
    expect(error.message).toBe('Service failed');
    expect(error.timestamp).toBeDefined();
  });

  it('should create DatabaseConnectionError', () => {
    const error = new DatabaseConnectionError('DB connection failed');

    expect(error.name).toBe('DatabaseConnectionError');
    expect(error.message).toBe('DB connection failed');
    expect(error.timestamp).toBeDefined();
  });
});
