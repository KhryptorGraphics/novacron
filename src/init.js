/**
 * NovaCron Platform Initialization Module
 *
 * This module handles the complete initialization sequence for the NovaCron platform,
 * including configuration loading, service bootstrapping, database connections,
 * and error handling setup.
 *
 * @module init
 */

const fs = require('fs');
const path = require('path');
const { EventEmitter } = require('events');
const { ConfigLoader } = require('./config/loader');
const { createPlatformValidator } = require('./config/validator');
const { createHealthChecker, HealthStatus } = require('./health/checker');

/**
 * Initialization states
 */
const InitState = {
  PENDING: 'pending',
  INITIALIZING: 'initializing',
  INITIALIZED: 'initialized',
  FAILED: 'failed'
};

/**
 * Main initialization class for NovaCron platform
 */
class PlatformInitializer extends EventEmitter {
  constructor(options = {}) {
    super();

    this.options = {
      configPath: options.configPath || path.join(process.cwd(), 'config'),
      environment: options.environment || process.env.NODE_ENV || 'development',
      enableLogging: options.enableLogging !== false,
      validateConfig: options.validateConfig !== false,
      autoConnect: options.autoConnect !== false,
      timeout: options.timeout || 30000, // 30 seconds default timeout
      ...options
    };

    this.state = InitState.PENDING;
    this.services = new Map();
    this.config = null;
    this.startTime = null;
    this.errors = [];
    this.configLoader = null;
    this.healthChecker = null;
    this.retryAttempts = options.retryAttempts || 3;
    this.retryDelay = options.retryDelay || 2000;
  }

  /**
   * Initialize the complete platform
   * @returns {Promise<Object>} Initialization result
   */
  async initialize() {
    this.startTime = Date.now();
    this.state = InitState.INITIALIZING;
    this.emit('init:start', { timestamp: this.startTime });

    try {
      // Step 1: Load configuration
      await this.loadConfiguration();
      this.emit('init:config-loaded', { config: this.config });

      // Step 2: Setup logging
      await this.setupLogging();
      this.emit('init:logging-setup');

      // Step 3: Validate environment
      await this.validateEnvironment();
      this.emit('init:environment-validated');

      // Step 4: Initialize core services
      await this.initializeCoreServices();
      this.emit('init:core-services-ready');

      // Step 5: Connect to databases
      if (this.options.autoConnect) {
        await this.connectDatabases();
        this.emit('init:databases-connected');
      }

      // Step 6: Initialize optional services
      await this.initializeOptionalServices();
      this.emit('init:optional-services-ready');

      // Step 7: Setup health checks
      await this.setupHealthChecks();
      this.emit('init:health-checks-setup');

      // Step 8: Setup error handlers
      await this.setupErrorHandlers();
      this.emit('init:error-handlers-setup');

      const duration = Date.now() - this.startTime;
      this.state = InitState.INITIALIZED;

      const result = {
        status: 'success',
        state: this.state,
        duration,
        services: Array.from(this.services.keys()),
        config: this.getSafeConfig(),
        timestamp: new Date().toISOString()
      };

      this.emit('init:complete', result);
      this.log('info', `Platform initialized successfully in ${duration}ms`);

      return result;

    } catch (error) {
      this.state = InitState.FAILED;
      this.errors.push(error);

      const result = {
        status: 'failed',
        state: this.state,
        error: error.message,
        stack: error.stack,
        errors: this.errors.map(e => ({
          message: e.message,
          code: e.code,
          timestamp: e.timestamp || new Date().toISOString()
        })),
        timestamp: new Date().toISOString()
      };

      this.emit('init:failed', result);
      this.log('error', `Platform initialization failed: ${error.message}`);

      throw new InitializationError('Platform initialization failed', error, result);
    }
  }

  /**
   * Load and validate configuration files
   */
  async loadConfiguration() {
    this.log('info', 'Loading configuration...');

    try {
      // Use ConfigLoader for advanced configuration management
      this.configLoader = new ConfigLoader({
        configPath: this.options.configPath,
        environment: this.options.environment,
        validateOnLoad: this.options.validateConfig
      });

      // Load configuration
      this.config = await this.configLoader.load();

      // Validate configuration with platform validator
      if (this.options.validateConfig) {
        const validator = createPlatformValidator();
        const validationResult = validator.validate(this.config);

        if (!validationResult.valid) {
          const errorMessages = validationResult.errors
            .map(e => `${e.field}: ${e.message}`)
            .join('; ');

          throw new ConfigurationError(
            `Configuration validation failed: ${errorMessages}`
          );
        }

        if (validationResult.warnings.length > 0) {
          this.log('warning', `Configuration warnings: ${validationResult.warnings.length}`, {
            warnings: validationResult.warnings
          });
        }
      }

      this.log('info', 'Configuration loaded and validated successfully');

    } catch (error) {
      throw new ConfigurationError(`Failed to load configuration: ${error.message}`, error);
    }
  }


  /**
   * Setup logging system
   */
  async setupLogging() {
    if (!this.options.enableLogging) {
      return;
    }

    this.log('info', 'Setting up logging system...');

    const loggingConfig = this.config.logging || {
      level: 'info',
      format: 'json',
      destination: 'console'
    };

    // In a real implementation, this would initialize a proper logging library
    // For now, we'll use a simple console-based logger
    this.logger = {
      level: loggingConfig.level,
      format: loggingConfig.format,
      log: (level, message, meta = {}) => {
        if (this.shouldLog(level)) {
          const logEntry = {
            timestamp: new Date().toISOString(),
            level,
            message,
            ...meta
          };
          console.log(JSON.stringify(logEntry));
        }
      }
    };

    this.log('info', 'Logging system initialized');
  }

  /**
   * Validate environment
   */
  async validateEnvironment() {
    this.log('info', 'Validating environment...');

    const checks = [
      this.checkNodeVersion(),
      this.checkRequiredDirectories(),
      this.checkFilePermissions()
    ];

    const results = await Promise.allSettled(checks);
    const failures = results.filter(r => r.status === 'rejected');

    if (failures.length > 0) {
      const errors = failures.map(f => f.reason.message).join('; ');
      throw new EnvironmentError(`Environment validation failed: ${errors}`);
    }

    this.log('info', 'Environment validated successfully');
  }

  /**
   * Check Node.js version
   */
  checkNodeVersion() {
    const required = '18.0.0';
    const current = process.version.substring(1); // Remove 'v' prefix

    if (this.compareVersions(current, required) < 0) {
      throw new EnvironmentError(
        `Node.js version ${required} or higher required (current: ${current})`
      );
    }

    return true;
  }

  /**
   * Check required directories exist
   */
  checkRequiredDirectories() {
    const requiredDirs = [
      'src',
      'config',
      'logs'
    ].map(dir => path.join(process.cwd(), dir));

    const missing = requiredDirs.filter(dir => !fs.existsSync(dir));

    if (missing.length > 0) {
      // Create missing directories
      missing.forEach(dir => {
        fs.mkdirSync(dir, { recursive: true });
        this.log('info', `Created missing directory: ${dir}`);
      });
    }

    return true;
  }

  /**
   * Check file permissions
   */
  checkFilePermissions() {
    const configPath = path.join(process.cwd(), 'config');

    try {
      fs.accessSync(configPath, fs.constants.R_OK);
      return true;
    } catch (error) {
      throw new EnvironmentError(`Cannot read configuration directory: ${configPath}`);
    }
  }

  /**
   * Initialize core services
   */
  async initializeCoreServices() {
    this.log('info', 'Initializing core services...');

    const coreServices = [
      { name: 'cache', path: './cache/cache-manager' },
      { name: 'workload-monitor', path: './services/workload-monitor' },
      { name: 'mcp-integration', path: './services/mcp-integration' }
    ];

    for (const service of coreServices) {
      try {
        await this.initializeService(service.name, service.path);
      } catch (error) {
        this.log('error', `Failed to initialize core service ${service.name}: ${error.message}`);
        throw new ServiceInitializationError(
          `Core service initialization failed: ${service.name}`,
          error
        );
      }
    }

    this.log('info', 'Core services initialized');
  }

  /**
   * Initialize a single service
   */
  async initializeService(name, modulePath) {
    this.log('info', `Initializing service: ${name}`);

    try {
      const ServiceModule = require(modulePath);
      const serviceConfig = this.config.services?.[name] || {};

      let service;
      if (typeof ServiceModule === 'function') {
        service = new ServiceModule(serviceConfig);
      } else if (ServiceModule.default && typeof ServiceModule.default === 'function') {
        service = new ServiceModule.default(serviceConfig);
      } else {
        service = ServiceModule;
      }

      // Initialize service if it has an init method
      if (typeof service.initialize === 'function') {
        await service.initialize();
      } else if (typeof service.init === 'function') {
        await service.init();
      }

      this.services.set(name, service);
      this.log('info', `Service initialized: ${name}`);

      return service;

    } catch (error) {
      if (error.code === 'MODULE_NOT_FOUND') {
        this.log('warning', `Service module not found: ${modulePath}`);
        return null;
      }
      throw error;
    }
  }

  /**
   * Connect to databases
   */
  async connectDatabases() {
    this.log('info', 'Connecting to databases...');

    const dbConfig = this.config.database;

    if (!dbConfig) {
      this.log('warning', 'No database configuration found');
      return;
    }

    // PostgreSQL connection
    if (dbConfig.postgres) {
      await this.connectPostgres(dbConfig.postgres);
    }

    // Redis connection
    if (dbConfig.redis) {
      await this.connectRedis(dbConfig.redis);
    }

    this.log('info', 'Database connections established');
  }

  /**
   * Connect to PostgreSQL with retry logic
   */
  async connectPostgres(config) {
    this.log('info', 'Connecting to PostgreSQL...');

    return this.retryOperation(
      async () => {
        const { Pool } = require('pg');

        const pool = new Pool({
          host: config.host,
          port: config.port,
          database: config.database,
          user: config.user,
          password: config.password,
          max: config.poolSize || 10,
          idleTimeoutMillis: config.idleTimeout || 30000,
          connectionTimeoutMillis: config.connectionTimeout || 10000
        });

        // Test connection
        const client = await pool.connect();
        await client.query('SELECT NOW()');
        client.release();

        this.services.set('postgres', pool);
        this.log('info', 'PostgreSQL connection established');
      },
      {
        operationName: 'PostgreSQL connection',
        critical: true
      }
    );
  }

  /**
   * Connect to Redis with retry logic
   */
  async connectRedis(config) {
    this.log('info', 'Connecting to Redis...');

    return this.retryOperation(
      async () => {
        const redis = require('redis');

        const client = redis.createClient({
          host: config.host,
          port: config.port,
          password: config.password,
          db: config.database || 0
        });

        await client.connect();
        await client.ping();

        this.services.set('redis', client);
        this.log('info', 'Redis connection established');
      },
      {
        operationName: 'Redis connection',
        critical: true
      }
    );
  }

  /**
   * Initialize optional services
   */
  async initializeOptionalServices() {
    this.log('info', 'Initializing optional services...');

    const optionalServices = [
      { name: 'smart-agent-spawner', path: './services/smart-agent-spawner' },
      { name: 'auto-spawning-orchestrator', path: './services/auto-spawning-orchestrator' }
    ];

    for (const service of optionalServices) {
      try {
        await this.initializeService(service.name, service.path);
      } catch (error) {
        this.log('warning', `Optional service ${service.name} not initialized: ${error.message}`);
        // Don't throw for optional services
      }
    }

    this.log('info', 'Optional services initialization complete');
  }

  /**
   * Setup health checks
   */
  async setupHealthChecks() {
    this.log('info', 'Setting up health checks...');

    try {
      // Create health checker with registered services
      this.healthChecker = createHealthChecker(
        {
          postgres: this.services.get('postgres'),
          redis: this.services.get('redis')
        },
        {
          checkInterval: this.config.health?.checkInterval || 30000,
          enableAutoCheck: this.config.health?.autoCheck !== false,
          retryAttempts: this.retryAttempts
        }
      );

      // Listen to health check events
      this.healthChecker.on('check:failed', (result) => {
        this.log('error', `Health check failed: ${result.name}`, {
          error: result.error,
          critical: result.critical
        });

        // Emit health degradation event
        if (result.critical) {
          this.emit('health:critical', result);
        } else {
          this.emit('health:degraded', result);
        }
      });

      this.healthChecker.on('check:success', (result) => {
        this.log('debug', `Health check passed: ${result.name}`, {
          duration: result.duration
        });
      });

      // Store health checker as a service
      this.services.set('health-checker', this.healthChecker);

      // Start health monitoring if enabled
      if (this.config.health?.autoCheck !== false) {
        this.healthChecker.start();
        this.log('info', 'Health monitoring started');
      }

      this.log('info', 'Health checks configured successfully');

    } catch (error) {
      this.log('warning', `Failed to setup health checks: ${error.message}`);
      // Don't throw - health checks are not critical for initialization
    }
  }

  /**
   * Retry operation with exponential backoff
   * @param {Function} operation - Async operation to retry
   * @param {Object} options - Retry options
   * @returns {Promise<any>} Operation result
   */
  async retryOperation(operation, options = {}) {
    const {
      operationName = 'Operation',
      critical = false,
      maxAttempts = this.retryAttempts,
      initialDelay = this.retryDelay,
      backoffFactor = 2,
      maxDelay = 30000
    } = options;

    let lastError = null;
    let delay = initialDelay;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        this.log('debug', `${operationName}: Attempt ${attempt}/${maxAttempts}`);

        const result = await operation();

        if (attempt > 1) {
          this.log('info', `${operationName}: Succeeded after ${attempt} attempts`);
        }

        return result;

      } catch (error) {
        lastError = error;

        this.log('warning', `${operationName}: Attempt ${attempt} failed`, {
          error: error.message,
          nextRetryIn: attempt < maxAttempts ? delay : null
        });

        if (attempt < maxAttempts) {
          // Wait before retrying
          await this.delay(delay);

          // Exponential backoff with max delay cap
          delay = Math.min(delay * backoffFactor, maxDelay);
        }
      }
    }

    // All retries exhausted
    const errorMsg = `${operationName}: Failed after ${maxAttempts} attempts: ${lastError.message}`;

    if (critical) {
      throw new Error(errorMsg);
    } else {
      this.log('error', errorMsg);
      this.errors.push({
        operation: operationName,
        error: lastError,
        attempts: maxAttempts,
        timestamp: new Date().toISOString()
      });
      return null;
    }
  }

  /**
   * Utility: Delay for specified milliseconds
   * @param {number} ms - Milliseconds to delay
   * @returns {Promise<void>}
   */
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Setup error handlers
   */
  async setupErrorHandlers() {
    this.log('info', 'Setting up error handlers...');

    // Unhandled promise rejections
    process.on('unhandledRejection', (reason, promise) => {
      this.log('error', 'Unhandled Promise Rejection', {
        reason: reason?.message || reason,
        stack: reason?.stack
      });
      this.emit('error:unhandled-rejection', { reason, promise });
    });

    // Uncaught exceptions
    process.on('uncaughtException', (error) => {
      this.log('error', 'Uncaught Exception', {
        message: error.message,
        stack: error.stack
      });
      this.emit('error:uncaught-exception', { error });

      // Give time to log before exiting
      setTimeout(() => {
        process.exit(1);
      }, 1000);
    });

    // SIGTERM and SIGINT handlers
    ['SIGTERM', 'SIGINT'].forEach(signal => {
      process.on(signal, async () => {
        this.log('info', `Received ${signal}, shutting down gracefully...`);
        await this.shutdown();
        process.exit(0);
      });
    });

    this.log('info', 'Error handlers configured');
  }

  /**
   * Graceful shutdown
   */
  async shutdown() {
    this.log('info', 'Initiating graceful shutdown...');
    this.emit('shutdown:start');

    try {
      // Close database connections
      const postgres = this.services.get('postgres');
      if (postgres && typeof postgres.end === 'function') {
        await postgres.end();
        this.log('info', 'PostgreSQL connection closed');
      }

      const redis = this.services.get('redis');
      if (redis && typeof redis.quit === 'function') {
        await redis.quit();
        this.log('info', 'Redis connection closed');
      }

      // Shutdown services
      for (const [name, service] of this.services.entries()) {
        if (typeof service.shutdown === 'function') {
          await service.shutdown();
          this.log('info', `Service ${name} shut down`);
        }
      }

      this.emit('shutdown:complete');
      this.log('info', 'Graceful shutdown complete');

    } catch (error) {
      this.log('error', `Error during shutdown: ${error.message}`);
      this.emit('shutdown:error', { error });
      throw error;
    }
  }

  /**
   * Get service by name
   */
  getService(name) {
    return this.services.get(name);
  }

  /**
   * Get all services
   */
  getAllServices() {
    return Object.fromEntries(this.services);
  }

  /**
   * Get safe configuration (without sensitive data)
   */
  getSafeConfig() {
    const safe = { ...this.config };

    // Remove sensitive fields
    if (safe.database) {
      if (safe.database.postgres) {
        safe.database.postgres = { ...safe.database.postgres };
        delete safe.database.postgres.password;
      }
      if (safe.database.redis) {
        safe.database.redis = { ...safe.database.redis };
        delete safe.database.redis.password;
      }
    }

    return safe;
  }

  /**
   * Check if should log at level
   */
  shouldLog(level) {
    const levels = ['debug', 'info', 'warning', 'error'];
    const currentLevel = this.logger?.level || 'info';
    return levels.indexOf(level) >= levels.indexOf(currentLevel);
  }

  /**
   * Log message
   */
  log(level, message, meta = {}) {
    if (this.logger) {
      this.logger.log(level, message, meta);
    } else if (this.options.enableLogging) {
      console.log(`[${level.toUpperCase()}] ${message}`);
    }
  }

  /**
   * Compare semantic versions
   */
  compareVersions(v1, v2) {
    const parts1 = v1.split('.').map(Number);
    const parts2 = v2.split('.').map(Number);

    for (let i = 0; i < 3; i++) {
      if (parts1[i] > parts2[i]) return 1;
      if (parts1[i] < parts2[i]) return -1;
    }
    return 0;
  }
}

/**
 * Custom error classes
 */
class InitializationError extends Error {
  constructor(message, cause, details = {}) {
    super(message);
    this.name = 'InitializationError';
    this.cause = cause;
    this.details = details;
    this.timestamp = new Date().toISOString();
  }
}

class ConfigurationError extends Error {
  constructor(message, cause) {
    super(message);
    this.name = 'ConfigurationError';
    this.cause = cause;
    this.timestamp = new Date().toISOString();
  }
}

class EnvironmentError extends Error {
  constructor(message) {
    super(message);
    this.name = 'EnvironmentError';
    this.timestamp = new Date().toISOString();
  }
}

class ServiceInitializationError extends Error {
  constructor(message, cause) {
    super(message);
    this.name = 'ServiceInitializationError';
    this.cause = cause;
    this.timestamp = new Date().toISOString();
  }
}

class DatabaseConnectionError extends Error {
  constructor(message, cause) {
    super(message);
    this.name = 'DatabaseConnectionError';
    this.cause = cause;
    this.timestamp = new Date().toISOString();
  }
}

/**
 * Factory function to create and initialize platform
 */
async function initializePlatform(options = {}) {
  const initializer = new PlatformInitializer(options);
  await initializer.initialize();
  return initializer;
}

/**
 * Export module
 */
module.exports = {
  PlatformInitializer,
  initializePlatform,
  InitState,
  InitializationError,
  ConfigurationError,
  EnvironmentError,
  ServiceInitializationError,
  DatabaseConnectionError
};
