/**
 * NovaCron Health Check System
 *
 * Provides comprehensive health checking for all platform services,
 * including databases, cache, distributed components, and external services.
 *
 * @module health/checker
 */

const { EventEmitter } = require('events');

/**
 * Health check status levels
 */
const HealthStatus = {
  HEALTHY: 'healthy',
  DEGRADED: 'degraded',
  UNHEALTHY: 'unhealthy',
  UNKNOWN: 'unknown'
};

/**
 * Component types for health checking
 */
const ComponentType = {
  DATABASE: 'database',
  CACHE: 'cache',
  SERVICE: 'service',
  EXTERNAL: 'external',
  DISTRIBUTED: 'distributed'
};

/**
 * Main health checker class
 */
class HealthChecker extends EventEmitter {
  constructor(options = {}) {
    super();

    this.options = {
      checkInterval: options.checkInterval || 30000, // 30 seconds
      timeout: options.timeout || 5000, // 5 seconds
      enableAutoCheck: options.enableAutoCheck !== false,
      retryAttempts: options.retryAttempts || 3,
      retryDelay: options.retryDelay || 1000,
      ...options
    };

    this.checks = new Map();
    this.results = new Map();
    this.intervalId = null;
    this.isRunning = false;
  }

  /**
   * Register a health check
   * @param {string} name - Check name
   * @param {Function} checkFn - Async function that performs the check
   * @param {Object} options - Check-specific options
   */
  registerCheck(name, checkFn, options = {}) {
    if (typeof checkFn !== 'function') {
      throw new Error(`Health check function for '${name}' must be a function`);
    }

    this.checks.set(name, {
      name,
      checkFn,
      type: options.type || ComponentType.SERVICE,
      critical: options.critical !== false,
      timeout: options.timeout || this.options.timeout,
      retryAttempts: options.retryAttempts || this.options.retryAttempts,
      retryDelay: options.retryDelay || this.options.retryDelay,
      metadata: options.metadata || {}
    });

    this.emit('check:registered', { name, type: options.type });
    return this;
  }

  /**
   * Unregister a health check
   * @param {string} name - Check name
   */
  unregisterCheck(name) {
    const removed = this.checks.delete(name);
    this.results.delete(name);

    if (removed) {
      this.emit('check:unregistered', { name });
    }

    return removed;
  }

  /**
   * Execute a single health check with retry logic
   * @param {string} name - Check name
   * @returns {Promise<Object>} Check result
   */
  async executeCheck(name) {
    const check = this.checks.get(name);

    if (!check) {
      return {
        name,
        status: HealthStatus.UNKNOWN,
        error: 'Check not registered',
        timestamp: new Date().toISOString()
      };
    }

    let lastError = null;

    for (let attempt = 1; attempt <= check.retryAttempts; attempt++) {
      try {
        const startTime = Date.now();

        // Execute check with timeout
        const result = await this.executeWithTimeout(
          check.checkFn(),
          check.timeout
        );

        const duration = Date.now() - startTime;

        const checkResult = {
          name,
          status: result.status || HealthStatus.HEALTHY,
          type: check.type,
          critical: check.critical,
          duration,
          attempt,
          message: result.message,
          metadata: {
            ...check.metadata,
            ...result.metadata
          },
          timestamp: new Date().toISOString()
        };

        this.results.set(name, checkResult);
        this.emit('check:success', checkResult);

        return checkResult;

      } catch (error) {
        lastError = error;

        if (attempt < check.retryAttempts) {
          this.emit('check:retry', {
            name,
            attempt,
            error: error.message,
            nextRetryIn: check.retryDelay
          });

          await this.delay(check.retryDelay);
        }
      }
    }

    // All retries failed
    const failedResult = {
      name,
      status: HealthStatus.UNHEALTHY,
      type: check.type,
      critical: check.critical,
      error: lastError.message,
      stack: lastError.stack,
      attempts: check.retryAttempts,
      timestamp: new Date().toISOString()
    };

    this.results.set(name, failedResult);
    this.emit('check:failed', failedResult);

    return failedResult;
  }

  /**
   * Execute function with timeout
   * @param {Promise} promise - Promise to execute
   * @param {number} timeout - Timeout in milliseconds
   * @returns {Promise<any>}
   */
  async executeWithTimeout(promise, timeout) {
    return Promise.race([
      promise,
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Health check timeout')), timeout)
      )
    ]);
  }

  /**
   * Execute all registered health checks
   * @returns {Promise<Object>} Aggregated results
   */
  async checkAll() {
    const startTime = Date.now();
    const checkNames = Array.from(this.checks.keys());

    this.emit('checks:started', { count: checkNames.length });

    // Execute all checks in parallel
    const results = await Promise.all(
      checkNames.map(name => this.executeCheck(name))
    );

    const duration = Date.now() - startTime;

    // Aggregate results
    const aggregated = this.aggregateResults(results, duration);

    this.emit('checks:completed', aggregated);

    return aggregated;
  }

  /**
   * Aggregate check results into overall health status
   * @param {Array} results - Individual check results
   * @param {number} duration - Total check duration
   * @returns {Object} Aggregated health status
   */
  aggregateResults(results, duration) {
    const checks = {};
    let overallStatus = HealthStatus.HEALTHY;
    const failures = [];
    const warnings = [];

    for (const result of results) {
      checks[result.name] = result;

      // Track failures and warnings
      if (result.status === HealthStatus.UNHEALTHY) {
        if (result.critical) {
          failures.push(result.name);
          overallStatus = HealthStatus.UNHEALTHY;
        } else {
          warnings.push(result.name);
          if (overallStatus === HealthStatus.HEALTHY) {
            overallStatus = HealthStatus.DEGRADED;
          }
        }
      } else if (result.status === HealthStatus.DEGRADED) {
        warnings.push(result.name);
        if (overallStatus === HealthStatus.HEALTHY) {
          overallStatus = HealthStatus.DEGRADED;
        }
      }
    }

    return {
      status: overallStatus,
      timestamp: new Date().toISOString(),
      duration,
      checks,
      summary: {
        total: results.length,
        healthy: results.filter(r => r.status === HealthStatus.HEALTHY).length,
        degraded: results.filter(r => r.status === HealthStatus.DEGRADED).length,
        unhealthy: results.filter(r => r.status === HealthStatus.UNHEALTHY).length,
        failures,
        warnings
      }
    };
  }

  /**
   * Get current health status
   * @returns {Object} Current health status
   */
  getStatus() {
    const results = Array.from(this.results.values());

    if (results.length === 0) {
      return {
        status: HealthStatus.UNKNOWN,
        message: 'No health checks have been executed',
        timestamp: new Date().toISOString()
      };
    }

    return this.aggregateResults(results, 0);
  }

  /**
   * Get result for a specific check
   * @param {string} name - Check name
   * @returns {Object|null} Check result or null
   */
  getCheckResult(name) {
    return this.results.get(name) || null;
  }

  /**
   * Start automatic health checking
   */
  start() {
    if (this.isRunning) {
      return;
    }

    this.isRunning = true;

    if (this.options.enableAutoCheck) {
      this.intervalId = setInterval(
        () => this.checkAll(),
        this.options.checkInterval
      );

      // Run initial check
      this.checkAll().catch(error => {
        this.emit('error', error);
      });
    }

    this.emit('checker:started');
  }

  /**
   * Stop automatic health checking
   */
  stop() {
    if (!this.isRunning) {
      return;
    }

    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }

    this.isRunning = false;
    this.emit('checker:stopped');
  }

  /**
   * Clear all check results
   */
  clearResults() {
    this.results.clear();
    this.emit('results:cleared');
  }

  /**
   * Utility: Delay for specified milliseconds
   * @param {number} ms - Milliseconds to delay
   * @returns {Promise<void>}
   */
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * Create common health checks for typical services
 */
class CommonHealthChecks {
  /**
   * PostgreSQL health check
   * @param {Object} pool - PostgreSQL pool instance
   * @returns {Function} Health check function
   */
  static postgres(pool) {
    return async () => {
      const client = await pool.connect();

      try {
        const result = await client.query('SELECT NOW(), version()');
        const version = result.rows[0].version;

        return {
          status: HealthStatus.HEALTHY,
          message: 'PostgreSQL connection healthy',
          metadata: {
            version: version.split(' ')[1],
            responseTime: result.duration || 0
          }
        };
      } finally {
        client.release();
      }
    };
  }

  /**
   * Redis health check
   * @param {Object} client - Redis client instance
   * @returns {Function} Health check function
   */
  static redis(client) {
    return async () => {
      const start = Date.now();
      const pong = await client.ping();
      const duration = Date.now() - start;

      if (pong !== 'PONG') {
        throw new Error('Redis ping failed');
      }

      const info = await client.info('server');
      const version = info.match(/redis_version:(\S+)/)?.[1];

      return {
        status: HealthStatus.HEALTHY,
        message: 'Redis connection healthy',
        metadata: {
          version,
          responseTime: duration
        }
      };
    };
  }

  /**
   * HTTP endpoint health check
   * @param {string} url - URL to check
   * @param {Object} options - Fetch options
   * @returns {Function} Health check function
   */
  static httpEndpoint(url, options = {}) {
    return async () => {
      const fetch = require('node-fetch');
      const start = Date.now();

      const response = await fetch(url, {
        method: options.method || 'GET',
        timeout: options.timeout || 5000,
        ...options
      });

      const duration = Date.now() - start;

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return {
        status: HealthStatus.HEALTHY,
        message: `Endpoint ${url} responding`,
        metadata: {
          statusCode: response.status,
          responseTime: duration
        }
      };
    };
  }

  /**
   * Memory usage health check
   * @param {number} threshold - Memory usage threshold (0-1)
   * @returns {Function} Health check function
   */
  static memoryUsage(threshold = 0.9) {
    return async () => {
      const usage = process.memoryUsage();
      const heapUsedPercent = usage.heapUsed / usage.heapTotal;

      let status = HealthStatus.HEALTHY;
      let message = 'Memory usage normal';

      if (heapUsedPercent > threshold) {
        status = HealthStatus.UNHEALTHY;
        message = 'Memory usage critical';
      } else if (heapUsedPercent > threshold * 0.8) {
        status = HealthStatus.DEGRADED;
        message = 'Memory usage elevated';
      }

      return {
        status,
        message,
        metadata: {
          heapUsed: Math.round(usage.heapUsed / 1024 / 1024) + 'MB',
          heapTotal: Math.round(usage.heapTotal / 1024 / 1024) + 'MB',
          heapUsedPercent: Math.round(heapUsedPercent * 100) + '%',
          rss: Math.round(usage.rss / 1024 / 1024) + 'MB'
        }
      };
    };
  }

  /**
   * Disk space health check
   * @param {string} path - Path to check
   * @param {number} threshold - Free space threshold (0-1)
   * @returns {Function} Health check function
   */
  static diskSpace(path = '/', threshold = 0.1) {
    return async () => {
      const checkDiskSpace = require('check-disk-space').default;
      const diskSpace = await checkDiskSpace(path);

      const freePercent = diskSpace.free / diskSpace.size;

      let status = HealthStatus.HEALTHY;
      let message = 'Disk space adequate';

      if (freePercent < threshold) {
        status = HealthStatus.UNHEALTHY;
        message = 'Disk space critical';
      } else if (freePercent < threshold * 2) {
        status = HealthStatus.DEGRADED;
        message = 'Disk space low';
      }

      return {
        status,
        message,
        metadata: {
          free: Math.round(diskSpace.free / 1024 / 1024 / 1024) + 'GB',
          total: Math.round(diskSpace.size / 1024 / 1024 / 1024) + 'GB',
          freePercent: Math.round(freePercent * 100) + '%'
        }
      };
    };
  }
}

/**
 * Create a health checker with default checks
 * @param {Object} services - Service instances
 * @param {Object} options - Health checker options
 * @returns {HealthChecker} Configured health checker
 */
function createHealthChecker(services = {}, options = {}) {
  const checker = new HealthChecker(options);

  // Register PostgreSQL check if available
  if (services.postgres) {
    checker.registerCheck(
      'postgresql',
      CommonHealthChecks.postgres(services.postgres),
      { type: ComponentType.DATABASE, critical: true }
    );
  }

  // Register Redis check if available
  if (services.redis) {
    checker.registerCheck(
      'redis',
      CommonHealthChecks.redis(services.redis),
      { type: ComponentType.CACHE, critical: true }
    );
  }

  // Register memory check
  checker.registerCheck(
    'memory',
    CommonHealthChecks.memoryUsage(0.9),
    { type: ComponentType.SERVICE, critical: false }
  );

  return checker;
}

module.exports = {
  HealthChecker,
  CommonHealthChecks,
  createHealthChecker,
  HealthStatus,
  ComponentType
};
