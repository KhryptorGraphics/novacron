/**
 * NovaCron Configuration Loader
 *
 * Handles loading, merging, and processing of configuration files
 * with support for environment-specific overrides and runtime variables.
 *
 * @module config/loader
 */

const fs = require('fs');
const path = require('path');
const { EventEmitter } = require('events');

/**
 * Configuration loader class
 */
class ConfigLoader extends EventEmitter {
  constructor(options = {}) {
    super();

    this.options = {
      configPath: options.configPath || path.join(process.cwd(), 'src', 'config'),
      environment: options.environment || process.env.NODE_ENV || 'development',
      envPrefix: options.envPrefix || 'NOVACRON_',
      allowMissingConfig: options.allowMissingConfig === true,
      validateOnLoad: options.validateOnLoad !== false,
      cacheConfig: options.cacheConfig !== false,
      ...options
    };

    this.config = null;
    this.loadedFiles = [];
    this.envVarsApplied = [];
  }

  /**
   * Load configuration from files and environment
   * @returns {Promise<Object>} Loaded configuration
   */
  async load() {
    this.emit('load:started', {
      environment: this.options.environment,
      configPath: this.options.configPath
    });

    try {
      // Load default configuration
      const defaultConfig = await this.loadConfigFile('config.default.json');

      // Load environment-specific configuration
      const envConfig = await this.loadConfigFile(
        `config.${this.options.environment}.json`,
        this.options.allowMissingConfig
      );

      // Merge configurations
      this.config = this.mergeConfigs(defaultConfig, envConfig);

      // Apply environment variables
      this.applyEnvironmentVariables();

      // Apply runtime overrides if provided
      if (this.options.overrides) {
        this.config = this.mergeConfigs(this.config, this.options.overrides);
      }

      this.emit('load:completed', {
        config: this.getSafeConfig(),
        filesLoaded: this.loadedFiles.length,
        envVarsApplied: this.envVarsApplied.length
      });

      return this.config;

    } catch (error) {
      this.emit('load:failed', { error: error.message });
      throw new ConfigurationLoadError(
        `Failed to load configuration: ${error.message}`,
        error
      );
    }
  }

  /**
   * Load a single configuration file
   * @param {string} filename - Configuration filename
   * @param {boolean} optional - Whether file is optional
   * @returns {Promise<Object>} Loaded configuration
   */
  async loadConfigFile(filename, optional = false) {
    const filePath = path.join(this.options.configPath, filename);

    try {
      // Check if file exists
      if (!fs.existsSync(filePath)) {
        if (optional) {
          this.emit('file:skipped', { filename, reason: 'not found' });
          return {};
        }
        throw new Error(`Configuration file not found: ${filePath}`);
      }

      // Read file content
      const content = fs.readFileSync(filePath, 'utf8');

      // Parse based on file extension
      let config;
      const ext = path.extname(filename).toLowerCase();

      switch (ext) {
        case '.json':
          config = JSON.parse(content);
          break;
        case '.js':
          config = require(filePath);
          break;
        case '.yaml':
        case '.yml':
          const yaml = require('js-yaml');
          config = yaml.load(content);
          break;
        default:
          throw new Error(`Unsupported config file format: ${ext}`);
      }

      this.loadedFiles.push({
        filename,
        path: filePath,
        size: content.length,
        timestamp: new Date().toISOString()
      });

      this.emit('file:loaded', { filename, path: filePath });

      return config;

    } catch (error) {
      if (optional) {
        this.emit('file:skipped', {
          filename,
          reason: error.message
        });
        return {};
      }

      throw new Error(`Failed to load ${filename}: ${error.message}`);
    }
  }

  /**
   * Merge multiple configuration objects
   * @param {Object} base - Base configuration
   * @param {Object} override - Override configuration
   * @returns {Object} Merged configuration
   */
  mergeConfigs(base, override) {
    if (!override || Object.keys(override).length === 0) {
      return base;
    }

    const merged = { ...base };

    for (const [key, value] of Object.entries(override)) {
      if (value === null || value === undefined) {
        continue;
      }

      // Deep merge for objects
      if (
        typeof value === 'object' &&
        !Array.isArray(value) &&
        typeof merged[key] === 'object' &&
        !Array.isArray(merged[key])
      ) {
        merged[key] = this.mergeConfigs(merged[key] || {}, value);
      } else {
        merged[key] = value;
      }
    }

    return merged;
  }

  /**
   * Apply environment variables to configuration
   */
  applyEnvironmentVariables() {
    const prefix = this.options.envPrefix;

    Object.keys(process.env)
      .filter(key => key.startsWith(prefix))
      .forEach(key => {
        const configKey = this.envKeyToConfigPath(key, prefix);
        const value = this.parseEnvValue(process.env[key]);

        this.setConfigValue(configKey, value);

        this.envVarsApplied.push({
          envVar: key,
          configPath: configKey,
          value: this.redactSensitiveValue(key, value)
        });
      });

    if (this.envVarsApplied.length > 0) {
      this.emit('env:applied', {
        count: this.envVarsApplied.length,
        variables: this.envVarsApplied.map(v => v.envVar)
      });
    }
  }

  /**
   * Convert environment variable key to config path
   * @param {string} envKey - Environment variable key
   * @param {string} prefix - Prefix to remove
   * @returns {string} Config path (dot notation)
   */
  envKeyToConfigPath(envKey, prefix) {
    return envKey
      .substring(prefix.length)
      .toLowerCase()
      .replace(/_/g, '.');
  }

  /**
   * Parse environment variable value
   * @param {string} value - String value
   * @returns {any} Parsed value
   */
  parseEnvValue(value) {
    // Try to parse as JSON
    if (value.startsWith('{') || value.startsWith('[')) {
      try {
        return JSON.parse(value);
      } catch {
        // Not valid JSON, return as string
      }
    }

    // Parse booleans
    if (value === 'true') return true;
    if (value === 'false') return false;

    // Parse numbers
    if (/^-?\d+(\.\d+)?$/.test(value)) {
      return Number(value);
    }

    // Return as string
    return value;
  }

  /**
   * Set configuration value by path
   * @param {string} path - Dot notation path
   * @param {any} value - Value to set
   */
  setConfigValue(path, value) {
    const parts = path.split('.');
    let current = this.config;

    for (let i = 0; i < parts.length - 1; i++) {
      const part = parts[i];

      if (!current[part] || typeof current[part] !== 'object') {
        current[part] = {};
      }

      current = current[part];
    }

    current[parts[parts.length - 1]] = value;
  }

  /**
   * Get configuration value by path
   * @param {string} path - Dot notation path
   * @param {any} defaultValue - Default value if not found
   * @returns {any} Configuration value
   */
  get(path, defaultValue = undefined) {
    if (!this.config) {
      throw new Error('Configuration not loaded. Call load() first.');
    }

    const parts = path.split('.');
    let current = this.config;

    for (const part of parts) {
      if (current === null || current === undefined || !(part in current)) {
        return defaultValue;
      }
      current = current[part];
    }

    return current;
  }

  /**
   * Check if configuration has a value at path
   * @param {string} path - Dot notation path
   * @returns {boolean} Whether value exists
   */
  has(path) {
    return this.get(path, undefined) !== undefined;
  }

  /**
   * Get entire configuration object
   * @returns {Object} Configuration
   */
  getAll() {
    if (!this.config) {
      throw new Error('Configuration not loaded. Call load() first.');
    }

    return { ...this.config };
  }

  /**
   * Get safe configuration (with sensitive data redacted)
   * @returns {Object} Safe configuration
   */
  getSafeConfig() {
    if (!this.config) {
      return {};
    }

    return this.redactSensitiveFields(this.config);
  }

  /**
   * Redact sensitive fields from configuration
   * @param {Object} obj - Object to redact
   * @returns {Object} Redacted object
   */
  redactSensitiveFields(obj) {
    const sensitiveKeys = [
      'password',
      'secret',
      'token',
      'apikey',
      'api_key',
      'privatekey',
      'private_key'
    ];

    const redacted = Array.isArray(obj) ? [...obj] : { ...obj };

    for (const [key, value] of Object.entries(redacted)) {
      const lowerKey = key.toLowerCase();

      if (sensitiveKeys.some(sk => lowerKey.includes(sk))) {
        redacted[key] = '***REDACTED***';
      } else if (typeof value === 'object' && value !== null) {
        redacted[key] = this.redactSensitiveFields(value);
      }
    }

    return redacted;
  }

  /**
   * Redact sensitive environment variable value
   * @param {string} key - Environment variable key
   * @param {any} value - Value
   * @returns {any} Redacted value
   */
  redactSensitiveValue(key, value) {
    const lowerKey = key.toLowerCase();
    const sensitiveKeys = ['password', 'secret', 'token', 'key'];

    if (sensitiveKeys.some(sk => lowerKey.includes(sk))) {
      return '***REDACTED***';
    }

    return value;
  }

  /**
   * Reload configuration
   * @returns {Promise<Object>} Reloaded configuration
   */
  async reload() {
    this.emit('reload:started');

    this.config = null;
    this.loadedFiles = [];
    this.envVarsApplied = [];

    const config = await this.load();

    this.emit('reload:completed', {
      filesLoaded: this.loadedFiles.length
    });

    return config;
  }

  /**
   * Export configuration to file
   * @param {string} outputPath - Output file path
   * @param {Object} options - Export options
   */
  async export(outputPath, options = {}) {
    const format = options.format || 'json';
    const safe = options.safe !== false;

    const config = safe ? this.getSafeConfig() : this.getAll();

    let content;

    switch (format) {
      case 'json':
        content = JSON.stringify(config, null, 2);
        break;
      case 'yaml':
        const yaml = require('js-yaml');
        content = yaml.dump(config);
        break;
      default:
        throw new Error(`Unsupported export format: ${format}`);
    }

    fs.writeFileSync(outputPath, content, 'utf8');

    this.emit('export:completed', {
      path: outputPath,
      format,
      safe
    });
  }

  /**
   * Validate configuration schema
   * @param {Object} schema - JSON schema
   * @returns {Object} Validation result
   */
  validate(schema) {
    if (!this.config) {
      throw new Error('Configuration not loaded. Call load() first.');
    }

    // This would typically use a library like ajv for JSON schema validation
    // For now, we'll do basic validation

    const errors = [];

    if (schema.required) {
      for (const field of schema.required) {
        if (!this.has(field)) {
          errors.push({
            field,
            error: 'Required field missing'
          });
        }
      }
    }

    const isValid = errors.length === 0;

    const result = {
      valid: isValid,
      errors,
      timestamp: new Date().toISOString()
    };

    this.emit('validation:completed', result);

    return result;
  }
}

/**
 * Custom error class for configuration loading errors
 */
class ConfigurationLoadError extends Error {
  constructor(message, cause) {
    super(message);
    this.name = 'ConfigurationLoadError';
    this.cause = cause;
    this.timestamp = new Date().toISOString();
  }
}

/**
 * Factory function to create and load configuration
 * @param {Object} options - Loader options
 * @returns {Promise<ConfigLoader>} Loaded config loader
 */
async function loadConfig(options = {}) {
  const loader = new ConfigLoader(options);
  await loader.load();
  return loader;
}

module.exports = {
  ConfigLoader,
  ConfigurationLoadError,
  loadConfig
};
