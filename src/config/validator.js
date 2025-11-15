/**
 * NovaCron Configuration Validator
 *
 * Provides comprehensive validation for configuration objects,
 * including schema validation, type checking, and business rule validation.
 *
 * @module config/validator
 */

const { EventEmitter } = require('events');

/**
 * Validation severity levels
 */
const ValidationSeverity = {
  ERROR: 'error',
  WARNING: 'warning',
  INFO: 'info'
};

/**
 * Configuration validator class
 */
class ConfigValidator extends EventEmitter {
  constructor(options = {}) {
    super();

    this.options = {
      strictMode: options.strictMode !== false,
      allowUnknownFields: options.allowUnknownFields === true,
      validateTypes: options.validateTypes !== false,
      validateRanges: options.validateRanges !== false,
      ...options
    };

    this.rules = new Map();
    this.validationResults = [];
  }

  /**
   * Add validation rule
   * @param {string} field - Field path (dot notation)
   * @param {Function} validator - Validation function
   * @param {Object} options - Rule options
   */
  addRule(field, validator, options = {}) {
    if (typeof validator !== 'function') {
      throw new Error(`Validator for '${field}' must be a function`);
    }

    this.rules.set(field, {
      field,
      validator,
      required: options.required !== false,
      type: options.type,
      severity: options.severity || ValidationSeverity.ERROR,
      message: options.message,
      ...options
    });

    return this;
  }

  /**
   * Validate configuration object
   * @param {Object} config - Configuration to validate
   * @returns {Object} Validation result
   */
  validate(config) {
    this.validationResults = [];

    this.emit('validation:started', {
      rulesCount: this.rules.size
    });

    // Validate required fields
    this.validateRequiredFields(config);

    // Validate field types
    if (this.options.validateTypes) {
      this.validateTypes(config);
    }

    // Validate custom rules
    this.validateCustomRules(config);

    // Validate ranges
    if (this.options.validateRanges) {
      this.validateRanges(config);
    }

    // Check for unknown fields in strict mode
    if (this.options.strictMode && !this.options.allowUnknownFields) {
      this.checkUnknownFields(config);
    }

    const errors = this.validationResults.filter(
      r => r.severity === ValidationSeverity.ERROR
    );

    const warnings = this.validationResults.filter(
      r => r.severity === ValidationSeverity.WARNING
    );

    const result = {
      valid: errors.length === 0,
      errors,
      warnings,
      info: this.validationResults.filter(
        r => r.severity === ValidationSeverity.INFO
      ),
      summary: {
        total: this.validationResults.length,
        errors: errors.length,
        warnings: warnings.length
      },
      timestamp: new Date().toISOString()
    };

    this.emit('validation:completed', result);

    return result;
  }

  /**
   * Validate required fields
   * @param {Object} config - Configuration object
   */
  validateRequiredFields(config) {
    for (const [field, rule] of this.rules.entries()) {
      if (rule.required) {
        const value = this.getFieldValue(config, field);

        if (value === undefined || value === null) {
          this.addValidationResult({
            field,
            severity: rule.severity,
            message: rule.message || `Required field '${field}' is missing`,
            rule: 'required'
          });
        }
      }
    }
  }

  /**
   * Validate field types
   * @param {Object} config - Configuration object
   */
  validateTypes(config) {
    for (const [field, rule] of this.rules.entries()) {
      if (!rule.type) continue;

      const value = this.getFieldValue(config, field);

      if (value === undefined || value === null) continue;

      const actualType = Array.isArray(value) ? 'array' : typeof value;

      if (actualType !== rule.type) {
        this.addValidationResult({
          field,
          severity: ValidationSeverity.ERROR,
          message: `Field '${field}' must be of type '${rule.type}', got '${actualType}'`,
          rule: 'type',
          expected: rule.type,
          actual: actualType
        });
      }
    }
  }

  /**
   * Validate custom rules
   * @param {Object} config - Configuration object
   */
  validateCustomRules(config) {
    for (const [field, rule] of this.rules.entries()) {
      const value = this.getFieldValue(config, field);

      if (value === undefined || value === null) continue;

      try {
        const result = rule.validator(value, config);

        if (result === false || (typeof result === 'object' && !result.valid)) {
          this.addValidationResult({
            field,
            severity: rule.severity,
            message:
              typeof result === 'object' && result.message
                ? result.message
                : rule.message || `Validation failed for '${field}'`,
            rule: 'custom',
            details: typeof result === 'object' ? result.details : undefined
          });
        }
      } catch (error) {
        this.addValidationResult({
          field,
          severity: ValidationSeverity.ERROR,
          message: `Validation error for '${field}': ${error.message}`,
          rule: 'custom',
          error: error.message
        });
      }
    }
  }

  /**
   * Validate value ranges
   * @param {Object} config - Configuration object
   */
  validateRanges(config) {
    for (const [field, rule] of this.rules.entries()) {
      if (!rule.min && !rule.max) continue;

      const value = this.getFieldValue(config, field);

      if (value === undefined || value === null) continue;

      if (typeof value !== 'number') continue;

      if (rule.min !== undefined && value < rule.min) {
        this.addValidationResult({
          field,
          severity: rule.severity,
          message: `Field '${field}' must be >= ${rule.min}, got ${value}`,
          rule: 'range',
          min: rule.min,
          actual: value
        });
      }

      if (rule.max !== undefined && value > rule.max) {
        this.addValidationResult({
          field,
          severity: rule.severity,
          message: `Field '${field}' must be <= ${rule.max}, got ${value}`,
          rule: 'range',
          max: rule.max,
          actual: value
        });
      }
    }
  }

  /**
   * Check for unknown fields
   * @param {Object} config - Configuration object
   * @param {string} prefix - Field path prefix
   */
  checkUnknownFields(config, prefix = '') {
    const knownFields = new Set(
      Array.from(this.rules.keys())
        .filter(key => key.startsWith(prefix))
        .map(key => key.split('.')[prefix ? prefix.split('.').length : 0])
    );

    for (const key of Object.keys(config)) {
      const fullPath = prefix ? `${prefix}.${key}` : key;

      if (!knownFields.has(key)) {
        this.addValidationResult({
          field: fullPath,
          severity: ValidationSeverity.WARNING,
          message: `Unknown field '${fullPath}'`,
          rule: 'unknown'
        });
      }

      // Recursively check nested objects
      if (typeof config[key] === 'object' && config[key] !== null && !Array.isArray(config[key])) {
        this.checkUnknownFields(config[key], fullPath);
      }
    }
  }

  /**
   * Get field value from config by path
   * @param {Object} config - Configuration object
   * @param {string} path - Dot notation path
   * @returns {any} Field value
   */
  getFieldValue(config, path) {
    const parts = path.split('.');
    let current = config;

    for (const part of parts) {
      if (current === null || current === undefined) {
        return undefined;
      }
      current = current[part];
    }

    return current;
  }

  /**
   * Add validation result
   * @param {Object} result - Validation result
   */
  addValidationResult(result) {
    this.validationResults.push({
      ...result,
      timestamp: new Date().toISOString()
    });

    this.emit('validation:issue', result);
  }

  /**
   * Clear all validation results
   */
  clearResults() {
    this.validationResults = [];
  }

  /**
   * Get validation results
   * @returns {Array} Validation results
   */
  getResults() {
    return [...this.validationResults];
  }
}

/**
 * Common validation rules
 */
class CommonValidators {
  /**
   * Validate string is not empty
   * @param {string} value - Value to validate
   * @returns {boolean} Is valid
   */
  static nonEmptyString(value) {
    return typeof value === 'string' && value.trim().length > 0;
  }

  /**
   * Validate positive number
   * @param {number} value - Value to validate
   * @returns {boolean} Is valid
   */
  static positiveNumber(value) {
    return typeof value === 'number' && value > 0;
  }

  /**
   * Validate port number
   * @param {number} value - Value to validate
   * @returns {boolean|Object} Validation result
   */
  static port(value) {
    if (typeof value !== 'number') {
      return { valid: false, message: 'Port must be a number' };
    }

    if (value < 1 || value > 65535) {
      return {
        valid: false,
        message: 'Port must be between 1 and 65535'
      };
    }

    return true;
  }

  /**
   * Validate URL
   * @param {string} value - Value to validate
   * @returns {boolean|Object} Validation result
   */
  static url(value) {
    if (typeof value !== 'string') {
      return { valid: false, message: 'URL must be a string' };
    }

    try {
      new URL(value);
      return true;
    } catch {
      return { valid: false, message: 'Invalid URL format' };
    }
  }

  /**
   * Validate email
   * @param {string} value - Value to validate
   * @returns {boolean} Is valid
   */
  static email(value) {
    if (typeof value !== 'string') return false;

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(value);
  }

  /**
   * Validate enum value
   * @param {Array} allowedValues - Allowed values
   * @returns {Function} Validator function
   */
  static enum(allowedValues) {
    return (value) => {
      if (!allowedValues.includes(value)) {
        return {
          valid: false,
          message: `Value must be one of: ${allowedValues.join(', ')}`
        };
      }
      return true;
    };
  }

  /**
   * Validate object has required keys
   * @param {Array} requiredKeys - Required keys
   * @returns {Function} Validator function
   */
  static hasKeys(requiredKeys) {
    return (value) => {
      if (typeof value !== 'object' || value === null) {
        return { valid: false, message: 'Value must be an object' };
      }

      const missing = requiredKeys.filter(key => !(key in value));

      if (missing.length > 0) {
        return {
          valid: false,
          message: `Missing required keys: ${missing.join(', ')}`
        };
      }

      return true;
    };
  }

  /**
   * Validate array is not empty
   * @param {Array} value - Value to validate
   * @returns {boolean} Is valid
   */
  static nonEmptyArray(value) {
    return Array.isArray(value) && value.length > 0;
  }

  /**
   * Validate IP address
   * @param {string} value - Value to validate
   * @returns {boolean} Is valid
   */
  static ipAddress(value) {
    if (typeof value !== 'string') return false;

    const ipv4Regex = /^(\d{1,3}\.){3}\d{1,3}$/;
    const ipv6Regex = /^([0-9a-f]{1,4}:){7}[0-9a-f]{1,4}$/i;

    return ipv4Regex.test(value) || ipv6Regex.test(value);
  }

  /**
   * Validate duration string (e.g., "30s", "5m", "1h")
   * @param {string} value - Value to validate
   * @returns {boolean|Object} Validation result
   */
  static duration(value) {
    if (typeof value !== 'string') {
      return { valid: false, message: 'Duration must be a string' };
    }

    const durationRegex = /^(\d+)(ms|s|m|h|d)$/;

    if (!durationRegex.test(value)) {
      return {
        valid: false,
        message: 'Invalid duration format (e.g., "30s", "5m", "1h")'
      };
    }

    return true;
  }
}

/**
 * Create NovaCron platform configuration validator
 * @returns {ConfigValidator} Configured validator
 */
function createPlatformValidator() {
  const validator = new ConfigValidator({ strictMode: true });

  // Database configuration
  validator.addRule('database.postgres.host', CommonValidators.nonEmptyString, {
    type: 'string',
    required: true,
    message: 'PostgreSQL host is required'
  });

  validator.addRule('database.postgres.port', CommonValidators.port, {
    type: 'number',
    required: true
  });

  validator.addRule('database.postgres.database', CommonValidators.nonEmptyString, {
    type: 'string',
    required: true
  });

  validator.addRule('database.postgres.poolSize', CommonValidators.positiveNumber, {
    type: 'number',
    min: 1,
    max: 100
  });

  // Redis configuration
  validator.addRule('database.redis.host', CommonValidators.nonEmptyString, {
    type: 'string',
    required: true
  });

  validator.addRule('database.redis.port', CommonValidators.port, {
    type: 'number',
    required: true
  });

  // Logging configuration
  validator.addRule(
    'logging.level',
    CommonValidators.enum(['debug', 'info', 'warning', 'error']),
    {
      type: 'string',
      required: true
    }
  );

  // API configuration
  validator.addRule('api.port', CommonValidators.port, {
    type: 'number',
    required: true
  });

  validator.addRule('api.host', CommonValidators.nonEmptyString, {
    type: 'string',
    required: true
  });

  return validator;
}

module.exports = {
  ConfigValidator,
  CommonValidators,
  ValidationSeverity,
  createPlatformValidator
};
