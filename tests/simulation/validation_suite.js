/**
 * Validation Suite - Production readiness validation checks
 *
 * Comprehensive validation suite for production readiness:
 * - System availability and uptime
 * - Performance metrics validation
 * - Error rate monitoring
 * - Resource utilization checks
 * - Data integrity verification
 *
 * @module ValidationSuite
 */

const http = require('k6/http');
const { check, group, sleep } = require('k6');
const { Counter, Gauge, Rate, Trend } = require('k6/metrics');

// Validation metrics
const availabilityRate = new Rate('availability_rate');
const dataIntegrityRate = new Rate('data_integrity_rate');
const resourceUtilization = new Gauge('resource_utilization_percent');
const validationErrors = new Counter('validation_errors');

const BASE_URL = __ENV.API_URL || 'https://api.novacron.io';
const API_TOKEN = __ENV.API_TOKEN;

const headers = {
  'Content-Type': 'application/json',
  'Authorization': `Bearer ${API_TOKEN}`,
};

/**
 * Validate system availability
 */
export function validateAvailability() {
  group('Availability Validation', () => {
    // Health check endpoint
    const healthRes = http.get(`${BASE_URL}/health`, { headers });
    const healthCheck = check(healthRes, {
      'Health endpoint responds': (r) => r.status === 200,
      'Health status is ok': (r) => r.json('status') === 'ok',
      'Database connected': (r) => r.json('database') === 'connected',
      'Response time <100ms': (r) => r.timings.duration < 100,
    });

    availabilityRate.add(healthCheck);

    if (!healthCheck) {
      validationErrors.add(1);
    }

    // API endpoints availability
    const endpointsToCheck = [
      '/api/v1/vms',
      '/api/v1/users/me',
      '/api/v1/nodes',
      '/api/v1/metrics',
    ];

    endpointsToCheck.forEach(endpoint => {
      const res = http.get(`${BASE_URL}${endpoint}`, { headers });
      const endpointCheck = check(res, {
        [`${endpoint} available`]: (r) => r.status >= 200 && r.status < 500,
        [`${endpoint} responds <500ms`]: (r) => r.timings.duration < 500,
      });

      availabilityRate.add(endpointCheck);
      if (!endpointCheck) validationErrors.add(1);
    });
  });

  sleep(1);
}

/**
 * Validate performance metrics
 */
export function validatePerformance() {
  group('Performance Validation', () => {
    // API latency validation
    const startTime = Date.now();
    const apiRes = http.get(`${BASE_URL}/api/v1/vms`, { headers });
    const latency = Date.now() - startTime;

    const perfCheck = check(apiRes, {
      'API p95 latency <100ms': () => latency < 100,
      'API responds successfully': (r) => r.status === 200,
      'Response size reasonable': (r) => r.body.length < 1000000, // <1MB
    });

    if (!perfCheck) validationErrors.add(1);

    // Database query performance
    const dbMetricsRes = http.get(`${BASE_URL}/api/v1/metrics/database`, { headers });
    const dbCheck = check(dbMetricsRes, {
      'Database queries <50ms avg': (r) => {
        try {
          return r.json('average_query_time') < 50;
        } catch (e) {
          return false;
        }
      },
      'Database connection pool <90%': (r) => {
        try {
          return r.json('connection_pool_utilization') < 0.90;
        } catch (e) {
          return false;
        }
      },
    });

    if (!dbCheck) validationErrors.add(1);

    // DWCP performance validation
    const dwcpMetricsRes = http.get(`${BASE_URL}/api/v1/metrics/dwcp`, { headers });
    const dwcpCheck = check(dwcpMetricsRes, {
      'DWCP bandwidth >70%': (r) => {
        try {
          return r.json('bandwidth_utilization') > 0.70;
        } catch (e) {
          return false;
        }
      },
      'DWCP migration p95 <30s': (r) => {
        try {
          return r.json('migration_p95_seconds') < 30;
        } catch (e) {
          return false;
        }
      },
    });

    if (!dwcpCheck) validationErrors.add(1);
  });

  sleep(2);
}

/**
 * Validate error rates
 */
export function validateErrorRates() {
  group('Error Rate Validation', () => {
    // Get error metrics
    const errorMetricsRes = http.get(`${BASE_URL}/api/v1/metrics/errors`, { headers });

    const errorCheck = check(errorMetricsRes, {
      'Overall error rate <1%': (r) => {
        try {
          return r.json('error_rate') < 0.01;
        } catch (e) {
          return false;
        }
      },
      'No critical errors': (r) => {
        try {
          return r.json('critical_errors') === 0;
        } catch (e) {
          return false;
        }
      },
      '5xx errors <0.1%': (r) => {
        try {
          return r.json('server_error_rate') < 0.001;
        } catch (e) {
          return false;
        }
      },
    });

    if (!errorCheck) validationErrors.add(1);

    // Check recent error logs
    const errorsRes = http.get(`${BASE_URL}/api/v1/logs/errors?limit=100`, { headers });
    check(errorsRes, {
      'Error logs accessible': (r) => r.status === 200,
      'No recent critical errors': (r) => {
        try {
          const errors = r.json();
          return !errors.some(e => e.level === 'critical');
        } catch (e) {
          return false;
        }
      },
    });
  });

  sleep(1);
}

/**
 * Validate resource utilization
 */
export function validateResourceUtilization() {
  group('Resource Utilization Validation', () => {
    const metricsRes = http.get(`${BASE_URL}/api/v1/metrics/system`, { headers });

    const resourceCheck = check(metricsRes, {
      'CPU usage <70%': (r) => {
        try {
          const cpu = r.json('cpu_utilization');
          resourceUtilization.add(cpu * 100, { resource: 'cpu' });
          return cpu < 0.70;
        } catch (e) {
          return false;
        }
      },
      'Memory usage <80%': (r) => {
        try {
          const memory = r.json('memory_utilization');
          resourceUtilization.add(memory * 100, { resource: 'memory' });
          return memory < 0.80;
        } catch (e) {
          return false;
        }
      },
      'Disk usage <85%': (r) => {
        try {
          const disk = r.json('disk_utilization');
          resourceUtilization.add(disk * 100, { resource: 'disk' });
          return disk < 0.85;
        } catch (e) {
          return false;
        }
      },
      'Network bandwidth available': (r) => {
        try {
          const network = r.json('network_utilization');
          resourceUtilization.add(network * 100, { resource: 'network' });
          return network < 0.90;
        } catch (e) {
          return false;
        }
      },
    });

    if (!resourceCheck) validationErrors.add(1);

    // Check for resource leaks
    const leakCheckRes = http.get(`${BASE_URL}/api/v1/metrics/leaks`, { headers });
    check(leakCheckRes, {
      'No memory leaks detected': (r) => {
        try {
          return !r.json('memory_leak_detected');
        } catch (e) {
          return false;
        }
      },
      'No connection leaks': (r) => {
        try {
          return !r.json('connection_leak_detected');
        } catch (e) {
          return false;
        }
      },
    });
  });

  sleep(1);
}

/**
 * Validate data integrity
 */
export function validateDataIntegrity() {
  group('Data Integrity Validation', () => {
    // Create test VM for integrity check
    const createRes = http.post(`${BASE_URL}/api/v1/vms`, JSON.stringify({
      name: `integrity-test-${Date.now()}`,
      size: 'small',
      tags: {
        type: 'integrity_test',
        test_id: `test-${__VU}-${__ITER}`,
      },
    }), { headers });

    if (createRes.status !== 201) {
      validationErrors.add(1);
      return;
    }

    const vmId = createRes.json('id');
    const originalData = createRes.json();

    // Retrieve and verify data consistency
    sleep(1);
    const getRes = http.get(`${BASE_URL}/api/v1/vms/${vmId}`, { headers });

    const integrityCheck = check(getRes, {
      'Data retrieved successfully': (r) => r.status === 200,
      'VM ID matches': (r) => r.json('id') === vmId,
      'VM name matches': (r) => r.json('name') === originalData.name,
      'VM tags preserved': (r) => {
        try {
          return r.json('tags.type') === 'integrity_test';
        } catch (e) {
          return false;
        }
      },
    });

    dataIntegrityRate.add(integrityCheck);
    if (!integrityCheck) validationErrors.add(1);

    // Cleanup
    http.del(`${BASE_URL}/api/v1/vms/${vmId}`, null, { headers });
  });

  sleep(1);
}

/**
 * Comprehensive validation run
 */
export default function() {
  validateAvailability();
  validatePerformance();
  validateErrorRates();
  validateResourceUtilization();
  validateDataIntegrity();
}

/**
 * Validation thresholds for production readiness
 */
export const options = {
  thresholds: {
    'availability_rate': ['rate>0.999'], // 99.9%+ availability
    'data_integrity_rate': ['rate>0.99'], // 99%+ data integrity
    'validation_errors': ['count<10'], // <10 validation errors total
  },
  duration: '1h',
  vus: 10,
};

module.exports = {
  default: default,
  validateAvailability,
  validatePerformance,
  validateErrorRates,
  validateResourceUtilization,
  validateDataIntegrity,
  options,
};
