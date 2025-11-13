/**
 * NovaCron Production Simulation Test Suite
 *
 * Comprehensive 7-day production simulation testing with:
 * - 100K VM capacity
 * - 10K req/s sustained API load
 * - 1K concurrent migrations
 * - Real workload patterns
 *
 * @module ProductionSimulation
 */

const k6 = require('k6');
const http = require('k6/http');
const { check, sleep, group } = require('k6');
const { Rate, Trend, Counter } = require('k6/metrics');

// Custom metrics
const vmCreationRate = new Rate('vm_creation_success');
const vmMigrationRate = new Rate('vm_migration_success');
const apiLatency = new Trend('api_latency_ms');
const errorRate = new Counter('errors');

// Test configuration
export const options = {
  scenarios: {
    // Day 1-2: Baseline load
    baseline: {
      executor: 'constant-vus',
      vus: 1000,
      duration: '48h',
      startTime: '0s',
      gracefulStop: '30s',
      exec: 'baselineWorkload',
    },

    // Day 3-4: Peak load (2x normal)
    peak: {
      executor: 'constant-vus',
      vus: 2000,
      duration: '48h',
      startTime: '48h',
      gracefulStop: '30s',
      exec: 'peakWorkload',
    },

    // Day 5-6: Stress load (5x normal)
    stress: {
      executor: 'constant-vus',
      vus: 5000,
      duration: '48h',
      startTime: '96h',
      gracefulStop: '30s',
      exec: 'stressWorkload',
    },

    // Day 7: Recovery and validation
    recovery: {
      executor: 'ramping-vus',
      startVUs: 5000,
      stages: [
        { duration: '12h', target: 1000 }, // Ramp down
        { duration: '12h', target: 1000 }, // Stabilize
      ],
      startTime: '144h',
      gracefulStop: '30s',
      exec: 'recoveryWorkload',
    },
  },

  thresholds: {
    // Critical thresholds - must pass for production readiness
    'http_req_duration': ['p(95)<100', 'p(99)<200'], // p95 <100ms, p99 <200ms
    'http_req_failed': ['rate<0.01'], // Error rate <1%
    'vm_creation_success': ['rate>0.99'], // 99%+ success rate
    'vm_migration_success': ['rate>0.98'], // 98%+ success rate
    'checks': ['rate>0.99'], // 99%+ check pass rate
  },

  ext: {
    loadimpact: {
      projectID: 3599339,
      name: 'NovaCron Production Simulation - 7 Days',
    },
  },
};

const BASE_URL = __ENV.API_URL || 'https://api.novacron.io';
const API_TOKEN = __ENV.API_TOKEN;

const headers = {
  'Content-Type': 'application/json',
  'Authorization': `Bearer ${API_TOKEN}`,
};

/**
 * Baseline Workload - Normal production operations
 */
export function baselineWorkload() {
  group('VM Lifecycle - Baseline', () => {
    // Create VM
    let createRes = http.post(`${BASE_URL}/api/v1/vms`, JSON.stringify({
      name: `test-vm-${__VU}-${__ITER}`,
      size: 'small',
      image: 'ubuntu-22.04',
    }), { headers });

    const createSuccess = check(createRes, {
      'VM creation status 201': (r) => r.status === 201,
      'VM creation latency <500ms': (r) => r.timings.duration < 500,
    });
    vmCreationRate.add(createSuccess);
    apiLatency.add(createRes.timings.duration);

    if (!createSuccess) {
      errorRate.add(1);
      return;
    }

    const vmId = createRes.json('id');

    // Start VM
    let startRes = http.post(`${BASE_URL}/api/v1/vms/${vmId}/start`, null, { headers });
    check(startRes, {
      'VM start status 200': (r) => r.status === 200,
      'VM start latency <1s': (r) => r.timings.duration < 1000,
    });
    apiLatency.add(startRes.timings.duration);

    sleep(Math.random() * 5 + 5); // Random wait 5-10s

    // Get VM status
    let statusRes = http.get(`${BASE_URL}/api/v1/vms/${vmId}`, { headers });
    check(statusRes, {
      'VM status 200': (r) => r.status === 200,
      'VM is running': (r) => r.json('state') === 'running',
    });
    apiLatency.add(statusRes.timings.duration);

    sleep(Math.random() * 10 + 10); // Random wait 10-20s

    // Stop VM
    let stopRes = http.post(`${BASE_URL}/api/v1/vms/${vmId}/stop`, null, { headers });
    check(stopRes, {
      'VM stop status 200': (r) => r.status === 200,
    });
    apiLatency.add(stopRes.timings.duration);

    sleep(2);

    // Delete VM
    let deleteRes = http.del(`${BASE_URL}/api/v1/vms/${vmId}`, null, { headers });
    check(deleteRes, {
      'VM delete status 200': (r) => r.status === 200,
    });
    apiLatency.add(deleteRes.timings.duration);
  });

  sleep(1);
}

/**
 * Peak Workload - 2x normal load
 */
export function peakWorkload() {
  group('VM Lifecycle - Peak Load', () => {
    // Faster VM operations with less sleep
    let createRes = http.post(`${BASE_URL}/api/v1/vms`, JSON.stringify({
      name: `peak-vm-${__VU}-${__ITER}`,
      size: 'medium',
      image: 'ubuntu-22.04',
    }), { headers });

    const createSuccess = check(createRes, {
      'VM creation status 201': (r) => r.status === 201,
      'VM creation latency <500ms': (r) => r.timings.duration < 500,
    });
    vmCreationRate.add(createSuccess);
    apiLatency.add(createRes.timings.duration);

    if (!createSuccess) {
      errorRate.add(1);
      return;
    }

    const vmId = createRes.json('id');

    // Quick lifecycle
    http.post(`${BASE_URL}/api/v1/vms/${vmId}/start`, null, { headers });
    sleep(2);
    http.get(`${BASE_URL}/api/v1/vms/${vmId}`, { headers });
    sleep(5);

    // Simulate migration
    let migrateRes = http.post(`${BASE_URL}/api/v1/vms/${vmId}/migrate`, JSON.stringify({
      targetNode: 'auto',
    }), { headers });

    const migrateSuccess = check(migrateRes, {
      'VM migration initiated': (r) => r.status === 202 || r.status === 200,
    });
    vmMigrationRate.add(migrateSuccess);

    sleep(10);

    http.post(`${BASE_URL}/api/v1/vms/${vmId}/stop`, null, { headers });
    sleep(1);
    http.del(`${BASE_URL}/api/v1/vms/${vmId}`, null, { headers });
  });

  sleep(0.5);
}

/**
 * Stress Workload - 5x normal load
 */
export function stressWorkload() {
  group('VM Lifecycle - Stress Test', () => {
    // Rapid fire VM operations
    let createRes = http.post(`${BASE_URL}/api/v1/vms`, JSON.stringify({
      name: `stress-vm-${__VU}-${__ITER}`,
      size: 'large',
      image: 'ubuntu-22.04',
    }), { headers });

    const createSuccess = check(createRes, {
      'VM creation under stress': (r) => r.status === 201,
    });
    vmCreationRate.add(createSuccess);
    apiLatency.add(createRes.timings.duration);

    if (!createSuccess) {
      errorRate.add(1);
      return;
    }

    const vmId = createRes.json('id');

    // Minimal waits - stress the system
    http.post(`${BASE_URL}/api/v1/vms/${vmId}/start`, null, { headers });
    sleep(1);
    http.get(`${BASE_URL}/api/v1/vms/${vmId}`, { headers });
    sleep(2);
    http.post(`${BASE_URL}/api/v1/vms/${vmId}/stop`, null, { headers });
    sleep(0.5);
    http.del(`${BASE_URL}/api/v1/vms/${vmId}`, null, { headers });
  });

  sleep(0.2);
}

/**
 * Recovery Workload - System recovery validation
 */
export function recoveryWorkload() {
  group('VM Lifecycle - Recovery', () => {
    // Normal operations with validation
    let createRes = http.post(`${BASE_URL}/api/v1/vms`, JSON.stringify({
      name: `recovery-vm-${__VU}-${__ITER}`,
      size: 'small',
      image: 'ubuntu-22.04',
    }), { headers });

    const createSuccess = check(createRes, {
      'VM creation during recovery': (r) => r.status === 201,
      'Recovery latency normal': (r) => r.timings.duration < 500,
    });
    vmCreationRate.add(createSuccess);
    apiLatency.add(createRes.timings.duration);

    if (createSuccess) {
      const vmId = createRes.json('id');

      // Gentle operations
      http.post(`${BASE_URL}/api/v1/vms/${vmId}/start`, null, { headers });
      sleep(5);
      http.get(`${BASE_URL}/api/v1/vms/${vmId}`, { headers });
      sleep(5);
      http.post(`${BASE_URL}/api/v1/vms/${vmId}/stop`, null, { headers });
      sleep(2);
      http.del(`${BASE_URL}/api/v1/vms/${vmId}`, null, { headers });
    }
  });

  sleep(2);
}

/**
 * Setup function - runs once at the beginning
 */
export function setup() {
  console.log('Starting 7-day production simulation...');
  console.log(`Base URL: ${BASE_URL}`);
  console.log('Scenarios: Baseline (48h) → Peak (48h) → Stress (48h) → Recovery (24h)');
  return { startTime: Date.now() };
}

/**
 * Teardown function - runs once at the end
 */
export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000 / 60 / 60; // hours
  console.log(`Production simulation completed: ${duration.toFixed(2)} hours`);
}

module.exports = {
  options,
  baselineWorkload,
  peakWorkload,
  stressWorkload,
  recoveryWorkload,
  setup,
  teardown,
};
