/**
 * Workload Generator - Realistic production workload patterns
 *
 * Generates various workload patterns to simulate real production usage:
 * - User activity patterns (morning peak, evening lull)
 * - Batch processing workloads
 * - Burst traffic patterns
 * - Long-running operations
 *
 * @module WorkloadGenerator
 */

const http = require('k6/http');
const { check, sleep } = require('k6');
const { SharedArray } = require('k6/data');
const { randomIntBetween, randomItem } = require('https://jslib.k6.io/k6-utils/1.2.0/index.js');

// VM configurations for realistic workloads
const vmConfigs = new SharedArray('vmConfigs', function () {
  return [
    { size: 'small', cpu: 2, memory: 4, disk: 50, use_case: 'web_server' },
    { size: 'medium', cpu: 4, memory: 8, disk: 100, use_case: 'application_server' },
    { size: 'large', cpu: 8, memory: 16, disk: 200, use_case: 'database' },
    { size: 'xlarge', cpu: 16, memory: 32, disk: 500, use_case: 'analytics' },
  ];
});

// Workload patterns throughout the day
const workloadPatterns = {
  // 00:00-06:00 - Low activity (20% of baseline)
  nighttime: {
    vusMultiplier: 0.2,
    operationDelay: 10,
    migrationProbability: 0.05,
  },

  // 06:00-09:00 - Morning ramp-up (50% → 100%)
  morning: {
    vusMultiplier: 0.75,
    operationDelay: 5,
    migrationProbability: 0.1,
  },

  // 09:00-12:00 - Peak hours (100%)
  peak: {
    vusMultiplier: 1.0,
    operationDelay: 2,
    migrationProbability: 0.2,
  },

  // 12:00-14:00 - Lunch dip (70%)
  lunch: {
    vusMultiplier: 0.7,
    operationDelay: 4,
    migrationProbability: 0.15,
  },

  // 14:00-18:00 - Afternoon peak (100%)
  afternoon: {
    vusMultiplier: 1.0,
    operationDelay: 2,
    migrationProbability: 0.25,
  },

  // 18:00-24:00 - Evening decline (50% → 20%)
  evening: {
    vusMultiplier: 0.5,
    operationDelay: 7,
    migrationProbability: 0.1,
  },
};

const BASE_URL = __ENV.API_URL || 'https://api.novacron.io';
const API_TOKEN = __ENV.API_TOKEN;

const headers = {
  'Content-Type': 'application/json',
  'Authorization': `Bearer ${API_TOKEN}`,
};

/**
 * Get current workload pattern based on time of day
 */
function getCurrentWorkloadPattern() {
  const hour = new Date().getHours();

  if (hour >= 0 && hour < 6) return workloadPatterns.nighttime;
  if (hour >= 6 && hour < 9) return workloadPatterns.morning;
  if (hour >= 9 && hour < 12) return workloadPatterns.peak;
  if (hour >= 12 && hour < 14) return workloadPatterns.lunch;
  if (hour >= 14 && hour < 18) return workloadPatterns.afternoon;
  return workloadPatterns.evening;
}

/**
 * Simulate realistic user workflow
 */
export function userWorkflow() {
  const pattern = getCurrentWorkloadPattern();
  const config = randomItem(vmConfigs);

  // Create VM with realistic configuration
  const createRes = http.post(`${BASE_URL}/api/v1/vms`, JSON.stringify({
    name: `user-vm-${__VU}-${__ITER}`,
    size: config.size,
    cpu: config.cpu,
    memory: config.memory,
    disk: config.disk,
    tags: {
      use_case: config.use_case,
      user_id: `user-${__VU}`,
      workload: 'production',
    },
  }), { headers });

  check(createRes, {
    'User VM created': (r) => r.status === 201,
  });

  if (createRes.status !== 201) return;

  const vmId = createRes.json('id');

  // Start VM
  http.post(`${BASE_URL}/api/v1/vms/${vmId}/start`, null, { headers });
  sleep(randomIntBetween(3, 8));

  // Simulate user activity
  const activityDuration = randomIntBetween(30, 300); // 30s to 5min
  const activityEnd = Date.now() + (activityDuration * 1000);

  while (Date.now() < activityEnd) {
    // Random API calls simulating user activity
    if (Math.random() < 0.3) {
      http.get(`${BASE_URL}/api/v1/vms/${vmId}`, { headers });
    }

    if (Math.random() < 0.1) {
      http.get(`${BASE_URL}/api/v1/vms/${vmId}/metrics`, { headers });
    }

    if (Math.random() < 0.05) {
      http.post(`${BASE_URL}/api/v1/vms/${vmId}/snapshot`, JSON.stringify({
        name: `snapshot-${Date.now()}`,
      }), { headers });
    }

    sleep(randomIntBetween(5, 15));
  }

  // Possibly migrate VM based on pattern
  if (Math.random() < pattern.migrationProbability) {
    http.post(`${BASE_URL}/api/v1/vms/${vmId}/migrate`, JSON.stringify({
      targetNode: 'auto',
    }), { headers });
    sleep(randomIntBetween(20, 60));
  }

  // Stop and delete VM
  http.post(`${BASE_URL}/api/v1/vms/${vmId}/stop`, null, { headers });
  sleep(2);
  http.del(`${BASE_URL}/api/v1/vms/${vmId}`, null, { headers });

  sleep(pattern.operationDelay);
}

/**
 * Simulate batch processing workload
 */
export function batchProcessingWorkload() {
  const batchSize = randomIntBetween(10, 50);
  const vmIds = [];

  // Create batch of VMs
  for (let i = 0; i < batchSize; i++) {
    const createRes = http.post(`${BASE_URL}/api/v1/vms`, JSON.stringify({
      name: `batch-vm-${__VU}-${__ITER}-${i}`,
      size: 'medium',
      tags: {
        batch_id: `batch-${__VU}-${__ITER}`,
        type: 'batch_processing',
      },
    }), { headers });

    if (createRes.status === 201) {
      vmIds.push(createRes.json('id'));
    }

    sleep(0.5); // Throttle creation
  }

  // Start all VMs
  vmIds.forEach(vmId => {
    http.post(`${BASE_URL}/api/v1/vms/${vmId}/start`, null, { headers });
  });

  // Simulate processing time
  sleep(randomIntBetween(120, 300)); // 2-5 minutes

  // Stop and delete all VMs
  vmIds.forEach(vmId => {
    http.post(`${BASE_URL}/api/v1/vms/${vmId}/stop`, null, { headers });
    sleep(0.5);
    http.del(`${BASE_URL}/api/v1/vms/${vmId}`, null, { headers });
  });
}

/**
 * Simulate burst traffic pattern
 */
export function burstTrafficWorkload() {
  const burstIntensity = randomIntBetween(20, 100);

  for (let i = 0; i < burstIntensity; i++) {
    const createRes = http.post(`${BASE_URL}/api/v1/vms`, JSON.stringify({
      name: `burst-vm-${__VU}-${__ITER}-${i}`,
      size: 'small',
    }), { headers });

    if (createRes.status === 201) {
      const vmId = createRes.json('id');
      http.post(`${BASE_URL}/api/v1/vms/${vmId}/start`, null, { headers });

      // Very short lifecycle
      sleep(randomIntBetween(10, 30));

      http.post(`${BASE_URL}/api/v1/vms/${vmId}/stop`, null, { headers });
      http.del(`${BASE_URL}/api/v1/vms/${vmId}`, null, { headers });
    }

    sleep(0.2); // Minimal delay between creates
  }
}

/**
 * Simulate long-running operation
 */
export function longRunningWorkload() {
  const createRes = http.post(`${BASE_URL}/api/v1/vms`, JSON.stringify({
    name: `longrun-vm-${__VU}-${__ITER}`,
    size: 'xlarge',
    tags: {
      type: 'long_running',
      purpose: 'analytics',
    },
  }), { headers });

  if (createRes.status !== 201) return;

  const vmId = createRes.json('id');

  http.post(`${BASE_URL}/api/v1/vms/${vmId}/start`, null, { headers });

  // Run for extended period with periodic checks
  const runDuration = randomIntBetween(1800, 3600); // 30-60 minutes
  const checkInterval = 300; // Check every 5 minutes
  const checks = Math.floor(runDuration / checkInterval);

  for (let i = 0; i < checks; i++) {
    sleep(checkInterval);
    http.get(`${BASE_URL}/api/v1/vms/${vmId}`, { headers });
    http.get(`${BASE_URL}/api/v1/vms/${vmId}/metrics`, { headers });
  }

  http.post(`${BASE_URL}/api/v1/vms/${vmId}/stop`, null, { headers });
  sleep(2);
  http.del(`${BASE_URL}/api/v1/vms/${vmId}`, null, { headers });
}

module.exports = {
  userWorkflow,
  batchProcessingWorkload,
  burstTrafficWorkload,
  longRunningWorkload,
};
