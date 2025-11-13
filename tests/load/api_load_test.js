import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const apiLatency = new Trend('api_latency');
const vmCreationTime = new Trend('vm_creation_time');
const failedRequests = new Counter('failed_requests');

export const options = {
  stages: [
    { duration: '2m', target: 100 },    // Warm up to 100 users
    { duration: '5m', target: 100 },    // Stay at 100 for baseline
    { duration: '2m', target: 1000 },   // Ramp to 1K users
    { duration: '5m', target: 1000 },   // Sustain 1K users
    { duration: '2m', target: 10000 },  // Ramp to 10K users
    { duration: '5m', target: 10000 },  // Sustain 10K users
    { duration: '2m', target: 0 },      // Ramp down gracefully
  ],
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'], // 95% < 500ms, 99% < 1s
    http_req_failed: ['rate<0.01'],                  // <1% errors
    errors: ['rate<0.05'],                           // <5% error rate
    api_latency: ['p(95)<600'],
    vm_creation_time: ['p(95)<2000'],
  },
};

const BASE_URL = __ENV.API_URL || 'http://localhost:8080';
const API_TOKEN = __ENV.API_TOKEN || 'test-token-123';

export function setup() {
  // Warm up the API
  console.log('Setting up load test environment...');
  const warmupRes = http.get(`${BASE_URL}/api/v1/health`);
  console.log(`Health check status: ${warmupRes.status}`);
  return { startTime: Date.now() };
}

export default function (data) {
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${API_TOKEN}`,
  };

  // Test 1: List VMs (most common operation)
  let response = http.get(`${BASE_URL}/api/v1/vms`, { headers });
  const listCheck = check(response, {
    'VM list status 200': (r) => r.status === 200,
    'VM list has data': (r) => r.json('data') !== undefined,
    'VM list response time < 500ms': (r) => r.timings.duration < 500,
  });

  if (!listCheck) {
    errorRate.add(1);
    failedRequests.add(1);
  }
  apiLatency.add(response.timings.duration);

  sleep(1);

  // Test 2: Get specific VM details
  const vmId = Math.floor(Math.random() * 1000) + 1;
  response = http.get(`${BASE_URL}/api/v1/vms/${vmId}`, { headers });
  check(response, {
    'VM detail status 200 or 404': (r) => r.status === 200 || r.status === 404,
    'VM detail response time < 300ms': (r) => r.timings.duration < 300,
  }) || errorRate.add(1);

  apiLatency.add(response.timings.duration);
  sleep(0.5);

  // Test 3: Create VM (write operation)
  const vmPayload = JSON.stringify({
    name: `test-vm-${Date.now()}-${__VU}`,
    cpu_cores: 4,
    memory_mb: 8192,
    disk_gb: 100,
    image: 'ubuntu-22.04',
    zone: 'us-west-1',
  });

  const startTime = Date.now();
  response = http.post(`${BASE_URL}/api/v1/vms`, vmPayload, { headers });
  const creationTime = Date.now() - startTime;

  const createCheck = check(response, {
    'VM create status 201': (r) => r.status === 201,
    'VM create has id': (r) => r.json('id') !== undefined,
    'VM create time < 2s': () => creationTime < 2000,
  });

  if (!createCheck) {
    errorRate.add(1);
    failedRequests.add(1);
  }
  vmCreationTime.add(creationTime);

  const newVmId = response.json('id');
  sleep(1);

  // Test 4: Update VM (if created successfully)
  if (newVmId) {
    const updatePayload = JSON.stringify({
      cpu_cores: 8,
      memory_mb: 16384,
    });

    response = http.patch(`${BASE_URL}/api/v1/vms/${newVmId}`, updatePayload, { headers });
    check(response, {
      'VM update status 200': (r) => r.status === 200,
      'VM update response time < 400ms': (r) => r.timings.duration < 400,
    }) || errorRate.add(1);

    apiLatency.add(response.timings.duration);
    sleep(0.5);

    // Test 5: Delete VM (cleanup)
    response = http.del(`${BASE_URL}/api/v1/vms/${newVmId}`, null, { headers });
    check(response, {
      'VM delete status 204': (r) => r.status === 204,
    }) || errorRate.add(1);
  }

  sleep(2);

  // Test 6: Search/Filter VMs
  response = http.get(`${BASE_URL}/api/v1/vms?zone=us-west-1&status=running`, { headers });
  check(response, {
    'VM search status 200': (r) => r.status === 200,
    'VM search response time < 600ms': (r) => r.timings.duration < 600,
  }) || errorRate.add(1);

  apiLatency.add(response.timings.duration);
  sleep(1);

  // Test 7: Get VM metrics/stats
  response = http.get(`${BASE_URL}/api/v1/vms/${vmId}/metrics`, { headers });
  check(response, {
    'VM metrics status 200 or 404': (r) => r.status === 200 || r.status === 404,
  }) || errorRate.add(1);

  sleep(1);
}

export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  console.log(`Load test completed in ${duration}s`);
}

export function handleSummary(data) {
  return {
    'stdout': textSummary(data, { indent: ' ', enableColors: true }),
    'results_api.json': JSON.stringify(data, null, 2),
  };
}

function textSummary(data, options) {
  const { indent = '', enableColors = false } = options;
  let summary = '\n' + indent + '=== API Load Test Summary ===\n\n';

  // HTTP metrics
  summary += indent + 'HTTP Requests:\n';
  summary += indent + `  Total: ${data.metrics.http_reqs.values.count}\n`;
  summary += indent + `  Failed: ${data.metrics.http_req_failed.values.passes} (${(data.metrics.http_req_failed.values.rate * 100).toFixed(2)}%)\n`;
  summary += indent + `  Duration (p95): ${data.metrics.http_req_duration.values['p(95)'].toFixed(2)}ms\n`;
  summary += indent + `  Duration (p99): ${data.metrics.http_req_duration.values['p(99)'].toFixed(2)}ms\n\n`;

  // Custom metrics
  summary += indent + 'Custom Metrics:\n';
  if (data.metrics.api_latency) {
    summary += indent + `  API Latency (p95): ${data.metrics.api_latency.values['p(95)'].toFixed(2)}ms\n`;
  }
  if (data.metrics.vm_creation_time) {
    summary += indent + `  VM Creation (p95): ${data.metrics.vm_creation_time.values['p(95)'].toFixed(2)}ms\n`;
  }
  summary += indent + `  Error Rate: ${(data.metrics.errors.values.rate * 100).toFixed(2)}%\n`;

  return summary;
}
