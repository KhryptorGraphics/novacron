import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import ws from 'k6/ws';
import { randomString } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Custom metrics for DWCP protocol
const dwcpErrorRate = new Rate('dwcp_errors');
const migrationTime = new Trend('migration_duration');
const protocolLatency = new Trend('dwcp_protocol_latency');
const vmMigrations = new Counter('vm_migrations_total');
const failedMigrations = new Counter('failed_migrations');

export const options = {
  stages: [
    { duration: '2m', target: 50 },     // Warm up - migrations are heavy
    { duration: '5m', target: 100 },    // Baseline: 100 concurrent migrations
    { duration: '2m', target: 500 },    // Ramp to 500
    { duration: '5m', target: 500 },    // Sustain 500
    { duration: '2m', target: 1000 },   // Stress test: 1K migrations
    { duration: '3m', target: 1000 },   // Sustain stress
    { duration: '2m', target: 0 },      // Ramp down
  ],
  thresholds: {
    dwcp_errors: ['rate<0.02'],              // <2% error rate
    migration_duration: ['p(95)<30000'],     // 95% of migrations < 30s
    dwcp_protocol_latency: ['p(95)<1000'],   // Protocol overhead < 1s
    vm_migrations_total: ['count>0'],        // At least some migrations
  },
};

const BASE_URL = __ENV.API_URL || 'http://localhost:8080';
const DWCP_WS_URL = __ENV.DWCP_WS_URL || 'ws://localhost:8080/dwcp/v3';
const API_TOKEN = __ENV.API_TOKEN || 'test-token-123';

export function setup() {
  console.log('Setting up DWCP load test...');

  // Verify DWCP endpoint is available
  const response = http.get(`${BASE_URL}/api/v1/dwcp/version`);
  console.log(`DWCP version check: ${response.status}`);

  return {
    startTime: Date.now(),
    dwcpVersion: response.status === 200 ? response.json('version') : 'unknown'
  };
}

export default function (data) {
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${API_TOKEN}`,
  };

  // Test 1: DWCP v3 Protocol - VM Migration via REST API
  const sourceZone = ['us-west-1', 'us-east-1', 'eu-west-1'][Math.floor(Math.random() * 3)];
  const targetZone = sourceZone === 'us-west-1' ? 'us-east-1' : 'us-west-1';

  const migrationPayload = JSON.stringify({
    vm_id: `test-vm-${randomString(8)}`,
    source_zone: sourceZone,
    target_zone: targetZone,
    protocol: 'dwcp-v3',
    live_migration: true,
    bandwidth_limit_mbps: 1000,
    compression: true,
    encryption: true,
  });

  const migrationStart = Date.now();
  let response = http.post(`${BASE_URL}/api/v1/dwcp/migrations`, migrationPayload, {
    headers,
    timeout: '60s',
  });

  const migrationCheck = check(response, {
    'Migration initiated (202)': (r) => r.status === 202,
    'Migration has job ID': (r) => r.json('job_id') !== undefined,
    'Protocol latency < 1s': (r) => r.timings.duration < 1000,
  });

  if (!migrationCheck) {
    dwcpErrorRate.add(1);
    failedMigrations.add(1);
  }

  protocolLatency.add(response.timings.duration);
  const jobId = response.json('job_id');

  sleep(1);

  // Test 2: Poll migration status
  if (jobId) {
    let migrationComplete = false;
    let attempts = 0;
    const maxAttempts = 30; // 30 seconds max

    while (!migrationComplete && attempts < maxAttempts) {
      response = http.get(`${BASE_URL}/api/v1/dwcp/migrations/${jobId}`, { headers });

      if (response.status === 200) {
        const status = response.json('status');

        if (status === 'completed') {
          migrationComplete = true;
          const duration = Date.now() - migrationStart;
          migrationTime.add(duration);
          vmMigrations.add(1);

          check(response, {
            'Migration completed successfully': (r) => r.json('status') === 'completed',
            'Migration duration < 30s': () => duration < 30000,
            'No data loss': (r) => r.json('data_loss') === false,
          }) || dwcpErrorRate.add(1);

        } else if (status === 'failed') {
          dwcpErrorRate.add(1);
          failedMigrations.add(1);
          break;
        }
      }

      attempts++;
      sleep(1);
    }

    if (!migrationComplete && attempts >= maxAttempts) {
      dwcpErrorRate.add(1);
      failedMigrations.add(1);
    }
  }

  sleep(2);

  // Test 3: DWCP Protocol Statistics
  response = http.get(`${BASE_URL}/api/v1/dwcp/stats`, { headers });
  check(response, {
    'DWCP stats available': (r) => r.status === 200,
    'Stats have active migrations': (r) => r.json('active_migrations') !== undefined,
  }) || dwcpErrorRate.add(1);

  sleep(1);

  // Test 4: WebSocket-based DWCP streaming (for live migration monitoring)
  const wsUrl = `${DWCP_WS_URL}/stream?job_id=${jobId || 'test'}`;

  ws.connect(wsUrl, { headers: { 'Authorization': `Bearer ${API_TOKEN}` } }, function (socket) {
    socket.on('open', () => {
      // Subscribe to migration events
      socket.send(JSON.stringify({
        action: 'subscribe',
        channel: 'migrations',
      }));
    });

    socket.on('message', (data) => {
      const message = JSON.parse(data);
      check(message, {
        'WS message has type': (m) => m.type !== undefined,
        'WS message has timestamp': (m) => m.timestamp !== undefined,
      }) || dwcpErrorRate.add(1);
    });

    socket.on('error', (e) => {
      dwcpErrorRate.add(1);
      console.log('WebSocket error:', e);
    });

    // Keep connection open for 5 seconds to receive updates
    socket.setTimeout(() => {
      socket.close();
    }, 5000);
  });

  sleep(2);

  // Test 5: Batch migration status check
  response = http.get(`${BASE_URL}/api/v1/dwcp/migrations?status=in_progress&limit=100`, { headers });
  check(response, {
    'Batch status check OK': (r) => r.status === 200,
    'Batch response has migrations': (r) => Array.isArray(r.json('migrations')),
  }) || dwcpErrorRate.add(1);

  sleep(1);
}

export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  console.log(`DWCP load test completed in ${duration}s`);
  console.log(`DWCP Version tested: ${data.dwcpVersion}`);
}

export function handleSummary(data) {
  return {
    'stdout': textSummary(data),
    'results_dwcp.json': JSON.stringify(data, null, 2),
  };
}

function textSummary(data) {
  let summary = '\n=== DWCP Protocol Load Test Summary ===\n\n';

  summary += 'DWCP Metrics:\n';
  summary += `  Total Migrations: ${data.metrics.vm_migrations_total ? data.metrics.vm_migrations_total.values.count : 0}\n`;
  summary += `  Failed Migrations: ${data.metrics.failed_migrations ? data.metrics.failed_migrations.values.count : 0}\n`;

  if (data.metrics.migration_duration) {
    summary += `  Migration Duration (p95): ${(data.metrics.migration_duration.values['p(95)'] / 1000).toFixed(2)}s\n`;
    summary += `  Migration Duration (avg): ${(data.metrics.migration_duration.values.avg / 1000).toFixed(2)}s\n`;
  }

  if (data.metrics.dwcp_protocol_latency) {
    summary += `  Protocol Latency (p95): ${data.metrics.dwcp_protocol_latency.values['p(95)'].toFixed(2)}ms\n`;
  }

  summary += `  Error Rate: ${(data.metrics.dwcp_errors.values.rate * 100).toFixed(2)}%\n\n`;

  summary += 'HTTP Requests:\n';
  summary += `  Total: ${data.metrics.http_reqs.values.count}\n`;
  summary += `  Failed: ${data.metrics.http_req_failed.values.passes}\n`;

  return summary;
}
