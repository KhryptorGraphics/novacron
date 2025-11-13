import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics for database operations
const dbErrorRate = new Rate('db_errors');
const queryLatency = new Trend('db_query_latency');
const writeLatency = new Trend('db_write_latency');
const readLatency = new Trend('db_read_latency');
const dbQueries = new Counter('db_queries_total');
const dbWrites = new Counter('db_writes_total');
const dbReads = new Counter('db_reads_total');

export const options = {
  stages: [
    { duration: '2m', target: 200 },    // Warm up database
    { duration: '5m', target: 500 },    // Baseline load
    { duration: '2m', target: 2000 },   // Ramp to 2K concurrent queries
    { duration: '5m', target: 2000 },   // Sustain 2K
    { duration: '2m', target: 5000 },   // High load
    { duration: '3m', target: 5000 },   // Sustain high load
    { duration: '2m', target: 0 },      // Ramp down
  ],
  thresholds: {
    db_errors: ['rate<0.01'],              // <1% error rate
    db_query_latency: ['p(95)<300'],       // 95% queries < 300ms
    db_write_latency: ['p(95)<500'],       // 95% writes < 500ms
    db_read_latency: ['p(95)<200'],        // 95% reads < 200ms
    http_req_duration: ['p(95)<600'],
  },
};

const BASE_URL = __ENV.API_URL || 'http://localhost:8080';
const API_TOKEN = __ENV.API_TOKEN || 'test-token-123';

export function setup() {
  console.log('Setting up database load test...');

  // Create test data set
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${API_TOKEN}`,
  };

  // Seed some VMs for querying
  console.log('Seeding test data...');
  for (let i = 0; i < 100; i++) {
    http.post(`${BASE_URL}/api/v1/vms`, JSON.stringify({
      name: `seed-vm-${i}`,
      cpu_cores: 2,
      memory_mb: 4096,
      disk_gb: 50,
    }), { headers });
  }

  return { startTime: Date.now() };
}

export default function (data) {
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${API_TOKEN}`,
  };

  // Test 1: Simple SELECT query (read-heavy)
  const readStart = Date.now();
  let response = http.get(`${BASE_URL}/api/v1/vms?limit=50`, { headers });
  const readDuration = Date.now() - readStart;

  const readCheck = check(response, {
    'DB read status 200': (r) => r.status === 200,
    'DB read has results': (r) => Array.isArray(r.json('data')),
    'DB read < 200ms': () => readDuration < 200,
  });

  if (!readCheck) {
    dbErrorRate.add(1);
  }

  readLatency.add(readDuration);
  dbReads.add(1);
  dbQueries.add(1);

  sleep(0.5);

  // Test 2: Complex JOIN query (heavy read)
  const complexQueryStart = Date.now();
  response = http.get(
    `${BASE_URL}/api/v1/vms?include=metrics,tags,network&zone=us-west-1&status=running&sort=created_at&order=desc&limit=100`,
    { headers }
  );
  const complexQueryDuration = Date.now() - complexQueryStart;

  check(response, {
    'Complex query status 200': (r) => r.status === 200,
    'Complex query < 300ms': () => complexQueryDuration < 300,
  }) || dbErrorRate.add(1);

  queryLatency.add(complexQueryDuration);
  dbReads.add(1);
  dbQueries.add(1);

  sleep(0.5);

  // Test 3: INSERT query (write operation)
  const writeStart = Date.now();
  const vmPayload = JSON.stringify({
    name: `load-test-vm-${Date.now()}-${__VU}`,
    cpu_cores: 4,
    memory_mb: 8192,
    disk_gb: 100,
    image: 'ubuntu-22.04',
    zone: 'us-west-1',
    tags: ['load-test', 'automated'],
  });

  response = http.post(`${BASE_URL}/api/v1/vms`, vmPayload, { headers });
  const writeDuration = Date.now() - writeStart;

  const writeCheck = check(response, {
    'DB write status 201': (r) => r.status === 201,
    'DB write has ID': (r) => r.json('id') !== undefined,
    'DB write < 500ms': () => writeDuration < 500,
  });

  if (!writeCheck) {
    dbErrorRate.add(1);
  }

  writeLatency.add(writeDuration);
  dbWrites.add(1);
  dbQueries.add(1);

  const newVmId = response.json('id');
  sleep(0.5);

  // Test 4: UPDATE query (write operation)
  if (newVmId) {
    const updateStart = Date.now();
    response = http.patch(
      `${BASE_URL}/api/v1/vms/${newVmId}`,
      JSON.stringify({
        cpu_cores: 8,
        memory_mb: 16384,
        tags: ['load-test', 'automated', 'updated'],
      }),
      { headers }
    );
    const updateDuration = Date.now() - updateStart;

    check(response, {
      'DB update status 200': (r) => r.status === 200,
      'DB update < 400ms': () => updateDuration < 400,
    }) || dbErrorRate.add(1);

    writeLatency.add(updateDuration);
    dbWrites.add(1);
    dbQueries.add(1);
  }

  sleep(0.5);

  // Test 5: Aggregation query
  const aggStart = Date.now();
  response = http.get(`${BASE_URL}/api/v1/vms/stats/aggregate`, { headers });
  const aggDuration = Date.now() - aggStart;

  check(response, {
    'Aggregation query status 200': (r) => r.status === 200,
    'Aggregation has totals': (r) => r.json('total_vms') !== undefined,
    'Aggregation < 500ms': () => aggDuration < 500,
  }) || dbErrorRate.add(1);

  queryLatency.add(aggDuration);
  dbReads.add(1);
  dbQueries.add(1);

  sleep(0.5);

  // Test 6: Full-text search (if supported)
  const searchStart = Date.now();
  response = http.get(`${BASE_URL}/api/v1/vms/search?q=load-test&limit=50`, { headers });
  const searchDuration = Date.now() - searchStart;

  check(response, {
    'Search query status 200': (r) => r.status === 200,
    'Search query < 400ms': () => searchDuration < 400,
  }) || dbErrorRate.add(1);

  queryLatency.add(searchDuration);
  dbReads.add(1);
  dbQueries.add(1);

  sleep(0.5);

  // Test 7: Transaction (multiple operations)
  const txStart = Date.now();
  response = http.post(
    `${BASE_URL}/api/v1/vms/batch`,
    JSON.stringify({
      operations: [
        { action: 'create', data: { name: 'batch-vm-1', cpu_cores: 2, memory_mb: 4096 } },
        { action: 'create', data: { name: 'batch-vm-2', cpu_cores: 2, memory_mb: 4096 } },
        { action: 'create', data: { name: 'batch-vm-3', cpu_cores: 2, memory_mb: 4096 } },
      ],
    }),
    { headers }
  );
  const txDuration = Date.now() - txStart;

  check(response, {
    'Transaction status 200': (r) => r.status === 200 || r.status === 201,
    'Transaction < 1000ms': () => txDuration < 1000,
  }) || dbErrorRate.add(1);

  writeLatency.add(txDuration);
  dbWrites.add(3); // 3 operations
  dbQueries.add(3);

  sleep(1);

  // Test 8: Connection pool stress (rapid sequential queries)
  for (let i = 0; i < 5; i++) {
    const rapidStart = Date.now();
    response = http.get(`${BASE_URL}/api/v1/vms?offset=${i * 10}&limit=10`, { headers });
    const rapidDuration = Date.now() - rapidStart;

    check(response, {
      'Rapid query OK': (r) => r.status === 200,
    }) || dbErrorRate.add(1);

    readLatency.add(rapidDuration);
    dbReads.add(1);
    dbQueries.add(1);

    sleep(0.1); // Very short sleep to stress connection pool
  }

  sleep(1);
}

export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  console.log(`Database load test completed in ${duration}s`);
}

export function handleSummary(data) {
  return {
    'stdout': textSummary(data),
    'results_database.json': JSON.stringify(data, null, 2),
  };
}

function textSummary(data) {
  let summary = '\n=== Database Load Test Summary ===\n\n';

  summary += 'Database Operations:\n';
  summary += `  Total Queries: ${data.metrics.db_queries_total ? data.metrics.db_queries_total.values.count : 0}\n`;
  summary += `  Total Reads: ${data.metrics.db_reads_total ? data.metrics.db_reads_total.values.count : 0}\n`;
  summary += `  Total Writes: ${data.metrics.db_writes_total ? data.metrics.db_writes_total.values.count : 0}\n\n`;

  if (data.metrics.db_query_latency) {
    summary += 'Query Performance:\n';
    summary += `  Overall (p95): ${data.metrics.db_query_latency.values['p(95)'].toFixed(2)}ms\n`;
    summary += `  Overall (avg): ${data.metrics.db_query_latency.values.avg.toFixed(2)}ms\n`;
  }

  if (data.metrics.db_read_latency) {
    summary += `  Read (p95): ${data.metrics.db_read_latency.values['p(95)'].toFixed(2)}ms\n`;
    summary += `  Read (avg): ${data.metrics.db_read_latency.values.avg.toFixed(2)}ms\n`;
  }

  if (data.metrics.db_write_latency) {
    summary += `  Write (p95): ${data.metrics.db_write_latency.values['p(95)'].toFixed(2)}ms\n`;
    summary += `  Write (avg): ${data.metrics.db_write_latency.values.avg.toFixed(2)}ms\n`;
  }

  summary += `  Error Rate: ${(data.metrics.db_errors.values.rate * 100).toFixed(2)}%\n\n`;

  summary += 'HTTP Requests:\n';
  summary += `  Total: ${data.metrics.http_reqs.values.count}\n`;
  summary += `  Failed: ${data.metrics.http_req_failed.values.passes}\n`;
  summary += `  Duration (p95): ${data.metrics.http_req_duration.values['p(95)'].toFixed(2)}ms\n`;

  return summary;
}
