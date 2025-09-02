import http from 'k6/http';
import ws from 'k6/ws';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter, Gauge } from 'k6/metrics';
import { config, getEnvironment, getTestData } from '../configs/test-config.js';

// Comprehensive benchmark metrics
const benchmarkMetrics = {
  // API Performance
  apiThroughput: new Rate('api_throughput_rps'),
  apiLatencyP50: new Trend('api_latency_p50'),
  apiLatencyP95: new Trend('api_latency_p95'),
  apiLatencyP99: new Trend('api_latency_p99'),
  
  // VM Operations
  vmCreationThroughput: new Rate('vm_creation_throughput'),
  vmOperationLatency: new Trend('vm_operation_latency'),
  maxConcurrentVMs: new Gauge('max_concurrent_vms'),
  
  // WebSocket Performance
  wsConnectionThroughput: new Rate('ws_connection_throughput'),
  wsMessageThroughput: new Rate('ws_message_throughput'),
  wsMaxConcurrentConnections: new Gauge('ws_max_concurrent_connections'),
  
  // Database Performance
  dbQueryThroughput: new Rate('db_query_throughput'),
  dbTransactionLatency: new Trend('db_transaction_latency'),
  dbConcurrentConnections: new Gauge('db_concurrent_connections'),
  
  // System Resource Utilization
  systemCPUUsage: new Gauge('system_cpu_usage_percent'),
  systemMemoryUsage: new Gauge('system_memory_usage_percent'),
  systemDiskIO: new Gauge('system_disk_io_ops'),
  systemNetworkIO: new Gauge('system_network_io_bps')
};

// Test configuration
export const options = {
  scenarios: {
    // API throughput benchmark
    api_benchmark: {
      executor: 'constant-arrival-rate',
      rate: 100, // 100 requests per second
      timeUnit: '1s',
      duration: '5m',
      preAllocatedVUs: 50,
      maxVUs: 200
    },
    
    // VM operations benchmark
    vm_benchmark: {
      executor: 'ramping-vus',
      startVUs: 10,
      stages: [
        { duration: '2m', target: 50 },
        { duration: '5m', target: 100 },
        { duration: '3m', target: 200 },
        { duration: '2m', target: 0 }
      ]
    },
    
    // WebSocket concurrency benchmark
    ws_benchmark: {
      executor: 'constant-vus',
      vus: 500,
      duration: '5m'
    }
  },
  thresholds: {
    'api_throughput_rps': ['rate>50'], // At least 50 RPS
    'vm_creation_throughput': ['rate>10'], // At least 10 VM/s
    'ws_connection_throughput': ['rate>100'], // At least 100 WS/s
    'api_latency_p95': ['p(95)<500'],
    'vm_operation_latency': ['p(95)<10000'],
    'db_transaction_latency': ['p(95)<100']
  }
};

const environment = getEnvironment();
const testData = getTestData();

// Benchmark state tracking
let benchmarkState = {
  totalAPIRequests: 0,
  totalVMOperations: 0,
  totalWSConnections: 0,
  startTime: Date.now(),
  peakConcurrentVMs: 0,
  peakWSConnections: 0
};

// Authentication helper
function authenticate(userType = 'admin') {
  const user = testData.users[userType];
  const loginPayload = {
    username: user.username,
    password: user.password
  };

  const response = http.post(`${environment.baseURL}/api/auth/login`, 
    JSON.stringify(loginPayload), {
      headers: { 'Content-Type': 'application/json' }
    }
  );

  if (response.status === 200) {
    return response.json().token;
  }
  return null;
}

// API performance benchmark
function benchmarkAPIPerformance(token) {
  const headers = { 'Authorization': `Bearer ${token}` };
  const requests = [];

  // Parallel API requests
  const endpoints = [
    '/api/vms',
    '/api/storage/volumes',
    '/api/cluster/nodes',
    '/api/monitoring/metrics',
    '/api/cluster/health'
  ];

  const batchStart = Date.now();
  
  for (const endpoint of endpoints) {
    const start = Date.now();
    const response = http.get(`${environment.baseURL}${endpoint}`, { headers });
    const duration = Date.now() - start;
    
    benchmarkMetrics.apiLatencyP50.add(duration);
    benchmarkMetrics.apiLatencyP95.add(duration);
    benchmarkMetrics.apiLatencyP99.add(duration);
    
    const success = check(response, {
      [`${endpoint} benchmark successful`]: (r) => r.status === 200,
      [`${endpoint} benchmark within SLA`]: (r) => r.timings.duration < 1000
    });
    
    benchmarkMetrics.apiThroughput.add(success);
    benchmarkState.totalAPIRequests++;
  }

  const batchDuration = Date.now() - batchStart;
  console.log(`API batch completed in ${batchDuration}ms`);
}

// VM operations benchmark
function benchmarkVMOperations(token) {
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  };

  // Create multiple VMs in parallel simulation
  const vmConfigs = [
    testData.vmConfigs[0], // small
    testData.vmConfigs[1]  // medium
  ];

  for (const config of vmConfigs) {
    const vmPayload = {
      name: `benchmark-vm-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      cpu_shares: config.cpu,
      memory_mb: config.memory,
      disk_size_gb: config.disk,
      command: '/bin/sleep',
      args: ['120'], // 2 minute lifespan
      tags: { 'test': 'benchmark', 'config': config.name }
    };

    const operationStart = Date.now();
    const response = http.post(`${environment.baseURL}/api/vms`, 
      JSON.stringify(vmPayload), { headers }
    );
    const duration = Date.now() - operationStart;

    benchmarkMetrics.vmOperationLatency.add(duration);
    
    const success = check(response, {
      'benchmark VM creation successful': (r) => r.status === 201,
      'benchmark VM creation within target': (r) => r.timings.duration < 20000
    });

    benchmarkMetrics.vmCreationThroughput.add(success);
    benchmarkState.totalVMOperations++;

    if (success) {
      benchmarkState.peakConcurrentVMs++;
      benchmarkMetrics.maxConcurrentVMs.add(benchmarkState.peakConcurrentVMs);
    }

    sleep(1);
  }
}

// WebSocket benchmark
function benchmarkWebSocketPerformance(token) {
  const wsEndpoints = [
    { path: '/ws/metrics', params: { interval: '1' } },
    { path: '/ws/alerts', params: { severity: 'warning,error' } }
  ];

  for (const endpoint of wsEndpoints) {
    const queryString = new URLSearchParams(endpoint.params).toString();
    const wsUrl = `${environment.wsURL}${endpoint.path}?${queryString}`;
    
    const connectionStart = Date.now();
    
    ws.connect(wsUrl, { 'Authorization': `Bearer ${token}` }, function (socket) {
      const connectionDuration = Date.now() - connectionStart;
      benchmarkMetrics.wsConnectionThroughput.add(1);
      benchmarkState.totalWSConnections++;
      benchmarkState.peakWSConnections++;
      benchmarkMetrics.wsMaxConcurrentConnections.add(benchmarkState.peakWSConnections);

      let messageCount = 0;
      const sessionStart = Date.now();

      socket.on('open', function () {
        console.log(`Benchmark WebSocket connected: ${endpoint.path}`);
      });

      socket.on('message', function (data) {
        messageCount++;
        benchmarkMetrics.wsMessageThroughput.add(1);
        
        // Validate message structure
        try {
          const message = JSON.parse(data);
          check(message, {
            'benchmark WS message valid': (m) => m.type !== undefined,
            'benchmark WS timestamp present': (m) => m.timestamp !== undefined
          });
        } catch (error) {
          console.error(`Invalid WebSocket message: ${error}`);
        }
      });

      socket.on('error', function (e) {
        console.error(`Benchmark WebSocket error: ${e.error()}`);
      });

      socket.on('close', function () {
        const sessionDuration = Date.now() - sessionStart;
        console.log(`Benchmark WebSocket session: ${sessionDuration}ms, ${messageCount} messages`);
        benchmarkState.peakWSConnections--;
      });

      // Keep connection alive for benchmark duration
      socket.setTimeout(function () {
        socket.close();
      }, 30000 + Math.random() * 10000); // 30-40 seconds
    });

    sleep(1);
  }
}

// System resource monitoring
function monitorSystemResources(token) {
  const headers = { 'Authorization': `Bearer ${token}` };

  const response = http.get(`${environment.baseURL}/api/monitoring/metrics`, { headers });
  
  if (response.status === 200) {
    const metrics = response.json();
    
    // Track system resource utilization during load test
    if (metrics.cpu) {
      benchmarkMetrics.systemCPUUsage.add(metrics.cpu.usage || 0);
    }
    
    if (metrics.memory) {
      const memoryUsagePercent = (metrics.memory.used / metrics.memory.total) * 100;
      benchmarkMetrics.systemMemoryUsage.add(memoryUsagePercent);
    }
    
    if (metrics.disk) {
      benchmarkMetrics.systemDiskIO.add(metrics.disk.iops || 0);
    }
    
    if (metrics.network) {
      const networkIO = (metrics.network.bytes_in + metrics.network.bytes_out) || 0;
      benchmarkMetrics.systemNetworkIO.add(networkIO);
    }
  }
}

// Main benchmark function
export default function() {
  const token = authenticate();
  if (!token) {
    console.error('Authentication failed for benchmark');
    return;
  }

  // Execute benchmark based on scenario
  const scenario = __VU % 100; // Distribute load across scenarios

  if (scenario < 40) { // 40% API benchmarking
    benchmarkAPIPerformance(token);
  } else if (scenario < 70) { // 30% VM benchmarking
    benchmarkVMOperations(token);
  } else if (scenario < 90) { // 20% WebSocket benchmarking
    benchmarkWebSocketPerformance(token);
  } else { // 10% System monitoring
    monitorSystemResources(token);
  }

  // Brief pause between benchmark iterations
  sleep(Math.random() + 0.5);
}

// Setup function
export function setup() {
  console.log('Starting comprehensive NovaCron benchmark suite');
  console.log(`Environment: ${environment.baseURL}`);
  console.log('Benchmark targets:');
  console.log('- API throughput: >50 RPS');
  console.log('- VM creation: >10 VMs/second');
  console.log('- WebSocket connections: >100 WS/second');
  console.log('- Database queries: >500 queries/second');
  
  // Verify all systems are operational
  const healthResponse = http.get(`${environment.baseURL}/api/cluster/health`);
  if (healthResponse.status !== 200) {
    throw new Error(`System not healthy for benchmarking: ${healthResponse.status}`);
  }

  const health = healthResponse.json();
  if (health.status !== 'healthy') {
    throw new Error(`Cluster not healthy: ${health.status}`);
  }
  
  return { 
    startTime: Date.now(),
    clusterHealth: health
  };
}

// Teardown function with comprehensive reporting
export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  
  console.log('\n=== NovaCron Performance Benchmark Results ===');
  console.log(`Total benchmark duration: ${duration} seconds`);
  console.log(`\nAPI Performance:`);
  console.log(`- Total API requests: ${benchmarkState.totalAPIRequests}`);
  console.log(`- Average API throughput: ${(benchmarkState.totalAPIRequests / duration).toFixed(2)} RPS`);
  
  console.log(`\nVM Operations:`);
  console.log(`- Total VM operations: ${benchmarkState.totalVMOperations}`);
  console.log(`- Peak concurrent VMs: ${benchmarkState.peakConcurrentVMs}`);
  console.log(`- VM creation throughput: ${(benchmarkState.totalVMOperations / duration).toFixed(2)} ops/sec`);
  
  console.log(`\nWebSocket Performance:`);
  console.log(`- Total WS connections: ${benchmarkState.totalWSConnections}`);
  console.log(`- Peak concurrent WS: ${benchmarkState.peakWSConnections}`);
  console.log(`- WS connection throughput: ${(benchmarkState.totalWSConnections / duration).toFixed(2)} conn/sec`);
  
  console.log(`\nBenchmark Summary:`);
  console.log(`- Test completed: ${new Date().toISOString()}`);
  console.log(`- Environment: ${environment.baseURL}`);
  console.log(`- Initial cluster health: ${data.clusterHealth.status}`);
  console.log('=== End Benchmark Results ===\n');
}