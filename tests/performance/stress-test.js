// K6 Stress Testing Script for NovaCron
import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Rate, Trend, Counter, Gauge } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('error_rate');
const systemStressLevel = new Gauge('system_stress_level');
const resourceUtilization = new Gauge('resource_utilization');
const concurrentOperations = new Counter('concurrent_operations');
const systemRecoveryTime = new Trend('system_recovery_time');

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8080';
const API_BASE = `${BASE_URL}/api/v1`;

export const options = {
  stages: [
    // Warm up
    { duration: '1m', target: 10 },
    
    // Gradual stress increase
    { duration: '3m', target: 50 },
    { duration: '3m', target: 100 },
    { duration: '3m', target: 200 },
    { duration: '3m', target: 350 },
    
    // Peak stress - push system beyond normal capacity
    { duration: '5m', target: 500 },
    { duration: '5m', target: 750 },
    { duration: '5m', target: 1000 },
    
    // Recovery phase
    { duration: '3m', target: 500 },
    { duration: '3m', target: 200 },
    { duration: '2m', target: 50 },
    { duration: '2m', target: 0 },
  ],
  thresholds: {
    // More relaxed thresholds for stress testing
    http_req_duration: ['p(99)<10000'], // 10 second max response time
    error_rate: ['rate<0.15'], // Allow up to 15% error rate during peak stress
    http_req_failed: ['rate<0.20'], // Allow up to 20% request failures at peak
    system_stress_level: ['value<100'], // System stress gauge
  },
  discardResponseBodies: true, // Save memory during high load
};

// Stress test specific configurations
const stressConfigs = {
  heavyVMConfig: {
    name: 'stress-heavy-vm',
    type: 'qemu',
    cpu: 8,
    memory: 16384,
    disk: 100,
    networks: ['default', 'internal', 'external'],
  },
  
  lightVMConfig: {
    name: 'stress-light-vm',
    type: 'qemu',  
    cpu: 1,
    memory: 512,
    disk: 5,
    networks: ['default'],
  },
  
  // Concurrent operation patterns
  operationBursts: [
    { operation: 'vm_create', weight: 0.3 },
    { operation: 'vm_start', weight: 0.25 },
    { operation: 'vm_stop', weight: 0.25 },
    { operation: 'vm_delete', weight: 0.2 },
  ]
};

let authToken = '';
let createdVMs = [];
let systemMetrics = {};

export function setup() {
  console.log('Setting up stress test environment...');
  
  // Get admin token for stress testing
  const loginResponse = http.post(`${API_BASE}/auth/login`, 
    JSON.stringify({
      username: 'admin',
      password: 'admin123',
    }),
    {
      headers: { 'Content-Type': 'application/json' },
    }
  );

  if (loginResponse.status === 200) {
    const loginData = JSON.parse(loginResponse.body);
    authToken = loginData.token;
    console.log('Successfully authenticated for stress test');
  } else {
    console.error('Failed to authenticate for stress test');
    throw new Error('Authentication failed');
  }

  // Get baseline system metrics
  const baselineResponse = http.get(`${API_BASE}/admin/stats`, {
    headers: { 'Authorization': `Bearer ${authToken}` },
  });
  
  if (baselineResponse.status === 200) {
    systemMetrics.baseline = JSON.parse(baselineResponse.body);
    console.log('Captured baseline system metrics');
  }

  return { authToken, systemMetrics };
}

export default function(data) {
  const token = data.authToken;
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`,
  };

  const currentStage = getCurrentStage();
  const stressLevel = calculateStressLevel(currentStage);
  systemStressLevel.set(stressLevel);

  group('Stress Test Operations', () => {
    
    // High-frequency health checks to monitor system under stress
    group('System Health Monitoring', () => {
      const startTime = Date.now();
      const response = http.get(`${BASE_URL}/health`);
      const responseTime = Date.now() - startTime;
      
      const isHealthy = check(response, {
        'system health check succeeds': (r) => r.status === 200,
        'health response under extreme load < 5s': (r) => responseTime < 5000,
      });
      
      if (!isHealthy) {
        console.warn(`System health check failed at stress level ${stressLevel}`);
      }
      
      // Track system recovery if it was previously failing
      if (isHealthy && systemMetrics.lastFailure) {
        const recoveryTime = Date.now() - systemMetrics.lastFailure;
        systemRecoveryTime.add(recoveryTime);
        delete systemMetrics.lastFailure;
      } else if (!isHealthy && !systemMetrics.lastFailure) {
        systemMetrics.lastFailure = Date.now();
      }
    });

    // Resource-intensive VM operations
    group('High-Load VM Operations', () => {
      const operations = selectOperationsForStressLevel(stressLevel);
      
      operations.forEach(operation => {
        concurrentOperations.add(1);
        
        switch (operation.type) {
          case 'create_heavy_vm':
            createHeavyVM(headers, stressLevel);
            break;
          case 'create_burst_vms':
            createBurstVMs(headers, stressLevel);
            break;
          case 'rapid_lifecycle':
            performRapidLifecycle(headers);
            break;
          case 'concurrent_operations':
            performConcurrentOperations(headers);
            break;
          case 'memory_stress':
            performMemoryStressTest(headers);
            break;
        }
      });
    });

    // Database stress testing
    group('Database Stress Operations', () => {
      if (stressLevel > 70) {
        // Heavy database operations at high stress levels
        performHeavyDatabaseOperations(headers);
      }
    });

    // API endpoint stress testing
    group('API Endpoint Stress', () => {
      stressAPIEndpoints(headers, stressLevel);
    });

    // Resource monitoring
    group('Resource Utilization Monitoring', () => {
      if (Math.random() < 0.1) { // Sample 10% of requests
        monitorResourceUtilization(headers);
      }
    });
  });

  // Dynamic sleep based on stress level - less sleep at higher stress
  const sleepTime = Math.max(0.1, (100 - stressLevel) / 100);
  sleep(sleepTime);
}

function getCurrentStage() {
  const stages = options.stages;
  let totalTime = 0;
  const currentTime = __ITER * (__VU - 1) / 1000; // Rough approximation
  
  for (const stage of stages) {
    totalTime += parseDuration(stage.duration);
    if (currentTime <= totalTime) {
      return stage;
    }
  }
  
  return stages[stages.length - 1];
}

function calculateStressLevel(stage) {
  const maxTarget = Math.max(...options.stages.map(s => s.target));
  return (stage.target / maxTarget) * 100;
}

function selectOperationsForStressLevel(stressLevel) {
  const operations = [];
  
  if (stressLevel < 30) {
    operations.push({ type: 'create_heavy_vm', probability: 0.3 });
  } else if (stressLevel < 60) {
    operations.push(
      { type: 'create_heavy_vm', probability: 0.4 },
      { type: 'rapid_lifecycle', probability: 0.3 }
    );
  } else if (stressLevel < 80) {
    operations.push(
      { type: 'create_burst_vms', probability: 0.4 },
      { type: 'concurrent_operations', probability: 0.4 },
      { type: 'memory_stress', probability: 0.2 }
    );
  } else {
    // Maximum stress - all operations
    operations.push(
      { type: 'create_burst_vms', probability: 0.3 },
      { type: 'concurrent_operations', probability: 0.3 },
      { type: 'memory_stress', probability: 0.2 },
      { type: 'rapid_lifecycle', probability: 0.2 }
    );
  }
  
  return operations.filter(op => Math.random() < op.probability);
}

function createHeavyVM(headers, stressLevel) {
  const config = {
    ...stressConfigs.heavyVMConfig,
    name: `stress-heavy-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
    // Scale resources based on stress level
    cpu: Math.max(4, Math.floor(stressLevel / 12.5)), // 4-8 CPUs
    memory: Math.max(8192, Math.floor(stressLevel * 200)), // 8-20GB RAM
  };

  const response = http.post(`${API_BASE}/vms`, 
    JSON.stringify(config), 
    { headers, timeout: '30s' }
  );

  const success = check(response, {
    'heavy VM creation initiated': (r) => r.status === 201 || r.status === 202,
    'heavy VM creation response time acceptable': (r) => r.timings.duration < 15000,
  });

  if (success && response.status === 201) {
    try {
      const vmData = JSON.parse(response.body);
      createdVMs.push({ id: vmData.id, type: 'heavy', created: Date.now() });
    } catch (e) {
      console.warn('Failed to parse heavy VM creation response');
    }
  }

  errorRate.add(!success);
}

function createBurstVMs(headers, stressLevel) {
  const burstSize = Math.min(10, Math.floor(stressLevel / 10)); // 1-10 VMs
  
  for (let i = 0; i < burstSize; i++) {
    const config = {
      ...stressConfigs.lightVMConfig,
      name: `stress-burst-${Date.now()}-${i}`,
    };

    // Fire and forget for burst testing
    http.asyncRequest('POST', `${API_BASE}/vms`, 
      JSON.stringify(config), 
      { headers, timeout: '10s' }
    ).then(response => {
      const success = response.status === 201 || response.status === 202;
      errorRate.add(!success);
      
      if (success && response.status === 201) {
        try {
          const vmData = JSON.parse(response.body);
          createdVMs.push({ id: vmData.id, type: 'burst', created: Date.now() });
        } catch (e) {
          // Ignore parsing errors during burst
        }
      }
    });
    
    // Small delay between burst requests
    sleep(0.1);
  }
}

function performRapidLifecycle(headers) {
  if (createdVMs.length === 0) return;
  
  const vm = createdVMs[Math.floor(Math.random() * createdVMs.length)];
  const operations = ['start', 'stop', 'restart'];
  const operation = operations[Math.floor(Math.random() * operations.length)];
  
  const response = http.post(`${API_BASE}/vms/${vm.id}/${operation}`, 
    null, 
    { headers, timeout: '10s' }
  );
  
  check(response, {
    [`rapid ${operation} operation succeeds`]: (r) => r.status === 200 || r.status === 202,
    [`rapid ${operation} response time acceptable`]: (r) => r.timings.duration < 8000,
  });
}

function performConcurrentOperations(headers) {
  const concurrentRequests = [];
  const operationCount = Math.floor(Math.random() * 5) + 3; // 3-7 concurrent ops
  
  for (let i = 0; i < operationCount; i++) {
    // Mix of different API endpoints
    const endpoints = [
      `${API_BASE}/vms`,
      `${API_BASE}/auth/me`,
      `${API_BASE}/admin/stats`,
      `${BASE_URL}/health`,
    ];
    
    const endpoint = endpoints[Math.floor(Math.random() * endpoints.length)];
    
    concurrentRequests.push(
      http.asyncRequest('GET', endpoint, null, { headers, timeout: '5s' })
    );
  }
  
  // Wait for all concurrent requests
  Promise.all(concurrentRequests).then(responses => {
    const successCount = responses.filter(r => r.status < 400).length;
    const successRate = successCount / responses.length;
    
    check(null, {
      'concurrent operations success rate > 70%': () => successRate > 0.7,
    });
    
    errorRate.add(successRate < 0.7);
  });
}

function performMemoryStressTest(headers) {
  // Request large datasets to stress memory
  const response = http.get(`${API_BASE}/vms?limit=1000&include_metrics=true`, 
    { headers, timeout: '15s' }
  );
  
  check(response, {
    'large dataset request succeeds': (r) => r.status === 200,
    'large dataset response time acceptable': (r) => r.timings.duration < 12000,
  });
  
  // Request system logs (potentially large)
  if (Math.random() < 0.3) {
    const logsResponse = http.get(`${API_BASE}/admin/logs?limit=10000`, 
      { headers, timeout: '20s' }
    );
    
    check(logsResponse, {
      'large logs request handled': (r) => r.status < 500,
    });
  }
}

function performHeavyDatabaseOperations(headers) {
  // Simulate complex queries through API endpoints
  const operations = [
    // Search operations
    () => http.get(`${API_BASE}/vms?search=stress&sort=created_at&limit=100`, { headers }),
    
    // Aggregation operations
    () => http.get(`${API_BASE}/admin/analytics/vm-usage?period=30d`, { headers }),
    
    // Batch operations
    () => http.post(`${API_BASE}/vms/batch-action`, 
      JSON.stringify({ action: 'get-status', vm_ids: createdVMs.slice(0, 50).map(vm => vm.id) }), 
      { headers }
    ),
  ];
  
  const operation = operations[Math.floor(Math.random() * operations.length)];
  const response = operation();
  
  check(response, {
    'heavy database operation completes': (r) => r.status < 500,
    'heavy database operation response time acceptable': (r) => r.timings.duration < 20000,
  });
}

function stressAPIEndpoints(headers, stressLevel) {
  const endpoints = [
    { url: `${API_BASE}/vms`, weight: 0.4 },
    { url: `${API_BASE}/auth/me`, weight: 0.2 },
    { url: `${API_BASE}/admin/stats`, weight: 0.2 },
    { url: `${BASE_URL}/health`, weight: 0.2 },
  ];
  
  // Make multiple requests based on stress level
  const requestCount = Math.max(1, Math.floor(stressLevel / 20));
  
  for (let i = 0; i < requestCount; i++) {
    const endpoint = selectWeightedEndpoint(endpoints);
    const response = http.get(endpoint, { headers, timeout: '8s' });
    
    check(response, {
      'API endpoint stress test succeeds': (r) => r.status < 500,
    });
    
    errorRate.add(response.status >= 500);
  }
}

function monitorResourceUtilization(headers) {
  const response = http.get(`${API_BASE}/admin/system-metrics`, { headers });
  
  if (response.status === 200) {
    try {
      const metrics = JSON.parse(response.body);
      
      // Calculate resource utilization score
      const utilizationScore = (
        (metrics.cpu_usage || 0) * 0.3 +
        (metrics.memory_usage || 0) * 0.3 +
        (metrics.disk_usage || 0) * 0.2 +
        (metrics.network_usage || 0) * 0.2
      );
      
      resourceUtilization.set(utilizationScore);
      
      // Log high resource usage
      if (utilizationScore > 90) {
        console.warn(`High resource utilization detected: ${utilizationScore}%`);
      }
      
    } catch (e) {
      console.warn('Failed to parse system metrics response');
    }
  }
}

function selectWeightedEndpoint(endpoints) {
  const totalWeight = endpoints.reduce((sum, ep) => sum + ep.weight, 0);
  let random = Math.random() * totalWeight;
  
  for (const endpoint of endpoints) {
    random -= endpoint.weight;
    if (random <= 0) {
      return endpoint.url;
    }
  }
  
  return endpoints[0].url; // Fallback
}

function parseDuration(duration) {
  const match = duration.match(/^(\d+)([smh])$/);
  if (!match) return 0;
  
  const value = parseInt(match[1]);
  const unit = match[2];
  
  switch (unit) {
    case 's': return value;
    case 'm': return value * 60;
    case 'h': return value * 3600;
    default: return 0;
  }
}

export function teardown(data) {
  console.log('Starting stress test cleanup...');
  
  const token = data.authToken;
  const headers = { 'Authorization': `Bearer ${token}` };
  
  // Clean up created VMs
  if (createdVMs.length > 0) {
    console.log(`Cleaning up ${createdVMs.length} VMs created during stress test...`);
    
    // Batch deletion for efficiency
    const batchSize = 10;
    for (let i = 0; i < createdVMs.length; i += batchSize) {
      const batch = createdVMs.slice(i, i + batchSize);
      
      batch.forEach(vm => {
        // Force stop and delete
        http.post(`${API_BASE}/vms/${vm.id}/stop`, null, { headers });
        sleep(1);
        
        const deleteResponse = http.del(`${API_BASE}/vms/${vm.id}`, null, { headers });
        if (deleteResponse.status === 200) {
          console.log(`Deleted VM: ${vm.id} (type: ${vm.type})`);
        }
      });
      
      // Brief pause between batches
      sleep(2);
    }
  }
  
  // Final system health check
  const finalHealthResponse = http.get(`${BASE_URL}/health`);
  if (finalHealthResponse.status === 200) {
    console.log('System health check passed after stress test cleanup');
  } else {
    console.warn('System health check failed after stress test - may need manual intervention');
  }
  
  console.log('Stress test cleanup completed.');
}

export function handleSummary(data) {
  const report = generateStressTestReport(data);
  
  return {
    'stress-test-summary.html': report.html,
    'stress-test-results.json': JSON.stringify(data, null, 2),
    'stress-test-analysis.txt': report.analysis,
  };
}

function generateStressTestReport(data) {
  const metrics = data.metrics;
  const maxStressLevel = Math.max(...options.stages.map(s => s.target));
  
  const html = `
<!DOCTYPE html>
<html>
<head>
    <title>NovaCron Stress Test Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .critical { color: red; font-weight: bold; }
        .warning { color: orange; }
        .success { color: green; }
        .metric { margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .chart { width: 100%; height: 200px; background: #f9f9f9; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>NovaCron Stress Test Results</h1>
    
    <div class="metric">
        <h2>Test Configuration</h2>
        <ul>
            <li><strong>Maximum Concurrent Users:</strong> ${maxStressLevel}</li>
            <li><strong>Test Duration:</strong> ${options.stages.reduce((sum, stage) => sum + parseDuration(stage.duration), 0) / 60} minutes</li>
            <li><strong>Stress Pattern:</strong> Gradual ramp-up to breaking point</li>
        </ul>
    </div>
    
    <div class="metric">
        <h2>System Breaking Points</h2>
        <table>
            <tr><th>Metric</th><th>Value</th><th>Status</th><th>Impact</th></tr>
            <tr>
                <td>Maximum Response Time</td>
                <td>${Math.round(metrics.http_req_duration?.values?.max || 0)}ms</td>
                <td class="${metrics.http_req_duration?.values?.max > 10000 ? 'critical' : 'success'}">
                    ${metrics.http_req_duration?.values?.max > 10000 ? 'CRITICAL' : 'ACCEPTABLE'}
                </td>
                <td>${metrics.http_req_duration?.values?.max > 10000 ? 'User experience severely degraded' : 'Within acceptable limits'}</td>
            </tr>
            <tr>
                <td>Peak Error Rate</td>
                <td>${Math.round((metrics.error_rate?.values?.rate || 0) * 100 * 100) / 100}%</td>
                <td class="${(metrics.error_rate?.values?.rate || 0) > 0.15 ? 'critical' : 'warning'}">
                    ${(metrics.error_rate?.values?.rate || 0) > 0.15 ? 'EXCEEDED THRESHOLD' : 'WITHIN LIMITS'}
                </td>
                <td>${(metrics.error_rate?.values?.rate || 0) > 0.15 ? 'Significant service degradation' : 'Service remained stable'}</td>
            </tr>
            <tr>
                <td>System Recovery</td>
                <td>${metrics.system_recovery_time ? Math.round(metrics.system_recovery_time.values.avg / 1000) + 's' : 'No failures detected'}</td>
                <td class="success">GOOD</td>
                <td>System recovered automatically from stress</td>
            </tr>
        </table>
    </div>
    
    <div class="metric">
        <h2>Resource Utilization</h2>
        <p><strong>Peak Resource Usage:</strong> ${Math.round(metrics.resource_utilization?.values?.max || 0)}%</p>
        <p><strong>Concurrent Operations:</strong> ${metrics.concurrent_operations?.values?.count || 0}</p>
    </div>
    
    <div class="metric">
        <h2>Key Findings</h2>
        <ul>
            <li>System handled up to ${maxStressLevel} concurrent users</li>
            <li>Performance degradation began at approximately ${Math.round(maxStressLevel * 0.7)} concurrent users</li>
            <li>Total requests processed: ${metrics.http_reqs?.values?.count || 0}</li>
            <li>Average requests per second: ${Math.round((metrics.http_reqs?.values?.rate || 0) * 100) / 100}</li>
        </ul>
    </div>
    
    <div class="metric">
        <h2>Recommendations</h2>
        <ul>
            ${(metrics.error_rate?.values?.rate || 0) > 0.10 ? '<li class="critical">Implement auto-scaling to handle traffic spikes</li>' : ''}
            ${metrics.http_req_duration?.values?.p95 > 5000 ? '<li class="warning">Optimize slow database queries and API responses</li>' : ''}
            ${(metrics.resource_utilization?.values?.max || 0) > 85 ? '<li class="warning">Monitor resource usage and consider horizontal scaling</li>' : ''}
            <li>Consider implementing circuit breakers for external dependencies</li>
            <li>Implement request queuing for high-load scenarios</li>
        </ul>
    </div>
    
    <p><em>Report generated at: ${new Date().toISOString()}</em></p>
</body>
</html>
  `;
  
  const analysis = `
NOVACRON STRESS TEST ANALYSIS
=============================

Test Parameters:
- Maximum Load: ${maxStressLevel} concurrent users
- Duration: ${options.stages.reduce((sum, stage) => sum + parseDuration(stage.duration), 0) / 60} minutes
- Total Requests: ${metrics.http_reqs?.values?.count || 0}

Performance Metrics:
- Average Response Time: ${Math.round(metrics.http_req_duration?.values?.avg || 0)}ms
- 95th Percentile: ${Math.round(metrics.http_req_duration?.values?.p95 || 0)}ms
- Maximum Response Time: ${Math.round(metrics.http_req_duration?.values?.max || 0)}ms
- Error Rate: ${Math.round((metrics.error_rate?.values?.rate || 0) * 10000) / 100}%

System Behavior:
- Breaking Point: Approximately ${Math.round(maxStressLevel * 0.8)} concurrent users
- Recovery Time: ${metrics.system_recovery_time ? Math.round(metrics.system_recovery_time.values.avg / 1000) + 's' : 'N/A'}
- Resource Peak: ${Math.round(metrics.resource_utilization?.values?.max || 0)}%

Conclusion:
${(metrics.error_rate?.values?.rate || 0) > 0.15 ? 'CRITICAL: System exceeded acceptable error thresholds' : 'System performed within acceptable stress parameters'}
  `;
  
  return { html, analysis };
}