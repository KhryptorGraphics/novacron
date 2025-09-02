import http from 'k6/http';
import ws from 'k6/ws';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter, Gauge } from 'k6/metrics';
import { config, getEnvironment, getTestData } from '../configs/test-config.js';

// Stress test specific metrics
const stressTestMetrics = {
  systemOverload: new Rate('system_overload_rate'),
  recoveryTime: new Trend('system_recovery_time'),
  failurePoints: new Counter('failure_points_detected'),
  maxLoadSupported: new Gauge('max_load_supported'),
  resourceExhaustion: new Counter('resource_exhaustion_events'),
  cascadingFailures: new Counter('cascading_failures'),
  gracefulDegradation: new Rate('graceful_degradation_rate')
};

// Test configuration for progressive stress testing
export const options = {
  scenarios: {
    stress_test: config.scenarios.stress_test,
  },
  thresholds: {
    'http_req_failed': ['rate<0.1'], // Allow 10% failure under stress
    'system_overload_rate': ['rate<0.2'], // System should handle 80% of peak load
    'graceful_degradation_rate': ['rate>0.8'], // 80% should degrade gracefully
    'max_load_supported': ['value>2000'] // Should support at least 2000 concurrent users
  }
};

const environment = getEnvironment();
const testData = getTestData();

// Stress test state tracking
let stressTestState = {
  currentLoad: 0,
  peakLoad: 0,
  failureThreshold: 0,
  recoveryStartTime: 0,
  systemFailures: 0,
  gracefulDegradations: 0
};

// Authentication with retry logic for stress conditions
function authenticateWithRetry(maxRetries = 3) {
  const user = testData.users.operator;
  const loginPayload = {
    username: user.username,
    password: user.password
  };

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const response = http.post(`${environment.baseURL}/api/auth/login`, 
        JSON.stringify(loginPayload), {
          headers: { 'Content-Type': 'application/json' },
          timeout: '30s'
        }
      );

      if (response.status === 200) {
        return response.json().token;
      } else if (response.status === 429) { // Rate limited
        console.log(`Authentication rate limited, attempt ${attempt}/${maxRetries}`);
        sleep(Math.pow(2, attempt)); // Exponential backoff
        continue;
      } else if (response.status >= 500) { // Server error
        console.log(`Server error during auth, attempt ${attempt}/${maxRetries}: ${response.status}`);
        stressTestMetrics.systemOverload.add(1);
        sleep(attempt * 2);
        continue;
      }
    } catch (error) {
      console.error(`Authentication attempt ${attempt} failed: ${error}`);
      stressTestMetrics.failurePoints.add(1);
      if (attempt < maxRetries) {
        sleep(attempt * 2);
      }
    }
  }

  return null;
}

// High-intensity VM operations under stress
function performStressVMOperations(token) {
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  };

  // Create multiple VMs rapidly to stress the system
  const vmBatch = [];
  const batchSize = Math.min(10, Math.floor(__VU / 10) + 1); // Scale batch size with VU count

  for (let i = 0; i < batchSize; i++) {
    const vmConfig = testData.vmConfigs[i % testData.vmConfigs.length];
    const vmPayload = {
      name: `stress-test-vm-${__VU}-${__ITER}-${i}`,
      cpu_shares: vmConfig.cpu,
      memory_mb: vmConfig.memory,
      disk_size_gb: vmConfig.disk,
      command: '/bin/stress',
      args: ['--cpu', '1', '--timeout', '60s'], // CPU stress test
      tags: { 
        'test': 'stress-test',
        'batch': `batch-${__VU}-${__ITER}`,
        'load-level': this.getCurrentLoadLevel()
      }
    };

    const createStart = Date.now();
    const response = http.post(`${environment.baseURL}/api/vms`, 
      JSON.stringify(vmPayload), { 
        headers,
        timeout: '60s' // Longer timeout under stress
      }
    );
    const createDuration = Date.now() - createStart;

    const success = check(response, {
      'stress VM creation handled': (r) => r.status === 201 || r.status === 429 || r.status === 503,
      'no server crash': (r) => r.status !== 500 && r.status !== 502 && r.status !== 504
    });

    if (response.status === 201) {
      vmBatch.push({
        id: response.json().id,
        created: true,
        createTime: createDuration
      });
      stressTestState.currentLoad++;
      stressTestMetrics.maxLoadSupported.add(stressTestState.currentLoad);
    } else if (response.status === 429 || response.status === 503) {
      // Graceful degradation - system is protecting itself
      stressTestMetrics.gracefulDegradation.add(1);
      stressTestState.gracefulDegradations++;
      console.log(`Graceful degradation detected: ${response.status}`);
    } else if (response.status >= 500) {
      // System failure
      stressTestMetrics.systemOverload.add(1);
      stressTestMetrics.failurePoints.add(1);
      stressTestState.systemFailures++;
      
      if (stressTestState.systemFailures > 5) {
        console.error('Multiple system failures detected - potential cascading failure');
        stressTestMetrics.cascadingFailures.add(1);
      }
    }

    // Brief pause between creations to allow system processing
    sleep(0.1);
  }

  // Track peak load
  if (stressTestState.currentLoad > stressTestState.peakLoad) {
    stressTestState.peakLoad = stressTestState.currentLoad;
  }

  // Perform operations on created VMs
  for (const vm of vmBatch) {
    if (vm.created) {
      this.performVMStressOperations(token, vm.id);
    }
  }

  return vmBatch;
}

// Intensive VM operations to stress the system
function performVMStressOperations(token, vmID) {
  const headers = { 'Authorization': `Bearer ${token}` };

  // Rapid-fire operations
  const operations = [
    () => http.post(`${environment.baseURL}/api/vms/${vmID}/start`, null, { headers, timeout: '30s' }),
    () => http.get(`${environment.baseURL}/api/vms/${vmID}/metrics`, { headers, timeout: '10s' }),
    () => http.post(`${environment.baseURL}/api/vms/${vmID}/pause`, null, { headers, timeout: '15s' }),
    () => http.post(`${environment.baseURL}/api/vms/${vmID}/resume`, null, { headers, timeout: '15s' }),
    () => http.post(`${environment.baseURL}/api/vms/${vmID}/stop`, null, { headers, timeout: '20s' })
  ];

  for (const operation of operations) {
    try {
      const response = operation();
      
      check(response, {
        'stress operation handled': (r) => r.status < 500 || r.status === 503,
        'no system crash': (r) => r.status !== 502 && r.status !== 504
      });

      if (response.status >= 500 && response.status !== 503) {
        stressTestMetrics.systemOverload.add(1);
      }

    } catch (error) {
      stressTestMetrics.failurePoints.add(1);
      console.error(`VM operation failed under stress: ${error}`);
    }

    sleep(0.2); // Brief pause between operations
  }

  // Cleanup VM
  try {
    const deleteResponse = http.del(`${environment.baseURL}/api/vms/${vmID}`, null, { 
      headers, 
      timeout: '30s' 
    });
    
    if (deleteResponse.status === 200) {
      stressTestState.currentLoad--;
    }
  } catch (error) {
    console.error(`VM cleanup failed: ${error}`);
  }
}

// WebSocket stress testing under high load
function performWebSocketStress(token) {
  const wsEndpoints = [
    { path: '/ws/metrics', params: { interval: '1' } },
    { path: '/ws/alerts', params: { severity: 'critical' } }
  ];

  for (const endpoint of wsEndpoints) {
    try {
      const queryString = new URLSearchParams(endpoint.params).toString();
      const wsUrl = `${environment.wsURL}${endpoint.path}?${queryString}`;
      
      ws.connect(wsUrl, { 'Authorization': `Bearer ${token}` }, function (socket) {
        let messageCount = 0;
        const connectionStart = Date.now();

        socket.on('open', function () {
          console.log(`Stress WebSocket connected: ${endpoint.path}`);
        });

        socket.on('message', function (data) {
          messageCount++;
          
          // Under stress, validate that messages are still well-formed
          try {
            const message = JSON.parse(data);
            check(message, {
              'stress WS message valid under load': (m) => m.type !== undefined
            });
          } catch (error) {
            stressTestMetrics.failurePoints.add(1);
          }
        });

        socket.on('error', function (e) {
          stressTestMetrics.systemOverload.add(1);
          console.error(`Stress WebSocket error: ${e.error()}`);
        });

        socket.on('close', function () {
          const sessionDuration = Date.now() - connectionStart;
          console.log(`Stress WebSocket closed: ${sessionDuration}ms, ${messageCount} messages`);
          
          // Check if connection lasted reasonable time under stress
          if (sessionDuration < 10000 && messageCount < 5) {
            stressTestMetrics.systemOverload.add(1);
          } else {
            stressTestMetrics.gracefulDegradation.add(1);
          }
        });

        // Shorter session under stress
        socket.setTimeout(function () {
          socket.close();
        }, 15000 + Math.random() * 5000); // 15-20 seconds
      });

    } catch (error) {
      stressTestMetrics.failurePoints.add(1);
      console.error(`WebSocket stress connection failed: ${error}`);
    }

    sleep(0.5); // Brief pause between WS connections
  }
}

// Database stress testing
function performDatabaseStress(token) {
  const headers = { 'Authorization': `Bearer ${token}` };

  // Rapid database queries to test connection pooling and query performance
  const queries = [
    '/api/vms?page=1&pageSize=100&sortBy=createdAt',
    '/api/storage/volumes?limit=100',
    '/api/monitoring/events?limit=200',
    '/api/cluster/nodes',
    '/api/monitoring/metrics?type=all'
  ];

  for (const query of queries) {
    try {
      const queryStart = Date.now();
      const response = http.get(`${environment.baseURL}${query}`, { 
        headers, 
        timeout: '45s' // Longer timeout under stress
      });
      const queryDuration = Date.now() - queryStart;

      const success = check(response, {
        'stress DB query handled': (r) => r.status < 500 || r.status === 503,
        'DB query reasonable time': (r) => r.timings.duration < 60000 // 60s max under stress
      });

      if (!success) {
        stressTestMetrics.systemOverload.add(1);
        
        if (response.status === 0) {
          stressTestMetrics.resourceExhaustion.add(1);
        }
      }

    } catch (error) {
      stressTestMetrics.failurePoints.add(1);
      console.error(`Database stress query failed: ${error}`);
    }

    sleep(0.1); // Minimal pause between queries
  }
}

// System recovery testing
function testSystemRecovery(token) {
  // Test system's ability to recover from high load
  if (stressTestState.systemFailures > 0 && stressTestState.recoveryStartTime === 0) {
    stressTestState.recoveryStartTime = Date.now();
    console.log('Starting system recovery testing...');
  }

  if (stressTestState.recoveryStartTime > 0) {
    // Test if system is recovering
    try {
      const healthResponse = http.get(`${environment.baseURL}/api/cluster/health`, {
        headers: { 'Authorization': `Bearer ${token}` },
        timeout: '10s'
      });

      if (healthResponse.status === 200) {
        const health = healthResponse.json();
        if (health.status === 'healthy') {
          const recoveryTime = Date.now() - stressTestState.recoveryStartTime;
          stressTestMetrics.recoveryTime.add(recoveryTime);
          stressTestState.recoveryStartTime = 0; // Reset
          console.log(`System recovered in ${recoveryTime}ms`);
        }
      }
    } catch (error) {
      console.error('Recovery test failed:', error);
    }
  }
}

// Main stress test function
export default function() {
  // Update current load level based on VU count and time
  const loadLevel = this.getCurrentLoadLevel();
  stressTestState.currentLoad = __VU;

  // Authenticate with retry logic
  const token = authenticateWithRetry();
  if (!token) {
    stressTestMetrics.failurePoints.add(1);
    console.error('Authentication failed under stress');
    return;
  }

  // Determine stress test intensity based on current load
  const intensity = Math.min(__VU / 100, 1.0); // Scale intensity with VU count

  try {
    // Progressive stress testing
    if (loadLevel === 'low') { // VUs 1-500
      performStressVMOperations(token);
      sleep(Math.random() * 2 + 1);
      
    } else if (loadLevel === 'medium') { // VUs 501-1500
      performStressVMOperations(token);
      performWebSocketStress(token);
      sleep(Math.random() * 1.5 + 0.5);
      
    } else if (loadLevel === 'high') { // VUs 1501-3000
      performStressVMOperations(token);
      performWebSocketStress(token);
      performDatabaseStress(token);
      sleep(Math.random() + 0.2);
      
    } else { // VUs 3000+ - Maximum stress
      performStressVMOperations(token);
      performWebSocketStress(token);
      performDatabaseStress(token);
      testSystemRecovery(token);
      sleep(Math.random() * 0.5); // Minimal pause
    }

  } catch (error) {
    stressTestMetrics.failurePoints.add(1);
    console.error(`Stress test iteration failed: ${error}`);
  }
}

function getCurrentLoadLevel() {
  if (__VU <= 500) return 'low';
  if (__VU <= 1500) return 'medium';  
  if (__VU <= 3000) return 'high';
  return 'extreme';
}

// Setup function
export function setup() {
  console.log('Starting progressive stress test');
  console.log(`Target environment: ${environment.baseURL}`);
  console.log('Stress test stages:');
  console.log('- Stage 1 (0-500 VUs): Basic stress testing');
  console.log('- Stage 2 (501-1500 VUs): Multi-component stress'); 
  console.log('- Stage 3 (1501-3000 VUs): High-intensity stress');
  console.log('- Stage 4 (3000+ VUs): Maximum load + recovery testing');
  
  // Verify system is healthy before stress testing
  const healthResponse = http.get(`${environment.baseURL}/api/cluster/health`);
  
  if (healthResponse.status !== 200) {
    throw new Error(`System not healthy for stress testing: ${healthResponse.status}`);
  }

  const health = healthResponse.json();
  if (health.status !== 'healthy') {
    throw new Error(`Cluster not healthy: ${health.status}`);
  }

  console.log(`Initial system status: ${health.status}`);
  console.log(`Initial cluster nodes: ${health.total_nodes}`);
  
  return { 
    startTime: Date.now(),
    initialHealth: health,
    testId: `stress-test-${Date.now()}`
  };
}

// Teardown function with comprehensive stress test analysis
export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  
  console.log('\n=== Stress Test Results ===');
  console.log(`Total duration: ${duration} seconds`);
  console.log(`Peak concurrent load: ${stressTestState.peakLoad} VUs`);
  console.log(`System failures detected: ${stressTestState.systemFailures}`);
  console.log(`Graceful degradations: ${stressTestState.gracefulDegradations}`);
  
  // Calculate stress test metrics
  const overloadRate = stressTestMetrics.systemOverload.rate;
  const degradationRate = stressTestMetrics.gracefulDegradation.rate;
  const failurePoints = stressTestMetrics.failurePoints.count;
  
  console.log(`System overload rate: ${(overloadRate * 100).toFixed(2)}%`);
  console.log(`Graceful degradation rate: ${(degradationRate * 100).toFixed(2)}%`);
  console.log(`Total failure points: ${failurePoints}`);
  
  // System resilience assessment
  let resilienceScore = 100;
  resilienceScore -= overloadRate * 30; // Penalty for overloads
  resilienceScore += degradationRate * 20; // Bonus for graceful degradation
  resilienceScore -= (failurePoints / 100) * 10; // Penalty for failures
  
  console.log(`System resilience score: ${Math.max(0, resilienceScore).toFixed(1)}/100`);
  
  // Recovery assessment
  if (stressTestState.recoveryStartTime > 0) {
    console.log('System still recovering from stress at test end');
  }
  
  // Recommendations based on stress test results
  console.log('\nStress Test Recommendations:');
  
  if (overloadRate > 0.2) {
    console.log('- HIGH: Implement better load balancing and circuit breakers');
  }
  
  if (degradationRate < 0.8) {
    console.log('- MEDIUM: Improve graceful degradation mechanisms');
  }
  
  if (failurePoints > 50) {
    console.log('- HIGH: Investigate and fix failure points in critical paths');
  }
  
  if (stressTestState.peakLoad < 2000) {
    console.log('- MEDIUM: System capacity may need scaling for production load');
  }
  
  console.log('=== End Stress Test Results ===\n');
  
  // Final system health check
  const finalHealthResponse = http.get(`${environment.baseURL}/api/cluster/health`);
  if (finalHealthResponse.status === 200) {
    const finalHealth = finalHealthResponse.json();
    console.log(`Final system status: ${finalHealth.status}`);
    
    if (finalHealth.status !== 'healthy') {
      console.warn('⚠️  System not fully recovered after stress test');
    }
  }
}