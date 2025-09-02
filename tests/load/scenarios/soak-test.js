import http from 'k6/http';
import ws from 'k6/ws';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter, Gauge } from 'k6/metrics';
import { config, getEnvironment, getTestData } from '../configs/test-config.js';

// Soak test specific metrics for long-term stability
const soakTestMetrics = {
  memoryLeakRate: new Trend('memory_leak_rate'),
  connectionLeakRate: new Trend('connection_leak_rate'),
  performanceDegradation: new Trend('performance_degradation'),
  resourceUsageGrowth: new Trend('resource_usage_growth'),
  errorRateOverTime: new Trend('error_rate_over_time'),
  systemStability: new Rate('system_stability_rate'),
  longRunningOperations: new Counter('long_running_operations'),
  memoryUsageBaseline: new Gauge('memory_usage_baseline'),
  currentMemoryUsage: new Gauge('current_memory_usage')
};

// Test configuration for 2-hour sustained load
export const options = {
  scenarios: {
    soak_test: config.scenarios.soak_test,
  },
  thresholds: {
    'http_req_failed': ['rate<0.02'], // Very low error rate for soak test
    'http_req_duration': ['p(95)<1000'], // Consistent performance over time
    'system_stability_rate': ['rate>0.98'], // 98% stability over 2 hours
    'memory_leak_rate': ['p(95)<5'], // Less than 5% memory growth per hour
    'performance_degradation': ['p(95)<10'] // Less than 10% performance degradation
  }
};

const environment = getEnvironment();
const testData = getTestData();

// Soak test state tracking
let soakTestState = {
  startTime: Date.now(),
  baselineMetrics: null,
  currentMetrics: null,
  hourlySnapshots: [],
  operationsPerformed: 0,
  longRunningVMs: new Map(),
  persistentConnections: new Map(),
  performanceBaseline: null,
  lastMemoryCheck: Date.now()
};

// Establish performance baseline
function establishBaseline(token) {
  const headers = { 'Authorization': `Bearer ${token}` };

  // Measure baseline performance
  const baselineStart = Date.now();
  
  const endpoints = [
    '/api/cluster/health',
    '/api/vms',
    '/api/storage/volumes',
    '/api/monitoring/metrics'
  ];

  let totalResponseTime = 0;
  let successCount = 0;

  for (const endpoint of endpoints) {
    try {
      const response = http.get(`${environment.baseURL}${endpoint}`, { headers });
      
      if (response.status === 200) {
        totalResponseTime += response.timings.duration;
        successCount++;
      }
    } catch (error) {
      console.error(`Baseline measurement failed for ${endpoint}: ${error}`);
    }
  }

  if (successCount > 0) {
    soakTestState.performanceBaseline = totalResponseTime / successCount;
    console.log(`Performance baseline established: ${soakTestState.performanceBaseline.toFixed(2)}ms`);
  }

  // Get system resource baseline
  soakTestState.baselineMetrics = {
    timestamp: Date.now(),
    memory: this.getCurrentMemoryUsage(),
    connections: this.getCurrentConnectionCount(),
    responseTime: soakTestState.performanceBaseline
  };

  soakTestMetrics.memoryUsageBaseline.add(soakTestState.baselineMetrics.memory);
}

// Long-running VM operations for soak testing
function performSoakVMOperations(token) {
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  };

  // Create long-running VMs that persist across test iterations
  const vmKey = `soak-vm-${__VU}`;
  
  if (!soakTestState.longRunningVMs.has(vmKey)) {
    // Create a long-running VM for this VU
    const vmConfig = testData.vmConfigs[0]; // Use small config for long-running
    const vmPayload = {
      name: `soak-test-vm-${__VU}-${Date.now()}`,
      cpu_shares: vmConfig.cpu,
      memory_mb: vmConfig.memory,
      disk_size_gb: vmConfig.disk,
      command: '/bin/bash',
      args: ['-c', 'while true; do sleep 300; echo "Soak test VM alive"; done'],
      tags: { 
        'test': 'soak-test',
        'vu': __VU.toString(),
        'long-running': 'true'
      }
    };

    const response = http.post(`${environment.baseURL}/api/vms`, 
      JSON.stringify(vmPayload), { headers }
    );

    if (response.status === 201) {
      const vm = response.json();
      soakTestState.longRunningVMs.set(vmKey, {
        id: vm.id,
        name: vm.name,
        createdAt: Date.now(),
        operations: 0
      });
      
      // Start the VM
      http.post(`${environment.baseURL}/api/vms/${vm.id}/start`, null, { headers });
      console.log(`Created long-running VM: ${vm.name}`);
    }
  }

  // Perform periodic operations on long-running VM
  const longRunningVM = soakTestState.longRunningVMs.get(vmKey);
  if (longRunningVM) {
    try {
      // Get VM metrics periodically
      const metricsResponse = http.get(`${environment.baseURL}/api/vms/${longRunningVM.id}/metrics`, { headers });
      
      const success = check(metricsResponse, {
        'soak VM metrics accessible': (r) => r.status === 200,
        'soak VM metrics consistent': (r) => {
          if (r.status === 200) {
            const metrics = r.json();
            return metrics.vm_id === longRunningVM.id;
          }
          return false;
        }
      });

      soakTestMetrics.systemStability.add(success);
      longRunningVM.operations++;
      soakTestMetrics.longRunningOperations.add(1);

      // Periodic VM health check
      if (longRunningVM.operations % 10 === 0) {
        const vmResponse = http.get(`${environment.baseURL}/api/vms/${longRunningVM.id}`, { headers });
        
        if (vmResponse.status === 200) {
          const vm = vmResponse.json();
          if (vm.state !== 'running') {
            console.warn(`Long-running VM ${longRunningVM.name} not in running state: ${vm.state}`);
            soakTestMetrics.systemStability.add(0);
          }
        }
      }

    } catch (error) {
      console.error(`Soak VM operation failed: ${error}`);
      soakTestMetrics.systemStability.add(0);
    }
  }
}

// Persistent WebSocket connections for soak testing
function maintainPersistentWebSockets(token) {
  const wsKey = `soak-ws-${__VU}`;
  
  if (!soakTestState.persistentConnections.has(wsKey)) {
    // Create persistent WebSocket connection
    const wsUrl = `${environment.wsURL}/ws/metrics?interval=30`;
    
    try {
      ws.connect(wsUrl, { 'Authorization': `Bearer ${token}` }, function (socket) {
        const connectionStart = Date.now();
        let messageCount = 0;
        let lastMessageTime = Date.now();

        soakTestState.persistentConnections.set(wsKey, {
          socket: socket,
          createdAt: connectionStart,
          messageCount: 0
        });

        socket.on('open', function () {
          console.log(`Soak test WebSocket connected: ${wsKey}`);
        });

        socket.on('message', function (data) {
          messageCount++;
          lastMessageTime = Date.now();
          
          try {
            const message = JSON.parse(data);
            check(message, {
              'soak WS message valid': (m) => m.type !== undefined && m.timestamp !== undefined
            });
          } catch (error) {
            console.error(`Invalid soak WebSocket message: ${error}`);
          }

          // Update connection info
          const connectionInfo = soakTestState.persistentConnections.get(wsKey);
          if (connectionInfo) {
            connectionInfo.messageCount = messageCount;
            connectionInfo.lastMessageTime = lastMessageTime;
          }
        });

        socket.on('error', function (e) {
          console.error(`Soak WebSocket error: ${e.error()}`);
          soakTestMetrics.systemStability.add(0);
        });

        socket.on('close', function () {
          const sessionDuration = Date.now() - connectionStart;
          console.log(`Soak WebSocket closed after ${sessionDuration}ms, ${messageCount} messages`);
          soakTestState.persistentConnections.delete(wsKey);
          
          // Check if connection lasted appropriate time for soak test
          if (sessionDuration < 300000) { // Less than 5 minutes is concerning
            soakTestMetrics.connectionLeakRate.add(1);
          }
        });

        // Send periodic heartbeat to maintain connection
        const heartbeatInterval = setInterval(function () {
          if (socket.readyState === 1) {
            socket.send(JSON.stringify({
              type: 'heartbeat',
              timestamp: new Date().toISOString(),
              vu: __VU
            }));
          } else {
            clearInterval(heartbeatInterval);
          }
        }, 60000); // Every minute
      });

    } catch (error) {
      console.error(`Failed to create persistent WebSocket: ${error}`);
    }
  }
}

// Monitor for memory leaks and resource growth
function monitorResourceUsage(token) {
  // Check memory usage every 5 minutes
  if (Date.now() - soakTestState.lastMemoryCheck > 300000) {
    soakTestState.lastMemoryCheck = Date.now();
    
    try {
      const headers = { 'Authorization': `Bearer ${token}` };
      const metricsResponse = http.get(`${environment.baseURL}/api/monitoring/metrics`, { headers });
      
      if (metricsResponse.status === 200) {
        const metrics = metricsResponse.json();
        const currentMemoryUsage = (metrics.memory.used / metrics.memory.total) * 100;
        
        soakTestMetrics.currentMemoryUsage.add(currentMemoryUsage);
        
        // Calculate memory growth rate
        if (soakTestState.baselineMetrics) {
          const memoryGrowth = currentMemoryUsage - soakTestState.baselineMetrics.memory;
          const timeElapsed = (Date.now() - soakTestState.startTime) / 3600000; // Hours
          const growthRate = timeElapsed > 0 ? memoryGrowth / timeElapsed : 0;
          
          soakTestMetrics.memoryLeakRate.add(growthRate);
          
          if (growthRate > 5) { // More than 5% growth per hour
            console.warn(`⚠️  Potential memory leak detected: ${growthRate.toFixed(2)}% growth/hour`);
          }
        }

        // Performance degradation check
        if (soakTestState.performanceBaseline) {
          const currentPerformance = metrics.response_time || 0;
          const degradation = ((currentPerformance - soakTestState.performanceBaseline) / soakTestState.performanceBaseline) * 100;
          
          soakTestMetrics.performanceDegradation.add(degradation);
          
          if (degradation > 20) { // More than 20% degradation
            console.warn(`⚠️  Performance degradation detected: ${degradation.toFixed(2)}%`);
          }
        }

        // Take hourly snapshots
        const hoursElapsed = Math.floor((Date.now() - soakTestState.startTime) / 3600000);
        if (hoursElapsed > soakTestState.hourlySnapshots.length) {
          soakTestState.hourlySnapshots.push({
            hour: hoursElapsed,
            timestamp: Date.now(),
            memoryUsage: currentMemoryUsage,
            vmCount: soakTestState.longRunningVMs.size,
            wsConnections: soakTestState.persistentConnections.size,
            operationsPerformed: soakTestState.operationsPerformed
          });
          
          console.log(`Hour ${hoursElapsed} snapshot: Memory ${currentMemoryUsage.toFixed(1)}%, VMs ${soakTestState.longRunningVMs.size}, WS ${soakTestState.persistentConnections.size}`);
        }
      }
    } catch (error) {
      console.error(`Resource monitoring failed: ${error}`);
    }
  }
}

// Continuous light operations for soak testing
function performSoakOperations(token) {
  const headers = { 'Authorization': `Bearer ${token}` };

  // Light, continuous operations
  const operations = [
    // Health checks
    () => http.get(`${environment.baseURL}/api/cluster/health`, { headers }),
    
    // VM listing with different parameters
    () => http.get(`${environment.baseURL}/api/vms?page=1&pageSize=20`, { headers }),
    
    // Storage monitoring
    () => http.get(`${environment.baseURL}/api/storage/metrics`, { headers }),
    
    // System metrics
    () => http.get(`${environment.baseURL}/api/monitoring/metrics`, { headers }),
    
    // Node status
    () => http.get(`${environment.baseURL}/api/cluster/nodes`, { headers })
  ];

  // Perform random operations
  const operation = operations[Math.floor(Math.random() * operations.length)];
  
  const operationStart = Date.now();
  const response = operation();
  const operationDuration = Date.now() - operationStart;

  const success = check(response, {
    'soak operation successful': (r) => r.status === 200,
    'soak operation timely': (r) => r.timings.duration < 2000 // 2 second timeout
  });

  soakTestMetrics.systemStability.add(success);
  soakTestState.operationsPerformed++;

  // Track performance over time
  if (soakTestState.performanceBaseline) {
    const performanceRatio = operationDuration / soakTestState.performanceBaseline;
    soakTestMetrics.performanceDegradation.add((performanceRatio - 1) * 100);
  }

  // Track error rate over time
  const currentTime = Date.now();
  const testDuration = (currentTime - soakTestState.startTime) / 3600000; // Hours
  const errorRate = success ? 0 : 1;
  soakTestMetrics.errorRateOverTime.add(errorRate);
}

// Main soak test function
export default function() {
  const token = authenticateWithRetry();
  if (!token) {
    console.error('Authentication failed in soak test');
    return;
  }

  // Establish baseline on first iteration
  if (__ITER === 0) {
    establishBaseline(token);
  }

  try {
    // Perform different types of soak test operations
    const operationType = __ITER % 5;

    switch (operationType) {
      case 0: // Continuous light operations
        performSoakOperations(token);
        break;

      case 1: // Long-running VM management
        performSoakVMOperations(token);
        break;

      case 2: // Persistent WebSocket maintenance
        maintainPersistentWebSockets(token);
        break;

      case 3: // Resource usage monitoring
        monitorResourceUsage(token);
        break;

      case 4: // System stability checks
        performStabilityChecks(token);
        break;
    }

  } catch (error) {
    console.error(`Soak test iteration failed: ${error}`);
    soakTestMetrics.systemStability.add(0);
  }

  // Consistent pacing for soak test
  sleep(Math.random() * 10 + 20); // 20-30 seconds between operations
}

function performStabilityChecks(token) {
  const headers = { 'Authorization': `Bearer ${token}` };

  // Check for system stability indicators
  const stabilityChecks = [
    // Memory usage trends
    async () => {
      const response = http.get(`${environment.baseURL}/api/monitoring/metrics`, { headers });
      if (response.status === 200) {
        const metrics = response.json();
        const memoryUsage = (metrics.memory.used / metrics.memory.total) * 100;
        
        if (soakTestState.baselineMetrics) {
          const memoryGrowth = memoryUsage - soakTestState.baselineMetrics.memory;
          if (memoryGrowth > 10) { // 10% memory growth
            console.warn(`Memory usage increased by ${memoryGrowth.toFixed(1)}% during soak test`);
            soakTestMetrics.memoryLeakRate.add(memoryGrowth);
          }
        }
      }
    },

    // Connection pool health
    async () => {
      const response = http.get(`${environment.baseURL}/api/cluster/nodes`, { headers });
      check(response, {
        'connection pool healthy': (r) => r.status === 200,
        'cluster connectivity stable': (r) => {
          if (r.status === 200) {
            const nodes = r.json();
            return Array.isArray(nodes) && nodes.length > 0;
          }
          return false;
        }
      });
    },

    // Application responsiveness
    async () => {
      const start = Date.now();
      const response = http.get(`${environment.baseURL}/api/cluster/health`, { headers });
      const duration = Date.now() - start;
      
      if (soakTestState.performanceBaseline) {
        const degradation = ((duration - soakTestState.performanceBaseline) / soakTestState.performanceBaseline) * 100;
        soakTestMetrics.performanceDegradation.add(degradation);
        
        if (degradation > 50) { // 50% performance degradation
          console.warn(`Significant performance degradation: ${degradation.toFixed(1)}%`);
        }
      }
    }
  ];

  // Execute stability checks
  for (const checkFn of stabilityChecks) {
    try {
      checkFn();
    } catch (error) {
      console.error(`Stability check failed: ${error}`);
      soakTestMetrics.systemStability.add(0);
    }
  }
}

// Helper function for authentication with retry
function authenticateWithRetry(maxRetries = 3) {
  const user = testData.users.viewer; // Use viewer for soak test
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
      }

      if (attempt < maxRetries) {
        sleep(attempt * 2); // Exponential backoff
      }
    } catch (error) {
      console.error(`Authentication attempt ${attempt} failed: ${error}`);
    }
  }

  return null;
}

// Helper functions
function getCurrentMemoryUsage() {
  // This would be implemented with actual system monitoring
  return Math.random() * 80 + 20; // Placeholder: 20-100%
}

function getCurrentConnectionCount() {
  // This would be implemented with actual connection monitoring
  return Math.floor(Math.random() * 100) + 50; // Placeholder: 50-150 connections
}

// Setup function
export function setup() {
  console.log('Starting 2-hour soak test for long-term stability');
  console.log(`Environment: ${environment.baseURL}`);
  console.log('Soak test objectives:');
  console.log('- Detect memory leaks and resource growth');
  console.log('- Monitor performance degradation over time');
  console.log('- Validate system stability under sustained load');
  console.log('- Test connection pooling and resource management');
  
  // Initial system health verification
  const healthResponse = http.get(`${environment.baseURL}/api/cluster/health`);
  
  if (healthResponse.status !== 200) {
    throw new Error(`System not ready for soak testing: ${healthResponse.status}`);
  }

  const health = healthResponse.json();
  console.log(`Initial system health: ${health.status}`);
  console.log(`Starting soak test at: ${new Date().toISOString()}`);
  
  return { 
    startTime: Date.now(),
    initialHealth: health,
    testDuration: '2 hours'
  };
}

// Teardown function with comprehensive soak test analysis
export function teardown(data) {
  const totalDuration = (Date.now() - data.startTime) / 1000;
  const hours = (totalDuration / 3600).toFixed(2);
  
  console.log('\n=== Soak Test Results ===');
  console.log(`Total duration: ${hours} hours (${totalDuration} seconds)`);
  console.log(`Operations performed: ${soakTestState.operationsPerformed}`);
  console.log(`Long-running VMs created: ${soakTestState.longRunningVMs.size}`);
  console.log(`Persistent WebSocket connections: ${soakTestState.persistentConnections.size}`);
  
  // Resource usage analysis
  console.log('\nResource Usage Analysis:');
  if (soakTestState.hourlySnapshots.length > 0) {
    soakTestState.hourlySnapshots.forEach((snapshot, index) => {
      console.log(`Hour ${snapshot.hour}: Memory ${snapshot.memoryUsage.toFixed(1)}%, VMs ${snapshot.vmCount}, WS ${snapshot.wsConnections}`);
    });
  }

  // Memory leak detection
  const memoryLeakRate = soakTestMetrics.memoryLeakRate.avg;
  if (memoryLeakRate > 2) {
    console.log(`⚠️  Potential memory leak: ${memoryLeakRate.toFixed(2)}% growth/hour`);
  } else {
    console.log(`✓ Memory usage stable: ${memoryLeakRate.toFixed(2)}% growth/hour`);
  }

  // Performance stability
  const performanceDegradation = soakTestMetrics.performanceDegradation.avg;
  if (performanceDegradation > 15) {
    console.log(`⚠️  Performance degradation: ${performanceDegradation.toFixed(2)}%`);
  } else {
    console.log(`✓ Performance stable: ${performanceDegradation.toFixed(2)}% change`);
  }

  // System stability
  const stabilityRate = soakTestMetrics.systemStability.rate;
  console.log(`System stability: ${(stabilityRate * 100).toFixed(2)}%`);
  
  // Long-term recommendations
  console.log('\nLong-term Stability Recommendations:');
  
  if (memoryLeakRate > 2) {
    console.log('- Investigate memory management in VM operations');
    console.log('- Review WebSocket connection lifecycle');
    console.log('- Implement memory monitoring alerts');
  }
  
  if (performanceDegradation > 10) {
    console.log('- Analyze database query optimization opportunities');
    console.log('- Review caching strategies for long-running operations');
    console.log('- Consider implementing performance monitoring');
  }
  
  if (stabilityRate < 0.98) {
    console.log('- Improve error handling and recovery mechanisms');
    console.log('- Implement health checks and auto-recovery');
    console.log('- Review system resource limits and scaling policies');
  }
  
  console.log('=== End Soak Test Results ===\n');
  
  // Cleanup long-running resources
  console.log('Cleaning up soak test resources...');
  const user = testData.users.operator;
  const cleanupAuthResponse = http.post(`${environment.baseURL}/api/auth/login`, 
    JSON.stringify({ username: user.username, password: user.password }), {
      headers: { 'Content-Type': 'application/json' }
    }
  );

  if (cleanupAuthResponse.status === 200) {
    const cleanupToken = cleanupAuthResponse.json().token;
    const cleanupHeaders = { 'Authorization': `Bearer ${cleanupToken}` };
    
    // Cleanup long-running VMs
    for (const [key, vm] of soakTestState.longRunningVMs) {
      try {
        http.post(`${environment.baseURL}/api/vms/${vm.id}/stop`, null, { headers: cleanupHeaders });
        sleep(1);
        http.del(`${environment.baseURL}/api/vms/${vm.id}`, null, { headers: cleanupHeaders });
        console.log(`Cleaned up soak test VM: ${vm.name}`);
      } catch (error) {
        console.error(`Failed to cleanup VM ${vm.name}: ${error}`);
      }
    }
    
    // Close persistent WebSocket connections
    for (const [key, connection] of soakTestState.persistentConnections) {
      try {
        if (connection.socket && connection.socket.readyState === 1) {
          connection.socket.close();
        }
      } catch (error) {
        console.error(`Failed to close WebSocket ${key}: ${error}`);
      }
    }
  }
  
  console.log('Soak test cleanup completed');
}