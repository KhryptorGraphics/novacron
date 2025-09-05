// K6 Load Testing Script for NovaCron
import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('error_rate');
const responseTime = new Trend('response_time');
const vmCreationTime = new Trend('vm_creation_time');
const apiErrorCount = new Counter('api_errors');

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8080';
const API_BASE = `${BASE_URL}/api/v1`;

export const options = {
  stages: [
    // Ramp up to 50 users over 2 minutes
    { duration: '2m', target: 50 },
    // Stay at 50 users for 5 minutes
    { duration: '5m', target: 50 },
    // Ramp up to 100 users over 3 minutes
    { duration: '3m', target: 100 },
    // Stay at 100 users for 5 minutes
    { duration: '5m', target: 100 },
    // Ramp down to 0 users over 2 minutes
    { duration: '2m', target: 0 },
  ],
  thresholds: {
    // 99% of requests should be below 2000ms
    http_req_duration: ['p(99)<2000'],
    // Error rate should be below 5%
    error_rate: ['rate<0.05'],
    // 95% of VM creation requests should be below 5000ms
    vm_creation_time: ['p(95)<5000'],
    // API error count should be below 100
    api_errors: ['count<100'],
  },
};

// Test data
const testUsers = [
  { username: 'loadtest1', password: 'password123', role: 'admin' },
  { username: 'loadtest2', password: 'password123', role: 'user' },
  { username: 'loadtest3', password: 'password123', role: 'user' },
];

const vmConfigs = [
  {
    name: 'load-test-vm-small',
    type: 'qemu',
    cpu: 1,
    memory: 1024,
    disk: 10,
    networks: ['default'],
  },
  {
    name: 'load-test-vm-medium',
    type: 'qemu',
    cpu: 2,
    memory: 2048,
    disk: 20,
    networks: ['default'],
  },
  {
    name: 'load-test-vm-large',
    type: 'qemu',
    cpu: 4,
    memory: 4096,
    disk: 40,
    networks: ['default', 'internal'],
  },
];

// Global state
let authTokens = {};
let createdVMs = [];

export function setup() {
  // Setup test users and get auth tokens
  console.log('Setting up load test environment...');
  
  for (const user of testUsers) {
    const loginResponse = http.post(`${API_BASE}/auth/login`, 
      JSON.stringify({
        username: user.username,
        password: user.password,
      }),
      {
        headers: { 'Content-Type': 'application/json' },
      }
    );

    if (loginResponse.status === 200) {
      const loginData = JSON.parse(loginResponse.body);
      authTokens[user.username] = loginData.token;
      console.log(`Successfully logged in user: ${user.username}`);
    } else {
      console.log(`Failed to login user: ${user.username}, status: ${loginResponse.status}`);
    }
  }

  return { authTokens };
}

export default function(data) {
  const user = testUsers[Math.floor(Math.random() * testUsers.length)];
  const token = data.authTokens[user.username];
  
  if (!token) {
    console.log(`No auth token for user: ${user.username}`);
    return;
  }

  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`,
  };

  group('Health Check', () => {
    const response = http.get(`${BASE_URL}/health`);
    
    const success = check(response, {
      'health check status is 200': (r) => r.status === 200,
      'health check response time < 500ms': (r) => r.timings.duration < 500,
    });
    
    errorRate.add(!success);
    responseTime.add(response.timings.duration);
  });

  group('Authentication', () => {
    const response = http.get(`${API_BASE}/auth/me`, { headers });
    
    const success = check(response, {
      'auth me status is 200': (r) => r.status === 200,
      'auth response contains user data': (r) => {
        try {
          const data = JSON.parse(r.body);
          return data.username && data.email;
        } catch {
          return false;
        }
      },
    });
    
    errorRate.add(!success);
    responseTime.add(response.timings.duration);
    
    if (!success) {
      apiErrorCount.add(1);
    }
  });

  group('VM Operations', () => {
    // List VMs
    group('List VMs', () => {
      const response = http.get(`${API_BASE}/vms`, { headers });
      
      const success = check(response, {
        'list VMs status is 200': (r) => r.status === 200,
        'list VMs response time < 1000ms': (r) => r.timings.duration < 1000,
        'list VMs returns array': (r) => {
          try {
            const data = JSON.parse(r.body);
            return Array.isArray(data);
          } catch {
            return false;
          }
        },
      });
      
      errorRate.add(!success);
      responseTime.add(response.timings.duration);
    });

    // Create VM (25% of users)
    if (Math.random() < 0.25) {
      group('Create VM', () => {
        const vmConfig = vmConfigs[Math.floor(Math.random() * vmConfigs.length)];
        const uniqueConfig = {
          ...vmConfig,
          name: `${vmConfig.name}-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
        };

        const startTime = Date.now();
        const response = http.post(`${API_BASE}/vms`, 
          JSON.stringify(uniqueConfig), 
          { headers }
        );
        const creationTime = Date.now() - startTime;

        const success = check(response, {
          'create VM status is 201': (r) => r.status === 201,
          'create VM response contains ID': (r) => {
            try {
              const data = JSON.parse(r.body);
              if (data.id) {
                createdVMs.push({ id: data.id, user: user.username });
                return true;
              }
              return false;
            } catch {
              return false;
            }
          },
        });

        errorRate.add(!success);
        vmCreationTime.add(creationTime);
        
        if (!success) {
          apiErrorCount.add(1);
        }
      });
    }

    // VM lifecycle operations (if we have created VMs)
    if (createdVMs.length > 0 && Math.random() < 0.3) {
      const randomVM = createdVMs[Math.floor(Math.random() * createdVMs.length)];
      
      group('VM Lifecycle Operations', () => {
        // Get VM details
        const getResponse = http.get(`${API_BASE}/vms/${randomVM.id}`, { headers });
        
        check(getResponse, {
          'get VM status is 200': (r) => r.status === 200,
          'get VM response contains details': (r) => {
            try {
              const data = JSON.parse(r.body);
              return data.id === randomVM.id;
            } catch {
              return false;
            }
          },
        });

        // Random VM operation
        const operations = ['start', 'stop', 'restart'];
        const operation = operations[Math.floor(Math.random() * operations.length)];
        
        const opResponse = http.post(`${API_BASE}/vms/${randomVM.id}/${operation}`, 
          null, 
          { headers }
        );
        
        const opSuccess = check(opResponse, {
          [`${operation} VM status is 200`]: (r) => r.status === 200,
          [`${operation} VM response time < 3000ms`]: (r) => r.timings.duration < 3000,
        });
        
        errorRate.add(!opSuccess);
        
        if (!opSuccess) {
          apiErrorCount.add(1);
        }
      });
    }
  });

  group('System Monitoring', () => {
    if (user.role === 'admin') {
      // Admin-only endpoints
      const statsResponse = http.get(`${API_BASE}/admin/stats`, { headers });
      
      const success = check(statsResponse, {
        'admin stats status is 200': (r) => r.status === 200,
        'admin stats response contains metrics': (r) => {
          try {
            const data = JSON.parse(r.body);
            return data.totalVMs !== undefined && data.runningVMs !== undefined;
          } catch {
            return false;
          }
        },
      });
      
      errorRate.add(!success);
      responseTime.add(statsResponse.timings.duration);
    }

    // System health endpoint (available to all users)
    const healthResponse = http.get(`${API_BASE}/system/health`, { headers });
    
    check(healthResponse, {
      'system health status is 200': (r) => r.status === 200,
      'system health response time < 1000ms': (r) => r.timings.duration < 1000,
    });
  });

  // Random think time between 1-3 seconds
  sleep(Math.random() * 2 + 1);
}

export function teardown(data) {
  console.log('Cleaning up load test environment...');
  
  // Clean up created VMs
  if (createdVMs.length > 0) {
    console.log(`Cleaning up ${createdVMs.length} created VMs...`);
    
    for (const vm of createdVMs) {
      const token = data.authTokens[vm.user];
      if (token) {
        const headers = {
          'Authorization': `Bearer ${token}`,
        };
        
        // Stop VM first
        http.post(`${API_BASE}/vms/${vm.id}/stop`, null, { headers });
        sleep(2);
        
        // Delete VM
        const deleteResponse = http.del(`${API_BASE}/vms/${vm.id}`, null, { headers });
        
        if (deleteResponse.status === 200) {
          console.log(`Successfully deleted VM: ${vm.id}`);
        } else {
          console.log(`Failed to delete VM: ${vm.id}, status: ${deleteResponse.status}`);
        }
      }
    }
  }
  
  console.log('Load test cleanup completed.');
}

// Utility functions for extended testing
export function handleSummary(data) {
  return {
    'load-test-summary.html': htmlReport(data),
    'load-test-results.json': JSON.stringify(data, null, 2),
  };
}

function htmlReport(data) {
  const metrics = data.metrics;
  
  return `
<!DOCTYPE html>
<html>
<head>
    <title>NovaCron Load Test Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric { margin: 10px 0; }
        .success { color: green; }
        .warning { color: orange; }
        .error { color: red; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>NovaCron Load Test Results</h1>
    
    <h2>Test Configuration</h2>
    <ul>
        <li>Base URL: ${BASE_URL}</li>
        <li>Max Users: 100</li>
        <li>Duration: 17 minutes</li>
        <li>Test Users: ${testUsers.length}</li>
    </ul>
    
    <h2>Key Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Status</th></tr>
        <tr>
            <td>Average Response Time</td>
            <td>${Math.round(metrics.http_req_duration.values.avg)}ms</td>
            <td>&lt; 2000ms</td>
            <td class="${metrics.http_req_duration.values.avg < 2000 ? 'success' : 'error'}">
                ${metrics.http_req_duration.values.avg < 2000 ? '✓ PASS' : '✗ FAIL'}
            </td>
        </tr>
        <tr>
            <td>95th Percentile Response Time</td>
            <td>${Math.round(metrics.http_req_duration.values['p(95)'])}ms</td>
            <td>&lt; 3000ms</td>
            <td class="${metrics.http_req_duration.values['p(95)'] < 3000 ? 'success' : 'error'}">
                ${metrics.http_req_duration.values['p(95)'] < 3000 ? '✓ PASS' : '✗ FAIL'}
            </td>
        </tr>
        <tr>
            <td>Error Rate</td>
            <td>${Math.round(metrics.error_rate.values.rate * 100 * 100) / 100}%</td>
            <td>&lt; 5%</td>
            <td class="${metrics.error_rate.values.rate < 0.05 ? 'success' : 'error'}">
                ${metrics.error_rate.values.rate < 0.05 ? '✓ PASS' : '✗ FAIL'}
            </td>
        </tr>
        <tr>
            <td>Total Requests</td>
            <td>${metrics.http_reqs.values.count}</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>Requests/Second</td>
            <td>${Math.round(metrics.http_reqs.values.rate * 100) / 100}</td>
            <td>-</td>
            <td>-</td>
        </tr>
    </table>
    
    <h2>VM Operations</h2>
    <div class="metric">
        <strong>VM Creation Time (95th percentile):</strong> 
        ${metrics.vm_creation_time ? Math.round(metrics.vm_creation_time.values['p(95)']) + 'ms' : 'N/A'}
    </div>
    <div class="metric">
        <strong>API Errors:</strong> 
        ${metrics.api_errors ? metrics.api_errors.values.count : 0}
    </div>
    
    <h2>Test Summary</h2>
    <p>
        The load test simulated ${options.stages.reduce((max, stage) => Math.max(max, stage.target), 0)} 
        concurrent users performing various operations on the NovaCron platform.
    </p>
    
    <p><em>Report generated at: ${new Date().toISOString()}</em></p>
</body>
</html>
  `;
}