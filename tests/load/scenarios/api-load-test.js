import http from 'k6/http';
import ws from 'k6/ws';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { config, getEnvironment, getTestData } from '../configs/test-config.js';

// Custom metrics
const apiResponseTime = new Trend('api_response_time');
const apiErrorRate = new Rate('api_error_rate');
const vmOperationsCounter = new Counter('vm_operations_total');
const authenticationRate = new Rate('authentication_success_rate');

// Test configuration
export const options = {
  scenarios: {
    api_load_test: config.scenarios.api_load,
  },
  thresholds: {
    'http_req_duration': config.thresholds.http_req_duration,
    'http_req_failed': config.thresholds.http_req_failed,
    'api_response_time': ['p(95)<500'],
    'api_error_rate': ['rate<0.01'],
    'authentication_success_rate': ['rate>0.99']
  }
};

const environment = getEnvironment();
const testData = getTestData();

// Authentication helper
function authenticate(user) {
  const loginPayload = {
    username: user.username,
    password: user.password
  };

  const response = http.post(`${environment.baseURL}/api/auth/login`, 
    JSON.stringify(loginPayload), {
      headers: { 'Content-Type': 'application/json' }
    }
  );

  const success = check(response, {
    'authentication successful': (r) => r.status === 200,
    'token received': (r) => r.json().token !== undefined
  });

  authenticationRate.add(success);

  if (success) {
    return response.json().token;
  }
  return null;
}

// VM operations helper
function performVMOperations(token) {
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  };

  // List VMs
  let response = http.get(`${environment.baseURL}/api/vms`, { headers });
  check(response, {
    'list VMs successful': (r) => r.status === 200,
    'list VMs response time OK': (r) => r.timings.duration < 1000
  });
  apiResponseTime.add(response.timings.duration);

  // Create VM
  const vmConfig = testData.vmConfigs[Math.floor(Math.random() * testData.vmConfigs.length)];
  const createPayload = {
    name: `${vmConfig.name}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    cpu_shares: vmConfig.cpu,
    memory_mb: vmConfig.memory,
    disk_size_gb: vmConfig.disk,
    command: '/bin/bash',
    tags: { 'test': 'load-test', 'environment': 'testing' }
  };

  response = http.post(`${environment.baseURL}/api/vms`, 
    JSON.stringify(createPayload), { headers }
  );

  const vmCreated = check(response, {
    'VM creation successful': (r) => r.status === 201,
    'VM creation response time OK': (r) => r.timings.duration < 30000,
    'VM ID returned': (r) => r.json().id !== undefined
  });

  if (!vmCreated) {
    apiErrorRate.add(1);
    return;
  }

  const vmID = response.json().id;
  vmOperationsCounter.add(1);
  apiResponseTime.add(response.timings.duration);

  // Get VM details
  response = http.get(`${environment.baseURL}/api/vms/${vmID}`, { headers });
  check(response, {
    'get VM successful': (r) => r.status === 200,
    'VM details correct': (r) => r.json().name === createPayload.name
  });
  apiResponseTime.add(response.timings.duration);

  // Start VM
  response = http.post(`${environment.baseURL}/api/vms/${vmID}/start`, null, { headers });
  check(response, {
    'VM start successful': (r) => r.status === 200,
    'VM start response time OK': (r) => r.timings.duration < 10000
  });
  apiResponseTime.add(response.timings.duration);

  sleep(2); // Allow VM to start

  // Get VM metrics
  response = http.get(`${environment.baseURL}/api/vms/${vmID}/metrics`, { headers });
  check(response, {
    'VM metrics successful': (r) => r.status === 200,
    'metrics data present': (r) => r.json().cpu_usage !== undefined
  });
  apiResponseTime.add(response.timings.duration);

  // Stop VM
  response = http.post(`${environment.baseURL}/api/vms/${vmID}/stop`, null, { headers });
  check(response, {
    'VM stop successful': (r) => r.status === 200,
    'VM stop response time OK': (r) => r.timings.duration < 5000
  });
  apiResponseTime.add(response.timings.duration);

  sleep(1); // Allow VM to stop

  // Delete VM
  response = http.del(`${environment.baseURL}/api/vms/${vmID}`, null, { headers });
  check(response, {
    'VM deletion successful': (r) => r.status === 200
  });
  apiResponseTime.add(response.timings.duration);
}

// Storage operations helper
function performStorageOperations(token) {
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  };

  // List storage tiers
  let response = http.get(`${environment.baseURL}/api/storage/tiers`, { headers });
  check(response, {
    'list storage tiers successful': (r) => r.status === 200
  });
  apiResponseTime.add(response.timings.duration);

  // List volumes
  response = http.get(`${environment.baseURL}/api/storage/volumes`, { headers });
  check(response, {
    'list volumes successful': (r) => r.status === 200
  });
  apiResponseTime.add(response.timings.duration);

  // Create volume
  const volumePayload = {
    name: `test-volume-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    size: 10 * 1024 * 1024 * 1024, // 10GB
    tier: 'hot'
  };

  response = http.post(`${environment.baseURL}/api/storage/volumes`, 
    JSON.stringify(volumePayload), { headers }
  );

  const volumeCreated = check(response, {
    'volume creation successful': (r) => r.status === 200,
    'volume ID returned': (r) => r.json().id !== undefined
  });

  if (volumeCreated) {
    const volumeID = response.json().id;
    apiResponseTime.add(response.timings.duration);

    // Get volume details
    response = http.get(`${environment.baseURL}/api/storage/volumes/${volumeID}`, { headers });
    check(response, {
      'get volume successful': (r) => r.status === 200
    });
    apiResponseTime.add(response.timings.duration);

    // Change volume tier
    const tierPayload = { new_tier: 'warm' };
    response = http.put(`${environment.baseURL}/api/storage/volumes/${volumeID}/tier`, 
      JSON.stringify(tierPayload), { headers }
    );
    check(response, {
      'volume tier change successful': (r) => r.status === 200
    });
    apiResponseTime.add(response.timings.duration);

    // Delete volume
    response = http.del(`${environment.baseURL}/api/storage/volumes/${volumeID}`, null, { headers });
    check(response, {
      'volume deletion successful': (r) => r.status === 200
    });
    apiResponseTime.add(response.timings.duration);
  }
}

// Monitoring operations helper
function performMonitoringOperations(token) {
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  };

  // Get system metrics
  let response = http.get(`${environment.baseURL}/api/monitoring/metrics`, { headers });
  check(response, {
    'system metrics successful': (r) => r.status === 200,
    'metrics data complete': (r) => {
      const data = r.json();
      return data.cpu && data.memory && data.disk && data.network;
    }
  });
  apiResponseTime.add(response.timings.duration);

  // Get alerts
  response = http.get(`${environment.baseURL}/api/monitoring/alerts`, { headers });
  check(response, {
    'alerts retrieval successful': (r) => r.status === 200
  });
  apiResponseTime.add(response.timings.duration);

  // Get events
  response = http.get(`${environment.baseURL}/api/monitoring/events?limit=50`, { headers });
  check(response, {
    'events retrieval successful': (r) => r.status === 200
  });
  apiResponseTime.add(response.timings.duration);

  // Get cluster health
  response = http.get(`${environment.baseURL}/api/cluster/health`, { headers });
  check(response, {
    'cluster health successful': (r) => r.status === 200,
    'cluster status healthy': (r) => r.json().status === 'healthy'
  });
  apiResponseTime.add(response.timings.duration);
}

// Main test function
export default function() {
  // Select random user for this iteration
  const userTypes = Object.keys(testData.users);
  const userType = userTypes[Math.floor(Math.random() * userTypes.length)];
  const user = testData.users[userType];

  // Authenticate
  const token = authenticate(user);
  if (!token) {
    apiErrorRate.add(1);
    return;
  }

  // Perform operations based on user role
  try {
    if (user.role === 'admin' || user.role === 'operator') {
      performVMOperations(token);
      performStorageOperations(token);
    }
    
    // All users can access monitoring data
    performMonitoringOperations(token);

  } catch (error) {
    console.error(`Test iteration failed: ${error}`);
    apiErrorRate.add(1);
  }

  // Think time between operations
  sleep(Math.random() * 3 + 1); // 1-4 seconds
}

// Setup function
export function setup() {
  console.log(`Starting API load test against: ${environment.baseURL}`);
  console.log(`Test scenario: ${JSON.stringify(config.scenarios.api_load)}`);
  
  // Verify API is accessible
  const response = http.get(`${environment.baseURL}/api/cluster/health`);
  if (response.status !== 200) {
    throw new Error(`API not accessible: ${response.status}`);
  }
  
  return { startTime: Date.now() };
}

// Teardown function  
export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  console.log(`API load test completed in ${duration} seconds`);
}