import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { config, getEnvironment, getTestData } from '../configs/test-config.js';

// Custom metrics for database performance
const dbQueryDuration = new Trend('db_query_duration');
const dbConnectionDuration = new Trend('db_connection_duration');
const dbQueryRate = new Rate('db_query_success_rate');
const dbTransactionCounter = new Counter('db_transactions_total');
const dbConcurrentOperations = new Counter('db_concurrent_operations');

// Test configuration
export const options = {
  scenarios: {
    database_performance_test: config.scenarios.database_performance,
  },
  thresholds: {
    'db_query_duration': config.thresholds.db_query_duration,
    'db_connection_duration': config.thresholds.db_connection_duration,
    'db_query_success_rate': ['rate>0.99'],
    'http_req_duration': ['p(95)<1000']
  }
};

const environment = getEnvironment();
const testData = getTestData();

// Authentication helper
function authenticate() {
  const user = testData.users.admin; // Use admin for database operations
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

// Database query patterns through API
function performComplexQueries(token) {
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  };

  // Complex VM listing with pagination and filters
  const queryStart = Date.now();
  let response = http.get(
    `${environment.baseURL}/api/vms?page=1&pageSize=50&sortBy=createdAt&sortDir=desc&state=running`,
    { headers }
  );
  let queryDuration = Date.now() - queryStart;
  
  dbQueryDuration.add(queryDuration);
  dbTransactionCounter.add(1, { query_type: 'vm_list_complex' });

  const vmListSuccess = check(response, {
    'complex VM list query successful': (r) => r.status === 200,
    'VM list query within SLA': (r) => r.timings.duration < 100,
    'pagination data present': (r) => r.headers['X-Pagination'] !== undefined
  });
  dbQueryRate.add(vmListSuccess);

  // Storage metrics aggregation
  const metricsStart = Date.now();
  response = http.get(`${environment.baseURL}/api/storage/metrics`, { headers });
  queryDuration = Date.now() - metricsStart;
  
  dbQueryDuration.add(queryDuration);
  dbTransactionCounter.add(1, { query_type: 'storage_metrics' });

  const metricsSuccess = check(response, {
    'storage metrics query successful': (r) => r.status === 200,
    'metrics aggregation within SLA': (r) => r.timings.duration < 200
  });
  dbQueryRate.add(metricsSuccess);

  // System monitoring data
  const monitoringStart = Date.now();
  response = http.get(`${environment.baseURL}/api/monitoring/metrics?type=all`, { headers });
  queryDuration = Date.now() - monitoringStart;
  
  dbQueryDuration.add(queryDuration);
  dbTransactionCounter.add(1, { query_type: 'system_monitoring' });

  const monitoringSuccess = check(response, {
    'system monitoring query successful': (r) => r.status === 200,
    'monitoring data complete': (r) => {
      const data = r.json();
      return data.cpu && data.memory && data.disk && data.network;
    }
  });
  dbQueryRate.add(monitoringSuccess);

  // Historical events query
  const eventsStart = Date.now();
  response = http.get(`${environment.baseURL}/api/monitoring/events?limit=100`, { headers });
  queryDuration = Date.now() - eventsStart;
  
  dbQueryDuration.add(queryDuration);
  dbTransactionCounter.add(1, { query_type: 'historical_events' });

  const eventsSuccess = check(response, {
    'events query successful': (r) => r.status === 200,
    'events query within SLA': (r) => r.timings.duration < 150
  });
  dbQueryRate.add(eventsSuccess);
}

// High-frequency read operations
function performHighFrequencyReads(token) {
  const headers = { 'Authorization': `Bearer ${token}` };

  // Rapid-fire VM status checks
  for (let i = 0; i < 10; i++) {
    const start = Date.now();
    const response = http.get(`${environment.baseURL}/api/vms`, { headers });
    const duration = Date.now() - start;
    
    dbQueryDuration.add(duration);
    dbTransactionCounter.add(1, { query_type: 'vm_status_check' });
    
    const success = check(response, {
      'rapid VM status check successful': (r) => r.status === 200
    });
    dbQueryRate.add(success);
    
    sleep(0.1); // 100ms between requests
  }

  // Cluster health monitoring
  for (let i = 0; i < 5; i++) {
    const start = Date.now();
    const response = http.get(`${environment.baseURL}/api/cluster/health`, { headers });
    const duration = Date.now() - start;
    
    dbQueryDuration.add(duration);
    dbTransactionCounter.add(1, { query_type: 'cluster_health' });
    
    const success = check(response, {
      'cluster health check successful': (r) => r.status === 200,
      'cluster status available': (r) => r.json().status !== undefined
    });
    dbQueryRate.add(success);
    
    sleep(0.2); // 200ms between requests
  }
}

// Write-heavy operations
function performWriteOperations(token) {
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  };

  // Create multiple VMs rapidly
  const vmPromises = [];
  for (let i = 0; i < 5; i++) {
    const vmConfig = testData.vmConfigs[Math.floor(Math.random() * testData.vmConfigs.length)];
    const vmPayload = {
      name: `db-test-vm-${Date.now()}-${i}-${__VU}`,
      cpu_shares: vmConfig.cpu,
      memory_mb: vmConfig.memory,
      disk_size_gb: vmConfig.disk,
      command: '/bin/sleep',
      args: ['300'], // 5 minute sleep
      tags: {
        'test': 'database-performance',
        'batch': `batch-${Date.now()}`,
        'vu': __VU.toString()
      }
    };

    const start = Date.now();
    const response = http.post(`${environment.baseURL}/api/vms`, 
      JSON.stringify(vmPayload), { headers }
    );
    const duration = Date.now() - start;
    
    dbQueryDuration.add(duration);
    dbTransactionCounter.add(1, { query_type: 'vm_create_batch' });
    dbConcurrentOperations.add(1);

    const success = check(response, {
      'batch VM creation successful': (r) => r.status === 201,
      'batch creation within SLA': (r) => r.timings.duration < 30000
    });
    dbQueryRate.add(success);

    if (success) {
      vmPromises.push(response.json().id);
    }

    sleep(0.5); // 500ms between creates
  }

  // Cleanup created VMs
  for (const vmID of vmPromises) {
    const deleteStart = Date.now();
    const response = http.del(`${environment.baseURL}/api/vms/${vmID}`, null, { headers });
    const duration = Date.now() - deleteStart;
    
    dbQueryDuration.add(duration);
    dbTransactionCounter.add(1, { query_type: 'vm_delete_batch' });
    
    const success = check(response, {
      'batch VM deletion successful': (r) => r.status === 200
    });
    dbQueryRate.add(success);
    
    sleep(0.2);
  }
}

// Transaction stress testing
function performTransactionStress(token) {
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  };

  // Create VM, start it, get metrics, stop it, delete it (transaction sequence)
  const vmConfig = testData.vmConfigs[0]; // Use smallest config for speed
  const vmPayload = {
    name: `txn-test-vm-${Date.now()}-${__VU}-${__ITER}`,
    cpu_shares: vmConfig.cpu,
    memory_mb: vmConfig.memory,
    disk_size_gb: vmConfig.disk,
    command: '/bin/echo',
    args: ['transaction test'],
    tags: { 'test': 'transaction-stress' }
  };

  const transactionStart = Date.now();
  
  // Step 1: Create VM
  let response = http.post(`${environment.baseURL}/api/vms`, 
    JSON.stringify(vmPayload), { headers }
  );
  
  if (!check(response, { 'transaction VM created': (r) => r.status === 201 })) {
    dbQueryRate.add(0);
    return;
  }
  
  const vmID = response.json().id;
  
  // Step 2: Start VM
  response = http.post(`${environment.baseURL}/api/vms/${vmID}/start`, null, { headers });
  
  if (!check(response, { 'transaction VM started': (r) => r.status === 200 })) {
    dbQueryRate.add(0);
    return;
  }

  sleep(1); // Allow VM to start

  // Step 3: Get metrics
  response = http.get(`${environment.baseURL}/api/vms/${vmID}/metrics`, { headers });
  check(response, { 'transaction metrics retrieved': (r) => r.status === 200 });

  // Step 4: Stop VM
  response = http.post(`${environment.baseURL}/api/vms/${vmID}/stop`, null, { headers });
  check(response, { 'transaction VM stopped': (r) => r.status === 200 });

  sleep(0.5);

  // Step 5: Delete VM
  response = http.del(`${environment.baseURL}/api/vms/${vmID}`, null, { headers });
  const success = check(response, { 'transaction VM deleted': (r) => r.status === 200 });

  const transactionDuration = Date.now() - transactionStart;
  dbQueryDuration.add(transactionDuration);
  dbTransactionCounter.add(1, { query_type: 'full_transaction' });
  dbQueryRate.add(success);
}

// Main test function
export default function() {
  const token = authenticate();
  if (!token) {
    console.error('Authentication failed for database test');
    return;
  }

  // Mix different database access patterns
  const testType = __ITER % 4;

  switch (testType) {
    case 0: // Complex analytical queries
      performComplexQueries(token);
      break;
      
    case 1: // High-frequency reads
      performHighFrequencyReads(token);
      break;
      
    case 2: // Write-heavy operations
      performWriteOperations(token);
      break;
      
    case 3: // Transaction stress
      performTransactionStress(token);
      break;
  }

  // Variable think time
  sleep(Math.random() * 2 + 0.5);
}

// Setup function
export function setup() {
  console.log(`Starting database performance test against: ${environment.baseURL}`);
  console.log(`Database target: ${environment.dbHost}`);
  console.log(`Test focus: Query performance, transaction integrity, concurrent access`);
  
  // Verify database connectivity through API
  const user = testData.users.admin;
  const loginPayload = {
    username: user.username,
    password: user.password
  };

  const authResponse = http.post(`${environment.baseURL}/api/auth/login`, 
    JSON.stringify(loginPayload), {
      headers: { 'Content-Type': 'application/json' }
    }
  );

  if (authResponse.status !== 200) {
    throw new Error(`Authentication failed: ${authResponse.status}`);
  }

  const token = authResponse.json().token;
  
  // Test database-backed endpoints
  const endpoints = [
    '/api/vms',
    '/api/storage/volumes',
    '/api/monitoring/metrics',
    '/api/cluster/health'
  ];

  for (const endpoint of endpoints) {
    const response = http.get(`${environment.baseURL}${endpoint}`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    
    if (response.status !== 200) {
      throw new Error(`Database-backed endpoint ${endpoint} not accessible: ${response.status}`);
    }
  }
  
  return { 
    startTime: Date.now(),
    token: token 
  };
}

// Teardown function
export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  console.log(`Database performance test completed in ${duration} seconds`);
  console.log(`Total database transactions: ${dbTransactionCounter.count}`);
  console.log(`Total concurrent operations: ${dbConcurrentOperations.count}`);
  
  // Generate database performance summary
  console.log('Database Performance Summary:');
  console.log(`- Average query duration: ${dbQueryDuration.avg}ms`);
  console.log(`- 95th percentile query duration: ${dbQueryDuration.p95}ms`);
  console.log(`- Query success rate: ${dbQueryRate.rate * 100}%`);
}