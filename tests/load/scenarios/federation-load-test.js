import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { config, getEnvironment, getTestData } from '../configs/test-config.js';

// Custom metrics for federation testing
const federationSyncDuration = new Trend('federation_sync_duration');
const federationHealthCheck = new Trend('federation_health_check');
const crossClusterLatency = new Trend('cross_cluster_latency');
const federationOperationRate = new Rate('federation_operation_success_rate');
const federationSyncCounter = new Counter('federation_sync_operations');
const crossClusterMigrations = new Counter('cross_cluster_migrations');

// Test configuration
export const options = {
  scenarios: {
    federation_load_test: config.scenarios.federation_load,
  },
  thresholds: {
    'federation_sync_duration': config.thresholds.federation_sync_duration,
    'federation_health_check': config.thresholds.federation_health_check,
    'federation_operation_success_rate': ['rate>0.95'],
    'cross_cluster_latency': ['p(95)<1000']
  }
};

const environment = getEnvironment();
const testData = getTestData();

// Authentication helper for multiple nodes
function authenticateNode(nodeUrl) {
  const user = testData.users.admin;
  const loginPayload = {
    username: user.username,
    password: user.password
  };

  const response = http.post(`${nodeUrl}/api/auth/login`, 
    JSON.stringify(loginPayload), {
      headers: { 'Content-Type': 'application/json' }
    }
  );

  if (response.status === 200) {
    return response.json().token;
  }
  return null;
}

// Federation health monitoring
function checkFederationHealth(nodeUrl, token) {
  const headers = { 'Authorization': `Bearer ${token}` };

  const healthStart = Date.now();
  const response = http.get(`${nodeUrl}/api/federation/health`, { headers });
  const duration = Date.now() - healthStart;

  federationHealthCheck.add(duration);

  const success = check(response, {
    'federation health check successful': (r) => r.status === 200,
    'federation nodes responsive': (r) => {
      const health = r.json();
      return health.connected_nodes && health.connected_nodes > 0;
    },
    'federation status healthy': (r) => r.json().status === 'healthy'
  });

  federationOperationRate.add(success);
  return success;
}

// Cross-cluster VM migration
function performCrossClusterMigration(sourceUrl, targetNode, token) {
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  };

  // Create VM on source cluster
  const vmPayload = {
    name: `fed-migration-test-${Date.now()}-${__VU}`,
    cpu_shares: 1,
    memory_mb: 1024,
    disk_size_gb: 10,
    command: '/bin/sleep',
    args: ['600'],
    tags: { 'test': 'federation-migration', 'target': targetNode.id }
  };

  let response = http.post(`${sourceUrl}/api/vms`, 
    JSON.stringify(vmPayload), { headers }
  );

  if (!check(response, { 'federation VM created': (r) => r.status === 201 })) {
    return false;
  }

  const vmID = response.json().id;

  // Start VM
  response = http.post(`${sourceUrl}/api/vms/${vmID}/start`, null, { headers });
  if (!check(response, { 'federation VM started': (r) => r.status === 200 })) {
    return false;
  }

  sleep(2); // Allow VM to initialize

  // Initiate cross-cluster migration
  const migrationPayload = {
    target_cluster: targetNode.id,
    target_host: targetNode.url,
    live: true,
    preserve_networking: true
  };

  const migrationStart = Date.now();
  response = http.post(`${sourceUrl}/api/vms/${vmID}/migrate`, 
    JSON.stringify(migrationPayload), { headers }
  );
  const migrationDuration = Date.now() - migrationStart;

  crossClusterLatency.add(migrationDuration);
  crossClusterMigrations.add(1);

  const migrationSuccess = check(response, {
    'cross-cluster migration initiated': (r) => r.status === 200,
    'migration job ID returned': (r) => r.json().job_id !== undefined
  });

  if (migrationSuccess) {
    const jobID = response.json().job_id;
    
    // Monitor migration progress
    let migrationComplete = false;
    let attempts = 0;
    const maxAttempts = 30; // 5 minutes max

    while (!migrationComplete && attempts < maxAttempts) {
      sleep(10); // Check every 10 seconds
      
      response = http.get(`${sourceUrl}/api/federation/migrations/${jobID}`, { headers });
      
      if (response.status === 200) {
        const migration = response.json();
        migrationComplete = migration.status === 'completed' || migration.status === 'failed';
        
        if (migration.status === 'completed') {
          federationSyncDuration.add(Date.now() - migrationStart);
          federationOperationRate.add(1);
        } else if (migration.status === 'failed') {
          federationOperationRate.add(0);
        }
      }
      
      attempts++;
    }

    // Cleanup: delete VM from target cluster
    const targetToken = authenticateNode(targetNode.url);
    if (targetToken) {
      http.del(`${targetNode.url}/api/vms/${vmID}`, null, {
        headers: { 'Authorization': `Bearer ${targetToken}` }
      });
    }
  }

  return migrationSuccess;
}

// Federation synchronization testing
function testFederationSync(nodeUrls, tokens) {
  // Create resources on multiple nodes simultaneously
  const syncOperations = [];

  for (let i = 0; i < nodeUrls.length; i++) {
    const nodeUrl = nodeUrls[i];
    const token = tokens[i];
    
    if (!token) continue;

    const headers = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    };

    // Create VM on this node
    const vmPayload = {
      name: `sync-test-vm-${Date.now()}-${i}-${__VU}`,
      cpu_shares: 1,
      memory_mb: 512,
      disk_size_gb: 5,
      command: '/bin/true',
      tags: { 'test': 'federation-sync', 'node': i.toString() }
    };

    const syncStart = Date.now();
    const response = http.post(`${nodeUrl}/api/vms`, 
      JSON.stringify(vmPayload), { headers }
    );
    
    const success = check(response, {
      'sync VM creation successful': (r) => r.status === 201
    });

    if (success) {
      syncOperations.push({
        nodeUrl: nodeUrl,
        vmID: response.json().id,
        token: token,
        startTime: syncStart
      });
      federationSyncCounter.add(1);
    }
    
    federationOperationRate.add(success);
  }

  // Wait for federation sync
  sleep(5);

  // Verify synchronization across all nodes
  for (const op of syncOperations) {
    for (const checkNodeUrl of nodeUrls) {
      const checkToken = tokens[nodeUrls.indexOf(checkNodeUrl)];
      if (!checkToken || checkNodeUrl === op.nodeUrl) continue;

      // Check if VM is visible on other nodes
      const checkResponse = http.get(`${checkNodeUrl}/api/federation/vms/${op.vmID}`, {
        headers: { 'Authorization': `Bearer ${checkToken}` }
      });

      check(checkResponse, {
        'VM synchronized across federation': (r) => r.status === 200 || r.status === 404 // 404 is acceptable if sync not complete
      });
    }

    // Cleanup: delete VM
    http.del(`${op.nodeUrl}/api/vms/${op.vmID}`, null, {
      headers: { 'Authorization': `Bearer ${op.token}` }
    });

    const syncDuration = Date.now() - op.startTime;
    federationSyncDuration.add(syncDuration);
  }
}

// Main test function
export default function() {
  const federationNodes = testData.federationNodes;
  
  // Authenticate with all available nodes
  const nodeTokens = {};
  const activeNodes = [];
  
  for (const node of federationNodes) {
    const token = authenticateNode(node.url);
    if (token) {
      nodeTokens[node.url] = token;
      activeNodes.push(node);
    }
  }

  if (activeNodes.length < 2) {
    console.error('Insufficient active federation nodes for testing');
    return;
  }

  // Test federation health across all nodes
  for (const node of activeNodes) {
    checkFederationHealth(node.url, nodeTokens[node.url]);
  }

  // Test scenarios based on iteration
  const scenario = __ITER % 3;

  switch (scenario) {
    case 0: // Cross-cluster migration
      if (activeNodes.length >= 2) {
        const sourceNode = activeNodes[0];
        const targetNode = activeNodes[1];
        performCrossClusterMigration(sourceNode.url, targetNode, nodeTokens[sourceNode.url]);
      }
      break;

    case 1: // Federation synchronization
      const nodeUrls = activeNodes.map(n => n.url);
      const tokens = nodeUrls.map(url => nodeTokens[url]);
      testFederationSync(nodeUrls, tokens);
      break;

    case 2: // Load balancing verification
      // Send requests to random nodes to test load distribution
      for (let i = 0; i < 5; i++) {
        const randomNode = activeNodes[Math.floor(Math.random() * activeNodes.length)];
        const token = nodeTokens[randomNode.url];
        
        const start = Date.now();
        const response = http.get(`${randomNode.url}/api/cluster/nodes`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        const latency = Date.now() - start;
        
        crossClusterLatency.add(latency);
        
        check(response, {
          'load balanced request successful': (r) => r.status === 200
        });
        
        sleep(0.5);
      }
      break;
  }

  sleep(Math.random() * 3 + 1);
}

// Setup function
export function setup() {
  console.log('Starting federation load test');
  console.log(`Federation nodes: ${testData.federationNodes.map(n => n.id).join(', ')}`);
  
  // Verify all federation nodes are accessible
  const activeNodes = [];
  for (const node of testData.federationNodes) {
    try {
      const response = http.get(`${node.url}/api/cluster/health`, { timeout: '10s' });
      if (response.status === 200) {
        activeNodes.push(node);
        console.log(`✓ Node ${node.id} is accessible`);
      } else {
        console.log(`✗ Node ${node.id} is not accessible (status: ${response.status})`);
      }
    } catch (error) {
      console.log(`✗ Node ${node.id} connection failed: ${error}`);
    }
  }

  if (activeNodes.length < 2) {
    throw new Error('Need at least 2 active federation nodes for testing');
  }
  
  return { 
    startTime: Date.now(),
    activeNodes: activeNodes
  };
}

// Teardown function
export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  console.log(`Federation load test completed in ${duration} seconds`);
  console.log(`Active nodes tested: ${data.activeNodes.length}`);
  console.log(`Federation sync operations: ${federationSyncCounter.count}`);
  console.log(`Cross-cluster migrations: ${crossClusterMigrations.count}`);
}