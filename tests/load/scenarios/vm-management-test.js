import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter, Gauge } from 'k6/metrics';
import { config, getEnvironment, getTestData } from '../configs/test-config.js';

// Custom metrics for VM operations
const vmCreationTime = new Trend('vm_creation_duration');
const vmStartTime = new Trend('vm_start_duration');
const vmStopTime = new Trend('vm_stop_duration');
const vmDeletionTime = new Trend('vm_deletion_duration');
const vmMigrationTime = new Trend('vm_migration_duration');
const vmSnapshotTime = new Trend('vm_snapshot_duration');

const vmCreationRate = new Rate('vm_creation_success_rate');
const vmOperationRate = new Rate('vm_operation_success_rate');
const activeVMs = new Gauge('active_vms_count');
const vmLifecycleCounter = new Counter('vm_lifecycle_operations');

// Test configuration
export const options = {
  scenarios: {
    vm_management_test: config.scenarios.vm_management,
  },
  thresholds: {
    'vm_creation_duration': config.thresholds.vm_creation_duration,
    'vm_start_duration': config.thresholds.vm_start_duration,
    'vm_stop_duration': config.thresholds.vm_stop_duration,
    'vm_creation_success_rate': ['rate>0.95'],
    'vm_operation_success_rate': ['rate>0.98']
  }
};

const environment = getEnvironment();
const testData = getTestData();

// VM management test state
let createdVMs = [];
let vmCounter = 0;

// Authentication helper
function authenticate() {
  const user = testData.users.operator; // Use operator role for VM management
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

// VM lifecycle operations
function createVM(token, config) {
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  };

  const vmPayload = {
    name: `${config.name}-${vmCounter++}-${__VU}-${__ITER}`,
    cpu_shares: config.cpu,
    memory_mb: config.memory,
    disk_size_gb: config.disk,
    command: '/bin/bash',
    args: ['-c', 'while true; do sleep 30; done'], // Keep VM running
    tags: {
      'test': 'vm-load-test',
      'vu': __VU.toString(),
      'iteration': __ITER.toString(),
      'config': config.name
    }
  };

  const startTime = Date.now();
  const response = http.post(`${environment.baseURL}/api/vms`, 
    JSON.stringify(vmPayload), { headers }
  );
  const duration = Date.now() - startTime;

  vmCreationTime.add(duration);
  vmLifecycleCounter.add(1, { operation: 'create' });

  const success = check(response, {
    'VM creation successful': (r) => r.status === 201,
    'VM creation within SLA': (r) => r.timings.duration < 30000,
    'VM ID returned': (r) => r.json().id !== undefined,
    'VM name correct': (r) => r.json().name === vmPayload.name
  });

  vmCreationRate.add(success);

  if (success) {
    const vm = response.json();
    createdVMs.push({
      id: vm.id,
      name: vm.name,
      config: config,
      createdAt: Date.now()
    });
    activeVMs.add(1);
    return vm;
  }

  return null;
}

function startVM(token, vmID) {
  const headers = { 'Authorization': `Bearer ${token}` };

  const startTime = Date.now();
  const response = http.post(`${environment.baseURL}/api/vms/${vmID}/start`, null, { headers });
  const duration = Date.now() - startTime;

  vmStartTime.add(duration);
  vmLifecycleCounter.add(1, { operation: 'start' });

  const success = check(response, {
    'VM start successful': (r) => r.status === 200,
    'VM start within SLA': (r) => r.timings.duration < 10000,
    'VM state is running': (r) => r.json().state === 'running'
  });

  vmOperationRate.add(success);
  return success;
}

function stopVM(token, vmID) {
  const headers = { 'Authorization': `Bearer ${token}` };

  const startTime = Date.now();
  const response = http.post(`${environment.baseURL}/api/vms/${vmID}/stop`, null, { headers });
  const duration = Date.now() - startTime;

  vmStopTime.add(duration);
  vmLifecycleCounter.add(1, { operation: 'stop' });

  const success = check(response, {
    'VM stop successful': (r) => r.status === 200,
    'VM stop within SLA': (r) => r.timings.duration < 5000,
    'VM state is stopped': (r) => r.json().state === 'stopped'
  });

  vmOperationRate.add(success);
  return success;
}

function restartVM(token, vmID) {
  const headers = { 'Authorization': `Bearer ${token}` };

  const startTime = Date.now();
  const response = http.post(`${environment.baseURL}/api/vms/${vmID}/restart`, null, { headers });
  const duration = Date.now() - startTime;

  vmLifecycleCounter.add(1, { operation: 'restart' });

  const success = check(response, {
    'VM restart successful': (r) => r.status === 200,
    'VM restart within SLA': (r) => r.timings.duration < 60000
  });

  vmOperationRate.add(success);
  return success;
}

function migrateVM(token, vmID) {
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  };

  const migrationPayload = {
    target_host: `node-${Math.floor(Math.random() * 3) + 1}`,
    live: true
  };

  const startTime = Date.now();
  const response = http.post(`${environment.baseURL}/api/vms/${vmID}/migrate`, 
    JSON.stringify(migrationPayload), { headers }
  );
  const duration = Date.now() - startTime;

  vmMigrationTime.add(duration);
  vmLifecycleCounter.add(1, { operation: 'migrate' });

  const success = check(response, {
    'VM migration successful': (r) => r.status === 200,
    'migration response complete': (r) => r.json().status !== undefined
  });

  vmOperationRate.add(success);
  return success;
}

function snapshotVM(token, vmID) {
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  };

  const snapshotPayload = {
    name: `snapshot-${vmID}-${Date.now()}`,
    description: `Load test snapshot for VM ${vmID}`
  };

  const startTime = Date.now();
  const response = http.post(`${environment.baseURL}/api/vms/${vmID}/snapshot`, 
    JSON.stringify(snapshotPayload), { headers }
  );
  const duration = Date.now() - startTime;

  vmSnapshotTime.add(duration);
  vmLifecycleCounter.add(1, { operation: 'snapshot' });

  const success = check(response, {
    'VM snapshot successful': (r) => r.status === 200,
    'snapshot ID returned': (r) => r.json().id !== undefined
  });

  vmOperationRate.add(success);
  return success;
}

function deleteVM(token, vmID) {
  const headers = { 'Authorization': `Bearer ${token}` };

  const startTime = Date.now();
  const response = http.del(`${environment.baseURL}/api/vms/${vmID}`, null, { headers });
  const duration = Date.now() - startTime;

  vmDeletionTime.add(duration);
  vmLifecycleCounter.add(1, { operation: 'delete' });

  const success = check(response, {
    'VM deletion successful': (r) => r.status === 200
  });

  vmOperationRate.add(success);

  if (success) {
    activeVMs.add(-1);
  }

  return success;
}

// Main test function
export default function() {
  const token = authenticate();
  if (!token) {
    console.error('Authentication failed');
    return;
  }

  // Determine test scenario based on iteration
  const scenario = __ITER % 4;

  switch (scenario) {
    case 0: // Full VM lifecycle
      const vmConfig = testData.vmConfigs[Math.floor(Math.random() * testData.vmConfigs.length)];
      const vm = createVM(token, vmConfig);
      
      if (vm) {
        sleep(2); // Allow VM to initialize
        
        if (startVM(token, vm.id)) {
          sleep(3); // Allow VM to run
          
          // 50% chance to migrate VM
          if (Math.random() < 0.5) {
            migrateVM(token, vm.id);
            sleep(2);
          }
          
          // 30% chance to create snapshot
          if (Math.random() < 0.3) {
            snapshotVM(token, vm.id);
            sleep(1);
          }
          
          stopVM(token, vm.id);
          sleep(1);
        }
        
        deleteVM(token, vm.id);
      }
      break;

    case 1: // VM creation burst
      const configs = [
        testData.vmConfigs[0], // small
        testData.vmConfigs[1]  // medium
      ];
      
      for (const config of configs) {
        const vm = createVM(token, config);
        if (vm) {
          createdVMs.push(vm);
          sleep(0.5);
        }
      }
      break;

    case 2: // VM operations on existing VMs
      if (createdVMs.length > 0) {
        const randomVM = createdVMs[Math.floor(Math.random() * createdVMs.length)];
        
        // Perform random operations
        const operations = ['start', 'stop', 'restart'];
        const operation = operations[Math.floor(Math.random() * operations.length)];
        
        switch (operation) {
          case 'start':
            startVM(token, randomVM.id);
            break;
          case 'stop':
            stopVM(token, randomVM.id);
            break;
          case 'restart':
            restartVM(token, randomVM.id);
            break;
        }
        
        sleep(1);
      }
      break;

    case 3: // VM cleanup
      if (createdVMs.length > 10) { // Clean up old VMs
        const oldVM = createdVMs.shift();
        if (oldVM && Date.now() - oldVM.createdAt > 60000) { // 1 minute old
          stopVM(token, oldVM.id);
          sleep(1);
          deleteVM(token, oldVM.id);
        }
      }
      break;
  }

  sleep(Math.random() * 2 + 0.5); // 0.5-2.5 seconds between operations
}

// Setup function
export function setup() {
  console.log(`Starting VM management load test against: ${environment.baseURL}`);
  console.log(`Target: Create and manage 1000+ VMs under load`);
  
  // Verify VM API is accessible
  const user = testData.users.operator;
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
  const vmResponse = http.get(`${environment.baseURL}/api/vms`, {
    headers: { 'Authorization': `Bearer ${token}` }
  });

  if (vmResponse.status !== 200) {
    throw new Error(`VM API not accessible: ${vmResponse.status}`);
  }
  
  return { 
    startTime: Date.now(),
    initialVMCount: vmResponse.json().length
  };
}

// Teardown function
export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  console.log(`VM management load test completed in ${duration} seconds`);
  console.log(`Created VMs during test: ${createdVMs.length}`);
  console.log(`VM operations performed: ${vmLifecycleCounter.count}`);
  
  // Cleanup any remaining test VMs
  const user = testData.users.operator;
  const loginPayload = {
    username: user.username,
    password: user.password
  };

  const authResponse = http.post(`${environment.baseURL}/api/auth/login`, 
    JSON.stringify(loginPayload), {
      headers: { 'Content-Type': 'application/json' }
    }
  );

  if (authResponse.status === 200) {
    const token = authResponse.json().token;
    const headers = { 'Authorization': `Bearer ${token}` };
    
    console.log('Cleaning up test VMs...');
    for (const vm of createdVMs) {
      // Try to stop and delete each remaining VM
      http.post(`${environment.baseURL}/api/vms/${vm.id}/stop`, null, { headers });
      sleep(1);
      http.del(`${environment.baseURL}/api/vms/${vm.id}`, null, { headers });
      sleep(0.5);
    }
  }
}