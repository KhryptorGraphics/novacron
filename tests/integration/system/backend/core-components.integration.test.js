/**
 * NovaCron Backend Core Components Integration Tests
 * 
 * Comprehensive integration tests for all backend core components including:
 * - Autoscaling service integration
 * - Consensus protocol validation
 * - Federation management
 * - VM lifecycle and migration
 * - Storage and networking
 * - Authentication and security
 */

const { describe, it, beforeAll, afterAll, beforeEach, afterEach, expect } = require('@jest/globals');
const axios = require('axios');
const WebSocket = require('ws');
const { Pool } = require('pg');
const Redis = require('redis');

// Test utilities
const TestEnvironment = require('../../utils/test-environment');
const APIClient = require('../../utils/api-client');
const MetricsCollector = require('../../utils/metrics-collector');

describe('Integration: Backend Core Components', () => {
  let testEnv;
  let apiClient;
  let dbPool;
  let redisClient;
  let metricsCollector;

  beforeAll(async () => {
    console.log('🚀 Starting Backend Core Components Integration Tests...');
    
    // Initialize test environment
    testEnv = new TestEnvironment();
    await testEnv.setup();
    
    // Setup database connection
    dbPool = new Pool({
      connectionString: process.env.DB_URL || 'postgresql://postgres:postgres@localhost:5432/novacron_test',
      max: 10,
    });
    
    // Setup Redis connection
    redisClient = Redis.createClient({
      url: process.env.REDIS_URL || 'redis://localhost:6379'
    });
    await redisClient.connect();
    
    // Initialize API client
    apiClient = new APIClient({
      baseURL: process.env.NOVACRON_API_URL || 'http://localhost:8090',
      timeout: 30000
    });
    
    // Initialize metrics collector
    metricsCollector = new MetricsCollector();
    
    // Wait for services to be ready
    await testEnv.waitForServices(['api-server', 'database', 'redis']);
    
    console.log('✅ Test environment initialized successfully');
  });

  afterAll(async () => {
    console.log('🧹 Cleaning up test environment...');
    
    await dbPool?.end();
    await redisClient?.quit();
    await testEnv?.cleanup();
    
    console.log('✅ Test environment cleaned up');
  });

  beforeEach(async () => {
    // Clear test data before each test
    await testEnv.cleanupTestData();
    metricsCollector.reset();
  });

  describe('Autoscaling Service Integration', () => {
    it('should trigger autoscaling when CPU threshold is exceeded', async () => {
      const startTime = Date.now();
      
      // Create a VM with high CPU load simulation
      const vmConfig = {
        name: 'test-autoscaling-vm',
        cpu: 2,
        memory: 4096,
        disk: 20,
        image: 'ubuntu-20.04',
        metadata: {
          'autoscaling-enabled': 'true',
          'cpu-threshold': '80'
        }
      };
      
      const response = await apiClient.post('/api/v1/vms', vmConfig);
      expect(response.status).toBe(201);
      
      const vmId = response.data.id;
      
      // Wait for VM to be running
      await testEnv.waitForVMState(vmId, 'running', 30000);
      
      // Simulate high CPU load
      await apiClient.post(`/api/v1/vms/${vmId}/simulate-load`, {
        cpu: 90,
        duration: 60000
      });
      
      // Wait for autoscaling to trigger (should happen within 2 minutes)
      let scalingTriggered = false;
      let attempts = 0;
      const maxAttempts = 24; // 2 minutes with 5-second intervals
      
      while (!scalingTriggered && attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 5000));
        
        const events = await apiClient.get(`/api/v1/vms/${vmId}/events`);
        scalingTriggered = events.data.some(event => 
          event.type === 'autoscaling' && event.action === 'scale-up'
        );
        attempts++;
      }
      
      expect(scalingTriggered).toBe(true);
      
      // Verify new VM instances were created
      const vms = await apiClient.get('/api/v1/vms');
      const scaledVMs = vms.data.filter(vm => 
        vm.metadata?.['autoscaling-group'] === vmConfig.name
      );
      
      expect(scaledVMs.length).toBeGreaterThan(1);
      
      // Record metrics
      metricsCollector.record('autoscaling-trigger-time', Date.now() - startTime);
      metricsCollector.record('scaled-instances', scaledVMs.length);
      
      // Cleanup
      await Promise.all(scaledVMs.map(vm => 
        apiClient.delete(`/api/v1/vms/${vm.id}`)
      ));
    }, 180000); // 3-minute timeout

    it('should scale down when load decreases', async () => {
      // Create multiple VMs in an autoscaling group
      const vmConfigs = Array.from({ length: 3 }, (_, i) => ({
        name: `scale-down-test-vm-${i}`,
        cpu: 1,
        memory: 2048,
        disk: 10,
        image: 'ubuntu-20.04',
        metadata: {
          'autoscaling-enabled': 'true',
          'autoscaling-group': 'scale-down-test',
          'cpu-threshold': '20'
        }
      }));
      
      const vms = await Promise.all(
        vmConfigs.map(config => apiClient.post('/api/v1/vms', config))
      );
      
      const vmIds = vms.map(vm => vm.data.id);
      
      // Wait for all VMs to be running
      await Promise.all(
        vmIds.map(id => testEnv.waitForVMState(id, 'running', 30000))
      );
      
      // Simulate low CPU load for extended period
      await Promise.all(
        vmIds.map(id => apiClient.post(`/api/v1/vms/${id}/simulate-load`, {
          cpu: 10,
          duration: 120000
        }))
      );
      
      // Wait for scale-down to trigger
      let scaleDownTriggered = false;
      let attempts = 0;
      const maxAttempts = 30; // 2.5 minutes
      
      while (!scaleDownTriggered && attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 5000));
        
        const currentVMs = await apiClient.get('/api/v1/vms');
        const activeVMs = currentVMs.data.filter(vm => 
          vm.metadata?.['autoscaling-group'] === 'scale-down-test' &&
          vm.state === 'running'
        );
        
        scaleDownTriggered = activeVMs.length < 3;
        attempts++;
      }
      
      expect(scaleDownTriggered).toBe(true);
      
      // Cleanup remaining VMs
      const remainingVMs = await apiClient.get('/api/v1/vms');
      const testVMs = remainingVMs.data.filter(vm => 
        vm.metadata?.['autoscaling-group'] === 'scale-down-test'
      );
      
      await Promise.all(testVMs.map(vm => 
        apiClient.delete(`/api/v1/vms/${vm.id}`)
      ));
    }, 200000);
  });

  describe('Consensus Protocol Validation', () => {
    it('should maintain data consistency across cluster nodes', async () => {
      // Get cluster node information
      const clusterInfo = await apiClient.get('/api/v1/cluster/nodes');
      expect(clusterInfo.data.nodes.length).toBeGreaterThan(1);
      
      const nodes = clusterInfo.data.nodes;
      const leaderId = nodes.find(node => node.role === 'leader')?.id;
      expect(leaderId).toBeDefined();
      
      // Create data on leader
      const testData = {
        key: `test-consensus-${Date.now()}`,
        value: 'test-value-for-consensus',
        timestamp: new Date().toISOString()
      };
      
      const createResponse = await apiClient.post('/api/v1/consensus/data', testData);
      expect(createResponse.status).toBe(201);
      
      // Wait for replication to complete
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Verify data exists on all nodes
      for (const node of nodes) {
        const nodeData = await apiClient.get(`/api/v1/consensus/data/${testData.key}`, {
          headers: { 'X-Target-Node': node.id }
        });
        
        expect(nodeData.data.value).toBe(testData.value);
        expect(nodeData.data.timestamp).toBe(testData.timestamp);
      }
      
      // Cleanup
      await apiClient.delete(`/api/v1/consensus/data/${testData.key}`);
    });

    it('should handle leader election correctly', async () => {
      const startTime = Date.now();
      
      // Get current leader
      const initialCluster = await apiClient.get('/api/v1/cluster/nodes');
      const currentLeader = initialCluster.data.nodes.find(node => node.role === 'leader');
      expect(currentLeader).toBeDefined();
      
      console.log(`Current leader: ${currentLeader.id}`);
      
      // Simulate leader failure
      await apiClient.post(`/api/v1/cluster/nodes/${currentLeader.id}/simulate-failure`);
      
      // Wait for new leader election
      let newLeaderElected = false;
      let attempts = 0;
      const maxAttempts = 20; // 1 minute with 3-second intervals
      
      while (!newLeaderElected && attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 3000));
        
        const clusterStatus = await apiClient.get('/api/v1/cluster/nodes');
        const leader = clusterStatus.data.nodes.find(node => node.role === 'leader');
        
        if (leader && leader.id !== currentLeader.id) {
          newLeaderElected = true;
          console.log(`New leader elected: ${leader.id}`);
        }
        attempts++;
      }
      
      expect(newLeaderElected).toBe(true);
      
      // Verify cluster is still functional
      const testData = {
        key: `leader-election-test-${Date.now()}`,
        value: 'post-election-test'
      };
      
      const response = await apiClient.post('/api/v1/consensus/data', testData);
      expect(response.status).toBe(201);
      
      // Record metrics
      metricsCollector.record('leader-election-time', Date.now() - startTime);
      
      // Restore original leader (if possible)
      await apiClient.post(`/api/v1/cluster/nodes/${currentLeader.id}/restore`);
      
      // Cleanup
      await apiClient.delete(`/api/v1/consensus/data/${testData.key}`);
    }, 120000);
  });

  describe('Federation Management', () => {
    it('should successfully federate with remote cluster', async () => {
      // Check if federation is configured
      const federationConfig = await apiClient.get('/api/v1/federation/config');
      
      if (!federationConfig.data.enabled) {
        console.log('⏭️ Skipping federation tests - federation not enabled');
        return;
      }
      
      const testFederationId = `test-federation-${Date.now()}`;
      
      // Create federation request
      const federationRequest = {
        id: testFederationId,
        remoteEndpoint: 'https://remote-cluster.example.com',
        authToken: 'test-federation-token',
        capabilities: ['vm-management', 'storage-replication']
      };
      
      const response = await apiClient.post('/api/v1/federation/requests', federationRequest);
      expect(response.status).toBe(201);
      
      // In a real environment, this would involve actual remote cluster communication
      // For testing, we'll simulate the federation establishment
      await apiClient.post(`/api/v1/federation/requests/${testFederationId}/simulate-acceptance`);
      
      // Wait for federation to be established
      await new Promise(resolve => setTimeout(resolve, 5000));
      
      // Verify federation is active
      const federations = await apiClient.get('/api/v1/federation/active');
      const testFederation = federations.data.find(fed => fed.id === testFederationId);
      
      expect(testFederation).toBeDefined();
      expect(testFederation.status).toBe('active');
      
      // Test cross-cluster VM creation
      const federatedVMConfig = {
        name: 'federated-test-vm',
        cpu: 1,
        memory: 1024,
        disk: 10,
        image: 'alpine-latest',
        federation: {
          targetCluster: testFederationId,
          replicationPolicy: 'async'
        }
      };
      
      const vmResponse = await apiClient.post('/api/v1/vms', federatedVMConfig);
      expect(vmResponse.status).toBe(201);
      
      // Cleanup
      await apiClient.delete(`/api/v1/vms/${vmResponse.data.id}`);
      await apiClient.delete(`/api/v1/federation/active/${testFederationId}`);
    });
  });

  describe('VM Lifecycle and Migration', () => {
    it('should complete full VM lifecycle successfully', async () => {
      const vmConfig = {
        name: 'lifecycle-test-vm',
        cpu: 2,
        memory: 2048,
        disk: 20,
        image: 'ubuntu-20.04',
        network: 'default',
        metadata: {
          'test-category': 'lifecycle'
        }
      };
      
      // Create VM
      const createResponse = await apiClient.post('/api/v1/vms', vmConfig);
      expect(createResponse.status).toBe(201);
      
      const vmId = createResponse.data.id;
      expect(vmId).toBeDefined();
      
      // Start VM
      const startResponse = await apiClient.post(`/api/v1/vms/${vmId}/start`);
      expect(startResponse.status).toBe(200);
      
      // Wait for VM to be running
      await testEnv.waitForVMState(vmId, 'running', 60000);
      
      // Verify VM is accessible
      const vmDetails = await apiClient.get(`/api/v1/vms/${vmId}`);
      expect(vmDetails.data.state).toBe('running');
      expect(vmDetails.data.ipAddress).toBeDefined();
      
      // Pause VM
      await apiClient.post(`/api/v1/vms/${vmId}/pause`);
      await testEnv.waitForVMState(vmId, 'paused', 30000);
      
      // Resume VM
      await apiClient.post(`/api/v1/vms/${vmId}/resume`);
      await testEnv.waitForVMState(vmId, 'running', 30000);
      
      // Stop VM
      await apiClient.post(`/api/v1/vms/${vmId}/stop`);
      await testEnv.waitForVMState(vmId, 'stopped', 30000);
      
      // Delete VM
      const deleteResponse = await apiClient.delete(`/api/v1/vms/${vmId}`);
      expect(deleteResponse.status).toBe(204);
      
      // Verify VM is deleted
      const getDeletedVM = apiClient.get(`/api/v1/vms/${vmId}`);
      await expect(getDeletedVM).rejects.toMatchObject({ response: { status: 404 } });
    });

    it('should migrate VM between hosts successfully', async () => {
      // Get available hosts
      const hostsResponse = await apiClient.get('/api/v1/hosts');
      const hosts = hostsResponse.data.filter(host => host.status === 'active');
      
      if (hosts.length < 2) {
        console.log('⏭️ Skipping migration test - need at least 2 active hosts');
        return;
      }
      
      const sourceHost = hosts[0];
      const targetHost = hosts[1];
      
      // Create VM on source host
      const vmConfig = {
        name: 'migration-test-vm',
        cpu: 1,
        memory: 1024,
        disk: 10,
        image: 'alpine-latest',
        hostId: sourceHost.id,
        metadata: {
          'migration-test': 'true'
        }
      };
      
      const createResponse = await apiClient.post('/api/v1/vms', vmConfig);
      expect(createResponse.status).toBe(201);
      
      const vmId = createResponse.data.id;
      
      // Start VM
      await apiClient.post(`/api/v1/vms/${vmId}/start`);
      await testEnv.waitForVMState(vmId, 'running', 60000);
      
      // Verify VM is on source host
      const vmBeforeMigration = await apiClient.get(`/api/v1/vms/${vmId}`);
      expect(vmBeforeMigration.data.hostId).toBe(sourceHost.id);
      
      const migrationStartTime = Date.now();
      
      // Initiate migration
      const migrationResponse = await apiClient.post(`/api/v1/vms/${vmId}/migrate`, {
        targetHostId: targetHost.id,
        migrationPolicy: 'live',
        timeout: 300000 // 5 minutes
      });
      
      expect(migrationResponse.status).toBe(202);
      const migrationId = migrationResponse.data.migrationId;
      
      // Wait for migration to complete
      let migrationCompleted = false;
      let attempts = 0;
      const maxAttempts = 60; // 5 minutes with 5-second intervals
      
      while (!migrationCompleted && attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 5000));
        
        const migrationStatus = await apiClient.get(`/api/v1/migrations/${migrationId}`);
        
        if (migrationStatus.data.status === 'completed') {
          migrationCompleted = true;
        } else if (migrationStatus.data.status === 'failed') {
          throw new Error(`Migration failed: ${migrationStatus.data.error}`);
        }
        
        attempts++;
      }
      
      expect(migrationCompleted).toBe(true);
      
      // Verify VM is on target host and still running
      const vmAfterMigration = await apiClient.get(`/api/v1/vms/${vmId}`);
      expect(vmAfterMigration.data.hostId).toBe(targetHost.id);
      expect(vmAfterMigration.data.state).toBe('running');
      
      // Record migration metrics
      metricsCollector.record('migration-time', Date.now() - migrationStartTime);
      
      // Cleanup
      await apiClient.delete(`/api/v1/vms/${vmId}`);
    }, 360000); // 6-minute timeout
  });

  describe('Storage and Networking', () => {
    it('should handle storage tiering correctly', async () => {
      // Create VM with tiered storage
      const vmConfig = {
        name: 'storage-tiering-test-vm',
        cpu: 1,
        memory: 1024,
        disks: [
          {
            name: 'boot-disk',
            size: 20,
            tier: 'ssd',
            primary: true
          },
          {
            name: 'data-disk',
            size: 100,
            tier: 'hdd',
            primary: false
          }
        ],
        image: 'ubuntu-20.04'
      };
      
      const response = await apiClient.post('/api/v1/vms', vmConfig);
      expect(response.status).toBe(201);
      
      const vmId = response.data.id;
      
      // Start VM and wait for it to be running
      await apiClient.post(`/api/v1/vms/${vmId}/start`);
      await testEnv.waitForVMState(vmId, 'running', 60000);
      
      // Verify storage configuration
      const vmDetails = await apiClient.get(`/api/v1/vms/${vmId}`);
      expect(vmDetails.data.disks).toHaveLength(2);
      
      const bootDisk = vmDetails.data.disks.find(disk => disk.name === 'boot-disk');
      const dataDisk = vmDetails.data.disks.find(disk => disk.name === 'data-disk');
      
      expect(bootDisk.tier).toBe('ssd');
      expect(dataDisk.tier).toBe('hdd');
      
      // Test storage tier migration
      await apiClient.post(`/api/v1/vms/${vmId}/disks/${dataDisk.id}/migrate-tier`, {
        targetTier: 'ssd'
      });
      
      // Wait for tier migration to complete
      await new Promise(resolve => setTimeout(resolve, 30000));
      
      const vmAfterMigration = await apiClient.get(`/api/v1/vms/${vmId}`);
      const migratedDisk = vmAfterMigration.data.disks.find(disk => disk.name === 'data-disk');
      expect(migratedDisk.tier).toBe('ssd');
      
      // Cleanup
      await apiClient.delete(`/api/v1/vms/${vmId}`);
    });

    it('should isolate network traffic between VMs', async () => {
      // Create two VMs in different network segments
      const vm1Config = {
        name: 'network-test-vm-1',
        cpu: 1,
        memory: 512,
        disk: 10,
        image: 'alpine-latest',
        networkSegment: 'segment-1'
      };
      
      const vm2Config = {
        name: 'network-test-vm-2',
        cpu: 1,
        memory: 512,
        disk: 10,
        image: 'alpine-latest',
        networkSegment: 'segment-2'
      };
      
      const [vm1Response, vm2Response] = await Promise.all([
        apiClient.post('/api/v1/vms', vm1Config),
        apiClient.post('/api/v1/vms', vm2Config)
      ]);
      
      const vm1Id = vm1Response.data.id;
      const vm2Id = vm2Response.data.id;
      
      // Start both VMs
      await Promise.all([
        apiClient.post(`/api/v1/vms/${vm1Id}/start`),
        apiClient.post(`/api/v1/vms/${vm2Id}/start`)
      ]);
      
      // Wait for VMs to be running
      await Promise.all([
        testEnv.waitForVMState(vm1Id, 'running', 60000),
        testEnv.waitForVMState(vm2Id, 'running', 60000)
      ]);
      
      // Get VM IP addresses
      const [vm1Details, vm2Details] = await Promise.all([
        apiClient.get(`/api/v1/vms/${vm1Id}`),
        apiClient.get(`/api/v1/vms/${vm2Id}`)
      ]);
      
      // Test network isolation
      const connectivityTest = await apiClient.post('/api/v1/network/connectivity-test', {
        sourceVmId: vm1Id,
        targetIp: vm2Details.data.ipAddress,
        port: 80,
        timeout: 5000
      });
      
      // VMs in different segments should not be able to communicate
      expect(connectivityTest.data.connected).toBe(false);
      
      // Create network policy to allow communication
      await apiClient.post('/api/v1/network/policies', {
        name: 'test-cross-segment-policy',
        sourceSegment: 'segment-1',
        targetSegment: 'segment-2',
        action: 'allow',
        ports: [80, 443]
      });
      
      // Wait for policy to be applied
      await new Promise(resolve => setTimeout(resolve, 10000));
      
      // Test connectivity again
      const connectivityTest2 = await apiClient.post('/api/v1/network/connectivity-test', {
        sourceVmId: vm1Id,
        targetIp: vm2Details.data.ipAddress,
        port: 80,
        timeout: 5000
      });
      
      expect(connectivityTest2.data.connected).toBe(true);
      
      // Cleanup
      await Promise.all([
        apiClient.delete(`/api/v1/vms/${vm1Id}`),
        apiClient.delete(`/api/v1/vms/${vm2Id}`),
        apiClient.delete('/api/v1/network/policies/test-cross-segment-policy')
      ]);
    });
  });

  describe('Authentication and Security', () => {
    it('should enforce role-based access control', async () => {
      // Create test users with different roles
      const adminUser = {
        username: 'test-admin',
        password: 'admin-password-123',
        email: 'admin@test.com',
        role: 'admin'
      };
      
      const regularUser = {
        username: 'test-user',
        password: 'user-password-123',
        email: 'user@test.com',
        role: 'user'
      };
      
      // Create users
      const adminResponse = await apiClient.post('/api/v1/auth/users', adminUser);
      const userResponse = await apiClient.post('/api/v1/auth/users', regularUser);
      
      expect(adminResponse.status).toBe(201);
      expect(userResponse.status).toBe(201);
      
      // Get authentication tokens
      const adminLogin = await apiClient.post('/api/v1/auth/login', {
        username: adminUser.username,
        password: adminUser.password
      });
      
      const userLogin = await apiClient.post('/api/v1/auth/login', {
        username: regularUser.username,
        password: regularUser.password
      });
      
      const adminToken = adminLogin.data.token;
      const userToken = userLogin.data.token;
      
      // Create API clients with different tokens
      const adminClient = new APIClient({
        baseURL: process.env.NOVACRON_API_URL || 'http://localhost:8090',
        headers: { 'Authorization': `Bearer ${adminToken}` }
      });
      
      const userClient = new APIClient({
        baseURL: process.env.NOVACRON_API_URL || 'http://localhost:8090',
        headers: { 'Authorization': `Bearer ${userToken}` }
      });
      
      // Test admin can access admin endpoints
      const usersListAdmin = await adminClient.get('/api/v1/auth/users');
      expect(usersListAdmin.status).toBe(200);
      
      // Test regular user cannot access admin endpoints
      const usersListUser = userClient.get('/api/v1/auth/users');
      await expect(usersListUser).rejects.toMatchObject({ response: { status: 403 } });
      
      // Test both can access general endpoints
      const vmListAdmin = await adminClient.get('/api/v1/vms');
      const vmListUser = await userClient.get('/api/v1/vms');
      
      expect(vmListAdmin.status).toBe(200);
      expect(vmListUser.status).toBe(200);
      
      // Cleanup users
      await Promise.all([
        adminClient.delete(`/api/v1/auth/users/${adminResponse.data.id}`),
        adminClient.delete(`/api/v1/auth/users/${userResponse.data.id}`)
      ]);
    });

    it('should validate API rate limiting', async () => {
      const startTime = Date.now();
      
      // Make many requests rapidly
      const promises = Array.from({ length: 50 }, () => 
        apiClient.get('/api/v1/vms').catch(err => err.response)
      );
      
      const responses = await Promise.all(promises);
      
      // Check for rate limiting responses (429 status)
      const rateLimitedResponses = responses.filter(response => 
        response && response.status === 429
      );
      
      expect(rateLimitedResponses.length).toBeGreaterThan(0);
      
      // Verify rate limit headers
      const rateLimitResponse = rateLimitedResponses[0];
      expect(rateLimitResponse.headers['x-rate-limit-limit']).toBeDefined();
      expect(rateLimitResponse.headers['x-rate-limit-remaining']).toBeDefined();
      expect(rateLimitResponse.headers['x-rate-limit-reset']).toBeDefined();
      
      metricsCollector.record('rate-limit-test-time', Date.now() - startTime);
    });
  });

  describe('System Health and Monitoring', () => {
    it('should report system health metrics accurately', async () => {
      const healthResponse = await apiClient.get('/api/v1/health');
      expect(healthResponse.status).toBe(200);
      
      const health = healthResponse.data;
      expect(health.status).toBe('healthy');
      expect(health.components).toBeDefined();
      
      // Verify required components
      const requiredComponents = ['database', 'redis', 'storage', 'compute'];
      
      for (const component of requiredComponents) {
        expect(health.components[component]).toBeDefined();
        expect(health.components[component].status).toBe('healthy');
      }
      
      // Get detailed metrics
      const metricsResponse = await apiClient.get('/api/v1/metrics');
      expect(metricsResponse.status).toBe(200);
      
      const metrics = metricsResponse.data;
      expect(metrics.cpu).toBeDefined();
      expect(metrics.memory).toBeDefined();
      expect(metrics.disk).toBeDefined();
      expect(metrics.network).toBeDefined();
      expect(metrics.vms).toBeDefined();
      
      // Verify metric ranges
      expect(metrics.cpu.usage).toBeGreaterThanOrEqual(0);
      expect(metrics.cpu.usage).toBeLessThanOrEqual(100);
      expect(metrics.memory.usage).toBeGreaterThanOrEqual(0);
      expect(metrics.memory.usage).toBeLessThanOrEqual(100);
    });

    it('should handle service degradation gracefully', async () => {
      // Simulate Redis failure
      await apiClient.post('/api/v1/system/simulate-failure', {
        service: 'redis',
        duration: 30000 // 30 seconds
      });
      
      // System should still be operational but degraded
      const healthResponse = await apiClient.get('/api/v1/health');
      expect(healthResponse.status).toBe(200);
      expect(healthResponse.data.status).toBe('degraded');
      expect(healthResponse.data.components.redis.status).toBe('unhealthy');
      
      // Core functionality should still work (with fallbacks)
      const vmListResponse = await apiClient.get('/api/v1/vms');
      expect(vmListResponse.status).toBe(200);
      
      // Wait for service recovery
      await new Promise(resolve => setTimeout(resolve, 35000));
      
      // Verify service recovery
      const recoveredHealthResponse = await apiClient.get('/api/v1/health');
      expect(recoveredHealthResponse.data.status).toBe('healthy');
      expect(recoveredHealthResponse.data.components.redis.status).toBe('healthy');
    });
  });

  afterEach(() => {
    // Log test metrics
    if (metricsCollector.hasMetrics()) {
      console.log('📊 Test Metrics:', metricsCollector.getMetrics());
    }
  });
});