/**
 * NovaCron Load Testing Suite
 * 
 * Comprehensive load testing for NovaCron system including:
 * - Concurrent VM operations
 * - API endpoint load testing
 * - Database performance under load
 * - WebSocket connection scaling
 * - Resource utilization monitoring
 * - Auto-scaling trigger validation
 */

const { describe, it, beforeAll, afterAll, beforeEach, afterEach, expect } = require('@jest/globals');
const axios = require('axios');
const WebSocket = require('ws');
const { performance } = require('perf_hooks');

// Test utilities
const TestEnvironment = require('../../utils/test-environment');
const APIClient = require('../../utils/api-client');
const LoadTestRunner = require('../../utils/load-test-runner');
const MetricsCollector = require('../../utils/metrics-collector');
const ResourceMonitor = require('../../utils/resource-monitor');

describe('Integration: Load Testing', () => {
  let testEnv;
  let apiClient;
  let loadTestRunner;
  let metricsCollector;
  let resourceMonitor;

  const LOAD_TEST_CONFIG = {
    concurrent_users: parseInt(process.env.LOAD_TEST_USERS) || 50,
    test_duration: parseInt(process.env.LOAD_TEST_DURATION) || 300000, // 5 minutes
    ramp_up_time: parseInt(process.env.LOAD_TEST_RAMPUP) || 60000, // 1 minute
    think_time: parseInt(process.env.LOAD_TEST_THINK_TIME) || 1000 // 1 second
  };

  beforeAll(async () => {
    console.log('🚀 Starting Load Testing Suite...');
    console.log('📊 Load Test Configuration:', LOAD_TEST_CONFIG);
    
    // Initialize test environment
    testEnv = new TestEnvironment();
    await testEnv.setup();
    
    // Initialize components
    apiClient = new APIClient({
      baseURL: process.env.NOVACRON_API_URL || 'http://localhost:8090',
      timeout: 30000
    });
    
    loadTestRunner = new LoadTestRunner(LOAD_TEST_CONFIG);
    metricsCollector = new MetricsCollector();
    resourceMonitor = new ResourceMonitor();
    
    // Wait for services to be ready
    await testEnv.waitForServices(['api-server', 'database', 'redis']);
    
    // Start resource monitoring
    await resourceMonitor.startMonitoring();
    
    console.log('✅ Load testing environment initialized');
  });

  afterAll(async () => {
    console.log('🧹 Cleaning up load testing environment...');
    
    await resourceMonitor?.stopMonitoring();
    await testEnv?.cleanup();
    
    // Generate final report
    const finalReport = {
      configuration: LOAD_TEST_CONFIG,
      metrics: metricsCollector.getMetrics(),
      resource_usage: resourceMonitor.getFinalReport()
    };
    
    console.log('📊 Final Load Test Report:', JSON.stringify(finalReport, null, 2));
    console.log('✅ Load testing environment cleaned up');
  });

  beforeEach(async () => {
    // Reset metrics for each test
    metricsCollector.reset();
    await testEnv.cleanupTestData();
  });

  describe('VM Operations Load Testing', () => {
    it('should handle concurrent VM creation requests', async () => {
      const vmCount = 100;
      const concurrency = 20;
      const startTime = performance.now();
      
      console.log(`🔄 Creating ${vmCount} VMs with ${concurrency} concurrent requests...`);
      
      const vmConfigs = Array.from({ length: vmCount }, (_, i) => ({
        name: `load-test-vm-${i}`,
        cpu: 1,
        memory: 1024,
        disk: 10,
        image: 'alpine-latest',
        metadata: {
          'test-type': 'load-test',
          'batch-id': Math.floor(i / concurrency)
        }
      }));
      
      const results = await loadTestRunner.executeConcurrentRequests(
        vmConfigs,
        async (vmConfig) => {
          const response = await apiClient.post('/api/v1/vms', vmConfig);
          return {
            success: response.status === 201,
            vmId: response.data.id,
            responseTime: response.responseTime
          };
        },
        { concurrency }
      );
      
      const totalTime = performance.now() - startTime;
      const successfulCreations = results.filter(r => r.success);
      const averageResponseTime = results.reduce((sum, r) => sum + (r.responseTime || 0), 0) / results.length;
      
      console.log(`📊 VM Creation Results:`);
      console.log(`  - Total time: ${totalTime.toFixed(2)}ms`);
      console.log(`  - Success rate: ${(successfulCreations.length / vmCount * 100).toFixed(2)}%`);
      console.log(`  - Average response time: ${averageResponseTime.toFixed(2)}ms`);
      console.log(`  - Throughput: ${(vmCount / (totalTime / 1000)).toFixed(2)} VMs/second`);
      
      // Assertions
      expect(successfulCreations.length).toBeGreaterThan(vmCount * 0.95); // 95% success rate
      expect(averageResponseTime).toBeLessThan(5000); // 5 second average
      expect(totalTime).toBeLessThan(120000); // Complete within 2 minutes
      
      // Record metrics
      metricsCollector.record('vm-creation-success-rate', successfulCreations.length / vmCount);
      metricsCollector.record('vm-creation-average-response-time', averageResponseTime);
      metricsCollector.record('vm-creation-throughput', vmCount / (totalTime / 1000));
      
      // Cleanup
      const vmIds = successfulCreations.map(r => r.vmId);
      await cleanupVMs(vmIds);
    }, 180000); // 3-minute timeout

    it('should handle concurrent VM lifecycle operations', async () => {
      const vmCount = 50;
      const startTime = performance.now();
      
      console.log(`🔄 Testing complete VM lifecycle for ${vmCount} VMs...`);
      
      // Create VMs
      const vmConfigs = Array.from({ length: vmCount }, (_, i) => ({
        name: `lifecycle-test-vm-${i}`,
        cpu: 1,
        memory: 512,
        disk: 5,
        image: 'alpine-latest'
      }));
      
      const createResults = await loadTestRunner.executeConcurrentRequests(
        vmConfigs,
        async (vmConfig) => {
          const response = await apiClient.post('/api/v1/vms', vmConfig);
          return { vmId: response.data.id, success: response.status === 201 };
        },
        { concurrency: 10 }
      );
      
      const vmIds = createResults.filter(r => r.success).map(r => r.vmId);
      console.log(`✅ Created ${vmIds.length} VMs`);
      
      // Start VMs
      const startResults = await loadTestRunner.executeConcurrentRequests(
        vmIds,
        async (vmId) => {
          const startTime = performance.now();
          const response = await apiClient.post(`/api/v1/vms/${vmId}/start`);
          return {
            success: response.status === 200,
            responseTime: performance.now() - startTime
          };
        },
        { concurrency: 15 }
      );
      
      const startSuccessCount = startResults.filter(r => r.success).length;
      console.log(`✅ Started ${startSuccessCount} VMs`);
      
      // Wait for VMs to be running
      await new Promise(resolve => setTimeout(resolve, 30000));
      
      // Stop VMs
      const stopResults = await loadTestRunner.executeConcurrentRequests(
        vmIds,
        async (vmId) => {
          const startTime = performance.now();
          const response = await apiClient.post(`/api/v1/vms/${vmId}/stop`);
          return {
            success: response.status === 200,
            responseTime: performance.now() - startTime
          };
        },
        { concurrency: 20 }
      );
      
      const stopSuccessCount = stopResults.filter(r => r.success).length;
      console.log(`✅ Stopped ${stopSuccessCount} VMs`);
      
      // Delete VMs
      const deleteResults = await loadTestRunner.executeConcurrentRequests(
        vmIds,
        async (vmId) => {
          const response = await apiClient.delete(`/api/v1/vms/${vmId}`);
          return { success: response.status === 204 };
        },
        { concurrency: 25 }
      );
      
      const deleteSuccessCount = deleteResults.filter(r => r.success).length;
      console.log(`✅ Deleted ${deleteSuccessCount} VMs`);
      
      const totalTime = performance.now() - startTime;
      
      // Calculate metrics
      const overallSuccessRate = (startSuccessCount + stopSuccessCount + deleteSuccessCount) / (vmIds.length * 3);
      const avgStartTime = startResults.reduce((sum, r) => sum + (r.responseTime || 0), 0) / startResults.length;
      const avgStopTime = stopResults.reduce((sum, r) => sum + (r.responseTime || 0), 0) / stopResults.length;
      
      console.log(`📊 Lifecycle Test Results:`);
      console.log(`  - Total time: ${(totalTime / 1000).toFixed(2)} seconds`);
      console.log(`  - Overall success rate: ${(overallSuccessRate * 100).toFixed(2)}%`);
      console.log(`  - Average start time: ${avgStartTime.toFixed(2)}ms`);
      console.log(`  - Average stop time: ${avgStopTime.toFixed(2)}ms`);
      
      // Assertions
      expect(overallSuccessRate).toBeGreaterThan(0.90); // 90% overall success rate
      expect(avgStartTime).toBeLessThan(10000); // 10 second average start time
      expect(avgStopTime).toBeLessThan(5000); // 5 second average stop time
      
      // Record metrics
      metricsCollector.record('lifecycle-success-rate', overallSuccessRate);
      metricsCollector.record('vm-start-time', avgStartTime);
      metricsCollector.record('vm-stop-time', avgStopTime);
    }, 300000); // 5-minute timeout
  });

  describe('API Endpoint Load Testing', () => {
    it('should handle high-volume GET requests', async () => {
      const requestCount = 1000;
      const concurrency = 50;
      const startTime = performance.now();
      
      console.log(`🔄 Executing ${requestCount} GET requests with ${concurrency} concurrent connections...`);
      
      // Create some test data first
      const testVMs = await createTestVMs(20);
      
      const requests = Array.from({ length: requestCount }, (_, i) => ({
        endpoint: i % 4 === 0 ? '/api/v1/vms' :
                  i % 4 === 1 ? '/api/v1/hosts' :
                  i % 4 === 2 ? '/api/v1/cluster/nodes' :
                  '/api/v1/metrics',
        requestId: i
      }));
      
      const results = await loadTestRunner.executeConcurrentRequests(
        requests,
        async (request) => {
          const requestStart = performance.now();
          try {
            const response = await apiClient.get(request.endpoint);
            return {
              success: response.status === 200,
              responseTime: performance.now() - requestStart,
              statusCode: response.status,
              dataSize: JSON.stringify(response.data).length
            };
          } catch (error) {
            return {
              success: false,
              responseTime: performance.now() - requestStart,
              statusCode: error.response?.status || 0,
              error: error.message
            };
          }
        },
        { concurrency }
      );
      
      const totalTime = performance.now() - startTime;
      const successfulRequests = results.filter(r => r.success);
      const averageResponseTime = results.reduce((sum, r) => sum + r.responseTime, 0) / results.length;
      const throughput = requestCount / (totalTime / 1000);
      const errorRate = (requestCount - successfulRequests.length) / requestCount;
      
      // Calculate percentiles
      const responseTimes = results.map(r => r.responseTime).sort((a, b) => a - b);
      const p95ResponseTime = responseTimes[Math.floor(responseTimes.length * 0.95)];
      const p99ResponseTime = responseTimes[Math.floor(responseTimes.length * 0.99)];
      
      console.log(`📊 GET Requests Load Test Results:`);
      console.log(`  - Success rate: ${(successfulRequests.length / requestCount * 100).toFixed(2)}%`);
      console.log(`  - Error rate: ${(errorRate * 100).toFixed(2)}%`);
      console.log(`  - Average response time: ${averageResponseTime.toFixed(2)}ms`);
      console.log(`  - 95th percentile: ${p95ResponseTime.toFixed(2)}ms`);
      console.log(`  - 99th percentile: ${p99ResponseTime.toFixed(2)}ms`);
      console.log(`  - Throughput: ${throughput.toFixed(2)} requests/second`);
      
      // Assertions
      expect(successfulRequests.length / requestCount).toBeGreaterThan(0.95); // 95% success rate
      expect(averageResponseTime).toBeLessThan(1000); // 1 second average
      expect(p95ResponseTime).toBeLessThan(2000); // 2 second 95th percentile
      expect(throughput).toBeGreaterThan(50); // At least 50 requests/second
      
      // Record metrics
      metricsCollector.record('get-requests-success-rate', successfulRequests.length / requestCount);
      metricsCollector.record('get-requests-avg-response-time', averageResponseTime);
      metricsCollector.record('get-requests-p95-response-time', p95ResponseTime);
      metricsCollector.record('get-requests-throughput', throughput);
      
      // Cleanup
      await cleanupVMs(testVMs);
    }, 120000); // 2-minute timeout

    it('should handle sustained load over time', async () => {
      const duration = 60000; // 1 minute
      const requestsPerSecond = 20;
      const startTime = performance.now();
      
      console.log(`🔄 Running sustained load test for ${duration / 1000} seconds at ${requestsPerSecond} RPS...`);
      
      const results = [];
      const resourceSnapshots = [];
      
      const endTime = Date.now() + duration;
      
      while (Date.now() < endTime) {
        const intervalStart = performance.now();
        
        // Make requests for this second
        const intervalRequests = Array.from({ length: requestsPerSecond }, (_, i) => ({
          endpoint: '/api/v1/vms',
          timestamp: Date.now()
        }));
        
        const intervalResults = await loadTestRunner.executeConcurrentRequests(
          intervalRequests,
          async (request) => {
            const requestStart = performance.now();
            try {
              const response = await apiClient.get(request.endpoint);
              return {
                success: response.status === 200,
                responseTime: performance.now() - requestStart,
                timestamp: request.timestamp
              };
            } catch (error) {
              return {
                success: false,
                responseTime: performance.now() - requestStart,
                timestamp: request.timestamp,
                error: error.message
              };
            }
          },
          { concurrency: requestsPerSecond }
        );
        
        results.push(...intervalResults);
        
        // Collect resource metrics
        const resourceSnapshot = await resourceMonitor.getSnapshot();
        resourceSnapshots.push({
          timestamp: Date.now(),
          ...resourceSnapshot
        });
        
        // Wait for the remainder of the second
        const intervalTime = performance.now() - intervalStart;
        if (intervalTime < 1000) {
          await new Promise(resolve => setTimeout(resolve, 1000 - intervalTime));
        }
      }
      
      const totalTime = performance.now() - startTime;
      const successRate = results.filter(r => r.success).length / results.length;
      const avgResponseTime = results.reduce((sum, r) => sum + r.responseTime, 0) / results.length;
      const actualRPS = results.length / (totalTime / 1000);
      
      // Analyze resource usage over time
      const avgCpuUsage = resourceSnapshots.reduce((sum, s) => sum + s.cpu, 0) / resourceSnapshots.length;
      const avgMemoryUsage = resourceSnapshots.reduce((sum, s) => sum + s.memory, 0) / resourceSnapshots.length;
      const maxCpuUsage = Math.max(...resourceSnapshots.map(s => s.cpu));
      const maxMemoryUsage = Math.max(...resourceSnapshots.map(s => s.memory));
      
      console.log(`📊 Sustained Load Test Results:`);
      console.log(`  - Duration: ${(totalTime / 1000).toFixed(2)} seconds`);
      console.log(`  - Success rate: ${(successRate * 100).toFixed(2)}%`);
      console.log(`  - Average response time: ${avgResponseTime.toFixed(2)}ms`);
      console.log(`  - Actual RPS: ${actualRPS.toFixed(2)}`);
      console.log(`  - Average CPU usage: ${avgCpuUsage.toFixed(2)}%`);
      console.log(`  - Average memory usage: ${avgMemoryUsage.toFixed(2)}%`);
      console.log(`  - Peak CPU usage: ${maxCpuUsage.toFixed(2)}%`);
      console.log(`  - Peak memory usage: ${maxMemoryUsage.toFixed(2)}%`);
      
      // Assertions
      expect(successRate).toBeGreaterThan(0.95);
      expect(avgResponseTime).toBeLessThan(2000);
      expect(actualRPS).toBeGreaterThan(requestsPerSecond * 0.9); // Within 10% of target
      expect(maxCpuUsage).toBeLessThan(90); // CPU should not max out
      expect(maxMemoryUsage).toBeLessThan(85); // Memory should not max out
      
      // Record metrics
      metricsCollector.record('sustained-load-success-rate', successRate);
      metricsCollector.record('sustained-load-avg-response-time', avgResponseTime);
      metricsCollector.record('sustained-load-rps', actualRPS);
      metricsCollector.record('sustained-load-max-cpu', maxCpuUsage);
      metricsCollector.record('sustained-load-max-memory', maxMemoryUsage);
    }, 90000); // 1.5-minute timeout
  });

  describe('WebSocket Connection Scaling', () => {
    it('should handle many concurrent WebSocket connections', async () => {
      const connectionCount = 100;
      const messagesPerConnection = 10;
      const startTime = performance.now();
      
      console.log(`🔄 Testing ${connectionCount} concurrent WebSocket connections...`);
      
      const connections = [];
      const connectionPromises = [];
      const messageResults = [];
      
      // Create connections
      for (let i = 0; i < connectionCount; i++) {
        const connectionPromise = new Promise((resolve, reject) => {
          const ws = new WebSocket(`ws://localhost:8090/ws?clientId=load-test-${i}`);
          
          let messagesReceived = 0;
          const connectionStart = performance.now();
          
          ws.on('open', () => {
            console.log(`Connection ${i} established`);
            resolve({
              connection: ws,
              connectionTime: performance.now() - connectionStart,
              clientId: i
            });
          });
          
          ws.on('message', (data) => {
            messagesReceived++;
            messageResults.push({
              clientId: i,
              message: JSON.parse(data),
              timestamp: Date.now()
            });
          });
          
          ws.on('error', (error) => {
            console.error(`Connection ${i} error:`, error);
            reject(error);
          });
          
          setTimeout(() => reject(new Error('Connection timeout')), 10000);
        });
        
        connectionPromises.push(connectionPromise);
      }
      
      // Wait for all connections to establish
      const connectionResults = await Promise.allSettled(connectionPromises);
      const successfulConnections = connectionResults
        .filter(r => r.status === 'fulfilled')
        .map(r => r.value);
      
      console.log(`✅ Established ${successfulConnections.length}/${connectionCount} connections`);
      
      // Send messages through each connection
      const messagingPromises = successfulConnections.map(async (conn, index) => {
        const messages = [];
        
        for (let i = 0; i < messagesPerConnection; i++) {
          const message = {
            type: 'test',
            data: `Test message ${i} from client ${conn.clientId}`,
            timestamp: Date.now()
          };
          
          const sendStart = performance.now();
          conn.connection.send(JSON.stringify(message));
          
          messages.push({
            sent: sendStart,
            message: message
          });
          
          // Wait a bit between messages
          await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        return messages;
      });
      
      await Promise.all(messagingPromises);
      
      // Wait for all messages to be processed
      await new Promise(resolve => setTimeout(resolve, 5000));
      
      const totalTime = performance.now() - startTime;
      const connectionSuccessRate = successfulConnections.length / connectionCount;
      const avgConnectionTime = successfulConnections.reduce((sum, conn) => sum + conn.connectionTime, 0) / successfulConnections.length;
      
      console.log(`📊 WebSocket Load Test Results:`);
      console.log(`  - Connection success rate: ${(connectionSuccessRate * 100).toFixed(2)}%`);
      console.log(`  - Average connection time: ${avgConnectionTime.toFixed(2)}ms`);
      console.log(`  - Messages sent: ${successfulConnections.length * messagesPerConnection}`);
      console.log(`  - Messages received: ${messageResults.length}`);
      console.log(`  - Total test time: ${(totalTime / 1000).toFixed(2)} seconds`);
      
      // Close all connections
      successfulConnections.forEach(conn => {
        if (conn.connection.readyState === WebSocket.OPEN) {
          conn.connection.close();
        }
      });
      
      // Assertions
      expect(connectionSuccessRate).toBeGreaterThan(0.90); // 90% connection success
      expect(avgConnectionTime).toBeLessThan(5000); // 5 second average connection time
      expect(messageResults.length).toBeGreaterThan(successfulConnections.length * messagesPerConnection * 0.8); // 80% message delivery
      
      // Record metrics
      metricsCollector.record('websocket-connection-success-rate', connectionSuccessRate);
      metricsCollector.record('websocket-avg-connection-time', avgConnectionTime);
      metricsCollector.record('websocket-message-delivery-rate', messageResults.length / (successfulConnections.length * messagesPerConnection));
    }, 120000); // 2-minute timeout
  });

  describe('Auto-scaling Trigger Validation', () => {
    it('should trigger auto-scaling under high load', async () => {
      console.log('🔄 Testing auto-scaling triggers under high load...');
      
      // Get initial system state
      const initialMetrics = await apiClient.get('/api/v1/metrics');
      const initialVMCount = (await apiClient.get('/api/v1/vms')).data.length;
      
      console.log(`📊 Initial state: ${initialVMCount} VMs`);
      
      // Create high load scenario
      const loadPromises = [
        // High CPU load simulation
        createCPULoadTest(80, 120000), // 80% CPU for 2 minutes
        
        // High memory usage
        createMemoryLoadTest(75, 120000), // 75% memory for 2 minutes
        
        // High API request load
        createAPILoadTest(100, 120000) // 100 RPS for 2 minutes
      ];
      
      const loadStartTime = performance.now();
      
      // Start all load tests
      const loadResultsPromise = Promise.allSettled(loadPromises);
      
      // Monitor for auto-scaling events
      let autoScalingTriggered = false;
      let newVMsCreated = 0;
      let monitoringAttempts = 0;
      const maxMonitoringAttempts = 48; // 4 minutes with 5-second intervals
      
      while (monitoringAttempts < maxMonitoringAttempts && !autoScalingTriggered) {
        await new Promise(resolve => setTimeout(resolve, 5000));
        
        // Check for new VMs (auto-scaling)
        const currentVMs = await apiClient.get('/api/v1/vms');
        const currentVMCount = currentVMs.data.length;
        
        if (currentVMCount > initialVMCount) {
          autoScalingTriggered = true;
          newVMsCreated = currentVMCount - initialVMCount;
          console.log(`✅ Auto-scaling triggered: ${newVMsCreated} new VMs created`);
        }
        
        // Check system metrics
        const currentMetrics = await apiClient.get('/api/v1/metrics');
        console.log(`📊 Current load - CPU: ${currentMetrics.data.cpu}%, Memory: ${currentMetrics.data.memory}%`);
        
        monitoringAttempts++;
      }
      
      // Wait for load tests to complete
      await loadResultsPromise;
      
      const totalTime = performance.now() - loadStartTime;
      
      console.log(`📊 Auto-scaling Test Results:`);
      console.log(`  - Auto-scaling triggered: ${autoScalingTriggered ? 'Yes' : 'No'}`);
      console.log(`  - New VMs created: ${newVMsCreated}`);
      console.log(`  - Total test time: ${(totalTime / 1000).toFixed(2)} seconds`);
      
      // Wait for system to stabilize
      await new Promise(resolve => setTimeout(resolve, 30000));
      
      // Check if system scales back down after load decreases
      let scaleDownDetected = false;
      let scaleDownAttempts = 0;
      const maxScaleDownAttempts = 24; // 2 minutes
      
      while (scaleDownAttempts < maxScaleDownAttempts && newVMsCreated > 0) {
        await new Promise(resolve => setTimeout(resolve, 5000));
        
        const currentVMs = await apiClient.get('/api/v1/vms');
        const currentVMCount = currentVMs.data.length;
        
        if (currentVMCount < initialVMCount + newVMsCreated) {
          scaleDownDetected = true;
          console.log(`✅ Scale-down detected: ${initialVMCount + newVMsCreated - currentVMCount} VMs removed`);
          break;
        }
        
        scaleDownAttempts++;
      }
      
      // Assertions
      expect(autoScalingTriggered).toBe(true);
      expect(newVMsCreated).toBeGreaterThan(0);
      
      // Record metrics
      metricsCollector.record('autoscaling-triggered', autoScalingTriggered ? 1 : 0);
      metricsCollector.record('autoscaling-vms-created', newVMsCreated);
      metricsCollector.record('autoscaling-response-time', totalTime);
      metricsCollector.record('scaledown-detected', scaleDownDetected ? 1 : 0);
    }, 600000); // 10-minute timeout

    async function createCPULoadTest(targetCPU, duration) {
      return loadTestRunner.simulateResourceLoad('cpu', targetCPU, duration);
    }

    async function createMemoryLoadTest(targetMemory, duration) {
      return loadTestRunner.simulateResourceLoad('memory', targetMemory, duration);
    }

    async function createAPILoadTest(rps, duration) {
      const requests = Array.from({ length: Math.ceil(rps * duration / 1000) }, () => ({
        endpoint: '/api/v1/vms'
      }));
      
      return loadTestRunner.executeConcurrentRequests(
        requests,
        async (request) => {
          const response = await apiClient.get(request.endpoint);
          return { success: response.status === 200 };
        },
        { 
          concurrency: Math.min(rps, 50),
          delayBetweenRequests: 1000 / rps
        }
      );
    }
  });

  // Utility functions
  async function createTestVMs(count) {
    const vmConfigs = Array.from({ length: count }, (_, i) => ({
      name: `load-test-data-vm-${i}`,
      cpu: 1,
      memory: 512,
      disk: 5,
      image: 'alpine-latest'
    }));
    
    const results = await Promise.all(
      vmConfigs.map(config => apiClient.post('/api/v1/vms', config))
    );
    
    return results.map(r => r.data.id);
  }

  async function cleanupVMs(vmIds) {
    if (!vmIds || vmIds.length === 0) return;
    
    console.log(`🧹 Cleaning up ${vmIds.length} test VMs...`);
    
    const cleanupPromises = vmIds.map(id =>
      apiClient.delete(`/api/v1/vms/${id}`).catch(error => {
        console.warn(`Failed to cleanup VM ${id}:`, error.message);
      })
    );
    
    await Promise.allSettled(cleanupPromises);
  }
});