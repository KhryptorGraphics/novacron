/**
 * NovaCron Frontend-Backend Integration Tests
 * 
 * Comprehensive integration tests for frontend-backend communication including:
 * - REST API integration
 * - WebSocket real-time updates
 * - Authentication flows
 * - State synchronization
 * - Error handling and recovery
 * - Performance validation
 */

const { describe, it, beforeAll, afterAll, beforeEach, afterEach, expect } = require('@jest/globals');
const puppeteer = require('puppeteer');
const WebSocket = require('ws');
const axios = require('axios');

// Test utilities
const TestEnvironment = require('../../utils/test-environment');
const APIClient = require('../../utils/api-client');
const FrontendTestHelper = require('../../utils/frontend-test-helper');

describe('Integration: Frontend-Backend Communication', () => {
  let testEnv;
  let browser;
  let page;
  let apiClient;
  let frontendHelper;
  let wsConnections = [];

  const FRONTEND_URL = process.env.NOVACRON_UI_URL || 'http://localhost:8092';
  const API_URL = process.env.NOVACRON_API_URL || 'http://localhost:8090';

  beforeAll(async () => {
    console.log('🚀 Starting Frontend-Backend Integration Tests...');
    
    // Initialize test environment
    testEnv = new TestEnvironment();
    await testEnv.setup();
    
    // Launch browser
    browser = await puppeteer.launch({
      headless: process.env.NOVACRON_TEST_HEADLESS !== 'false',
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    
    page = await browser.newPage();
    await page.setViewport({ width: 1920, height: 1080 });
    
    // Enable console logging in tests
    page.on('console', msg => {
      if (msg.type() === 'error') {
        console.log('🔴 Frontend Error:', msg.text());
      }
    });
    
    // Initialize API client
    apiClient = new APIClient({
      baseURL: API_URL,
      timeout: 30000
    });
    
    // Initialize frontend helper
    frontendHelper = new FrontendTestHelper(page, FRONTEND_URL);
    
    // Wait for services to be ready
    await testEnv.waitForServices(['api-server', 'frontend', 'database']);
    
    console.log('✅ Test environment initialized successfully');
  });

  afterAll(async () => {
    console.log('🧹 Cleaning up test environment...');
    
    // Close all WebSocket connections
    for (const ws of wsConnections) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    }
    
    await browser?.close();
    await testEnv?.cleanup();
    
    console.log('✅ Test environment cleaned up');
  });

  beforeEach(async () => {
    // Clear test data and reset page state
    await testEnv.cleanupTestData();
    await page.goto(FRONTEND_URL);
    await frontendHelper.waitForAppLoad();
  });

  afterEach(async () => {
    // Close any WebSocket connections created during test
    for (const ws of wsConnections.splice(0)) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    }
  });

  describe('REST API Integration', () => {
    it('should load VM dashboard with real data from API', async () => {
      // Create test VMs via API
      const testVMs = [
        {
          name: 'frontend-test-vm-1',
          cpu: 2,
          memory: 4096,
          disk: 20,
          image: 'ubuntu-20.04',
          state: 'running'
        },
        {
          name: 'frontend-test-vm-2',
          cpu: 1,
          memory: 2048,
          disk: 10,
          image: 'alpine-latest',
          state: 'stopped'
        }
      ];
      
      const createdVMs = await Promise.all(
        testVMs.map(vm => apiClient.post('/api/v1/vms', vm))
      );
      
      const vmIds = createdVMs.map(response => response.data.id);
      
      // Navigate to VM dashboard
      await frontendHelper.navigateTo('/dashboard/vms');
      
      // Wait for VM list to load
      await page.waitForSelector('[data-testid="vm-list"]', { timeout: 10000 });
      
      // Verify VMs are displayed correctly
      const vmElements = await page.$$('[data-testid="vm-item"]');
      expect(vmElements.length).toBeGreaterThanOrEqual(2);
      
      // Check VM details are rendered correctly
      for (const testVM of testVMs) {
        const vmElement = await page.waitForSelector(`[data-testid="vm-${testVM.name}"]`);
        expect(vmElement).toBeTruthy();
        
        // Verify VM details
        const nameElement = await vmElement.$('[data-testid="vm-name"]');
        const name = await nameElement.evaluate(el => el.textContent);
        expect(name).toBe(testVM.name);
        
        const stateElement = await vmElement.$('[data-testid="vm-state"]');
        const state = await stateElement.evaluate(el => el.textContent.toLowerCase());
        expect(state).toContain(testVM.state);
      }
      
      // Cleanup
      await Promise.all(vmIds.map(id => apiClient.delete(`/api/v1/vms/${id}`)));
    });

    it('should handle VM creation through frontend form', async () => {
      // Navigate to VM creation page
      await frontendHelper.navigateTo('/vms/create');
      
      // Fill VM creation form
      await page.waitForSelector('[data-testid="vm-create-form"]');
      
      const vmData = {
        name: 'frontend-created-vm',
        cpu: '2',
        memory: '4096',
        disk: '20',
        image: 'ubuntu-20.04'
      };
      
      await page.type('[data-testid="vm-name-input"]', vmData.name);
      await page.select('[data-testid="vm-cpu-select"]', vmData.cpu);
      await page.type('[data-testid="vm-memory-input"]', vmData.memory);
      await page.type('[data-testid="vm-disk-input"]', vmData.disk);
      await page.select('[data-testid="vm-image-select"]', vmData.image);
      
      // Submit form
      await page.click('[data-testid="vm-create-submit"]');
      
      // Wait for success notification
      await page.waitForSelector('[data-testid="success-notification"]', { timeout: 10000 });
      
      // Verify VM was created in backend
      await new Promise(resolve => setTimeout(resolve, 2000)); // Allow for async processing
      
      const vmsResponse = await apiClient.get('/api/v1/vms');
      const createdVM = vmsResponse.data.find(vm => vm.name === vmData.name);
      
      expect(createdVM).toBeDefined();
      expect(createdVM.cpu).toBe(parseInt(vmData.cpu));
      expect(createdVM.memory).toBe(parseInt(vmData.memory));
      expect(createdVM.disk).toBe(parseInt(vmData.disk));
      
      // Cleanup
      await apiClient.delete(`/api/v1/vms/${createdVM.id}`);
    });

    it('should display appropriate error messages for API failures', async () => {
      // Navigate to VMs page
      await frontendHelper.navigateTo('/dashboard/vms');
      
      // Simulate API server failure
      await apiClient.post('/api/v1/system/simulate-failure', {
        service: 'api-server',
        duration: 10000
      });
      
      // Try to refresh VM list
      await page.click('[data-testid="refresh-vms-button"]');
      
      // Wait for error message to appear
      await page.waitForSelector('[data-testid="error-notification"]', { timeout: 15000 });
      
      const errorMessage = await page.$eval('[data-testid="error-notification"]', el => el.textContent);
      expect(errorMessage).toContain('Unable to load VMs');
      
      // Wait for service recovery
      await new Promise(resolve => setTimeout(resolve, 12000));
      
      // Try refresh again - should work now
      await page.click('[data-testid="refresh-vms-button"]');
      
      // Error message should disappear
      await page.waitForFunction(
        () => !document.querySelector('[data-testid="error-notification"]'),
        { timeout: 10000 }
      );
    });
  });

  describe('WebSocket Real-time Updates', () => {
    it('should receive real-time VM state updates', async () => {
      // Create a test VM
      const vmResponse = await apiClient.post('/api/v1/vms', {
        name: 'realtime-test-vm',
        cpu: 1,
        memory: 1024,
        disk: 10,
        image: 'alpine-latest'
      });
      
      const vmId = vmResponse.data.id;
      
      // Navigate to VM details page
      await frontendHelper.navigateTo(`/vms/${vmId}`);
      
      // Wait for WebSocket connection to be established
      await page.waitForFunction(
        () => window.wsConnection && window.wsConnection.readyState === 1,
        { timeout: 10000 }
      );
      
      // Start the VM via API
      await apiClient.post(`/api/v1/vms/${vmId}/start`);
      
      // Wait for state update to appear in UI
      await page.waitForFunction(
        (vmId) => {
          const stateElement = document.querySelector(`[data-testid="vm-${vmId}-state"]`);
          return stateElement && stateElement.textContent.toLowerCase().includes('starting');
        },
        { timeout: 15000 },
        vmId
      );
      
      // Wait for running state
      await page.waitForFunction(
        (vmId) => {
          const stateElement = document.querySelector(`[data-testid="vm-${vmId}-state"]`);
          return stateElement && stateElement.textContent.toLowerCase().includes('running');
        },
        { timeout: 60000 },
        vmId
      );
      
      // Stop the VM
      await apiClient.post(`/api/v1/vms/${vmId}/stop`);
      
      // Wait for stopped state
      await page.waitForFunction(
        (vmId) => {
          const stateElement = document.querySelector(`[data-testid="vm-${vmId}-state"]`);
          return stateElement && stateElement.textContent.toLowerCase().includes('stopped');
        },
        { timeout: 30000 },
        vmId
      );
      
      // Cleanup
      await apiClient.delete(`/api/v1/vms/${vmId}`);
    });

    it('should handle WebSocket connection loss and reconnection', async () => {
      // Navigate to dashboard
      await frontendHelper.navigateTo('/dashboard');
      
      // Wait for WebSocket connection
      await page.waitForFunction(
        () => window.wsConnection && window.wsConnection.readyState === 1,
        { timeout: 10000 }
      );
      
      // Check connection indicator shows connected
      await page.waitForSelector('[data-testid="connection-status-connected"]');
      
      // Simulate WebSocket server failure
      await page.evaluate(() => {
        window.wsConnection.close();
      });
      
      // Wait for disconnected indicator
      await page.waitForSelector('[data-testid="connection-status-disconnected"]', { timeout: 5000 });
      
      // Wait for automatic reconnection
      await page.waitForSelector('[data-testid="connection-status-connected"]', { timeout: 15000 });
      
      // Verify WebSocket is functional again
      const isConnected = await page.evaluate(() => {
        return window.wsConnection && window.wsConnection.readyState === 1;
      });
      
      expect(isConnected).toBe(true);
    });

    it('should receive system-wide notifications', async () => {
      // Navigate to dashboard
      await frontendHelper.navigateTo('/dashboard');
      
      // Wait for WebSocket connection
      await page.waitForFunction(
        () => window.wsConnection && window.wsConnection.readyState === 1,
        { timeout: 10000 }
      );
      
      // Trigger system notification via API
      await apiClient.post('/api/v1/notifications/broadcast', {
        type: 'system',
        level: 'info',
        title: 'Test System Notification',
        message: 'This is a test system notification',
        duration: 5000
      });
      
      // Wait for notification to appear
      await page.waitForSelector('[data-testid="system-notification"]', { timeout: 10000 });
      
      const notificationTitle = await page.$eval(
        '[data-testid="notification-title"]',
        el => el.textContent
      );
      const notificationMessage = await page.$eval(
        '[data-testid="notification-message"]',
        el => el.textContent
      );
      
      expect(notificationTitle).toBe('Test System Notification');
      expect(notificationMessage).toBe('This is a test system notification');
      
      // Wait for notification to auto-dismiss
      await page.waitForFunction(
        () => !document.querySelector('[data-testid="system-notification"]'),
        { timeout: 7000 }
      );
    });
  });

  describe('Authentication Flows', () => {
    it('should handle login flow correctly', async () => {
      // Create test user
      const testUser = {
        username: 'frontend-test-user',
        password: 'test-password-123',
        email: 'frontend-test@example.com',
        role: 'user'
      };
      
      const userResponse = await apiClient.post('/api/v1/auth/users', testUser);
      const userId = userResponse.data.id;
      
      // Go to login page
      await frontendHelper.navigateTo('/login');
      
      // Fill login form
      await page.waitForSelector('[data-testid="login-form"]');
      await page.type('[data-testid="username-input"]', testUser.username);
      await page.type('[data-testid="password-input"]', testUser.password);
      
      // Submit login
      await page.click('[data-testid="login-submit"]');
      
      // Wait for redirect to dashboard
      await page.waitForNavigation();
      expect(page.url()).toContain('/dashboard');
      
      // Verify user is logged in
      const userMenuButton = await page.waitForSelector('[data-testid="user-menu-button"]');
      expect(userMenuButton).toBeTruthy();
      
      // Verify user info in menu
      await page.click('[data-testid="user-menu-button"]');
      await page.waitForSelector('[data-testid="user-menu"]');
      
      const displayedUsername = await page.$eval('[data-testid="username-display"]', el => el.textContent);
      expect(displayedUsername).toBe(testUser.username);
      
      // Test logout
      await page.click('[data-testid="logout-button"]');
      
      // Should redirect to login page
      await page.waitForNavigation();
      expect(page.url()).toContain('/login');
      
      // Cleanup
      await apiClient.delete(`/api/v1/auth/users/${userId}`);
    });

    it('should handle authentication token expiration', async () => {
      // Login with a user
      await frontendHelper.loginAsUser({
        username: 'token-test-user',
        password: 'test-password',
        email: 'token-test@example.com'
      });
      
      // Navigate to protected page
      await frontendHelper.navigateTo('/dashboard/vms');
      
      // Simulate token expiration by clearing localStorage
      await page.evaluate(() => {
        localStorage.removeItem('authToken');
        sessionStorage.removeItem('authToken');
      });
      
      // Try to make an API call that requires authentication
      await page.click('[data-testid="refresh-vms-button"]');
      
      // Should be redirected to login page
      await page.waitForNavigation({ timeout: 10000 });
      expect(page.url()).toContain('/login');
      
      // Should show session expired message
      const sessionMessage = await page.waitForSelector('[data-testid="session-expired-message"]');
      expect(sessionMessage).toBeTruthy();
    });
  });

  describe('State Synchronization', () => {
    it('should synchronize VM states across multiple browser tabs', async () => {
      // Create test VM
      const vmResponse = await apiClient.post('/api/v1/vms', {
        name: 'sync-test-vm',
        cpu: 1,
        memory: 1024,
        disk: 10,
        image: 'alpine-latest'
      });
      
      const vmId = vmResponse.data.id;
      
      // Open second browser tab
      const page2 = await browser.newPage();
      const frontendHelper2 = new FrontendTestHelper(page2, FRONTEND_URL);
      
      // Navigate both tabs to VM details
      await Promise.all([
        frontendHelper.navigateTo(`/vms/${vmId}`),
        frontendHelper2.navigateTo(`/vms/${vmId}`)
      ]);
      
      // Wait for WebSocket connections on both tabs
      await Promise.all([
        page.waitForFunction(() => window.wsConnection && window.wsConnection.readyState === 1),
        page2.waitForFunction(() => window.wsConnection && window.wsConnection.readyState === 1)
      ]);
      
      // Start VM from first tab
      await page.click('[data-testid="start-vm-button"]');
      
      // Verify state updates on both tabs
      await Promise.all([
        page.waitForFunction(
          (vmId) => {
            const stateElement = document.querySelector(`[data-testid="vm-${vmId}-state"]`);
            return stateElement && !stateElement.textContent.toLowerCase().includes('stopped');
          },
          { timeout: 30000 },
          vmId
        ),
        page2.waitForFunction(
          (vmId) => {
            const stateElement = document.querySelector(`[data-testid="vm-${vmId}-state"]`);
            return stateElement && !stateElement.textContent.toLowerCase().includes('stopped');
          },
          { timeout: 30000 },
          vmId
        )
      ]);
      
      // Close second tab
      await page2.close();
      
      // Cleanup
      await apiClient.delete(`/api/v1/vms/${vmId}`);
    });
  });

  describe('Performance Validation', () => {
    it('should load dashboard within acceptable time limits', async () => {
      // Create multiple test VMs for realistic load
      const testVMs = Array.from({ length: 20 }, (_, i) => ({
        name: `perf-test-vm-${i}`,
        cpu: 1,
        memory: 1024,
        disk: 10,
        image: 'alpine-latest'
      }));
      
      const createdVMs = await Promise.all(
        testVMs.map(vm => apiClient.post('/api/v1/vms', vm))
      );
      
      const vmIds = createdVMs.map(response => response.data.id);
      
      // Measure dashboard load time
      const startTime = Date.now();
      
      await frontendHelper.navigateTo('/dashboard');
      
      // Wait for all components to load
      await page.waitForSelector('[data-testid="vm-stats"]');
      await page.waitForSelector('[data-testid="system-metrics"]');
      await page.waitForSelector('[data-testid="recent-activities"]');
      
      const loadTime = Date.now() - startTime;
      
      console.log(`📊 Dashboard load time: ${loadTime}ms`);
      expect(loadTime).toBeLessThan(5000); // Should load within 5 seconds
      
      // Measure VM list load time
      const vmListStartTime = Date.now();
      
      await frontendHelper.navigateTo('/dashboard/vms');
      await page.waitForSelector('[data-testid="vm-list"]');
      
      // Wait for all VM items to be rendered
      await page.waitForFunction(
        (expectedCount) => {
          const vmItems = document.querySelectorAll('[data-testid="vm-item"]');
          return vmItems.length >= expectedCount;
        },
        { timeout: 10000 },
        20
      );
      
      const vmListLoadTime = Date.now() - vmListStartTime;
      
      console.log(`📊 VM list load time: ${vmListLoadTime}ms`);
      expect(vmListLoadTime).toBeLessThan(3000); // Should load within 3 seconds
      
      // Cleanup
      await Promise.all(vmIds.map(id => apiClient.delete(`/api/v1/vms/${id}`)));
    });

    it('should handle large datasets without performance degradation', async () => {
      // Skip if running in fast mode
      if (process.env.NOVACRON_TEST_FAST === 'true') {
        console.log('⏭️ Skipping large dataset test in fast mode');
        return;
      }
      
      // Create many VMs (simulate large dataset)
      const vmCount = 100;
      const batchSize = 10;
      const vmIds = [];
      
      console.log(`Creating ${vmCount} test VMs...`);
      
      for (let i = 0; i < vmCount; i += batchSize) {
        const batch = Array.from({ length: Math.min(batchSize, vmCount - i) }, (_, j) => ({
          name: `large-dataset-vm-${i + j}`,
          cpu: 1,
          memory: 512,
          disk: 5,
          image: 'alpine-latest'
        }));
        
        const responses = await Promise.all(
          batch.map(vm => apiClient.post('/api/v1/vms', vm))
        );
        
        vmIds.push(...responses.map(r => r.data.id));
        
        console.log(`Created ${vmIds.length}/${vmCount} VMs`);
      }
      
      // Measure performance with large dataset
      const startTime = Date.now();
      
      await frontendHelper.navigateTo('/dashboard/vms');
      
      // Wait for pagination or virtualization to handle large dataset
      await page.waitForSelector('[data-testid="vm-list"]');
      await page.waitForSelector('[data-testid="vm-pagination"], [data-testid="virtual-scroll"]');
      
      const renderTime = Date.now() - startTime;
      console.log(`📊 Large dataset render time: ${renderTime}ms`);
      
      // Should handle large datasets efficiently
      expect(renderTime).toBeLessThan(8000); // 8 seconds max for 100 VMs
      
      // Test scrolling/pagination performance
      const scrollStartTime = Date.now();
      
      if (await page.$('[data-testid="vm-pagination"]')) {
        // Test pagination
        await page.click('[data-testid="next-page-button"]');
        await page.waitForSelector('[data-testid="vm-list"]');
      } else {
        // Test virtual scrolling
        await page.evaluate(() => {
          const scrollContainer = document.querySelector('[data-testid="virtual-scroll"]');
          if (scrollContainer) {
            scrollContainer.scrollTop = 2000;
          }
        });
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
      
      const scrollTime = Date.now() - scrollStartTime;
      console.log(`📊 Navigation/scroll time: ${scrollTime}ms`);
      expect(scrollTime).toBeLessThan(2000);
      
      // Cleanup (in batches to avoid overwhelming the API)
      console.log('Cleaning up test VMs...');
      for (let i = 0; i < vmIds.length; i += batchSize) {
        const batch = vmIds.slice(i, i + batchSize);
        await Promise.all(batch.map(id => 
          apiClient.delete(`/api/v1/vms/${id}`).catch(() => {}) // Ignore cleanup errors
        ));
        console.log(`Deleted ${Math.min(i + batchSize, vmIds.length)}/${vmIds.length} VMs`);
      }
    }, 300000); // 5-minute timeout for large dataset test
  });
});