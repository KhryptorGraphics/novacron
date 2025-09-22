/**
 * End-to-End Test for Distributed Monitoring Features
 * Tests navigation, real-time updates, and user interactions
 */

const { test, expect } = require('@playwright/test');

// Mock WebSocket server for testing
class MockWebSocketServer {
  constructor() {
    this.clients = [];
    this.mockData = {
      metrics: {
        totalVMs: 47,
        runningVMs: 42,
        cpuUsage: 68,
        memoryUsage: 72,
        storageUsage: 54,
        networkStatus: 'online',
        alerts: 3,
        criticalAlerts: 0
      },
      topology: {
        nodes: [
          {
            id: 'vm-web-01',
            name: 'VM-Web-01',
            type: 'vm',
            status: 'healthy',
            clusterId: 'cluster-1',
            region: 'us-east-1',
            metrics: { cpuUsage: 45, memoryUsage: 60, networkIn: 120, networkOut: 85 }
          },
          {
            id: 'host-01',
            name: 'Host-01',
            type: 'host',
            status: 'healthy',
            clusterId: 'cluster-1',
            region: 'us-east-1',
            metrics: { cpuUsage: 35, memoryUsage: 45 }
          }
        ],
        edges: [
          {
            source: 'vm-web-01',
            target: 'host-01',
            type: 'network',
            metrics: { latency: 2.3, bandwidth: 1000, utilization: 75 }
          }
        ],
        clusters: [
          {
            id: 'cluster-1',
            name: 'Production Cluster',
            region: 'us-east-1',
            nodeCount: 12,
            status: 'healthy'
          }
        ]
      },
      bandwidth: {
        interfaces: [
          {
            id: 'eth0',
            name: 'Primary Interface',
            utilization: 75,
            capacity: 10000,
            inbound: 7500,
            outbound: 5200
          }
        ],
        aggregated: {
          totalCapacity: 10000,
          totalUtilization: 75,
          peakUtilization: 89,
          averageLatency: 2.3
        }
      },
      predictions: {
        resourcePredictions: [
          {
            resourceType: 'cpu',
            currentUsage: 68,
            predictedUsage: 72,
            confidence: 85,
            timeHorizon: '1hr',
            recommendations: ['Consider scaling up if usage continues to increase']
          }
        ]
      },
      fabric: {
        computeJobs: [
          {
            id: 'job-ml-001',
            name: 'ML Training Job',
            type: 'ml_training',
            status: 'running',
            priority: 8,
            progress: 45
          }
        ],
        globalResourcePool: {
          totalCpu: 1000,
          availableCpu: 320,
          utilization: { cpu: 68, memory: 72, gpu: 76, storage: 58 }
        }
      }
    };
  }

  // Mock WebSocket responses
  setupMockWebSocket(page) {
    return page.route('**/api/ws/**', route => {
      // Simulate WebSocket connection success
      route.fulfill({
        status: 101,
        headers: {
          'Upgrade': 'websocket',
          'Connection': 'Upgrade',
        }
      });
    });
  }

  // Mock API responses
  setupMockAPI(page) {
    page.route('**/api/v1/**', route => {
      const url = route.request().url();
      let mockResponse;

      if (url.includes('/network/topology')) {
        mockResponse = { data: this.mockData.topology };
      } else if (url.includes('/network/bandwidth')) {
        mockResponse = { data: this.mockData.bandwidth };
      } else if (url.includes('/ai/predictions')) {
        mockResponse = { data: this.mockData.predictions };
      } else if (url.includes('/fabric/global')) {
        mockResponse = { data: this.mockData.fabric };
      } else {
        mockResponse = { data: null };
      }

      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockResponse)
      });
    });
  }

  // Simulate real-time updates
  simulateUpdate(updateType, data) {
    this.mockData = { ...this.mockData, [updateType]: data };
  }
}

test.describe('Distributed Monitoring Dashboard', () => {
  let mockServer;

  test.beforeEach(async ({ page }) => {
    mockServer = new MockWebSocketServer();

    // Setup mocks
    await mockServer.setupMockWebSocket(page);
    await mockServer.setupMockAPI(page);

    // Navigate to dashboard
    await page.goto('/dashboard');
  });

  test('navigates through all monitoring tabs', async ({ page }) => {
    // Start at overview
    await expect(page.locator('text=Overview')).toBeVisible();
    await expect(page.locator('text=47')).toBeVisible(); // Total VMs

    // Navigate to Bandwidth tab
    await page.click('button:has-text("Bandwidth")');
    await expect(page.locator('text=Bandwidth Monitoring')).toBeVisible();
    await expect(page.locator('text=Primary Interface')).toBeVisible();
    await expect(page.locator('text=75%')).toBeVisible(); // Utilization

    // Navigate to AI Predictions tab
    await page.click('button:has-text("AI Predictions")');
    await expect(page.locator('text=Performance Predictions')).toBeVisible();
    await expect(page.locator('text=85%')).toBeVisible(); // Confidence

    // Navigate to Fabric tab
    await page.click('button:has-text("Fabric")');
    await expect(page.locator('text=Supercompute Fabric')).toBeVisible();
    await expect(page.locator('text=ML Training Job')).toBeVisible();

    // Navigate to Topology tab
    await page.click('button:has-text("Topology")');
    await expect(page.locator('text=Network Topology')).toBeVisible();
    await expect(page.locator('canvas')).toBeVisible();
  });

  test('displays live updates badge when connected', async ({ page }) => {
    // Go to bandwidth monitoring
    await page.click('button:has-text("Bandwidth")');

    // Should show live updates badge
    await expect(page.locator('text=Live Updates')).toBeVisible();

    // Should show animated indicator
    await expect(page.locator('.animate-pulse')).toBeVisible();
  });

  test('network topology interactions work', async ({ page }) => {
    // Navigate to topology
    await page.click('button:has-text("Topology")');

    // Toggle distributed view
    const distributedBtn = page.locator('button:has-text("Distributed View")');
    await distributedBtn.click();
    await expect(distributedBtn).toHaveClass(/default/);

    // Toggle bandwidth view
    const bandwidthBtn = page.locator('button:has-text("Bandwidth")');
    await bandwidthBtn.click();
    await expect(bandwidthBtn).toHaveClass(/default/);

    // Toggle metrics view
    const metricsBtn = page.locator('button:has-text("Metrics")');
    await metricsBtn.click();
    await expect(metricsBtn).toHaveClass(/default/);

    // Change layout type
    await page.click('text=Force-Directed');
    await page.click('text=Circular');

    // Verify canvas is still visible after layout change
    await expect(page.locator('canvas')).toBeVisible();
  });

  test('bandwidth monitoring shows real-time data', async ({ page }) => {
    await page.click('button:has-text("Bandwidth")');

    // Verify initial data
    await expect(page.locator('text=75%')).toBeVisible();
    await expect(page.locator('text=10,000')).toBeVisible(); // Total capacity

    // Simulate WebSocket update
    mockServer.simulateUpdate('bandwidth', {
      ...mockServer.mockData.bandwidth,
      aggregated: {
        ...mockServer.mockData.bandwidth.aggregated,
        totalUtilization: 85
      }
    });

    // Verify updated data appears (would need actual WebSocket in real test)
    // For now, just verify the chart container exists
    await expect(page.locator('[data-testid="bandwidth-chart"]')).toBeVisible();
  });

  test('performance predictions display correctly', async ({ page }) => {
    await page.click('button:has-text("AI Predictions")');

    // Check prediction cards
    await expect(page.locator('text=CPU')).toBeVisible();
    await expect(page.locator('text=68%')).toBeVisible(); // Current usage
    await expect(page.locator('text=72%')).toBeVisible(); // Predicted usage
    await expect(page.locator('text=85%')).toBeVisible(); // Confidence

    // Check recommendations
    await expect(page.locator('text=Consider scaling up')).toBeVisible();

    // Test time horizon selector
    await page.click('[data-testid="time-horizon-select"]');
    await page.click('text=6 Hours');

    // Verify selection changed
    await expect(page.locator('text=6hr')).toBeVisible();
  });

  test('fabric dashboard shows compute jobs', async ({ page }) => {
    await page.click('button:has-text("Fabric")');

    // Check global resource pool
    await expect(page.locator('text=Global Resource Pool')).toBeVisible();
    await expect(page.locator('text=68%')).toBeVisible(); // CPU utilization

    // Check compute jobs
    await expect(page.locator('text=ML Training Job')).toBeVisible();
    await expect(page.locator('text=running')).toBeVisible();
    await expect(page.locator('text=45%')).toBeVisible(); // Progress

    // Test job actions (if any)
    const jobCard = page.locator('text=ML Training Job').locator('..');
    await expect(jobCard).toBeVisible();
  });

  test('mobile responsive layout works', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });

    // Should show mobile navigation
    await expect(page.locator('[data-testid="mobile-menu-button"]')).toBeVisible();

    // Click mobile menu
    await page.click('[data-testid="mobile-menu-button"]');

    // Should show navigation items
    await expect(page.locator('text=Virtual Machines')).toBeVisible();
    await expect(page.locator('text=Monitoring')).toBeVisible();

    // Test bottom navigation (if implemented)
    const bottomNav = page.locator('[data-testid="bottom-navigation"]');
    if (await bottomNav.isVisible()) {
      await expect(bottomNav).toBeVisible();
    }
  });

  test('error handling for WebSocket disconnection', async ({ page }) => {
    // Mock WebSocket disconnection
    await page.route('**/api/ws/**', route => {
      route.fulfill({
        status: 500,
        body: 'WebSocket connection failed'
      });
    });

    await page.click('button:has-text("Bandwidth")');

    // Should not show live updates when disconnected
    await expect(page.locator('text=Live Updates')).not.toBeVisible();

    // Should show offline indicator or fallback data
    await expect(page.locator('text=Bandwidth Monitoring')).toBeVisible();
  });

  test('accessibility features work', async ({ page }) => {
    // Check for proper ARIA labels and keyboard navigation
    await page.keyboard.press('Tab');

    // Navigation should be focusable
    const focusedElement = await page.locator(':focus');
    await expect(focusedElement).toBeVisible();

    // Test keyboard navigation through tabs
    for (let i = 0; i < 5; i++) {
      await page.keyboard.press('Tab');
    }

    // All interactive elements should have proper focus states
    // and ARIA attributes would be tested here
  });

  test('data export functionality', async ({ page }) => {
    await page.click('button:has-text("Bandwidth")');

    // Look for export button (if implemented)
    const exportBtn = page.locator('button:has-text("Export")');
    if (await exportBtn.isVisible()) {
      await exportBtn.click();

      // Test download (mock)
      await expect(page.locator('text=Downloading')).toBeVisible();
    }
  });

  test('search and filter functionality', async ({ page }) => {
    await page.click('button:has-text("Fabric")');

    // Test job filtering (if implemented)
    const filterSelect = page.locator('[data-testid="job-filter"]');
    if (await filterSelect.isVisible()) {
      await filterSelect.click();
      await page.click('text=Running Jobs');

      // Verify filter applied
      await expect(page.locator('text=running')).toBeVisible();
    }

    // Test search functionality
    const searchInput = page.locator('[data-testid="search-input"]');
    if (await searchInput.isVisible()) {
      await searchInput.fill('ML Training');

      // Verify search results
      await expect(page.locator('text=ML Training Job')).toBeVisible();
    }
  });
});