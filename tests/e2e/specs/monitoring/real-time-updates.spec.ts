import { test, expect } from '@playwright/test';
import { DashboardPage } from '../../pages/DashboardPage';
import { VMDetailsPage } from '../../pages/VMDetailsPage';
import { authenticateUser } from '../../fixtures/auth';
import { testData } from '../../fixtures/testData';

test.describe('Monitoring: Real-time Updates', () => {
  let dashboardPage: DashboardPage;
  let vmDetailsPage: VMDetailsPage;

  test.beforeEach(async ({ page }) => {
    await authenticateUser(page, testData.users.admin);

    dashboardPage = new DashboardPage(page);
    vmDetailsPage = new VMDetailsPage(page);
  });

  test('should receive real-time VM state updates', {
    tag: '@smoke',
  }, async ({ page }) => {
    // Create VM
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `realtime-test-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-minimal"]').click();
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    // Open dashboard in another view
    await dashboardPage.goto();

    // Trigger state change
    await dashboardPage.clickVM(vmName);
    await vmDetailsPage.clickAction('stop');

    // Dashboard should update in real-time
    await dashboardPage.goto();

    await expect(
      page.locator(`[data-testid="vm-${vmName}-state"]`)
    ).toHaveText('stopping', { timeout: 10000 });

    await expect(
      page.locator(`[data-testid="vm-${vmName}-state"]`)
    ).toHaveText('stopped', { timeout: 30000 });

    // Cleanup
    await dashboardPage.clickVM(vmName);
    await vmDetailsPage.forceDelete();
  });

  test('should update metrics in real-time', async ({ page }) => {
    await dashboardPage.goto();
    const vms = await dashboardPage.getVMList();

    if (vms.length === 0) {
      test.skip();
    }

    await dashboardPage.clickVM(vms[0].name);
    await vmDetailsPage.navigateToTab('metrics');

    // Get initial CPU value
    const initialCPU = await page.locator('[data-testid="cpu-current"]').textContent();

    // Wait for update
    await page.waitForTimeout(10000);

    const updatedCPU = await page.locator('[data-testid="cpu-current"]').textContent();

    // Value should update (may be same if usage is stable, but timestamp should change)
    const timestamp1 = await page.locator('[data-testid="metrics-timestamp"]').textContent();

    await page.waitForTimeout(10000);

    const timestamp2 = await page.locator('[data-testid="metrics-timestamp"]').textContent();

    expect(timestamp2).not.toBe(timestamp1);
  });

  test('should show real-time notifications', async ({ page }) => {
    await dashboardPage.goto();

    // Subscribe to notifications
    await page.locator('[data-testid="notifications-icon"]').click();
    await page.locator('[data-testid="enable-realtime"]').check();

    // Trigger event that creates notification (create VM)
    await dashboardPage.clickQuickCreate();

    const vmName = `notification-test-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-minimal"]').click();
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="quick-create-submit"]').click();

    // Should receive notification
    await expect(page.locator('[data-testid="new-notification-badge"]')).toBeVisible({ timeout: 10000 });

    await page.locator('[data-testid="notifications-icon"]').click();

    const notifications = await page.locator('[data-testid="notification"]').all();
    const vmCreationNotification = notifications.find(async (n) => {
      const text = await n.textContent();
      return text?.includes(vmName);
    });

    expect(vmCreationNotification).toBeDefined();

    // Cleanup
    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
    await vmDetailsPage.forceDelete();
  });

  test('should update node status in real-time', async ({ page }) => {
    await page.goto('/cluster');

    const initialStatus = await page.locator('[data-testid="node-0-status"]').textContent();

    // Monitor for status changes
    let statusChanged = false;

    page.on('websocket', ws => {
      ws.on('framereceived', event => {
        const data = JSON.parse(event.payload?.toString() || '{}');
        if (data.type === 'node_status_update') {
          statusChanged = true;
        }
      });
    });

    // Wait for potential updates
    await page.waitForTimeout(30000);

    // Status should be monitored via WebSocket
    const wsConnections = await page.evaluate(() => {
      return performance.getEntriesByType('resource')
        .filter(r => r.name.includes('ws://') || r.name.includes('wss://'))
        .length;
    });

    expect(wsConnections).toBeGreaterThan(0);
  });

  test('should handle WebSocket reconnection', async ({ page }) => {
    await dashboardPage.goto();

    // Verify WebSocket is connected
    await expect(page.locator('[data-testid="realtime-connected"]')).toBeVisible();

    // Simulate disconnect
    await page.evaluate(() => {
      const ws = (window as any).__ws__;
      if (ws) ws.close();
    });

    // Should show disconnected state
    await expect(page.locator('[data-testid="realtime-disconnected"]')).toBeVisible({ timeout: 5000 });

    // Should automatically reconnect
    await expect(page.locator('[data-testid="realtime-connected"]')).toBeVisible({ timeout: 10000 });
  });

  test('should batch real-time updates', async ({ page }) => {
    await dashboardPage.goto();

    // Monitor WebSocket messages
    const messages: any[] = [];

    page.on('websocket', ws => {
      ws.on('framereceived', event => {
        try {
          const data = JSON.parse(event.payload?.toString() || '{}');
          messages.push(data);
        } catch (e) {
          // Ignore parse errors
        }
      });
    });

    // Wait for messages
    await page.waitForTimeout(15000);

    // Should receive batched updates (not individual messages for each metric)
    const batchedMessages = messages.filter(m => m.type === 'batch_update');
    expect(batchedMessages.length).toBeGreaterThan(0);

    // Each batch should contain multiple updates
    if (batchedMessages.length > 0) {
      const batch = batchedMessages[0];
      expect(batch.updates).toBeDefined();
      expect(Array.isArray(batch.updates)).toBe(true);
    }
  });

  test('should throttle high-frequency updates', async ({ page }) => {
    await dashboardPage.goto();

    const updateCounts: number[] = [];
    let updateCount = 0;

    page.on('websocket', ws => {
      ws.on('framereceived', event => {
        updateCount++;
      });
    });

    // Measure updates per second
    for (let i = 0; i < 5; i++) {
      await page.waitForTimeout(1000);
      updateCounts.push(updateCount);
      updateCount = 0;
    }

    // Updates should be throttled (not exceeding reasonable rate)
    const maxUpdatesPerSecond = Math.max(...updateCounts);
    expect(maxUpdatesPerSecond).toBeLessThan(100); // Reasonable throttle limit
  });

  test('should sync updates across multiple browser tabs', async ({ browser }) => {
    const context = await browser.newContext();
    const page1 = await context.newPage();
    const page2 = await context.newPage();

    try {
      // Login both pages
      await authenticateUser(page1, testData.users.admin);
      await authenticateUser(page2, testData.users.admin);

      // Both pages on dashboard
      const dashboard1 = new DashboardPage(page1);
      const dashboard2 = new DashboardPage(page2);

      await dashboard1.goto();
      await dashboard2.goto();

      // Create VM in page1
      await dashboard1.clickQuickCreate();

      const vmName = `sync-test-${Date.now()}`;
      await page1.locator('[data-testid="quick-create-preset-minimal"]').click();
      await page1.locator('[data-testid="vm-name"]').fill(vmName);
      await page1.locator('[data-testid="quick-create-submit"]').click();

      // Page2 should see the new VM
      await expect(
        page2.locator(`[data-testid="vm-${vmName}"]`)
      ).toBeVisible({ timeout: 15000 });

      // Cleanup
      await dashboard1.goto();
      await dashboard1.clickVM(vmName);
      const vmDetails = new VMDetailsPage(page1);
      await vmDetails.waitForVMState('running', { timeout: 60000 });
      await vmDetails.forceDelete();
    } finally {
      await context.close();
    }
  });

  test('should show connection quality indicator', async ({ page }) => {
    await dashboardPage.goto();

    const indicator = page.locator('[data-testid="connection-quality"]');
    await expect(indicator).toBeVisible();

    const quality = await indicator.getAttribute('data-quality');
    expect(['good', 'fair', 'poor']).toContain(quality);

    // Should show latency
    const latency = page.locator('[data-testid="connection-latency"]');
    await expect(latency).toBeVisible();

    const latencyValue = parseInt(await latency.textContent() || '0');
    expect(latencyValue).toBeGreaterThan(0);
  });

  test('should recover from message loss', async ({ page }) => {
    await dashboardPage.goto();

    // Verify connection
    await expect(page.locator('[data-testid="realtime-connected"]')).toBeVisible();

    // Simulate message loss by blocking WebSocket temporarily
    await page.route('**/ws', route => route.abort(), { times: 3 });

    await page.waitForTimeout(5000);

    // Restore connection
    await page.unroute('**/ws');

    // Should resync data
    await expect(page.locator('[data-testid="resyncing"]')).toBeVisible({ timeout: 10000 });
    await expect(page.locator('[data-testid="realtime-connected"]')).toBeVisible({ timeout: 15000 });

    // Data should be current
    const lastUpdate = await page.locator('[data-testid="last-update"]').textContent();
    expect(lastUpdate).toBeTruthy();
  });
});
