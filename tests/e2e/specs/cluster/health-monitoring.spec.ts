import { test, expect } from '@playwright/test';
import { ClusterPage } from '../../pages/ClusterPage';
import { NodeDetailsPage } from '../../pages/NodeDetailsPage';
import { authenticateUser } from '../../fixtures/auth';
import { testData } from '../../fixtures/testData';

test.describe('Cluster: Health Monitoring', () => {
  let clusterPage: ClusterPage;
  let nodeDetailsPage: NodeDetailsPage;

  test.beforeEach(async ({ page }) => {
    await authenticateUser(page, testData.users.admin);

    clusterPage = new ClusterPage(page);
    nodeDetailsPage = new NodeDetailsPage(page);

    await clusterPage.goto();
  });

  test('should display cluster health overview', {
    tag: '@smoke',
  }, async ({ page }) => {
    await clusterPage.waitForLoad();

    const health = await clusterPage.getClusterHealth();

    expect(health).toHaveProperty('status');
    expect(['healthy', 'warning', 'critical']).toContain(health.status);

    expect(health).toHaveProperty('totalNodes');
    expect(health).toHaveProperty('healthyNodes');
    expect(health).toHaveProperty('warnings');
    expect(health).toHaveProperty('errors');

    // Verify health indicator is visible
    await expect(page.locator('[data-testid="cluster-health-indicator"]')).toBeVisible();
  });

  test('should show active alerts', async ({ page }) => {
    await clusterPage.navigateToTab('alerts');

    const alerts = await clusterPage.getActiveAlerts();

    // Verify alert structure
    for (const alert of alerts) {
      expect(alert).toHaveProperty('severity');
      expect(alert).toHaveProperty('message');
      expect(alert).toHaveProperty('timestamp');
      expect(['info', 'warning', 'critical']).toContain(alert.severity);
    }

    // Verify alerts are displayed
    if (alerts.length > 0) {
      await expect(page.locator('[data-testid="alert"]').first()).toBeVisible();
    }
  });

  test('should configure health checks', async ({ page }) => {
    await clusterPage.navigateToTab('health');

    await page.locator('[data-testid="configure-health-checks"]').click();

    // Configure CPU threshold
    await page.locator('[data-testid="cpu-warning-threshold"]').fill('75');
    await page.locator('[data-testid="cpu-critical-threshold"]').fill('90');

    // Configure memory threshold
    await page.locator('[data-testid="memory-warning-threshold"]').fill('80');
    await page.locator('[data-testid="memory-critical-threshold"]').fill('95');

    // Configure disk threshold
    await page.locator('[data-testid="disk-warning-threshold"]').fill('75');
    await page.locator('[data-testid="disk-critical-threshold"]').fill('90');

    // Configure check interval
    await page.locator('[data-testid="check-interval"]').fill('60');

    await page.locator('[data-testid="save-health-config"]').click();

    await expect(page.locator('[data-testid="config-saved"]')).toBeVisible();
  });

  test('should configure alert notifications', async ({ page }) => {
    await clusterPage.navigateToTab('alerts');

    await page.locator('[data-testid="configure-notifications"]').click();

    // Enable email notifications
    await page.locator('[data-testid="enable-email-notifications"]').check();
    await page.locator('[data-testid="alert-email"]').fill('admin@example.com');

    // Configure severity levels
    await page.locator('[data-testid="notify-warning"]').check();
    await page.locator('[data-testid="notify-critical"]').check();

    // Enable webhook
    await page.locator('[data-testid="enable-webhook"]').check();
    await page.locator('[data-testid="webhook-url"]').fill('https://hooks.example.com/alerts');

    await page.locator('[data-testid="save-notifications"]').click();

    await expect(page.locator('[data-testid="notifications-saved"]')).toBeVisible();
  });

  test('should acknowledge alerts', async ({ page }) => {
    await clusterPage.navigateToTab('alerts');

    const alerts = await clusterPage.getActiveAlerts();

    if (alerts.length === 0) {
      test.skip();
    }

    const firstAlert = alerts[0];

    await page.locator(`[data-testid="alert-${firstAlert.id}"]`).click();
    await page.locator('[data-testid="acknowledge-alert"]').click();

    await page.locator('[data-testid="acknowledgment-note"]').fill('Investigating the issue');
    await page.locator('[data-testid="confirm-acknowledge"]').click();

    await expect(page.locator('[data-testid="alert-acknowledged"]')).toBeVisible();

    // Verify alert is marked as acknowledged
    const updatedAlert = await page.locator(`[data-testid="alert-${firstAlert.id}"]`);
    await expect(updatedAlert.locator('[data-testid="acknowledged-badge"]')).toBeVisible();
  });

  test('should resolve alerts', async ({ page }) => {
    await clusterPage.navigateToTab('alerts');

    const alerts = await clusterPage.getActiveAlerts();

    if (alerts.length === 0) {
      test.skip();
    }

    const firstAlert = alerts[0];

    await page.locator(`[data-testid="alert-${firstAlert.id}"]`).click();
    await page.locator('[data-testid="resolve-alert"]').click();

    await page.locator('[data-testid="resolution-note"]').fill('Issue has been resolved');
    await page.locator('[data-testid="confirm-resolve"]').click();

    await expect(page.locator('[data-testid="alert-resolved"]')).toBeVisible();

    // Alert should be removed from active list
    await expect(page.locator(`[data-testid="alert-${firstAlert.id}"]`)).not.toBeVisible();
  });

  test('should view alert history', async ({ page }) => {
    await clusterPage.navigateToTab('alerts');

    await page.locator('[data-testid="view-history"]').click();

    await expect(page.locator('[data-testid="alert-history"]')).toBeVisible();

    const historyItems = await page.locator('[data-testid="history-alert"]').all();
    expect(historyItems.length).toBeGreaterThan(0);

    // Verify history includes resolved alerts
    for (const item of historyItems) {
      await expect(item.locator('[data-testid="alert-message"]')).toBeVisible();
      await expect(item.locator('[data-testid="alert-timestamp"]')).toBeVisible();
      await expect(item.locator('[data-testid="alert-resolution"]')).toBeVisible();
    }
  });

  test('should monitor node health status', async ({ page }) => {
    const nodes = await clusterPage.getNodeList();
    const targetNode = nodes[0];

    await clusterPage.clickNode(targetNode.id);

    await nodeDetailsPage.navigateToTab('health');

    // Verify system health metrics
    await expect(page.locator('[data-testid="cpu-health"]')).toBeVisible();
    await expect(page.locator('[data-testid="memory-health"]')).toBeVisible();
    await expect(page.locator('[data-testid="disk-health"]')).toBeVisible();
    await expect(page.locator('[data-testid="network-health"]')).toBeVisible();

    // Verify service health checks
    const services = await page.locator('[data-testid="service-health"]').all();
    expect(services.length).toBeGreaterThan(0);

    for (const service of services) {
      const status = await service.getAttribute('data-status');
      expect(['healthy', 'degraded', 'down']).toContain(status);
    }
  });

  test('should export health report', async ({ page }) => {
    await clusterPage.navigateToTab('health');

    await page.locator('[data-testid="export-health-report"]').click();

    // Select report options
    await page.locator('[data-testid="include-nodes"]').check();
    await page.locator('[data-testid="include-alerts"]').check();
    await page.locator('[data-testid="include-metrics"]').check();

    await page.locator('[data-testid="report-format"]').selectOption('pdf');

    const downloadPromise = page.waitForEvent('download');
    await page.locator('[data-testid="generate-report"]').click();

    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/health-report.*\.pdf/);
  });

  test('should create custom health dashboard', async ({ page }) => {
    await clusterPage.navigateToTab('health');

    await page.locator('[data-testid="customize-dashboard"]').click();

    // Add widgets
    await page.locator('[data-testid="add-widget"]').click();
    await page.locator('[data-testid="widget-cpu-usage"]').click();

    await page.locator('[data-testid="add-widget"]').click();
    await page.locator('[data-testid="widget-memory-usage"]').click();

    await page.locator('[data-testid="add-widget"]').click();
    await page.locator('[data-testid="widget-active-alerts"]').click();

    await page.locator('[data-testid="save-dashboard"]').click();

    await expect(page.locator('[data-testid="dashboard-saved"]')).toBeVisible();

    // Verify widgets are displayed
    await expect(page.locator('[data-testid="widget-cpu-usage"]')).toBeVisible();
    await expect(page.locator('[data-testid="widget-memory-usage"]')).toBeVisible();
    await expect(page.locator('[data-testid="widget-active-alerts"]')).toBeVisible();
  });
});
