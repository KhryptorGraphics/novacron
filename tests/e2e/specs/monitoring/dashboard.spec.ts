import { test, expect } from '@playwright/test';
import { DashboardPage } from '../../pages/DashboardPage';
import { authenticateUser } from '../../fixtures/auth';
import { testData } from '../../fixtures/testData';

test.describe('Monitoring: Dashboard', () => {
  let dashboardPage: DashboardPage;

  test.beforeEach(async ({ page }) => {
    await authenticateUser(page, testData.users.admin);

    dashboardPage = new DashboardPage(page);
    await dashboardPage.goto();
  });

  test('should display dashboard overview', {
    tag: '@smoke',
  }, async ({ page }) => {
    await dashboardPage.waitForLoad();

    // Verify key metrics are visible
    await expect(page.locator('[data-testid="total-vms"]')).toBeVisible();
    await expect(page.locator('[data-testid="running-vms"]')).toBeVisible();
    await expect(page.locator('[data-testid="total-nodes"]')).toBeVisible();
    await expect(page.locator('[data-testid="cluster-health"]')).toBeVisible();

    // Verify resource usage charts
    await expect(page.locator('[data-testid="cpu-usage-chart"]')).toBeVisible();
    await expect(page.locator('[data-testid="memory-usage-chart"]')).toBeVisible();
    await expect(page.locator('[data-testid="storage-usage-chart"]')).toBeVisible();
  });

  test('should show recent activity', async ({ page }) => {
    await expect(page.locator('[data-testid="recent-activity"]')).toBeVisible();

    const activities = await page.locator('[data-testid="activity-item"]').all();
    expect(activities.length).toBeGreaterThan(0);

    for (const activity of activities) {
      await expect(activity.locator('[data-testid="activity-type"]')).toBeVisible();
      await expect(activity.locator('[data-testid="activity-timestamp"]')).toBeVisible();
    }
  });

  test('should display VM statistics', async ({ page }) => {
    await expect(page.locator('[data-testid="vm-stats"]')).toBeVisible();

    const stats = await dashboardPage.getVMStatistics();

    expect(stats).toHaveProperty('total');
    expect(stats).toHaveProperty('running');
    expect(stats).toHaveProperty('stopped');
    expect(stats).toHaveProperty('paused');

    expect(stats.total).toBeGreaterThanOrEqual(
      stats.running + stats.stopped + stats.paused
    );
  });

  test('should filter dashboard by time range', async ({ page }) => {
    await page.locator('[data-testid="time-range-selector"]').click();
    await page.locator('[data-testid="time-range-1h"]').click();

    await expect(page.locator('[data-testid="selected-range"]')).toContainText('1 hour');

    // Charts should update
    await expect(page.locator('[data-testid="chart-loading"]')).toBeVisible();
    await expect(page.locator('[data-testid="chart-loading"]')).not.toBeVisible({ timeout: 10000 });

    // Try different ranges
    await page.locator('[data-testid="time-range-selector"]').click();
    await page.locator('[data-testid="time-range-24h"]').click();

    await expect(page.locator('[data-testid="selected-range"]')).toContainText('24 hours');
  });

  test('should customize dashboard layout', async ({ page }) => {
    await page.locator('[data-testid="customize-dashboard"]').click();

    // Drag and drop widgets
    await page.locator('[data-testid="widget-cpu-usage"]').dragTo(
      page.locator('[data-testid="grid-position-0"]')
    );

    await page.locator('[data-testid="widget-memory-usage"]').dragTo(
      page.locator('[data-testid="grid-position-1"]')
    );

    await page.locator('[data-testid="save-layout"]').click();

    await expect(page.locator('[data-testid="layout-saved"]')).toBeVisible();

    // Reload and verify layout persisted
    await page.reload();

    const widget1 = page.locator('[data-testid="grid-position-0"] [data-testid="widget-cpu-usage"]');
    const widget2 = page.locator('[data-testid="grid-position-1"] [data-testid="widget-memory-usage"]');

    await expect(widget1).toBeVisible();
    await expect(widget2).toBeVisible();
  });

  test('should add custom widgets', async ({ page }) => {
    await page.locator('[data-testid="customize-dashboard"]').click();
    await page.locator('[data-testid="add-widget"]').click();

    await page.locator('[data-testid="widget-type-alert-summary"]').click();

    await page.locator('[data-testid="widget-title"]').fill('Critical Alerts');
    await page.locator('[data-testid="widget-severity-filter"]').selectOption('critical');
    await page.locator('[data-testid="add-widget-submit"]').click();

    await expect(page.locator('[data-testid="widget-Critical-Alerts"]')).toBeVisible();

    await page.locator('[data-testid="save-layout"]').click();
  });

  test('should remove widgets', async ({ page }) => {
    await page.locator('[data-testid="customize-dashboard"]').click();

    const widget = page.locator('[data-testid="widget-cpu-usage"]').first();
    await widget.hover();

    await widget.locator('[data-testid="remove-widget"]').click();
    await page.locator('[data-testid="confirm-remove"]').click();

    await expect(widget).not.toBeVisible();

    await page.locator('[data-testid="save-layout"]').click();
  });

  test('should export dashboard data', async ({ page }) => {
    await page.locator('[data-testid="export-dashboard"]').click();

    await page.locator('[data-testid="export-format"]').selectOption('csv');

    const downloadPromise = page.waitForEvent('download');
    await page.locator('[data-testid="confirm-export"]').click();

    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/dashboard-data.*\.csv/);
  });

  test('should show quick actions', async ({ page }) => {
    await expect(page.locator('[data-testid="quick-actions"]')).toBeVisible();

    const actions = await page.locator('[data-testid="quick-action"]').all();
    expect(actions.length).toBeGreaterThan(0);

    // Verify common actions are present
    await expect(page.locator('[data-testid="quick-action-create-vm"]')).toBeVisible();
    await expect(page.locator('[data-testid="quick-action-view-alerts"]')).toBeVisible();
  });

  test('should display system notifications', async ({ page }) => {
    await page.locator('[data-testid="notifications-icon"]').click();

    await expect(page.locator('[data-testid="notifications-panel"]')).toBeVisible();

    const notifications = await page.locator('[data-testid="notification"]').all();

    if (notifications.length > 0) {
      const firstNotification = notifications[0];
      await expect(firstNotification.locator('[data-testid="notification-title"]')).toBeVisible();
      await expect(firstNotification.locator('[data-testid="notification-timestamp"]')).toBeVisible();
    }
  });

  test('should refresh dashboard data', async ({ page }) => {
    const initialVMCount = await page.locator('[data-testid="total-vms"]').textContent();

    await page.locator('[data-testid="refresh-dashboard"]').click();

    await expect(page.locator('[data-testid="refreshing"]')).toBeVisible();
    await expect(page.locator('[data-testid="refreshing"]')).not.toBeVisible({ timeout: 5000 });

    // Timestamp should update
    const lastUpdated = page.locator('[data-testid="last-updated"]');
    await expect(lastUpdated).toBeVisible();
  });

  test('should auto-refresh dashboard', async ({ page }) => {
    await page.locator('[data-testid="auto-refresh"]').click();
    await page.locator('[data-testid="refresh-interval-30s"]').click();

    await expect(page.locator('[data-testid="auto-refresh-active"]')).toBeVisible();

    // Wait for refresh
    await page.waitForTimeout(31000);

    await expect(page.locator('[data-testid="last-updated"]')).toBeVisible();
  });
});
