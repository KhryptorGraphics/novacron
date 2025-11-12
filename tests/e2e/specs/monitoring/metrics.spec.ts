import { test, expect } from '@playwright/test';
import { DashboardPage } from '../../pages/DashboardPage';
import { VMDetailsPage } from '../../pages/VMDetailsPage';
import { MetricsPage } from '../../pages/MetricsPage';
import { authenticateUser } from '../../fixtures/auth';
import { testData } from '../../fixtures/testData';

test.describe('Monitoring: Metrics', () => {
  let dashboardPage: DashboardPage;
  let vmDetailsPage: VMDetailsPage;
  let metricsPage: MetricsPage;

  test.beforeEach(async ({ page }) => {
    await authenticateUser(page, testData.users.admin);

    dashboardPage = new DashboardPage(page);
    vmDetailsPage = new VMDetailsPage(page);
    metricsPage = new MetricsPage(page);
  });

  test('should display VM metrics', {
    tag: '@smoke',
  }, async ({ page }) => {
    // Create test VM
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `metrics-test-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-web"]').click();
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    await vmDetailsPage.navigateToTab('metrics');

    // Verify key metrics are visible
    await expect(page.locator('[data-testid="cpu-usage-metric"]')).toBeVisible();
    await expect(page.locator('[data-testid="memory-usage-metric"]')).toBeVisible();
    await expect(page.locator('[data-testid="disk-io-metric"]')).toBeVisible();
    await expect(page.locator('[data-testid="network-io-metric"]')).toBeVisible();

    // Verify metrics have values
    const cpuUsage = await page.locator('[data-testid="cpu-usage-value"]').textContent();
    expect(parseFloat(cpuUsage || '0')).toBeGreaterThanOrEqual(0);

    // Cleanup
    await vmDetailsPage.forceDelete();
  });

  test('should display historical metrics', async ({ page }) => {
    await dashboardPage.goto();
    const vms = await dashboardPage.getVMList();

    if (vms.length === 0) {
      test.skip();
    }

    await dashboardPage.clickVM(vms[0].name);
    await vmDetailsPage.navigateToTab('metrics');

    // Select time range
    await page.locator('[data-testid="time-range"]').selectOption('24h');

    // Verify charts display historical data
    await expect(page.locator('[data-testid="cpu-history-chart"]')).toBeVisible();
    await expect(page.locator('[data-testid="memory-history-chart"]')).toBeVisible();

    // Verify data points
    const dataPoints = await page.locator('[data-testid="chart-data-point"]').all();
    expect(dataPoints.length).toBeGreaterThan(0);
  });

  test('should compare metrics across time ranges', async ({ page }) => {
    await dashboardPage.goto();
    const vms = await dashboardPage.getVMList();

    if (vms.length === 0) {
      test.skip();
    }

    await dashboardPage.clickVM(vms[0].name);
    await vmDetailsPage.navigateToTab('metrics');

    await page.locator('[data-testid="enable-comparison"]').check();

    await page.locator('[data-testid="compare-range-1"]').selectOption('1h');
    await page.locator('[data-testid="compare-range-2"]').selectOption('24h');

    await expect(page.locator('[data-testid="comparison-chart"]')).toBeVisible();

    // Verify legend shows both time ranges
    await expect(page.locator('[data-testid="legend-1h"]')).toBeVisible();
    await expect(page.locator('[data-testid="legend-24h"]')).toBeVisible();
  });

  test('should aggregate metrics by node', async ({ page }) => {
    await metricsPage.goto();

    await page.locator('[data-testid="group-by"]').selectOption('node');

    const nodeMetrics = await page.locator('[data-testid="node-metric-group"]').all();
    expect(nodeMetrics.length).toBeGreaterThan(0);

    for (const metric of nodeMetrics) {
      await expect(metric.locator('[data-testid="node-name"]')).toBeVisible();
      await expect(metric.locator('[data-testid="total-cpu"]')).toBeVisible();
      await expect(metric.locator('[data-testid="total-memory"]')).toBeVisible();
      await expect(metric.locator('[data-testid="vm-count"]')).toBeVisible();
    }
  });

  test('should export metrics data', async ({ page }) => {
    await dashboardPage.goto();
    const vms = await dashboardPage.getVMList();

    if (vms.length === 0) {
      test.skip();
    }

    await dashboardPage.clickVM(vms[0].name);
    await vmDetailsPage.navigateToTab('metrics');

    await page.locator('[data-testid="export-metrics"]').click();

    // Configure export
    await page.locator('[data-testid="export-metric-cpu"]').check();
    await page.locator('[data-testid="export-metric-memory"]').check();
    await page.locator('[data-testid="export-time-range"]').selectOption('1h');
    await page.locator('[data-testid="export-format"]').selectOption('csv');

    const downloadPromise = page.waitForEvent('download');
    await page.locator('[data-testid="confirm-export"]').click();

    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/metrics.*\.csv/);
  });

  test('should set metric thresholds', async ({ page }) => {
    await dashboardPage.goto();
    const vms = await dashboardPage.getVMList();

    if (vms.length === 0) {
      test.skip();
    }

    await dashboardPage.clickVM(vms[0].name);
    await vmDetailsPage.navigateToTab('metrics');

    await page.locator('[data-testid="configure-thresholds"]').click();

    // Set CPU threshold
    await page.locator('[data-testid="cpu-warning-threshold"]').fill('70');
    await page.locator('[data-testid="cpu-critical-threshold"]').fill('90');

    // Set memory threshold
    await page.locator('[data-testid="memory-warning-threshold"]').fill('75');
    await page.locator('[data-testid="memory-critical-threshold"]').fill('90');

    await page.locator('[data-testid="save-thresholds"]').click();

    await expect(page.locator('[data-testid="thresholds-saved"]')).toBeVisible();

    // Verify threshold lines appear on charts
    await expect(page.locator('[data-testid="cpu-warning-line"]')).toBeVisible();
    await expect(page.locator('[data-testid="cpu-critical-line"]')).toBeVisible();
  });

  test('should display custom metrics', async ({ page }) => {
    await metricsPage.goto();

    await page.locator('[data-testid="add-custom-metric"]').click();

    await page.locator('[data-testid="metric-name"]').fill('Custom Disk IOPS');
    await page.locator('[data-testid="metric-query"]').fill('rate(disk_io_operations[5m])');
    await page.locator('[data-testid="metric-type"]').selectOption('gauge');

    await page.locator('[data-testid="add-metric"]').click();

    await expect(page.locator('[data-testid="custom-metric-Custom-Disk-IOPS"]')).toBeVisible();
  });

  test('should zoom and pan time series charts', async ({ page }) => {
    await dashboardPage.goto();
    const vms = await dashboardPage.getVMList();

    if (vms.length === 0) {
      test.skip();
    }

    await dashboardPage.clickVM(vms[0].name);
    await vmDetailsPage.navigateToTab('metrics');

    const chart = page.locator('[data-testid="cpu-history-chart"]');

    // Zoom in
    await chart.click({ position: { x: 100, y: 100 } });
    await page.mouse.wheel(0, -100);

    await expect(page.locator('[data-testid="zoom-level"]')).not.toHaveText('100%');

    // Pan
    await chart.click({ position: { x: 200, y: 100 } });
    await page.mouse.down();
    await page.mouse.move(100, 100);
    await page.mouse.up();

    // Reset zoom
    await page.locator('[data-testid="reset-zoom"]').click();
    await expect(page.locator('[data-testid="zoom-level"]')).toHaveText('100%');
  });

  test('should show metric percentiles', async ({ page }) => {
    await metricsPage.goto();

    await page.locator('[data-testid="view-mode"]').selectOption('percentiles');

    await expect(page.locator('[data-testid="p50-metric"]')).toBeVisible();
    await expect(page.locator('[data-testid="p95-metric"]')).toBeVisible();
    await expect(page.locator('[data-testid="p99-metric"]')).toBeVisible();

    // Verify values are in ascending order
    const p50 = parseFloat(await page.locator('[data-testid="p50-value"]').textContent() || '0');
    const p95 = parseFloat(await page.locator('[data-testid="p95-value"]').textContent() || '0');
    const p99 = parseFloat(await page.locator('[data-testid="p99-value"]').textContent() || '0');

    expect(p95).toBeGreaterThanOrEqual(p50);
    expect(p99).toBeGreaterThanOrEqual(p95);
  });

  test('should correlate metrics', async ({ page }) => {
    await dashboardPage.goto();
    const vms = await dashboardPage.getVMList();

    if (vms.length === 0) {
      test.skip();
    }

    await dashboardPage.clickVM(vms[0].name);
    await vmDetailsPage.navigateToTab('metrics');

    await page.locator('[data-testid="correlate-metrics"]').click();

    await page.locator('[data-testid="metric-x"]').selectOption('cpu');
    await page.locator('[data-testid="metric-y"]').selectOption('memory');

    await expect(page.locator('[data-testid="correlation-chart"]')).toBeVisible();

    // Should show correlation coefficient
    const correlation = page.locator('[data-testid="correlation-coefficient"]');
    await expect(correlation).toBeVisible();

    const coefficientValue = parseFloat(await correlation.textContent() || '0');
    expect(coefficientValue).toBeGreaterThanOrEqual(-1);
    expect(coefficientValue).toBeLessThanOrEqual(1);
  });
});
