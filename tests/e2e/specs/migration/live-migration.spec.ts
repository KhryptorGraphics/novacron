import { test, expect } from '@playwright/test';
import { DashboardPage } from '../../pages/DashboardPage';
import { VMDetailsPage } from '../../pages/VMDetailsPage';
import { MigrationWizard } from '../../pages/MigrationWizard';
import { ClusterPage } from '../../pages/ClusterPage';
import { authenticateUser } from '../../fixtures/auth';
import { testData } from '../../fixtures/testData';

test.describe('Migration: Live Migration', () => {
  let dashboardPage: DashboardPage;
  let vmDetailsPage: VMDetailsPage;
  let migrationWizard: MigrationWizard;
  let clusterPage: ClusterPage;

  test.beforeEach(async ({ page }) => {
    await authenticateUser(page, testData.users.admin);

    dashboardPage = new DashboardPage(page);
    vmDetailsPage = new VMDetailsPage(page);
    migrationWizard = new MigrationWizard(page);
    clusterPage = new ClusterPage(page);
  });

  test('should perform zero-downtime live migration', {
    tag: '@smoke',
  }, async ({ page }) => {
    // Create VM
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `live-migrate-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-web"]').click();
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    const sourceNode = await vmDetailsPage.getCurrentNode();

    // Start live migration
    await vmDetailsPage.clickAction('migrate');
    await migrationWizard.selectMigrationType('live');

    const nodes = await migrationWizard.getAvailableNodes();
    const targetNode = nodes.find(n => n.name !== sourceNode);

    await migrationWizard.selectTargetNode(targetNode!.name);
    await migrationWizard.submit();

    // Monitor migration - VM should stay running
    const progressDialog = page.locator('[data-testid="migration-progress"]');
    await expect(progressDialog).toBeVisible();

    // Verify VM state remains running throughout
    let state = await vmDetailsPage.getVMStatus();
    expect(state).toBe('running');

    // Wait for migration phases
    await expect(page.locator('[data-testid="phase-preparing"]')).toHaveClass(/active/);

    state = await vmDetailsPage.getVMStatus();
    expect(state).toBe('running');

    await expect(page.locator('[data-testid="phase-transferring"]')).toHaveClass(/active/, { timeout: 30000 });

    state = await vmDetailsPage.getVMStatus();
    expect(state).toBe('running');

    await expect(page.locator('[data-testid="phase-completing"]')).toHaveClass(/active/, { timeout: 60000 });

    // Wait for completion
    await expect(page.locator('[data-testid="migration-complete"]')).toBeVisible({ timeout: 120000 });

    // Verify VM is on target node and still running
    state = await vmDetailsPage.getVMStatus();
    expect(state).toBe('running');

    const currentNode = await vmDetailsPage.getCurrentNode();
    expect(currentNode).toBe(targetNode!.name);

    // Verify uptime was not significantly affected
    const uptime = await vmDetailsPage.getUptime();
    expect(uptime).toBeGreaterThan(60); // Should be greater than migration time

    // Cleanup
    await vmDetailsPage.forceDelete();
  });

  test('should track memory dirty rate during migration', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `dirty-rate-test-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-database"]').click();
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    await vmDetailsPage.clickAction('migrate');
    await migrationWizard.selectMigrationType('live');

    const nodes = await migrationWizard.getAvailableNodes();
    await migrationWizard.selectTargetNode(nodes[0].name);

    // Enable detailed monitoring
    await migrationWizard.enableDetailedMonitoring();

    await migrationWizard.submit();

    // Check dirty rate metrics
    const metricsPanel = page.locator('[data-testid="migration-metrics"]');
    await expect(metricsPanel).toBeVisible();

    await expect(metricsPanel.locator('[data-testid="dirty-rate"]')).toBeVisible();
    await expect(metricsPanel.locator('[data-testid="transfer-rate"]')).toBeVisible();
    await expect(metricsPanel.locator('[data-testid="remaining-data"]')).toBeVisible();

    await expect(page.locator('[data-testid="migration-complete"]')).toBeVisible({ timeout: 180000 });

    // Cleanup
    await vmDetailsPage.forceDelete();
  });

  test('should handle high memory pressure during migration', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `high-memory-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-custom"]').click();
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="cpu-input"]').fill('4');
    await page.locator('[data-testid="memory-input"]').fill('16384');
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    await vmDetailsPage.clickAction('migrate');
    await migrationWizard.selectMigrationType('live');

    const nodes = await migrationWizard.getAvailableNodes();
    await migrationWizard.selectTargetNode(nodes[0].name);

    // Configure memory compression
    await migrationWizard.enableMemoryCompression();
    await migrationWizard.setCompressionLevel('high');

    await migrationWizard.submit();

    await expect(page.locator('[data-testid="migration-progress"]')).toBeVisible();

    // Should show compression stats
    const compressionRatio = page.locator('[data-testid="compression-ratio"]');
    await expect(compressionRatio).toBeVisible({ timeout: 30000 });

    await expect(page.locator('[data-testid="migration-complete"]')).toBeVisible({ timeout: 300000 });

    // Cleanup
    await vmDetailsPage.forceDelete();
  });

  test('should perform post-copy migration on failure', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `postcopy-test-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-web"]').click();
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    await vmDetailsPage.clickAction('migrate');
    await migrationWizard.selectMigrationType('live');

    const nodes = await migrationWizard.getAvailableNodes();
    await migrationWizard.selectTargetNode(nodes[0].name);

    // Enable post-copy fallback
    await migrationWizard.enablePostCopyFallback();

    await migrationWizard.submit();

    // Simulate high dirty rate that triggers post-copy
    await page.route('**/api/migrations/*/trigger-postcopy', route => {
      route.fulfill({ status: 200, body: JSON.stringify({ triggered: true }) });
    });

    const progressDialog = page.locator('[data-testid="migration-progress"]');
    await expect(progressDialog).toBeVisible();

    // Should switch to post-copy mode
    await expect(page.locator('[data-testid="postcopy-active"]')).toBeVisible({ timeout: 60000 });

    await expect(page.locator('[data-testid="migration-complete"]')).toBeVisible({ timeout: 120000 });

    // Cleanup
    await vmDetailsPage.forceDelete();
  });

  test('should cancel live migration', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `cancel-migrate-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-minimal"]').click();
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    const sourceNode = await vmDetailsPage.getCurrentNode();

    await vmDetailsPage.clickAction('migrate');
    await migrationWizard.selectMigrationType('live');

    const nodes = await migrationWizard.getAvailableNodes();
    await migrationWizard.selectTargetNode(nodes[0].name);
    await migrationWizard.submit();

    // Wait for migration to start
    await expect(page.locator('[data-testid="phase-transferring"]')).toHaveClass(/active/, { timeout: 30000 });

    // Cancel migration
    await page.locator('[data-testid="cancel-migration"]').click();
    await page.locator('[data-testid="confirm-cancel"]').click();

    await expect(page.locator('[data-testid="migration-cancelled"]')).toBeVisible();

    // Verify VM is still on source node
    const currentNode = await vmDetailsPage.getCurrentNode();
    expect(currentNode).toBe(sourceNode);

    // Verify VM is still running
    const state = await vmDetailsPage.getVMStatus();
    expect(state).toBe('running');

    // Cleanup
    await vmDetailsPage.forceDelete();
  });

  test('should perform parallel live migrations', async ({ page }) => {
    const vmNames: string[] = [];

    // Create multiple VMs
    for (let i = 0; i < 3; i++) {
      await dashboardPage.goto();
      await dashboardPage.clickQuickCreate();

      const vmName = `parallel-migrate-${i}-${Date.now()}`;
      vmNames.push(vmName);

      await page.locator('[data-testid="quick-create-preset-minimal"]').click();
      await page.locator('[data-testid="vm-name"]').fill(vmName);
      await page.locator('[data-testid="quick-create-submit"]').click();

      await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
    }

    // Select all VMs for batch migration
    await dashboardPage.goto();

    for (const vmName of vmNames) {
      await page.locator(`[data-testid="vm-checkbox-${vmName}"]`).check();
    }

    await page.locator('[data-testid="bulk-actions"]').click();
    await page.locator('[data-testid="bulk-migrate"]').click();

    await migrationWizard.selectMigrationType('live');

    const nodes = await migrationWizard.getAvailableNodes();
    await migrationWizard.selectTargetNode(nodes[0].name);

    await migrationWizard.enableParallelMigration(true);
    await migrationWizard.setMaxParallel(2);

    await migrationWizard.submit();

    // Monitor parallel migrations
    const batchProgress = page.locator('[data-testid="batch-migration-progress"]');
    await expect(batchProgress).toBeVisible();

    const activeCount = page.locator('[data-testid="active-migrations-count"]');
    await expect(activeCount).toHaveText('2'); // Max parallel

    await expect(page.locator('[data-testid="batch-complete"]')).toBeVisible({ timeout: 300000 });

    // Verify all migrations succeeded
    const results = await page.locator('[data-testid="migration-result"]').all();
    expect(results).toHaveLength(3);

    for (const result of results) {
      await expect(result).toContainText('completed');
    }

    // Cleanup
    for (const vmName of vmNames) {
      await dashboardPage.goto();
      await dashboardPage.clickVM(vmName);
      await vmDetailsPage.forceDelete();
    }
  });

  test('should measure downtime during live migration', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `downtime-measure-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-web"]').click();
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    await vmDetailsPage.clickAction('migrate');
    await migrationWizard.selectMigrationType('live');

    const nodes = await migrationWizard.getAvailableNodes();
    await migrationWizard.selectTargetNode(nodes[0].name);

    // Enable downtime measurement
    await migrationWizard.enableDowntimeMeasurement();

    await migrationWizard.submit();

    await expect(page.locator('[data-testid="migration-complete"]')).toBeVisible({ timeout: 120000 });

    // Check downtime metrics
    const downtime = await page.locator('[data-testid="measured-downtime"]').textContent();
    const downtimeMs = parseInt(downtime || '0');

    // Live migration downtime should be < 100ms
    expect(downtimeMs).toBeLessThan(100);

    // Cleanup
    await vmDetailsPage.forceDelete();
  });
});
