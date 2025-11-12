import { test, expect } from '@playwright/test';
import { DashboardPage } from '../../pages/DashboardPage';
import { VMDetailsPage } from '../../pages/VMDetailsPage';
import { MigrationWizard } from '../../pages/MigrationWizard';
import { authenticateUser } from '../../fixtures/auth';
import { testData } from '../../fixtures/testData';

test.describe('Migration: Cold Migration', () => {
  let dashboardPage: DashboardPage;
  let vmDetailsPage: VMDetailsPage;
  let migrationWizard: MigrationWizard;

  test.beforeEach(async ({ page }) => {
    await authenticateUser(page, testData.users.admin);

    dashboardPage = new DashboardPage(page);
    vmDetailsPage = new VMDetailsPage(page);
    migrationWizard = new MigrationWizard(page);
  });

  test('should perform cold migration with data verification', {
    tag: '@smoke',
  }, async ({ page }) => {
    // Create VM
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `cold-migrate-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-database"]').click();
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    const sourceNode = await vmDetailsPage.getCurrentNode();

    // Stop VM
    await vmDetailsPage.clickAction('stop');
    await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });

    // Migrate
    await vmDetailsPage.clickAction('migrate');
    await migrationWizard.selectMigrationType('cold');

    const nodes = await migrationWizard.getAvailableNodes();
    const targetNode = nodes.find(n => n.name !== sourceNode);

    await migrationWizard.selectTargetNode(targetNode!.name);

    // Enable data verification
    await migrationWizard.enableDataVerification();

    await migrationWizard.submit();

    // Monitor migration
    await expect(page.locator('[data-testid="migration-progress"]')).toBeVisible();

    // Cold migration phases
    await expect(page.locator('[data-testid="phase-copying-disk"]')).toHaveClass(/active/, { timeout: 10000 });
    await expect(page.locator('[data-testid="phase-verifying"]')).toHaveClass(/active/, { timeout: 120000 });

    await expect(page.locator('[data-testid="migration-complete"]')).toBeVisible({ timeout: 180000 });

    // Verify node change
    const currentNode = await vmDetailsPage.getCurrentNode();
    expect(currentNode).toBe(targetNode!.name);

    // Start VM on new node
    await vmDetailsPage.clickAction('start');
    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    // Cleanup
    await vmDetailsPage.forceDelete();
  });

  test('should optimize cold migration with compression', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `cold-compressed-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-web"]').click();
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
    await vmDetailsPage.clickAction('stop');
    await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });

    await vmDetailsPage.clickAction('migrate');
    await migrationWizard.selectMigrationType('cold');

    const nodes = await migrationWizard.getAvailableNodes();
    await migrationWizard.selectTargetNode(nodes[0].name);

    // Enable compression
    await migrationWizard.enableCompression();
    await migrationWizard.setCompressionLevel('maximum');

    await migrationWizard.submit();

    await expect(page.locator('[data-testid="migration-progress"]')).toBeVisible();

    // Check compression stats
    const compressionRatio = page.locator('[data-testid="compression-ratio"]');
    await expect(compressionRatio).toBeVisible({ timeout: 30000 });

    const ratio = await compressionRatio.textContent();
    const ratioValue = parseFloat(ratio || '1.0');
    expect(ratioValue).toBeGreaterThan(1.0); // Should achieve some compression

    await expect(page.locator('[data-testid="migration-complete"]')).toBeVisible({ timeout: 180000 });

    // Cleanup
    await vmDetailsPage.forceDelete();
  });

  test('should perform incremental cold migration', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `incremental-migrate-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-custom"]').click();
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="disk-input"]').fill('100');
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    // Create snapshot as base
    await vmDetailsPage.navigateToTab('snapshots');
    await vmDetailsPage.clickCreateSnapshot();
    await page.locator('[data-testid="snapshot-name"]').fill('migration-base');
    await page.locator('[data-testid="snapshot-submit"]').click();
    await expect(page.locator('[data-testid="snapshot-success"]')).toBeVisible({ timeout: 30000 });

    await vmDetailsPage.navigateToTab('overview');
    await vmDetailsPage.clickAction('stop');
    await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });

    await vmDetailsPage.clickAction('migrate');
    await migrationWizard.selectMigrationType('cold');

    const nodes = await migrationWizard.getAvailableNodes();
    await migrationWizard.selectTargetNode(nodes[0].name);

    // Enable incremental migration
    await migrationWizard.enableIncrementalMigration();
    await migrationWizard.selectBaseSnapshot('migration-base');

    await migrationWizard.submit();

    await expect(page.locator('[data-testid="migration-progress"]')).toBeVisible();

    // Should show incremental transfer
    await expect(page.locator('[data-testid="incremental-mode"]')).toBeVisible();

    await expect(page.locator('[data-testid="migration-complete"]')).toBeVisible({ timeout: 120000 });

    // Cleanup
    await vmDetailsPage.forceDelete();
  });

  test('should handle cold migration network interruption', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `network-fail-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-minimal"]').click();
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
    await vmDetailsPage.clickAction('stop');
    await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });

    await vmDetailsPage.clickAction('migrate');
    await migrationWizard.selectMigrationType('cold');

    const nodes = await migrationWizard.getAvailableNodes();
    await migrationWizard.selectTargetNode(nodes[0].name);

    // Enable auto-resume on failure
    await migrationWizard.enableAutoResume();

    await migrationWizard.submit();

    // Simulate network interruption after migration starts
    await page.waitForTimeout(5000);

    await page.route('**/api/migrations/**', route => route.abort(), { times: 3 });

    // Should show retry attempts
    await expect(page.locator('[data-testid="migration-retrying"]')).toBeVisible({ timeout: 10000 });

    // Restore network
    await page.unroute('**/api/migrations/**');

    // Migration should resume and complete
    await expect(page.locator('[data-testid="migration-resumed"]')).toBeVisible({ timeout: 30000 });
    await expect(page.locator('[data-testid="migration-complete"]')).toBeVisible({ timeout: 180000 });

    // Cleanup
    await vmDetailsPage.forceDelete();
  });

  test('should show accurate progress during cold migration', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `progress-test-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-web"]').click();
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
    await vmDetailsPage.clickAction('stop');
    await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });

    await vmDetailsPage.clickAction('migrate');
    await migrationWizard.selectMigrationType('cold');

    const nodes = await migrationWizard.getAvailableNodes();
    await migrationWizard.selectTargetNode(nodes[0].name);

    await migrationWizard.submit();

    const progressBar = page.locator('[data-testid="migration-progress-bar"]');
    await expect(progressBar).toBeVisible();

    // Progress should increase over time
    const initialProgress = await progressBar.getAttribute('aria-valuenow');
    await page.waitForTimeout(10000);
    const laterProgress = await progressBar.getAttribute('aria-valuenow');

    expect(parseInt(laterProgress || '0')).toBeGreaterThan(parseInt(initialProgress || '0'));

    // Should show estimated time remaining
    await expect(page.locator('[data-testid="time-remaining"]')).toBeVisible();

    // Should show transfer rate
    await expect(page.locator('[data-testid="transfer-rate"]')).toBeVisible();

    await expect(page.locator('[data-testid="migration-complete"]')).toBeVisible({ timeout: 180000 });

    // Cleanup
    await vmDetailsPage.forceDelete();
  });
});
