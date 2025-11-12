import { test, expect } from '@playwright/test';
import { DashboardPage } from '../../pages/DashboardPage';
import { VMDetailsPage } from '../../pages/VMDetailsPage';
import { MigrationWizard } from '../../pages/MigrationWizard';
import { authenticateUser } from '../../fixtures/auth';
import { testData } from '../../fixtures/testData';

test.describe('Migration: Failure Recovery', () => {
  let dashboardPage: DashboardPage;
  let vmDetailsPage: VMDetailsPage;
  let migrationWizard: MigrationWizard;

  test.beforeEach(async ({ page }) => {
    await authenticateUser(page, testData.users.admin);

    dashboardPage = new DashboardPage(page);
    vmDetailsPage = new VMDetailsPage(page);
    migrationWizard = new MigrationWizard(page);
  });

  test('should recover from target node failure during migration', {
    tag: '@smoke',
  }, async ({ page }) => {
    // Create VM
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `failure-recovery-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-minimal"]').click();
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    const sourceNode = await vmDetailsPage.getCurrentNode();

    // Start migration
    await vmDetailsPage.clickAction('migrate');
    await migrationWizard.selectMigrationType('live');

    const nodes = await migrationWizard.getAvailableNodes();
    const targetNode = nodes.find(n => n.name !== sourceNode);

    await migrationWizard.selectTargetNode(targetNode!.name);
    await migrationWizard.submit();

    // Wait for migration to start
    await expect(page.locator('[data-testid="phase-transferring"]')).toHaveClass(/active/, { timeout: 30000 });

    // Simulate target node failure
    await page.route(`**/api/nodes/${targetNode!.id}/health`, route => {
      route.fulfill({
        status: 500,
        body: JSON.stringify({ error: 'Node unreachable' })
      });
    });

    // Should detect failure and offer recovery options
    await expect(page.locator('[data-testid="migration-failed"]')).toBeVisible({ timeout: 30000 });
    await expect(page.locator('[data-testid="recovery-options"]')).toBeVisible();

    // Select alternative target node
    await page.locator('[data-testid="recovery-option-retry-different-node"]').click();

    const alternativeNodes = await migrationWizard.getAvailableNodes();
    const alternativeNode = alternativeNodes.find(n => n.name !== sourceNode && n.name !== targetNode!.name);

    await migrationWizard.selectTargetNode(alternativeNode!.name);
    await page.locator('[data-testid="confirm-recovery"]').click();

    // Migration should restart to alternative node
    await expect(page.locator('[data-testid="migration-restarted"]')).toBeVisible();
    await expect(page.locator('[data-testid="migration-complete"]')).toBeVisible({ timeout: 120000 });

    // Verify VM is on alternative node
    const currentNode = await vmDetailsPage.getCurrentNode();
    expect(currentNode).toBe(alternativeNode!.name);

    // Cleanup
    await vmDetailsPage.forceDelete();
  });

  test('should rollback failed live migration', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `rollback-test-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-web"]').click();
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    const sourceNode = await vmDetailsPage.getCurrentNode();

    await vmDetailsPage.clickAction('migrate');
    await migrationWizard.selectMigrationType('live');

    const nodes = await migrationWizard.getAvailableNodes();
    await migrationWizard.selectTargetNode(nodes[0].name);

    // Enable automatic rollback on failure
    await migrationWizard.enableAutoRollback();

    await migrationWizard.submit();

    // Wait for migration to start
    await expect(page.locator('[data-testid="phase-transferring"]')).toHaveClass(/active/, { timeout: 30000 });

    // Simulate migration failure
    await page.route('**/api/migrations/*/complete', route => {
      route.fulfill({
        status: 500,
        body: JSON.stringify({ error: 'Migration failed' })
      });
    });

    // Should automatically rollback
    await expect(page.locator('[data-testid="migration-failed"]')).toBeVisible({ timeout: 60000 });
    await expect(page.locator('[data-testid="rolling-back"]')).toBeVisible();
    await expect(page.locator('[data-testid="rollback-complete"]')).toBeVisible({ timeout: 60000 });

    // VM should still be on source node and running
    const currentNode = await vmDetailsPage.getCurrentNode();
    expect(currentNode).toBe(sourceNode);

    const state = await vmDetailsPage.getVMStatus();
    expect(state).toBe('running');

    // Cleanup
    await vmDetailsPage.forceDelete();
  });

  test('should handle disk space exhaustion during migration', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `disk-space-test-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-database"]').click();
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

    // Simulate disk space exhaustion
    await page.waitForTimeout(10000);

    await page.route('**/api/migrations/**', route => {
      route.fulfill({
        status: 507,
        body: JSON.stringify({ error: 'Insufficient storage space' })
      });
    });

    // Should show error with recovery options
    await expect(page.locator('[data-testid="migration-error"]')).toBeVisible({ timeout: 30000 });
    await expect(page.locator('[data-testid="error-disk-space"]')).toBeVisible();

    // Should suggest cleanup or alternative storage
    await expect(page.locator('[data-testid="recovery-suggestions"]')).toBeVisible();

    // Cleanup
    await page.locator('[data-testid="cancel-migration"]').click();
    await vmDetailsPage.forceDelete();
  });

  test('should handle network partition during migration', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `network-partition-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-minimal"]').click();
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    await vmDetailsPage.clickAction('migrate');
    await migrationWizard.selectMigrationType('live');

    const nodes = await migrationWizard.getAvailableNodes();
    await migrationWizard.selectTargetNode(nodes[0].name);

    // Enable network failure handling
    await migrationWizard.enableNetworkFailureHandling();
    await migrationWizard.setNetworkTimeout(30);

    await migrationWizard.submit();

    // Wait for migration to start
    await expect(page.locator('[data-testid="phase-transferring"]')).toHaveClass(/active/, { timeout: 30000 });

    // Simulate network partition
    await page.route('**/api/migrations/**', route => route.abort(), { times: 5 });

    // Should detect network issue
    await expect(page.locator('[data-testid="network-issue-detected"]')).toBeVisible({ timeout: 40000 });

    // Restore network
    await page.unroute('**/api/migrations/**');

    // Should recover and continue
    await expect(page.locator('[data-testid="network-recovered"]')).toBeVisible({ timeout: 30000 });
    await expect(page.locator('[data-testid="migration-resuming"]')).toBeVisible();

    await expect(page.locator('[data-testid="migration-complete"]')).toBeVisible({ timeout: 120000 });

    // Cleanup
    await vmDetailsPage.forceDelete();
  });

  test('should provide detailed failure diagnostics', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `diagnostics-test-${Date.now()}`;
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

    await migrationWizard.submit();

    // Simulate failure
    await page.route('**/api/migrations/**', route => {
      route.fulfill({
        status: 500,
        body: JSON.stringify({
          error: 'Migration failed',
          details: {
            phase: 'transferring',
            errorCode: 'DISK_IO_ERROR',
            message: 'Failed to write to target disk',
            suggestions: [
              'Check target disk health',
              'Verify storage permissions',
              'Ensure sufficient IOPS'
            ]
          }
        })
      });
    });

    // Should show detailed diagnostics
    await expect(page.locator('[data-testid="migration-error"]')).toBeVisible({ timeout: 30000 });

    const diagnostics = page.locator('[data-testid="failure-diagnostics"]');
    await expect(diagnostics).toBeVisible();

    await expect(diagnostics.locator('[data-testid="error-phase"]')).toContainText('transferring');
    await expect(diagnostics.locator('[data-testid="error-code"]')).toContainText('DISK_IO_ERROR');

    const suggestions = await diagnostics.locator('[data-testid="suggestion"]').all();
    expect(suggestions.length).toBeGreaterThan(0);

    // Should allow downloading logs
    const downloadPromise = page.waitForEvent('download');
    await page.locator('[data-testid="download-migration-logs"]').click();

    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/migration-logs.*\.txt/);

    // Cleanup
    await page.locator('[data-testid="close-diagnostics"]').click();
    await vmDetailsPage.forceDelete();
  });

  test('should handle source node failure after migration completes', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `source-fail-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-minimal"]').click();
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    const sourceNode = await vmDetailsPage.getCurrentNode();

    await vmDetailsPage.clickAction('migrate');
    await migrationWizard.selectMigrationType('live');

    const nodes = await migrationWizard.getAvailableNodes();
    const targetNode = nodes.find(n => n.name !== sourceNode);

    await migrationWizard.selectTargetNode(targetNode!.name);
    await migrationWizard.submit();

    await expect(page.locator('[data-testid="migration-complete"]')).toBeVisible({ timeout: 120000 });

    // Simulate source node failure during cleanup
    await page.route(`**/api/nodes/${sourceNode}/cleanup`, route => {
      route.fulfill({ status: 500 });
    });

    // Should still complete successfully
    const currentNode = await vmDetailsPage.getCurrentNode();
    expect(currentNode).toBe(targetNode!.name);

    const state = await vmDetailsPage.getVMStatus();
    expect(state).toBe('running');

    // Should log cleanup failure but not affect VM
    await vmDetailsPage.navigateToTab('events');
    const events = await page.locator('[data-testid="event"]').all();
    const cleanupEvent = events.find(async (e) => {
      const text = await e.textContent();
      return text?.includes('cleanup');
    });

    expect(cleanupEvent).toBeDefined();

    // Cleanup
    await vmDetailsPage.forceDelete();
  });
});
