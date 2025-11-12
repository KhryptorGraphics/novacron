import { test, expect } from '@playwright/test';
import { DashboardPage } from '../../pages/DashboardPage';
import { VMDetailsPage } from '../../pages/VMDetailsPage';
import { authenticateUser } from '../../fixtures/auth';
import { testData } from '../../fixtures/testData';

test.describe('VM Management: Snapshots', () => {
  let dashboardPage: DashboardPage;
  let vmDetailsPage: VMDetailsPage;
  let testVMName: string;

  test.beforeAll(async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();

    await authenticateUser(page, testData.users.admin);

    const dashboard = new DashboardPage(page);
    await dashboard.goto();
    await dashboard.clickQuickCreate();

    testVMName = 'snapshot-test-vm';
    await page.locator('[data-testid="quick-create-preset-minimal"]').click();
    await page.locator('[data-testid="vm-name"]').fill(testVMName);
    await page.locator('[data-testid="quick-create-submit"]').click();

    const vmDetails = new VMDetailsPage(page);
    await vmDetails.waitForVMState('running', { timeout: 60000 });

    await context.close();
  });

  test.beforeEach(async ({ page }) => {
    await authenticateUser(page, testData.users.admin);

    dashboardPage = new DashboardPage(page);
    vmDetailsPage = new VMDetailsPage(page);

    await dashboardPage.goto();
    await dashboardPage.clickVM(testVMName);
    await vmDetailsPage.navigateToTab('snapshots');
  });

  test.afterAll(async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();

    await authenticateUser(page, testData.users.admin);

    const dashboard = new DashboardPage(page);
    await dashboard.goto();

    const vm = await dashboard.findVM(testVMName);
    if (vm) {
      await dashboard.clickVM(testVMName);
      const vmDetails = new VMDetailsPage(page);
      await vmDetails.forceDelete();
    }

    await context.close();
  });

  test('should create snapshot of running VM', {
    tag: '@smoke',
  }, async ({ page }) => {
    await vmDetailsPage.clickCreateSnapshot();

    await page.locator('[data-testid="snapshot-name"]').fill('running-state-snapshot');
    await page.locator('[data-testid="snapshot-description"]').fill('Snapshot of VM in running state');
    await page.locator('[data-testid="snapshot-submit"]').click();

    await expect(page.locator('[data-testid="snapshot-creating"]')).toBeVisible();
    await expect(page.locator('[data-testid="snapshot-success"]')).toBeVisible({ timeout: 30000 });

    const snapshots = await vmDetailsPage.getSnapshotList();
    expect(snapshots).toContainEqual(
      expect.objectContaining({
        name: 'running-state-snapshot',
        state: 'running'
      })
    );
  });

  test('should create snapshot of stopped VM', async ({ page }) => {
    await vmDetailsPage.navigateToTab('overview');
    await vmDetailsPage.clickAction('stop');
    await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });

    await vmDetailsPage.navigateToTab('snapshots');
    await vmDetailsPage.clickCreateSnapshot();

    await page.locator('[data-testid="snapshot-name"]').fill('stopped-state-snapshot');
    await page.locator('[data-testid="snapshot-description"]').fill('Snapshot of stopped VM');
    await page.locator('[data-testid="snapshot-submit"]').click();

    await expect(page.locator('[data-testid="snapshot-success"]')).toBeVisible({ timeout: 30000 });

    const snapshots = await vmDetailsPage.getSnapshotList();
    expect(snapshots).toContainEqual(
      expect.objectContaining({
        name: 'stopped-state-snapshot',
        state: 'stopped'
      })
    );

    // Restart VM for other tests
    await vmDetailsPage.navigateToTab('overview');
    await vmDetailsPage.clickAction('start');
    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
  });

  test('should restore VM from snapshot', async ({ page }) => {
    // Create snapshot
    await vmDetailsPage.clickCreateSnapshot();
    await page.locator('[data-testid="snapshot-name"]').fill('restore-test-snapshot');
    await page.locator('[data-testid="snapshot-submit"]').click();
    await expect(page.locator('[data-testid="snapshot-success"]')).toBeVisible({ timeout: 30000 });

    // Make changes to VM
    await vmDetailsPage.navigateToTab('overview');
    await vmDetailsPage.clickAction('stop');
    await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });

    await vmDetailsPage.navigateToTab('configuration');
    await page.locator('[data-testid="edit-memory"]').click();
    await page.locator('[data-testid="memory-size"]').fill('4096');
    await page.locator('[data-testid="save-memory"]').click();

    // Restore from snapshot
    await vmDetailsPage.navigateToTab('snapshots');
    await page.locator('[data-testid="snapshot-restore-test-snapshot"]').click();
    await page.locator('[data-testid="snapshot-restore"]').click();
    await page.locator('[data-testid="confirm-restore"]').click();

    await expect(page.locator('[data-testid="restoring"]')).toBeVisible();
    await expect(page.locator('[data-testid="restore-complete"]')).toBeVisible({ timeout: 60000 });

    // Verify configuration was restored
    await vmDetailsPage.navigateToTab('configuration');
    const memory = await vmDetailsPage.getMemory();
    expect(memory).toBeLessThan(4096); // Should be back to original
  });

  test('should delete snapshot', async ({ page }) => {
    // Create snapshot to delete
    await vmDetailsPage.clickCreateSnapshot();
    await page.locator('[data-testid="snapshot-name"]').fill('delete-me-snapshot');
    await page.locator('[data-testid="snapshot-submit"]').click();
    await expect(page.locator('[data-testid="snapshot-success"]')).toBeVisible({ timeout: 30000 });

    // Delete snapshot
    await page.locator('[data-testid="snapshot-delete-me-snapshot"]').click();
    await page.locator('[data-testid="snapshot-delete"]').click();
    await page.locator('[data-testid="confirm-delete"]').click();

    await expect(page.locator('[data-testid="snapshot-deleted"]')).toBeVisible();

    const snapshots = await vmDetailsPage.getSnapshotList();
    expect(snapshots).not.toContainEqual(
      expect.objectContaining({
        name: 'delete-me-snapshot'
      })
    );
  });

  test('should create snapshot tree', async ({ page }) => {
    // Create parent snapshot
    await vmDetailsPage.clickCreateSnapshot();
    await page.locator('[data-testid="snapshot-name"]').fill('parent-snapshot');
    await page.locator('[data-testid="snapshot-submit"]').click();
    await expect(page.locator('[data-testid="snapshot-success"]')).toBeVisible({ timeout: 30000 });

    // Restore to parent
    await page.locator('[data-testid="snapshot-parent-snapshot"]').click();
    await page.locator('[data-testid="snapshot-restore"]').click();
    await page.locator('[data-testid="confirm-restore"]').click();
    await expect(page.locator('[data-testid="restore-complete"]')).toBeVisible({ timeout: 60000 });

    // Create child snapshot
    await vmDetailsPage.clickCreateSnapshot();
    await page.locator('[data-testid="snapshot-name"]').fill('child-snapshot-1');
    await page.locator('[data-testid="snapshot-submit"]').click();
    await expect(page.locator('[data-testid="snapshot-success"]')).toBeVisible({ timeout: 30000 });

    // Restore to parent again
    await page.locator('[data-testid="snapshot-parent-snapshot"]').click();
    await page.locator('[data-testid="snapshot-restore"]').click();
    await page.locator('[data-testid="confirm-restore"]').click();
    await expect(page.locator('[data-testid="restore-complete"]')).toBeVisible({ timeout: 60000 });

    // Create another child
    await vmDetailsPage.clickCreateSnapshot();
    await page.locator('[data-testid="snapshot-name"]').fill('child-snapshot-2');
    await page.locator('[data-testid="snapshot-submit"]').click();
    await expect(page.locator('[data-testid="snapshot-success"]')).toBeVisible({ timeout: 30000 });

    // Verify tree structure
    await page.locator('[data-testid="snapshot-tree-view"]').click();

    const treeItems = await page.locator('[data-testid^="tree-node-"]').all();
    expect(treeItems.length).toBeGreaterThanOrEqual(3);
  });

  test('should handle snapshot with memory', async ({ page }) => {
    await vmDetailsPage.clickCreateSnapshot();

    await page.locator('[data-testid="snapshot-name"]').fill('memory-snapshot');
    await page.locator('[data-testid="include-memory"]').check();
    await page.locator('[data-testid="snapshot-submit"]').click();

    await expect(page.locator('[data-testid="snapshot-creating"]')).toBeVisible();
    await expect(page.locator('[data-testid="snapshot-success"]')).toBeVisible({ timeout: 60000 });

    const snapshots = await vmDetailsPage.getSnapshotList();
    const memorySnapshot = snapshots.find(s => s.name === 'memory-snapshot');

    expect(memorySnapshot).toBeDefined();
    expect(memorySnapshot?.hasMemory).toBe(true);
    expect(memorySnapshot?.size).toBeGreaterThan(0);
  });

  test('should validate snapshot name', async ({ page }) => {
    await vmDetailsPage.clickCreateSnapshot();

    const invalidNames = [
      '',
      'a',
      'name with spaces',
      'name_with_underscore',
      'name-that-is-way-too-long-and-exceeds-maximum-length-limit'
    ];

    for (const name of invalidNames) {
      await page.locator('[data-testid="snapshot-name"]').fill(name);
      await page.keyboard.press('Tab');

      await expect(page.locator('[data-testid="snapshot-name-error"]')).toBeVisible();
      await page.locator('[data-testid="snapshot-name"]').clear();
    }

    await page.locator('[data-testid="snapshot-name"]').fill('valid-snapshot-name');
    await expect(page.locator('[data-testid="snapshot-name-error"]')).not.toBeVisible();
  });

  test('should prevent duplicate snapshot names', async ({ page }) => {
    const snapshotName = 'duplicate-test';

    // Create first snapshot
    await vmDetailsPage.clickCreateSnapshot();
    await page.locator('[data-testid="snapshot-name"]').fill(snapshotName);
    await page.locator('[data-testid="snapshot-submit"]').click();
    await expect(page.locator('[data-testid="snapshot-success"]')).toBeVisible({ timeout: 30000 });

    // Try to create second with same name
    await vmDetailsPage.clickCreateSnapshot();
    await page.locator('[data-testid="snapshot-name"]').fill(snapshotName);
    await page.keyboard.press('Tab');

    await expect(page.locator('[data-testid="snapshot-name-error"]')).toContainText('already exists');
  });

  test('should export snapshot', async ({ page }) => {
    // Create snapshot to export
    await vmDetailsPage.clickCreateSnapshot();
    await page.locator('[data-testid="snapshot-name"]').fill('export-test-snapshot');
    await page.locator('[data-testid="snapshot-submit"]').click();
    await expect(page.locator('[data-testid="snapshot-success"]')).toBeVisible({ timeout: 30000 });

    // Export snapshot
    await page.locator('[data-testid="snapshot-export-test-snapshot"]').click();
    await page.locator('[data-testid="snapshot-export"]').click();

    const downloadPromise = page.waitForEvent('download');
    await page.locator('[data-testid="confirm-export"]').click();

    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/snapshot.*\.qcow2/);
  });

  test('should show snapshot details', async ({ page }) => {
    // Create snapshot
    await vmDetailsPage.clickCreateSnapshot();
    await page.locator('[data-testid="snapshot-name"]').fill('details-test-snapshot');
    await page.locator('[data-testid="snapshot-description"]').fill('Test snapshot for details view');
    await page.locator('[data-testid="snapshot-submit"]').click();
    await expect(page.locator('[data-testid="snapshot-success"]')).toBeVisible({ timeout: 30000 });

    // View details
    await page.locator('[data-testid="snapshot-details-test-snapshot"]').click();
    await page.locator('[data-testid="snapshot-details"]').click();

    const detailsPanel = page.locator('[data-testid="snapshot-details-panel"]');
    await expect(detailsPanel).toBeVisible();

    await expect(detailsPanel.locator('[data-testid="snapshot-name"]')).toContainText('details-test-snapshot');
    await expect(detailsPanel.locator('[data-testid="snapshot-description"]')).toContainText('Test snapshot for details view');
    await expect(detailsPanel.locator('[data-testid="snapshot-size"]')).toBeVisible();
    await expect(detailsPanel.locator('[data-testid="snapshot-created"]')).toBeVisible();
    await expect(detailsPanel.locator('[data-testid="snapshot-state"]')).toBeVisible();
  });

  test('should handle automatic snapshots', async ({ page }) => {
    await vmDetailsPage.navigateToTab('configuration');

    await page.locator('[data-testid="edit-snapshot-policy"]').click();

    await page.locator('[data-testid="enable-auto-snapshots"]').check();
    await page.locator('[data-testid="snapshot-schedule"]').selectOption('daily');
    await page.locator('[data-testid="snapshot-retention"]').fill('7');
    await page.locator('[data-testid="save-snapshot-policy"]').click();

    await expect(page.locator('[data-testid="policy-saved"]')).toBeVisible();

    // Verify policy is active
    await vmDetailsPage.navigateToTab('snapshots');

    const policyIndicator = page.locator('[data-testid="auto-snapshot-policy"]');
    await expect(policyIndicator).toBeVisible();
    await expect(policyIndicator).toContainText('Daily');
  });

  test('should handle snapshot failure', async ({ page }) => {
    // Simulate snapshot failure
    await page.route('**/api/snapshots', route => {
      route.fulfill({
        status: 500,
        body: JSON.stringify({
          error: 'Insufficient disk space'
        })
      });
    });

    await vmDetailsPage.clickCreateSnapshot();
    await page.locator('[data-testid="snapshot-name"]').fill('failed-snapshot');
    await page.locator('[data-testid="snapshot-submit"]').click();

    await expect(page.locator('[data-testid="snapshot-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="snapshot-error"]')).toContainText('Insufficient disk space');
  });
});
