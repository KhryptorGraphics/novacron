import { test, expect } from '@playwright/test';
import { DashboardPage } from '../../pages/DashboardPage';
import { VMDetailsPage } from '../../pages/VMDetailsPage';
import { authenticateUser } from '../../fixtures/auth';
import { testData } from '../../fixtures/testData';

test.describe('VM Management: Operations', () => {
  let dashboardPage: DashboardPage;
  let vmDetailsPage: VMDetailsPage;
  let testVMName: string;

  test.beforeAll(async ({ browser }) => {
    // Create test VM once for all operations tests
    const context = await browser.newContext();
    const page = await context.newPage();

    await authenticateUser(page, testData.users.admin);

    const dashboard = new DashboardPage(page);
    await dashboard.goto();
    await dashboard.clickQuickCreate();

    testVMName = 'operations-test-vm';
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
  });

  test.afterAll(async ({ browser }) => {
    // Cleanup test VM
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

  test('should start VM', { tag: '@smoke' }, async () => {
    // Ensure VM is stopped
    const currentState = await vmDetailsPage.getVMStatus();
    if (currentState !== 'stopped') {
      await vmDetailsPage.clickAction('stop');
      await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });
    }

    await vmDetailsPage.clickAction('start');

    await vmDetailsPage.waitForVMState('starting', { timeout: 5000 });
    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    const status = await vmDetailsPage.getVMStatus();
    expect(status).toBe('running');
  });

  test('should stop VM gracefully', async () => {
    // Ensure VM is running
    const currentState = await vmDetailsPage.getVMStatus();
    if (currentState !== 'running') {
      await vmDetailsPage.clickAction('start');
      await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
    }

    await vmDetailsPage.clickAction('stop');
    await vmDetailsPage.confirmAction();

    await vmDetailsPage.waitForVMState('stopping', { timeout: 5000 });
    await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });

    const status = await vmDetailsPage.getVMStatus();
    expect(status).toBe('stopped');
  });

  test('should force stop VM', async () => {
    // Ensure VM is running
    const currentState = await vmDetailsPage.getVMStatus();
    if (currentState !== 'running') {
      await vmDetailsPage.clickAction('start');
      await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
    }

    await vmDetailsPage.clickAction('force-stop');
    await vmDetailsPage.confirmAction();

    await vmDetailsPage.waitForVMState('stopped', { timeout: 15000 });

    const status = await vmDetailsPage.getVMStatus();
    expect(status).toBe('stopped');
  });

  test('should restart VM', async () => {
    // Ensure VM is running
    const currentState = await vmDetailsPage.getVMStatus();
    if (currentState !== 'running') {
      await vmDetailsPage.clickAction('start');
      await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
    }

    const uptimeBefore = await vmDetailsPage.getUptime();

    await vmDetailsPage.clickAction('restart');
    await vmDetailsPage.confirmAction();

    await vmDetailsPage.waitForVMState('restarting', { timeout: 5000 });
    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    // Wait for uptime to reset
    await vmDetailsPage.page.waitForTimeout(5000);

    const uptimeAfter = await vmDetailsPage.getUptime();
    expect(uptimeAfter).toBeLessThan(uptimeBefore);
  });

  test('should pause and resume VM', async () => {
    // Ensure VM is running
    const currentState = await vmDetailsPage.getVMStatus();
    if (currentState !== 'running') {
      await vmDetailsPage.clickAction('start');
      await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
    }

    // Pause VM
    await vmDetailsPage.clickAction('pause');

    await vmDetailsPage.waitForVMState('paused', { timeout: 10000 });

    let status = await vmDetailsPage.getVMStatus();
    expect(status).toBe('paused');

    // Resume VM
    await vmDetailsPage.clickAction('resume');

    await vmDetailsPage.waitForVMState('running', { timeout: 10000 });

    status = await vmDetailsPage.getVMStatus();
    expect(status).toBe('running');
  });

  test('should suspend and restore VM', async () => {
    // Ensure VM is running
    const currentState = await vmDetailsPage.getVMStatus();
    if (currentState !== 'running') {
      await vmDetailsPage.clickAction('start');
      await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
    }

    // Suspend VM
    await vmDetailsPage.clickAction('suspend');
    await vmDetailsPage.confirmAction();

    await vmDetailsPage.waitForVMState('suspending', { timeout: 5000 });
    await vmDetailsPage.waitForVMState('suspended', { timeout: 30000 });

    const status = await vmDetailsPage.getVMStatus();
    expect(status).toBe('suspended');

    // Restore VM
    await vmDetailsPage.clickAction('restore');

    await vmDetailsPage.waitForVMState('restoring', { timeout: 5000 });
    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    const restoredStatus = await vmDetailsPage.getVMStatus();
    expect(restoredStatus).toBe('running');
  });

  test('should prevent invalid state transitions', async () => {
    // Ensure VM is stopped
    const currentState = await vmDetailsPage.getVMStatus();
    if (currentState !== 'stopped') {
      await vmDetailsPage.clickAction('stop');
      await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });
    }

    // Try to pause stopped VM
    const pauseButton = vmDetailsPage.page.locator('[data-testid="action-pause"]');
    await expect(pauseButton).toBeDisabled();

    // Try to restart stopped VM
    const restartButton = vmDetailsPage.page.locator('[data-testid="action-restart"]');
    await expect(restartButton).toBeDisabled();
  });

  test('should show operation progress', async ({ page }) => {
    // Ensure VM is stopped
    const currentState = await vmDetailsPage.getVMStatus();
    if (currentState !== 'stopped') {
      await vmDetailsPage.clickAction('stop');
      await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });
    }

    await vmDetailsPage.clickAction('start');

    // Check progress indicator
    const progressIndicator = page.locator('[data-testid="vm-operation-progress"]');
    await expect(progressIndicator).toBeVisible();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    await expect(progressIndicator).not.toBeVisible();
  });

  test('should handle operation failures', async ({ page }) => {
    // Simulate operation failure
    await page.route('**/api/vms/*/start', route => {
      route.fulfill({
        status: 500,
        body: JSON.stringify({
          error: 'Insufficient resources'
        })
      });
    });

    // Ensure VM is stopped
    const currentState = await vmDetailsPage.getVMStatus();
    if (currentState !== 'stopped') {
      await vmDetailsPage.clickAction('stop');
      await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });
    }

    await vmDetailsPage.clickAction('start');

    const errorMessage = page.locator('[data-testid="operation-error"]');
    await expect(errorMessage).toBeVisible();
    await expect(errorMessage).toContainText('Insufficient resources');

    // VM should remain in stopped state
    const status = await vmDetailsPage.getVMStatus();
    expect(status).toBe('stopped');
  });

  test('should support bulk operations', async ({ page }) => {
    await dashboardPage.goto();

    // Select multiple VMs
    await page.locator(`[data-testid="vm-checkbox-${testVMName}"]`).check();
    await page.locator('[data-testid="vm-checkbox-operations-test-vm-2"]').check();

    await page.locator('[data-testid="bulk-actions"]').click();
    await page.locator('[data-testid="bulk-stop"]').click();
    await page.locator('[data-testid="confirm-bulk-operation"]').click();

    // Wait for bulk operation to complete
    await expect(page.locator('[data-testid="bulk-operation-progress"]')).toBeVisible();
    await expect(page.locator('[data-testid="bulk-operation-complete"]')).toBeVisible({ timeout: 60000 });

    // Verify both VMs are stopped
    const vm1 = await dashboardPage.findVM(testVMName);
    expect(vm1?.state).toBe('stopped');
  });

  test('should queue operations when busy', async ({ page }) => {
    // Ensure VM is running
    const currentState = await vmDetailsPage.getVMStatus();
    if (currentState !== 'running') {
      await vmDetailsPage.clickAction('start');
      await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
    }

    // Start restart operation
    await vmDetailsPage.clickAction('restart');
    await vmDetailsPage.confirmAction();

    // Try to perform another operation immediately
    await page.waitForTimeout(1000);

    const stopButton = page.locator('[data-testid="action-stop"]');
    await expect(stopButton).toBeDisabled();

    // Wait for restart to complete
    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    // Now operation should be available
    await expect(stopButton).toBeEnabled();
  });
});
