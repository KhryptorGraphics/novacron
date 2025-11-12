import { test, expect } from '@playwright/test';
import { DashboardPage } from '../../pages/DashboardPage';
import { VMCreationWizard } from '../../pages/VMCreationWizard';
import { VMDetailsPage } from '../../pages/VMDetailsPage';
import { authenticateUser } from '../../fixtures/auth';
import { testData } from '../../fixtures/testData';

test.describe('Critical Path: Complete VM Lifecycle', () => {
  let dashboardPage: DashboardPage;
  let vmCreationWizard: VMCreationWizard;
  let vmDetailsPage: VMDetailsPage;
  let vmId: string;

  test.beforeEach(async ({ page }) => {
    await authenticateUser(page, testData.users.admin);

    dashboardPage = new DashboardPage(page);
    vmCreationWizard = new VMCreationWizard(page);
    vmDetailsPage = new VMDetailsPage(page);

    await dashboardPage.goto();
  });

  test('should complete full VM lifecycle: create, start, stop, snapshot, restore, delete', {
    tag: '@smoke @critical',
  }, async ({ page }) => {
    // Step 1: Create VM
    await test.step('Create new VM', async () => {
      await dashboardPage.clickCreateVM();

      await vmCreationWizard.setVMName('lifecycle-test-vm');
      await vmCreationWizard.selectTemplate('debian-12');
      await vmCreationWizard.nextStep();

      await vmCreationWizard.setCPU(4);
      await vmCreationWizard.setMemory(8192);
      await vmCreationWizard.setDisk(100);
      await vmCreationWizard.nextStep();

      await vmCreationWizard.selectNetwork('default');
      await vmCreationWizard.nextStep();

      await vmCreationWizard.submit();

      const url = page.url();
      vmId = url.match(/\/vms\/([a-f0-9-]+)/)?.[1] || '';
      expect(vmId).toBeTruthy();
    });

    // Step 2: Wait for VM to start
    await test.step('Wait for VM to reach running state', async () => {
      await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

      const status = await vmDetailsPage.getVMStatus();
      expect(status).toBe('running');

      // Verify resource metrics are being collected
      await expect(vmDetailsPage.cpuUsageChart).toBeVisible();
      await expect(vmDetailsPage.memoryUsageChart).toBeVisible();
    });

    // Step 3: Perform VM operations
    await test.step('Perform stop operation', async () => {
      await vmDetailsPage.clickAction('stop');
      await vmDetailsPage.confirmAction();

      await vmDetailsPage.waitForVMState('stopping', { timeout: 5000 });
      await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });

      const status = await vmDetailsPage.getVMStatus();
      expect(status).toBe('stopped');
    });

    await test.step('Restart VM', async () => {
      await vmDetailsPage.clickAction('start');

      await vmDetailsPage.waitForVMState('starting', { timeout: 5000 });
      await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

      const status = await vmDetailsPage.getVMStatus();
      expect(status).toBe('running');
    });

    // Step 4: Create snapshot
    await test.step('Create VM snapshot', async () => {
      await vmDetailsPage.navigateToTab('snapshots');
      await vmDetailsPage.clickCreateSnapshot();

      await page.locator('[data-testid="snapshot-name"]').fill('pre-config-snapshot');
      await page.locator('[data-testid="snapshot-description"]').fill('Snapshot before configuration');
      await page.locator('[data-testid="snapshot-submit"]').click();

      await expect(page.locator('[data-testid="snapshot-success"]')).toBeVisible();

      // Verify snapshot appears in list
      const snapshots = await vmDetailsPage.getSnapshotList();
      expect(snapshots).toContainEqual(
        expect.objectContaining({
          name: 'pre-config-snapshot'
        })
      );
    });

    // Step 5: Make changes to VM
    await test.step('Modify VM configuration', async () => {
      await vmDetailsPage.navigateToTab('settings');

      // Change memory
      await vmDetailsPage.clickAction('edit-resources');
      await page.locator('[data-testid="memory-input"]').fill('16384');
      await page.locator('[data-testid="save-resources"]').click();

      await expect(page.locator('[data-testid="config-updated"]')).toBeVisible();
    });

    // Step 6: Restore from snapshot
    await test.step('Restore VM from snapshot', async () => {
      await vmDetailsPage.navigateToTab('snapshots');

      await page.locator('[data-testid="snapshot-pre-config-snapshot"]').click();
      await page.locator('[data-testid="snapshot-restore"]').click();
      await page.locator('[data-testid="confirm-restore"]').click();

      await vmDetailsPage.waitForVMState('restoring', { timeout: 5000 });
      await vmDetailsPage.waitForVMState('stopped', { timeout: 60000 });

      // Verify configuration is restored
      await vmDetailsPage.navigateToTab('overview');
      const memory = await vmDetailsPage.getMemory();
      expect(memory).toBe(8192);
    });

    // Step 7: Clone VM
    await test.step('Clone VM', async () => {
      await vmDetailsPage.clickAction('clone');

      await page.locator('[data-testid="clone-name"]').fill('lifecycle-test-vm-clone');
      await page.locator('[data-testid="clone-submit"]').click();

      await expect(page.locator('[data-testid="clone-success"]')).toBeVisible();

      // Verify clone appears in dashboard
      await dashboardPage.goto();
      const vmList = await dashboardPage.getVMList();
      expect(vmList).toContainEqual(
        expect.objectContaining({
          name: 'lifecycle-test-vm-clone'
        })
      );
    });

    // Step 8: Delete clone
    await test.step('Delete cloned VM', async () => {
      await dashboardPage.clickVM('lifecycle-test-vm-clone');

      await vmDetailsPage.clickAction('delete');
      await page.locator('[data-testid="delete-confirm-checkbox"]').check();
      await page.locator('[data-testid="delete-confirm"]').click();

      await expect(page).toHaveURL(/\/dashboard/);

      // Verify clone is removed
      const vmList = await dashboardPage.getVMList();
      expect(vmList).not.toContainEqual(
        expect.objectContaining({
          name: 'lifecycle-test-vm-clone'
        })
      );
    });

    // Step 9: Delete original VM
    await test.step('Delete original VM', async () => {
      await dashboardPage.clickVM('lifecycle-test-vm');

      await vmDetailsPage.clickAction('delete');
      await page.locator('[data-testid="delete-confirm-checkbox"]').check();
      await page.locator('[data-testid="delete-snapshots-checkbox"]').check();
      await page.locator('[data-testid="delete-confirm"]').click();

      await expect(page).toHaveURL(/\/dashboard/);

      // Verify VM is removed
      const vmList = await dashboardPage.getVMList();
      expect(vmList).not.toContainEqual(
        expect.objectContaining({
          name: 'lifecycle-test-vm'
        })
      );
    });
  });

  test('should handle VM lifecycle with power operations', async ({ page }) => {
    await test.step('Create and start VM', async () => {
      await dashboardPage.clickCreateVM();

      await vmCreationWizard.quickCreate({
        name: 'power-ops-vm',
        template: 'alpine-3.18',
        cpu: 2,
        memory: 2048,
        disk: 20
      });

      await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
    });

    await test.step('Test restart operation', async () => {
      await vmDetailsPage.clickAction('restart');
      await vmDetailsPage.confirmAction();

      await vmDetailsPage.waitForVMState('restarting', { timeout: 5000 });
      await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
    });

    await test.step('Test forced shutdown', async () => {
      await vmDetailsPage.clickAction('force-stop');
      await vmDetailsPage.confirmAction();

      await vmDetailsPage.waitForVMState('stopped', { timeout: 15000 });
    });

    await test.step('Test pause/resume', async () => {
      await vmDetailsPage.clickAction('start');
      await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

      await vmDetailsPage.clickAction('pause');
      await vmDetailsPage.waitForVMState('paused', { timeout: 10000 });

      await vmDetailsPage.clickAction('resume');
      await vmDetailsPage.waitForVMState('running', { timeout: 10000 });
    });

    await test.step('Cleanup', async () => {
      await vmDetailsPage.clickAction('delete');
      await page.locator('[data-testid="delete-confirm-checkbox"]').check();
      await page.locator('[data-testid="delete-confirm"]').click();
    });
  });

  test('should handle VM lifecycle errors gracefully', async ({ page }) => {
    await test.step('Create VM', async () => {
      await dashboardPage.clickCreateVM();

      await vmCreationWizard.quickCreate({
        name: 'error-test-vm',
        template: 'ubuntu-22.04',
        cpu: 2,
        memory: 4096,
        disk: 30
      });

      await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
    });

    await test.step('Attempt invalid operation', async () => {
      // Try to delete running VM without force
      await vmDetailsPage.clickAction('delete');

      const errorMsg = page.locator('[data-testid="delete-error"]');
      await expect(errorMsg).toBeVisible();
      await expect(errorMsg).toContainText('must be stopped');

      await page.locator('[data-testid="cancel-delete"]').click();
    });

    await test.step('Attempt rapid state changes', async () => {
      // Stop VM
      await vmDetailsPage.clickAction('stop');

      // Try to stop again before it finishes
      await page.waitForTimeout(1000);
      const stopButton = page.locator('[data-testid="action-stop"]');
      await expect(stopButton).toBeDisabled();
    });

    await test.step('Cleanup', async () => {
      await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });
      await vmDetailsPage.clickAction('delete');
      await page.locator('[data-testid="delete-confirm-checkbox"]').check();
      await page.locator('[data-testid="delete-confirm"]').click();
    });
  });
});
