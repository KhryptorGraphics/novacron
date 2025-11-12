import { test, expect } from '@playwright/test';
import { DashboardPage } from '../../pages/DashboardPage';
import { VMDetailsPage } from '../../pages/VMDetailsPage';
import { MigrationWizard } from '../../pages/MigrationWizard';
import { ClusterPage } from '../../pages/ClusterPage';
import { authenticateUser } from '../../fixtures/auth';
import { testData } from '../../fixtures/testData';

test.describe('Critical Path: VM Migration Workflows', () => {
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

  test('should perform live migration successfully', {
    tag: '@smoke @critical',
  }, async ({ page }) => {
    let sourceNode: string;
    let targetNode: string;

    await test.step('Verify cluster has multiple nodes', async () => {
      await clusterPage.goto();

      const nodes = await clusterPage.getNodeList();
      expect(nodes.length).toBeGreaterThanOrEqual(2);

      const healthyNodes = nodes.filter(n => n.state === 'ready');
      expect(healthyNodes.length).toBeGreaterThanOrEqual(2);

      sourceNode = healthyNodes[0].name;
      targetNode = healthyNodes[1].name;
    });

    await test.step('Create VM on source node', async () => {
      await dashboardPage.goto();
      await dashboardPage.clickCreateVM();

      await page.locator('[data-testid="vm-name"]').fill('migration-test-vm');
      await page.locator('[data-testid="template-ubuntu-22.04"]').click();
      await page.locator('[data-testid="wizard-next"]').click();

      await page.locator('[data-testid="cpu-input"]').fill('4');
      await page.locator('[data-testid="memory-input"]').fill('8192');
      await page.locator('[data-testid="wizard-next"]').click();

      // Pin to source node
      await page.locator('[data-testid="advanced-options"]').click();
      await page.locator('[data-testid="pin-to-node"]').check();
      await page.locator('[data-testid="node-select"]').selectOption(sourceNode);
      await page.locator('[data-testid="wizard-next"]').click();

      await page.locator('[data-testid="wizard-submit"]').click();

      await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

      // Verify VM is on source node
      const currentNode = await vmDetailsPage.getCurrentNode();
      expect(currentNode).toBe(sourceNode);
    });

    await test.step('Start live migration', async () => {
      await vmDetailsPage.clickAction('migrate');

      await migrationWizard.selectMigrationType('live');
      await migrationWizard.selectTargetNode(targetNode);

      // Review migration plan
      const plan = await migrationWizard.getMigrationPlan();
      expect(plan).toMatchObject({
        type: 'live',
        sourceNode,
        targetNode,
        estimatedDowntime: expect.any(String)
      });

      await migrationWizard.submit();
    });

    await test.step('Monitor migration progress', async () => {
      const progressDialog = page.locator('[data-testid="migration-progress"]');
      await expect(progressDialog).toBeVisible();

      // Wait for phases
      await expect(page.locator('[data-testid="phase-preparing"]')).toHaveClass(/active/);
      await expect(page.locator('[data-testid="phase-transferring"]')).toHaveClass(/active/, { timeout: 30000 });
      await expect(page.locator('[data-testid="phase-completing"]')).toHaveClass(/active/, { timeout: 60000 });

      // Wait for completion
      await expect(page.locator('[data-testid="migration-complete"]')).toBeVisible({ timeout: 120000 });
    });

    await test.step('Verify migration success', async () => {
      await page.locator('[data-testid="close-migration-dialog"]').click();

      // VM should still be running
      const status = await vmDetailsPage.getVMStatus();
      expect(status).toBe('running');

      // Verify VM is on target node
      const currentNode = await vmDetailsPage.getCurrentNode();
      expect(currentNode).toBe(targetNode);

      // Check migration history
      await vmDetailsPage.navigateToTab('history');
      const history = await vmDetailsPage.getMigrationHistory();
      expect(history[0]).toMatchObject({
        type: 'live',
        from: sourceNode,
        to: targetNode,
        status: 'completed'
      });
    });

    await test.step('Cleanup', async () => {
      await vmDetailsPage.clickAction('delete');
      await page.locator('[data-testid="delete-confirm-checkbox"]').check();
      await page.locator('[data-testid="delete-confirm"]').click();
    });
  });

  test('should perform cold migration with minimal downtime', async ({ page }) => {
    await test.step('Create VM', async () => {
      await dashboardPage.goto();
      await dashboardPage.clickCreateVM();

      await page.locator('[data-testid="vm-name"]').fill('cold-migration-vm');
      await page.locator('[data-testid="quick-create-alpine"]').click();

      await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
    });

    const sourceNode = await vmDetailsPage.getCurrentNode();

    await test.step('Stop VM for cold migration', async () => {
      await vmDetailsPage.clickAction('stop');
      await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });
    });

    await test.step('Perform cold migration', async () => {
      await vmDetailsPage.clickAction('migrate');

      await migrationWizard.selectMigrationType('cold');

      const nodes = await migrationWizard.getAvailableNodes();
      const targetNode = nodes.find(n => n.name !== sourceNode);

      await migrationWizard.selectTargetNode(targetNode!.name);
      await migrationWizard.submit();

      await expect(page.locator('[data-testid="migration-complete"]')).toBeVisible({ timeout: 60000 });
    });

    await test.step('Verify and restart VM', async () => {
      const currentNode = await vmDetailsPage.getCurrentNode();
      expect(currentNode).not.toBe(sourceNode);

      await vmDetailsPage.clickAction('start');
      await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
    });

    await test.step('Cleanup', async () => {
      await vmDetailsPage.clickAction('delete');
      await page.locator('[data-testid="delete-confirm-checkbox"]').check();
      await page.locator('[data-testid="delete-confirm"]').click();
    });
  });

  test('should handle migration failures gracefully', async ({ page }) => {
    await test.step('Create VM', async () => {
      await dashboardPage.goto();
      await dashboardPage.clickCreateVM();

      await page.locator('[data-testid="vm-name"]').fill('migration-fail-vm');
      await page.locator('[data-testid="quick-create-debian"]').click();

      await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
    });

    await test.step('Attempt migration to unavailable node', async () => {
      await vmDetailsPage.clickAction('migrate');
      await migrationWizard.selectMigrationType('live');

      // Select node and simulate failure
      await page.route('**/api/migrations', async route => {
        if (route.request().method() === 'POST') {
          await route.fulfill({
            status: 500,
            body: JSON.stringify({
              error: 'Target node not reachable'
            })
          });
        }
      });

      const nodes = await migrationWizard.getAvailableNodes();
      await migrationWizard.selectTargetNode(nodes[0].name);
      await migrationWizard.submit();

      // Should show error
      await expect(page.locator('[data-testid="migration-error"]')).toBeVisible();
      await expect(page.locator('[data-testid="migration-error"]')).toContainText('not reachable');
    });

    await test.step('Verify VM state unchanged', async () => {
      await page.locator('[data-testid="close-migration-dialog"]').click();

      const status = await vmDetailsPage.getVMStatus();
      expect(status).toBe('running');
    });

    await test.step('Cleanup', async () => {
      await vmDetailsPage.clickAction('delete');
      await page.locator('[data-testid="delete-confirm-checkbox"]').check();
      await page.locator('[data-testid="delete-confirm"]').click();
    });
  });

  test('should support batch migration of multiple VMs', async ({ page }) => {
    const vmNames = ['batch-vm-1', 'batch-vm-2', 'batch-vm-3'];

    await test.step('Create multiple VMs', async () => {
      for (const name of vmNames) {
        await dashboardPage.goto();
        await dashboardPage.clickCreateVM();

        await page.locator('[data-testid="vm-name"]').fill(name);
        await page.locator('[data-testid="quick-create-alpine"]').click();

        await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
      }
    });

    await test.step('Select VMs for batch migration', async () => {
      await dashboardPage.goto();

      for (const name of vmNames) {
        await page.locator(`[data-testid="vm-checkbox-${name}"]`).check();
      }

      await page.locator('[data-testid="bulk-actions"]').click();
      await page.locator('[data-testid="bulk-migrate"]').click();
    });

    await test.step('Configure batch migration', async () => {
      await migrationWizard.selectMigrationType('live');

      const nodes = await migrationWizard.getAvailableNodes();
      await migrationWizard.selectTargetNode(nodes[0].name);

      await migrationWizard.enableParallelMigration(true);
      await migrationWizard.setMaxParallel(2);

      await migrationWizard.submit();
    });

    await test.step('Monitor batch migration', async () => {
      const batchProgress = page.locator('[data-testid="batch-migration-progress"]');
      await expect(batchProgress).toBeVisible();

      // Wait for all migrations to complete
      await expect(page.locator('[data-testid="batch-complete"]')).toBeVisible({ timeout: 300000 });

      const results = await page.locator('[data-testid="migration-result"]').all();
      expect(results).toHaveLength(3);

      for (const result of results) {
        await expect(result).toContainText('completed');
      }
    });

    await test.step('Cleanup', async () => {
      await page.locator('[data-testid="close-batch-dialog"]').click();
      await dashboardPage.goto();

      for (const name of vmNames) {
        await page.locator(`[data-testid="vm-checkbox-${name}"]`).check();
      }

      await page.locator('[data-testid="bulk-actions"]').click();
      await page.locator('[data-testid="bulk-delete"]').click();
      await page.locator('[data-testid="delete-confirm-checkbox"]').check();
      await page.locator('[data-testid="delete-confirm"]').click();
    });
  });
});
