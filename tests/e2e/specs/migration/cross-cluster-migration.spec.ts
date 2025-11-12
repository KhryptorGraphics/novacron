import { test, expect } from '@playwright/test';
import { DashboardPage } from '../../pages/DashboardPage';
import { VMDetailsPage } from '../../pages/VMDetailsPage';
import { MigrationWizard } from '../../pages/MigrationWizard';
import { ClusterPage } from '../../pages/ClusterPage';
import { authenticateUser } from '../../fixtures/auth';
import { testData } from '../../fixtures/testData';

test.describe('Migration: Cross-Cluster Migration', () => {
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

  test('should perform cross-cluster migration', {
    tag: '@smoke',
  }, async ({ page }) => {
    // Verify federation is set up
    await clusterPage.goto();
    await clusterPage.navigateToTab('federation');

    const federations = await clusterPage.getFederationList();
    expect(federations.length).toBeGreaterThan(0);

    const targetCluster = federations[0];

    // Create VM
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `cross-cluster-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-minimal"]').click();
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    const sourceCluster = await vmDetailsPage.getClusterName();

    // Stop VM for cross-cluster migration
    await vmDetailsPage.clickAction('stop');
    await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });

    // Start cross-cluster migration
    await vmDetailsPage.clickAction('migrate');
    await migrationWizard.selectMigrationType('cross-cluster');

    await migrationWizard.selectTargetCluster(targetCluster.name);

    // Select target node in remote cluster
    const remoteNodes = await migrationWizard.getRemoteClusterNodes();
    expect(remoteNodes.length).toBeGreaterThan(0);

    await migrationWizard.selectTargetNode(remoteNodes[0].name);

    // Configure network mapping
    await migrationWizard.mapNetwork('default', 'default');

    // Configure storage mapping
    await migrationWizard.mapStorage('local-lvm', 'remote-lvm');

    await migrationWizard.submit();

    // Monitor cross-cluster migration
    await expect(page.locator('[data-testid="migration-progress"]')).toBeVisible();

    await expect(page.locator('[data-testid="phase-exporting"]')).toHaveClass(/active/, { timeout: 30000 });
    await expect(page.locator('[data-testid="phase-transferring"]')).toHaveClass(/active/, { timeout: 60000 });
    await expect(page.locator('[data-testid="phase-importing"]')).toHaveClass(/active/, { timeout: 120000 });

    await expect(page.locator('[data-testid="migration-complete"]')).toBeVisible({ timeout: 300000 });

    // Verify VM is on remote cluster
    const currentCluster = await vmDetailsPage.getClusterName();
    expect(currentCluster).toBe(targetCluster.name);

    // Verify VM can start on remote cluster
    await vmDetailsPage.clickAction('start');
    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    // Cleanup
    await vmDetailsPage.forceDelete();
  });

  test('should validate network compatibility for cross-cluster migration', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `network-compat-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-web"]').click();
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
    await vmDetailsPage.clickAction('stop');
    await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });

    await vmDetailsPage.clickAction('migrate');
    await migrationWizard.selectMigrationType('cross-cluster');

    const federations = await migrationWizard.getTargetClusters();
    await migrationWizard.selectTargetCluster(federations[0].name);

    // Should show network compatibility check
    const networkCheck = page.locator('[data-testid="network-compatibility"]');
    await expect(networkCheck).toBeVisible();

    // Should suggest network mappings
    const suggestions = await migrationWizard.getNetworkMappingSuggestions();
    expect(suggestions.length).toBeGreaterThan(0);

    // Cleanup
    await page.locator('[data-testid="cancel-migration"]').click();
    await vmDetailsPage.forceDelete();
  });

  test('should handle storage format conversion during cross-cluster migration', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `storage-convert-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-database"]').click();
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
    await vmDetailsPage.clickAction('stop');
    await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });

    await vmDetailsPage.clickAction('migrate');
    await migrationWizard.selectMigrationType('cross-cluster');

    const federations = await migrationWizard.getTargetClusters();
    await migrationWizard.selectTargetCluster(federations[0].name);

    const remoteNodes = await migrationWizard.getRemoteClusterNodes();
    await migrationWizard.selectTargetNode(remoteNodes[0].name);

    // Configure storage with format conversion
    await migrationWizard.mapStorage('local-lvm', 'ceph-pool');
    await migrationWizard.enableFormatConversion();
    await migrationWizard.selectTargetFormat('qcow2');

    await migrationWizard.submit();

    await expect(page.locator('[data-testid="migration-progress"]')).toBeVisible();

    // Should show conversion progress
    await expect(page.locator('[data-testid="converting-format"]')).toBeVisible({ timeout: 60000 });

    await expect(page.locator('[data-testid="migration-complete"]')).toBeVisible({ timeout: 300000 });

    // Cleanup
    await vmDetailsPage.forceDelete();
  });

  test('should migrate VM with multiple disks across clusters', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.clickCreateVM();

    const vmName = `multi-disk-cross-${Date.now()}`;
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="template-ubuntu-22.04"]').click();
    await page.locator('[data-testid="wizard-next"]').click();

    await page.locator('[data-testid="cpu-input"]').fill('2');
    await page.locator('[data-testid="memory-input"]').fill('4096');
    await page.locator('[data-testid="disk-input"]').fill('50');

    // Add additional disks
    await page.locator('[data-testid="add-disk"]').click();
    await page.locator('[data-testid="disk-size"]').fill('100');
    await page.locator('[data-testid="add-disk-submit"]').click();

    await page.locator('[data-testid="wizard-next"]').click();
    await page.locator('[data-testid="wizard-next"]').click();
    await page.locator('[data-testid="wizard-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
    await vmDetailsPage.clickAction('stop');
    await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });

    await vmDetailsPage.clickAction('migrate');
    await migrationWizard.selectMigrationType('cross-cluster');

    const federations = await migrationWizard.getTargetClusters();
    await migrationWizard.selectTargetCluster(federations[0].name);

    const remoteNodes = await migrationWizard.getRemoteClusterNodes();
    await migrationWizard.selectTargetNode(remoteNodes[0].name);

    // Map each disk separately
    const disks = await migrationWizard.getDiskList();
    expect(disks.length).toBe(2);

    for (let i = 0; i < disks.length; i++) {
      await migrationWizard.mapDiskStorage(i, 'remote-storage');
    }

    await migrationWizard.submit();

    await expect(page.locator('[data-testid="migration-progress"]')).toBeVisible();

    // Should show progress for each disk
    for (let i = 0; i < disks.length; i++) {
      await expect(page.locator(`[data-testid="disk-${i}-progress"]`)).toBeVisible();
    }

    await expect(page.locator('[data-testid="migration-complete"]')).toBeVisible({ timeout: 300000 });

    // Cleanup
    await vmDetailsPage.forceDelete();
  });

  test('should handle federation connectivity issues', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `federation-fail-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-minimal"]').click();
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
    await vmDetailsPage.clickAction('stop');
    await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });

    await vmDetailsPage.clickAction('migrate');
    await migrationWizard.selectMigrationType('cross-cluster');

    // Simulate federation connectivity error
    await page.route('**/api/federation/*/nodes', route => {
      route.fulfill({
        status: 503,
        body: JSON.stringify({
          error: 'Target cluster unreachable'
        })
      });
    });

    const federations = await migrationWizard.getTargetClusters();
    await migrationWizard.selectTargetCluster(federations[0].name);

    await expect(page.locator('[data-testid="federation-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="federation-error"]')).toContainText('unreachable');

    // Cleanup
    await page.locator('[data-testid="cancel-migration"]').click();
    await vmDetailsPage.forceDelete();
  });
});
