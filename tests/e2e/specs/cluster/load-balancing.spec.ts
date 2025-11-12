import { test, expect } from '@playwright/test';
import { ClusterPage } from '../../pages/ClusterPage';
import { DashboardPage } from '../../pages/DashboardPage';
import { VMDetailsPage } from '../../pages/VMDetailsPage';
import { authenticateUser } from '../../fixtures/auth';
import { testData } from '../../fixtures/testData';

test.describe('Cluster: Load Balancing', () => {
  let clusterPage: ClusterPage;
  let dashboardPage: DashboardPage;
  let vmDetailsPage: VMDetailsPage;

  test.beforeEach(async ({ page }) => {
    await authenticateUser(page, testData.users.admin);

    clusterPage = new ClusterPage(page);
    dashboardPage = new DashboardPage(page);
    vmDetailsPage = new VMDetailsPage(page);
  });

  test('should distribute VMs across nodes automatically', {
    tag: '@smoke',
  }, async ({ page }) => {
    const vmNames: string[] = [];

    // Create multiple VMs
    for (let i = 0; i < 5; i++) {
      await dashboardPage.goto();
      await dashboardPage.clickQuickCreate();

      const vmName = `lb-test-${i}-${Date.now()}`;
      vmNames.push(vmName);

      await page.locator('[data-testid="quick-create-preset-minimal"]').click();
      await page.locator('[data-testid="vm-name"]').fill(vmName);
      await page.locator('[data-testid="quick-create-submit"]').click();

      await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
    }

    // Check distribution across nodes
    const nodeDistribution = new Map<string, number>();

    for (const vmName of vmNames) {
      await dashboardPage.goto();
      await dashboardPage.clickVM(vmName);

      const nodeId = await vmDetailsPage.getCurrentNode();
      nodeDistribution.set(nodeId, (nodeDistribution.get(nodeId) || 0) + 1);
    }

    // VMs should be distributed (not all on same node)
    expect(nodeDistribution.size).toBeGreaterThan(1);

    // Cleanup
    for (const vmName of vmNames) {
      await dashboardPage.goto();
      await dashboardPage.clickVM(vmName);
      await vmDetailsPage.forceDelete();
    }
  });

  test('should rebalance cluster load', async ({ page }) => {
    await clusterPage.goto();
    await clusterPage.navigateToTab('load-balancing');

    // Check current balance score
    const balanceScoreBefore = await clusterPage.getBalanceScore();

    await page.locator('[data-testid="rebalance-cluster"]').click();
    await page.locator('[data-testid="confirm-rebalance"]').click();

    await expect(page.locator('[data-testid="rebalancing"]')).toBeVisible();

    // Monitor rebalancing
    await expect(page.locator('[data-testid="rebalance-progress"]')).toBeVisible();
    await expect(page.locator('[data-testid="rebalance-complete"]')).toBeVisible({ timeout: 300000 });

    // Balance score should improve
    const balanceScoreAfter = await clusterPage.getBalanceScore();
    expect(balanceScoreAfter).toBeGreaterThanOrEqual(balanceScoreBefore);
  });

  test('should configure load balancing policy', async ({ page }) => {
    await clusterPage.goto();
    await clusterPage.navigateToTab('load-balancing');

    await page.locator('[data-testid="configure-policy"]').click();

    // Configure balancing strategy
    await page.locator('[data-testid="strategy"]').selectOption('balanced');

    // Configure thresholds
    await page.locator('[data-testid="cpu-threshold"]').fill('80');
    await page.locator('[data-testid="memory-threshold"]').fill('85');

    // Enable auto-rebalancing
    await page.locator('[data-testid="enable-auto-rebalance"]').check();
    await page.locator('[data-testid="rebalance-interval"]').fill('3600');

    await page.locator('[data-testid="save-policy"]').click();

    await expect(page.locator('[data-testid="policy-saved"]')).toBeVisible();

    // Verify policy is active
    const policy = await clusterPage.getLoadBalancingPolicy();
    expect(policy.strategy).toBe('balanced');
    expect(policy.autoRebalance).toBe(true);
  });

  test('should respect VM affinity rules during balancing', async ({ page }) => {
    // Create two VMs with affinity
    await dashboardPage.goto();
    await dashboardPage.clickCreateVM();

    const vm1Name = `affinity-vm-1-${Date.now()}`;
    await page.locator('[data-testid="vm-name"]').fill(vm1Name);
    await page.locator('[data-testid="template-alpine-3.18"]').click();
    await page.locator('[data-testid="wizard-next"]').click();
    await page.locator('[data-testid="wizard-next"]').click();
    await page.locator('[data-testid="wizard-next"]').click();
    await page.locator('[data-testid="wizard-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    const node1 = await vmDetailsPage.getCurrentNode();

    // Create second VM with affinity to first
    await dashboardPage.goto();
    await dashboardPage.clickCreateVM();

    const vm2Name = `affinity-vm-2-${Date.now()}`;
    await page.locator('[data-testid="vm-name"]').fill(vm2Name);
    await page.locator('[data-testid="template-alpine-3.18"]').click();
    await page.locator('[data-testid="wizard-next"]').click();

    // Configure affinity
    await page.locator('[data-testid="advanced-options"]').click();
    await page.locator('[data-testid="enable-affinity"]').check();
    await page.locator('[data-testid="affinity-target"]').selectOption(vm1Name);
    await page.locator('[data-testid="affinity-type"]').selectOption('same-node');

    await page.locator('[data-testid="wizard-next"]').click();
    await page.locator('[data-testid="wizard-next"]').click();
    await page.locator('[data-testid="wizard-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    const node2 = await vmDetailsPage.getCurrentNode();

    // VMs should be on same node
    expect(node2).toBe(node1);

    // Trigger rebalancing
    await clusterPage.goto();
    await clusterPage.navigateToTab('load-balancing');
    await page.locator('[data-testid="rebalance-cluster"]').click();
    await page.locator('[data-testid="confirm-rebalance"]').click();

    await expect(page.locator('[data-testid="rebalance-complete"]')).toBeVisible({ timeout: 300000 });

    // Verify VMs still on same node after rebalancing
    await dashboardPage.goto();
    await dashboardPage.clickVM(vm1Name);
    const finalNode1 = await vmDetailsPage.getCurrentNode();

    await dashboardPage.goto();
    await dashboardPage.clickVM(vm2Name);
    const finalNode2 = await vmDetailsPage.getCurrentNode();

    expect(finalNode2).toBe(finalNode1);

    // Cleanup
    await vmDetailsPage.forceDelete();
    await dashboardPage.goto();
    await dashboardPage.clickVM(vm1Name);
    await vmDetailsPage.forceDelete();
  });

  test('should respect anti-affinity rules', async ({ page }) => {
    // Create two VMs with anti-affinity
    await dashboardPage.goto();
    await dashboardPage.clickCreateVM();

    const vm1Name = `anti-affinity-vm-1-${Date.now()}`;
    await page.locator('[data-testid="vm-name"]').fill(vm1Name);
    await page.locator('[data-testid="template-alpine-3.18"]').click();
    await page.locator('[data-testid="wizard-next"]').click();
    await page.locator('[data-testid="wizard-next"]').click();
    await page.locator('[data-testid="wizard-next"]').click();
    await page.locator('[data-testid="wizard-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    const node1 = await vmDetailsPage.getCurrentNode();

    // Create second VM with anti-affinity
    await dashboardPage.goto();
    await dashboardPage.clickCreateVM();

    const vm2Name = `anti-affinity-vm-2-${Date.now()}`;
    await page.locator('[data-testid="vm-name"]').fill(vm2Name);
    await page.locator('[data-testid="template-alpine-3.18"]').click();
    await page.locator('[data-testid="wizard-next"]').click();

    await page.locator('[data-testid="advanced-options"]').click();
    await page.locator('[data-testid="enable-affinity"]').check();
    await page.locator('[data-testid="affinity-target"]').selectOption(vm1Name);
    await page.locator('[data-testid="affinity-type"]').selectOption('different-node');

    await page.locator('[data-testid="wizard-next"]').click();
    await page.locator('[data-testid="wizard-next"]').click();
    await page.locator('[data-testid="wizard-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    const node2 = await vmDetailsPage.getCurrentNode();

    // VMs should be on different nodes
    expect(node2).not.toBe(node1);

    // Cleanup
    await vmDetailsPage.forceDelete();
    await dashboardPage.goto();
    await dashboardPage.clickVM(vm1Name);
    await vmDetailsPage.forceDelete();
  });

  test('should show load distribution metrics', async ({ page }) => {
    await clusterPage.goto();
    await clusterPage.navigateToTab('load-balancing');

    // Verify distribution charts
    await expect(page.locator('[data-testid="cpu-distribution-chart"]')).toBeVisible();
    await expect(page.locator('[data-testid="memory-distribution-chart"]')).toBeVisible();
    await expect(page.locator('[data-testid="vm-distribution-chart"]')).toBeVisible();

    // Verify per-node metrics
    const nodeMetrics = await page.locator('[data-testid="node-metric"]').all();
    expect(nodeMetrics.length).toBeGreaterThan(0);

    for (const metric of nodeMetrics) {
      await expect(metric.locator('[data-testid="node-name"]')).toBeVisible();
      await expect(metric.locator('[data-testid="cpu-usage"]')).toBeVisible();
      await expect(metric.locator('[data-testid="memory-usage"]')).toBeVisible();
      await expect(metric.locator('[data-testid="vm-count"]')).toBeVisible();
    }
  });
});
