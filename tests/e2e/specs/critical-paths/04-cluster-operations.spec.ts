import { test, expect } from '@playwright/test';
import { ClusterPage } from '../../pages/ClusterPage';
import { NodeDetailsPage } from '../../pages/NodeDetailsPage';
import { DashboardPage } from '../../pages/DashboardPage';
import { authenticateUser } from '../../fixtures/auth';
import { testData } from '../../fixtures/testData';

test.describe('Critical Path: Cluster Management Operations', () => {
  let clusterPage: ClusterPage;
  let nodeDetailsPage: NodeDetailsPage;
  let dashboardPage: DashboardPage;

  test.beforeEach(async ({ page }) => {
    await authenticateUser(page, testData.users.admin);

    clusterPage = new ClusterPage(page);
    nodeDetailsPage = new NodeDetailsPage(page);
    dashboardPage = new DashboardPage(page);

    await clusterPage.goto();
  });

  test('should manage cluster lifecycle: create, scale, drain, remove nodes', {
    tag: '@smoke @critical',
  }, async ({ page }) => {
    let newNodeId: string;

    await test.step('Verify initial cluster state', async () => {
      await clusterPage.waitForLoad();

      const nodes = await clusterPage.getNodeList();
      expect(nodes.length).toBeGreaterThan(0);

      const clusterHealth = await clusterPage.getClusterHealth();
      expect(clusterHealth.status).toBe('healthy');
    });

    await test.step('Add new node to cluster', async () => {
      await clusterPage.clickAddNode();

      await page.locator('[data-testid="node-hostname"]').fill('node-test-001.local');
      await page.locator('[data-testid="node-ip"]').fill('192.168.1.101');
      await page.locator('[data-testid="node-ssh-port"]').fill('22');

      // Provide SSH credentials
      await page.locator('[data-testid="node-ssh-user"]').fill('root');
      await page.locator('[data-testid="node-ssh-key"]').selectOption('default-key');

      // Set node labels
      await page.locator('[data-testid="add-label"]').click();
      await page.locator('[data-testid="label-key"]').fill('environment');
      await page.locator('[data-testid="label-value"]').fill('test');

      await page.locator('[data-testid="submit-add-node"]').click();

      // Wait for node to be added
      await expect(page.locator('[data-testid="node-provisioning"]')).toBeVisible();
      await expect(page.locator('[data-testid="node-ready"]')).toBeVisible({ timeout: 120000 });

      newNodeId = await page.locator('[data-testid="new-node-id"]').textContent() || '';
    });

    await test.step('Verify node is operational', async () => {
      const nodes = await clusterPage.getNodeList();
      const newNode = nodes.find(n => n.id === newNodeId);

      expect(newNode).toBeDefined();
      expect(newNode?.state).toBe('ready');
      expect(newNode?.labels).toContain('environment=test');

      // Check node resources
      await clusterPage.clickNode(newNodeId);

      const resources = await nodeDetailsPage.getResources();
      expect(resources.cpu.total).toBeGreaterThan(0);
      expect(resources.memory.total).toBeGreaterThan(0);
      expect(resources.storage.total).toBeGreaterThan(0);
    });

    await test.step('Create VMs on new node', async () => {
      await dashboardPage.goto();
      await dashboardPage.clickCreateVM();

      await page.locator('[data-testid="vm-name"]').fill('cluster-test-vm-1');
      await page.locator('[data-testid="quick-create-alpine"]').click();

      // Pin to new node
      await page.locator('[data-testid="advanced-options"]').click();
      await page.locator('[data-testid="pin-to-node"]').check();
      await page.locator('[data-testid="node-select"]').selectOption(newNodeId);
      await page.locator('[data-testid="wizard-submit"]').click();

      await page.locator('[data-testid="vm-status"]').waitFor({ state: 'visible' });
      await expect(page.locator('[data-testid="vm-status"]')).toHaveText('running', { timeout: 60000 });
    });

    await test.step('Drain node (evacuate VMs)', async () => {
      await clusterPage.goto();
      await clusterPage.clickNode(newNodeId);

      await nodeDetailsPage.clickAction('drain');
      await page.locator('[data-testid="drain-confirm"]').click();

      // Monitor draining process
      await expect(nodeDetailsPage.nodeStatus).toHaveText('draining', { timeout: 10000 });

      // Wait for VMs to migrate
      await expect(page.locator('[data-testid="drain-progress"]')).toBeVisible();
      await expect(page.locator('[data-testid="drain-complete"]')).toBeVisible({ timeout: 120000 });

      // Verify node is drained
      const vmCount = await nodeDetailsPage.getRunningVMCount();
      expect(vmCount).toBe(0);

      await expect(nodeDetailsPage.nodeStatus).toHaveText('drained');
    });

    await test.step('Verify VMs migrated to other nodes', async () => {
      await dashboardPage.goto();

      const vm = await dashboardPage.findVM('cluster-test-vm-1');
      expect(vm).toBeDefined();
      expect(vm?.state).toBe('running');
      expect(vm?.nodeId).not.toBe(newNodeId);
    });

    await test.step('Uncordon node', async () => {
      await clusterPage.goto();
      await clusterPage.clickNode(newNodeId);

      await nodeDetailsPage.clickAction('uncordon');

      await expect(nodeDetailsPage.nodeStatus).toHaveText('ready', { timeout: 10000 });

      // Verify node can accept workloads again
      const schedulable = await nodeDetailsPage.isSchedulable();
      expect(schedulable).toBe(true);
    });

    await test.step('Remove node from cluster', async () => {
      await nodeDetailsPage.clickAction('remove');

      await page.locator('[data-testid="remove-force"]').uncheck();
      await page.locator('[data-testid="remove-confirm-text"]').fill('REMOVE');
      await page.locator('[data-testid="confirm-remove"]').click();

      // Wait for removal
      await expect(page.locator('[data-testid="node-removing"]')).toBeVisible();
      await expect(page).toHaveURL(/\/cluster/, { timeout: 60000 });

      // Verify node is removed
      const nodes = await clusterPage.getNodeList();
      const removedNode = nodes.find(n => n.id === newNodeId);
      expect(removedNode).toBeUndefined();
    });

    await test.step('Cleanup VMs', async () => {
      await dashboardPage.goto();
      await dashboardPage.clickVM('cluster-test-vm-1');

      await page.locator('[data-testid="action-delete"]').click();
      await page.locator('[data-testid="delete-confirm-checkbox"]').check();
      await page.locator('[data-testid="delete-confirm"]').click();
    });
  });

  test('should handle node failure and recovery', async ({ page }) => {
    let healthyNodeId: string;

    await test.step('Identify healthy node', async () => {
      const nodes = await clusterPage.getNodeList();
      const healthyNode = nodes.find(n => n.state === 'ready');

      expect(healthyNode).toBeDefined();
      healthyNodeId = healthyNode!.id;
    });

    await test.step('Create VMs on healthy node', async () => {
      await dashboardPage.goto();

      for (let i = 1; i <= 3; i++) {
        await dashboardPage.clickCreateVM();
        await page.locator('[data-testid="vm-name"]').fill(`failover-vm-${i}`);
        await page.locator('[data-testid="quick-create-alpine"]').click();
        await page.locator('[data-testid="vm-status"]').waitFor({ state: 'visible' });
        await dashboardPage.goto();
      }
    });

    await test.step('Simulate node failure', async () => {
      await clusterPage.goto();
      await clusterPage.clickNode(healthyNodeId);

      // Simulate network partition
      await nodeDetailsPage.clickAction('simulate-failure');
      await page.locator('[data-testid="failure-type-network"]').click();
      await page.locator('[data-testid="confirm-simulate"]').click();

      // Node should be marked as unreachable
      await expect(nodeDetailsPage.nodeStatus).toHaveText('unreachable', { timeout: 30000 });
    });

    await test.step('Verify automatic failover', async () => {
      // Check cluster detected the failure
      await clusterPage.goto();

      const alerts = await clusterPage.getActiveAlerts();
      expect(alerts).toContainEqual(
        expect.objectContaining({
          severity: 'critical',
          message: expect.stringContaining('Node unreachable')
        })
      );

      // VMs should be rescheduled
      await dashboardPage.goto();

      // Wait for HA to kick in
      await page.waitForTimeout(60000);

      const vms = await dashboardPage.getVMList();
      const failoverVMs = vms.filter(vm => vm.name.startsWith('failover-vm-'));

      for (const vm of failoverVMs) {
        // VMs should be running on different nodes
        expect(vm.state).toBe('running');
        expect(vm.nodeId).not.toBe(healthyNodeId);
      }
    });

    await test.step('Recover failed node', async () => {
      await clusterPage.goto();
      await clusterPage.clickNode(healthyNodeId);

      await nodeDetailsPage.clickAction('recover');
      await page.locator('[data-testid="confirm-recovery"]').click();

      await expect(nodeDetailsPage.nodeStatus).toHaveText('ready', { timeout: 60000 });
    });

    await test.step('Cleanup', async () => {
      await dashboardPage.goto();

      for (let i = 1; i <= 3; i++) {
        const vm = await dashboardPage.findVM(`failover-vm-${i}`);
        if (vm) {
          await dashboardPage.clickVM(`failover-vm-${i}`);
          await page.locator('[data-testid="action-delete"]').click();
          await page.locator('[data-testid="delete-confirm-checkbox"]').check();
          await page.locator('[data-testid="delete-confirm"]').click();
          await dashboardPage.goto();
        }
      }
    });
  });

  test('should manage cluster federation', async ({ page }) => {
    await test.step('Navigate to federation settings', async () => {
      await clusterPage.goto();
      await clusterPage.navigateToTab('federation');

      await expect(page.locator('[data-testid="federation-panel"]')).toBeVisible();
    });

    await test.step('Create federation with remote cluster', async () => {
      await page.locator('[data-testid="add-federation"]').click();

      await page.locator('[data-testid="federation-name"]').fill('test-federation');
      await page.locator('[data-testid="remote-cluster-url"]').fill('https://cluster2.example.com');
      await page.locator('[data-testid="federation-token"]').fill('test-token-12345');

      await page.locator('[data-testid="submit-federation"]').click();

      await expect(page.locator('[data-testid="federation-establishing"]')).toBeVisible();
      await expect(page.locator('[data-testid="federation-active"]')).toBeVisible({ timeout: 30000 });
    });

    await test.step('Verify federation capabilities', async () => {
      const federations = await clusterPage.getFederationList();

      expect(federations).toContainEqual(
        expect.objectContaining({
          name: 'test-federation',
          status: 'active'
        })
      );

      // Check cross-cluster visibility
      await page.locator('[data-testid="federation-test-federation"]').click();

      const remoteResources = await page.locator('[data-testid="remote-resource"]').all();
      expect(remoteResources.length).toBeGreaterThan(0);
    });

    await test.step('Remove federation', async () => {
      await page.locator('[data-testid="federation-actions"]').click();
      await page.locator('[data-testid="remove-federation"]').click();
      await page.locator('[data-testid="confirm-remove-federation"]').click();

      await expect(page.locator('[data-testid="federation-test-federation"]')).not.toBeVisible();
    });
  });
});
