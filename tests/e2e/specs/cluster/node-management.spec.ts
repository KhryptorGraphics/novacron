import { test, expect } from '@playwright/test';
import { ClusterPage } from '../../pages/ClusterPage';
import { NodeDetailsPage } from '../../pages/NodeDetailsPage';
import { authenticateUser } from '../../fixtures/auth';
import { testData } from '../../fixtures/testData';

test.describe('Cluster: Node Management', () => {
  let clusterPage: ClusterPage;
  let nodeDetailsPage: NodeDetailsPage;

  test.beforeEach(async ({ page }) => {
    await authenticateUser(page, testData.users.admin);

    clusterPage = new ClusterPage(page);
    nodeDetailsPage = new NodeDetailsPage(page);

    await clusterPage.goto();
  });

  test('should list all cluster nodes', {
    tag: '@smoke',
  }, async ({ page }) => {
    await clusterPage.waitForLoad();

    const nodes = await clusterPage.getNodeList();

    expect(nodes.length).toBeGreaterThan(0);

    for (const node of nodes) {
      expect(node).toHaveProperty('id');
      expect(node).toHaveProperty('name');
      expect(node).toHaveProperty('state');
      expect(node).toHaveProperty('cpuUsage');
      expect(node).toHaveProperty('memoryUsage');
    }
  });

  test('should view node details', async ({ page }) => {
    const nodes = await clusterPage.getNodeList();
    const firstNode = nodes[0];

    await clusterPage.clickNode(firstNode.id);

    await expect(nodeDetailsPage.nodeName).toContainText(firstNode.name);

    // Verify resource information
    await expect(nodeDetailsPage.cpuInfo).toBeVisible();
    await expect(nodeDetailsPage.memoryInfo).toBeVisible();
    await expect(nodeDetailsPage.storageInfo).toBeVisible();

    // Verify running VMs list
    await expect(page.locator('[data-testid="node-vms"]')).toBeVisible();
  });

  test('should drain node', async ({ page }) => {
    const nodes = await clusterPage.getNodeList();
    const targetNode = nodes.find(n => n.state === 'ready');

    expect(targetNode).toBeDefined();

    await clusterPage.clickNode(targetNode!.id);

    const vmCountBefore = await nodeDetailsPage.getRunningVMCount();

    // Drain node
    await nodeDetailsPage.clickAction('drain');
    await page.locator('[data-testid="drain-confirm"]').click();

    await expect(nodeDetailsPage.nodeStatus).toHaveText('draining', { timeout: 10000 });

    // Monitor draining progress
    await expect(page.locator('[data-testid="drain-progress"]')).toBeVisible();

    if (vmCountBefore > 0) {
      await expect(page.locator('[data-testid="drain-complete"]')).toBeVisible({ timeout: 180000 });

      const vmCountAfter = await nodeDetailsPage.getRunningVMCount();
      expect(vmCountAfter).toBe(0);
    }

    await expect(nodeDetailsPage.nodeStatus).toHaveText('drained');
  });

  test('should uncordon node', async ({ page }) => {
    const nodes = await clusterPage.getNodeList();
    const drainedNode = nodes.find(n => n.state === 'drained');

    if (!drainedNode) {
      // Drain a node first
      const targetNode = nodes.find(n => n.state === 'ready');
      await clusterPage.clickNode(targetNode!.id);
      await nodeDetailsPage.clickAction('drain');
      await page.locator('[data-testid="drain-confirm"]').click();
      await expect(nodeDetailsPage.nodeStatus).toHaveText('drained', { timeout: 180000 });
    } else {
      await clusterPage.clickNode(drainedNode.id);
    }

    await nodeDetailsPage.clickAction('uncordon');

    await expect(nodeDetailsPage.nodeStatus).toHaveText('ready', { timeout: 10000 });

    const schedulable = await nodeDetailsPage.isSchedulable();
    expect(schedulable).toBe(true);
  });

  test('should monitor node health', async ({ page }) => {
    const nodes = await clusterPage.getNodeList();
    const healthyNode = nodes.find(n => n.state === 'ready');

    expect(healthyNode).toBeDefined();

    await clusterPage.clickNode(healthyNode!.id);

    await nodeDetailsPage.navigateToTab('health');

    // Verify health metrics
    await expect(page.locator('[data-testid="health-status"]')).toBeVisible();
    await expect(page.locator('[data-testid="uptime"]')).toBeVisible();
    await expect(page.locator('[data-testid="load-average"]')).toBeVisible();

    // Verify system checks
    const checks = await page.locator('[data-testid="health-check"]').all();
    expect(checks.length).toBeGreaterThan(0);

    for (const check of checks) {
      const status = await check.getAttribute('data-status');
      expect(['passing', 'warning', 'critical']).toContain(status);
    }
  });

  test('should update node labels', async ({ page }) => {
    const nodes = await clusterPage.getNodeList();
    const targetNode = nodes[0];

    await clusterPage.clickNode(targetNode.id);

    await nodeDetailsPage.clickAction('edit-labels');

    // Add new label
    await page.locator('[data-testid="add-label"]').click();
    await page.locator('[data-testid="label-key"]').fill('environment');
    await page.locator('[data-testid="label-value"]').fill('production');
    await page.locator('[data-testid="add-label-submit"]').click();

    await page.locator('[data-testid="save-labels"]').click();

    await expect(page.locator('[data-testid="labels-updated"]')).toBeVisible();

    // Verify label appears
    const labels = await nodeDetailsPage.getLabels();
    expect(labels).toHaveProperty('environment', 'production');
  });

  test('should configure node taints', async ({ page }) => {
    const nodes = await clusterPage.getNodeList();
    const targetNode = nodes[0];

    await clusterPage.clickNode(targetNode.id);

    await nodeDetailsPage.clickAction('edit-taints');

    // Add taint
    await page.locator('[data-testid="add-taint"]').click();
    await page.locator('[data-testid="taint-key"]').fill('dedicated');
    await page.locator('[data-testid="taint-value"]').fill('database');
    await page.locator('[data-testid="taint-effect"]').selectOption('NoSchedule');
    await page.locator('[data-testid="add-taint-submit"]').click();

    await page.locator('[data-testid="save-taints"]').click();

    await expect(page.locator('[data-testid="taints-updated"]')).toBeVisible();

    // Verify taint appears
    const taints = await nodeDetailsPage.getTaints();
    expect(taints).toContainEqual(
      expect.objectContaining({
        key: 'dedicated',
        value: 'database',
        effect: 'NoSchedule'
      })
    );
  });

  test('should view node resource allocation', async ({ page }) => {
    const nodes = await clusterPage.getNodeList();
    const targetNode = nodes[0];

    await clusterPage.clickNode(targetNode.id);

    await nodeDetailsPage.navigateToTab('allocation');

    // Verify allocation charts
    await expect(page.locator('[data-testid="cpu-allocation-chart"]')).toBeVisible();
    await expect(page.locator('[data-testid="memory-allocation-chart"]')).toBeVisible();
    await expect(page.locator('[data-testid="storage-allocation-chart"]')).toBeVisible();

    // Verify allocation details
    const allocation = await nodeDetailsPage.getResourceAllocation();

    expect(allocation.cpu).toHaveProperty('total');
    expect(allocation.cpu).toHaveProperty('allocated');
    expect(allocation.cpu).toHaveProperty('available');

    expect(allocation.memory).toHaveProperty('total');
    expect(allocation.memory).toHaveProperty('allocated');
    expect(allocation.memory).toHaveProperty('available');
  });

  test('should filter nodes by labels', async ({ page }) => {
    await clusterPage.clickFilter();

    await page.locator('[data-testid="filter-by-label"]').click();
    await page.locator('[data-testid="label-key-input"]').fill('environment');
    await page.locator('[data-testid="label-value-input"]').fill('production');
    await page.locator('[data-testid="apply-filter"]').click();

    const filteredNodes = await clusterPage.getNodeList();

    for (const node of filteredNodes) {
      await clusterPage.clickNode(node.id);
      const labels = await nodeDetailsPage.getLabels();
      expect(labels.environment).toBe('production');
      await clusterPage.goto();
    }
  });

  test('should show node events', async ({ page }) => {
    const nodes = await clusterPage.getNodeList();
    const targetNode = nodes[0];

    await clusterPage.clickNode(targetNode.id);

    await nodeDetailsPage.navigateToTab('events');

    const events = await page.locator('[data-testid="node-event"]').all();
    expect(events.length).toBeGreaterThan(0);

    // Verify event structure
    const firstEvent = events[0];
    await expect(firstEvent.locator('[data-testid="event-type"]')).toBeVisible();
    await expect(firstEvent.locator('[data-testid="event-message"]')).toBeVisible();
    await expect(firstEvent.locator('[data-testid="event-timestamp"]')).toBeVisible();
  });

  test('should export node configuration', async ({ page }) => {
    const nodes = await clusterPage.getNodeList();
    const targetNode = nodes[0];

    await clusterPage.clickNode(targetNode.id);

    await nodeDetailsPage.clickAction('export-config');

    const downloadPromise = page.waitForEvent('download');
    await page.locator('[data-testid="confirm-export"]').click();

    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/node-config.*\.json/);
  });
});
