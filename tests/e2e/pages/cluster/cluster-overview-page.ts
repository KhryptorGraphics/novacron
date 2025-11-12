import { Page, Locator } from '@playwright/test';
import { BasePage } from '../base-page';

/**
 * Cluster health status
 */
export type ClusterHealth = 'healthy' | 'warning' | 'critical' | 'unknown';

/**
 * Cluster Overview Page Object Model
 * Handles cluster dashboard and overview operations
 */
export class ClusterOverviewPage extends BasePage {
  // Header
  private readonly clusterName: Locator;
  private readonly clusterHealth: Locator;
  private readonly refreshButton: Locator;
  private readonly settingsButton: Locator;

  // Overview metrics
  private readonly totalNodes: Locator;
  private readonly activeNodes: Locator;
  private readonly totalVMs: Locator;
  private readonly runningVMs: Locator;
  private readonly cpuUtilization: Locator;
  private readonly memoryUtilization: Locator;
  private readonly storageUtilization: Locator;
  private readonly networkThroughput: Locator;

  // Resource charts
  private readonly cpuChart: Locator;
  private readonly memoryChart: Locator;
  private readonly storageChart: Locator;
  private readonly networkChart: Locator;
  private readonly chartTimeRange: Locator;

  // Node list
  private readonly nodeList: Locator;
  private readonly nodeFilter: Locator;
  private readonly nodeSortDropdown: Locator;

  // VM list
  private readonly vmList: Locator;
  private readonly vmFilter: Locator;

  // Alerts
  private readonly alertsPanel: Locator;
  private readonly alertsList: Locator;
  private readonly alertsCount: Locator;

  // Federation status
  private readonly federationStatus: Locator;
  private readonly federatedClusters: Locator;

  constructor(page: Page) {
    super(page);

    // Header
    this.clusterName = this.page.getByTestId('cluster-overview-name');
    this.clusterHealth = this.page.getByTestId('cluster-overview-health');
    this.refreshButton = this.page.getByTestId('cluster-overview-refresh');
    this.settingsButton = this.page.getByTestId('cluster-overview-settings');

    // Overview metrics
    this.totalNodes = this.page.getByTestId('cluster-overview-total-nodes');
    this.activeNodes = this.page.getByTestId('cluster-overview-active-nodes');
    this.totalVMs = this.page.getByTestId('cluster-overview-total-vms');
    this.runningVMs = this.page.getByTestId('cluster-overview-running-vms');
    this.cpuUtilization = this.page.getByTestId('cluster-overview-cpu-utilization');
    this.memoryUtilization = this.page.getByTestId('cluster-overview-memory-utilization');
    this.storageUtilization = this.page.getByTestId('cluster-overview-storage-utilization');
    this.networkThroughput = this.page.getByTestId('cluster-overview-network-throughput');

    // Resource charts
    this.cpuChart = this.page.getByTestId('cluster-overview-cpu-chart');
    this.memoryChart = this.page.getByTestId('cluster-overview-memory-chart');
    this.storageChart = this.page.getByTestId('cluster-overview-storage-chart');
    this.networkChart = this.page.getByTestId('cluster-overview-network-chart');
    this.chartTimeRange = this.page.getByTestId('cluster-overview-chart-time-range');

    // Node list
    this.nodeList = this.page.getByTestId('cluster-overview-node-list');
    this.nodeFilter = this.page.getByTestId('cluster-overview-node-filter');
    this.nodeSortDropdown = this.page.getByTestId('cluster-overview-node-sort');

    // VM list
    this.vmList = this.page.getByTestId('cluster-overview-vm-list');
    this.vmFilter = this.page.getByTestId('cluster-overview-vm-filter');

    // Alerts
    this.alertsPanel = this.page.getByTestId('cluster-overview-alerts-panel');
    this.alertsList = this.page.getByTestId('cluster-overview-alerts-list');
    this.alertsCount = this.page.getByTestId('cluster-overview-alerts-count');

    // Federation
    this.federationStatus = this.page.getByTestId('cluster-overview-federation-status');
    this.federatedClusters = this.page.getByTestId('cluster-overview-federated-clusters');
  }

  /**
   * Navigate to cluster overview page
   * @param clusterId - Cluster ID (optional, uses default if not provided)
   * @returns ClusterOverviewPage instance for chaining
   */
  async navigate(clusterId?: string): Promise<this> {
    const path = clusterId ? `/cluster/${clusterId}` : '/cluster';
    await this.goto(path);
    await this.waitForLoad();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Refresh cluster data
   * @returns ClusterOverviewPage instance for chaining
   */
  async refresh(): Promise<this> {
    await this.refreshButton.click();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Get cluster name
   * @returns Cluster name
   */
  async getClusterName(): Promise<string> {
    return (await this.clusterName.textContent()) || '';
  }

  /**
   * Get cluster health status
   * @returns Health status
   */
  async getClusterHealth(): Promise<ClusterHealth> {
    const healthText = await this.clusterHealth.getAttribute('data-health');
    return (healthText as ClusterHealth) || 'unknown';
  }

  /**
   * Get total nodes count
   * @returns Total nodes
   */
  async getTotalNodes(): Promise<number> {
    const text = await this.totalNodes.textContent();
    return parseInt(text || '0');
  }

  /**
   * Get active nodes count
   * @returns Active nodes
   */
  async getActiveNodes(): Promise<number> {
    const text = await this.activeNodes.textContent();
    return parseInt(text || '0');
  }

  /**
   * Get total VMs count
   * @returns Total VMs
   */
  async getTotalVMs(): Promise<number> {
    const text = await this.totalVMs.textContent();
    return parseInt(text || '0');
  }

  /**
   * Get running VMs count
   * @returns Running VMs
   */
  async getRunningVMs(): Promise<number> {
    const text = await this.runningVMs.textContent();
    return parseInt(text || '0');
  }

  /**
   * Get CPU utilization percentage
   * @returns CPU utilization
   */
  async getCPUUtilization(): Promise<number> {
    const text = await this.cpuUtilization.textContent();
    const match = text?.match(/(\d+(\.\d+)?)/);
    return match ? parseFloat(match[1]) : 0;
  }

  /**
   * Get memory utilization percentage
   * @returns Memory utilization
   */
  async getMemoryUtilization(): Promise<number> {
    const text = await this.memoryUtilization.textContent();
    const match = text?.match(/(\d+(\.\d+)?)/);
    return match ? parseFloat(match[1]) : 0;
  }

  /**
   * Get storage utilization percentage
   * @returns Storage utilization
   */
  async getStorageUtilization(): Promise<number> {
    const text = await this.storageUtilization.textContent();
    const match = text?.match(/(\d+(\.\d+)?)/);
    return match ? parseFloat(match[1]) : 0;
  }

  /**
   * Get network throughput
   * @returns Network throughput text
   */
  async getNetworkThroughput(): Promise<string> {
    return (await this.networkThroughput.textContent()) || '';
  }

  /**
   * Set chart time range
   * @param range - Time range (1h, 6h, 24h, 7d, 30d)
   * @returns ClusterOverviewPage instance for chaining
   */
  async setChartTimeRange(range: '1h' | '6h' | '24h' | '7d' | '30d'): Promise<this> {
    await this.chartTimeRange.selectOption(range);
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Filter nodes by status
   * @param status - Node status filter
   * @returns ClusterOverviewPage instance for chaining
   */
  async filterNodes(status: string): Promise<this> {
    await this.nodeFilter.selectOption(status);
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Sort nodes
   * @param sortBy - Sort field
   * @returns ClusterOverviewPage instance for chaining
   */
  async sortNodes(sortBy: string): Promise<this> {
    await this.nodeSortDropdown.selectOption(sortBy);
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Get node by name
   * @param nodeName - Node name
   * @returns Locator for the node
   */
  getNode(nodeName: string): Locator {
    return this.page.getByTestId(`cluster-node-${nodeName}`);
  }

  /**
   * Click on a node to view details
   * @param nodeName - Node name
   * @returns ClusterOverviewPage instance for chaining
   */
  async clickNode(nodeName: string): Promise<this> {
    await this.getNode(nodeName).click();
    await this.page.waitForURL(`**/cluster/nodes/${nodeName}`);
    return this;
  }

  /**
   * Get VM by name
   * @param vmName - VM name
   * @returns Locator for the VM
   */
  getVM(vmName: string): Locator {
    return this.page.getByTestId(`cluster-vm-${vmName}`);
  }

  /**
   * Click on a VM to view details
   * @param vmName - VM name
   * @returns ClusterOverviewPage instance for chaining
   */
  async clickVM(vmName: string): Promise<this> {
    await this.getVM(vmName).click();
    await this.page.waitForURL(`**/vms/${vmName}`);
    return this;
  }

  /**
   * Get alerts count
   * @returns Number of alerts
   */
  async getAlertsCount(): Promise<number> {
    const text = await this.alertsCount.textContent();
    return parseInt(text || '0');
  }

  /**
   * Get alert by index
   * @param index - Alert index
   * @returns Alert text
   */
  async getAlert(index: number): Promise<string> {
    const alert = this.alertsList.locator('[data-testid="alert-item"]').nth(index);
    return (await alert.textContent()) || '';
  }

  /**
   * Get all alerts
   * @returns Array of alert texts
   */
  async getAllAlerts(): Promise<string[]> {
    const alerts = await this.alertsList.locator('[data-testid="alert-item"]').all();
    const alertTexts: string[] = [];

    for (const alert of alerts) {
      const text = await alert.textContent();
      if (text) alertTexts.push(text);
    }

    return alertTexts;
  }

  /**
   * Dismiss alert
   * @param index - Alert index
   * @returns ClusterOverviewPage instance for chaining
   */
  async dismissAlert(index: number): Promise<this> {
    const alert = this.alertsList.locator('[data-testid="alert-item"]').nth(index);
    await alert.getByTestId('alert-dismiss-button').click();
    return this;
  }

  /**
   * Get federation status
   * @returns Federation status text
   */
  async getFederationStatus(): Promise<string> {
    return (await this.federationStatus.textContent()) || '';
  }

  /**
   * Get federated clusters count
   * @returns Number of federated clusters
   */
  async getFederatedClustersCount(): Promise<number> {
    const text = await this.federatedClusters.textContent();
    const match = text?.match(/(\d+)/);
    return match ? parseInt(match[1]) : 0;
  }

  /**
   * Open cluster settings
   * @returns ClusterOverviewPage instance for chaining
   */
  async openSettings(): Promise<this> {
    await this.settingsButton.click();
    await this.page.waitForURL('**/cluster/settings');
    return this;
  }

  /**
   * Check if cluster is healthy
   * @returns True if cluster is healthy
   */
  async isClusterHealthy(): Promise<boolean> {
    const health = await this.getClusterHealth();
    return health === 'healthy';
  }

  /**
   * Wait for cluster health to change
   * @param expectedHealth - Expected health status
   * @param timeout - Maximum time to wait
   * @returns ClusterOverviewPage instance for chaining
   */
  async waitForHealthStatus(
    expectedHealth: ClusterHealth,
    timeout: number = 30000
  ): Promise<this> {
    await this.page.waitForFunction(
      (health) => {
        const healthElement = document.querySelector(
          '[data-testid="cluster-overview-health"]'
        );
        return healthElement?.getAttribute('data-health') === health;
      },
      expectedHealth,
      { timeout }
    );
    return this;
  }

  /**
   * Check if CPU chart is visible
   * @returns True if chart is visible
   */
  async isCPUChartVisible(): Promise<boolean> {
    return await this.cpuChart.isVisible();
  }

  /**
   * Check if memory chart is visible
   * @returns True if chart is visible
   */
  async isMemoryChartVisible(): Promise<boolean> {
    return await this.memoryChart.isVisible();
  }

  /**
   * Check if storage chart is visible
   * @returns True if chart is visible
   */
  async isStorageChartVisible(): Promise<boolean> {
    return await this.storageChart.isVisible();
  }

  /**
   * Check if network chart is visible
   * @returns True if chart is visible
   */
  async isNetworkChartVisible(): Promise<boolean> {
    return await this.networkChart.isVisible();
  }
}
