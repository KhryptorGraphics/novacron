import { Page, Locator } from '@playwright/test';
import { BasePage } from '../base-page';

/**
 * Node status type
 */
export type NodeStatus = 'online' | 'offline' | 'maintenance' | 'degraded';

/**
 * Node operation type
 */
export type NodeOperation = 'enable' | 'disable' | 'drain' | 'cordon' | 'uncordon' | 'reboot' | 'remove';

/**
 * Node Management Page Object Model
 * Handles node listing, operations, and management
 */
export class NodeManagementPage extends BasePage {
  // Header
  private readonly pageTitle: Locator;
  private readonly addNodeButton: Locator;
  private readonly refreshButton: Locator;
  private readonly bulkActionsDropdown: Locator;

  // Filters
  private readonly searchInput: Locator;
  private readonly statusFilter: Locator;
  private readonly roleFilter: Locator;
  private readonly clearFiltersButton: Locator;

  // Node list
  private readonly nodeList: Locator;
  private readonly nodeCards: Locator;
  private readonly emptyState: Locator;

  // Pagination
  private readonly paginationContainer: Locator;
  private readonly prevPageButton: Locator;
  private readonly nextPageButton: Locator;
  private readonly pageInfo: Locator;

  // Bulk selection
  private readonly selectAllCheckbox: Locator;

  constructor(page: Page) {
    super(page);

    // Header
    this.pageTitle = this.page.getByTestId('node-management-title');
    this.addNodeButton = this.page.getByTestId('node-management-add-button');
    this.refreshButton = this.page.getByTestId('node-management-refresh-button');
    this.bulkActionsDropdown = this.page.getByTestId('node-management-bulk-actions');

    // Filters
    this.searchInput = this.page.getByTestId('node-management-search-input');
    this.statusFilter = this.page.getByTestId('node-management-status-filter');
    this.roleFilter = this.page.getByTestId('node-management-role-filter');
    this.clearFiltersButton = this.page.getByTestId('node-management-clear-filters');

    // Node list
    this.nodeList = this.page.getByTestId('node-management-list');
    this.nodeCards = this.page.getByTestId('node-card');
    this.emptyState = this.page.getByTestId('node-management-empty-state');

    // Pagination
    this.paginationContainer = this.page.getByTestId('node-management-pagination');
    this.prevPageButton = this.page.getByTestId('node-management-prev-page');
    this.nextPageButton = this.page.getByTestId('node-management-next-page');
    this.pageInfo = this.page.getByTestId('node-management-page-info');

    // Bulk selection
    this.selectAllCheckbox = this.page.getByTestId('node-management-select-all');
  }

  /**
   * Navigate to node management page
   * @returns NodeManagementPage instance for chaining
   */
  async navigate(): Promise<this> {
    await this.goto('/cluster/nodes');
    await this.waitForLoad();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Click add node button
   * @returns NodeManagementPage instance for chaining
   */
  async clickAddNode(): Promise<this> {
    await this.addNodeButton.click();
    await this.page.waitForURL('**/cluster/nodes/add');
    return this;
  }

  /**
   * Refresh node list
   * @returns NodeManagementPage instance for chaining
   */
  async refresh(): Promise<this> {
    await this.refreshButton.click();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Search for nodes
   * @param query - Search query
   * @returns NodeManagementPage instance for chaining
   */
  async search(query: string): Promise<this> {
    await this.searchInput.fill(query);
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Filter nodes by status
   * @param status - Node status
   * @returns NodeManagementPage instance for chaining
   */
  async filterByStatus(status: NodeStatus): Promise<this> {
    await this.statusFilter.selectOption(status);
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Filter nodes by role
   * @param role - Node role
   * @returns NodeManagementPage instance for chaining
   */
  async filterByRole(role: string): Promise<this> {
    await this.roleFilter.selectOption(role);
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Clear all filters
   * @returns NodeManagementPage instance for chaining
   */
  async clearFilters(): Promise<this> {
    await this.clearFiltersButton.click();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Get node by name
   * @param nodeName - Node name
   * @returns Locator for the node
   */
  getNode(nodeName: string): Locator {
    return this.page.getByTestId(`node-card-${nodeName}`);
  }

  /**
   * Click on a node to view details
   * @param nodeName - Node name
   * @returns NodeManagementPage instance for chaining
   */
  async clickNode(nodeName: string): Promise<this> {
    await this.getNode(nodeName).click();
    await this.page.waitForURL(`**/cluster/nodes/${nodeName}`);
    return this;
  }

  /**
   * Get node status
   * @param nodeName - Node name
   * @returns Node status
   */
  async getNodeStatus(nodeName: string): Promise<string> {
    const statusLocator = this.getNode(nodeName).getByTestId('node-status');
    return (await statusLocator.textContent()) || '';
  }

  /**
   * Get node CPU usage
   * @param nodeName - Node name
   * @returns CPU usage percentage
   */
  async getNodeCPUUsage(nodeName: string): Promise<number> {
    const cpuLocator = this.getNode(nodeName).getByTestId('node-cpu-usage');
    const text = await cpuLocator.textContent();
    const match = text?.match(/(\d+(\.\d+)?)/);
    return match ? parseFloat(match[1]) : 0;
  }

  /**
   * Get node memory usage
   * @param nodeName - Node name
   * @returns Memory usage percentage
   */
  async getNodeMemoryUsage(nodeName: string): Promise<number> {
    const memoryLocator = this.getNode(nodeName).getByTestId('node-memory-usage');
    const text = await memoryLocator.textContent();
    const match = text?.match(/(\d+(\.\d+)?)/);
    return match ? parseFloat(match[1]) : 0;
  }

  /**
   * Perform node operation
   * @param nodeName - Node name
   * @param operation - Operation to perform
   * @param confirm - Whether to confirm operation
   * @returns NodeManagementPage instance for chaining
   */
  async performOperation(
    nodeName: string,
    operation: NodeOperation,
    confirm: boolean = true
  ): Promise<this> {
    const node = this.getNode(nodeName);
    await node.getByTestId('node-actions-menu').click();
    await this.page.getByTestId(`node-action-${operation}`).click();

    if (confirm) {
      await this.page.getByTestId('confirm-operation-button').click();
      await this.waitForToast(`Node ${operation} initiated`);
    } else {
      await this.page.getByTestId('cancel-operation-button').click();
    }

    return this;
  }

  /**
   * Enable node
   * @param nodeName - Node name
   * @returns NodeManagementPage instance for chaining
   */
  async enableNode(nodeName: string): Promise<this> {
    return await this.performOperation(nodeName, 'enable');
  }

  /**
   * Disable node
   * @param nodeName - Node name
   * @returns NodeManagementPage instance for chaining
   */
  async disableNode(nodeName: string): Promise<this> {
    return await this.performOperation(nodeName, 'disable');
  }

  /**
   * Drain node (move VMs off node)
   * @param nodeName - Node name
   * @returns NodeManagementPage instance for chaining
   */
  async drainNode(nodeName: string): Promise<this> {
    return await this.performOperation(nodeName, 'drain');
  }

  /**
   * Cordon node (prevent new VMs)
   * @param nodeName - Node name
   * @returns NodeManagementPage instance for chaining
   */
  async cordonNode(nodeName: string): Promise<this> {
    return await this.performOperation(nodeName, 'cordon');
  }

  /**
   * Uncordon node (allow new VMs)
   * @param nodeName - Node name
   * @returns NodeManagementPage instance for chaining
   */
  async uncordonNode(nodeName: string): Promise<this> {
    return await this.performOperation(nodeName, 'uncordon');
  }

  /**
   * Reboot node
   * @param nodeName - Node name
   * @param confirm - Whether to confirm reboot
   * @returns NodeManagementPage instance for chaining
   */
  async rebootNode(nodeName: string, confirm: boolean = true): Promise<this> {
    return await this.performOperation(nodeName, 'reboot', confirm);
  }

  /**
   * Remove node from cluster
   * @param nodeName - Node name
   * @param confirm - Whether to confirm removal
   * @returns NodeManagementPage instance for chaining
   */
  async removeNode(nodeName: string, confirm: boolean = true): Promise<this> {
    return await this.performOperation(nodeName, 'remove', confirm);
  }

  /**
   * Select node checkbox
   * @param nodeName - Node name
   * @returns NodeManagementPage instance for chaining
   */
  async selectNode(nodeName: string): Promise<this> {
    const checkbox = this.getNode(nodeName).getByTestId('node-checkbox');
    await checkbox.check();
    return this;
  }

  /**
   * Select multiple nodes
   * @param nodeNames - Array of node names
   * @returns NodeManagementPage instance for chaining
   */
  async selectNodes(nodeNames: string[]): Promise<this> {
    for (const nodeName of nodeNames) {
      await this.selectNode(nodeName);
    }
    return this;
  }

  /**
   * Select all nodes
   * @returns NodeManagementPage instance for chaining
   */
  async selectAll(): Promise<this> {
    await this.selectAllCheckbox.check();
    return this;
  }

  /**
   * Perform bulk action
   * @param action - Action to perform
   * @param confirm - Whether to confirm action
   * @returns NodeManagementPage instance for chaining
   */
  async performBulkAction(action: NodeOperation, confirm: boolean = true): Promise<this> {
    await this.bulkActionsDropdown.click();
    await this.page.getByTestId(`bulk-action-${action}`).click();

    if (confirm) {
      await this.page.getByTestId('confirm-bulk-action-button').click();
      await this.waitForToast(`Bulk ${action} completed`);
    } else {
      await this.page.getByTestId('cancel-bulk-action-button').click();
    }

    return this;
  }

  /**
   * Get total node count
   * @returns Total number of nodes
   */
  async getNodeCount(): Promise<number> {
    return await this.nodeCards.count();
  }

  /**
   * Check if node exists
   * @param nodeName - Node name
   * @returns True if node exists
   */
  async nodeExists(nodeName: string): Promise<boolean> {
    return await this.getNode(nodeName).isVisible();
  }

  /**
   * Wait for node to appear
   * @param nodeName - Node name
   * @param timeout - Maximum time to wait
   * @returns NodeManagementPage instance for chaining
   */
  async waitForNode(nodeName: string, timeout: number = 10000): Promise<this> {
    await this.getNode(nodeName).waitFor({ state: 'visible', timeout });
    return this;
  }

  /**
   * Wait for node status to change
   * @param nodeName - Node name
   * @param expectedStatus - Expected status
   * @param timeout - Maximum time to wait
   * @returns NodeManagementPage instance for chaining
   */
  async waitForNodeStatus(
    nodeName: string,
    expectedStatus: NodeStatus,
    timeout: number = 30000
  ): Promise<this> {
    await this.page.waitForFunction(
      ({ nodeName, status }) => {
        const statusElement = document.querySelector(
          `[data-testid="node-card-${nodeName}"] [data-testid="node-status"]`
        );
        return statusElement?.textContent?.toLowerCase() === status.toLowerCase();
      },
      { nodeName, status: expectedStatus },
      { timeout }
    );
    return this;
  }

  /**
   * Go to next page
   * @returns NodeManagementPage instance for chaining
   */
  async nextPage(): Promise<this> {
    await this.nextPageButton.click();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Go to previous page
   * @returns NodeManagementPage instance for chaining
   */
  async prevPage(): Promise<this> {
    await this.prevPageButton.click();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Check if empty state is visible
   * @returns True if empty state is visible
   */
  async isEmptyStateVisible(): Promise<boolean> {
    return await this.emptyState.isVisible();
  }
}
