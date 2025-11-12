import { Page, Locator } from '@playwright/test';
import { BasePage } from '../base-page';

/**
 * VM filter options
 */
export interface VMFilterOptions {
  status?: 'running' | 'stopped' | 'paused' | 'error';
  cluster?: string;
  tag?: string;
  search?: string;
}

/**
 * VM sort options
 */
export type VMSortField = 'name' | 'status' | 'cpu' | 'memory' | 'created' | 'updated';
export type SortDirection = 'asc' | 'desc';

/**
 * VM List Page Object Model
 * Handles VM listing, filtering, sorting, and bulk operations
 */
export class VMListPage extends BasePage {
  // Header controls
  private readonly createVMButton: Locator;
  private readonly refreshButton: Locator;
  private readonly bulkActionsDropdown: Locator;
  private readonly viewToggle: Locator;

  // Search and filters
  private readonly searchInput: Locator;
  private readonly statusFilter: Locator;
  private readonly clusterFilter: Locator;
  private readonly tagFilter: Locator;
  private readonly clearFiltersButton: Locator;

  // Sort controls
  private readonly sortDropdown: Locator;

  // VM list
  private readonly vmList: Locator;
  private readonly vmCards: Locator;
  private readonly vmTable: Locator;
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
    this.createVMButton = this.page.getByTestId('vm-list-create-button');
    this.refreshButton = this.page.getByTestId('vm-list-refresh-button');
    this.bulkActionsDropdown = this.page.getByTestId('vm-list-bulk-actions');
    this.viewToggle = this.page.getByTestId('vm-list-view-toggle');

    // Search and filters
    this.searchInput = this.page.getByTestId('vm-list-search-input');
    this.statusFilter = this.page.getByTestId('vm-list-status-filter');
    this.clusterFilter = this.page.getByTestId('vm-list-cluster-filter');
    this.tagFilter = this.page.getByTestId('vm-list-tag-filter');
    this.clearFiltersButton = this.page.getByTestId('vm-list-clear-filters');

    // Sort
    this.sortDropdown = this.page.getByTestId('vm-list-sort-dropdown');

    // VM list
    this.vmList = this.page.getByTestId('vm-list-container');
    this.vmCards = this.page.getByTestId('vm-card');
    this.vmTable = this.page.getByTestId('vm-table');
    this.emptyState = this.page.getByTestId('vm-list-empty-state');

    // Pagination
    this.paginationContainer = this.page.getByTestId('vm-list-pagination');
    this.prevPageButton = this.page.getByTestId('vm-list-prev-page');
    this.nextPageButton = this.page.getByTestId('vm-list-next-page');
    this.pageInfo = this.page.getByTestId('vm-list-page-info');

    // Bulk selection
    this.selectAllCheckbox = this.page.getByTestId('vm-list-select-all');
  }

  /**
   * Navigate to VM list page
   * @returns VMListPage instance for chaining
   */
  async navigate(): Promise<this> {
    await this.goto('/vms');
    await this.waitForLoad();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Click create VM button
   * @returns VMListPage instance for chaining
   */
  async clickCreateVM(): Promise<this> {
    await this.createVMButton.click();
    await this.page.waitForURL('**/vms/create');
    return this;
  }

  /**
   * Refresh VM list
   * @returns VMListPage instance for chaining
   */
  async refresh(): Promise<this> {
    await this.refreshButton.click();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Search for VMs
   * @param query - Search query
   * @returns VMListPage instance for chaining
   */
  async search(query: string): Promise<this> {
    await this.searchInput.fill(query);
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Apply filters to VM list
   * @param filters - Filter options
   * @returns VMListPage instance for chaining
   */
  async applyFilters(filters: VMFilterOptions): Promise<this> {
    if (filters.status) {
      await this.statusFilter.selectOption(filters.status);
    }
    if (filters.cluster) {
      await this.clusterFilter.selectOption(filters.cluster);
    }
    if (filters.tag) {
      await this.tagFilter.selectOption(filters.tag);
    }
    if (filters.search) {
      await this.search(filters.search);
    }
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Clear all filters
   * @returns VMListPage instance for chaining
   */
  async clearFilters(): Promise<this> {
    await this.clearFiltersButton.click();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Sort VM list
   * @param field - Field to sort by
   * @param direction - Sort direction
   * @returns VMListPage instance for chaining
   */
  async sortBy(field: VMSortField, direction: SortDirection = 'asc'): Promise<this> {
    await this.sortDropdown.selectOption(`${field}-${direction}`);
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Toggle between card and table view
   * @param view - View type ('card' or 'table')
   * @returns VMListPage instance for chaining
   */
  async toggleView(view: 'card' | 'table'): Promise<this> {
    const currentView = await this.viewToggle.getAttribute('data-view');
    if (currentView !== view) {
      await this.viewToggle.click();
    }
    return this;
  }

  /**
   * Get VM by name
   * @param vmName - VM name
   * @returns Locator for the VM card/row
   */
  getVM(vmName: string): Locator {
    return this.page.getByTestId(`vm-item-${vmName}`);
  }

  /**
   * Click on a VM to view details
   * @param vmName - VM name
   * @returns VMListPage instance for chaining
   */
  async clickVM(vmName: string): Promise<this> {
    await this.getVM(vmName).click();
    await this.page.waitForURL(`**/vms/${vmName}`);
    return this;
  }

  /**
   * Get VM status
   * @param vmName - VM name
   * @returns VM status
   */
  async getVMStatus(vmName: string): Promise<string> {
    const statusLocator = this.getVM(vmName).getByTestId('vm-status');
    return (await statusLocator.textContent()) || '';
  }

  /**
   * Start a VM
   * @param vmName - VM name
   * @returns VMListPage instance for chaining
   */
  async startVM(vmName: string): Promise<this> {
    const vmItem = this.getVM(vmName);
    await vmItem.getByTestId('vm-action-start').click();
    await this.waitForToast('VM started successfully');
    return this;
  }

  /**
   * Stop a VM
   * @param vmName - VM name
   * @returns VMListPage instance for chaining
   */
  async stopVM(vmName: string): Promise<this> {
    const vmItem = this.getVM(vmName);
    await vmItem.getByTestId('vm-action-stop').click();
    await this.waitForToast('VM stopped successfully');
    return this;
  }

  /**
   * Restart a VM
   * @param vmName - VM name
   * @returns VMListPage instance for chaining
   */
  async restartVM(vmName: string): Promise<this> {
    const vmItem = this.getVM(vmName);
    await vmItem.getByTestId('vm-action-restart').click();
    await this.waitForToast('VM restarted successfully');
    return this;
  }

  /**
   * Delete a VM
   * @param vmName - VM name
   * @param confirm - Whether to confirm deletion
   * @returns VMListPage instance for chaining
   */
  async deleteVM(vmName: string, confirm: boolean = true): Promise<this> {
    const vmItem = this.getVM(vmName);
    await vmItem.getByTestId('vm-action-delete').click();

    if (confirm) {
      await this.page.getByTestId('confirm-delete-button').click();
      await this.waitForToast('VM deleted successfully');
    } else {
      await this.page.getByTestId('cancel-delete-button').click();
    }
    return this;
  }

  /**
   * Select VM checkbox
   * @param vmName - VM name
   * @returns VMListPage instance for chaining
   */
  async selectVM(vmName: string): Promise<this> {
    const checkbox = this.getVM(vmName).getByTestId('vm-checkbox');
    await checkbox.check();
    return this;
  }

  /**
   * Select multiple VMs
   * @param vmNames - Array of VM names
   * @returns VMListPage instance for chaining
   */
  async selectVMs(vmNames: string[]): Promise<this> {
    for (const vmName of vmNames) {
      await this.selectVM(vmName);
    }
    return this;
  }

  /**
   * Select all VMs
   * @returns VMListPage instance for chaining
   */
  async selectAll(): Promise<this> {
    await this.selectAllCheckbox.check();
    return this;
  }

  /**
   * Deselect all VMs
   * @returns VMListPage instance for chaining
   */
  async deselectAll(): Promise<this> {
    await this.selectAllCheckbox.uncheck();
    return this;
  }

  /**
   * Perform bulk action
   * @param action - Action to perform (start, stop, delete)
   * @param confirm - Whether to confirm action
   * @returns VMListPage instance for chaining
   */
  async performBulkAction(
    action: 'start' | 'stop' | 'restart' | 'delete',
    confirm: boolean = true
  ): Promise<this> {
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
   * Get total VM count
   * @returns Total number of VMs
   */
  async getVMCount(): Promise<number> {
    const count = await this.vmCards.count();
    return count;
  }

  /**
   * Get selected VM count
   * @returns Number of selected VMs
   */
  async getSelectedCount(): Promise<number> {
    const selectedText = await this.page
      .getByTestId('vm-list-selected-count')
      .textContent();
    const match = selectedText?.match(/(\d+)/);
    return match ? parseInt(match[1]) : 0;
  }

  /**
   * Check if empty state is visible
   * @returns True if empty state is visible
   */
  async isEmptyStateVisible(): Promise<boolean> {
    return await this.emptyState.isVisible();
  }

  /**
   * Go to next page
   * @returns VMListPage instance for chaining
   */
  async nextPage(): Promise<this> {
    await this.nextPageButton.click();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Go to previous page
   * @returns VMListPage instance for chaining
   */
  async prevPage(): Promise<this> {
    await this.prevPageButton.click();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Get current page info
   * @returns Page info text (e.g., "1-10 of 50")
   */
  async getPageInfo(): Promise<string> {
    return (await this.pageInfo.textContent()) || '';
  }

  /**
   * Check if VM exists in list
   * @param vmName - VM name
   * @returns True if VM exists
   */
  async vmExists(vmName: string): Promise<boolean> {
    return await this.getVM(vmName).isVisible();
  }

  /**
   * Wait for VM to appear in list
   * @param vmName - VM name
   * @param timeout - Maximum time to wait
   * @returns VMListPage instance for chaining
   */
  async waitForVM(vmName: string, timeout: number = 10000): Promise<this> {
    await this.getVM(vmName).waitFor({ state: 'visible', timeout });
    return this;
  }

  /**
   * Wait for VM status to change
   * @param vmName - VM name
   * @param expectedStatus - Expected status
   * @param timeout - Maximum time to wait
   * @returns VMListPage instance for chaining
   */
  async waitForVMStatus(
    vmName: string,
    expectedStatus: string,
    timeout: number = 30000
  ): Promise<this> {
    await this.page.waitForFunction(
      ({ vmName, status }) => {
        const statusElement = document.querySelector(
          `[data-testid="vm-item-${vmName}"] [data-testid="vm-status"]`
        );
        return statusElement?.textContent?.toLowerCase() === status.toLowerCase();
      },
      { vmName, status: expectedStatus },
      { timeout }
    );
    return this;
  }
}
