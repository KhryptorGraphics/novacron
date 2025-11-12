import { Page, Locator } from '@playwright/test';
import { BasePage } from '../base-page';

/**
 * VM operation type
 */
export type VMOperation = 'start' | 'stop' | 'restart' | 'pause' | 'resume' | 'delete' | 'clone' | 'migrate';

/**
 * VM tab type
 */
export type VMTab = 'overview' | 'metrics' | 'console' | 'storage' | 'network' | 'snapshots' | 'events' | 'settings';

/**
 * VM Details Page Object Model
 * Handles VM details view and operations
 */
export class VMDetailsPage extends BasePage {
  // Header
  private readonly vmName: Locator;
  private readonly vmStatus: Locator;
  private readonly backButton: Locator;
  private readonly actionsMenu: Locator;

  // Tabs
  private readonly tabContainer: Locator;

  // Overview Tab
  private readonly cpuUsage: Locator;
  private readonly memoryUsage: Locator;
  private readonly diskUsage: Locator;
  private readonly networkUsage: Locator;
  private readonly uptime: Locator;
  private readonly ipAddress: Locator;
  private readonly clusterInfo: Locator;

  // Metrics Tab
  private readonly metricsChart: Locator;
  private readonly metricsTimeRange: Locator;
  private readonly metricsRefresh: Locator;

  // Storage Tab
  private readonly diskList: Locator;
  private readonly addDiskButton: Locator;
  private readonly resizeDiskButton: Locator;

  // Network Tab
  private readonly networkInterfaceList: Locator;
  private readonly addNetworkButton: Locator;

  // Snapshots Tab
  private readonly snapshotList: Locator;
  private readonly createSnapshotButton: Locator;

  // Events Tab
  private readonly eventsList: Locator;
  private readonly eventsFilter: Locator;

  // Settings Tab
  private readonly settingsForm: Locator;
  private readonly saveSettingsButton: Locator;

  // Operations
  private readonly confirmDialog: Locator;

  constructor(page: Page) {
    super(page);

    // Header
    this.vmName = this.page.getByTestId('vm-details-name');
    this.vmStatus = this.page.getByTestId('vm-details-status');
    this.backButton = this.page.getByTestId('vm-details-back-button');
    this.actionsMenu = this.page.getByTestId('vm-details-actions-menu');

    // Tabs
    this.tabContainer = this.page.getByTestId('vm-details-tabs');

    // Overview Tab
    this.cpuUsage = this.page.getByTestId('vm-overview-cpu-usage');
    this.memoryUsage = this.page.getByTestId('vm-overview-memory-usage');
    this.diskUsage = this.page.getByTestId('vm-overview-disk-usage');
    this.networkUsage = this.page.getByTestId('vm-overview-network-usage');
    this.uptime = this.page.getByTestId('vm-overview-uptime');
    this.ipAddress = this.page.getByTestId('vm-overview-ip-address');
    this.clusterInfo = this.page.getByTestId('vm-overview-cluster-info');

    // Metrics Tab
    this.metricsChart = this.page.getByTestId('vm-metrics-chart');
    this.metricsTimeRange = this.page.getByTestId('vm-metrics-time-range');
    this.metricsRefresh = this.page.getByTestId('vm-metrics-refresh');

    // Storage Tab
    this.diskList = this.page.getByTestId('vm-storage-disk-list');
    this.addDiskButton = this.page.getByTestId('vm-storage-add-disk-button');
    this.resizeDiskButton = this.page.getByTestId('vm-storage-resize-disk-button');

    // Network Tab
    this.networkInterfaceList = this.page.getByTestId('vm-network-interface-list');
    this.addNetworkButton = this.page.getByTestId('vm-network-add-button');

    // Snapshots Tab
    this.snapshotList = this.page.getByTestId('vm-snapshots-list');
    this.createSnapshotButton = this.page.getByTestId('vm-snapshots-create-button');

    // Events Tab
    this.eventsList = this.page.getByTestId('vm-events-list');
    this.eventsFilter = this.page.getByTestId('vm-events-filter');

    // Settings Tab
    this.settingsForm = this.page.getByTestId('vm-settings-form');
    this.saveSettingsButton = this.page.getByTestId('vm-settings-save-button');

    // Operations
    this.confirmDialog = this.page.getByTestId('confirm-dialog');
  }

  /**
   * Navigate to VM details page
   * @param vmId - VM ID or name
   * @returns VMDetailsPage instance for chaining
   */
  async navigate(vmId: string): Promise<this> {
    await this.goto(`/vms/${vmId}`);
    await this.waitForLoad();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Click back button to return to VM list
   * @returns VMDetailsPage instance for chaining
   */
  async clickBack(): Promise<this> {
    await this.backButton.click();
    await this.page.waitForURL('**/vms');
    return this;
  }

  /**
   * Switch to a specific tab
   * @param tab - Tab name
   * @returns VMDetailsPage instance for chaining
   */
  async switchToTab(tab: VMTab): Promise<this> {
    await this.page.getByTestId(`vm-details-tab-${tab}`).click();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Get VM name
   * @returns VM name
   */
  async getVMName(): Promise<string> {
    return (await this.vmName.textContent()) || '';
  }

  /**
   * Get VM status
   * @returns VM status
   */
  async getVMStatus(): Promise<string> {
    return (await this.vmStatus.textContent()) || '';
  }

  /**
   * Perform VM operation
   * @param operation - Operation type
   * @param confirm - Whether to confirm operation
   * @returns VMDetailsPage instance for chaining
   */
  async performOperation(operation: VMOperation, confirm: boolean = true): Promise<this> {
    await this.actionsMenu.click();
    await this.page.getByTestId(`vm-action-${operation}`).click();

    if (confirm) {
      await this.page.getByTestId('confirm-operation-button').click();
      await this.waitForToast(`VM ${operation} initiated`);
    } else {
      await this.page.getByTestId('cancel-operation-button').click();
    }

    return this;
  }

  /**
   * Start VM
   * @returns VMDetailsPage instance for chaining
   */
  async startVM(): Promise<this> {
    return await this.performOperation('start');
  }

  /**
   * Stop VM
   * @returns VMDetailsPage instance for chaining
   */
  async stopVM(): Promise<this> {
    return await this.performOperation('stop');
  }

  /**
   * Restart VM
   * @returns VMDetailsPage instance for chaining
   */
  async restartVM(): Promise<this> {
    return await this.performOperation('restart');
  }

  /**
   * Delete VM
   * @param confirm - Whether to confirm deletion
   * @returns VMDetailsPage instance for chaining
   */
  async deleteVM(confirm: boolean = true): Promise<this> {
    return await this.performOperation('delete', confirm);
  }

  /**
   * Clone VM
   * @param newName - Name for cloned VM
   * @returns VMDetailsPage instance for chaining
   */
  async cloneVM(newName: string): Promise<this> {
    await this.actionsMenu.click();
    await this.page.getByTestId('vm-action-clone').click();
    await this.page.getByTestId('clone-vm-name-input').fill(newName);
    await this.page.getByTestId('confirm-clone-button').click();
    await this.waitForToast('VM cloned successfully');
    return this;
  }

  /**
   * Migrate VM
   * @param targetCluster - Target cluster name
   * @returns VMDetailsPage instance for chaining
   */
  async migrateVM(targetCluster: string): Promise<this> {
    await this.actionsMenu.click();
    await this.page.getByTestId('vm-action-migrate').click();
    await this.page.getByTestId('migrate-target-cluster-select').selectOption(targetCluster);
    await this.page.getByTestId('confirm-migrate-button').click();
    await this.waitForToast('VM migration initiated');
    return this;
  }

  /**
   * Get CPU usage percentage
   * @returns CPU usage percentage
   */
  async getCPUUsage(): Promise<number> {
    const usageText = await this.cpuUsage.textContent();
    const match = usageText?.match(/(\d+(\.\d+)?)/);
    return match ? parseFloat(match[1]) : 0;
  }

  /**
   * Get memory usage percentage
   * @returns Memory usage percentage
   */
  async getMemoryUsage(): Promise<number> {
    const usageText = await this.memoryUsage.textContent();
    const match = usageText?.match(/(\d+(\.\d+)?)/);
    return match ? parseFloat(match[1]) : 0;
  }

  /**
   * Get disk usage percentage
   * @returns Disk usage percentage
   */
  async getDiskUsage(): Promise<number> {
    const usageText = await this.diskUsage.textContent();
    const match = usageText?.match(/(\d+(\.\d+)?)/);
    return match ? parseFloat(match[1]) : 0;
  }

  /**
   * Get VM IP address
   * @returns IP address
   */
  async getIPAddress(): Promise<string> {
    return (await this.ipAddress.textContent()) || '';
  }

  /**
   * Get VM uptime
   * @returns Uptime text
   */
  async getUptime(): Promise<string> {
    return (await this.uptime.textContent()) || '';
  }

  /**
   * Create snapshot
   * @param snapshotName - Snapshot name
   * @param description - Snapshot description (optional)
   * @returns VMDetailsPage instance for chaining
   */
  async createSnapshot(snapshotName: string, description?: string): Promise<this> {
    await this.switchToTab('snapshots');
    await this.createSnapshotButton.click();
    await this.page.getByTestId('snapshot-name-input').fill(snapshotName);

    if (description) {
      await this.page.getByTestId('snapshot-description-input').fill(description);
    }

    await this.page.getByTestId('confirm-snapshot-button').click();
    await this.waitForToast('Snapshot created successfully');
    return this;
  }

  /**
   * Restore from snapshot
   * @param snapshotName - Snapshot name
   * @param confirm - Whether to confirm restoration
   * @returns VMDetailsPage instance for chaining
   */
  async restoreSnapshot(snapshotName: string, confirm: boolean = true): Promise<this> {
    await this.switchToTab('snapshots');
    const snapshot = this.page.getByTestId(`snapshot-${snapshotName}`);
    await snapshot.getByTestId('snapshot-restore-button').click();

    if (confirm) {
      await this.page.getByTestId('confirm-restore-button').click();
      await this.waitForToast('Snapshot restored successfully');
    } else {
      await this.page.getByTestId('cancel-restore-button').click();
    }

    return this;
  }

  /**
   * Delete snapshot
   * @param snapshotName - Snapshot name
   * @param confirm - Whether to confirm deletion
   * @returns VMDetailsPage instance for chaining
   */
  async deleteSnapshot(snapshotName: string, confirm: boolean = true): Promise<this> {
    await this.switchToTab('snapshots');
    const snapshot = this.page.getByTestId(`snapshot-${snapshotName}`);
    await snapshot.getByTestId('snapshot-delete-button').click();

    if (confirm) {
      await this.page.getByTestId('confirm-delete-button').click();
      await this.waitForToast('Snapshot deleted successfully');
    } else {
      await this.page.getByTestId('cancel-delete-button').click();
    }

    return this;
  }

  /**
   * Add disk to VM
   * @param diskSize - Disk size in GB
   * @param storageProfile - Storage profile (optional)
   * @returns VMDetailsPage instance for chaining
   */
  async addDisk(diskSize: number, storageProfile?: string): Promise<this> {
    await this.switchToTab('storage');
    await this.addDiskButton.click();
    await this.page.getByTestId('add-disk-size-input').fill(diskSize.toString());

    if (storageProfile) {
      await this.page.getByTestId('add-disk-storage-profile-select').selectOption(storageProfile);
    }

    await this.page.getByTestId('confirm-add-disk-button').click();
    await this.waitForToast('Disk added successfully');
    return this;
  }

  /**
   * Add network interface
   * @param networkName - Network name
   * @returns VMDetailsPage instance for chaining
   */
  async addNetworkInterface(networkName: string): Promise<this> {
    await this.switchToTab('network');
    await this.addNetworkButton.click();
    await this.page.getByTestId('add-network-select').selectOption(networkName);
    await this.page.getByTestId('confirm-add-network-button').click();
    await this.waitForToast('Network interface added successfully');
    return this;
  }

  /**
   * Update VM settings
   * @param settings - Settings object (key-value pairs)
   * @returns VMDetailsPage instance for chaining
   */
  async updateSettings(settings: Record<string, any>): Promise<this> {
    await this.switchToTab('settings');

    for (const [key, value] of Object.entries(settings)) {
      const input = this.page.getByTestId(`vm-setting-${key}-input`);
      await input.fill(value.toString());
    }

    await this.saveSettingsButton.click();
    await this.waitForToast('Settings updated successfully');
    return this;
  }

  /**
   * Get events list
   * @param filter - Event type filter (optional)
   * @returns Array of event texts
   */
  async getEvents(filter?: string): Promise<string[]> {
    await this.switchToTab('events');

    if (filter) {
      await this.eventsFilter.selectOption(filter);
      await this.waitForLoadingComplete();
    }

    const events = await this.eventsList.locator('[data-testid="event-item"]').all();
    const eventTexts: string[] = [];

    for (const event of events) {
      const text = await event.textContent();
      if (text) eventTexts.push(text);
    }

    return eventTexts;
  }

  /**
   * Wait for VM status to change
   * @param expectedStatus - Expected status
   * @param timeout - Maximum time to wait
   * @returns VMDetailsPage instance for chaining
   */
  async waitForStatus(expectedStatus: string, timeout: number = 30000): Promise<this> {
    await this.page.waitForFunction(
      (status) => {
        const statusElement = document.querySelector('[data-testid="vm-details-status"]');
        return statusElement?.textContent?.toLowerCase() === status.toLowerCase();
      },
      expectedStatus,
      { timeout }
    );
    return this;
  }

  /**
   * Refresh page data
   * @returns VMDetailsPage instance for chaining
   */
  async refresh(): Promise<this> {
    await this.reload();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Open console
   * @returns VMDetailsPage instance for chaining
   */
  async openConsole(): Promise<this> {
    await this.switchToTab('console');
    return this;
  }
}
