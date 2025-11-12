import { Page, Locator } from '@playwright/test';
import { BasePage } from '../base-page';

/**
 * Federation configuration interface
 */
export interface FederationConfig {
  clusterName: string;
  endpoint: string;
  authToken?: string;
  syncInterval?: number;
  enableWorkloadSharing?: boolean;
}

/**
 * Federation status type
 */
export type FederationStatus = 'connected' | 'disconnected' | 'syncing' | 'error';

/**
 * Federation Page Object Model
 * Handles cluster federation management and operations
 */
export class FederationPage extends BasePage {
  // Header
  private readonly pageTitle: Locator;
  private readonly addFederationButton: Locator;
  private readonly refreshButton: Locator;
  private readonly globalSearchToggle: Locator;

  // Federation list
  private readonly federationList: Locator;
  private readonly federationCards: Locator;
  private readonly emptyState: Locator;

  // Add federation dialog
  private readonly addDialog: Locator;
  private readonly clusterNameInput: Locator;
  private readonly endpointInput: Locator;
  private readonly authTokenInput: Locator;
  private readonly syncIntervalInput: Locator;
  private readonly workloadSharingCheckbox: Locator;
  private readonly confirmAddButton: Locator;
  private readonly cancelAddButton: Locator;

  // Federation details
  private readonly detailsPanel: Locator;
  private readonly federationStatus: Locator;
  private readonly lastSyncTime: Locator;
  private readonly sharedWorkloadsCount: Locator;

  // Workload sharing
  private readonly workloadList: Locator;
  private readonly shareWorkloadButton: Locator;
  private readonly reclaimWorkloadButton: Locator;

  // Settings
  private readonly settingsButton: Locator;
  private readonly settingsPanel: Locator;

  constructor(page: Page) {
    super(page);

    // Header
    this.pageTitle = this.page.getByTestId('federation-page-title');
    this.addFederationButton = this.page.getByTestId('federation-add-button');
    this.refreshButton = this.page.getByTestId('federation-refresh-button');
    this.globalSearchToggle = this.page.getByTestId('federation-global-search-toggle');

    // Federation list
    this.federationList = this.page.getByTestId('federation-list');
    this.federationCards = this.page.getByTestId('federation-card');
    this.emptyState = this.page.getByTestId('federation-empty-state');

    // Add federation dialog
    this.addDialog = this.page.getByTestId('federation-add-dialog');
    this.clusterNameInput = this.page.getByTestId('federation-cluster-name-input');
    this.endpointInput = this.page.getByTestId('federation-endpoint-input');
    this.authTokenInput = this.page.getByTestId('federation-auth-token-input');
    this.syncIntervalInput = this.page.getByTestId('federation-sync-interval-input');
    this.workloadSharingCheckbox = this.page.getByTestId('federation-workload-sharing-checkbox');
    this.confirmAddButton = this.page.getByTestId('federation-confirm-add-button');
    this.cancelAddButton = this.page.getByTestId('federation-cancel-add-button');

    // Details
    this.detailsPanel = this.page.getByTestId('federation-details-panel');
    this.federationStatus = this.page.getByTestId('federation-status');
    this.lastSyncTime = this.page.getByTestId('federation-last-sync');
    this.sharedWorkloadsCount = this.page.getByTestId('federation-shared-workloads');

    // Workload sharing
    this.workloadList = this.page.getByTestId('federation-workload-list');
    this.shareWorkloadButton = this.page.getByTestId('federation-share-workload-button');
    this.reclaimWorkloadButton = this.page.getByTestId('federation-reclaim-workload-button');

    // Settings
    this.settingsButton = this.page.getByTestId('federation-settings-button');
    this.settingsPanel = this.page.getByTestId('federation-settings-panel');
  }

  /**
   * Navigate to federation page
   * @returns FederationPage instance for chaining
   */
  async navigate(): Promise<this> {
    await this.goto('/cluster/federation');
    await this.waitForLoad();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Click add federation button
   * @returns FederationPage instance for chaining
   */
  async clickAddFederation(): Promise<this> {
    await this.addFederationButton.click();
    await this.addDialog.waitFor({ state: 'visible' });
    return this;
  }

  /**
   * Add federated cluster
   * @param config - Federation configuration
   * @returns FederationPage instance for chaining
   */
  async addFederation(config: FederationConfig): Promise<this> {
    await this.clickAddFederation();

    await this.clusterNameInput.fill(config.clusterName);
    await this.endpointInput.fill(config.endpoint);

    if (config.authToken) {
      await this.authTokenInput.fill(config.authToken);
    }

    if (config.syncInterval) {
      await this.syncIntervalInput.fill(config.syncInterval.toString());
    }

    if (config.enableWorkloadSharing) {
      await this.workloadSharingCheckbox.check();
    }

    await this.confirmAddButton.click();
    await this.waitForToast('Federation added successfully');
    return this;
  }

  /**
   * Cancel adding federation
   * @returns FederationPage instance for chaining
   */
  async cancelAddFederation(): Promise<this> {
    await this.cancelAddButton.click();
    await this.addDialog.waitFor({ state: 'hidden' });
    return this;
  }

  /**
   * Get federation by cluster name
   * @param clusterName - Cluster name
   * @returns Locator for the federation card
   */
  getFederation(clusterName: string): Locator {
    return this.page.getByTestId(`federation-card-${clusterName}`);
  }

  /**
   * Click on a federation to view details
   * @param clusterName - Cluster name
   * @returns FederationPage instance for chaining
   */
  async clickFederation(clusterName: string): Promise<this> {
    await this.getFederation(clusterName).click();
    await this.detailsPanel.waitFor({ state: 'visible' });
    return this;
  }

  /**
   * Get federation status
   * @param clusterName - Cluster name
   * @returns Federation status
   */
  async getFederationStatus(clusterName: string): Promise<FederationStatus> {
    const statusLocator = this.getFederation(clusterName).getByTestId('federation-status');
    const status = await statusLocator.getAttribute('data-status');
    return (status as FederationStatus) || 'disconnected';
  }

  /**
   * Remove federated cluster
   * @param clusterName - Cluster name
   * @param confirm - Whether to confirm removal
   * @returns FederationPage instance for chaining
   */
  async removeFederation(clusterName: string, confirm: boolean = true): Promise<this> {
    const federation = this.getFederation(clusterName);
    await federation.getByTestId('federation-remove-button').click();

    if (confirm) {
      await this.page.getByTestId('confirm-remove-button').click();
      await this.waitForToast('Federation removed successfully');
    } else {
      await this.page.getByTestId('cancel-remove-button').click();
    }

    return this;
  }

  /**
   * Sync with federated cluster
   * @param clusterName - Cluster name
   * @returns FederationPage instance for chaining
   */
  async syncFederation(clusterName: string): Promise<this> {
    const federation = this.getFederation(clusterName);
    await federation.getByTestId('federation-sync-button').click();
    await this.waitForToast('Sync initiated');
    return this;
  }

  /**
   * Enable workload sharing
   * @param clusterName - Cluster name
   * @returns FederationPage instance for chaining
   */
  async enableWorkloadSharing(clusterName: string): Promise<this> {
    await this.clickFederation(clusterName);
    const toggle = this.detailsPanel.getByTestId('workload-sharing-toggle');
    await toggle.check();
    await this.waitForToast('Workload sharing enabled');
    return this;
  }

  /**
   * Disable workload sharing
   * @param clusterName - Cluster name
   * @returns FederationPage instance for chaining
   */
  async disableWorkloadSharing(clusterName: string): Promise<this> {
    await this.clickFederation(clusterName);
    const toggle = this.detailsPanel.getByTestId('workload-sharing-toggle');
    await toggle.uncheck();
    await this.waitForToast('Workload sharing disabled');
    return this;
  }

  /**
   * Share workload with federated cluster
   * @param workloadId - Workload ID
   * @param targetCluster - Target cluster name
   * @returns FederationPage instance for chaining
   */
  async shareWorkload(workloadId: string, targetCluster: string): Promise<this> {
    await this.shareWorkloadButton.click();
    await this.page.getByTestId('share-workload-id-input').fill(workloadId);
    await this.page.getByTestId('share-target-cluster-select').selectOption(targetCluster);
    await this.page.getByTestId('confirm-share-button').click();
    await this.waitForToast('Workload shared successfully');
    return this;
  }

  /**
   * Reclaim shared workload
   * @param workloadId - Workload ID
   * @param confirm - Whether to confirm reclaim
   * @returns FederationPage instance for chaining
   */
  async reclaimWorkload(workloadId: string, confirm: boolean = true): Promise<this> {
    const workload = this.workloadList.getByTestId(`workload-${workloadId}`);
    await workload.getByTestId('workload-reclaim-button').click();

    if (confirm) {
      await this.page.getByTestId('confirm-reclaim-button').click();
      await this.waitForToast('Workload reclaimed successfully');
    } else {
      await this.page.getByTestId('cancel-reclaim-button').click();
    }

    return this;
  }

  /**
   * Toggle global search
   * @param enabled - Whether to enable global search
   * @returns FederationPage instance for chaining
   */
  async toggleGlobalSearch(enabled: boolean = true): Promise<this> {
    const isChecked = await this.globalSearchToggle.isChecked();

    if (enabled && !isChecked) {
      await this.globalSearchToggle.check();
    } else if (!enabled && isChecked) {
      await this.globalSearchToggle.uncheck();
    }

    return this;
  }

  /**
   * Get last sync time for federation
   * @param clusterName - Cluster name
   * @returns Last sync time text
   */
  async getLastSyncTime(clusterName: string): Promise<string> {
    await this.clickFederation(clusterName);
    return (await this.lastSyncTime.textContent()) || '';
  }

  /**
   * Get shared workloads count
   * @param clusterName - Cluster name
   * @returns Number of shared workloads
   */
  async getSharedWorkloadsCount(clusterName: string): Promise<number> {
    await this.clickFederation(clusterName);
    const text = await this.sharedWorkloadsCount.textContent();
    return parseInt(text || '0');
  }

  /**
   * Refresh federation list
   * @returns FederationPage instance for chaining
   */
  async refresh(): Promise<this> {
    await this.refreshButton.click();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Check if federation exists
   * @param clusterName - Cluster name
   * @returns True if federation exists
   */
  async federationExists(clusterName: string): Promise<boolean> {
    return await this.getFederation(clusterName).isVisible();
  }

  /**
   * Wait for federation status to change
   * @param clusterName - Cluster name
   * @param expectedStatus - Expected status
   * @param timeout - Maximum time to wait
   * @returns FederationPage instance for chaining
   */
  async waitForFederationStatus(
    clusterName: string,
    expectedStatus: FederationStatus,
    timeout: number = 30000
  ): Promise<this> {
    await this.page.waitForFunction(
      ({ clusterName, status }) => {
        const statusElement = document.querySelector(
          `[data-testid="federation-card-${clusterName}"] [data-testid="federation-status"]`
        );
        return statusElement?.getAttribute('data-status') === status;
      },
      { clusterName, status: expectedStatus },
      { timeout }
    );
    return this;
  }

  /**
   * Get total federations count
   * @returns Total number of federations
   */
  async getFederationsCount(): Promise<number> {
    return await this.federationCards.count();
  }

  /**
   * Check if empty state is visible
   * @returns True if empty state is visible
   */
  async isEmptyStateVisible(): Promise<boolean> {
    return await this.emptyState.isVisible();
  }

  /**
   * Open settings
   * @returns FederationPage instance for chaining
   */
  async openSettings(): Promise<this> {
    await this.settingsButton.click();
    await this.settingsPanel.waitFor({ state: 'visible' });
    return this;
  }
}
