import { Page, Locator } from '@playwright/test';
import { BasePage } from '../base-page';

/**
 * Migration status type
 */
export type MigrationStatus =
  | 'pending'
  | 'preparing'
  | 'in_progress'
  | 'completing'
  | 'completed'
  | 'failed'
  | 'cancelled';

/**
 * Migration filter options
 */
export interface MigrationFilterOptions {
  status?: MigrationStatus;
  sourceCluster?: string;
  targetCluster?: string;
  dateRange?: string;
}

/**
 * Migration Status Page Object Model
 * Handles migration monitoring and status tracking
 */
export class MigrationStatusPage extends BasePage {
  // Header
  private readonly pageTitle: Locator;
  private readonly newMigrationButton: Locator;
  private readonly refreshButton: Locator;
  private readonly autoRefreshToggle: Locator;

  // Filters
  private readonly statusFilter: Locator;
  private readonly sourceClusterFilter: Locator;
  private readonly targetClusterFilter: Locator;
  private readonly dateRangeFilter: Locator;
  private readonly searchInput: Locator;
  private readonly clearFiltersButton: Locator;

  // Migrations list
  private readonly migrationsList: Locator;
  private readonly migrationCards: Locator;
  private readonly emptyState: Locator;

  // Active migrations panel
  private readonly activeMigrationsPanel: Locator;
  private readonly activeMigrationsCount: Locator;

  // Migration details
  private readonly detailsPanel: Locator;
  private readonly migrationProgress: Locator;
  private readonly progressPercentage: Locator;
  private readonly currentPhase: Locator;
  private readonly elapsedTime: Locator;
  private readonly remainingTime: Locator;
  private readonly transferredData: Locator;
  private readonly transferRate: Locator;

  // Actions
  private readonly pauseButton: Locator;
  private readonly resumeButton: Locator;
  private readonly cancelButton: Locator;
  private readonly rollbackButton: Locator;

  // Migration logs
  private readonly logsPanel: Locator;
  private readonly logsContent: Locator;
  private readonly downloadLogsButton: Locator;

  // Statistics
  private readonly statsPanel: Locator;
  private readonly totalMigrations: Locator;
  private readonly successfulMigrations: Locator;
  private readonly failedMigrations: Locator;
  private readonly avgDuration: Locator;

  // Pagination
  private readonly paginationContainer: Locator;
  private readonly prevPageButton: Locator;
  private readonly nextPageButton: Locator;

  constructor(page: Page) {
    super(page);

    // Header
    this.pageTitle = this.page.getByTestId('migration-status-title');
    this.newMigrationButton = this.page.getByTestId('migration-status-new-button');
    this.refreshButton = this.page.getByTestId('migration-status-refresh-button');
    this.autoRefreshToggle = this.page.getByTestId('migration-status-auto-refresh-toggle');

    // Filters
    this.statusFilter = this.page.getByTestId('migration-status-filter');
    this.sourceClusterFilter = this.page.getByTestId('migration-source-cluster-filter');
    this.targetClusterFilter = this.page.getByTestId('migration-target-cluster-filter');
    this.dateRangeFilter = this.page.getByTestId('migration-date-range-filter');
    this.searchInput = this.page.getByTestId('migration-search-input');
    this.clearFiltersButton = this.page.getByTestId('migration-clear-filters');

    // Migrations list
    this.migrationsList = this.page.getByTestId('migrations-list');
    this.migrationCards = this.page.getByTestId('migration-card');
    this.emptyState = this.page.getByTestId('migrations-empty-state');

    // Active migrations
    this.activeMigrationsPanel = this.page.getByTestId('active-migrations-panel');
    this.activeMigrationsCount = this.page.getByTestId('active-migrations-count');

    // Details
    this.detailsPanel = this.page.getByTestId('migration-details-panel');
    this.migrationProgress = this.page.getByTestId('migration-progress');
    this.progressPercentage = this.page.getByTestId('migration-progress-percentage');
    this.currentPhase = this.page.getByTestId('migration-current-phase');
    this.elapsedTime = this.page.getByTestId('migration-elapsed-time');
    this.remainingTime = this.page.getByTestId('migration-remaining-time');
    this.transferredData = this.page.getByTestId('migration-transferred-data');
    this.transferRate = this.page.getByTestId('migration-transfer-rate');

    // Actions
    this.pauseButton = this.page.getByTestId('migration-pause-button');
    this.resumeButton = this.page.getByTestId('migration-resume-button');
    this.cancelButton = this.page.getByTestId('migration-cancel-button');
    this.rollbackButton = this.page.getByTestId('migration-rollback-button');

    // Logs
    this.logsPanel = this.page.getByTestId('migration-logs-panel');
    this.logsContent = this.page.getByTestId('migration-logs-content');
    this.downloadLogsButton = this.page.getByTestId('migration-download-logs-button');

    // Statistics
    this.statsPanel = this.page.getByTestId('migration-stats-panel');
    this.totalMigrations = this.page.getByTestId('migration-stats-total');
    this.successfulMigrations = this.page.getByTestId('migration-stats-successful');
    this.failedMigrations = this.page.getByTestId('migration-stats-failed');
    this.avgDuration = this.page.getByTestId('migration-stats-avg-duration');

    // Pagination
    this.paginationContainer = this.page.getByTestId('migration-pagination');
    this.prevPageButton = this.page.getByTestId('migration-prev-page');
    this.nextPageButton = this.page.getByTestId('migration-next-page');
  }

  /**
   * Navigate to migration status page
   * @returns MigrationStatusPage instance for chaining
   */
  async navigate(): Promise<this> {
    await this.goto('/migration');
    await this.waitForLoad();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Click new migration button
   * @returns MigrationStatusPage instance for chaining
   */
  async clickNewMigration(): Promise<this> {
    await this.newMigrationButton.click();
    await this.page.waitForURL('**/migration/new');
    return this;
  }

  /**
   * Refresh migrations list
   * @returns MigrationStatusPage instance for chaining
   */
  async refresh(): Promise<this> {
    await this.refreshButton.click();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Toggle auto-refresh
   * @param enabled - Whether to enable auto-refresh
   * @returns MigrationStatusPage instance for chaining
   */
  async toggleAutoRefresh(enabled: boolean = true): Promise<this> {
    const isChecked = await this.autoRefreshToggle.isChecked();

    if (enabled && !isChecked) {
      await this.autoRefreshToggle.check();
    } else if (!enabled && isChecked) {
      await this.autoRefreshToggle.uncheck();
    }

    return this;
  }

  /**
   * Apply filters
   * @param filters - Filter options
   * @returns MigrationStatusPage instance for chaining
   */
  async applyFilters(filters: MigrationFilterOptions): Promise<this> {
    if (filters.status) {
      await this.statusFilter.selectOption(filters.status);
    }

    if (filters.sourceCluster) {
      await this.sourceClusterFilter.selectOption(filters.sourceCluster);
    }

    if (filters.targetCluster) {
      await this.targetClusterFilter.selectOption(filters.targetCluster);
    }

    if (filters.dateRange) {
      await this.dateRangeFilter.selectOption(filters.dateRange);
    }

    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Search migrations
   * @param query - Search query
   * @returns MigrationStatusPage instance for chaining
   */
  async search(query: string): Promise<this> {
    await this.searchInput.fill(query);
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Clear filters
   * @returns MigrationStatusPage instance for chaining
   */
  async clearFilters(): Promise<this> {
    await this.clearFiltersButton.click();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Get migration by ID
   * @param migrationId - Migration ID
   * @returns Locator for the migration
   */
  getMigration(migrationId: string): Locator {
    return this.page.getByTestId(`migration-card-${migrationId}`);
  }

  /**
   * Click on migration to view details
   * @param migrationId - Migration ID
   * @returns MigrationStatusPage instance for chaining
   */
  async clickMigration(migrationId: string): Promise<this> {
    await this.getMigration(migrationId).click();
    await this.detailsPanel.waitFor({ state: 'visible' });
    return this;
  }

  /**
   * Get migration status
   * @param migrationId - Migration ID
   * @returns Migration status
   */
  async getMigrationStatus(migrationId: string): Promise<MigrationStatus> {
    const statusLocator = this.getMigration(migrationId).getByTestId('migration-status');
    const status = await statusLocator.getAttribute('data-status');
    return (status as MigrationStatus) || 'pending';
  }

  /**
   * Get progress percentage
   * @param migrationId - Migration ID
   * @returns Progress percentage (0-100)
   */
  async getProgressPercentage(migrationId: string): Promise<number> {
    await this.clickMigration(migrationId);
    const text = await this.progressPercentage.textContent();
    const match = text?.match(/(\d+)/);
    return match ? parseInt(match[1]) : 0;
  }

  /**
   * Get current phase
   * @param migrationId - Migration ID
   * @returns Current migration phase
   */
  async getCurrentPhase(migrationId: string): Promise<string> {
    await this.clickMigration(migrationId);
    return (await this.currentPhase.textContent()) || '';
  }

  /**
   * Pause migration
   * @param migrationId - Migration ID
   * @param confirm - Whether to confirm pause
   * @returns MigrationStatusPage instance for chaining
   */
  async pauseMigration(migrationId: string, confirm: boolean = true): Promise<this> {
    await this.clickMigration(migrationId);
    await this.pauseButton.click();

    if (confirm) {
      await this.page.getByTestId('confirm-pause-migration').click();
      await this.waitForToast('Migration paused');
    } else {
      await this.page.getByTestId('cancel-pause-migration').click();
    }

    return this;
  }

  /**
   * Resume migration
   * @param migrationId - Migration ID
   * @returns MigrationStatusPage instance for chaining
   */
  async resumeMigration(migrationId: string): Promise<this> {
    await this.clickMigration(migrationId);
    await this.resumeButton.click();
    await this.waitForToast('Migration resumed');
    return this;
  }

  /**
   * Cancel migration
   * @param migrationId - Migration ID
   * @param confirm - Whether to confirm cancellation
   * @returns MigrationStatusPage instance for chaining
   */
  async cancelMigration(migrationId: string, confirm: boolean = true): Promise<this> {
    await this.clickMigration(migrationId);
    await this.cancelButton.click();

    if (confirm) {
      await this.page.getByTestId('confirm-cancel-migration').click();
      await this.waitForToast('Migration cancelled');
    } else {
      await this.page.getByTestId('dismiss-cancel-migration').click();
    }

    return this;
  }

  /**
   * Rollback migration
   * @param migrationId - Migration ID
   * @param confirm - Whether to confirm rollback
   * @returns MigrationStatusPage instance for chaining
   */
  async rollbackMigration(migrationId: string, confirm: boolean = true): Promise<this> {
    await this.clickMigration(migrationId);
    await this.rollbackButton.click();

    if (confirm) {
      await this.page.getByTestId('confirm-rollback-migration').click();
      await this.waitForToast('Migration rollback initiated');
    } else {
      await this.page.getByTestId('cancel-rollback-migration').click();
    }

    return this;
  }

  /**
   * Download migration logs
   * @param migrationId - Migration ID
   * @returns MigrationStatusPage instance for chaining
   */
  async downloadLogs(migrationId: string): Promise<this> {
    await this.clickMigration(migrationId);
    await this.downloadLogsButton.click();
    return this;
  }

  /**
   * Get migration logs
   * @param migrationId - Migration ID
   * @returns Logs content
   */
  async getLogs(migrationId: string): Promise<string> {
    await this.clickMigration(migrationId);
    return (await this.logsContent.textContent()) || '';
  }

  /**
   * Get active migrations count
   * @returns Number of active migrations
   */
  async getActiveMigrationsCount(): Promise<number> {
    const text = await this.activeMigrationsCount.textContent();
    return parseInt(text || '0');
  }

  /**
   * Wait for migration to complete
   * @param migrationId - Migration ID
   * @param timeout - Maximum time to wait
   * @returns MigrationStatusPage instance for chaining
   */
  async waitForMigrationComplete(
    migrationId: string,
    timeout: number = 300000
  ): Promise<this> {
    await this.page.waitForFunction(
      ({ migrationId }) => {
        const statusElement = document.querySelector(
          `[data-testid="migration-card-${migrationId}"] [data-testid="migration-status"]`
        );
        const status = statusElement?.getAttribute('data-status');
        return status === 'completed' || status === 'failed' || status === 'cancelled';
      },
      { migrationId },
      { timeout }
    );
    return this;
  }

  /**
   * Get total migrations
   * @returns Total migrations count
   */
  async getTotalMigrations(): Promise<number> {
    const text = await this.totalMigrations.textContent();
    return parseInt(text || '0');
  }

  /**
   * Get successful migrations
   * @returns Successful migrations count
   */
  async getSuccessfulMigrations(): Promise<number> {
    const text = await this.successfulMigrations.textContent();
    return parseInt(text || '0');
  }

  /**
   * Get failed migrations
   * @returns Failed migrations count
   */
  async getFailedMigrations(): Promise<number> {
    const text = await this.failedMigrations.textContent();
    return parseInt(text || '0');
  }

  /**
   * Go to next page
   * @returns MigrationStatusPage instance for chaining
   */
  async nextPage(): Promise<this> {
    await this.nextPageButton.click();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Go to previous page
   * @returns MigrationStatusPage instance for chaining
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
