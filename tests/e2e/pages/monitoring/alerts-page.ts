import { Page, Locator } from '@playwright/test';
import { BasePage } from '../base-page';

/**
 * Alert severity type
 */
export type AlertSeverity = 'info' | 'warning' | 'critical';

/**
 * Alert status type
 */
export type AlertStatus = 'active' | 'acknowledged' | 'resolved';

/**
 * Alert rule configuration
 */
export interface AlertRuleConfig {
  name: string;
  metric: string;
  condition: 'greater_than' | 'less_than' | 'equals';
  threshold: number;
  duration: number;
  severity: AlertSeverity;
  notificationChannels?: string[];
}

/**
 * Alerts Page Object Model
 * Handles alert management, rules, and notifications
 */
export class AlertsPage extends BasePage {
  // Header
  private readonly pageTitle: Locator;
  private readonly createRuleButton: Locator;
  private readonly refreshButton: Locator;
  private readonly bulkActionsDropdown: Locator;

  // Filters
  private readonly severityFilter: Locator;
  private readonly statusFilter: Locator;
  private readonly searchInput: Locator;
  private readonly clearFiltersButton: Locator;

  // Alerts list
  private readonly alertsList: Locator;
  private readonly alertCards: Locator;
  private readonly emptyState: Locator;

  // Tabs
  private readonly activeTab: Locator;
  private readonly acknowledgedTab: Locator;
  private readonly resolvedTab: Locator;
  private readonly rulesTab: Locator;

  // Create rule dialog
  private readonly createRuleDialog: Locator;
  private readonly ruleNameInput: Locator;
  private readonly metricSelect: Locator;
  private readonly conditionSelect: Locator;
  private readonly thresholdInput: Locator;
  private readonly durationInput: Locator;
  private readonly severitySelect: Locator;
  private readonly notificationChannelsSelect: Locator;
  private readonly saveRuleButton: Locator;

  // Alert details
  private readonly detailsPanel: Locator;
  private readonly acknowledgeButton: Locator;
  private readonly resolveButton: Locator;
  private readonly muteButton: Locator;

  // Bulk selection
  private readonly selectAllCheckbox: Locator;

  // Pagination
  private readonly paginationContainer: Locator;
  private readonly prevPageButton: Locator;
  private readonly nextPageButton: Locator;

  constructor(page: Page) {
    super(page);

    // Header
    this.pageTitle = this.page.getByTestId('alerts-page-title');
    this.createRuleButton = this.page.getByTestId('alerts-create-rule-button');
    this.refreshButton = this.page.getByTestId('alerts-refresh-button');
    this.bulkActionsDropdown = this.page.getByTestId('alerts-bulk-actions');

    // Filters
    this.severityFilter = this.page.getByTestId('alerts-severity-filter');
    this.statusFilter = this.page.getByTestId('alerts-status-filter');
    this.searchInput = this.page.getByTestId('alerts-search-input');
    this.clearFiltersButton = this.page.getByTestId('alerts-clear-filters');

    // Alerts list
    this.alertsList = this.page.getByTestId('alerts-list');
    this.alertCards = this.page.getByTestId('alert-card');
    this.emptyState = this.page.getByTestId('alerts-empty-state');

    // Tabs
    this.activeTab = this.page.getByTestId('alerts-tab-active');
    this.acknowledgedTab = this.page.getByTestId('alerts-tab-acknowledged');
    this.resolvedTab = this.page.getByTestId('alerts-tab-resolved');
    this.rulesTab = this.page.getByTestId('alerts-tab-rules');

    // Create rule dialog
    this.createRuleDialog = this.page.getByTestId('alerts-create-rule-dialog');
    this.ruleNameInput = this.page.getByTestId('alert-rule-name-input');
    this.metricSelect = this.page.getByTestId('alert-rule-metric-select');
    this.conditionSelect = this.page.getByTestId('alert-rule-condition-select');
    this.thresholdInput = this.page.getByTestId('alert-rule-threshold-input');
    this.durationInput = this.page.getByTestId('alert-rule-duration-input');
    this.severitySelect = this.page.getByTestId('alert-rule-severity-select');
    this.notificationChannelsSelect = this.page.getByTestId('alert-rule-channels-select');
    this.saveRuleButton = this.page.getByTestId('alert-rule-save-button');

    // Alert details
    this.detailsPanel = this.page.getByTestId('alert-details-panel');
    this.acknowledgeButton = this.page.getByTestId('alert-acknowledge-button');
    this.resolveButton = this.page.getByTestId('alert-resolve-button');
    this.muteButton = this.page.getByTestId('alert-mute-button');

    // Bulk selection
    this.selectAllCheckbox = this.page.getByTestId('alerts-select-all');

    // Pagination
    this.paginationContainer = this.page.getByTestId('alerts-pagination');
    this.prevPageButton = this.page.getByTestId('alerts-prev-page');
    this.nextPageButton = this.page.getByTestId('alerts-next-page');
  }

  /**
   * Navigate to alerts page
   * @returns AlertsPage instance for chaining
   */
  async navigate(): Promise<this> {
    await this.goto('/monitoring/alerts');
    await this.waitForLoad();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Switch to tab
   * @param tab - Tab name
   * @returns AlertsPage instance for chaining
   */
  async switchToTab(tab: 'active' | 'acknowledged' | 'resolved' | 'rules'): Promise<this> {
    await this.page.getByTestId(`alerts-tab-${tab}`).click();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Filter by severity
   * @param severity - Alert severity
   * @returns AlertsPage instance for chaining
   */
  async filterBySeverity(severity: AlertSeverity): Promise<this> {
    await this.severityFilter.selectOption(severity);
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Filter by status
   * @param status - Alert status
   * @returns AlertsPage instance for chaining
   */
  async filterByStatus(status: AlertStatus): Promise<this> {
    await this.statusFilter.selectOption(status);
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Search alerts
   * @param query - Search query
   * @returns AlertsPage instance for chaining
   */
  async search(query: string): Promise<this> {
    await this.searchInput.fill(query);
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Clear filters
   * @returns AlertsPage instance for chaining
   */
  async clearFilters(): Promise<this> {
    await this.clearFiltersButton.click();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Get alert by ID
   * @param alertId - Alert ID
   * @returns Locator for the alert
   */
  getAlert(alertId: string): Locator {
    return this.page.getByTestId(`alert-card-${alertId}`);
  }

  /**
   * Click on alert to view details
   * @param alertId - Alert ID
   * @returns AlertsPage instance for chaining
   */
  async clickAlert(alertId: string): Promise<this> {
    await this.getAlert(alertId).click();
    await this.detailsPanel.waitFor({ state: 'visible' });
    return this;
  }

  /**
   * Acknowledge alert
   * @param alertId - Alert ID
   * @param note - Acknowledgment note (optional)
   * @returns AlertsPage instance for chaining
   */
  async acknowledgeAlert(alertId: string, note?: string): Promise<this> {
    await this.clickAlert(alertId);
    await this.acknowledgeButton.click();

    if (note) {
      await this.page.getByTestId('acknowledge-note-input').fill(note);
    }

    await this.page.getByTestId('confirm-acknowledge').click();
    await this.waitForToast('Alert acknowledged');
    return this;
  }

  /**
   * Resolve alert
   * @param alertId - Alert ID
   * @param note - Resolution note (optional)
   * @returns AlertsPage instance for chaining
   */
  async resolveAlert(alertId: string, note?: string): Promise<this> {
    await this.clickAlert(alertId);
    await this.resolveButton.click();

    if (note) {
      await this.page.getByTestId('resolve-note-input').fill(note);
    }

    await this.page.getByTestId('confirm-resolve').click();
    await this.waitForToast('Alert resolved');
    return this;
  }

  /**
   * Mute alert
   * @param alertId - Alert ID
   * @param duration - Mute duration in minutes
   * @returns AlertsPage instance for chaining
   */
  async muteAlert(alertId: string, duration: number): Promise<this> {
    await this.clickAlert(alertId);
    await this.muteButton.click();
    await this.page.getByTestId('mute-duration-input').fill(duration.toString());
    await this.page.getByTestId('confirm-mute').click();
    await this.waitForToast('Alert muted');
    return this;
  }

  /**
   * Select alert checkbox
   * @param alertId - Alert ID
   * @returns AlertsPage instance for chaining
   */
  async selectAlert(alertId: string): Promise<this> {
    const checkbox = this.getAlert(alertId).getByTestId('alert-checkbox');
    await checkbox.check();
    return this;
  }

  /**
   * Select all alerts
   * @returns AlertsPage instance for chaining
   */
  async selectAll(): Promise<this> {
    await this.selectAllCheckbox.check();
    return this;
  }

  /**
   * Perform bulk action
   * @param action - Action to perform
   * @param confirm - Whether to confirm action
   * @returns AlertsPage instance for chaining
   */
  async performBulkAction(
    action: 'acknowledge' | 'resolve' | 'mute' | 'delete',
    confirm: boolean = true
  ): Promise<this> {
    await this.bulkActionsDropdown.click();
    await this.page.getByTestId(`bulk-action-${action}`).click();

    if (confirm) {
      await this.page.getByTestId('confirm-bulk-action').click();
      await this.waitForToast(`Bulk ${action} completed`);
    } else {
      await this.page.getByTestId('cancel-bulk-action').click();
    }

    return this;
  }

  /**
   * Click create rule button
   * @returns AlertsPage instance for chaining
   */
  async clickCreateRule(): Promise<this> {
    await this.createRuleButton.click();
    await this.createRuleDialog.waitFor({ state: 'visible' });
    return this;
  }

  /**
   * Create alert rule
   * @param config - Alert rule configuration
   * @returns AlertsPage instance for chaining
   */
  async createRule(config: AlertRuleConfig): Promise<this> {
    await this.clickCreateRule();

    await this.ruleNameInput.fill(config.name);
    await this.metricSelect.selectOption(config.metric);
    await this.conditionSelect.selectOption(config.condition);
    await this.thresholdInput.fill(config.threshold.toString());
    await this.durationInput.fill(config.duration.toString());
    await this.severitySelect.selectOption(config.severity);

    if (config.notificationChannels) {
      for (const channel of config.notificationChannels) {
        await this.notificationChannelsSelect.selectOption(channel);
      }
    }

    await this.saveRuleButton.click();
    await this.waitForToast('Alert rule created');
    return this;
  }

  /**
   * Get alert rule by name
   * @param ruleName - Rule name
   * @returns Locator for the rule
   */
  getRule(ruleName: string): Locator {
    return this.page.getByTestId(`alert-rule-${ruleName}`);
  }

  /**
   * Edit alert rule
   * @param ruleName - Rule name
   * @param updates - Fields to update
   * @returns AlertsPage instance for chaining
   */
  async editRule(ruleName: string, updates: Partial<AlertRuleConfig>): Promise<this> {
    await this.switchToTab('rules');
    const rule = this.getRule(ruleName);
    await rule.getByTestId('rule-edit-button').click();

    if (updates.threshold !== undefined) {
      await this.thresholdInput.clear();
      await this.thresholdInput.fill(updates.threshold.toString());
    }

    if (updates.severity) {
      await this.severitySelect.selectOption(updates.severity);
    }

    await this.saveRuleButton.click();
    await this.waitForToast('Alert rule updated');
    return this;
  }

  /**
   * Delete alert rule
   * @param ruleName - Rule name
   * @param confirm - Whether to confirm deletion
   * @returns AlertsPage instance for chaining
   */
  async deleteRule(ruleName: string, confirm: boolean = true): Promise<this> {
    await this.switchToTab('rules');
    const rule = this.getRule(ruleName);
    await rule.getByTestId('rule-delete-button').click();

    if (confirm) {
      await this.page.getByTestId('confirm-delete-rule').click();
      await this.waitForToast('Alert rule deleted');
    } else {
      await this.page.getByTestId('cancel-delete-rule').click();
    }

    return this;
  }

  /**
   * Get alerts count
   * @returns Total number of alerts
   */
  async getAlertsCount(): Promise<number> {
    return await this.alertCards.count();
  }

  /**
   * Refresh alerts
   * @returns AlertsPage instance for chaining
   */
  async refresh(): Promise<this> {
    await this.refreshButton.click();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Go to next page
   * @returns AlertsPage instance for chaining
   */
  async nextPage(): Promise<this> {
    await this.nextPageButton.click();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Go to previous page
   * @returns AlertsPage instance for chaining
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
