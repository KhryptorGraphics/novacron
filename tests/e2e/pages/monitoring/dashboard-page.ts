import { Page, Locator } from '@playwright/test';
import { BasePage } from '../base-page';

/**
 * Monitoring time range
 */
export type TimeRange = '15m' | '1h' | '6h' | '24h' | '7d' | '30d' | 'custom';

/**
 * Monitoring Dashboard Page Object Model
 * Handles main monitoring dashboard with metrics overview
 */
export class MonitoringDashboardPage extends BasePage {
  // Header
  private readonly pageTitle: Locator;
  private readonly timeRangeSelect: Locator;
  private readonly refreshButton: Locator;
  private readonly autoRefreshToggle: Locator;

  // Overview metrics
  private readonly systemHealth: Locator;
  private readonly totalAlerts: Locator;
  private readonly criticalAlerts: Locator;
  private readonly avgCPU: Locator;
  private readonly avgMemory: Locator;
  private readonly avgStorage: Locator;

  // Charts
  private readonly cpuChart: Locator;
  private readonly memoryChart: Locator;
  private readonly networkChart: Locator;
  private readonly diskIOChart: Locator;

  // Resource widgets
  private readonly topCPUConsumers: Locator;
  private readonly topMemoryConsumers: Locator;
  private readonly topNetworkConsumers: Locator;

  // Alerts panel
  private readonly alertsPanel: Locator;
  private readonly recentAlerts: Locator;

  // Custom dashboard
  private readonly addWidgetButton: Locator;
  private readonly customizeButton: Locator;
  private readonly resetLayoutButton: Locator;

  constructor(page: Page) {
    super(page);

    // Header
    this.pageTitle = this.page.getByTestId('monitoring-dashboard-title');
    this.timeRangeSelect = this.page.getByTestId('monitoring-time-range-select');
    this.refreshButton = this.page.getByTestId('monitoring-refresh-button');
    this.autoRefreshToggle = this.page.getByTestId('monitoring-auto-refresh-toggle');

    // Overview metrics
    this.systemHealth = this.page.getByTestId('monitoring-system-health');
    this.totalAlerts = this.page.getByTestId('monitoring-total-alerts');
    this.criticalAlerts = this.page.getByTestId('monitoring-critical-alerts');
    this.avgCPU = this.page.getByTestId('monitoring-avg-cpu');
    this.avgMemory = this.page.getByTestId('monitoring-avg-memory');
    this.avgStorage = this.page.getByTestId('monitoring-avg-storage');

    // Charts
    this.cpuChart = this.page.getByTestId('monitoring-cpu-chart');
    this.memoryChart = this.page.getByTestId('monitoring-memory-chart');
    this.networkChart = this.page.getByTestId('monitoring-network-chart');
    this.diskIOChart = this.page.getByTestId('monitoring-disk-io-chart');

    // Resource widgets
    this.topCPUConsumers = this.page.getByTestId('monitoring-top-cpu-consumers');
    this.topMemoryConsumers = this.page.getByTestId('monitoring-top-memory-consumers');
    this.topNetworkConsumers = this.page.getByTestId('monitoring-top-network-consumers');

    // Alerts
    this.alertsPanel = this.page.getByTestId('monitoring-alerts-panel');
    this.recentAlerts = this.page.getByTestId('monitoring-recent-alerts');

    // Customization
    this.addWidgetButton = this.page.getByTestId('monitoring-add-widget-button');
    this.customizeButton = this.page.getByTestId('monitoring-customize-button');
    this.resetLayoutButton = this.page.getByTestId('monitoring-reset-layout-button');
  }

  /**
   * Navigate to monitoring dashboard
   * @returns MonitoringDashboardPage instance for chaining
   */
  async navigate(): Promise<this> {
    await this.goto('/monitoring');
    await this.waitForLoad();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Set time range for metrics
   * @param range - Time range
   * @returns MonitoringDashboardPage instance for chaining
   */
  async setTimeRange(range: TimeRange): Promise<this> {
    await this.timeRangeSelect.selectOption(range);
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Set custom time range
   * @param startDate - Start date
   * @param endDate - End date
   * @returns MonitoringDashboardPage instance for chaining
   */
  async setCustomTimeRange(startDate: string, endDate: string): Promise<this> {
    await this.timeRangeSelect.selectOption('custom');
    await this.page.getByTestId('monitoring-custom-start-date').fill(startDate);
    await this.page.getByTestId('monitoring-custom-end-date').fill(endDate);
    await this.page.getByTestId('monitoring-apply-custom-range').click();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Refresh dashboard data
   * @returns MonitoringDashboardPage instance for chaining
   */
  async refresh(): Promise<this> {
    await this.refreshButton.click();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Toggle auto-refresh
   * @param enabled - Whether to enable auto-refresh
   * @returns MonitoringDashboardPage instance for chaining
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
   * Get system health status
   * @returns Health status
   */
  async getSystemHealth(): Promise<string> {
    return (await this.systemHealth.textContent()) || '';
  }

  /**
   * Get total alerts count
   * @returns Total alerts
   */
  async getTotalAlerts(): Promise<number> {
    const text = await this.totalAlerts.textContent();
    return parseInt(text || '0');
  }

  /**
   * Get critical alerts count
   * @returns Critical alerts
   */
  async getCriticalAlerts(): Promise<number> {
    const text = await this.criticalAlerts.textContent();
    return parseInt(text || '0');
  }

  /**
   * Get average CPU usage
   * @returns Average CPU percentage
   */
  async getAvgCPU(): Promise<number> {
    const text = await this.avgCPU.textContent();
    const match = text?.match(/(\d+(\.\d+)?)/);
    return match ? parseFloat(match[1]) : 0;
  }

  /**
   * Get average memory usage
   * @returns Average memory percentage
   */
  async getAvgMemory(): Promise<number> {
    const text = await this.avgMemory.textContent();
    const match = text?.match(/(\d+(\.\d+)?)/);
    return match ? parseFloat(match[1]) : 0;
  }

  /**
   * Get average storage usage
   * @returns Average storage percentage
   */
  async getAvgStorage(): Promise<number> {
    const text = await this.avgStorage.textContent();
    const match = text?.match(/(\d+(\.\d+)?)/);
    return match ? parseFloat(match[1]) : 0;
  }

  /**
   * Get top CPU consumers
   * @param count - Number of top consumers to return
   * @returns Array of consumer names
   */
  async getTopCPUConsumers(count: number = 5): Promise<string[]> {
    const consumers = await this.topCPUConsumers
      .locator('[data-testid="consumer-item"]')
      .all();
    const names: string[] = [];

    for (let i = 0; i < Math.min(count, consumers.length); i++) {
      const name = await consumers[i].textContent();
      if (name) names.push(name);
    }

    return names;
  }

  /**
   * Get top memory consumers
   * @param count - Number of top consumers to return
   * @returns Array of consumer names
   */
  async getTopMemoryConsumers(count: number = 5): Promise<string[]> {
    const consumers = await this.topMemoryConsumers
      .locator('[data-testid="consumer-item"]')
      .all();
    const names: string[] = [];

    for (let i = 0; i < Math.min(count, consumers.length); i++) {
      const name = await consumers[i].textContent();
      if (name) names.push(name);
    }

    return names;
  }

  /**
   * Get recent alerts
   * @param count - Number of alerts to return
   * @returns Array of alert texts
   */
  async getRecentAlerts(count: number = 5): Promise<string[]> {
    const alerts = await this.recentAlerts.locator('[data-testid="alert-item"]').all();
    const alertTexts: string[] = [];

    for (let i = 0; i < Math.min(count, alerts.length); i++) {
      const text = await alerts[i].textContent();
      if (text) alertTexts.push(text);
    }

    return alertTexts;
  }

  /**
   * Click on alert to view details
   * @param alertIndex - Alert index
   * @returns MonitoringDashboardPage instance for chaining
   */
  async clickAlert(alertIndex: number): Promise<this> {
    const alert = this.recentAlerts.locator('[data-testid="alert-item"]').nth(alertIndex);
    await alert.click();
    await this.page.waitForURL('**/monitoring/alerts/**');
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
   * Check if network chart is visible
   * @returns True if chart is visible
   */
  async isNetworkChartVisible(): Promise<boolean> {
    return await this.networkChart.isVisible();
  }

  /**
   * Check if disk I/O chart is visible
   * @returns True if chart is visible
   */
  async isDiskIOChartVisible(): Promise<boolean> {
    return await this.diskIOChart.isVisible();
  }

  /**
   * Add widget to dashboard
   * @param widgetType - Widget type
   * @returns MonitoringDashboardPage instance for chaining
   */
  async addWidget(widgetType: string): Promise<this> {
    await this.addWidgetButton.click();
    await this.page.getByTestId(`widget-type-${widgetType}`).click();
    await this.waitForToast('Widget added');
    return this;
  }

  /**
   * Enter customize mode
   * @returns MonitoringDashboardPage instance for chaining
   */
  async enterCustomizeMode(): Promise<this> {
    await this.customizeButton.click();
    return this;
  }

  /**
   * Exit customize mode
   * @returns MonitoringDashboardPage instance for chaining
   */
  async exitCustomizeMode(): Promise<this> {
    await this.page.getByTestId('monitoring-exit-customize').click();
    return this;
  }

  /**
   * Save dashboard layout
   * @returns MonitoringDashboardPage instance for chaining
   */
  async saveLayout(): Promise<this> {
    await this.page.getByTestId('monitoring-save-layout').click();
    await this.waitForToast('Layout saved');
    return this;
  }

  /**
   * Reset dashboard layout
   * @param confirm - Whether to confirm reset
   * @returns MonitoringDashboardPage instance for chaining
   */
  async resetLayout(confirm: boolean = true): Promise<this> {
    await this.resetLayoutButton.click();

    if (confirm) {
      await this.page.getByTestId('confirm-reset-layout').click();
      await this.waitForToast('Layout reset');
    } else {
      await this.page.getByTestId('cancel-reset-layout').click();
    }

    return this;
  }

  /**
   * Navigate to metrics page
   * @returns MonitoringDashboardPage instance for chaining
   */
  async goToMetrics(): Promise<this> {
    await this.page.getByTestId('monitoring-nav-metrics').click();
    await this.page.waitForURL('**/monitoring/metrics');
    return this;
  }

  /**
   * Navigate to alerts page
   * @returns MonitoringDashboardPage instance for chaining
   */
  async goToAlerts(): Promise<this> {
    await this.page.getByTestId('monitoring-nav-alerts').click();
    await this.page.waitForURL('**/monitoring/alerts');
    return this;
  }
}
