import { Page, Locator } from '@playwright/test';
import { BasePage } from '../base-page';

/**
 * Metric type
 */
export type MetricType = 'cpu' | 'memory' | 'storage' | 'network' | 'custom';

/**
 * Chart type
 */
export type ChartType = 'line' | 'area' | 'bar' | 'gauge' | 'heatmap';

/**
 * Metrics Page Object Model
 * Handles detailed metrics viewing and analysis
 */
export class MetricsPage extends BasePage {
  // Header
  private readonly pageTitle: Locator;
  private readonly metricTypeSelect: Locator;
  private readonly timeRangeSelect: Locator;
  private readonly exportButton: Locator;

  // Filters
  private readonly resourceFilter: Locator;
  private readonly clusterFilter: Locator;
  private readonly nodeFilter: Locator;
  private readonly vmFilter: Locator;

  // Main chart
  private readonly mainChart: Locator;
  private readonly chartTypeSelect: Locator;
  private readonly zoomInButton: Locator;
  private readonly zoomOutButton: Locator;
  private readonly resetZoomButton: Locator;

  // Metrics list
  private readonly metricsList: Locator;
  private readonly metricsSearch: Locator;
  private readonly metricsSort: Locator;

  // Statistics
  private readonly statsPanel: Locator;
  private readonly avgValue: Locator;
  private readonly minValue: Locator;
  private readonly maxValue: Locator;
  private readonly p95Value: Locator;
  private readonly p99Value: Locator;

  // Query builder
  private readonly queryBuilderToggle: Locator;
  private readonly queryInput: Locator;
  private readonly runQueryButton: Locator;

  // Compare mode
  private readonly compareModeToggle: Locator;
  private readonly addComparisonButton: Locator;

  constructor(page: Page) {
    super(page);

    // Header
    this.pageTitle = this.page.getByTestId('metrics-page-title');
    this.metricTypeSelect = this.page.getByTestId('metrics-type-select');
    this.timeRangeSelect = this.page.getByTestId('metrics-time-range-select');
    this.exportButton = this.page.getByTestId('metrics-export-button');

    // Filters
    this.resourceFilter = this.page.getByTestId('metrics-resource-filter');
    this.clusterFilter = this.page.getByTestId('metrics-cluster-filter');
    this.nodeFilter = this.page.getByTestId('metrics-node-filter');
    this.vmFilter = this.page.getByTestId('metrics-vm-filter');

    // Main chart
    this.mainChart = this.page.getByTestId('metrics-main-chart');
    this.chartTypeSelect = this.page.getByTestId('metrics-chart-type-select');
    this.zoomInButton = this.page.getByTestId('metrics-zoom-in');
    this.zoomOutButton = this.page.getByTestId('metrics-zoom-out');
    this.resetZoomButton = this.page.getByTestId('metrics-reset-zoom');

    // Metrics list
    this.metricsList = this.page.getByTestId('metrics-list');
    this.metricsSearch = this.page.getByTestId('metrics-search-input');
    this.metricsSort = this.page.getByTestId('metrics-sort-select');

    // Statistics
    this.statsPanel = this.page.getByTestId('metrics-stats-panel');
    this.avgValue = this.page.getByTestId('metrics-avg-value');
    this.minValue = this.page.getByTestId('metrics-min-value');
    this.maxValue = this.page.getByTestId('metrics-max-value');
    this.p95Value = this.page.getByTestId('metrics-p95-value');
    this.p99Value = this.page.getByTestId('metrics-p99-value');

    // Query builder
    this.queryBuilderToggle = this.page.getByTestId('metrics-query-builder-toggle');
    this.queryInput = this.page.getByTestId('metrics-query-input');
    this.runQueryButton = this.page.getByTestId('metrics-run-query-button');

    // Compare mode
    this.compareModeToggle = this.page.getByTestId('metrics-compare-toggle');
    this.addComparisonButton = this.page.getByTestId('metrics-add-comparison');
  }

  /**
   * Navigate to metrics page
   * @returns MetricsPage instance for chaining
   */
  async navigate(): Promise<this> {
    await this.goto('/monitoring/metrics');
    await this.waitForLoad();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Select metric type
   * @param type - Metric type
   * @returns MetricsPage instance for chaining
   */
  async selectMetricType(type: MetricType): Promise<this> {
    await this.metricTypeSelect.selectOption(type);
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Set time range
   * @param range - Time range
   * @returns MetricsPage instance for chaining
   */
  async setTimeRange(range: string): Promise<this> {
    await this.timeRangeSelect.selectOption(range);
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Filter by resource
   * @param resource - Resource name
   * @returns MetricsPage instance for chaining
   */
  async filterByResource(resource: string): Promise<this> {
    await this.resourceFilter.selectOption(resource);
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Filter by cluster
   * @param cluster - Cluster name
   * @returns MetricsPage instance for chaining
   */
  async filterByCluster(cluster: string): Promise<this> {
    await this.clusterFilter.selectOption(cluster);
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Filter by node
   * @param node - Node name
   * @returns MetricsPage instance for chaining
   */
  async filterByNode(node: string): Promise<this> {
    await this.nodeFilter.selectOption(node);
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Filter by VM
   * @param vm - VM name
   * @returns MetricsPage instance for chaining
   */
  async filterByVM(vm: string): Promise<this> {
    await this.vmFilter.selectOption(vm);
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Change chart type
   * @param type - Chart type
   * @returns MetricsPage instance for chaining
   */
  async setChartType(type: ChartType): Promise<this> {
    await this.chartTypeSelect.selectOption(type);
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Zoom in on chart
   * @returns MetricsPage instance for chaining
   */
  async zoomIn(): Promise<this> {
    await this.zoomInButton.click();
    return this;
  }

  /**
   * Zoom out on chart
   * @returns MetricsPage instance for chaining
   */
  async zoomOut(): Promise<this> {
    await this.zoomOutButton.click();
    return this;
  }

  /**
   * Reset chart zoom
   * @returns MetricsPage instance for chaining
   */
  async resetZoom(): Promise<this> {
    await this.resetZoomButton.click();
    return this;
  }

  /**
   * Search metrics
   * @param query - Search query
   * @returns MetricsPage instance for chaining
   */
  async searchMetrics(query: string): Promise<this> {
    await this.metricsSearch.fill(query);
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Sort metrics
   * @param sortBy - Sort field
   * @returns MetricsPage instance for chaining
   */
  async sortMetrics(sortBy: string): Promise<this> {
    await this.metricsSort.selectOption(sortBy);
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Get average value
   * @returns Average value
   */
  async getAvgValue(): Promise<number> {
    const text = await this.avgValue.textContent();
    const match = text?.match(/(\d+(\.\d+)?)/);
    return match ? parseFloat(match[1]) : 0;
  }

  /**
   * Get minimum value
   * @returns Minimum value
   */
  async getMinValue(): Promise<number> {
    const text = await this.minValue.textContent();
    const match = text?.match(/(\d+(\.\d+)?)/);
    return match ? parseFloat(match[1]) : 0;
  }

  /**
   * Get maximum value
   * @returns Maximum value
   */
  async getMaxValue(): Promise<number> {
    const text = await this.maxValue.textContent();
    const match = text?.match(/(\d+(\.\d+)?)/);
    return match ? parseFloat(match[1]) : 0;
  }

  /**
   * Get 95th percentile value
   * @returns P95 value
   */
  async getP95Value(): Promise<number> {
    const text = await this.p95Value.textContent();
    const match = text?.match(/(\d+(\.\d+)?)/);
    return match ? parseFloat(match[1]) : 0;
  }

  /**
   * Get 99th percentile value
   * @returns P99 value
   */
  async getP99Value(): Promise<number> {
    const text = await this.p99Value.textContent();
    const match = text?.match(/(\d+(\.\d+)?)/);
    return match ? parseFloat(match[1]) : 0;
  }

  /**
   * Toggle query builder
   * @param show - Whether to show query builder
   * @returns MetricsPage instance for chaining
   */
  async toggleQueryBuilder(show: boolean = true): Promise<this> {
    const isVisible = await this.queryInput.isVisible();

    if (show && !isVisible) {
      await this.queryBuilderToggle.click();
    } else if (!show && isVisible) {
      await this.queryBuilderToggle.click();
    }

    return this;
  }

  /**
   * Run custom query
   * @param query - Query string
   * @returns MetricsPage instance for chaining
   */
  async runQuery(query: string): Promise<this> {
    await this.toggleQueryBuilder(true);
    await this.queryInput.fill(query);
    await this.runQueryButton.click();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Toggle compare mode
   * @param enabled - Whether to enable compare mode
   * @returns MetricsPage instance for chaining
   */
  async toggleCompareMode(enabled: boolean = true): Promise<this> {
    const isChecked = await this.compareModeToggle.isChecked();

    if (enabled && !isChecked) {
      await this.compareModeToggle.check();
    } else if (!enabled && isChecked) {
      await this.compareModeToggle.uncheck();
    }

    return this;
  }

  /**
   * Add comparison
   * @param resource - Resource to compare
   * @returns MetricsPage instance for chaining
   */
  async addComparison(resource: string): Promise<this> {
    await this.toggleCompareMode(true);
    await this.addComparisonButton.click();
    await this.page.getByTestId('comparison-resource-select').selectOption(resource);
    await this.page.getByTestId('confirm-comparison').click();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Export metrics data
   * @param format - Export format (csv, json, png)
   * @returns MetricsPage instance for chaining
   */
  async exportMetrics(format: 'csv' | 'json' | 'png'): Promise<this> {
    await this.exportButton.click();
    await this.page.getByTestId(`export-format-${format}`).click();
    await this.waitForToast('Export completed');
    return this;
  }

  /**
   * Check if main chart is visible
   * @returns True if chart is visible
   */
  async isChartVisible(): Promise<boolean> {
    return await this.mainChart.isVisible();
  }

  /**
   * Check if stats panel is visible
   * @returns True if stats panel is visible
   */
  async isStatsPanelVisible(): Promise<boolean> {
    return await this.statsPanel.isVisible();
  }
}
