/**
 * Automated Performance Regression Detection System
 * Monitors performance metrics and detects regressions using statistical analysis
 */

const { PerformanceMonitor } = require('../monitoring/performance-monitor');
const fs = require('fs').promises;
const path = require('path');

class PerformanceRegressionDetector {
  constructor(config = {}) {
    this.config = {
      // Baseline configuration
      baselineMethod: config.baselineMethod || 'rolling_average',
      baselinePeriod: config.baselinePeriod || 604800000, // 7 days in ms
      minimumSamples: config.minimumSamples || 100,
      updateFrequency: config.updateFrequency || 86400000, // 24 hours in ms
      
      // Detection parameters
      sensitivity: config.sensitivity || 'medium',
      minChangeThreshold: config.minChangeThreshold || 10, // 10% minimum change
      statisticalConfidence: config.statisticalConfidence || 0.95,
      consecutiveViolations: config.consecutiveViolations || 3,
      
      // Data storage
      baselineStorage: config.baselineStorage || './tests/performance/regression/baselines.json',
      detectionHistory: config.detectionHistory || './tests/performance/regression/detections.json',
      
      ...config
    };
    
    this.baselines = new Map();
    this.detectionHistory = [];
    this.currentViolations = new Map();
    this.lastBaselineUpdate = new Map();
    
    this.sensitivitySettings = {
      low: { zScoreThreshold: 2.0, changeThreshold: 20 },
      medium: { zScoreThreshold: 1.96, changeThreshold: 15 },
      high: { zScoreThreshold: 1.5, changeThreshold: 10 }
    };
  }

  /**
   * Initialize the regression detection system
   */
  async initialize() {
    console.log('🔍 Initializing Performance Regression Detection System...');
    
    // Load existing baselines
    await this.loadBaselines();
    
    // Load detection history
    await this.loadDetectionHistory();
    
    console.log('✅ Regression detection system initialized');
    console.log(`   Loaded ${this.baselines.size} baselines`);
    console.log(`   Detection history: ${this.detectionHistory.length} entries`);
  }

  /**
   * Analyze current metrics for regressions
   */
  async analyzeMetrics(currentMetrics, timestamp = Date.now()) {
    const regressions = [];
    
    for (const [metricName, metricValue] of Object.entries(currentMetrics)) {
      try {
        const regression = await this.analyzeMetric(metricName, metricValue, timestamp);
        if (regression) {
          regressions.push(regression);
        }
      } catch (error) {
        console.error(`Error analyzing metric ${metricName}:`, error);
      }
    }
    
    // Update violation tracking
    this.updateViolationTracking(regressions);
    
    // Check for persistent violations
    const persistentRegressions = this.checkPersistentViolations();
    
    if (persistentRegressions.length > 0) {
      await this.triggerRegressionAlerts(persistentRegressions);
    }
    
    return {
      timestamp,
      currentMetrics,
      regressions,
      persistentRegressions
    };
  }

  /**
   * Analyze individual metric for regression
   */
  async analyzeMetric(metricName, currentValue, timestamp) {
    // Get or create baseline for this metric
    let baseline = this.baselines.get(metricName);
    
    if (!baseline) {
      baseline = await this.createBaseline(metricName, currentValue, timestamp);
      this.baselines.set(metricName, baseline);
    }
    
    // Check if baseline needs updating
    if (this.shouldUpdateBaseline(metricName, timestamp)) {
      baseline = await this.updateBaseline(metricName, currentValue, timestamp);
    }
    
    // Perform regression analysis
    const analysis = this.performRegressionAnalysis(metricName, currentValue, baseline);
    
    if (analysis.isRegression) {
      return {
        metric: metricName,
        currentValue,
        baseline: baseline.value,
        timestamp,
        ...analysis
      };
    }
    
    return null;
  }

  /**
   * Create initial baseline for a metric
   */
  async createBaseline(metricName, initialValue, timestamp) {
    console.log(`📊 Creating initial baseline for metric: ${metricName}`);
    
    const baseline = {
      metric: metricName,
      value: initialValue,
      standardDeviation: 0,
      sampleCount: 1,
      createdAt: timestamp,
      updatedAt: timestamp,
      history: [{ value: initialValue, timestamp }],
      method: this.config.baselineMethod
    };
    
    await this.saveBaselines();
    return baseline;
  }

  /**
   * Update baseline with new data
   */
  async updateBaseline(metricName, newValue, timestamp) {
    const baseline = this.baselines.get(metricName);
    
    if (!baseline) {
      return await this.createBaseline(metricName, newValue, timestamp);
    }
    
    // Add new data point
    baseline.history.push({ value: newValue, timestamp });
    
    // Keep only data within baseline period
    const cutoff = timestamp - this.config.baselinePeriod;
    baseline.history = baseline.history.filter(h => h.timestamp >= cutoff);
    
    // Recalculate baseline statistics
    const values = baseline.history.map(h => h.value);
    
    baseline.value = this.calculateBaselineValue(values, this.config.baselineMethod);
    baseline.standardDeviation = this.calculateStandardDeviation(values);
    baseline.sampleCount = values.length;
    baseline.updatedAt = timestamp;
    
    this.lastBaselineUpdate.set(metricName, timestamp);
    
    console.log(`📊 Updated baseline for ${metricName}: ${baseline.value.toFixed(2)} ± ${baseline.standardDeviation.toFixed(2)}`);
    
    await this.saveBaselines();
    return baseline;
  }

  /**
   * Calculate baseline value using specified method
   */
  calculateBaselineValue(values, method) {
    switch (method) {
      case 'mean':
        return values.reduce((sum, v) => sum + v, 0) / values.length;
        
      case 'median':
        const sorted = [...values].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2 === 0 ? 
          (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
          
      case 'rolling_average':
        // Exponential moving average with more weight on recent values
        return this.calculateExponentialMovingAverage(values, 0.2);
        
      case 'percentile':
        // Use 95th percentile as baseline
        return this.calculatePercentile(values, 95);
        
      default:
        return this.calculateExponentialMovingAverage(values, 0.1);
    }
  }

  /**
   * Calculate exponential moving average
   */
  calculateExponentialMovingAverage(values, alpha) {
    let ema = values[0];
    for (let i = 1; i < values.length; i++) {
      ema = alpha * values[i] + (1 - alpha) * ema;
    }
    return ema;
  }

  /**
   * Calculate percentile value
   */
  calculatePercentile(values, percentile) {
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.ceil((percentile / 100) * sorted.length) - 1;
    return sorted[Math.max(0, index)];
  }

  /**
   * Calculate standard deviation
   */
  calculateStandardDeviation(values) {
    const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
    const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
    return Math.sqrt(variance);
  }

  /**
   * Perform regression analysis on a metric
   */
  performRegressionAnalysis(metricName, currentValue, baseline) {
    const settings = this.sensitivitySettings[this.config.sensitivity];
    
    // Calculate deviation from baseline
    const absoluteChange = currentValue - baseline.value;
    const percentChange = baseline.value !== 0 ? (absoluteChange / baseline.value) * 100 : 0;
    
    // Determine if change is in the "bad" direction for this metric
    const isBadDirection = this.isBadDirection(metricName, absoluteChange);
    
    // Statistical significance test (Z-score)
    const zScore = baseline.standardDeviation > 0 ? 
      Math.abs(absoluteChange / baseline.standardDeviation) : 0;
    
    // Check regression criteria
    const isStatisticallySignificant = zScore > settings.zScoreThreshold;
    const isMagnitudeSignificant = Math.abs(percentChange) > settings.changeThreshold;
    const isRegression = isBadDirection && isStatisticallySignificant && isMagnitudeSignificant;
    
    return {
      isRegression,
      absoluteChange,
      percentChange,
      zScore,
      isStatisticallySignificant,
      isMagnitudeSignificant,
      severity: this.calculateSeverity(Math.abs(percentChange), zScore),
      confidence: this.calculateConfidence(zScore)
    };
  }

  /**
   * Determine if change is in bad direction for metric
   */
  isBadDirection(metricName, change) {
    // Define which direction is "bad" for each metric type
    const badDirections = {
      // Higher is bad
      response_time: change > 0,
      latency: change > 0,
      error_rate: change > 0,
      cpu_usage: change > 0,
      memory_usage: change > 0,
      connection_count: change > 0,
      
      // Lower is bad  
      throughput: change < 0,
      queries_per_second: change < 0,
      cache_hit_ratio: change < 0,
      availability: change < 0,
      
      // Default: higher values are bad
      default: change > 0
    };
    
    // Check for exact match first
    if (badDirections.hasOwnProperty(metricName)) {
      return badDirections[metricName];
    }
    
    // Check for partial matches
    for (const [pattern, isBad] of Object.entries(badDirections)) {
      if (metricName.includes(pattern)) {
        return isBad;
      }
    }
    
    return badDirections.default;
  }

  /**
   * Calculate regression severity
   */
  calculateSeverity(percentChange, zScore) {
    if (percentChange > 50 || zScore > 4) return 'critical';
    if (percentChange > 25 || zScore > 3) return 'high';
    if (percentChange > 10 || zScore > 2) return 'medium';
    return 'low';
  }

  /**
   * Calculate confidence in regression detection
   */
  calculateConfidence(zScore) {
    if (zScore > 3) return 0.999;
    if (zScore > 2.58) return 0.99;
    if (zScore > 1.96) return 0.95;
    if (zScore > 1.65) return 0.90;
    return Math.max(0.5, 0.5 + (zScore / 4));
  }

  /**
   * Check if baseline should be updated
   */
  shouldUpdateBaseline(metricName, timestamp) {
    const lastUpdate = this.lastBaselineUpdate.get(metricName) || 0;
    return timestamp - lastUpdate >= this.config.updateFrequency;
  }

  /**
   * Update violation tracking for persistent regression detection
   */
  updateViolationTracking(regressions) {
    const currentTime = Date.now();
    
    // Track new violations
    for (const regression of regressions) {
      const key = regression.metric;
      
      if (!this.currentViolations.has(key)) {
        this.currentViolations.set(key, {
          metric: regression.metric,
          firstSeen: currentTime,
          consecutiveCount: 1,
          violations: [regression]
        });
      } else {
        const violation = this.currentViolations.get(key);
        violation.consecutiveCount++;
        violation.violations.push(regression);
      }
    }
    
    // Clear violations for metrics that are no longer regressing
    const regressingMetrics = new Set(regressions.map(r => r.metric));
    
    for (const [metric, violation] of this.currentViolations.entries()) {
      if (!regressingMetrics.has(metric)) {
        this.currentViolations.delete(metric);
      }
    }
  }

  /**
   * Check for persistent violations that should trigger alerts
   */
  checkPersistentViolations() {
    const persistentRegressions = [];
    
    for (const [metric, violation] of this.currentViolations.entries()) {
      if (violation.consecutiveCount >= this.config.consecutiveViolations) {
        persistentRegressions.push({
          ...violation,
          isPersistent: true,
          duration: Date.now() - violation.firstSeen
        });
      }
    }
    
    return persistentRegressions;
  }

  /**
   * Trigger alerts for regression detections
   */
  async triggerRegressionAlerts(persistentRegressions) {
    for (const regression of persistentRegressions) {
      const alert = {
        id: `regression-${regression.metric}-${Date.now()}`,
        type: 'performance_regression',
        metric: regression.metric,
        severity: regression.violations[regression.violations.length - 1].severity,
        timestamp: Date.now(),
        description: this.generateRegressionDescription(regression),
        data: regression
      };
      
      // Log alert
      console.log(`🚨 PERFORMANCE REGRESSION ALERT: ${regression.metric}`);
      console.log(`   Severity: ${alert.severity.toUpperCase()}`);
      console.log(`   Duration: ${Math.round((alert.timestamp - regression.firstSeen) / 60000)} minutes`);
      console.log(`   Description: ${alert.description}`);
      
      // Save to detection history
      this.detectionHistory.push(alert);
      
      // In a real implementation, you would integrate with your alerting system here
      // await this.sendAlert(alert);
    }
    
    await this.saveDetectionHistory();
  }

  /**
   * Generate human-readable regression description
   */
  generateRegressionDescription(regression) {
    const latestViolation = regression.violations[regression.violations.length - 1];
    const direction = latestViolation.absoluteChange > 0 ? 'increased' : 'decreased';
    
    return `Metric ${regression.metric} has ${direction} by ${Math.abs(latestViolation.percentChange).toFixed(1)}% ` +
           `over ${regression.consecutiveCount} consecutive measurements. ` +
           `Statistical confidence: ${(latestViolation.confidence * 100).toFixed(1)}%`;
  }

  /**
   * Generate regression detection report
   */
  async generateRegressionReport(startTime, endTime) {
    const reportPeriodAlerts = this.detectionHistory.filter(
      alert => alert.timestamp >= startTime && alert.timestamp <= endTime
    );
    
    const report = {
      period: {
        start: new Date(startTime).toISOString(),
        end: new Date(endTime).toISOString()
      },
      summary: {
        totalRegressions: reportPeriodAlerts.length,
        criticalRegressions: reportPeriodAlerts.filter(a => a.severity === 'critical').length,
        highRegressions: reportPeriodAlerts.filter(a => a.severity === 'high').length,
        affectedMetrics: [...new Set(reportPeriodAlerts.map(a => a.metric))]
      },
      regressions: reportPeriodAlerts,
      baselines: this.getBaselinesSummary(),
      configuration: this.config
    };
    
    // Save report
    const reportPath = path.join(
      path.dirname(this.config.detectionHistory),
      `regression-report-${Date.now()}.json`
    );
    
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    console.log(`📊 Regression report saved: ${reportPath}`);
    
    return report;
  }

  /**
   * Get summary of current baselines
   */
  getBaselinesSummary() {
    const summary = {};
    
    for (const [metric, baseline] of this.baselines.entries()) {
      summary[metric] = {
        value: baseline.value,
        standardDeviation: baseline.standardDeviation,
        sampleCount: baseline.sampleCount,
        age: Date.now() - baseline.updatedAt,
        method: baseline.method
      };
    }
    
    return summary;
  }

  /**
   * Load baselines from storage
   */
  async loadBaselines() {
    try {
      const data = await fs.readFile(this.config.baselineStorage, 'utf8');
      const baselines = JSON.parse(data);
      
      for (const [metric, baseline] of Object.entries(baselines)) {
        this.baselines.set(metric, baseline);
        this.lastBaselineUpdate.set(metric, baseline.updatedAt);
      }
      
    } catch (error) {
      if (error.code !== 'ENOENT') {
        console.error('Error loading baselines:', error);
      }
    }
  }

  /**
   * Save baselines to storage
   */
  async saveBaselines() {
    try {
      const baselineData = {};
      for (const [metric, baseline] of this.baselines.entries()) {
        baselineData[metric] = baseline;
      }
      
      await fs.mkdir(path.dirname(this.config.baselineStorage), { recursive: true });
      await fs.writeFile(
        this.config.baselineStorage, 
        JSON.stringify(baselineData, null, 2)
      );
      
    } catch (error) {
      console.error('Error saving baselines:', error);
    }
  }

  /**
   * Load detection history from storage
   */
  async loadDetectionHistory() {
    try {
      const data = await fs.readFile(this.config.detectionHistory, 'utf8');
      this.detectionHistory = JSON.parse(data);
      
    } catch (error) {
      if (error.code !== 'ENOENT') {
        console.error('Error loading detection history:', error);
      }
    }
  }

  /**
   * Save detection history to storage
   */
  async saveDetectionHistory() {
    try {
      await fs.mkdir(path.dirname(this.config.detectionHistory), { recursive: true });
      await fs.writeFile(
        this.config.detectionHistory,
        JSON.stringify(this.detectionHistory, null, 2)
      );
      
    } catch (error) {
      console.error('Error saving detection history:', error);
    }
  }

  /**
   * Reset baselines (use with caution)
   */
  async resetBaselines(metrics = null) {
    if (metrics) {
      // Reset specific metrics
      for (const metric of metrics) {
        this.baselines.delete(metric);
        this.lastBaselineUpdate.delete(metric);
        console.log(`🔄 Reset baseline for metric: ${metric}`);
      }
    } else {
      // Reset all baselines
      this.baselines.clear();
      this.lastBaselineUpdate.clear();
      console.log('🔄 Reset all baselines');
    }
    
    await this.saveBaselines();
  }

  /**
   * Get regression detection status
   */
  getStatus() {
    return {
      initialized: this.baselines.size > 0,
      activeBaselines: this.baselines.size,
      currentViolations: this.currentViolations.size,
      totalDetections: this.detectionHistory.length,
      lastDetection: this.detectionHistory.length > 0 ? 
        this.detectionHistory[this.detectionHistory.length - 1].timestamp : null,
      configuration: this.config
    };
  }
}

module.exports = { PerformanceRegressionDetector };