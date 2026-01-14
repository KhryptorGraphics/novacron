/**
 * Historical Trend Analysis System
 * Advanced analytics for historical performance data with predictive capabilities
 */

const EventEmitter = require('events');
const fs = require('fs').promises;
const path = require('path');

class HistoricalTrendAnalyzer extends EventEmitter {
  constructor(metricsCollector, config = {}) {
    super();
    
    this.metricsCollector = metricsCollector;
    this.config = {
      analysisInterval: config.analysisInterval || 300000, // 5 minutes
      trendWindow: config.trendWindow || 24 * 60 * 60 * 1000, // 24 hours
      predictionWindow: config.predictionWindow || 4 * 60 * 60 * 1000, // 4 hours
      significanceThreshold: config.significanceThreshold || 0.05,
      alertThresholds: config.alertThresholds || {},
      seasonalityPeriods: config.seasonalityPeriods || [60 * 60 * 1000, 24 * 60 * 60 * 1000], // 1h, 24h
      enablePredictions: config.enablePredictions !== false,
      enableAnomalyDetection: config.enableAnomalyDetection !== false,
      storageLocation: config.storageLocation || './trend-analysis',
      ...config
    };

    this.trendData = new Map();
    this.anomalies = [];
    this.predictions = new Map();
    this.models = new Map();
    this.analysisInterval = null;
    
    this.initializeModels();
  }

  initializeModels() {
    // Initialize trend models for different metric types
    this.models.set('system', new SystemTrendModel(this.config));
    this.models.set('application', new ApplicationTrendModel(this.config));
    this.models.set('database', new DatabaseTrendModel(this.config));
    this.models.set('network', new NetworkTrendModel(this.config));
    this.models.set('ml_workflow', new MLWorkflowTrendModel(this.config));
    
    console.log(`Initialized ${this.models.size} trend analysis models`);
  }

  async start() {
    console.log('Starting historical trend analysis...');
    
    // Ensure storage directory exists
    await fs.mkdir(this.config.storageLocation, { recursive: true });
    
    // Load historical trend data
    await this.loadHistoricalTrends();
    
    // Start periodic analysis
    this.analysisInterval = setInterval(async () => {
      await this.performTrendAnalysis();
    }, this.config.analysisInterval);
    
    // Perform initial analysis
    await this.performTrendAnalysis();
    
    this.emit('analysis:started');
    console.log('Trend analysis started');
  }

  async stop() {
    console.log('Stopping trend analysis...');
    
    if (this.analysisInterval) {
      clearInterval(this.analysisInterval);
      this.analysisInterval = null;
    }
    
    // Save current trend data
    await this.saveTrendData();
    
    this.emit('analysis:stopped');
    console.log('Trend analysis stopped');
  }

  async performTrendAnalysis() {
    try {
      console.log('Performing trend analysis...');
      
      const endTime = Date.now();
      const startTime = endTime - this.config.trendWindow;
      
      // Get historical metrics
      const metrics = await this.metricsCollector.getMetrics({
        startTime,
        endTime,
        limit: 5000
      });

      if (metrics.length === 0) {
        console.log('No metrics available for trend analysis');
        return;
      }

      // Analyze trends for each metric type
      const analysisResults = {};
      
      for (const [modelName, model] of this.models) {
        try {
          const modelMetrics = this.extractMetricsForModel(metrics, modelName);
          if (modelMetrics.length > 0) {
            const analysis = await model.analyzeTrends(modelMetrics);
            analysisResults[modelName] = analysis;
            
            // Store trend data
            this.trendData.set(modelName, {
              timestamp: Date.now(),
              analysis,
              dataPoints: modelMetrics.length
            });
          }
        } catch (error) {
          console.error(`Error analyzing ${modelName} trends:`, error);
        }
      }

      // Detect anomalies
      if (this.config.enableAnomalyDetection) {
        await this.detectAnomalies(analysisResults);
      }

      // Generate predictions
      if (this.config.enablePredictions) {
        await this.generatePredictions(analysisResults);
      }

      // Check for trend alerts
      await this.checkTrendAlerts(analysisResults);

      // Save analysis results
      await this.saveAnalysisResults(analysisResults);

      this.emit('analysis:completed', analysisResults);

    } catch (error) {
      console.error('Error during trend analysis:', error);
      this.emit('analysis:error', error);
    }
  }

  extractMetricsForModel(metrics, modelName) {
    return metrics
      .filter(m => m.collectors && m.collectors[modelName])
      .map(m => ({
        timestamp: m.timestamp,
        ...m.collectors[modelName]
      }))
      .filter(m => !m.error);
  }

  async detectAnomalies(analysisResults) {
    const newAnomalies = [];
    const timestamp = Date.now();

    for (const [modelName, analysis] of Object.entries(analysisResults)) {
      if (analysis.anomalies && analysis.anomalies.length > 0) {
        for (const anomaly of analysis.anomalies) {
          newAnomalies.push({
            id: `${modelName}_${timestamp}_${Math.random().toString(36).substr(2, 9)}`,
            model: modelName,
            timestamp,
            type: anomaly.type,
            severity: anomaly.severity,
            metric: anomaly.metric,
            value: anomaly.value,
            expected: anomaly.expected,
            deviation: anomaly.deviation,
            description: anomaly.description
          });
        }
      }
    }

    if (newAnomalies.length > 0) {
      this.anomalies.push(...newAnomalies);
      
      // Keep only recent anomalies (last 7 days)
      const cutoffTime = timestamp - (7 * 24 * 60 * 60 * 1000);
      this.anomalies = this.anomalies.filter(a => a.timestamp > cutoffTime);
      
      this.emit('anomalies:detected', newAnomalies);
      console.log(`Detected ${newAnomalies.length} new anomalies`);
    }
  }

  async generatePredictions(analysisResults) {
    const timestamp = Date.now();
    const predictionHorizon = this.config.predictionWindow;

    for (const [modelName, analysis] of Object.entries(analysisResults)) {
      if (analysis.trend && analysis.trend.confidence > 0.6) {
        try {
          const model = this.models.get(modelName);
          const predictions = await model.generatePredictions(analysis, predictionHorizon);
          
          this.predictions.set(modelName, {
            timestamp,
            horizon: predictionHorizon,
            predictions,
            confidence: analysis.trend.confidence
          });
          
        } catch (error) {
          console.error(`Error generating ${modelName} predictions:`, error);
        }
      }
    }

    if (this.predictions.size > 0) {
      this.emit('predictions:generated', Object.fromEntries(this.predictions));
    }
  }

  async checkTrendAlerts(analysisResults) {
    const alerts = [];
    const timestamp = Date.now();

    for (const [modelName, analysis] of Object.entries(analysisResults)) {
      const modelAlerts = this.checkModelAlerts(modelName, analysis, timestamp);
      alerts.push(...modelAlerts);
    }

    if (alerts.length > 0) {
      this.emit('alerts:triggered', alerts);
      console.log(`Generated ${alerts.length} trend alerts`);
    }
  }

  checkModelAlerts(modelName, analysis, timestamp) {
    const alerts = [];
    const thresholds = this.config.alertThresholds[modelName] || {};

    // Check trend direction alerts
    if (analysis.trend) {
      if (analysis.trend.direction === 'increasing' && thresholds.increasingTrend) {
        if (analysis.trend.slope > thresholds.increasingTrend.slope) {
          alerts.push({
            id: `trend_${modelName}_increasing_${timestamp}`,
            type: 'trend',
            model: modelName,
            severity: 'warning',
            message: `${modelName} metrics showing increasing trend (slope: ${analysis.trend.slope.toFixed(4)})`,
            metric: 'trend_slope',
            value: analysis.trend.slope,
            threshold: thresholds.increasingTrend.slope,
            timestamp
          });
        }
      }

      if (analysis.trend.direction === 'decreasing' && thresholds.decreasingTrend) {
        if (Math.abs(analysis.trend.slope) > thresholds.decreasingTrend.slope) {
          alerts.push({
            id: `trend_${modelName}_decreasing_${timestamp}`,
            type: 'trend',
            model: modelName,
            severity: 'info',
            message: `${modelName} metrics showing decreasing trend (slope: ${analysis.trend.slope.toFixed(4)})`,
            metric: 'trend_slope',
            value: analysis.trend.slope,
            threshold: -thresholds.decreasingTrend.slope,
            timestamp
          });
        }
      }
    }

    // Check seasonality alerts
    if (analysis.seasonality && thresholds.seasonality) {
      if (analysis.seasonality.strength > thresholds.seasonality.maxStrength) {
        alerts.push({
          id: `seasonality_${modelName}_${timestamp}`,
          type: 'seasonality',
          model: modelName,
          severity: 'info',
          message: `Strong seasonality detected in ${modelName} metrics (strength: ${analysis.seasonality.strength.toFixed(2)})`,
          metric: 'seasonality_strength',
          value: analysis.seasonality.strength,
          threshold: thresholds.seasonality.maxStrength,
          timestamp
        });
      }
    }

    // Check volatility alerts
    if (analysis.volatility && thresholds.volatility) {
      if (analysis.volatility.coefficient > thresholds.volatility.maxCoefficient) {
        alerts.push({
          id: `volatility_${modelName}_${timestamp}`,
          type: 'volatility',
          model: modelName,
          severity: 'warning',
          message: `High volatility detected in ${modelName} metrics (CV: ${analysis.volatility.coefficient.toFixed(2)})`,
          metric: 'volatility_coefficient',
          value: analysis.volatility.coefficient,
          threshold: thresholds.volatility.maxCoefficient,
          timestamp
        });
      }
    }

    return alerts;
  }

  async getTrendSummary(timeRange = '24h') {
    const endTime = Date.now();
    let startTime;
    
    switch (timeRange) {
      case '1h':
        startTime = endTime - (60 * 60 * 1000);
        break;
      case '24h':
        startTime = endTime - (24 * 60 * 60 * 1000);
        break;
      case '7d':
        startTime = endTime - (7 * 24 * 60 * 60 * 1000);
        break;
      default:
        startTime = endTime - (24 * 60 * 60 * 1000);
    }

    const summary = {
      timeRange,
      period: { startTime, endTime },
      trends: {},
      anomalies: this.anomalies.filter(a => a.timestamp >= startTime),
      predictions: Object.fromEntries(this.predictions),
      overallHealth: 'good'
    };

    // Summarize trends for each model
    for (const [modelName, trendData] of this.trendData) {
      if (trendData.timestamp >= startTime) {
        summary.trends[modelName] = {
          direction: trendData.analysis.trend?.direction || 'stable',
          confidence: trendData.analysis.trend?.confidence || 0,
          slope: trendData.analysis.trend?.slope || 0,
          volatility: trendData.analysis.volatility?.coefficient || 0,
          seasonality: trendData.analysis.seasonality?.strength || 0,
          dataPoints: trendData.dataPoints
        };
      }
    }

    // Calculate overall health
    summary.overallHealth = this.calculateOverallHealth(summary);

    return summary;
  }

  calculateOverallHealth(summary) {
    let healthScore = 100;
    let issues = 0;

    // Penalize for anomalies
    const criticalAnomalies = summary.anomalies.filter(a => a.severity === 'critical').length;
    const warningAnomalies = summary.anomalies.filter(a => a.severity === 'warning').length;
    
    healthScore -= criticalAnomalies * 20;
    healthScore -= warningAnomalies * 10;
    issues += criticalAnomalies + warningAnomalies;

    // Penalize for concerning trends
    for (const trend of Object.values(summary.trends)) {
      if (trend.direction === 'increasing' && trend.slope > 0.01) {
        healthScore -= 15;
        issues++;
      }
      
      if (trend.volatility > 0.5) {
        healthScore -= 10;
        issues++;
      }
    }

    // Determine health category
    if (healthScore >= 90 && issues === 0) return 'excellent';
    if (healthScore >= 75) return 'good';
    if (healthScore >= 50) return 'fair';
    if (healthScore >= 25) return 'poor';
    return 'critical';
  }

  async saveAnalysisResults(results) {
    const timestamp = Date.now();
    const filename = `trend_analysis_${timestamp}.json`;
    const filepath = path.join(this.config.storageLocation, filename);
    
    const analysisData = {
      timestamp,
      results,
      anomalies: this.anomalies.slice(-100), // Last 100 anomalies
      predictions: Object.fromEntries(this.predictions),
      summary: await this.getTrendSummary()
    };

    await fs.writeFile(filepath, JSON.stringify(analysisData, null, 2));
    console.log(`Saved trend analysis results to ${filename}`);
  }

  async loadHistoricalTrends() {
    try {
      const files = await fs.readdir(this.config.storageLocation);
      const trendFiles = files
        .filter(f => f.startsWith('trend_analysis_'))
        .sort()
        .reverse()
        .slice(0, 10); // Load last 10 analyses

      for (const file of trendFiles) {
        const filepath = path.join(this.config.storageLocation, file);
        const content = await fs.readFile(filepath, 'utf8');
        const data = JSON.parse(content);
        
        // Restore anomalies
        if (data.anomalies) {
          this.anomalies.push(...data.anomalies);
        }
        
        // Restore predictions
        if (data.predictions) {
          for (const [model, prediction] of Object.entries(data.predictions)) {
            this.predictions.set(model, prediction);
          }
        }
      }

      // Remove duplicate anomalies
      const uniqueAnomalies = new Map();
      for (const anomaly of this.anomalies) {
        uniqueAnomalies.set(anomaly.id, anomaly);
      }
      this.anomalies = Array.from(uniqueAnomalies.values());

      console.log(`Loaded historical trends from ${trendFiles.length} files`);

    } catch (error) {
      console.log('No historical trend data found, starting fresh');
    }
  }

  async saveTrendData() {
    const data = {
      timestamp: Date.now(),
      trendData: Object.fromEntries(this.trendData),
      anomalies: this.anomalies,
      predictions: Object.fromEntries(this.predictions)
    };

    const filename = 'trend_data_snapshot.json';
    const filepath = path.join(this.config.storageLocation, filename);
    
    await fs.writeFile(filepath, JSON.stringify(data, null, 2));
  }

  getStatus() {
    return {
      isRunning: this.analysisInterval !== null,
      modelsActive: this.models.size,
      trendDataEntries: this.trendData.size,
      anomaliesCount: this.anomalies.length,
      predictionsCount: this.predictions.size,
      config: {
        analysisInterval: this.config.analysisInterval,
        trendWindow: this.config.trendWindow,
        predictionWindow: this.config.predictionWindow
      }
    };
  }
}

// Base trend model class
class TrendModel {
  constructor(config) {
    this.config = config;
  }

  async analyzeTrends(metrics) {
    if (metrics.length < 10) {
      return { error: 'Insufficient data points for trend analysis' };
    }

    const timestamps = metrics.map(m => m.timestamp);
    const values = this.extractValues(metrics);
    
    const analysis = {
      dataPoints: metrics.length,
      timeSpan: timestamps[timestamps.length - 1] - timestamps[0],
      trend: this.calculateTrend(timestamps, values),
      seasonality: this.detectSeasonality(timestamps, values),
      volatility: this.calculateVolatility(values),
      anomalies: this.detectAnomalies(timestamps, values),
      statistics: this.calculateStatistics(values)
    };

    return analysis;
  }

  extractValues(metrics) {
    // Override in subclasses to extract relevant values
    return metrics.map(m => m.value || 0);
  }

  calculateTrend(timestamps, values) {
    const n = values.length;
    const x = timestamps.map((t, i) => i); // Normalize to indices
    const y = values;
    
    // Linear regression
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    // Calculate R-squared
    const yMean = sumY / n;
    const totalSumSquares = y.reduce((sum, yi) => sum + Math.pow(yi - yMean, 2), 0);
    const residualSumSquares = y.reduce((sum, yi, i) => {
      const predicted = slope * x[i] + intercept;
      return sum + Math.pow(yi - predicted, 2);
    }, 0);
    const rSquared = 1 - (residualSumSquares / totalSumSquares);
    
    return {
      slope,
      intercept,
      rSquared,
      confidence: Math.min(rSquared, 1),
      direction: Math.abs(slope) < 0.001 ? 'stable' : slope > 0 ? 'increasing' : 'decreasing',
      significance: Math.abs(slope) > this.config.significanceThreshold
    };
  }

  detectSeasonality(timestamps, values) {
    if (values.length < 48) return null; // Need at least 48 points for seasonality
    
    // Simple seasonality detection using autocorrelation
    const periods = this.config.seasonalityPeriods;
    let maxCorrelation = 0;
    let bestPeriod = null;
    
    for (const period of periods) {
      const correlation = this.calculateAutocorrelation(values, Math.floor(period / 60000)); // Convert to minutes
      if (correlation > maxCorrelation) {
        maxCorrelation = correlation;
        bestPeriod = period;
      }
    }
    
    return {
      detected: maxCorrelation > 0.3,
      strength: maxCorrelation,
      period: bestPeriod,
      periodLabel: this.formatPeriod(bestPeriod)
    };
  }

  calculateAutocorrelation(values, lag) {
    if (lag >= values.length) return 0;
    
    const n = values.length - lag;
    const mean1 = values.slice(0, n).reduce((a, b) => a + b, 0) / n;
    const mean2 = values.slice(lag).reduce((a, b) => a + b, 0) / n;
    
    let numerator = 0;
    let denominator1 = 0;
    let denominator2 = 0;
    
    for (let i = 0; i < n; i++) {
      const x1 = values[i] - mean1;
      const x2 = values[i + lag] - mean2;
      numerator += x1 * x2;
      denominator1 += x1 * x1;
      denominator2 += x2 * x2;
    }
    
    return numerator / Math.sqrt(denominator1 * denominator2);
  }

  calculateVolatility(values) {
    if (values.length < 2) return { coefficient: 0 };
    
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (values.length - 1);
    const stdDev = Math.sqrt(variance);
    const coefficient = mean !== 0 ? stdDev / Math.abs(mean) : 0;
    
    return {
      mean,
      standardDeviation: stdDev,
      variance,
      coefficient,
      level: coefficient < 0.1 ? 'low' : coefficient < 0.3 ? 'medium' : 'high'
    };
  }

  detectAnomalies(timestamps, values) {
    const anomalies = [];
    
    if (values.length < 10) return anomalies;
    
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const stdDev = Math.sqrt(values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length);
    
    // Z-score based anomaly detection
    for (let i = 0; i < values.length; i++) {
      const zScore = Math.abs((values[i] - mean) / stdDev);
      
      if (zScore > 3) { // More than 3 standard deviations
        anomalies.push({
          type: 'statistical_outlier',
          timestamp: timestamps[i],
          metric: this.getMetricName(),
          value: values[i],
          expected: mean,
          deviation: zScore,
          severity: zScore > 4 ? 'critical' : 'warning',
          description: `Value ${values[i].toFixed(2)} deviates ${zScore.toFixed(2)} standard deviations from mean`
        });
      }
    }
    
    return anomalies;
  }

  calculateStatistics(values) {
    if (values.length === 0) return {};
    
    const sorted = [...values].sort((a, b) => a - b);
    const n = values.length;
    
    return {
      count: n,
      min: sorted[0],
      max: sorted[n - 1],
      mean: values.reduce((a, b) => a + b, 0) / n,
      median: n % 2 === 0 ? (sorted[n/2 - 1] + sorted[n/2]) / 2 : sorted[Math.floor(n/2)],
      q1: sorted[Math.floor(n * 0.25)],
      q3: sorted[Math.floor(n * 0.75)],
      p95: sorted[Math.floor(n * 0.95)],
      p99: sorted[Math.floor(n * 0.99)]
    };
  }

  async generatePredictions(analysis, horizon) {
    // Simple linear extrapolation
    const predictions = [];
    const trend = analysis.trend;
    
    if (!trend || trend.confidence < 0.5) {
      return { error: 'Insufficient trend confidence for predictions' };
    }
    
    const stepSize = horizon / 10; // 10 prediction points
    const lastValue = analysis.statistics.mean;
    
    for (let i = 1; i <= 10; i++) {
      const futureTime = Date.now() + (stepSize * i);
      const predictedValue = lastValue + (trend.slope * i);
      
      predictions.push({
        timestamp: futureTime,
        value: predictedValue,
        confidence: Math.max(0.1, trend.confidence - (i * 0.05)) // Decreasing confidence
      });
    }
    
    return {
      predictions,
      method: 'linear_extrapolation',
      confidence: trend.confidence,
      horizon
    };
  }

  getMetricName() {
    return 'unknown';
  }

  formatPeriod(ms) {
    const hours = ms / (1000 * 60 * 60);
    if (hours < 1) return `${Math.round(ms / (1000 * 60))}m`;
    if (hours < 24) return `${Math.round(hours)}h`;
    return `${Math.round(hours / 24)}d`;
  }
}

// Specific model implementations
class SystemTrendModel extends TrendModel {
  extractValues(metrics) {
    // Combine CPU and memory usage for system trend
    return metrics.map(m => {
      const cpu = m.cpu?.usage || 0;
      const memory = m.memory?.usage || 0;
      return (cpu + memory) / 2; // Average of CPU and memory
    });
  }
  
  getMetricName() {
    return 'system_performance';
  }
}

class ApplicationTrendModel extends TrendModel {
  extractValues(metrics) {
    return metrics.map(m => m.responseTime || m.avgResponseTime || 0);
  }
  
  getMetricName() {
    return 'response_time';
  }
}

class DatabaseTrendModel extends TrendModel {
  extractValues(metrics) {
    return metrics.map(m => m.queryTime || m.avgQueryTime || 0);
  }
  
  getMetricName() {
    return 'query_time';
  }
}

class NetworkTrendModel extends TrendModel {
  extractValues(metrics) {
    return metrics.map(m => m.latency || 0);
  }
  
  getMetricName() {
    return 'network_latency';
  }
}

class MLWorkflowTrendModel extends TrendModel {
  extractValues(metrics) {
    return metrics.map(m => m.trainingTime || m.inferenceTime || 0).filter(v => v > 0);
  }
  
  getMetricName() {
    return 'ml_execution_time';
  }
}

module.exports = {
  HistoricalTrendAnalyzer,
  TrendModel,
  SystemTrendModel,
  ApplicationTrendModel,
  DatabaseTrendModel,
  NetworkTrendModel,
  MLWorkflowTrendModel
};