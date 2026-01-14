/**
 * Database Performance Monitoring System
 * Real-time metrics collection and alerting
 */

const EventEmitter = require('events');
const fs = require('fs').promises;
const path = require('path');
const { performance } = require('perf_hooks');

class PerformanceMonitor extends EventEmitter {
  constructor(config = {}) {
    super();
    
    this.config = {
      metricsInterval: config.metricsInterval || 1000, // 1 second
      alertThresholds: {
        responseTime: config.alertThresholds?.responseTime || 1000, // 1 second
        errorRate: config.alertThresholds?.errorRate || 5, // 5%
        throughput: config.alertThresholds?.throughput || 100, // queries/sec
        memoryUsage: config.alertThresholds?.memoryUsage || 500, // MB
        connectionCount: config.alertThresholds?.connectionCount || 100
      },
      retentionPeriod: config.retentionPeriod || 3600000, // 1 hour
      outputDir: config.outputDir || './tests/performance/monitoring/data',
      ...config
    };
    
    this.metrics = {
      current: {
        timestamp: Date.now(),
        queryMetrics: {
          totalQueries: 0,
          successfulQueries: 0,
          failedQueries: 0,
          avgResponseTime: 0,
          minResponseTime: Infinity,
          maxResponseTime: 0,
          p50ResponseTime: 0,
          p95ResponseTime: 0,
          p99ResponseTime: 0
        },
        systemMetrics: {
          cpuUsage: 0,
          memoryUsage: 0,
          diskIO: { read: 0, write: 0 },
          networkIO: { in: 0, out: 0 },
          activeConnections: 0,
          connectionPoolSize: 0
        },
        databaseMetrics: {
          cacheHitRatio: 0,
          lockWaits: 0,
          deadlocks: 0,
          slowQueries: 0,
          indexUsage: 0
        }
      },
      history: [],
      alerts: []
    };
    
    this.responseTimeBuffer = [];
    this.isMonitoring = false;
    this.monitoringInterval = null;
    this.alertCooldowns = new Map();
  }

  /**
   * Start performance monitoring
   */
  async start() {
    if (this.isMonitoring) {
      console.log('⚠️ Performance monitoring is already running');
      return;
    }

    console.log('🚀 Starting database performance monitoring...');
    
    // Ensure output directory exists
    await fs.mkdir(this.config.outputDir, { recursive: true });
    
    this.isMonitoring = true;
    this.monitoringInterval = setInterval(() => {
      this.collectMetrics();
    }, this.config.metricsInterval);
    
    // Start alert checking
    this.alertInterval = setInterval(() => {
      this.checkAlerts();
    }, this.config.metricsInterval * 2);
    
    // Cleanup old metrics
    this.cleanupInterval = setInterval(() => {
      this.cleanupOldMetrics();
    }, 60000); // Every minute
    
    this.emit('monitoring:started');
    console.log('✅ Performance monitoring started');
  }

  /**
   * Stop performance monitoring
   */
  async stop() {
    if (!this.isMonitoring) {
      console.log('⚠️ Performance monitoring is not running');
      return;
    }

    console.log('⏹️ Stopping database performance monitoring...');
    
    this.isMonitoring = false;
    
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
    
    if (this.alertInterval) {
      clearInterval(this.alertInterval);
      this.alertInterval = null;
    }
    
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
    
    // Save final metrics
    await this.saveMetricsSnapshot();
    
    this.emit('monitoring:stopped');
    console.log('✅ Performance monitoring stopped');
  }

  /**
   * Record query execution metrics
   */
  recordQuery(queryMetrics) {
    const current = this.metrics.current.queryMetrics;
    
    current.totalQueries++;
    
    if (queryMetrics.success) {
      current.successfulQueries++;
    } else {
      current.failedQueries++;
    }
    
    if (queryMetrics.responseTime !== undefined) {
      this.responseTimeBuffer.push(queryMetrics.responseTime);
      
      // Keep only recent response times for percentile calculation
      if (this.responseTimeBuffer.length > 1000) {
        this.responseTimeBuffer = this.responseTimeBuffer.slice(-1000);
      }
      
      current.minResponseTime = Math.min(current.minResponseTime, queryMetrics.responseTime);
      current.maxResponseTime = Math.max(current.maxResponseTime, queryMetrics.responseTime);
      
      // Calculate moving average
      const totalTime = current.avgResponseTime * (current.totalQueries - 1) + queryMetrics.responseTime;
      current.avgResponseTime = totalTime / current.totalQueries;
      
      // Update percentiles
      this.updateResponseTimePercentiles();
    }
    
    // Track slow queries
    if (queryMetrics.responseTime > this.config.alertThresholds.responseTime) {
      current.slowQueries++;
      
      this.emit('slow:query', {
        query: queryMetrics.query,
        responseTime: queryMetrics.responseTime,
        timestamp: Date.now()
      });
    }
  }

  /**
   * Record system metrics
   */
  recordSystemMetrics(systemMetrics) {
    Object.assign(this.metrics.current.systemMetrics, systemMetrics);
  }

  /**
   * Record database-specific metrics
   */
  recordDatabaseMetrics(dbMetrics) {
    Object.assign(this.metrics.current.databaseMetrics, dbMetrics);
  }

  /**
   * Collect all metrics
   */
  async collectMetrics() {
    if (!this.isMonitoring) return;
    
    const timestamp = Date.now();
    
    // Collect system metrics
    await this.collectSystemMetrics();
    
    // Update timestamp
    this.metrics.current.timestamp = timestamp;
    
    // Create snapshot for history
    const snapshot = JSON.parse(JSON.stringify(this.metrics.current));
    this.metrics.history.push(snapshot);
    
    // Emit metrics update
    this.emit('metrics:collected', snapshot);
    
    // Reset counters for next interval (if needed)
    // Note: We keep cumulative counters for some metrics
  }

  /**
   * Collect system-level metrics
   */
  async collectSystemMetrics() {
    const memoryUsage = process.memoryUsage();
    
    this.metrics.current.systemMetrics.memoryUsage = Math.round(memoryUsage.heapUsed / 1024 / 1024); // MB
    
    // Simulate CPU usage (in real implementation, use system monitoring)
    this.metrics.current.systemMetrics.cpuUsage = Math.random() * 100;
    
    // Simulate other system metrics
    this.metrics.current.systemMetrics.diskIO = {
      read: Math.random() * 1000,
      write: Math.random() * 500
    };
    
    this.metrics.current.systemMetrics.networkIO = {
      in: Math.random() * 10000,
      out: Math.random() * 5000
    };
  }

  /**
   * Update response time percentiles
   */
  updateResponseTimePercentiles() {
    if (this.responseTimeBuffer.length === 0) return;
    
    const sorted = [...this.responseTimeBuffer].sort((a, b) => a - b);
    const current = this.metrics.current.queryMetrics;
    
    current.p50ResponseTime = this.calculatePercentile(sorted, 50);
    current.p95ResponseTime = this.calculatePercentile(sorted, 95);
    current.p99ResponseTime = this.calculatePercentile(sorted, 99);
  }

  /**
   * Calculate percentile value
   */
  calculatePercentile(sortedArray, percentile) {
    const index = Math.ceil((percentile / 100) * sortedArray.length) - 1;
    return sortedArray[Math.max(0, index)];
  }

  /**
   * Check for alerts
   */
  checkAlerts() {
    const current = this.metrics.current;
    const thresholds = this.config.alertThresholds;
    const now = Date.now();
    
    // Response time alert
    if (current.queryMetrics.avgResponseTime > thresholds.responseTime) {
      this.triggerAlert('high_response_time', {
        current: current.queryMetrics.avgResponseTime,
        threshold: thresholds.responseTime,
        severity: 'warning'
      });
    }
    
    // Error rate alert
    const errorRate = current.queryMetrics.totalQueries > 0 ? 
      (current.queryMetrics.failedQueries / current.queryMetrics.totalQueries) * 100 : 0;
    
    if (errorRate > thresholds.errorRate) {
      this.triggerAlert('high_error_rate', {
        current: errorRate,
        threshold: thresholds.errorRate,
        severity: 'critical'
      });
    }
    
    // Memory usage alert
    if (current.systemMetrics.memoryUsage > thresholds.memoryUsage) {
      this.triggerAlert('high_memory_usage', {
        current: current.systemMetrics.memoryUsage,
        threshold: thresholds.memoryUsage,
        severity: 'warning'
      });
    }
    
    // Connection count alert
    if (current.systemMetrics.activeConnections > thresholds.connectionCount) {
      this.triggerAlert('high_connection_count', {
        current: current.systemMetrics.activeConnections,
        threshold: thresholds.connectionCount,
        severity: 'warning'
      });
    }
  }

  /**
   * Trigger alert with cooldown
   */
  triggerAlert(alertType, alertData) {
    const cooldownKey = alertType;
    const now = Date.now();
    const cooldownPeriod = 300000; // 5 minutes
    
    // Check if alert is in cooldown
    if (this.alertCooldowns.has(cooldownKey)) {
      const lastAlert = this.alertCooldowns.get(cooldownKey);
      if (now - lastAlert < cooldownPeriod) {
        return; // Skip alert due to cooldown
      }
    }
    
    const alert = {
      type: alertType,
      timestamp: now,
      data: alertData,
      id: `alert-${now}-${Math.random().toString(36).substr(2, 9)}`
    };
    
    this.metrics.alerts.push(alert);
    this.alertCooldowns.set(cooldownKey, now);
    
    console.log(`🚨 ALERT [${alertData.severity.toUpperCase()}]: ${alertType}`);
    console.log(`   Current: ${alertData.current}, Threshold: ${alertData.threshold}`);
    
    this.emit('alert:triggered', alert);
  }

  /**
   * Get current performance summary
   */
  getCurrentSummary() {
    const current = this.metrics.current;
    const errorRate = current.queryMetrics.totalQueries > 0 ? 
      (current.queryMetrics.failedQueries / current.queryMetrics.totalQueries) * 100 : 0;
    
    return {
      timestamp: current.timestamp,
      uptime: Date.now() - (this.metrics.history[0]?.timestamp || Date.now()),
      queries: {
        total: current.queryMetrics.totalQueries,
        successful: current.queryMetrics.successfulQueries,
        failed: current.queryMetrics.failedQueries,
        errorRate: errorRate.toFixed(2),
        avgResponseTime: current.queryMetrics.avgResponseTime.toFixed(2),
        p95ResponseTime: current.queryMetrics.p95ResponseTime.toFixed(2),
        slowQueries: current.queryMetrics.slowQueries
      },
      system: {
        memoryUsage: current.systemMetrics.memoryUsage,
        cpuUsage: current.systemMetrics.cpuUsage.toFixed(1),
        activeConnections: current.systemMetrics.activeConnections
      },
      database: {
        cacheHitRatio: current.databaseMetrics.cacheHitRatio.toFixed(2),
        lockWaits: current.databaseMetrics.lockWaits,
        deadlocks: current.databaseMetrics.deadlocks
      },
      alerts: {
        total: this.metrics.alerts.length,
        recent: this.metrics.alerts.filter(a => Date.now() - a.timestamp < 300000).length // Last 5 minutes
      }
    };
  }

  /**
   * Get metrics history
   */
  getMetricsHistory(startTime, endTime) {
    let history = this.metrics.history;
    
    if (startTime) {
      history = history.filter(m => m.timestamp >= startTime);
    }
    
    if (endTime) {
      history = history.filter(m => m.timestamp <= endTime);
    }
    
    return history;
  }

  /**
   * Get performance trends
   */
  getPerformanceTrends(periodMinutes = 60) {
    const periodMs = periodMinutes * 60 * 1000;
    const cutoff = Date.now() - periodMs;
    const recentMetrics = this.metrics.history.filter(m => m.timestamp >= cutoff);
    
    if (recentMetrics.length === 0) {
      return null;
    }
    
    const responseTimesTrend = recentMetrics.map(m => m.queryMetrics.avgResponseTime);
    const throughputTrend = recentMetrics.map(m => m.queryMetrics.totalQueries);
    const errorRatesTrend = recentMetrics.map(m => 
      m.queryMetrics.totalQueries > 0 ? 
        (m.queryMetrics.failedQueries / m.queryMetrics.totalQueries) * 100 : 0
    );
    
    return {
      period: `${periodMinutes} minutes`,
      responseTime: {
        trend: this.calculateTrend(responseTimesTrend),
        min: Math.min(...responseTimesTrend),
        max: Math.max(...responseTimesTrend),
        avg: responseTimesTrend.reduce((sum, rt) => sum + rt, 0) / responseTimesTrend.length
      },
      throughput: {
        trend: this.calculateTrend(throughputTrend),
        min: Math.min(...throughputTrend),
        max: Math.max(...throughputTrend),
        avg: throughputTrend.reduce((sum, tp) => sum + tp, 0) / throughputTrend.length
      },
      errorRate: {
        trend: this.calculateTrend(errorRatesTrend),
        min: Math.min(...errorRatesTrend),
        max: Math.max(...errorRatesTrend),
        avg: errorRatesTrend.reduce((sum, er) => sum + er, 0) / errorRatesTrend.length
      }
    };
  }

  /**
   * Calculate trend (increasing, decreasing, stable)
   */
  calculateTrend(values) {
    if (values.length < 2) return 'stable';
    
    const firstHalf = values.slice(0, Math.floor(values.length / 2));
    const secondHalf = values.slice(Math.floor(values.length / 2));
    
    const firstAvg = firstHalf.reduce((sum, v) => sum + v, 0) / firstHalf.length;
    const secondAvg = secondHalf.reduce((sum, v) => sum + v, 0) / secondHalf.length;
    
    const changePercent = ((secondAvg - firstAvg) / firstAvg) * 100;
    
    if (changePercent > 10) return 'increasing';
    if (changePercent < -10) return 'decreasing';
    return 'stable';
  }

  /**
   * Save metrics snapshot to file
   */
  async saveMetricsSnapshot() {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `metrics-snapshot-${timestamp}.json`;
    const filepath = path.join(this.config.outputDir, filename);
    
    const snapshot = {
      timestamp: Date.now(),
      current: this.metrics.current,
      summary: this.getCurrentSummary(),
      trends: this.getPerformanceTrends(60),
      alerts: this.metrics.alerts.slice(-100), // Last 100 alerts
      config: this.config
    };
    
    await fs.writeFile(filepath, JSON.stringify(snapshot, null, 2));
    console.log(`📊 Metrics snapshot saved: ${filepath}`);
    
    return filepath;
  }

  /**
   * Clean up old metrics to prevent memory leaks
   */
  cleanupOldMetrics() {
    const cutoff = Date.now() - this.config.retentionPeriod;
    
    this.metrics.history = this.metrics.history.filter(m => m.timestamp >= cutoff);
    this.metrics.alerts = this.metrics.alerts.filter(a => a.timestamp >= cutoff);
    
    // Clean up alert cooldowns
    for (const [key, timestamp] of this.alertCooldowns.entries()) {
      if (Date.now() - timestamp > 3600000) { // 1 hour
        this.alertCooldowns.delete(key);
      }
    }
  }

  /**
   * Generate real-time dashboard data
   */
  getDashboardData() {
    const summary = this.getCurrentSummary();
    const trends = this.getPerformanceTrends(30); // Last 30 minutes
    const recentMetrics = this.getMetricsHistory(Date.now() - 1800000); // Last 30 minutes
    
    return {
      summary,
      trends,
      chartData: {
        responseTime: recentMetrics.map(m => ({
          timestamp: m.timestamp,
          value: m.queryMetrics.avgResponseTime
        })),
        throughput: recentMetrics.map(m => ({
          timestamp: m.timestamp,
          value: m.queryMetrics.totalQueries
        })),
        errorRate: recentMetrics.map(m => ({
          timestamp: m.timestamp,
          value: m.queryMetrics.totalQueries > 0 ? 
            (m.queryMetrics.failedQueries / m.queryMetrics.totalQueries) * 100 : 0
        })),
        memoryUsage: recentMetrics.map(m => ({
          timestamp: m.timestamp,
          value: m.systemMetrics.memoryUsage
        }))
      },
      alerts: this.metrics.alerts.slice(-10), // Last 10 alerts
      healthScore: this.calculateHealthScore()
    };
  }

  /**
   * Calculate overall health score (0-100)
   */
  calculateHealthScore() {
    const current = this.metrics.current;
    const thresholds = this.config.alertThresholds;
    
    let score = 100;
    
    // Response time impact
    const responseTimeRatio = current.queryMetrics.avgResponseTime / thresholds.responseTime;
    if (responseTimeRatio > 1) {
      score -= Math.min(50, (responseTimeRatio - 1) * 50);
    }
    
    // Error rate impact
    const errorRate = current.queryMetrics.totalQueries > 0 ? 
      (current.queryMetrics.failedQueries / current.queryMetrics.totalQueries) * 100 : 0;
    if (errorRate > thresholds.errorRate) {
      score -= Math.min(30, ((errorRate - thresholds.errorRate) / thresholds.errorRate) * 30);
    }
    
    // Memory usage impact
    const memoryRatio = current.systemMetrics.memoryUsage / thresholds.memoryUsage;
    if (memoryRatio > 1) {
      score -= Math.min(20, (memoryRatio - 1) * 20);
    }
    
    return Math.max(0, Math.round(score));
  }
}

module.exports = { PerformanceMonitor };