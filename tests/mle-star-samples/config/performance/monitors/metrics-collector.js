/**
 * Performance Metrics Collection System
 * Comprehensive system for collecting, aggregating, and storing performance metrics
 */

const EventEmitter = require('events');
const os = require('os');
const fs = require('fs').promises;
const path = require('path');

class PerformanceMetricsCollector extends EventEmitter {
  constructor(config = {}) {
    super();
    
    this.config = {
      collectionInterval: config.collectionInterval || 5000, // 5 seconds
      storageInterval: config.storageInterval || 60000, // 1 minute  
      retentionPeriod: config.retentionPeriod || 7 * 24 * 60 * 60 * 1000, // 7 days
      storageLocation: config.storageLocation || './metrics-data',
      enableRealtime: config.enableRealtime !== false,
      enableHistorical: config.enableHistorical !== false,
      enableAggregation: config.enableAggregation !== false,
      batchSize: config.batchSize || 100,
      compressionEnabled: config.compressionEnabled || true,
      ...config
    };

    this.isCollecting = false;
    this.metricsBuffer = [];
    this.aggregatedMetrics = new Map();
    this.collectionInterval = null;
    this.storageInterval = null;
    this.collectors = new Map();
    
    this.initializeCollectors();
  }

  initializeCollectors() {
    // System metrics collector
    this.collectors.set('system', new SystemMetricsCollector(this.config));
    
    // Application metrics collector
    this.collectors.set('application', new ApplicationMetricsCollector(this.config));
    
    // Database metrics collector
    this.collectors.set('database', new DatabaseMetricsCollector(this.config));
    
    // Network metrics collector
    this.collectors.set('network', new NetworkMetricsCollector(this.config));
    
    // ML workflow metrics collector
    this.collectors.set('ml_workflow', new MLWorkflowMetricsCollector(this.config));
    
    // Custom metrics collector
    this.collectors.set('custom', new CustomMetricsCollector(this.config));
    
    console.log(`Initialized ${this.collectors.size} metrics collectors`);
  }

  async start() {
    if (this.isCollecting) {
      console.log('Metrics collection already running');
      return;
    }

    console.log('Starting performance metrics collection...');
    
    // Ensure storage directory exists
    await fs.mkdir(this.config.storageLocation, { recursive: true });
    
    this.isCollecting = true;
    
    // Start collection interval
    this.collectionInterval = setInterval(async () => {
      await this.collectMetrics();
    }, this.config.collectionInterval);
    
    // Start storage interval
    if (this.config.enableHistorical) {
      this.storageInterval = setInterval(async () => {
        await this.storeMetrics();
      }, this.config.storageInterval);
    }
    
    // Start aggregation if enabled
    if (this.config.enableAggregation) {
      setInterval(async () => {
        await this.aggregateMetrics();
      }, this.config.storageInterval * 2); // Aggregate every 2 minutes
    }
    
    this.emit('collection:started');
    console.log('Metrics collection started');
  }

  async stop() {
    if (!this.isCollecting) {
      console.log('Metrics collection not running');
      return;
    }

    console.log('Stopping performance metrics collection...');
    
    this.isCollecting = false;
    
    if (this.collectionInterval) {
      clearInterval(this.collectionInterval);
      this.collectionInterval = null;
    }
    
    if (this.storageInterval) {
      clearInterval(this.storageInterval);
      this.storageInterval = null;
    }
    
    // Store any remaining metrics
    if (this.metricsBuffer.length > 0) {
      await this.storeMetrics();
    }
    
    this.emit('collection:stopped');
    console.log('Metrics collection stopped');
  }

  async collectMetrics() {
    try {
      const timestamp = Date.now();
      const metrics = {
        timestamp,
        collectors: {}
      };

      // Collect from all registered collectors
      for (const [name, collector] of this.collectors) {
        try {
          const collectorMetrics = await collector.collect();
          metrics.collectors[name] = collectorMetrics;
        } catch (error) {
          console.error(`Error collecting ${name} metrics:`, error);
          metrics.collectors[name] = { error: error.message };
        }
      }

      // Add to buffer
      this.metricsBuffer.push(metrics);

      // Emit real-time metrics if enabled
      if (this.config.enableRealtime) {
        this.emit('metrics:collected', metrics);
      }

      // Maintain buffer size
      if (this.metricsBuffer.length > this.config.batchSize * 2) {
        this.metricsBuffer = this.metricsBuffer.slice(-this.config.batchSize);
      }

    } catch (error) {
      console.error('Error during metrics collection:', error);
      this.emit('collection:error', error);
    }
  }

  async storeMetrics() {
    if (this.metricsBuffer.length === 0) return;

    try {
      const timestamp = Date.now();
      const filename = `metrics_${timestamp}.json`;
      const filepath = path.join(this.config.storageLocation, filename);
      
      const dataToStore = {
        collectionTime: timestamp,
        metricsCount: this.metricsBuffer.length,
        metrics: [...this.metricsBuffer]
      };

      // Compress if enabled
      if (this.config.compressionEnabled) {
        const compressed = await this.compressData(dataToStore);
        await fs.writeFile(filepath + '.gz', compressed);
      } else {
        await fs.writeFile(filepath, JSON.stringify(dataToStore, null, 2));
      }

      console.log(`Stored ${this.metricsBuffer.length} metrics to ${filename}`);
      
      // Clear buffer
      this.metricsBuffer = [];
      
      // Cleanup old files
      await this.cleanupOldMetrics();
      
      this.emit('metrics:stored', { filename, count: dataToStore.metricsCount });

    } catch (error) {
      console.error('Error storing metrics:', error);
      this.emit('storage:error', error);
    }
  }

  async aggregateMetrics() {
    try {
      const now = Date.now();
      const hourlyKey = Math.floor(now / (60 * 60 * 1000)); // Hour bucket
      const dailyKey = Math.floor(now / (24 * 60 * 60 * 1000)); // Day bucket

      // Aggregate recent metrics
      const recentMetrics = this.metricsBuffer.slice(-60); // Last 60 samples (5 minutes)
      
      if (recentMetrics.length === 0) return;

      // Calculate aggregations
      const hourlyAgg = this.calculateAggregation(recentMetrics, 'hourly');
      const dailyAgg = this.calculateAggregation(recentMetrics, 'daily');

      // Store aggregations
      this.aggregatedMetrics.set(`hourly_${hourlyKey}`, hourlyAgg);
      this.aggregatedMetrics.set(`daily_${dailyKey}`, dailyAgg);

      // Store to file
      await this.storeAggregatedMetrics(hourlyKey, hourlyAgg, 'hourly');
      await this.storeAggregatedMetrics(dailyKey, dailyAgg, 'daily');

      this.emit('metrics:aggregated', { hourly: hourlyAgg, daily: dailyAgg });

    } catch (error) {
      console.error('Error aggregating metrics:', error);
      this.emit('aggregation:error', error);
    }
  }

  calculateAggregation(metrics, period) {
    const aggregation = {
      period,
      timestamp: Date.now(),
      sampleCount: metrics.length,
      system: this.aggregateSystemMetrics(metrics),
      application: this.aggregateApplicationMetrics(metrics),
      database: this.aggregateDatabaseMetrics(metrics),
      network: this.aggregateNetworkMetrics(metrics),
      ml_workflow: this.aggregateMLWorkflowMetrics(metrics)
    };

    return aggregation;
  }

  aggregateSystemMetrics(metrics) {
    const systemMetrics = metrics
      .map(m => m.collectors.system)
      .filter(s => s && !s.error);

    if (systemMetrics.length === 0) return null;

    const cpuValues = systemMetrics.map(s => s.cpu?.usage || 0);
    const memoryValues = systemMetrics.map(s => s.memory?.usage || 0);

    return {
      cpu: {
        avg: this.average(cpuValues),
        min: Math.min(...cpuValues),
        max: Math.max(...cpuValues),
        p95: this.percentile(cpuValues, 95)
      },
      memory: {
        avg: this.average(memoryValues),
        min: Math.min(...memoryValues),
        max: Math.max(...memoryValues),
        p95: this.percentile(memoryValues, 95)
      },
      samples: systemMetrics.length
    };
  }

  aggregateApplicationMetrics(metrics) {
    const appMetrics = metrics
      .map(m => m.collectors.application)
      .filter(a => a && !a.error);

    if (appMetrics.length === 0) return null;

    const responseTimeValues = appMetrics.map(a => a.responseTime || 0);
    const throughputValues = appMetrics.map(a => a.throughput || 0);
    const errorRateValues = appMetrics.map(a => a.errorRate || 0);

    return {
      responseTime: {
        avg: this.average(responseTimeValues),
        p50: this.percentile(responseTimeValues, 50),
        p95: this.percentile(responseTimeValues, 95),
        p99: this.percentile(responseTimeValues, 99)
      },
      throughput: {
        avg: this.average(throughputValues),
        max: Math.max(...throughputValues)
      },
      errorRate: {
        avg: this.average(errorRateValues),
        max: Math.max(...errorRateValues)
      },
      samples: appMetrics.length
    };
  }

  aggregateDatabaseMetrics(metrics) {
    const dbMetrics = metrics
      .map(m => m.collectors.database)
      .filter(d => d && !d.error);

    if (dbMetrics.length === 0) return null;

    const queryTimeValues = dbMetrics.map(d => d.queryTime || 0);
    const connectionValues = dbMetrics.map(d => d.connections || 0);
    const cacheHitValues = dbMetrics.map(d => d.cacheHitRatio || 0);

    return {
      queryTime: {
        avg: this.average(queryTimeValues),
        p95: this.percentile(queryTimeValues, 95)
      },
      connections: {
        avg: this.average(connectionValues),
        max: Math.max(...connectionValues)
      },
      cacheHitRatio: {
        avg: this.average(cacheHitValues),
        min: Math.min(...cacheHitValues)
      },
      samples: dbMetrics.length
    };
  }

  aggregateNetworkMetrics(metrics) {
    const netMetrics = metrics
      .map(m => m.collectors.network)
      .filter(n => n && !n.error);

    if (netMetrics.length === 0) return null;

    const bandwidthValues = netMetrics.map(n => n.bandwidth || 0);
    const latencyValues = netMetrics.map(n => n.latency || 0);

    return {
      bandwidth: {
        avg: this.average(bandwidthValues),
        max: Math.max(...bandwidthValues)
      },
      latency: {
        avg: this.average(latencyValues),
        p95: this.percentile(latencyValues, 95)
      },
      samples: netMetrics.length
    };
  }

  aggregateMLWorkflowMetrics(metrics) {
    const mlMetrics = metrics
      .map(m => m.collectors.ml_workflow)
      .filter(ml => ml && !ml.error);

    if (mlMetrics.length === 0) return null;

    const trainingTimeValues = mlMetrics.map(ml => ml.trainingTime || 0).filter(t => t > 0);
    const inferenceTimeValues = mlMetrics.map(ml => ml.inferenceTime || 0).filter(t => t > 0);
    const modelAccuracyValues = mlMetrics.map(ml => ml.modelAccuracy || 0).filter(a => a > 0);

    return {
      trainingTime: trainingTimeValues.length > 0 ? {
        avg: this.average(trainingTimeValues),
        min: Math.min(...trainingTimeValues),
        max: Math.max(...trainingTimeValues)
      } : null,
      inferenceTime: inferenceTimeValues.length > 0 ? {
        avg: this.average(inferenceTimeValues),
        p95: this.percentile(inferenceTimeValues, 95)
      } : null,
      modelAccuracy: modelAccuracyValues.length > 0 ? {
        avg: this.average(modelAccuracyValues),
        latest: modelAccuracyValues[modelAccuracyValues.length - 1]
      } : null,
      samples: mlMetrics.length
    };
  }

  async storeAggregatedMetrics(key, aggregation, period) {
    const filename = `aggregated_${period}_${key}.json`;
    const filepath = path.join(this.config.storageLocation, 'aggregated', filename);
    
    await fs.mkdir(path.dirname(filepath), { recursive: true });
    await fs.writeFile(filepath, JSON.stringify(aggregation, null, 2));
  }

  async cleanupOldMetrics() {
    try {
      const files = await fs.readdir(this.config.storageLocation);
      const cutoffTime = Date.now() - this.config.retentionPeriod;

      for (const file of files) {
        if (file.startsWith('metrics_')) {
          const timestampMatch = file.match(/metrics_(\d+)/);
          if (timestampMatch) {
            const fileTimestamp = parseInt(timestampMatch[1]);
            if (fileTimestamp < cutoffTime) {
              await fs.unlink(path.join(this.config.storageLocation, file));
              console.log(`Cleaned up old metrics file: ${file}`);
            }
          }
        }
      }
    } catch (error) {
      console.error('Error cleaning up old metrics:', error);
    }
  }

  async getMetrics(query = {}) {
    const {
      startTime,
      endTime,
      collector,
      aggregated = false,
      limit = 1000
    } = query;

    try {
      if (aggregated) {
        return await this.getAggregatedMetrics(query);
      }

      // Get raw metrics
      const files = await fs.readdir(this.config.storageLocation);
      const metricsFiles = files
        .filter(f => f.startsWith('metrics_'))
        .sort()
        .reverse()
        .slice(0, limit);

      const allMetrics = [];

      for (const file of metricsFiles) {
        const filepath = path.join(this.config.storageLocation, file);
        const content = await fs.readFile(filepath, 'utf8');
        const data = JSON.parse(content);

        // Filter by time range
        const filteredMetrics = data.metrics.filter(metric => {
          if (startTime && metric.timestamp < startTime) return false;
          if (endTime && metric.timestamp > endTime) return false;
          return true;
        });

        // Filter by collector
        if (collector) {
          filteredMetrics.forEach(metric => {
            if (metric.collectors[collector]) {
              allMetrics.push({
                timestamp: metric.timestamp,
                [collector]: metric.collectors[collector]
              });
            }
          });
        } else {
          allMetrics.push(...filteredMetrics);
        }

        if (allMetrics.length >= limit) break;
      }

      return allMetrics.slice(0, limit);

    } catch (error) {
      console.error('Error retrieving metrics:', error);
      throw error;
    }
  }

  async getAggregatedMetrics(query = {}) {
    const { period = 'hourly', startTime, endTime } = query;
    
    try {
      const aggregatedDir = path.join(this.config.storageLocation, 'aggregated');
      const files = await fs.readdir(aggregatedDir);
      
      const aggregatedFiles = files
        .filter(f => f.startsWith(`aggregated_${period}_`))
        .sort()
        .reverse();

      const aggregatedMetrics = [];

      for (const file of aggregatedFiles) {
        const filepath = path.join(aggregatedDir, file);
        const content = await fs.readFile(filepath, 'utf8');
        const data = JSON.parse(content);

        // Filter by time range
        if (startTime && data.timestamp < startTime) continue;
        if (endTime && data.timestamp > endTime) continue;

        aggregatedMetrics.push(data);
      }

      return aggregatedMetrics;

    } catch (error) {
      console.error('Error retrieving aggregated metrics:', error);
      return [];
    }
  }

  async compressData(data) {
    const zlib = require('zlib');
    const jsonString = JSON.stringify(data);
    return zlib.gzipSync(jsonString);
  }

  // Utility functions
  average(values) {
    return values.length > 0 ? values.reduce((sum, val) => sum + val, 0) / values.length : 0;
  }

  percentile(values, percentile) {
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.ceil((percentile / 100) * sorted.length) - 1;
    return sorted[Math.max(0, Math.min(index, sorted.length - 1))];
  }

  getStatus() {
    return {
      isCollecting: this.isCollecting,
      bufferSize: this.metricsBuffer.length,
      collectors: Array.from(this.collectors.keys()),
      aggregatedMetrics: this.aggregatedMetrics.size,
      config: {
        collectionInterval: this.config.collectionInterval,
        storageInterval: this.config.storageInterval,
        retentionPeriod: this.config.retentionPeriod
      }
    };
  }
}

// Individual collector classes
class SystemMetricsCollector {
  constructor(config) {
    this.config = config;
  }

  async collect() {
    const timestamp = Date.now();
    
    return {
      timestamp,
      cpu: {
        usage: this.getCPUUsage(),
        loadAverage: os.loadavg(),
        cores: os.cpus().length
      },
      memory: {
        used: process.memoryUsage().heapUsed,
        total: process.memoryUsage().heapTotal,
        free: os.freemem(),
        totalSystem: os.totalmem(),
        usage: (process.memoryUsage().rss / os.totalmem()) * 100
      },
      uptime: os.uptime(),
      platform: os.platform()
    };
  }

  getCPUUsage() {
    // Mock CPU usage - in production, use proper CPU monitoring
    const loadAvg = os.loadavg()[0];
    const cores = os.cpus().length;
    return Math.min(100, (loadAvg / cores) * 100);
  }
}

class ApplicationMetricsCollector {
  constructor(config) {
    this.config = config;
    this.requestCount = 0;
    this.errorCount = 0;
    this.responseTimeSum = 0;
    this.lastCollectionTime = Date.now();
  }

  async collect() {
    const now = Date.now();
    const timeDiff = (now - this.lastCollectionTime) / 1000; // seconds
    
    // Mock application metrics
    const newRequests = Math.floor(Math.random() * 100) + 50;
    const newErrors = Math.floor(Math.random() * 5);
    const avgResponseTime = Math.random() * 100 + 20;
    
    this.requestCount += newRequests;
    this.errorCount += newErrors;
    this.responseTimeSum += avgResponseTime * newRequests;
    
    const metrics = {
      timestamp: now,
      requestCount: this.requestCount,
      errorCount: this.errorCount,
      throughput: timeDiff > 0 ? newRequests / timeDiff : 0,
      errorRate: this.requestCount > 0 ? (this.errorCount / this.requestCount) * 100 : 0,
      responseTime: avgResponseTime,
      avgResponseTime: this.requestCount > 0 ? this.responseTimeSum / this.requestCount : 0
    };
    
    this.lastCollectionTime = now;
    return metrics;
  }
}

class DatabaseMetricsCollector {
  constructor(config) {
    this.config = config;
    this.queryCount = 0;
    this.totalQueryTime = 0;
  }

  async collect() {
    // Mock database metrics
    const newQueries = Math.floor(Math.random() * 50) + 10;
    const avgQueryTime = Math.random() * 50 + 5;
    
    this.queryCount += newQueries;
    this.totalQueryTime += avgQueryTime * newQueries;
    
    return {
      timestamp: Date.now(),
      connections: Math.floor(Math.random() * 50) + 20,
      queryCount: this.queryCount,
      queryTime: avgQueryTime,
      avgQueryTime: this.queryCount > 0 ? this.totalQueryTime / this.queryCount : 0,
      cacheHitRatio: 0.7 + Math.random() * 0.25,
      lockWaits: Math.floor(Math.random() * 5),
      deadlocks: Math.floor(Math.random() * 2)
    };
  }
}

class NetworkMetricsCollector {
  constructor(config) {
    this.config = config;
  }

  async collect() {
    return {
      timestamp: Date.now(),
      bandwidth: Math.random() * 100 + 50, // MB/s
      latency: Math.random() * 50 + 10, // ms
      packetLoss: Math.random() * 0.1, // %
      connections: Math.floor(Math.random() * 200) + 100,
      bytesIn: Math.floor(Math.random() * 1000000),
      bytesOut: Math.floor(Math.random() * 1000000)
    };
  }
}

class MLWorkflowMetricsCollector {
  constructor(config) {
    this.config = config;
    this.currentTraining = null;
    this.modelCache = new Map();
  }

  async collect() {
    const metrics = {
      timestamp: Date.now(),
      activeTrainingJobs: Math.floor(Math.random() * 5),
      trainingTime: this.currentTraining ? Date.now() - this.currentTraining.startTime : 0,
      inferenceTime: Math.random() * 100 + 10, // ms
      modelAccuracy: Math.random() * 0.3 + 0.7, // 70-100%
      datasetSize: Math.floor(Math.random() * 100000) + 10000,
      featureCount: Math.floor(Math.random() * 500) + 50,
      memoryUsage: Math.random() * 2000 + 500, // MB
      gpuUtilization: Math.random() * 100 // %
    };

    // Simulate training job lifecycle
    if (!this.currentTraining && Math.random() > 0.9) {
      this.currentTraining = { startTime: Date.now(), duration: Math.random() * 300000 + 60000 };
    }

    if (this.currentTraining && Date.now() - this.currentTraining.startTime > this.currentTraining.duration) {
      this.currentTraining = null;
    }

    return metrics;
  }
}

class CustomMetricsCollector {
  constructor(config) {
    this.config = config;
    this.customMetrics = new Map();
  }

  async collect() {
    // Return any custom metrics that have been registered
    const metrics = {
      timestamp: Date.now(),
      metrics: Object.fromEntries(this.customMetrics)
    };

    // Clear one-time metrics
    for (const [key, value] of this.customMetrics) {
      if (value.oneTime) {
        this.customMetrics.delete(key);
      }
    }

    return metrics;
  }

  addMetric(name, value, options = {}) {
    this.customMetrics.set(name, {
      value,
      timestamp: Date.now(),
      ...options
    });
  }

  removeMetric(name) {
    this.customMetrics.delete(name);
  }
}

module.exports = {
  PerformanceMetricsCollector,
  SystemMetricsCollector,
  ApplicationMetricsCollector,
  DatabaseMetricsCollector,
  NetworkMetricsCollector,
  MLWorkflowMetricsCollector,
  CustomMetricsCollector
};