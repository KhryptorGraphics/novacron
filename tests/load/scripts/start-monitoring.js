const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');

class LoadTestMonitor {
  constructor() {
    this.monitoringInterval = 5000; // 5 seconds
    this.metricsHistory = [];
    this.alertThresholds = {
      cpuUsage: 80,
      memoryUsage: 85,
      diskUsage: 90,
      responseTime: 1000,
      errorRate: 5,
      activeConnections: 1000
    };
    this.isMonitoring = false;
    this.monitoringProcesses = [];
    this.reportDir = path.join(__dirname, '../reports');
    this.monitoringStartTime = Date.now();
  }

  async startMonitoring() {
    console.log('Starting comprehensive load test monitoring...');
    this.isMonitoring = true;

    // Ensure reports directory exists
    if (!fs.existsSync(this.reportDir)) {
      fs.mkdirSync(this.reportDir, { recursive: true });
    }

    try {
      // Start monitoring components in parallel
      await Promise.all([
        this.startSystemMonitoring(),
        this.startApplicationMonitoring(),
        this.startNetworkMonitoring(),
        this.startDatabaseMonitoring(),
        this.startWebSocketMonitoring(),
        this.startPrometheusCollection(),
        this.startLogAggregation()
      ]);

      console.log('All monitoring components started successfully');
      console.log(`Monitoring data will be saved to: ${this.reportDir}`);
      
      // Start the main monitoring loop
      this.monitoringLoop();

    } catch (error) {
      console.error('Failed to start monitoring:', error);
      this.stopMonitoring();
      process.exit(1);
    }
  }

  async startSystemMonitoring() {
    console.log('Starting system resource monitoring...');
    
    // Monitor CPU, Memory, Disk, Network
    const systemMetricsFile = path.join(this.reportDir, 'system-metrics.jsonl');
    
    setInterval(() => {
      if (!this.isMonitoring) return;

      try {
        // Get system metrics (Linux-specific commands)
        const cpuUsage = this.getCPUUsage();
        const memoryUsage = this.getMemoryUsage();
        const diskUsage = this.getDiskUsage();
        const networkStats = this.getNetworkStats();
        const processStats = this.getProcessStats();

        const systemMetrics = {
          timestamp: new Date().toISOString(),
          cpu: cpuUsage,
          memory: memoryUsage,
          disk: diskUsage,
          network: networkStats,
          processes: processStats
        };

        // Append to JSONL file for time-series data
        fs.appendFileSync(systemMetricsFile, JSON.stringify(systemMetrics) + '\n');
        
        // Check for alerts
        this.checkSystemAlerts(systemMetrics);

      } catch (error) {
        console.error('System monitoring error:', error);
      }
    }, this.monitoringInterval);
  }

  async startApplicationMonitoring() {
    console.log('Starting application performance monitoring...');
    
    const appMetricsFile = path.join(this.reportDir, 'application-metrics.jsonl');
    const environment = process.env.ENVIRONMENT || 'local';
    const baseURL = this.getBaseURL(environment);

    setInterval(async () => {
      if (!this.isMonitoring) return;

      try {
        // Test application health and performance
        const healthCheck = await this.performHealthCheck(baseURL);
        const apiMetrics = await this.collectAPIMetrics(baseURL);
        const vmMetrics = await this.collectVMMetrics(baseURL);

        const appMetrics = {
          timestamp: new Date().toISOString(),
          health: healthCheck,
          api: apiMetrics,
          vms: vmMetrics
        };

        fs.appendFileSync(appMetricsFile, JSON.stringify(appMetrics) + '\n');
        
        // Check application alerts
        this.checkApplicationAlerts(appMetrics);

      } catch (error) {
        console.error('Application monitoring error:', error);
      }
    }, this.monitoringInterval * 2); // Less frequent for API calls
  }

  async startNetworkMonitoring() {
    console.log('Starting network performance monitoring...');
    
    const networkMetricsFile = path.join(this.reportDir, 'network-metrics.jsonl');

    setInterval(() => {
      if (!this.isMonitoring) return;

      try {
        const networkMetrics = {
          timestamp: new Date().toISOString(),
          interfaces: this.getNetworkInterfaceStats(),
          connections: this.getNetworkConnections(),
          latency: this.measureNetworkLatency(),
          bandwidth: this.measureBandwidthUtilization()
        };

        fs.appendFileSync(networkMetricsFile, JSON.stringify(networkMetrics) + '\n');

      } catch (error) {
        console.error('Network monitoring error:', error);
      }
    }, this.monitoringInterval);
  }

  async startDatabaseMonitoring() {
    console.log('Starting database performance monitoring...');
    
    const dbMetricsFile = path.join(this.reportDir, 'database-metrics.jsonl');

    setInterval(async () => {
      if (!this.isMonitoring) return;

      try {
        const dbMetrics = {
          timestamp: new Date().toISOString(),
          connections: this.getDatabaseConnections(),
          queryPerformance: await this.measureDatabasePerformance(),
          lockStats: this.getDatabaseLockStats(),
          cacheHitRatio: this.getDatabaseCacheStats()
        };

        fs.appendFileSync(dbMetricsFile, JSON.stringify(dbMetrics) + '\n');
        
        // Check database alerts
        this.checkDatabaseAlerts(dbMetrics);

      } catch (error) {
        console.error('Database monitoring error:', error);
      }
    }, this.monitoringInterval * 3); // Less frequent for DB queries
  }

  async startWebSocketMonitoring() {
    console.log('Starting WebSocket connection monitoring...');
    
    const wsMetricsFile = path.join(this.reportDir, 'websocket-metrics.jsonl');

    setInterval(() => {
      if (!this.isMonitoring) return;

      try {
        const wsMetrics = {
          timestamp: new Date().toISOString(),
          activeConnections: this.getWebSocketConnections(),
          messageQueues: this.getWebSocketQueueStats(),
          connectionPool: this.getWebSocketPoolStats(),
          protocolStats: this.getWebSocketProtocolStats()
        };

        fs.appendFileSync(wsMetricsFile, JSON.stringify(wsMetrics) + '\n');

      } catch (error) {
        console.error('WebSocket monitoring error:', error);
      }
    }, this.monitoringInterval);
  }

  async startPrometheusCollection() {
    console.log('Starting Prometheus metrics collection...');
    
    const prometheusURL = process.env.PROMETHEUS_URL || 'http://localhost:9090';
    const prometheusFile = path.join(this.reportDir, 'prometheus-metrics.jsonl');

    setInterval(async () => {
      if (!this.isMonitoring) return;

      try {
        // Query key Prometheus metrics
        const queries = [
          'novacron_http_requests_total',
          'novacron_vm_operations_total',
          'novacron_websocket_connections_active',
          'novacron_database_queries_total',
          'go_memstats_alloc_bytes',
          'go_goroutines'
        ];

        const metrics = {};
        for (const query of queries) {
          const result = await this.queryPrometheus(prometheusURL, query);
          metrics[query] = result;
        }

        const prometheusMetrics = {
          timestamp: new Date().toISOString(),
          metrics: metrics
        };

        fs.appendFileSync(prometheusFile, JSON.stringify(prometheusMetrics) + '\n');

      } catch (error) {
        console.error('Prometheus collection error:', error);
      }
    }, this.monitoringInterval * 2);
  }

  async startLogAggregation() {
    console.log('Starting log aggregation...');
    
    const logFile = path.join(this.reportDir, 'test-logs.jsonl');
    
    // Monitor application logs
    const logSources = [
      '/var/log/novacron/api.log',
      '/var/log/novacron/vm-manager.log',
      '/var/log/novacron/websocket.log',
      './backend/logs/application.log'
    ];

    // Use tail -f to monitor log files
    for (const logSource of logSources) {
      if (fs.existsSync(logSource)) {
        const tailProcess = spawn('tail', ['-f', logSource]);
        
        tailProcess.stdout.on('data', (data) => {
          const logEntry = {
            timestamp: new Date().toISOString(),
            source: logSource,
            message: data.toString().trim()
          };
          
          fs.appendFileSync(logFile, JSON.stringify(logEntry) + '\n');
        });

        this.monitoringProcesses.push(tailProcess);
      }
    }
  }

  monitoringLoop() {
    const loopInterval = setInterval(() => {
      if (!this.isMonitoring) {
        clearInterval(loopInterval);
        return;
      }

      // Generate real-time monitoring dashboard data
      this.generateRealtimeDashboard();
      
      // Check for critical alerts
      this.performAlertCheck();
      
      // Update monitoring status
      const duration = Date.now() - this.monitoringStartTime;
      console.log(`Monitoring active for ${Math.floor(duration / 1000)} seconds`);

    }, 30000); // Every 30 seconds
  }

  generateRealtimeDashboard() {
    try {
      const latestMetrics = this.getLatestMetrics();
      const dashboardData = {
        timestamp: new Date().toISOString(),
        status: this.getOverallSystemStatus(),
        metrics: latestMetrics,
        alerts: this.getActiveAlerts(),
        uptime: Date.now() - this.monitoringStartTime
      };

      const dashboardFile = path.join(this.reportDir, 'realtime-dashboard.json');
      fs.writeFileSync(dashboardFile, JSON.stringify(dashboardData, null, 2));

    } catch (error) {
      console.error('Dashboard generation error:', error);
    }
  }

  getLatestMetrics() {
    return {
      system: this.getLatestSystemMetrics(),
      application: this.getLatestApplicationMetrics(),
      network: this.getLatestNetworkMetrics(),
      database: this.getLatestDatabaseMetrics()
    };
  }

  getLatestSystemMetrics() {
    try {
      return {
        cpu: this.getCPUUsage(),
        memory: this.getMemoryUsage(),
        disk: this.getDiskUsage(),
        load: this.getLoadAverage()
      };
    } catch (error) {
      return { error: error.message };
    }
  }

  getLatestApplicationMetrics() {
    // Get application metrics from last health check
    return {
      responseTime: 'N/A',
      throughput: 'N/A',
      errorRate: 'N/A',
      activeUsers: 'N/A'
    };
  }

  getLatestNetworkMetrics() {
    try {
      return this.getNetworkStats();
    } catch (error) {
      return { error: error.message };
    }
  }

  getLatestDatabaseMetrics() {
    return {
      connections: 'N/A',
      queryTime: 'N/A',
      cacheHitRatio: 'N/A'
    };
  }

  // System metric collection methods
  getCPUUsage() {
    try {
      const output = execSync("top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | sed 's/%us,//'", { encoding: 'utf8' });
      return parseFloat(output.trim()) || 0;
    } catch (error) {
      return 0;
    }
  }

  getMemoryUsage() {
    try {
      const output = execSync("free -m | awk 'NR==2{printf \"%.2f\", $3*100/$2}'", { encoding: 'utf8' });
      return {
        usagePercent: parseFloat(output) || 0,
        total: this.getMemoryTotal(),
        used: this.getMemoryUsed(),
        free: this.getMemoryFree()
      };
    } catch (error) {
      return { usagePercent: 0, total: 0, used: 0, free: 0 };
    }
  }

  getDiskUsage() {
    try {
      const output = execSync("df -h / | awk 'NR==2{print $5}' | sed 's/%//'", { encoding: 'utf8' });
      return {
        usagePercent: parseInt(output.trim()) || 0,
        total: this.getDiskTotal(),
        used: this.getDiskUsed(),
        free: this.getDiskFree()
      };
    } catch (error) {
      return { usagePercent: 0, total: 0, used: 0, free: 0 };
    }
  }

  getNetworkStats() {
    try {
      // Get network interface statistics
      const interfaces = execSync("cat /proc/net/dev | tail -n +3", { encoding: 'utf8' });
      const stats = {};
      
      interfaces.split('\n').forEach(line => {
        if (line.trim()) {
          const parts = line.trim().split(/\s+/);
          const interfaceName = parts[0].replace(':', '');
          stats[interfaceName] = {
            bytesReceived: parseInt(parts[1]) || 0,
            bytesTransmitted: parseInt(parts[9]) || 0,
            packetsReceived: parseInt(parts[2]) || 0,
            packetsTransmitted: parseInt(parts[10]) || 0
          };
        }
      });
      
      return stats;
    } catch (error) {
      return {};
    }
  }

  getProcessStats() {
    try {
      // Get NovaCron process statistics
      const processes = execSync("ps aux | grep novacron | grep -v grep", { encoding: 'utf8' });
      const stats = [];
      
      processes.split('\n').forEach(line => {
        if (line.trim()) {
          const parts = line.trim().split(/\s+/);
          stats.push({
            pid: parts[1],
            cpu: parseFloat(parts[2]) || 0,
            memory: parseFloat(parts[3]) || 0,
            command: parts.slice(10).join(' ')
          });
        }
      });
      
      return stats;
    } catch (error) {
      return [];
    }
  }

  async performHealthCheck(baseURL) {
    return new Promise((resolve) => {
      const startTime = Date.now();
      
      const req = http.get(`${baseURL}/api/cluster/health`, (res) => {
        const responseTime = Date.now() - startTime;
        let data = '';
        
        res.on('data', chunk => data += chunk);
        res.on('end', () => {
          resolve({
            status: res.statusCode,
            responseTime: responseTime,
            data: data ? JSON.parse(data) : null
          });
        });
      });

      req.on('error', (error) => {
        resolve({
          status: 0,
          responseTime: Date.now() - startTime,
          error: error.message
        });
      });

      req.setTimeout(10000, () => {
        req.destroy();
        resolve({
          status: 0,
          responseTime: 10000,
          error: 'Timeout'
        });
      });
    });
  }

  async collectAPIMetrics(baseURL) {
    const endpoints = [
      '/api/vms',
      '/api/storage/volumes',
      '/api/cluster/nodes',
      '/api/monitoring/metrics'
    ];

    const metrics = {
      endpoints: {},
      averageResponseTime: 0,
      totalRequests: 0,
      errors: 0
    };

    for (const endpoint of endpoints) {
      try {
        const startTime = Date.now();
        const health = await this.performHealthCheck(`${baseURL}${endpoint}`);
        const responseTime = Date.now() - startTime;
        
        metrics.endpoints[endpoint] = {
          responseTime: responseTime,
          status: health.status,
          success: health.status === 200
        };
        
        metrics.totalRequests++;
        metrics.averageResponseTime += responseTime;
        
        if (health.status !== 200) {
          metrics.errors++;
        }
        
      } catch (error) {
        metrics.endpoints[endpoint] = {
          responseTime: 0,
          status: 0,
          error: error.message
        };
        metrics.errors++;
      }
    }

    if (metrics.totalRequests > 0) {
      metrics.averageResponseTime /= metrics.totalRequests;
    }

    return metrics;
  }

  async collectVMMetrics(baseURL) {
    try {
      const vmHealth = await this.performHealthCheck(`${baseURL}/api/vms`);
      const vmData = vmHealth.data || [];
      
      return {
        totalVMs: Array.isArray(vmData) ? vmData.length : 0,
        runningVMs: Array.isArray(vmData) ? vmData.filter(vm => vm.state === 'running').length : 0,
        vmStates: this.aggregateVMStates(vmData)
      };
    } catch (error) {
      return { error: error.message };
    }
  }

  checkSystemAlerts(metrics) {
    const alerts = [];

    if (metrics.cpu > this.alertThresholds.cpuUsage) {
      alerts.push({
        type: 'system',
        severity: 'warning',
        message: `High CPU usage: ${metrics.cpu}%`,
        threshold: this.alertThresholds.cpuUsage
      });
    }

    if (metrics.memory.usagePercent > this.alertThresholds.memoryUsage) {
      alerts.push({
        type: 'system',
        severity: 'warning',
        message: `High memory usage: ${metrics.memory.usagePercent}%`,
        threshold: this.alertThresholds.memoryUsage
      });
    }

    if (metrics.disk.usagePercent > this.alertThresholds.diskUsage) {
      alerts.push({
        type: 'system',
        severity: 'critical',
        message: `High disk usage: ${metrics.disk.usagePercent}%`,
        threshold: this.alertThresholds.diskUsage
      });
    }

    if (alerts.length > 0) {
      this.logAlerts(alerts);
    }
  }

  checkApplicationAlerts(metrics) {
    const alerts = [];

    if (metrics.api.averageResponseTime > this.alertThresholds.responseTime) {
      alerts.push({
        type: 'application',
        severity: 'warning',
        message: `High API response time: ${metrics.api.averageResponseTime}ms`,
        threshold: this.alertThresholds.responseTime
      });
    }

    const errorRate = metrics.api.totalRequests > 0 ? 
      (metrics.api.errors / metrics.api.totalRequests) * 100 : 0;
    
    if (errorRate > this.alertThresholds.errorRate) {
      alerts.push({
        type: 'application',
        severity: 'critical',
        message: `High error rate: ${errorRate.toFixed(2)}%`,
        threshold: this.alertThresholds.errorRate
      });
    }

    if (alerts.length > 0) {
      this.logAlerts(alerts);
    }
  }

  logAlerts(alerts) {
    const alertsFile = path.join(this.reportDir, 'monitoring-alerts.jsonl');
    
    alerts.forEach(alert => {
      const alertEntry = {
        ...alert,
        timestamp: new Date().toISOString(),
        monitoringSession: this.monitoringStartTime
      };
      
      fs.appendFileSync(alertsFile, JSON.stringify(alertEntry) + '\n');
      console.warn(`🚨 ALERT [${alert.severity.toUpperCase()}]: ${alert.message}`);
    });
  }

  getBaseURL(environment) {
    const urls = {
      local: 'http://localhost:8080',
      staging: 'https://staging.novacron.com',
      production: 'https://api.novacron.com'
    };
    return urls[environment] || urls.local;
  }

  stopMonitoring() {
    console.log('Stopping load test monitoring...');
    this.isMonitoring = false;

    // Stop all monitoring processes
    this.monitoringProcesses.forEach(process => {
      try {
        process.kill('SIGTERM');
      } catch (error) {
        console.error('Error stopping monitoring process:', error);
      }
    });

    // Generate final monitoring report
    this.generateMonitoringReport();
    
    console.log('Monitoring stopped successfully');
  }

  generateMonitoringReport() {
    const duration = Date.now() - this.monitoringStartTime;
    const report = {
      monitoringDuration: duration,
      metricsCollected: this.metricsHistory.length,
      alertsGenerated: this.countAlerts(),
      finalSystemState: this.getLatestMetrics(),
      summary: 'Monitoring completed successfully'
    };

    const reportFile = path.join(this.reportDir, `monitoring-report-${new Date().toISOString().replace(/[:.]/g, '-')}.json`);
    fs.writeFileSync(reportFile, JSON.stringify(report, null, 2));
    
    console.log(`Monitoring report saved: ${reportFile}`);
  }

  countAlerts() {
    const alertsFile = path.join(this.reportDir, 'monitoring-alerts.jsonl');
    if (fs.existsSync(alertsFile)) {
      return fs.readFileSync(alertsFile, 'utf8').split('\n').filter(line => line.trim()).length;
    }
    return 0;
  }

  // Placeholder methods for metrics collection
  getMemoryTotal() { return 0; }
  getMemoryUsed() { return 0; }
  getMemoryFree() { return 0; }
  getDiskTotal() { return 0; }
  getDiskUsed() { return 0; }
  getDiskFree() { return 0; }
  getLoadAverage() { return [0, 0, 0]; }
  getNetworkInterfaceStats() { return {}; }
  getNetworkConnections() { return 0; }
  measureNetworkLatency() { return 0; }
  measureBandwidthUtilization() { return 0; }
  getDatabaseConnections() { return 0; }
  async measureDatabasePerformance() { return {}; }
  getDatabaseLockStats() { return {}; }
  getDatabaseCacheStats() { return 0; }
  getWebSocketConnections() { return 0; }
  getWebSocketQueueStats() { return {}; }
  getWebSocketPoolStats() { return {}; }
  getWebSocketProtocolStats() { return {}; }
  async queryPrometheus(url, query) { return {}; }
  aggregateVMStates(vms) { return {}; }
  getOverallSystemStatus() { return 'healthy'; }
  getActiveAlerts() { return []; }
  checkDatabaseAlerts() { }
  performAlertCheck() { }
}

// CLI execution
if (require.main === module) {
  const monitor = new LoadTestMonitor();
  
  // Handle graceful shutdown
  process.on('SIGINT', () => {
    console.log('\nReceived SIGINT, stopping monitoring...');
    monitor.stopMonitoring();
    process.exit(0);
  });

  process.on('SIGTERM', () => {
    console.log('\nReceived SIGTERM, stopping monitoring...');
    monitor.stopMonitoring();
    process.exit(0);
  });

  monitor.startMonitoring()
    .then(() => {
      console.log('Monitoring started successfully. Press Ctrl+C to stop.');
    })
    .catch((error) => {
      console.error('Failed to start monitoring:', error);
      process.exit(1);
    });
}

module.exports = LoadTestMonitor;