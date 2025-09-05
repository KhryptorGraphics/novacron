/**
 * Real-time Performance Monitoring Dashboard
 * Interactive dashboard for real-time performance monitoring and visualization
 */

const EventEmitter = require('events');
const WebSocket = require('ws');
const fs = require('fs').promises;
const path = require('path');

class PerformanceMonitoringDashboard extends EventEmitter {
  constructor(framework, config = {}) {
    super();
    
    this.framework = framework;
    this.config = {
      port: config.port || 8080,
      wsPort: config.wsPort || 8081,
      updateInterval: config.updateInterval || 2000, // 2 seconds
      dataRetentionPeriod: config.dataRetentionPeriod || 24 * 60 * 60 * 1000, // 24 hours
      maxDataPoints: config.maxDataPoints || 1000,
      alertThresholds: config.alertThresholds || {},
      ...config
    };

    this.wsServer = null;
    this.connectedClients = new Set();
    this.metricsBuffer = new Map();
    this.realtimeData = {
      system: {},
      benchmarks: {},
      optimizations: {},
      alerts: []
    };
    
    this.isRunning = false;
    this.metricsInterval = null;
  }

  async start() {
    console.log('Starting Performance Monitoring Dashboard...');
    
    // Initialize WebSocket server
    await this.initializeWebSocketServer();
    
    // Start metrics collection
    this.startMetricsCollection();
    
    // Generate dashboard HTML
    await this.generateDashboardHTML();
    
    this.isRunning = true;
    this.emit('dashboard:started', { port: this.config.port, wsPort: this.config.wsPort });
    
    console.log(`Dashboard available at: http://localhost:${this.config.port}`);
    console.log(`WebSocket server listening on port: ${this.config.wsPort}`);
  }

  async stop() {
    console.log('Stopping Performance Monitoring Dashboard...');
    
    this.isRunning = false;
    
    if (this.metricsInterval) {
      clearInterval(this.metricsInterval);
      this.metricsInterval = null;
    }
    
    if (this.wsServer) {
      this.wsServer.close();
    }
    
    this.emit('dashboard:stopped');
    console.log('Dashboard stopped');
  }

  async initializeWebSocketServer() {
    this.wsServer = new WebSocket.Server({ port: this.config.wsPort });
    
    this.wsServer.on('connection', (ws, request) => {
      const clientId = this.generateClientId();
      console.log(`Dashboard client connected: ${clientId}`);
      
      ws.clientId = clientId;
      this.connectedClients.add(ws);
      
      // Send initial data
      this.sendToClient(ws, {
        type: 'initial_data',
        data: this.realtimeData
      });
      
      ws.on('message', (message) => {
        try {
          const data = JSON.parse(message);
          this.handleClientMessage(ws, data);
        } catch (error) {
          console.error('Error parsing client message:', error);
        }
      });
      
      ws.on('close', () => {
        console.log(`Dashboard client disconnected: ${clientId}`);
        this.connectedClients.delete(ws);
      });
      
      ws.on('error', (error) => {
        console.error(`WebSocket error for client ${clientId}:`, error);
        this.connectedClients.delete(ws);
      });
    });
  }

  generateClientId() {
    return `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  handleClientMessage(ws, message) {
    switch (message.type) {
      case 'request_historical_data':
        this.sendHistoricalData(ws, message.params);
        break;
      case 'trigger_benchmark':
        this.triggerBenchmark(message.benchmarkName, message.config);
        break;
      case 'trigger_optimization':
        this.triggerOptimization(message.optimizerType, message.config);
        break;
      case 'update_thresholds':
        this.updateAlertThresholds(message.thresholds);
        break;
      default:
        console.log('Unknown message type:', message.type);
    }
  }

  startMetricsCollection() {
    this.metricsInterval = setInterval(async () => {
      try {
        const metrics = await this.collectRealtimeMetrics();
        this.updateRealtimeData(metrics);
        this.broadcastToClients({
          type: 'metrics_update',
          data: metrics,
          timestamp: Date.now()
        });
      } catch (error) {
        console.error('Error collecting metrics:', error);
      }
    }, this.config.updateInterval);
  }

  async collectRealtimeMetrics() {
    const timestamp = Date.now();
    
    // Collect system metrics
    const systemMetrics = await this.collectSystemMetrics();
    
    // Collect performance metrics
    const performanceMetrics = await this.collectPerformanceMetrics();
    
    // Collect resource utilization
    const resourceMetrics = await this.collectResourceMetrics();
    
    // Check alerts
    const alerts = this.checkAlerts({ systemMetrics, performanceMetrics, resourceMetrics });
    
    return {
      timestamp,
      system: systemMetrics,
      performance: performanceMetrics,
      resources: resourceMetrics,
      alerts
    };
  }

  async collectSystemMetrics() {
    const os = require('os');
    const process = require('process');
    
    return {
      cpu: {
        usage: this.calculateCPUUsage(),
        loadAverage: os.loadavg(),
        cores: os.cpus().length
      },
      memory: {
        used: process.memoryUsage().heapUsed,
        total: process.memoryUsage().heapTotal,
        free: os.freemem(),
        totalSystem: os.totalmem(),
        usage: (process.memoryUsage().heapUsed / os.totalmem()) * 100
      },
      uptime: os.uptime(),
      platform: os.platform(),
      arch: os.arch()
    };
  }

  calculateCPUUsage() {
    // Mock CPU usage calculation
    // In a real implementation, you would use actual CPU monitoring
    return Math.random() * 100;
  }

  async collectPerformanceMetrics() {
    // Mock performance metrics - would integrate with actual benchmark results
    return {
      throughput: {
        current: 150 + Math.random() * 50, // ops/sec
        target: 200,
        trend: 'stable'
      },
      latency: {
        current: 45 + Math.random() * 20, // ms
        p95: 80 + Math.random() * 30,
        p99: 120 + Math.random() * 50,
        target: 50,
        trend: 'improving'
      },
      errorRate: {
        current: Math.random() * 5, // %
        target: 1,
        trend: 'stable'
      },
      availability: {
        current: 99.5 + Math.random() * 0.5, // %
        target: 99.9,
        trend: 'stable'
      }
    };
  }

  async collectResourceMetrics() {
    return {
      database: {
        connections: 45 + Math.floor(Math.random() * 30),
        qps: 120 + Math.floor(Math.random() * 80), // queries per second
        cacheHitRatio: 0.75 + Math.random() * 0.2,
        replicationLag: Math.random() * 100 // ms
      },
      cache: {
        hitRatio: 0.8 + Math.random() * 0.15,
        memoryUsage: 60 + Math.random() * 30, // %
        evictionRate: Math.random() * 10 // evictions/sec
      },
      storage: {
        iops: 800 + Math.floor(Math.random() * 400),
        throughput: 85 + Math.random() * 30, // MB/s
        utilization: 40 + Math.random() * 40, // %
        queueDepth: Math.floor(Math.random() * 20)
      },
      network: {
        bandwidth: 70 + Math.random() * 50, // MB/s
        latency: 25 + Math.random() * 25, // ms
        packetLoss: Math.random() * 0.1, // %
        connections: 100 + Math.floor(Math.random() * 200)
      }
    };
  }

  checkAlerts(metrics) {
    const alerts = [];
    const timestamp = Date.now();
    
    // CPU alerts
    if (metrics.systemMetrics.cpu.usage > 90) {
      alerts.push({
        id: `cpu_high_${timestamp}`,
        type: 'cpu',
        severity: 'critical',
        message: `High CPU usage: ${metrics.systemMetrics.cpu.usage.toFixed(1)}%`,
        value: metrics.systemMetrics.cpu.usage,
        threshold: 90,
        timestamp
      });
    }
    
    // Memory alerts
    if (metrics.systemMetrics.memory.usage > 85) {
      alerts.push({
        id: `memory_high_${timestamp}`,
        type: 'memory',
        severity: 'warning',
        message: `High memory usage: ${metrics.systemMetrics.memory.usage.toFixed(1)}%`,
        value: metrics.systemMetrics.memory.usage,
        threshold: 85,
        timestamp
      });
    }
    
    // Performance alerts
    if (metrics.performanceMetrics.latency.current > metrics.performanceMetrics.latency.target * 2) {
      alerts.push({
        id: `latency_high_${timestamp}`,
        type: 'performance',
        severity: 'warning',
        message: `High latency: ${metrics.performanceMetrics.latency.current.toFixed(1)}ms`,
        value: metrics.performanceMetrics.latency.current,
        threshold: metrics.performanceMetrics.latency.target * 2,
        timestamp
      });
    }
    
    // Cache alerts
    if (metrics.resourceMetrics.cache.hitRatio < 0.7) {
      alerts.push({
        id: `cache_low_${timestamp}`,
        type: 'cache',
        severity: 'info',
        message: `Low cache hit ratio: ${(metrics.resourceMetrics.cache.hitRatio * 100).toFixed(1)}%`,
        value: metrics.resourceMetrics.cache.hitRatio,
        threshold: 0.7,
        timestamp
      });
    }
    
    return alerts;
  }

  updateRealtimeData(metrics) {
    // Store metrics with timestamp
    const key = metrics.timestamp;
    
    // Update system data
    this.realtimeData.system = metrics.system;
    
    // Add to metrics buffer
    if (!this.metricsBuffer.has('system')) {
      this.metricsBuffer.set('system', []);
    }
    
    const systemBuffer = this.metricsBuffer.get('system');
    systemBuffer.push({ timestamp: key, ...metrics.system });
    
    // Maintain buffer size
    if (systemBuffer.length > this.config.maxDataPoints) {
      systemBuffer.shift();
    }
    
    // Update performance data
    if (!this.metricsBuffer.has('performance')) {
      this.metricsBuffer.set('performance', []);
    }
    
    const perfBuffer = this.metricsBuffer.get('performance');
    perfBuffer.push({ timestamp: key, ...metrics.performance });
    
    if (perfBuffer.length > this.config.maxDataPoints) {
      perfBuffer.shift();
    }
    
    // Update resource data
    if (!this.metricsBuffer.has('resources')) {
      this.metricsBuffer.set('resources', []);
    }
    
    const resourceBuffer = this.metricsBuffer.get('resources');
    resourceBuffer.push({ timestamp: key, ...metrics.resources });
    
    if (resourceBuffer.length > this.config.maxDataPoints) {
      resourceBuffer.shift();
    }
    
    // Update alerts (keep last 100)
    this.realtimeData.alerts.push(...metrics.alerts);
    if (this.realtimeData.alerts.length > 100) {
      this.realtimeData.alerts = this.realtimeData.alerts.slice(-100);
    }
  }

  sendToClient(client, message) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify(message));
    }
  }

  broadcastToClients(message) {
    const messageStr = JSON.stringify(message);
    
    this.connectedClients.forEach(client => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(messageStr);
      }
    });
  }

  sendHistoricalData(client, params) {
    const { type, timeRange } = params;
    const endTime = Date.now();
    const startTime = endTime - (timeRange || 3600000); // Default 1 hour
    
    if (this.metricsBuffer.has(type)) {
      const data = this.metricsBuffer.get(type).filter(
        metric => metric.timestamp >= startTime && metric.timestamp <= endTime
      );
      
      this.sendToClient(client, {
        type: 'historical_data',
        dataType: type,
        data,
        timeRange: { startTime, endTime }
      });
    }
  }

  async triggerBenchmark(benchmarkName, config) {
    console.log(`Dashboard triggered benchmark: ${benchmarkName}`);
    
    try {
      // This would integrate with the benchmark framework
      this.broadcastToClients({
        type: 'benchmark_started',
        benchmarkName,
        config,
        timestamp: Date.now()
      });
      
      // Mock benchmark execution
      setTimeout(() => {
        const mockResult = {
          benchmarkName,
          duration: Math.random() * 60000 + 10000, // 10-70 seconds
          success: Math.random() > 0.2, // 80% success rate
          metrics: {
            throughput: Math.random() * 200 + 100,
            latency: Math.random() * 100 + 20,
            errorRate: Math.random() * 5
          }
        };
        
        this.broadcastToClients({
          type: 'benchmark_completed',
          result: mockResult,
          timestamp: Date.now()
        });
        
      }, Math.random() * 30000 + 5000); // 5-35 seconds
      
    } catch (error) {
      this.broadcastToClients({
        type: 'benchmark_failed',
        benchmarkName,
        error: error.message,
        timestamp: Date.now()
      });
    }
  }

  async triggerOptimization(optimizerType, config) {
    console.log(`Dashboard triggered optimization: ${optimizerType}`);
    
    try {
      this.broadcastToClients({
        type: 'optimization_started',
        optimizerType,
        config,
        timestamp: Date.now()
      });
      
      // Mock optimization execution
      setTimeout(() => {
        const mockResult = {
          optimizerType,
          success: Math.random() > 0.15, // 85% success rate
          improvements: {
            performanceGain: Math.random() * 20 + 5, // 5-25% improvement
            resourceSaving: Math.random() * 15 + 3 // 3-18% saving
          },
          strategies: Math.floor(Math.random() * 5) + 2 // 2-6 strategies applied
        };
        
        this.broadcastToClients({
          type: 'optimization_completed',
          result: mockResult,
          timestamp: Date.now()
        });
        
      }, Math.random() * 20000 + 3000); // 3-23 seconds
      
    } catch (error) {
      this.broadcastToClients({
        type: 'optimization_failed',
        optimizerType,
        error: error.message,
        timestamp: Date.now()
      });
    }
  }

  updateAlertThresholds(thresholds) {
    console.log('Updating alert thresholds:', thresholds);
    
    Object.assign(this.config.alertThresholds, thresholds);
    
    this.broadcastToClients({
      type: 'thresholds_updated',
      thresholds: this.config.alertThresholds,
      timestamp: Date.now()
    });
  }

  async generateDashboardHTML() {
    const htmlContent = this.createDashboardHTML();
    const htmlPath = path.join(__dirname, 'dashboard.html');
    
    await fs.writeFile(htmlPath, htmlContent);
    console.log(`Dashboard HTML generated at: ${htmlPath}`);
  }

  createDashboardHTML() {
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NovaCron Performance Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f1419;
            color: #e6e6e6;
            overflow-x: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
            padding: 1rem 2rem;
            border-bottom: 1px solid #4a5568;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        
        .header h1 {
            color: #63b3ed;
            font-size: 1.8rem;
            font-weight: 600;
        }
        
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #48bb78;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
            padding: 1.5rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .card {
            background: linear-gradient(145deg, #1a202c 0%, #2d3748 100%);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #4a5568;
            box-shadow: 0 4px 20px rgba(0,0,0,0.25);
            transition: all 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.35);
        }
        
        .card-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #63b3ed;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 0;
            border-bottom: 1px solid #4a5568;
        }
        
        .metric:last-child {
            border-bottom: none;
        }
        
        .metric-label {
            color: #a0aec0;
            font-weight: 500;
        }
        
        .metric-value {
            font-weight: 600;
            font-size: 1.1rem;
        }
        
        .metric-good { color: #48bb78; }
        .metric-warning { color: #ed8936; }
        .metric-critical { color: #f56565; }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 1rem;
        }
        
        .alerts-container {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .alert {
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            border-radius: 8px;
            border-left: 4px solid;
        }
        
        .alert-critical {
            background: rgba(245, 101, 101, 0.1);
            border-left-color: #f56565;
        }
        
        .alert-warning {
            background: rgba(237, 137, 54, 0.1);
            border-left-color: #ed8936;
        }
        
        .alert-info {
            background: rgba(99, 179, 237, 0.1);
            border-left-color: #63b3ed;
        }
        
        .controls {
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
            flex-wrap: wrap;
        }
        
        .btn {
            background: linear-gradient(145deg, #4299e1, #3182ce);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(66, 153, 225, 0.4);
        }
        
        .btn-secondary {
            background: linear-gradient(145deg, #718096, #4a5568);
        }
        
        .loading {
            text-align: center;
            color: #a0aec0;
            padding: 2rem;
        }
        
        .connection-status {
            position: fixed;
            top: 1rem;
            right: 1rem;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 500;
            z-index: 1000;
        }
        
        .connected {
            background: rgba(72, 187, 120, 0.2);
            color: #48bb78;
            border: 1px solid #48bb78;
        }
        
        .disconnected {
            background: rgba(245, 101, 101, 0.2);
            color: #f56565;
            border: 1px solid #f56565;
        }
    </style>
</head>
<body>
    <div class="connection-status disconnected" id="connection-status">
        Connecting...
    </div>
    
    <header class="header">
        <h1>
            <span class="status-indicator"></span>
            NovaCron Performance Dashboard
        </h1>
    </header>
    
    <div class="container">
        <!-- System Metrics -->
        <div class="card">
            <h2 class="card-title">🖥️ System Metrics</h2>
            <div id="system-metrics" class="loading">Loading...</div>
        </div>
        
        <!-- Performance Metrics -->
        <div class="card">
            <h2 class="card-title">⚡ Performance</h2>
            <div id="performance-metrics" class="loading">Loading...</div>
        </div>
        
        <!-- Resource Utilization -->
        <div class="card">
            <h2 class="card-title">📊 Resources</h2>
            <div id="resource-metrics" class="loading">Loading...</div>
        </div>
        
        <!-- CPU Chart -->
        <div class="card">
            <h2 class="card-title">📈 CPU Usage</h2>
            <div class="chart-container">
                <canvas id="cpu-chart"></canvas>
            </div>
        </div>
        
        <!-- Memory Chart -->
        <div class="card">
            <h2 class="card-title">🧠 Memory Usage</h2>
            <div class="chart-container">
                <canvas id="memory-chart"></canvas>
            </div>
        </div>
        
        <!-- Performance Chart -->
        <div class="card">
            <h2 class="card-title">🎯 Latency & Throughput</h2>
            <div class="chart-container">
                <canvas id="performance-chart"></canvas>
            </div>
        </div>
        
        <!-- Alerts -->
        <div class="card">
            <h2 class="card-title">🚨 Active Alerts</h2>
            <div id="alerts-container" class="alerts-container"></div>
        </div>
        
        <!-- Controls -->
        <div class="card">
            <h2 class="card-title">🎛️ Controls</h2>
            <div class="controls">
                <button class="btn" onclick="triggerBenchmark('comprehensive')">
                    Run Comprehensive Benchmark
                </button>
                <button class="btn" onclick="triggerBenchmark('health-check')">
                    Health Check
                </button>
                <button class="btn btn-secondary" onclick="triggerOptimization('cache')">
                    Optimize Cache
                </button>
                <button class="btn btn-secondary" onclick="triggerOptimization('memory')">
                    Optimize Memory
                </button>
            </div>
        </div>
    </div>
    
    <script>
        // WebSocket connection
        const wsUrl = 'ws://localhost:${this.config.wsPort}';
        let ws;
        let isConnected = false;
        let reconnectInterval;
        
        // Charts
        let cpuChart, memoryChart, performanceChart;
        
        // Data buffers
        const dataBuffers = {
            cpu: [],
            memory: [],
            throughput: [],
            latency: []
        };
        
        function connectWebSocket() {
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                console.log('Connected to dashboard');
                isConnected = true;
                updateConnectionStatus(true);
                
                if (reconnectInterval) {
                    clearInterval(reconnectInterval);
                    reconnectInterval = null;
                }
            };
            
            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                handleMessage(message);
            };
            
            ws.onclose = () => {
                console.log('Disconnected from dashboard');
                isConnected = false;
                updateConnectionStatus(false);
                
                // Attempt to reconnect
                if (!reconnectInterval) {
                    reconnectInterval = setInterval(connectWebSocket, 5000);
                }
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }
        
        function updateConnectionStatus(connected) {
            const statusEl = document.getElementById('connection-status');
            if (connected) {
                statusEl.className = 'connection-status connected';
                statusEl.textContent = 'Connected';
            } else {
                statusEl.className = 'connection-status disconnected';
                statusEl.textContent = 'Disconnected';
            }
        }
        
        function handleMessage(message) {
            switch (message.type) {
                case 'initial_data':
                    initializeUI(message.data);
                    break;
                case 'metrics_update':
                    updateMetrics(message.data);
                    break;
                case 'benchmark_started':
                    showNotification(\`Benchmark started: \${message.benchmarkName}\`, 'info');
                    break;
                case 'benchmark_completed':
                    showNotification(\`Benchmark completed: \${message.result.benchmarkName}\`, 'success');
                    break;
                case 'optimization_started':
                    showNotification(\`Optimization started: \${message.optimizerType}\`, 'info');
                    break;
                case 'optimization_completed':
                    showNotification(\`Optimization completed: \${message.result.optimizerType}\`, 'success');
                    break;
            }
        }
        
        function initializeUI(data) {
            initializeCharts();
            if (data.system) updateSystemMetrics(data.system);
            if (data.performance) updatePerformanceMetrics(data.performance);
            if (data.resources) updateResourceMetrics(data.resources);
        }
        
        function updateMetrics(data) {
            if (data.system) {
                updateSystemMetrics(data.system);
                updateCPUChart(data.system.cpu.usage);
                updateMemoryChart(data.system.memory.usage);
            }
            
            if (data.performance) {
                updatePerformanceMetrics(data.performance);
                updatePerformanceChart(data.performance);
            }
            
            if (data.resources) {
                updateResourceMetrics(data.resources);
            }
            
            if (data.alerts && data.alerts.length > 0) {
                updateAlerts(data.alerts);
            }
        }
        
        function updateSystemMetrics(system) {
            const container = document.getElementById('system-metrics');
            container.innerHTML = \`
                <div class="metric">
                    <span class="metric-label">CPU Usage</span>
                    <span class="metric-value \${getMetricClass(system.cpu.usage, 80, 90)}">\${system.cpu.usage.toFixed(1)}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Memory Usage</span>
                    <span class="metric-value \${getMetricClass(system.memory.usage, 70, 85)}">\${system.memory.usage.toFixed(1)}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Load Average</span>
                    <span class="metric-value">\${system.cpu.loadAverage[0].toFixed(2)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Uptime</span>
                    <span class="metric-value">\${formatUptime(system.uptime)}</span>
                </div>
            \`;
        }
        
        function updatePerformanceMetrics(performance) {
            const container = document.getElementById('performance-metrics');
            container.innerHTML = \`
                <div class="metric">
                    <span class="metric-label">Throughput</span>
                    <span class="metric-value \${performance.throughput.current > performance.throughput.target * 0.8 ? 'metric-good' : 'metric-warning'}">\${performance.throughput.current.toFixed(0)} ops/sec</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Latency</span>
                    <span class="metric-value \${getMetricClass(performance.latency.current, performance.latency.target, performance.latency.target * 2)}">\${performance.latency.current.toFixed(1)} ms</span>
                </div>
                <div class="metric">
                    <span class="metric-label">P99 Latency</span>
                    <span class="metric-value \${getMetricClass(performance.latency.p99, performance.latency.target * 2, performance.latency.target * 4)}">\${performance.latency.p99.toFixed(1)} ms</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Error Rate</span>
                    <span class="metric-value \${getMetricClass(performance.errorRate.current, 2, 5)}">\${performance.errorRate.current.toFixed(2)}%</span>
                </div>
            \`;
        }
        
        function updateResourceMetrics(resources) {
            const container = document.getElementById('resource-metrics');
            container.innerHTML = \`
                <div class="metric">
                    <span class="metric-label">Cache Hit Ratio</span>
                    <span class="metric-value \${resources.cache.hitRatio > 0.8 ? 'metric-good' : 'metric-warning'}">\${(resources.cache.hitRatio * 100).toFixed(1)}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">DB Connections</span>
                    <span class="metric-value">\${resources.database.connections}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Storage IOPS</span>
                    <span class="metric-value \${resources.storage.iops > 1000 ? 'metric-good' : 'metric-warning'}">\${resources.storage.iops.toFixed(0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Network Latency</span>
                    <span class="metric-value \${getMetricClass(resources.network.latency, 50, 100)}">\${resources.network.latency.toFixed(1)} ms</span>
                </div>
            \`;
        }
        
        function initializeCharts() {
            const commonOptions = {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#e6e6e6' } } },
                scales: {
                    x: { ticks: { color: '#a0aec0' }, grid: { color: '#4a5568' } },
                    y: { ticks: { color: '#a0aec0' }, grid: { color: '#4a5568' } }
                }
            };
            
            // CPU Chart
            cpuChart = new Chart(document.getElementById('cpu-chart'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'CPU Usage %',
                        data: [],
                        borderColor: '#63b3ed',
                        backgroundColor: 'rgba(99, 179, 237, 0.1)',
                        tension: 0.4
                    }]
                },
                options: { ...commonOptions, scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, max: 100 } } }
            });
            
            // Memory Chart
            memoryChart = new Chart(document.getElementById('memory-chart'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Memory Usage %',
                        data: [],
                        borderColor: '#ed8936',
                        backgroundColor: 'rgba(237, 137, 54, 0.1)',
                        tension: 0.4
                    }]
                },
                options: { ...commonOptions, scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, max: 100 } } }
            });
            
            // Performance Chart
            performanceChart = new Chart(document.getElementById('performance-chart'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Latency (ms)',
                        data: [],
                        borderColor: '#f56565',
                        backgroundColor: 'rgba(245, 101, 101, 0.1)',
                        yAxisID: 'y'
                    }, {
                        label: 'Throughput (ops/sec)',
                        data: [],
                        borderColor: '#48bb78',
                        backgroundColor: 'rgba(72, 187, 120, 0.1)',
                        yAxisID: 'y1'
                    }]
                },
                options: {
                    ...commonOptions,
                    scales: {
                        ...commonOptions.scales,
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            ticks: { color: '#a0aec0' },
                            grid: { drawOnChartArea: false }
                        }
                    }
                }
            });
        }
        
        function updateCPUChart(cpuUsage) {
            const now = new Date().toLocaleTimeString();
            dataBuffers.cpu.push({ time: now, value: cpuUsage });
            
            if (dataBuffers.cpu.length > 50) dataBuffers.cpu.shift();
            
            cpuChart.data.labels = dataBuffers.cpu.map(d => d.time);
            cpuChart.data.datasets[0].data = dataBuffers.cpu.map(d => d.value);
            cpuChart.update('none');
        }
        
        function updateMemoryChart(memoryUsage) {
            const now = new Date().toLocaleTimeString();
            dataBuffers.memory.push({ time: now, value: memoryUsage });
            
            if (dataBuffers.memory.length > 50) dataBuffers.memory.shift();
            
            memoryChart.data.labels = dataBuffers.memory.map(d => d.time);
            memoryChart.data.datasets[0].data = dataBuffers.memory.map(d => d.value);
            memoryChart.update('none');
        }
        
        function updatePerformanceChart(performance) {
            const now = new Date().toLocaleTimeString();
            dataBuffers.throughput.push({ time: now, value: performance.throughput.current });
            dataBuffers.latency.push({ time: now, value: performance.latency.current });
            
            if (dataBuffers.throughput.length > 50) {
                dataBuffers.throughput.shift();
                dataBuffers.latency.shift();
            }
            
            performanceChart.data.labels = dataBuffers.throughput.map(d => d.time);
            performanceChart.data.datasets[0].data = dataBuffers.latency.map(d => d.value);
            performanceChart.data.datasets[1].data = dataBuffers.throughput.map(d => d.value);
            performanceChart.update('none');
        }
        
        function updateAlerts(alerts) {
            const container = document.getElementById('alerts-container');
            
            if (alerts.length === 0) {
                container.innerHTML = '<div class="loading">No active alerts</div>';
                return;
            }
            
            container.innerHTML = alerts.map(alert => \`
                <div class="alert alert-\${alert.severity}">
                    <strong>[\${alert.severity.toUpperCase()}]</strong> \${alert.message}
                    <div style="font-size: 0.875rem; color: #a0aec0; margin-top: 0.25rem;">
                        \${new Date(alert.timestamp).toLocaleString()}
                    </div>
                </div>
            \`).join('');
        }
        
        function getMetricClass(value, warning, critical) {
            if (value >= critical) return 'metric-critical';
            if (value >= warning) return 'metric-warning';
            return 'metric-good';
        }
        
        function formatUptime(seconds) {
            const days = Math.floor(seconds / 86400);
            const hours = Math.floor((seconds % 86400) / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            
            if (days > 0) return \`\${days}d \${hours}h \${minutes}m\`;
            if (hours > 0) return \`\${hours}h \${minutes}m\`;
            return \`\${minutes}m\`;
        }
        
        function showNotification(message, type) {
            // Simple notification system
            const notification = document.createElement('div');
            notification.style.cssText = \`
                position: fixed;
                top: 80px;
                right: 20px;
                padding: 1rem 1.5rem;
                border-radius: 8px;
                color: white;
                font-weight: 500;
                z-index: 2000;
                animation: slideIn 0.3s ease;
                background: \${type === 'success' ? '#48bb78' : type === 'info' ? '#63b3ed' : '#f56565'};
            \`;
            notification.textContent = message;
            
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.style.animation = 'slideOut 0.3s ease';
                setTimeout(() => document.body.removeChild(notification), 300);
            }, 3000);
        }
        
        // Control functions
        function triggerBenchmark(type) {
            if (!isConnected) {
                showNotification('Not connected to server', 'error');
                return;
            }
            
            ws.send(JSON.stringify({
                type: 'trigger_benchmark',
                benchmarkName: type,
                config: {}
            }));
        }
        
        function triggerOptimization(type) {
            if (!isConnected) {
                showNotification('Not connected to server', 'error');
                return;
            }
            
            ws.send(JSON.stringify({
                type: 'trigger_optimization',
                optimizerType: type,
                config: {}
            }));
        }
        
        // Add CSS animations
        const style = document.createElement('style');
        style.textContent = \`
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        \`;
        document.head.appendChild(style);
        
        // Initialize
        connectWebSocket();
    </script>
</body>
</html>`;
  }

  getStatus() {
    return {
      running: this.isRunning,
      connectedClients: this.connectedClients.size,
      port: this.config.port,
      wsPort: this.config.wsPort,
      updateInterval: this.config.updateInterval,
      bufferSizes: {
        system: this.metricsBuffer.get('system')?.length || 0,
        performance: this.metricsBuffer.get('performance')?.length || 0,
        resources: this.metricsBuffer.get('resources')?.length || 0
      }
    };
  }
}

module.exports = PerformanceMonitoringDashboard;