/**
 * Database Performance Testing Suite
 * Comprehensive performance benchmarking and load testing framework
 */

const { performance } = require('perf_hooks');
const EventEmitter = require('events');
const fs = require('fs').promises;
const path = require('path');

class DatabasePerformanceTestSuite extends EventEmitter {
  constructor(config = {}) {
    super();
    this.config = {
      maxConcurrency: config.maxConcurrency || 100,
      testDuration: config.testDuration || 60000, // 1 minute
      warmupDuration: config.warmupDuration || 10000, // 10 seconds
      cooldownDuration: config.cooldownDuration || 5000, // 5 seconds
      metricsInterval: config.metricsInterval || 1000, // 1 second
      outputDir: config.outputDir || './tests/performance/reports',
      ...config
    };
    
    this.metrics = {
      queries: [],
      throughput: [],
      latency: [],
      errors: [],
      resources: [],
      connections: []
    };
    
    this.activeConnections = 0;
    this.totalQueries = 0;
    this.totalErrors = 0;
    this.testStartTime = null;
    this.isRunning = false;
  }

  /**
   * Initialize the performance testing suite
   */
  async initialize() {
    console.log('🚀 Initializing Database Performance Testing Suite...');
    
    // Ensure output directory exists
    await fs.mkdir(this.config.outputDir, { recursive: true });
    
    // Initialize metrics collection
    this.metricsCollector = setInterval(() => {
      this.collectMetrics();
    }, this.config.metricsInterval);
    
    this.emit('initialized');
    console.log('✅ Performance testing suite initialized');
  }

  /**
   * Run comprehensive performance benchmark
   */
  async runBenchmark(testScenarios) {
    console.log('🎯 Starting comprehensive performance benchmark...');
    this.isRunning = true;
    this.testStartTime = performance.now();
    
    const results = {
      testSuite: 'Database Performance Benchmark',
      startTime: new Date().toISOString(),
      scenarios: []
    };

    try {
      // Warmup phase
      console.log('🔥 Warming up database connections...');
      await this.warmupPhase();
      
      // Run each test scenario
      for (const scenario of testScenarios) {
        console.log(`📊 Running scenario: ${scenario.name}`);
        const scenarioResult = await this.runScenario(scenario);
        results.scenarios.push(scenarioResult);
        
        // Cooldown between scenarios
        if (scenario !== testScenarios[testScenarios.length - 1]) {
          console.log('❄️ Cooling down between scenarios...');
          await this.sleep(this.config.cooldownDuration);
        }
      }
      
      results.endTime = new Date().toISOString();
      results.totalDuration = performance.now() - this.testStartTime;
      results.summary = this.generateSummary();
      
      // Save results
      const reportPath = path.join(
        this.config.outputDir, 
        `benchmark-${Date.now()}.json`
      );
      await fs.writeFile(reportPath, JSON.stringify(results, null, 2));
      console.log(`📄 Benchmark results saved to: ${reportPath}`);
      
      return results;
      
    } catch (error) {
      console.error('❌ Benchmark failed:', error);
      throw error;
    } finally {
      this.isRunning = false;
      clearInterval(this.metricsCollector);
    }
  }

  /**
   * Run individual test scenario
   */
  async runScenario(scenario) {
    const scenarioStartTime = performance.now();
    const scenarioMetrics = {
      name: scenario.name,
      type: scenario.type,
      concurrency: scenario.concurrency || 10,
      duration: scenario.duration || this.config.testDuration,
      queries: [],
      latency: { min: Infinity, max: 0, avg: 0, p95: 0, p99: 0 },
      throughput: 0,
      errorRate: 0,
      resourceUsage: {}
    };

    const workers = [];
    const startTime = performance.now();
    
    // Create worker promises
    for (let i = 0; i < scenarioMetrics.concurrency; i++) {
      workers.push(this.runWorker(scenario, scenarioMetrics.duration, i));
    }
    
    // Wait for all workers to complete
    const workerResults = await Promise.allSettled(workers);
    
    const endTime = performance.now();
    const totalDuration = endTime - startTime;
    
    // Process worker results
    const allQueries = [];
    let totalErrors = 0;
    
    workerResults.forEach((result, index) => {
      if (result.status === 'fulfilled') {
        allQueries.push(...result.value.queries);
        totalErrors += result.value.errors;
      } else {
        console.error(`Worker ${index} failed:`, result.reason);
        totalErrors++;
      }
    });
    
    // Calculate metrics
    if (allQueries.length > 0) {
      const latencies = allQueries.map(q => q.latency).sort((a, b) => a - b);
      scenarioMetrics.latency = {
        min: Math.min(...latencies),
        max: Math.max(...latencies),
        avg: latencies.reduce((sum, l) => sum + l, 0) / latencies.length,
        p95: latencies[Math.floor(latencies.length * 0.95)],
        p99: latencies[Math.floor(latencies.length * 0.99)]
      };
      
      scenarioMetrics.throughput = (allQueries.length / totalDuration) * 1000; // queries per second
      scenarioMetrics.errorRate = (totalErrors / (allQueries.length + totalErrors)) * 100;
    }
    
    scenarioMetrics.queries = allQueries;
    scenarioMetrics.totalDuration = totalDuration;
    
    console.log(`✅ Scenario '${scenario.name}' completed:`);
    console.log(`   Throughput: ${scenarioMetrics.throughput.toFixed(2)} queries/sec`);
    console.log(`   Avg Latency: ${scenarioMetrics.latency.avg.toFixed(2)}ms`);
    console.log(`   Error Rate: ${scenarioMetrics.errorRate.toFixed(2)}%`);
    
    return scenarioMetrics;
  }

  /**
   * Run individual worker thread
   */
  async runWorker(scenario, duration, workerId) {
    const queries = [];
    let errors = 0;
    const startTime = performance.now();
    
    while (performance.now() - startTime < duration) {
      try {
        const queryStart = performance.now();
        
        // Execute the scenario's operation
        await this.executeOperation(scenario.operation, workerId);
        
        const queryEnd = performance.now();
        const latency = queryEnd - queryStart;
        
        queries.push({
          workerId,
          timestamp: queryEnd,
          latency,
          operation: scenario.operation.type
        });
        
        this.totalQueries++;
        
        // Optional delay between queries
        if (scenario.delayMs) {
          await this.sleep(scenario.delayMs);
        }
        
      } catch (error) {
        errors++;
        this.totalErrors++;
        console.error(`Worker ${workerId} query failed:`, error.message);
      }
    }
    
    return { queries, errors };
  }

  /**
   * Execute database operation
   */
  async executeOperation(operation, workerId) {
    switch (operation.type) {
      case 'select':
        return await this.executeSelect(operation, workerId);
      case 'insert':
        return await this.executeInsert(operation, workerId);
      case 'update':
        return await this.executeUpdate(operation, workerId);
      case 'delete':
        return await this.executeDelete(operation, workerId);
      case 'transaction':
        return await this.executeTransaction(operation, workerId);
      case 'join':
        return await this.executeJoin(operation, workerId);
      case 'aggregate':
        return await this.executeAggregate(operation, workerId);
      default:
        throw new Error(`Unknown operation type: ${operation.type}`);
    }
  }

  /**
   * Execute SELECT operations
   */
  async executeSelect(operation, workerId) {
    // Simulate database connection and query
    const connection = await this.getConnection(workerId);
    
    try {
      // Simulate query execution time based on complexity
      const baseLatency = operation.complexity === 'high' ? 50 : 
                         operation.complexity === 'medium' ? 20 : 5;
      const jitter = Math.random() * 10;
      
      await this.sleep(baseLatency + jitter);
      
      // Simulate different result set sizes
      const resultSize = operation.resultSize || 'medium';
      const rowCount = resultSize === 'large' ? 1000 : 
                      resultSize === 'medium' ? 100 : 10;
      
      return { rowCount, cached: Math.random() < 0.1 }; // 10% cache hit rate
      
    } finally {
      this.releaseConnection(connection, workerId);
    }
  }

  /**
   * Execute INSERT operations
   */
  async executeInsert(operation, workerId) {
    const connection = await this.getConnection(workerId);
    
    try {
      // Simulate insert latency
      const batchSize = operation.batchSize || 1;
      const latency = 10 + (batchSize * 2) + Math.random() * 5;
      
      await this.sleep(latency);
      
      return { rowsAffected: batchSize };
      
    } finally {
      this.releaseConnection(connection, workerId);
    }
  }

  /**
   * Execute UPDATE operations
   */
  async executeUpdate(operation, workerId) {
    const connection = await this.getConnection(workerId);
    
    try {
      const latency = 15 + Math.random() * 10;
      await this.sleep(latency);
      
      const rowsAffected = operation.whereCondition ? 
        Math.floor(Math.random() * 10) : Math.floor(Math.random() * 100);
      
      return { rowsAffected };
      
    } finally {
      this.releaseConnection(connection, workerId);
    }
  }

  /**
   * Execute DELETE operations
   */
  async executeDelete(operation, workerId) {
    const connection = await this.getConnection(workerId);
    
    try {
      const latency = 12 + Math.random() * 8;
      await this.sleep(latency);
      
      const rowsAffected = Math.floor(Math.random() * 5);
      
      return { rowsAffected };
      
    } finally {
      this.releaseConnection(connection, workerId);
    }
  }

  /**
   * Execute transaction operations
   */
  async executeTransaction(operation, workerId) {
    const connection = await this.getConnection(workerId);
    
    try {
      // Begin transaction
      await this.sleep(2);
      
      // Execute multiple operations
      for (const op of operation.operations) {
        await this.executeOperation(op, workerId);
      }
      
      // Commit transaction
      await this.sleep(5);
      
      return { committed: true, operationCount: operation.operations.length };
      
    } catch (error) {
      // Rollback on error
      await this.sleep(3);
      throw error;
    } finally {
      this.releaseConnection(connection, workerId);
    }
  }

  /**
   * Execute JOIN operations
   */
  async executeJoin(operation, workerId) {
    const connection = await this.getConnection(workerId);
    
    try {
      // Complex joins take longer
      const joinCount = operation.tableCount || 2;
      const baseLatency = joinCount * 15;
      const complexity = operation.complexity === 'high' ? 2 : 1;
      
      await this.sleep(baseLatency * complexity + Math.random() * 20);
      
      return { 
        rowCount: Math.floor(Math.random() * 500),
        tableCount: joinCount 
      };
      
    } finally {
      this.releaseConnection(connection, workerId);
    }
  }

  /**
   * Execute aggregate operations
   */
  async executeAggregate(operation, workerId) {
    const connection = await this.getConnection(workerId);
    
    try {
      // Aggregates are typically slower
      const complexity = operation.groupByColumns ? operation.groupByColumns.length : 0;
      const latency = 30 + (complexity * 10) + Math.random() * 25;
      
      await this.sleep(latency);
      
      return { 
        aggregateType: operation.aggregateType,
        groupCount: Math.pow(10, complexity) 
      };
      
    } finally {
      this.releaseConnection(connection, workerId);
    }
  }

  /**
   * Simulate database connection management
   */
  async getConnection(workerId) {
    this.activeConnections++;
    
    // Simulate connection establishment delay
    if (Math.random() < 0.1) { // 10% chance of new connection
      await this.sleep(5 + Math.random() * 5);
    }
    
    return {
      id: `conn-${workerId}-${Date.now()}`,
      workerId,
      establishedAt: Date.now()
    };
  }

  /**
   * Release database connection
   */
  releaseConnection(connection, workerId) {
    this.activeConnections--;
  }

  /**
   * Warmup phase
   */
  async warmupPhase() {
    const warmupWorkers = [];
    const warmupConcurrency = Math.min(10, this.config.maxConcurrency);
    
    for (let i = 0; i < warmupConcurrency; i++) {
      warmupWorkers.push(this.runWarmupWorker(i));
    }
    
    await Promise.allSettled(warmupWorkers);
    console.log('✅ Warmup phase completed');
  }

  /**
   * Warmup worker
   */
  async runWarmupWorker(workerId) {
    const startTime = performance.now();
    
    while (performance.now() - startTime < this.config.warmupDuration) {
      try {
        await this.executeOperation({
          type: 'select',
          complexity: 'low',
          resultSize: 'small'
        }, workerId);
        
        await this.sleep(100); // 100ms between warmup queries
        
      } catch (error) {
        // Ignore warmup errors
      }
    }
  }

  /**
   * Collect system metrics
   */
  collectMetrics() {
    if (!this.isRunning) return;
    
    const currentTime = Date.now();
    const memoryUsage = process.memoryUsage();
    
    this.metrics.resources.push({
      timestamp: currentTime,
      memory: {
        rss: memoryUsage.rss,
        heapUsed: memoryUsage.heapUsed,
        heapTotal: memoryUsage.heapTotal,
        external: memoryUsage.external
      },
      connections: this.activeConnections,
      queries: this.totalQueries,
      errors: this.totalErrors
    });
  }

  /**
   * Generate performance summary
   */
  generateSummary() {
    const totalQueries = this.metrics.resources.length > 0 ? 
      this.metrics.resources[this.metrics.resources.length - 1].queries : 0;
    const totalErrors = this.metrics.resources.length > 0 ? 
      this.metrics.resources[this.metrics.resources.length - 1].errors : 0;
    
    const memoryUsages = this.metrics.resources.map(r => r.memory.heapUsed);
    const avgMemoryUsage = memoryUsages.length > 0 ? 
      memoryUsages.reduce((sum, mem) => sum + mem, 0) / memoryUsages.length : 0;
    
    return {
      totalQueries,
      totalErrors,
      errorRate: totalQueries > 0 ? (totalErrors / totalQueries) * 100 : 0,
      avgMemoryUsage: Math.round(avgMemoryUsage / 1024 / 1024), // MB
      maxConnections: Math.max(...this.metrics.resources.map(r => r.connections), 0)
    };
  }

  /**
   * Utility sleep function
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Generate HTML performance report
   */
  async generateHTMLReport(results) {
    const htmlContent = this.generateHTMLContent(results);
    const reportPath = path.join(this.config.outputDir, `report-${Date.now()}.html`);
    await fs.writeFile(reportPath, htmlContent);
    console.log(`📊 HTML report generated: ${reportPath}`);
    return reportPath;
  }

  /**
   * Generate HTML content for reports
   */
  generateHTMLContent(results) {
    return `
<!DOCTYPE html>
<html>
<head>
    <title>Database Performance Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        .header { text-align: center; margin-bottom: 30px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric-card { background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }
        .metric-title { font-size: 14px; color: #666; margin-bottom: 5px; }
        .metric-value { font-size: 28px; font-weight: bold; color: #333; }
        .chart-container { width: 100%; height: 400px; margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; }
        .scenario-good { background-color: #d4edda; }
        .scenario-warning { background-color: #fff3cd; }
        .scenario-error { background-color: #f8d7da; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Database Performance Benchmark Report</h1>
            <p>Generated: ${new Date(results.startTime).toLocaleString()}</p>
            <p>Total Duration: ${(results.totalDuration / 1000).toFixed(2)} seconds</p>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-title">Total Queries</div>
                <div class="metric-value">${results.summary.totalQueries.toLocaleString()}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Error Rate</div>
                <div class="metric-value">${results.summary.errorRate.toFixed(2)}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Avg Memory</div>
                <div class="metric-value">${results.summary.avgMemoryUsage}MB</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Max Connections</div>
                <div class="metric-value">${results.summary.maxConnections}</div>
            </div>
        </div>
        
        <h2>Scenario Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Scenario</th>
                    <th>Type</th>
                    <th>Concurrency</th>
                    <th>Throughput (q/s)</th>
                    <th>Avg Latency (ms)</th>
                    <th>P95 Latency (ms)</th>
                    <th>Error Rate (%)</th>
                </tr>
            </thead>
            <tbody>
                ${results.scenarios.map(scenario => `
                    <tr class="${this.getScenarioClass(scenario)}">
                        <td>${scenario.name}</td>
                        <td>${scenario.type}</td>
                        <td>${scenario.concurrency}</td>
                        <td>${scenario.throughput.toFixed(2)}</td>
                        <td>${scenario.latency.avg.toFixed(2)}</td>
                        <td>${scenario.latency.p95.toFixed(2)}</td>
                        <td>${scenario.errorRate.toFixed(2)}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
        
        <div class="chart-container">
            <canvas id="throughputChart"></canvas>
        </div>
        
        <div class="chart-container">
            <canvas id="latencyChart"></canvas>
        </div>
    </div>
    
    <script>
        // Throughput chart
        const throughputCtx = document.getElementById('throughputChart').getContext('2d');
        new Chart(throughputCtx, {
            type: 'bar',
            data: {
                labels: ${JSON.stringify(results.scenarios.map(s => s.name))},
                datasets: [{
                    label: 'Throughput (queries/second)',
                    data: ${JSON.stringify(results.scenarios.map(s => s.throughput))},
                    backgroundColor: 'rgba(54, 162, 235, 0.8)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { title: { display: true, text: 'Throughput by Scenario' } }
            }
        });
        
        // Latency chart
        const latencyCtx = document.getElementById('latencyChart').getContext('2d');
        new Chart(latencyCtx, {
            type: 'line',
            data: {
                labels: ${JSON.stringify(results.scenarios.map(s => s.name))},
                datasets: [
                    {
                        label: 'Average Latency',
                        data: ${JSON.stringify(results.scenarios.map(s => s.latency.avg))},
                        borderColor: 'rgba(255, 99, 132, 1)',
                        tension: 0.4
                    },
                    {
                        label: 'P95 Latency',
                        data: ${JSON.stringify(results.scenarios.map(s => s.latency.p95))},
                        borderColor: 'rgba(255, 159, 64, 1)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { title: { display: true, text: 'Latency by Scenario' } }
            }
        });
    </script>
</body>
</html>
    `;
  }

  /**
   * Get CSS class for scenario based on performance
   */
  getScenarioClass(scenario) {
    if (scenario.errorRate > 5) return 'scenario-error';
    if (scenario.latency.avg > 100) return 'scenario-warning';
    return 'scenario-good';
  }
}

module.exports = { DatabasePerformanceTestSuite };