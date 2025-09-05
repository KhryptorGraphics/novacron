#!/usr/bin/env node
/**
 * Database Performance Benchmark Runner
 * Command-line interface for running comprehensive database performance tests
 */

const { DatabasePerformanceTestSuite } = require('../database-performance-suite');
const { loadTestScenarios, scenarioUtils } = require('../scenarios/load-test-scenarios');
const { PerformanceMonitor } = require('../monitoring/performance-monitor');
const fs = require('fs').promises;
const path = require('path');

class BenchmarkRunner {
  constructor() {
    this.testSuite = null;
    this.monitor = null;
    this.config = {
      outputDir: './tests/performance/reports',
      monitoringDir: './tests/performance/monitoring/data',
      comparisonBaseline: null,
      includeMonitoring: true,
      generateReports: true,
      verbose: false
    };
  }

  /**
   * Run performance benchmarks with specified configuration
   */
  async run(options = {}) {
    try {
      console.log('🚀 Starting Database Performance Benchmark Runner...\n');
      
      // Merge options with default config
      this.config = { ...this.config, ...options };
      
      // Initialize test suite
      this.testSuite = new DatabasePerformanceTestSuite({
        outputDir: this.config.outputDir,
        maxConcurrency: this.config.maxConcurrency || 200,
        testDuration: this.config.testDuration || 60000,
        verbose: this.config.verbose
      });
      
      // Initialize performance monitor if requested
      if (this.config.includeMonitoring) {
        this.monitor = new PerformanceMonitor({
          outputDir: this.config.monitoringDir,
          alertThresholds: this.config.alertThresholds || {}
        });
        
        await this.monitor.start();
      }
      
      // Initialize test suite
      await this.testSuite.initialize();
      
      // Select test scenarios
      const scenarios = this.selectScenarios(this.config.scenarioTypes);
      
      if (scenarios.length === 0) {
        throw new Error('No test scenarios selected');
      }
      
      console.log(`📊 Running ${scenarios.length} performance scenarios:\n`);
      scenarios.forEach(scenario => {
        console.log(`   • ${scenario.name} (${scenario.type})`);
      });
      console.log('');
      
      // Run benchmarks
      const results = await this.testSuite.runBenchmark(scenarios);
      
      // Stop monitoring
      if (this.monitor) {
        await this.monitor.stop();
      }
      
      // Generate reports
      if (this.config.generateReports) {
        await this.generateReports(results);
      }
      
      // Display summary
      this.displaySummary(results);
      
      // Comparison with baseline (if provided)
      if (this.config.comparisonBaseline) {
        await this.compareWithBaseline(results);
      }
      
      console.log('\n✅ Benchmark run completed successfully!');
      return results;
      
    } catch (error) {
      console.error('\n❌ Benchmark run failed:', error.message);
      if (this.config.verbose) {
        console.error(error.stack);
      }
      throw error;
    }
  }

  /**
   * Select scenarios based on configuration
   */
  selectScenarios(scenarioTypes) {
    if (!scenarioTypes || scenarioTypes.length === 0) {
      // Default scenario selection
      return scenarioUtils.getBaselineScenarios();
    }
    
    let selectedScenarios = [];
    
    for (const type of scenarioTypes) {
      switch (type) {
        case 'all':
          selectedScenarios = scenarioUtils.getAll();
          break;
        case 'baseline':
          selectedScenarios.push(...scenarioUtils.getBaselineScenarios());
          break;
        case 'stress':
          selectedScenarios.push(...scenarioUtils.getStressScenarios());
          break;
        case 'oltp':
          selectedScenarios.push(...scenarioUtils.getByType('oltp'));
          break;
        case 'olap':
          selectedScenarios.push(...scenarioUtils.getByType('olap'));
          break;
        case 'mixed':
          selectedScenarios.push(...scenarioUtils.getByType('mixed'));
          break;
        case 'regression':
          selectedScenarios.push(...scenarioUtils.getByType('regression'));
          break;
        case 'scalability':
          selectedScenarios.push(...scenarioUtils.getByType('scalability'));
          break;
        default:
          if (loadTestScenarios[type]) {
            selectedScenarios.push(...loadTestScenarios[type]);
          }
      }
    }
    
    // Remove duplicates
    const uniqueScenarios = [];
    const seenNames = new Set();
    
    for (const scenario of selectedScenarios) {
      if (!seenNames.has(scenario.name)) {
        uniqueScenarios.push(scenario);
        seenNames.add(scenario.name);
      }
    }
    
    return uniqueScenarios;
  }

  /**
   * Generate comprehensive performance reports
   */
  async generateReports(results) {
    console.log('📄 Generating performance reports...');
    
    // Generate JSON report
    const jsonReportPath = path.join(this.config.outputDir, `benchmark-detailed-${Date.now()}.json`);
    await fs.writeFile(jsonReportPath, JSON.stringify(results, null, 2));
    console.log(`   • JSON report: ${jsonReportPath}`);
    
    // Generate HTML report
    const htmlReportPath = await this.testSuite.generateHTMLReport(results);
    console.log(`   • HTML report: ${htmlReportPath}`);
    
    // Generate CSV report for data analysis
    const csvReportPath = await this.generateCSVReport(results);
    console.log(`   • CSV report: ${csvReportPath}`);
    
    // Generate performance insights
    const insightsPath = await this.generatePerformanceInsights(results);
    console.log(`   • Insights report: ${insightsPath}`);
  }

  /**
   * Generate CSV report for data analysis
   */
  async generateCSVReport(results) {
    const csvData = [];
    
    // Header
    csvData.push([
      'Scenario Name',
      'Type', 
      'Concurrency',
      'Duration (ms)',
      'Total Queries',
      'Throughput (q/s)',
      'Avg Latency (ms)',
      'P95 Latency (ms)',
      'P99 Latency (ms)',
      'Error Rate (%)',
      'Min Latency (ms)',
      'Max Latency (ms)'
    ].join(','));
    
    // Data rows
    for (const scenario of results.scenarios) {
      csvData.push([
        `"${scenario.name}"`,
        scenario.type,
        scenario.concurrency,
        scenario.totalDuration,
        scenario.queries.length,
        scenario.throughput.toFixed(2),
        scenario.latency.avg.toFixed(2),
        scenario.latency.p95.toFixed(2),
        scenario.latency.p99.toFixed(2),
        scenario.errorRate.toFixed(2),
        scenario.latency.min.toFixed(2),
        scenario.latency.max.toFixed(2)
      ].join(','));
    }
    
    const csvContent = csvData.join('\n');
    const csvPath = path.join(this.config.outputDir, `benchmark-data-${Date.now()}.csv`);
    await fs.writeFile(csvPath, csvContent);
    
    return csvPath;
  }

  /**
   * Generate performance insights and recommendations
   */
  async generatePerformanceInsights(results) {
    const insights = {
      timestamp: new Date().toISOString(),
      summary: results.summary,
      insights: [],
      recommendations: [],
      bottlenecks: [],
      strengths: []
    };
    
    // Analyze each scenario for insights
    for (const scenario of results.scenarios) {
      this.analyzeScenarioPerformance(scenario, insights);
    }
    
    // Overall performance analysis
    this.analyzeOverallPerformance(results, insights);
    
    // Generate recommendations
    this.generateRecommendations(results, insights);
    
    const insightsPath = path.join(this.config.outputDir, `performance-insights-${Date.now()}.json`);
    await fs.writeFile(insightsPath, JSON.stringify(insights, null, 2));
    
    return insightsPath;
  }

  /**
   * Analyze individual scenario performance
   */
  analyzeScenarioPerformance(scenario, insights) {
    // High error rate analysis
    if (scenario.errorRate > 5) {
      insights.bottlenecks.push({
        type: 'high_error_rate',
        scenario: scenario.name,
        value: scenario.errorRate,
        severity: scenario.errorRate > 10 ? 'critical' : 'warning',
        description: `High error rate of ${scenario.errorRate.toFixed(2)}% in ${scenario.name}`
      });
    }
    
    // High latency analysis
    if (scenario.latency.p95 > 1000) { // 1 second
      insights.bottlenecks.push({
        type: 'high_latency',
        scenario: scenario.name,
        value: scenario.latency.p95,
        severity: scenario.latency.p95 > 5000 ? 'critical' : 'warning',
        description: `High P95 latency of ${scenario.latency.p95.toFixed(2)}ms in ${scenario.name}`
      });
    }
    
    // Low throughput analysis
    if (scenario.throughput < 10) {
      insights.bottlenecks.push({
        type: 'low_throughput',
        scenario: scenario.name,
        value: scenario.throughput,
        severity: scenario.throughput < 1 ? 'critical' : 'warning',
        description: `Low throughput of ${scenario.throughput.toFixed(2)} q/s in ${scenario.name}`
      });
    }
    
    // Good performance identification
    if (scenario.errorRate < 1 && scenario.latency.p95 < 100 && scenario.throughput > 100) {
      insights.strengths.push({
        type: 'excellent_performance',
        scenario: scenario.name,
        description: `Excellent performance: ${scenario.throughput.toFixed(2)} q/s with ${scenario.latency.p95.toFixed(2)}ms P95 latency`
      });
    }
  }

  /**
   * Analyze overall performance patterns
   */
  analyzeOverallPerformance(results, insights) {
    const scenarios = results.scenarios;
    
    // Calculate averages across all scenarios
    const avgThroughput = scenarios.reduce((sum, s) => sum + s.throughput, 0) / scenarios.length;
    const avgLatency = scenarios.reduce((sum, s) => sum + s.latency.avg, 0) / scenarios.length;
    const avgErrorRate = scenarios.reduce((sum, s) => sum + s.errorRate, 0) / scenarios.length;
    
    insights.insights.push({
      type: 'overall_performance',
      metrics: {
        avgThroughput: avgThroughput.toFixed(2),
        avgLatency: avgLatency.toFixed(2),
        avgErrorRate: avgErrorRate.toFixed(2)
      }
    });
    
    // Identify performance patterns
    const olapScenarios = scenarios.filter(s => s.type === 'olap');
    const oltpScenarios = scenarios.filter(s => s.type === 'oltp');
    
    if (olapScenarios.length > 0) {
      const avgOlapLatency = olapScenarios.reduce((sum, s) => sum + s.latency.avg, 0) / olapScenarios.length;
      insights.insights.push({
        type: 'olap_performance',
        avgLatency: avgOlapLatency.toFixed(2),
        description: 'Analytical query performance analysis'
      });
    }
    
    if (oltpScenarios.length > 0) {
      const avgOltpThroughput = oltpScenarios.reduce((sum, s) => sum + s.throughput, 0) / oltpScenarios.length;
      insights.insights.push({
        type: 'oltp_performance',
        avgThroughput: avgOltpThroughput.toFixed(2),
        description: 'Transactional workload performance analysis'
      });
    }
  }

  /**
   * Generate performance recommendations
   */
  generateRecommendations(results, insights) {
    // High error rate recommendations
    const highErrorRateBottlenecks = insights.bottlenecks.filter(b => b.type === 'high_error_rate');
    if (highErrorRateBottlenecks.length > 0) {
      insights.recommendations.push({
        priority: 'high',
        category: 'reliability',
        title: 'Address High Error Rates',
        description: 'Multiple scenarios showing elevated error rates',
        actions: [
          'Review database connection pool configuration',
          'Check for deadlock issues in transaction-heavy workloads',
          'Analyze error logs for specific failure patterns',
          'Consider implementing circuit breaker pattern for resilience'
        ]
      });
    }
    
    // High latency recommendations
    const highLatencyBottlenecks = insights.bottlenecks.filter(b => b.type === 'high_latency');
    if (highLatencyBottlenecks.length > 0) {
      insights.recommendations.push({
        priority: 'high',
        category: 'performance',
        title: 'Optimize Query Performance',
        description: 'High latency detected in multiple scenarios',
        actions: [
          'Review and optimize slow-running queries',
          'Ensure proper indexing strategy',
          'Consider query result caching',
          'Analyze execution plans for optimization opportunities',
          'Review database statistics and update if needed'
        ]
      });
    }
    
    // Low throughput recommendations
    const lowThroughputBottlenecks = insights.bottlenecks.filter(b => b.type === 'low_throughput');
    if (lowThroughputBottlenecks.length > 0) {
      insights.recommendations.push({
        priority: 'medium',
        category: 'scalability',
        title: 'Improve System Throughput',
        description: 'Low throughput limiting system capacity',
        actions: [
          'Optimize connection pool size and configuration',
          'Consider read replica distribution for read-heavy workloads',
          'Review hardware resources (CPU, memory, I/O)',
          'Implement connection pooling if not already in use',
          'Consider horizontal scaling options'
        ]
      });
    }
    
    // General performance recommendations
    insights.recommendations.push({
      priority: 'medium',
      category: 'monitoring',
      title: 'Implement Continuous Performance Monitoring',
      description: 'Establish ongoing performance visibility',
      actions: [
        'Set up real-time performance dashboards',
        'Implement automated performance regression detection',
        'Configure alerts for key performance thresholds',
        'Schedule regular performance benchmark runs',
        'Track performance trends over time'
      ]
    });
  }

  /**
   * Compare results with baseline
   */
  async compareWithBaseline(results) {
    try {
      console.log('📈 Comparing with baseline performance...');
      
      const baselineData = await fs.readFile(this.config.comparisonBaseline, 'utf8');
      const baseline = JSON.parse(baselineData);
      
      const comparison = {
        timestamp: new Date().toISOString(),
        baseline: {
          file: this.config.comparisonBaseline,
          timestamp: baseline.startTime
        },
        current: {
          timestamp: results.startTime
        },
        scenarios: []
      };
      
      // Compare scenarios
      for (const currentScenario of results.scenarios) {
        const baselineScenario = baseline.scenarios.find(s => s.name === currentScenario.name);
        
        if (baselineScenario) {
          const scenarioComparison = {
            name: currentScenario.name,
            throughput: {
              baseline: baselineScenario.throughput,
              current: currentScenario.throughput,
              change: ((currentScenario.throughput - baselineScenario.throughput) / baselineScenario.throughput) * 100
            },
            latency: {
              baseline: baselineScenario.latency.avg,
              current: currentScenario.latency.avg,
              change: ((currentScenario.latency.avg - baselineScenario.latency.avg) / baselineScenario.latency.avg) * 100
            },
            errorRate: {
              baseline: baselineScenario.errorRate,
              current: currentScenario.errorRate,
              change: currentScenario.errorRate - baselineScenario.errorRate
            }
          };
          
          comparison.scenarios.push(scenarioComparison);
        }
      }
      
      // Save comparison report
      const comparisonPath = path.join(this.config.outputDir, `baseline-comparison-${Date.now()}.json`);
      await fs.writeFile(comparisonPath, JSON.stringify(comparison, null, 2));
      console.log(`   • Comparison report: ${comparisonPath}`);
      
      // Display comparison summary
      this.displayComparisonSummary(comparison);
      
    } catch (error) {
      console.error('⚠️ Failed to compare with baseline:', error.message);
    }
  }

  /**
   * Display performance summary
   */
  displaySummary(results) {
    console.log('\n📊 Performance Benchmark Summary');
    console.log('═'.repeat(50));
    console.log(`Total Duration: ${(results.totalDuration / 1000).toFixed(2)} seconds`);
    console.log(`Total Queries: ${results.summary.totalQueries.toLocaleString()}`);
    console.log(`Error Rate: ${results.summary.errorRate.toFixed(2)}%`);
    console.log(`Average Memory Usage: ${results.summary.avgMemoryUsage}MB`);
    console.log(`Max Connections: ${results.summary.maxConnections}`);
    
    console.log('\n🎯 Scenario Performance:');
    console.log('─'.repeat(80));
    console.log('Scenario Name                    | Throughput  | Avg Latency | Error Rate');
    console.log('─'.repeat(80));
    
    for (const scenario of results.scenarios) {
      const name = scenario.name.padEnd(32);
      const throughput = `${scenario.throughput.toFixed(1)} q/s`.padEnd(11);
      const latency = `${scenario.latency.avg.toFixed(1)}ms`.padEnd(11);
      const errorRate = `${scenario.errorRate.toFixed(1)}%`;
      
      console.log(`${name} | ${throughput} | ${latency} | ${errorRate}`);
    }
  }

  /**
   * Display baseline comparison summary
   */
  displayComparisonSummary(comparison) {
    console.log('\n📈 Baseline Comparison Summary');
    console.log('═'.repeat(60));
    
    for (const scenario of comparison.scenarios) {
      console.log(`\n${scenario.name}:`);
      
      const throughputChange = scenario.throughput.change > 0 ? '📈' : '📉';
      console.log(`   Throughput: ${throughputChange} ${scenario.throughput.change.toFixed(1)}%`);
      
      const latencyChange = scenario.latency.change < 0 ? '📈' : '📉';
      console.log(`   Latency: ${latencyChange} ${scenario.latency.change.toFixed(1)}%`);
      
      if (Math.abs(scenario.errorRate.change) > 0.1) {
        const errorChange = scenario.errorRate.change < 0 ? '📈' : '📉';
        console.log(`   Error Rate: ${errorChange} ${scenario.errorRate.change.toFixed(2)}pp`);
      }
    }
  }
}

// CLI interface
async function main() {
  const args = process.argv.slice(2);
  const options = {};
  
  // Parse command line arguments
  for (let i = 0; i < args.length; i += 2) {
    const key = args[i].replace(/^--/, '');
    const value = args[i + 1];
    
    switch (key) {
      case 'scenarios':
        options.scenarioTypes = value ? value.split(',') : ['baseline'];
        break;
      case 'output':
        options.outputDir = value;
        break;
      case 'baseline':
        options.comparisonBaseline = value;
        break;
      case 'concurrency':
        options.maxConcurrency = parseInt(value);
        break;
      case 'duration':
        options.testDuration = parseInt(value) * 1000; // Convert to milliseconds
        break;
      case 'no-monitoring':
        options.includeMonitoring = false;
        i--; // No value for this flag
        break;
      case 'no-reports':
        options.generateReports = false;
        i--; // No value for this flag
        break;
      case 'verbose':
        options.verbose = true;
        i--; // No value for this flag
        break;
    }
  }
  
  const runner = new BenchmarkRunner();
  await runner.run(options);
}

// Run if called directly
if (require.main === module) {
  main().catch(error => {
    console.error('❌ Benchmark runner failed:', error);
    process.exit(1);
  });
}

module.exports = { BenchmarkRunner };