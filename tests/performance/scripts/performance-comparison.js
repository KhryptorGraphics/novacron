/**
 * Database Performance Comparison Tool
 * Compares performance between current and optimized database architectures
 */

const { DatabasePerformanceTestSuite } = require('../database-performance-suite');
const { loadTestScenarios, scenarioUtils } = require('../scenarios/load-test-scenarios');
const { PerformanceMonitor } = require('../monitoring/performance-monitor');
const fs = require('fs').promises;
const path = require('path');

class PerformanceComparison {
  constructor(config = {}) {
    this.config = {
      outputDir: config.outputDir || './tests/performance/reports/comparisons',
      currentArchitecture: config.currentArchitecture || 'current',
      optimizedArchitecture: config.optimizedArchitecture || 'optimized',
      scenarios: config.scenarios || 'baseline',
      iterations: config.iterations || 3,
      confidenceLevel: config.confidenceLevel || 0.95,
      significanceThreshold: config.significanceThreshold || 0.05,
      ...config
    };
    
    this.results = {
      current: null,
      optimized: null,
      comparison: null,
      statisticalAnalysis: null
    };
  }

  /**
   * Run comprehensive performance comparison
   */
  async runComparison() {
    console.log('🔬 Starting Database Performance Architecture Comparison...\n');
    
    try {
      // Ensure output directory exists
      await fs.mkdir(this.config.outputDir, { recursive: true });
      
      // Phase 1: Test current architecture
      console.log('📊 Phase 1: Testing Current Architecture');
      console.log('━'.repeat(50));
      this.results.current = await this.testArchitecture('current');
      
      // Wait between tests to ensure clean state
      console.log('\n⏳ Waiting 30 seconds between architecture tests...');
      await this.sleep(30000);
      
      // Phase 2: Test optimized architecture
      console.log('\n📊 Phase 2: Testing Optimized Architecture');
      console.log('━'.repeat(50));
      this.results.optimized = await this.testArchitecture('optimized');
      
      // Phase 3: Statistical comparison
      console.log('\n📈 Phase 3: Statistical Analysis');
      console.log('━'.repeat(50));
      this.results.comparison = this.compareResults();
      this.results.statisticalAnalysis = this.performStatisticalAnalysis();
      
      // Phase 4: Generate reports
      console.log('\n📄 Phase 4: Generating Comparison Reports');
      console.log('━'.repeat(50));
      await this.generateComparisonReports();
      
      // Display summary
      this.displayComparisonSummary();
      
      console.log('\n✅ Performance comparison completed successfully!');
      return this.results;
      
    } catch (error) {
      console.error('\n❌ Performance comparison failed:', error.message);
      throw error;
    }
  }

  /**
   * Test specific architecture configuration
   */
  async testArchitecture(architectureType) {
    console.log(`\n🏗️ Testing ${architectureType} architecture...`);
    
    const testSuite = new DatabasePerformanceTestSuite({
      outputDir: path.join(this.config.outputDir, architectureType),
      architecture: architectureType,
      testDuration: 45000, // 45 seconds per scenario
      maxConcurrency: 100
    });
    
    const monitor = new PerformanceMonitor({
      outputDir: path.join(this.config.outputDir, architectureType, 'monitoring')
    });
    
    try {
      // Initialize test suite and monitoring
      await testSuite.initialize();
      await monitor.start();
      
      // Select scenarios for comparison
      const scenarios = this.selectComparisonScenarios();
      console.log(`   Running ${scenarios.length} comparison scenarios`);
      
      // Run multiple iterations for statistical significance
      const iterationResults = [];
      
      for (let iteration = 1; iteration <= this.config.iterations; iteration++) {
        console.log(`   Iteration ${iteration}/${this.config.iterations}`);
        
        const iterationResult = await testSuite.runBenchmark(scenarios);
        iterationResults.push(iterationResult);
        
        // Brief pause between iterations
        if (iteration < this.config.iterations) {
          await this.sleep(10000); // 10 seconds
        }
      }
      
      // Stop monitoring
      await monitor.stop();
      
      // Aggregate results from multiple iterations
      const aggregatedResults = this.aggregateIterationResults(iterationResults);
      
      console.log(`✅ ${architectureType} architecture testing completed`);
      return {
        architecture: architectureType,
        iterations: this.config.iterations,
        rawResults: iterationResults,
        aggregated: aggregatedResults,
        timestamp: new Date().toISOString()
      };
      
    } catch (error) {
      console.error(`❌ Failed to test ${architectureType} architecture:`, error.message);
      throw error;
    }
  }

  /**
   * Select scenarios specifically for architecture comparison
   */
  selectComparisonScenarios() {
    // Use a focused set of scenarios that highlight architectural differences
    return [
      // OLTP scenarios - test transaction processing improvements
      ...loadTestScenarios.oltp.slice(0, 2),
      
      // OLAP scenarios - test analytical query optimizations  
      ...loadTestScenarios.olap.slice(0, 1),
      
      // Mixed workload - test overall system balance
      ...loadTestScenarios.mixed.slice(0, 1),
      
      // Stress test - test system limits
      ...loadTestScenarios.stress.slice(0, 1)
    ];
  }

  /**
   * Aggregate results from multiple test iterations
   */
  aggregateIterationResults(iterationResults) {
    const aggregated = {
      scenarios: {},
      summary: {
        totalQueries: 0,
        avgThroughput: 0,
        avgLatency: 0,
        avgErrorRate: 0
      },
      statistics: {}
    };
    
    // Group results by scenario
    const scenarioGroups = {};
    
    iterationResults.forEach(iterationResult => {
      iterationResult.scenarios.forEach(scenario => {
        if (!scenarioGroups[scenario.name]) {
          scenarioGroups[scenario.name] = [];
        }
        scenarioGroups[scenario.name].push(scenario);
      });
    });
    
    // Calculate statistics for each scenario
    Object.keys(scenarioGroups).forEach(scenarioName => {
      const scenarios = scenarioGroups[scenarioName];
      
      aggregated.scenarios[scenarioName] = {
        name: scenarioName,
        type: scenarios[0].type,
        iterations: scenarios.length,
        throughput: this.calculateStatistics(scenarios.map(s => s.throughput)),
        latency: {
          avg: this.calculateStatistics(scenarios.map(s => s.latency.avg)),
          p95: this.calculateStatistics(scenarios.map(s => s.latency.p95)),
          p99: this.calculateStatistics(scenarios.map(s => s.latency.p99))
        },
        errorRate: this.calculateStatistics(scenarios.map(s => s.errorRate))
      };
    });
    
    // Calculate overall summary statistics
    const allThroughputs = Object.values(aggregated.scenarios).map(s => s.throughput.mean);
    const allLatencies = Object.values(aggregated.scenarios).map(s => s.latency.avg.mean);
    const allErrorRates = Object.values(aggregated.scenarios).map(s => s.errorRate.mean);
    
    aggregated.summary = {
      avgThroughput: allThroughputs.reduce((sum, t) => sum + t, 0) / allThroughputs.length,
      avgLatency: allLatencies.reduce((sum, l) => sum + l, 0) / allLatencies.length,
      avgErrorRate: allErrorRates.reduce((sum, e) => sum + e, 0) / allErrorRates.length,
      totalQueries: iterationResults.reduce((sum, r) => sum + r.summary.totalQueries, 0)
    };
    
    return aggregated;
  }

  /**
   * Calculate statistical measures for a dataset
   */
  calculateStatistics(data) {
    const sorted = [...data].sort((a, b) => a - b);
    const n = data.length;
    
    const mean = data.reduce((sum, val) => sum + val, 0) / n;
    const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (n - 1);
    const stdDev = Math.sqrt(variance);
    
    // Standard error of the mean
    const sem = stdDev / Math.sqrt(n);
    
    // Confidence interval (95% by default)
    const tValue = this.getTValue(n - 1, this.config.confidenceLevel);
    const marginOfError = tValue * sem;
    
    return {
      mean: mean,
      median: sorted[Math.floor(n / 2)],
      min: sorted[0],
      max: sorted[n - 1],
      stdDev: stdDev,
      variance: variance,
      sem: sem,
      confidenceInterval: {
        lower: mean - marginOfError,
        upper: mean + marginOfError,
        level: this.config.confidenceLevel
      },
      count: n
    };
  }

  /**
   * Get t-value for confidence interval calculation
   */
  getTValue(degreesOfFreedom, confidenceLevel) {
    // Simplified t-table values for common cases
    const tTable = {
      0.95: { 1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306 }
    };
    
    const alpha = 1 - confidenceLevel;
    const df = Math.min(degreesOfFreedom, 8);
    
    return tTable[confidenceLevel]?.[df] || 2.0; // Default to 2.0 if not found
  }

  /**
   * Compare results between architectures
   */
  compareResults() {
    const comparison = {
      timestamp: new Date().toISOString(),
      scenarios: {},
      summary: {},
      significantImprovements: [],
      regressions: []
    };
    
    // Compare each scenario
    Object.keys(this.results.current.aggregated.scenarios).forEach(scenarioName => {
      const current = this.results.current.aggregated.scenarios[scenarioName];
      const optimized = this.results.optimized.aggregated.scenarios[scenarioName];
      
      if (!optimized) {
        console.warn(`⚠️ Scenario ${scenarioName} not found in optimized results`);
        return;
      }
      
      const scenarioComparison = {
        name: scenarioName,
        type: current.type,
        throughput: this.compareMetric(current.throughput, optimized.throughput, 'higher_better'),
        latency: {
          avg: this.compareMetric(current.latency.avg, optimized.latency.avg, 'lower_better'),
          p95: this.compareMetric(current.latency.p95, optimized.latency.p95, 'lower_better'),
          p99: this.compareMetric(current.latency.p99, optimized.latency.p99, 'lower_better')
        },
        errorRate: this.compareMetric(current.errorRate, optimized.errorRate, 'lower_better')
      };
      
      comparison.scenarios[scenarioName] = scenarioComparison;
      
      // Identify significant changes
      this.identifySignificantChanges(scenarioComparison, comparison);
    });
    
    // Overall summary comparison
    comparison.summary = {
      throughput: this.compareMetric(
        { mean: this.results.current.aggregated.summary.avgThroughput },
        { mean: this.results.optimized.aggregated.summary.avgThroughput },
        'higher_better'
      ),
      latency: this.compareMetric(
        { mean: this.results.current.aggregated.summary.avgLatency },
        { mean: this.results.optimized.aggregated.summary.avgLatency },
        'lower_better'
      ),
      errorRate: this.compareMetric(
        { mean: this.results.current.aggregated.summary.avgErrorRate },
        { mean: this.results.optimized.aggregated.summary.avgErrorRate },
        'lower_better'
      )
    };
    
    return comparison;
  }

  /**
   * Compare individual metrics between architectures
   */
  compareMetric(currentStat, optimizedStat, betterDirection) {
    const current = currentStat.mean;
    const optimized = optimizedStat.mean;
    
    const absoluteChange = optimized - current;
    const percentChange = current !== 0 ? (absoluteChange / current) * 100 : 0;
    
    // Determine if change is improvement based on direction
    const isImprovement = betterDirection === 'higher_better' ? 
      absoluteChange > 0 : absoluteChange < 0;
    
    return {
      current: current,
      optimized: optimized,
      absoluteChange: absoluteChange,
      percentChange: percentChange,
      isImprovement: isImprovement,
      magnitude: Math.abs(percentChange),
      significance: this.assessSignificance(Math.abs(percentChange))
    };
  }

  /**
   * Assess significance level of changes
   */
  assessSignificance(percentChange) {
    if (percentChange >= 25) return 'high';
    if (percentChange >= 10) return 'medium';
    if (percentChange >= 5) return 'low';
    return 'negligible';
  }

  /**
   * Identify significant improvements and regressions
   */
  identifySignificantChanges(scenarioComparison, comparison) {
    const scenario = scenarioComparison.name;
    
    // Check throughput improvement
    if (scenarioComparison.throughput.isImprovement && scenarioComparison.throughput.significance !== 'negligible') {
      comparison.significantImprovements.push({
        scenario: scenario,
        metric: 'throughput',
        improvement: scenarioComparison.throughput.percentChange,
        significance: scenarioComparison.throughput.significance
      });
    }
    
    // Check latency improvement
    if (scenarioComparison.latency.avg.isImprovement && scenarioComparison.latency.avg.significance !== 'negligible') {
      comparison.significantImprovements.push({
        scenario: scenario,
        metric: 'latency',
        improvement: Math.abs(scenarioComparison.latency.avg.percentChange),
        significance: scenarioComparison.latency.avg.significance
      });
    }
    
    // Check for regressions
    if (!scenarioComparison.throughput.isImprovement && scenarioComparison.throughput.significance !== 'negligible') {
      comparison.regressions.push({
        scenario: scenario,
        metric: 'throughput',
        regression: Math.abs(scenarioComparison.throughput.percentChange),
        significance: scenarioComparison.throughput.significance
      });
    }
  }

  /**
   * Perform statistical analysis
   */
  performStatisticalAnalysis() {
    const analysis = {
      timestamp: new Date().toISOString(),
      sampleSize: this.config.iterations,
      confidenceLevel: this.config.confidenceLevel,
      statisticalTests: {},
      overallAssessment: {
        improvementConfidence: 0,
        regressionRisk: 0,
        recommendations: []
      }
    };
    
    // Analyze confidence intervals for overlapping
    let improvementCount = 0;
    let totalComparisons = 0;
    
    Object.values(this.results.comparison.scenarios).forEach(scenario => {
      totalComparisons++;
      
      // Count significant improvements
      if (scenario.throughput.isImprovement && scenario.throughput.significance !== 'negligible') {
        improvementCount++;
      }
      if (scenario.latency.avg.isImprovement && scenario.latency.avg.significance !== 'negligible') {
        improvementCount++;
      }
    });
    
    // Calculate overall improvement confidence
    analysis.overallAssessment.improvementConfidence = totalComparisons > 0 ? 
      (improvementCount / totalComparisons) * 100 : 0;
    
    // Generate recommendations based on analysis
    this.generateStatisticalRecommendations(analysis);
    
    return analysis;
  }

  /**
   * Generate recommendations based on statistical analysis
   */
  generateStatisticalRecommendations(analysis) {
    const improvementConfidence = analysis.overallAssessment.improvementConfidence;
    
    if (improvementConfidence >= 75) {
      analysis.overallAssessment.recommendations.push({
        type: 'deployment',
        priority: 'high',
        title: 'Proceed with Optimized Architecture',
        description: `High confidence (${improvementConfidence.toFixed(1)}%) in performance improvements`,
        actions: [
          'Deploy optimized architecture to production',
          'Monitor performance closely during rollout',
          'Implement gradual rollback plan as safety measure'
        ]
      });
    } else if (improvementConfidence >= 50) {
      analysis.overallAssessment.recommendations.push({
        type: 'testing',
        priority: 'medium',
        title: 'Extended Testing Recommended',
        description: `Moderate confidence (${improvementConfidence.toFixed(1)}%) requires more validation`,
        actions: [
          'Increase test iterations for better statistical power',
          'Test with production-like data volumes',
          'Conduct staged rollout with performance monitoring'
        ]
      });
    } else {
      analysis.overallAssessment.recommendations.push({
        type: 'optimization',
        priority: 'high',
        title: 'Further Optimization Required',
        description: `Low confidence (${improvementConfidence.toFixed(1)}%) suggests limited improvements`,
        actions: [
          'Review optimization strategies',
          'Identify additional bottlenecks',
          'Consider alternative architectural approaches',
          'Validate optimization implementation'
        ]
      });
    }
    
    // Check for any regressions
    if (this.results.comparison.regressions.length > 0) {
      analysis.overallAssessment.recommendations.push({
        type: 'regression',
        priority: 'critical',
        title: 'Address Performance Regressions',
        description: `${this.results.comparison.regressions.length} regressions detected`,
        actions: [
          'Investigate root cause of performance regressions',
          'Validate optimization implementation',
          'Consider rollback if regressions are severe',
          'Re-run tests after fixes'
        ]
      });
    }
  }

  /**
   * Generate comprehensive comparison reports
   */
  async generateComparisonReports() {
    const timestamp = Date.now();
    
    // Detailed JSON report
    const detailedReport = {
      metadata: {
        timestamp: new Date().toISOString(),
        configurations: {
          current: this.config.currentArchitecture,
          optimized: this.config.optimizedArchitecture
        },
        testParameters: {
          iterations: this.config.iterations,
          confidenceLevel: this.config.confidenceLevel,
          scenarios: this.config.scenarios
        }
      },
      results: this.results
    };
    
    const jsonReportPath = path.join(this.config.outputDir, `detailed-comparison-${timestamp}.json`);
    await fs.writeFile(jsonReportPath, JSON.stringify(detailedReport, null, 2));
    console.log(`   📄 Detailed JSON report: ${jsonReportPath}`);
    
    // Executive summary report
    const executiveSummary = this.generateExecutiveSummary();
    const summaryPath = path.join(this.config.outputDir, `executive-summary-${timestamp}.json`);
    await fs.writeFile(summaryPath, JSON.stringify(executiveSummary, null, 2));
    console.log(`   📊 Executive summary: ${summaryPath}`);
    
    // CSV data export
    const csvData = this.generateComparisonCSV();
    const csvPath = path.join(this.config.outputDir, `comparison-data-${timestamp}.csv`);
    await fs.writeFile(csvPath, csvData);
    console.log(`   📈 CSV data export: ${csvPath}`);
  }

  /**
   * Generate executive summary
   */
  generateExecutiveSummary() {
    const improvements = this.results.comparison.significantImprovements;
    const regressions = this.results.comparison.regressions;
    
    return {
      timestamp: new Date().toISOString(),
      executiveSummary: {
        recommendation: this.results.statisticalAnalysis.overallAssessment.improvementConfidence >= 75 ? 
          'PROCEED' : this.results.statisticalAnalysis.overallAssessment.improvementConfidence >= 50 ? 
          'PROCEED_WITH_CAUTION' : 'REQUIRES_FURTHER_OPTIMIZATION',
        
        confidence: `${this.results.statisticalAnalysis.overallAssessment.improvementConfidence.toFixed(1)}%`,
        
        keyFindings: {
          totalImprovements: improvements.length,
          totalRegressions: regressions.length,
          majorImprovements: improvements.filter(i => i.significance === 'high').length,
          significantRegressions: regressions.filter(r => r.significance === 'high').length
        },
        
        performanceGains: {
          throughput: this.results.comparison.summary.throughput.percentChange,
          latency: Math.abs(this.results.comparison.summary.latency.percentChange),
          errorRate: Math.abs(this.results.comparison.summary.errorRate.percentChange)
        },
        
        riskAssessment: {
          deploymentRisk: regressions.length > 0 ? 'MEDIUM' : 'LOW',
          rollbackComplexity: 'MEDIUM',
          monitoringRequired: 'HIGH'
        },
        
        nextSteps: this.results.statisticalAnalysis.overallAssessment.recommendations.map(r => r.title)
      }
    };
  }

  /**
   * Generate CSV comparison data
   */
  generateComparisonCSV() {
    const csvData = [];
    
    // Header
    csvData.push([
      'Scenario',
      'Architecture',
      'Throughput (q/s)',
      'Avg Latency (ms)',
      'P95 Latency (ms)',
      'Error Rate (%)',
      'Throughput Change (%)',
      'Latency Change (%)',
      'Significance'
    ].join(','));
    
    // Data rows
    Object.values(this.results.comparison.scenarios).forEach(scenario => {
      // Current architecture row
      csvData.push([
        `"${scenario.name}"`,
        'Current',
        scenario.throughput.current.toFixed(2),
        scenario.latency.avg.current.toFixed(2),
        scenario.latency.p95.current.toFixed(2),
        scenario.errorRate.current.toFixed(2),
        '0.00',
        '0.00',
        'baseline'
      ].join(','));
      
      // Optimized architecture row
      csvData.push([
        `"${scenario.name}"`,
        'Optimized',
        scenario.throughput.optimized.toFixed(2),
        scenario.latency.avg.optimized.toFixed(2),
        scenario.latency.p95.optimized.toFixed(2),
        scenario.errorRate.optimized.toFixed(2),
        scenario.throughput.percentChange.toFixed(2),
        scenario.latency.avg.percentChange.toFixed(2),
        scenario.throughput.significance
      ].join(','));
    });
    
    return csvData.join('\n');
  }

  /**
   * Display comparison summary to console
   */
  displayComparisonSummary() {
    console.log('\n🎯 Performance Comparison Summary');
    console.log('═'.repeat(60));
    
    const summary = this.results.comparison.summary;
    const stats = this.results.statisticalAnalysis;
    
    console.log(`Overall Confidence: ${stats.overallAssessment.improvementConfidence.toFixed(1)}%`);
    console.log(`Test Iterations: ${this.config.iterations}`);
    console.log(`Confidence Level: ${(this.config.confidenceLevel * 100)}%`);
    
    console.log('\n📊 Performance Changes:');
    console.log('─'.repeat(40));
    console.log(`Throughput: ${this.formatChange(summary.throughput)} q/s`);
    console.log(`Latency: ${this.formatChange(summary.latency)} ms`);
    console.log(`Error Rate: ${this.formatChange(summary.errorRate)} %`);
    
    if (this.results.comparison.significantImprovements.length > 0) {
      console.log('\n✅ Significant Improvements:');
      this.results.comparison.significantImprovements.forEach(improvement => {
        console.log(`   • ${improvement.scenario}: ${improvement.metric} +${improvement.improvement.toFixed(1)}% (${improvement.significance})`);
      });
    }
    
    if (this.results.comparison.regressions.length > 0) {
      console.log('\n⚠️ Performance Regressions:');
      this.results.comparison.regressions.forEach(regression => {
        console.log(`   • ${regression.scenario}: ${regression.metric} -${regression.regression.toFixed(1)}% (${regression.significance})`);
      });
    }
    
    console.log('\n🎯 Recommendation:');
    const mainRecommendation = stats.overallAssessment.recommendations[0];
    console.log(`   ${mainRecommendation.title}`);
    console.log(`   Priority: ${mainRecommendation.priority.toUpperCase()}`);
    console.log(`   ${mainRecommendation.description}`);
  }

  /**
   * Format performance change for display
   */
  formatChange(metric) {
    const change = metric.percentChange;
    const symbol = change > 0 ? '+' : '';
    const arrow = metric.isImprovement ? '📈' : change === 0 ? '➡️' : '📉';
    return `${arrow} ${symbol}${change.toFixed(1)}%`;
  }

  /**
   * Utility sleep function
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

module.exports = { PerformanceComparison };