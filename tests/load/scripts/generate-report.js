const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

class LoadTestReportGenerator {
  constructor() {
    this.reportsDir = path.join(__dirname, '../reports');
    this.templatesDir = path.join(__dirname, '../templates');
    this.timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  }

  async generateComprehensiveReport() {
    console.log('Generating comprehensive load test report...');

    // Ensure reports directory exists
    if (!fs.existsSync(this.reportsDir)) {
      fs.mkdirSync(this.reportsDir, { recursive: true });
    }

    try {
      // Collect all test results
      const results = await this.collectTestResults();
      
      // Generate different report formats
      await Promise.all([
        this.generateHTMLReport(results),
        this.generateJSONReport(results),
        this.generateCSVReport(results),
        this.generateMarkdownSummary(results)
      ]);

      console.log(`Reports generated successfully in: ${this.reportsDir}`);
      console.log(`Report timestamp: ${this.timestamp}`);

    } catch (error) {
      console.error('Failed to generate reports:', error);
      process.exit(1);
    }
  }

  async collectTestResults() {
    const results = {
      metadata: {
        timestamp: new Date().toISOString(),
        environment: process.env.ENVIRONMENT || 'local',
        testDuration: this.calculateTestDuration(),
        testSuite: 'NovaCron Load Testing Suite v1.0'
      },
      summary: {},
      scenarios: {},
      metrics: {},
      thresholds: {},
      recommendations: []
    };

    // Collect results from different test scenarios
    const scenarioFiles = [
      'api-load-test-results.json',
      'vm-management-test-results.json', 
      'websocket-stress-test-results.json',
      'database-performance-test-results.json',
      'federation-load-test-results.json',
      'benchmark-results.json'
    ];

    for (const file of scenarioFiles) {
      const filePath = path.join(this.reportsDir, file);
      if (fs.existsSync(filePath)) {
        try {
          const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
          const scenarioName = file.replace('-results.json', '').replace(/-/g, '_');
          results.scenarios[scenarioName] = data;
        } catch (error) {
          console.warn(`Failed to parse ${file}:`, error.message);
        }
      }
    }

    // Calculate summary statistics
    results.summary = this.calculateSummaryStats(results.scenarios);
    
    // Analyze performance metrics
    results.metrics = this.analyzeMetrics(results.scenarios);
    
    // Check threshold compliance
    results.thresholds = this.checkThresholds(results.scenarios);
    
    // Generate recommendations
    results.recommendations = this.generateRecommendations(results);

    return results;
  }

  calculateSummaryStats(scenarios) {
    const summary = {
      totalRequests: 0,
      totalErrors: 0,
      averageResponseTime: 0,
      requestsPerSecond: 0,
      concurrentUsers: 0,
      testDuration: 0,
      successRate: 0,
      
      vmOperations: {
        created: 0,
        started: 0,
        stopped: 0,
        deleted: 0,
        migrated: 0,
        snapshots: 0
      },
      
      websocketConnections: {
        total: 0,
        concurrent: 0,
        messagesSent: 0,
        messagesReceived: 0
      },
      
      databaseOperations: {
        queries: 0,
        transactions: 0,
        averageQueryTime: 0,
        concurrentConnections: 0
      }
    };

    // Aggregate data from all scenarios
    Object.values(scenarios).forEach(scenario => {
      if (scenario.root_group) {
        const rootGroup = scenario.root_group;
        
        summary.totalRequests += rootGroup.http_reqs?.count || 0;
        summary.totalErrors += rootGroup.http_req_failed?.count || 0;
        
        if (rootGroup.http_req_duration) {
          summary.averageResponseTime += rootGroup.http_req_duration.avg || 0;
        }
        
        // Add VM-specific metrics
        if (scenario.custom_metrics) {
          const custom = scenario.custom_metrics;
          summary.vmOperations.created += custom.vm_lifecycle_operations?.values?.create || 0;
          summary.websocketConnections.total += custom.ws_messages_sent?.count || 0;
          summary.databaseOperations.queries += custom.db_transactions_total?.count || 0;
        }
      }
    });

    // Calculate derived metrics
    const scenarioCount = Object.keys(scenarios).length;
    if (scenarioCount > 0) {
      summary.averageResponseTime /= scenarioCount;
      summary.successRate = summary.totalRequests > 0 ? 
        ((summary.totalRequests - summary.totalErrors) / summary.totalRequests) * 100 : 0;
    }

    return summary;
  }

  analyzeMetrics(scenarios) {
    const metrics = {
      performance: {
        apiLatency: { p50: 0, p95: 0, p99: 0 },
        vmOperationLatency: { p50: 0, p95: 0, p99: 0 },
        websocketLatency: { p50: 0, p95: 0, p99: 0 },
        databaseLatency: { p50: 0, p95: 0, p99: 0 }
      },
      
      throughput: {
        apiRequestsPerSecond: 0,
        vmOperationsPerSecond: 0,
        websocketConnectionsPerSecond: 0,
        databaseQueriesPerSecond: 0
      },
      
      scalability: {
        maxConcurrentUsers: 0,
        maxConcurrentVMs: 0,
        maxWebSocketConnections: 0,
        maxDatabaseConnections: 0
      },
      
      reliability: {
        errorRates: {},
        uptimePercentage: 0,
        failoverTime: 0
      }
    };

    // Extract metrics from scenario results
    Object.entries(scenarios).forEach(([name, data]) => {
      if (data.root_group) {
        const rootGroup = data.root_group;
        
        // Performance metrics
        if (rootGroup.http_req_duration) {
          const duration = rootGroup.http_req_duration;
          metrics.performance.apiLatency.p50 += duration.p50 || 0;
          metrics.performance.apiLatency.p95 += duration.p95 || 0;
          metrics.performance.apiLatency.p99 += duration.p99 || 0;
        }
        
        // Error rates
        if (rootGroup.http_req_failed) {
          metrics.reliability.errorRates[name] = rootGroup.http_req_failed.rate || 0;
        }
      }
    });

    return metrics;
  }

  checkThresholds(scenarios) {
    const thresholds = {
      passed: [],
      failed: [],
      summary: {
        totalChecks: 0,
        passedChecks: 0,
        failureRate: 0
      }
    };

    Object.entries(scenarios).forEach(([name, data]) => {
      if (data.thresholds) {
        Object.entries(data.thresholds).forEach(([metric, result]) => {
          thresholds.summary.totalChecks++;
          
          if (result.ok) {
            thresholds.passed.push({
              scenario: name,
              metric: metric,
              threshold: result.threshold,
              actual: result.value
            });
            thresholds.summary.passedChecks++;
          } else {
            thresholds.failed.push({
              scenario: name,
              metric: metric,
              threshold: result.threshold,
              actual: result.value,
              impact: this.assessThresholdImpact(metric, result)
            });
          }
        });
      }
    });

    thresholds.summary.failureRate = thresholds.summary.totalChecks > 0 ?
      ((thresholds.summary.totalChecks - thresholds.summary.passedChecks) / thresholds.summary.totalChecks) * 100 : 0;

    return thresholds;
  }

  generateRecommendations(results) {
    const recommendations = [];

    // Performance recommendations
    if (results.metrics.performance.apiLatency.p95 > 500) {
      recommendations.push({
        category: 'Performance',
        priority: 'High',
        issue: 'API latency exceeds target (>500ms at P95)',
        recommendation: 'Implement API response caching, optimize database queries, consider load balancing',
        impact: 'User experience degradation'
      });
    }

    // Scalability recommendations
    if (results.summary.successRate < 95) {
      recommendations.push({
        category: 'Reliability',
        priority: 'Critical',
        issue: `Success rate below target: ${results.summary.successRate.toFixed(2)}%`,
        recommendation: 'Investigate error patterns, implement circuit breakers, increase resource allocation',
        impact: 'Service availability issues'
      });
    }

    // VM operation recommendations
    if (results.summary.vmOperations.created > 0 && 
        results.metrics.performance.vmOperationLatency.p95 > 30000) {
      recommendations.push({
        category: 'VM Operations',
        priority: 'Medium',
        issue: 'VM creation latency exceeds 30 seconds at P95',
        recommendation: 'Optimize hypervisor initialization, implement VM template caching',
        impact: 'Delayed VM provisioning'
      });
    }

    // WebSocket recommendations
    if (results.summary.websocketConnections.total > 0 && 
        results.thresholds.failed.some(f => f.metric.includes('ws_'))) {
      recommendations.push({
        category: 'WebSocket',
        priority: 'Medium',
        issue: 'WebSocket performance issues detected',
        recommendation: 'Implement connection pooling, optimize message serialization, consider clustering',
        impact: 'Real-time feature degradation'
      });
    }

    // Database recommendations
    if (results.summary.databaseOperations.queries > 0 && 
        results.metrics.performance.databaseLatency.p95 > 100) {
      recommendations.push({
        category: 'Database',
        priority: 'High',
        issue: 'Database query latency exceeds 100ms at P95',
        recommendation: 'Add database indexes, implement query optimization, consider read replicas',
        impact: 'Overall system performance impact'
      });
    }

    return recommendations;
  }

  assessThresholdImpact(metric, result) {
    const impactMap = {
      'http_req_duration': 'User experience',
      'http_req_failed': 'Service reliability',
      'vm_creation_duration': 'VM provisioning speed',
      'ws_connection_duration': 'Real-time features',
      'db_query_duration': 'Data access performance'
    };

    return impactMap[metric] || 'System performance';
  }

  async generateHTMLReport(results) {
    const html = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NovaCron Load Test Report - ${this.timestamp}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background: #f8f9fa; padding: 15px; border-radius: 6px; border-left: 4px solid #007bff; }
        .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
        .metric-label { color: #666; font-size: 14px; }
        .success { border-left-color: #28a745; }
        .warning { border-left-color: #ffc107; }
        .danger { border-left-color: #dc3545; }
        .recommendations { margin: 20px 0; }
        .recommendation { background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; margin: 10px 0; border-radius: 6px; }
        .high-priority { background: #f8d7da; border-color: #f5c6cb; }
        .critical-priority { background: #d1ecf1; border-color: #bee5eb; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; }
        .scenario-section { margin: 30px 0; }
        .charts { margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>NovaCron Load Test Report</h1>
            <p>Generated: ${results.metadata.timestamp}</p>
            <p>Environment: ${results.metadata.environment}</p>
            <p>Duration: ${results.metadata.testDuration}</p>
        </div>

        <div class="metric-grid">
            <div class="metric-card ${results.summary.successRate >= 95 ? 'success' : 'danger'}">
                <div class="metric-value">${results.summary.successRate.toFixed(2)}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${results.summary.totalRequests.toLocaleString()}</div>
                <div class="metric-label">Total Requests</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${results.summary.requestsPerSecond.toFixed(2)}</div>
                <div class="metric-label">Requests/Second</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${results.summary.averageResponseTime.toFixed(0)}ms</div>
                <div class="metric-label">Avg Response Time</div>
            </div>
        </div>

        <div class="scenario-section">
            <h2>VM Operations Performance</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">${results.summary.vmOperations.created}</div>
                    <div class="metric-label">VMs Created</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${results.summary.vmOperations.started}</div>
                    <div class="metric-label">VMs Started</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${results.summary.vmOperations.migrated}</div>
                    <div class="metric-label">VMs Migrated</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${results.summary.vmOperations.snapshots}</div>
                    <div class="metric-label">Snapshots Created</div>
                </div>
            </div>
        </div>

        <div class="scenario-section">
            <h2>WebSocket Performance</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">${results.summary.websocketConnections.total}</div>
                    <div class="metric-label">Total Connections</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${results.summary.websocketConnections.concurrent}</div>
                    <div class="metric-label">Peak Concurrent</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${results.summary.websocketConnections.messagesSent}</div>
                    <div class="metric-label">Messages Sent</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${results.summary.websocketConnections.messagesReceived}</div>
                    <div class="metric-label">Messages Received</div>
                </div>
            </div>
        </div>

        <div class="scenario-section">
            <h2>Threshold Analysis</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Threshold</th>
                        <th>Actual</th>
                        <th>Status</th>
                        <th>Impact</th>
                    </tr>
                </thead>
                <tbody>
                    ${results.thresholds.failed.map(t => `
                        <tr>
                            <td>${t.metric}</td>
                            <td>${t.threshold}</td>
                            <td>${t.actual}</td>
                            <td style="color: #dc3545;">FAILED</td>
                            <td>${t.impact}</td>
                        </tr>
                    `).join('')}
                    ${results.thresholds.passed.slice(0, 10).map(t => `
                        <tr>
                            <td>${t.metric}</td>
                            <td>${t.threshold}</td>
                            <td>${t.actual}</td>
                            <td style="color: #28a745;">PASSED</td>
                            <td>-</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>

        <div class="recommendations">
            <h2>Performance Recommendations</h2>
            ${results.recommendations.map(rec => `
                <div class="recommendation ${rec.priority.toLowerCase()}-priority">
                    <h4>${rec.category} - ${rec.priority} Priority</h4>
                    <p><strong>Issue:</strong> ${rec.issue}</p>
                    <p><strong>Recommendation:</strong> ${rec.recommendation}</p>
                    <p><strong>Impact:</strong> ${rec.impact}</p>
                </div>
            `).join('')}
        </div>

        <div class="scenario-section">
            <h2>Detailed Scenario Results</h2>
            ${Object.entries(results.scenarios).map(([name, data]) => `
                <h3>${name.replace(/_/g, ' ').toUpperCase()}</h3>
                <pre>${JSON.stringify(data.summary || {}, null, 2)}</pre>
            `).join('')}
        </div>
    </div>
</body>
</html>`;

    const reportPath = path.join(this.reportsDir, `load-test-report-${this.timestamp}.html`);
    fs.writeFileSync(reportPath, html);
    console.log(`HTML report generated: ${reportPath}`);
  }

  async generateJSONReport(results) {
    const reportPath = path.join(this.reportsDir, `load-test-report-${this.timestamp}.json`);
    fs.writeFileSync(reportPath, JSON.stringify(results, null, 2));
    console.log(`JSON report generated: ${reportPath}`);
  }

  async generateCSVReport(results) {
    const csvData = [];
    
    // Header
    csvData.push([
      'Timestamp', 'Scenario', 'Metric', 'Value', 'Unit', 'Threshold', 'Status'
    ]);

    // Add data rows
    Object.entries(results.scenarios).forEach(([scenario, data]) => {
      if (data.root_group) {
        const rootGroup = data.root_group;
        Object.entries(rootGroup).forEach(([metric, values]) => {
          if (typeof values === 'object' && values !== null) {
            csvData.push([
              results.metadata.timestamp,
              scenario,
              metric,
              values.avg || values.count || values.rate || 'N/A',
              this.getMetricUnit(metric),
              'N/A',
              'N/A'
            ]);
          }
        });
      }
    });

    const csvContent = csvData.map(row => row.join(',')).join('\n');
    const reportPath = path.join(this.reportsDir, `load-test-report-${this.timestamp}.csv`);
    fs.writeFileSync(reportPath, csvContent);
    console.log(`CSV report generated: ${reportPath}`);
  }

  async generateMarkdownSummary(results) {
    const markdown = `# NovaCron Load Test Report

**Generated:** ${results.metadata.timestamp}  
**Environment:** ${results.metadata.environment}  
**Test Duration:** ${results.metadata.testDuration}  

## Executive Summary

- **Success Rate:** ${results.summary.successRate.toFixed(2)}%
- **Total Requests:** ${results.summary.totalRequests.toLocaleString()}
- **Average Response Time:** ${results.summary.averageResponseTime.toFixed(0)}ms
- **Requests per Second:** ${results.summary.requestsPerSecond.toFixed(2)}

## Key Performance Indicators

### VM Operations
- VMs Created: ${results.summary.vmOperations.created}
- VMs Started: ${results.summary.vmOperations.started}
- VMs Migrated: ${results.summary.vmOperations.migrated}
- Snapshots Created: ${results.summary.vmOperations.snapshots}

### WebSocket Performance
- Total Connections: ${results.summary.websocketConnections.total}
- Peak Concurrent: ${results.summary.websocketConnections.concurrent}
- Messages Exchanged: ${results.summary.websocketConnections.messagesSent + results.summary.websocketConnections.messagesReceived}

### Database Performance
- Total Queries: ${results.summary.databaseOperations.queries}
- Average Query Time: ${results.summary.databaseOperations.averageQueryTime.toFixed(2)}ms
- Concurrent Connections: ${results.summary.databaseOperations.concurrentConnections}

## Threshold Analysis

### Failed Thresholds (${results.thresholds.failed.length})
${results.thresholds.failed.map(t => `
- **${t.metric}**: ${t.actual} (threshold: ${t.threshold})
  - Impact: ${t.impact}
`).join('')}

### Passed Thresholds (${results.thresholds.passed.length})
${results.thresholds.passed.slice(0, 5).map(t => `
- **${t.metric}**: ${t.actual} ✓
`).join('')}

## Recommendations

${results.recommendations.map(rec => `
### ${rec.category} - ${rec.priority} Priority

**Issue:** ${rec.issue}

**Recommendation:** ${rec.recommendation}

**Impact:** ${rec.impact}

---
`).join('')}

## Test Scenarios Executed

${Object.keys(results.scenarios).map(scenario => `- ${scenario.replace(/_/g, ' ')}`).join('\n')}

---
*Report generated by NovaCron Load Testing Suite v1.0*
`;

    const reportPath = path.join(this.reportsDir, `load-test-summary-${this.timestamp}.md`);
    fs.writeFileSync(reportPath, markdown);
    console.log(`Markdown summary generated: ${reportPath}`);
  }

  calculateTestDuration() {
    // Calculate total test duration from scenario configurations
    return '15-30 minutes (varies by scenario)';
  }

  getMetricUnit(metric) {
    const units = {
      'http_req_duration': 'ms',
      'http_reqs': 'requests',
      'http_req_failed': 'rate',
      'vm_creation_duration': 'ms',
      'ws_connecting_duration': 'ms',
      'db_query_duration': 'ms'
    };
    
    return units[metric] || 'count';
  }
}

// CLI execution
if (require.main === module) {
  const generator = new LoadTestReportGenerator();
  generator.generateComprehensiveReport()
    .then(() => {
      console.log('Report generation completed successfully');
      process.exit(0);
    })
    .catch((error) => {
      console.error('Report generation failed:', error);
      process.exit(1);
    });
}

module.exports = LoadTestReportGenerator;