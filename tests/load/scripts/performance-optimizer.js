#!/usr/bin/env node

/**
 * NovaCron Performance Optimizer
 * Real-time performance analysis and optimization recommendations
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

class PerformanceOptimizer {
    constructor() {
        this.config = this.loadConfig();
        this.metrics = new Map();
        this.recommendations = [];
        this.thresholds = {
            responseTime: {
                excellent: 100,
                good: 500,
                acceptable: 1000,
                poor: 2000
            },
            throughput: {
                excellent: 1000,
                good: 500,
                acceptable: 200,
                poor: 100
            },
            errorRate: {
                excellent: 0.001,
                good: 0.01,
                acceptable: 0.05,
                poor: 0.1
            },
            resourceUsage: {
                cpu: 80,
                memory: 85,
                disk: 90
            }
        };
    }

    loadConfig() {
        try {
            return JSON.parse(fs.readFileSync(path.join(__dirname, '../configs/test-config.js'), 'utf8'));
        } catch (error) {
            console.warn('Could not load test config, using defaults');
            return {};
        }
    }

    /**
     * Analyze performance metrics from k6 results
     */
    async analyzePerformanceMetrics(resultsFile) {
        console.log(`\n🔍 Analyzing performance metrics from ${resultsFile}...`);
        
        if (!fs.existsSync(resultsFile)) {
            throw new Error(`Results file not found: ${resultsFile}`);
        }

        const results = JSON.parse(fs.readFileSync(resultsFile, 'utf8'));
        
        // Extract key metrics
        const metrics = {
            responseTime: this.extractResponseTimeMetrics(results),
            throughput: this.extractThroughputMetrics(results),
            errorRate: this.extractErrorRateMetrics(results),
            resourceUsage: this.extractResourceMetrics(results)
        };

        this.metrics.set('current', metrics);
        return metrics;
    }

    extractResponseTimeMetrics(results) {
        const httpReqDuration = results.metrics?.http_req_duration;
        return {
            avg: httpReqDuration?.avg || 0,
            p50: httpReqDuration?.p50 || 0,
            p95: httpReqDuration?.p95 || 0,
            p99: httpReqDuration?.p99 || 0,
            max: httpReqDuration?.max || 0,
            rating: this.rateResponseTime(httpReqDuration?.p95 || 0)
        };
    }

    extractThroughputMetrics(results) {
        const httpReqs = results.metrics?.http_reqs;
        const vus = results.metrics?.vus;
        const iterations = results.metrics?.iterations;
        
        return {
            requestsPerSecond: httpReqs?.rate || 0,
            totalRequests: httpReqs?.count || 0,
            maxVirtualUsers: vus?.max || 0,
            iterationsPerSecond: iterations?.rate || 0,
            rating: this.rateThroughput(httpReqs?.rate || 0)
        };
    }

    extractErrorRateMetrics(results) {
        const httpReqFailed = results.metrics?.http_req_failed;
        const httpReqs = results.metrics?.http_reqs;
        
        const errorRate = httpReqFailed?.rate || 0;
        const totalErrors = httpReqFailed?.count || 0;
        const totalRequests = httpReqs?.count || 0;
        
        return {
            rate: errorRate,
            count: totalErrors,
            percentage: totalRequests > 0 ? (totalErrors / totalRequests) * 100 : 0,
            rating: this.rateErrorRate(errorRate)
        };
    }

    extractResourceMetrics(results) {
        // Extract from custom metrics if available
        const customMetrics = results.metrics || {};
        
        return {
            cpu: customMetrics.cpu_usage?.avg || 0,
            memory: customMetrics.memory_usage?.avg || 0,
            disk: customMetrics.disk_usage?.avg || 0,
            network: customMetrics.network_io?.avg || 0
        };
    }

    rateResponseTime(p95) {
        const thresholds = this.thresholds.responseTime;
        if (p95 <= thresholds.excellent) return 'excellent';
        if (p95 <= thresholds.good) return 'good';
        if (p95 <= thresholds.acceptable) return 'acceptable';
        return 'poor';
    }

    rateThroughput(rps) {
        const thresholds = this.thresholds.throughput;
        if (rps >= thresholds.excellent) return 'excellent';
        if (rps >= thresholds.good) return 'good';
        if (rps >= thresholds.acceptable) return 'acceptable';
        return 'poor';
    }

    rateErrorRate(rate) {
        const thresholds = this.thresholds.errorRate;
        if (rate <= thresholds.excellent) return 'excellent';
        if (rate <= thresholds.good) return 'good';
        if (rate <= thresholds.acceptable) return 'acceptable';
        return 'poor';
    }

    /**
     * Generate optimization recommendations based on metrics
     */
    generateRecommendations(metrics) {
        console.log('\n💡 Generating optimization recommendations...');
        
        this.recommendations = [];

        // Response time recommendations
        if (metrics.responseTime.rating === 'poor') {
            this.recommendations.push({
                category: 'Response Time',
                severity: 'high',
                issue: `High response times (P95: ${metrics.responseTime.p95}ms)`,
                recommendations: [
                    'Enable response compression (gzip/brotli)',
                    'Implement database connection pooling',
                    'Add Redis caching for frequently accessed data',
                    'Optimize database queries with proper indexing',
                    'Consider CDN for static assets'
                ]
            });
        } else if (metrics.responseTime.rating === 'acceptable') {
            this.recommendations.push({
                category: 'Response Time',
                severity: 'medium',
                issue: `Moderate response times (P95: ${metrics.responseTime.p95}ms)`,
                recommendations: [
                    'Implement HTTP/2 or HTTP/3',
                    'Add database query optimization',
                    'Consider response caching strategies',
                    'Profile slow endpoints for bottlenecks'
                ]
            });
        }

        // Throughput recommendations
        if (metrics.throughput.rating === 'poor') {
            this.recommendations.push({
                category: 'Throughput',
                severity: 'high',
                issue: `Low throughput (${metrics.throughput.requestsPerSecond} RPS)`,
                recommendations: [
                    'Scale API server horizontally',
                    'Implement load balancing',
                    'Optimize goroutine pools in Go backend',
                    'Add connection pooling',
                    'Consider async processing for heavy operations'
                ]
            });
        }

        // Error rate recommendations
        if (metrics.errorRate.rating === 'poor') {
            this.recommendations.push({
                category: 'Error Rate',
                severity: 'critical',
                issue: `High error rate (${(metrics.errorRate.percentage).toFixed(2)}%)`,
                recommendations: [
                    'Implement circuit breaker patterns',
                    'Add comprehensive error handling',
                    'Increase timeout values for slow operations',
                    'Implement retry logic with exponential backoff',
                    'Add health check endpoints'
                ]
            });
        }

        // Resource usage recommendations
        Object.entries(metrics.resourceUsage).forEach(([resource, usage]) => {
            if (usage > this.thresholds.resourceUsage[resource]) {
                this.recommendations.push({
                    category: 'Resource Usage',
                    severity: 'medium',
                    issue: `High ${resource} usage (${usage}%)`,
                    recommendations: [
                        `Optimize ${resource} usage patterns`,
                        `Add ${resource} monitoring alerts`,
                        `Consider horizontal scaling`,
                        `Profile ${resource} bottlenecks`
                    ]
                });
            }
        });

        return this.recommendations;
    }

    /**
     * Compare current metrics with baseline
     */
    compareWithBaseline(currentMetrics, baselineFile) {
        console.log('\n📊 Comparing with baseline performance...');
        
        if (!fs.existsSync(baselineFile)) {
            console.log('⚠️ No baseline found, establishing current metrics as baseline');
            this.saveBaseline(currentMetrics, baselineFile);
            return null;
        }

        const baseline = JSON.parse(fs.readFileSync(baselineFile, 'utf8'));
        
        const comparison = {
            responseTime: {
                current: currentMetrics.responseTime.p95,
                baseline: baseline.responseTime.p95,
                change: ((currentMetrics.responseTime.p95 - baseline.responseTime.p95) / baseline.responseTime.p95) * 100
            },
            throughput: {
                current: currentMetrics.throughput.requestsPerSecond,
                baseline: baseline.throughput.requestsPerSecond,
                change: ((currentMetrics.throughput.requestsPerSecond - baseline.throughput.requestsPerSecond) / baseline.throughput.requestsPerSecond) * 100
            },
            errorRate: {
                current: currentMetrics.errorRate.rate,
                baseline: baseline.errorRate.rate,
                change: ((currentMetrics.errorRate.rate - baseline.errorRate.rate) / (baseline.errorRate.rate || 0.001)) * 100
            }
        };

        this.analyzeRegression(comparison);
        return comparison;
    }

    analyzeRegression(comparison) {
        const regressionThreshold = 10; // 10% degradation threshold
        
        Object.entries(comparison).forEach(([metric, data]) => {
            if (metric === 'responseTime' && data.change > regressionThreshold) {
                this.recommendations.push({
                    category: 'Performance Regression',
                    severity: 'critical',
                    issue: `Response time regression: +${data.change.toFixed(1)}%`,
                    recommendations: [
                        'Investigate recent code changes affecting performance',
                        'Profile application for new bottlenecks',
                        'Check database query performance',
                        'Verify infrastructure changes'
                    ]
                });
            } else if (metric === 'throughput' && data.change < -regressionThreshold) {
                this.recommendations.push({
                    category: 'Performance Regression',
                    severity: 'critical',
                    issue: `Throughput regression: ${data.change.toFixed(1)}%`,
                    recommendations: [
                        'Check for resource constraints',
                        'Verify connection pool sizing',
                        'Review recent algorithm changes',
                        'Monitor system resource usage'
                    ]
                });
            } else if (metric === 'errorRate' && data.change > regressionThreshold) {
                this.recommendations.push({
                    category: 'Performance Regression',
                    severity: 'critical',
                    issue: `Error rate regression: +${data.change.toFixed(1)}%`,
                    recommendations: [
                        'Investigate new error patterns',
                        'Check API endpoint stability',
                        'Verify database connection health',
                        'Review recent deployment changes'
                    ]
                });
            }
        });
    }

    saveBaseline(metrics, baselineFile) {
        const baselineDir = path.dirname(baselineFile);
        if (!fs.existsSync(baselineDir)) {
            fs.mkdirSync(baselineDir, { recursive: true });
        }
        
        fs.writeFileSync(baselineFile, JSON.stringify(metrics, null, 2));
        console.log(`✅ Baseline saved to ${baselineFile}`);
    }

    /**
     * Generate performance optimization report
     */
    generateOptimizationReport(metrics, comparison = null) {
        console.log('\n📝 Generating optimization report...');
        
        const report = {
            timestamp: new Date().toISOString(),
            summary: this.generateSummary(metrics),
            metrics: metrics,
            comparison: comparison,
            recommendations: this.recommendations,
            actionPlan: this.generateActionPlan()
        };

        // Save detailed JSON report
        const reportFile = path.join(__dirname, '../reports', `optimization-report-${Date.now()}.json`);
        const reportDir = path.dirname(reportFile);
        
        if (!fs.existsSync(reportDir)) {
            fs.mkdirSync(reportDir, { recursive: true });
        }
        
        fs.writeFileSync(reportFile, JSON.stringify(report, null, 2));
        
        // Generate markdown summary
        const markdownReport = this.generateMarkdownReport(report);
        const markdownFile = reportFile.replace('.json', '.md');
        fs.writeFileSync(markdownFile, markdownReport);
        
        console.log(`✅ Optimization report saved:`);
        console.log(`   JSON: ${reportFile}`);
        console.log(`   Markdown: ${markdownFile}`);
        
        return report;
    }

    generateSummary(metrics) {
        const overallRating = this.calculateOverallRating(metrics);
        
        return {
            overallRating,
            responseTime: `${metrics.responseTime.p95}ms (${metrics.responseTime.rating})`,
            throughput: `${metrics.throughput.requestsPerSecond} RPS (${metrics.throughput.rating})`,
            errorRate: `${(metrics.errorRate.percentage).toFixed(3)}% (${metrics.errorRate.rating})`,
            recommendationCount: this.recommendations.length,
            urgentIssues: this.recommendations.filter(r => r.severity === 'critical').length
        };
    }

    calculateOverallRating(metrics) {
        const ratings = ['excellent', 'good', 'acceptable', 'poor'];
        const scores = {
            responseTime: ratings.indexOf(metrics.responseTime.rating),
            throughput: ratings.indexOf(metrics.throughput.rating),
            errorRate: ratings.indexOf(metrics.errorRate.rating)
        };
        
        const avgScore = (scores.responseTime + scores.throughput + scores.errorRate) / 3;
        return ratings[Math.round(avgScore)];
    }

    generateActionPlan() {
        const criticalItems = this.recommendations.filter(r => r.severity === 'critical');
        const highItems = this.recommendations.filter(r => r.severity === 'high');
        const mediumItems = this.recommendations.filter(r => r.severity === 'medium');
        
        return {
            immediate: criticalItems.map(item => ({
                priority: 1,
                category: item.category,
                action: item.recommendations[0] // First recommendation as immediate action
            })),
            shortTerm: highItems.map(item => ({
                priority: 2,
                category: item.category,
                action: item.recommendations[0]
            })),
            longTerm: mediumItems.map(item => ({
                priority: 3,
                category: item.category,
                action: item.recommendations[0]
            }))
        };
    }

    generateMarkdownReport(report) {
        const { summary, metrics, comparison, recommendations } = report;
        
        let markdown = `# NovaCron Performance Optimization Report\n\n`;
        markdown += `**Generated:** ${new Date(report.timestamp).toLocaleString()}\n\n`;
        
        // Executive Summary
        markdown += `## Executive Summary\n\n`;
        markdown += `- **Overall Rating:** ${summary.overallRating.toUpperCase()}\n`;
        markdown += `- **Response Time:** ${summary.responseTime}\n`;
        markdown += `- **Throughput:** ${summary.throughput}\n`;
        markdown += `- **Error Rate:** ${summary.errorRate}\n`;
        markdown += `- **Recommendations:** ${summary.recommendationCount} total, ${summary.urgentIssues} urgent\n\n`;
        
        // Performance Metrics
        markdown += `## Performance Metrics\n\n`;
        markdown += `### Response Time Analysis\n`;
        markdown += `| Metric | Value | Rating |\n`;
        markdown += `|--------|-------|--------|\n`;
        markdown += `| Average | ${metrics.responseTime.avg}ms | ${metrics.responseTime.rating} |\n`;
        markdown += `| P50 | ${metrics.responseTime.p50}ms | - |\n`;
        markdown += `| P95 | ${metrics.responseTime.p95}ms | - |\n`;
        markdown += `| P99 | ${metrics.responseTime.p99}ms | - |\n`;
        markdown += `| Max | ${metrics.responseTime.max}ms | - |\n\n`;
        
        markdown += `### Throughput Analysis\n`;
        markdown += `| Metric | Value | Rating |\n`;
        markdown += `|--------|-------|--------|\n`;
        markdown += `| Requests/sec | ${metrics.throughput.requestsPerSecond} | ${metrics.throughput.rating} |\n`;
        markdown += `| Total Requests | ${metrics.throughput.totalRequests} | - |\n`;
        markdown += `| Max VUs | ${metrics.throughput.maxVirtualUsers} | - |\n`;
        markdown += `| Iterations/sec | ${metrics.throughput.iterationsPerSecond} | - |\n\n`;
        
        // Baseline Comparison
        if (comparison) {
            markdown += `## Baseline Comparison\n\n`;
            markdown += `| Metric | Current | Baseline | Change |\n`;
            markdown += `|--------|---------|----------|--------|\n`;
            markdown += `| Response Time (P95) | ${comparison.responseTime.current}ms | ${comparison.responseTime.baseline}ms | ${comparison.responseTime.change > 0 ? '+' : ''}${comparison.responseTime.change.toFixed(1)}% |\n`;
            markdown += `| Throughput | ${comparison.throughput.current} RPS | ${comparison.throughput.baseline} RPS | ${comparison.throughput.change > 0 ? '+' : ''}${comparison.throughput.change.toFixed(1)}% |\n`;
            markdown += `| Error Rate | ${(comparison.errorRate.current * 100).toFixed(3)}% | ${(comparison.errorRate.baseline * 100).toFixed(3)}% | ${comparison.errorRate.change > 0 ? '+' : ''}${comparison.errorRate.change.toFixed(1)}% |\n\n`;
        }
        
        // Recommendations
        if (recommendations.length > 0) {
            markdown += `## Optimization Recommendations\n\n`;
            
            const groupedRecs = this.groupRecommendationsBySeverity(recommendations);
            
            ['critical', 'high', 'medium'].forEach(severity => {
                if (groupedRecs[severity] && groupedRecs[severity].length > 0) {
                    markdown += `### ${severity.toUpperCase()} Priority\n\n`;
                    groupedRecs[severity].forEach((rec, index) => {
                        markdown += `#### ${index + 1}. ${rec.category}: ${rec.issue}\n\n`;
                        rec.recommendations.forEach(recommendation => {
                            markdown += `- ${recommendation}\n`;
                        });
                        markdown += `\n`;
                    });
                }
            });
        }
        
        // Action Plan
        markdown += `## Action Plan\n\n`;
        const actionPlan = report.actionPlan;
        
        if (actionPlan.immediate.length > 0) {
            markdown += `### Immediate Actions (Critical)\n\n`;
            actionPlan.immediate.forEach((action, index) => {
                markdown += `${index + 1}. **${action.category}**: ${action.action}\n`;
            });
            markdown += `\n`;
        }
        
        if (actionPlan.shortTerm.length > 0) {
            markdown += `### Short-term Improvements (1-2 weeks)\n\n`;
            actionPlan.shortTerm.forEach((action, index) => {
                markdown += `${index + 1}. **${action.category}**: ${action.action}\n`;
            });
            markdown += `\n`;
        }
        
        if (actionPlan.longTerm.length > 0) {
            markdown += `### Long-term Optimizations (1+ months)\n\n`;
            actionPlan.longTerm.forEach((action, index) => {
                markdown += `${index + 1}. **${action.category}**: ${action.action}\n`;
            });
            markdown += `\n`;
        }
        
        return markdown;
    }

    groupRecommendationsBySeverity(recommendations) {
        return recommendations.reduce((groups, rec) => {
            if (!groups[rec.severity]) {
                groups[rec.severity] = [];
            }
            groups[rec.severity].push(rec);
            return groups;
        }, {});
    }

    /**
     * Real-time performance monitoring
     */
    async startRealTimeMonitoring(duration = 60000) {
        console.log(`\n📊 Starting real-time performance monitoring for ${duration/1000}s...`);
        
        const startTime = Date.now();
        const interval = 5000; // 5 second intervals
        const dataPoints = [];
        
        const monitorInterval = setInterval(async () => {
            try {
                const currentTime = Date.now();
                const elapsedTime = currentTime - startTime;
                
                // Collect system metrics
                const systemMetrics = await this.collectSystemMetrics();
                
                // Collect application metrics (if Prometheus is available)
                let appMetrics = {};
                try {
                    appMetrics = await this.collectApplicationMetrics();
                } catch (error) {
                    console.warn('Application metrics not available:', error.message);
                }
                
                const dataPoint = {
                    timestamp: currentTime,
                    elapsedTime,
                    system: systemMetrics,
                    application: appMetrics
                };
                
                dataPoints.push(dataPoint);
                
                console.log(`[${elapsedTime/1000}s] CPU: ${systemMetrics.cpu}%, Memory: ${systemMetrics.memory}%, Load: ${systemMetrics.loadAvg}`);
                
                if (elapsedTime >= duration) {
                    clearInterval(monitorInterval);
                    this.saveMonitoringData(dataPoints);
                    console.log('✅ Real-time monitoring completed');
                }
            } catch (error) {
                console.error('Monitoring error:', error.message);
            }
        }, interval);
        
        return dataPoints;
    }

    async collectSystemMetrics() {
        try {
            // CPU usage
            const cpuUsage = execSync("grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$3+$4+$5)} END {print usage}'", { encoding: 'utf8' }).trim();
            
            // Memory usage
            const memInfo = execSync("free | grep Mem | awk '{print ($3/$2) * 100.0}'", { encoding: 'utf8' }).trim();
            
            // Load average
            const loadAvg = execSync("uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//'", { encoding: 'utf8' }).trim();
            
            // Disk usage
            const diskUsage = execSync("df / | tail -1 | awk '{print $5}' | sed 's/%//'", { encoding: 'utf8' }).trim();
            
            return {
                cpu: parseFloat(cpuUsage) || 0,
                memory: parseFloat(memInfo) || 0,
                loadAvg: parseFloat(loadAvg) || 0,
                disk: parseFloat(diskUsage) || 0
            };
        } catch (error) {
            console.warn('Could not collect system metrics:', error.message);
            return { cpu: 0, memory: 0, loadAvg: 0, disk: 0 };
        }
    }

    async collectApplicationMetrics() {
        // Try to fetch from Prometheus if available
        try {
            const { spawn } = require('child_process');
            
            // Check if Prometheus is running
            const prometheusUrl = 'http://localhost:9090';
            
            return new Promise((resolve, reject) => {
                const curl = spawn('curl', ['-s', `${prometheusUrl}/api/v1/query?query=up`]);
                
                let data = '';
                curl.stdout.on('data', (chunk) => {
                    data += chunk;
                });
                
                curl.on('close', (code) => {
                    if (code === 0) {
                        try {
                            const result = JSON.parse(data);
                            resolve({
                                prometheusUp: result.status === 'success',
                                targets: result.data?.result?.length || 0
                            });
                        } catch (error) {
                            resolve({ prometheusUp: false, targets: 0 });
                        }
                    } else {
                        resolve({ prometheusUp: false, targets: 0 });
                    }
                });
            });
        } catch (error) {
            return { prometheusUp: false, targets: 0 };
        }
    }

    saveMonitoringData(dataPoints) {
        const monitoringFile = path.join(__dirname, '../reports', `monitoring-data-${Date.now()}.json`);
        const reportDir = path.dirname(monitoringFile);
        
        if (!fs.existsSync(reportDir)) {
            fs.mkdirSync(reportDir, { recursive: true });
        }
        
        fs.writeFileSync(monitoringFile, JSON.stringify(dataPoints, null, 2));
        console.log(`📊 Monitoring data saved to ${monitoringFile}`);
    }

    /**
     * Optimize test configuration based on current system capabilities
     */
    optimizeTestConfiguration() {
        console.log('\n⚙️ Optimizing test configuration...');
        
        const systemMetrics = execSync("nproc && free -m | grep Mem | awk '{print $2}'", { encoding: 'utf8' }).trim().split('\n');
        const cpuCores = parseInt(systemMetrics[0]);
        const memoryMB = parseInt(systemMetrics[1]);
        
        const recommendations = {
            maxVirtualUsers: Math.min(cpuCores * 250, memoryMB * 2),
            optimalBatchSize: Math.min(cpuCores * 50, 500),
            recommendedDuration: memoryMB > 8000 ? '30m' : '15m',
            parallelScenarios: Math.min(cpuCores, 4)
        };
        
        console.log('System Analysis:');
        console.log(`  CPU Cores: ${cpuCores}`);
        console.log(`  Memory: ${memoryMB}MB`);
        console.log('\nOptimized Configuration:');
        console.log(`  Max VUs: ${recommendations.maxVirtualUsers}`);
        console.log(`  Batch Size: ${recommendations.optimalBatchSize}`);
        console.log(`  Duration: ${recommendations.recommendedDuration}`);
        console.log(`  Parallel Scenarios: ${recommendations.parallelScenarios}`);
        
        return recommendations;
    }
}

// CLI Interface
async function main() {
    const args = process.argv.slice(2);
    const command = args[0];
    
    const optimizer = new PerformanceOptimizer();
    
    try {
        switch (command) {
            case 'analyze':
                const resultsFile = args[1];
                if (!resultsFile) {
                    console.error('Usage: performance-optimizer.js analyze <results-file.json>');
                    process.exit(1);
                }
                
                const metrics = await optimizer.analyzePerformanceMetrics(resultsFile);
                const recommendations = optimizer.generateRecommendations(metrics);
                
                // Try to compare with baseline
                const baselineFile = path.join(__dirname, '../reports/baseline-metrics.json');
                const comparison = optimizer.compareWithBaseline(metrics, baselineFile);
                
                const report = optimizer.generateOptimizationReport(metrics, comparison);
                
                console.log('\n🎯 Performance Analysis Complete!');
                console.log(`Overall Rating: ${report.summary.overallRating.toUpperCase()}`);
                console.log(`Recommendations: ${report.summary.recommendationCount}`);
                break;
                
            case 'monitor':
                const duration = parseInt(args[1]) || 60000;
                await optimizer.startRealTimeMonitoring(duration);
                break;
                
            case 'optimize-config':
                const optimized = optimizer.optimizeTestConfiguration();
                console.log('\n✅ Configuration optimization complete');
                break;
                
            case 'baseline':
                const baselineResults = args[1];
                if (!baselineResults) {
                    console.error('Usage: performance-optimizer.js baseline <results-file.json>');
                    process.exit(1);
                }
                
                const baselineMetrics = await optimizer.analyzePerformanceMetrics(baselineResults);
                const baselinePath = path.join(__dirname, '../reports/baseline-metrics.json');
                optimizer.saveBaseline(baselineMetrics, baselinePath);
                break;
                
            default:
                console.log(`
NovaCron Performance Optimizer

Usage: node performance-optimizer.js <command> [options]

Commands:
  analyze <file>     Analyze k6 results and generate recommendations
  monitor [duration] Start real-time performance monitoring (default: 60s)
  optimize-config    Optimize test configuration for current system
  baseline <file>    Set performance baseline from results file

Examples:
  node performance-optimizer.js analyze reports/api-load-test-results.json
  node performance-optimizer.js monitor 120000
  node performance-optimizer.js optimize-config
  node performance-optimizer.js baseline reports/baseline-results.json
                `);
                break;
        }
    } catch (error) {
        console.error('❌ Error:', error.message);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}

module.exports = PerformanceOptimizer;