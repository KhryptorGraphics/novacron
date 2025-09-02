#!/usr/bin/env node

/**
 * NovaCron Benchmark Comparator
 * Advanced performance comparison and regression analysis
 */

const fs = require('fs');
const path = require('path');

class BenchmarkComparator {
    constructor() {
        this.comparisonThresholds = {
            regression: {
                responseTime: 15, // 15% increase is regression
                throughput: 15,   // 15% decrease is regression
                errorRate: 100    // 100% increase (doubling) is regression
            },
            improvement: {
                responseTime: 10, // 10% decrease is improvement
                throughput: 10,   // 10% increase is improvement
                errorRate: 50     // 50% decrease is improvement
            }
        };
    }

    /**
     * Compare multiple benchmark results
     */
    async compareResults(resultFiles) {
        console.log('🔍 Comparing benchmark results...');
        
        const results = [];
        
        for (const file of resultFiles) {
            if (!fs.existsSync(file)) {
                console.warn(`⚠️ File not found: ${file}`);
                continue;
            }
            
            try {
                const data = JSON.parse(fs.readFileSync(file, 'utf8'));
                const metrics = this.extractComparableMetrics(data, file);
                results.push(metrics);
            } catch (error) {
                console.error(`❌ Error processing ${file}:`, error.message);
            }
        }
        
        if (results.length < 2) {
            throw new Error('Need at least 2 result files for comparison');
        }
        
        return this.performComparison(results);
    }

    /**
     * Extract comparable metrics from k6 results
     */
    extractComparableMetrics(data, sourceFile) {
        const metrics = data.metrics || {};
        const fileName = path.basename(sourceFile, '.json');
        
        // Parse timestamp from filename if possible
        const timestampMatch = fileName.match(/(\d{13})/);
        const timestamp = timestampMatch ? parseInt(timestampMatch[1]) : Date.now();
        
        return {
            source: sourceFile,
            timestamp,
            name: fileName,
            responseTime: {
                avg: metrics.http_req_duration?.avg || 0,
                p50: metrics.http_req_duration?.p50 || 0,
                p95: metrics.http_req_duration?.p95 || 0,
                p99: metrics.http_req_duration?.p99 || 0,
                max: metrics.http_req_duration?.max || 0
            },
            throughput: {
                requestsPerSecond: metrics.http_reqs?.rate || 0,
                totalRequests: metrics.http_reqs?.count || 0,
                iterationsPerSecond: metrics.iterations?.rate || 0
            },
            errorRate: {
                rate: metrics.http_req_failed?.rate || 0,
                count: metrics.http_req_failed?.count || 0
            },
            virtualUsers: {
                max: metrics.vus?.max || 0,
                avg: metrics.vus?.avg || 0
            },
            testDuration: metrics.test_duration?.value || 0
        };
    }

    /**
     * Perform detailed comparison analysis
     */
    performComparison(results) {
        // Sort by timestamp
        results.sort((a, b) => a.timestamp - b.timestamp);
        
        const comparison = {
            metadata: {
                compareCount: results.length,
                timeRange: {
                    start: new Date(results[0].timestamp).toISOString(),
                    end: new Date(results[results.length - 1].timestamp).toISOString()
                }
            },
            results: results,
            analysis: {
                trends: this.analyzeTrends(results),
                regressions: this.detectRegressions(results),
                improvements: this.detectImprovements(results),
                stability: this.analyzeStability(results)
            },
            recommendations: this.generateComparisonRecommendations(results)
        };
        
        return comparison;
    }

    /**
     * Analyze performance trends over time
     */
    analyzeTrends(results) {
        if (results.length < 3) {
            return { message: 'Not enough data points for trend analysis' };
        }
        
        const trends = {
            responseTime: this.calculateTrend(results.map(r => r.responseTime.p95)),
            throughput: this.calculateTrend(results.map(r => r.throughput.requestsPerSecond)),
            errorRate: this.calculateTrend(results.map(r => r.errorRate.rate))
        };
        
        return {
            responseTime: {
                trend: trends.responseTime,
                direction: trends.responseTime > 5 ? 'worsening' : trends.responseTime < -5 ? 'improving' : 'stable'
            },
            throughput: {
                trend: trends.throughput,
                direction: trends.throughput > 5 ? 'improving' : trends.throughput < -5 ? 'worsening' : 'stable'
            },
            errorRate: {
                trend: trends.errorRate,
                direction: trends.errorRate > 5 ? 'worsening' : trends.errorRate < -5 ? 'improving' : 'stable'
            }
        };
    }

    calculateTrend(values) {
        if (values.length < 2) return 0;
        
        // Simple linear trend calculation
        const n = values.length;
        const sumX = (n * (n - 1)) / 2;
        const sumY = values.reduce((a, b) => a + b, 0);
        const sumXY = values.reduce((sum, y, x) => sum + (x * y), 0);
        const sumX2 = values.reduce((sum, _, x) => sum + (x * x), 0);
        
        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        const avgY = sumY / n;
        
        // Return trend as percentage change
        return avgY !== 0 ? (slope / avgY) * 100 : 0;
    }

    /**
     * Detect performance regressions
     */
    detectRegressions(results) {
        if (results.length < 2) return [];
        
        const regressions = [];
        const baseline = results[0];
        const latest = results[results.length - 1];
        
        // Response time regression
        const responseTimeChange = this.calculatePercentageChange(
            baseline.responseTime.p95,
            latest.responseTime.p95
        );
        
        if (responseTimeChange > this.comparisonThresholds.regression.responseTime) {
            regressions.push({
                metric: 'Response Time (P95)',
                baseline: `${baseline.responseTime.p95.toFixed(0)}ms`,
                current: `${latest.responseTime.p95.toFixed(0)}ms`,
                change: `+${responseTimeChange.toFixed(1)}%`,
                severity: responseTimeChange > 30 ? 'critical' : 'high',
                impact: 'User experience degradation'
            });
        }
        
        // Throughput regression
        const throughputChange = this.calculatePercentageChange(
            baseline.throughput.requestsPerSecond,
            latest.throughput.requestsPerSecond
        );
        
        if (throughputChange < -this.comparisonThresholds.regression.throughput) {
            regressions.push({
                metric: 'Throughput',
                baseline: `${baseline.throughput.requestsPerSecond.toFixed(0)} RPS`,
                current: `${latest.throughput.requestsPerSecond.toFixed(0)} RPS`,
                change: `${throughputChange.toFixed(1)}%`,
                severity: throughputChange < -40 ? 'critical' : 'high',
                impact: 'Reduced system capacity'
            });
        }
        
        // Error rate regression
        const errorRateChange = this.calculatePercentageChange(
            baseline.errorRate.rate,
            latest.errorRate.rate
        );
        
        if (errorRateChange > this.comparisonThresholds.regression.errorRate) {
            regressions.push({
                metric: 'Error Rate',
                baseline: `${(baseline.errorRate.rate * 100).toFixed(3)}%`,
                current: `${(latest.errorRate.rate * 100).toFixed(3)}%`,
                change: `+${errorRateChange.toFixed(1)}%`,
                severity: 'critical',
                impact: 'System reliability degradation'
            });
        }
        
        return regressions;
    }

    /**
     * Detect performance improvements
     */
    detectImprovements(results) {
        if (results.length < 2) return [];
        
        const improvements = [];
        const baseline = results[0];
        const latest = results[results.length - 1];
        
        // Response time improvement
        const responseTimeChange = this.calculatePercentageChange(
            baseline.responseTime.p95,
            latest.responseTime.p95
        );
        
        if (responseTimeChange < -this.comparisonThresholds.improvement.responseTime) {
            improvements.push({
                metric: 'Response Time (P95)',
                baseline: `${baseline.responseTime.p95.toFixed(0)}ms`,
                current: `${latest.responseTime.p95.toFixed(0)}ms`,
                change: `${responseTimeChange.toFixed(1)}%`,
                impact: 'Better user experience'
            });
        }
        
        // Throughput improvement
        const throughputChange = this.calculatePercentageChange(
            baseline.throughput.requestsPerSecond,
            latest.throughput.requestsPerSecond
        );
        
        if (throughputChange > this.comparisonThresholds.improvement.throughput) {
            improvements.push({
                metric: 'Throughput',
                baseline: `${baseline.throughput.requestsPerSecond.toFixed(0)} RPS`,
                current: `${latest.throughput.requestsPerSecond.toFixed(0)} RPS`,
                change: `+${throughputChange.toFixed(1)}%`,
                impact: 'Increased system capacity'
            });
        }
        
        return improvements;
    }

    calculatePercentageChange(baseline, current) {
        if (baseline === 0) return current > 0 ? 100 : 0;
        return ((current - baseline) / baseline) * 100;
    }

    /**
     * Analyze performance stability
     */
    analyzeStability(results) {
        if (results.length < 3) {
            return { message: 'Not enough data for stability analysis' };
        }
        
        const responseTimesP95 = results.map(r => r.responseTime.p95);
        const throughputs = results.map(r => r.throughput.requestsPerSecond);
        const errorRates = results.map(r => r.errorRate.rate);
        
        return {
            responseTime: {
                variability: this.calculateVariability(responseTimesP95),
                stability: this.assessStability(responseTimesP95)
            },
            throughput: {
                variability: this.calculateVariability(throughputs),
                stability: this.assessStability(throughputs)
            },
            errorRate: {
                variability: this.calculateVariability(errorRates),
                stability: this.assessStability(errorRates)
            }
        };
    }

    calculateVariability(values) {
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / values.length;
        const stdDev = Math.sqrt(variance);
        
        return {
            mean,
            stdDev,
            coefficientOfVariation: mean !== 0 ? (stdDev / mean) * 100 : 0
        };
    }

    assessStability(values) {
        const variability = this.calculateVariability(values);
        const cv = variability.coefficientOfVariation;
        
        if (cv < 10) return 'excellent';
        if (cv < 20) return 'good';
        if (cv < 30) return 'fair';
        return 'poor';
    }

    /**
     * Generate recommendations based on comparison
     */
    generateComparisonRecommendations(results) {
        const recommendations = [];
        const analysis = this.performComparison(results).analysis;
        
        // Trend-based recommendations
        if (analysis.trends.responseTime.direction === 'worsening') {
            recommendations.push({
                category: 'Performance Trend',
                priority: 'high',
                issue: 'Response time is trending upward',
                actions: [
                    'Profile application for performance bottlenecks',
                    'Monitor database query performance',
                    'Check for memory leaks',
                    'Review recent code changes for performance impact'
                ]
            });
        }
        
        if (analysis.trends.throughput.direction === 'worsening') {
            recommendations.push({
                category: 'Capacity Trend',
                priority: 'high',
                issue: 'Throughput is trending downward',
                actions: [
                    'Scale API servers horizontally',
                    'Optimize connection pooling',
                    'Review rate limiting configuration',
                    'Check system resource constraints'
                ]
            });
        }
        
        // Stability-based recommendations
        if (analysis.stability.responseTime.stability === 'poor') {
            recommendations.push({
                category: 'Stability',
                priority: 'medium',
                issue: 'Response time variability is high',
                actions: [
                    'Implement consistent caching strategies',
                    'Add connection pooling',
                    'Review garbage collection settings',
                    'Monitor external dependency performance'
                ]
            });
        }
        
        return recommendations;
    }

    /**
     * Generate comprehensive comparison report
     */
    generateComparisonReport(results, outputDir) {
        console.log('📝 Generating comparison report...');
        
        const comparison = this.performComparison(results);
        
        // Create output directory
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }
        
        // Save detailed JSON report
        const jsonFile = path.join(outputDir, `benchmark-comparison-${Date.now()}.json`);
        fs.writeFileSync(jsonFile, JSON.stringify(comparison, null, 2));
        
        // Generate HTML report
        const htmlReport = this.generateHTMLReport(comparison);
        const htmlFile = jsonFile.replace('.json', '.html');
        fs.writeFileSync(htmlFile, htmlReport);
        
        // Generate markdown summary
        const markdownReport = this.generateMarkdownReport(comparison);
        const markdownFile = jsonFile.replace('.json', '.md');
        fs.writeFileSync(markdownFile, markdownReport);
        
        console.log(`📊 Comparison report generated:`);
        console.log(`   JSON: ${jsonFile}`);
        console.log(`   HTML: ${htmlFile}`);
        console.log(`   Markdown: ${markdownFile}`);
        
        return comparison;
    }

    generateHTMLReport(comparison) {
        const { results, analysis, recommendations } = comparison;
        
        let html = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NovaCron Benchmark Comparison Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            border-bottom: 2px solid #eee;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #007bff;
        }
        .chart-container {
            position: relative;
            height: 400px;
            margin: 20px 0;
        }
        .regression {
            background-color: #fff5f5;
            border-left-color: #e53e3e;
        }
        .improvement {
            background-color: #f0fff4;
            border-left-color: #38a169;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        .trend-up { color: #e53e3e; }
        .trend-down { color: #38a169; }
        .trend-stable { color: #718096; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 NovaCron Benchmark Comparison</h1>
            <p><strong>Generated:</strong> ${new Date().toLocaleString()}</p>
            <p><strong>Results Compared:</strong> ${results.length}</p>
            <p><strong>Time Range:</strong> ${comparison.metadata.timeRange.start} to ${comparison.metadata.timeRange.end}</p>
        </div>
        
        <div class="metric-grid">
            ${this.generateMetricCards(results, analysis)}
        </div>
        
        <h2>📈 Performance Trends</h2>
        <div class="chart-container">
            <canvas id="trendsChart"></canvas>
        </div>
        
        <h2>📊 Detailed Comparison</h2>
        ${this.generateComparisonTable(results)}
        
        ${analysis.regressions.length > 0 ? `
        <h2>⚠️ Performance Regressions</h2>
        ${this.generateRegressionsTable(analysis.regressions)}
        ` : ''}
        
        ${analysis.improvements.length > 0 ? `
        <h2>✅ Performance Improvements</h2>
        ${this.generateImprovementsTable(analysis.improvements)}
        ` : ''}
        
        ${recommendations.length > 0 ? `
        <h2>💡 Recommendations</h2>
        ${this.generateRecommendationsHTML(recommendations)}
        ` : ''}
    </div>
    
    <script>
        // Generate trends chart
        const ctx = document.getElementById('trendsChart').getContext('2d');
        const chartData = ${JSON.stringify(this.prepareChartData(results))};
        
        new Chart(ctx, {
            type: 'line',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Test Run'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Response Time (ms)'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Performance Trends Over Time'
                    }
                }
            }
        });
    </script>
</body>
</html>`;
        
        return html;
    }

    generateMetricCards(results, analysis) {
        const latest = results[results.length - 1];
        const baseline = results[0];
        
        return [
            this.createMetricCard('Response Time (P95)', 
                `${latest.responseTime.p95.toFixed(0)}ms`,
                this.calculatePercentageChange(baseline.responseTime.p95, latest.responseTime.p95),
                analysis.trends.responseTime.direction),
            
            this.createMetricCard('Throughput', 
                `${latest.throughput.requestsPerSecond.toFixed(0)} RPS`,
                this.calculatePercentageChange(baseline.throughput.requestsPerSecond, latest.throughput.requestsPerSecond),
                analysis.trends.throughput.direction),
            
            this.createMetricCard('Error Rate', 
                `${(latest.errorRate.rate * 100).toFixed(3)}%`,
                this.calculatePercentageChange(baseline.errorRate.rate, latest.errorRate.rate),
                analysis.trends.errorRate.direction)
        ].join('');
    }

    createMetricCard(title, value, change, direction) {
        const cardClass = direction === 'improving' ? 'improvement' : direction === 'worsening' ? 'regression' : '';
        const changeStr = change > 0 ? `+${change.toFixed(1)}%` : `${change.toFixed(1)}%`;
        const trendClass = direction === 'improving' ? 'trend-down' : direction === 'worsening' ? 'trend-up' : 'trend-stable';
        
        return `
        <div class="metric-card ${cardClass}">
            <h3>${title}</h3>
            <p style="font-size: 2em; margin: 10px 0; font-weight: bold;">${value}</p>
            <p class="${trendClass}">${changeStr} vs baseline (${direction})</p>
        </div>`;
    }

    generateComparisonTable(results) {
        let table = `
        <table>
            <thead>
                <tr>
                    <th>Test Run</th>
                    <th>Response Time (P95)</th>
                    <th>Throughput (RPS)</th>
                    <th>Error Rate</th>
                    <th>Virtual Users</th>
                    <th>Duration</th>
                </tr>
            </thead>
            <tbody>`;
        
        results.forEach((result, index) => {
            table += `
                <tr>
                    <td>${result.name}</td>
                    <td>${result.responseTime.p95.toFixed(0)}ms</td>
                    <td>${result.throughput.requestsPerSecond.toFixed(0)}</td>
                    <td>${(result.errorRate.rate * 100).toFixed(3)}%</td>
                    <td>${result.virtualUsers.max}</td>
                    <td>${Math.round(result.testDuration)}s</td>
                </tr>`;
        });
        
        table += `
            </tbody>
        </table>`;
        
        return table;
    }

    generateRegressionsTable(regressions) {
        let table = `
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Baseline</th>
                    <th>Current</th>
                    <th>Change</th>
                    <th>Severity</th>
                    <th>Impact</th>
                </tr>
            </thead>
            <tbody>`;
        
        regressions.forEach(regression => {
            table += `
                <tr>
                    <td>${regression.metric}</td>
                    <td>${regression.baseline}</td>
                    <td>${regression.current}</td>
                    <td class="trend-up">${regression.change}</td>
                    <td><span style="color: ${regression.severity === 'critical' ? '#e53e3e' : '#d69e2e'}">${regression.severity}</span></td>
                    <td>${regression.impact}</td>
                </tr>`;
        });
        
        table += `
            </tbody>
        </table>`;
        
        return table;
    }

    generateImprovementsTable(improvements) {
        let table = `
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Baseline</th>
                    <th>Current</th>
                    <th>Change</th>
                    <th>Impact</th>
                </tr>
            </thead>
            <tbody>`;
        
        improvements.forEach(improvement => {
            table += `
                <tr>
                    <td>${improvement.metric}</td>
                    <td>${improvement.baseline}</td>
                    <td>${improvement.current}</td>
                    <td class="trend-down">${improvement.change}</td>
                    <td>${improvement.impact}</td>
                </tr>`;
        });
        
        table += `
            </tbody>
        </table>`;
        
        return table;
    }

    generateRecommendationsHTML(recommendations) {
        let html = '<div class="recommendations">';
        
        recommendations.forEach((rec, index) => {
            html += `
            <div class="metric-card">
                <h4>${index + 1}. ${rec.category}</h4>
                <p><strong>Issue:</strong> ${rec.issue}</p>
                <p><strong>Priority:</strong> ${rec.priority}</p>
                <ul>`;
            
            rec.actions.forEach(action => {
                html += `<li>${action}</li>`;
            });
            
            html += `
                </ul>
            </div>`;
        });
        
        html += '</div>';
        return html;
    }

    prepareChartData(results) {
        return {
            labels: results.map((r, i) => `Run ${i + 1}`),
            datasets: [
                {
                    label: 'Response Time (P95)',
                    data: results.map(r => r.responseTime.p95),
                    borderColor: '#e53e3e',
                    backgroundColor: 'rgba(229, 62, 62, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'Throughput (RPS)',
                    data: results.map(r => r.throughput.requestsPerSecond),
                    borderColor: '#3182ce',
                    backgroundColor: 'rgba(49, 130, 206, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y1'
                }
            ]
        };
    }

    generateMarkdownReport(comparison) {
        const { results, analysis, recommendations } = comparison;
        
        let markdown = `# NovaCron Benchmark Comparison Report\n\n`;
        markdown += `**Generated:** ${new Date().toLocaleString()}\n`;
        markdown += `**Results Compared:** ${results.length}\n`;
        markdown += `**Time Range:** ${comparison.metadata.timeRange.start} to ${comparison.metadata.timeRange.end}\n\n`;
        
        // Summary
        const latest = results[results.length - 1];
        const baseline = results[0];
        
        markdown += `## Executive Summary\n\n`;
        markdown += `| Metric | Baseline | Current | Change | Trend |\n`;
        markdown += `|--------|----------|---------|-----------|-------|\n`;
        markdown += `| Response Time (P95) | ${baseline.responseTime.p95.toFixed(0)}ms | ${latest.responseTime.p95.toFixed(0)}ms | ${this.calculatePercentageChange(baseline.responseTime.p95, latest.responseTime.p95).toFixed(1)}% | ${analysis.trends.responseTime.direction} |\n`;
        markdown += `| Throughput | ${baseline.throughput.requestsPerSecond.toFixed(0)} RPS | ${latest.throughput.requestsPerSecond.toFixed(0)} RPS | ${this.calculatePercentageChange(baseline.throughput.requestsPerSecond, latest.throughput.requestsPerSecond).toFixed(1)}% | ${analysis.trends.throughput.direction} |\n`;
        markdown += `| Error Rate | ${(baseline.errorRate.rate * 100).toFixed(3)}% | ${(latest.errorRate.rate * 100).toFixed(3)}% | ${this.calculatePercentageChange(baseline.errorRate.rate, latest.errorRate.rate).toFixed(1)}% | ${analysis.trends.errorRate.direction} |\n\n`;
        
        // Detailed Results
        markdown += `## Detailed Results\n\n`;
        markdown += `| Run | Response Time (P95) | Throughput (RPS) | Error Rate | VUs | Duration |\n`;
        markdown += `|-----|-------------------|----------------|------------|-----|----------|\n`;
        
        results.forEach((result, index) => {
            markdown += `| ${index + 1} | ${result.responseTime.p95.toFixed(0)}ms | ${result.throughput.requestsPerSecond.toFixed(0)} | ${(result.errorRate.rate * 100).toFixed(3)}% | ${result.virtualUsers.max} | ${Math.round(result.testDuration)}s |\n`;
        });
        markdown += `\n`;
        
        // Regressions
        if (analysis.regressions.length > 0) {
            markdown += `## ⚠️ Performance Regressions\n\n`;
            analysis.regressions.forEach((regression, index) => {
                markdown += `### ${index + 1}. ${regression.metric}\n`;
                markdown += `- **Baseline:** ${regression.baseline}\n`;
                markdown += `- **Current:** ${regression.current}\n`;
                markdown += `- **Change:** ${regression.change}\n`;
                markdown += `- **Severity:** ${regression.severity}\n`;
                markdown += `- **Impact:** ${regression.impact}\n\n`;
            });
        }
        
        // Improvements
        if (analysis.improvements.length > 0) {
            markdown += `## ✅ Performance Improvements\n\n`;
            analysis.improvements.forEach((improvement, index) => {
                markdown += `### ${index + 1}. ${improvement.metric}\n`;
                markdown += `- **Baseline:** ${improvement.baseline}\n`;
                markdown += `- **Current:** ${improvement.current}\n`;
                markdown += `- **Change:** ${improvement.change}\n`;
                markdown += `- **Impact:** ${improvement.impact}\n\n`;
            });
        }
        
        // Recommendations
        if (recommendations.length > 0) {
            markdown += `## 💡 Recommendations\n\n`;
            recommendations.forEach((rec, index) => {
                markdown += `### ${index + 1}. ${rec.category}\n`;
                markdown += `**Issue:** ${rec.issue}\n`;
                markdown += `**Priority:** ${rec.priority}\n\n`;
                markdown += `**Actions:**\n`;
                rec.actions.forEach(action => {
                    markdown += `- ${action}\n`;
                });
                markdown += `\n`;
            });
        }
        
        return markdown;
    }
}

// CLI Interface
async function main() {
    const args = process.argv.slice(2);
    const command = args[0] || 'compare';
    
    try {
        switch (command) {
            case 'compare':
                if (args.length < 3) {
                    console.error('Usage: benchmark-comparator.js compare <file1> <file2> [file3...]');
                    console.log('\nExample:');
                    console.log('  benchmark-comparator.js compare reports/baseline.json reports/after-optimization.json');
                    process.exit(1);
                }
                
                const resultFiles = args.slice(1);
                const comparator = new BenchmarkComparator();
                
                const comparison = await comparator.compareResults(resultFiles);
                const outputDir = path.join(__dirname, '../reports');
                
                comparator.generateComparisonReport(comparison.results, outputDir);
                
                // Print summary to console
                console.log('\n📊 Comparison Summary:');
                console.log(`Results: ${comparison.results.length}`);
                console.log(`Regressions: ${comparison.analysis.regressions.length}`);
                console.log(`Improvements: ${comparison.analysis.improvements.length}`);
                console.log(`Recommendations: ${comparison.recommendations.length}`);
                
                break;
                
            case 'trend':
                const trendFiles = args.slice(1);
                if (trendFiles.length < 3) {
                    console.error('Usage: benchmark-comparator.js trend <file1> <file2> <file3> [file4...]');
                    console.log('Need at least 3 files for trend analysis');
                    process.exit(1);
                }
                
                const trendComparator = new BenchmarkComparator();
                const trendComparison = await trendComparator.compareResults(trendFiles);
                
                console.log('\n📈 Trend Analysis:');
                console.log('Response Time:', trendComparison.analysis.trends.responseTime);
                console.log('Throughput:', trendComparison.analysis.trends.throughput);
                console.log('Error Rate:', trendComparison.analysis.trends.errorRate);
                
                break;
                
            default:
                console.log(`
NovaCron Benchmark Comparator

Usage: node benchmark-comparator.js <command> [options]

Commands:
  compare <file1> <file2> [file3...]  Compare benchmark results
  trend <file1> <file2> <file3...>    Analyze performance trends (min 3 files)

Examples:
  benchmark-comparator.js compare baseline.json current.json
  benchmark-comparator.js trend week1.json week2.json week3.json week4.json

Features:
  - Performance regression detection
  - Improvement identification
  - Trend analysis over time
  - Stability assessment
  - HTML/Markdown report generation
  - Automated recommendations
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

module.exports = BenchmarkComparator;