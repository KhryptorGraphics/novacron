#!/usr/bin/env node

/**
 * NovaCron Continuous Performance Monitoring
 * Real-time performance tracking with alerting and auto-scaling recommendations
 */

const EventEmitter = require('events');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

class ContinuousMonitor extends EventEmitter {
    constructor(config = {}) {
        super();
        
        this.config = {
            interval: config.interval || 10000, // 10 seconds
            alertThresholds: {
                responseTime: config.responseTimeThreshold || 1000,
                errorRate: config.errorRateThreshold || 0.05,
                throughput: config.throughputThreshold || 100
            },
            endpoints: config.endpoints || [
                '/api/cluster/health',
                '/api/vms',
                '/api/auth/validate'
            ],
            ...config
        };
        
        this.isRunning = false;
        this.metrics = [];
        this.alerts = [];
        this.monitorInterval = null;
    }

    /**
     * Start continuous monitoring
     */
    async start(apiTarget = 'http://localhost:8080') {
        if (this.isRunning) {
            console.log('⚠️ Monitoring is already running');
            return;
        }

        console.log(`🚀 Starting continuous monitoring for ${apiTarget}`);
        console.log(`📊 Monitoring interval: ${this.config.interval}ms`);
        console.log(`🎯 Endpoints: ${this.config.endpoints.join(', ')}`);
        
        this.apiTarget = apiTarget;
        this.isRunning = true;
        this.startTime = Date.now();
        
        // Initial system check
        await this.validateSystem();
        
        // Start monitoring loop
        this.monitorInterval = setInterval(async () => {
            try {
                await this.collectMetrics();
                await this.analyzeMetrics();
                this.checkAlerts();
                this.emit('metrics', this.getLatestMetrics());
            } catch (error) {
                console.error('❌ Monitoring error:', error.message);
                this.emit('error', error);
            }
        }, this.config.interval);
        
        // Handle graceful shutdown
        process.on('SIGINT', () => this.stop());
        process.on('SIGTERM', () => this.stop());
        
        this.emit('started');
        console.log('✅ Continuous monitoring started');
    }

    /**
     * Stop monitoring
     */
    stop() {
        if (!this.isRunning) {
            return;
        }

        console.log('\n🛑 Stopping continuous monitoring...');
        
        this.isRunning = false;
        if (this.monitorInterval) {
            clearInterval(this.monitorInterval);
            this.monitorInterval = null;
        }
        
        // Save final report
        this.generateFinalReport();
        
        this.emit('stopped');
        console.log('✅ Monitoring stopped');
    }

    /**
     * Validate system before starting
     */
    async validateSystem() {
        console.log('🔍 Validating system...');
        
        // Check if API target is accessible
        try {
            const response = await this.makeRequest(`${this.apiTarget}/api/cluster/health`);
            if (response.status !== 200) {
                throw new Error(`API health check failed: ${response.status}`);
            }
            console.log('✅ API target is accessible');
        } catch (error) {
            console.error('❌ API validation failed:', error.message);
            throw error;
        }
        
        // Check system resources
        const systemInfo = await this.getSystemInfo();
        console.log(`💻 System: ${systemInfo.cpu} cores, ${systemInfo.memory}MB RAM`);
        
        if (systemInfo.memory < 2048) {
            console.warn('⚠️ Low memory detected, consider reducing monitoring frequency');
        }
    }

    /**
     * Collect performance metrics
     */
    async collectMetrics() {
        const timestamp = Date.now();
        const systemMetrics = await this.getSystemMetrics();
        
        // Test each endpoint
        const endpointMetrics = await Promise.all(
            this.config.endpoints.map(endpoint => this.testEndpoint(endpoint))
        );
        
        const metrics = {
            timestamp,
            elapsedTime: timestamp - this.startTime,
            system: systemMetrics,
            endpoints: endpointMetrics,
            summary: this.calculateSummaryMetrics(endpointMetrics)
        };
        
        this.metrics.push(metrics);
        
        // Keep only last 1000 data points to prevent memory issues
        if (this.metrics.length > 1000) {
            this.metrics = this.metrics.slice(-1000);
        }
        
        return metrics;
    }

    /**
     * Test individual endpoint performance
     */
    async testEndpoint(endpoint) {
        const url = `${this.apiTarget}${endpoint}`;
        const startTime = Date.now();
        
        try {
            const response = await this.makeRequest(url);
            const endTime = Date.now();
            const responseTime = endTime - startTime;
            
            return {
                endpoint,
                responseTime,
                status: response.status,
                success: response.status >= 200 && response.status < 400,
                contentLength: response.contentLength || 0,
                timestamp: startTime
            };
        } catch (error) {
            const endTime = Date.now();
            return {
                endpoint,
                responseTime: endTime - startTime,
                status: 0,
                success: false,
                error: error.message,
                timestamp: startTime
            };
        }
    }

    /**
     * Make HTTP request with timeout
     */
    makeRequest(url, timeout = 10000) {
        return new Promise((resolve, reject) => {
            const startTime = Date.now();
            
            const curl = spawn('curl', [
                '-s',
                '-w', '%{http_code},%{size_download},%{time_total}',
                '-o', '/dev/null',
                '--connect-timeout', '10',
                '--max-time', String(timeout / 1000),
                url
            ]);
            
            let output = '';
            curl.stdout.on('data', (data) => {
                output += data.toString();
            });
            
            curl.on('close', (code) => {
                const endTime = Date.now();
                
                if (code === 0) {
                    const parts = output.trim().split(',');
                    resolve({
                        status: parseInt(parts[0]) || 0,
                        contentLength: parseInt(parts[1]) || 0,
                        totalTime: parseFloat(parts[2]) || 0,
                        responseTime: endTime - startTime
                    });
                } else {
                    reject(new Error(`Request failed with code ${code}`));
                }
            });
            
            // Handle timeout
            setTimeout(() => {
                curl.kill('SIGTERM');
                reject(new Error('Request timeout'));
            }, timeout);
        });
    }

    /**
     * Get system performance metrics
     */
    async getSystemMetrics() {
        try {
            const { spawn } = require('child_process');
            
            return new Promise((resolve) => {
                const script = `
                    echo "$(grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$3+$4+$5)} END {print usage}')"
                    echo "$(free | grep Mem | awk '{print ($3/$2) * 100.0}')"
                    echo "$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')"
                    echo "$(df / | tail -1 | awk '{print $5}' | sed 's/%//')"
                `;
                
                const bash = spawn('bash', ['-c', script]);
                let output = '';
                
                bash.stdout.on('data', (data) => {
                    output += data.toString();
                });
                
                bash.on('close', () => {
                    const lines = output.trim().split('\n');
                    resolve({
                        cpu: parseFloat(lines[0]) || 0,
                        memory: parseFloat(lines[1]) || 0,
                        loadAvg: parseFloat(lines[2]) || 0,
                        disk: parseFloat(lines[3]) || 0
                    });
                });
            });
        } catch (error) {
            console.warn('Could not collect system metrics:', error.message);
            return { cpu: 0, memory: 0, loadAvg: 0, disk: 0 };
        }
    }

    async getSystemInfo() {
        try {
            const { spawn } = require('child_process');
            
            return new Promise((resolve) => {
                const script = `
                    echo "$(nproc)"
                    echo "$(free -m | grep Mem | awk '{print $2}')"
                `;
                
                const bash = spawn('bash', ['-c', script]);
                let output = '';
                
                bash.stdout.on('data', (data) => {
                    output += data.toString();
                });
                
                bash.on('close', () => {
                    const lines = output.trim().split('\n');
                    resolve({
                        cpu: parseInt(lines[0]) || 1,
                        memory: parseInt(lines[1]) || 1024
                    });
                });
            });
        } catch (error) {
            return { cpu: 1, memory: 1024 };
        }
    }

    /**
     * Calculate summary metrics from endpoint tests
     */
    calculateSummaryMetrics(endpointMetrics) {
        const successfulRequests = endpointMetrics.filter(m => m.success);
        const totalRequests = endpointMetrics.length;
        
        if (successfulRequests.length === 0) {
            return {
                avgResponseTime: 0,
                maxResponseTime: 0,
                minResponseTime: 0,
                successRate: 0,
                errorRate: 1
            };
        }
        
        const responseTimes = successfulRequests.map(m => m.responseTime);
        
        return {
            avgResponseTime: responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length,
            maxResponseTime: Math.max(...responseTimes),
            minResponseTime: Math.min(...responseTimes),
            successRate: successfulRequests.length / totalRequests,
            errorRate: 1 - (successfulRequests.length / totalRequests)
        };
    }

    /**
     * Analyze metrics for trends and anomalies
     */
    async analyzeMetrics() {
        if (this.metrics.length < 3) {
            return; // Need at least 3 data points for trend analysis
        }
        
        const recent = this.metrics.slice(-3);
        const current = recent[recent.length - 1];
        const previous = recent[recent.length - 2];
        
        // Detect response time degradation
        const responseTimeTrend = this.calculateTrend(
            recent.map(m => m.summary.avgResponseTime)
        );
        
        if (responseTimeTrend > 20) { // 20% increase
            this.createAlert('response_time_degradation', {
                message: `Response time increased by ${responseTimeTrend.toFixed(1)}%`,
                current: current.summary.avgResponseTime,
                previous: previous.summary.avgResponseTime,
                severity: 'warning'
            });
        }
        
        // Detect error rate spikes
        if (current.summary.errorRate > this.config.alertThresholds.errorRate) {
            this.createAlert('high_error_rate', {
                message: `Error rate exceeded threshold: ${(current.summary.errorRate * 100).toFixed(2)}%`,
                threshold: this.config.alertThresholds.errorRate * 100,
                current: current.summary.errorRate * 100,
                severity: 'critical'
            });
        }
        
        // Detect system resource issues
        if (current.system.cpu > 90) {
            this.createAlert('high_cpu_usage', {
                message: `High CPU usage: ${current.system.cpu.toFixed(1)}%`,
                current: current.system.cpu,
                severity: 'warning'
            });
        }
        
        if (current.system.memory > 95) {
            this.createAlert('high_memory_usage', {
                message: `High memory usage: ${current.system.memory.toFixed(1)}%`,
                current: current.system.memory,
                severity: 'critical'
            });
        }
    }

    /**
     * Calculate trend percentage between values
     */
    calculateTrend(values) {
        if (values.length < 2) return 0;
        
        const first = values[0];
        const last = values[values.length - 1];
        
        if (first === 0) return 0;
        
        return ((last - first) / first) * 100;
    }

    /**
     * Create and manage alerts
     */
    createAlert(type, data) {
        const alert = {
            id: `${type}_${Date.now()}`,
            type,
            timestamp: Date.now(),
            ...data
        };
        
        this.alerts.push(alert);
        console.log(`🚨 ALERT [${data.severity.toUpperCase()}]: ${data.message}`);
        
        this.emit('alert', alert);
        
        // Keep only last 100 alerts
        if (this.alerts.length > 100) {
            this.alerts = this.alerts.slice(-100);
        }
    }

    /**
     * Check for threshold breaches and create alerts
     */
    checkAlerts() {
        if (this.metrics.length === 0) return;
        
        const latest = this.metrics[this.metrics.length - 1];
        const { summary } = latest;
        
        // Response time alerts
        if (summary.avgResponseTime > this.config.alertThresholds.responseTime) {
            this.createAlert('response_time_threshold', {
                message: `Response time exceeded threshold: ${summary.avgResponseTime.toFixed(0)}ms`,
                threshold: this.config.alertThresholds.responseTime,
                current: summary.avgResponseTime,
                severity: 'warning'
            });
        }
        
        // Throughput alerts (inverted - low throughput is bad)
        const estimatedThroughput = 1000 / (summary.avgResponseTime || 1000); // Rough estimate
        if (estimatedThroughput < this.config.alertThresholds.throughput) {
            this.createAlert('low_throughput', {
                message: `Low throughput detected: ~${estimatedThroughput.toFixed(0)} RPS`,
                threshold: this.config.alertThresholds.throughput,
                current: estimatedThroughput,
                severity: 'warning'
            });
        }
    }

    /**
     * Get latest metrics
     */
    getLatestMetrics() {
        return this.metrics.length > 0 ? this.metrics[this.metrics.length - 1] : null;
    }

    /**
     * Get performance summary
     */
    getPerformanceSummary() {
        if (this.metrics.length === 0) {
            return null;
        }
        
        const responseTimes = this.metrics.map(m => m.summary.avgResponseTime);
        const errorRates = this.metrics.map(m => m.summary.errorRate);
        const successRates = this.metrics.map(m => m.summary.successRate);
        
        return {
            duration: Date.now() - this.startTime,
            dataPoints: this.metrics.length,
            responseTime: {
                avg: responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length,
                min: Math.min(...responseTimes),
                max: Math.max(...responseTimes)
            },
            errorRate: {
                avg: errorRates.reduce((a, b) => a + b, 0) / errorRates.length,
                max: Math.max(...errorRates)
            },
            availability: {
                avg: successRates.reduce((a, b) => a + b, 0) / successRates.length * 100,
                min: Math.min(...successRates) * 100
            },
            alertCount: this.alerts.length
        };
    }

    /**
     * Generate auto-scaling recommendations
     */
    generateScalingRecommendations() {
        const summary = this.getPerformanceSummary();
        if (!summary) return [];
        
        const recommendations = [];
        
        // High response time recommendations
        if (summary.responseTime.avg > 1000) {
            recommendations.push({
                type: 'scale_out',
                reason: 'High average response time',
                action: 'Increase API server instances',
                priority: 'high',
                estimatedImpact: 'Reduce response time by 30-50%'
            });
        }
        
        // Low availability recommendations
        if (summary.availability.avg < 99) {
            recommendations.push({
                type: 'reliability',
                reason: 'Low availability detected',
                action: 'Implement health checks and auto-restart',
                priority: 'critical',
                estimatedImpact: 'Improve availability to 99.9%'
            });
        }
        
        // High error rate recommendations
        if (summary.errorRate.avg > 0.01) {
            recommendations.push({
                type: 'stability',
                reason: 'High error rate',
                action: 'Implement circuit breaker and retry logic',
                priority: 'high',
                estimatedImpact: 'Reduce error rate to <1%'
            });
        }
        
        return recommendations;
    }

    /**
     * Generate final monitoring report
     */
    generateFinalReport() {
        const summary = this.getPerformanceSummary();
        const recommendations = this.generateScalingRecommendations();
        
        const report = {
            metadata: {
                startTime: this.startTime,
                endTime: Date.now(),
                duration: summary?.duration || 0,
                apiTarget: this.apiTarget,
                endpoints: this.config.endpoints
            },
            performance: summary,
            alerts: this.alerts,
            recommendations,
            rawMetrics: this.metrics.slice(-100) // Last 100 data points
        };
        
        // Save report
        const reportDir = path.join(__dirname, '../reports');
        if (!fs.existsSync(reportDir)) {
            fs.mkdirSync(reportDir, { recursive: true });
        }
        
        const reportFile = path.join(reportDir, `continuous-monitoring-${Date.now()}.json`);
        fs.writeFileSync(reportFile, JSON.stringify(report, null, 2));
        
        // Generate markdown summary
        const markdownReport = this.generateMarkdownSummary(report);
        const markdownFile = reportFile.replace('.json', '.md');
        fs.writeFileSync(markdownFile, markdownReport);
        
        console.log(`📄 Final report saved:`);
        console.log(`   JSON: ${reportFile}`);
        console.log(`   Markdown: ${markdownFile}`);
        
        return report;
    }

    generateMarkdownSummary(report) {
        const { metadata, performance, alerts, recommendations } = report;
        
        let markdown = `# Continuous Monitoring Report\n\n`;
        markdown += `**Target:** ${metadata.apiTarget}\n`;
        markdown += `**Duration:** ${Math.round(metadata.duration / 1000)}s\n`;
        markdown += `**Data Points:** ${performance?.dataPoints || 0}\n\n`;
        
        if (performance) {
            markdown += `## Performance Summary\n\n`;
            markdown += `| Metric | Value |\n`;
            markdown += `|--------|-------|\n`;
            markdown += `| Avg Response Time | ${performance.responseTime.avg.toFixed(0)}ms |\n`;
            markdown += `| Max Response Time | ${performance.responseTime.max.toFixed(0)}ms |\n`;
            markdown += `| Availability | ${performance.availability.avg.toFixed(2)}% |\n`;
            markdown += `| Error Rate | ${(performance.errorRate.avg * 100).toFixed(3)}% |\n\n`;
        }
        
        if (alerts.length > 0) {
            markdown += `## Alerts (${alerts.length} total)\n\n`;
            const criticalAlerts = alerts.filter(a => a.severity === 'critical');
            const warningAlerts = alerts.filter(a => a.severity === 'warning');
            
            if (criticalAlerts.length > 0) {
                markdown += `### Critical (${criticalAlerts.length})\n`;
                criticalAlerts.slice(-5).forEach(alert => {
                    markdown += `- ${alert.message}\n`;
                });
                markdown += `\n`;
            }
            
            if (warningAlerts.length > 0) {
                markdown += `### Warnings (${warningAlerts.length})\n`;
                warningAlerts.slice(-3).forEach(alert => {
                    markdown += `- ${alert.message}\n`;
                });
                markdown += `\n`;
            }
        }
        
        if (recommendations.length > 0) {
            markdown += `## Scaling Recommendations\n\n`;
            recommendations.forEach((rec, index) => {
                markdown += `### ${index + 1}. ${rec.action}\n`;
                markdown += `- **Reason:** ${rec.reason}\n`;
                markdown += `- **Priority:** ${rec.priority}\n`;
                markdown += `- **Impact:** ${rec.estimatedImpact}\n\n`;
            });
        }
        
        return markdown;
    }

    /**
     * Export metrics for external analysis
     */
    exportMetrics(format = 'json') {
        const exportDir = path.join(__dirname, '../reports/exports');
        if (!fs.existsSync(exportDir)) {
            fs.mkdirSync(exportDir, { recursive: true });
        }
        
        const timestamp = Date.now();
        
        switch (format.toLowerCase()) {
            case 'csv':
                return this.exportCSV(exportDir, timestamp);
            case 'prometheus':
                return this.exportPrometheus(exportDir, timestamp);
            default:
                return this.exportJSON(exportDir, timestamp);
        }
    }

    exportCSV(exportDir, timestamp) {
        const csvFile = path.join(exportDir, `metrics-${timestamp}.csv`);
        
        let csv = 'timestamp,endpoint,response_time,status,success,cpu,memory,load_avg\n';
        
        this.metrics.forEach(metric => {
            metric.endpoints.forEach(endpoint => {
                csv += [
                    metric.timestamp,
                    endpoint.endpoint,
                    endpoint.responseTime,
                    endpoint.status,
                    endpoint.success,
                    metric.system.cpu,
                    metric.system.memory,
                    metric.system.loadAvg
                ].join(',') + '\n';
            });
        });
        
        fs.writeFileSync(csvFile, csv);
        console.log(`📊 CSV export saved: ${csvFile}`);
        return csvFile;
    }

    exportJSON(exportDir, timestamp) {
        const jsonFile = path.join(exportDir, `metrics-${timestamp}.json`);
        fs.writeFileSync(jsonFile, JSON.stringify(this.metrics, null, 2));
        console.log(`📊 JSON export saved: ${jsonFile}`);
        return jsonFile;
    }
}

// CLI Interface
async function main() {
    const args = process.argv.slice(2);
    const command = args[0] || 'start';
    
    const monitor = new ContinuousMonitor({
        interval: parseInt(args[1]) || 10000,
        responseTimeThreshold: parseInt(args[2]) || 1000
    });
    
    // Setup event listeners
    monitor.on('started', () => {
        console.log('📊 Monitoring started - Press Ctrl+C to stop');
    });
    
    monitor.on('metrics', (metrics) => {
        if (metrics) {
            const summary = metrics.summary;
            console.log(`[${new Date().toLocaleTimeString()}] ` +
                       `RT: ${summary.avgResponseTime.toFixed(0)}ms | ` +
                       `SR: ${(summary.successRate * 100).toFixed(1)}% | ` +
                       `CPU: ${metrics.system.cpu.toFixed(1)}% | ` +
                       `MEM: ${metrics.system.memory.toFixed(1)}%`);
        }
    });
    
    monitor.on('alert', (alert) => {
        console.log(`🚨 ALERT: ${alert.message}`);
    });
    
    monitor.on('stopped', () => {
        const summary = monitor.getPerformanceSummary();
        if (summary) {
            console.log('\n📊 Session Summary:');
            console.log(`   Duration: ${Math.round(summary.duration / 1000)}s`);
            console.log(`   Avg Response Time: ${summary.responseTime.avg.toFixed(0)}ms`);
            console.log(`   Availability: ${summary.availability.avg.toFixed(2)}%`);
            console.log(`   Alerts: ${summary.alertCount}`);
            
            const recommendations = monitor.generateScalingRecommendations();
            if (recommendations.length > 0) {
                console.log('\n💡 Scaling Recommendations:');
                recommendations.forEach((rec, index) => {
                    console.log(`   ${index + 1}. ${rec.action} (${rec.priority})`);
                });
            }
        }
    });
    
    try {
        switch (command) {
            case 'start':
                const apiTarget = args[1] || 'http://localhost:8080';
                await monitor.start(apiTarget);
                break;
                
            case 'analyze':
                const reportFile = args[1];
                if (!reportFile) {
                    console.error('Usage: continuous-monitoring.js analyze <report-file.json>');
                    process.exit(1);
                }
                
                const reportData = JSON.parse(fs.readFileSync(reportFile, 'utf8'));
                monitor.metrics = reportData.rawMetrics || [];
                monitor.alerts = reportData.alerts || [];
                
                const summary = monitor.getPerformanceSummary();
                const recommendations = monitor.generateScalingRecommendations();
                
                console.log('📊 Performance Analysis:');
                console.log(JSON.stringify({ summary, recommendations }, null, 2));
                break;
                
            default:
                console.log(`
NovaCron Continuous Performance Monitor

Usage: node continuous-monitoring.js <command> [options]

Commands:
  start [api-target] [interval] [threshold]  Start monitoring (default: localhost:8080, 10s, 1000ms)
  analyze <report-file>                      Analyze existing monitoring report

Examples:
  node continuous-monitoring.js start
  node continuous-monitoring.js start http://staging.novacron.com 5000 500
  node continuous-monitoring.js analyze reports/continuous-monitoring-*.json

Features:
  - Real-time performance tracking
  - Automatic alerting on threshold breaches
  - Trend analysis and anomaly detection
  - Auto-scaling recommendations
  - Export to multiple formats (JSON, CSV)
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

module.exports = ContinuousMonitor;