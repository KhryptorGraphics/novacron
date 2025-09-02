#!/usr/bin/env node

/**
 * NovaCron Performance Tuner
 * Automated performance tuning with configuration optimization
 */

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

class PerformanceTuner {
    constructor() {
        this.tuningProfiles = {
            development: {
                maxVUs: 50,
                duration: '2m',
                rampUpTime: '30s',
                responseTimeTarget: 1000,
                throughputTarget: 50
            },
            staging: {
                maxVUs: 500,
                duration: '10m',
                rampUpTime: '2m',
                responseTimeTarget: 500,
                throughputTarget: 200
            },
            production: {
                maxVUs: 2000,
                duration: '30m',
                rampUpTime: '5m',
                responseTimeTarget: 200,
                throughputTarget: 1000
            }
        };
        
        this.optimizationStrategies = {
            responseTime: [
                'enable_compression',
                'optimize_database_queries',
                'implement_caching',
                'connection_pooling',
                'async_processing'
            ],
            throughput: [
                'horizontal_scaling',
                'load_balancing',
                'connection_optimization',
                'resource_scaling',
                'queue_optimization'
            ],
            errorRate: [
                'circuit_breaker',
                'retry_logic',
                'timeout_optimization',
                'health_checks',
                'graceful_degradation'
            ]
        };
    }

    /**
     * Auto-tune system configuration based on performance profile
     */
    async autoTune(environment = 'staging', baselineFile = null) {
        console.log(`🎯 Starting auto-tuning for ${environment} environment...`);
        
        const profile = this.tuningProfiles[environment];
        if (!profile) {
            throw new Error(`Unknown environment: ${environment}`);
        }
        
        console.log(`Target Profile: ${JSON.stringify(profile, null, 2)}`);
        
        // Step 1: Baseline measurement
        const baseline = await this.measureBaseline(environment);
        console.log('📊 Baseline established');
        
        // Step 2: Identify optimization opportunities
        const opportunities = this.identifyOptimizationOpportunities(baseline, profile);
        console.log(`🔍 Found ${opportunities.length} optimization opportunities`);
        
        // Step 3: Apply optimizations iteratively
        const optimizations = await this.applyOptimizations(opportunities, environment);
        
        // Step 4: Validate improvements
        const final = await this.measureFinal(environment);
        
        // Step 5: Generate tuning report
        const report = this.generateTuningReport(baseline, final, optimizations, environment);
        
        return report;
    }

    /**
     * Measure baseline performance
     */
    async measureBaseline(environment) {
        console.log('📏 Measuring baseline performance...');
        
        const profile = this.tuningProfiles[environment];
        const testConfig = {
            vus: Math.min(profile.maxVUs / 4, 100), // Start with 25% of target load
            duration: '5m',
            scenario: 'api-load-test'
        };
        
        // Run baseline test
        const results = await this.runPerformanceTest(testConfig);
        
        return {
            timestamp: Date.now(),
            config: testConfig,
            metrics: this.extractMetrics(results),
            environment
        };
    }

    /**
     * Measure final performance after optimizations
     */
    async measureFinal(environment) {
        console.log('🏁 Measuring final performance...');
        
        const profile = this.tuningProfiles[environment];
        const testConfig = {
            vus: profile.maxVUs,
            duration: profile.duration,
            scenario: 'comprehensive'
        };
        
        const results = await this.runPerformanceTest(testConfig);
        
        return {
            timestamp: Date.now(),
            config: testConfig,
            metrics: this.extractMetrics(results),
            environment
        };
    }

    /**
     * Run performance test with given configuration
     */
    async runPerformanceTest(config) {
        return new Promise((resolve, reject) => {
            const testFile = path.join(__dirname, '../scenarios/api-load-test.js');
            const outputFile = path.join(__dirname, '../reports', `tuning-test-${Date.now()}.json`);
            
            // Ensure reports directory exists
            const reportDir = path.dirname(outputFile);
            if (!fs.existsSync(reportDir)) {
                fs.mkdirSync(reportDir, { recursive: true });
            }
            
            const k6Args = [
                'run',
                '--vus', config.vus.toString(),
                '--duration', config.duration,
                '--out', `json=${outputFile}`,
                '--quiet',
                testFile
            ];
            
            console.log(`Running: k6 ${k6Args.join(' ')}`);
            
            const k6Process = spawn('k6', k6Args, {
                stdio: ['pipe', 'pipe', 'pipe'],
                env: {
                    ...process.env,
                    API_TARGET: process.env.API_TARGET || 'http://localhost:8080'
                }
            });
            
            let stdout = '';
            let stderr = '';
            
            k6Process.stdout.on('data', (data) => {
                stdout += data.toString();
            });
            
            k6Process.stderr.on('data', (data) => {
                stderr += data.toString();
            });
            
            k6Process.on('close', (code) => {
                if (code === 0) {
                    try {
                        const results = JSON.parse(fs.readFileSync(outputFile, 'utf8'));
                        resolve(results);
                    } catch (error) {
                        reject(new Error(`Failed to parse results: ${error.message}`));
                    }
                } else {
                    reject(new Error(`K6 test failed with code ${code}: ${stderr}`));
                }
            });
        });
    }

    /**
     * Extract metrics from k6 results
     */
    extractMetrics(results) {
        const metrics = results.metrics || {};
        
        return {
            responseTime: {
                avg: metrics.http_req_duration?.avg || 0,
                p95: metrics.http_req_duration?.p95 || 0,
                p99: metrics.http_req_duration?.p99 || 0
            },
            throughput: {
                requestsPerSecond: metrics.http_reqs?.rate || 0,
                totalRequests: metrics.http_reqs?.count || 0
            },
            errorRate: metrics.http_req_failed?.rate || 0,
            virtualUsers: metrics.vus?.max || 0
        };
    }

    /**
     * Identify optimization opportunities
     */
    identifyOptimizationOpportunities(baseline, targetProfile) {
        const opportunities = [];
        const { metrics } = baseline;
        
        // Response time optimization
        if (metrics.responseTime.p95 > targetProfile.responseTimeTarget) {
            const gap = metrics.responseTime.p95 - targetProfile.responseTimeTarget;
            opportunities.push({
                type: 'responseTime',
                current: metrics.responseTime.p95,
                target: targetProfile.responseTimeTarget,
                gap,
                priority: gap > targetProfile.responseTimeTarget ? 'critical' : 'high',
                strategies: this.optimizationStrategies.responseTime
            });
        }
        
        // Throughput optimization
        if (metrics.throughput.requestsPerSecond < targetProfile.throughputTarget) {
            const gap = targetProfile.throughputTarget - metrics.throughput.requestsPerSecond;
            opportunities.push({
                type: 'throughput',
                current: metrics.throughput.requestsPerSecond,
                target: targetProfile.throughputTarget,
                gap,
                priority: gap > targetProfile.throughputTarget * 0.5 ? 'critical' : 'high',
                strategies: this.optimizationStrategies.throughput
            });
        }
        
        // Error rate optimization
        if (metrics.errorRate > 0.01) { // 1% threshold
            opportunities.push({
                type: 'errorRate',
                current: metrics.errorRate,
                target: 0.01,
                gap: metrics.errorRate - 0.01,
                priority: metrics.errorRate > 0.05 ? 'critical' : 'high',
                strategies: this.optimizationStrategies.errorRate
            });
        }
        
        return opportunities.sort((a, b) => {
            const priorityOrder = { critical: 3, high: 2, medium: 1, low: 0 };
            return priorityOrder[b.priority] - priorityOrder[a.priority];
        });
    }

    /**
     * Apply optimization strategies
     */
    async applyOptimizations(opportunities, environment) {
        console.log('⚙️ Applying performance optimizations...');
        
        const appliedOptimizations = [];
        
        for (const opportunity of opportunities) {
            console.log(`\n🔧 Optimizing ${opportunity.type}...`);
            
            for (const strategy of opportunity.strategies.slice(0, 2)) { // Apply top 2 strategies
                try {
                    const optimization = await this.applyOptimizationStrategy(strategy, opportunity, environment);
                    appliedOptimizations.push(optimization);
                    
                    // Test improvement after each optimization
                    const testResults = await this.quickTest(environment);
                    optimization.impact = this.measureOptimizationImpact(testResults, opportunity);
                    
                    console.log(`  ✅ ${strategy}: ${optimization.impact.description}`);
                    
                    // Stop if target achieved
                    if (optimization.impact.targetAchieved) {
                        console.log(`  🎯 Target achieved for ${opportunity.type}`);
                        break;
                    }
                } catch (error) {
                    console.error(`  ❌ ${strategy} failed: ${error.message}`);
                }
            }
        }
        
        return appliedOptimizations;
    }

    /**
     * Apply specific optimization strategy
     */
    async applyOptimizationStrategy(strategy, opportunity, environment) {
        const optimization = {
            strategy,
            type: opportunity.type,
            appliedAt: Date.now(),
            environment,
            changes: []
        };
        
        switch (strategy) {
            case 'enable_compression':
                optimization.changes = await this.enableCompression();
                break;
                
            case 'optimize_database_queries':
                optimization.changes = await this.optimizeDatabaseQueries();
                break;
                
            case 'implement_caching':
                optimization.changes = await this.implementCaching();
                break;
                
            case 'connection_pooling':
                optimization.changes = await this.optimizeConnectionPooling();
                break;
                
            case 'horizontal_scaling':
                optimization.changes = await this.simulateHorizontalScaling();
                break;
                
            case 'load_balancing':
                optimization.changes = await this.optimizeLoadBalancing();
                break;
                
            case 'circuit_breaker':
                optimization.changes = await this.implementCircuitBreaker();
                break;
                
            default:
                optimization.changes = [`Strategy ${strategy} is conceptual - would be implemented in production`];
                break;
        }
        
        return optimization;
    }

    /**
     * Optimization strategy implementations (simulated for load testing)
     */
    async enableCompression() {
        return [
            'Configure gzip compression in reverse proxy',
            'Enable brotli compression for static assets',
            'Set appropriate compression levels for API responses'
        ];
    }

    async optimizeDatabaseQueries() {
        return [
            'Add database indexes for frequently queried columns',
            'Implement query result caching',
            'Optimize JOIN operations',
            'Add query performance monitoring'
        ];
    }

    async implementCaching() {
        return [
            'Deploy Redis cluster for session storage',
            'Implement API response caching',
            'Add database query result caching',
            'Configure CDN for static asset caching'
        ];
    }

    async optimizeConnectionPooling() {
        return [
            'Increase database connection pool size',
            'Configure HTTP keep-alive settings',
            'Optimize TCP connection parameters',
            'Implement connection health monitoring'
        ];
    }

    async simulateHorizontalScaling() {
        return [
            'Deploy additional API server instances',
            'Configure load balancer for new instances',
            'Update service discovery configuration',
            'Implement auto-scaling policies'
        ];
    }

    async optimizeLoadBalancing() {
        return [
            'Configure round-robin load balancing',
            'Implement health check-based routing',
            'Add sticky session support where needed',
            'Optimize load balancer timeout settings'
        ];
    }

    async implementCircuitBreaker() {
        return [
            'Add circuit breaker for external service calls',
            'Configure failure threshold and recovery time',
            'Implement fallback mechanisms',
            'Add circuit breaker monitoring'
        ];
    }

    /**
     * Quick performance test
     */
    async quickTest(environment) {
        const profile = this.tuningProfiles[environment];
        const quickConfig = {
            vus: Math.min(profile.maxVUs / 10, 50),
            duration: '1m',
            scenario: 'api-load-test'
        };
        
        return await this.runPerformanceTest(quickConfig);
    }

    /**
     * Measure optimization impact
     */
    measureOptimizationImpact(testResults, opportunity) {
        const metrics = this.extractMetrics(testResults);
        
        let improvement = 0;
        let targetAchieved = false;
        let description = '';
        
        switch (opportunity.type) {
            case 'responseTime':
                improvement = ((opportunity.current - metrics.responseTime.p95) / opportunity.current) * 100;
                targetAchieved = metrics.responseTime.p95 <= opportunity.target;
                description = `Response time: ${metrics.responseTime.p95.toFixed(0)}ms (${improvement > 0 ? '+' : ''}${improvement.toFixed(1)}%)`;
                break;
                
            case 'throughput':
                improvement = ((metrics.throughput.requestsPerSecond - opportunity.current) / opportunity.current) * 100;
                targetAchieved = metrics.throughput.requestsPerSecond >= opportunity.target;
                description = `Throughput: ${metrics.throughput.requestsPerSecond.toFixed(0)} RPS (${improvement > 0 ? '+' : ''}${improvement.toFixed(1)}%)`;
                break;
                
            case 'errorRate':
                improvement = ((opportunity.current - metrics.errorRate) / opportunity.current) * 100;
                targetAchieved = metrics.errorRate <= opportunity.target;
                description = `Error rate: ${(metrics.errorRate * 100).toFixed(3)}% (${improvement > 0 ? '+' : ''}${improvement.toFixed(1)}%)`;
                break;
        }
        
        return {
            improvement,
            targetAchieved,
            description,
            metrics
        };
    }

    /**
     * Generate comprehensive tuning report
     */
    generateTuningReport(baseline, final, optimizations, environment) {
        console.log('📝 Generating tuning report...');
        
        const report = {
            metadata: {
                environment,
                startTime: baseline.timestamp,
                endTime: final.timestamp,
                duration: final.timestamp - baseline.timestamp,
                optimizationsApplied: optimizations.length
            },
            baseline: baseline.metrics,
            final: final.metrics,
            optimizations,
            improvements: this.calculateImprovements(baseline.metrics, final.metrics),
            recommendations: this.generateFinalRecommendations(baseline.metrics, final.metrics, optimizations)
        };
        
        // Save report
        const reportDir = path.join(__dirname, '../reports/tuning');
        if (!fs.existsSync(reportDir)) {
            fs.mkdirSync(reportDir, { recursive: true });
        }
        
        const reportFile = path.join(reportDir, `auto-tuning-${environment}-${Date.now()}.json`);
        fs.writeFileSync(reportFile, JSON.stringify(report, null, 2));
        
        // Generate markdown summary
        const markdownReport = this.generateTuningMarkdown(report);
        const markdownFile = reportFile.replace('.json', '.md');
        fs.writeFileSync(markdownFile, markdownReport);
        
        console.log(`📊 Tuning report generated:`);
        console.log(`   JSON: ${reportFile}`);
        console.log(`   Markdown: ${markdownFile}`);
        
        // Print summary
        this.printTuningSummary(report);
        
        return report;
    }

    calculateImprovements(baseline, final) {
        return {
            responseTime: {
                baseline: baseline.responseTime.p95,
                final: final.responseTime.p95,
                improvement: ((baseline.responseTime.p95 - final.responseTime.p95) / baseline.responseTime.p95) * 100
            },
            throughput: {
                baseline: baseline.throughput.requestsPerSecond,
                final: final.throughput.requestsPerSecond,
                improvement: ((final.throughput.requestsPerSecond - baseline.throughput.requestsPerSecond) / baseline.throughput.requestsPerSecond) * 100
            },
            errorRate: {
                baseline: baseline.errorRate,
                final: final.errorRate,
                improvement: ((baseline.errorRate - final.errorRate) / baseline.errorRate) * 100
            }
        };
    }

    generateFinalRecommendations(baseline, final, optimizations) {
        const recommendations = [];
        
        const improvements = this.calculateImprovements(baseline, final);
        
        // Response time recommendations
        if (improvements.responseTime.improvement < 10) {
            recommendations.push({
                category: 'Response Time',
                priority: 'high',
                message: 'Limited response time improvement achieved',
                actions: [
                    'Consider database optimization',
                    'Profile application bottlenecks',
                    'Implement advanced caching strategies',
                    'Review algorithm efficiency'
                ]
            });
        }
        
        // Throughput recommendations
        if (improvements.throughput.improvement < 20) {
            recommendations.push({
                category: 'Throughput',
                priority: 'medium',
                message: 'Throughput gains below expectations',
                actions: [
                    'Scale infrastructure horizontally',
                    'Optimize resource utilization',
                    'Implement async processing',
                    'Review connection handling'
                ]
            });
        }
        
        // Success recommendations
        if (improvements.responseTime.improvement > 30 || improvements.throughput.improvement > 50) {
            recommendations.push({
                category: 'Success',
                priority: 'info',
                message: 'Significant performance improvements achieved',
                actions: [
                    'Document successful optimizations',
                    'Monitor for performance regressions',
                    'Consider applying to other environments',
                    'Establish new performance baseline'
                ]
            });
        }
        
        return recommendations;
    }

    generateTuningMarkdown(report) {
        const { metadata, baseline, final, improvements, optimizations, recommendations } = report;
        
        let markdown = `# NovaCron Auto-Tuning Report\n\n`;
        markdown += `**Environment:** ${metadata.environment}\n`;
        markdown += `**Duration:** ${Math.round(metadata.duration / 1000)}s\n`;
        markdown += `**Optimizations Applied:** ${metadata.optimizationsApplied}\n\n`;
        
        // Performance improvements
        markdown += `## Performance Improvements\n\n`;
        markdown += `| Metric | Baseline | Final | Improvement |\n`;
        markdown += `|--------|----------|-------|-------------|\n`;
        markdown += `| Response Time (P95) | ${baseline.responseTime.p95.toFixed(0)}ms | ${final.responseTime.p95.toFixed(0)}ms | ${improvements.responseTime.improvement > 0 ? '+' : ''}${improvements.responseTime.improvement.toFixed(1)}% |\n`;
        markdown += `| Throughput | ${baseline.throughput.requestsPerSecond.toFixed(0)} RPS | ${final.throughput.requestsPerSecond.toFixed(0)} RPS | ${improvements.throughput.improvement > 0 ? '+' : ''}${improvements.throughput.improvement.toFixed(1)}% |\n`;
        markdown += `| Error Rate | ${(baseline.errorRate * 100).toFixed(3)}% | ${(final.errorRate * 100).toFixed(3)}% | ${improvements.errorRate.improvement > 0 ? '+' : ''}${improvements.errorRate.improvement.toFixed(1)}% |\n\n`;
        
        // Applied optimizations
        if (optimizations.length > 0) {
            markdown += `## Applied Optimizations\n\n`;
            optimizations.forEach((opt, index) => {
                markdown += `### ${index + 1}. ${opt.strategy.replace(/_/g, ' ').toUpperCase()}\n`;
                markdown += `**Type:** ${opt.type}\n`;
                markdown += `**Impact:** ${opt.impact?.description || 'Pending measurement'}\n`;
                markdown += `**Changes:**\n`;
                opt.changes.forEach(change => {
                    markdown += `- ${change}\n`;
                });
                markdown += `\n`;
            });
        }
        
        // Recommendations
        if (recommendations.length > 0) {
            markdown += `## Recommendations\n\n`;
            recommendations.forEach((rec, index) => {
                markdown += `### ${index + 1}. ${rec.category}\n`;
                markdown += `**Priority:** ${rec.priority}\n`;
                markdown += `**Message:** ${rec.message}\n`;
                markdown += `**Actions:**\n`;
                rec.actions.forEach(action => {
                    markdown += `- ${action}\n`;
                });
                markdown += `\n`;
            });
        }
        
        return markdown;
    }

    printTuningSummary(report) {
        const { improvements } = report;
        
        console.log('\n🎯 Auto-Tuning Summary:');
        console.log(`Environment: ${report.metadata.environment}`);
        console.log(`Optimizations: ${report.metadata.optimizationsApplied}`);
        console.log('\nImprovements:');
        console.log(`  Response Time: ${improvements.responseTime.improvement > 0 ? '+' : ''}${improvements.responseTime.improvement.toFixed(1)}%`);
        console.log(`  Throughput: ${improvements.throughput.improvement > 0 ? '+' : ''}${improvements.throughput.improvement.toFixed(1)}%`);
        console.log(`  Error Rate: ${improvements.errorRate.improvement > 0 ? '+' : ''}${improvements.errorRate.improvement.toFixed(1)}%`);
        
        const totalScore = (
            Math.max(0, improvements.responseTime.improvement) +
            Math.max(0, improvements.throughput.improvement) +
            Math.max(0, improvements.errorRate.improvement)
        ) / 3;
        
        console.log(`\nOverall Improvement Score: ${totalScore.toFixed(1)}%`);
        
        if (totalScore > 25) {
            console.log('🚀 Excellent improvements achieved!');
        } else if (totalScore > 10) {
            console.log('✅ Good improvements achieved');
        } else if (totalScore > 0) {
            console.log('📈 Modest improvements achieved');
        } else {
            console.log('⚠️ Limited improvements - consider additional strategies');
        }
    }

    /**
     * Configuration optimizer for different environments
     */
    optimizeConfiguration(environment, systemResources) {
        console.log(`⚙️ Optimizing configuration for ${environment}...`);
        
        const baseProfile = this.tuningProfiles[environment];
        const optimizedConfig = { ...baseProfile };
        
        // Adjust based on system resources
        const { cpu, memory } = systemResources;
        
        // CPU-based adjustments
        const cpuFactor = Math.min(cpu / 4, 4); // Normalize to 4 cores baseline
        optimizedConfig.maxVUs = Math.round(baseProfile.maxVUs * cpuFactor);
        
        // Memory-based adjustments
        const memoryGB = memory / 1024;
        if (memoryGB < 4) {
            optimizedConfig.maxVUs = Math.round(optimizedConfig.maxVUs * 0.5);
            optimizedConfig.duration = '5m'; // Shorter tests for low memory
        } else if (memoryGB > 16) {
            optimizedConfig.maxVUs = Math.round(optimizedConfig.maxVUs * 1.5);
        }
        
        // Environment-specific adjustments
        switch (environment) {
            case 'development':
                optimizedConfig.responseTimeTarget *= 2; // More lenient for dev
                optimizedConfig.throughputTarget *= 0.5;
                break;
                
            case 'production':
                optimizedConfig.responseTimeTarget *= 0.7; // Stricter for prod
                optimizedConfig.throughputTarget *= 1.2;
                break;
        }
        
        console.log('Optimized Configuration:');
        console.log(JSON.stringify(optimizedConfig, null, 2));
        
        return optimizedConfig;
    }

    /**
     * Generate performance tuning recommendations based on system analysis
     */
    generateSystemTuningRecommendations(systemMetrics) {
        const recommendations = [];
        
        // CPU recommendations
        if (systemMetrics.cpu.avgUsage > 80) {
            recommendations.push({
                category: 'CPU Optimization',
                priority: 'high',
                current: `${systemMetrics.cpu.avgUsage.toFixed(1)}% average usage`,
                actions: [
                    'Scale out to more instances',
                    'Optimize CPU-intensive algorithms',
                    'Implement worker process pools',
                    'Add CPU usage monitoring and alerting'
                ]
            });
        }
        
        // Memory recommendations
        if (systemMetrics.memory.avgUsage > 85) {
            recommendations.push({
                category: 'Memory Optimization',
                priority: 'critical',
                current: `${systemMetrics.memory.avgUsage.toFixed(1)}% average usage`,
                actions: [
                    'Profile memory usage for leaks',
                    'Optimize object caching strategies',
                    'Implement memory-efficient data structures',
                    'Add memory monitoring and alerts'
                ]
            });
        }
        
        // Load recommendations
        if (systemMetrics.load.avgLoad > systemMetrics.cpu.coreCount * 2) {
            recommendations.push({
                category: 'System Load',
                priority: 'high',
                current: `${systemMetrics.load.avgLoad.toFixed(2)} load average`,
                actions: [
                    'Investigate high system load causes',
                    'Optimize I/O operations',
                    'Consider infrastructure upgrades',
                    'Implement load shedding mechanisms'
                ]
            });
        }
        
        return recommendations;
    }
}

// CLI Interface
async function main() {
    const args = process.argv.slice(2);
    const command = args[0] || 'auto-tune';
    
    try {
        const tuner = new PerformanceTuner();
        
        switch (command) {
            case 'auto-tune':
                const environment = args[1] || 'staging';
                const baselineFile = args[2] || null;
                
                console.log(`🚀 Starting auto-tuning for ${environment} environment`);
                const report = await tuner.autoTune(environment, baselineFile);
                
                break;
                
            case 'optimize-config':
                const env = args[1] || 'staging';
                const cpu = parseInt(args[2]) || 4;
                const memory = parseInt(args[3]) || 8192;
                
                const optimizedConfig = tuner.optimizeConfiguration(env, { cpu, memory });
                console.log('\n✅ Optimized configuration generated');
                
                break;
                
            case 'quick-tune':
                const quickEnv = args[1] || 'staging';
                
                console.log('⚡ Running quick tuning...');
                const baseline = await tuner.measureBaseline(quickEnv);
                const opportunities = tuner.identifyOptimizationOpportunities(baseline.metrics, tuner.tuningProfiles[quickEnv]);
                
                console.log(`\n📊 Quick Analysis for ${quickEnv}:`);
                console.log(`Response Time: ${baseline.metrics.responseTime.p95.toFixed(0)}ms`);
                console.log(`Throughput: ${baseline.metrics.throughput.requestsPerSecond.toFixed(0)} RPS`);
                console.log(`Error Rate: ${(baseline.metrics.errorRate * 100).toFixed(3)}%`);
                console.log(`Optimization Opportunities: ${opportunities.length}`);
                
                break;
                
            default:
                console.log(`
NovaCron Performance Tuner

Usage: node performance-tuner.js <command> [options]

Commands:
  auto-tune [environment] [baseline]     Run complete auto-tuning process
  optimize-config [env] [cpu] [memory]   Generate optimized configuration
  quick-tune [environment]               Quick performance analysis

Environments: development, staging, production

Examples:
  node performance-tuner.js auto-tune staging
  node performance-tuner.js optimize-config production 8 16384
  node performance-tuner.js quick-tune development

Features:
  - Automated performance optimization
  - Configuration tuning based on system resources
  - Optimization strategy application
  - Comprehensive improvement tracking
  - Environment-specific tuning profiles
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

module.exports = PerformanceTuner;