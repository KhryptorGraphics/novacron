#!/usr/bin/env node

/**
 * NovaCron Performance Profiler
 * Deep performance profiling with bottleneck identification and flame graph generation
 */

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

class PerformanceProfiler {
    constructor() {
        this.profileData = [];
        this.bottlenecks = [];
        this.config = {
            samplingRate: 100, // samples per second
            profilingDuration: 60000, // 1 minute default
            endpoints: [
                '/api/cluster/health',
                '/api/vms',
                '/api/auth/validate',
                '/api/storage/volumes',
                '/api/monitoring/metrics'
            ]
        };
    }

    /**
     * Start comprehensive performance profiling
     */
    async startProfiling(apiTarget, options = {}) {
        console.log('🔬 Starting performance profiling...');
        
        const config = { ...this.config, ...options };
        const startTime = Date.now();
        
        // Initialize profiling session
        const session = {
            id: `profile_${startTime}`,
            startTime,
            apiTarget,
            config,
            profiles: []
        };

        console.log(`Target: ${apiTarget}`);
        console.log(`Duration: ${config.profilingDuration / 1000}s`);
        console.log(`Sampling Rate: ${config.samplingRate} Hz`);
        
        // Start concurrent profiling
        const profiles = await Promise.all([
            this.profileEndpoints(apiTarget, config),
            this.profileSystemResources(config),
            this.profileNetworkLatency(apiTarget, config),
            this.profileConcurrency(apiTarget, config)
        ]);
        
        session.profiles = profiles;
        session.endTime = Date.now();
        session.duration = session.endTime - session.startTime;
        
        // Analyze results
        const analysis = await this.analyzeProfiles(session);
        session.analysis = analysis;
        
        // Generate profiling report
        this.generateProfilingReport(session);
        
        return session;
    }

    /**
     * Profile API endpoint performance
     */
    async profileEndpoints(apiTarget, config) {
        console.log('📊 Profiling API endpoints...');
        
        const endpointData = [];
        const startTime = Date.now();
        const interval = 1000 / config.samplingRate; // Convert Hz to ms
        
        return new Promise((resolve) => {
            const profileInterval = setInterval(async () => {
                const currentTime = Date.now();
                
                if (currentTime - startTime >= config.profilingDuration) {
                    clearInterval(profileInterval);
                    resolve({
                        type: 'endpoints',
                        data: endpointData,
                        summary: this.summarizeEndpointData(endpointData)
                    });
                    return;
                }
                
                // Test all endpoints concurrently
                const endpointTests = await Promise.all(
                    config.endpoints.map(endpoint => this.measureEndpoint(apiTarget + endpoint))
                );
                
                endpointData.push({
                    timestamp: currentTime,
                    endpoints: endpointTests
                });
            }, interval);
        });
    }

    async measureEndpoint(url) {
        const startTime = process.hrtime.bigint();
        
        try {
            const response = await this.makeTimedRequest(url);
            const endTime = process.hrtime.bigint();
            const responseTime = Number(endTime - startTime) / 1000000; // Convert to ms
            
            return {
                url,
                responseTime,
                status: response.status,
                contentLength: response.contentLength || 0,
                success: response.status >= 200 && response.status < 400,
                dnsTime: response.dnsTime || 0,
                connectTime: response.connectTime || 0,
                sslTime: response.sslTime || 0,
                transferTime: response.transferTime || 0
            };
        } catch (error) {
            const endTime = process.hrtime.bigint();
            const responseTime = Number(endTime - startTime) / 1000000;
            
            return {
                url,
                responseTime,
                status: 0,
                success: false,
                error: error.message
            };
        }
    }

    makeTimedRequest(url) {
        return new Promise((resolve, reject) => {
            const curl = spawn('curl', [
                '-s',
                '-w', JSON.stringify({
                    'http_code': '%{http_code}',
                    'size_download': '%{size_download}',
                    'time_namelookup': '%{time_namelookup}',
                    'time_connect': '%{time_connect}',
                    'time_appconnect': '%{time_appconnect}',
                    'time_pretransfer': '%{time_pretransfer}',
                    'time_starttransfer': '%{time_starttransfer}',
                    'time_total': '%{time_total}'
                }),
                '-o', '/dev/null',
                '--connect-timeout', '10',
                '--max-time', '30',
                url
            ]);
            
            let output = '';
            curl.stdout.on('data', (data) => {
                output += data.toString();
            });
            
            curl.on('close', (code) => {
                if (code === 0) {
                    try {
                        const timing = JSON.parse(output.trim());
                        resolve({
                            status: parseInt(timing.http_code) || 0,
                            contentLength: parseInt(timing.size_download) || 0,
                            dnsTime: parseFloat(timing.time_namelookup) * 1000,
                            connectTime: parseFloat(timing.time_connect) * 1000,
                            sslTime: parseFloat(timing.time_appconnect) * 1000,
                            transferTime: parseFloat(timing.time_starttransfer) * 1000,
                            totalTime: parseFloat(timing.time_total) * 1000
                        });
                    } catch (error) {
                        reject(new Error('Failed to parse curl timing output'));
                    }
                } else {
                    reject(new Error(`Curl failed with code ${code}`));
                }
            });
            
            curl.on('error', reject);
        });
    }

    /**
     * Profile system resource usage
     */
    async profileSystemResources(config) {
        console.log('💻 Profiling system resources...');
        
        const resourceData = [];
        const startTime = Date.now();
        
        return new Promise((resolve) => {
            const resourceInterval = setInterval(async () => {
                const currentTime = Date.now();
                
                if (currentTime - startTime >= config.profilingDuration) {
                    clearInterval(resourceInterval);
                    resolve({
                        type: 'system_resources',
                        data: resourceData,
                        summary: this.summarizeResourceData(resourceData)
                    });
                    return;
                }
                
                const resources = await this.getDetailedSystemMetrics();
                resourceData.push({
                    timestamp: currentTime,
                    ...resources
                });
            }, 1000); // 1 second intervals for system metrics
        });
    }

    async getDetailedSystemMetrics() {
        try {
            const script = `
                # CPU usage per core
                grep '^cpu[0-9]' /proc/stat | awk '{print $1, ($2+$4)*100/($2+$3+$4+$5)}'
                
                # Memory breakdown
                awk '/^MemTotal|^MemFree|^MemAvailable|^Buffers|^Cached|^SwapTotal|^SwapFree/ {print $1, $2}' /proc/meminfo
                
                # Load average
                awk '{print "load_1min", $1; print "load_5min", $2; print "load_15min", $3}' /proc/loadavg
                
                # Network stats
                awk '/^[[:space:]]*eth0|^[[:space:]]*ens|^[[:space:]]*en/ {print $1, $2, $10}' /proc/net/dev | head -1
                
                # Disk I/O
                awk '/^[[:space:]]*[sv]da/ {print $1, $4, $8}' /proc/diskstats | head -1
            `;
            
            return new Promise((resolve) => {
                const bash = spawn('bash', ['-c', script]);
                let output = '';
                
                bash.stdout.on('data', (data) => {
                    output += data.toString();
                });
                
                bash.on('close', () => {
                    const metrics = this.parseSystemMetrics(output);
                    resolve(metrics);
                });
            });
        } catch (error) {
            console.warn('Could not collect detailed system metrics:', error.message);
            return {
                cpu: { overall: 0, cores: [] },
                memory: { total: 0, free: 0, available: 0, used: 0 },
                load: { min1: 0, min5: 0, min15: 0 },
                network: { interface: '', rxBytes: 0, txBytes: 0 },
                disk: { device: '', readOps: 0, writeOps: 0 }
            };
        }
    }

    parseSystemMetrics(output) {
        const lines = output.trim().split('\n');
        const metrics = {
            cpu: { overall: 0, cores: [] },
            memory: { total: 0, free: 0, available: 0, used: 0 },
            load: { min1: 0, min5: 0, min15: 0 },
            network: { interface: '', rxBytes: 0, txBytes: 0 },
            disk: { device: '', readOps: 0, writeOps: 0 }
        };
        
        lines.forEach(line => {
            const parts = line.trim().split(/\s+/);
            if (parts.length < 2) return;
            
            const key = parts[0];
            const value = parts[1];
            
            if (key.startsWith('cpu')) {
                if (key === 'cpu') {
                    metrics.cpu.overall = parseFloat(value) || 0;
                } else {
                    metrics.cpu.cores.push({
                        core: key,
                        usage: parseFloat(value) || 0
                    });
                }
            } else if (key.includes('Mem')) {
                const field = key.replace(':', '').toLowerCase();
                metrics.memory[field] = parseInt(value) || 0;
            } else if (key.includes('load_')) {
                const field = key.replace('load_', '');
                metrics.load[field] = parseFloat(value) || 0;
            }
        });
        
        // Calculate memory usage
        if (metrics.memory.total > 0) {
            metrics.memory.used = metrics.memory.total - metrics.memory.free;
            metrics.memory.usagePercent = (metrics.memory.used / metrics.memory.total) * 100;
        }
        
        return metrics;
    }

    /**
     * Profile network latency and connectivity
     */
    async profileNetworkLatency(apiTarget, config) {
        console.log('🌐 Profiling network latency...');
        
        const latencyData = [];
        const startTime = Date.now();
        
        // Extract hostname from API target
        const hostname = new URL(apiTarget).hostname;
        
        return new Promise((resolve) => {
            const latencyInterval = setInterval(async () => {
                const currentTime = Date.now();
                
                if (currentTime - startTime >= config.profilingDuration) {
                    clearInterval(latencyInterval);
                    resolve({
                        type: 'network_latency',
                        data: latencyData,
                        summary: this.summarizeLatencyData(latencyData)
                    });
                    return;
                }
                
                const latency = await this.measureNetworkLatency(hostname);
                latencyData.push({
                    timestamp: currentTime,
                    ...latency
                });
            }, 2000); // 2 second intervals
        });
    }

    async measureNetworkLatency(hostname) {
        try {
            return new Promise((resolve) => {
                const ping = spawn('ping', ['-c', '1', '-W', '5', hostname]);
                let output = '';
                
                ping.stdout.on('data', (data) => {
                    output += data.toString();
                });
                
                ping.on('close', (code) => {
                    if (code === 0) {
                        const timeMatch = output.match(/time=([0-9.]+) ms/);
                        const time = timeMatch ? parseFloat(timeMatch[1]) : 0;
                        
                        resolve({
                            hostname,
                            latency: time,
                            success: true,
                            packetLoss: 0
                        });
                    } else {
                        resolve({
                            hostname,
                            latency: 0,
                            success: false,
                            packetLoss: 100
                        });
                    }
                });
            });
        } catch (error) {
            return {
                hostname,
                latency: 0,
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Profile concurrency and connection handling
     */
    async profileConcurrency(apiTarget, config) {
        console.log('🔄 Profiling concurrency performance...');
        
        const concurrencyData = [];
        const concurrencyLevels = [1, 5, 10, 25, 50, 100, 200];
        
        for (const concurrency of concurrencyLevels) {
            console.log(`  Testing concurrency level: ${concurrency}`);
            
            const results = await this.testConcurrencyLevel(apiTarget, concurrency);
            concurrencyData.push({
                concurrency,
                ...results
            });
            
            // Brief pause between tests
            await new Promise(resolve => setTimeout(resolve, 2000));
        }
        
        return {
            type: 'concurrency',
            data: concurrencyData,
            summary: this.analyzeConcurrencyData(concurrencyData)
        };
    }

    async testConcurrencyLevel(apiTarget, concurrency) {
        const promises = [];
        const startTime = Date.now();
        
        // Create concurrent requests
        for (let i = 0; i < concurrency; i++) {
            promises.push(this.measureEndpoint(apiTarget + '/api/cluster/health'));
        }
        
        try {
            const results = await Promise.all(promises);
            const endTime = Date.now();
            
            const successfulRequests = results.filter(r => r.success);
            const responseTimes = successfulRequests.map(r => r.responseTime);
            
            return {
                totalTime: endTime - startTime,
                successCount: successfulRequests.length,
                errorCount: results.length - successfulRequests.length,
                successRate: successfulRequests.length / results.length,
                responseTime: {
                    avg: responseTimes.length > 0 ? responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length : 0,
                    min: responseTimes.length > 0 ? Math.min(...responseTimes) : 0,
                    max: responseTimes.length > 0 ? Math.max(...responseTimes) : 0,
                    p95: responseTimes.length > 0 ? this.calculatePercentile(responseTimes, 95) : 0
                },
                throughput: responseTimes.length > 0 ? (successfulRequests.length / ((endTime - startTime) / 1000)) : 0
            };
        } catch (error) {
            return {
                totalTime: Date.now() - startTime,
                successCount: 0,
                errorCount: concurrency,
                successRate: 0,
                error: error.message
            };
        }
    }

    calculatePercentile(values, percentile) {
        if (values.length === 0) return 0;
        
        const sorted = [...values].sort((a, b) => a - b);
        const index = Math.ceil((percentile / 100) * sorted.length) - 1;
        return sorted[Math.max(0, index)];
    }

    /**
     * Analyze profiles and identify bottlenecks
     */
    async analyzeProfiles(session) {
        console.log('🔍 Analyzing performance profiles...');
        
        const analysis = {
            bottlenecks: [],
            recommendations: [],
            performanceProfile: {},
            scalabilityAnalysis: {}
        };
        
        // Analyze endpoint performance
        const endpointProfile = session.profiles.find(p => p.type === 'endpoints');
        if (endpointProfile) {
            analysis.performanceProfile.endpoints = this.analyzeEndpointPerformance(endpointProfile);
            analysis.bottlenecks.push(...this.identifyEndpointBottlenecks(endpointProfile));
        }
        
        // Analyze system resources
        const systemProfile = session.profiles.find(p => p.type === 'system_resources');
        if (systemProfile) {
            analysis.performanceProfile.system = this.analyzeSystemPerformance(systemProfile);
            analysis.bottlenecks.push(...this.identifySystemBottlenecks(systemProfile));
        }
        
        // Analyze concurrency
        const concurrencyProfile = session.profiles.find(p => p.type === 'concurrency');
        if (concurrencyProfile) {
            analysis.scalabilityAnalysis = this.analyzeConcurrencyData(concurrencyProfile.data);
            analysis.bottlenecks.push(...this.identifyScalabilityBottlenecks(concurrencyProfile));
        }
        
        // Generate recommendations
        analysis.recommendations = this.generateProfilingRecommendations(analysis.bottlenecks);
        
        return analysis;
    }

    analyzeEndpointPerformance(endpointProfile) {
        const { data, summary } = endpointProfile;
        
        // Calculate statistics for each endpoint
        const endpointStats = {};
        
        summary.forEach(endpoint => {
            const endpointData = data.flatMap(d => 
                d.endpoints.filter(e => e.url.includes(endpoint.path))
            ).filter(e => e.success);
            
            if (endpointData.length > 0) {
                const responseTimes = endpointData.map(e => e.responseTime);
                
                endpointStats[endpoint.path] = {
                    avgResponseTime: responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length,
                    p95ResponseTime: this.calculatePercentile(responseTimes, 95),
                    minResponseTime: Math.min(...responseTimes),
                    maxResponseTime: Math.max(...responseTimes),
                    samples: responseTimes.length,
                    variability: this.calculateVariability(responseTimes)
                };
            }
        });
        
        return endpointStats;
    }

    identifyEndpointBottlenecks(endpointProfile) {
        const bottlenecks = [];
        const stats = this.analyzeEndpointPerformance(endpointProfile);
        
        Object.entries(stats).forEach(([endpoint, data]) => {
            // High response time bottleneck
            if (data.p95ResponseTime > 1000) {
                bottlenecks.push({
                    type: 'endpoint_performance',
                    severity: data.p95ResponseTime > 2000 ? 'critical' : 'high',
                    endpoint,
                    issue: `High response time: P95 = ${data.p95ResponseTime.toFixed(0)}ms`,
                    evidence: {
                        p95: data.p95ResponseTime,
                        avg: data.avgResponseTime,
                        max: data.maxResponseTime,
                        samples: data.samples
                    },
                    potentialCauses: [
                        'Database query performance',
                        'External API calls',
                        'Inefficient algorithms',
                        'Resource contention'
                    ]
                });
            }
            
            // High variability bottleneck
            if (data.variability.coefficientOfVariation > 50) {
                bottlenecks.push({
                    type: 'endpoint_stability',
                    severity: 'medium',
                    endpoint,
                    issue: `High response time variability: CV = ${data.variability.coefficientOfVariation.toFixed(1)}%`,
                    evidence: {
                        cv: data.variability.coefficientOfVariation,
                        stdDev: data.variability.stdDev,
                        min: data.minResponseTime,
                        max: data.maxResponseTime
                    },
                    potentialCauses: [
                        'Inconsistent caching',
                        'Garbage collection pauses',
                        'Network latency variations',
                        'Resource contention spikes'
                    ]
                });
            }
        });
        
        return bottlenecks;
    }

    identifySystemBottlenecks(systemProfile) {
        const bottlenecks = [];
        const { summary } = systemProfile;
        
        // CPU bottlenecks
        if (summary.cpu.maxUsage > 90) {
            bottlenecks.push({
                type: 'cpu_bottleneck',
                severity: 'critical',
                issue: `High CPU usage: max ${summary.cpu.maxUsage.toFixed(1)}%`,
                evidence: {
                    max: summary.cpu.maxUsage,
                    avg: summary.cpu.avgUsage,
                    cores: summary.cpu.coreCount
                },
                potentialCauses: [
                    'CPU-intensive algorithms',
                    'Insufficient horizontal scaling',
                    'Blocking I/O operations',
                    'Inefficient request processing'
                ]
            });
        }
        
        // Memory bottlenecks
        if (summary.memory.maxUsage > 85) {
            bottlenecks.push({
                type: 'memory_bottleneck',
                severity: summary.memory.maxUsage > 95 ? 'critical' : 'high',
                issue: `High memory usage: max ${summary.memory.maxUsage.toFixed(1)}%`,
                evidence: {
                    max: summary.memory.maxUsage,
                    avg: summary.memory.avgUsage,
                    total: summary.memory.total
                },
                potentialCauses: [
                    'Memory leaks',
                    'Large object caching',
                    'Insufficient garbage collection',
                    'Resource pooling issues'
                ]
            });
        }
        
        return bottlenecks;
    }

    identifyScalabilityBottlenecks(concurrencyProfile) {
        const bottlenecks = [];
        const { summary } = concurrencyProfile;
        
        // Find concurrency breaking point
        const breakingPoint = summary.optimalConcurrency;
        if (breakingPoint < 100) {
            bottlenecks.push({
                type: 'scalability_limit',
                severity: 'high',
                issue: `Low concurrency limit: optimal at ${breakingPoint} users`,
                evidence: {
                    optimalConcurrency: breakingPoint,
                    maxTested: Math.max(...concurrencyProfile.data.map(d => d.concurrency)),
                    performanceDrop: summary.performanceDropoff
                },
                potentialCauses: [
                    'Connection pool limits',
                    'Database connection limits',
                    'File descriptor limits',
                    'Memory constraints'
                ]
            });
        }
        
        return bottlenecks;
    }

    calculateVariability(values) {
        if (values.length === 0) return { stdDev: 0, coefficientOfVariation: 0 };
        
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / values.length;
        const stdDev = Math.sqrt(variance);
        
        return {
            mean,
            stdDev,
            variance,
            coefficientOfVariation: mean !== 0 ? (stdDev / mean) * 100 : 0
        };
    }

    summarizeEndpointData(data) {
        const endpoints = new Set();
        data.forEach(d => d.endpoints.forEach(e => endpoints.add(e.url)));
        
        return Array.from(endpoints).map(url => {
            const endpointData = data.flatMap(d => 
                d.endpoints.filter(e => e.url === url)
            );
            
            const successfulRequests = endpointData.filter(e => e.success);
            const responseTimes = successfulRequests.map(e => e.responseTime);
            
            return {
                path: new URL(url).pathname,
                url,
                totalRequests: endpointData.length,
                successfulRequests: successfulRequests.length,
                successRate: endpointData.length > 0 ? successfulRequests.length / endpointData.length : 0,
                avgResponseTime: responseTimes.length > 0 ? responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length : 0,
                p95ResponseTime: responseTimes.length > 0 ? this.calculatePercentile(responseTimes, 95) : 0
            };
        });
    }

    summarizeResourceData(data) {
        if (data.length === 0) return {};
        
        const cpuUsages = data.map(d => d.cpu.overall).filter(v => v > 0);
        const memoryUsages = data.map(d => d.memory.usagePercent).filter(v => v > 0);
        const loads = data.map(d => d.load.min1).filter(v => v > 0);
        
        return {
            cpu: {
                avgUsage: cpuUsages.length > 0 ? cpuUsages.reduce((a, b) => a + b, 0) / cpuUsages.length : 0,
                maxUsage: cpuUsages.length > 0 ? Math.max(...cpuUsages) : 0,
                coreCount: data[0]?.cpu?.cores?.length || 0
            },
            memory: {
                avgUsage: memoryUsages.length > 0 ? memoryUsages.reduce((a, b) => a + b, 0) / memoryUsages.length : 0,
                maxUsage: memoryUsages.length > 0 ? Math.max(...memoryUsages) : 0,
                total: data[0]?.memory?.total || 0
            },
            load: {
                avgLoad: loads.length > 0 ? loads.reduce((a, b) => a + b, 0) / loads.length : 0,
                maxLoad: loads.length > 0 ? Math.max(...loads) : 0
            }
        };
    }

    summarizeLatencyData(data) {
        const latencies = data.filter(d => d.success).map(d => d.latency);
        
        if (latencies.length === 0) {
            return { avgLatency: 0, packetLoss: 100 };
        }
        
        const packetLossCount = data.filter(d => !d.success).length;
        
        return {
            avgLatency: latencies.reduce((a, b) => a + b, 0) / latencies.length,
            minLatency: Math.min(...latencies),
            maxLatency: Math.max(...latencies),
            packetLoss: (packetLossCount / data.length) * 100,
            samples: data.length
        };
    }

    analyzeConcurrencyData(data) {
        // Find optimal concurrency level
        let optimalConcurrency = 1;
        let bestThroughput = 0;
        
        data.forEach(result => {
            if (result.successRate > 0.95 && result.throughput > bestThroughput) {
                bestThroughput = result.throughput;
                optimalConcurrency = result.concurrency;
            }
        });
        
        // Calculate performance dropoff point
        const maxThroughput = Math.max(...data.map(d => d.throughput || 0));
        const dropoffPoint = data.find(d => d.throughput < maxThroughput * 0.8);
        
        return {
            optimalConcurrency,
            maxTestedConcurrency: Math.max(...data.map(d => d.concurrency)),
            bestThroughput,
            performanceDropoff: dropoffPoint ? dropoffPoint.concurrency : null,
            scalabilityRating: this.assessScalability(data)
        };
    }

    assessScalability(data) {
        const maxConcurrency = Math.max(...data.map(d => d.concurrency));
        const optimalResult = data.find(d => d.concurrency === Math.max(...data.filter(r => r.successRate > 0.95).map(r => r.concurrency)));
        
        if (!optimalResult) return 'poor';
        
        const optimalConcurrency = optimalResult.concurrency;
        
        if (optimalConcurrency >= 200) return 'excellent';
        if (optimalConcurrency >= 100) return 'good';
        if (optimalConcurrency >= 50) return 'fair';
        return 'poor';
    }

    /**
     * Generate profiling recommendations
     */
    generateProfilingRecommendations(bottlenecks) {
        const recommendations = [];
        
        // Group bottlenecks by type
        const groupedBottlenecks = bottlenecks.reduce((groups, bottleneck) => {
            if (!groups[bottleneck.type]) {
                groups[bottleneck.type] = [];
            }
            groups[bottleneck.type].push(bottleneck);
            return groups;
        }, {});
        
        // Generate type-specific recommendations
        Object.entries(groupedBottlenecks).forEach(([type, bottleneckList]) => {
            switch (type) {
                case 'endpoint_performance':
                    recommendations.push({
                        category: 'API Performance',
                        priority: 'high',
                        bottlenecks: bottleneckList.length,
                        actions: [
                            'Profile slow endpoints with application profiler',
                            'Implement response caching for read-heavy endpoints',
                            'Optimize database queries and add indexes',
                            'Consider async processing for heavy operations',
                            'Add connection pooling if not present'
                        ]
                    });
                    break;
                    
                case 'cpu_bottleneck':
                    recommendations.push({
                        category: 'CPU Optimization',
                        priority: 'critical',
                        bottlenecks: bottleneckList.length,
                        actions: [
                            'Scale horizontally to distribute CPU load',
                            'Profile CPU-intensive code paths',
                            'Implement caching to reduce computation',
                            'Optimize algorithms for better complexity',
                            'Consider worker processes for CPU-bound tasks'
                        ]
                    });
                    break;
                    
                case 'memory_bottleneck':
                    recommendations.push({
                        category: 'Memory Optimization',
                        priority: 'critical',
                        bottlenecks: bottleneckList.length,
                        actions: [
                            'Profile memory usage and identify leaks',
                            'Implement efficient caching strategies',
                            'Optimize data structures and object lifecycle',
                            'Add memory monitoring and alerts',
                            'Consider vertical scaling or memory optimization'
                        ]
                    });
                    break;
                    
                case 'scalability_limit':
                    recommendations.push({
                        category: 'Scalability',
                        priority: 'high',
                        bottlenecks: bottleneckList.length,
                        actions: [
                            'Increase connection pool sizes',
                            'Implement load balancing',
                            'Add horizontal scaling capabilities',
                            'Optimize connection handling',
                            'Review system resource limits'
                        ]
                    });
                    break;
            }
        });
        
        return recommendations;
    }

    /**
     * Generate comprehensive profiling report
     */
    generateProfilingReport(session) {
        const reportDir = path.join(__dirname, '../reports/profiling');
        if (!fs.existsSync(reportDir)) {
            fs.mkdirSync(reportDir, { recursive: true });
        }
        
        // Save raw session data
        const sessionFile = path.join(reportDir, `profiling-session-${session.id}.json`);
        fs.writeFileSync(sessionFile, JSON.stringify(session, null, 2));
        
        // Generate HTML report
        const htmlReport = this.generateProfilingHTML(session);
        const htmlFile = sessionFile.replace('.json', '.html');
        fs.writeFileSync(htmlFile, htmlReport);
        
        // Generate markdown summary
        const markdownReport = this.generateProfilingMarkdown(session);
        const markdownFile = sessionFile.replace('.json', '.md');
        fs.writeFileSync(markdownFile, markdownReport);
        
        console.log(`🔬 Profiling report generated:`);
        console.log(`   Session Data: ${sessionFile}`);
        console.log(`   HTML Report: ${htmlFile}`);
        console.log(`   Markdown: ${markdownFile}`);
        
        return {
            sessionFile,
            htmlFile,
            markdownFile,
            session
        };
    }

    generateProfilingHTML(session) {
        const { analysis } = session;
        
        return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NovaCron Performance Profiling Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { border-bottom: 2px solid #eee; padding-bottom: 20px; margin-bottom: 30px; }
        .bottleneck { background: #fff5f5; border-left: 4px solid #e53e3e; padding: 15px; margin: 10px 0; border-radius: 4px; }
        .bottleneck.medium { background: #fffbf0; border-left-color: #d69e2e; }
        .chart-container { position: relative; height: 400px; margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: 600; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 NovaCron Performance Profiling Report</h1>
            <p><strong>Session ID:</strong> ${session.id}</p>
            <p><strong>Target:</strong> ${session.apiTarget}</p>
            <p><strong>Duration:</strong> ${Math.round(session.duration / 1000)}s</p>
            <p><strong>Bottlenecks Found:</strong> ${analysis.bottlenecks.length}</p>
        </div>
        
        <h2>🎯 Performance Bottlenecks</h2>
        ${analysis.bottlenecks.map(bottleneck => `
        <div class="bottleneck ${bottleneck.severity}">
            <h3>${bottleneck.type.replace(/_/g, ' ').toUpperCase()}</h3>
            <p><strong>Issue:</strong> ${bottleneck.issue}</p>
            <p><strong>Severity:</strong> ${bottleneck.severity}</p>
            <p><strong>Potential Causes:</strong></p>
            <ul>
                ${bottleneck.potentialCauses.map(cause => `<li>${cause}</li>`).join('')}
            </ul>
        </div>
        `).join('')}
        
        <h2>💡 Optimization Recommendations</h2>
        ${analysis.recommendations.map((rec, index) => `
        <div class="bottleneck medium">
            <h3>${index + 1}. ${rec.category}</h3>
            <p><strong>Priority:</strong> ${rec.priority}</p>
            <p><strong>Bottlenecks Addressed:</strong> ${rec.bottlenecks}</p>
            <ul>
                ${rec.actions.map(action => `<li>${action}</li>`).join('')}
            </ul>
        </div>
        `).join('')}
    </div>
</body>
</html>`;
    }

    generateProfilingMarkdown(session) {
        const { analysis } = session;
        
        let markdown = `# NovaCron Performance Profiling Report\n\n`;
        markdown += `**Session ID:** ${session.id}\n`;
        markdown += `**Target:** ${session.apiTarget}\n`;
        markdown += `**Duration:** ${Math.round(session.duration / 1000)}s\n`;
        markdown += `**Profiles Generated:** ${session.profiles.length}\n`;
        markdown += `**Bottlenecks Found:** ${analysis.bottlenecks.length}\n\n`;
        
        // Bottlenecks
        if (analysis.bottlenecks.length > 0) {
            markdown += `## 🎯 Performance Bottlenecks\n\n`;
            
            analysis.bottlenecks.forEach((bottleneck, index) => {
                markdown += `### ${index + 1}. ${bottleneck.type.replace(/_/g, ' ').toUpperCase()}\n`;
                markdown += `- **Issue:** ${bottleneck.issue}\n`;
                markdown += `- **Severity:** ${bottleneck.severity}\n`;
                if (bottleneck.endpoint) {
                    markdown += `- **Endpoint:** ${bottleneck.endpoint}\n`;
                }
                markdown += `- **Potential Causes:**\n`;
                bottleneck.potentialCauses.forEach(cause => {
                    markdown += `  - ${cause}\n`;
                });
                markdown += `\n`;
            });
        }
        
        // Recommendations
        if (analysis.recommendations.length > 0) {
            markdown += `## 💡 Optimization Recommendations\n\n`;
            
            analysis.recommendations.forEach((rec, index) => {
                markdown += `### ${index + 1}. ${rec.category}\n`;
                markdown += `- **Priority:** ${rec.priority}\n`;
                markdown += `- **Bottlenecks Addressed:** ${rec.bottlenecks}\n`;
                markdown += `- **Actions:**\n`;
                rec.actions.forEach(action => {
                    markdown += `  - ${action}\n`;
                });
                markdown += `\n`;
            });
        }
        
        // Scalability Analysis
        if (analysis.scalabilityAnalysis.optimalConcurrency) {
            markdown += `## 📈 Scalability Analysis\n\n`;
            markdown += `- **Optimal Concurrency:** ${analysis.scalabilityAnalysis.optimalConcurrency} users\n`;
            markdown += `- **Best Throughput:** ${analysis.scalabilityAnalysis.bestThroughput.toFixed(0)} RPS\n`;
            markdown += `- **Scalability Rating:** ${analysis.scalabilityAnalysis.scalabilityRating}\n\n`;
        }
        
        return markdown;
    }
}

// CLI Interface
async function main() {
    const args = process.argv.slice(2);
    const command = args[0] || 'profile';
    
    try {
        switch (command) {
            case 'profile':
                const apiTarget = args[1] || 'http://localhost:8080';
                const duration = parseInt(args[2]) || 60000;
                
                const profiler = new PerformanceProfiler();
                const session = await profiler.startProfiling(apiTarget, {
                    profilingDuration: duration
                });
                
                console.log('\n📋 Profiling Summary:');
                console.log(`Duration: ${Math.round(session.duration / 1000)}s`);
                console.log(`Bottlenecks: ${session.analysis.bottlenecks.length}`);
                console.log(`Recommendations: ${session.analysis.recommendations.length}`);
                
                break;
                
            case 'bottlenecks':
                const reportFile = args[1];
                if (!reportFile) {
                    console.error('Usage: profiler.js bottlenecks <profiling-session.json>');
                    process.exit(1);
                }
                
                const sessionData = JSON.parse(fs.readFileSync(reportFile, 'utf8'));
                
                console.log('\n🎯 Identified Bottlenecks:');
                sessionData.analysis.bottlenecks.forEach((bottleneck, index) => {
                    console.log(`${index + 1}. ${bottleneck.issue} (${bottleneck.severity})`);
                });
                
                break;
                
            default:
                console.log(`
NovaCron Performance Profiler

Usage: node profiler.js <command> [options]

Commands:
  profile [target] [duration]    Start comprehensive profiling (default: localhost:8080, 60s)
  bottlenecks <session-file>     Analyze bottlenecks from existing session

Examples:
  node profiler.js profile
  node profiler.js profile http://staging.novacron.com 120000
  node profiler.js bottlenecks reports/profiling/profiling-session-*.json

Features:
  - Endpoint performance profiling
  - System resource monitoring
  - Network latency analysis
  - Concurrency testing
  - Bottleneck identification
  - Optimization recommendations
  - HTML/Markdown reports
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

module.exports = PerformanceProfiler;