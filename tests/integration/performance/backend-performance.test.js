/**
 * Backend Performance Integration Tests
 * Target: Validate 10x performance improvement goals
 */

const request = require('supertest');
const { performance } = require('perf_hooks');
const app = require('../../../backend/cmd/api-server/main');

describe('Backend Performance Integration Tests', () => {
    let server;
    let baseURL;
    
    beforeAll(async () => {
        // Start test server
        server = app.listen(0); // Use random available port
        const port = server.address().port;
        baseURL = `http://localhost:${port}`;
        
        // Wait for server to be ready
        await new Promise(resolve => setTimeout(resolve, 1000));
    });
    
    afterAll(async () => {
        if (server) {
            await new Promise(resolve => server.close(resolve));
        }
    });

    describe('API Response Time Performance', () => {
        test('VM list endpoint should respond in <100ms (target: <50ms)', async () => {
            const iterations = 10;
            const responseTimes = [];
            
            for (let i = 0; i < iterations; i++) {
                const start = performance.now();
                const response = await request(baseURL)
                    .get('/api/v1/vms')
                    .expect(200);
                const end = performance.now();
                
                const responseTime = end - start;
                responseTimes.push(responseTime);
                
                expect(response.headers['x-response-time']).toBeDefined();
            }
            
            const averageTime = responseTimes.reduce((a, b) => a + b) / responseTimes.length;
            const p90Time = responseTimes.sort((a, b) => a - b)[Math.floor(0.9 * responseTimes.length)];
            
            console.log(`Average response time: ${averageTime.toFixed(2)}ms`);
            console.log(`90th percentile: ${p90Time.toFixed(2)}ms`);
            
            // Current target: <100ms (final target: <50ms)
            expect(averageTime).toBeLessThan(100);
            expect(p90Time).toBeLessThan(150);
        });

        test('Dashboard endpoint should respond in <200ms (target: <100ms)', async () => {
            const start = performance.now();
            const response = await request(baseURL)
                .get('/api/v1/dashboard')
                .query({ organizationId: 'test-org' })
                .expect(200);
            const end = performance.now();
            
            const responseTime = end - start;
            console.log(`Dashboard response time: ${responseTime.toFixed(2)}ms`);
            
            // Current target: <200ms (final target: <100ms)
            expect(responseTime).toBeLessThan(200);
            expect(response.body).toHaveProperty('realtime');
        });

        test('VM metrics endpoint should respond in <150ms (target: <50ms)', async () => {
            const start = performance.now();
            const response = await request(baseURL)
                .get('/api/v1/vms/test-vm-id/metrics')
                .query({
                    start: new Date(Date.now() - 3600000).toISOString(),
                    end: new Date().toISOString()
                })
                .expect(200);
            const end = performance.now();
            
            const responseTime = end - start;
            console.log(`VM metrics response time: ${responseTime.toFixed(2)}ms`);
            
            // Current target: <150ms (final target: <50ms)
            expect(responseTime).toBeLessThan(150);
        });
    });

    describe('Database Performance', () => {
        test('connection pool should handle concurrent requests efficiently', async () => {
            const concurrentRequests = 50;
            const requests = [];
            
            const startTime = performance.now();
            
            for (let i = 0; i < concurrentRequests; i++) {
                requests.push(
                    request(baseURL)
                        .get('/api/v1/vms')
                        .expect(200)
                );
            }
            
            const responses = await Promise.all(requests);
            const endTime = performance.now();
            const totalTime = endTime - startTime;
            
            console.log(`${concurrentRequests} concurrent requests completed in ${totalTime.toFixed(2)}ms`);
            console.log(`Average time per request: ${(totalTime / concurrentRequests).toFixed(2)}ms`);
            
            // All requests should complete successfully
            expect(responses).toHaveLength(concurrentRequests);
            
            // Total time should be reasonable for concurrent execution
            expect(totalTime).toBeLessThan(5000); // 5 seconds max for 50 requests
        });

        test('bulk metrics ingestion should handle large payloads efficiently', async () => {
            const metricsCount = 1000;
            const metrics = Array.from({ length: metricsCount }, (_, i) => ({
                vm_id: `test-vm-${i % 10}`,
                cpu_usage: Math.random() * 100,
                memory_percent: Math.random() * 100,
                disk_io: Math.random() * 1000,
                network_io: Math.random() * 1000,
                timestamp: new Date(Date.now() - i * 1000).toISOString()
            }));
            
            const start = performance.now();
            const response = await request(baseURL)
                .post('/api/v1/metrics/bulk')
                .send(metrics)
                .expect(200);
            const end = performance.now();
            
            const responseTime = end - start;
            console.log(`Bulk metrics ingestion (${metricsCount} records): ${responseTime.toFixed(2)}ms`);
            
            // Should handle 1000 metrics in <2 seconds
            expect(responseTime).toBeLessThan(2000);
            expect(response.body.count).toBe(metricsCount);
        });
    });

    describe('Memory and Resource Usage', () => {
        test('memory usage should remain stable during load', async () => {
            const memoryBefore = process.memoryUsage();
            
            // Generate load
            const requests = Array.from({ length: 100 }, () =>
                request(baseURL)
                    .get('/api/v1/vms')
                    .expect(200)
            );
            
            await Promise.all(requests);
            
            // Wait for GC
            if (global.gc) {
                global.gc();
            }
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            const memoryAfter = process.memoryUsage();
            const heapGrowth = memoryAfter.heapUsed - memoryBefore.heapUsed;
            
            console.log(`Memory usage before: ${(memoryBefore.heapUsed / 1024 / 1024).toFixed(2)}MB`);
            console.log(`Memory usage after: ${(memoryAfter.heapUsed / 1024 / 1024).toFixed(2)}MB`);
            console.log(`Heap growth: ${(heapGrowth / 1024 / 1024).toFixed(2)}MB`);
            
            // Memory growth should be minimal (<50MB for 100 requests)
            expect(heapGrowth).toBeLessThan(50 * 1024 * 1024);
        });

        test('cache effectiveness should improve response times', async () => {
            const endpoint = '/api/v1/dashboard?organizationId=cache-test';
            
            // First request (cache miss)
            const start1 = performance.now();
            const response1 = await request(baseURL)
                .get(endpoint)
                .expect(200);
            const time1 = performance.now() - start1;
            
            expect(response1.headers['x-cache-hit']).toBe('false');
            
            // Second request (cache hit)
            const start2 = performance.now();
            const response2 = await request(baseURL)
                .get(endpoint)
                .expect(200);
            const time2 = performance.now() - start2;
            
            expect(response2.headers['x-cache-hit']).toBe('true');
            
            console.log(`Cache miss time: ${time1.toFixed(2)}ms`);
            console.log(`Cache hit time: ${time2.toFixed(2)}ms`);
            console.log(`Cache speedup: ${(time1 / time2).toFixed(2)}x`);
            
            // Cache should provide significant speedup
            expect(time2).toBeLessThan(time1 * 0.5); // At least 2x faster
        });
    });

    describe('Error Handling Performance', () => {
        test('error responses should be fast', async () => {
            const start = performance.now();
            const response = await request(baseURL)
                .get('/api/v1/vms/non-existent-vm')
                .expect(404);
            const end = performance.now();
            
            const responseTime = end - start;
            console.log(`Error response time: ${responseTime.toFixed(2)}ms`);
            
            // Error responses should be very fast (<50ms)
            expect(responseTime).toBeLessThan(50);
        });

        test('rate limiting should not significantly impact performance', async () => {
            // Test rate limiting behavior under load
            const requests = Array.from({ length: 10 }, () =>
                request(baseURL)
                    .get('/api/v1/vms')
            );
            
            const start = performance.now();
            const responses = await Promise.allSettled(requests);
            const end = performance.now();
            
            const totalTime = end - start;
            const successCount = responses.filter(r => r.status === 'fulfilled').length;
            
            console.log(`Rate limiting test: ${successCount}/${requests.length} succeeded in ${totalTime.toFixed(2)}ms`);
            
            // Should handle reasonable load without major slowdown
            expect(totalTime).toBeLessThan(3000); // 3 seconds for 10 requests
        });
    });

    describe('GraphQL Performance', () => {
        test('GraphQL queries should be optimized', async () => {
            const query = `
                query {
                    vms(first: 10) {
                        edges {
                            node {
                                id
                                name
                                state
                                metrics {
                                    cpuUsage
                                    memoryUsage
                                }
                            }
                        }
                    }
                }
            `;
            
            const start = performance.now();
            const response = await request(baseURL)
                .post('/graphql')
                .send({ query })
                .expect(200);
            const end = performance.now();
            
            const responseTime = end - start;
            console.log(`GraphQL query time: ${responseTime.toFixed(2)}ms`);
            
            // GraphQL should be efficient
            expect(responseTime).toBeLessThan(200);
            expect(response.body.errors).toBeUndefined();
        });
    });
});

/**
 * Performance Benchmarking Helper Functions
 */

class PerformanceBenchmark {
    static async measureEndpoint(baseURL, endpoint, options = {}) {
        const iterations = options.iterations || 10;
        const results = [];
        
        for (let i = 0; i < iterations; i++) {
            const start = performance.now();
            const response = await request(baseURL).get(endpoint);
            const end = performance.now();
            
            results.push({
                iteration: i + 1,
                responseTime: end - start,
                status: response.status,
                cacheHit: response.headers['x-cache-hit'] === 'true'
            });
            
            if (options.delay) {
                await new Promise(resolve => setTimeout(resolve, options.delay));
            }
        }
        
        return this.calculateStats(results);
    }
    
    static calculateStats(results) {
        const times = results.map(r => r.responseTime);
        const sorted = [...times].sort((a, b) => a - b);
        
        return {
            count: results.length,
            average: times.reduce((a, b) => a + b) / times.length,
            median: sorted[Math.floor(sorted.length / 2)],
            p90: sorted[Math.floor(0.9 * sorted.length)],
            p99: sorted[Math.floor(0.99 * sorted.length)],
            min: Math.min(...times),
            max: Math.max(...times),
            successRate: results.filter(r => r.status === 200).length / results.length
        };
    }
}