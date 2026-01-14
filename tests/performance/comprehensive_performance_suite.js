import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter, Gauge } from 'k6/metrics';
import { SharedArray } from 'k6/data';
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.1/index.js';
import { htmlReport } from 'https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js';

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8080';
const API_TOKEN = __ENV.API_TOKEN || 'test-token';
const TEST_DURATION = __ENV.TEST_DURATION || '30m';
const TARGET_VUS = __ENV.TARGET_VUS || '1000';

// Custom metrics for comprehensive performance analysis
const apiLatency = new Trend('api_latency', true);
const vmProvisioningTime = new Trend('vm_provisioning_time', true);
const migrationDuration = new Trend('migration_duration', true);
const p2pNetworkFormation = new Trend('p2p_network_formation', true);
const bandwidthUtilization = new Gauge('bandwidth_utilization');
const cpuUtilization = new Gauge('cpu_utilization');
const memoryUtilization = new Gauge('memory_utilization');
const diskIOPS = new Gauge('disk_iops');
const networkThroughput = new Gauge('network_throughput');
const errorRate = new Rate('error_rate');
const successRate = new Rate('success_rate');
const transactionRate = new Counter('transaction_rate');
const aiPredictionAccuracy = new Rate('ai_prediction_accuracy');
const cacheHitRate = new Rate('cache_hit_rate');
const queueDepth = new Gauge('queue_depth');
const connectionPoolUtilization = new Gauge('connection_pool_utilization');

// Helper function to safely load test data
function safeLoadTestData(filename, envVar) {
    // First try to use environment variable data
    if (__ENV[envVar]) {
        try {
            return JSON.parse(__ENV[envVar]);
        } catch (e) {
            console.warn(`Failed to parse ${envVar} from environment: ${e.message}`);
        }
    }

    // Then try to load from file
    try {
        return JSON.parse(open(filename));
    } catch (e) {
        console.warn(`Failed to load ${filename}: ${e.message}, using empty array`);
        return [];
    }
}

// Test data
const testVMs = new SharedArray('vms', function() {
    return safeLoadTestData('./test-data/vms.json', 'VMS_JSON');
});

const testClusters = new SharedArray('clusters', function() {
    return safeLoadTestData('./test-data/clusters.json', 'CLUSTERS_JSON');
});

// Test scenarios configuration
export const options = {
    scenarios: {
        // Baseline performance test
        baseline: {
            executor: 'constant-vus',
            vus: 10,
            duration: '5m',
            tags: { scenario: 'baseline' },
            exec: 'baselinePerformance',
        },

        // Load test with gradual ramp-up
        loadTest: {
            executor: 'ramping-vus',
            startVUs: 0,
            stages: [
                { duration: '5m', target: 100 },
                { duration: '10m', target: 500 },
                { duration: '10m', target: 1000 },
                { duration: '5m', target: 0 },
            ],
            tags: { scenario: 'load' },
            exec: 'loadTest',
            startTime: '5m',
        },

        // Stress test
        stressTest: {
            executor: 'ramping-arrival-rate',
            startRate: 10,
            timeUnit: '1s',
            preAllocatedVUs: 500,
            maxVUs: 2000,
            stages: [
                { duration: '2m', target: 100 },
                { duration: '5m', target: 500 },
                { duration: '2m', target: 1000 },
                { duration: '5m', target: 1500 },
                { duration: '2m', target: 0 },
            ],
            tags: { scenario: 'stress' },
            exec: 'stressTest',
            startTime: '35m',
        },

        // Spike test
        spikeTest: {
            executor: 'ramping-vus',
            stages: [
                { duration: '10s', target: 2000 },
                { duration: '1m', target: 2000 },
                { duration: '10s', target: 0 },
            ],
            tags: { scenario: 'spike' },
            exec: 'spikeTest',
            startTime: '51m',
        },

        // Soak test
        soakTest: {
            executor: 'constant-vus',
            vus: 200,
            duration: '2h',
            tags: { scenario: 'soak' },
            exec: 'soakTest',
            startTime: '53m',
        },

        // P2P fabric performance
        p2pFabricTest: {
            executor: 'per-vu-iterations',
            vus: 50,
            iterations: 10,
            tags: { scenario: 'p2p' },
            exec: 'p2pFabricPerformance',
            startTime: '173m',
        },

        // AI subsystem performance
        aiPerformanceTest: {
            executor: 'constant-arrival-rate',
            rate: 100,
            timeUnit: '1s',
            duration: '10m',
            preAllocatedVUs: 50,
            tags: { scenario: 'ai' },
            exec: 'aiSubsystemPerformance',
            startTime: '178m',
        },

        // Database performance
        databasePerformanceTest: {
            executor: 'ramping-arrival-rate',
            startRate: 50,
            timeUnit: '1s',
            preAllocatedVUs: 100,
            stages: [
                { duration: '2m', target: 200 },
                { duration: '5m', target: 500 },
                { duration: '2m', target: 100 },
            ],
            tags: { scenario: 'database' },
            exec: 'databasePerformance',
            startTime: '188m',
        },
    },

    thresholds: {
        'http_req_duration': ['p(95)<500', 'p(99)<1000'],
        'api_latency': ['p(95)<300', 'p(99)<500'],
        'vm_provisioning_time': ['p(95)<30000', 'p(99)<60000'],
        'migration_duration': ['p(95)<120000', 'p(99)<180000'],
        'error_rate': ['rate<0.01'],
        'success_rate': ['rate>0.99'],
        'cpu_utilization': ['value<80'],
        'memory_utilization': ['value<85'],
        'cache_hit_rate': ['rate>0.8'],
    },
};

// Setup function
export function setup() {
    console.log('Setting up performance test suite...');

    // Verify connectivity
    const healthCheck = http.get(`${BASE_URL}/health`);
    check(healthCheck, {
        'Health check successful': (r) => r.status === 200,
    });

    // Initialize test environment
    const initResponse = http.post(
        `${BASE_URL}/api/v1/test/init`,
        JSON.stringify({
            environment: 'performance-test',
            reset: true,
        }),
        {
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${API_TOKEN}`,
            },
        }
    );

    check(initResponse, {
        'Test environment initialized': (r) => r.status === 200,
    });

    return {
        testStartTime: new Date().toISOString(),
        baselineMetrics: collectBaselineMetrics(),
    };
}

// Baseline performance test
export function baselinePerformance() {
    group('Baseline API Performance', () => {
        // Test core API endpoints
        const endpoints = [
            '/api/v1/vms',
            '/api/v1/clusters',
            '/api/v1/networks',
            '/api/v1/storage',
            '/api/v1/metrics',
        ];

        endpoints.forEach(endpoint => {
            const start = Date.now();
            const response = http.get(`${BASE_URL}${endpoint}`, {
                headers: { 'Authorization': `Bearer ${API_TOKEN}` },
                tags: { endpoint: endpoint },
            });
            const duration = Date.now() - start;

            apiLatency.add(duration);

            check(response, {
                [`${endpoint} status 200`]: (r) => r.status === 200,
                [`${endpoint} response time OK`]: (r) => r.timings.duration < 500,
            });

            successRate.add(response.status === 200);
            errorRate.add(response.status >= 400);
        });
    });

    sleep(1);
}

// Load test
export function loadTest() {
    group('Load Test Operations', () => {
        // VM provisioning under load
        const vmRequest = {
            name: `vm-load-${__VU}-${__ITER}`,
            cpus: 2,
            memory: 4096,
            disk: 50,
            network: 'default',
        };

        const start = Date.now();
        const response = http.post(
            `${BASE_URL}/api/v1/vms`,
            JSON.stringify(vmRequest),
            {
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${API_TOKEN}`,
                },
                tags: { operation: 'vm_provision' },
            }
        );
        const provisionTime = Date.now() - start;

        vmProvisioningTime.add(provisionTime);
        transactionRate.add(1);

        check(response, {
            'VM provisioned': (r) => r.status === 201,
            'Provision time acceptable': () => provisionTime < 30000,
        });

        if (response.status === 201) {
            const vm = JSON.parse(response.body);

            // Test VM operations
            testVMOperations(vm.id);

            // Cleanup
            http.del(`${BASE_URL}/api/v1/vms/${vm.id}`, {
                headers: { 'Authorization': `Bearer ${API_TOKEN}` },
            });
        }
    });

    sleep(Math.random() * 2);
}

// Stress test
export function stressTest() {
    group('Stress Test Operations', () => {
        // Concurrent operations
        const operations = [
            () => createVM(),
            () => migrateVM(),
            () => scaleCluster(),
            () => runBackup(),
            () => queryMetrics(),
        ];

        const operation = operations[Math.floor(Math.random() * operations.length)];
        const result = operation();

        check(result, {
            'Operation successful under stress': (r) => r.success,
            'Response time under stress': (r) => r.duration < 2000,
        });

        // Monitor resource utilization
        const metrics = http.get(`${BASE_URL}/api/v1/metrics/current`, {
            headers: { 'Authorization': `Bearer ${API_TOKEN}` },
        });

        if (metrics.status === 200) {
            const data = JSON.parse(metrics.body);
            cpuUtilization.add(data.cpu);
            memoryUtilization.add(data.memory);
            diskIOPS.add(data.diskIOPS);
            networkThroughput.add(data.networkThroughput);
        }
    });

    sleep(0.1);
}

// Spike test
export function spikeTest() {
    group('Spike Test Operations', () => {
        // Sudden burst of requests
        const batchSize = 10;
        const batch = http.batch(
            Array.from({ length: batchSize }, (_, i) => [
                'GET',
                `${BASE_URL}/api/v1/vms?page=${i}`,
                null,
                { headers: { 'Authorization': `Bearer ${API_TOKEN}` } },
            ])
        );

        batch.forEach(response => {
            check(response, {
                'Handled spike load': (r) => r.status < 500,
            });
        });

        // Test queue depth during spike
        const queueMetrics = http.get(`${BASE_URL}/api/v1/metrics/queue`, {
            headers: { 'Authorization': `Bearer ${API_TOKEN}` },
        });

        if (queueMetrics.status === 200) {
            const data = JSON.parse(queueMetrics.body);
            queueDepth.add(data.depth);
        }
    });
}

// Soak test
export function soakTest() {
    group('Soak Test Operations', () => {
        // Long-running operations
        const operations = [
            'vm_lifecycle',
            'data_processing',
            'backup_restore',
            'migration',
            'monitoring',
        ];

        const operation = operations[__ITER % operations.length];

        switch(operation) {
            case 'vm_lifecycle':
                performVMLifecycle();
                break;
            case 'data_processing':
                performDataProcessing();
                break;
            case 'backup_restore':
                performBackupRestore();
                break;
            case 'migration':
                performMigration();
                break;
            case 'monitoring':
                performMonitoring();
                break;
        }

        // Check for memory leaks
        const memoryStats = http.get(`${BASE_URL}/api/v1/metrics/memory`, {
            headers: { 'Authorization': `Bearer ${API_TOKEN}` },
        });

        if (memoryStats.status === 200) {
            const data = JSON.parse(memoryStats.body);
            check(data, {
                'No memory leak detected': (d) => d.heapUsed < d.heapTotal * 0.9,
            });
        }
    });

    sleep(5);
}

// P2P fabric performance test
export function p2pFabricPerformance() {
    group('P2P Fabric Performance', () => {
        // Test network formation
        const start = Date.now();
        const response = http.post(
            `${BASE_URL}/api/v1/p2p/network/form`,
            JSON.stringify({
                nodes: 100,
                topology: 'mesh',
                redundancy: 3,
            }),
            {
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${API_TOKEN}`,
                },
            }
        );
        const formationTime = Date.now() - start;

        p2pNetworkFormation.add(formationTime);

        check(response, {
            'P2P network formed': (r) => r.status === 200,
            'Formation time acceptable': () => formationTime < 10000,
        });

        if (response.status === 200) {
            const network = JSON.parse(response.body);

            // Test P2P operations
            testP2POperations(network.id);

            // Cleanup
            http.del(`${BASE_URL}/api/v1/p2p/network/${network.id}`, {
                headers: { 'Authorization': `Bearer ${API_TOKEN}` },
            });
        }
    });
}

// AI subsystem performance test
export function aiSubsystemPerformance() {
    group('AI Subsystem Performance', () => {
        // Test prediction accuracy
        const predictionRequest = {
            model: 'resource-predictor',
            input: {
                cpu_history: Array.from({ length: 100 }, () => Math.random() * 100),
                memory_history: Array.from({ length: 100 }, () => Math.random() * 100),
                timestamp: Date.now(),
            },
        };

        const response = http.post(
            `${BASE_URL}/api/v1/ai/predict`,
            JSON.stringify(predictionRequest),
            {
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${API_TOKEN}`,
                },
            }
        );

        check(response, {
            'AI prediction successful': (r) => r.status === 200,
            'Prediction latency OK': (r) => r.timings.duration < 100,
        });

        if (response.status === 200) {
            const prediction = JSON.parse(response.body);

            // Validate prediction accuracy
            const validation = http.post(
                `${BASE_URL}/api/v1/ai/validate`,
                JSON.stringify({
                    prediction_id: prediction.id,
                    actual_values: {
                        cpu: Math.random() * 100,
                        memory: Math.random() * 100,
                    },
                }),
                {
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${API_TOKEN}`,
                    },
                }
            );

            if (validation.status === 200) {
                const result = JSON.parse(validation.body);
                aiPredictionAccuracy.add(result.accuracy > 0.8);
            }
        }
    });

    sleep(0.5);
}

// Database performance test
export function databasePerformance() {
    group('Database Performance', () => {
        // Test database operations
        const operations = [
            { type: 'read', query: 'SELECT * FROM vms LIMIT 100' },
            { type: 'write', query: 'INSERT INTO metrics ...' },
            { type: 'update', query: 'UPDATE vms SET ...' },
            { type: 'delete', query: 'DELETE FROM old_metrics ...' },
            { type: 'aggregate', query: 'SELECT COUNT(*) ...' },
        ];

        const operation = operations[Math.floor(Math.random() * operations.length)];

        const response = http.post(
            `${BASE_URL}/api/v1/database/query`,
            JSON.stringify(operation),
            {
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${API_TOKEN}`,
                },
                tags: { db_operation: operation.type },
            }
        );

        check(response, {
            'Database query successful': (r) => r.status === 200,
            'Query performance OK': (r) => r.timings.duration < 100,
        });

        // Test cache effectiveness
        const cacheStats = http.get(`${BASE_URL}/api/v1/cache/stats`, {
            headers: { 'Authorization': `Bearer ${API_TOKEN}` },
        });

        if (cacheStats.status === 200) {
            const stats = JSON.parse(cacheStats.body);
            cacheHitRate.add(stats.hits / (stats.hits + stats.misses));
        }

        // Test connection pool
        const poolStats = http.get(`${BASE_URL}/api/v1/database/pool/stats`, {
            headers: { 'Authorization': `Bearer ${API_TOKEN}` },
        });

        if (poolStats.status === 200) {
            const stats = JSON.parse(poolStats.body);
            connectionPoolUtilization.add(stats.active / stats.total);
        }
    });

    sleep(0.2);
}

// Helper functions
function createVM() {
    const start = Date.now();
    const response = http.post(
        `${BASE_URL}/api/v1/vms`,
        JSON.stringify({
            name: `vm-stress-${Date.now()}`,
            cpus: 1,
            memory: 2048,
        }),
        {
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${API_TOKEN}`,
            },
        }
    );

    return {
        success: response.status === 201,
        duration: Date.now() - start,
    };
}

function migrateVM() {
    // Use mock VM data if no test data available
    const vms = testVMs.length > 0
        ? testVMs[Math.floor(Math.random() * testVMs.length)]
        : { id: `mock-vm-${Math.floor(Math.random() * 1000)}` };

    const start = Date.now();

    const response = http.post(
        `${BASE_URL}/api/v1/vms/${vms.id}/migrate`,
        JSON.stringify({ target_host: 'host-2' }),
        {
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${API_TOKEN}`,
            },
        }
    );

    const duration = Date.now() - start;
    migrationDuration.add(duration);

    return {
        success: response.status === 200,
        duration: duration,
    };
}

function scaleCluster() {
    // Use mock cluster data if no test data available
    const cluster = testClusters.length > 0
        ? testClusters[Math.floor(Math.random() * testClusters.length)]
        : { id: `mock-cluster-${Math.floor(Math.random() * 1000)}` };

    const start = Date.now();

    const response = http.post(
        `${BASE_URL}/api/v1/clusters/${cluster.id}/scale`,
        JSON.stringify({ replicas: Math.floor(Math.random() * 10) + 1 }),
        {
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${API_TOKEN}`,
            },
        }
    );

    return {
        success: response.status === 200,
        duration: Date.now() - start,
    };
}

function runBackup() {
    const start = Date.now();
    const response = http.post(
        `${BASE_URL}/api/v1/backup/create`,
        JSON.stringify({ type: 'incremental' }),
        {
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${API_TOKEN}`,
            },
        }
    );

    return {
        success: response.status === 202,
        duration: Date.now() - start,
    };
}

function queryMetrics() {
    const start = Date.now();
    const response = http.get(
        `${BASE_URL}/api/v1/metrics?start=${Date.now() - 3600000}&end=${Date.now()}`,
        {
            headers: { 'Authorization': `Bearer ${API_TOKEN}` },
        }
    );

    return {
        success: response.status === 200,
        duration: Date.now() - start,
    };
}

function testVMOperations(vmId) {
    // Start VM
    http.post(`${BASE_URL}/api/v1/vms/${vmId}/start`, null, {
        headers: { 'Authorization': `Bearer ${API_TOKEN}` },
    });

    sleep(1);

    // Stop VM
    http.post(`${BASE_URL}/api/v1/vms/${vmId}/stop`, null, {
        headers: { 'Authorization': `Bearer ${API_TOKEN}` },
    });

    sleep(1);

    // Restart VM
    http.post(`${BASE_URL}/api/v1/vms/${vmId}/restart`, null, {
        headers: { 'Authorization': `Bearer ${API_TOKEN}` },
    });
}

function testP2POperations(networkId) {
    // Test peer discovery
    const peers = http.get(`${BASE_URL}/api/v1/p2p/network/${networkId}/peers`, {
        headers: { 'Authorization': `Bearer ${API_TOKEN}` },
    });

    check(peers, {
        'Peers discovered': (r) => r.status === 200 && JSON.parse(r.body).length > 0,
    });

    // Test DHT operations
    const dhtPut = http.put(
        `${BASE_URL}/api/v1/p2p/network/${networkId}/dht`,
        JSON.stringify({ key: 'test-key', value: 'test-value' }),
        {
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${API_TOKEN}`,
            },
        }
    );

    check(dhtPut, {
        'DHT put successful': (r) => r.status === 200,
    });

    const dhtGet = http.get(`${BASE_URL}/api/v1/p2p/network/${networkId}/dht/test-key`, {
        headers: { 'Authorization': `Bearer ${API_TOKEN}` },
    });

    check(dhtGet, {
        'DHT get successful': (r) => r.status === 200,
    });
}

function performVMLifecycle() {
    const vmRequest = {
        name: `vm-soak-${Date.now()}`,
        cpus: 2,
        memory: 4096,
    };

    // Create
    const create = http.post(`${BASE_URL}/api/v1/vms`, JSON.stringify(vmRequest), {
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${API_TOKEN}`,
        },
    });

    if (create.status === 201) {
        const vm = JSON.parse(create.body);

        // Update
        http.patch(
            `${BASE_URL}/api/v1/vms/${vm.id}`,
            JSON.stringify({ memory: 8192 }),
            {
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${API_TOKEN}`,
                },
            }
        );

        // Delete
        http.del(`${BASE_URL}/api/v1/vms/${vm.id}`, {
            headers: { 'Authorization': `Bearer ${API_TOKEN}` },
        });
    }
}

function performDataProcessing() {
    const data = {
        dataset: Array.from({ length: 1000 }, () => Math.random()),
        operation: 'aggregate',
    };

    http.post(`${BASE_URL}/api/v1/process`, JSON.stringify(data), {
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${API_TOKEN}`,
        },
    });
}

function performBackupRestore() {
    // Create backup
    const backup = http.post(`${BASE_URL}/api/v1/backup/create`, JSON.stringify({ type: 'full' }), {
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${API_TOKEN}`,
        },
    });

    if (backup.status === 202) {
        const backupData = JSON.parse(backup.body);

        // Wait for backup completion
        sleep(10);

        // Restore
        http.post(
            `${BASE_URL}/api/v1/backup/restore`,
            JSON.stringify({ backup_id: backupData.id }),
            {
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${API_TOKEN}`,
                },
            }
        );
    }
}

function performMigration() {
    // Use mock VM data if no test data available
    const vms = testVMs.length > 0
        ? testVMs[Math.floor(Math.random() * testVMs.length)]
        : { id: `mock-vm-${Math.floor(Math.random() * 1000)}` };

    http.post(
        `${BASE_URL}/api/v1/vms/${vms.id}/migrate`,
        JSON.stringify({ target_host: 'host-3', live: true }),
        {
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${API_TOKEN}`,
            },
        }
    );
}

function performMonitoring() {
    // Collect various metrics
    const endpoints = [
        '/api/v1/metrics/cpu',
        '/api/v1/metrics/memory',
        '/api/v1/metrics/disk',
        '/api/v1/metrics/network',
    ];

    endpoints.forEach(endpoint => {
        http.get(`${BASE_URL}${endpoint}`, {
            headers: { 'Authorization': `Bearer ${API_TOKEN}` },
        });
    });
}

function collectBaselineMetrics() {
    const response = http.get(`${BASE_URL}/api/v1/metrics/baseline`, {
        headers: { 'Authorization': `Bearer ${API_TOKEN}` },
    });

    if (response.status === 200) {
        return JSON.parse(response.body);
    }

    return null;
}

// Teardown function
export function teardown(data) {
    console.log('Tearing down performance test suite...');

    // Cleanup test environment
    http.post(
        `${BASE_URL}/api/v1/test/cleanup`,
        JSON.stringify({
            environment: 'performance-test',
            preserve_results: true,
        }),
        {
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${API_TOKEN}`,
            },
        }
    );

    // Calculate performance degradation
    if (data.baselineMetrics) {
        const currentMetrics = collectBaselineMetrics();
        if (currentMetrics) {
            console.log('Performance Analysis:');
            console.log(`CPU degradation: ${currentMetrics.cpu - data.baselineMetrics.cpu}%`);
            console.log(`Memory degradation: ${currentMetrics.memory - data.baselineMetrics.memory}%`);
            console.log(`Response time increase: ${currentMetrics.responseTime - data.baselineMetrics.responseTime}ms`);
        }
    }

    console.log(`Test completed at: ${new Date().toISOString()}`);
    console.log(`Total duration: ${(Date.now() - new Date(data.testStartTime).getTime()) / 1000}s`);
}

// Custom summary handler
export function handleSummary(data) {
    return {
        'stdout': textSummary(data, { indent: ' ', enableColors: true }),
        'performance-report.html': htmlReport(data),
        'performance-report.json': JSON.stringify(data, null, 2),
    };
}