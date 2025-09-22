import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter, Gauge } from 'k6/metrics';
import { htmlReport } from "https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js";
import { textSummary } from "https://jslib.k6.io/k6-summary/0.0.1/index.js";

// Custom metrics for bandwidth optimization
const qosPolicyApplicationTime = new Trend('qos_policy_application_time', true);
const trafficClassificationAccuracy = new Rate('traffic_classification_accuracy');
const rateLimitingEffectiveness = new Rate('rate_limiting_effectiveness');
const policyUpdatePropagationTime = new Trend('policy_update_propagation_time', true);
const bandwidthMonitoringOverhead = new Gauge('bandwidth_monitoring_overhead');
const alertGenerationLatency = new Trend('alert_generation_latency', true);
const trafficShapingAccuracy = new Rate('traffic_shaping_accuracy');
const burstHandlingCapacity = new Gauge('burst_handling_capacity');
const congestionControlEffectiveness = new Rate('congestion_control_effectiveness');
const aiPredictionAccuracy = new Rate('ai_prediction_accuracy');
const routeOptimizationTime = new Trend('route_optimization_time', true);
const adaptiveQosAdjustmentTime = new Trend('adaptive_qos_adjustment_time', true);
const crossClusterBandwidthEfficiency = new Rate('cross_cluster_bandwidth_efficiency');
const wanOptimizationGain = new Gauge('wan_optimization_gain');
const adaptationResponseTime = new Trend('adaptation_response_time', true);

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8080';
const API_TOKEN = __ENV.API_TOKEN || '';

export const options = {
  scenarios: {
    // QoS Policy Performance
    qos_policy: {
      executor: 'constant-arrival-rate',
      rate: 50,
      timeUnit: '1s',
      duration: '5m',
      preAllocatedVUs: 20,
      maxVUs: 100,
      tags: { scenario: 'qos_policy' },
      exec: 'qosPolicyPerformance',
    },
    // Bandwidth Monitoring Efficiency
    bandwidth_monitoring: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '30s', target: 10 },
        { duration: '1m', target: 50 },
        { duration: '2m', target: 100 },
        { duration: '1m', target: 50 },
        { duration: '30s', target: 0 },
      ],
      tags: { scenario: 'bandwidth_monitoring' },
      exec: 'bandwidthMonitoring',
    },
    // Traffic Shaping
    traffic_shaping: {
      executor: 'per-vu-iterations',
      vus: 50,
      iterations: 20,
      maxDuration: '10m',
      tags: { scenario: 'traffic_shaping' },
      exec: 'trafficShaping',
    },
    // Network Optimization
    network_optimization: {
      executor: 'shared-iterations',
      vus: 30,
      iterations: 300,
      maxDuration: '15m',
      tags: { scenario: 'network_optimization' },
      exec: 'networkOptimization',
    },
    // Cross-Cluster Bandwidth
    cross_cluster_bandwidth: {
      executor: 'constant-vus',
      vus: 25,
      duration: '10m',
      tags: { scenario: 'cross_cluster' },
      exec: 'crossClusterBandwidth',
    },
    // Real-time Adaptation
    realtime_adaptation: {
      executor: 'ramping-arrival-rate',
      startRate: 10,
      timeUnit: '1s',
      preAllocatedVUs: 30,
      stages: [
        { duration: '1m', target: 20 },
        { duration: '2m', target: 50 },
        { duration: '2m', target: 100 },
        { duration: '1m', target: 20 },
      ],
      tags: { scenario: 'realtime_adaptation' },
      exec: 'realtimeAdaptation',
    },
    // Scalability Tests
    scalability: {
      executor: 'constant-vus',
      vus: 20,
      duration: '5m',
      tags: { scenario: 'scalability' },
      exec: 'scalabilityTests',
    },
  },
  thresholds: {
    'qos_policy_application_time': ['p(95)<100', 'p(99)<200'],
    'traffic_classification_accuracy': ['rate>0.95'],
    'rate_limiting_effectiveness': ['rate>0.98'],
    'policy_update_propagation_time': ['p(95)<500', 'p(99)<1000'],
    'bandwidth_monitoring_overhead': ['value<5'], // Less than 5% overhead
    'alert_generation_latency': ['p(95)<1000', 'p(99)<2000'],
    'traffic_shaping_accuracy': ['rate>0.95'],
    'congestion_control_effectiveness': ['rate>0.90'],
    'ai_prediction_accuracy': ['rate>0.85'],
    'route_optimization_time': ['p(95)<2000', 'p(99)<5000'],
    'adaptive_qos_adjustment_time': ['p(95)<1000', 'p(99)<2000'],
    'cross_cluster_bandwidth_efficiency': ['rate>0.80'],
    'adaptation_response_time': ['p(95)<500', 'p(99)<1000'],
    'http_req_duration': ['p(95)<300', 'p(99)<500'],
    'http_req_failed': ['rate<0.05'],
  },
};

const headers = {
  'Content-Type': 'application/json',
  'Authorization': `Bearer ${API_TOKEN}`,
};

// QoS Policy Performance Tests
export function qosPolicyPerformance() {
  group('QoS Policy Performance', () => {
    const policyStart = Date.now();

    // Apply QoS policy
    const policyTypes = ['guaranteed', 'assured', 'best-effort'];
    const policyType = policyTypes[__ITER % policyTypes.length];

    const applyResponse = http.post(
      `${BASE_URL}/api/network/qos/policy`,
      JSON.stringify({
        policyId: `policy-${__VU}-${__ITER}`,
        type: policyType,
        rules: [
          {
            priority: 1,
            match: { protocol: 'tcp', port: 443 },
            action: { bandwidth: '10Mbps', latency: '10ms' },
          },
          {
            priority: 2,
            match: { protocol: 'udp', port: 53 },
            action: { bandwidth: '1Mbps', latency: '5ms' },
          },
        ],
        targets: [`vm-${__VU}`, `vm-${__VU + 1}`],
      }),
      { headers, tags: { operation: 'qos_apply' } }
    );

    check(applyResponse, {
      'QoS policy applied': (r) => r.status === 200,
      'policy active': (r) => r.json('status') === 'active',
    });

    qosPolicyApplicationTime.add(Date.now() - policyStart);

    // Test traffic classification
    const classifyResponse = http.post(
      `${BASE_URL}/api/network/traffic/classify`,
      JSON.stringify({
        flows: [
          { src: '10.0.0.1', dst: '10.0.0.2', protocol: 'tcp', port: 443 },
          { src: '10.0.0.3', dst: '10.0.0.4', protocol: 'udp', port: 53 },
        ],
      }),
      { headers, tags: { operation: 'traffic_classify' } }
    );

    check(classifyResponse, {
      'traffic classified': (r) => r.status === 200,
    });

    if (classifyResponse.status === 200) {
      const accuracy = classifyResponse.json('accuracy');
      trafficClassificationAccuracy.add(accuracy);
    }

    // Test rate limiting
    const rateLimitResponse = http.post(
      `${BASE_URL}/api/network/qos/ratelimit/test`,
      JSON.stringify({
        targetRate: '10Mbps',
        duration: 5000,
        vmId: `vm-${__VU}`,
      }),
      { headers, tags: { operation: 'rate_limit_test' } }
    );

    check(rateLimitResponse, {
      'rate limiting effective': (r) => r.status === 200,
      'within tolerance': (r) => Math.abs(r.json('actualRate') - 10) < 0.5,
    });

    if (rateLimitResponse.status === 200) {
      const effectiveness = rateLimitResponse.json('actualRate') / 10;
      rateLimitingEffectiveness.add(effectiveness > 0.95 && effectiveness < 1.05 ? 1 : 0);
    }

    // Test policy propagation
    const propagationStart = Date.now();
    const propagationResponse = http.get(
      `${BASE_URL}/api/network/qos/policy/${applyResponse.json('policyId')}/status`,
      { headers, tags: { operation: 'policy_propagation' } }
    );

    check(propagationResponse, {
      'policy propagated': (r) => r.status === 200,
      'all nodes updated': (r) => r.json('propagatedNodes') === r.json('totalNodes'),
    });

    policyUpdatePropagationTime.add(Date.now() - propagationStart);
  });
}

// Bandwidth Monitoring Tests
export function bandwidthMonitoring() {
  group('Bandwidth Monitoring Efficiency', () => {
    // Start monitoring
    const monitorResponse = http.post(
      `${BASE_URL}/api/network/bandwidth/monitor`,
      JSON.stringify({
        vmId: `vm-${__VU}`,
        interval: 1000, // 1 second
        metrics: ['throughput', 'utilization', 'packet_loss', 'latency'],
      }),
      { headers, tags: { operation: 'bandwidth_monitor' } }
    );

    check(monitorResponse, {
      'monitoring started': (r) => r.status === 200,
    });

    const monitorId = monitorResponse.json('monitorId');

    // Collect metrics for 10 seconds
    sleep(10);

    // Get monitoring overhead
    const overheadResponse = http.get(
      `${BASE_URL}/api/network/bandwidth/monitor/${monitorId}/overhead`,
      { headers, tags: { operation: 'monitor_overhead' } }
    );

    check(overheadResponse, {
      'overhead measured': (r) => r.status === 200,
      'overhead acceptable': (r) => r.json('cpuOverhead') < 5 && r.json('memoryOverhead') < 5,
    });

    if (overheadResponse.status === 200) {
      bandwidthMonitoringOverhead.add(overheadResponse.json('cpuOverhead'));
    }

    // Test alert generation
    const alertStart = Date.now();
    const alertResponse = http.post(
      `${BASE_URL}/api/network/bandwidth/alert`,
      JSON.stringify({
        monitorId: monitorId,
        threshold: { metric: 'utilization', value: 80, condition: 'greater' },
      }),
      { headers, tags: { operation: 'alert_setup' } }
    );

    check(alertResponse, {
      'alert configured': (r) => r.status === 200,
    });

    // Trigger alert condition
    const triggerResponse = http.post(
      `${BASE_URL}/api/network/bandwidth/simulate`,
      JSON.stringify({
        vmId: `vm-${__VU}`,
        utilization: 85,
      }),
      { headers, tags: { operation: 'alert_trigger' } }
    );

    // Check alert was generated
    const alertCheckResponse = http.get(
      `${BASE_URL}/api/network/bandwidth/alerts?monitorId=${monitorId}`,
      { headers, tags: { operation: 'alert_check' } }
    );

    check(alertCheckResponse, {
      'alert generated': (r) => r.status === 200 && r.json('alerts').length > 0,
    });

    if (alertCheckResponse.status === 200 && alertCheckResponse.json('alerts').length > 0) {
      alertGenerationLatency.add(Date.now() - alertStart);
    }
  });
}

// Traffic Shaping Tests
export function trafficShaping() {
  group('Traffic Shaping Effectiveness', () => {
    // Configure traffic shaping
    const shapeResponse = http.post(
      `${BASE_URL}/api/network/traffic/shape`,
      JSON.stringify({
        vmId: `vm-${__VU}`,
        profile: {
          inbound: { rate: '100Mbps', burst: '10MB', priority: 'high' },
          outbound: { rate: '50Mbps', burst: '5MB', priority: 'medium' },
        },
      }),
      { headers, tags: { operation: 'traffic_shape' } }
    );

    check(shapeResponse, {
      'shaping configured': (r) => r.status === 200,
    });

    // Test burst handling
    const burstResponse = http.post(
      `${BASE_URL}/api/network/traffic/burst/test`,
      JSON.stringify({
        vmId: `vm-${__VU}`,
        burstSize: '15MB',
        duration: 1000,
      }),
      { headers, tags: { operation: 'burst_test' } }
    );

    check(burstResponse, {
      'burst handled': (r) => r.status === 200,
      'within burst limit': (r) => r.json('droppedPackets') === 0,
    });

    if (burstResponse.status === 200) {
      burstHandlingCapacity.add(burstResponse.json('throughput'));
    }

    // Test shaping accuracy
    const accuracyResponse = http.post(
      `${BASE_URL}/api/network/traffic/shape/verify`,
      JSON.stringify({
        vmId: `vm-${__VU}`,
        expectedRate: '100Mbps',
        tolerance: 5, // 5% tolerance
      }),
      { headers, tags: { operation: 'shaping_verify' } }
    );

    check(accuracyResponse, {
      'shaping accurate': (r) => r.status === 200,
      'within tolerance': (r) => r.json('withinTolerance') === true,
    });

    if (accuracyResponse.status === 200) {
      trafficShapingAccuracy.add(accuracyResponse.json('withinTolerance') ? 1 : 0);
    }

    // Test congestion control
    const congestionResponse = http.post(
      `${BASE_URL}/api/network/traffic/congestion/test`,
      JSON.stringify({
        vmId: `vm-${__VU}`,
        loadLevel: 'high',
        duration: 5000,
      }),
      { headers, tags: { operation: 'congestion_test' } }
    );

    check(congestionResponse, {
      'congestion handled': (r) => r.status === 200,
      'no packet loss': (r) => r.json('packetLoss') < 0.01,
    });

    if (congestionResponse.status === 200) {
      congestionControlEffectiveness.add(congestionResponse.json('packetLoss') < 0.01 ? 1 : 0);
    }
  });
}

// Network Optimization Tests
export function networkOptimization() {
  group('Network Optimization Algorithms', () => {
    // AI-driven bandwidth prediction
    const predictionResponse = http.post(
      `${BASE_URL}/api/network/ai/predict`,
      JSON.stringify({
        vmId: `vm-${__VU}`,
        historicalData: true,
        predictionWindow: '5m',
      }),
      { headers, tags: { operation: 'ai_prediction' } }
    );

    check(predictionResponse, {
      'prediction generated': (r) => r.status === 200,
      'confidence high': (r) => r.json('confidence') > 0.8,
    });

    if (predictionResponse.status === 200) {
      // Validate prediction accuracy after window
      sleep(5);
      const validationResponse = http.get(
        `${BASE_URL}/api/network/ai/validate/${predictionResponse.json('predictionId')}`,
        { headers, tags: { operation: 'prediction_validate' } }
      );

      if (validationResponse.status === 200) {
        aiPredictionAccuracy.add(validationResponse.json('accuracy') > 0.85 ? 1 : 0);
      }
    }

    // Route optimization
    const routeStart = Date.now();
    const routeResponse = http.post(
      `${BASE_URL}/api/network/route/optimize`,
      JSON.stringify({
        source: `vm-${__VU}`,
        destination: `vm-${__VU + 10}`,
        constraints: {
          maxLatency: 50,
          minBandwidth: 100,
          preferredPath: 'shortest',
        },
      }),
      { headers, tags: { operation: 'route_optimize' } }
    );

    check(routeResponse, {
      'route optimized': (r) => r.status === 200,
      'constraints met': (r) => r.json('meetsConstraints') === true,
    });

    routeOptimizationTime.add(Date.now() - routeStart);

    // Adaptive QoS adjustment
    const adaptiveStart = Date.now();
    const adaptiveResponse = http.post(
      `${BASE_URL}/api/network/qos/adaptive`,
      JSON.stringify({
        vmId: `vm-${__VU}`,
        mode: 'auto',
        objectives: {
          minimizeLatency: true,
          maximizeThroughput: true,
          fairness: 0.8,
        },
      }),
      { headers, tags: { operation: 'adaptive_qos' } }
    );

    check(adaptiveResponse, {
      'adaptive QoS enabled': (r) => r.status === 200,
      'objectives achievable': (r) => r.json('feasible') === true,
    });

    adaptiveQosAdjustmentTime.add(Date.now() - adaptiveStart);
  });
}

// Cross-Cluster Bandwidth Management Tests
export function crossClusterBandwidth() {
  group('Cross-Cluster Bandwidth Management', () => {
    const clusters = ['us-east-1', 'eu-west-1', 'ap-southeast-1'];
    const sourceCluster = clusters[__VU % clusters.length];
    const targetCluster = clusters[(__VU + 1) % clusters.length];

    // Allocate cross-cluster bandwidth
    const allocateResponse = http.post(
      `${BASE_URL}/api/network/cross-cluster/allocate`,
      JSON.stringify({
        sourceCluster: sourceCluster,
        targetCluster: targetCluster,
        bandwidth: '1Gbps',
        priority: 'high',
        duration: '1h',
      }),
      { headers, tags: { operation: 'cross_cluster_allocate' } }
    );

    check(allocateResponse, {
      'bandwidth allocated': (r) => r.status === 200,
      'allocation confirmed': (r) => r.json('allocated') === true,
    });

    // Test WAN optimization
    const wanResponse = http.post(
      `${BASE_URL}/api/network/wan/optimize`,
      JSON.stringify({
        sourceCluster: sourceCluster,
        targetCluster: targetCluster,
        techniques: ['compression', 'deduplication', 'caching'],
      }),
      { headers, tags: { operation: 'wan_optimize' } }
    );

    check(wanResponse, {
      'WAN optimization enabled': (r) => r.status === 200,
      'optimization gain': (r) => r.json('optimizationGain') > 1.5,
    });

    if (wanResponse.status === 200) {
      wanOptimizationGain.add(wanResponse.json('optimizationGain'));
    }

    // Test cross-cluster traffic prioritization
    const priorityResponse = http.post(
      `${BASE_URL}/api/network/cross-cluster/prioritize`,
      JSON.stringify({
        flows: [
          { type: 'migration', priority: 1 },
          { type: 'replication', priority: 2 },
          { type: 'backup', priority: 3 },
        ],
      }),
      { headers, tags: { operation: 'traffic_prioritize' } }
    );

    check(priorityResponse, {
      'priorities set': (r) => r.status === 200,
    });

    // Verify bandwidth efficiency
    const efficiencyResponse = http.get(
      `${BASE_URL}/api/network/cross-cluster/efficiency?source=${sourceCluster}&target=${targetCluster}`,
      { headers, tags: { operation: 'bandwidth_efficiency' } }
    );

    check(efficiencyResponse, {
      'efficiency measured': (r) => r.status === 200,
      'efficiency good': (r) => r.json('efficiency') > 0.8,
    });

    if (efficiencyResponse.status === 200) {
      crossClusterBandwidthEfficiency.add(efficiencyResponse.json('efficiency') > 0.8 ? 1 : 0);
    }
  });
}

// Real-time Adaptation Tests
export function realtimeAdaptation() {
  group('Real-time Adaptation Performance', () => {
    const adaptStart = Date.now();

    // Simulate network condition change
    const conditionResponse = http.post(
      `${BASE_URL}/api/network/simulate/condition`,
      JSON.stringify({
        type: 'congestion',
        severity: Math.random() * 0.5 + 0.3, // 30-80% severity
        affectedVMs: [`vm-${__VU}`, `vm-${__VU + 1}`, `vm-${__VU + 2}`],
      }),
      { headers, tags: { operation: 'condition_change' } }
    );

    check(conditionResponse, {
      'condition simulated': (r) => r.status === 200,
    });

    // Check adaptation response
    const adaptationResponse = http.get(
      `${BASE_URL}/api/network/adaptation/status`,
      { headers, tags: { operation: 'adaptation_check' } }
    );

    check(adaptationResponse, {
      'adaptation triggered': (r) => r.status === 200,
      'adaptation active': (r) => r.json('adapting') === true,
    });

    adaptationResponseTime.add(Date.now() - adaptStart);

    // Verify automatic bandwidth reallocation
    sleep(2);

    const reallocationResponse = http.get(
      `${BASE_URL}/api/network/bandwidth/allocation`,
      { headers, tags: { operation: 'reallocation_check' } }
    );

    check(reallocationResponse, {
      'bandwidth reallocated': (r) => r.status === 200,
      'allocation optimal': (r) => r.json('optimality') > 0.9,
    });

    // Test policy adjustment effectiveness
    const policyCheckResponse = http.get(
      `${BASE_URL}/api/network/qos/policy/current`,
      { headers, tags: { operation: 'policy_effectiveness' } }
    );

    check(policyCheckResponse, {
      'policies updated': (r) => r.status === 200,
      'policies effective': (r) => r.json('effectiveness') > 0.85,
    });
  });
}

// Scalability Tests
export function scalabilityTests() {
  group('Bandwidth Optimization Scalability', () => {
    // Test with increasing VM counts
    const vmCount = Math.min(__VU * 10, 500);
    const interfaceCount = vmCount * 2;
    const flowCount = vmCount * 5;

    // Test bandwidth optimization with scale
    const scaleResponse = http.post(
      `${BASE_URL}/api/network/bandwidth/optimize/scale`,
      JSON.stringify({
        vms: vmCount,
        interfaces: interfaceCount,
        flows: flowCount,
        optimizationLevel: 'aggressive',
      }),
      { headers, tags: { operation: 'scale_test' } }
    );

    check(scaleResponse, {
      'scale test completed': (r) => r.status === 200,
      'performance acceptable': (r) => r.json('processingTime') < 5000,
      'optimization effective': (r) => r.json('optimizationRatio') > 1.3,
    });

    // Test monitoring at scale
    const monitorScaleResponse = http.post(
      `${BASE_URL}/api/network/monitor/scale`,
      JSON.stringify({
        targets: vmCount,
        metricsPerTarget: 10,
        collectionInterval: 1000,
      }),
      { headers, tags: { operation: 'monitor_scale' } }
    );

    check(monitorScaleResponse, {
      'monitoring scaled': (r) => r.status === 200,
      'collection efficient': (r) => r.json('collectionLatency') < 100,
      'no data loss': (r) => r.json('dataLoss') === 0,
    });
  });
}

// Main scenario execution
export default function () {
  const scenario = __ENV.SCENARIO || 'all';

  switch (scenario) {
    case 'qos':
      qosPolicyPerformance();
      break;
    case 'monitoring':
      bandwidthMonitoring();
      break;
    case 'shaping':
      trafficShaping();
      break;
    case 'optimization':
      networkOptimization();
      break;
    case 'cross_cluster':
      crossClusterBandwidth();
      break;
    case 'adaptation':
      realtimeAdaptation();
      break;
    case 'scalability':
      scalabilityTests();
      break;
    default:
      // Run all scenarios
      qosPolicyPerformance();
      sleep(1);
      bandwidthMonitoring();
      sleep(1);
      trafficShaping();
      sleep(1);
      networkOptimization();
      sleep(1);
      crossClusterBandwidth();
      sleep(1);
      realtimeAdaptation();
  }

  sleep(1);
}

// Helper functions for safe metric access
function m(data, metric, key, def) {
  const mm = (data.metrics || {})[metric];
  const v = mm && mm.values && mm.values[key];
  return typeof v === 'number' ? v : def;
}

function c(data, metric, key, def) {
  const mm = (data.metrics || {})[metric];
  const v = mm && mm.values && mm.values[key];
  return (typeof v === 'number' ? v : def);
}

// Custom summary generation
export function handleSummary(data) {
  const customData = {
    'Bandwidth Optimization Performance Summary': {
      'QoS Policy': {
        'Application Time': `${m(data, 'qos_policy_application_time', 'avg', 0).toFixed(2)}ms`,
        'Classification Accuracy': `${(c(data, 'traffic_classification_accuracy', 'rate', 0) * 100).toFixed(2)}%`,
        'Rate Limiting': `${(c(data, 'rate_limiting_effectiveness', 'rate', 0) * 100).toFixed(2)}%`,
      },
      'Bandwidth Monitoring': {
        'CPU Overhead': `${c(data, 'bandwidth_monitoring_overhead', 'value', 0).toFixed(2)}%`,
        'Alert Latency': `${m(data, 'alert_generation_latency', 'avg', 0).toFixed(2)}ms`,
      },
      'Traffic Shaping': {
        'Shaping Accuracy': `${(c(data, 'traffic_shaping_accuracy', 'rate', 0) * 100).toFixed(2)}%`,
        'Burst Capacity': `${c(data, 'burst_handling_capacity', 'value', 0).toFixed(2)}Mbps`,
        'Congestion Control': `${(c(data, 'congestion_control_effectiveness', 'rate', 0) * 100).toFixed(2)}%`,
      },
      'Network Optimization': {
        'AI Prediction': `${(c(data, 'ai_prediction_accuracy', 'rate', 0) * 100).toFixed(2)}%`,
        'Route Optimization': `${m(data, 'route_optimization_time', 'avg', 0).toFixed(2)}ms`,
        'Adaptive QoS': `${m(data, 'adaptive_qos_adjustment_time', 'avg', 0).toFixed(2)}ms`,
      },
      'Cross-Cluster': {
        'Bandwidth Efficiency': `${(c(data, 'cross_cluster_bandwidth_efficiency', 'rate', 0) * 100).toFixed(2)}%`,
        'WAN Optimization Gain': `${c(data, 'wan_optimization_gain', 'value', 0).toFixed(2)}x`,
      },
      'Real-time Adaptation': {
        'Response Time': `${m(data, 'adaptation_response_time', 'avg', 0).toFixed(2)}ms`,
        'P95 Response': `${m(data, 'adaptation_response_time', 'p(95)', 0).toFixed(2)}ms`,
      },
    },
  };

  return {
    'stdout': textSummary(data, { indent: ' ', enableColors: true }) +
              '\n' + JSON.stringify(customData, null, 2),
    'summary.html': htmlReport(data),
    'summary.json': JSON.stringify(data),
  };
}