import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter, Gauge } from 'k6/metrics';
import { htmlReport } from "https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js";
import { textSummary } from "https://jslib.k6.io/k6-summary/0.0.1/index.js";

// Custom metrics for P2P fabric performance
const peerDiscoveryTime = new Trend('peer_discovery_time', true);
const dhtLookupTime = new Trend('dht_lookup_time', true);
const natTraversalSuccess = new Rate('nat_traversal_success');
const connectionEstablishmentTime = new Trend('connection_establishment_time', true);
const gossipPropagationTime = new Trend('gossip_propagation_time', true);
const networkFormationTime = new Trend('network_formation_time', true);
const peerChurnHandling = new Trend('peer_churn_handling_time', true);
const crossClusterLatency = new Trend('cross_cluster_latency', true);
const bandwidthUtilization = new Gauge('bandwidth_utilization');
const activeConnections = new Gauge('active_connections');
const failedConnections = new Counter('failed_connections');

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8080';
const API_TOKEN = __ENV.API_TOKEN || '';

export const options = {
  scenarios: {
    // P2P Network Formation Benchmark
    network_formation: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '30s', target: 10 },  // Warm-up with 10 peers
        { duration: '1m', target: 50 },   // Scale to 50 peers
        { duration: '2m', target: 100 },  // Scale to 100 peers
        { duration: '3m', target: 500 },  // Scale to 500 peers
        { duration: '2m', target: 100 },  // Scale down
        { duration: '1m', target: 0 },    // Cool down
      ],
      gracefulRampDown: '30s',
      tags: { scenario: 'network_formation' },
      exec: 'networkFormation',
    },
    // DHT Performance Benchmark
    dht_performance: {
      executor: 'constant-arrival-rate',
      rate: 100,
      timeUnit: '1s',
      duration: '5m',
      preAllocatedVUs: 50,
      maxVUs: 200,
      tags: { scenario: 'dht_performance' },
      exec: 'gossipEfficiency',
    },
    // NAT Traversal Benchmark
    nat_traversal: {
      executor: 'per-vu-iterations',
      vus: 100,
      iterations: 10,
      maxDuration: '10m',
      tags: { scenario: 'nat_traversal' },
      exec: 'natTraversal',
    },
    // Gossip Protocol Efficiency
    gossip_efficiency: {
      executor: 'shared-iterations',
      vus: 50,
      iterations: 500,
      maxDuration: '15m',
      tags: { scenario: 'gossip_efficiency' },
      exec: 'gossipEfficiency',
    },
    // Peer Churn Resilience
    peer_churn: {
      executor: 'ramping-arrival-rate',
      startRate: 10,
      timeUnit: '1s',
      preAllocatedVUs: 50,
      stages: [
        { duration: '1m', target: 10 },
        { duration: '2m', target: 50 },  // High churn rate
        { duration: '2m', target: 100 }, // Very high churn
        { duration: '1m', target: 10 },
      ],
      tags: { scenario: 'peer_churn' },
      exec: 'peerChurn',
    },
  },
  thresholds: {
    'peer_discovery_time': ['p(95)<5000', 'p(99)<10000'],
    'dht_lookup_time': ['p(95)<1000', 'p(99)<2000'],
    'nat_traversal_success': ['rate>0.95'],
    'connection_establishment_time': ['p(95)<3000', 'p(99)<5000'],
    'gossip_propagation_time': ['p(95)<500', 'p(99)<1000'],
    'network_formation_time': ['p(95)<30000', 'p(99)<60000'],
    'peer_churn_handling_time': ['p(95)<2000', 'p(99)<5000'],
    'cross_cluster_latency': ['p(95)<100', 'p(99)<200'],
    'http_req_duration': ['p(95)<500', 'p(99)<1000'],
    'http_req_failed': ['rate<0.05'],
  },
};

const headers = {
  'Content-Type': 'application/json',
  'Authorization': `Bearer ${API_TOKEN}`,
};

// P2P Network Formation Tests
export function networkFormation() {
  group('P2P Network Formation', () => {
    const startTime = Date.now();

    // Initialize peer
    const initResponse = http.post(
      `${BASE_URL}/api/p2p/peer/init`,
      JSON.stringify({
        nodeId: `peer-${__VU}-${__ITER}`,
        capabilities: ['compute', 'storage', 'network'],
        region: `region-${Math.floor(__VU / 100)}`,
      }),
      { headers, tags: { operation: 'peer_init' } }
    );

    check(initResponse, {
      'peer initialization successful': (r) => r.status === 200,
    });

    if (initResponse.status !== 200) {
      failedConnections.add(1);
      return;
    }

    const peerId = initResponse.json('peerId');

    // Discover peers
    const discoverStart = Date.now();
    const discoverResponse = http.get(
      `${BASE_URL}/api/p2p/peer/${peerId}/discover`,
      { headers, tags: { operation: 'peer_discovery' } }
    );

    check(discoverResponse, {
      'peer discovery successful': (r) => r.status === 200,
      'discovered peers': (r) => r.json('peers') && r.json('peers').length > 0,
    });

    peerDiscoveryTime.add(Date.now() - discoverStart);

    // DHT lookup
    const dhtStart = Date.now();
    const dhtResponse = http.get(
      `${BASE_URL}/api/p2p/dht/lookup/${peerId}`,
      { headers, tags: { operation: 'dht_lookup' } }
    );

    check(dhtResponse, {
      'DHT lookup successful': (r) => r.status === 200,
    });

    dhtLookupTime.add(Date.now() - dhtStart);

    // Establish connections
    const peers = discoverResponse.json('peers') || [];
    let successfulConnections = 0;

    peers.slice(0, 5).forEach((targetPeer) => {
      const connectStart = Date.now();
      const connectResponse = http.post(
        `${BASE_URL}/api/p2p/peer/${peerId}/connect`,
        JSON.stringify({ targetPeerId: targetPeer.id }),
        { headers, tags: { operation: 'peer_connect' } }
      );

      if (connectResponse.status === 200) {
        successfulConnections++;
        connectionEstablishmentTime.add(Date.now() - connectStart);
      } else {
        failedConnections.add(1);
      }
    });

    activeConnections.add(successfulConnections);
    networkFormationTime.add(Date.now() - startTime);
  });
}

// NAT Traversal Tests
export function natTraversal() {
  group('NAT Traversal', () => {
    const peerId = `nat-peer-${__VU}-${__ITER}`;

    // STUN server interaction
    const stunResponse = http.post(
      `${BASE_URL}/api/p2p/nat/stun`,
      JSON.stringify({
        peerId: peerId,
        localAddress: `192.168.${__VU}.${__ITER}`,
      }),
      { headers, tags: { operation: 'stun_request' } }
    );

    check(stunResponse, {
      'STUN successful': (r) => r.status === 200,
      'public address obtained': (r) => r.json('publicAddress') !== null,
    });

    // TURN relay if needed
    if (stunResponse.json('natType') === 'symmetric') {
      const turnResponse = http.post(
        `${BASE_URL}/api/p2p/nat/turn`,
        JSON.stringify({
          peerId: peerId,
          relayRequired: true,
        }),
        { headers, tags: { operation: 'turn_request' } }
      );

      check(turnResponse, {
        'TURN relay established': (r) => r.status === 200,
      });
    }

    // Hole punching attempt
    const holePunchResponse = http.post(
      `${BASE_URL}/api/p2p/nat/holepunch`,
      JSON.stringify({
        sourcePeer: peerId,
        targetPeer: `peer-target-${__ITER}`,
      }),
      { headers, tags: { operation: 'hole_punching' } }
    );

    const success = holePunchResponse.status === 200;
    natTraversalSuccess.add(success ? 1 : 0);

    check(holePunchResponse, {
      'hole punching successful': (r) => r.status === 200,
    });
  });
}

// Gossip Protocol Tests
export function gossipEfficiency() {
  group('Gossip Protocol Efficiency', () => {
    const message = {
      type: 'state_update',
      payload: {
        timestamp: Date.now(),
        data: `message-${__VU}-${__ITER}`,
        ttl: 5,
      },
    };

    const propagateStart = Date.now();

    // Initiate gossip
    const gossipResponse = http.post(
      `${BASE_URL}/api/p2p/gossip/broadcast`,
      JSON.stringify(message),
      { headers, tags: { operation: 'gossip_broadcast' } }
    );

    check(gossipResponse, {
      'gossip initiated': (r) => r.status === 200,
    });

    // Check propagation
    sleep(1);

    const verifyResponse = http.get(
      `${BASE_URL}/api/p2p/gossip/verify/${message.payload.data}`,
      { headers, tags: { operation: 'gossip_verify' } }
    );

    check(verifyResponse, {
      'message propagated': (r) => r.status === 200,
      'coverage complete': (r) => r.json('coverage') > 0.95,
    });

    gossipPropagationTime.add(Date.now() - propagateStart);
  });
}

// Peer Churn Handling Tests
export function peerChurn() {
  group('Peer Churn Resilience', () => {
    const churnStart = Date.now();

    // Simulate peer leaving
    if (Math.random() < 0.3) {
      const leaveResponse = http.post(
        `${BASE_URL}/api/p2p/peer/leave`,
        JSON.stringify({
          peerId: `churn-peer-${__VU}`,
          graceful: Math.random() > 0.5,
        }),
        { headers, tags: { operation: 'peer_leave' } }
      );

      check(leaveResponse, {
        'peer leave handled': (r) => r.status === 200,
      });
    }

    // Simulate peer joining
    const joinResponse = http.post(
      `${BASE_URL}/api/p2p/peer/join`,
      JSON.stringify({
        peerId: `churn-peer-new-${__VU}-${__ITER}`,
        bootstrap: ['bootstrap1.novacron.io', 'bootstrap2.novacron.io'],
      }),
      { headers, tags: { operation: 'peer_join' } }
    );

    check(joinResponse, {
      'peer join successful': (r) => r.status === 200,
      'network stable': (r) => r.json('networkStable') === true,
    });

    peerChurnHandling.add(Date.now() - churnStart);
  });
}

// Cross-Cluster Communication Tests
export function crossClusterCommunication() {
  group('Cross-Cluster Communication', () => {
    const clusters = ['us-east', 'eu-west', 'ap-south'];
    const sourceCluster = clusters[__VU % clusters.length];
    const targetCluster = clusters[(__VU + 1) % clusters.length];

    const messageStart = Date.now();

    const response = http.post(
      `${BASE_URL}/api/p2p/cluster/message`,
      JSON.stringify({
        sourceCluster: sourceCluster,
        targetCluster: targetCluster,
        message: {
          type: 'federation_sync',
          payload: { data: `cross-cluster-${__ITER}` },
        },
      }),
      { headers, tags: { operation: 'cross_cluster_message' } }
    );

    check(response, {
      'cross-cluster message sent': (r) => r.status === 200,
      'latency acceptable': (r) => r.json('latency') < 200,
    });

    if (response.status === 200) {
      crossClusterLatency.add(response.json('latency'));
    }
  });
}

// Network Fabric Scalability Tests
export function networkScalability() {
  group('Network Fabric Scalability', () => {
    // Test with increasing peer counts
    const peerCount = Math.min(__VU * 10, 1000);

    const scaleResponse = http.get(
      `${BASE_URL}/api/p2p/fabric/stats?peers=${peerCount}`,
      { headers, tags: { operation: 'fabric_stats' } }
    );

    check(scaleResponse, {
      'fabric stats retrieved': (r) => r.status === 200,
      'peer count correct': (r) => r.json('activePeers') >= peerCount * 0.95,
      'bandwidth efficient': (r) => r.json('bandwidthEfficiency') > 0.8,
    });

    if (scaleResponse.status === 200) {
      bandwidthUtilization.add(scaleResponse.json('bandwidthUtilization'));
      activeConnections.add(scaleResponse.json('activeConnections'));
    }
  });
}

// Failure Scenario Tests
export function failureScenarios() {
  group('Failure Scenarios', () => {
    const failureTypes = ['network_partition', 'peer_failure', 'bandwidth_limit', 'packet_loss'];
    const failureType = failureTypes[__ITER % failureTypes.length];

    // Inject failure
    const injectResponse = http.post(
      `${BASE_URL}/api/p2p/chaos/inject`,
      JSON.stringify({
        type: failureType,
        duration: 10000, // 10 seconds
        severity: Math.random() * 0.5 + 0.3, // 30-80% severity
      }),
      { headers, tags: { operation: 'chaos_inject' } }
    );

    check(injectResponse, {
      'failure injected': (r) => r.status === 200,
    });

    sleep(5);

    // Check recovery
    const recoveryResponse = http.get(
      `${BASE_URL}/api/p2p/fabric/health`,
      { headers, tags: { operation: 'health_check' } }
    );

    check(recoveryResponse, {
      'fabric recovering': (r) => r.status === 200,
      'connections restoring': (r) => r.json('status') !== 'critical',
    });

    sleep(10);

    // Verify full recovery
    const verifyResponse = http.get(
      `${BASE_URL}/api/p2p/fabric/health`,
      { headers, tags: { operation: 'recovery_verify' } }
    );

    check(verifyResponse, {
      'fabric recovered': (r) => r.status === 200 && r.json('status') === 'healthy',
      'all peers reconnected': (r) => r.json('disconnectedPeers') === 0,
    });
  });
}

// Main scenario execution
export default function () {
  const scenario = __ENV.SCENARIO || 'all';

  switch (scenario) {
    case 'network_formation':
      networkFormation();
      break;
    case 'nat_traversal':
      natTraversal();
      break;
    case 'gossip':
      gossipEfficiency();
      break;
    case 'churn':
      peerChurn();
      break;
    case 'cross_cluster':
      crossClusterCommunication();
      break;
    case 'scalability':
      networkScalability();
      break;
    case 'failure':
      failureScenarios();
      break;
    default:
      // Run all scenarios
      networkFormation();
      sleep(1);
      natTraversal();
      sleep(1);
      gossipEfficiency();
      sleep(1);
      crossClusterCommunication();
      sleep(1);
      networkScalability();
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
    'P2P Fabric Performance Summary': {
      'Network Formation': {
        'Average Time': `${m(data, 'network_formation_time', 'avg', 0).toFixed(2)}ms`,
        'P95 Time': `${m(data, 'network_formation_time', 'p(95)', 0).toFixed(2)}ms`,
      },
      'Peer Discovery': {
        'Average Time': `${m(data, 'peer_discovery_time', 'avg', 0).toFixed(2)}ms`,
        'P95 Time': `${m(data, 'peer_discovery_time', 'p(95)', 0).toFixed(2)}ms`,
      },
      'NAT Traversal': {
        'Success Rate': `${(c(data, 'nat_traversal_success', 'rate', 0) * 100).toFixed(2)}%`,
        'Failed Connections': c(data, 'failed_connections', 'count', 0),
      },
      'DHT Performance': {
        'Average Lookup': `${m(data, 'dht_lookup_time', 'avg', 0).toFixed(2)}ms`,
        'P99 Lookup': `${m(data, 'dht_lookup_time', 'p(99)', 0).toFixed(2)}ms`,
      },
      'Gossip Protocol': {
        'Average Propagation': `${m(data, 'gossip_propagation_time', 'avg', 0).toFixed(2)}ms`,
        'P95 Propagation': `${m(data, 'gossip_propagation_time', 'p(95)', 0).toFixed(2)}ms`,
      },
      'Cross-Cluster': {
        'Average Latency': `${m(data, 'cross_cluster_latency', 'avg', 0).toFixed(2)}ms`,
        'P99 Latency': `${m(data, 'cross_cluster_latency', 'p(99)', 0).toFixed(2)}ms`,
      },
      'Network Utilization': {
        'Bandwidth Usage': `${c(data, 'bandwidth_utilization', 'value', 0).toFixed(2)}%`,
        'Active Connections': c(data, 'active_connections', 'value', 0),
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