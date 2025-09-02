// NovaCron Load Testing Configuration
export const config = {
  // Environment settings
  environments: {
    local: {
      baseURL: 'http://localhost:8080',
      wsURL: 'ws://localhost:8080',
      dbHost: 'localhost:5432'
    },
    staging: {
      baseURL: 'https://staging.novacron.com',
      wsURL: 'wss://staging.novacron.com',
      dbHost: 'staging-db.novacron.com:5432'
    },
    production: {
      baseURL: 'https://api.novacron.com',
      wsURL: 'wss://api.novacron.com',
      dbHost: 'prod-db.novacron.com:5432'
    }
  },

  // Load test scenarios
  scenarios: {
    // API endpoint load testing
    api_load: {
      executor: 'constant-vus',
      vus: 1000,
      duration: '10m',
      gracefulRampDown: '30s'
    },
    
    // VM management load testing
    vm_management: {
      executor: 'ramping-vus',
      stages: [
        { duration: '2m', target: 100 },
        { duration: '5m', target: 500 },
        { duration: '10m', target: 1000 },
        { duration: '2m', target: 0 }
      ]
    },
    
    // WebSocket stress testing
    websocket_stress: {
      executor: 'constant-vus',
      vus: 2000,
      duration: '15m',
      gracefulRampDown: '1m'
    },
    
    // Database performance testing
    database_performance: {
      executor: 'ramping-arrival-rate',
      startRate: 10,
      timeUnit: '1s',
      preAllocatedVUs: 50,
      maxVUs: 500,
      stages: [
        { duration: '2m', target: 50 },
        { duration: '5m', target: 200 },
        { duration: '10m', target: 500 },
        { duration: '2m', target: 0 }
      ]
    },
    
    // Federation load testing
    federation_load: {
      executor: 'shared-iterations',
      vus: 100,
      iterations: 10000,
      maxDuration: '30m'
    },
    
    // Stress testing
    stress_test: {
      executor: 'ramping-vus',
      stages: [
        { duration: '5m', target: 500 },
        { duration: '10m', target: 1500 },
        { duration: '20m', target: 3000 },
        { duration: '10m', target: 5000 },
        { duration: '5m', target: 0 }
      ]
    },
    
    // Soak testing
    soak_test: {
      executor: 'constant-vus',
      vus: 200,
      duration: '2h'
    }
  },

  // Performance thresholds
  thresholds: {
    // Response time thresholds
    http_req_duration: ['p(95)<500', 'p(99)<1000'],
    http_req_failed: ['rate<0.01'],
    
    // WebSocket thresholds
    ws_connecting_duration: ['p(95)<1000'],
    ws_session_duration: ['p(95)<30000'],
    
    // VM operation thresholds
    vm_creation_duration: ['p(95)<30000'],
    vm_start_duration: ['p(95)<10000'],
    vm_stop_duration: ['p(95)<5000'],
    
    // Database thresholds
    db_query_duration: ['p(95)<100', 'p(99)<500'],
    db_connection_duration: ['p(95)<1000'],
    
    // Federation thresholds
    federation_sync_duration: ['p(95)<2000'],
    federation_health_check: ['p(95)<500']
  },

  // Test data configuration
  testData: {
    // VM configurations for testing
    vmConfigs: [
      { cpu: 1, memory: 1024, disk: 10, name: 'test-vm-small' },
      { cpu: 2, memory: 2048, disk: 20, name: 'test-vm-medium' },
      { cpu: 4, memory: 4096, disk: 40, name: 'test-vm-large' },
      { cpu: 8, memory: 8192, disk: 80, name: 'test-vm-xlarge' }
    ],
    
    // User authentication data
    users: {
      viewer: { username: 'viewer', password: 'viewer123', role: 'viewer' },
      operator: { username: 'operator', password: 'operator123', role: 'operator' },
      admin: { username: 'admin', password: 'admin123', role: 'admin' }
    },
    
    // Federation nodes
    federationNodes: [
      { id: 'node-us-east', region: 'us-east-1', url: 'https://us-east.novacron.com' },
      { id: 'node-us-west', region: 'us-west-1', url: 'https://us-west.novacron.com' },
      { id: 'node-eu-central', region: 'eu-central-1', url: 'https://eu-central.novacron.com' }
    ]
  },

  // Monitoring configuration
  monitoring: {
    prometheus: {
      url: 'http://localhost:9090',
      queryInterval: '30s'
    },
    grafana: {
      url: 'http://localhost:3000',
      dashboardId: 'novacron-load-test'
    },
    influxdb: {
      url: 'http://localhost:8086',
      database: 'k6',
      measurement: 'novacron_load_test'
    }
  },

  // Report configuration
  reporting: {
    outputDir: './reports',
    formats: ['html', 'json', 'csv'],
    includeGraphs: true,
    includeRawData: false
  }
};

// Environment selection helper
export function getEnvironment() {
  const env = __ENV.ENVIRONMENT || 'local';
  return config.environments[env] || config.environments.local;
}

// Load scenario by name
export function getScenario(name) {
  return config.scenarios[name];
}

// Get test data
export function getTestData() {
  return config.testData;
}

// Get monitoring config
export function getMonitoringConfig() {
  return config.monitoring;
}