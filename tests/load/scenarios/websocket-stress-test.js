import ws from 'k6/ws';
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter, Gauge } from 'k6/metrics';
import { config, getEnvironment, getTestData } from '../configs/test-config.js';

// Custom metrics for WebSocket testing
const wsConnectionTime = new Trend('ws_connecting_duration');
const wsSessionDuration = new Trend('ws_session_duration');
const wsMessagesSent = new Counter('ws_messages_sent');
const wsMessagesReceived = new Counter('ws_messages_received');
const wsConnectionRate = new Rate('ws_connection_success_rate');
const wsActiveConnections = new Gauge('ws_active_connections');
const wsReconnections = new Counter('ws_reconnections');

// Test configuration
export const options = {
  scenarios: {
    websocket_stress_test: config.scenarios.websocket_stress,
  },
  thresholds: {
    'ws_connecting_duration': config.thresholds.ws_connecting_duration,
    'ws_session_duration': config.thresholds.ws_session_duration,
    'ws_connection_success_rate': ['rate>0.95'],
    'ws_messages_sent': ['count>1000'],
    'ws_messages_received': ['count>1000']
  }
};

const environment = getEnvironment();
const testData = getTestData();

// Authentication helper
function authenticate() {
  const user = testData.users.viewer; // Use viewer role for WebSocket testing
  const loginPayload = {
    username: user.username,
    password: user.password
  };

  const response = http.post(`${environment.baseURL}/api/auth/login`, 
    JSON.stringify(loginPayload), {
      headers: { 'Content-Type': 'application/json' }
    }
  );

  if (response.status === 200) {
    return response.json().token;
  }
  return null;
}

// WebSocket connection helper
function connectWebSocket(endpoint, token, params = {}) {
  const queryString = new URLSearchParams(params).toString();
  const wsUrl = `${environment.wsURL}${endpoint}${queryString ? '?' + queryString : ''}`;
  
  const connectStart = Date.now();
  const connectionHeaders = token ? { 'Authorization': `Bearer ${token}` } : {};
  
  const response = ws.connect(wsUrl, connectionHeaders, function (socket) {
    const connectDuration = Date.now() - connectStart;
    wsConnectionTime.add(connectDuration);
    wsActiveConnections.add(1);
    
    const sessionStart = Date.now();
    let messageCount = 0;
    let reconnectAttempts = 0;
    
    socket.on('open', function () {
      console.log(`WebSocket connected to ${endpoint}`);
      wsConnectionRate.add(1);
    });

    socket.on('message', function (data) {
      try {
        const message = JSON.parse(data);
        wsMessagesReceived.add(1);
        messageCount++;
        
        // Handle different message types
        switch (message.type) {
          case 'metrics_update':
            // Validate metrics structure
            check(message, {
              'metrics message valid': (m) => m.metrics !== undefined,
              'metrics timestamp present': (m) => m.timestamp !== undefined
            });
            break;
            
          case 'alert':
            // Validate alert structure
            check(message, {
              'alert message valid': (m) => m.alert_id !== undefined,
              'alert severity present': (m) => m.severity !== undefined
            });
            break;
            
          case 'log_entry':
            // Validate log structure
            check(message, {
              'log message valid': (m) => m.message !== undefined,
              'log level present': (m) => m.level !== undefined
            });
            break;
            
          case 'console_output':
            // Validate console structure
            check(message, {
              'console message valid': (m) => m.data !== undefined,
              'VM ID present': (m) => m.vm_id !== undefined
            });
            break;
        }
      } catch (error) {
        console.error(`Failed to parse WebSocket message: ${error}`);
      }
    });

    socket.on('error', function (e) {
      console.error(`WebSocket error on ${endpoint}: ${e.error()}`);
      wsConnectionRate.add(0);
      reconnectAttempts++;
      wsReconnections.add(1);
    });

    socket.on('close', function () {
      const sessionDuration = Date.now() - sessionStart;
      wsSessionDuration.add(sessionDuration);
      wsActiveConnections.add(-1);
      
      console.log(`WebSocket ${endpoint} closed after ${sessionDuration}ms, ${messageCount} messages, ${reconnectAttempts} reconnects`);
    });

    // Send periodic messages to keep connection active
    const interval = setInterval(function () {
      if (socket.readyState === 1) { // WebSocket.OPEN
        const testMessage = {
          type: 'ping',
          timestamp: new Date().toISOString(),
          vu: __VU,
          iteration: __ITER
        };
        
        socket.send(JSON.stringify(testMessage));
        wsMessagesSent.add(1);
      }
    }, 5000); // Send ping every 5 seconds

    // Test different filter updates
    setTimeout(function () {
      if (socket.readyState === 1) {
        const filterUpdate = {
          type: 'update_filters',
          filters: {
            severity: ['warning', 'error'],
            sources: ['vm', 'system']
          },
          timestamp: new Date().toISOString()
        };
        
        socket.send(JSON.stringify(filterUpdate));
        wsMessagesSent.add(1);
      }
    }, 10000);

    // Clean up interval when test ends
    socket.setTimeout(function () {
      clearInterval(interval);
      socket.close();
    }, 60000 + Math.random() * 30000); // 60-90 seconds
  });

  check(response, {
    'WebSocket connection established': (r) => r && r.status === 101
  });

  return response;
}

// Main test function
export default function() {
  const token = authenticate();
  if (!token) {
    console.error('Authentication failed for WebSocket test');
    return;
  }

  // Test different WebSocket endpoints
  const scenarios = [
    {
      endpoint: '/ws/metrics',
      params: { sources: 'cpu,memory,disk', interval: '5' },
      weight: 0.4 // 40% of connections
    },
    {
      endpoint: '/ws/alerts', 
      params: { severity: 'warning,error,critical' },
      weight: 0.3 // 30% of connections
    },
    {
      endpoint: '/ws/logs',
      params: { level: 'info,warning,error', components: 'vm,api' },
      weight: 0.2 // 20% of connections
    },
    {
      endpoint: '/ws/logs/vm',
      params: { level: 'error,warning' },
      weight: 0.1 // 10% of connections
    }
  ];

  // Select scenario based on weights
  const random = Math.random();
  let cumulativeWeight = 0;
  let selectedScenario = scenarios[0];

  for (const scenario of scenarios) {
    cumulativeWeight += scenario.weight;
    if (random <= cumulativeWeight) {
      selectedScenario = scenario;
      break;
    }
  }

  // Connect to selected WebSocket endpoint
  try {
    connectWebSocket(selectedScenario.endpoint, token, selectedScenario.params);
  } catch (error) {
    console.error(`WebSocket connection failed: ${error}`);
    wsConnectionRate.add(0);
  }

  // Simulate user behavior - keep connection alive
  sleep(Math.random() * 10 + 30); // 30-40 seconds
}

// Setup function
export function setup() {
  console.log(`Starting WebSocket stress test against: ${environment.wsURL}`);
  console.log(`Target: 2000+ concurrent WebSocket connections`);
  
  // Verify WebSocket endpoints are accessible
  const user = testData.users.viewer;
  const loginPayload = {
    username: user.username,
    password: user.password
  };

  const authResponse = http.post(`${environment.baseURL}/api/auth/login`, 
    JSON.stringify(loginPayload), {
      headers: { 'Content-Type': 'application/json' }
    }
  );

  if (authResponse.status !== 200) {
    throw new Error(`Authentication failed: ${authResponse.status}`);
  }

  // Test basic WebSocket connectivity
  const token = authResponse.json().token;
  const testWs = ws.connect(`${environment.wsURL}/ws/metrics?interval=30`, {
    'Authorization': `Bearer ${token}`
  }, function (socket) {
    socket.setTimeout(function () {
      socket.close();
    }, 5000);
  });

  check(testWs, {
    'WebSocket test connection successful': (r) => r.status === 101
  });
  
  return { 
    startTime: Date.now(),
    token: token
  };
}

// Teardown function
export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  console.log(`WebSocket stress test completed in ${duration} seconds`);
  console.log(`Total messages sent: ${wsMessagesSent.count}`);
  console.log(`Total messages received: ${wsMessagesReceived.count}`);
  console.log(`Total reconnections: ${wsReconnections.count}`);
}