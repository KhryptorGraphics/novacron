import ws from 'k6/ws';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { randomString } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Custom metrics for WebSocket testing
const wsErrorRate = new Rate('ws_errors');
const messageLatency = new Trend('ws_message_latency');
const connectionTime = new Trend('ws_connection_time');
const messagesReceived = new Counter('ws_messages_received');
const messagesSent = new Counter('ws_messages_sent');
const connectionsEstablished = new Counter('ws_connections_established');
const connectionsFailed = new Counter('ws_connections_failed');

export const options = {
  stages: [
    { duration: '2m', target: 100 },    // 100 concurrent connections
    { duration: '5m', target: 1000 },   // Ramp to 1K connections
    { duration: '5m', target: 1000 },   // Sustain 1K
    { duration: '2m', target: 10000 },  // Spike to 10K
    { duration: '5m', target: 10000 },  // Sustain 10K
    { duration: '2m', target: 100000 }, // Extreme stress: 100K connections
    { duration: '3m', target: 100000 }, // Sustain stress
    { duration: '2m', target: 0 },      // Ramp down
  ],
  thresholds: {
    ws_errors: ['rate<0.05'],                    // <5% error rate
    ws_message_latency: ['p(95)<500'],           // 95% < 500ms
    ws_connection_time: ['p(95)<2000'],          // Connection < 2s
    ws_connections_established: ['count>0'],      // Some connections succeed
  },
};

const WS_URL = __ENV.WS_URL || 'ws://localhost:8080/ws';
const API_TOKEN = __ENV.API_TOKEN || 'test-token-123';

export function setup() {
  console.log('Setting up WebSocket load test...');
  return { startTime: Date.now() };
}

export default function (data) {
  const url = `${WS_URL}/realtime?token=${API_TOKEN}`;
  const params = {
    headers: {
      'Authorization': `Bearer ${API_TOKEN}`,
    },
    tags: {
      vu: __VU,
    },
  };

  const connectionStart = Date.now();

  const response = ws.connect(url, params, function (socket) {
    const connectionDuration = Date.now() - connectionStart;
    connectionTime.add(connectionDuration);

    socket.on('open', () => {
      connectionsEstablished.add(1);

      // Subscribe to various event channels
      const subscriptions = [
        { action: 'subscribe', channel: 'vm_status' },
        { action: 'subscribe', channel: 'system_metrics' },
        { action: 'subscribe', channel: 'alerts' },
      ];

      subscriptions.forEach(sub => {
        socket.send(JSON.stringify(sub));
        messagesSent.add(1);
      });
    });

    socket.on('message', (message) => {
      const receiveTime = Date.now();
      messagesReceived.add(1);

      try {
        const data = JSON.parse(message);

        const msgCheck = check(data, {
          'Message has type': (d) => d.type !== undefined,
          'Message has channel': (d) => d.channel !== undefined,
          'Message has timestamp': (d) => d.timestamp !== undefined,
          'Message timestamp is recent': (d) => {
            if (d.timestamp) {
              const latency = receiveTime - new Date(d.timestamp).getTime();
              messageLatency.add(Math.abs(latency));
              return latency < 5000; // Within 5 seconds
            }
            return false;
          },
        });

        if (!msgCheck) {
          wsErrorRate.add(1);
        }

      } catch (e) {
        wsErrorRate.add(1);
        console.log('Failed to parse message:', e);
      }
    });

    socket.on('error', (e) => {
      wsErrorRate.add(1);
      connectionsFailed.add(1);
      console.log('WebSocket error:', e);
    });

    socket.on('close', () => {
      // Connection closed normally
    });

    // Send periodic heartbeat/ping messages
    socket.setInterval(() => {
      const ping = JSON.stringify({
        type: 'ping',
        timestamp: Date.now(),
        client_id: `client-${__VU}-${randomString(8)}`,
      });
      socket.send(ping);
      messagesSent.add(1);
    }, 5000); // Every 5 seconds

    // Send test data to simulate real-world usage
    socket.setInterval(() => {
      const testMessages = [
        {
          type: 'vm_query',
          action: 'get_status',
          vm_id: `vm-${Math.floor(Math.random() * 1000)}`,
        },
        {
          type: 'metric_query',
          action: 'get_metrics',
          resource: 'cpu',
        },
        {
          type: 'event_subscribe',
          filters: { severity: 'high' },
        },
      ];

      const msg = testMessages[Math.floor(Math.random() * testMessages.length)];
      socket.send(JSON.stringify(msg));
      messagesSent.add(1);
    }, 10000); // Every 10 seconds

    // Keep connection alive for 30-60 seconds
    const connectionDuration = Math.floor(Math.random() * 30000) + 30000;
    socket.setTimeout(() => {
      // Send disconnect message
      socket.send(JSON.stringify({ type: 'disconnect' }));
      messagesSent.add(1);

      // Close connection gracefully
      socket.close();
    }, connectionDuration);
  });

  check(response, {
    'WebSocket connection established': (r) => r && r.status === 101,
  }) || wsErrorRate.add(1);

  sleep(1);
}

export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  console.log(`WebSocket load test completed in ${duration}s`);
}

export function handleSummary(data) {
  return {
    'stdout': textSummary(data),
    'results_websocket.json': JSON.stringify(data, null, 2),
  };
}

function textSummary(data) {
  let summary = '\n=== WebSocket Load Test Summary ===\n\n';

  summary += 'WebSocket Metrics:\n';
  summary += `  Connections Established: ${data.metrics.ws_connections_established ? data.metrics.ws_connections_established.values.count : 0}\n`;
  summary += `  Connections Failed: ${data.metrics.ws_connections_failed ? data.metrics.ws_connections_failed.values.count : 0}\n`;
  summary += `  Messages Sent: ${data.metrics.ws_messages_sent ? data.metrics.ws_messages_sent.values.count : 0}\n`;
  summary += `  Messages Received: ${data.metrics.ws_messages_received ? data.metrics.ws_messages_received.values.count : 0}\n`;

  if (data.metrics.ws_connection_time) {
    summary += `  Connection Time (p95): ${data.metrics.ws_connection_time.values['p(95)'].toFixed(2)}ms\n`;
    summary += `  Connection Time (avg): ${data.metrics.ws_connection_time.values.avg.toFixed(2)}ms\n`;
  }

  if (data.metrics.ws_message_latency) {
    summary += `  Message Latency (p95): ${data.metrics.ws_message_latency.values['p(95)'].toFixed(2)}ms\n`;
    summary += `  Message Latency (avg): ${data.metrics.ws_message_latency.values.avg.toFixed(2)}ms\n`;
  }

  summary += `  Error Rate: ${(data.metrics.ws_errors.values.rate * 100).toFixed(2)}%\n\n`;

  summary += 'Connection Stats:\n';
  if (data.metrics.ws_connections_established && data.metrics.ws_connections_failed) {
    const total = data.metrics.ws_connections_established.values.count + data.metrics.ws_connections_failed.values.count;
    const successRate = total > 0 ? (data.metrics.ws_connections_established.values.count / total * 100) : 0;
    summary += `  Success Rate: ${successRate.toFixed(2)}%\n`;
  }

  if (data.metrics.ws_messages_received && data.metrics.ws_messages_sent) {
    const sent = data.metrics.ws_messages_sent.values.count;
    const received = data.metrics.ws_messages_received.values.count;
    summary += `  Message Throughput: ${sent} sent, ${received} received\n`;
  }

  return summary;
}
