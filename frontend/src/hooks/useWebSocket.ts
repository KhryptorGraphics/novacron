import { useEffect, useState, useCallback, useMemo, useRef } from 'react';
import { buildWebSocketUrls } from '@/lib/api/origin';

interface WebSocketOptions {
  enabled?: boolean;
  reconnect?: boolean;
  reconnectInterval?: number;
  reconnectAttempts?: number;
  heartbeat?: boolean;
  heartbeatInterval?: number;
  queue?: boolean; // Use message queue for high-frequency data
}

interface WebSocketState<T = any> {
  data: T | null;
  isConnected: boolean;
  error: Error | null;
  send: (data: any) => void;
  close: () => void;
  reconnect: () => void;
}

export function useWebSocket<T = any>(
  url: string,
  options: WebSocketOptions = {}
): WebSocketState<T> {
  const {
    enabled = true,
    reconnect = true,
    reconnectInterval = 5000,
    reconnectAttempts = 5,
    heartbeat = true,
    heartbeatInterval = 30000,
    queue = false,
  } = options;

  const [data, setData] = useState<T | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  
  const ws = useRef<WebSocket | null>(null);
  const reconnectCount = useRef(0);
  const candidateIndex = useRef(0);
  const openedOnce = useRef(false);
  const heartbeatTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const queueUnsubscribeRef = useRef<(() => void) | null>(null);
  const candidateUrls = useMemo(() => buildWebSocketUrls(url), [url]);

  const clearTimers = useCallback(() => {
    if (heartbeatTimer.current) {
      clearInterval(heartbeatTimer.current);
      heartbeatTimer.current = null;
    }
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }
  }, []);

  const startHeartbeat = useCallback(() => {
    if (!heartbeat || !ws.current) return;
    
    heartbeatTimer.current = setInterval(() => {
      if (ws.current?.readyState === WebSocket.OPEN) {
        ws.current.send(JSON.stringify({ type: 'ping' }));
      }
    }, heartbeatInterval);
  }, [heartbeat, heartbeatInterval]);

  const connect = useCallback(() => {
    if (!enabled || candidateUrls.length === 0) {
      return;
    }

    try {
      const fullUrl = candidateUrls[Math.min(candidateIndex.current, candidateUrls.length - 1)];
      ws.current = new WebSocket(fullUrl);

      ws.current.onopen = () => {
        openedOnce.current = true;
        candidateIndex.current = 0;
        setIsConnected(true);
        setError(null);
        reconnectCount.current = 0;
        startHeartbeat();
      };

      ws.current.onmessage = (event) => {
        try {
          const parsedData = JSON.parse(event.data);
          // Ignore pong messages
          if (parsedData.type !== 'pong') {
            if (options.queue) {
              wsMessageQueue.enqueue(parsedData);
            } else {
              setData(parsedData);
            }
          }
        } catch (err) {
          // If not JSON, set raw data
          const rawData = event.data;
          if (options.queue) {
            wsMessageQueue.enqueue(rawData);
          } else {
            setData(rawData);
          }
        }
      };

      ws.current.onerror = () => {
        setError(new Error('WebSocket error'));
      };

      ws.current.onclose = () => {
        setIsConnected(false);
        clearTimers();

        if (!openedOnce.current && candidateIndex.current < candidateUrls.length - 1) {
          candidateIndex.current += 1;
          connect();
          return;
        }

        openedOnce.current = false;
        candidateIndex.current = 0;
        
        // Attempt reconnection if enabled
        if (reconnect && reconnectCount.current < reconnectAttempts) {
          reconnectCount.current++;
          reconnectTimer.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      };
    } catch (err) {
      setError(err as Error);
    }
  }, [
    enabled,
    candidateUrls,
    reconnect,
    reconnectInterval,
    reconnectAttempts,
    startHeartbeat,
    clearTimers,
  ]);

  const send = useCallback((data: any) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(typeof data === 'string' ? data : JSON.stringify(data));
    }
  }, []);

  const close = useCallback(() => {
    clearTimers();
    reconnectCount.current = reconnectAttempts; // Prevent reconnection
    candidateIndex.current = 0;
    openedOnce.current = false;
    ws.current?.close();
  }, [reconnectAttempts, clearTimers]);

  const reconnectManually = useCallback(() => {
    reconnectCount.current = 0;
    candidateIndex.current = 0;
    openedOnce.current = false;
    close();
    setTimeout(connect, 100);
  }, [close, connect]);

  useEffect(() => {
    if (!enabled) {
      clearTimers();
      ws.current?.close();
      setIsConnected(false);
      return;
    }

    connect();

    // Subscribe to message queue if queue option is enabled
    if (queue) {
      queueUnsubscribeRef.current = wsMessageQueue.subscribe((queuedData) => {
        setData(queuedData);
      });
    }

    return () => {
      clearTimers();
      ws.current?.close();
      if (queueUnsubscribeRef.current) {
        queueUnsubscribeRef.current();
        queueUnsubscribeRef.current = null;
      }
    };
  }, [enabled, connect, clearTimers, queue]);

  return {
    data,
    isConnected,
    error,
    send,
    close,
    reconnect: reconnectManually,
  };
}

function unavailableRealtimeChannel<T = any>(message: string): WebSocketState<T> {
  return {
    data: null,
    isConnected: false,
    error: new Error(message),
    send: () => {},
    close: () => {},
    reconnect: () => {},
  };
}

// Specialized hooks for different WebSocket endpoints
export function useMonitoringWebSocket() {
  return useWebSocket('/api/ws/metrics', {
    heartbeatInterval: 20000,
  });
}

export function useVMWebSocket(vmId?: string) {
  return unavailableRealtimeChannel(
    `VM realtime updates are not available on the canonical websocket contract${vmId ? ` for ${vmId}` : ''}.`
  );
}

export function useNetworkWebSocket() {
  return unavailableRealtimeChannel(
    'Network realtime updates are not available on the canonical websocket contract.'
  );
}

export function useStorageWebSocket() {
  return unavailableRealtimeChannel(
    'Storage realtime updates are not available on the canonical websocket contract.'
  );
}

export function useSecurityWebSocket() {
  return useWebSocket('/api/ws/security/events', {
    heartbeatInterval: 10000,
    reconnectAttempts: 10, // More attempts for security monitoring
  });
}

// Enhanced WebSocket hooks for distributed system updates
export function useDistributedTopologyWebSocket() {
  return unavailableRealtimeChannel(
    'Distributed topology updates are not available on the canonical websocket contract.'
  );
}

export function useBandwidthMonitoringWebSocket() {
  return useWebSocket('/api/ws/metrics?sources=bandwidth,network_io,qos&interval=5', {
    heartbeatInterval: 5000, // High frequency for bandwidth data
    reconnectAttempts: 12,
  });
}

export function usePerformancePredictionWebSocket() {
  return useWebSocket('/api/ws/metrics?sources=cpu_usage,memory_usage,disk_usage,predictions&interval=30', {
    heartbeatInterval: 30000,
    reconnectAttempts: 6,
  });
}

export function useSupercomputeFabricWebSocket() {
  return useWebSocket('/api/ws/metrics?sources=fabric,compute_jobs,memory_fabric,processing&interval=20', {
    heartbeatInterval: 20000,
    reconnectAttempts: 10,
  });
}

export function useFederationWebSocket() {
  return useWebSocket('/api/ws/alerts?sources=federation', {
    heartbeatInterval: 25000,
    reconnectAttempts: 8,
  });
}

export function useCrossClusterWebSocket() {
  return useWebSocket('/api/ws/metrics?sources=cluster,cross_cluster,replication&interval=15', {
    heartbeatInterval: 15000,
    reconnectAttempts: 10,
  });
}

// Specialized hooks for different data streams
export function useMetricsStreamWebSocket(endpoint: string, options?: Partial<WebSocketOptions>) {
  return useWebSocket(`/api/ws/metrics/${endpoint}`, {
    heartbeatInterval: 10000,
    reconnectAttempts: 8,
    ...options,
  });
}

export function useAIModelWebSocket(modelId: string) {
  return unavailableRealtimeChannel(
    `AI model realtime updates are not available on the canonical websocket contract for ${modelId}.`
  );
}

export function useJobMonitoringWebSocket(jobId?: string) {
  return unavailableRealtimeChannel(
    `Job monitoring updates are not available on the canonical websocket contract${jobId ? ` for ${jobId}` : ''}.`
  );
}

// Connection pool manager for multiple WebSocket connections
class WebSocketConnectionPool {
  private connections: Map<string, WebSocket> = new Map();
  private maxConnections = 10;

  getConnection(url: string): WebSocket | null {
    if (this.connections.has(url)) {
      return this.connections.get(url) || null;
    }

    if (this.connections.size >= this.maxConnections) {
      // Close oldest connection
      const firstKey = this.connections.keys().next().value;
      const oldConnection = this.connections.get(firstKey);
      if (oldConnection) {
        oldConnection.close();
        this.connections.delete(firstKey);
      }
    }

    try {
      const ws = new WebSocket(url);
      this.connections.set(url, ws);
      return ws;
    } catch {
      return null;
    }
  }

  closeConnection(url: string) {
    const connection = this.connections.get(url);
    if (connection) {
      connection.close();
      this.connections.delete(url);
    }
  }

  closeAllConnections() {
    this.connections.forEach((ws) => ws.close());
    this.connections.clear();
  }
}

export const wsConnectionPool = new WebSocketConnectionPool();

// WebSocket message queue for handling high-frequency updates
class WebSocketMessageQueue {
  private queue: Array<{ timestamp: number; data: any }> = [];
  private maxSize = 1000;
  private processingInterval = 100; // Process every 100ms
  private processor: ReturnType<typeof setInterval> | null = null;
  private subscribers: Array<(data: any) => void> = [];

  start() {
    if (this.processor) return;

    this.processor = setInterval(() => {
      this.processQueue();
    }, this.processingInterval);
  }

  stop() {
    if (this.processor) {
      clearInterval(this.processor);
      this.processor = null;
    }
  }

  enqueue(data: any) {
    if (this.subscribers.length === 0) return; // Don't queue if no subscribers

    this.queue.push({ timestamp: Date.now(), data });

    if (this.queue.length > this.maxSize) {
      this.queue.shift(); // Remove oldest message
    }
  }

  private processQueue() {
    if (this.queue.length === 0 || this.subscribers.length === 0) return;

    const messages = this.queue.splice(0, Math.min(10, this.queue.length));

    messages.forEach(({ data }) => {
      this.subscribers.forEach(callback => callback(data));
    });
  }

  subscribe(callback: (data: any) => void) {
    this.subscribers.push(callback);

    // Start processing if this is the first subscriber
    if (this.subscribers.length === 1 && !this.processor) {
      this.start();
    }

    return () => {
      const index = this.subscribers.indexOf(callback);
      if (index > -1) {
        this.subscribers.splice(index, 1);
      }

      // Stop processing if no subscribers
      if (this.subscribers.length === 0) {
        this.stop();
      }
    };
  }
}

export const wsMessageQueue = new WebSocketMessageQueue();

// Clean up on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    wsMessageQueue.stop();
    wsConnectionPool.closeAllConnections();
  });
}
