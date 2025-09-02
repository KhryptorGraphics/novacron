import { useEffect, useState, useCallback, useRef } from 'react';

interface WebSocketOptions {
  reconnect?: boolean;
  reconnectInterval?: number;
  reconnectAttempts?: number;
  heartbeat?: boolean;
  heartbeatInterval?: number;
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
    reconnect = true,
    reconnectInterval = 5000,
    reconnectAttempts = 5,
    heartbeat = true,
    heartbeatInterval = 30000,
  } = options;

  const [data, setData] = useState<T | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  
  const ws = useRef<WebSocket | null>(null);
  const reconnectCount = useRef(0);
  const heartbeatTimer = useRef<NodeJS.Timeout | null>(null);
  const reconnectTimer = useRef<NodeJS.Timeout | null>(null);

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
    try {
      // Construct full WebSocket URL
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.host;
      const fullUrl = url.startsWith('/') ? `${protocol}//${host}${url}` : url;
      
      ws.current = new WebSocket(fullUrl);

      ws.current.onopen = () => {
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
            setData(parsedData);
          }
        } catch (err) {
          // If not JSON, set raw data
          setData(event.data);
        }
      };

      ws.current.onerror = (event) => {
        setError(new Error('WebSocket error'));
      };

      ws.current.onclose = () => {
        setIsConnected(false);
        clearTimers();
        
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
  }, [url, reconnect, reconnectInterval, reconnectAttempts, startHeartbeat, clearTimers]);

  const send = useCallback((data: any) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(typeof data === 'string' ? data : JSON.stringify(data));
    }
  }, []);

  const close = useCallback(() => {
    clearTimers();
    reconnectCount.current = reconnectAttempts; // Prevent reconnection
    ws.current?.close();
  }, [reconnectAttempts, clearTimers]);

  const reconnectManually = useCallback(() => {
    reconnectCount.current = 0;
    close();
    setTimeout(connect, 100);
  }, [close, connect]);

  useEffect(() => {
    connect();

    return () => {
      clearTimers();
      ws.current?.close();
    };
  }, [connect, clearTimers]);

  return {
    data,
    isConnected,
    error,
    send,
    close,
    reconnect: reconnectManually,
  };
}

// Specialized hooks for different WebSocket endpoints
export function useMonitoringWebSocket() {
  return useWebSocket('/api/ws/monitoring', {
    heartbeatInterval: 20000,
  });
}

export function useVMWebSocket(vmId?: string) {
  return useWebSocket(vmId ? `/api/ws/vms/${vmId}` : '/api/ws/vms', {
    heartbeatInterval: 15000,
  });
}

export function useNetworkWebSocket() {
  return useWebSocket('/api/ws/network', {
    heartbeatInterval: 25000,
  });
}

export function useStorageWebSocket() {
  return useWebSocket('/api/ws/storage', {
    heartbeatInterval: 30000,
  });
}

export function useSecurityWebSocket() {
  return useWebSocket('/api/ws/security', {
    heartbeatInterval: 10000,
    reconnectAttempts: 10, // More attempts for security monitoring
  });
}