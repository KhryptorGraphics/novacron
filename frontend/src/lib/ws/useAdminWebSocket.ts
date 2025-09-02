import { useEffect, useState, useCallback, useRef } from 'react';
import useWebSocket, { ReadyState } from 'react-use-websocket';
import { useQueryClient } from '@tanstack/react-query';
import { ADMIN_QUERY_KEYS } from '../api/hooks/useAdmin';
import { useToast } from '@/components/ui/use-toast';

export interface AdminWebSocketMessage {
  type: 'system_metrics' | 'security_alert' | 'user_activity' | 'vm_status' | 'audit_log' | 'config_change';
  data: any;
  timestamp: string;
}

export interface WebSocketConnectionState {
  isConnected: boolean;
  connectionState: ReadyState;
  lastMessage: AdminWebSocketMessage | null;
  reconnectAttempts: number;
  error: string | null;
}

export const useAdminWebSocket = () => {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [connectionState, setConnectionState] = useState<WebSocketConnectionState>({
    isConnected: false,
    connectionState: ReadyState.UNINSTANTIATED,
    lastMessage: null,
    reconnectAttempts: 0,
    error: null
  });
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  
  const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8091';
  const socketUrl = `${wsUrl}/admin/ws`;
  
  const {
    sendMessage,
    lastMessage,
    readyState,
    getWebSocket
  } = useWebSocket(socketUrl, {
    onOpen: () => {
      console.log('Admin WebSocket connected');
      setConnectionState(prev => ({
        ...prev,
        isConnected: true,
        connectionState: readyState,
        error: null,
        reconnectAttempts: 0
      }));
      
      // Send authentication token if available
      const token = localStorage.getItem('novacron_token');
      if (token) {
        sendMessage(JSON.stringify({
          type: 'auth',
          token
        }));
      }
      
      // Subscribe to admin events
      sendMessage(JSON.stringify({
        type: 'subscribe',
        channels: [
          'system_metrics',
          'security_alerts', 
          'user_activity',
          'vm_status',
          'audit_logs',
          'config_changes'
        ]
      }));
    },
    onClose: (event) => {
      console.log('Admin WebSocket disconnected:', event);
      setConnectionState(prev => ({
        ...prev,
        isConnected: false,
        connectionState: readyState,
        error: event.reason || 'Connection closed'
      }));
      
      // Attempt to reconnect with exponential backoff
      if (event.code !== 1000) { // Not a normal closure
        const attempts = connectionState.reconnectAttempts + 1;
        const delay = Math.min(1000 * Math.pow(2, attempts), 30000); // Max 30 seconds
        
        setConnectionState(prev => ({
          ...prev,
          reconnectAttempts: attempts
        }));
        
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log(`Attempting to reconnect (attempt ${attempts})`);
          getWebSocket()?.close();
        }, delay);
      }
    },
    onError: (event) => {
      console.error('Admin WebSocket error:', event);
      setConnectionState(prev => ({
        ...prev,
        error: 'Connection error occurred'
      }));
    },
    onMessage: (event) => {
      try {
        const message: AdminWebSocketMessage = JSON.parse(event.data);
        handleWebSocketMessage(message);
        
        setConnectionState(prev => ({
          ...prev,
          lastMessage: message
        }));
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    },
    shouldReconnect: (closeEvent) => {
      // Reconnect unless it's a normal closure or authentication failure
      return closeEvent.code !== 1000 && closeEvent.code !== 4001;
    },
    reconnectInterval: (attemptNumber) => {
      return Math.min(1000 * Math.pow(2, attemptNumber), 30000);
    },
    reconnectAttempts: 10,
    heartbeat: {
      message: JSON.stringify({ type: 'ping' }),
      returnMessage: JSON.stringify({ type: 'pong' }),
      timeout: 60000, // 1 minute
      interval: 30000  // 30 seconds
    }
  });
  
  const handleWebSocketMessage = useCallback((message: AdminWebSocketMessage) => {
    switch (message.type) {
      case 'system_metrics':
        // Update system metrics cache
        queryClient.setQueryData(
          ADMIN_QUERY_KEYS.SYSTEM_METRICS,
          (oldData: any) => {
            if (!oldData) return [message.data];
            return [...oldData.slice(-49), message.data]; // Keep last 50 points
          }
        );
        break;
        
      case 'security_alert':
        // Add new security alert
        queryClient.setQueryData(
          ADMIN_QUERY_KEYS.SECURITY_ALERTS,
          (oldData: any) => {
            if (!oldData) return { alerts: [message.data], total: 1 };
            return {
              ...oldData,
              alerts: [message.data, ...oldData.alerts],
              total: oldData.total + 1
            };
          }
        );
        
        // Show toast notification for high/critical alerts
        if (['high', 'critical'].includes(message.data.severity)) {
          toast({
            title: "Security Alert",
            description: message.data.title,
            variant: message.data.severity === 'critical' ? 'destructive' : 'default'
          });
        }
        break;
        
      case 'user_activity':
        // Invalidate user-related queries to trigger refetch
        queryClient.invalidateQueries({ queryKey: ADMIN_QUERY_KEYS.USERS });
        
        if (message.data.action === 'login_failed' && message.data.count > 5) {
          toast({
            title: "Suspicious Activity",
            description: `Multiple failed login attempts detected for ${message.data.email}`,
            variant: "destructive"
          });
        }
        break;
        
      case 'vm_status':
        // Update VM status in real-time
        queryClient.setQueryData(
          ['vms'],
          (oldData: any) => {
            if (!oldData) return oldData;
            return {
              ...oldData,
              vms: oldData.vms?.map((vm: any) => 
                vm.id === message.data.id 
                  ? { ...vm, ...message.data }
                  : vm
              )
            };
          }
        );
        
        // Notify about VM state changes
        if (message.data.previous_status && message.data.status !== message.data.previous_status) {
          toast({
            title: "VM Status Changed",
            description: `${message.data.name}: ${message.data.previous_status} → ${message.data.status}`,
          });
        }
        break;
        
      case 'audit_log':
        // Add new audit log entry
        queryClient.setQueryData(
          ADMIN_QUERY_KEYS.AUDIT_LOGS,
          (oldData: any) => {
            if (!oldData) return { logs: [message.data], total: 1 };
            return {
              ...oldData,
              logs: [message.data, ...oldData.logs.slice(0, 99)], // Keep last 100
              total: oldData.total + 1
            };
          }
        );
        break;
        
      case 'config_change':
        // Invalidate system config queries
        queryClient.invalidateQueries({ queryKey: ADMIN_QUERY_KEYS.SYSTEM_CONFIG });
        
        toast({
          title: "Configuration Updated",
          description: `${message.data.key} has been changed by ${message.data.updated_by}`,
        });
        break;
        
      default:
        console.log('Unhandled WebSocket message type:', message.type);
    }
  }, [queryClient, toast]);
  
  // Update connection state when readyState changes
  useEffect(() => {
    setConnectionState(prev => ({
      ...prev,
      connectionState: readyState,
      isConnected: readyState === ReadyState.OPEN
    }));
  }, [readyState]);
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);
  
  // Methods to send specific commands
  const subscribeToChannel = useCallback((channel: string) => {
    if (readyState === ReadyState.OPEN) {
      sendMessage(JSON.stringify({
        type: 'subscribe',
        channel
      }));
    }
  }, [readyState, sendMessage]);
  
  const unsubscribeFromChannel = useCallback((channel: string) => {
    if (readyState === ReadyState.OPEN) {
      sendMessage(JSON.stringify({
        type: 'unsubscribe',
        channel
      }));
    }
  }, [readyState, sendMessage]);
  
  const requestMetricsSnapshot = useCallback(() => {
    if (readyState === ReadyState.OPEN) {
      sendMessage(JSON.stringify({
        type: 'request_snapshot',
        data_type: 'system_metrics'
      }));
    }
  }, [readyState, sendMessage]);
  
  const sendCommand = useCallback((command: string, data?: any) => {
    if (readyState === ReadyState.OPEN) {
      sendMessage(JSON.stringify({
        type: 'command',
        command,
        data
      }));
    }
  }, [readyState, sendMessage]);
  
  return {
    connectionState,
    lastMessage: connectionState.lastMessage,
    sendMessage,
    subscribeToChannel,
    unsubscribeFromChannel,
    requestMetricsSnapshot,
    sendCommand
  };
};

// Hook for specific admin dashboard real-time features
export const useAdminRealTimeUpdates = () => {
  const queryClient = useQueryClient();
  const { connectionState, subscribeToChannel, unsubscribeFromChannel } = useAdminWebSocket();
  const [metrics, setMetrics] = useState<any[]>([]);
  const [alerts, setAlerts] = useState<any[]>([]);
  
  useEffect(() => {
    // Subscribe to real-time updates when component mounts
    subscribeToChannel('system_metrics');
    subscribeToChannel('security_alerts');
    subscribeToChannel('user_activity');
    
    return () => {
      // Cleanup subscriptions
      unsubscribeFromChannel('system_metrics');
      unsubscribeFromChannel('security_alerts');
      unsubscribeFromChannel('user_activity');
    };
  }, [subscribeToChannel, unsubscribeFromChannel]);
  
  // Listen for query cache updates
  useEffect(() => {
    const unsubscribe = queryClient.getQueryCache().subscribe((event) => {
      if (event.type === 'updated') {
        if (event.query.queryKey[0] === 'admin' && event.query.queryKey[1] === 'system-metrics') {
          setMetrics(event.query.state.data as any[] || []);
        }
        if (event.query.queryKey[0] === 'admin' && event.query.queryKey[1] === 'security-alerts') {
          const data = event.query.state.data as any;
          setAlerts(data?.alerts || []);
        }
      }
    });
    
    return unsubscribe;
  }, [queryClient]);
  
  return {
    isConnected: connectionState.isConnected,
    metrics,
    alerts,
    connectionState: connectionState.connectionState,
    error: connectionState.error
  };
};

// Connection status component helper
export const getConnectionStatusInfo = (state: ReadyState) => {
  switch (state) {
    case ReadyState.CONNECTING:
      return { status: 'Connecting...', color: 'text-yellow-600', icon: '⚡' };
    case ReadyState.OPEN:
      return { status: 'Connected', color: 'text-green-600', icon: '🟢' };
    case ReadyState.CLOSING:
      return { status: 'Disconnecting...', color: 'text-yellow-600', icon: '⚠️' };
    case ReadyState.CLOSED:
      return { status: 'Disconnected', color: 'text-red-600', icon: '🔴' };
    default:
      return { status: 'Unknown', color: 'text-gray-600', icon: '❓' };
  }
};