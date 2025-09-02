export const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8090/ws/orchestration";

// Get auth token from various sources
function getAuthToken(): string | null {
  try {
    // Try localStorage first
    const token = localStorage.getItem('auth_token') || localStorage.getItem('access_token');
    if (token) return token;
    
    // Try sessionStorage
    const sessionToken = sessionStorage.getItem('auth_token') || sessionStorage.getItem('access_token');
    if (sessionToken) return sessionToken;
    
    // Try cookies
    const cookies = document.cookie.split(';');
    for (const cookie of cookies) {
      const [name, value] = cookie.trim().split('=');
      if (name === 'auth_token' || name === 'access_token') {
        return decodeURIComponent(value);
      }
    }
  } catch (error) {
    console.warn('Failed to get auth token:', error);
  }
  return null;
}

export interface SecureWebSocketOptions {
  onWelcome?: (msg: unknown) => void;
  onError?: (error: Event) => void;
  requireAuth?: boolean;
  reconnect?: boolean;
  reconnectAttempts?: number;
  reconnectDelay?: number;
}

export function connectEvents(options: SecureWebSocketOptions | ((msg: unknown) => void) = {}): WebSocket {
  // Handle backward compatibility - if first arg is a function, use it as onWelcome
  if (typeof options === 'function') {
    options = { onWelcome: options };
  }

  const {
    onWelcome,
    onError,
    requireAuth = true,
    reconnect = true,
    reconnectAttempts = 5,
    reconnectDelay = 5000
  } = options;

  try {
    let wsUrl = WS_URL;
    const authToken = getAuthToken();
    
    // Check if authentication is required
    if (requireAuth && !authToken) {
      console.error('Authentication required but no token found');
      throw new Error('Authentication required');
    }

    // Create WebSocket with authentication
    let ws: WebSocket;
    if (authToken) {
      // Use Sec-WebSocket-Protocol header for secure token transmission
      ws = new WebSocket(wsUrl, [`access_token.${authToken}`]);
    } else {
      ws = new WebSocket(wsUrl);
    }

    let first = true;
    let reconnectCount = 0;
    
    ws.addEventListener("open", () => {
      console.log("Secure WebSocket connected");
      reconnectCount = 0; // Reset on successful connection
    });
    
    ws.addEventListener("error", (error) => {
      console.warn("WebSocket error:", error);
      onError?.(error);
    });
    
    ws.addEventListener("close", (event) => {
      console.log("WebSocket disconnected", { code: event.code, reason: event.reason });
      
      // Handle different close codes
      if (event.code === 1006) {
        console.warn('WebSocket connection closed abnormally');
      } else if (event.code === 1001) {
        console.info('WebSocket connection closed by server');
      } else if (event.code === 4001) {
        console.error('WebSocket authentication failed');
        return; // Don't reconnect on auth failure
      } else if (event.code === 4003) {
        console.error('WebSocket rate limited');
        return; // Don't reconnect immediately on rate limit
      }
      
      // Attempt reconnection if enabled
      if (reconnect && reconnectCount < reconnectAttempts) {
        reconnectCount++;
        console.log(`Attempting to reconnect (${reconnectCount}/${reconnectAttempts})`);
        setTimeout(() => {
          try {
            const newWs = connectEvents(options as SecureWebSocketOptions);
            // Copy event listeners to new WebSocket
            Object.assign(ws, newWs);
          } catch (error) {
            console.error('Reconnection failed:', error);
          }
        }, reconnectDelay * reconnectCount); // Exponential backoff
      }
    });
    
    ws.addEventListener("message", (ev) => {
      try {
        const payload = JSON.parse(ev.data as string);
        
        // Handle different message types
        if (payload.type === 'connected') {
          console.log("WebSocket authenticated:", {
            clientId: payload.data?.client_id,
            authenticated: payload.data?.authenticated,
            userId: payload.data?.user_id
          });
          if (first) {
            first = false;
            onWelcome?.(payload);
          }
        } else if (payload.type === 'ping') {
          // Respond to ping with pong
          ws.send(JSON.stringify({ type: 'pong' }));
        } else if (payload.type === 'error') {
          console.error('WebSocket server error:', payload.error);
        } else {
          console.log("WebSocket message:", payload);
        }
      } catch (e) {
        console.warn("WebSocket message parse error:", e);
      }
    });
    
    return ws;
  } catch (error) {
    console.error("Failed to create secure WebSocket:", error);
    onError?.(error as Event);
    
    // Return a mock WebSocket that won't cause crashes
    return {
      close: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      send: () => {},
      readyState: WebSocket.CLOSED,
      url: WS_URL,
      protocol: '',
      extensions: '',
      bufferedAmount: 0,
      binaryType: 'blob' as BinaryType,
      onopen: null,
      onmessage: null,
      onerror: null,
      onclose: null,
      CONNECTING: WebSocket.CONNECTING,
      OPEN: WebSocket.OPEN,
      CLOSING: WebSocket.CLOSING,
      CLOSED: WebSocket.CLOSED
    } as WebSocket;
  }
}

// Subscribe to specific event types
export function subscribeToEvents(ws: WebSocket, eventTypes: string[], sources?: string[]) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'subscribe',
      filters: {
        event_types: eventTypes,
        sources: sources
      }
    }));
  }
}

// Unsubscribe from events
export function unsubscribeFromEvents(ws: WebSocket) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'unsubscribe'
    }));
  }
}

