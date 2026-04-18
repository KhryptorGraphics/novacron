import { buildWebSocketUrls } from '@/lib/api/origin';

const EVENT_STREAM_PATH = '/api/ws/metrics';

export const WS_URL = buildWebSocketUrls(EVENT_STREAM_PATH)[0] || 'ws://localhost:8090/api/ws/metrics';

/**
 * Get auth token from localStorage
 */
function getAuthToken(): string | null {
  if (typeof window !== 'undefined') {
    return localStorage.getItem('novacron_token');
  }
  return null;
}

/**
 * Connect to WebSocket with authentication
 */
export function connectEvents(onWelcome?: (msg: unknown) => void): WebSocket {
  try {
    // Get auth token and add to URL
    const token = getAuthToken();
    const baseUrl = buildWebSocketUrls(EVENT_STREAM_PATH)[0] || WS_URL;
    const wsUrl = token ? `${baseUrl}?token=${encodeURIComponent(token)}` : baseUrl;

    const ws = new WebSocket(wsUrl);
    let first = true;

    ws.addEventListener("open", () => {
      console.log("WS connected");

      // Send authentication message if token exists
      if (token) {
        ws.send(JSON.stringify({
          type: 'auth',
          token: token
        }));
      }
    });

    ws.addEventListener("error", (error) => console.warn("WS error:", error));
    ws.addEventListener("close", () => console.log("WS disconnected"));
    ws.addEventListener("message", (ev) => {
      try {
        const payload = JSON.parse(ev.data as string);
        console.log("WS message:", payload);
        if (first) {
          first = false;
          onWelcome?.(payload);
        }
      } catch (e) {
        console.warn("WS message parse error", e);
      }
    });

    return ws;
  } catch (error) {
    console.error("Failed to create WebSocket:", error);
    // Return a mock WebSocket that won't cause crashes
    return {
      close: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      send: () => {},
      readyState: WebSocket.CLOSED
    } as WebSocket;
  }
}
