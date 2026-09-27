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
 * Connect to WebSocket with authentication. The token travels as a
 * `Sec-WebSocket-Protocol: bearer, <token>` negotiation (never a `?token=`
 * query param, which would leak into proxy access logs) because browsers
 * cannot set an Authorization header on a WebSocket handshake.
 */
export function connectEvents(onWelcome?: (msg: unknown) => void): WebSocket {
  try {
    const token = getAuthToken();
    const wsUrl = buildWebSocketUrls(EVENT_STREAM_PATH)[0] || WS_URL;
    const ws = token ? new WebSocket(wsUrl, ['bearer', token]) : new WebSocket(wsUrl);
    let first = true;

    ws.addEventListener('open', () => {
      console.log('WS connected');
    });

    ws.addEventListener('error', (error) => console.warn('WS error:', error));
    ws.addEventListener('close', () => console.log('WS disconnected'));
    ws.addEventListener('message', (ev) => {
      try {
        const payload = JSON.parse(ev.data as string);
        console.log('WS message:', payload);
        if (first) {
          first = false;
          onWelcome?.(payload);
        }
      } catch (e) {
        console.warn('WS message parse error', e);
      }
    });

    return ws;
  } catch (error) {
    console.error('Failed to create WebSocket:', error);
    // Return a mock WebSocket that won't cause crashes
    return {
      close: () => undefined,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      send: () => undefined,
      readyState: WebSocket.CLOSED
    } as unknown as WebSocket;
  }
}
