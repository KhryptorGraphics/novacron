import { act, renderHook, waitFor } from '@testing-library/react';
import { useWebSocket } from '../../hooks/useWebSocket';

jest.mock('../../lib/api/origin', () => ({
  buildWebSocketUrls: jest.fn(() => [
    'ws://localhost:8090/api/ws/security/events',
    'ws://localhost:8090/api/security/events/stream',
  ]),
}));

class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
  static instances: MockWebSocket[] = [];

  readonly url: string;
  readyState = MockWebSocket.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  send = jest.fn();
  close = jest.fn();

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  emitOpen() {
    this.readyState = MockWebSocket.OPEN;
    this.onopen?.(new Event('open'));
  }

  emitClose(reason = 'failed') {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.({ code: 1006, reason } as CloseEvent);
  }
}

describe('useWebSocket', () => {
  beforeEach(() => {
    MockWebSocket.instances = [];
    (global as typeof globalThis & { WebSocket: typeof WebSocket }).WebSocket =
      MockWebSocket as unknown as typeof WebSocket;
  });

  it('uses the canonical API origin and falls back to the legacy security stream alias', async () => {
    const { result } = renderHook(() =>
      useWebSocket('/api/ws/security/events', { reconnect: false }),
    );

    expect(MockWebSocket.instances[0]?.url).toBe(
      'ws://localhost:8090/api/ws/security/events',
    );

    act(() => {
      MockWebSocket.instances[0].emitClose();
    });

    await waitFor(() => {
      expect(MockWebSocket.instances).toHaveLength(2);
    });

    expect(MockWebSocket.instances[1]?.url).toBe(
      'ws://localhost:8090/api/security/events/stream',
    );

    act(() => {
      MockWebSocket.instances[1].emitOpen();
    });

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
    });
  });

  it('does not open a socket when disabled', () => {
    renderHook(() =>
      useWebSocket('/api/ws/security/events', { enabled: false, reconnect: false }),
    );

    expect(MockWebSocket.instances).toHaveLength(0);
  });
});
