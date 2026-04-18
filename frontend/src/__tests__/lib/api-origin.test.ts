describe('API origin helpers', () => {
  const originalApiUrl = process.env.NEXT_PUBLIC_API_URL;
  const originalWsUrl = process.env.NEXT_PUBLIC_WS_URL;

  afterEach(() => {
    process.env.NEXT_PUBLIC_API_URL = originalApiUrl;
    process.env.NEXT_PUBLIC_WS_URL = originalWsUrl;
    jest.resetModules();
  });

  it('derives websocket candidates from the canonical API origin when no websocket origin is configured', () => {
    process.env.NEXT_PUBLIC_API_URL = 'http://localhost:8090/api';
    delete process.env.NEXT_PUBLIC_WS_URL;

    const { buildWebSocketUrls } = require('../../lib/api/origin');

    expect(buildWebSocketUrls('/api/ws/security/events')).toEqual([
      'ws://localhost:8090/api/ws/security/events',
      'ws://localhost:8090/api/security/events/stream',
    ]);
  });

  it('normalizes canonical API routes onto the API origin', () => {
    process.env.NEXT_PUBLIC_API_URL = 'http://localhost:8090/api';
    delete process.env.NEXT_PUBLIC_WS_URL;

    const { buildApiUrl } = require('../../lib/api/origin');

    expect(buildApiUrl('/api/security/events')).toBe(
      'http://localhost:8090/api/security/events',
    );
  });
});
