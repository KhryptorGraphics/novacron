describe('API origin helpers', () => {
  const originalApiUrl = process.env.NEXT_PUBLIC_API_URL;
  const originalWsUrl = process.env.NEXT_PUBLIC_WS_URL;
  const originalApiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

  afterEach(() => {
    process.env.NEXT_PUBLIC_API_URL = originalApiUrl;
    process.env.NEXT_PUBLIC_WS_URL = originalWsUrl;
    process.env.NEXT_PUBLIC_API_BASE_URL = originalApiBaseUrl;
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

  it('always derives api/v1 URLs from the canonical API origin', () => {
    process.env.NEXT_PUBLIC_API_URL = 'http://localhost:8090/api';
    process.env.NEXT_PUBLIC_API_BASE_URL = 'https://legacy.example.invalid/api/v1';

    const { buildApiV1Url, API_V1_BASE } = require('../../lib/api/origin');

    expect(API_V1_BASE).toBe('http://localhost:8090/api/v1');
    expect(buildApiV1Url('/vms')).toBe('http://localhost:8090/api/v1/vms');
  });
});
