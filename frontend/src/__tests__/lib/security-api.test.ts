import securityAPI, {
} from '../../lib/api/security';

describe('securityAPI auth token handling', () => {
  const mockFetch = jest.fn();

  beforeEach(() => {
    mockFetch.mockReset();
    global.fetch = mockFetch as unknown as typeof fetch;
    window.localStorage.clear();
  });

  it('uses the canonical novacron_token for authorization', async () => {
    window.localStorage.setItem('authToken', 'legacy-token');
    window.localStorage.setItem('novacron_token', 'canonical-token');

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ events: [], total: 0 }),
    });

    await securityAPI.getSecurityEvents();

    expect(mockFetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: 'Bearer canonical-token',
        }),
      }),
    );
  });

  it('calls the live event acknowledgement route on the canonical server', async () => {
    window.localStorage.setItem('novacron_token', 'canonical-token');
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ acknowledged: true }),
    });

    await securityAPI.acknowledgeSecurityEvent('event-1');

    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/security/events/event-1/acknowledge'),
      expect.objectContaining({
        method: 'POST',
      }),
    );
  });

  it('creates manual security incidents on the canonical server', async () => {
    window.localStorage.setItem('novacron_token', 'canonical-token');
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ incidentId: 'incident-1' }),
    });

    await expect(
      securityAPI.createSecurityIncident({
        title: 'Manual incident',
        description: 'Operator escalation',
        severity: 'high',
        type: 'manual',
      }),
    ).resolves.toEqual({ incidentId: 'incident-1' });

    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/security/incidents'),
      expect.objectContaining({
        method: 'POST',
      }),
    );
  });
});
