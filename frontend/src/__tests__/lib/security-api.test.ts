import securityAPI from '../../lib/api/security';

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
});
