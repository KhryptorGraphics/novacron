import authService from '../../lib/auth';

const encodeSegment = (value: Record<string, unknown>) =>
  window.btoa(JSON.stringify(value)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');

const createToken = (payload: Record<string, unknown>) =>
  `${encodeSegment({ alg: 'HS256', typ: 'JWT' })}.${encodeSegment(payload)}.signature`;

const okJson = (payload: unknown) => ({
  ok: true,
  json: async () => payload,
});

describe('authService role handling', () => {
  const mockFetch = jest.fn();

  beforeEach(() => {
    mockFetch.mockReset();
    global.fetch = mockFetch as unknown as typeof fetch;
    window.localStorage.clear();
  });

  it('hydrates role and roles from JWT claims', () => {
    const token = createToken({
      sub: 'user-1',
      email: 'admin@novacron.io',
      first_name: 'Admin',
      last_name: 'User',
      role: 'admin',
      roles: ['operator'],
      exp: Math.floor(Date.now() / 1000) + 60,
    });

    authService.setToken(token);

    expect(authService.getCurrentUser()).toMatchObject({
      id: 'user-1',
      role: 'admin',
      roles: ['admin', 'operator'],
    });
  });

  it('stores normalized roles when a login response omits them', () => {
    const token = createToken({
      sub: 'user-2',
      email: 'admin@novacron.io',
      first_name: 'Admin',
      last_name: 'User',
      role: 'admin',
      exp: Math.floor(Date.now() / 1000) + 60,
    });

    authService.setToken(token, {
      id: 'user-2',
      email: 'admin@novacron.io',
      firstName: 'Admin',
      lastName: 'User',
      status: 'active',
    });

    expect(JSON.parse(window.localStorage.getItem('authUser') || '{}')).toMatchObject({
      role: 'admin',
      roles: ['admin'],
    });
  });

  it('builds the GitHub authorization URL using the canonical runtime endpoint', async () => {
    mockFetch.mockResolvedValueOnce(
      okJson({
        provider: 'github',
        authorizationUrl: 'https://github.com/login/oauth/authorize?client_id=test',
      }),
    );

    const response = await authService.getGitHubAuthorizationUrl('/clusters/local');

    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/auth/oauth/github/url?redirect_to=%2Fclusters%2Flocal'),
      expect.objectContaining({
        headers: expect.objectContaining({
          'Content-Type': 'application/json',
        }),
      }),
    );
    expect(response).toEqual({
      provider: 'github',
      authorizationUrl: 'https://github.com/login/oauth/authorize?client_id=test',
    });
  });

  it('merges cached user data with JWT claims and metadata fallbacks', () => {
    window.localStorage.setItem(
      'authUser',
      JSON.stringify({
        id: 'user-3',
        email: 'cached@novacron.io',
        firstName: 'Cached',
        lastName: 'User',
        tenantId: 'tenant-cached',
        status: 'active',
        role: 'member',
        roles: ['member'],
      }),
    );

    const token = createToken({
      sub: 'user-3',
      tenant_id: 'tenant-from-token',
      metadata: {
        email: 'oauth@novacron.io',
        first_name: 'OAuth',
        last_name: 'User',
        status: 'pending',
      },
      exp: Math.floor(Date.now() / 1000) + 60,
    });

    authService.setToken(token);

    expect(authService.getCurrentUser()).toMatchObject({
      id: 'user-3',
      email: 'oauth@novacron.io',
      firstName: 'OAuth',
      lastName: 'User',
      tenantId: 'tenant-from-token',
      status: 'pending',
      role: 'member',
      roles: ['member'],
    });
  });
});
