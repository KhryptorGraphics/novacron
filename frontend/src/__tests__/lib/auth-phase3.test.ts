import authService from '../../lib/auth';

const encodeSegment = (value: Record<string, unknown>) =>
  window.btoa(JSON.stringify(value)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');

const createToken = (payload: Record<string, unknown>) =>
  `${encodeSegment({ alg: 'HS256', typ: 'JWT' })}.${encodeSegment(payload)}.signature`;

const okJson = (payload: unknown) => ({
  ok: true,
  json: async () => payload,
});

describe('authService canonical phase 3 auth flows', () => {
  const mockFetch = jest.fn();

  beforeEach(() => {
    mockFetch.mockReset();
    global.fetch = mockFetch as unknown as typeof fetch;
    window.localStorage.clear();
  });

  function authenticateUser() {
    const token = createToken({
      sub: 'user-7',
      email: 'admin@novacron.io',
      first_name: 'Admin',
      last_name: 'User',
      role: 'admin',
      exp: Math.floor(Date.now() / 1000) + 60,
    });

    authService.setToken(token);

    return token;
  }

  it('uses canonical password recovery routes', async () => {
    mockFetch
      .mockResolvedValueOnce(okJson({ message: 'sent' }))
      .mockResolvedValueOnce(okJson({ message: 'reset' }));

    await authService.forgotPassword({ email: 'admin@novacron.io' });
    await authService.resetPassword({ token: 'reset-token', password: 'new-password' });

    expect(mockFetch).toHaveBeenNthCalledWith(
      1,
      expect.stringContaining('/api/auth/forgot-password'),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ email: 'admin@novacron.io' }),
      }),
    );

    expect(mockFetch).toHaveBeenNthCalledWith(
      2,
      expect.stringContaining('/api/auth/reset-password'),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ token: 'reset-token', password: 'new-password' }),
      }),
    );
  });

  it('uses the authenticated user for 2FA setup and status routes', async () => {
    const token = authenticateUser();
    mockFetch
      .mockResolvedValueOnce(okJson({ qr_code: 'qr', secret: 'secret', backup_codes: ['A'] }))
      .mockResolvedValueOnce(okJson({ enabled: true, setup: true, setup_at: '2026-04-18T00:00:00Z' }));

    await authService.setup2FA();
    const status = await authService.get2FAStatus();

    expect(mockFetch).toHaveBeenNthCalledWith(
      1,
      expect.stringContaining('/api/auth/2fa/setup'),
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          Authorization: `Bearer ${token}`,
        }),
        body: JSON.stringify({
          user_id: 'user-7',
          account_name: 'admin@novacron.io',
        }),
      }),
    );

    expect(mockFetch).toHaveBeenNthCalledWith(
      2,
      expect.stringContaining('/api/auth/2fa/status?user_id=user-7'),
      expect.objectContaining({
        method: 'GET',
        headers: expect.objectContaining({
          Authorization: `Bearer ${token}`,
        }),
      }),
    );

    expect(status).toEqual({
      enabled: true,
      backup_codes_remaining: 0,
      setup: true,
      setup_at: '2026-04-18T00:00:00Z',
      last_used: undefined,
    });
  });

  it('uses canonical authenticated routes for 2FA verification and backup code management', async () => {
    const token = authenticateUser();
    mockFetch
      .mockResolvedValueOnce(okJson({ valid: true, token: 'verified-token' }))
      .mockResolvedValueOnce(okJson({ success: true, message: 'enabled' }))
      .mockResolvedValueOnce(okJson({ success: true, message: 'disabled' }))
      .mockResolvedValueOnce(okJson({ backup_codes: ['B1', 'B2'] }));

    await authService.verify2FASetup({ user_id: 'user-7', code: '123456' });
    await authService.enable2FA('123456');
    await authService.disable2FA();
    await authService.generateBackupCodes();

    expect(mockFetch).toHaveBeenNthCalledWith(
      1,
      expect.stringContaining('/api/auth/2fa/verify'),
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          Authorization: `Bearer ${token}`,
        }),
        body: JSON.stringify({
          user_id: 'user-7',
          code: '123456',
          is_backup_code: false,
        }),
      }),
    );

    expect(mockFetch).toHaveBeenNthCalledWith(
      2,
      expect.stringContaining('/api/auth/2fa/enable'),
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          Authorization: `Bearer ${token}`,
        }),
        body: JSON.stringify({
          user_id: 'user-7',
          code: '123456',
        }),
      }),
    );

    expect(mockFetch).toHaveBeenNthCalledWith(
      3,
      expect.stringContaining('/api/auth/2fa/disable'),
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          Authorization: `Bearer ${token}`,
        }),
        body: JSON.stringify({
          user_id: 'user-7',
        }),
      }),
    );

    expect(mockFetch).toHaveBeenNthCalledWith(
      4,
      expect.stringContaining('/api/auth/2fa/backup-codes'),
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          Authorization: `Bearer ${token}`,
        }),
        body: JSON.stringify({
          user_id: 'user-7',
        }),
      }),
    );
  });

  it('uses the canonical login verification route for 2FA challenge completion', async () => {
    mockFetch.mockResolvedValueOnce(okJson({ token: 'verified-session', expiresAt: 'tomorrow', user: { id: 'user-7' } }));

    await authService.verify2FALogin({
      user_id: 'user-7',
      code: '654321',
      temp_token: 'temp-session',
      is_backup_code: true,
    });

    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/auth/2fa/verify-login'),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          user_id: 'user-7',
          code: '654321',
          is_backup_code: true,
          temp_token: 'temp-session',
        }),
      }),
    );
  });
});
