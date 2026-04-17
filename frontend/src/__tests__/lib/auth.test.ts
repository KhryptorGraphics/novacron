import authService from '../../lib/auth';

const encodeSegment = (value: Record<string, unknown>) =>
  window.btoa(JSON.stringify(value)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');

const createToken = (payload: Record<string, unknown>) =>
  `${encodeSegment({ alg: 'HS256', typ: 'JWT' })}.${encodeSegment(payload)}.signature`;

describe('authService role handling', () => {
  beforeEach(() => {
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
});
