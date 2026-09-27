import { render, screen, waitFor } from '@testing-library/react';

import LoginPage from '@/app/auth/login/page';
import { authService } from '@/lib/auth';

jest.mock('@/lib/auth', () => ({
  authService: {
    getServerInfo: jest.fn(),
    getGitHubAuthorizationUrl: jest.fn(),
  },
}));

jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: jest.fn(),
  }),
}));

jest.mock('@/components/ui/use-toast', () => ({
  useToast: () => ({
    toast: jest.fn(),
  }),
}));

jest.mock('@/hooks/useAuth', () => ({
  useAuth: () => ({
    login: jest.fn(),
    verify2FA: jest.fn(),
    requires2FA: false,
    tempToken: null,
  }),
}));

describe('LoginPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('hides the GitHub button when the backend does not advertise the github provider', async () => {
    (authService.getServerInfo as jest.Mock).mockResolvedValue({
      name: 'NovaCron API',
      version: '1.0.0',
      description: '',
      auth: { providers: ['password'] },
    });

    render(<LoginPage />);

    await waitFor(() => expect(authService.getServerInfo).toHaveBeenCalled());
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /continue with github/i })).toBeNull();
  });

  it('shows the GitHub button when the backend advertises the github provider', async () => {
    (authService.getServerInfo as jest.Mock).mockResolvedValue({
      name: 'NovaCron API',
      version: '1.0.0',
      description: '',
      auth: { providers: ['password', 'github'] },
    });

    render(<LoginPage />);

    expect(await screen.findByRole('button', { name: /continue with github/i })).toBeInTheDocument();
  });

  it('keeps the GitHub button hidden when /api/info is unreachable', async () => {
    (authService.getServerInfo as jest.Mock).mockRejectedValue(new Error('offline'));

    render(<LoginPage />);

    await waitFor(() => expect(authService.getServerInfo).toHaveBeenCalled());
    expect(screen.queryByRole('button', { name: /continue with github/i })).toBeNull();
  });
});
