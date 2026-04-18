import { render, screen, waitFor } from '@testing-library/react';

import UsersPage from '@/app/users/page';

jest.mock('@/components/auth/AuthGuard', () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

jest.mock('@/components/ui/use-toast', () => ({
  useToast: () => ({ toast: jest.fn() }),
}));

jest.mock('@/hooks/useAuth', () => ({
  useAuth: () => ({
    user: {
      role: 'admin',
      roles: ['admin'],
    },
  }),
}));

jest.mock('@/lib/api/admin', () => ({
  adminApi: {
    users: {
      list: jest.fn().mockResolvedValue({
        users: [
          {
            id: 1,
            username: 'alice',
            email: 'alice@novacron.test',
            role: 'admin',
            active: true,
            created_at: '2026-04-18T00:00:00Z',
            updated_at: '2026-04-18T00:00:00Z',
          },
        ],
      }),
      create: jest.fn(),
      update: jest.fn(),
      delete: jest.fn(),
      assignRoles: jest.fn(),
    },
  },
}));

describe('UsersPage', () => {
  it('renders the canonical admin user inventory', async () => {
    render(<UsersPage />);

    expect(screen.getByText('Users')).toBeInTheDocument();
    expect(screen.getByText(/admin-only user management/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('alice')).toBeInTheDocument();
      expect(screen.getByText('alice@novacron.test')).toBeInTheDocument();
    });
  });
});
