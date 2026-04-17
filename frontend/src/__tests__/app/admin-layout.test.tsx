import type { ReactNode } from 'react';
import { render, screen } from '@testing-library/react';
import AdminLayout from '@/app/admin/layout';

jest.mock('@/components/auth/AuthGuard', () => ({
  __esModule: true,
  default: ({ children }: { children: ReactNode }) => (
    <div data-testid="auth-guard">{children}</div>
  ),
}));

jest.mock('@/components/auth/RBACGuard', () => ({
  __esModule: true,
  default: ({
    children,
    requiredRoles,
  }: {
    children: ReactNode;
    requiredRoles?: string[];
  }) => (
    <div data-testid="rbac-guard" data-roles={(requiredRoles || []).join(',')}>
      {children}
    </div>
  ),
}));

describe('AdminLayout', () => {
  it('wraps admin routes in auth and admin-role guards', () => {
    render(
      <AdminLayout>
        <div>Protected admin content</div>
      </AdminLayout>,
    );

    expect(screen.getByTestId('auth-guard')).toBeInTheDocument();
    expect(screen.getByTestId('rbac-guard')).toHaveAttribute('data-roles', 'admin,super-admin');
    expect(screen.getByText('Protected admin content')).toBeInTheDocument();
  });
});
