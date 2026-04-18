import { render, screen } from '@testing-library/react';
import RBACGuard from '@/components/auth/RBACGuard';
import { useRBAC } from '@/contexts/RBACContext';

jest.mock('@/contexts/RBACContext', () => ({
  useRBAC: jest.fn(),
}));

describe('RBACGuard', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('passes role and permission requirements through the canonical canAccess contract', () => {
    const canAccess = jest.fn().mockReturnValue(true);
    (useRBAC as jest.Mock).mockReturnValue({ canAccess });

    render(
      <RBACGuard
        requiredRoles={['admin']}
        requiredPermissions={[{ resource: 'security', action: 'read' }]}
      >
        <div>Protected content</div>
      </RBACGuard>,
    );

    expect(canAccess).toHaveBeenCalledWith(['admin'], [
      { resource: 'security', action: 'read' },
    ]);
    expect(screen.getByText('Protected content')).toBeInTheDocument();
  });

  it('renders the access denied state when the canonical RBAC context rejects access', () => {
    (useRBAC as jest.Mock).mockReturnValue({
      canAccess: jest.fn().mockReturnValue(false),
    });

    render(
      <RBACGuard requiredRoles={['admin']} fallback={<div>Access denied fallback</div>}>
        <div>Protected content</div>
      </RBACGuard>,
    );

    expect(screen.getByText('Access denied fallback')).toBeInTheDocument();
  });
});
