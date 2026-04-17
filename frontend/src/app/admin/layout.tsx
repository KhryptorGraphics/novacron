import AuthGuard from '@/components/auth/AuthGuard';
import RBACGuard from '@/components/auth/RBACGuard';

export default function AdminLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <AuthGuard>
      <RBACGuard requiredRoles={['admin', 'super-admin']}>
        {children}
      </RBACGuard>
    </AuthGuard>
  );
}
