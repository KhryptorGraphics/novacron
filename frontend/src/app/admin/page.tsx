'use client';

import { useEffect, useState } from 'react';
import { AlertTriangle, Bell, FileText, Loader2, Shield, UserCheck } from 'lucide-react';

import { ErrorBoundary } from '@/components/error-boundary';
import { SkipToMain } from '@/components/accessibility/a11y-components';
import RolePermissionManager from '@/components/admin/RolePermissionManager';
import SecurityComplianceDashboard from '@/components/security/SecurityComplianceDashboard';
import { MobileNavigation, DesktopSidebar } from '@/components/ui/mobile-navigation';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { LoadingStates, RefreshIndicator, DashboardSkeleton } from '@/components/ui/loading-states';
import { LazyTabs } from '@/components/ui/progressive-disclosure';
import { ThemeToggle } from '@/components/theme/theme-toggle';
import { useToast } from '@/components/ui/use-toast';
import { cn } from '@/lib/utils';
import { useAuth } from '@/hooks/useAuth';
import { securityAPI, type SecurityEvent } from '@/lib/api/security';

function CanonicalAuditPanel() {
  const { toast } = useToast();
  const [events, setEvents] = useState<SecurityEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadAuditTrail = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await securityAPI.getAuditTrail(50, 0);
      setEvents(response.events);
    } catch (auditError) {
      setError(auditError instanceof Error ? auditError.message : 'Failed to load canonical audit events.');
      setEvents([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadAuditTrail();
  }, []);

  const exportAudit = async () => {
    try {
      await securityAPI.exportSecurityReport('audit', 'json');
      toast({
        title: 'Audit export started',
        description: 'The canonical audit export download has started.',
      });
    } catch (exportError) {
      toast({
        title: 'Audit export failed',
        description: exportError instanceof Error ? exportError.message : 'Failed to export canonical audit events.',
        variant: 'destructive',
      });
    }
  };

  return (
    <Card>
      <CardHeader className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <CardTitle>Audit Trail</CardTitle>
          <CardDescription>Canonical audit events and export, scoped to the release candidate surface.</CardDescription>
        </div>
        <Button variant="outline" onClick={exportAudit}>
          <FileText className="mr-2 h-4 w-4" />
          Export Audit
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        {loading ? (
          <div className="flex items-center text-sm text-muted-foreground">
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Loading canonical audit events…
          </div>
        ) : null}
        {error ? <div className="text-sm text-destructive">{error}</div> : null}
        {!loading && !error && events.length === 0 ? (
          <div className="text-sm text-muted-foreground">No audit events available on the canonical backend.</div>
        ) : null}
        {!loading && !error && events.length > 0 ? (
          <div className="space-y-3">
            {events.map((event) => (
              <div key={event.id} className="rounded-lg border p-4">
                <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
                  <div>
                    <div className="font-medium">{event.action}</div>
                    <div className="text-sm text-muted-foreground">{event.details}</div>
                  </div>
                  <Badge variant="outline">{event.result}</Badge>
                </div>
                <div className="mt-3 grid gap-1 text-xs text-muted-foreground md:grid-cols-4">
                  <span>Type: {event.type}</span>
                  <span>Source: {event.source}</span>
                  <span>User: {event.user || 'system'}</span>
                  <span>{new Date(event.timestamp).toLocaleString()}</span>
                </div>
              </div>
            ))}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

export default function AdminDashboard() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const [loadingState, setLoadingState] = useState({
    isLoading: true,
    isError: false,
    isSuccess: false,
    message: '',
  });
  const { user, logout } = useAuth();
  const { toast } = useToast();

  useEffect(() => {
    const timer = setTimeout(() => {
      setLoadingState({
        isLoading: false,
        isError: false,
        isSuccess: true,
        message: '',
      });
    }, 400);

    return () => clearTimeout(timer);
  }, []);

  const handleRefresh = async () => {
    setRefreshing(true);
    await new Promise((resolve) => setTimeout(resolve, 500));
    setLastUpdated(new Date());
    setRefreshing(false);
  };

  const handleLogout = () => {
    logout();
    toast({ title: 'Logged out' });
  };

  const navigationUser = {
    name: [user?.firstName, user?.lastName].filter(Boolean).join(' ') || user?.email || 'Administrator',
    email: user?.email || '',
    role: user?.role || user?.roles?.[0] || 'admin',
  };

  const adminTabs = [
    {
      id: 'security',
      label: 'Security',
      icon: <Shield className="h-4 w-4" />,
      content: <SecurityComplianceDashboard />,
    },
    {
      id: 'permissions',
      label: 'Roles & Permissions',
      icon: <UserCheck className="h-4 w-4" />,
      content: <RolePermissionManager />,
    },
    {
      id: 'audit',
      label: 'Audit',
      icon: <FileText className="h-4 w-4" />,
      content: <CanonicalAuditPanel />,
    },
  ];

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        <SkipToMain />
        <MobileNavigation user={navigationUser} onLogout={handleLogout} />
        <DesktopSidebar user={navigationUser} collapsed={sidebarCollapsed} onCollapse={setSidebarCollapsed} />

        <main
          id="main-content"
          className={cn(
            'transition-all duration-300 lg:ml-64 pb-16 sm:pb-0',
            sidebarCollapsed && 'lg:ml-16',
          )}
        >
          <header className="sticky top-0 z-30 border-b bg-white dark:border-gray-800 dark:bg-gray-900">
            <div className="px-4 sm:px-6 lg:px-8">
              <div className="flex h-16 items-center justify-between">
                <div className="flex items-center gap-4">
                  <h1 className="text-2xl font-bold text-red-600 dark:text-red-400">Admin Dashboard</h1>
                  <Badge variant="destructive" className="animate-pulse">
                    <AlertTriangle className="mr-1 h-3 w-3" />
                    Elevated Access
                  </Badge>
                </div>

                <div className="flex items-center gap-2">
                  <RefreshIndicator
                    isRefreshing={refreshing}
                    onRefresh={handleRefresh}
                    lastUpdated={lastUpdated}
                  />
                  <Button variant="ghost" size="icon" className="relative">
                    <Bell className="h-5 w-5" />
                    <span className="absolute -top-1 -right-1 flex h-5 w-5 items-center justify-center rounded-full bg-red-500 text-xs text-white">
                      1
                    </span>
                  </Button>
                  <ThemeToggle />
                </div>
              </div>
            </div>
          </header>

          <div className="px-4 py-8 sm:px-6 lg:px-8">
            <LoadingStates state={loadingState} loadingComponent={<DashboardSkeleton />}>
              <LazyTabs tabs={adminTabs} defaultTab="security" className="w-full" />
            </LoadingStates>
          </div>
        </main>
      </div>
    </ErrorBoundary>
  );
}
