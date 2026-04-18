'use client';

import Link from 'next/link';
import { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  ArrowRight,
  Database,
  HardDrive,
  Loader2,
  Network,
  Server,
  Settings,
  Shield,
  Users,
} from 'lucide-react';

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { buildApiV1Url } from '@/lib/api/origin';
import { adminApi } from '@/lib/api/admin';
import { networkApi } from '@/lib/api/networks';
import { useSecurityMetrics } from '@/hooks/useSecurity';
import { useVolumes } from '@/hooks/useVolumes';
import { useVMs } from '@/lib/api/hooks/useVMs';

type MonitoringSummary = {
  currentCpuUsage: number;
  currentMemoryUsage: number;
  currentDiskUsage: number;
  currentNetworkUsage: number;
};

function OverviewCard({
  title,
  value,
  description,
  icon,
}: {
  title: string;
  value: string;
  description: string;
  icon: React.ReactNode;
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {icon}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-semibold">{value}</div>
        <p className="text-xs text-muted-foreground">{description}</p>
      </CardContent>
    </Card>
  );
}

export default function UnifiedDashboard() {
  const { items: vms, isLoading: vmsLoading, error: vmsError } = useVMs({ page: 1, pageSize: 100 });
  const { volumes, loading: volumesLoading, error: volumesError } = useVolumes();
  const { metrics: securityMetrics, loading: securityLoading, error: securityError } = useSecurityMetrics();
  const [monitoring, setMonitoring] = useState<MonitoringSummary | null>(null);
  const [monitoringError, setMonitoringError] = useState<string | null>(null);
  const [networkCount, setNetworkCount] = useState<number | null>(null);
  const [userCount, setUserCount] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadOverview() {
      try {
        const metricsResponse = await fetch(buildApiV1Url('/monitoring/metrics'));
        if (!metricsResponse.ok) {
          throw new Error(`Monitoring request failed with ${metricsResponse.status}`);
        }
        const metricsPayload = (await metricsResponse.json()) as MonitoringSummary;

        const [networks, users] = await Promise.all([
          networkApi.listNetworks().catch(() => []),
          adminApi.users.list({ page: 1, page_size: 1 }).catch(() => null),
        ]);

        if (!cancelled) {
          setMonitoring(metricsPayload);
          setNetworkCount(Array.isArray(networks) ? networks.length : null);
          setUserCount(users?.total ?? null);
        }
      } catch (requestError) {
        if (!cancelled) {
          setMonitoringError(
            requestError instanceof Error
              ? requestError.message
              : 'Failed to load canonical monitoring overview.',
          );
        }
      }
    }

    void loadOverview();
    return () => {
      cancelled = true;
    };
  }, []);

  const vmSummary = useMemo(() => {
    const total = vms.length;
    const running = vms.filter((vm) => vm.state === 'running').length;
    const stopped = vms.filter((vm) => vm.state === 'stopped').length;
    return { total, running, stopped };
  }, [vms]);

  const storageSummary = useMemo(() => {
    const provisioned = volumes.reduce((sum, volume) => sum + volume.size, 0);
    return {
      total: volumes.length,
      provisioned,
    };
  }, [volumes]);

  const statusNotes = [
    vmsError instanceof Error ? `VMs: ${vmsError.message}` : null,
    volumesError ? `Storage: ${volumesError}` : null,
    securityError ? `Security: ${securityError}` : null,
    monitoringError ? `Monitoring: ${monitoringError}` : null,
  ].filter(Boolean) as string[];

  const quickLinks = [
    { href: '/vms', label: 'Virtual Machines', description: 'Manage live VM inventory and actions', icon: Server },
    { href: '/monitoring', label: 'Monitoring', description: 'Inspect canonical metrics, alerts, and VM telemetry', icon: Activity },
    { href: '/storage', label: 'Storage', description: 'Use the volume-only GraphQL release surface', icon: HardDrive },
    { href: '/security', label: 'Security', description: 'Review live security posture and compliance', icon: Shield },
    { href: '/users', label: 'Users', description: 'Admin user management on the canonical `/api/admin/users*` surface', icon: Users },
    { href: '/network', label: 'Network', description: 'Minimal network inventory and VM interface operations', icon: Network },
    { href: '/analytics', label: 'Analytics', description: 'Current-state operational analytics without fabricated trends', icon: Database },
    { href: '/settings', label: 'Settings', description: 'Account and security preferences only', icon: Settings },
  ];

  return (
    <div className="space-y-6 p-6">
      <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <h1 className="text-3xl font-bold">Dashboard</h1>
          <p className="text-muted-foreground">
            Live operational overview for the release-candidate surface.
          </p>
        </div>
        <Badge variant="outline">No Mock Data</Badge>
      </div>

      {statusNotes.length > 0 && (
        <Alert>
          <AlertTitle>Some overview tiles are partially unavailable</AlertTitle>
          <AlertDescription>{statusNotes.join(' ')}</AlertDescription>
        </Alert>
      )}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <OverviewCard
          title="Virtual Machines"
          value={vmsLoading ? '…' : String(vmSummary.total)}
          description={`${vmSummary.running} running, ${vmSummary.stopped} stopped`}
          icon={<Server className="h-4 w-4 text-muted-foreground" />}
        />
        <OverviewCard
          title="Monitoring"
          value={monitoring ? `${monitoring.currentCpuUsage.toFixed(1)}% CPU` : '…'}
          description={monitoring ? `${monitoring.currentMemoryUsage.toFixed(1)}% memory in use` : 'Loading canonical metrics'}
          icon={<Activity className="h-4 w-4 text-muted-foreground" />}
        />
        <OverviewCard
          title="Storage"
          value={volumesLoading ? '…' : `${storageSummary.total} volumes`}
          description={`${storageSummary.provisioned.toLocaleString()} GiB provisioned`}
          icon={<HardDrive className="h-4 w-4 text-muted-foreground" />}
        />
        <OverviewCard
          title="Security"
          value={securityLoading || !securityMetrics ? '…' : `${securityMetrics.securityScore}%`}
          description={
            securityMetrics
              ? `${securityMetrics.activeThreats} active threats, ${securityMetrics.complianceScore}% compliance`
              : 'Loading security posture'
          }
          icon={<Shield className="h-4 w-4 text-muted-foreground" />}
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Live Signals</CardTitle>
            <CardDescription>Current-state summaries collected from the canonical APIs.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span>CPU usage</span>
                <span>{monitoring ? `${monitoring.currentCpuUsage.toFixed(1)}%` : 'Loading…'}</span>
              </div>
              <Progress value={monitoring?.currentCpuUsage ?? 0} />
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span>Memory usage</span>
                <span>{monitoring ? `${monitoring.currentMemoryUsage.toFixed(1)}%` : 'Loading…'}</span>
              </div>
              <Progress value={monitoring?.currentMemoryUsage ?? 0} />
            </div>
            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-lg border p-4">
                <div className="text-sm font-medium text-muted-foreground">Networks</div>
                <div className="mt-2 text-2xl font-semibold">{networkCount ?? '…'}</div>
              </div>
              <div className="rounded-lg border p-4">
                <div className="text-sm font-medium text-muted-foreground">Users</div>
                <div className="mt-2 text-2xl font-semibold">{userCount ?? '…'}</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Operational Notes</CardTitle>
            <CardDescription>The release dashboard is a launchpad, not a fake multi-tab control center.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-muted-foreground">
            <p>Historical analytics remain intentionally unavailable until a real analytics backend exists.</p>
            <p>Storage is volume-only on the routed release path.</p>
            <p>Network management is limited to inventory plus VM interface operations.</p>
            <p>Settings are narrowed to account and security preferences instead of global system configuration.</p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Quick Links</CardTitle>
          <CardDescription>Use the dedicated routed pages for deeper actions and detail.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          {quickLinks.map((link) => {
            const Icon = link.icon;
            return (
              <Link
                key={link.href}
                href={link.href}
                className="group rounded-lg border p-4 transition-colors hover:border-primary hover:bg-muted/40"
              >
                <div className="flex items-start justify-between">
                  <Icon className="h-5 w-5 text-muted-foreground" />
                  <ArrowRight className="h-4 w-4 text-muted-foreground transition-transform group-hover:translate-x-1" />
                </div>
                <div className="mt-4 space-y-1">
                  <div className="font-medium">{link.label}</div>
                  <p className="text-sm text-muted-foreground">{link.description}</p>
                </div>
              </Link>
            );
          })}
        </CardContent>
      </Card>
    </div>
  );
}
