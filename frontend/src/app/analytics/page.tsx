"use client";

export const dynamic = "force-dynamic";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { Activity, ArrowRight, Database, HardDrive, Loader2, Network, Server, Shield } from "lucide-react";

import AuthGuard from "@/components/auth/AuthGuard";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { buildApiV1Url } from "@/lib/api/origin";
import { networkApi } from "@/lib/api/networks";
import { useSecurityMetrics } from "@/hooks/useSecurity";
import { useVolumes } from "@/hooks/useVolumes";
import { useVMs } from "@/lib/api/hooks/useVMs";

type MonitoringSummary = {
  currentCpuUsage: number;
  currentMemoryUsage: number;
  currentDiskUsage: number;
  currentNetworkUsage: number;
};

function MetricPanel({
  title,
  value,
  description,
  progress,
  icon,
}: {
  title: string;
  value: string;
  description: string;
  progress?: number;
  icon: React.ReactNode;
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {icon}
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="text-2xl font-semibold">{value}</div>
        {progress !== undefined ? <Progress value={progress} /> : null}
        <p className="text-xs text-muted-foreground">{description}</p>
      </CardContent>
    </Card>
  );
}

export default function AnalyticsPage() {
  const { items: vms, isLoading: vmsLoading } = useVMs({ page: 1, pageSize: 100 });
  const { volumes, loading: volumesLoading } = useVolumes();
  const { metrics: securityMetrics, loading: securityLoading } = useSecurityMetrics();
  const [monitoring, setMonitoring] = useState<MonitoringSummary | null>(null);
  const [networkCount, setNetworkCount] = useState<number | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadAnalytics() {
      try {
        const [metricsResponse, networks] = await Promise.all([
          fetch(buildApiV1Url('/monitoring/metrics')),
          networkApi.listNetworks().catch(() => []),
        ]);

        if (!metricsResponse.ok) {
          throw new Error(`Monitoring request failed with ${metricsResponse.status}`);
        }

        const monitoringPayload = (await metricsResponse.json()) as MonitoringSummary;
        if (!cancelled) {
          setMonitoring(monitoringPayload);
          setNetworkCount(Array.isArray(networks) ? networks.length : 0);
        }
      } catch (requestError) {
        if (!cancelled) {
          setLoadError(
            requestError instanceof Error
              ? requestError.message
              : 'Failed to load the operational analytics summary.',
          );
        }
      }
    }

    void loadAnalytics();
    return () => {
      cancelled = true;
    };
  }, []);

  const vmSummary = useMemo(() => ({
    total: vms.length,
    running: vms.filter((vm) => vm.state === 'running').length,
    stopped: vms.filter((vm) => vm.state === 'stopped').length,
  }), [vms]);

  const storageSummary = useMemo(() => ({
    totalVolumes: volumes.length,
    totalProvisioned: volumes.reduce((sum, volume) => sum + volume.size, 0),
  }), [volumes]);

  return (
    <AuthGuard>
      <div className="container mx-auto space-y-6 p-6">
        <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
          <div>
            <h1 className="text-3xl font-bold">Analytics</h1>
            <p className="text-muted-foreground">
              Current-state operational analytics composed from live canonical APIs only.
            </p>
          </div>
          <Badge variant="outline">Read Only</Badge>
        </div>

        <Alert>
          <Database className="h-4 w-4" />
          <AlertTitle>Historical trends are intentionally unavailable</AlertTitle>
          <AlertDescription>
            This page no longer fabricates time-series charts. It summarizes live VM, monitoring, storage, security, and network state instead.
          </AlertDescription>
        </Alert>

        {loadError ? (
          <Alert variant="destructive">
            <AlertTitle>Analytics load failed</AlertTitle>
            <AlertDescription>{loadError}</AlertDescription>
          </Alert>
        ) : null}

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <MetricPanel
            title="CPU"
            value={monitoring ? `${monitoring.currentCpuUsage.toFixed(1)}%` : vmsLoading ? '…' : 'n/a'}
            description="Live aggregate CPU usage from `/api/v1/monitoring/metrics`."
            progress={monitoring?.currentCpuUsage}
            icon={<Activity className="h-4 w-4 text-muted-foreground" />}
          />
          <MetricPanel
            title="Memory"
            value={monitoring ? `${monitoring.currentMemoryUsage.toFixed(1)}%` : '…'}
            description="Current memory allocation on the canonical monitoring surface."
            progress={monitoring?.currentMemoryUsage}
            icon={<Server className="h-4 w-4 text-muted-foreground" />}
          />
          <MetricPanel
            title="Storage"
            value={volumesLoading ? '…' : `${storageSummary.totalVolumes} volumes`}
            description={`${storageSummary.totalProvisioned.toLocaleString()} GiB provisioned.`}
            icon={<HardDrive className="h-4 w-4 text-muted-foreground" />}
          />
          <MetricPanel
            title="Security"
            value={securityLoading || !securityMetrics ? '…' : `${securityMetrics.securityScore}%`}
            description={securityMetrics ? `${securityMetrics.activeThreats} active threats.` : 'Loading live security posture.'}
            progress={securityMetrics?.complianceScore}
            icon={<Shield className="h-4 w-4 text-muted-foreground" />}
          />
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Operational Snapshot</CardTitle>
              <CardDescription>Raw current-state counts pulled from the active release surfaces.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4 sm:grid-cols-2">
              <div className="rounded-lg border p-4">
                <div className="text-sm text-muted-foreground">Virtual machines</div>
                <div className="mt-2 text-2xl font-semibold">{vmSummary.total}</div>
                <div className="mt-2 text-sm text-muted-foreground">{vmSummary.running} running / {vmSummary.stopped} stopped</div>
              </div>
              <div className="rounded-lg border p-4">
                <div className="text-sm text-muted-foreground">Networks</div>
                <div className="mt-2 text-2xl font-semibold">{networkCount ?? '…'}</div>
                <div className="mt-2 text-sm text-muted-foreground">Minimal live inventory only</div>
              </div>
              <div className="rounded-lg border p-4">
                <div className="text-sm text-muted-foreground">Disk usage</div>
                <div className="mt-2 text-2xl font-semibold">
                  {monitoring ? `${monitoring.currentDiskUsage.toFixed(1)}%` : '…'}
                </div>
              </div>
              <div className="rounded-lg border p-4">
                <div className="text-sm text-muted-foreground">Network usage</div>
                <div className="mt-2 text-2xl font-semibold">
                  {monitoring ? `${monitoring.currentNetworkUsage.toFixed(1)}` : '…'}
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Next Actions</CardTitle>
              <CardDescription>Use dedicated pages for deeper operational detail and mutations.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {[
                { href: '/monitoring', label: 'Open Monitoring', note: 'Investigate live metrics, alerts, and VM telemetry.' },
                { href: '/storage', label: 'Open Storage', note: 'Manage the volume-only GraphQL release surface.' },
                { href: '/security', label: 'Open Security', note: 'Review compliance and active security events.' },
                { href: '/network', label: 'Open Network', note: 'Inspect network inventory and VM interfaces.' },
              ].map((entry) => (
                <Link
                  key={entry.href}
                  href={entry.href}
                  className="flex items-center justify-between rounded-lg border p-4 transition-colors hover:bg-muted/40"
                >
                  <div>
                    <div className="font-medium">{entry.label}</div>
                    <div className="text-sm text-muted-foreground">{entry.note}</div>
                  </div>
                  <ArrowRight className="h-4 w-4 text-muted-foreground" />
                </Link>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>
    </AuthGuard>
  );
}
