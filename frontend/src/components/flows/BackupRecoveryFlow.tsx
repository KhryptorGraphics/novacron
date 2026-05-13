'use client';

import { type ReactNode, useEffect, useState } from 'react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Archive, CheckCircle, Database, RefreshCw, RotateCcw, Shield } from 'lucide-react';
import { backupApi, type BackupPolicy, type BackupRun, type BackupStatus } from '@/lib/api/backup';
import { useVMs } from '@/lib/api/hooks/useVMs';
import { cn } from '@/lib/utils';

const emptyStatus: BackupStatus = {
  activeBackups: 0,
  lastBackupTime: '',
  backupHealth: 'unknown',
  totalBackupSize: 0,
};

function formatBytes(value: number) {
  if (!Number.isFinite(value) || value <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let next = value;
  let unit = 0;
  while (next >= 1024 && unit < units.length - 1) {
    next /= 1024;
    unit += 1;
  }
  return `${next.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function formatDate(value: string) {
  if (!value) return 'not available';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

function healthVariant(health: string): 'default' | 'secondary' | 'destructive' | 'outline' {
  switch (health.toLowerCase()) {
    case 'healthy':
      return 'default';
    case 'degraded':
    case 'not_configured':
      return 'secondary';
    case 'failed':
    case 'critical':
      return 'destructive';
    default:
      return 'outline';
  }
}

export function BackupRecoveryFlow() {
  const { items: vms, isLoading: vmsLoading, error: vmsError } = useVMs({ page: 1, pageSize: 100 });
  const [status, setStatus] = useState<BackupStatus>(emptyStatus);
  const [policies, setPolicies] = useState<BackupPolicy[]>([]);
  const [backups, setBackups] = useState<BackupRun[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const loadBackupStatus = async () => {
    setLoading(true);
    setError(null);
    try {
      const [response, policyResponse, backupResponse] = await Promise.all([
        backupApi.getStatus(),
        backupApi.listPolicies(),
        backupApi.listBackups(),
      ]);
      setStatus(response || emptyStatus);
      setPolicies(Array.isArray(policyResponse) ? policyResponse : []);
      setBackups(Array.isArray(backupResponse) ? backupResponse : []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load backup status.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadBackupStatus();
  }, []);

  const createPolicy = async () => {
    setLoading(true);
    setError(null);
    setMessage(null);
    try {
      const created = await backupApi.createPolicy({
        name: `Daily VM policy ${policies.length + 1}`,
        schedule: 'daily',
        retentionDays: 30,
        target: 'local',
        enabled: true,
      });
      setPolicies((current) => [created, ...current]);
      setMessage('Backup policy created through the canonical backup API.');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create backup policy.');
    } finally {
      setLoading(false);
    }
  };

  const runPolicy = async (policyId: string) => {
    setLoading(true);
    setError(null);
    setMessage(null);
    try {
      const run = await backupApi.runPolicy(policyId);
      setBackups((current) => [run, ...current]);
      setMessage(`Backup run ${run.id} queued.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to run backup policy.');
    } finally {
      setLoading(false);
    }
  };

  const restoreBackup = async (backupId: string) => {
    setLoading(true);
    setError(null);
    setMessage(null);
    try {
      const restore = await backupApi.restore({ backupId, target: 'original' });
      setMessage(`Restore job ${restore.id} queued.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to queue restore.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="text-3xl font-bold">Backup & Recovery</h2>
          <p className="text-gray-600 mt-1">
            Operate backup policy, run, and restore contracts through canonical NovaCron APIs
          </p>
        </div>
        <Button variant="outline" onClick={loadBackupStatus} disabled={loading}>
          <RefreshCw className={cn('mr-2 h-4 w-4', loading && 'animate-spin')} />
          Refresh
        </Button>
      </div>

      <Alert className="border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-950">
        <CheckCircle className="h-4 w-4 text-green-700 dark:text-green-300" />
        <AlertDescription className="text-green-800 dark:text-green-200">
          Backup health, policies, manual runs, restore points, and recovery execution use live canonical backup contracts.
        </AlertDescription>
      </Alert>

      {message && (
        <Alert>
          <CheckCircle className="h-4 w-4" />
          <AlertDescription>{message}</AlertDescription>
        </Alert>
      )}

      {(error || Boolean(vmsError)) && (
        <Alert variant="destructive">
          <AlertDescription>
            {error || 'Failed to load VM inventory from the canonical API.'}
          </AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <StatusCard
          title="Backup Health"
          value={status.backupHealth}
          icon={<Shield className="h-4 w-4 text-muted-foreground" />}
          badge={<Badge variant={healthVariant(status.backupHealth)}>{status.backupHealth}</Badge>}
        />
        <StatusCard
          title="Active Backups"
          value={String(status.activeBackups)}
          icon={<Archive className="h-4 w-4 text-muted-foreground" />}
        />
        <StatusCard
          title="Protected VMs"
          value={vmsLoading ? '...' : String(vms.length)}
          icon={<Database className="h-4 w-4 text-muted-foreground" />}
        />
        <StatusCard
          title="Stored Backup Data"
          value={formatBytes(status.totalBackupSize)}
          icon={<RotateCcw className="h-4 w-4 text-muted-foreground" />}
        />
      </div>

      <Tabs defaultValue="status" className="space-y-4">
        <TabsList>
          <TabsTrigger value="status">Status</TabsTrigger>
          <TabsTrigger value="policies">Policies</TabsTrigger>
          <TabsTrigger value="restore">Restore</TabsTrigger>
        </TabsList>

        <TabsContent value="status">
          <Card>
            <CardHeader>
              <CardTitle>Backup Status Contract</CardTitle>
              <CardDescription>Live response from `/api/v1/backup/status`</CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <Detail label="Health" value={status.backupHealth} />
              <Detail label="Active jobs" value={String(status.activeBackups)} />
              <Detail label="Last backup" value={formatDate(status.lastBackupTime)} />
              <Detail label="Stored data" value={formatBytes(status.totalBackupSize)} />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="policies">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle>Backup Policies</CardTitle>
                <CardDescription>Live records from `/api/v1/backup/policies`</CardDescription>
              </div>
              <Button onClick={createPolicy} disabled={loading}>Create Policy</Button>
            </CardHeader>
            <CardContent className="space-y-3">
              {policies.length === 0 && (
                <div className="rounded-md border border-dashed p-6 text-sm text-muted-foreground">
                  {loading ? 'Loading policies...' : 'No backup policies returned by the canonical API.'}
                </div>
              )}
              {policies.map((policy) => (
                <div key={policy.id} className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <div className="font-medium">{policy.name}</div>
                    <div className="text-sm text-muted-foreground">
                      {policy.schedule}, {policy.retentionDays} days, target {policy.target}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">{policy.status}</Badge>
                    <Button variant="outline" size="sm" onClick={() => runPolicy(policy.id)} disabled={loading}>
                      Run
                    </Button>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="restore">
          <Card>
            <CardHeader>
              <CardTitle>Restore Operations</CardTitle>
              <CardDescription>Live records from `/api/v1/backup/backups` and `/api/v1/backup/restore`</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {backups.length === 0 && (
                <div className="rounded-md border border-dashed p-6 text-sm text-muted-foreground">
                  {loading ? 'Loading restore points...' : 'No backup runs returned by the canonical API.'}
                </div>
              )}
              {backups.map((backup) => (
                <div key={backup.id} className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <div className="font-medium">{backup.id}</div>
                    <div className="text-sm text-muted-foreground">
                      Policy {backup.policyId || 'unknown'} queued {formatDate(backup.createdAt)}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">{backup.status}</Badge>
                    <Button variant="outline" size="sm" onClick={() => restoreBackup(backup.id)} disabled={loading}>
                      Restore
                    </Button>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

function StatusCard({
  title,
  value,
  icon,
  badge,
}: {
  title: string;
  value: string;
  icon: ReactNode;
  badge?: ReactNode;
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {icon}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {badge && <div className="mt-2">{badge}</div>}
      </CardContent>
    </Card>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border p-4">
      <div className="text-sm text-muted-foreground">{label}</div>
      <div className="mt-1 font-medium">{value}</div>
    </div>
  );
}
