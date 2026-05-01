"use client";

import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Archive, CheckCircle, Database, RefreshCw, RotateCcw, Shield } from "lucide-react";
import { apiClient } from "@/lib/api/client";
import { useVMs } from "@/lib/api/hooks/useVMs";
import { cn } from "@/lib/utils";

type BackupStatus = {
  activeBackups: number;
  lastBackupTime: string;
  backupHealth: string;
  totalBackupSize: number;
};

const emptyStatus: BackupStatus = {
  activeBackups: 0,
  lastBackupTime: "",
  backupHealth: "unknown",
  totalBackupSize: 0,
};

function formatBytes(value: number) {
  if (!Number.isFinite(value) || value <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let next = value;
  let unit = 0;
  while (next >= 1024 && unit < units.length - 1) {
    next /= 1024;
    unit += 1;
  }
  return `${next.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function formatDate(value: string) {
  if (!value) return "not available";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

function healthVariant(health: string): "default" | "secondary" | "destructive" | "outline" {
  switch (health.toLowerCase()) {
    case "healthy":
      return "default";
    case "degraded":
    case "not_configured":
      return "secondary";
    case "failed":
    case "critical":
      return "destructive";
    default:
      return "outline";
  }
}

export function BackupRecoveryFlow() {
  const { items: vms, isLoading: vmsLoading, error: vmsError } = useVMs({ page: 1, pageSize: 100 });
  const [status, setStatus] = useState<BackupStatus>(emptyStatus);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadBackupStatus = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiClient.get<BackupStatus>("/api/v1/backup/status");
      setStatus(response || emptyStatus);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load backup status.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadBackupStatus();
  }, []);

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="text-3xl font-bold">Backup & Recovery</h2>
          <p className="text-gray-600 mt-1">
            Monitor backup readiness through canonical NovaCron backup contracts
          </p>
        </div>
        <Button variant="outline" onClick={loadBackupStatus} disabled={loading}>
          <RefreshCw className={cn("mr-2 h-4 w-4", loading && "animate-spin")} />
          Refresh
        </Button>
      </div>

      <Alert className="border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-950">
        <CheckCircle className="h-4 w-4 text-green-700 dark:text-green-300" />
        <AlertDescription className="text-green-800 dark:text-green-200">
          Backup health is backed by live `GET /api/v1/backup/status`. Policy, run, restore, and point-in-time recovery controls stay disabled until their canonical APIs exist.
        </AlertDescription>
      </Alert>

      {(error || Boolean(vmsError)) && (
        <Alert variant="destructive">
          <AlertDescription>
            {error || "Failed to load VM inventory from the canonical API."}
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
          value={vmsLoading ? "..." : String(vms.length)}
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
          <PendingContractCard
            title="Backup Policies"
            description="Policy create/update, manual backup run, retention, encryption, and storage target controls are disabled until canonical backup policy APIs are implemented."
          />
        </TabsContent>

        <TabsContent value="restore">
          <PendingContractCard
            title="Restore Operations"
            description="Restore-point browsing, target VM selection, rollback, and recovery execution are disabled until canonical restore APIs are implemented."
          />
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

function PendingContractCard({ title, description }: { title: string; description: string }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>Canonical contract pending</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="rounded-md border border-dashed p-6 text-sm text-muted-foreground">
          {description}
        </div>
      </CardContent>
    </Card>
  );
}
