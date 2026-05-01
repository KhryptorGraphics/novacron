"use client";

import { useEffect, useState } from "react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { CheckCircle, RefreshCw, Send, Server } from "lucide-react";
import { apiClient } from "@/lib/api/client";
import { useVMs } from "@/lib/api/hooks/useVMs";
import { cn } from "@/lib/utils";

type MigrationJob = {
  id?: string;
  jobId?: string;
  status: string;
  sourceCluster?: string;
  targetCluster?: string;
  vmCount?: number;
  migrationStrategy?: string;
  createdAt?: string;
  startTime?: string;
};

function jobId(job: MigrationJob) {
  return job.jobId || job.id || "unknown";
}

function formatDate(value?: string) {
  if (!value) return "unknown";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

export function MigrationWorkflow() {
  const { items: vms, isLoading: vmsLoading, error: vmsError } = useVMs({ page: 1, pageSize: 100 });
  const [selectedVMs, setSelectedVMs] = useState<string[]>([]);
  const [sourceCluster, setSourceCluster] = useState("local");
  const [targetCluster, setTargetCluster] = useState("");
  const [migrationStrategy, setMigrationStrategy] = useState("cold");
  const [jobs, setJobs] = useState<MigrationJob[]>([]);
  const [loadingJobs, setLoadingJobs] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadJobs = async () => {
    setLoadingJobs(true);
    setError(null);
    try {
      const response = await apiClient.get<MigrationJob[]>("/api/v1/migration/jobs");
      setJobs(Array.isArray(response) ? response : []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load migration jobs.");
    } finally {
      setLoadingJobs(false);
    }
  };

  useEffect(() => {
    loadJobs();
  }, []);

  const toggleVM = (vmId: string) => {
    setSelectedVMs((current) =>
      current.includes(vmId)
        ? current.filter((id) => id !== vmId)
        : [...current, vmId]
    );
  };

  const submitMigration = async () => {
    if (!targetCluster.trim()) {
      setError("Target cluster is required.");
      return;
    }
    if (selectedVMs.length === 0) {
      setError("Select at least one VM.");
      return;
    }

    setSubmitting(true);
    setError(null);
    setMessage(null);
    try {
      const created = await apiClient.post<MigrationJob>("/api/v1/migration/initiate", {
        sourceCluster: sourceCluster.trim() || "local",
        targetCluster: targetCluster.trim(),
        vmIds: selectedVMs,
        migrationStrategy,
      });
      setJobs((current) => [created, ...current]);
      setSelectedVMs([]);
      setMessage(`Migration ${jobId(created)} queued through the canonical API.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to queue migration.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="text-3xl font-bold">VM Migration Workflow</h2>
          <p className="text-gray-600 mt-1">
            Queue VM migrations through the canonical migration API
          </p>
        </div>
        <Button variant="outline" onClick={loadJobs} disabled={loadingJobs}>
          <RefreshCw className={cn("mr-2 h-4 w-4", loadingJobs && "animate-spin")} />
          Refresh Jobs
        </Button>
      </div>

      <Alert className="border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-950">
        <CheckCircle className="h-4 w-4 text-green-700 dark:text-green-300" />
        <AlertDescription className="text-green-800 dark:text-green-200">
          This workflow uses live VM inventory and `POST /api/v1/migration/initiate`. Detailed migration planning and preflight APIs are still pending.
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
            {error || "Failed to load VM inventory from the canonical API."}
          </AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle>Migration Request</CardTitle>
            <CardDescription>Canonical request payload</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="source-cluster">Source Cluster</Label>
              <Input
                id="source-cluster"
                value={sourceCluster}
                onChange={(event) => setSourceCluster(event.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="target-cluster">Target Cluster</Label>
              <Input
                id="target-cluster"
                value={targetCluster}
                onChange={(event) => setTargetCluster(event.target.value)}
                placeholder="trusted-west"
              />
            </div>
            <div className="space-y-2">
              <Label>Migration Strategy</Label>
              <Select value={migrationStrategy} onValueChange={setMigrationStrategy}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="cold">Cold migration</SelectItem>
                  <SelectItem value="checkpoint">Checkpoint and restore</SelectItem>
                  <SelectItem value="live">Live migration feature gate</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <Button
              className="w-full"
              onClick={submitMigration}
              disabled={submitting || vmsLoading || selectedVMs.length === 0}
            >
              <Send className="mr-2 h-4 w-4" />
              {submitting ? "Queuing..." : "Queue Migration"}
            </Button>
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Source VMs</CardTitle>
            <CardDescription>Live records from `/api/v1/vms`</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {vms.length === 0 && (
                <div className="rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">
                  {vmsLoading ? "Loading VMs..." : "No VMs returned by the canonical API."}
                </div>
              )}

              {vms.map((vm) => (
                <label
                  key={vm.id}
                  className="flex cursor-pointer items-center justify-between rounded-lg border p-4 hover:bg-muted/50"
                >
                  <div className="flex items-center gap-3">
                    <Checkbox
                      checked={selectedVMs.includes(vm.id)}
                      onCheckedChange={() => toggleVM(vm.id)}
                    />
                    <Server className="h-4 w-4 text-muted-foreground" />
                    <div>
                      <div className="font-medium">{vm.name}</div>
                      <div className="text-sm text-muted-foreground">
                        {vm.id} on {vm.node_id || "unassigned"}
                      </div>
                    </div>
                  </div>
                  <Badge variant={vm.state === "running" ? "default" : "secondary"}>
                    {vm.state}
                  </Badge>
                </label>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Migration Jobs</CardTitle>
          <CardDescription>Live queue state from `/api/v1/migration/jobs`</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Job</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Source</TableHead>
                <TableHead>Target</TableHead>
                <TableHead>VMs</TableHead>
                <TableHead>Strategy</TableHead>
                <TableHead>Created</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {jobs.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="text-center text-sm text-muted-foreground">
                    {loadingJobs ? "Loading migration jobs..." : "No migration jobs returned by the canonical API."}
                  </TableCell>
                </TableRow>
              ) : (
                jobs.map((job) => (
                  <TableRow key={jobId(job)}>
                    <TableCell className="font-mono text-sm">{jobId(job)}</TableCell>
                    <TableCell>
                      <Badge variant="outline">{job.status}</Badge>
                    </TableCell>
                    <TableCell>{job.sourceCluster || "unknown"}</TableCell>
                    <TableCell>{job.targetCluster || "unknown"}</TableCell>
                    <TableCell>{job.vmCount ?? "unknown"}</TableCell>
                    <TableCell>{job.migrationStrategy || "unknown"}</TableCell>
                    <TableCell>{formatDate(job.createdAt || job.startTime)}</TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
