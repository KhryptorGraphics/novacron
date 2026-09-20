'use client';

export const dynamic = 'force-dynamic';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { Loader2, Play, RefreshCw, Server, XCircle } from 'lucide-react';

import AuthGuard from '@/components/auth/AuthGuard';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Textarea } from '@/components/ui/textarea';
import { useToast } from '@/components/ui/use-toast';
import {
  fabricApi,
  type FabricJob,
  type FabricJobDetail,
  type FabricJobRequest,
  type FabricNode,
  type FabricTransfer,
  type FabricTransferDetail,
} from '@/lib/api/fabric';

const POLL_INTERVAL_MS = 2500;
const ANY_NODE = 'any';

type StatusDisplay = {
  variant: 'default' | 'secondary' | 'success' | 'destructive' | 'outline';
  active: boolean;
};

// Only running/queued work keeps the page polling.
const STATUS_DISPLAY: Record<string, StatusDisplay> = {
  running: { variant: 'default', active: true },
  queued: { variant: 'secondary', active: true },
  completed: { variant: 'success', active: false },
  failed: { variant: 'destructive', active: false },
  cancelled: { variant: 'outline', active: false },
};

const UNKNOWN_STATUS: StatusDisplay = { variant: 'outline', active: false };

function statusDisplay(status: string): StatusDisplay {
  return STATUS_DISPLAY[status] ?? UNKNOWN_STATUS;
}

const BYTE_UNITS = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];

function formatBytes(value: number | null | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    return '—';
  }

  let scaled = value;
  let unit = 0;
  while (scaled >= 1024 && unit < BYTE_UNITS.length - 1) {
    scaled /= 1024;
    unit += 1;
  }

  return `${unit === 0 ? scaled : scaled.toFixed(1)} ${BYTE_UNITS[unit]}`;
}

type JobFormState = {
  command: string;
  args: string;
  env: string;
  nodeId: string;
  memoryMb: string;
  vcpus: string;
  bytesToMove: string;
};

const emptyJobForm: JobFormState = {
  command: '',
  args: '',
  env: '',
  nodeId: ANY_NODE,
  memoryMb: '',
  vcpus: '',
  bytesToMove: '',
};

// Whitespace-separated, with single/double quotes grouping one argument.
function splitArgs(input: string): string[] {
  const args: string[] = [];
  let current = '';
  let quote: string | null = null;

  for (const char of input) {
    if (quote) {
      if (char === quote) {
        quote = null;
      } else {
        current += char;
      }
      continue;
    }

    if (char === '"' || char === "'") {
      quote = char;
      continue;
    }

    if (/\s/.test(char)) {
      if (current) {
        args.push(current);
        current = '';
      }
      continue;
    }

    current += char;
  }

  if (current) {
    args.push(current);
  }

  return args;
}

function parseEnvLines(input: string): { env: Record<string, string>; error: string | null } {
  const env: Record<string, string> = {};

  for (const rawLine of input.split('\n')) {
    const line = rawLine.trim();
    if (!line) {
      continue;
    }

    const separator = line.indexOf('=');
    const key = separator > 0 ? line.slice(0, separator).trim() : '';

    if (!key) {
      return { env, error: `Environment line "${line}" is not KEY=VALUE.` };
    }

    env[key] = line.slice(separator + 1).trim();
  }

  return { env, error: null };
}

function parseOptionalInt(input: string, label: string): { value?: number; error: string | null } {
  const trimmed = input.trim();
  if (!trimmed) {
    return { error: null };
  }

  const parsed = Number(trimmed);
  if (!Number.isInteger(parsed) || parsed < 0) {
    return { error: `${label} must be a whole number of 0 or more.` };
  }

  return { value: parsed, error: null };
}

type BuildJobRequestResult =
  | { ok: true; request: FabricJobRequest }
  | { ok: false; error: string };

function buildJobRequest(form: JobFormState): BuildJobRequestResult {
  const command = form.command.trim();
  if (!command) {
    return { ok: false, error: 'A command is required to submit a fabric job.' };
  }

  const envResult = parseEnvLines(form.env);
  if (envResult.error) {
    return { ok: false, error: envResult.error };
  }

  const memory = parseOptionalInt(form.memoryMb, 'memory_mb');
  const vcpus = parseOptionalInt(form.vcpus, 'vcpus');
  const bytesToMove = parseOptionalInt(form.bytesToMove, 'bytes_to_move');
  const invalid = [memory.error, vcpus.error, bytesToMove.error].find((entry) => entry !== null);
  if (invalid) {
    return { ok: false, error: invalid };
  }

  const request: FabricJobRequest = { command };
  const args = splitArgs(form.args);
  if (args.length > 0) {
    request.args = args;
  }
  if (Object.keys(envResult.env).length > 0) {
    request.env = envResult.env;
  }
  if (form.nodeId !== ANY_NODE) {
    request.node_id = form.nodeId;
  }
  if (memory.value !== undefined) {
    request.memory_mb = memory.value;
  }
  if (vcpus.value !== undefined) {
    request.vcpus = vcpus.value;
  }
  if (bytesToMove.value !== undefined) {
    request.bytes_to_move = bytesToMove.value;
  }

  return { ok: true, request };
}

export default function FabricPage() {
  const { toast } = useToast();
  const [nodes, setNodes] = useState<FabricNode[]>([]);
  const [jobs, setJobs] = useState<FabricJob[]>([]);
  const [transfers, setTransfers] = useState<FabricTransfer[]>([]);
  const [nodesError, setNodesError] = useState<string | null>(null);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [transfersError, setTransfersError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [jobForm, setJobForm] = useState<JobFormState>(emptyJobForm);
  const [formError, setFormError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<FabricJobDetail | null>(null);
  const [jobDetailError, setJobDetailError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [selectedTransferId, setSelectedTransferId] = useState<string | null>(null);
  const [selectedTransfer, setSelectedTransfer] = useState<FabricTransferDetail | null>(null);
  const [transferDetailError, setTransferDetailError] = useState<string | null>(null);

  const loadNodes = useCallback(async () => {
    try {
      setNodes(await fabricApi.listNodes());
      setNodesError(null);
    } catch (error) {
      setNodesError(`Failed to load fabric nodes: ${error instanceof Error ? error.message : 'unknown error'}`);
    }
  }, []);

  const loadJobs = useCallback(async () => {
    try {
      setJobs(await fabricApi.listJobs());
      setJobsError(null);
    } catch (error) {
      setJobsError(`Failed to load fabric jobs: ${error instanceof Error ? error.message : 'unknown error'}`);
    }
  }, []);

  const loadTransfers = useCallback(async () => {
    try {
      setTransfers(await fabricApi.listTransfers());
      setTransfersError(null);
    } catch (error) {
      setTransfersError(`Failed to load transfers: ${error instanceof Error ? error.message : 'unknown error'}`);
    }
  }, []);

  const loadJobDetail = useCallback(async (jobId: string) => {
    try {
      setSelectedJob(await fabricApi.getJob(jobId));
      setJobDetailError(null);
    } catch (error) {
      setSelectedJob(null);
      setJobDetailError(`Failed to load job ${jobId}: ${error instanceof Error ? error.message : 'unknown error'}`);
    }
  }, []);

  const loadTransferDetail = useCallback(async (transferId: string) => {
    try {
      setSelectedTransfer(await fabricApi.getTransfer(transferId));
      setTransferDetailError(null);
    } catch (error) {
      setSelectedTransfer(null);
      setTransferDetailError(`Failed to load transfer ${transferId}: ${error instanceof Error ? error.message : 'unknown error'}`);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    const loadInitialState = async () => {
      await Promise.all([loadNodes(), loadJobs(), loadTransfers()]);
      if (!cancelled) {
        setLoading(false);
      }
    };

    void loadInitialState();

    return () => {
      cancelled = true;
    };
  }, [loadJobs, loadNodes, loadTransfers]);

  const selectedJobIsActive = selectedJob !== null && statusDisplay(selectedJob.status).active;
  const selectedTransferIsActive = selectedTransfer !== null && statusDisplay(selectedTransfer.status).active;
  const jobsAreActive = jobs.some((job) => statusDisplay(job.status).active) || selectedJobIsActive;
  const transfersAreActive = transfers.some((entry) => statusDisplay(entry.status).active) || selectedTransferIsActive;

  useEffect(() => {
    if (!jobsAreActive && !transfersAreActive) {
      return undefined;
    }

    const timer = window.setInterval(() => {
      if (jobsAreActive) {
        void loadJobs();
        if (selectedJobIsActive && selectedJobId) {
          void loadJobDetail(selectedJobId);
        }
      }
      if (transfersAreActive) {
        void loadTransfers();
        if (selectedTransferIsActive && selectedTransferId) {
          void loadTransferDetail(selectedTransferId);
        }
      }
    }, POLL_INTERVAL_MS);

    return () => window.clearInterval(timer);
  }, [
    jobsAreActive,
    loadJobDetail,
    loadJobs,
    loadTransferDetail,
    loadTransfers,
    selectedJobId,
    selectedJobIsActive,
    selectedTransferId,
    selectedTransferIsActive,
    transfersAreActive,
  ]);

  const runningJobs = useMemo(() => jobs.filter((job) => statusDisplay(job.status).active).length, [jobs]);

  const refreshAll = async () => {
    setRefreshing(true);
    await Promise.all([loadNodes(), loadJobs(), loadTransfers()]);
    if (selectedJobId) {
      await loadJobDetail(selectedJobId);
    }
    if (selectedTransferId) {
      await loadTransferDetail(selectedTransferId);
    }
    setRefreshing(false);
  };

  const selectJob = (jobId: string) => {
    setSelectedJobId(jobId);
    setSelectedJob(null);
    setJobDetailError(null);
    void loadJobDetail(jobId);
  };

  const selectTransfer = (transferId: string) => {
    setSelectedTransferId(transferId);
    setSelectedTransfer(null);
    setTransferDetailError(null);
    void loadTransferDetail(transferId);
  };

  const submitJob = async () => {
    const built = buildJobRequest(jobForm);
    if (!built.ok) {
      setFormError(built.error);
      return;
    }

    setFormError(null);
    setSubmitting(true);

    try {
      const submission = await fabricApi.submitJob(built.request);
      setJobForm(emptyJobForm);
      setSelectedJobId(submission.job_id);
      setSelectedJob(null);
      await Promise.all([loadJobs(), loadJobDetail(submission.job_id)]);
      toast({
        title: 'Job submitted',
        description: `${submission.job_id} is ${submission.status} on ${submission.node_id}${
          submission.placement?.reason ? ` — ${submission.placement.reason}` : ''
        }`,
      });
    } catch (error) {
      toast({
        title: 'Job submission failed',
        description: error instanceof Error ? error.message : 'Failed to submit the fabric job.',
        variant: 'destructive',
      });
    } finally {
      setSubmitting(false);
    }
  };

  const cancelSelectedJob = async () => {
    if (!selectedJobId) {
      return;
    }

    setCancelling(true);

    try {
      const result = await fabricApi.cancelJob(selectedJobId);
      await Promise.all([loadJobs(), loadJobDetail(selectedJobId)]);
      toast({
        title: result.cancelled ? 'Job cancelled' : 'Cancel rejected',
        description: result.cancelled
          ? `${selectedJobId} is now ${result.status}.`
          : result.error || `${selectedJobId} reported ${result.status}.`,
        variant: result.cancelled ? 'default' : 'destructive',
      });
    } catch (error) {
      toast({
        title: 'Cancel failed',
        description: error instanceof Error ? error.message : `Failed to cancel ${selectedJobId}.`,
        variant: 'destructive',
      });
    } finally {
      setCancelling(false);
    }
  };

  return (
    <AuthGuard>
      <div className="container mx-auto space-y-6 p-6">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Fabric</h1>
            <p className="text-muted-foreground">
              Bandwidth-aware peer compute fabric backed by `/api/cluster/nodes`, `/api/compute/jobs*` and `/api/transfers*`.
            </p>
          </div>
          <Button variant="outline" onClick={refreshAll} disabled={refreshing}>
            {refreshing ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-2 h-4 w-4" />}
            Refresh
          </Button>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Server className="h-4 w-4 text-muted-foreground" />
              Nodes
            </CardTitle>
            <CardDescription>
              Live capacity and link profiles from GET /api/cluster/nodes. A missing or stale link profile means the peer
              has not been probed recently.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {nodesError ? <div className="text-sm text-destructive">{nodesError}</div> : null}
            {loading ? (
              <div className="flex items-center text-sm text-muted-foreground">
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Loading fabric nodes…
              </div>
            ) : nodes.length === 0 ? (
              nodesError ? null : (
                <div className="text-sm text-muted-foreground">
                  No fabric nodes have registered yet. Local capacity appears once the api-server reports its node.
                </div>
              )
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Node</TableHead>
                    <TableHead>Arch</TableHead>
                    <TableHead>Cores</TableHead>
                    <TableHead>Memory</TableHead>
                    <TableHead>Storage</TableHead>
                    <TableHead>VMs</TableHead>
                    <TableHead>Reachable</TableHead>
                    <TableHead>Link</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {nodes.map((node) => (
                    <TableRow key={node.node_id}>
                      <TableCell>
                        <div className="font-medium">{node.node_id}</div>
                        <div className="text-xs text-muted-foreground">{node.addr}</div>
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline">{node.arch || '—'}</Badge>
                      </TableCell>
                      <TableCell>{node.cores}</TableCell>
                      <TableCell>
                        <div>{Math.max(0, node.mem_total_mb - node.mem_allocated_mb)} MB free</div>
                        <div className="text-xs text-muted-foreground">
                          {node.mem_total_mb} MB total · {node.mem_allocated_mb} MB allocated
                        </div>
                      </TableCell>
                      <TableCell>
                        <div>{node.storage_free_gb} GB free</div>
                        <div className="text-xs text-muted-foreground">{node.storage_total_gb} GB total</div>
                      </TableCell>
                      <TableCell>{node.vm_count}</TableCell>
                      <TableCell>
                        <Badge variant={node.reachable ? 'success' : 'destructive'}>
                          {node.reachable ? 'reachable' : 'unreachable'}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        {!node.link ? (
                          <Badge variant="warning">No link profile</Badge>
                        ) : node.link.stale ? (
                          <div className="space-y-1">
                            <div>{node.link.rtt_ms.toFixed(2)} ms</div>
                            <Badge variant="warning">Stale link profile</Badge>
                          </div>
                        ) : (
                          <div>
                            <div>{node.link.rtt_ms.toFixed(2)} ms</div>
                            <div className="text-xs text-muted-foreground">heartbeat {node.link.last_heartbeat}</div>
                          </div>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Submit a job</CardTitle>
            <CardDescription>
              POST /api/compute/jobs — the fabric scheduler places the job, or pins it to the selected node.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="job-command">Command</Label>
                <Input
                  id="job-command"
                  value={jobForm.command}
                  onChange={(event) => setJobForm((current) => ({ ...current, command: event.target.value }))}
                  placeholder="/bin/echo"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="job-args">Arguments</Label>
                <Input
                  id="job-args"
                  value={jobForm.args}
                  onChange={(event) => setJobForm((current) => ({ ...current, args: event.target.value }))}
                  placeholder={'hello "two words"'}
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="job-env">Environment</Label>
              <Textarea
                id="job-env"
                value={jobForm.env}
                onChange={(event) => setJobForm((current) => ({ ...current, env: event.target.value }))}
                placeholder={'LOG_LEVEL=debug\nRETRIES=3'}
              />
              <p className="text-xs text-muted-foreground">One KEY=VALUE per line. Blank lines are ignored.</p>
            </div>
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              <div className="space-y-2">
                <Label htmlFor="job-node">Pin to node</Label>
                <Select
                  value={jobForm.nodeId}
                  onValueChange={(value) => setJobForm((current) => ({ ...current, nodeId: value }))}
                >
                  <SelectTrigger id="job-node">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={ANY_NODE}>Any node</SelectItem>
                    {nodes.map((node) => (
                      <SelectItem key={node.node_id} value={node.node_id}>
                        {node.reachable ? node.node_id : `${node.node_id} (unreachable)`}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="job-memory">memory_mb</Label>
                <Input
                  id="job-memory"
                  type="number"
                  min={0}
                  value={jobForm.memoryMb}
                  onChange={(event) => setJobForm((current) => ({ ...current, memoryMb: event.target.value }))}
                  placeholder="512"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="job-vcpus">vcpus</Label>
                <Input
                  id="job-vcpus"
                  type="number"
                  min={0}
                  value={jobForm.vcpus}
                  onChange={(event) => setJobForm((current) => ({ ...current, vcpus: event.target.value }))}
                  placeholder="2"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="job-bytes">bytes_to_move</Label>
                <Input
                  id="job-bytes"
                  type="number"
                  min={0}
                  value={jobForm.bytesToMove}
                  onChange={(event) => setJobForm((current) => ({ ...current, bytesToMove: event.target.value }))}
                  placeholder="1048576"
                />
              </div>
            </div>
            {formError ? <div className="text-sm text-destructive">{formError}</div> : null}
            <Button onClick={submitJob} disabled={submitting}>
              {submitting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
              Submit job
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Jobs</CardTitle>
            <CardDescription>
              GET /api/compute/jobs — {runningJobs} running or queued. The list refreshes every {POLL_INTERVAL_MS / 1000} s
              while work is active.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {jobsError ? <div className="text-sm text-destructive">{jobsError}</div> : null}
            {loading ? (
              <div className="flex items-center text-sm text-muted-foreground">
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Loading fabric jobs…
              </div>
            ) : jobs.length === 0 ? (
              jobsError ? null : (
                <div className="text-sm text-muted-foreground">No fabric jobs have been submitted yet.</div>
              )
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Job</TableHead>
                    <TableHead>Command</TableHead>
                    <TableHead>Node</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Created</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {jobs.map((job) => (
                    <TableRow key={job.job_id}>
                      <TableCell>
                        <Button
                          variant="link"
                          className="h-auto p-0 font-medium"
                          onClick={() => selectJob(job.job_id)}
                        >
                          {job.job_id}
                        </Button>
                        {job.name ? <div className="text-xs text-muted-foreground">{job.name}</div> : null}
                      </TableCell>
                      <TableCell className="font-mono text-xs">{job.command}</TableCell>
                      <TableCell>{job.node_id || '—'}</TableCell>
                      <TableCell>
                        <Badge variant={statusDisplay(job.status).variant}>{job.status}</Badge>
                      </TableCell>
                      <TableCell className="text-xs text-muted-foreground">{job.created_at}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}

            {selectedJobId ? (
              <div className="rounded-lg border p-4">
                <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{selectedJobId}</span>
                      {selectedJob ? (
                        <Badge variant={statusDisplay(selectedJob.status).variant}>
                          {selectedJob.status}
                        </Badge>
                      ) : null}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Job detail from GET /api/compute/jobs/{selectedJobId}
                    </div>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={cancelSelectedJob}
                    disabled={cancelling || !selectedJobIsActive}
                  >
                    {cancelling ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <XCircle className="mr-2 h-4 w-4" />}
                    Cancel job
                  </Button>
                </div>

                {jobDetailError ? (
                  <div className="text-sm text-destructive">{jobDetailError}</div>
                ) : !selectedJob ? (
                  <div className="flex items-center text-sm text-muted-foreground">
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Loading job detail…
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="grid gap-4 md:grid-cols-4">
                      <div>
                        <div className="text-xs text-muted-foreground">Node</div>
                        <div>{selectedJob.node_id || '—'}</div>
                      </div>
                      <div>
                        <div className="text-xs text-muted-foreground">VM</div>
                        <div>{selectedJob.vm_id || '—'}</div>
                      </div>
                      <div className="md:col-span-2">
                        <div className="text-xs text-muted-foreground">Command</div>
                        <div className="font-mono text-xs">{selectedJob.command}</div>
                      </div>
                    </div>
                    {selectedJob.error ? <div className="text-sm text-destructive">{selectedJob.error}</div> : null}
                    <div className="grid gap-4 lg:grid-cols-2">
                      <div>
                        <div className="mb-2 text-sm font-medium">stdout (tail)</div>
                        <pre className="max-h-64 overflow-auto rounded-md border bg-muted/40 p-3 text-xs">
                          {selectedJob.logs?.stdout || 'No stdout recorded.'}
                        </pre>
                      </div>
                      <div>
                        <div className="mb-2 text-sm font-medium">stderr (tail)</div>
                        <pre className="max-h-64 overflow-auto rounded-md border bg-muted/40 p-3 text-xs">
                          {selectedJob.logs?.stderr || 'No stderr recorded.'}
                        </pre>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : null}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Transfers</CardTitle>
            <CardDescription>
              GET /api/transfers — measured bandwidth, compression choice and scheduling inputs for every fabric move.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {transfersError ? <div className="text-sm text-destructive">{transfersError}</div> : null}
            {loading ? (
              <div className="flex items-center text-sm text-muted-foreground">
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Loading transfers…
              </div>
            ) : transfers.length === 0 ? (
              transfersError ? null : (
                <div className="text-sm text-muted-foreground">
                  No transfers have been recorded yet. Transfers appear here once the fabric schedules a migration or data move.
                </div>
              )
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Transfer</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Moved</TableHead>
                    <TableHead>Measured rate</TableHead>
                    <TableHead>Compression</TableHead>
                    <TableHead>ETA</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {transfers.map((entry) => (
                    <TableRow key={entry.transfer_id}>
                      <TableCell>
                        <Button
                          variant="link"
                          className="h-auto p-0 font-medium"
                          onClick={() => selectTransfer(entry.transfer_id)}
                        >
                          {entry.transfer_id}
                        </Button>
                        {entry.target_node ? (
                          <div className="text-xs text-muted-foreground">target {entry.target_node}</div>
                        ) : null}
                      </TableCell>
                      <TableCell>
                        <Badge variant={statusDisplay(entry.status).variant}>{entry.status}</Badge>
                      </TableCell>
                      <TableCell>
                        {formatBytes(entry.bytes_moved)} / {formatBytes(entry.bytes_total)}
                      </TableCell>
                      <TableCell>
                        {typeof entry.measured_bps === 'number' ? `${formatBytes(entry.measured_bps)}/s` : '—'}
                      </TableCell>
                      <TableCell>{entry.compression || '—'}</TableCell>
                      <TableCell>
                        {typeof entry.eta_seconds === 'number' ? `${entry.eta_seconds} s` : '—'}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}

            {selectedTransferId ? (
              <div className="rounded-lg border p-4">
                <div className="mb-3">
                  <div className="flex items-center gap-2">
                    <span className="font-medium">{selectedTransferId}</span>
                    {selectedTransfer ? (
                      <Badge variant={statusDisplay(selectedTransfer.status).variant}>
                        {selectedTransfer.status}
                      </Badge>
                    ) : null}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Transfer detail from GET /api/transfers/{selectedTransferId}
                  </div>
                </div>

                {transferDetailError ? (
                  <div className="text-sm text-destructive">{transferDetailError}</div>
                ) : !selectedTransfer ? (
                  <div className="flex items-center text-sm text-muted-foreground">
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Loading transfer detail…
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="grid gap-4 md:grid-cols-4">
                      <div>
                        <div className="text-xs text-muted-foreground">Target node</div>
                        <div>{selectedTransfer.target_node || '—'}</div>
                      </div>
                      <div>
                        <div className="text-xs text-muted-foreground">Kind</div>
                        <div>{selectedTransfer.kind || '—'}</div>
                      </div>
                      <div>
                        <div className="text-xs text-muted-foreground">Compression</div>
                        <div>{selectedTransfer.compression || '—'}</div>
                      </div>
                      <div>
                        <div className="text-xs text-muted-foreground">ETA</div>
                        <div>{typeof selectedTransfer.eta_seconds === 'number' ? `${selectedTransfer.eta_seconds} s` : '—'}</div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Moved</span>
                        <span>
                          {formatBytes(selectedTransfer.bytes_moved)} / {formatBytes(selectedTransfer.bytes_total)}
                        </span>
                      </div>
                      {typeof selectedTransfer.bytes_total === 'number' && selectedTransfer.bytes_total > 0 ? (
                        <Progress
                          value={Math.min(
                            100,
                            (Math.max(0, selectedTransfer.bytes_moved ?? 0) / selectedTransfer.bytes_total) * 100,
                          )}
                        />
                      ) : null}
                      <div className="text-sm">
                        Measured rate: {typeof selectedTransfer.measured_bps === 'number' ? `${formatBytes(selectedTransfer.measured_bps)}/s` : '—'}
                      </div>
                    </div>

                    <div>
                      <div className="mb-2 text-sm font-medium">Decision inputs</div>
                      {selectedTransfer.decision_inputs ? (
                        <div className="grid gap-4 md:grid-cols-3">
                          <div>
                            <div className="text-xs text-muted-foreground">link_bps</div>
                            <div>{`${formatBytes(selectedTransfer.decision_inputs.link_bps)}/s`}</div>
                          </div>
                          <div>
                            <div className="text-xs text-muted-foreground">sample_ratio</div>
                            <div>{selectedTransfer.decision_inputs.sample_ratio}</div>
                          </div>
                          <div>
                            <div className="text-xs text-muted-foreground">threshold</div>
                            <div>{selectedTransfer.decision_inputs.threshold}</div>
                          </div>
                        </div>
                      ) : (
                        <div className="text-sm text-muted-foreground">
                          The server has not reported decision inputs for this transfer yet.
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ) : null}
          </CardContent>
        </Card>
      </div>
    </AuthGuard>
  );
}