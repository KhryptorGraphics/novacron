/**
 * DWCP TypeScript SDK - Fabric Compute
 *
 * The fabric surface is the NovaCron api-server's HTTP/JSON API (authed with a
 * Bearer JWT). It is a different transport from the DWCP TCP protocol in
 * ./client — jobs are placed on the node the capacity/bandwidth cost picks and
 * canceled through the peer that owns their VM — so it is a separate client
 * with its own config and auth, sharing the SDK's error model.
 *
 * Endpoint shapes are the fixed fabric contract v1 (2026-09-20).
 */

import {
  AuthenticationError,
  ConnectionError,
  DWCPError,
  TimeoutError,
} from './client';

// Default request timeout, mirroring the DWCP client's defaultConfig.
export const DEFAULT_FABRIC_TIMEOUT = 60000;

// Link profile of a peer, as measured by the cluster heartbeat loop.
export interface FabricLink {
  rtt_ms: number;
  last_heartbeat: string;
  stale: boolean;
}

// One node's live capacity and VM reservation.
export interface FabricNode {
  node_id: string;
  /** Dial address; absent for the local node. */
  addr?: string;
  arch: string;
  cores: number;
  mem_total_mb: number;
  mem_allocated_mb: number;
  storage_total_gb: number;
  storage_free_gb: number;
  vm_count: number;
  reachable: boolean;
  /** null for the local node / a peer that has never been probed. */
  link: FabricLink | null;
}

// Job outcome mirrored from the VM lifecycle that executes it. "pending" is the
// window between create and start.
export type FabricJobStatus =
  | 'pending'
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled';

// Why the job landed on its node. cost_estimate_s is the placement cost in
// seconds (Go: placementDecision.cost_estimate_s, omitempty).
export interface FabricPlacement {
  decision: string;
  cost_estimate_s?: number;
  reason: string;
}

// Submit payload for a fabric job.
export interface FabricJobRequest {
  command: string;
  args?: string[];
  env?: Record<string, string>;
  /** Pin the job to a node; an unreachable pin fails, it is never moved. */
  node_id?: string;
  bytes_to_move?: number;
  memory_mb?: number;
  vcpus?: number;
  name?: string;
}

export interface FabricJobSubmission {
  job_id: string;
  vm_id: string;
  node_id: string;
  status: FabricJobStatus;
  placement: FabricPlacement;
}

// Tails (<=64KiB) of the job process' captured output.
export interface FabricJobLogs {
  stdout: string;
  stderr: string;
}

export interface FabricJob {
  job_id: string;
  name: string;
  command: string;
  status: FabricJobStatus;
  node_id: string;
  vm_id: string;
  created_at: string;
  error?: string;
  /** Present on getJob: the executing node's captured process output. */
  logs?: FabricJobLogs;
}

export interface FabricJobCancelResult {
  cancelled: boolean;
  status: 'cancelled' | 'failed-to-cancel';
  error?: string;
}

export type TransferKind = 'migration' | 'data' | 'job';
export type TransferCompression = 'none' | 'zstd-multifd' | 'xbzrle';
export type TransferStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled';

// Submit payload for a bulk data move between nodes.
export interface TransferRequest {
  kind: TransferKind;
  target_node: string;
  vm_id?: string;
  job_id?: string;
  bytes_estimated?: number;
}

export interface TransferSubmission {
  transfer_id: string;
  status: 'queued' | 'running';
  eta_seconds?: number;
  queue_position?: number;
}

// The inputs the transfer's compression/stream decision was made from.
export interface TransferDecisionInputs {
  link_bps: number;
  sample_ratio: number;
  threshold: number;
}

export interface Transfer {
  transfer_id: string;
  status: TransferStatus;
  bytes_total: number;
  bytes_moved: number;
  measured_bps: number;
  compression: TransferCompression;
  eta_seconds?: number;
  decision_inputs: TransferDecisionInputs;
}

// Fabric client configuration
export interface FabricClientConfig {
  /** api-server origin, e.g. "http://127.0.0.1:18090". */
  baseUrl: string;
  /** JWT from POST /api/auth/login to authenticate with. */
  token?: string;
  /** Credentials used to log in lazily on the first authed call. */
  email?: string;
  password?: string;
  requestTimeout?: number;
  /** fetch implementation; defaults to the global fetch (Node >=18, browser). */
  fetch?: typeof fetch;
}

// Error for a non-2xx fabric response. The parsed body is kept so callers can
// read fields the status code alone does not carry (e.g. cancel's
// {"cancelled":false,"status":"failed-to-cancel","error":...}).
export class FabricAPIError extends DWCPError {
  constructor(
    message: string,
    public readonly status: number,
    public readonly body?: unknown
  ) {
    super(message);
    this.name = 'FabricAPIError';
  }
}

// Fabric client: node inventory, compute jobs, and transfers.
export class FabricClient {
  private baseUrl: string;
  private token?: string;
  private email?: string;
  private password?: string;
  private requestTimeout: number;
  private fetchImpl?: typeof fetch;

  constructor(config: FabricClientConfig) {
    if (!config.baseUrl) {
      throw new DWCPError('baseUrl is required');
    }

    this.baseUrl = config.baseUrl.replace(/\/+$/, '');
    this.token = config.token;
    this.email = config.email;
    this.password = config.password;
    this.requestTimeout = config.requestTimeout || DEFAULT_FABRIC_TIMEOUT;
    this.fetchImpl =
      config.fetch ||
      (typeof fetch === 'function' ? fetch : undefined);
  }

  /**
   * Exchange the configured credentials for a JWT and adopt it for later
   * calls. Called automatically on the first authed call when email/password
   * are configured.
   */
  async login(): Promise<string> {
    if (!this.email || !this.password) {
      throw new DWCPError('email and password are required to login');
    }

    const data = await this.request<{ token?: string }>(
      'POST',
      '/api/auth/login',
      { email: this.email, password: this.password },
      { authenticated: false }
    );

    if (!data || typeof data.token !== 'string' || data.token === '') {
      throw new AuthenticationError('Login response did not include a token');
    }

    this.token = data.token;
    return this.token;
  }

  // Nodes in the fabric with their live capacity and link profile.
  async listNodes(): Promise<FabricNode[]> {
    const data = await this.request<{ nodes: FabricNode[] }>(
      'GET',
      '/api/cluster/nodes'
    );

    return data.nodes;
  }

  // Place and start a job; the response carries the placement decision.
  async submitJob(req: FabricJobRequest): Promise<FabricJobSubmission> {
    return this.request<FabricJobSubmission>(
      'POST',
      '/api/compute/jobs',
      req
    );
  }

  // One job; status is derived live from its VM, logs are its output tails.
  async getJob(id: string): Promise<FabricJob> {
    return this.request<FabricJob>(
      'GET',
      `/api/compute/jobs/${encodeURIComponent(id)}`
    );
  }

  // Recent jobs, newest first.
  async listJobs(): Promise<FabricJob[]> {
    const data = await this.request<{ jobs: FabricJob[] }>(
      'GET',
      '/api/compute/jobs'
    );

    return data.jobs;
  }

  // Stop the job's VM on the node that owns it.
  async cancelJob(id: string): Promise<FabricJobCancelResult> {
    return this.request<FabricJobCancelResult>(
      'POST',
      `/api/compute/jobs/${encodeURIComponent(id)}/cancel`
    );
  }

  // Transfers in flight.
  async listTransfers(): Promise<Transfer[]> {
    const data = await this.request<{ transfers: Transfer[] }>(
      'GET',
      '/api/transfers'
    );

    return data.transfers;
  }

  // One transfer's progress and the inputs its streaming decision used.
  async getTransfer(id: string): Promise<Transfer> {
    return this.request<Transfer>(
      'GET',
      `/api/transfers/${encodeURIComponent(id)}`
    );
  }

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
    options: { authenticated?: boolean } = {}
  ): Promise<T> {
    const fetchImpl = this.fetchImpl;
    if (!fetchImpl) {
      throw new ConnectionError(
        'No fetch implementation available: pass `fetch` in the client config'
      );
    }

    const headers: Record<string, string> = { Accept: 'application/json' };
    if (options.authenticated !== false) {
      headers.Authorization = `Bearer ${await this.requireToken()}`;
    }
    if (body !== undefined) {
      headers['Content-Type'] = 'application/json';
    }

    const url = this.baseUrl + path;
    const signal =
      this.requestTimeout > 0 && typeof AbortSignal !== 'undefined'
        ? AbortSignal.timeout(this.requestTimeout)
        : undefined;

    let response: Response;
    try {
      response = await fetchImpl(url, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal,
      });
    } catch (error) {
      const cause = error as Error;
      if (cause.name === 'TimeoutError' || cause.name === 'AbortError') {
        throw new TimeoutError(
          `Request to ${path} timed out after ${this.requestTimeout}ms`
        );
      }
      throw new ConnectionError(`Request to ${url} failed: ${cause.message}`);
    }

    const text = await response.text();
    let payload: unknown;
    let invalidJson = false;
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        invalidJson = true;
      }
    }

    if (!response.ok) {
      let message = `${response.status} ${response.statusText}`;
      if (payload && typeof payload === 'object' && 'error' in payload) {
        const reported = payload.error;
        if (typeof reported === 'string' && reported !== '') {
          message = reported;
        }
      } else if (invalidJson) {
        // A non-JSON error body (gateway 404, HTML error page) still belongs
        // to this response: keep the status and carry the text as the message.
        message = text.trim().slice(0, 200) || message;
      }

      if (response.status === 401 || response.status === 403) {
        throw new AuthenticationError(
          `Authentication failed for ${method} ${path}: ${message}`
        );
      }

      throw new FabricAPIError(
        `${method} ${path} failed (HTTP ${response.status}): ${message}`,
        response.status,
        payload
      );
    }

    if (invalidJson) {
      throw new DWCPError(
        `Invalid JSON response from ${method} ${path} (HTTP ${response.status})`
      );
    }

    // The contract fixes the response shape per endpoint; this is the only
    // place the JSON crosses the wire boundary, so it is the one assertion.
    return payload as T;
  }

  private async requireToken(): Promise<string> {
    if (this.token) {
      return this.token;
    }

    if (this.email && this.password) {
      await this.login();
      if (this.token) {
        return this.token;
      }
    }

    throw new AuthenticationError(
      'Not authenticated: pass token or email/password to FabricClient, or call login()'
    );
  }
}