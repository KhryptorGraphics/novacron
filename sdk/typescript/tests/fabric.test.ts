/**
 * FabricClient tests.
 *
 * Every test drives the client against a real HTTP server bound to
 * 127.0.0.1:0: the assertions cover the request the client builds (method,
 * path, headers, JSON body), the typed value it parses back, and the error it
 * raises on a non-2xx response. No live api-server is involved.
 */

import * as http from 'http';
import { AddressInfo } from 'net';

import {
  AuthenticationError,
  ConnectionError,
  TimeoutError,
} from '../src/client';
import {
  FabricAPIError,
  FabricClient,
  FabricJob,
  FabricJobSubmission,
  FabricNode,
  Transfer,
} from '../src/fabric';

interface RecordedRequest {
  method: string;
  path: string;
  headers: http.IncomingHttpHeaders;
  body: unknown;
}

interface MockResponse {
  status?: number;
  body?: unknown;
  /** Verbatim response body (used for non-JSON error pages). */
  raw?: string;
  delayMs?: number;
}

type MockHandler = (
  req: RecordedRequest
) => MockResponse | Promise<MockResponse>;

interface MockServer {
  url: string;
  requests: RecordedRequest[];
  close(): Promise<void>;
}

// Servers started by a test, closed after it.
const startedServers: MockServer[] = [];

afterEach(async () => {
  await Promise.all(
    startedServers.splice(0).map((server) => server.close())
  );
});

async function startMockServer(handler: MockHandler): Promise<MockServer> {
  const requests: RecordedRequest[] = [];

  const server = http.createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on('data', (chunk: Buffer) => chunks.push(chunk));
    req.on('end', () => {
      void (async () => {
        const raw = Buffer.concat(chunks).toString('utf8');
        const recorded: RecordedRequest = {
          method: req.method ?? '',
          path: req.url ?? '',
          headers: req.headers,
          body: raw === '' ? undefined : JSON.parse(raw),
        };
        requests.push(recorded);

        const response = await handler(recorded);
        if (response.delayMs) {
          // A real delay is required only by the abort test: the client's
          // AbortSignal.timeout fires on the platform clock, which fake
          // timers cannot drive.
          const { promise, resolve } = Promise.withResolvers<void>();
          setTimeout(resolve, response.delayMs);
          await promise;
        }

        if (res.destroyed) {
          return;
        }

        res.writeHead(response.status ?? 200, {
          'Content-Type': 'application/json',
        });
        res.end(
          response.raw !== undefined
            ? response.raw
            : response.body === undefined
              ? ''
              : JSON.stringify(response.body)
        );
      })();
    });
  });

  const listening = Promise.withResolvers<void>();
  server.listen(0, '127.0.0.1', listening.resolve);
  await listening.promise;

  const address = server.address();
  if (address === null || typeof address === 'string') {
    throw new Error('mock server did not bind a TCP address');
  }

  const mock: MockServer = {
    url: `http://127.0.0.1:${(address as AddressInfo).port}`,
    requests,
    close: () => {
      const closed = Promise.withResolvers<void>();
      server.close((error) => {
        // Closing an already-closed mock server is a no-op: the
        // unreachable-server test closes it before afterEach runs.
        const code = (error as NodeJS.ErrnoException | undefined)?.code;
        if (error && code !== 'ERR_SERVER_NOT_RUNNING') {
          closed.reject(error);
          return;
        }
        closed.resolve();
      });
      return closed.promise;
    },
  };

  startedServers.push(mock);
  return mock;
}

const nodeA: FabricNode = {
  node_id: 'node-a',
  arch: 'arm64',
  cores: 14,
  mem_total_mb: 125748,
  mem_allocated_mb: 0,
  storage_total_gb: 3752,
  storage_free_gb: 912,
  vm_count: 0,
  reachable: true,
  link: null,
};

const nodeB: FabricNode = {
  ...nodeA,
  node_id: 'node-b',
  addr: '127.0.0.1:18091',
  link: {
    rtt_ms: 0.341,
    last_heartbeat: '2026-09-20T11:29:24Z',
    stale: false,
  },
};

const submission: FabricJobSubmission = {
  job_id: 'job-1',
  vm_id: 'vm-1',
  node_id: 'node-b',
  status: 'running',
  placement: {
    decision: 'cost',
    cost_estimate_s: 0.012,
    reason: 'lowest estimated move cost over reachable nodes',
  },
};

const job: FabricJob = {
  job_id: 'job-1',
  name: 'e2e',
  command: '/bin/echo',
  status: 'completed',
  node_id: 'node-b',
  vm_id: 'vm-1',
  created_at: '2026-09-20T11:30:00Z',
  logs: { stdout: 'hi\n', stderr: '' },
};

const transfer: Transfer = {
  transfer_id: 'xfer-1',
  status: 'running',
  bytes_total: 1073741824,
  bytes_moved: 268435456,
  measured_bps: 94371840,
  compression: 'zstd-multifd',
  eta_seconds: 8.5,
  decision_inputs: { link_bps: 125000000, sample_ratio: 0.25, threshold: 0.8 },
};

describe('FabricClient authentication', () => {
  it('sends the configured token as a bearer header', async () => {
    const server = await startMockServer(() => ({
      body: { nodes: [nodeA, nodeB] },
    }));
    const client = new FabricClient({ baseUrl: server.url, token: 'jwt-1' });

    await client.listNodes();

    expect(server.requests).toHaveLength(1);
    expect(server.requests[0].headers.authorization).toBe('Bearer jwt-1');
  });

  it('refuses an authed call with no token and no credentials', async () => {
    const server = await startMockServer(() => ({ body: { nodes: [] } }));
    const client = new FabricClient({ baseUrl: server.url });

    await expect(client.listNodes()).rejects.toThrow(AuthenticationError);
    expect(server.requests).toHaveLength(0);
  });

  it('logs in with configured credentials on the first call, then reuses the token', async () => {
    const server = await startMockServer((req) =>
      req.path === '/api/auth/login'
        ? { body: { token: 'jwt-login' } }
        : { body: { jobs: [] } }
    );
    const client = new FabricClient({
      baseUrl: server.url,
      email: 'fabric@novacron.test',
      password: 'Fabr1c!Pass',
    });

    await client.listJobs();
    await client.listJobs();

    expect(server.requests).toHaveLength(3);
    expect(server.requests[0]).toMatchObject({
      method: 'POST',
      path: '/api/auth/login',
      body: { email: 'fabric@novacron.test', password: 'Fabr1c!Pass' },
    });
    expect(server.requests[0].headers.authorization).toBeUndefined();
    expect(server.requests[0].headers['content-type']).toBe(
      'application/json'
    );
    expect(server.requests[1].path).toBe('/api/compute/jobs');
    expect(server.requests[1].headers.authorization).toBe('Bearer jwt-login');
    expect(server.requests[2].headers.authorization).toBe('Bearer jwt-login');
  });

  it('surfaces a login without a token as an authentication failure', async () => {
    const server = await startMockServer(() => ({ body: { user: 'x' } }));
    const client = new FabricClient({
      baseUrl: server.url,
      email: 'fabric@novacron.test',
      password: 'Fabr1c!Pass',
    });

    await expect(client.login()).rejects.toThrow(AuthenticationError);
  });

  it('maps a 401 response to AuthenticationError with the server message', async () => {
    const server = await startMockServer(() => ({
      status: 401,
      body: { error: 'authorization header required' },
    }));
    const client = new FabricClient({ baseUrl: server.url, token: 'stale' });

    const error = await client.listNodes().catch((err: unknown) => err);

    expect(error).toBeInstanceOf(AuthenticationError);
    expect(error).toMatchObject({
      message: expect.stringContaining('authorization header required'),
    });
  });
});

describe('FabricClient nodes', () => {
  it('lists nodes and parses both a null link and a measured link', async () => {
    const server = await startMockServer(() => ({
      body: { nodes: [nodeA, nodeB] },
    }));
    const client = new FabricClient({ baseUrl: server.url, token: 'jwt-1' });

    const nodes = await client.listNodes();

    expect(server.requests[0]).toMatchObject({
      method: 'GET',
      path: '/api/cluster/nodes',
    });
    expect(server.requests[0].body).toBeUndefined();
    expect(nodes).toEqual([nodeA, nodeB]);
    expect(nodes[0].link).toBeNull();
    expect(nodes[1].link?.rtt_ms).toBe(0.341);
  });
});

describe('FabricClient jobs', () => {
  it('submits a job with every optional field and parses the placement', async () => {
    const server = await startMockServer(() => ({
      status: 201,
      body: submission,
    }));
    const client = new FabricClient({ baseUrl: server.url, token: 'jwt-1' });

    const result = await client.submitJob({
      name: 'e2e',
      command: '/bin/echo',
      args: ['hi'],
      env: { GREETING: 'hi' },
      node_id: 'node-b',
      bytes_to_move: 1048576,
      memory_mb: 512,
      vcpus: 2,
    });

    expect(server.requests[0]).toMatchObject({
      method: 'POST',
      path: '/api/compute/jobs',
    });
    expect(server.requests[0].headers['content-type']).toBe(
      'application/json'
    );
    expect(server.requests[0].body).toEqual({
      name: 'e2e',
      command: '/bin/echo',
      args: ['hi'],
      env: { GREETING: 'hi' },
      node_id: 'node-b',
      bytes_to_move: 1048576,
      memory_mb: 512,
      vcpus: 2,
    });
    expect(result).toEqual(submission);
    expect(result.placement.cost_estimate_s).toBe(0.012);
  });

  it('sends only the command when no optional field is given', async () => {
    const server = await startMockServer(() => ({
      status: 201,
      body: submission,
    }));
    const client = new FabricClient({ baseUrl: server.url, token: 'jwt-1' });

    await client.submitJob({ command: '/bin/echo' });

    expect(server.requests[0].body).toEqual({ command: '/bin/echo' });
  });

  it('gets one job with its log tails', async () => {
    const server = await startMockServer(() => ({ body: job }));
    const client = new FabricClient({ baseUrl: server.url, token: 'jwt-1' });

    const fetched = await client.getJob('job-1');

    expect(server.requests[0]).toMatchObject({
      method: 'GET',
      path: '/api/compute/jobs/job-1',
    });
    expect(fetched).toEqual(job);
    expect(fetched.logs?.stdout).toBe('hi\n');
  });

  it('encodes the job id in the request path', async () => {
    const server = await startMockServer(() => ({ body: job }));
    const client = new FabricClient({ baseUrl: server.url, token: 'jwt-1' });

    await client.getJob('job/../1');

    expect(server.requests[0].path).toBe('/api/compute/jobs/job%2F..%2F1');
  });

  it('lists jobs newest first', async () => {
    const older: FabricJob = { ...job, job_id: 'job-0', status: 'failed' };
    const server = await startMockServer(() => ({
      body: { jobs: [job, older] },
    }));
    const client = new FabricClient({ baseUrl: server.url, token: 'jwt-1' });

    const jobs = await client.listJobs();

    expect(server.requests[0]).toMatchObject({
      method: 'GET',
      path: '/api/compute/jobs',
    });
    expect(jobs.map((entry) => entry.job_id)).toEqual(['job-1', 'job-0']);
    expect(jobs[1].status).toBe('failed');
  });

  it('cancels a job through the owning node', async () => {
    const server = await startMockServer(() => ({
      body: { cancelled: true, status: 'cancelled' },
    }));
    const client = new FabricClient({ baseUrl: server.url, token: 'jwt-1' });

    const result = await client.cancelJob('job-1');

    expect(server.requests[0]).toMatchObject({
      method: 'POST',
      path: '/api/compute/jobs/job-1/cancel',
    });
    expect(server.requests[0].body).toBeUndefined();
    expect(result).toEqual({ cancelled: true, status: 'cancelled' });
  });

  it('keeps the server body of a failed cancel on the error', async () => {
    const body = {
      cancelled: false,
      status: 'failed-to-cancel',
      error: 'job\'s node "node-b" is not registered',
    };
    const server = await startMockServer(() => ({ status: 502, body }));
    const client = new FabricClient({ baseUrl: server.url, token: 'jwt-1' });

    const error = await client.cancelJob('job-1').catch((err: unknown) => err);

    expect(error).toBeInstanceOf(FabricAPIError);
    expect(error).toMatchObject({ status: 502, body });
  });
});

describe('FabricClient transfers', () => {
  it('lists transfers in flight', async () => {
    const server = await startMockServer(() => ({
      body: { transfers: [transfer] },
    }));
    const client = new FabricClient({ baseUrl: server.url, token: 'jwt-1' });

    const transfers = await client.listTransfers();

    expect(server.requests[0]).toMatchObject({
      method: 'GET',
      path: '/api/transfers',
    });
    expect(transfers).toEqual([transfer]);
  });

  it('gets one transfer with its decision inputs', async () => {
    const server = await startMockServer(() => ({ body: transfer }));
    const client = new FabricClient({ baseUrl: server.url, token: 'jwt-1' });

    const fetched = await client.getTransfer('xfer-1');

    expect(server.requests[0]).toMatchObject({
      method: 'GET',
      path: '/api/transfers/xfer-1',
    });
    expect(fetched.compression).toBe('zstd-multifd');
    expect(fetched.decision_inputs).toEqual({
      link_bps: 125000000,
      sample_ratio: 0.25,
      threshold: 0.8,
    });
  });
});

describe('FabricClient usage', () => {
  it('fetches the usage summary for the caller org', async () => {
    const summary = {
      from: '2026-09-21T00:00:00Z',
      to: '2026-09-22T00:00:00Z',
      totals: {
        egress_bytes: 268435456,
        egress_gb: 0.25,
        migrations: 1,
        job_seconds: 12.5,
        vcpu_seconds: 75000,
        vcpu_hours: 20.83,
        estimated_cost_usd: 0.56,
      },
      rate_card: {
        usd_per_gb_egress: 0.04,
        usd_per_vcpu_hour: 0.027,
        usd_per_job_second: 0,
        usd_per_migration: 0.5,
      },
    };
    const server = await startMockServer(() => ({ body: summary }));
    const client = new FabricClient({ baseUrl: server.url, token: 'jwt-1' });

    const fetched = await client.usageSummary();

    expect(server.requests[0]).toMatchObject({
      method: 'GET',
      path: '/api/billing/usage/summary',
    });
    expect(fetched.totals.egress_bytes).toBe(268435456);
    expect(fetched.totals.migrations).toBe(1);
  });

  it('scopes the summary to a specific org when asked', async () => {
    const server = await startMockServer(() => ({
      body: { from: 'x', to: 'y', totals: {}, rate_card: {} },
    }));
    const client = new FabricClient({ baseUrl: server.url, token: 'jwt-1' });

    await client.usageSummary('00000000-0000-0000-0000-000000000001');

    expect(server.requests[0].path).toBe(
      '/api/billing/usage/summary?org_id=00000000-0000-0000-0000-000000000001'
    );
  });
});

describe('FabricClient errors', () => {
  it('throws FabricAPIError carrying the 400 status and message', async () => {
    const server = await startMockServer(() => ({
      status: 400,
      body: { error: 'command is required' },
    }));
    const client = new FabricClient({ baseUrl: server.url, token: 'jwt-1' });

    const error = await client
      .submitJob({ command: '   ' })
      .catch((err: unknown) => err);

    expect(error).toBeInstanceOf(FabricAPIError);
    expect(error).toMatchObject({
      status: 400,
      body: { error: 'command is required' },
      message: expect.stringContaining('command is required'),
    });
  });

  it('throws FabricAPIError carrying the 503 status when no node fits', async () => {
    const server = await startMockServer(() => ({
      status: 503,
      body: { error: 'pinned node is not reachable' },
    }));
    const client = new FabricClient({ baseUrl: server.url, token: 'jwt-1' });

    const error = await client
      .submitJob({ command: '/bin/echo', node_id: 'node-z' })
      .catch((err: unknown) => err);

    expect(error).toBeInstanceOf(FabricAPIError);
    expect(error).toMatchObject({
      status: 503,
      message: expect.stringContaining('pinned node is not reachable'),
    });
  });

  it('keeps the status of a non-JSON error body (unrouted P3 path)', async () => {
    const server = await startMockServer(() => ({
      status: 404,
      raw: '<html><body>404 page not found</body></html>',
    }));
    const client = new FabricClient({ baseUrl: server.url, token: 'jwt-1' });

    const error = await client.listTransfers().catch((err: unknown) => err);

    expect(error).toBeInstanceOf(FabricAPIError);
    expect(error).toMatchObject({
      status: 404,
      message: expect.stringContaining('404 page not found'),
    });
  });

  it('reports an unreachable server as ConnectionError', async () => {
    const server = await startMockServer(() => ({ body: { nodes: [] } }));
    const url = server.url;
    await server.close();

    const client = new FabricClient({ baseUrl: url, token: 'jwt-1' });

    await expect(client.listNodes()).rejects.toThrow(ConnectionError);
  });

  it('aborts and reports TimeoutError when the server is too slow', async () => {
    const server = await startMockServer(() => ({
      body: { nodes: [nodeA] },
      delayMs: 250,
    }));
    const client = new FabricClient({
      baseUrl: server.url,
      token: 'jwt-1',
      requestTimeout: 25,
    });

    await expect(client.listNodes()).rejects.toThrow(TimeoutError);
  });
});