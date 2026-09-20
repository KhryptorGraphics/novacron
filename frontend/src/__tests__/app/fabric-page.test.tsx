import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import FabricPage from '@/app/fabric/page';

jest.mock('@/components/auth/AuthGuard', () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

jest.mock('@/components/ui/use-toast', () => ({
  useToast: () => ({ toast: jest.fn() }),
}));

const nodeFixtures = [
  {
    node_id: 'node-a',
    addr: '10.0.0.5:18090',
    arch: 'arm64',
    cores: 14,
    mem_total_mb: 131072,
    mem_allocated_mb: 16384,
    storage_total_gb: 3752,
    storage_free_gb: 912,
    vm_count: 3,
    reachable: true,
    link: null,
  },
  {
    node_id: 'node-b',
    addr: '127.0.0.1:18091',
    arch: 'amd64',
    cores: 8,
    mem_total_mb: 65536,
    mem_allocated_mb: 0,
    storage_total_gb: 1024,
    storage_free_gb: 512,
    vm_count: 0,
    reachable: false,
    link: { rtt_ms: 12.5, last_heartbeat: '2026-09-20T11:00:00Z', stale: true },
  },
];

const jobFixtures = [
  {
    job_id: 'job-1',
    name: 'echo',
    command: '/bin/echo',
    status: 'running',
    node_id: 'node-a',
    vm_id: 'vm-1',
    created_at: '2026-09-20T11:29:24Z',
  },
];

const transferFixtures = [
  {
    transfer_id: 'tr-1',
    status: 'running',
    kind: 'data',
    target_node: 'node-b',
    bytes_total: 1048576,
    bytes_moved: 262144,
    measured_bps: 1048576,
    compression: 'zstd-multifd',
    eta_seconds: 1,
  },
];

function jsonResponse(payload: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => payload,
    text: async () => JSON.stringify(payload),
  } as Response;
}

function mockFabricFetch() {
  const fetchMock = jest.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = typeof input === 'string' ? input : input.toString();
    const method = (init?.method || 'GET').toUpperCase();

    if (url.endsWith('/api/cluster/nodes')) {
      return jsonResponse({ nodes: nodeFixtures });
    }

    if (url.endsWith('/api/compute/jobs') && method === 'GET') {
      return jsonResponse({ jobs: jobFixtures });
    }

    if (url.endsWith('/api/compute/jobs') && method === 'POST') {
      return jsonResponse(
        {
          job_id: 'job-9',
          vm_id: 'vm-9',
          node_id: 'node-a',
          status: 'running',
          placement: { decision: 'node-a', cost_estimate_s: 4, reason: 'node-a has the lowest measured rtt' },
        },
        201,
      );
    }

    if (url.endsWith('/cancel')) {
      return jsonResponse({ cancelled: true, status: 'cancelled' });
    }

    const detailMatch = url.match(/\/api\/compute\/jobs\/([^/]+)$/);
    if (detailMatch && method === 'GET') {
      return jsonResponse({
        ...jobFixtures[0],
        job_id: detailMatch[1],
        logs: { stdout: `job ${detailMatch[1]} finished\n`, stderr: '' },
      });
    }

    if (url.endsWith('/api/transfers')) {
      return jsonResponse({ transfers: transferFixtures });
    }

    const transferMatch = url.match(/\/api\/transfers\/([^/]+)$/);
    if (transferMatch && method === 'GET') {
      return jsonResponse({
        ...transferFixtures[0],
        transfer_id: transferMatch[1],
        decision_inputs: { link_bps: 1048576, sample_ratio: 0.05, threshold: 65536 },
      });
    }

    throw new Error(`Unexpected fetch: ${method} ${url}`);
  });

  global.fetch = fetchMock as unknown as typeof fetch;
  return fetchMock;
}

describe('FabricPage', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it('renders live nodes, their link profiles, jobs and transfers', async () => {
    mockFabricFetch();
    render(<FabricPage />);

    expect(screen.getByText('Fabric')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('10.0.0.5:18090')).toBeInTheDocument();
    });

    expect(screen.getByText('127.0.0.1:18091')).toBeInTheDocument();
    expect(screen.getByText('arm64')).toBeInTheDocument();
    expect(screen.getByText('amd64')).toBeInTheDocument();
    expect(screen.getByText('114688 MB free')).toBeInTheDocument();
    expect(screen.getByText('131072 MB total · 16384 MB allocated')).toBeInTheDocument();
    expect(screen.getByText('912 GB free')).toBeInTheDocument();
    expect(screen.getByText('reachable')).toBeInTheDocument();
    expect(screen.getByText('unreachable')).toBeInTheDocument();
    expect(screen.getByText('No link profile')).toBeInTheDocument();
    expect(screen.getByText('Stale link profile')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'job-1' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'tr-1' })).toBeInTheDocument();
  });

  it('POSTs the submitted job as JSON to the compute jobs endpoint', async () => {
    const fetchMock = mockFabricFetch();
    const user = userEvent.setup();
    render(<FabricPage />);

    await waitFor(() => {
      expect(screen.getByText('10.0.0.5:18090')).toBeInTheDocument();
    });

    await user.type(screen.getByLabelText('Command'), '/bin/echo');
    await user.type(screen.getByLabelText('Arguments'), 'hello "two words"');
    await user.type(screen.getByLabelText('Environment'), 'LOG_LEVEL=debug\nRETRIES=3');
    await user.type(screen.getByLabelText('memory_mb'), '512');
    await user.click(screen.getByRole('button', { name: /submit job/i }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        'http://localhost:8090/api/compute/jobs',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            command: '/bin/echo',
            args: ['hello', 'two words'],
            env: { LOG_LEVEL: 'debug', RETRIES: '3' },
            memory_mb: 512,
          }),
        }),
      );
    });
  });

  it('blocks submission when an environment line is not KEY=VALUE', async () => {
    const fetchMock = mockFabricFetch();
    const user = userEvent.setup();
    render(<FabricPage />);

    await waitFor(() => {
      expect(screen.getByText('10.0.0.5:18090')).toBeInTheDocument();
    });

    await user.type(screen.getByLabelText('Command'), '/bin/echo');
    await user.type(screen.getByLabelText('Environment'), 'BROKEN LINE');
    await user.click(screen.getByRole('button', { name: /submit job/i }));

    expect(await screen.findByText(/is not KEY=VALUE/)).toBeInTheDocument();
    expect(fetchMock.mock.calls.filter((call) => call[1]?.method === 'POST')).toHaveLength(0);
  });

  it('loads the selected job log tail and cancels the job', async () => {
    const fetchMock = mockFabricFetch();
    const user = userEvent.setup();
    render(<FabricPage />);

    await waitFor(() => {
      expect(screen.getByText('10.0.0.5:18090')).toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: 'job-1' }));

    expect(await screen.findByText('job job-1 finished')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /cancel job/i }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        'http://localhost:8090/api/compute/jobs/job-1/cancel',
        expect.objectContaining({ method: 'POST' }),
      );
    });
  });
});