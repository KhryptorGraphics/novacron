import { render, screen, waitFor } from '@testing-library/react';

import BillingPage from '@/app/billing/page';

jest.mock('@/components/auth/AuthGuard', () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

const summaryFixture = {
  org_id: '00000000-0000-0000-0000-000000000001',
  from: '2026-08-23T00:00:00Z',
  to: '2026-09-22T00:00:00Z',
  totals: {
    egress_bytes: 1610612736,
    egress_gb: 1.5,
    migrations: 4,
    job_seconds: 5400,
    vcpu_seconds: 7200,
    vcpu_hours: 2,
    estimated_cost_usd: 12.3456,
  },
  rate_card: {
    usd_per_gb_egress: 0.05,
    usd_per_vcpu_hour: 0.25,
    usd_per_job_second: 0.0005,
    usd_per_migration: 1.5,
  },
  note: 'measured consumption; rates are operator-configured via NOVACRON_RATE_* env (0 defaults mean unpriced, not free)',
};

const emptySummaryFixture = {
  ...summaryFixture,
  totals: {
    egress_bytes: 0,
    egress_gb: 0,
    migrations: 0,
    job_seconds: 0,
    vcpu_seconds: 0,
    vcpu_hours: 0,
    estimated_cost_usd: 0,
  },
  rate_card: {
    usd_per_gb_egress: 0,
    usd_per_vcpu_hour: 0,
    usd_per_job_second: 0,
    usd_per_migration: 0,
  },
};

function jsonResponse(payload: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => payload,
    text: async () => JSON.stringify(payload),
  } as Response;
}

describe('BillingPage', () => {
  beforeEach(() => {
    window.localStorage.setItem('novacron_token', 'test-token');
  });

  afterEach(() => {
    window.localStorage.clear();
    jest.resetAllMocks();
  });

  it('renders the metered totals, window and configured rate card', async () => {
    const fetchMock = jest
      .fn()
      .mockResolvedValue(jsonResponse(summaryFixture)) as unknown as typeof fetch;
    global.fetch = fetchMock;

    render(<BillingPage />);

    expect(screen.getByText('Billing')).toBeInTheDocument();
    expect(screen.getByText(/Loading metered usage/)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('Total egress')).toBeInTheDocument();
    });

    expect(screen.getByText('1.5 GiB')).toBeInTheDocument();
    expect(screen.getByText(/1\.500 GiB billed/)).toBeInTheDocument();
    expect(screen.getByText('4')).toBeInTheDocument();
    expect(screen.getByText('5,400.0')).toBeInTheDocument();
    expect(screen.getByText('2.00')).toBeInTheDocument();
    expect(screen.getByText('$12.3456')).toBeInTheDocument();
    expect(screen.getByText('2026-08-23T00:00:00Z')).toBeInTheDocument();
    expect(screen.getByText('2026-09-22T00:00:00Z')).toBeInTheDocument();
    expect(screen.getByText('00000000-0000-0000-0000-000000000001')).toBeInTheDocument();

    // Rate card rows, including the unpriced-note for the configured rates.
    expect(screen.getByText('$0.05')).toBeInTheDocument();
    expect(screen.getByText('$0.25')).toBeInTheDocument();
    expect(screen.getByText('$0.0005')).toBeInTheDocument();
    expect(screen.getByText('$1.50')).toBeInTheDocument();
    expect(screen.queryByText(/No metered activity yet/)).not.toBeInTheDocument();

    expect(fetchMock).toHaveBeenCalledWith(
      'http://localhost:8090/api/billing/usage/summary',
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: 'Bearer test-token' }),
      }),
    );
  });

  it('renders the failure message when the summary request fails', async () => {
    global.fetch = jest
      .fn()
      .mockResolvedValue(jsonResponse({ error: 'usage summary query failed' }, 500)) as unknown as typeof fetch;

    render(<BillingPage />);

    await waitFor(() => {
      expect(screen.getByText('Billing usage load failed')).toBeInTheDocument();
    });

    expect(screen.getByText('Failed to load billing usage: usage summary query failed')).toBeInTheDocument();
    expect(screen.queryByText('Total egress')).not.toBeInTheDocument();
  });

  it('states the empty case explicitly when nothing was metered', async () => {
    global.fetch = jest
      .fn()
      .mockResolvedValue(jsonResponse(emptySummaryFixture)) as unknown as typeof fetch;

    render(<BillingPage />);

    await waitFor(() => {
      expect(
        screen.getByText('No metered activity yet; rates are operator-configured'),
      ).toBeInTheDocument();
    });

    expect(screen.getAllByText('$0.00').length).toBeGreaterThanOrEqual(5);
    expect(screen.queryByText('Billing usage load failed')).not.toBeInTheDocument();
  });
});