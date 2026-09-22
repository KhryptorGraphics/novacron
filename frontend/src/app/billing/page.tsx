'use client';

export const dynamic = 'force-dynamic';

import { useCallback, useEffect, useState } from 'react';
import { Activity, ArrowLeftRight, DollarSign, Gauge, HardDrive, Loader2, Receipt, RefreshCw } from 'lucide-react';

import AuthGuard from '@/components/auth/AuthGuard';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { buildApiUrl } from '@/lib/api/origin';
import { authService } from '@/lib/auth';

const USAGE_SUMMARY_PATH = '/api/billing/usage/summary';

// Mirrors usageTotals in backend/cmd/api-server/billing_usage.go.
type BillingUsageTotals = {
  egress_bytes: number;
  egress_gb: number;
  migrations: number;
  job_seconds: number;
  vcpu_seconds: number;
  vcpu_hours: number;
  estimated_cost_usd: number;
};

// Mirrors usageRates in backend/cmd/api-server/billing_usage.go; every rate is
// USD per unit and defaults to 0 (unpriced, not free).
type BillingRateCard = {
  usd_per_gb_egress: number;
  usd_per_vcpu_hour: number;
  usd_per_job_second: number;
  usd_per_migration: number;
};

type BillingUsageSummary = {
  org_id: string | null;
  from: string;
  to: string;
  totals: BillingUsageTotals;
  rate_card: BillingRateCard;
  note?: string;
};

const RATE_CARD_ROWS: { key: keyof BillingRateCard; label: string; unit: string; env: string }[] = [
  { key: 'usd_per_gb_egress', label: 'Egress', unit: 'per GB', env: 'NOVACRON_RATE_PER_GB_EGRESS' },
  { key: 'usd_per_vcpu_hour', label: 'vCPU', unit: 'per vCPU hour', env: 'NOVACRON_RATE_PER_VCPU_HOUR' },
  { key: 'usd_per_job_second', label: 'Job seconds', unit: 'per job second', env: 'NOVACRON_RATE_PER_JOB_SECOND' },
  { key: 'usd_per_migration', label: 'Migrations', unit: 'per migration', env: 'NOVACRON_RATE_PER_MIGRATION' },
];

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

function formatNumber(value: number | null | undefined, digits = 0): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return '—';
  }

  return value.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function formatUsd(value: number | null | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return '—';
  }

  return value.toLocaleString('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 4,
  });
}

function usageTotal(summary: BillingUsageSummary | null): number {
  if (!summary) {
    return 0;
  }

  const { totals } = summary;
  return [totals.egress_bytes, totals.migrations, totals.job_seconds, totals.vcpu_seconds].reduce<number>(
    (sum, value) => sum + (typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : 0),
    0,
  );
}

function isUnpriced(card: BillingRateCard | null | undefined): boolean {
  if (!card) {
    return true;
  }

  return RATE_CARD_ROWS.every((row) => {
    const rate = card[row.key];
    return !(typeof rate === 'number' && Number.isFinite(rate) && rate > 0);
  });
}

// The billing route answers failures with `{"error": "..."}` (401/500), so
// surface that message instead of the raw status when the body carries one.
async function describeFailure(response: Response): Promise<string> {
  const body = await response.text();

  if (body) {
    try {
      const parsed = JSON.parse(body) as { error?: unknown; message?: unknown };
      if (typeof parsed.error === 'string' && parsed.error) {
        return parsed.error;
      }
      if (typeof parsed.message === 'string' && parsed.message) {
        return parsed.message;
      }
    } catch {
      // Non-JSON bodies fall through to the raw text.
    }

    return body;
  }

  return `Billing usage request failed (HTTP ${response.status})`;
}

async function fetchUsageSummary(): Promise<BillingUsageSummary> {
  const token = authService.getToken();
  const response = await fetch(buildApiUrl(USAGE_SUMMARY_PATH), {
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });

  if (!response.ok) {
    throw new Error(await describeFailure(response));
  }

  return (await response.json()) as BillingUsageSummary;
}

function MetricCard({
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
      <CardContent className="space-y-2">
        <div className="text-2xl font-semibold">{value}</div>
        <p className="text-xs text-muted-foreground">{description}</p>
      </CardContent>
    </Card>
  );
}

export default function BillingPage() {
  const [summary, setSummary] = useState<BillingUsageSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const loadSummary = useCallback(async () => {
    try {
      setSummary(await fetchUsageSummary());
      setError(null);
    } catch (loadError) {
      setSummary(null);
      setError(
        `Failed to load billing usage: ${loadError instanceof Error ? loadError.message : 'unknown error'}`,
      );
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    const loadInitialSummary = async () => {
      await loadSummary();
      if (!cancelled) {
        setLoading(false);
      }
    };

    void loadInitialSummary();

    return () => {
      cancelled = true;
    };
  }, [loadSummary]);

  const refreshSummary = async () => {
    setRefreshing(true);
    try {
      await loadSummary();
    } finally {
      setRefreshing(false);
    }
  };

  const totals = summary?.totals ?? null;
  const hasActivity = usageTotal(summary) > 0;
  const unpriced = isUnpriced(summary?.rate_card);

  return (
    <AuthGuard>
      <div className="container mx-auto space-y-6 p-6">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Billing</h1>
            <p className="text-muted-foreground">
              Measured consumption for the trailing window from GET /api/billing/usage/summary — egress, migrations, job
              seconds and vCPU hours with the operator rate card applied.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline">Read Only</Badge>
            <Button variant="outline" onClick={refreshSummary} disabled={refreshing || loading}>
              {refreshing ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-2 h-4 w-4" />}
              Refresh
            </Button>
          </div>
        </div>

        {error ? (
          <Alert variant="destructive">
            <AlertTitle>Billing usage load failed</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : null}

        {loading ? (
          <Card>
            <CardContent className="flex items-center pt-6 text-sm text-muted-foreground">
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Loading metered usage…
            </CardContent>
          </Card>
        ) : null}

        {summary && !hasActivity ? (
          <Alert>
            <Receipt className="h-4 w-4" />
            <AlertTitle>No metered activity yet; rates are operator-configured</AlertTitle>
            <AlertDescription>
              The summary endpoint returned zero measured egress, migrations, job seconds and vCPU seconds for this
              window. Rates are set by the operator from NOVACRON_RATE_* environment variables; a zero rate means
              unpriced, not free.
            </AlertDescription>
          </Alert>
        ) : null}

        {summary ? (
          <>
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
              <MetricCard
                title="Total egress"
                value={formatBytes(totals?.egress_bytes)}
                description={`${formatNumber(totals?.egress_gb, 3)} GiB billed from measured transfer bytes.`}
                icon={<ArrowLeftRight className="h-4 w-4 text-muted-foreground" />}
              />
              <MetricCard
                title="Migrations"
                value={formatNumber(totals?.migrations)}
                description="Completed migrations recorded as `migration` usage events."
                icon={<HardDrive className="h-4 w-4 text-muted-foreground" />}
              />
              <MetricCard
                title="Job seconds"
                value={formatNumber(totals?.job_seconds, 1)}
                description="Fabric job wall-clock seconds observed at terminal transition."
                icon={<Activity className="h-4 w-4 text-muted-foreground" />}
              />
              <MetricCard
                title="vCPU hours"
                value={formatNumber(totals?.vcpu_hours, 2)}
                description={`${formatNumber(totals?.vcpu_seconds, 0)} vCPU seconds from VM row timestamps.`}
                icon={<Gauge className="h-4 w-4 text-muted-foreground" />}
              />
              <MetricCard
                title="Estimated cost (USD)"
                value={formatUsd(totals?.estimated_cost_usd)}
                description={
                  unpriced
                    ? 'Computed with the configured rate card; every rate is 0, so this total is unpriced.'
                    : 'Computed with the configured rate card below.'
                }
                icon={<DollarSign className="h-4 w-4 text-muted-foreground" />}
              />
            </div>

            <div className="grid gap-4 lg:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Window</CardTitle>
                  <CardDescription>
                    Inclusive start and exclusive end of the window the summary aggregated, as returned by the endpoint.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center justify-between gap-4 rounded-lg border p-4">
                    <div className="text-sm text-muted-foreground">From</div>
                    <div className="font-mono text-sm">{summary.from}</div>
                  </div>
                  <div className="flex items-center justify-between gap-4 rounded-lg border p-4">
                    <div className="text-sm text-muted-foreground">To</div>
                    <div className="font-mono text-sm">{summary.to}</div>
                  </div>
                  <div className="flex items-center justify-between gap-4 rounded-lg border p-4">
                    <div className="text-sm text-muted-foreground">Scope</div>
                    <div className="font-mono text-sm">{summary.org_id ?? 'all organizations (admin scope)'}</div>
                  </div>
                  {summary.note ? <p className="text-xs text-muted-foreground">{summary.note}</p> : null}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Receipt className="h-4 w-4 text-muted-foreground" />
                    Rate card
                  </CardTitle>
                  <CardDescription>
                    Operator-configured USD rates applied to the measured totals. Zero means unpriced, not free.
                    {unpriced ? ' Every rate is currently 0.' : ''}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Meter</TableHead>
                        <TableHead>Unit</TableHead>
                        <TableHead className="text-right">Rate (USD)</TableHead>
                        <TableHead>Environment</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {RATE_CARD_ROWS.map((row) => (
                        <TableRow key={row.key}>
                          <TableCell className="font-medium">{row.label}</TableCell>
                          <TableCell>{row.unit}</TableCell>
                          <TableCell className="text-right font-mono">
                            {formatUsd(summary.rate_card[row.key])}
                          </TableCell>
                          <TableCell className="font-mono text-xs text-muted-foreground">{row.env}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            </div>
          </>
        ) : null}
      </div>
    </AuthGuard>
  );
}