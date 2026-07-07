import { render, screen } from '@/__tests__/utils/test-utils';
import { MetricsCard } from '@/components/monitoring/MetricsCard';

describe('MetricsCard', () => {
  it('renders title, value and unit', () => {
    render(<MetricsCard title="CPU Usage" value="75" unit="%" />);

    expect(screen.getByText('CPU Usage')).toBeInTheDocument();
    expect(screen.getByText('75')).toBeInTheDocument();
    expect(screen.getByText('%')).toBeInTheDocument();
  });

  it('renders the change percentage and change label', () => {
    render(
      <MetricsCard title="CPU Usage" value="75" change={5} changeLabel="vs last hour" />
    );

    expect(screen.getByText('5%')).toBeInTheDocument();
    expect(screen.getByText('vs last hour')).toBeInTheDocument();
  });

  it('shows the absolute change for negative deltas', () => {
    render(<MetricsCard title="Errors" value="12" change={-3} />);

    expect(screen.getByText('3%')).toBeInTheDocument();
  });

  it('omits the change block when no change or changeLabel is given', () => {
    const { container } = render(<MetricsCard title="CPU Usage" value="75" />);

    expect(container.textContent).not.toContain('%');
  });

  it('renders a provided icon', () => {
    render(
      <MetricsCard title="CPU" value="1" icon={<span data-testid="metric-icon" />} />
    );

    expect(screen.getByTestId('metric-icon')).toBeInTheDocument();
  });

  it('shows a status indicator bar for non-neutral status', () => {
    const { container } = render(<MetricsCard title="CPU" value="1" status="success" />);

    expect(container.querySelector('.bg-green-500')).toBeInTheDocument();
  });
});
