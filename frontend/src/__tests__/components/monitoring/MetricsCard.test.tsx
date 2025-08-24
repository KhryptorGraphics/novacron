import React from 'react';
import { render, screen } from '@/src/__tests__/utils/test-utils';
import { MetricsCard } from '@/components/monitoring/MetricsCard';

describe('MetricsCard', () => {
  const mockMetric = {
    title: 'CPU Usage',
    value: '75%',
    change: '+5%',
    trend: 'up' as const,
    color: 'blue',
  };

  it('renders metric information correctly', () => {
    render(<MetricsCard {...mockMetric} />);

    expect(screen.getByText('CPU Usage')).toBeInTheDocument();
    expect(screen.getByText('75%')).toBeInTheDocument();
    expect(screen.getByText('+5%')).toBeInTheDocument();
  });

  it('shows positive trend with up arrow', () => {
    render(<MetricsCard {...mockMetric} trend="up" />);

    const trendIcon = screen.getByTestId('trend-up-icon');
    expect(trendIcon).toBeInTheDocument();
  });

  it('shows negative trend with down arrow', () => {
    render(<MetricsCard {...mockMetric} trend="down" change="-3%" />);

    const trendIcon = screen.getByTestId('trend-down-icon');
    expect(trendIcon).toBeInTheDocument();
    expect(screen.getByText('-3%')).toBeInTheDocument();
  });

  it('handles missing change data', () => {
    const { change, ...metricWithoutChange } = mockMetric;
    render(<MetricsCard {...metricWithoutChange} />);

    expect(screen.getByText('CPU Usage')).toBeInTheDocument();
    expect(screen.getByText('75%')).toBeInTheDocument();
    expect(screen.queryByText('+5%')).not.toBeInTheDocument();
  });

  it('applies correct color classes', () => {
    const { container } = render(<MetricsCard {...mockMetric} color="green" />);

    const card = container.firstChild as HTMLElement;
    expect(card).toHaveClass('border-green-200');
  });

  it('has proper accessibility attributes', () => {
    render(<MetricsCard {...mockMetric} />);

    const card = screen.getByRole('article');
    expect(card).toHaveAttribute('aria-label', 'CPU Usage metric: 75%');
  });
});
