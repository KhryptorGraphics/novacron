import React from 'react';
import { render, screen } from '@testing-library/react';
import Dashboard from '@/components/dashboard/page';

test('renders dashboard charts', () => {
  render(<Dashboard />);
  const title = screen.getByText(/Dashboard/i);
  expect(title).toBeInTheDocument();
});
