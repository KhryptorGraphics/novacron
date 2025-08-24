import React from 'react';
import { render, screen } from '@/src/__tests__/utils/test-utils';
import { PasswordStrengthIndicator } from '@/components/auth/PasswordStrengthIndicator';

describe('PasswordStrengthIndicator', () => {
  it('shows weak strength for short password', () => {
    render(<PasswordStrengthIndicator password="123" />);
    
    expect(screen.getByText('Weak')).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '25');
  });

  it('shows medium strength for moderately complex password', () => {
    render(<PasswordStrengthIndicator password="Password1" />);
    
    expect(screen.getByText('Medium')).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '50');
  });

  it('shows strong strength for complex password', () => {
    render(<PasswordStrengthIndicator password="ComplexP@ssw0rd!" />);
    
    expect(screen.getByText('Strong')).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '100');
  });

  it('provides helpful feedback for password requirements', () => {
    render(<PasswordStrengthIndicator password="short" />);
    
    expect(screen.getByText(/at least 8 characters/i)).toBeInTheDocument();
    expect(screen.getByText(/uppercase letter/i)).toBeInTheDocument();
    expect(screen.getByText(/number/i)).toBeInTheDocument();
    expect(screen.getByText(/special character/i)).toBeInTheDocument();
  });

  it('shows all requirements met for strong password', () => {
    render(<PasswordStrengthIndicator password="StrongP@ssw0rd!" />);
    
    const requirements = screen.getAllByText('✓');
    expect(requirements.length).toBeGreaterThan(0);
  });

  it('updates in real-time as password changes', () => {
    const { rerender } = render(<PasswordStrengthIndicator password="weak" />);
    expect(screen.getByText('Weak')).toBeInTheDocument();

    rerender(<PasswordStrengthIndicator password="StrongerP@ssw0rd!" />);
    expect(screen.getByText('Strong')).toBeInTheDocument();
  });
});
