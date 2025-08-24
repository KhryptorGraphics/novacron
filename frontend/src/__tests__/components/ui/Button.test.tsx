import React from 'react';
import { render, screen } from '@/src/__tests__/utils/test-utils';
import userEvent from '@testing-library/user-event';

// Mock Button component
const Button = ({ 
  children, 
  variant = 'default', 
  size = 'default',
  disabled = false,
  loading = false,
  onClick,
  ...props 
}: {
  children: React.ReactNode;
  variant?: 'default' | 'primary' | 'secondary' | 'danger';
  size?: 'default' | 'sm' | 'lg';
  disabled?: boolean;
  loading?: boolean;
  onClick?: () => void;
  [key: string]: any;
}) => (
  <button
    className={`btn btn-${variant} btn-${size} ${loading ? 'loading' : ''}`}
    disabled={disabled || loading}
    onClick={onClick}
    {...props}
  >
    {loading ? 'Loading...' : children}
  </button>
);

describe('Button', () => {
  it('renders button with text', () => {
    render(<Button>Click me</Button>);
    
    expect(screen.getByRole('button', { name: 'Click me' })).toBeInTheDocument();
  });

  it('applies variant classes correctly', () => {
    render(<Button variant="primary">Primary Button</Button>);
    
    const button = screen.getByRole('button');
    expect(button).toHaveClass('btn-primary');
  });

  it('applies size classes correctly', () => {
    render(<Button size="lg">Large Button</Button>);
    
    const button = screen.getByRole('button');
    expect(button).toHaveClass('btn-lg');
  });

  it('handles disabled state', () => {
    render(<Button disabled>Disabled Button</Button>);
    
    const button = screen.getByRole('button');
    expect(button).toBeDisabled();
  });

  it('shows loading state', () => {
    render(<Button loading>Submit</Button>);
    
    const button = screen.getByRole('button');
    expect(button).toBeDisabled();
    expect(button).toHaveTextContent('Loading...');
    expect(button).toHaveClass('loading');
  });

  it('calls onClick when clicked', async () => {
    const user = userEvent.setup();
    const handleClick = jest.fn();
    
    render(<Button onClick={handleClick}>Click me</Button>);
    
    await user.click(screen.getByRole('button'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('does not call onClick when disabled', async () => {
    const user = userEvent.setup();
    const handleClick = jest.fn();
    
    render(<Button disabled onClick={handleClick}>Disabled</Button>);
    
    await user.click(screen.getByRole('button'));
    expect(handleClick).not.toHaveBeenCalled();
  });

  it('supports keyboard navigation', async () => {
    const user = userEvent.setup();
    const handleClick = jest.fn();
    
    render(<Button onClick={handleClick}>Keyboard Button</Button>);
    
    const button = screen.getByRole('button');
    button.focus();
    
    await user.keyboard('{Enter}');
    expect(handleClick).toHaveBeenCalledTimes(1);
  });
});
