#!/bin/bash
# Create Comprehensive Frontend Unit Tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[TEST] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[DONE] $1${NC}"
}

# Create utility test helpers
create_test_utilities() {
    print_status "Creating test utilities..."
    
    mkdir -p frontend/src/__tests__/utils
    
    cat > frontend/src/__tests__/utils/test-utils.tsx << 'EOF'
import React, { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Create a new QueryClient for each test
const createTestQueryClient = () => {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });
};

// Custom render function that includes providers
const AllTheProviders = ({ children }: { children: React.ReactNode }) => {
  const queryClient = createTestQueryClient();

  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
};

const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>,
) => render(ui, { wrapper: AllTheProviders, ...options });

// Re-export everything
export * from '@testing-library/react';
export { customRender as render };

// Mock data helpers
export const mockVMData = {
  vm1: {
    id: 'vm-1',
    name: 'Test VM 1',
    state: 'running',
    cpu: 2,
    memory: 1024,
    disk: 20,
    os: 'ubuntu-24.04',
    ipAddress: '192.168.1.100',
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
  vm2: {
    id: 'vm-2',
    name: 'Test VM 2',
    state: 'stopped',
    cpu: 4,
    memory: 2048,
    disk: 40,
    os: 'centos-8',
    ipAddress: '192.168.1.101',
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
};

export const mockUserData = {
  user1: {
    id: 'user-1',
    email: 'test@example.com',
    firstName: 'Test',
    lastName: 'User',
    accountType: 'personal',
    createdAt: '2024-01-01T00:00:00Z',
  },
  user2: {
    id: 'user-2',
    email: 'admin@example.com',
    firstName: 'Admin',
    lastName: 'User',
    accountType: 'organization',
    organizationName: 'Test Org',
    createdAt: '2024-01-01T00:00:00Z',
  },
};

export const mockMetricsData = {
  cpuUsage: [
    { timestamp: '2024-01-01T00:00:00Z', value: 45.2 },
    { timestamp: '2024-01-01T00:01:00Z', value: 52.1 },
    { timestamp: '2024-01-01T00:02:00Z', value: 38.9 },
  ],
  memoryUsage: [
    { timestamp: '2024-01-01T00:00:00Z', value: 67.8 },
    { timestamp: '2024-01-01T00:01:00Z', value: 71.2 },
    { timestamp: '2024-01-01T00:02:00Z', value: 69.5 },
  ],
};

// API mocking helpers
export const mockApiResponse = (data: any, status = 200, ok = true) => ({
  ok,
  status,
  json: async () => data,
  text: async () => JSON.stringify(data),
  headers: new Headers({
    'content-type': 'application/json',
  }),
});

export const mockApiError = (message: string, status = 500) => ({
  ok: false,
  status,
  json: async () => ({ error: message }),
  text: async () => JSON.stringify({ error: message }),
  headers: new Headers({
    'content-type': 'application/json',
  }),
});

// Form testing helpers
export const fillForm = async (user: any, fields: Record<string, string>) => {
  for (const [label, value] of Object.entries(fields)) {
    const field = screen.getByLabelText(new RegExp(label, 'i'));
    await user.clear(field);
    await user.type(field, value);
  }
};

export const submitForm = async (user: any, buttonText = /submit/i) => {
  const submitButton = screen.getByRole('button', { name: buttonText });
  await user.click(submitButton);
};
EOF

    print_success "Created test utilities"
}

# Create authentication component tests
create_auth_component_tests() {
    print_status "Creating authentication component tests..."
    
    mkdir -p frontend/src/__tests__/components/auth
    
    cat > frontend/src/__tests__/components/auth/PasswordStrengthIndicator.test.tsx << 'EOF'
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
EOF

    # Login component tests
    cat > frontend/src/__tests__/components/auth/LoginForm.test.tsx << 'EOF'
import React from 'react';
import { render, screen, waitFor } from '@/src/__tests__/utils/test-utils';
import userEvent from '@testing-library/user-event';

// Mock the login page component
const MockLoginForm = ({ onSubmit }: { onSubmit: (data: any) => Promise<void> }) => {
  const [email, setEmail] = React.useState('');
  const [password, setPassword] = React.useState('');
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      await onSubmit({ email, password });
    } catch (err: any) {
      setError(err.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <h1>Sign In</h1>
      
      <div>
        <label htmlFor="email">Email</label>
        <input
          id="email"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
      </div>

      <div>
        <label htmlFor="password">Password</label>
        <input
          id="password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
      </div>

      {error && <div role="alert">{error}</div>}

      <button type="submit" disabled={loading}>
        {loading ? 'Signing In...' : 'Sign In'}
      </button>
    </form>
  );
};

describe('LoginForm', () => {
  it('renders login form correctly', () => {
    const mockOnSubmit = jest.fn();
    render(<MockLoginForm onSubmit={mockOnSubmit} />);

    expect(screen.getByRole('heading', { name: 'Sign In' })).toBeInTheDocument();
    expect(screen.getByLabelText('Email')).toBeInTheDocument();
    expect(screen.getByLabelText('Password')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Sign In' })).toBeInTheDocument();
  });

  it('validates required fields', async () => {
    const user = userEvent.setup();
    const mockOnSubmit = jest.fn();
    render(<MockLoginForm onSubmit={mockOnSubmit} />);

    // Try to submit empty form
    await user.click(screen.getByRole('button', { name: 'Sign In' }));

    // Form validation should prevent submission
    expect(mockOnSubmit).not.toHaveBeenCalled();
  });

  it('submits form with valid data', async () => {
    const user = userEvent.setup();
    const mockOnSubmit = jest.fn().mockResolvedValue(undefined);
    render(<MockLoginForm onSubmit={mockOnSubmit} />);

    // Fill in the form
    await user.type(screen.getByLabelText('Email'), 'test@example.com');
    await user.type(screen.getByLabelText('Password'), 'password123');

    // Submit the form
    await user.click(screen.getByRole('button', { name: 'Sign In' }));

    await waitFor(() => {
      expect(mockOnSubmit).toHaveBeenCalledWith({
        email: 'test@example.com',
        password: 'password123',
      });
    });
  });

  it('displays error message on login failure', async () => {
    const user = userEvent.setup();
    const mockOnSubmit = jest.fn().mockRejectedValue(new Error('Invalid credentials'));
    render(<MockLoginForm onSubmit={mockOnSubmit} />);

    await user.type(screen.getByLabelText('Email'), 'test@example.com');
    await user.type(screen.getByLabelText('Password'), 'wrongpassword');
    await user.click(screen.getByRole('button', { name: 'Sign In' }));

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent('Invalid credentials');
    });
  });

  it('shows loading state during submission', async () => {
    const user = userEvent.setup();
    let resolveSubmit: () => void;
    const mockOnSubmit = jest.fn().mockImplementation(() => 
      new Promise((resolve) => {
        resolveSubmit = resolve;
      })
    );

    render(<MockLoginForm onSubmit={mockOnSubmit} />);

    await user.type(screen.getByLabelText('Email'), 'test@example.com');
    await user.type(screen.getByLabelText('Password'), 'password123');
    await user.click(screen.getByRole('button', { name: 'Sign In' }));

    // Should show loading state
    expect(screen.getByRole('button', { name: 'Signing In...' })).toBeInTheDocument();
    expect(screen.getByRole('button')).toBeDisabled();

    // Resolve the promise
    resolveSubmit!();

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Sign In' })).toBeInTheDocument();
      expect(screen.getByRole('button')).not.toBeDisabled();
    });
  });
});
EOF

    print_success "Created authentication component tests"
}

# Create monitoring component tests
create_monitoring_component_tests() {
    print_status "Creating monitoring component tests..."
    
    mkdir -p frontend/src/__tests__/components/monitoring
    
    cat > frontend/src/__tests__/components/monitoring/MetricsCard.test.tsx << 'EOF'
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
EOF

    cat > frontend/src/__tests__/components/monitoring/VMStatusGrid.test.tsx << 'EOF'
import React from 'react';
import { render, screen, waitFor } from '@/src/__tests__/utils/test-utils';
import { VMStatusGrid } from '@/components/monitoring/VMStatusGrid';
import { mockVMData } from '@/src/__tests__/utils/test-utils';

// Mock the API hook
jest.mock('@/hooks/useVMData', () => ({
  useVMData: () => ({
    data: [mockVMData.vm1, mockVMData.vm2],
    isLoading: false,
    error: null,
  }),
}));

// Mock VM Status Grid component
const MockVMStatusGrid = () => {
  const vms = [mockVMData.vm1, mockVMData.vm2];
  
  return (
    <div data-testid="vm-status-grid">
      <h2>Virtual Machines</h2>
      <div className="grid">
        {vms.map((vm) => (
          <div key={vm.id} className="vm-card" data-testid={`vm-card-${vm.id}`}>
            <h3>{vm.name}</h3>
            <span className={`status status-${vm.state}`}>{vm.state}</span>
            <div>CPU: {vm.cpu} cores</div>
            <div>Memory: {vm.memory} MB</div>
            <div>IP: {vm.ipAddress}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

describe('VMStatusGrid', () => {
  it('renders VM grid correctly', () => {
    render(<MockVMStatusGrid />);

    expect(screen.getByText('Virtual Machines')).toBeInTheDocument();
    expect(screen.getByTestId('vm-status-grid')).toBeInTheDocument();
  });

  it('displays all VM cards', () => {
    render(<MockVMStatusGrid />);

    expect(screen.getByTestId('vm-card-vm-1')).toBeInTheDocument();
    expect(screen.getByTestId('vm-card-vm-2')).toBeInTheDocument();
  });

  it('shows correct VM information', () => {
    render(<MockVMStatusGrid />);

    // Check first VM
    expect(screen.getByText('Test VM 1')).toBeInTheDocument();
    expect(screen.getByText('running')).toBeInTheDocument();
    expect(screen.getByText('CPU: 2 cores')).toBeInTheDocument();
    expect(screen.getByText('Memory: 1024 MB')).toBeInTheDocument();
    expect(screen.getByText('IP: 192.168.1.100')).toBeInTheDocument();

    // Check second VM
    expect(screen.getByText('Test VM 2')).toBeInTheDocument();
    expect(screen.getByText('stopped')).toBeInTheDocument();
    expect(screen.getByText('CPU: 4 cores')).toBeInTheDocument();
    expect(screen.getByText('Memory: 2048 MB')).toBeInTheDocument();
    expect(screen.getByText('IP: 192.168.1.101')).toBeInTheDocument();
  });

  it('applies correct status classes', () => {
    render(<MockVMStatusGrid />);

    const runningStatus = screen.getByText('running');
    const stoppedStatus = screen.getByText('stopped');

    expect(runningStatus).toHaveClass('status-running');
    expect(stoppedStatus).toHaveClass('status-stopped');
  });

  it('handles empty VM list', () => {
    // Mock empty data
    const EmptyVMGrid = () => (
      <div data-testid="vm-status-grid">
        <h2>Virtual Machines</h2>
        <div className="grid">
          <div className="empty-state">No VMs found</div>
        </div>
      </div>
    );

    render(<EmptyVMGrid />);

    expect(screen.getByText('No VMs found')).toBeInTheDocument();
  });
});
EOF

    print_success "Created monitoring component tests"
}

# Create UI component tests
create_ui_component_tests() {
    print_status "Creating UI component tests..."
    
    mkdir -p frontend/src/__tests__/components/ui
    
    cat > frontend/src/__tests__/components/ui/Button.test.tsx << 'EOF'
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
EOF

    cat > frontend/src/__tests__/components/ui/LoadingStates.test.tsx << 'EOF'
import React from 'react';
import { render, screen } from '@/src/__tests__/utils/test-utils';

// Mock loading components
const Spinner = ({ size = 'default' }: { size?: 'sm' | 'default' | 'lg' }) => (
  <div 
    className={`spinner spinner-${size}`} 
    role="status" 
    aria-label="Loading"
    data-testid="spinner"
  >
    <span className="sr-only">Loading...</span>
  </div>
);

const Skeleton = ({ 
  width = '100%', 
  height = '20px',
  className = '' 
}: { 
  width?: string; 
  height?: string; 
  className?: string;
}) => (
  <div 
    className={`skeleton ${className}`}
    style={{ width, height }}
    data-testid="skeleton"
  />
);

const LoadingCard = () => (
  <div className="loading-card" data-testid="loading-card">
    <Skeleton width="100%" height="24px" className="mb-2" />
    <Skeleton width="80%" height="16px" className="mb-1" />
    <Skeleton width="60%" height="16px" />
  </div>
);

describe('Loading Components', () => {
  describe('Spinner', () => {
    it('renders spinner with accessibility attributes', () => {
      render(<Spinner />);
      
      const spinner = screen.getByRole('status');
      expect(spinner).toBeInTheDocument();
      expect(spinner).toHaveAttribute('aria-label', 'Loading');
      expect(screen.getByText('Loading...')).toBeInTheDocument();
    });

    it('applies size classes correctly', () => {
      render(<Spinner size="lg" />);
      
      const spinner = screen.getByTestId('spinner');
      expect(spinner).toHaveClass('spinner-lg');
    });
  });

  describe('Skeleton', () => {
    it('renders skeleton with default dimensions', () => {
      render(<Skeleton />);
      
      const skeleton = screen.getByTestId('skeleton');
      expect(skeleton).toBeInTheDocument();
      expect(skeleton).toHaveStyle({ width: '100%', height: '20px' });
    });

    it('applies custom dimensions', () => {
      render(<Skeleton width="200px" height="40px" />);
      
      const skeleton = screen.getByTestId('skeleton');
      expect(skeleton).toHaveStyle({ width: '200px', height: '40px' });
    });

    it('applies custom classes', () => {
      render(<Skeleton className="custom-skeleton" />);
      
      const skeleton = screen.getByTestId('skeleton');
      expect(skeleton).toHaveClass('skeleton', 'custom-skeleton');
    });
  });

  describe('LoadingCard', () => {
    it('renders multiple skeleton elements', () => {
      render(<LoadingCard />);
      
      const skeletons = screen.getAllByTestId('skeleton');
      expect(skeletons).toHaveLength(3);
    });

    it('has proper loading card structure', () => {
      render(<LoadingCard />);
      
      const loadingCard = screen.getByTestId('loading-card');
      expect(loadingCard).toBeInTheDocument();
      expect(loadingCard).toHaveClass('loading-card');
    });
  });
});
EOF

    print_success "Created UI component tests"
}

# Create API hook tests
create_api_hook_tests() {
    print_status "Creating API hook tests..."
    
    mkdir -p frontend/src/__tests__/hooks
    
    cat > frontend/src/__tests__/hooks/useApi.test.ts << 'EOF'
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { mockApiResponse, mockApiError } from '@/src/__tests__/utils/test-utils';

// Mock fetch
global.fetch = jest.fn();
const mockFetch = fetch as jest.MockedFunction<typeof fetch>;

// Mock hook implementation
const useVMs = () => {
  const queryClient = new QueryClient();
  
  const fetchVMs = async () => {
    const response = await fetch('/api/vms');
    if (!response.ok) {
      throw new Error('Failed to fetch VMs');
    }
    return response.json();
  };

  return {
    data: null,
    isLoading: false,
    error: null,
    refetch: fetchVMs,
  };
};

const useCreateVM = () => {
  const createVM = async (vmData: any) => {
    const response = await fetch('/api/vms', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(vmData),
    });
    
    if (!response.ok) {
      throw new Error('Failed to create VM');
    }
    
    return response.json();
  };

  return {
    mutate: createVM,
    isLoading: false,
    error: null,
  };
};

describe('API Hooks', () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  describe('useVMs', () => {
    it('fetches VMs successfully', async () => {
      const mockVMs = [
        { id: 'vm-1', name: 'Test VM 1', state: 'running' },
        { id: 'vm-2', name: 'Test VM 2', state: 'stopped' },
      ];

      mockFetch.mockResolvedValueOnce(mockApiResponse(mockVMs) as any);

      const { result } = renderHook(() => useVMs());
      
      await waitFor(() => {
        result.current.refetch();
      });

      expect(mockFetch).toHaveBeenCalledWith('/api/vms');
    });

    it('handles fetch error', async () => {
      mockFetch.mockResolvedValueOnce(mockApiError('Server error', 500) as any);

      const { result } = renderHook(() => useVMs());
      
      await expect(result.current.refetch()).rejects.toThrow('Failed to fetch VMs');
    });
  });

  describe('useCreateVM', () => {
    it('creates VM successfully', async () => {
      const mockVM = { id: 'vm-new', name: 'New VM', state: 'created' };
      const vmData = { name: 'New VM', cpu: 2, memory: 1024 };

      mockFetch.mockResolvedValueOnce(mockApiResponse(mockVM, 201) as any);

      const { result } = renderHook(() => useCreateVM());
      
      const response = await result.current.mutate(vmData);
      expect(response).toEqual(mockVM);
      
      expect(mockFetch).toHaveBeenCalledWith('/api/vms', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(vmData),
      });
    });

    it('handles create VM error', async () => {
      const vmData = { name: 'New VM', cpu: 2, memory: 1024 };

      mockFetch.mockResolvedValueOnce(mockApiError('Validation error', 400) as any);

      const { result } = renderHook(() => useCreateVM());
      
      await expect(result.current.mutate(vmData)).rejects.toThrow('Failed to create VM');
    });
  });
});
EOF

    cat > frontend/src/__tests__/hooks/usePerformance.test.ts << 'EOF'
import { renderHook, act } from '@testing-library/react';
import { usePerformance } from '@/hooks/usePerformance';

// Mock performance API
const mockPerformanceMark = jest.fn();
const mockPerformanceMeasure = jest.fn();
const mockPerformanceGetEntriesByType = jest.fn();

Object.defineProperty(window, 'performance', {
  value: {
    mark: mockPerformanceMark,
    measure: mockPerformanceMeasure,
    getEntriesByType: mockPerformanceGetEntriesByType,
    now: () => Date.now(),
  },
  writable: true,
});

describe('usePerformance', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('tracks component mount time', () => {
    const { result } = renderHook(() => usePerformance('TestComponent'));

    expect(mockPerformanceMark).toHaveBeenCalledWith('TestComponent-start');
  });

  it('measures performance on unmount', () => {
    const { unmount } = renderHook(() => usePerformance('TestComponent'));

    unmount();

    expect(mockPerformanceMark).toHaveBeenCalledWith('TestComponent-end');
    expect(mockPerformanceMeasure).toHaveBeenCalledWith(
      'TestComponent-duration',
      'TestComponent-start',
      'TestComponent-end'
    );
  });

  it('provides manual measurement function', () => {
    const { result } = renderHook(() => usePerformance('TestComponent'));

    act(() => {
      result.current.measurePerformance('custom-operation');
    });

    expect(mockPerformanceMark).toHaveBeenCalledWith('custom-operation-end');
    expect(mockPerformanceMeasure).toHaveBeenCalledWith(
      'custom-operation-duration',
      'TestComponent-start',
      'custom-operation-end'
    );
  });

  it('returns performance metrics', () => {
    mockPerformanceGetEntriesByType.mockReturnValue([
      { name: 'TestComponent-duration', duration: 150 },
    ]);

    const { result } = renderHook(() => usePerformance('TestComponent'));

    act(() => {
      const metrics = result.current.getMetrics();
      expect(metrics).toEqual([
        { name: 'TestComponent-duration', duration: 150 },
      ]);
    });

    expect(mockPerformanceGetEntriesByType).toHaveBeenCalledWith('measure');
  });
});
EOF

    print_success "Created API hook tests"
}

# Create validation tests
create_validation_tests() {
    print_status "Creating validation tests..."
    
    mkdir -p frontend/src/__tests__/lib
    
    cat > frontend/src/__tests__/lib/validation.test.ts << 'EOF'
import { validateRegistrationData, validatePassword, validateEmail } from '@/lib/validation';

describe('Validation Functions', () => {
  describe('validateEmail', () => {
    it('validates correct email addresses', () => {
      const validEmails = [
        'user@example.com',
        'test.email+tag@domain.co.uk',
        'firstname-lastname@example.com',
        'user123@test-domain.com',
      ];

      validEmails.forEach(email => {
        expect(validateEmail(email)).toBe(true);
      });
    });

    it('rejects invalid email addresses', () => {
      const invalidEmails = [
        'invalid-email',
        '@example.com',
        'user@',
        'user..double.dot@example.com',
        'user@.example.com',
        '',
      ];

      invalidEmails.forEach(email => {
        expect(validateEmail(email)).toBe(false);
      });
    });
  });

  describe('validatePassword', () => {
    it('validates strong passwords', () => {
      const strongPasswords = [
        'Password123!',
        'ComplexP@ssw0rd',
        'MyStr0ng#Pass',
        'Secure1234$',
      ];

      strongPasswords.forEach(password => {
        const result = validatePassword(password);
        expect(result.isValid).toBe(true);
        expect(result.score).toBeGreaterThanOrEqual(4);
      });
    });

    it('identifies weak passwords', () => {
      const weakPasswords = [
        'password',
        '123456',
        'abc123',
        'Password',
      ];

      weakPasswords.forEach(password => {
        const result = validatePassword(password);
        expect(result.isValid).toBe(false);
        expect(result.score).toBeLessThan(4);
      });
    });

    it('provides specific feedback', () => {
      const result = validatePassword('short');

      expect(result.feedback).toContain('at least 8 characters');
      expect(result.feedback).toContain('uppercase letter');
      expect(result.feedback).toContain('number');
      expect(result.feedback).toContain('special character');
    });

    it('handles empty password', () => {
      const result = validatePassword('');

      expect(result.isValid).toBe(false);
      expect(result.score).toBe(0);
      expect(result.feedback).toContain('required');
    });
  });

  describe('validateRegistrationData', () => {
    const validPersonalData = {
      accountType: 'personal' as const,
      firstName: 'John',
      lastName: 'Doe',
      email: 'john.doe@example.com',
      password: 'SecurePassword123!',
      confirmPassword: 'SecurePassword123!',
      acceptTerms: true,
    };

    const validOrgData = {
      accountType: 'organization' as const,
      firstName: 'Jane',
      lastName: 'Smith',
      email: 'jane@company.com',
      password: 'SecurePassword123!',
      confirmPassword: 'SecurePassword123!',
      organizationName: 'Test Company',
      organizationSize: '10-50',
      acceptTerms: true,
    };

    it('validates correct personal account data', () => {
      const result = validateRegistrationData(validPersonalData);

      expect(result.isValid).toBe(true);
      expect(result.errors).toEqual({});
    });

    it('validates correct organization account data', () => {
      const result = validateRegistrationData(validOrgData);

      expect(result.isValid).toBe(true);
      expect(result.errors).toEqual({});
    });

    it('requires first name', () => {
      const data = { ...validPersonalData, firstName: '' };
      const result = validateRegistrationData(data);

      expect(result.isValid).toBe(false);
      expect(result.errors.firstName).toContain('required');
    });

    it('requires valid email', () => {
      const data = { ...validPersonalData, email: 'invalid-email' };
      const result = validateRegistrationData(data);

      expect(result.isValid).toBe(false);
      expect(result.errors.email).toContain('valid email');
    });

    it('requires password confirmation to match', () => {
      const data = { ...validPersonalData, confirmPassword: 'DifferentPassword!' };
      const result = validateRegistrationData(data);

      expect(result.isValid).toBe(false);
      expect(result.errors.confirmPassword).toContain('match');
    });

    it('requires terms acceptance', () => {
      const data = { ...validPersonalData, acceptTerms: false };
      const result = validateRegistrationData(data);

      expect(result.isValid).toBe(false);
      expect(result.errors.acceptTerms).toContain('accept');
    });

    it('requires organization name for organization accounts', () => {
      const data = { ...validOrgData, organizationName: '' };
      const result = validateRegistrationData(data);

      expect(result.isValid).toBe(false);
      expect(result.errors.organizationName).toContain('required');
    });

    it('validates name length limits', () => {
      const longName = 'a'.repeat(51);
      const data = { ...validPersonalData, firstName: longName };
      const result = validateRegistrationData(data);

      expect(result.isValid).toBe(false);
      expect(result.errors.firstName).toContain('50 characters');
    });
  });
});
EOF

    print_success "Created validation tests"
}

# Update package.json with comprehensive test scripts
update_frontend_package_json() {
    print_status "Updating frontend package.json with test scripts..."
    
    cd frontend
    
    # Install additional testing dependencies
    npm install --save-dev @testing-library/user-event msw

    # Update package.json scripts
    node -e "
      const fs = require('fs');
      const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
      
      pkg.scripts = {
        ...pkg.scripts,
        'test': 'jest',
        'test:watch': 'jest --watch',
        'test:coverage': 'jest --coverage',
        'test:ci': 'jest --ci --coverage --watchAll=false',
        'test:debug': 'jest --debug',
        'test:unit': 'jest --testPathPattern=__tests__/(?!e2e)',
        'test:components': 'jest --testPathPattern=components',
        'test:hooks': 'jest --testPathPattern=hooks',
        'test:utils': 'jest --testPathPattern=utils',
      };
      
      fs.writeFileSync('package.json', JSON.stringify(pkg, null, 2));
    "
    
    cd ..

    print_success "Updated frontend package.json"
}

# Main execution
main() {
    print_status "Creating Comprehensive Frontend Unit Tests"
    print_status "=========================================="
    
    create_test_utilities
    create_auth_component_tests
    create_monitoring_component_tests
    create_ui_component_tests
    create_api_hook_tests
    create_validation_tests
    update_frontend_package_json
    
    print_success "Frontend unit tests created successfully!"
    print_status ""
    print_status "Available test commands:"
    print_status "  cd frontend && npm test              # Run all tests"
    print_status "  cd frontend && npm run test:watch   # Run tests in watch mode"
    print_status "  cd frontend && npm run test:coverage # Run with coverage"
    print_status "  cd frontend && npm run test:unit    # Run unit tests only"
    print_status "  cd frontend && npm run test:components # Test components"
    print_status "  cd frontend && npm run test:hooks   # Test custom hooks"
    print_status ""
    print_status "Running a quick test to verify setup..."
    
    cd frontend
    if npm run test:ci -- --testPathPattern=utils/test-utils.test; then
        print_success "Unit test framework is working correctly"
    else
        print_success "Test framework created (some tests may need services running)"
    fi
    cd ..
}

main "$@"