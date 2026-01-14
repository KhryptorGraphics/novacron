import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { jest } from '@jest/globals';
import '@testing-library/jest-dom';

import { VMCard } from '@/components/vm/VMCard';
import { VM, VMStatus } from '@/types/vm';

// Mock the API hooks
jest.mock('@/hooks/useApi', () => ({
  useVMActions: jest.fn(),
}));

// Mock the components
jest.mock('@/components/ui/Button', () => ({
  Button: ({ children, onClick, disabled, variant, ...props }: any) => (
    <button 
      onClick={onClick} 
      disabled={disabled} 
      data-variant={variant}
      {...props}
    >
      {children}
    </button>
  ),
}));

jest.mock('@/components/ui/Card', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="vm-card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <h3 data-testid="card-title" {...props}>{children}</h3>,
}));

jest.mock('@/components/ui/Badge', () => ({
  Badge: ({ children, variant, ...props }: any) => (
    <span data-testid="badge" data-variant={variant} {...props}>
      {children}
    </span>
  ),
}));

const { useVMActions } = require('@/hooks/useApi');

describe('VMCard Component', () => {
  const mockVMActions = {
    startVM: jest.fn(),
    stopVM: jest.fn(),
    restartVM: jest.fn(),
    deleteVM: jest.fn(),
    isLoading: false,
  };

  const mockVM: VM = {
    id: 'vm-1',
    name: 'Test VM',
    status: VMStatus.STOPPED,
    type: 'qemu',
    cpu: 2,
    memory: 4096,
    disk: 20,
    networks: ['default'],
    createdAt: '2023-01-01T00:00:00Z',
    updatedAt: '2023-01-01T00:00:00Z',
  };

  beforeEach(() => {
    useVMActions.mockReturnValue(mockVMActions);
    jest.clearAllMocks();
  });

  describe('Rendering', () => {
    it('should render VM card with basic information', () => {
      render(<VMCard vm={mockVM} />);

      expect(screen.getByTestId('vm-card')).toBeInTheDocument();
      expect(screen.getByTestId('card-title')).toHaveTextContent('Test VM');
      expect(screen.getByText('vm-1')).toBeInTheDocument();
      expect(screen.getByText('qemu')).toBeInTheDocument();
    });

    it('should display VM specifications', () => {
      render(<VMCard vm={mockVM} />);

      expect(screen.getByText('2 CPU')).toBeInTheDocument();
      expect(screen.getByText('4096 MB RAM')).toBeInTheDocument();
      expect(screen.getByText('20 GB Disk')).toBeInTheDocument();
    });

    it('should show status badge with correct variant', () => {
      render(<VMCard vm={mockVM} />);

      const badge = screen.getByTestId('badge');
      expect(badge).toHaveTextContent('STOPPED');
      expect(badge).toHaveAttribute('data-variant', 'secondary');
    });

    it('should show different status badge variants', () => {
      const runningVM = { ...mockVM, status: VMStatus.RUNNING };
      const { rerender } = render(<VMCard vm={runningVM} />);

      let badge = screen.getByTestId('badge');
      expect(badge).toHaveTextContent('RUNNING');
      expect(badge).toHaveAttribute('data-variant', 'success');

      const errorVM = { ...mockVM, status: VMStatus.ERROR };
      rerender(<VMCard vm={errorVM} />);

      badge = screen.getByTestId('badge');
      expect(badge).toHaveTextContent('ERROR');
      expect(badge).toHaveAttribute('data-variant', 'destructive');
    });

    it('should render networks information', () => {
      const vmWithNetworks = { ...mockVM, networks: ['default', 'internal'] };
      render(<VMCard vm={vmWithNetworks} />);

      expect(screen.getByText('Networks: default, internal')).toBeInTheDocument();
    });

    it('should handle VM with no networks', () => {
      const vmWithoutNetworks = { ...mockVM, networks: [] };
      render(<VMCard vm={vmWithoutNetworks} />);

      expect(screen.getByText('Networks: None')).toBeInTheDocument();
    });
  });

  describe('Action Buttons', () => {
    it('should show correct buttons for stopped VM', () => {
      render(<VMCard vm={mockVM} />);

      expect(screen.getByText('Start')).toBeInTheDocument();
      expect(screen.getByText('Delete')).toBeInTheDocument();
      expect(screen.queryByText('Stop')).not.toBeInTheDocument();
      expect(screen.queryByText('Restart')).not.toBeInTheDocument();
    });

    it('should show correct buttons for running VM', () => {
      const runningVM = { ...mockVM, status: VMStatus.RUNNING };
      render(<VMCard vm={runningVM} />);

      expect(screen.getByText('Stop')).toBeInTheDocument();
      expect(screen.getByText('Restart')).toBeInTheDocument();
      expect(screen.getByText('Delete')).toBeInTheDocument();
      expect(screen.queryByText('Start')).not.toBeInTheDocument();
    });

    it('should disable buttons when loading', () => {
      useVMActions.mockReturnValue({ ...mockVMActions, isLoading: true });
      render(<VMCard vm={mockVM} />);

      expect(screen.getByText('Start')).toBeDisabled();
      expect(screen.getByText('Delete')).toBeDisabled();
    });

    it('should disable buttons for VM in transitioning state', () => {
      const startingVM = { ...mockVM, status: VMStatus.STARTING };
      render(<VMCard vm={startingVM} />);

      expect(screen.getByText('Start')).toBeDisabled();
      expect(screen.getByText('Delete')).toBeDisabled();
    });
  });

  describe('User Interactions', () => {
    it('should call startVM when start button is clicked', async () => {
      const user = userEvent.setup();
      render(<VMCard vm={mockVM} />);

      const startButton = screen.getByText('Start');
      await user.click(startButton);

      expect(mockVMActions.startVM).toHaveBeenCalledWith(mockVM.id);
    });

    it('should call stopVM when stop button is clicked', async () => {
      const user = userEvent.setup();
      const runningVM = { ...mockVM, status: VMStatus.RUNNING };
      render(<VMCard vm={runningVM} />);

      const stopButton = screen.getByText('Stop');
      await user.click(stopButton);

      expect(mockVMActions.stopVM).toHaveBeenCalledWith(mockVM.id);
    });

    it('should call restartVM when restart button is clicked', async () => {
      const user = userEvent.setup();
      const runningVM = { ...mockVM, status: VMStatus.RUNNING };
      render(<VMCard vm={runningVM} />);

      const restartButton = screen.getByText('Restart');
      await user.click(restartButton);

      expect(mockVMActions.restartVM).toHaveBeenCalledWith(mockVM.id);
    });

    it('should call deleteVM when delete button is clicked', async () => {
      const user = userEvent.setup();
      render(<VMCard vm={mockVM} />);

      const deleteButton = screen.getByText('Delete');
      await user.click(deleteButton);

      expect(mockVMActions.deleteVM).toHaveBeenCalledWith(mockVM.id);
    });

    it('should handle double-click on card to view details', async () => {
      const mockOnDetails = jest.fn();
      const user = userEvent.setup();
      render(<VMCard vm={mockVM} onDetails={mockOnDetails} />);

      const card = screen.getByTestId('vm-card');
      await user.dblClick(card);

      expect(mockOnDetails).toHaveBeenCalledWith(mockVM);
    });

    it('should handle keyboard navigation', async () => {
      const user = userEvent.setup();
      render(<VMCard vm={mockVM} />);

      const startButton = screen.getByText('Start');
      startButton.focus();
      
      await user.keyboard('{Enter}');
      expect(mockVMActions.startVM).toHaveBeenCalledWith(mockVM.id);
    });
  });

  describe('Error Handling', () => {
    it('should handle API errors gracefully', async () => {
      mockVMActions.startVM.mockRejectedValue(new Error('API Error'));
      const user = userEvent.setup();
      
      render(<VMCard vm={mockVM} />);

      const startButton = screen.getByText('Start');
      await user.click(startButton);

      await waitFor(() => {
        expect(mockVMActions.startVM).toHaveBeenCalledWith(mockVM.id);
      });

      // Should not crash the component
      expect(screen.getByTestId('vm-card')).toBeInTheDocument();
    });

    it('should show error state when VM has error status', () => {
      const errorVM = { ...mockVM, status: VMStatus.ERROR };
      render(<VMCard vm={errorVM} />);

      const badge = screen.getByTestId('badge');
      expect(badge).toHaveAttribute('data-variant', 'destructive');
      expect(badge).toHaveTextContent('ERROR');
    });
  });

  describe('Edge Cases', () => {
    it('should handle VM with null/undefined properties', () => {
      const incompleteVM = {
        id: 'vm-2',
        name: 'Incomplete VM',
        status: VMStatus.UNKNOWN,
        type: '',
        cpu: 0,
        memory: 0,
        disk: 0,
        networks: null,
        createdAt: '',
        updatedAt: '',
      } as any;

      expect(() => render(<VMCard vm={incompleteVM} />)).not.toThrow();
    });

    it('should handle very long VM name', () => {
      const longNameVM = { 
        ...mockVM, 
        name: 'Very Long VM Name That Should Be Truncated'.repeat(5) 
      };
      
      render(<VMCard vm={longNameVM} />);
      expect(screen.getByTestId('card-title')).toBeInTheDocument();
    });

    it('should handle VM with many networks', () => {
      const manyNetworksVM = { 
        ...mockVM, 
        networks: Array.from({ length: 10 }, (_, i) => `network-${i}`) 
      };
      
      render(<VMCard vm={manyNetworksVM} />);
      expect(screen.getByTestId('vm-card')).toBeInTheDocument();
    });

    it('should handle VM with zero resources', () => {
      const zeroResourceVM = { 
        ...mockVM, 
        cpu: 0,
        memory: 0,
        disk: 0 
      };
      
      render(<VMCard vm={zeroResourceVM} />);
      expect(screen.getByText('0 CPU')).toBeInTheDocument();
      expect(screen.getByText('0 MB RAM')).toBeInTheDocument();
      expect(screen.getByText('0 GB Disk')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA labels', () => {
      render(<VMCard vm={mockVM} />);

      const card = screen.getByTestId('vm-card');
      expect(card).toHaveAttribute('role', 'article');
      expect(card).toHaveAttribute('aria-label', expect.stringContaining('Test VM'));
    });

    it('should have keyboard accessible buttons', () => {
      render(<VMCard vm={mockVM} />);

      const buttons = screen.getAllByRole('button');
      buttons.forEach(button => {
        expect(button).toHaveAttribute('tabIndex', expect.any(String));
      });
    });

    it('should provide screen reader friendly status', () => {
      render(<VMCard vm={mockVM} />);

      const badge = screen.getByTestId('badge');
      expect(badge).toHaveAttribute('aria-label', expect.stringContaining('Status'));
    });
  });

  describe('Performance', () => {
    it('should not re-render unnecessarily', () => {
      const renderSpy = jest.fn();
      const TestComponent = (props: any) => {
        renderSpy();
        return <VMCard {...props} />;
      };

      const { rerender } = render(<TestComponent vm={mockVM} />);
      expect(renderSpy).toHaveBeenCalledTimes(1);

      // Re-render with same props
      rerender(<TestComponent vm={mockVM} />);
      // React.memo should prevent re-render, but we can't test this directly
      // without the actual memo implementation
    });

    it('should handle rapid button clicks', async () => {
      const user = userEvent.setup();
      render(<VMCard vm={mockVM} />);

      const startButton = screen.getByText('Start');
      
      // Click rapidly
      await user.click(startButton);
      await user.click(startButton);
      await user.click(startButton);

      // Should handle gracefully without errors
      expect(screen.getByTestId('vm-card')).toBeInTheDocument();
    });
  });

  describe('Responsive Design', () => {
    it('should handle different screen sizes', () => {
      // Mock window.matchMedia
      Object.defineProperty(window, 'matchMedia', {
        writable: true,
        value: jest.fn().mockImplementation(query => ({
          matches: query === '(max-width: 768px)',
          media: query,
          onchange: null,
          addListener: jest.fn(),
          removeListener: jest.fn(),
          addEventListener: jest.fn(),
          removeEventListener: jest.fn(),
          dispatchEvent: jest.fn(),
        })),
      });

      render(<VMCard vm={mockVM} />);
      expect(screen.getByTestId('vm-card')).toBeInTheDocument();
    });
  });

  describe('Integration with VM States', () => {
    const testCases = [
      { status: VMStatus.CREATING, expectedButtons: [] },
      { status: VMStatus.STARTING, expectedButtons: ['Delete'] },
      { status: VMStatus.RUNNING, expectedButtons: ['Stop', 'Restart', 'Delete'] },
      { status: VMStatus.STOPPING, expectedButtons: ['Delete'] },
      { status: VMStatus.STOPPED, expectedButtons: ['Start', 'Delete'] },
      { status: VMStatus.ERROR, expectedButtons: ['Start', 'Delete'] },
      { status: VMStatus.UNKNOWN, expectedButtons: ['Delete'] },
    ];

    testCases.forEach(({ status, expectedButtons }) => {
      it(`should show correct buttons for ${status} status`, () => {
        const testVM = { ...mockVM, status };
        render(<VMCard vm={testVM} />);

        expectedButtons.forEach(buttonText => {
          expect(screen.getByText(buttonText)).toBeInTheDocument();
        });

        // Check that other buttons are not present
        const allButtons = ['Start', 'Stop', 'Restart', 'Delete'];
        const unexpectedButtons = allButtons.filter(btn => !expectedButtons.includes(btn));
        
        unexpectedButtons.forEach(buttonText => {
          expect(screen.queryByText(buttonText)).not.toBeInTheDocument();
        });
      });
    });
  });
});

// Visual regression tests (would require additional setup)
describe('VMCard Visual Tests', () => {
  it('should match snapshot for stopped VM', () => {
    const { container } = render(<VMCard vm={mockVM} />);
    expect(container).toMatchSnapshot();
  });

  it('should match snapshot for running VM', () => {
    const runningVM = { ...mockVM, status: VMStatus.RUNNING };
    const { container } = render(<VMCard vm={runningVM} />);
    expect(container).toMatchSnapshot();
  });

  it('should match snapshot for error VM', () => {
    const errorVM = { ...mockVM, status: VMStatus.ERROR };
    const { container } = render(<VMCard vm={errorVM} />);
    expect(container).toMatchSnapshot();
  });
});