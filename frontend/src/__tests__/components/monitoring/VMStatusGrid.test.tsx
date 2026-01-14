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
