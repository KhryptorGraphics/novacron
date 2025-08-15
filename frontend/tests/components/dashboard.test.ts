import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { JobList } from '@/components/dashboard/job-list';
import { WorkflowList } from '@/components/dashboard/workflow-list';
import { useJobs, useWorkflows } from '@/hooks/useAPI';

// Mock the API hooks
jest.mock('@/hooks/useAPI', () => ({
  useJobs: jest.fn(),
  useWorkflows: jest.fn(),
  useWebSocket: jest.fn().mockReturnValue({ connected: true, lastMessage: null })
}));

// Mock useRouter
jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: jest.fn(),
  }),
}));

describe('JobList Component', () => {
  const mockJobs = [
    {
      id: '1',
      name: 'Daily Backup',
      schedule: '0 2 * * *',
      timezone: 'UTC',
      enabled: true,
      priority: 5,
      max_retries: 3,
      timeout: 30000,
      created_at: '2023-01-01T00:00:00Z',
      next_run_at: '2023-01-02T02:00:00Z'
    },
    {
      id: '2',
      name: 'Weekly Report',
      schedule: '0 0 * * 1',
      timezone: 'UTC',
      enabled: false,
      priority: 3,
      max_retries: 3,
      timeout: 30000,
      created_at: '2023-01-01T00:00:00Z'
    }
  ];

  const mockCreateJob = jest.fn();
  const mockUpdateJob = jest.fn();
  const mockDeleteJob = jest.fn();
  const mockExecuteJob = jest.fn();
  const mockRefetch = jest.fn();

  beforeEach(() => {
    (useJobs as jest.Mock).mockReturnValue({
      jobs: mockJobs,
      loading: false,
      error: null,
      createJob: mockCreateJob,
      updateJob: mockUpdateJob,
      deleteJob: mockDeleteJob,
      executeJob: mockExecuteJob,
      refetch: mockRefetch
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('renders job list correctly', () => {
    render(<JobList />);
    
    expect(screen.getByText('Job Management')).toBeInTheDocument();
    expect(screen.getByText('Daily Backup')).toBeInTheDocument();
    expect(screen.getByText('Weekly Report')).toBeInTheDocument();
    expect(screen.getByText('Create Job')).toBeInTheDocument();
  });

  it('shows loading state', () => {
    (useJobs as jest.Mock).mockReturnValue({
      jobs: null,
      loading: true,
      error: null,
      createJob: mockCreateJob,
      updateJob: mockUpdateJob,
      deleteJob: mockDeleteJob,
      executeJob: mockExecuteJob,
      refetch: mockRefetch
    });
    
    render(<JobList />);
    expect(screen.getByText('Loading jobs...')).toBeInTheDocument();
  });

  it('shows error state', () => {
    (useJobs as jest.Mock).mockReturnValue({
      jobs: null,
      loading: false,
      error: 'Failed to load jobs',
      createJob: mockCreateJob,
      updateJob: mockUpdateJob,
      deleteJob: mockDeleteJob,
      executeJob: mockExecuteJob,
      refetch: mockRefetch
    });
    
    render(<JobList />);
    expect(screen.getByText('Error loading jobs: Failed to load jobs')).toBeInTheDocument();
  });

  it('opens create job dialog', async () => {
    const user = userEvent.setup();
    render(<JobList />);
    
    const createButton = screen.getByText('Create Job');
    await user.click(createButton);
    
    expect(screen.getByText('Create New Job')).toBeInTheDocument();
  });

  it('executes a job', async () => {
    const user = userEvent.setup();
    render(<JobList />);
    
    const runButtons = screen.getAllByText('Run');
    await user.click(runButtons[0]);
    
    expect(mockExecuteJob).toHaveBeenCalledWith('1');
  });

  it('deletes a job', async () => {
    const user = userEvent.setup();
    window.confirm = jest.fn(() => true);
    
    render(<JobList />);
    
    const deleteButtons = screen.getAllByText('Delete');
    await user.click(deleteButtons[0]);
    
    expect(mockDeleteJob).toHaveBeenCalledWith('1');
  });
});

describe('WorkflowList Component', () => {
  const mockWorkflows = [
    {
      id: '1',
      name: 'Data Pipeline',
      description: 'Process daily data',
      nodes: [],
      edges: [],
      enabled: true,
      createdAt: '2023-01-01T00:00:00Z',
      updatedAt: '2023-01-01T00:00:00Z'
    }
  ];

  const mockCreateWorkflow = jest.fn();
  const mockUpdateWorkflow = jest.fn();
  const mockDeleteWorkflow = jest.fn();
  const mockExecuteWorkflow = jest.fn();
  const mockRefetch = jest.fn();

  beforeEach(() => {
    (useWorkflows as jest.Mock).mockReturnValue({
      workflows: mockWorkflows,
      loading: false,
      error: null,
      createWorkflow: mockCreateWorkflow,
      updateWorkflow: mockUpdateWorkflow,
      deleteWorkflow: mockDeleteWorkflow,
      executeWorkflow: mockExecuteWorkflow,
      refetch: mockRefetch
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('renders workflow list correctly', () => {
    render(<WorkflowList />);
    
    expect(screen.getByText('Workflow Management')).toBeInTheDocument();
    expect(screen.getByText('Data Pipeline')).toBeInTheDocument();
    expect(screen.getByText('Create Workflow')).toBeInTheDocument();
  });

  it('opens create workflow dialog', async () => {
    const user = userEvent.setup();
    render(<WorkflowList />);
    
    const createButton = screen.getByText('Create Workflow');
    await user.click(createButton);
    
    expect(screen.getByText('Create New Workflow')).toBeInTheDocument();
  });

  it('executes a workflow', async () => {
    const user = userEvent.setup();
    render(<WorkflowList />);
    
    const runButtons = screen.getAllByText('Run');
    await user.click(runButtons[0]);
    
    expect(mockExecuteWorkflow).toHaveBeenCalledWith('1');
  });
});