import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { RegistrationWizard } from '@/components/auth/RegistrationWizard';
import { apiService } from '@/lib/api';

// Mock the API service
jest.mock('@/lib/api', () => ({
  apiService: {
    checkEmailAvailability: jest.fn(),
    register: jest.fn(),
  },
}));

// Mock router
const mockPush = jest.fn();
jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: mockPush,
  }),
}));

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  },
  AnimatePresence: ({ children }: any) => children,
}));

describe('RegistrationWizard', () => {
  const mockOnComplete = jest.fn();
  const user = userEvent.setup();

  beforeEach(() => {
    jest.clearAllMocks();
    (apiService.checkEmailAvailability as jest.Mock).mockResolvedValue({ available: true });
    (apiService.register as jest.Mock).mockResolvedValue({ success: true });
  });

  it('renders initial step with account type selection', () => {
    render(<RegistrationWizard onComplete={mockOnComplete} />);
    
    expect(screen.getByText('Create Your NovaCron Account')).toBeInTheDocument();
    expect(screen.getByText('Step 1 of 3: Choose your account type')).toBeInTheDocument();
    expect(screen.getByText('Personal Account')).toBeInTheDocument();
    expect(screen.getByText('Organization Account')).toBeInTheDocument();
  });

  it('progresses through personal account registration flow', async () => {
    render(<RegistrationWizard onComplete={mockOnComplete} />);
    
    // Step 1: Select personal account
    const personalRadio = screen.getByLabelText('Personal Account');
    await user.click(personalRadio);
    
    const nextButton = screen.getByRole('button', { name: /next/i });
    await user.click(nextButton);

    // Step 2: Personal information
    await waitFor(() => {
      expect(screen.getByText('Step 2 of 3')).toBeInTheDocument();
    });

    // Fill out personal information
    await user.type(screen.getByLabelText('First Name *'), 'John');
    await user.type(screen.getByLabelText('Last Name *'), 'Doe');
    await user.type(screen.getByLabelText('Email Address *'), 'john.doe@test.local');

    await user.click(nextButton);

    // Step 3: Security
    await waitFor(() => {
      expect(screen.getByText('Step 3 of 3')).toBeInTheDocument();
    });

    // Fill out password
    await user.type(screen.getByLabelText('Password *'), 'SecurePassword123!');
    await user.type(screen.getByLabelText('Confirm Password *'), 'SecurePassword123!');

    // Accept terms
    const termsCheckbox = screen.getByLabelText(/I accept the Terms of Service/);
    await user.click(termsCheckbox);

    // Complete registration
    const completeButton = screen.getByRole('button', { name: /complete registration/i });
    await user.click(completeButton);

    await waitFor(() => {
      expect(apiService.register).toHaveBeenCalledWith({
        firstName: 'John',
        lastName: 'Doe',
        email: 'john.doe@test.local',
        password: 'SecurePassword123!',
        accountType: 'personal',
        organizationName: '',
        organizationSize: '',
        phone: '',
        enableTwoFactor: false,
      });
    });
  });

  it('shows organization step for organization account', async () => {
    render(<RegistrationWizard onComplete={mockOnComplete} />);
    
    // Select organization account
    const orgRadio = screen.getByLabelText('Organization Account');
    await user.click(orgRadio);

    expect(screen.getByText('Step 1 of 4')).toBeInTheDocument();
    
    const nextButton = screen.getByRole('button', { name: /next/i });
    await user.click(nextButton);

    // Fill personal info
    await user.type(screen.getByLabelText('First Name *'), 'Jane');
    await user.type(screen.getByLabelText('Last Name *'), 'Smith');
    await user.type(screen.getByLabelText('Email Address *'), 'jane@company.com');

    await user.click(nextButton);

    // Should show organization step
    await waitFor(() => {
      expect(screen.getByText('Step 3 of 4')).toBeInTheDocument();
      expect(screen.getByLabelText('Organization Name *')).toBeInTheDocument();
      expect(screen.getByLabelText('Organization Size *')).toBeInTheDocument();
    });
  });

  it('validates required fields', async () => {
    render(<RegistrationWizard onComplete={mockOnComplete} />);
    
    const nextButton = screen.getByRole('button', { name: /next/i });
    await user.click(nextButton);

    await waitFor(() => {
      expect(screen.getByText('Please select an account type')).toBeInTheDocument();
    });
  });

  it('validates email availability', async () => {
    (apiService.checkEmailAvailability as jest.Mock).mockResolvedValue({ available: false });
    
    render(<RegistrationWizard onComplete={mockOnComplete} />);
    
    // Select personal account and go to next step
    const personalRadio = screen.getByLabelText('Personal Account');
    await user.click(personalRadio);
    await user.click(screen.getByRole('button', { name: /next/i }));

    // Type email that's not available
    const emailInput = screen.getByLabelText('Email Address *');
    await user.type(emailInput, 'taken@test.local');

    await waitFor(() => {
      expect(screen.getByText('This email is already registered')).toBeInTheDocument();
    });

    expect(apiService.checkEmailAvailability).toHaveBeenCalledWith('taken@test.local');
  });

  it('validates password strength', async () => {
    render(<RegistrationWizard onComplete={mockOnComplete} />);
    
    // Navigate to security step
    await user.click(screen.getByLabelText('Personal Account'));
    await user.click(screen.getByRole('button', { name: /next/i }));
    
    await user.type(screen.getByLabelText('First Name *'), 'John');
    await user.type(screen.getByLabelText('Last Name *'), 'Doe');
    await user.type(screen.getByLabelText('Email Address *'), 'john@test.local');
    await user.click(screen.getByRole('button', { name: /next/i }));

    // Type weak password
    await user.type(screen.getByLabelText('Password *'), '123');

    await waitFor(() => {
      expect(screen.getByText('Very Weak')).toBeInTheDocument();
    });
  });

  it('shows password mismatch error', async () => {
    render(<RegistrationWizard onComplete={mockOnComplete} />);
    
    // Navigate to security step
    await user.click(screen.getByLabelText('Personal Account'));
    await user.click(screen.getByRole('button', { name: /next/i }));
    
    await user.type(screen.getByLabelText('First Name *'), 'John');
    await user.type(screen.getByLabelText('Last Name *'), 'Doe');
    await user.type(screen.getByLabelText('Email Address *'), 'john@test.local');
    await user.click(screen.getByRole('button', { name: /next/i }));

    // Type mismatched passwords
    await user.type(screen.getByLabelText('Password *'), 'Password123!');
    await user.type(screen.getByLabelText('Confirm Password *'), 'Different123!');

    await waitFor(() => {
      expect(screen.getByText('Passwords do not match')).toBeInTheDocument();
    });
  });

  it('handles registration errors gracefully', async () => {
    (apiService.register as jest.Mock).mockRejectedValue(new Error('Registration failed'));
    
    render(<RegistrationWizard onComplete={mockOnComplete} />);
    
    // Complete the form
    await user.click(screen.getByLabelText('Personal Account'));
    await user.click(screen.getByRole('button', { name: /next/i }));
    
    await user.type(screen.getByLabelText('First Name *'), 'John');
    await user.type(screen.getByLabelText('Last Name *'), 'Doe');
    await user.type(screen.getByLabelText('Email Address *'), 'john@test.local');
    await user.click(screen.getByRole('button', { name: /next/i }));

    await user.type(screen.getByLabelText('Password *'), 'SecurePassword123!');
    await user.type(screen.getByLabelText('Confirm Password *'), 'SecurePassword123!');
    await user.click(screen.getByLabelText(/I accept the Terms of Service/));

    const completeButton = screen.getByRole('button', { name: /complete registration/i });
    await user.click(completeButton);

    await waitFor(() => {
      expect(screen.getByText('Registration failed. Please try again.')).toBeInTheDocument();
    });
  });

  it('can navigate back through steps', async () => {
    render(<RegistrationWizard onComplete={mockOnComplete} />);
    
    // Go to step 2
    await user.click(screen.getByLabelText('Personal Account'));
    await user.click(screen.getByRole('button', { name: /next/i }));

    expect(screen.getByText('Step 2 of 3')).toBeInTheDocument();

    // Go back to step 1
    const backButton = screen.getByRole('button', { name: /back/i });
    await user.click(backButton);

    expect(screen.getByText('Step 1 of 3')).toBeInTheDocument();
  });
});