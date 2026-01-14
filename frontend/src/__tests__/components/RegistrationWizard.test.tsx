import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { RegistrationWizard } from '@/components/auth/RegistrationWizard';
import { RegistrationData } from '@/lib/validation';

// Mock next/navigation
jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: jest.fn(),
    back: jest.fn(),
  }),
}));

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

describe('RegistrationWizard', () => {
  const mockOnComplete = jest.fn();
  
  beforeEach(() => {
    jest.clearAllMocks();
  });
  
  it('renders the first step correctly', () => {
    render(<RegistrationWizard onComplete={mockOnComplete} />);
    
    expect(screen.getByText('Create Your NovaCron Account')).toBeInTheDocument();
    expect(screen.getByText('Step 1 of 3: Choose your account type')).toBeInTheDocument();
    expect(screen.getByText('Personal Account')).toBeInTheDocument();
    expect(screen.getByText('Organization Account')).toBeInTheDocument();
  });
  
  it('navigates through personal account flow', async () => {
    const user = userEvent.setup();
    render(<RegistrationWizard onComplete={mockOnComplete} />);
    
    // Step 1: Select Personal Account
    const personalRadio = screen.getByLabelText('Personal Account');
    await user.click(personalRadio);
    await user.click(screen.getByText('Next'));
    
    // Step 2: Personal Information
    await waitFor(() => {
      expect(screen.getByText('Step 2 of 3: Tell us about yourself')).toBeInTheDocument();
    });
    
    await user.type(screen.getByLabelText('First Name *'), 'John');
    await user.type(screen.getByLabelText('Last Name *'), 'Doe');
    await user.type(screen.getByLabelText('Email Address *'), 'john.doe@test.local');
    await user.click(screen.getByText('Next'));
    
    // Step 3: Security
    await waitFor(() => {
      expect(screen.getByText('Step 3 of 3: Secure your account')).toBeInTheDocument();
    });
    
    await user.type(screen.getByLabelText('Password *'), 'SecurePass123!');
    await user.type(screen.getByLabelText('Confirm Password *'), 'SecurePass123!');
    await user.click(screen.getByLabelText(/I accept the Terms/));
    
    await user.click(screen.getByText('Complete Registration'));
    
    await waitFor(() => {
      expect(mockOnComplete).toHaveBeenCalledWith(
        expect.objectContaining({
          accountType: 'personal',
          firstName: 'John',
          lastName: 'Doe',
          email: 'john.doe@test.local',
          password: 'SecurePass123!',
          confirmPassword: 'SecurePass123!',
          acceptTerms: true,
        })
      );
    });
  });
  
  it('shows organization fields for organization account', async () => {
    const user = userEvent.setup();
    render(<RegistrationWizard onComplete={mockOnComplete} />);
    
    // Select Organization Account
    const orgRadio = screen.getByLabelText('Organization Account');
    await user.click(orgRadio);
    await user.click(screen.getByText('Next'));
    
    // Fill personal info
    await waitFor(() => {
      expect(screen.getByText('Step 2 of 4: Tell us about yourself')).toBeInTheDocument();
    });
    
    await user.type(screen.getByLabelText('First Name *'), 'Jane');
    await user.type(screen.getByLabelText('Last Name *'), 'Smith');
    await user.type(screen.getByLabelText('Email Address *'), 'jane@company.com');
    await user.click(screen.getByText('Next'));
    
    // Organization details step should appear
    await waitFor(() => {
      expect(screen.getByText('Step 3 of 4: Organization details')).toBeInTheDocument();
      expect(screen.getByLabelText('Organization Name *')).toBeInTheDocument();
      expect(screen.getByLabelText('Organization Size *')).toBeInTheDocument();
    });
  });
  
  it('validates required fields', async () => {
    const user = userEvent.setup();
    render(<RegistrationWizard onComplete={mockOnComplete} />);
    
    // Try to proceed without selecting account type
    await user.click(screen.getByText('Next'));
    
    await waitFor(() => {
      expect(screen.getByText(/Please select an account type/)).toBeInTheDocument();
    });
  });
  
  it('shows password strength indicator', async () => {
    const user = userEvent.setup();
    render(<RegistrationWizard onComplete={mockOnComplete} />);
    
    // Navigate to security step
    const personalRadio = screen.getByLabelText('Personal Account');
    await user.click(personalRadio);
    await user.click(screen.getByText('Next'));
    
    await user.type(screen.getByLabelText('First Name *'), 'John');
    await user.type(screen.getByLabelText('Last Name *'), 'Doe');
    await user.type(screen.getByLabelText('Email Address *'), 'john@test.local');
    await user.click(screen.getByText('Next'));
    
    // Type weak password
    const passwordInput = screen.getByLabelText('Password *');
    await user.type(passwordInput, 'weak');
    
    await waitFor(() => {
      expect(screen.getByText(/Weak/)).toBeInTheDocument();
    });
    
    // Type strong password
    await user.clear(passwordInput);
    await user.type(passwordInput, 'StrongP@ssw0rd123!');
    
    await waitFor(() => {
      expect(screen.getByText(/Strong/)).toBeInTheDocument();
    });
  });
  
  it('validates password match', async () => {
    const user = userEvent.setup();
    render(<RegistrationWizard onComplete={mockOnComplete} />);
    
    // Navigate to security step
    const personalRadio = screen.getByLabelText('Personal Account');
    await user.click(personalRadio);
    await user.click(screen.getByText('Next'));
    
    await user.type(screen.getByLabelText('First Name *'), 'John');
    await user.type(screen.getByLabelText('Last Name *'), 'Doe');
    await user.type(screen.getByLabelText('Email Address *'), 'john@test.local');
    await user.click(screen.getByText('Next'));
    
    await user.type(screen.getByLabelText('Password *'), 'Password123!');
    await user.type(screen.getByLabelText('Confirm Password *'), 'DifferentPassword123!');
    
    await waitFor(() => {
      expect(screen.getByText('Passwords do not match')).toBeInTheDocument();
    });
  });
  
  it('allows navigation back to previous steps', async () => {
    const user = userEvent.setup();
    render(<RegistrationWizard onComplete={mockOnComplete} />);
    
    // Go to step 2
    const personalRadio = screen.getByLabelText('Personal Account');
    await user.click(personalRadio);
    await user.click(screen.getByText('Next'));
    
    await waitFor(() => {
      expect(screen.getByText('Step 2 of 3: Tell us about yourself')).toBeInTheDocument();
    });
    
    // Go back to step 1
    await user.click(screen.getByText('Back'));
    
    await waitFor(() => {
      expect(screen.getByText('Step 1 of 3: Choose your account type')).toBeInTheDocument();
    });
  });
  
  it('enables two-factor authentication option', async () => {
    const user = userEvent.setup();
    render(<RegistrationWizard onComplete={mockOnComplete} />);
    
    // Navigate to security step
    const personalRadio = screen.getByLabelText('Personal Account');
    await user.click(personalRadio);
    await user.click(screen.getByText('Next'));
    
    await user.type(screen.getByLabelText('First Name *'), 'John');
    await user.type(screen.getByLabelText('Last Name *'), 'Doe');
    await user.type(screen.getByLabelText('Email Address *'), 'john@test.local');
    await user.click(screen.getByText('Next'));
    
    // Enable 2FA
    const twoFactorCheckbox = screen.getByLabelText('Enable Two-Factor Authentication');
    await user.click(twoFactorCheckbox);
    
    await waitFor(() => {
      expect(screen.getByText(/You'll set up two-factor authentication/)).toBeInTheDocument();
    });
  });
});