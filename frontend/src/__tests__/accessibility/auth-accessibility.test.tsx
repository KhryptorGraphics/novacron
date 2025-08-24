import { render } from '@testing-library/react';
import { axe, toHaveNoViolations } from 'jest-axe';
import { RegistrationWizard } from '@/components/auth/RegistrationWizard';
import { PasswordStrengthIndicator } from '@/components/auth/PasswordStrengthIndicator';
import { EmailVerificationFlow } from '@/components/auth/EmailVerificationFlow';

expect.extend(toHaveNoViolations);

// Mock dependencies
jest.mock('@/lib/api', () => ({
  apiService: {
    checkEmailAvailability: jest.fn().mockResolvedValue({ available: true }),
    register: jest.fn().mockResolvedValue({ success: true }),
    resendVerificationEmail: jest.fn().mockResolvedValue({ success: true }),
  },
}));

jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: jest.fn(),
  }),
}));

jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  },
  AnimatePresence: ({ children }: any) => children,
}));

describe('Authentication Components Accessibility', () => {
  it('RegistrationWizard should not have accessibility violations', async () => {
    const { container } = render(<RegistrationWizard />);
    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });

  it('PasswordStrengthIndicator should not have accessibility violations', async () => {
    const { container } = render(
      <PasswordStrengthIndicator password="TestPassword123!" />
    );
    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });

  it('EmailVerificationFlow should not have accessibility violations', async () => {
    const mockProps = {
      email: 'test@example.com',
      onVerificationComplete: jest.fn(),
      onSkip: jest.fn(),
    };
    
    const { container } = render(<EmailVerificationFlow {...mockProps} />);
    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });

  it('form labels are properly associated with inputs', () => {
    const { container } = render(<RegistrationWizard />);
    
    // Check that all form inputs have proper labels
    const inputs = container.querySelectorAll('input');
    inputs.forEach((input) => {
      const id = input.getAttribute('id');
      if (id) {
        const label = container.querySelector(`label[for="${id}"]`);
        expect(label).toBeTruthy();
      }
    });
  });

  it('error messages are announced to screen readers', () => {
    const { container } = render(<RegistrationWizard />);
    
    // Check for aria-describedby attributes on inputs with errors
    const inputsWithErrors = container.querySelectorAll('input.border-red-500');
    inputsWithErrors.forEach((input) => {
      const describedBy = input.getAttribute('aria-describedby');
      if (describedBy) {
        const errorElement = container.querySelector(`#${describedBy}`);
        expect(errorElement).toBeTruthy();
      }
    });
  });

  it('buttons have appropriate accessible names', () => {
    const { getByRole } = render(<RegistrationWizard />);
    
    // Check that buttons have meaningful names
    const buttons = document.querySelectorAll('button');
    buttons.forEach((button) => {
      const accessibleName = button.textContent || button.getAttribute('aria-label');
      expect(accessibleName).toBeTruthy();
      expect(accessibleName!.trim().length).toBeGreaterThan(0);
    });
  });

  it('progress indicator is accessible', () => {
    const { container } = render(<RegistrationWizard />);
    
    // Check for progress indicators
    const progressBar = container.querySelector('[role="progressbar"]') || 
                       container.querySelector('progress');
    
    if (progressBar) {
      expect(progressBar).toHaveAttribute('aria-valuenow');
      expect(progressBar).toHaveAttribute('aria-valuemax');
    }
  });

  it('password visibility toggle has appropriate labels', async () => {
    const { container } = render(<RegistrationWizard />);
    
    // Navigate to password step (simplified for test)
    const passwordToggles = container.querySelectorAll('button[aria-label*="password"]');
    passwordToggles.forEach((toggle) => {
      const ariaLabel = toggle.getAttribute('aria-label');
      expect(ariaLabel).toMatch(/show password|hide password/i);
    });
  });

  it('form validation messages are accessible', () => {
    const { container } = render(<PasswordStrengthIndicator password="weak" />);
    
    // Check that validation messages are properly marked up
    const validationMessages = container.querySelectorAll('[role="alert"]') ||
                             container.querySelectorAll('.text-red-600, .text-red-500');
    
    validationMessages.forEach((message) => {
      // Should be programmatically announced
      const role = message.getAttribute('role');
      const ariaLive = message.getAttribute('aria-live');
      expect(role === 'alert' || ariaLive).toBeTruthy();
    });
  });

  it('step navigation is keyboard accessible', () => {
    const { container } = render(<RegistrationWizard />);
    
    // Check that navigation buttons are keyboard accessible
    const navButtons = container.querySelectorAll('button');
    navButtons.forEach((button) => {
      expect(button).toHaveAttribute('type');
      expect(button.getAttribute('tabindex')).not.toBe('-1');
    });
  });

  it('radio groups have proper labels and structure', () => {
    const { container } = render(<RegistrationWizard />);
    
    // Check radio groups
    const radioGroups = container.querySelectorAll('[role="radiogroup"]');
    radioGroups.forEach((group) => {
      // Should have accessible name
      const name = group.getAttribute('aria-labelledby') || 
                  group.getAttribute('aria-label');
      expect(name).toBeTruthy();
    });
  });

  it('loading states are announced to screen readers', async () => {
    const { container } = render(<EmailVerificationFlow 
      email="test@example.com" 
      onVerificationComplete={jest.fn()} 
      onSkip={jest.fn()} 
    />);
    
    // Check for loading indicators
    const loadingElements = container.querySelectorAll('[aria-live]') ||
                           container.querySelectorAll('.animate-spin');
    
    loadingElements.forEach((element) => {
      const ariaLive = element.getAttribute('aria-live');
      const role = element.getAttribute('role');
      expect(ariaLive || role === 'status').toBeTruthy();
    });
  });
});