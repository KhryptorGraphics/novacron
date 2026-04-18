import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import RegisterPage from '@/app/auth/register/page';
import { authService } from '@/lib/auth';

const mockToast = jest.fn();

jest.mock('@/lib/auth', () => ({
  authService: {
    register: jest.fn(),
  },
}));

jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: jest.fn(),
  }),
}));

jest.mock('@/components/ui/use-toast', () => ({
  useToast: () => ({
    toast: mockToast,
  }),
}));

jest.mock('@/components/auth/RegistrationWizard', () => ({
  RegistrationWizard: ({ onComplete }: { onComplete: (data: any) => Promise<void> }) => (
    <button
      type="button"
      onClick={() =>
        onComplete({
          accountType: 'organization',
          firstName: 'Jane',
          lastName: 'Doe',
          email: 'jane@example.com',
          password: 'SecurePassword123!',
          confirmPassword: 'SecurePassword123!',
          organizationName: 'NovaCorp',
          organizationSize: '51-200',
          phone: '+1-555-0100',
          acceptTerms: true,
          enableTwoFactor: true,
        })
      }
    >
      Complete mocked registration
    </button>
  ),
}));

describe('RegisterPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (authService.register as jest.Mock).mockResolvedValue({
      id: 'user-1',
      email: 'jane@example.com',
      firstName: 'Jane',
      lastName: 'Doe',
      status: 'active',
    });
  });

  it('submits canonical registration payload through authService', async () => {
    const user = userEvent.setup();

    render(<RegisterPage />);

    await user.click(screen.getByRole('button', { name: 'Complete mocked registration' }));

    await waitFor(() => {
      expect(authService.register).toHaveBeenCalledWith({
        firstName: 'Jane',
        lastName: 'Doe',
        email: 'jane@example.com',
        password: 'SecurePassword123!',
      });
    });
  });
});
