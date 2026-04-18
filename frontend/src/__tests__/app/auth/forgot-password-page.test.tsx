import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import ForgotPasswordPage from '@/app/auth/forgot-password/page';
import { authService } from '@/lib/auth';

const mockPush = jest.fn();
const mockToast = jest.fn();

jest.mock('@/lib/auth', () => ({
  authService: {
    forgotPassword: jest.fn(),
  },
}));

jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: mockPush,
  }),
}));

jest.mock('@/components/ui/use-toast', () => ({
  useToast: () => ({
    toast: mockToast,
  }),
}));

describe('ForgotPasswordPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (authService.forgotPassword as jest.Mock).mockResolvedValue({
      message: 'Password reset email sent',
    });
  });

  it('calls the canonical forgot-password route via authService', async () => {
    const user = userEvent.setup();

    render(<ForgotPasswordPage />);

    await user.type(screen.getByLabelText('Email'), 'user@example.com');
    await user.click(screen.getByRole('button', { name: /send reset instructions/i }));

    await waitFor(() => {
      expect(authService.forgotPassword).toHaveBeenCalledWith({ email: 'user@example.com' });
    });

    expect(screen.getByText(/please check your email for password reset instructions/i)).toBeInTheDocument();
  });
});
