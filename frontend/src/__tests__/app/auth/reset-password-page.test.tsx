import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import ResetPasswordPage from '@/app/auth/reset-password/page';
import { authService } from '@/lib/auth';

const mockPush = jest.fn();
const mockToast = jest.fn();
const mockSearchParams = new URLSearchParams();

jest.mock('@/lib/auth', () => ({
  authService: {
    resetPassword: jest.fn(),
  },
}));

jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: mockPush,
  }),
  useSearchParams: () => mockSearchParams,
}));

jest.mock('@/components/ui/use-toast', () => ({
  useToast: () => ({
    toast: mockToast,
  }),
}));

describe('ResetPasswordPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockSearchParams.set('token', 'valid-reset-token');
    (authService.resetPassword as jest.Mock).mockResolvedValue({
      message: 'Password reset successfully',
    });
  });

  it('submits the token and password to the canonical reset-password route', async () => {
    const user = userEvent.setup();

    render(<ResetPasswordPage />);

    await user.type(screen.getByLabelText('New Password'), 'NewPassw0rd!');
    await user.type(screen.getByLabelText('Confirm Password'), 'NewPassw0rd!');
    await user.click(screen.getByRole('button', { name: /reset password/i }));

    await waitFor(() => {
      expect(authService.resetPassword).toHaveBeenCalledWith({
        token: 'valid-reset-token',
        password: 'NewPassw0rd!',
      });
    });

    expect(screen.getByText(/your password has been reset successfully/i)).toBeInTheDocument();
  });

  it('rejects mismatched password confirmation without calling the API', async () => {
    const user = userEvent.setup();

    render(<ResetPasswordPage />);

    await user.type(screen.getByLabelText('New Password'), 'NewPassw0rd!');
    await user.type(screen.getByLabelText('Confirm Password'), 'Different1!');
    await user.click(screen.getByRole('button', { name: /reset password/i }));

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({ title: 'Error' }),
      );
    });

    expect(authService.resetPassword).not.toHaveBeenCalled();
  });

  it('shows the invalid-link message when the token is missing', () => {
    mockSearchParams.delete('token');

    render(<ResetPasswordPage />);

    expect(screen.getByText(/this reset link is invalid/i)).toBeInTheDocument();
    expect(authService.resetPassword).not.toHaveBeenCalled();
  });
});