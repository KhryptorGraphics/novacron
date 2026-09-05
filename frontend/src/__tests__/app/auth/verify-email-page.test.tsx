import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import VerifyEmailPage from '@/app/auth/verify-email/page';
import { apiService } from '@/lib/api';

const mockPush = jest.fn();
const mockToast = jest.fn();
const mockSearchParams = new URLSearchParams();

jest.mock('@/lib/api', () => ({
  apiService: {
    verifyEmail: jest.fn(),
    resendVerificationEmail: jest.fn(),
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

describe('VerifyEmailPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockSearchParams.delete('token');
    (apiService.verifyEmail as jest.Mock).mockResolvedValue({ success: true });
    (apiService.resendVerificationEmail as jest.Mock).mockResolvedValue({ success: true });
  });

  it('calls verifyEmail with the token on mount and shows success', async () => {
    mockSearchParams.set('token', 'valid-verify-token');

    render(<VerifyEmailPage />);

    await waitFor(() => {
      expect(apiService.verifyEmail).toHaveBeenCalledWith('valid-verify-token');
    });

    await waitFor(() => {
      expect(screen.getByText(/your email has been verified/i)).toBeInTheDocument();
    });
  });

  it('shows the invalid-link error when verification fails', async () => {
    mockSearchParams.set('token', 'expired-token');
    (apiService.verifyEmail as jest.Mock).mockRejectedValue(
      new Error('invalid or expired token'),
    );

    render(<VerifyEmailPage />);

    await waitFor(() => {
      expect(apiService.verifyEmail).toHaveBeenCalledWith('expired-token');
    });

    await waitFor(() => {
      expect(screen.getByText(/invalid or expired token/i)).toBeInTheDocument();
    });
  });

  it('sends the email through the canonical resend-verification route when no token is present', async () => {
    const user = userEvent.setup();

    render(<VerifyEmailPage />);

    await user.type(screen.getByLabelText('Email'), 'user@example.com');
    await user.click(screen.getByRole('button', { name: /resend verification email/i }));

    await waitFor(() => {
      expect(apiService.resendVerificationEmail).toHaveBeenCalledWith('user@example.com');
    });

    expect(screen.getByText(/verification instructions sent/i)).toBeInTheDocument();
  });
});