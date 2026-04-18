import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { EmailVerificationFlow } from '@/components/auth/EmailVerificationFlow';

describe('EmailVerificationFlow', () => {
  it('renders an explicit unavailable state and allows skipping', async () => {
    const user = userEvent.setup();
    const onSkip = jest.fn();

    render(
      <EmailVerificationFlow
        email="test@test.local"
        onVerificationComplete={jest.fn()}
        onSkip={onSkip}
      />
    );

    expect(
      screen.getByText(/email verification is not available on the canonical release-candidate server yet/i)
    ).toBeInTheDocument();
    expect(screen.getByText('test@test.local')).toBeInTheDocument();

    const resendButton = screen.getByRole('button', { name: /resend email unavailable/i });
    expect(resendButton).toBeDisabled();

    await user.click(screen.getByRole('button', { name: /continue without verification/i }));
    expect(onSkip).toHaveBeenCalledTimes(1);
  });
});
