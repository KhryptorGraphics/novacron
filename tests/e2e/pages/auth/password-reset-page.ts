import { Page, Locator } from '@playwright/test';
import { BasePage } from '../base-page';

/**
 * Password reset flow type
 */
export type PasswordResetFlow = 'request' | 'verify' | 'reset';

/**
 * Password Reset Page Object Model
 * Handles password reset request, verification, and reset operations
 */
export class PasswordResetPage extends BasePage {
  // Request Reset locators
  private readonly emailInput: Locator;
  private readonly requestResetButton: Locator;

  // Verify Code locators
  private readonly verificationCodeInput: Locator;
  private readonly verifyCodeButton: Locator;
  private readonly resendCodeButton: Locator;

  // Reset Password locators
  private readonly newPasswordInput: Locator;
  private readonly confirmNewPasswordInput: Locator;
  private readonly showPasswordToggle: Locator;
  private readonly resetPasswordButton: Locator;
  private readonly passwordStrengthIndicator: Locator;

  // Common locators
  private readonly successMessage: Locator;
  private readonly errorMessage: Locator;
  private readonly backToLoginLink: Locator;
  private readonly flowIndicator: Locator;

  constructor(page: Page) {
    super(page);

    // Request Reset
    this.emailInput = this.page.getByTestId('reset-email-input');
    this.requestResetButton = this.page.getByTestId('reset-request-button');

    // Verify Code
    this.verificationCodeInput = this.page.getByTestId('reset-verification-code-input');
    this.verifyCodeButton = this.page.getByTestId('reset-verify-code-button');
    this.resendCodeButton = this.page.getByTestId('reset-resend-code-button');

    // Reset Password
    this.newPasswordInput = this.page.getByTestId('reset-new-password-input');
    this.confirmNewPasswordInput = this.page.getByTestId('reset-confirm-password-input');
    this.showPasswordToggle = this.page.getByTestId('reset-show-password-toggle');
    this.resetPasswordButton = this.page.getByTestId('reset-password-button');
    this.passwordStrengthIndicator = this.page.getByTestId('reset-password-strength');

    // Common
    this.successMessage = this.page.getByTestId('reset-success-message');
    this.errorMessage = this.page.getByTestId('reset-error-message');
    this.backToLoginLink = this.page.getByTestId('reset-back-to-login-link');
    this.flowIndicator = this.page.getByTestId('reset-flow-indicator');
  }

  /**
   * Navigate to password reset page
   * @returns PasswordResetPage instance for chaining
   */
  async navigate(): Promise<this> {
    await this.goto('/auth/forgot-password');
    await this.waitForLoad();
    return this;
  }

  /**
   * Request password reset
   * @param email - Email address
   * @returns PasswordResetPage instance for chaining
   */
  async requestReset(email: string): Promise<this> {
    await this.emailInput.fill(email);
    await this.requestResetButton.click();
    return this;
  }

  /**
   * Request reset and wait for verification step
   * @param email - Email address
   * @returns PasswordResetPage instance for chaining
   */
  async requestResetAndWaitForVerification(email: string): Promise<this> {
    await this.requestReset(email);
    await this.waitForStep('verify');
    return this;
  }

  /**
   * Enter verification code
   * @param code - Verification code
   * @returns PasswordResetPage instance for chaining
   */
  async enterVerificationCode(code: string): Promise<this> {
    await this.verificationCodeInput.fill(code);
    return this;
  }

  /**
   * Verify code
   * @returns PasswordResetPage instance for chaining
   */
  async verifyCode(): Promise<this> {
    await this.verifyCodeButton.click();
    return this;
  }

  /**
   * Verify code and wait for reset step
   * @param code - Verification code
   * @returns PasswordResetPage instance for chaining
   */
  async verifyCodeAndWaitForReset(code: string): Promise<this> {
    await this.enterVerificationCode(code);
    await this.verifyCode();
    await this.waitForStep('reset');
    return this;
  }

  /**
   * Resend verification code
   * @returns PasswordResetPage instance for chaining
   */
  async resendCode(): Promise<this> {
    await this.resendCodeButton.click();
    await this.waitForSuccessMessage();
    return this;
  }

  /**
   * Enter new password
   * @param password - New password
   * @param confirmPassword - Password confirmation
   * @returns PasswordResetPage instance for chaining
   */
  async enterNewPassword(password: string, confirmPassword: string): Promise<this> {
    await this.newPasswordInput.fill(password);
    await this.confirmNewPasswordInput.fill(confirmPassword);
    return this;
  }

  /**
   * Reset password
   * @returns PasswordResetPage instance for chaining
   */
  async resetPassword(): Promise<this> {
    await this.resetPasswordButton.click();
    return this;
  }

  /**
   * Complete password reset and wait for success
   * @param password - New password
   * @param confirmPassword - Password confirmation
   * @returns PasswordResetPage instance for chaining
   */
  async resetPasswordAndWaitForSuccess(
    password: string,
    confirmPassword: string
  ): Promise<this> {
    await this.enterNewPassword(password, confirmPassword);
    await this.resetPassword();
    await this.waitForSuccessMessage();
    return this;
  }

  /**
   * Complete full password reset flow
   * @param email - Email address
   * @param verificationCode - Verification code
   * @param newPassword - New password
   * @param confirmPassword - Password confirmation
   * @returns PasswordResetPage instance for chaining
   */
  async completePasswordReset(
    email: string,
    verificationCode: string,
    newPassword: string,
    confirmPassword: string
  ): Promise<this> {
    await this.requestResetAndWaitForVerification(email);
    await this.verifyCodeAndWaitForReset(verificationCode);
    await this.resetPasswordAndWaitForSuccess(newPassword, confirmPassword);
    return this;
  }

  /**
   * Toggle password visibility
   * @returns PasswordResetPage instance for chaining
   */
  async togglePasswordVisibility(): Promise<this> {
    await this.showPasswordToggle.click();
    return this;
  }

  /**
   * Click back to login link
   * @returns PasswordResetPage instance for chaining
   */
  async clickBackToLogin(): Promise<this> {
    await this.backToLoginLink.click();
    await this.page.waitForURL('**/auth/login');
    return this;
  }

  /**
   * Get current flow step
   * @returns Current flow step (request, verify, reset)
   */
  async getCurrentFlow(): Promise<PasswordResetFlow> {
    const flowText = await this.flowIndicator.getAttribute('data-flow');
    return (flowText as PasswordResetFlow) || 'request';
  }

  /**
   * Wait for specific flow step
   * @param flow - Flow step to wait for
   * @param timeout - Maximum time to wait
   * @returns PasswordResetPage instance for chaining
   */
  async waitForStep(flow: PasswordResetFlow, timeout: number = 10000): Promise<this> {
    await this.page.waitForFunction(
      (expectedFlow) => {
        const indicator = document.querySelector('[data-testid="reset-flow-indicator"]');
        return indicator?.getAttribute('data-flow') === expectedFlow;
      },
      flow,
      { timeout }
    );
    return this;
  }

  /**
   * Get password strength level
   * @returns Password strength (weak, fair, good, strong)
   */
  async getPasswordStrength(): Promise<string> {
    const strengthClass = await this.passwordStrengthIndicator.getAttribute('data-strength');
    return strengthClass || '';
  }

  /**
   * Get success message text
   * @returns Success message text
   */
  async getSuccessMessage(): Promise<string> {
    await this.successMessage.waitFor({ state: 'visible' });
    return (await this.successMessage.textContent()) || '';
  }

  /**
   * Get error message text
   * @returns Error message text
   */
  async getErrorMessage(): Promise<string> {
    await this.errorMessage.waitFor({ state: 'visible' });
    return (await this.errorMessage.textContent()) || '';
  }

  /**
   * Wait for success message to appear
   * @param timeout - Maximum time to wait
   * @returns PasswordResetPage instance for chaining
   */
  async waitForSuccessMessage(timeout: number = 10000): Promise<this> {
    await this.successMessage.waitFor({ state: 'visible', timeout });
    return this;
  }

  /**
   * Check if error message is visible
   * @returns True if error message is visible
   */
  async hasError(): Promise<boolean> {
    return await this.errorMessage.isVisible();
  }

  /**
   * Get validation error for specific field
   * @param fieldName - Field name
   * @returns Validation error text
   */
  async getFieldValidationError(fieldName: string): Promise<string> {
    const errorLocator = this.page.getByTestId(`reset-${fieldName}-error`);
    return (await errorLocator.textContent()) || '';
  }

  /**
   * Check if request button is disabled
   * @returns True if request button is disabled
   */
  async isRequestButtonDisabled(): Promise<boolean> {
    return await this.requestResetButton.isDisabled();
  }

  /**
   * Check if verify button is disabled
   * @returns True if verify button is disabled
   */
  async isVerifyButtonDisabled(): Promise<boolean> {
    return await this.verifyCodeButton.isDisabled();
  }

  /**
   * Check if reset button is disabled
   * @returns True if reset button is disabled
   */
  async isResetButtonDisabled(): Promise<boolean> {
    return await this.resetPasswordButton.isDisabled();
  }

  /**
   * Verify password reset page is loaded
   * @returns True if page is loaded
   */
  async isLoaded(): Promise<boolean> {
    try {
      await this.flowIndicator.waitFor({ state: 'visible', timeout: 5000 });
      return true;
    } catch {
      return false;
    }
  }
}
