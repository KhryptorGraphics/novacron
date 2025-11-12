import { Page, Locator } from '@playwright/test';
import { BasePage } from '../base-page';

/**
 * Login credentials interface
 */
export interface LoginCredentials {
  email: string;
  password: string;
  rememberMe?: boolean;
}

/**
 * Login Page Object Model
 * Handles login functionality and related operations
 */
export class LoginPage extends BasePage {
  // Locators
  private readonly emailInput: Locator;
  private readonly passwordInput: Locator;
  private readonly rememberMeCheckbox: Locator;
  private readonly loginButton: Locator;
  private readonly forgotPasswordLink: Locator;
  private readonly signUpLink: Locator;
  private readonly errorMessage: Locator;
  private readonly showPasswordToggle: Locator;
  private readonly ssoButtons: Locator;

  constructor(page: Page) {
    super(page);

    // Initialize locators using test IDs
    this.emailInput = this.page.getByTestId('login-email-input');
    this.passwordInput = this.page.getByTestId('login-password-input');
    this.rememberMeCheckbox = this.page.getByTestId('login-remember-me-checkbox');
    this.loginButton = this.page.getByTestId('login-submit-button');
    this.forgotPasswordLink = this.page.getByTestId('login-forgot-password-link');
    this.signUpLink = this.page.getByTestId('login-signup-link');
    this.errorMessage = this.page.getByTestId('login-error-message');
    this.showPasswordToggle = this.page.getByTestId('login-show-password-toggle');
    this.ssoButtons = this.page.getByTestId('login-sso-buttons');
  }

  /**
   * Navigate to login page
   * @returns LoginPage instance for chaining
   */
  async navigate(): Promise<this> {
    await this.goto('/auth/login');
    await this.waitForLoad();
    return this;
  }

  /**
   * Fill email input
   * @param email - Email address
   * @returns LoginPage instance for chaining
   */
  async enterEmail(email: string): Promise<this> {
    await this.emailInput.fill(email);
    return this;
  }

  /**
   * Fill password input
   * @param password - Password
   * @returns LoginPage instance for chaining
   */
  async enterPassword(password: string): Promise<this> {
    await this.passwordInput.fill(password);
    return this;
  }

  /**
   * Toggle remember me checkbox
   * @param checked - Whether to check or uncheck
   * @returns LoginPage instance for chaining
   */
  async toggleRememberMe(checked: boolean = true): Promise<this> {
    const isChecked = await this.rememberMeCheckbox.isChecked();
    if (isChecked !== checked) {
      await this.rememberMeCheckbox.click();
    }
    return this;
  }

  /**
   * Click login button
   * @returns LoginPage instance for chaining
   */
  async clickLogin(): Promise<this> {
    await this.loginButton.click();
    return this;
  }

  /**
   * Complete login flow
   * @param credentials - Login credentials
   * @returns LoginPage instance for chaining
   */
  async login(credentials: LoginCredentials): Promise<this> {
    await this.enterEmail(credentials.email);
    await this.enterPassword(credentials.password);

    if (credentials.rememberMe !== undefined) {
      await this.toggleRememberMe(credentials.rememberMe);
    }

    await this.clickLogin();
    return this;
  }

  /**
   * Login and wait for navigation to dashboard
   * @param credentials - Login credentials
   * @returns LoginPage instance for chaining
   */
  async loginAndWaitForDashboard(credentials: LoginCredentials): Promise<this> {
    await this.login(credentials);
    await this.page.waitForURL('**/dashboard', { timeout: 10000 });
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Click forgot password link
   * @returns LoginPage instance for chaining
   */
  async clickForgotPassword(): Promise<this> {
    await this.forgotPasswordLink.click();
    await this.page.waitForURL('**/auth/forgot-password');
    return this;
  }

  /**
   * Click sign up link
   * @returns LoginPage instance for chaining
   */
  async clickSignUp(): Promise<this> {
    await this.signUpLink.click();
    await this.page.waitForURL('**/auth/register');
    return this;
  }

  /**
   * Toggle password visibility
   * @returns LoginPage instance for chaining
   */
  async togglePasswordVisibility(): Promise<this> {
    await this.showPasswordToggle.click();
    return this;
  }

  /**
   * Login with SSO provider
   * @param provider - SSO provider name (google, github, microsoft)
   * @returns LoginPage instance for chaining
   */
  async loginWithSSO(provider: 'google' | 'github' | 'microsoft'): Promise<this> {
    await this.page.getByTestId(`login-sso-${provider}-button`).click();
    return this;
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
   * Check if error message is visible
   * @returns True if error message is visible
   */
  async hasError(): Promise<boolean> {
    return await this.errorMessage.isVisible();
  }

  /**
   * Check if login button is disabled
   * @returns True if login button is disabled
   */
  async isLoginButtonDisabled(): Promise<boolean> {
    return await this.loginButton.isDisabled();
  }

  /**
   * Check if login button is loading
   * @returns True if login button shows loading state
   */
  async isLoginButtonLoading(): Promise<boolean> {
    const loadingIcon = this.loginButton.locator('[data-testid="loading-icon"]');
    return await loadingIcon.isVisible();
  }

  /**
   * Clear login form
   * @returns LoginPage instance for chaining
   */
  async clearForm(): Promise<this> {
    await this.emailInput.clear();
    await this.passwordInput.clear();
    return this;
  }

  /**
   * Verify login page is loaded
   * @returns True if login page is loaded
   */
  async isLoaded(): Promise<boolean> {
    try {
      await this.emailInput.waitFor({ state: 'visible', timeout: 5000 });
      await this.passwordInput.waitFor({ state: 'visible', timeout: 5000 });
      await this.loginButton.waitFor({ state: 'visible', timeout: 5000 });
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get input validation error for email
   * @returns Validation error text
   */
  async getEmailValidationError(): Promise<string> {
    const errorLocator = this.page.getByTestId('login-email-error');
    return (await errorLocator.textContent()) || '';
  }

  /**
   * Get input validation error for password
   * @returns Validation error text
   */
  async getPasswordValidationError(): Promise<string> {
    const errorLocator = this.page.getByTestId('login-password-error');
    return (await errorLocator.textContent()) || '';
  }
}
