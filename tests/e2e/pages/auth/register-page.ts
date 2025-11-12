import { Page, Locator } from '@playwright/test';
import { BasePage } from '../base-page';

/**
 * Registration data interface
 */
export interface RegistrationData {
  email: string;
  password: string;
  confirmPassword: string;
  firstName: string;
  lastName: string;
  organization?: string;
  agreeToTerms: boolean;
  subscribeToNewsletter?: boolean;
}

/**
 * Registration wizard step
 */
export type RegistrationStep = 'account' | 'profile' | 'organization' | 'confirmation';

/**
 * Register Page Object Model
 * Handles registration wizard and related operations
 */
export class RegisterPage extends BasePage {
  // Step 1: Account Information
  private readonly emailInput: Locator;
  private readonly passwordInput: Locator;
  private readonly confirmPasswordInput: Locator;
  private readonly showPasswordToggle: Locator;
  private readonly passwordStrengthIndicator: Locator;

  // Step 2: Profile Information
  private readonly firstNameInput: Locator;
  private readonly lastNameInput: Locator;
  private readonly phoneInput: Locator;

  // Step 3: Organization Information
  private readonly organizationNameInput: Locator;
  private readonly organizationSizeSelect: Locator;
  private readonly industrySelect: Locator;

  // Step 4: Confirmation
  private readonly termsCheckbox: Locator;
  private readonly newsletterCheckbox: Locator;
  private readonly submitButton: Locator;

  // Navigation
  private readonly nextButton: Locator;
  private readonly backButton: Locator;
  private readonly stepIndicator: Locator;

  // Common
  private readonly errorMessage: Locator;
  private readonly loginLink: Locator;

  constructor(page: Page) {
    super(page);

    // Step 1 locators
    this.emailInput = this.page.getByTestId('register-email-input');
    this.passwordInput = this.page.getByTestId('register-password-input');
    this.confirmPasswordInput = this.page.getByTestId('register-confirm-password-input');
    this.showPasswordToggle = this.page.getByTestId('register-show-password-toggle');
    this.passwordStrengthIndicator = this.page.getByTestId('register-password-strength');

    // Step 2 locators
    this.firstNameInput = this.page.getByTestId('register-first-name-input');
    this.lastNameInput = this.page.getByTestId('register-last-name-input');
    this.phoneInput = this.page.getByTestId('register-phone-input');

    // Step 3 locators
    this.organizationNameInput = this.page.getByTestId('register-organization-input');
    this.organizationSizeSelect = this.page.getByTestId('register-organization-size-select');
    this.industrySelect = this.page.getByTestId('register-industry-select');

    // Step 4 locators
    this.termsCheckbox = this.page.getByTestId('register-terms-checkbox');
    this.newsletterCheckbox = this.page.getByTestId('register-newsletter-checkbox');
    this.submitButton = this.page.getByTestId('register-submit-button');

    // Navigation locators
    this.nextButton = this.page.getByTestId('register-next-button');
    this.backButton = this.page.getByTestId('register-back-button');
    this.stepIndicator = this.page.getByTestId('register-step-indicator');

    // Common locators
    this.errorMessage = this.page.getByTestId('register-error-message');
    this.loginLink = this.page.getByTestId('register-login-link');
  }

  /**
   * Navigate to registration page
   * @returns RegisterPage instance for chaining
   */
  async navigate(): Promise<this> {
    await this.goto('/auth/register');
    await this.waitForLoad();
    return this;
  }

  /**
   * Fill account information (Step 1)
   * @param email - Email address
   * @param password - Password
   * @param confirmPassword - Password confirmation
   * @returns RegisterPage instance for chaining
   */
  async fillAccountInfo(
    email: string,
    password: string,
    confirmPassword: string
  ): Promise<this> {
    await this.emailInput.fill(email);
    await this.passwordInput.fill(password);
    await this.confirmPasswordInput.fill(confirmPassword);
    return this;
  }

  /**
   * Fill profile information (Step 2)
   * @param firstName - First name
   * @param lastName - Last name
   * @param phone - Phone number (optional)
   * @returns RegisterPage instance for chaining
   */
  async fillProfileInfo(
    firstName: string,
    lastName: string,
    phone?: string
  ): Promise<this> {
    await this.firstNameInput.fill(firstName);
    await this.lastNameInput.fill(lastName);
    if (phone) {
      await this.phoneInput.fill(phone);
    }
    return this;
  }

  /**
   * Fill organization information (Step 3)
   * @param organizationName - Organization name
   * @param size - Organization size
   * @param industry - Industry type
   * @returns RegisterPage instance for chaining
   */
  async fillOrganizationInfo(
    organizationName: string,
    size: string,
    industry: string
  ): Promise<this> {
    await this.organizationNameInput.fill(organizationName);
    await this.organizationSizeSelect.selectOption(size);
    await this.industrySelect.selectOption(industry);
    return this;
  }

  /**
   * Accept terms and conditions
   * @param subscribe - Whether to subscribe to newsletter
   * @returns RegisterPage instance for chaining
   */
  async acceptTerms(subscribe: boolean = false): Promise<this> {
    await this.termsCheckbox.check();
    if (subscribe) {
      await this.newsletterCheckbox.check();
    }
    return this;
  }

  /**
   * Click next button to proceed to next step
   * @returns RegisterPage instance for chaining
   */
  async clickNext(): Promise<this> {
    await this.nextButton.click();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Click back button to go to previous step
   * @returns RegisterPage instance for chaining
   */
  async clickBack(): Promise<this> {
    await this.backButton.click();
    return this;
  }

  /**
   * Submit registration form
   * @returns RegisterPage instance for chaining
   */
  async submit(): Promise<this> {
    await this.submitButton.click();
    return this;
  }

  /**
   * Complete full registration flow
   * @param data - Registration data
   * @returns RegisterPage instance for chaining
   */
  async completeRegistration(data: RegistrationData): Promise<this> {
    // Step 1: Account Info
    await this.fillAccountInfo(data.email, data.password, data.confirmPassword);
    await this.clickNext();

    // Step 2: Profile Info
    await this.fillProfileInfo(data.firstName, data.lastName);
    await this.clickNext();

    // Step 3: Organization Info (optional)
    if (data.organization) {
      await this.organizationNameInput.fill(data.organization);
      await this.clickNext();
    } else {
      await this.clickNext();
    }

    // Step 4: Terms and Submit
    await this.acceptTerms(data.subscribeToNewsletter);
    await this.submit();

    return this;
  }

  /**
   * Complete registration and wait for confirmation
   * @param data - Registration data
   * @returns RegisterPage instance for chaining
   */
  async registerAndWaitForConfirmation(data: RegistrationData): Promise<this> {
    await this.completeRegistration(data);
    await this.page.waitForURL('**/auth/verify-email', { timeout: 10000 });
    return this;
  }

  /**
   * Get current registration step
   * @returns Current step name
   */
  async getCurrentStep(): Promise<string> {
    const stepText = await this.stepIndicator.textContent();
    return stepText?.toLowerCase() || '';
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
   * Toggle password visibility
   * @returns RegisterPage instance for chaining
   */
  async togglePasswordVisibility(): Promise<this> {
    await this.showPasswordToggle.click();
    return this;
  }

  /**
   * Click login link
   * @returns RegisterPage instance for chaining
   */
  async clickLoginLink(): Promise<this> {
    await this.loginLink.click();
    await this.page.waitForURL('**/auth/login');
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
   * Get validation error for specific field
   * @param fieldName - Field name (email, password, firstName, etc.)
   * @returns Validation error text
   */
  async getFieldValidationError(fieldName: string): Promise<string> {
    const errorLocator = this.page.getByTestId(`register-${fieldName}-error`);
    return (await errorLocator.textContent()) || '';
  }

  /**
   * Check if submit button is disabled
   * @returns True if submit button is disabled
   */
  async isSubmitButtonDisabled(): Promise<boolean> {
    return await this.submitButton.isDisabled();
  }

  /**
   * Check if next button is disabled
   * @returns True if next button is disabled
   */
  async isNextButtonDisabled(): Promise<boolean> {
    return await this.nextButton.isDisabled();
  }

  /**
   * Verify registration page is loaded
   * @returns True if registration page is loaded
   */
  async isLoaded(): Promise<boolean> {
    try {
      await this.emailInput.waitFor({ state: 'visible', timeout: 5000 });
      await this.stepIndicator.waitFor({ state: 'visible', timeout: 5000 });
      return true;
    } catch {
      return false;
    }
  }
}
