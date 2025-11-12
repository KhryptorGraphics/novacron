import { test, expect } from '@playwright/test';
import { LoginPage } from '../../pages/LoginPage';
import { RegistrationPage } from '../../pages/RegistrationPage';
import { testData } from '../../fixtures/testData';

test.describe('Authentication: Registration', () => {
  let loginPage: LoginPage;
  let registrationPage: RegistrationPage;

  test.beforeEach(async ({ page }) => {
    loginPage = new LoginPage(page);
    registrationPage = new RegistrationPage(page);

    await loginPage.goto();
    await loginPage.clickRegisterLink();
  });

  test('should complete registration wizard successfully', {
    tag: '@smoke',
  }, async ({ page }) => {
    const newUser = testData.users.generate();

    await test.step('Fill personal information', async () => {
      await registrationPage.fillName(newUser.name);
      await registrationPage.fillEmail(newUser.email);
      await registrationPage.fillPassword(newUser.password);
      await registrationPage.fillConfirmPassword(newUser.password);
      await registrationPage.nextStep();

      await expect(registrationPage.currentStep).toContainText('Organization');
    });

    await test.step('Fill organization details', async () => {
      await registrationPage.fillOrganization('Test Corp');
      await registrationPage.selectRole('developer');
      await registrationPage.nextStep();

      await expect(registrationPage.currentStep).toContainText('Terms');
    });

    await test.step('Accept terms and submit', async () => {
      await registrationPage.acceptTerms();
      await registrationPage.acceptPrivacy();
      await registrationPage.submit();

      await expect(registrationPage.successMessage).toBeVisible();
      await expect(page).toHaveURL(/\/verify-email/);
    });
  });

  test('should validate password requirements', async () => {
    await registrationPage.fillEmail('test@example.com');

    const weakPasswords = [
      'short',
      'nouppercase1!',
      'NOLOWERCASE1!',
      'NoNumbers!',
      'NoSpecial1'
    ];

    for (const password of weakPasswords) {
      await registrationPage.fillPassword(password);
      await registrationPage.page.keyboard.press('Tab');

      await expect(registrationPage.page.locator('[data-testid="password-error"]')).toBeVisible();
    }

    await registrationPage.fillPassword('ValidP@ssw0rd!');
    await expect(registrationPage.page.locator('[data-testid="password-error"]')).not.toBeVisible();
  });

  test('should validate password confirmation match', async () => {
    await registrationPage.fillPassword('ValidP@ssw0rd!');
    await registrationPage.fillConfirmPassword('DifferentP@ssw0rd!');

    await registrationPage.page.keyboard.press('Tab');

    await expect(registrationPage.page.locator('[data-testid="confirm-password-error"]'))
      .toContainText('Passwords do not match');
  });

  test('should detect duplicate email', async () => {
    const existingUser = testData.users.admin;

    await registrationPage.fillName('New User');
    await registrationPage.fillEmail(existingUser.email);
    await registrationPage.fillPassword('ValidP@ssw0rd!');
    await registrationPage.fillConfirmPassword('ValidP@ssw0rd!');
    await registrationPage.nextStep();

    await expect(registrationPage.errorMessage).toContainText('Email already registered');
  });

  test('should validate email format', async () => {
    const invalidEmails = [
      'notanemail',
      'missing@domain',
      '@nodomain.com',
      'spaces in@email.com'
    ];

    for (const email of invalidEmails) {
      await registrationPage.fillEmail(email);
      await registrationPage.page.keyboard.press('Tab');

      await expect(registrationPage.page.locator('[data-testid="email-error"]'))
        .toContainText('valid email');

      await registrationPage.page.locator('[data-testid="email-input"]').clear();
    }
  });

  test('should support social registration', async ({ page }) => {
    await registrationPage.clickSocialRegister('google');

    await expect(page).toHaveURL(/accounts\.google\.com/);
  });

  test('should handle registration errors gracefully', async () => {
    // Simulate server error
    await registrationPage.page.route('**/api/auth/register', route => {
      route.fulfill({
        status: 500,
        body: JSON.stringify({ error: 'Server error' })
      });
    });

    const newUser = testData.users.generate();
    await registrationPage.fillRegistrationForm(newUser);
    await registrationPage.acceptTerms();
    await registrationPage.submit();

    await expect(registrationPage.errorMessage).toBeVisible();
    await expect(registrationPage.errorMessage).toContainText('error occurred');
  });

  test('should save progress on wizard navigation', async ({ page }) => {
    const newUser = testData.users.generate();

    await registrationPage.fillName(newUser.name);
    await registrationPage.fillEmail(newUser.email);
    await registrationPage.nextStep();

    await registrationPage.fillOrganization('Test Corp');
    await registrationPage.previousStep();

    // Values should be preserved
    await expect(registrationPage.page.locator('[data-testid="name-input"]'))
      .toHaveValue(newUser.name);
    await expect(registrationPage.page.locator('[data-testid="email-input"]'))
      .toHaveValue(newUser.email);
  });

  test('should require terms acceptance', async () => {
    const newUser = testData.users.generate();

    await registrationPage.fillRegistrationForm(newUser);
    await registrationPage.nextStep();
    await registrationPage.fillOrganization('Test Corp');
    await registrationPage.nextStep();

    // Try to submit without accepting terms
    await registrationPage.submit();

    await expect(registrationPage.page.locator('[data-testid="terms-error"]'))
      .toContainText('accept the terms');
  });
});
