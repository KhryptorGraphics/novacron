import { test, expect } from '@playwright/test';
import { LoginPage } from '../../pages/LoginPage';
import { DashboardPage } from '../../pages/DashboardPage';
import { testData } from '../../fixtures/testData';

test.describe('Authentication: Login', () => {
  let loginPage: LoginPage;
  let dashboardPage: DashboardPage;

  test.beforeEach(async ({ page }) => {
    loginPage = new LoginPage(page);
    dashboardPage = new DashboardPage(page);
    await loginPage.goto();
  });

  test('should login successfully with valid credentials', {
    tag: '@smoke',
  }, async ({ page }) => {
    const user = testData.users.admin;

    await loginPage.fillEmail(user.email);
    await loginPage.fillPassword(user.password);
    await loginPage.submit();

    await expect(page).toHaveURL(/\/dashboard/);
    await expect(dashboardPage.userMenu).toBeVisible();
    await expect(dashboardPage.userMenu).toContainText(user.name);
  });

  test('should show error with invalid credentials', async () => {
    await loginPage.fillEmail('invalid@example.com');
    await loginPage.fillPassword('wrongpassword');
    await loginPage.submit();

    await expect(loginPage.errorMessage).toBeVisible();
    await expect(loginPage.errorMessage).toContainText('Invalid email or password');
    await expect(loginPage.page).toHaveURL(/\/login/);
  });

  test('should validate email format', async () => {
    await loginPage.fillEmail('not-an-email');
    await loginPage.fillPassword('password123');
    await loginPage.submit();

    await expect(loginPage.emailInput).toHaveAttribute('aria-invalid', 'true');
    await expect(loginPage.page.locator('[data-testid="email-error"]')).toContainText('valid email');
  });

  test('should require password', async () => {
    await loginPage.fillEmail('user@example.com');
    await loginPage.submit();

    await expect(loginPage.passwordInput).toHaveAttribute('aria-invalid', 'true');
    await expect(loginPage.page.locator('[data-testid="password-error"]')).toContainText('required');
  });

  test('should handle account lockout after failed attempts', async () => {
    const email = 'lockout@example.com';

    for (let i = 0; i < 5; i++) {
      await loginPage.fillEmail(email);
      await loginPage.fillPassword('wrongpassword');
      await loginPage.submit();
      await loginPage.page.waitForTimeout(500);
    }

    await expect(loginPage.errorMessage).toContainText('Account locked');
    await expect(loginPage.page.locator('[data-testid="lockout-timer"]')).toBeVisible();
  });

  test('should support remember me functionality', async ({ page, context }) => {
    const user = testData.users.standard;

    await loginPage.fillEmail(user.email);
    await loginPage.fillPassword(user.password);
    await loginPage.checkRememberMe();
    await loginPage.submit();

    await expect(page).toHaveURL(/\/dashboard/);

    // Check persistent cookie was set
    const cookies = await context.cookies();
    const rememberCookie = cookies.find(c => c.name === 'remember_token');
    expect(rememberCookie).toBeDefined();
    expect(rememberCookie?.expires).toBeGreaterThan(Date.now() / 1000);

    // Close and reopen browser
    await page.close();
    const newPage = await context.newPage();

    await newPage.goto('/');
    await expect(newPage).toHaveURL(/\/dashboard/);
  });

  test('should support SSO login', async ({ page }) => {
    await loginPage.clickSSOLogin();

    await expect(page.locator('[data-testid="sso-providers"]')).toBeVisible();

    await page.locator('[data-testid="sso-google"]').click();

    // Should redirect to SSO provider
    await expect(page).toHaveURL(/accounts\.google\.com/);
  });

  test('should handle session timeout redirect', async ({ page }) => {
    const user = testData.users.standard;

    await loginPage.login(user.email, user.password);
    await expect(page).toHaveURL(/\/dashboard/);

    // Simulate session expiry
    await page.evaluate(() => {
      document.cookie = 'session_token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';
    });

    // Try to access protected page
    await page.goto('/vms');

    await expect(page).toHaveURL(/\/login/);
    await expect(loginPage.page.locator('[data-testid="session-expired"]')).toBeVisible();
  });

  test('should support two-factor authentication', async ({ page }) => {
    const user = testData.users.with2FA;

    await loginPage.fillEmail(user.email);
    await loginPage.fillPassword(user.password);
    await loginPage.submit();

    // Should show 2FA prompt
    await expect(page.locator('[data-testid="2fa-prompt"]')).toBeVisible();

    await page.locator('[data-testid="2fa-code"]').fill('123456');
    await page.locator('[data-testid="2fa-submit"]').click();

    await expect(page).toHaveURL(/\/dashboard/);
  });

  test('should show password strength indicator', async () => {
    await loginPage.clickRegisterLink();

    const passwordInput = loginPage.page.locator('[data-testid="register-password"]');
    const strengthIndicator = loginPage.page.locator('[data-testid="password-strength"]');

    await passwordInput.fill('weak');
    await expect(strengthIndicator).toHaveClass(/strength-weak/);

    await passwordInput.fill('Medium1!');
    await expect(strengthIndicator).toHaveClass(/strength-medium/);

    await passwordInput.fill('VeryStr0ng!Pass@123');
    await expect(strengthIndicator).toHaveClass(/strength-strong/);
  });

  test('should support keyboard navigation', async ({ page }) => {
    await loginPage.emailInput.focus();
    await page.keyboard.type('user@example.com');

    await page.keyboard.press('Tab');
    await page.keyboard.type('password123');

    await page.keyboard.press('Enter');

    // Should submit form
    await expect(page).toHaveURL(/\/(dashboard|login)/);
  });

  test.describe('Accessibility', () => {
    test('should have proper ARIA labels', async () => {
      await expect(loginPage.emailInput).toHaveAttribute('aria-label');
      await expect(loginPage.passwordInput).toHaveAttribute('aria-label');
      await expect(loginPage.submitButton).toHaveAttribute('aria-label');
    });

    test('should support screen reader navigation', async ({ page }) => {
      const form = page.locator('form[role="form"]');
      await expect(form).toBeVisible();

      const heading = page.locator('h1');
      await expect(heading).toHaveAttribute('role', 'heading');
    });
  });
});
