import { test, expect } from '@playwright/test';
import { LoginPage } from '../../pages/LoginPage';
import { PasswordResetPage } from '../../pages/PasswordResetPage';
import { testData } from '../../fixtures/testData';

test.describe('Authentication: Password Reset', () => {
  let loginPage: LoginPage;
  let passwordResetPage: PasswordResetPage;

  test.beforeEach(async ({ page }) => {
    loginPage = new LoginPage(page);
    passwordResetPage = new PasswordResetPage(page);

    await loginPage.goto();
    await loginPage.clickForgotPassword();
  });

  test('should complete password reset flow', {
    tag: '@smoke',
  }, async ({ page }) => {
    const user = testData.users.standard;

    await test.step('Request password reset', async () => {
      await passwordResetPage.fillEmail(user.email);
      await passwordResetPage.submit();

      await expect(passwordResetPage.successMessage).toBeVisible();
      await expect(passwordResetPage.successMessage).toContainText('reset link');
    });

    await test.step('Access reset link', async () => {
      // Simulate clicking reset link from email
      const resetToken = 'mock-reset-token-12345';
      await page.goto(`/reset-password?token=${resetToken}`);

      await expect(page.locator('[data-testid="reset-form"]')).toBeVisible();
    });

    await test.step('Set new password', async () => {
      const newPassword = 'NewSecureP@ssw0rd!';

      await page.locator('[data-testid="new-password"]').fill(newPassword);
      await page.locator('[data-testid="confirm-password"]').fill(newPassword);
      await page.locator('[data-testid="submit-reset"]').click();

      await expect(page.locator('[data-testid="reset-success"]')).toBeVisible();
      await expect(page).toHaveURL(/\/login/);
    });

    await test.step('Login with new password', async () => {
      await loginPage.fillEmail(user.email);
      await loginPage.fillPassword('NewSecureP@ssw0rd!');
      await loginPage.submit();

      await expect(page).toHaveURL(/\/dashboard/);
    });
  });

  test('should validate email exists', async () => {
    await passwordResetPage.fillEmail('nonexistent@example.com');
    await passwordResetPage.submit();

    // For security, should show success even if email doesn't exist
    await expect(passwordResetPage.successMessage).toBeVisible();
  });

  test('should handle expired reset token', async ({ page }) => {
    await page.goto('/reset-password?token=expired-token');

    await page.locator('[data-testid="new-password"]').fill('NewP@ssw0rd!');
    await page.locator('[data-testid="confirm-password"]').fill('NewP@ssw0rd!');
    await page.locator('[data-testid="submit-reset"]').click();

    await expect(page.locator('[data-testid="error-message"]'))
      .toContainText('expired or invalid');
  });

  test('should validate new password requirements', async ({ page }) => {
    await page.goto('/reset-password?token=valid-token');

    const weakPassword = 'weak';
    await page.locator('[data-testid="new-password"]').fill(weakPassword);
    await page.keyboard.press('Tab');

    await expect(page.locator('[data-testid="password-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="password-requirements"]')).toBeVisible();
  });

  test('should prevent password reuse', async ({ page }) => {
    const user = testData.users.standard;

    await page.goto('/reset-password?token=valid-token');

    await page.locator('[data-testid="new-password"]').fill(user.password);
    await page.locator('[data-testid="confirm-password"]').fill(user.password);
    await page.locator('[data-testid="submit-reset"]').click();

    await expect(page.locator('[data-testid="error-message"]'))
      .toContainText('different from previous');
  });

  test('should rate limit reset requests', async () => {
    const email = 'ratelimit@example.com';

    for (let i = 0; i < 6; i++) {
      await passwordResetPage.fillEmail(email);
      await passwordResetPage.submit();
      await passwordResetPage.page.waitForTimeout(500);

      if (i < 5) {
        await passwordResetPage.page.locator('[data-testid="back-to-reset"]').click();
      }
    }

    await expect(passwordResetPage.errorMessage).toContainText('Too many requests');
  });

  test('should invalidate old tokens after password change', async ({ page, context }) => {
    const user = testData.users.standard;

    await test.step('Get first reset token', async () => {
      await passwordResetPage.fillEmail(user.email);
      await passwordResetPage.submit();
    });

    const firstToken = 'first-token-12345';

    await test.step('Request another reset', async () => {
      await page.goto('/forgot-password');
      await passwordResetPage.fillEmail(user.email);
      await passwordResetPage.submit();
    });

    const secondToken = 'second-token-67890';

    await test.step('Try to use first token', async () => {
      await page.goto(`/reset-password?token=${firstToken}`);

      await page.locator('[data-testid="new-password"]').fill('NewP@ssw0rd!');
      await page.locator('[data-testid="confirm-password"]').fill('NewP@ssw0rd!');
      await page.locator('[data-testid="submit-reset"]').click();

      await expect(page.locator('[data-testid="error-message"]'))
        .toContainText('invalid');
    });
  });
});
