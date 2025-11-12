import { test, expect } from '@playwright/test';
import { LoginPage } from '../../pages/LoginPage';
import { DashboardPage } from '../../pages/DashboardPage';
import { SettingsPage } from '../../pages/SettingsPage';
import { testData } from '../../fixtures/testData';

test.describe('Authentication: Session Management', () => {
  let loginPage: LoginPage;
  let dashboardPage: DashboardPage;
  let settingsPage: SettingsPage;

  test.beforeEach(async ({ page }) => {
    loginPage = new LoginPage(page);
    dashboardPage = new DashboardPage(page);
    settingsPage = new SettingsPage(page);
  });

  test('should handle session timeout', {
    tag: '@smoke',
  }, async ({ page }) => {
    const user = testData.users.standard;

    await test.step('Login successfully', async () => {
      await loginPage.goto();
      await loginPage.login(user.email, user.password);
      await expect(page).toHaveURL(/\/dashboard/);
    });

    await test.step('Simulate session expiry', async () => {
      // Set session timeout to 30 seconds for test
      await settingsPage.goto();
      await settingsPage.navigateToTab('security');
      await page.locator('[data-testid="session-timeout"]').fill('30');
      await page.locator('[data-testid="save-settings"]').click();

      await dashboardPage.goto();
    });

    await test.step('Verify timeout warning', async () => {
      await page.waitForTimeout(25000);

      const warningDialog = page.locator('[data-testid="session-timeout-warning"]');
      await expect(warningDialog).toBeVisible();
      await expect(warningDialog).toContainText('expire soon');
    });

    await test.step('Extend session', async () => {
      await page.locator('[data-testid="extend-session"]').click();
      await expect(page.locator('[data-testid="session-timeout-warning"]')).not.toBeVisible();

      // Should remain on dashboard
      await expect(page).toHaveURL(/\/dashboard/);
    });
  });

  test('should handle concurrent sessions', async ({ browser }) => {
    const user = testData.users.standard;

    const context1 = await browser.newContext();
    const page1 = await context1.newPage();
    const login1 = new LoginPage(page1);

    const context2 = await browser.newContext();
    const page2 = await context2.newPage();
    const login2 = new LoginPage(page2);

    await test.step('Login from first browser', async () => {
      await login1.goto();
      await login1.login(user.email, user.password);
      await expect(page1).toHaveURL(/\/dashboard/);
    });

    await test.step('Login from second browser', async () => {
      await login2.goto();
      await login2.login(user.email, user.password);
      await expect(page2).toHaveURL(/\/dashboard/);
    });

    await test.step('Verify both sessions active', async () => {
      const dashboard1 = new DashboardPage(page1);
      const dashboard2 = new DashboardPage(page2);

      await dashboard1.goto();
      await expect(dashboard1.userMenu).toBeVisible();

      await dashboard2.goto();
      await expect(dashboard2.userMenu).toBeVisible();
    });

    await test.step('Check active sessions in settings', async () => {
      const settings1 = new SettingsPage(page1);
      await settings1.goto();
      await settings1.navigateToTab('sessions');

      const sessions = await page1.locator('[data-testid="active-session"]').all();
      expect(sessions.length).toBeGreaterThanOrEqual(2);
    });

    await context1.close();
    await context2.close();
  });

  test('should allow session termination', async ({ browser }) => {
    const user = testData.users.standard;

    const context1 = await browser.newContext();
    const page1 = await context1.newPage();

    const context2 = await browser.newContext();
    const page2 = await context2.newPage();

    await test.step('Create two sessions', async () => {
      await new LoginPage(page1).goto();
      await new LoginPage(page1).login(user.email, user.password);

      await new LoginPage(page2).goto();
      await new LoginPage(page2).login(user.email, user.password);
    });

    await test.step('Terminate other session', async () => {
      const settings = new SettingsPage(page1);
      await settings.goto();
      await settings.navigateToTab('sessions');

      const sessions = await page1.locator('[data-testid="active-session"]').all();
      expect(sessions.length).toBeGreaterThanOrEqual(2);

      // Terminate second session
      await page1.locator('[data-testid="session-terminate"]:nth-child(2)').click();
      await page1.locator('[data-testid="confirm-terminate"]').click();

      await expect(page1.locator('[data-testid="session-terminated"]')).toBeVisible();
    });

    await test.step('Verify session terminated', async () => {
      await page2.reload();
      await expect(page2).toHaveURL(/\/login/);
    });

    await context1.close();
    await context2.close();
  });

  test('should enforce maximum concurrent sessions', async ({ browser }) => {
    const user = testData.users.withSessionLimit;
    const maxSessions = 3;
    const contexts: any[] = [];

    try {
      await test.step('Configure session limit', async () => {
        const tempContext = await browser.newContext();
        const tempPage = await tempContext.newPage();

        await new LoginPage(tempPage).goto();
        await new LoginPage(tempPage).login(user.email, user.password);

        const settings = new SettingsPage(tempPage);
        await settings.goto();
        await settings.navigateToTab('security');
        await tempPage.locator('[data-testid="max-sessions"]').fill(String(maxSessions));
        await tempPage.locator('[data-testid="save-settings"]').click();

        await tempContext.close();
      });

      await test.step('Create maximum sessions', async () => {
        for (let i = 0; i < maxSessions; i++) {
          const context = await browser.newContext();
          const page = await context.newPage();

          await new LoginPage(page).goto();
          await new LoginPage(page).login(user.email, user.password);
          await expect(page).toHaveURL(/\/dashboard/);

          contexts.push(context);
        }
      });

      await test.step('Attempt to exceed limit', async () => {
        const context = await browser.newContext();
        const page = await context.newPage();

        await new LoginPage(page).goto();
        await new LoginPage(page).login(user.email, user.password);

        // Should show error or force logout oldest session
        const error = page.locator('[data-testid="session-limit-error"]');
        const forceLogout = page.locator('[data-testid="force-logout-prompt"]');

        const errorVisible = await error.isVisible();
        const forceLogoutVisible = await forceLogout.isVisible();

        expect(errorVisible || forceLogoutVisible).toBe(true);

        await context.close();
      });
    } finally {
      for (const context of contexts) {
        await context.close();
      }
    }
  });

  test('should maintain session across page refreshes', async ({ page }) => {
    const user = testData.users.standard;

    await loginPage.goto();
    await loginPage.login(user.email, user.password);
    await expect(page).toHaveURL(/\/dashboard/);

    await page.reload();
    await expect(page).toHaveURL(/\/dashboard/);
    await expect(dashboardPage.userMenu).toBeVisible();

    await page.goto('/vms');
    await page.reload();
    await expect(page).toHaveURL(/\/vms/);
  });

  test('should handle logout correctly', async ({ page }) => {
    const user = testData.users.standard;

    await loginPage.goto();
    await loginPage.login(user.email, user.password);
    await expect(page).toHaveURL(/\/dashboard/);

    await dashboardPage.clickUserMenu();
    await dashboardPage.clickLogout();

    await expect(page).toHaveURL(/\/login/);

    // Try to access protected page
    await page.goto('/dashboard');
    await expect(page).toHaveURL(/\/login/);
  });

  test('should revoke all sessions on password change', async ({ browser }) => {
    const user = testData.users.standard;

    const context1 = await browser.newContext();
    const page1 = await context1.newPage();

    const context2 = await browser.newContext();
    const page2 = await context2.newPage();

    await test.step('Create multiple sessions', async () => {
      await new LoginPage(page1).goto();
      await new LoginPage(page1).login(user.email, user.password);

      await new LoginPage(page2).goto();
      await new LoginPage(page2).login(user.email, user.password);
    });

    await test.step('Change password', async () => {
      const settings = new SettingsPage(page1);
      await settings.goto();
      await settings.navigateToTab('security');

      await page1.locator('[data-testid="current-password"]').fill(user.password);
      await page1.locator('[data-testid="new-password"]').fill('NewSecureP@ss123!');
      await page1.locator('[data-testid="confirm-password"]').fill('NewSecureP@ss123!');
      await page1.locator('[data-testid="save-password"]').click();

      await expect(page1.locator('[data-testid="password-changed"]')).toBeVisible();
    });

    await test.step('Verify other sessions invalidated', async () => {
      await page2.reload();
      await expect(page2).toHaveURL(/\/login/);
    });

    await context1.close();
    await context2.close();
  });
});
