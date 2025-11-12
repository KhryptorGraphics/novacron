import { test, expect } from '@playwright/test';
import {
  navigateAndWait,
  fillInput,
  clickWithRetry,
  waitForNetworkIdle,
  takeScreenshot,
} from './utils/playwright-helpers';
import {
  waitForText,
  assertUrlContains,
  assertPerformance,
} from './utils/test-helpers';
import { generateUser } from './utils/data-generators';

/**
 * Example Playwright Test Suite for NovaCron
 *
 * This file demonstrates:
 * - Test organization with describe blocks
 * - Using helper utilities
 * - Test data generation
 * - Custom assertions
 * - Performance testing
 * - Screenshots
 */

test.describe('NovaCron Example Tests', () => {
  test.describe('Homepage', () => {
    test('should load homepage successfully', async ({ page }) => {
      // Navigate to homepage with performance tracking
      await assertPerformance(
        async () => {
          await navigateAndWait(page, '/');
          await waitForNetworkIdle(page);
        },
        3000,
        'Homepage load'
      );

      // Verify page title
      await expect(page).toHaveTitle(/NovaCron/);

      // Take screenshot
      await takeScreenshot(page, 'homepage');
    });

    test('should display navigation menu', async ({ page }) => {
      await page.goto('/');

      // Check navigation elements
      const nav = page.locator('[data-testid="main-nav"]');
      await expect(nav).toBeVisible();

      // Verify menu items
      const menuItems = ['Home', 'VMs', 'Projects', 'Settings'];
      for (const item of menuItems) {
        await expect(page.locator(`nav >> text=${item}`)).toBeVisible();
      }
    });
  });

  test.describe('User Authentication', () => {
    test('should navigate to login page', async ({ page }) => {
      await page.goto('/');

      // Click login button
      await clickWithRetry(page.locator('[data-testid="login-link"]'));

      // Verify URL
      await assertUrlContains(page, '/login');

      // Verify login form is visible
      await expect(page.locator('[data-testid="login-form"]')).toBeVisible();
    });

    test('should show error for invalid credentials', async ({ page }) => {
      await page.goto('/login');

      // Fill invalid credentials
      await fillInput(page.locator('[data-testid="username-input"]'), 'invalid@example.com');
      await fillInput(page.locator('[data-testid="password-input"]'), 'wrongpassword');

      // Submit form
      await clickWithRetry(page.locator('[data-testid="login-button"]'));

      // Verify error message
      const errorMessage = page.locator('[data-testid="error-message"]');
      await waitForText(errorMessage, /Invalid credentials/);
    });

    test.skip('should login with valid credentials', async ({ page }) => {
      // This test requires actual authentication setup
      await page.goto('/login');

      const user = generateUser();

      await fillInput(page.locator('[data-testid="username-input"]'), user.email);
      await fillInput(page.locator('[data-testid="password-input"]'), user.password);

      await clickWithRetry(page.locator('[data-testid="login-button"]'));

      // Should redirect to dashboard
      await assertUrlContains(page, '/dashboard');
    });
  });

  test.describe('VM Management', () => {
    // Use authenticated state for these tests
    test.use({ storageState: 'tests/e2e/fixtures/auth-state.json' });

    test('should display VM list page', async ({ page }) => {
      await page.goto('/vms');

      // Verify page title
      await expect(page.locator('h1')).toContainText('Virtual Machines');

      // Verify table or list exists
      const vmList = page.locator('[data-testid="vm-list"]');
      await expect(vmList).toBeVisible();
    });

    test.skip('should create new VM', async ({ page }) => {
      // This is a placeholder - requires actual VM creation flow
      await page.goto('/vms/new');

      const vmData = {
        name: 'test-vm-example',
        cpu: '4',
        memory: '8',
      };

      // Fill VM form
      await fillInput(page.locator('[data-testid="vm-name"]'), vmData.name);
      await page.selectOption('[data-testid="vm-cpu"]', vmData.cpu);
      await page.selectOption('[data-testid="vm-memory"]', vmData.memory);

      // Submit
      await clickWithRetry(page.locator('[data-testid="create-vm-button"]'));

      // Verify success
      await waitForText(
        page.locator('[data-testid="success-message"]'),
        /VM created successfully/
      );
    });
  });

  test.describe('Responsive Design', () => {
    test('should be mobile responsive', async ({ page }) => {
      // Set mobile viewport
      await page.setViewportSize({ width: 375, height: 667 });

      await page.goto('/');

      // Verify mobile menu
      const mobileMenu = page.locator('[data-testid="mobile-menu"]');
      await expect(mobileMenu).toBeVisible();

      // Take mobile screenshot
      await takeScreenshot(page, 'mobile-homepage', { fullPage: true });
    });

    test('should be tablet responsive', async ({ page }) => {
      // Set tablet viewport
      await page.setViewportSize({ width: 768, height: 1024 });

      await page.goto('/');

      // Take tablet screenshot
      await takeScreenshot(page, 'tablet-homepage', { fullPage: true });
    });
  });

  test.describe('Accessibility', () => {
    test('should have proper heading hierarchy', async ({ page }) => {
      await page.goto('/');

      // Check for h1
      const h1 = page.locator('h1');
      await expect(h1).toBeVisible();
      expect(await h1.count()).toBeGreaterThan(0);
    });

    test('should have alt text for images', async ({ page }) => {
      await page.goto('/');

      // Get all images
      const images = page.locator('img');
      const count = await images.count();

      // Check each image has alt text
      for (let i = 0; i < count; i++) {
        const img = images.nth(i);
        const alt = await img.getAttribute('alt');
        expect(alt).toBeTruthy();
      }
    });

    test('should be keyboard navigable', async ({ page }) => {
      await page.goto('/');

      // Tab through elements
      await page.keyboard.press('Tab');

      // Verify focus is visible
      const focused = page.locator(':focus');
      await expect(focused).toBeVisible();
    });
  });
});
