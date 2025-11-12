import { test, expect } from '@playwright/test';
import { DashboardPage } from '../../pages/DashboardPage';
import { VMDetailsPage } from '../../pages/VMDetailsPage';
import { VMConsolePage } from '../../pages/VMConsolePage';
import { authenticateUser } from '../../fixtures/auth';
import { testData } from '../../fixtures/testData';

test.describe('VM Management: Console Access', () => {
  let dashboardPage: DashboardPage;
  let vmDetailsPage: VMDetailsPage;
  let vmConsolePage: VMConsolePage;
  let testVMName: string;

  test.beforeAll(async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();

    await authenticateUser(page, testData.users.admin);

    const dashboard = new DashboardPage(page);
    await dashboard.goto();
    await dashboard.clickQuickCreate();

    testVMName = 'console-test-vm';
    await page.locator('[data-testid="quick-create-preset-minimal"]').click();
    await page.locator('[data-testid="vm-name"]').fill(testVMName);
    await page.locator('[data-testid="quick-create-submit"]').click();

    const vmDetails = new VMDetailsPage(page);
    await vmDetails.waitForVMState('running', { timeout: 60000 });

    await context.close();
  });

  test.beforeEach(async ({ page }) => {
    await authenticateUser(page, testData.users.admin);

    dashboardPage = new DashboardPage(page);
    vmDetailsPage = new VMDetailsPage(page);
    vmConsolePage = new VMConsolePage(page);

    await dashboardPage.goto();
    await dashboardPage.clickVM(testVMName);
  });

  test.afterAll(async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();

    await authenticateUser(page, testData.users.admin);

    const dashboard = new DashboardPage(page);
    await dashboard.goto();

    const vm = await dashboard.findVM(testVMName);
    if (vm) {
      await dashboard.clickVM(testVMName);
      const vmDetails = new VMDetailsPage(page);
      await vmDetails.forceDelete();
    }

    await context.close();
  });

  test('should open console and display terminal', {
    tag: '@smoke',
  }, async ({ page }) => {
    await vmDetailsPage.openConsole();

    const consoleFrame = page.frameLocator('[data-testid="vm-console-frame"]');
    await expect(consoleFrame.locator('.terminal')).toBeVisible({ timeout: 15000 });

    // Wait for login prompt
    await expect(consoleFrame.locator('text=login:')).toBeVisible({ timeout: 30000 });
  });

  test('should support console keyboard input', async ({ page }) => {
    await vmDetailsPage.openConsole();

    const consoleFrame = page.frameLocator('[data-testid="vm-console-frame"]');
    const terminal = consoleFrame.locator('.terminal');

    await expect(terminal).toBeVisible({ timeout: 15000 });
    await terminal.click();

    await page.keyboard.type('root');
    await page.keyboard.press('Enter');

    // Should see password prompt
    await expect(consoleFrame.locator('text=Password:')).toBeVisible({ timeout: 5000 });
  });

  test('should support console copy and paste', async ({ page }) => {
    await vmDetailsPage.openConsole();

    const consoleFrame = page.frameLocator('[data-testid="vm-console-frame"]');
    await expect(consoleFrame.locator('.terminal')).toBeVisible({ timeout: 15000 });

    await vmConsolePage.openClipboard();

    const testText = 'echo "Hello from clipboard"';
    await page.locator('[data-testid="clipboard-input"]').fill(testText);
    await page.locator('[data-testid="paste-to-console"]').click();

    // Text should appear in console
    await expect(consoleFrame.locator(`text=${testText}`)).toBeVisible();
  });

  test('should support fullscreen mode', async ({ page }) => {
    await vmDetailsPage.openConsole();

    await vmConsolePage.toggleFullscreen();

    await expect(page.locator('[data-testid="console-fullscreen"]')).toHaveClass(/fullscreen/);

    await vmConsolePage.toggleFullscreen();

    await expect(page.locator('[data-testid="console-fullscreen"]')).not.toHaveClass(/fullscreen/);
  });

  test('should support VNC console', async ({ page }) => {
    await vmDetailsPage.openConsole();

    await vmConsolePage.selectConsoleType('vnc');

    const vncCanvas = page.locator('[data-testid="vnc-canvas"]');
    await expect(vncCanvas).toBeVisible({ timeout: 15000 });

    // Verify VNC is interactive
    await vncCanvas.click({ position: { x: 100, y: 100 } });
  });

  test('should support SPICE console', async ({ page }) => {
    await vmDetailsPage.openConsole();

    await vmConsolePage.selectConsoleType('spice');

    await expect(page.locator('[data-testid="spice-viewer"]')).toBeVisible({ timeout: 15000 });
  });

  test('should send Ctrl+Alt+Del', async ({ page }) => {
    await vmDetailsPage.openConsole();

    const consoleFrame = page.frameLocator('[data-testid="vm-console-frame"]');
    await expect(consoleFrame.locator('.terminal')).toBeVisible({ timeout: 15000 });

    await vmConsolePage.sendCtrlAltDel();

    // Should trigger reboot or login screen
    await expect(consoleFrame.locator('text=reboot')).toBeVisible({ timeout: 10000 });
  });

  test('should support console screenshot', async ({ page }) => {
    await vmDetailsPage.openConsole();

    const consoleFrame = page.frameLocator('[data-testid="vm-console-frame"]');
    await expect(consoleFrame.locator('.terminal')).toBeVisible({ timeout: 15000 });

    const downloadPromise = page.waitForEvent('download');
    await vmConsolePage.takeScreenshot();

    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/screenshot.*\.png/);
  });

  test('should handle console disconnection', async ({ page }) => {
    await vmDetailsPage.openConsole();

    const consoleFrame = page.frameLocator('[data-testid="vm-console-frame"]');
    await expect(consoleFrame.locator('.terminal')).toBeVisible({ timeout: 15000 });

    // Simulate network disconnect
    await page.route('**/api/console/**', route => route.abort());

    await expect(page.locator('[data-testid="console-disconnected"]')).toBeVisible({ timeout: 10000 });

    // Should show reconnect button
    await expect(page.locator('[data-testid="console-reconnect"]')).toBeVisible();
  });

  test('should support console resize', async ({ page }) => {
    await vmDetailsPage.openConsole();

    const consoleFrame = page.frameLocator('[data-testid="vm-console-frame"]');
    await expect(consoleFrame.locator('.terminal')).toBeVisible({ timeout: 15000 });

    // Change terminal size
    await page.locator('[data-testid="console-settings"]').click();
    await page.locator('[data-testid="terminal-rows"]').fill('40');
    await page.locator('[data-testid="terminal-cols"]').fill('120');
    await page.locator('[data-testid="apply-settings"]').click();

    // Terminal should resize
    const terminal = consoleFrame.locator('.terminal');
    const size = await terminal.evaluate((el: any) => ({
      rows: el.rows,
      cols: el.cols
    }));

    expect(size.rows).toBe(40);
    expect(size.cols).toBe(120);
  });

  test('should support console text selection', async ({ page }) => {
    await vmDetailsPage.openConsole();

    const consoleFrame = page.frameLocator('[data-testid="vm-console-frame"]');
    const terminal = consoleFrame.locator('.terminal');

    await expect(terminal).toBeVisible({ timeout: 15000 });

    // Wait for some text to appear
    await expect(consoleFrame.locator('text=login:')).toBeVisible({ timeout: 30000 });

    // Select text
    await terminal.click({ position: { x: 10, y: 10 } });
    await page.mouse.down();
    await page.mouse.move(100, 10);
    await page.mouse.up();

    // Copy should be enabled
    const copyButton = page.locator('[data-testid="console-copy"]');
    await expect(copyButton).toBeEnabled();
  });

  test('should support multiple console sessions', async ({ browser }) => {
    const context1 = await browser.newContext();
    const page1 = await context1.newPage();

    const context2 = await browser.newContext();
    const page2 = await context2.newPage();

    try {
      await test.step('Open first console session', async () => {
        await authenticateUser(page1, testData.users.admin);

        const dashboard1 = new DashboardPage(page1);
        await dashboard1.goto();
        await dashboard1.clickVM(testVMName);

        const vmDetails1 = new VMDetailsPage(page1);
        await vmDetails1.openConsole();

        const consoleFrame1 = page1.frameLocator('[data-testid="vm-console-frame"]');
        await expect(consoleFrame1.locator('.terminal')).toBeVisible({ timeout: 15000 });
      });

      await test.step('Open second console session', async () => {
        await authenticateUser(page2, testData.users.standard);

        const dashboard2 = new DashboardPage(page2);
        await dashboard2.goto();
        await dashboard2.clickVM(testVMName);

        const vmDetails2 = new VMDetailsPage(page2);
        await vmDetails2.openConsole();

        const consoleFrame2 = page2.frameLocator('[data-testid="vm-console-frame"]');
        await expect(consoleFrame2.locator('.terminal')).toBeVisible({ timeout: 15000 });
      });

      await test.step('Verify concurrent access', async () => {
        // Both consoles should be active
        const consoleFrame1 = page1.frameLocator('[data-testid="vm-console-frame"]');
        const consoleFrame2 = page2.frameLocator('[data-testid="vm-console-frame"]');

        await expect(consoleFrame1.locator('.terminal')).toBeVisible();
        await expect(consoleFrame2.locator('.terminal')).toBeVisible();

        // Type in one console
        await consoleFrame1.locator('.terminal').click();
        await page1.keyboard.type('test');

        // Should appear in both
        await expect(consoleFrame2.locator('text=test')).toBeVisible({ timeout: 5000 });
      });
    } finally {
      await context1.close();
      await context2.close();
    }
  });

  test('should support console logging', async ({ page }) => {
    await vmDetailsPage.openConsole();

    const consoleFrame = page.frameLocator('[data-testid="vm-console-frame"]');
    await expect(consoleFrame.locator('.terminal')).toBeVisible({ timeout: 15000 });

    await vmConsolePage.enableLogging();

    // Interact with console
    await consoleFrame.locator('.terminal').click();
    await page.keyboard.type('echo "test logging"');
    await page.keyboard.press('Enter');

    // Stop logging and download
    const downloadPromise = page.waitForEvent('download');
    await vmConsolePage.stopLoggingAndDownload();

    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/console-log.*\.txt/);
  });

  test('should handle console permissions', async ({ page }) => {
    // Logout and login as read-only user
    await dashboardPage.clickUserMenu();
    await dashboardPage.clickLogout();

    await authenticateUser(page, testData.users.readonly);

    await dashboardPage.goto();
    await dashboardPage.clickVM(testVMName);

    await vmDetailsPage.openConsole();

    // Console should be in read-only mode
    const consoleFrame = page.frameLocator('[data-testid="vm-console-frame"]');
    await expect(consoleFrame.locator('.terminal')).toBeVisible({ timeout: 15000 });

    await expect(page.locator('[data-testid="console-readonly-notice"]')).toBeVisible();

    // Keyboard input should be disabled
    await consoleFrame.locator('.terminal').click();
    await page.keyboard.type('test');

    // Text should not appear (read-only)
    await expect(consoleFrame.locator('text=test')).not.toBeVisible({ timeout: 2000 });
  });
});
