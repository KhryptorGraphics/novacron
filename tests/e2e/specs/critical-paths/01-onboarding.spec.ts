import { test, expect } from '@playwright/test';
import { LoginPage } from '../../pages/LoginPage';
import { RegistrationPage } from '../../pages/RegistrationPage';
import { DashboardPage } from '../../pages/DashboardPage';
import { VMCreationWizard } from '../../pages/VMCreationWizard';
import { VMDetailsPage } from '../../pages/VMDetailsPage';
import { testData } from '../../fixtures/testData';

test.describe('Critical Path: User Onboarding to First VM', () => {
  let loginPage: LoginPage;
  let registrationPage: RegistrationPage;
  let dashboardPage: DashboardPage;
  let vmCreationWizard: VMCreationWizard;
  let vmDetailsPage: VMDetailsPage;

  test.beforeEach(async ({ page }) => {
    loginPage = new LoginPage(page);
    registrationPage = new RegistrationPage(page);
    dashboardPage = new DashboardPage(page);
    vmCreationWizard = new VMCreationWizard(page);
    vmDetailsPage = new VMDetailsPage(page);
  });

  test('should complete full onboarding journey from registration to first VM', {
    tag: '@smoke @critical',
  }, async ({ page }) => {
    // Step 1: Register new user
    await test.step('Register new user account', async () => {
      await loginPage.goto();
      await loginPage.clickRegisterLink();

      const newUser = testData.users.generate();
      await registrationPage.fillRegistrationForm(newUser);
      await registrationPage.acceptTerms();
      await registrationPage.submit();

      await expect(registrationPage.successMessage).toBeVisible();
      await expect(page).toHaveURL(/\/welcome/);
    });

    // Step 2: Complete welcome wizard
    await test.step('Complete welcome wizard', async () => {
      await page.locator('[data-testid="wizard-next"]').click();

      // Select organization type
      await page.locator('[data-testid="org-type-enterprise"]').click();
      await page.locator('[data-testid="wizard-next"]').click();

      // Set preferences
      await page.locator('[data-testid="pref-notifications-email"]').check();
      await page.locator('[data-testid="wizard-finish"]').click();

      await expect(page).toHaveURL(/\/dashboard/);
    });

    // Step 3: Navigate to VM creation
    await test.step('Navigate to VM creation', async () => {
      await dashboardPage.waitForLoad();
      await dashboardPage.clickCreateVM();

      await expect(vmCreationWizard.wizardTitle).toBeVisible();
      await expect(vmCreationWizard.wizardTitle).toContainText('Create Virtual Machine');
    });

    // Step 4: Configure VM basic settings
    await test.step('Configure VM basic settings', async () => {
      await vmCreationWizard.setVMName('my-first-vm');
      await vmCreationWizard.selectTemplate('ubuntu-22.04');
      await vmCreationWizard.nextStep();

      await expect(vmCreationWizard.currentStep).toContainText('Resources');
    });

    // Step 5: Configure resources
    await test.step('Configure VM resources', async () => {
      await vmCreationWizard.setCPU(2);
      await vmCreationWizard.setMemory(4096);
      await vmCreationWizard.setDisk(50);
      await vmCreationWizard.nextStep();

      await expect(vmCreationWizard.currentStep).toContainText('Network');
    });

    // Step 6: Configure networking
    await test.step('Configure networking', async () => {
      await vmCreationWizard.selectNetwork('default');
      await vmCreationWizard.enableDHCP();
      await vmCreationWizard.nextStep();

      await expect(vmCreationWizard.currentStep).toContainText('Review');
    });

    // Step 7: Review and create
    await test.step('Review and create VM', async () => {
      const summary = await vmCreationWizard.getConfigSummary();

      expect(summary).toMatchObject({
        name: 'my-first-vm',
        template: 'ubuntu-22.04',
        cpu: 2,
        memory: 4096,
        disk: 50,
        network: 'default'
      });

      await vmCreationWizard.submit();

      await expect(page.locator('[data-testid="creation-success"]')).toBeVisible();
    });

    // Step 8: Verify VM is created and starting
    await test.step('Verify VM creation and startup', async () => {
      await expect(page).toHaveURL(/\/vms\/[a-f0-9-]+/);

      await vmDetailsPage.waitForVMState('starting', { timeout: 10000 });
      await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

      const status = await vmDetailsPage.getVMStatus();
      expect(status).toBe('running');

      // Verify VM appears in dashboard
      await dashboardPage.goto();
      const vmList = await dashboardPage.getVMList();
      expect(vmList).toContainEqual(
        expect.objectContaining({
          name: 'my-first-vm',
          state: 'running'
        })
      );
    });

    // Step 9: Access VM console
    await test.step('Access VM console', async () => {
      await dashboardPage.clickVM('my-first-vm');
      await vmDetailsPage.openConsole();

      const consoleFrame = page.frameLocator('[data-testid="vm-console-frame"]');
      await expect(consoleFrame.locator('.terminal')).toBeVisible({ timeout: 15000 });

      // Wait for login prompt
      await expect(consoleFrame.locator('text=login:')).toBeVisible({ timeout: 30000 });
    });

    // Step 10: Show onboarding completion
    await test.step('Verify onboarding completion badge', async () => {
      await dashboardPage.goto();

      const completionBadge = page.locator('[data-testid="onboarding-complete"]');
      await expect(completionBadge).toBeVisible();

      const tutorialPrompt = page.locator('[data-testid="tutorial-next-steps"]');
      await expect(tutorialPrompt).toBeVisible();
    });
  });

  test('should handle onboarding interruption and resume', async ({ page }) => {
    await test.step('Start registration', async () => {
      await loginPage.goto();
      await loginPage.clickRegisterLink();

      const newUser = testData.users.generate();
      await registrationPage.fillRegistrationForm(newUser);
      await registrationPage.acceptTerms();
      await registrationPage.submit();
    });

    await test.step('Interrupt during welcome wizard', async () => {
      await page.locator('[data-testid="wizard-next"]').click();

      // Simulate browser close/refresh
      await page.reload();

      // Should resume at same step
      await expect(page.locator('[data-testid="wizard-step-2"]')).toBeVisible();
    });

    await test.step('Complete wizard after interruption', async () => {
      await page.locator('[data-testid="org-type-startup"]').click();
      await page.locator('[data-testid="wizard-next"]').click();
      await page.locator('[data-testid="wizard-finish"]').click();

      await expect(page).toHaveURL(/\/dashboard/);
    });
  });

  test('should provide helpful guidance for first-time users', async ({ page }) => {
    const newUser = testData.users.generate();

    await test.step('Register and login', async () => {
      await loginPage.goto();
      await loginPage.clickRegisterLink();
      await registrationPage.fillRegistrationForm(newUser);
      await registrationPage.acceptTerms();
      await registrationPage.submit();

      // Skip wizard
      await page.locator('[data-testid="wizard-skip"]').click();
    });

    await test.step('Verify tutorial tooltips', async () => {
      await dashboardPage.waitForLoad();

      // Check for tutorial overlay
      const tutorial = page.locator('[data-testid="tutorial-overlay"]');
      await expect(tutorial).toBeVisible();

      // Verify key features are highlighted
      await expect(page.locator('[data-testid="highlight-create-vm"]')).toBeVisible();
      await expect(page.locator('[data-testid="highlight-dashboard"]')).toBeVisible();
      await expect(page.locator('[data-testid="highlight-clusters"]')).toBeVisible();
    });

    await test.step('Complete interactive tutorial', async () => {
      await page.locator('[data-testid="tutorial-next"]').click();
      await page.locator('[data-testid="tutorial-next"]').click();
      await page.locator('[data-testid="tutorial-finish"]').click();

      await expect(page.locator('[data-testid="tutorial-overlay"]')).not.toBeVisible();
    });
  });

  test.afterEach(async ({ page }, testInfo) => {
    if (testInfo.status !== 'passed') {
      // Capture screenshot and logs on failure
      await page.screenshot({
        path: `tests/e2e/screenshots/onboarding-failure-${Date.now()}.png`,
        fullPage: true
      });
    }
  });
});
