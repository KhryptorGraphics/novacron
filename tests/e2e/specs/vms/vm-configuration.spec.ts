import { test, expect } from '@playwright/test';
import { DashboardPage } from '../../pages/DashboardPage';
import { VMDetailsPage } from '../../pages/VMDetailsPage';
import { VMConfigurationPage } from '../../pages/VMConfigurationPage';
import { authenticateUser } from '../../fixtures/auth';
import { testData } from '../../fixtures/testData';

test.describe('VM Management: Configuration', () => {
  let dashboardPage: DashboardPage;
  let vmDetailsPage: VMDetailsPage;
  let vmConfigPage: VMConfigurationPage;
  let testVMId: string;

  test.beforeEach(async ({ page }) => {
    await authenticateUser(page, testData.users.admin);

    dashboardPage = new DashboardPage(page);
    vmDetailsPage = new VMDetailsPage(page);
    vmConfigPage = new VMConfigurationPage(page);

    // Create test VM
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    await page.locator('[data-testid="quick-create-preset-minimal"]').click();
    await page.locator('[data-testid="vm-name"]').fill(`config-test-${Date.now()}`);
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    testVMId = page.url().match(/\/vms\/([a-f0-9-]+)/)?.[1] || '';
  });

  test.afterEach(async () => {
    if (testVMId) {
      await vmDetailsPage.forceDelete();
    }
  });

  test('should update CPU configuration', {
    tag: '@smoke',
  }, async ({ page }) => {
    // Stop VM for configuration change
    await vmDetailsPage.clickAction('stop');
    await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });

    await vmDetailsPage.navigateToTab('configuration');
    await vmConfigPage.editCPU();

    await page.locator('[data-testid="cpu-count"]').fill('4');
    await page.locator('[data-testid="cpu-sockets"]').fill('2');
    await page.locator('[data-testid="cpu-cores"]').fill('2');
    await page.locator('[data-testid="save-cpu"]').click();

    await expect(page.locator('[data-testid="config-saved"]')).toBeVisible();

    // Verify changes
    const cpuInfo = await vmConfigPage.getCPUInfo();
    expect(cpuInfo.count).toBe(4);
    expect(cpuInfo.sockets).toBe(2);
    expect(cpuInfo.cores).toBe(2);
  });

  test('should update memory configuration', async ({ page }) => {
    await vmDetailsPage.clickAction('stop');
    await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });

    await vmDetailsPage.navigateToTab('configuration');
    await vmConfigPage.editMemory();

    await page.locator('[data-testid="memory-size"]').fill('8192');
    await page.locator('[data-testid="save-memory"]').click();

    await expect(page.locator('[data-testid="config-saved"]')).toBeVisible();

    const memoryInfo = await vmConfigPage.getMemoryInfo();
    expect(memoryInfo.size).toBe(8192);
  });

  test('should enable memory hotplug', async ({ page }) => {
    await vmDetailsPage.clickAction('stop');
    await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });

    await vmDetailsPage.navigateToTab('configuration');
    await vmConfigPage.editMemory();

    await page.locator('[data-testid="enable-hotplug"]').check();
    await page.locator('[data-testid="max-memory"]').fill('16384');
    await page.locator('[data-testid="save-memory"]').click();

    await expect(page.locator('[data-testid="config-saved"]')).toBeVisible();

    // Start VM and test hotplug
    await vmDetailsPage.navigateToTab('overview');
    await vmDetailsPage.clickAction('start');
    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    await vmDetailsPage.navigateToTab('configuration');
    await vmConfigPage.editMemory();

    // Should allow memory increase while running
    await page.locator('[data-testid="memory-size"]').fill('12288');
    await page.locator('[data-testid="save-memory"]').click();

    await expect(page.locator('[data-testid="config-saved"]')).toBeVisible();
  });

  test('should add and remove disks', async ({ page }) => {
    await vmDetailsPage.clickAction('stop');
    await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });

    await vmDetailsPage.navigateToTab('storage');

    // Add new disk
    await page.locator('[data-testid="add-disk"]').click();

    await page.locator('[data-testid="disk-size"]').fill('100');
    await page.locator('[data-testid="disk-type"]').selectOption('ssd');
    await page.locator('[data-testid="disk-bus"]').selectOption('virtio');
    await page.locator('[data-testid="save-disk"]').click();

    await expect(page.locator('[data-testid="disk-added"]')).toBeVisible();

    const disks = await vmDetailsPage.getDiskList();
    expect(disks).toHaveLength(2);

    // Remove disk
    await page.locator('[data-testid="disk-actions-1"]').click();
    await page.locator('[data-testid="remove-disk"]').click();
    await page.locator('[data-testid="confirm-remove"]').click();

    await expect(page.locator('[data-testid="disk-removed"]')).toBeVisible();

    const updatedDisks = await vmDetailsPage.getDiskList();
    expect(updatedDisks).toHaveLength(1);
  });

  test('should resize disk', async ({ page }) => {
    await vmDetailsPage.clickAction('stop');
    await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });

    await vmDetailsPage.navigateToTab('storage');

    const originalSize = await vmDetailsPage.getDiskSize(0);

    await page.locator('[data-testid="disk-actions-0"]').click();
    await page.locator('[data-testid="resize-disk"]').click();

    const newSize = originalSize + 50;
    await page.locator('[data-testid="new-disk-size"]').fill(String(newSize));
    await page.locator('[data-testid="confirm-resize"]').click();

    await expect(page.locator('[data-testid="disk-resized"]')).toBeVisible();

    const updatedSize = await vmDetailsPage.getDiskSize(0);
    expect(updatedSize).toBe(newSize);
  });

  test('should configure boot order', async ({ page }) => {
    await vmDetailsPage.clickAction('stop');
    await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });

    await vmDetailsPage.navigateToTab('configuration');
    await vmConfigPage.editBootOrder();

    // Reorder boot devices
    await page.locator('[data-testid="boot-device-cdrom"]').dragTo(
      page.locator('[data-testid="boot-position-0"]')
    );

    await page.locator('[data-testid="save-boot-order"]').click();

    await expect(page.locator('[data-testid="config-saved"]')).toBeVisible();

    const bootOrder = await vmConfigPage.getBootOrder();
    expect(bootOrder[0]).toBe('cdrom');
  });

  test('should update network configuration', async ({ page }) => {
    await vmDetailsPage.navigateToTab('network');

    // Edit existing interface
    await page.locator('[data-testid="nic-actions-0"]').click();
    await page.locator('[data-testid="edit-nic"]').click();

    await page.locator('[data-testid="nic-network"]').selectOption('management');
    await page.locator('[data-testid="nic-type"]').selectOption('virtio');
    await page.locator('[data-testid="save-nic"]').click();

    await expect(page.locator('[data-testid="nic-updated"]')).toBeVisible();

    // Add new interface
    await page.locator('[data-testid="add-nic"]').click();

    await page.locator('[data-testid="nic-network"]').selectOption('storage');
    await page.locator('[data-testid="nic-type"]').selectOption('e1000');
    await page.locator('[data-testid="save-nic"]').click();

    await expect(page.locator('[data-testid="nic-added"]')).toBeVisible();

    const nics = await vmDetailsPage.getNetworkInterfaces();
    expect(nics).toHaveLength(2);
  });

  test('should configure VM metadata', async ({ page }) => {
    await vmDetailsPage.navigateToTab('configuration');

    await page.locator('[data-testid="edit-metadata"]').click();

    await page.locator('[data-testid="vm-description"]').fill('Test VM for configuration');
    await page.locator('[data-testid="add-tag"]').click();
    await page.locator('[data-testid="tag-input"]').fill('test');
    await page.locator('[data-testid="add-tag-button"]').click();

    await page.locator('[data-testid="add-label"]').click();
    await page.locator('[data-testid="label-key"]').fill('environment');
    await page.locator('[data-testid="label-value"]').fill('testing');
    await page.locator('[data-testid="add-label-button"]').click();

    await page.locator('[data-testid="save-metadata"]').click();

    await expect(page.locator('[data-testid="metadata-saved"]')).toBeVisible();

    // Verify metadata
    const metadata = await vmConfigPage.getMetadata();
    expect(metadata.description).toBe('Test VM for configuration');
    expect(metadata.tags).toContain('test');
    expect(metadata.labels).toHaveProperty('environment', 'testing');
  });

  test('should configure VM options', async ({ page }) => {
    await vmDetailsPage.clickAction('stop');
    await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });

    await vmDetailsPage.navigateToTab('configuration');
    await vmConfigPage.editOptions();

    await page.locator('[data-testid="enable-acpi"]').check();
    await page.locator('[data-testid="enable-apic"]').check();
    await page.locator('[data-testid="enable-kvm"]').check();

    await page.locator('[data-testid="machine-type"]').selectOption('q35');
    await page.locator('[data-testid="bios-type"]').selectOption('uefi');

    await page.locator('[data-testid="save-options"]').click();

    await expect(page.locator('[data-testid="config-saved"]')).toBeVisible();

    const options = await vmConfigPage.getOptions();
    expect(options.acpi).toBe(true);
    expect(options.apic).toBe(true);
    expect(options.kvm).toBe(true);
    expect(options.machineType).toBe('q35');
    expect(options.biosType).toBe('uefi');
  });

  test('should prevent invalid configurations', async ({ page }) => {
    await vmDetailsPage.clickAction('stop');
    await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });

    await vmDetailsPage.navigateToTab('configuration');
    await vmConfigPage.editMemory();

    // Try to set memory below minimum
    await page.locator('[data-testid="memory-size"]').fill('128');
    await page.locator('[data-testid="save-memory"]').click();

    await expect(page.locator('[data-testid="validation-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="validation-error"]')).toContainText('minimum');
  });

  test('should export and import VM configuration', async ({ page }) => {
    await vmDetailsPage.navigateToTab('configuration');

    // Export configuration
    await page.locator('[data-testid="export-config"]').click();

    const downloadPromise = page.waitForEvent('download');
    await page.locator('[data-testid="confirm-export"]').click();

    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/config.*\.json/);

    // Import configuration (on another VM)
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();
    await page.locator('[data-testid="quick-create-preset-minimal"]').click();
    await page.locator('[data-testid="vm-name"]').fill(`import-test-${Date.now()}`);
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
    await vmDetailsPage.clickAction('stop');
    await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });

    await vmDetailsPage.navigateToTab('configuration');
    await page.locator('[data-testid="import-config"]').click();

    const fileInput = page.locator('[data-testid="config-file-input"]');
    await fileInput.setInputFiles(await download.path());

    await page.locator('[data-testid="confirm-import"]').click();

    await expect(page.locator('[data-testid="config-imported"]')).toBeVisible();

    // Cleanup
    await vmDetailsPage.forceDelete();
  });
});
