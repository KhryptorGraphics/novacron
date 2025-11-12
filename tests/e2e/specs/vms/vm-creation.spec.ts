import { test, expect } from '@playwright/test';
import { DashboardPage } from '../../pages/DashboardPage';
import { VMCreationWizard } from '../../pages/VMCreationWizard';
import { VMDetailsPage } from '../../pages/VMDetailsPage';
import { authenticateUser } from '../../fixtures/auth';
import { testData } from '../../fixtures/testData';

test.describe('VM Management: Creation', () => {
  let dashboardPage: DashboardPage;
  let vmCreationWizard: VMCreationWizard;
  let vmDetailsPage: VMDetailsPage;

  test.beforeEach(async ({ page }) => {
    await authenticateUser(page, testData.users.admin);

    dashboardPage = new DashboardPage(page);
    vmCreationWizard = new VMCreationWizard(page);
    vmDetailsPage = new VMDetailsPage(page);

    await dashboardPage.goto();
  });

  test.afterEach(async ({ page }, testInfo) => {
    if (testInfo.status === 'passed') {
      // Cleanup created VMs
      const vmName = testInfo.title.match(/\[([^\]]+)\]/)?.[1];
      if (vmName) {
        await dashboardPage.goto();
        const vm = await dashboardPage.findVM(vmName);
        if (vm) {
          await dashboardPage.clickVM(vmName);
          await vmDetailsPage.forceDelete();
        }
      }
    }
  });

  test('should create VM with basic configuration [basic-vm]', {
    tag: '@smoke',
  }, async ({ page }) => {
    await dashboardPage.clickCreateVM();

    await vmCreationWizard.setVMName('basic-vm');
    await vmCreationWizard.selectTemplate('ubuntu-22.04');
    await vmCreationWizard.nextStep();

    await vmCreationWizard.setCPU(2);
    await vmCreationWizard.setMemory(4096);
    await vmCreationWizard.setDisk(50);
    await vmCreationWizard.nextStep();

    await vmCreationWizard.selectNetwork('default');
    await vmCreationWizard.nextStep();

    await vmCreationWizard.submit();

    await expect(page).toHaveURL(/\/vms\/[a-f0-9-]+/);
    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
  });

  test('should create VM with custom resources [custom-resources-vm]', async ({ page }) => {
    await dashboardPage.clickCreateVM();

    await vmCreationWizard.setVMName('custom-resources-vm');
    await vmCreationWizard.selectTemplate('debian-12');
    await vmCreationWizard.nextStep();

    await vmCreationWizard.setCPU(8);
    await vmCreationWizard.setMemory(16384);
    await vmCreationWizard.setDisk(200);

    // Enable CPU pinning
    await vmCreationWizard.enableAdvancedOptions();
    await vmCreationWizard.enableCPUPinning();
    await vmCreationWizard.selectCPUCores([0, 1, 2, 3, 4, 5, 6, 7]);

    // Set NUMA node
    await vmCreationWizard.selectNUMANode(0);

    await vmCreationWizard.nextStep();
    await vmCreationWizard.selectNetwork('default');
    await vmCreationWizard.nextStep();
    await vmCreationWizard.submit();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    // Verify resources
    const resources = await vmDetailsPage.getResources();
    expect(resources.cpu).toBe(8);
    expect(resources.memory).toBe(16384);
    expect(resources.disk).toBe(200);
  });

  test('should create VM with multiple disks [multi-disk-vm]', async ({ page }) => {
    await dashboardPage.clickCreateVM();

    await vmCreationWizard.setVMName('multi-disk-vm');
    await vmCreationWizard.selectTemplate('ubuntu-22.04');
    await vmCreationWizard.nextStep();

    await vmCreationWizard.setCPU(4);
    await vmCreationWizard.setMemory(8192);
    await vmCreationWizard.setDisk(100);

    // Add additional disks
    await vmCreationWizard.addDisk({
      size: 200,
      type: 'ssd',
      bus: 'virtio',
      cache: 'writeback'
    });

    await vmCreationWizard.addDisk({
      size: 500,
      type: 'hdd',
      bus: 'scsi',
      cache: 'none'
    });

    await vmCreationWizard.nextStep();
    await vmCreationWizard.selectNetwork('default');
    await vmCreationWizard.nextStep();
    await vmCreationWizard.submit();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    // Verify disks
    await vmDetailsPage.navigateToTab('storage');
    const disks = await vmDetailsPage.getDiskList();
    expect(disks).toHaveLength(3);
  });

  test('should create VM with multiple network interfaces [multi-nic-vm]', async ({ page }) => {
    await dashboardPage.clickCreateVM();

    await vmCreationWizard.setVMName('multi-nic-vm');
    await vmCreationWizard.selectTemplate('alpine-3.18');
    await vmCreationWizard.nextStep();

    await vmCreationWizard.setCPU(2);
    await vmCreationWizard.setMemory(2048);
    await vmCreationWizard.nextStep();

    // Configure multiple networks
    await vmCreationWizard.selectNetwork('default');

    await vmCreationWizard.addNetworkInterface({
      network: 'management',
      type: 'virtio',
      ipv4: 'dhcp'
    });

    await vmCreationWizard.addNetworkInterface({
      network: 'storage',
      type: 'e1000',
      ipv4: 'static',
      address: '10.10.10.100',
      netmask: '255.255.255.0',
      gateway: '10.10.10.1'
    });

    await vmCreationWizard.nextStep();
    await vmCreationWizard.submit();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    // Verify network interfaces
    await vmDetailsPage.navigateToTab('network');
    const nics = await vmDetailsPage.getNetworkInterfaces();
    expect(nics).toHaveLength(3);
  });

  test('should create VM from custom ISO [iso-vm]', async ({ page }) => {
    await dashboardPage.clickCreateVM();

    await vmCreationWizard.selectCreationType('iso');

    await vmCreationWizard.setVMName('iso-vm');
    await vmCreationWizard.selectISO('custom-os.iso');
    await vmCreationWizard.selectOSType('linux');
    await vmCreationWizard.selectOSVariant('generic');
    await vmCreationWizard.nextStep();

    await vmCreationWizard.setCPU(2);
    await vmCreationWizard.setMemory(4096);
    await vmCreationWizard.setDisk(80);
    await vmCreationWizard.nextStep();

    await vmCreationWizard.selectNetwork('default');
    await vmCreationWizard.nextStep();
    await vmCreationWizard.submit();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    // Verify boot order includes CD-ROM
    await vmDetailsPage.navigateToTab('settings');
    const bootOrder = await vmDetailsPage.getBootOrder();
    expect(bootOrder[0]).toContain('cdrom');
  });

  test('should create VM with cloud-init configuration [cloudinit-vm]', async ({ page }) => {
    await dashboardPage.clickCreateVM();

    await vmCreationWizard.setVMName('cloudinit-vm');
    await vmCreationWizard.selectTemplate('ubuntu-22.04-cloud');
    await vmCreationWizard.nextStep();

    await vmCreationWizard.setCPU(2);
    await vmCreationWizard.setMemory(4096);
    await vmCreationWizard.nextStep();

    await vmCreationWizard.selectNetwork('default');
    await vmCreationWizard.nextStep();

    // Configure cloud-init
    await vmCreationWizard.enableCloudInit();
    await vmCreationWizard.setCloudInitHostname('cloudinit-test');
    await vmCreationWizard.setCloudInitUser('ubuntu');
    await vmCreationWizard.setCloudInitPassword('SecureP@ss123!');
    await vmCreationWizard.addCloudInitSSHKey('ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQ...');

    await vmCreationWizard.setCloudInitUserData(`#cloud-config
packages:
  - nginx
  - docker.io
runcmd:
  - systemctl start nginx
  - systemctl enable docker
`);

    await vmCreationWizard.submit();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
  });

  test('should validate VM name requirements', async ({ page }) => {
    await dashboardPage.clickCreateVM();

    const invalidNames = [
      '',
      'a',
      'name with spaces',
      'name_with_underscore',
      'UPPERCASE',
      'name-that-is-way-too-long-and-exceeds-maximum-length-limit'
    ];

    for (const name of invalidNames) {
      await vmCreationWizard.setVMName(name);
      await page.keyboard.press('Tab');

      await expect(page.locator('[data-testid="vm-name-error"]')).toBeVisible();
      await page.locator('[data-testid="vm-name-input"]').clear();
    }

    await vmCreationWizard.setVMName('valid-vm-name');
    await expect(page.locator('[data-testid="vm-name-error"]')).not.toBeVisible();
  });

  test('should prevent duplicate VM names', async ({ page }) => {
    // Create first VM
    await dashboardPage.clickCreateVM();
    await vmCreationWizard.quickCreate({
      name: 'duplicate-test',
      template: 'alpine-3.18',
      cpu: 1,
      memory: 1024,
      disk: 10
    });

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    // Try to create second VM with same name
    await dashboardPage.goto();
    await dashboardPage.clickCreateVM();

    await vmCreationWizard.setVMName('duplicate-test');
    await page.keyboard.press('Tab');

    await expect(page.locator('[data-testid="vm-name-error"]'))
      .toContainText('already exists');

    // Cleanup
    await dashboardPage.goto();
    await dashboardPage.clickVM('duplicate-test');
    await vmDetailsPage.forceDelete();
  });

  test('should validate resource constraints', async ({ page }) => {
    await dashboardPage.clickCreateVM();

    await vmCreationWizard.setVMName('resource-test');
    await vmCreationWizard.selectTemplate('ubuntu-22.04');
    await vmCreationWizard.nextStep();

    // Test CPU limits
    await vmCreationWizard.setCPU(9999);
    await expect(page.locator('[data-testid="cpu-error"]')).toBeVisible();

    await vmCreationWizard.setCPU(0);
    await expect(page.locator('[data-testid="cpu-error"]')).toBeVisible();

    await vmCreationWizard.setCPU(4);
    await expect(page.locator('[data-testid="cpu-error"]')).not.toBeVisible();

    // Test memory limits
    await vmCreationWizard.setMemory(999999999);
    await expect(page.locator('[data-testid="memory-error"]')).toBeVisible();

    await vmCreationWizard.setMemory(128);
    await expect(page.locator('[data-testid="memory-error"]')).toBeVisible();

    await vmCreationWizard.setMemory(4096);
    await expect(page.locator('[data-testid="memory-error"]')).not.toBeVisible();

    // Test disk limits
    await vmCreationWizard.setDisk(999999);
    await expect(page.locator('[data-testid="disk-error"]')).toBeVisible();

    await vmCreationWizard.setDisk(1);
    await expect(page.locator('[data-testid="disk-error"]')).toBeVisible();

    await vmCreationWizard.setDisk(50);
    await expect(page.locator('[data-testid="disk-error"]')).not.toBeVisible();
  });

  test('should create VM from template with preserved settings [template-vm]', async ({ page }) => {
    await dashboardPage.clickCreateVM();

    await vmCreationWizard.setVMName('template-vm');

    // Select template with predefined settings
    await vmCreationWizard.selectTemplate('web-server-template');

    // Verify template settings are applied
    await vmCreationWizard.nextStep();

    const cpu = await vmCreationWizard.getCPUValue();
    const memory = await vmCreationWizard.getMemoryValue();

    expect(cpu).toBe(4);
    expect(memory).toBe(8192);

    await vmCreationWizard.nextStep();
    await vmCreationWizard.nextStep();
    await vmCreationWizard.submit();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
  });

  test('should support quick create for common configurations', async ({ page }) => {
    await dashboardPage.clickQuickCreate();

    await page.locator('[data-testid="quick-create-preset-web"]').click();

    await page.locator('[data-testid="vm-name"]').fill('quick-web-vm');
    await page.locator('[data-testid="quick-create-submit"]').click();

    await expect(page).toHaveURL(/\/vms\/[a-f0-9-]+/);
    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    // Cleanup
    await vmDetailsPage.forceDelete();
  });
});
