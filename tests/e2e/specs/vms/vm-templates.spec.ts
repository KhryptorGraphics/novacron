import { test, expect } from '@playwright/test';
import { DashboardPage } from '../../pages/DashboardPage';
import { VMDetailsPage } from '../../pages/VMDetailsPage';
import { TemplatePage } from '../../pages/TemplatePage';
import { authenticateUser } from '../../fixtures/auth';
import { testData } from '../../fixtures/testData';

test.describe('VM Management: Templates', () => {
  let dashboardPage: DashboardPage;
  let vmDetailsPage: VMDetailsPage;
  let templatePage: TemplatePage;

  test.beforeEach(async ({ page }) => {
    await authenticateUser(page, testData.users.admin);

    dashboardPage = new DashboardPage(page);
    vmDetailsPage = new VMDetailsPage(page);
    templatePage = new TemplatePage(page);
  });

  test('should create template from existing VM', {
    tag: '@smoke',
  }, async ({ page }) => {
    // Create a VM first
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `template-source-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-minimal"]').click();
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    // Stop VM before creating template
    await vmDetailsPage.clickAction('stop');
    await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });

    // Create template
    await vmDetailsPage.clickAction('create-template');

    await page.locator('[data-testid="template-name"]').fill('my-custom-template');
    await page.locator('[data-testid="template-description"]').fill('Custom template created from VM');
    await page.locator('[data-testid="template-category"]').selectOption('custom');
    await page.locator('[data-testid="create-template-submit"]').click();

    await expect(page.locator('[data-testid="template-creating"]')).toBeVisible();
    await expect(page.locator('[data-testid="template-created"]')).toBeVisible({ timeout: 60000 });

    // Verify template exists
    await templatePage.goto();

    const templates = await templatePage.getTemplateList();
    expect(templates).toContainEqual(
      expect.objectContaining({
        name: 'my-custom-template'
      })
    );

    // Cleanup
    await dashboardPage.goto();
    await dashboardPage.clickVM(vmName);
    await vmDetailsPage.forceDelete();
  });

  test('should create VM from template', async ({ page }) => {
    await templatePage.goto();

    const templateName = 'ubuntu-22.04';
    await templatePage.selectTemplate(templateName);
    await templatePage.clickCreateFromTemplate();

    await page.locator('[data-testid="vm-name"]').fill(`from-template-${Date.now()}`);
    await page.locator('[data-testid="wizard-next"]').click();

    // Template settings should be pre-filled
    const cpu = await page.locator('[data-testid="cpu-input"]').inputValue();
    const memory = await page.locator('[data-testid="memory-input"]').inputValue();

    expect(parseInt(cpu)).toBeGreaterThan(0);
    expect(parseInt(memory)).toBeGreaterThan(0);

    await page.locator('[data-testid="wizard-next"]').click();
    await page.locator('[data-testid="wizard-next"]').click();
    await page.locator('[data-testid="wizard-submit"]').click();

    await expect(page).toHaveURL(/\/vms\/[a-f0-9-]+/);
    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    // Cleanup
    await vmDetailsPage.forceDelete();
  });

  test('should edit template', async ({ page }) => {
    await templatePage.goto();

    const templateName = 'debian-12';
    await templatePage.selectTemplate(templateName);
    await templatePage.clickEditTemplate();

    await page.locator('[data-testid="template-description"]').fill('Updated Debian 12 template');
    await page.locator('[data-testid="template-cpu"]').fill('4');
    await page.locator('[data-testid="template-memory"]').fill('8192');
    await page.locator('[data-testid="save-template"]').click();

    await expect(page.locator('[data-testid="template-updated"]')).toBeVisible();

    // Verify changes
    const template = await templatePage.getTemplateDetails(templateName);
    expect(template.description).toContain('Updated Debian 12 template');
    expect(template.cpu).toBe(4);
    expect(template.memory).toBe(8192);
  });

  test('should delete template', async ({ page }) => {
    // Create a template to delete
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `delete-template-source-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-minimal"]').click();
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });
    await vmDetailsPage.clickAction('stop');
    await vmDetailsPage.waitForVMState('stopped', { timeout: 30000 });

    const templateName = `delete-me-template-${Date.now()}`;
    await vmDetailsPage.clickAction('create-template');
    await page.locator('[data-testid="template-name"]').fill(templateName);
    await page.locator('[data-testid="create-template-submit"]').click();
    await expect(page.locator('[data-testid="template-created"]')).toBeVisible({ timeout: 60000 });

    // Delete template
    await templatePage.goto();
    await templatePage.selectTemplate(templateName);
    await templatePage.clickDeleteTemplate();

    await page.locator('[data-testid="confirm-delete-template"]').click();

    await expect(page.locator('[data-testid="template-deleted"]')).toBeVisible();

    // Verify template is removed
    const templates = await templatePage.getTemplateList();
    expect(templates).not.toContainEqual(
      expect.objectContaining({
        name: templateName
      })
    );

    // Cleanup VM
    await dashboardPage.goto();
    await dashboardPage.clickVM(vmName);
    await vmDetailsPage.forceDelete();
  });

  test('should clone template', async ({ page }) => {
    await templatePage.goto();

    const sourceTemplate = 'alpine-3.18';
    await templatePage.selectTemplate(sourceTemplate);
    await templatePage.clickCloneTemplate();

    const cloneName = `alpine-clone-${Date.now()}`;
    await page.locator('[data-testid="clone-name"]').fill(cloneName);
    await page.locator('[data-testid="clone-description"]').fill('Cloned from Alpine template');
    await page.locator('[data-testid="clone-submit"]').click();

    await expect(page.locator('[data-testid="template-cloned"]')).toBeVisible();

    // Verify clone exists
    const templates = await templatePage.getTemplateList();
    expect(templates).toContainEqual(
      expect.objectContaining({
        name: cloneName
      })
    );

    // Cleanup - delete cloned template
    await templatePage.selectTemplate(cloneName);
    await templatePage.clickDeleteTemplate();
    await page.locator('[data-testid="confirm-delete-template"]').click();
  });

  test('should export template', async ({ page }) => {
    await templatePage.goto();

    const templateName = 'ubuntu-22.04';
    await templatePage.selectTemplate(templateName);
    await templatePage.clickExportTemplate();

    const downloadPromise = page.waitForEvent('download');
    await page.locator('[data-testid="confirm-export"]').click();

    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/template.*\.tar\.gz/);
  });

  test('should import template', async ({ page }) => {
    await templatePage.goto();

    await templatePage.clickImportTemplate();

    // Mock file upload
    const fileInput = page.locator('[data-testid="template-file-input"]');
    await fileInput.setInputFiles({
      name: 'imported-template.tar.gz',
      mimeType: 'application/gzip',
      buffer: Buffer.from('mock template data')
    });

    await page.locator('[data-testid="import-name"]').fill(`imported-${Date.now()}`);
    await page.locator('[data-testid="import-submit"]').click();

    await expect(page.locator('[data-testid="template-importing"]')).toBeVisible();
    await expect(page.locator('[data-testid="template-imported"]')).toBeVisible({ timeout: 60000 });
  });

  test('should manage template categories', async ({ page }) => {
    await templatePage.goto();

    // Filter by category
    await templatePage.selectCategory('linux');

    const linuxTemplates = await templatePage.getTemplateList();
    expect(linuxTemplates.every(t => t.category === 'linux')).toBe(true);

    await templatePage.selectCategory('windows');

    const windowsTemplates = await templatePage.getTemplateList();
    expect(windowsTemplates.every(t => t.category === 'windows')).toBe(true);
  });

  test('should manage template tags', async ({ page }) => {
    await templatePage.goto();

    const templateName = 'centos-9';
    await templatePage.selectTemplate(templateName);
    await templatePage.clickEditTemplate();

    await page.locator('[data-testid="add-tag"]').click();
    await page.locator('[data-testid="tag-input"]').fill('production');
    await page.locator('[data-testid="add-tag-button"]').click();

    await page.locator('[data-testid="add-tag"]').click();
    await page.locator('[data-testid="tag-input"]').fill('web-server');
    await page.locator('[data-testid="add-tag-button"]').click();

    await page.locator('[data-testid="save-template"]').click();

    await expect(page.locator('[data-testid="template-updated"]')).toBeVisible();

    // Filter by tag
    await templatePage.goto();
    await templatePage.filterByTag('production');

    const productionTemplates = await templatePage.getTemplateList();
    expect(productionTemplates).toContainEqual(
      expect.objectContaining({
        name: templateName
      })
    );
  });

  test('should handle template versioning', async ({ page }) => {
    await templatePage.goto();

    const templateName = 'ubuntu-22.04';
    await templatePage.selectTemplate(templateName);
    await templatePage.viewVersionHistory();

    const versions = await templatePage.getVersionList();
    expect(versions.length).toBeGreaterThan(0);

    // Each version should have metadata
    expect(versions[0]).toHaveProperty('version');
    expect(versions[0]).toHaveProperty('created');
    expect(versions[0]).toHaveProperty('size');
  });

  test('should share template with users', async ({ page }) => {
    await templatePage.goto();

    const templateName = 'alpine-3.18';
    await templatePage.selectTemplate(templateName);
    await templatePage.clickShareTemplate();

    await page.locator('[data-testid="share-with-user"]').fill('user@example.com');
    await page.locator('[data-testid="share-permission"]').selectOption('read');
    await page.locator('[data-testid="add-share"]').click();

    await expect(page.locator('[data-testid="share-added"]')).toBeVisible();

    // Verify share appears in list
    const shares = await templatePage.getShareList();
    expect(shares).toContainEqual(
      expect.objectContaining({
        user: 'user@example.com',
        permission: 'read'
      })
    );
  });

  test('should validate template requirements', async ({ page }) => {
    await dashboardPage.goto();
    await dashboardPage.clickQuickCreate();

    const vmName = `validate-template-${Date.now()}`;
    await page.locator('[data-testid="quick-create-preset-minimal"]').click();
    await page.locator('[data-testid="vm-name"]').fill(vmName);
    await page.locator('[data-testid="quick-create-submit"]').click();

    await vmDetailsPage.waitForVMState('running', { timeout: 60000 });

    // Try to create template from running VM
    await vmDetailsPage.clickAction('create-template');

    await expect(page.locator('[data-testid="template-validation-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="template-validation-error"]')).toContainText('must be stopped');

    // Cleanup
    await page.locator('[data-testid="cancel-create-template"]').click();
    await vmDetailsPage.forceDelete();
  });
});
