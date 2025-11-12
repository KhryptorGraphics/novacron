import { test, expect } from '@playwright/test';
import { ClusterPage } from '../../pages/ClusterPage';
import { authenticateUser } from '../../fixtures/auth';
import { testData } from '../../fixtures/testData';

test.describe('Cluster: Federation', () => {
  let clusterPage: ClusterPage;

  test.beforeEach(async ({ page }) => {
    await authenticateUser(page, testData.users.admin);

    clusterPage = new ClusterPage(page);

    await clusterPage.goto();
    await clusterPage.navigateToTab('federation');
  });

  test('should create federation', {
    tag: '@smoke',
  }, async ({ page }) => {
    await page.locator('[data-testid="add-federation"]').click();

    const federationName = `test-federation-${Date.now()}`;
    await page.locator('[data-testid="federation-name"]').fill(federationName);
    await page.locator('[data-testid="remote-cluster-url"]').fill('https://cluster2.example.com');
    await page.locator('[data-testid="federation-token"]').fill('mock-token-12345');

    await page.locator('[data-testid="submit-federation"]').click();

    await expect(page.locator('[data-testid="federation-establishing"]')).toBeVisible();
    await expect(page.locator('[data-testid="federation-active"]')).toBeVisible({ timeout: 30000 });

    const federations = await clusterPage.getFederationList();
    expect(federations).toContainEqual(
      expect.objectContaining({
        name: federationName,
        status: 'active'
      })
    );
  });

  test('should view federated resources', async ({ page }) => {
    const federations = await clusterPage.getFederationList();

    if (federations.length === 0) {
      test.skip();
    }

    const federation = federations[0];

    await page.locator(`[data-testid="federation-${federation.name}"]`).click();

    // Verify remote cluster information
    await expect(page.locator('[data-testid="remote-cluster-info"]')).toBeVisible();

    // Verify remote resources
    await expect(page.locator('[data-testid="remote-nodes"]')).toBeVisible();
    await expect(page.locator('[data-testid="remote-vms"]')).toBeVisible();

    const remoteNodes = await page.locator('[data-testid="remote-node"]').all();
    expect(remoteNodes.length).toBeGreaterThan(0);
  });

  test('should sync resources across federation', async ({ page }) => {
    const federations = await clusterPage.getFederationList();

    if (federations.length === 0) {
      test.skip();
    }

    const federation = federations[0];

    await page.locator(`[data-testid="federation-${federation.name}"]`).click();

    await page.locator('[data-testid="sync-resources"]').click();

    await expect(page.locator('[data-testid="syncing"]')).toBeVisible();
    await expect(page.locator('[data-testid="sync-complete"]')).toBeVisible({ timeout: 30000 });

    // Verify sync timestamp updated
    const lastSync = await page.locator('[data-testid="last-sync"]').textContent();
    expect(lastSync).toBeTruthy();
  });

  test('should configure federation policies', async ({ page }) => {
    const federations = await clusterPage.getFederationList();

    if (federations.length === 0) {
      test.skip();
    }

    const federation = federations[0];

    await page.locator(`[data-testid="federation-${federation.name}"]`).click();

    await page.locator('[data-testid="edit-policies"]').click();

    // Configure resource sharing
    await page.locator('[data-testid="allow-vm-migration"]').check();
    await page.locator('[data-testid="allow-resource-sharing"]').check();

    // Configure quotas
    await page.locator('[data-testid="max-shared-cpu"]').fill('100');
    await page.locator('[data-testid="max-shared-memory"]').fill('204800');

    await page.locator('[data-testid="save-policies"]').click();

    await expect(page.locator('[data-testid="policies-updated"]')).toBeVisible();
  });

  test('should handle federation connectivity issues', async ({ page }) => {
    const federations = await clusterPage.getFederationList();

    if (federations.length === 0) {
      test.skip();
    }

    const federation = federations[0];

    // Simulate connectivity error
    await page.route(`**/api/federation/${federation.id}/health`, route => {
      route.fulfill({
        status: 503,
        body: JSON.stringify({ error: 'Connection timeout' })
      });
    });

    await page.locator(`[data-testid="federation-${federation.name}"]`).click();

    await expect(page.locator('[data-testid="federation-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="federation-status"]')).toContainText('disconnected');

    // Verify retry mechanism
    await page.locator('[data-testid="retry-connection"]').click();

    await expect(page.locator('[data-testid="reconnecting"]')).toBeVisible();
  });

  test('should remove federation', async ({ page }) => {
    const federations = await clusterPage.getFederationList();

    if (federations.length === 0) {
      test.skip();
    }

    const federation = federations[0];

    await page.locator(`[data-testid="federation-${federation.name}"]`).click();

    await page.locator('[data-testid="federation-actions"]').click();
    await page.locator('[data-testid="remove-federation"]').click();

    await page.locator('[data-testid="confirm-remove-text"]').fill('REMOVE');
    await page.locator('[data-testid="confirm-remove-federation"]').click();

    await expect(page.locator('[data-testid="federation-removing"]')).toBeVisible();
    await expect(page.locator(`[data-testid="federation-${federation.name}"]`)).not.toBeVisible({ timeout: 30000 });

    const updatedFederations = await clusterPage.getFederationList();
    expect(updatedFederations).not.toContainEqual(
      expect.objectContaining({
        name: federation.name
      })
    );
  });
});
