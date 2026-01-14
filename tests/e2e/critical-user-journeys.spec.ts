import { test, expect, Page, BrowserContext } from '@playwright/test';

// Test data and configuration
const TEST_CONFIG = {
  baseURL: process.env.BASE_URL || 'http://localhost:3000',
  apiURL: process.env.API_URL || 'http://localhost:8080',
  testTimeout: 30000,
  adminUser: {
    username: 'admin',
    password: 'admin123',
    email: 'admin@novacron.test'
  },
  regularUser: {
    username: 'testuser', 
    password: 'user123',
    email: 'user@novacron.test'
  }
};

test.describe('Critical User Journeys - VM Management', () => {
  let context: BrowserContext;
  let page: Page;

  test.beforeAll(async ({ browser }) => {
    context = await browser.newContext({
      viewport: { width: 1920, height: 1080 },
      recordVideo: { dir: 'test-results/videos' },
    });
    page = await context.newPage();
    
    // Set up test data and API mocks if needed
    await setupTestEnvironment(page);
  });

  test.afterAll(async () => {
    await context.close();
  });

  test.beforeEach(async () => {
    await page.goto(TEST_CONFIG.baseURL);
  });

  test.describe('Authentication Flow', () => {
    test('should allow admin user to login and access dashboard', async () => {
      // Navigate to login page
      await page.click('[data-testid="login-button"]');
      await expect(page).toHaveURL(/.*\/login/);

      // Fill login form
      await page.fill('[data-testid="username-input"]', TEST_CONFIG.adminUser.username);
      await page.fill('[data-testid="password-input"]', TEST_CONFIG.adminUser.password);
      
      // Submit form
      await page.click('[data-testid="login-submit"]');

      // Should redirect to dashboard
      await expect(page).toHaveURL(/.*\/dashboard/);
      await expect(page.locator('[data-testid="dashboard-title"]')).toBeVisible();
      await expect(page.locator('[data-testid="welcome-message"]')).toContainText('admin');

      // Verify admin-specific elements
      await expect(page.locator('[data-testid="admin-panel-link"]')).toBeVisible();
      await expect(page.locator('[data-testid="system-stats"]')).toBeVisible();
    });

    test('should reject invalid credentials', async () => {
      await page.click('[data-testid="login-button"]');
      
      await page.fill('[data-testid="username-input"]', 'invalid');
      await page.fill('[data-testid="password-input"]', 'wrongpassword');
      await page.click('[data-testid="login-submit"]');

      // Should show error message
      await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
      await expect(page.locator('[data-testid="error-message"]')).toContainText('Invalid credentials');
      
      // Should stay on login page
      await expect(page).toHaveURL(/.*\/login/);
    });

    test('should handle session timeout gracefully', async () => {
      // Login first
      await loginAsAdmin(page);
      
      // Mock session expiry by intercepting API calls
      await page.route('**/api/v1/**', (route) => {
        route.fulfill({
          status: 401,
          body: JSON.stringify({ message: 'Session expired' })
        });
      });

      // Try to access protected resource
      await page.click('[data-testid="vms-nav-link"]');

      // Should redirect to login
      await expect(page).toHaveURL(/.*\/login/);
      await expect(page.locator('[data-testid="session-expired-message"]')).toBeVisible();
    });

    test('should allow user to logout', async () => {
      await loginAsAdmin(page);
      
      // Click logout
      await page.click('[data-testid="user-menu"]');
      await page.click('[data-testid="logout-button"]');

      // Should redirect to home page
      await expect(page).toHaveURL(TEST_CONFIG.baseURL);
      await expect(page.locator('[data-testid="login-button"]')).toBeVisible();
    });
  });

  test.describe('VM Creation Workflow', () => {
    test.beforeEach(async () => {
      await loginAsAdmin(page);
    });

    test('should create a new VM with all configurations', async () => {
      // Navigate to VM creation
      await page.click('[data-testid="vms-nav-link"]');
      await page.click('[data-testid="create-vm-button"]');

      await expect(page).toHaveURL(/.*\/vms\/create/);

      // Fill VM configuration form
      await page.fill('[data-testid="vm-name-input"]', 'Test VM E2E');
      await page.selectOption('[data-testid="vm-type-select"]', 'qemu');
      await page.fill('[data-testid="vm-cpu-input"]', '4');
      await page.fill('[data-testid="vm-memory-input"]', '8192');
      await page.fill('[data-testid="vm-disk-input"]', '50');

      // Configure networking
      await page.click('[data-testid="add-network-button"]');
      await page.selectOption('[data-testid="network-select-0"]', 'default');

      // Set advanced options
      await page.click('[data-testid="advanced-options-toggle"]');
      await page.check('[data-testid="enable-nested-virtualization"]');
      await page.selectOption('[data-testid="boot-order-select"]', 'disk');

      // Submit form
      await page.click('[data-testid="create-vm-submit"]');

      // Should show success message and redirect
      await expect(page.locator('[data-testid="success-notification"]')).toBeVisible();
      await expect(page.locator('[data-testid="success-notification"]')).toContainText('VM created successfully');
      
      // Should redirect to VM details page
      await expect(page).toHaveURL(/.*\/vms\/[a-zA-Z0-9-]+/);
      
      // Verify VM details
      await expect(page.locator('[data-testid="vm-name"]')).toContainText('Test VM E2E');
      await expect(page.locator('[data-testid="vm-status"]')).toContainText('Created');
      await expect(page.locator('[data-testid="vm-cpu"]')).toContainText('4');
      await expect(page.locator('[data-testid="vm-memory"]')).toContainText('8192');
    });

    test('should validate required fields', async () => {
      await page.click('[data-testid="vms-nav-link"]');
      await page.click('[data-testid="create-vm-button"]');

      // Try to submit without required fields
      await page.click('[data-testid="create-vm-submit"]');

      // Should show validation errors
      await expect(page.locator('[data-testid="vm-name-error"]')).toBeVisible();
      await expect(page.locator('[data-testid="vm-name-error"]')).toContainText('Name is required');
      
      // Form should not submit
      await expect(page).toHaveURL(/.*\/vms\/create/);
    });

    test('should handle form validation edge cases', async () => {
      await page.click('[data-testid="vms-nav-link"]');
      await page.click('[data-testid="create-vm-button"]');

      // Test negative values
      await page.fill('[data-testid="vm-cpu-input"]', '-1');
      await page.fill('[data-testid="vm-memory-input"]', '0');
      await page.fill('[data-testid="vm-disk-input"]', '-10');

      await page.click('[data-testid="create-vm-submit"]');

      // Should show validation errors
      await expect(page.locator('[data-testid="vm-cpu-error"]')).toContainText('CPU must be positive');
      await expect(page.locator('[data-testid="vm-memory-error"]')).toContainText('Memory must be greater than 0');
      await expect(page.locator('[data-testid="vm-disk-error"]')).toContainText('Disk size must be positive');
    });

    test('should save draft configuration', async () => {
      await page.click('[data-testid="vms-nav-link"]');
      await page.click('[data-testid="create-vm-button"]');

      // Fill partial form
      await page.fill('[data-testid="vm-name-input"]', 'Draft VM');
      await page.fill('[data-testid="vm-cpu-input"]', '2');

      // Save as draft
      await page.click('[data-testid="save-draft-button"]');

      await expect(page.locator('[data-testid="draft-saved-notification"]')).toBeVisible();

      // Navigate away and back
      await page.click('[data-testid="vms-nav-link"]');
      await page.click('[data-testid="create-vm-button"]');

      // Should restore draft
      await expect(page.locator('[data-testid="vm-name-input"]')).toHaveValue('Draft VM');
      await expect(page.locator('[data-testid="vm-cpu-input"]')).toHaveValue('2');
    });
  });

  test.describe('VM Management Operations', () => {
    let vmId: string;

    test.beforeEach(async () => {
      await loginAsAdmin(page);
      vmId = await createTestVM(page);
    });

    test.afterEach(async () => {
      // Clean up test VM
      await deleteTestVM(page, vmId);
    });

    test('should start, stop, and restart VM', async () => {
      await page.goto(`${TEST_CONFIG.baseURL}/vms/${vmId}`);

      // Start VM
      await page.click('[data-testid="start-vm-button"]');
      await expect(page.locator('[data-testid="vm-status"]')).toContainText('Starting');
      
      // Wait for VM to be running (with timeout)
      await page.waitForSelector('[data-testid="vm-status"]:has-text("Running")', { timeout: 30000 });
      await expect(page.locator('[data-testid="start-vm-button"]')).not.toBeVisible();
      await expect(page.locator('[data-testid="stop-vm-button"]')).toBeVisible();

      // Stop VM
      await page.click('[data-testid="stop-vm-button"]');
      await expect(page.locator('[data-testid="vm-status"]')).toContainText('Stopping');
      
      await page.waitForSelector('[data-testid="vm-status"]:has-text("Stopped")', { timeout: 30000 });
      await expect(page.locator('[data-testid="stop-vm-button"]')).not.toBeVisible();
      await expect(page.locator('[data-testid="start-vm-button"]')).toBeVisible();

      // Restart VM (start it first)
      await page.click('[data-testid="start-vm-button"]');
      await page.waitForSelector('[data-testid="vm-status"]:has-text("Running")', { timeout: 30000 });
      
      await page.click('[data-testid="restart-vm-button"]');
      await expect(page.locator('[data-testid="vm-status"]')).toContainText('Restarting');
      
      await page.waitForSelector('[data-testid="vm-status"]:has-text("Running")', { timeout: 30000 });
    });

    test('should update VM configuration', async () => {
      await page.goto(`${TEST_CONFIG.baseURL}/vms/${vmId}`);

      // Click edit button
      await page.click('[data-testid="edit-vm-button"]');
      await expect(page).toHaveURL(/.*\/vms\/[a-zA-Z0-9-]+\/edit/);

      // Update configuration
      await page.fill('[data-testid="vm-name-input"]', 'Updated Test VM');
      await page.fill('[data-testid="vm-cpu-input"]', '4');
      await page.fill('[data-testid="vm-memory-input"]', '8192');

      // Save changes
      await page.click('[data-testid="save-vm-button"]');

      // Should redirect back to VM details
      await expect(page).toHaveURL(`${TEST_CONFIG.baseURL}/vms/${vmId}`);
      await expect(page.locator('[data-testid="success-notification"]')).toBeVisible();

      // Verify changes
      await expect(page.locator('[data-testid="vm-name"]')).toContainText('Updated Test VM');
      await expect(page.locator('[data-testid="vm-cpu"]')).toContainText('4');
      await expect(page.locator('[data-testid="vm-memory"]')).toContainText('8192');
    });

    test('should clone VM successfully', async () => {
      await page.goto(`${TEST_CONFIG.baseURL}/vms/${vmId}`);

      // Click clone button
      await page.click('[data-testid="clone-vm-button"]');

      // Should open clone dialog
      await expect(page.locator('[data-testid="clone-vm-dialog"]')).toBeVisible();

      // Fill clone form
      await page.fill('[data-testid="clone-vm-name-input"]', 'Cloned Test VM');
      await page.check('[data-testid="clone-with-data-checkbox"]');

      // Confirm clone
      await page.click('[data-testid="confirm-clone-button"]');

      // Should show success message
      await expect(page.locator('[data-testid="success-notification"]')).toBeVisible();
      await expect(page.locator('[data-testid="success-notification"]')).toContainText('VM cloned successfully');

      // Should navigate to cloned VM
      await expect(page).toHaveURL(/.*\/vms\/[a-zA-Z0-9-]+/);
      await expect(page.locator('[data-testid="vm-name"]')).toContainText('Cloned Test VM');
    });

    test('should create VM snapshot', async () => {
      await page.goto(`${TEST_CONFIG.baseURL}/vms/${vmId}`);

      // Navigate to snapshots tab
      await page.click('[data-testid="snapshots-tab"]');

      // Create snapshot
      await page.click('[data-testid="create-snapshot-button"]');
      
      await page.fill('[data-testid="snapshot-name-input"]', 'Test Snapshot');
      await page.fill('[data-testid="snapshot-description-input"]', 'E2E test snapshot');
      
      await page.click('[data-testid="create-snapshot-confirm"]');

      // Should show in snapshots list
      await expect(page.locator('[data-testid="snapshot-list"]')).toContainText('Test Snapshot');
      await expect(page.locator('[data-testid="snapshot-list"]')).toContainText('E2E test snapshot');
    });
  });

  test.describe('VM Monitoring and Metrics', () => {
    test.beforeEach(async () => {
      await loginAsAdmin(page);
    });

    test('should display real-time VM metrics', async () => {
      const vmId = await createTestVM(page);
      await page.goto(`${TEST_CONFIG.baseURL}/vms/${vmId}`);

      // Start VM to generate metrics
      await page.click('[data-testid="start-vm-button"]');
      await page.waitForSelector('[data-testid="vm-status"]:has-text("Running")', { timeout: 30000 });

      // Navigate to monitoring tab
      await page.click('[data-testid="monitoring-tab"]');

      // Should show metrics charts
      await expect(page.locator('[data-testid="cpu-usage-chart"]')).toBeVisible();
      await expect(page.locator('[data-testid="memory-usage-chart"]')).toBeVisible();
      await expect(page.locator('[data-testid="disk-io-chart"]')).toBeVisible();
      await expect(page.locator('[data-testid="network-io-chart"]')).toBeVisible();

      // Metrics should update periodically
      const initialCpuValue = await page.locator('[data-testid="cpu-usage-value"]').textContent();
      
      // Wait for metric update
      await page.waitForTimeout(5000);
      
      const updatedCpuValue = await page.locator('[data-testid="cpu-usage-value"]').textContent();
      // Values might be the same in test environment, but elements should be present
      
      await deleteTestVM(page, vmId);
    });

    test('should show system-wide dashboard metrics', async () => {
      await page.goto(`${TEST_CONFIG.baseURL}/dashboard`);

      // Should show system overview
      await expect(page.locator('[data-testid="total-vms-metric"]')).toBeVisible();
      await expect(page.locator('[data-testid="running-vms-metric"]')).toBeVisible();
      await expect(page.locator('[data-testid="system-load-metric"]')).toBeVisible();
      await expect(page.locator('[data-testid="memory-usage-metric"]')).toBeVisible();

      // Should show recent activity
      await expect(page.locator('[data-testid="recent-activity-list"]')).toBeVisible();

      // Should show alerts if any
      await expect(page.locator('[data-testid="alerts-section"]')).toBeVisible();
    });
  });

  test.describe('User Management (Admin)', () => {
    test.beforeEach(async () => {
      await loginAsAdmin(page);
    });

    test('should create new user account', async () => {
      await page.goto(`${TEST_CONFIG.baseURL}/admin/users`);

      await page.click('[data-testid="create-user-button"]');

      // Fill user form
      await page.fill('[data-testid="user-username-input"]', 'newtestuser');
      await page.fill('[data-testid="user-email-input"]', 'newuser@test.com');
      await page.fill('[data-testid="user-password-input"]', 'newpassword123');
      await page.fill('[data-testid="user-confirm-password-input"]', 'newpassword123');
      
      // Assign role
      await page.selectOption('[data-testid="user-role-select"]', 'user');

      await page.click('[data-testid="create-user-submit"]');

      // Should show success and redirect to users list
      await expect(page.locator('[data-testid="success-notification"]')).toBeVisible();
      await expect(page).toHaveURL(/.*\/admin\/users/);
      
      // Should appear in users list
      await expect(page.locator('[data-testid="users-table"]')).toContainText('newtestuser');
      await expect(page.locator('[data-testid="users-table"]')).toContainText('newuser@test.com');
    });

    test('should update user permissions', async () => {
      await page.goto(`${TEST_CONFIG.baseURL}/admin/users`);

      // Find and edit test user
      await page.click('[data-testid="edit-user-testuser"]');

      // Update permissions
      await page.check('[data-testid="permission-vm-create"]');
      await page.check('[data-testid="permission-vm-delete"]');
      await page.uncheck('[data-testid="permission-vm-admin"]');

      await page.click('[data-testid="save-user-permissions"]');

      await expect(page.locator('[data-testid="success-notification"]')).toBeVisible();
    });

    test('should disable/enable user account', async () => {
      await page.goto(`${TEST_CONFIG.baseURL}/admin/users`);

      // Disable user
      await page.click('[data-testid="disable-user-testuser"]');
      await page.click('[data-testid="confirm-disable-user"]');

      // Should show as disabled
      await expect(page.locator('[data-testid="user-status-testuser"]')).toContainText('Disabled');

      // Enable user
      await page.click('[data-testid="enable-user-testuser"]');
      
      await expect(page.locator('[data-testid="user-status-testuser"]')).toContainText('Active');
    });
  });

  test.describe('System Administration', () => {
    test.beforeEach(async () => {
      await loginAsAdmin(page);
    });

    test('should view system logs', async () => {
      await page.goto(`${TEST_CONFIG.baseURL}/admin/logs`);

      // Should show log entries
      await expect(page.locator('[data-testid="logs-table"]')).toBeVisible();
      
      // Should be able to filter logs
      await page.selectOption('[data-testid="log-level-filter"]', 'ERROR');
      await page.click('[data-testid="apply-log-filter"]');

      // Should update results
      await expect(page.locator('[data-testid="logs-table"] [data-level="ERROR"]')).toBeVisible();

      // Should be able to search logs
      await page.fill('[data-testid="log-search-input"]', 'VM');
      await page.press('[data-testid="log-search-input"]', 'Enter');

      // Results should be filtered
      await expect(page.locator('[data-testid="logs-table"]')).toContainText('VM');
    });

    test('should manage system settings', async () => {
      await page.goto(`${TEST_CONFIG.baseURL}/admin/settings`);

      // Update system settings
      await page.fill('[data-testid="max-vms-per-user-input"]', '10');
      await page.fill('[data-testid="default-vm-memory-input"]', '2048');
      await page.check('[data-testid="enable-auto-backup-checkbox"]');

      await page.click('[data-testid="save-settings-button"]');

      await expect(page.locator('[data-testid="success-notification"]')).toBeVisible();

      // Settings should persist after reload
      await page.reload();
      await expect(page.locator('[data-testid="max-vms-per-user-input"]')).toHaveValue('10');
      await expect(page.locator('[data-testid="enable-auto-backup-checkbox"]')).toBeChecked();
    });

    test('should backup and restore system', async () => {
      await page.goto(`${TEST_CONFIG.baseURL}/admin/backup`);

      // Create backup
      await page.click('[data-testid="create-backup-button"]');
      
      await page.fill('[data-testid="backup-name-input"]', 'E2E Test Backup');
      await page.check('[data-testid="include-vm-data-checkbox"]');
      
      await page.click('[data-testid="create-backup-confirm"]');

      // Should show backup progress
      await expect(page.locator('[data-testid="backup-progress"]')).toBeVisible();

      // Wait for backup completion (with timeout)
      await page.waitForSelector('[data-testid="backup-completed"]', { timeout: 60000 });

      // Should appear in backups list
      await expect(page.locator('[data-testid="backups-list"]')).toContainText('E2E Test Backup');
    });
  });

  test.describe('Error Handling and Edge Cases', () => {
    test.beforeEach(async () => {
      await loginAsAdmin(page);
    });

    test('should handle network errors gracefully', async () => {
      // Simulate network error
      await page.route('**/api/v1/vms', (route) => {
        route.abort('failed');
      });

      await page.goto(`${TEST_CONFIG.baseURL}/vms`);

      // Should show error message
      await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
      await expect(page.locator('[data-testid="error-message"]')).toContainText('Failed to load');

      // Should show retry button
      await expect(page.locator('[data-testid="retry-button"]')).toBeVisible();
    });

    test('should handle server errors (500)', async () => {
      await page.route('**/api/v1/vms/*/start', (route) => {
        route.fulfill({
          status: 500,
          body: JSON.stringify({ error: 'Internal server error' })
        });
      });

      const vmId = await createTestVM(page);
      await page.goto(`${TEST_CONFIG.baseURL}/vms/${vmId}`);

      await page.click('[data-testid="start-vm-button"]');

      // Should show error notification
      await expect(page.locator('[data-testid="error-notification"]')).toBeVisible();
      await expect(page.locator('[data-testid="error-notification"]')).toContainText('Failed to start VM');

      await deleteTestVM(page, vmId);
    });

    test('should handle concurrent operations conflicts', async () => {
      const vmId = await createTestVM(page);

      // Open two tabs with same VM
      const page2 = await context.newPage();
      await page2.goto(`${TEST_CONFIG.baseURL}/vms/${vmId}`);

      // Simulate concurrent start attempts
      await Promise.all([
        page.click('[data-testid="start-vm-button"]'),
        page2.click('[data-testid="start-vm-button"]')
      ]);

      // One should succeed, one should show conflict error
      const errorVisible = await Promise.race([
        page.locator('[data-testid="error-notification"]').isVisible(),
        page2.locator('[data-testid="error-notification"]').isVisible()
      ]);

      expect(errorVisible).toBe(true);

      await page2.close();
      await deleteTestVM(page, vmId);
    });

    test('should maintain functionality during high load', async () => {
      // Create multiple VMs concurrently to simulate load
      const vmCreationPromises = Array.from({ length: 5 }, (_, i) => 
        createTestVM(page, `Load Test VM ${i}`)
      );

      const vmIds = await Promise.all(vmCreationPromises);

      // All VMs should be created successfully
      expect(vmIds).toHaveLength(5);
      vmIds.forEach(id => expect(id).toBeTruthy());

      // Clean up
      await Promise.all(vmIds.map(id => deleteTestVM(page, id)));
    });
  });

  test.describe('Accessibility and Usability', () => {
    test.beforeEach(async () => {
      await loginAsAdmin(page);
    });

    test('should be keyboard navigable', async () => {
      await page.goto(`${TEST_CONFIG.baseURL}/vms`);

      // Tab through navigation
      await page.keyboard.press('Tab');
      await expect(page.locator(':focus')).toHaveAttribute('data-testid', 'dashboard-nav-link');

      await page.keyboard.press('Tab');
      await expect(page.locator(':focus')).toHaveAttribute('data-testid', 'vms-nav-link');

      // Enter should activate focused element
      await page.keyboard.press('Enter');
      await expect(page).toHaveURL(/.*\/vms/);

      // Should be able to navigate VM cards with keyboard
      await page.keyboard.press('Tab');
      const focusedElement = page.locator(':focus');
      await expect(focusedElement).toBeVisible();
    });

    test('should have proper ARIA labels', async () => {
      await page.goto(`${TEST_CONFIG.baseURL}/dashboard`);

      // Check important elements have ARIA labels
      await expect(page.locator('[data-testid="main-navigation"]')).toHaveAttribute('aria-label');
      await expect(page.locator('[data-testid="user-menu"]')).toHaveAttribute('aria-label');
      await expect(page.locator('[data-testid="vm-stats"]')).toHaveAttribute('aria-label');
    });

    test('should work with screen reader simulation', async () => {
      // Enable screen reader mode
      await page.emulateMedia({ reducedMotion: 'reduce' });
      
      await page.goto(`${TEST_CONFIG.baseURL}/dashboard`);

      // Check that important content is available to screen readers
      const announcement = await page.locator('[aria-live="polite"]');
      await expect(announcement).toBeVisible();

      // Status changes should be announced
      const vmId = await createTestVM(page);
      await page.goto(`${TEST_CONFIG.baseURL}/vms/${vmId}`);
      
      await page.click('[data-testid="start-vm-button"]');
      
      // Status change should be announced
      await expect(page.locator('[aria-live="polite"]')).toContainText('starting');

      await deleteTestVM(page, vmId);
    });
  });

  test.describe('Mobile Responsiveness', () => {
    test.beforeEach(async () => {
      // Set mobile viewport
      await page.setViewportSize({ width: 375, height: 667 });
      await loginAsAdmin(page);
    });

    test('should work on mobile devices', async () => {
      await page.goto(`${TEST_CONFIG.baseURL}/dashboard`);

      // Mobile navigation should be available
      await expect(page.locator('[data-testid="mobile-menu-button"]')).toBeVisible();
      
      // Open mobile menu
      await page.click('[data-testid="mobile-menu-button"]');
      await expect(page.locator('[data-testid="mobile-menu"]')).toBeVisible();

      // Should be able to navigate
      await page.click('[data-testid="mobile-vms-link"]');
      await expect(page).toHaveURL(/.*\/vms/);
    });

    test('should handle touch interactions', async () => {
      const vmId = await createTestVM(page);
      await page.goto(`${TEST_CONFIG.baseURL}/vms`);

      // Should be able to tap VM card
      const vmCard = page.locator(`[data-testid="vm-card-${vmId}"]`);
      await vmCard.tap();

      await expect(page).toHaveURL(`${TEST_CONFIG.baseURL}/vms/${vmId}`);

      await deleteTestVM(page, vmId);
    });
  });
});

// Helper functions
async function setupTestEnvironment(page: Page) {
  // Mock any external APIs or services
  await page.route('**/api/v1/health', (route) => {
    route.fulfill({
      status: 200,
      body: JSON.stringify({ status: 'healthy', timestamp: Date.now() })
    });
  });
}

async function loginAsAdmin(page: Page) {
  await page.goto(`${TEST_CONFIG.baseURL}/login`);
  await page.fill('[data-testid="username-input"]', TEST_CONFIG.adminUser.username);
  await page.fill('[data-testid="password-input"]', TEST_CONFIG.adminUser.password);
  await page.click('[data-testid="login-submit"]');
  await page.waitForURL(/.*\/dashboard/);
}

async function createTestVM(page: Page, name: string = 'E2E Test VM'): Promise<string> {
  await page.goto(`${TEST_CONFIG.baseURL}/vms/create`);
  
  await page.fill('[data-testid="vm-name-input"]', name);
  await page.selectOption('[data-testid="vm-type-select"]', 'qemu');
  await page.fill('[data-testid="vm-cpu-input"]', '2');
  await page.fill('[data-testid="vm-memory-input"]', '4096');
  await page.fill('[data-testid="vm-disk-input"]', '20');
  
  await page.click('[data-testid="create-vm-submit"]');
  
  // Wait for redirect to VM details page
  await page.waitForURL(/.*\/vms\/[a-zA-Z0-9-]+/);
  
  // Extract VM ID from URL
  const url = page.url();
  const vmId = url.split('/').pop();
  
  if (!vmId) {
    throw new Error('Failed to get VM ID from URL');
  }
  
  return vmId;
}

async function deleteTestVM(page: Page, vmId: string) {
  await page.goto(`${TEST_CONFIG.baseURL}/vms/${vmId}`);
  
  // Stop VM if running
  const stopButton = page.locator('[data-testid="stop-vm-button"]');
  if (await stopButton.isVisible()) {
    await stopButton.click();
    await page.waitForSelector('[data-testid="vm-status"]:has-text("Stopped")', { timeout: 30000 });
  }
  
  await page.click('[data-testid="delete-vm-button"]');
  await page.click('[data-testid="confirm-delete-vm"]');
  
  // Should redirect to VMs list
  await page.waitForURL(/.*\/vms$/);
}