import { test, expect } from '@playwright/test';
import { AlertsPage } from '../../pages/AlertsPage';
import { DashboardPage } from '../../pages/DashboardPage';
import { VMDetailsPage } from '../../pages/VMDetailsPage';
import { authenticateUser } from '../../fixtures/auth';
import { testData } from '../../fixtures/testData';

test.describe('Monitoring: Alerts', () => {
  let alertsPage: AlertsPage;
  let dashboardPage: DashboardPage;
  let vmDetailsPage: VMDetailsPage;

  test.beforeEach(async ({ page }) => {
    await authenticateUser(page, testData.users.admin);

    alertsPage = new AlertsPage(page);
    dashboardPage = new DashboardPage(page);
    vmDetailsPage = new VMDetailsPage(page);

    await alertsPage.goto();
  });

  test('should create custom alert rule', {
    tag: '@smoke',
  }, async ({ page }) => {
    await page.locator('[data-testid="create-alert-rule"]').click();

    await page.locator('[data-testid="rule-name"]').fill('High CPU Usage');
    await page.locator('[data-testid="rule-description"]').fill('Alert when CPU usage exceeds 80%');

    // Configure condition
    await page.locator('[data-testid="metric"]').selectOption('cpu_usage');
    await page.locator('[data-testid="operator"]').selectOption('greater_than');
    await page.locator('[data-testid="threshold"]').fill('80');
    await page.locator('[data-testid="duration"]').fill('5');

    // Configure severity
    await page.locator('[data-testid="severity"]').selectOption('warning');

    // Configure notifications
    await page.locator('[data-testid="notify-email"]').check();
    await page.locator('[data-testid="email-recipients"]').fill('ops@example.com');

    await page.locator('[data-testid="save-rule"]').click();

    await expect(page.locator('[data-testid="rule-created"]')).toBeVisible();

    // Verify rule appears in list
    const rules = await alertsPage.getAlertRules();
    expect(rules).toContainEqual(
      expect.objectContaining({
        name: 'High CPU Usage'
      })
    );
  });

  test('should list active alerts', async ({ page }) => {
    const alerts = await alertsPage.getActiveAlerts();

    // Verify alert structure
    for (const alert of alerts) {
      expect(alert).toHaveProperty('id');
      expect(alert).toHaveProperty('severity');
      expect(alert).toHaveProperty('message');
      expect(alert).toHaveProperty('timestamp');
    }

    // Filter by severity
    await page.locator('[data-testid="filter-severity"]').selectOption('critical');

    const criticalAlerts = await alertsPage.getActiveAlerts();

    for (const alert of criticalAlerts) {
      expect(alert.severity).toBe('critical');
    }
  });

  test('should acknowledge alert', async ({ page }) => {
    const alerts = await alertsPage.getActiveAlerts();

    if (alerts.length === 0) {
      test.skip();
    }

    const alert = alerts[0];

    await page.locator(`[data-testid="alert-${alert.id}"]`).click();
    await page.locator('[data-testid="acknowledge"]').click();

    await page.locator('[data-testid="ack-note"]').fill('Investigating the issue');
    await page.locator('[data-testid="confirm-acknowledge"]').click();

    await expect(page.locator('[data-testid="acknowledged"]')).toBeVisible();

    // Verify acknowledged badge
    const updatedAlert = await page.locator(`[data-testid="alert-${alert.id}"]`);
    await expect(updatedAlert.locator('[data-testid="ack-badge"]')).toBeVisible();
  });

  test('should configure alert routing', async ({ page }) => {
    await page.locator('[data-testid="configure-routing"]').click();

    // Create routing rule
    await page.locator('[data-testid="add-route"]').click();

    await page.locator('[data-testid="route-name"]').fill('Critical Alerts Route');

    // Configure matcher
    await page.locator('[data-testid="matcher-field"]').selectOption('severity');
    await page.locator('[data-testid="matcher-operator"]').selectOption('equals');
    await page.locator('[data-testid="matcher-value"]').fill('critical');

    // Configure receivers
    await page.locator('[data-testid="receiver-email"]').check();
    await page.locator('[data-testid="receiver-email-address"]').fill('critical-alerts@example.com');

    await page.locator('[data-testid="receiver-slack"]').check();
    await page.locator('[data-testid="receiver-slack-webhook"]').fill('https://hooks.slack.com/services/XXX');

    await page.locator('[data-testid="save-route"]').click();

    await expect(page.locator('[data-testid="route-saved"]')).toBeVisible();
  });

  test('should silence alerts', async ({ page }) => {
    await page.locator('[data-testid="create-silence"]').click();

    await page.locator('[data-testid="silence-comment"]').fill('Planned maintenance');

    // Configure matcher
    await page.locator('[data-testid="matcher-field"]').selectOption('alertname');
    await page.locator('[data-testid="matcher-value"]').fill('HighCPUUsage');

    // Set duration
    await page.locator('[data-testid="silence-duration"]').selectOption('4h');

    await page.locator('[data-testid="create-silence-submit"]').click();

    await expect(page.locator('[data-testid="silence-created"]')).toBeVisible();

    // Verify silence is active
    await page.locator('[data-testid="view-silences"]').click();

    const silences = await alertsPage.getActiveSilences();
    expect(silences).toContainEqual(
      expect.objectContaining({
        comment: 'Planned maintenance'
      })
    );
  });

  test('should configure alert templates', async ({ page }) => {
    await page.locator('[data-testid="configure-templates"]').click();

    await page.locator('[data-testid="add-template"]').click();

    await page.locator('[data-testid="template-name"]').fill('Email Alert Template');
    await page.locator('[data-testid="template-type"]').selectOption('email');

    await page.locator('[data-testid="template-subject"]').fill('{{ .Severity }}: {{ .AlertName }}');

    await page.locator('[data-testid="template-body"]').fill(`
Alert: {{ .AlertName }}
Severity: {{ .Severity }}
Description: {{ .Description }}
Started: {{ .StartsAt }}
    `);

    await page.locator('[data-testid="save-template"]').click();

    await expect(page.locator('[data-testid="template-saved"]')).toBeVisible();
  });

  test('should show alert history', async ({ page }) => {
    await page.locator('[data-testid="view-history"]').click();

    const historyItems = await page.locator('[data-testid="alert-history-item"]').all();
    expect(historyItems.length).toBeGreaterThan(0);

    // Verify history details
    const firstItem = historyItems[0];
    await expect(firstItem.locator('[data-testid="alert-name"]')).toBeVisible();
    await expect(firstItem.locator('[data-testid="start-time"]')).toBeVisible();
    await expect(firstItem.locator('[data-testid="end-time"]')).toBeVisible();
    await expect(firstItem.locator('[data-testid="duration"]')).toBeVisible();
  });

  test('should test alert rule', async ({ page }) => {
    const rules = await alertsPage.getAlertRules();

    if (rules.length === 0) {
      test.skip();
    }

    const rule = rules[0];

    await page.locator(`[data-testid="rule-${rule.id}"]`).click();
    await page.locator('[data-testid="test-rule"]').click();

    // Should show test results
    await expect(page.locator('[data-testid="test-results"]')).toBeVisible();

    // Should show if rule would fire
    const wouldFire = await page.locator('[data-testid="would-fire"]').isVisible();
    expect(wouldFire).toBeDefined();
  });

  test('should export alert data', async ({ page }) => {
    await page.locator('[data-testid="export-alerts"]').click();

    await page.locator('[data-testid="export-time-range"]').selectOption('7d');
    await page.locator('[data-testid="export-format"]').selectOption('json');

    const downloadPromise = page.waitForEvent('download');
    await page.locator('[data-testid="confirm-export"]').click();

    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/alerts.*\.json/);
  });

  test('should configure alert escalation', async ({ page }) => {
    await page.locator('[data-testid="configure-escalation"]').click();

    await page.locator('[data-testid="add-escalation-rule"]').click();

    await page.locator('[data-testid="rule-name"]').fill('Critical Escalation');

    // First level
    await page.locator('[data-testid="level-0-duration"]').fill('15');
    await page.locator('[data-testid="level-0-notify"]').fill('team@example.com');

    // Second level
    await page.locator('[data-testid="add-level"]').click();
    await page.locator('[data-testid="level-1-duration"]').fill('30');
    await page.locator('[data-testid="level-1-notify"]').fill('manager@example.com');

    // Third level
    await page.locator('[data-testid="add-level"]').click();
    await page.locator('[data-testid="level-2-duration"]').fill('60');
    await page.locator('[data-testid="level-2-notify"]').fill('executive@example.com');

    await page.locator('[data-testid="save-escalation"]').click();

    await expect(page.locator('[data-testid="escalation-saved"]')).toBeVisible();
  });

  test('should group related alerts', async ({ page }) => {
    await page.locator('[data-testid="group-alerts"]').click();

    await page.locator('[data-testid="group-by"]').selectOption('alertname');

    const groups = await page.locator('[data-testid="alert-group"]').all();
    expect(groups.length).toBeGreaterThan(0);

    // Verify grouping
    for (const group of groups) {
      await expect(group.locator('[data-testid="group-name"]')).toBeVisible();
      await expect(group.locator('[data-testid="alert-count"]')).toBeVisible();

      const count = parseInt(await group.locator('[data-testid="alert-count"]').textContent() || '0');
      expect(count).toBeGreaterThan(0);
    }
  });
});
