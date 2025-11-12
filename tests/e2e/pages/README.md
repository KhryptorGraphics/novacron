# NovaCron Page Object Models (POMs)

Comprehensive Page Object Models for NovaCron's E2E testing using Playwright.

## 📁 Structure

```
tests/e2e/pages/
├── base-page.ts                    # Base POM with common utilities
├── index.ts                        # Centralized exports
├── types.ts                        # Shared TypeScript types
├── auth/                          # Authentication pages
│   ├── login-page.ts
│   ├── register-page.ts
│   └── password-reset-page.ts
├── vms/                           # VM management pages
│   ├── vm-list-page.ts
│   ├── vm-create-page.ts
│   ├── vm-details-page.ts
│   └── vm-console-page.ts
├── cluster/                       # Cluster management pages
│   ├── cluster-overview-page.ts
│   ├── node-management-page.ts
│   └── federation-page.ts
├── monitoring/                    # Monitoring pages
│   ├── dashboard-page.ts
│   ├── metrics-page.ts
│   └── alerts-page.ts
└── migration/                     # Migration pages
    ├── migration-wizard-page.ts
    └── migration-status-page.ts
```

## 🎯 Key Features

### Base Page (`base-page.ts`)

The foundation for all POMs with common utilities:

- **Navigation**: `goto()`, `goBack()`, `goForward()`, `reload()`
- **Waiting**: `waitForLoad()`, `waitForSelector()`, `waitForTestId()`, `waitForText()`
- **Interactions**: `clickByTestId()`, `fillByTestId()`, `selectByTestId()`
- **Network**: `waitForApiResponse()`, `interceptRequest()`, `mockApiResponse()`
- **Screenshots**: `takeScreenshot()`
- **Toasts**: `waitForToast()`, `assertSuccessToast()`, `assertErrorToast()`

### Authentication Pages

#### LoginPage
```typescript
import { LoginPage } from './pages';

const loginPage = new LoginPage(page);
await loginPage.navigate();
await loginPage.login({
  email: 'user@example.com',
  password: 'password123',
  rememberMe: true
});
// or
await loginPage.loginAndWaitForDashboard(credentials);
```

#### RegisterPage
```typescript
import { RegisterPage } from './pages';

const registerPage = new RegisterPage(page);
await registerPage.navigate();
await registerPage.completeRegistration({
  email: 'newuser@example.com',
  password: 'SecurePass123!',
  confirmPassword: 'SecurePass123!',
  firstName: 'John',
  lastName: 'Doe',
  agreeToTerms: true
});
```

#### PasswordResetPage
```typescript
import { PasswordResetPage } from './pages';

const resetPage = new PasswordResetPage(page);
await resetPage.navigate();
await resetPage.completePasswordReset(
  'user@example.com',
  '123456',
  'NewPassword123!',
  'NewPassword123!'
);
```

### VM Management Pages

#### VMListPage
```typescript
import { VMListPage } from './pages';

const vmListPage = new VMListPage(page);
await vmListPage.navigate();

// Filter and search
await vmListPage.applyFilters({
  status: 'running',
  cluster: 'cluster-1',
  search: 'web-server'
});

// VM operations
await vmListPage.startVM('vm-001');
await vmListPage.stopVM('vm-002');
await vmListPage.deleteVM('vm-003');

// Bulk operations
await vmListPage.selectVMs(['vm-001', 'vm-002']);
await vmListPage.performBulkAction('restart');
```

#### VMCreatePage
```typescript
import { VMCreatePage } from './pages';

const createPage = new VMCreatePage(page);
await createPage.navigate();

await createPage.createVMAndWaitForCompletion({
  name: 'web-server-01',
  cluster: 'cluster-1',
  cpu: 4,
  memory: 8,
  disk: 100,
  networkInterface: 'eth0',
  tags: ['production', 'web'],
  autoStart: true
});
```

#### VMDetailsPage
```typescript
import { VMDetailsPage } from './pages';

const detailsPage = new VMDetailsPage(page);
await detailsPage.navigate('vm-001');

// View metrics
const cpuUsage = await detailsPage.getCPUUsage();
const memoryUsage = await detailsPage.getMemoryUsage();

// Perform operations
await detailsPage.restartVM();
await detailsPage.createSnapshot('before-update');

// Add resources
await detailsPage.addDisk(50);
await detailsPage.addNetworkInterface('eth1');
```

#### VMConsolePage
```typescript
import { VMConsolePage } from './pages';

const consolePage = new VMConsolePage(page);
await consolePage.navigate('vm-001');
await consolePage.connect('vnc');

// Interact with console
await consolePage.typeInConsole('ls -la\n');
await consolePage.sendCtrlAltDel();
await consolePage.takeConsoleScreenshot();
```

### Cluster Management Pages

#### ClusterOverviewPage
```typescript
import { ClusterOverviewPage } from './pages';

const clusterPage = new ClusterOverviewPage(page);
await clusterPage.navigate();

// Get metrics
const health = await clusterPage.getClusterHealth();
const cpuUtil = await clusterPage.getCPUUtilization();
const nodeCount = await clusterPage.getTotalNodes();

// Set time range
await clusterPage.setChartTimeRange('24h');
```

#### NodeManagementPage
```typescript
import { NodeManagementPage } from './pages';

const nodePage = new NodeManagementPage(page);
await nodePage.navigate();

// Node operations
await nodePage.drainNode('node-01');
await nodePage.cordonNode('node-02');
await nodePage.uncordonNode('node-02');

// Bulk operations
await nodePage.selectNodes(['node-01', 'node-02']);
await nodePage.performBulkAction('disable');
```

#### FederationPage
```typescript
import { FederationPage } from './pages';

const federationPage = new FederationPage(page);
await federationPage.navigate();

// Add federation
await federationPage.addFederation({
  clusterName: 'remote-cluster-1',
  endpoint: 'https://cluster1.example.com',
  authToken: 'token123',
  enableWorkloadSharing: true
});

// Manage workloads
await federationPage.shareWorkload('workload-1', 'remote-cluster-1');
```

### Monitoring Pages

#### MonitoringDashboardPage
```typescript
import { MonitoringDashboardPage } from './pages';

const dashboardPage = new MonitoringDashboardPage(page);
await dashboardPage.navigate();

// Set time range
await dashboardPage.setTimeRange('6h');

// Get metrics
const avgCPU = await dashboardPage.getAvgCPU();
const criticalAlerts = await dashboardPage.getCriticalAlerts();

// Customize dashboard
await dashboardPage.addWidget('cpu-chart');
await dashboardPage.saveLayout();
```

#### MetricsPage
```typescript
import { MetricsPage } from './pages';

const metricsPage = new MetricsPage(page);
await metricsPage.navigate();

// View specific metric
await metricsPage.selectMetricType('cpu');
await metricsPage.filterByNode('node-01');

// Compare metrics
await metricsPage.toggleCompareMode(true);
await metricsPage.addComparison('node-02');

// Export data
await metricsPage.exportMetrics('csv');
```

#### AlertsPage
```typescript
import { AlertsPage } from './pages';

const alertsPage = new AlertsPage(page);
await alertsPage.navigate();

// Create alert rule
await alertsPage.createRule({
  name: 'High CPU Alert',
  metric: 'cpu_usage',
  condition: 'greater_than',
  threshold: 90,
  duration: 300,
  severity: 'critical',
  notificationChannels: ['email', 'slack']
});

// Manage alerts
await alertsPage.acknowledgeAlert('alert-123');
await alertsPage.resolveAlert('alert-456', 'Fixed by restart');
```

### Migration Pages

#### MigrationWizardPage
```typescript
import { MigrationWizardPage } from './pages';

const wizardPage = new MigrationWizardPage(page);
await wizardPage.navigate();

await wizardPage.completeMigration({
  sourceVM: 'vm-001',
  targetCluster: 'cluster-2',
  targetNode: 'node-05',
  migrationType: 'live',
  priority: 'high',
  bandwidthLimit: 1000
});
```

#### MigrationStatusPage
```typescript
import { MigrationStatusPage } from './pages';

const statusPage = new MigrationStatusPage(page);
await statusPage.navigate();

// Monitor migrations
await statusPage.applyFilters({ status: 'in_progress' });
const progress = await statusPage.getProgressPercentage('migration-123');

// Control migrations
await statusPage.pauseMigration('migration-123');
await statusPage.resumeMigration('migration-123');

// Wait for completion
await statusPage.waitForMigrationComplete('migration-123', 600000);
```

## 🔧 Best Practices

### 1. Use Test IDs
All POMs use `data-testid` attributes for reliable element selection:
```typescript
// Good
await page.getByTestId('login-submit-button').click();

// Avoid
await page.locator('.btn-primary').click();
```

### 2. Method Chaining
Most methods return `this` for fluent API:
```typescript
await loginPage
  .navigate()
  .enterEmail('user@example.com')
  .enterPassword('password')
  .clickLogin();
```

### 3. Wait Strategies
Built-in waiting for stability:
```typescript
// Waits for loading to complete
await vmListPage.refresh();

// Waits for toast notification
await vmListPage.deleteVM('vm-001');
```

### 4. Error Handling
POMs include error assertions:
```typescript
// Check for validation errors
const error = await loginPage.getErrorMessage();
expect(error).toContain('Invalid credentials');

// Verify success
await loginPage.assertSuccessToast('Login successful');
```

### 5. TypeScript Types
Use provided interfaces for type safety:
```typescript
import { VMConfiguration } from './pages';

const vmConfig: VMConfiguration = {
  name: 'test-vm',
  cluster: 'cluster-1',
  cpu: 2,
  memory: 4,
  disk: 50
};
```

## 📝 Example Test

```typescript
import { test, expect } from '@playwright/test';
import { LoginPage, VMListPage, VMCreatePage } from './pages';

test.describe('VM Management', () => {
  test('create and verify new VM', async ({ page }) => {
    // Login
    const loginPage = new LoginPage(page);
    await loginPage.navigate();
    await loginPage.loginAndWaitForDashboard({
      email: 'admin@example.com',
      password: 'admin123'
    });

    // Navigate to VMs
    const vmListPage = new VMListPage(page);
    await vmListPage.navigate();

    // Create new VM
    const createPage = new VMCreatePage(page);
    await vmListPage.clickCreateVM();
    await createPage.createVMAndWaitForCompletion({
      name: 'test-vm-001',
      cluster: 'default',
      cpu: 2,
      memory: 4,
      disk: 50,
      tags: ['test']
    });

    // Verify VM appears in list
    await vmListPage.navigate();
    const exists = await vmListPage.vmExists('test-vm-001');
    expect(exists).toBeTruthy();

    // Verify VM status
    const status = await vmListPage.getVMStatus('test-vm-001');
    expect(status).toBe('running');
  });
});
```

## 🔄 Updating POMs

When adding new features:

1. Update the appropriate page object
2. Add TypeScript types to `types.ts`
3. Export from `index.ts`
4. Update this README with examples
5. Add corresponding tests

## 🚀 Tips

- **Parallel Execution**: POMs are designed for parallel test execution
- **Reusability**: Common patterns are in `BasePage`
- **Maintainability**: Change selectors in one place
- **Readability**: Tests read like user workflows
- **Type Safety**: Full TypeScript support

## 📚 Additional Resources

- [Playwright Documentation](https://playwright.dev)
- [Page Object Model Pattern](https://playwright.dev/docs/pom)
- [Best Practices](https://playwright.dev/docs/best-practices)
