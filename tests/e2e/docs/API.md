# E2E Test Utilities API Reference

This document provides a complete reference for all helper functions, page objects, and utilities available in the E2E test suite.

## Table of Contents

- [Helper Functions](#helper-functions)
- [Page Object Models](#page-object-models)
- [Custom Fixtures](#custom-fixtures)
- [Configuration](#configuration)
- [Utilities](#utilities)

## Helper Functions

### Authentication Helpers

**File:** `helpers/auth.js`

#### `loginAsUser(page, email, password)`

Logs in as a regular user.

**Parameters:**
- `page` (Page): Playwright page instance
- `email` (string): User email
- `password` (string): User password

**Returns:** `Promise<void>`

**Example:**
```javascript
const { loginAsUser } = require('../helpers/auth');

test('user dashboard', async ({ page }) => {
  await loginAsUser(page, 'user@example.com', 'password123');
  await expect(page).toHaveURL('/dashboard');
});
```

#### `loginAsAdmin(page)`

Logs in as an admin user using credentials from environment variables.

**Parameters:**
- `page` (Page): Playwright page instance

**Returns:** `Promise<void>`

**Example:**
```javascript
const { loginAsAdmin } = require('../helpers/auth');

test('admin panel', async ({ page }) => {
  await loginAsAdmin(page);
  await expect(page.locator('[data-role="admin"]')).toBeVisible();
});
```

#### `logout(page)`

Logs out the current user.

**Parameters:**
- `page` (Page): Playwright page instance

**Returns:** `Promise<void>`

**Example:**
```javascript
const { logout } = require('../helpers/auth');

test('logout flow', async ({ page }) => {
  await loginAsUser(page, 'user@example.com', 'password');
  await logout(page);
  await expect(page).toHaveURL('/login');
});
```

#### `createAuthState(email, password, statePath)`

Creates and saves authentication state for reuse across tests.

**Parameters:**
- `email` (string): User email
- `password` (string): User password
- `statePath` (string): Path to save auth state (default: 'auth.json')

**Returns:** `Promise<string>` - Path to saved state file

**Example:**
```javascript
const { createAuthState } = require('../helpers/auth');

// In global setup
await createAuthState('user@example.com', 'password', 'user-auth.json');

// Use in tests
test.use({ storageState: 'user-auth.json' });
```

#### `isLoggedIn(page)`

Checks if a user is currently logged in.

**Parameters:**
- `page` (Page): Playwright page instance

**Returns:** `Promise<boolean>`

**Example:**
```javascript
const { isLoggedIn } = require('../helpers/auth');

test('check login state', async ({ page }) => {
  const loggedIn = await isLoggedIn(page);
  expect(loggedIn).toBe(false);
});
```

### API Helpers

**File:** `helpers/api.js`

#### `apiRequest(method, endpoint, data, options)`

Makes an API request with authentication.

**Parameters:**
- `method` (string): HTTP method (GET, POST, PUT, DELETE)
- `endpoint` (string): API endpoint path
- `data` (object): Request body data (optional)
- `options` (object): Additional request options (optional)

**Returns:** `Promise<Response>`

**Example:**
```javascript
const { apiRequest } = require('../helpers/api');

test('create node via API', async ({ page }) => {
  const response = await apiRequest('POST', '/api/nodes', {
    name: 'test-node',
    region: 'us-east-1',
  });

  expect(response.status()).toBe(201);
  const data = await response.json();
  expect(data.name).toBe('test-node');
});
```

#### `createTestNode(nodeData)`

Creates a test node via API.

**Parameters:**
- `nodeData` (object): Node configuration

**Returns:** `Promise<object>` - Created node data

**Example:**
```javascript
const { createTestNode } = require('../helpers/api');

test('node management', async ({ page }) => {
  const node = await createTestNode({
    name: 'test-node-1',
    region: 'us-west-2',
  });

  await page.goto(`/nodes/${node.id}`);
  await expect(page.locator('h1')).toHaveText('test-node-1');
});
```

#### `deleteTestNode(nodeId)`

Deletes a test node via API.

**Parameters:**
- `nodeId` (string): Node ID to delete

**Returns:** `Promise<void>`

**Example:**
```javascript
const { createTestNode, deleteTestNode } = require('../helpers/api');

test.afterEach(async () => {
  await deleteTestNode(testNodeId);
});
```

#### `createTestUser(userData)`

Creates a test user via API.

**Parameters:**
- `userData` (object): User data

**Returns:** `Promise<object>` - Created user data

**Example:**
```javascript
const { createTestUser } = require('../helpers/api');

test('user creation', async ({ page }) => {
  const user = await createTestUser({
    email: 'newuser@example.com',
    password: 'password123',
    role: 'user',
  });

  expect(user.email).toBe('newuser@example.com');
});
```

#### `waitForApiResponse(page, urlPattern, options)`

Waits for a specific API response.

**Parameters:**
- `page` (Page): Playwright page instance
- `urlPattern` (string|RegExp): URL pattern to match
- `options` (object): Wait options (timeout, status code)

**Returns:** `Promise<Response>`

**Example:**
```javascript
const { waitForApiResponse } = require('../helpers/api');

test('data loading', async ({ page }) => {
  await page.goto('/dashboard');

  const response = await waitForApiResponse(page, /api\/stats/);
  const data = await response.json();

  expect(data.nodes).toBeGreaterThan(0);
});
```

### Custom Assertions

**File:** `helpers/assertions.js`

#### `expectVisible(page, selector, options)`

Asserts that an element is visible.

**Parameters:**
- `page` (Page): Playwright page instance
- `selector` (string): Element selector
- `options` (object): Assertion options

**Returns:** `Promise<void>`

**Example:**
```javascript
const { expectVisible } = require('../helpers/assertions');

test('modal visibility', async ({ page }) => {
  await page.click('[data-open-modal]');
  await expectVisible(page, '.modal');
});
```

#### `expectNotVisible(page, selector, options)`

Asserts that an element is not visible.

**Parameters:**
- `page` (Page): Playwright page instance
- `selector` (string): Element selector
- `options` (object): Assertion options

**Returns:** `Promise<void>`

**Example:**
```javascript
const { expectNotVisible } = require('../helpers/assertions');

test('loading spinner', async ({ page }) => {
  await page.waitForLoadState('networkidle');
  await expectNotVisible(page, '.loading-spinner');
});
```

#### `expectFormError(page, fieldSelector, errorMessage)`

Asserts that a form field has a specific error message.

**Parameters:**
- `page` (Page): Playwright page instance
- `fieldSelector` (string): Form field selector
- `errorMessage` (string): Expected error message

**Returns:** `Promise<void>`

**Example:**
```javascript
const { expectFormError } = require('../helpers/assertions');

test('email validation', async ({ page }) => {
  await page.fill('#email', 'invalid-email');
  await page.click('button[type="submit"]');

  await expectFormError(page, '#email', 'Invalid email address');
});
```

#### `expectTableRowCount(page, tableSelector, expectedCount)`

Asserts the number of rows in a table.

**Parameters:**
- `page` (Page): Playwright page instance
- `tableSelector` (string): Table selector
- `expectedCount` (number): Expected number of rows

**Returns:** `Promise<void>`

**Example:**
```javascript
const { expectTableRowCount } = require('../helpers/assertions');

test('node list', async ({ page }) => {
  await page.goto('/nodes');
  await expectTableRowCount(page, '.nodes-table', 5);
});
```

#### `expectToastMessage(page, message, type)`

Asserts that a toast notification appears with specific message.

**Parameters:**
- `page` (Page): Playwright page instance
- `message` (string|RegExp): Expected message
- `type` (string): Toast type (success, error, warning, info)

**Returns:** `Promise<void>`

**Example:**
```javascript
const { expectToastMessage } = require('../helpers/assertions');

test('node creation success', async ({ page }) => {
  await createNode(page, 'new-node');
  await expectToastMessage(page, 'Node created successfully', 'success');
});
```

## Page Object Models

### LoginPage

**File:** `pages/LoginPage.js`

```javascript
class LoginPage {
  constructor(page);

  // Navigation
  async goto();

  // Actions
  async login(email, password, rememberMe = false);
  async loginAsAdmin();
  async forgotPassword(email);

  // Assertions
  async expectLoginSuccess();
  async expectLoginError(message);

  // Queries
  async getErrorMessage();
  async isRememberMeChecked();
}
```

**Example:**
```javascript
const { LoginPage } = require('../pages/LoginPage');

test('login', async ({ page }) => {
  const loginPage = new LoginPage(page);
  await loginPage.goto();
  await loginPage.login('user@example.com', 'password');
  await loginPage.expectLoginSuccess();
});
```

### DashboardPage

**File:** `pages/DashboardPage.js`

```javascript
class DashboardPage {
  constructor(page);

  // Navigation
  async goto();
  async navigateTo(section); // 'home', 'nodes', 'workloads', 'settings'

  // Actions
  async logout();
  async refreshStats();

  // Queries
  async getStats(); // Returns { totalNodes, activeWorkloads, cpuUsage, memoryUsage }
  async getUserName();
  async getNotificationCount();

  // Assertions
  async expectStatsVisible();
  async expectUserLoggedIn(userName);
}
```

**Example:**
```javascript
const { DashboardPage } = require('../pages/DashboardPage');

test('dashboard stats', async ({ page }) => {
  const dashboard = new DashboardPage(page);
  await dashboard.goto();
  await dashboard.expectStatsVisible();

  const stats = await dashboard.getStats();
  expect(stats.totalNodes).toMatch(/\d+/);
});
```

### NodesPage

**File:** `pages/NodesPage.js`

```javascript
class NodesPage {
  constructor(page);

  // Navigation
  async goto();
  async gotoNodeDetails(nodeId);

  // Actions
  async createNode(nodeData);
  async deleteNode(nodeName);
  async searchNodes(query);
  async filterByRegion(region);
  async sortBy(column); // 'name', 'region', 'status', 'created'

  // Queries
  async getNodeCount();
  async getNodeByName(name);
  async getNodeStatus(nodeName);

  // Assertions
  async expectNodeExists(nodeName);
  async expectNodeStatus(nodeName, status);
  async expectNodesTableVisible();
}
```

**Example:**
```javascript
const { NodesPage } = require('../pages/NodesPage');

test('create node', async ({ page }) => {
  const nodesPage = new NodesPage(page);
  await nodesPage.goto();

  await nodesPage.createNode({
    name: 'test-node',
    region: 'us-east-1',
    capacity: { cpu: 4, memory: 8 },
  });

  await nodesPage.expectNodeExists('test-node');
  await nodesPage.expectNodeStatus('test-node', 'active');
});
```

### WorkloadsPage

**File:** `pages/WorkloadsPage.js`

```javascript
class WorkloadsPage {
  constructor(page);

  // Navigation
  async goto();
  async gotoWorkloadDetails(workloadId);

  // Actions
  async deployWorkload(workloadData);
  async stopWorkload(workloadName);
  async deleteWorkload(workloadName);
  async scaleWorkload(workloadName, replicas);

  // Queries
  async getWorkloadCount();
  async getWorkloadStatus(workloadName);
  async getWorkloadLogs(workloadName);

  // Assertions
  async expectWorkloadDeployed(workloadName);
  async expectWorkloadRunning(workloadName);
  async expectWorkloadStopped(workloadName);
}
```

**Example:**
```javascript
const { WorkloadsPage } = require('../pages/WorkloadsPage');

test('deploy workload', async ({ page }) => {
  const workloads = new WorkloadsPage(page);
  await workloads.goto();

  await workloads.deployWorkload({
    name: 'test-app',
    image: 'nginx:latest',
    replicas: 3,
  });

  await workloads.expectWorkloadDeployed('test-app');
  await workloads.expectWorkloadRunning('test-app');
});
```

## Custom Fixtures

**File:** `fixtures/index.js`

### authenticatedPage

Provides a page with user already logged in.

**Example:**
```javascript
const { test } = require('../fixtures');

test('authenticated action', async ({ authenticatedPage }) => {
  await authenticatedPage.goto('/dashboard');
  await expect(authenticatedPage).toHaveURL('/dashboard');
});
```

### adminPage

Provides a page with admin user logged in.

**Example:**
```javascript
const { test } = require('../fixtures');

test('admin action', async ({ adminPage }) => {
  await adminPage.goto('/admin');
  await expect(adminPage.locator('[data-role="admin"]')).toBeVisible();
});
```

### testNode

Provides a pre-created test node.

**Example:**
```javascript
const { test } = require('../fixtures');

test('node management', async ({ page, testNode }) => {
  await page.goto(`/nodes/${testNode.id}`);
  await expect(page.locator('h1')).toHaveText(testNode.name);
});
```

## Configuration

### Environment Variables

**File:** `config/test.env`

```bash
# Application URLs
BASE_URL=http://localhost:3000
API_URL=http://localhost:8080/api

# Test Credentials
TEST_USER_EMAIL=test.user@example.com
TEST_USER_PASSWORD=SecurePass123!
TEST_ADMIN_EMAIL=admin@example.com
TEST_ADMIN_PASSWORD=AdminPass123!

# Database
TEST_DB_HOST=localhost
TEST_DB_PORT=5432
TEST_DB_NAME=novacron_test

# Test Configuration
HEADLESS=true
SLOW_MO=0
TIMEOUT=30000
SCREENSHOT_ON_FAILURE=true
VIDEO_ON_FAILURE=true
```

### Playwright Config

**File:** `playwright.config.js`

Key configuration options:

```javascript
{
  testDir: './tests/e2e/specs',
  timeout: 30000,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 2 : undefined,

  use: {
    baseURL: process.env.BASE_URL,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },

  projects: [
    { name: 'chromium' },
    { name: 'firefox' },
    { name: 'webkit' },
  ],
}
```

## Utilities

### Test Data Factories

**File:** `helpers/factories.js`

#### `createUser(overrides)`

Creates test user data.

**Example:**
```javascript
const { createUser } = require('../helpers/factories');

const user = createUser({ role: 'admin' });
// { email: 'random@example.com', password: 'xyz', role: 'admin', ... }
```

#### `createNode(overrides)`

Creates test node data.

**Example:**
```javascript
const { createNode } = require('../helpers/factories');

const node = createNode({ region: 'us-west-2' });
// { name: 'node-abc123', ip: '192.168.1.1', region: 'us-west-2', ... }
```

### Database Utilities

**File:** `helpers/database.js`

#### `seedTestData()`

Seeds database with test data.

#### `cleanupTestData()`

Removes all test data from database.

#### `resetDatabase()`

Resets database to initial state.

**Example:**
```javascript
const { seedTestData, cleanupTestData } = require('../helpers/database');

test.beforeAll(async () => {
  await seedTestData();
});

test.afterAll(async () => {
  await cleanupTestData();
});
```

---

**Last Updated**: 2025-01-10
**Playwright Version**: 1.40+
