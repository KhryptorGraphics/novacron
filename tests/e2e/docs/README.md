# E2E Testing Documentation

## Overview

This directory contains end-to-end (E2E) tests for the NovaCron platform using [Playwright](https://playwright.dev/). Our E2E test suite validates critical user workflows across multiple browsers and ensures the platform functions correctly from a user's perspective.

### Why Playwright?

We migrated from Puppeteer to Playwright for:
- **Multi-browser support**: Test across Chromium, Firefox, and WebKit
- **Better API**: Modern, intuitive testing API with auto-waiting
- **Built-in test runner**: Integrated test runner with parallelization
- **Advanced features**: Network interception, mobile emulation, trace viewer
- **Better CI/CD**: First-class support for continuous integration

### Test Coverage

Our E2E tests cover:
- **Authentication**: Login, logout, registration flows
- **Dashboard**: User dashboard and navigation
- **Node Management**: Edge node creation, monitoring, and lifecycle
- **Workload Deployment**: Container deployment and orchestration
- **Data Synchronization**: Cross-node data sync verification
- **Performance**: Load times and responsiveness

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Supported OS: Linux, macOS, or Windows
- Recommended: 8GB RAM minimum

### Installation

```bash
# Install dependencies
npm install

# Install Playwright browsers
npx playwright install

# Install system dependencies (Linux only)
npx playwright install-deps
```

### Running Tests

```bash
# Run all E2E tests
npm run test:e2e

# Run tests in headed mode (see browser)
npm run test:e2e:headed

# Run specific test file
npx playwright test tests/e2e/specs/auth.spec.js

# Run tests in debug mode
npm run test:e2e:debug

# Run tests for specific browser
npx playwright test --project=chromium
npx playwright test --project=firefox
npx playwright test --project=webkit
```

### Viewing Test Results

```bash
# Open HTML test report
npx playwright show-report

# View trace for failed tests
npx playwright show-trace trace.zip
```

## Project Structure

```
tests/e2e/
├── docs/                    # Documentation (you are here)
│   ├── README.md           # This file
│   ├── SETUP.md            # Setup guide
│   ├── WRITING_TESTS.md    # Test writing guide
│   ├── CI_CD.md            # CI/CD documentation
│   ├── MIGRATION_GUIDE.md  # Puppeteer to Playwright migration
│   └── API.md              # Test utilities API reference
├── specs/                   # Test specifications
│   ├── auth.spec.js        # Authentication tests
│   ├── dashboard.spec.js   # Dashboard tests
│   ├── nodes.spec.js       # Node management tests
│   └── workloads.spec.js   # Workload deployment tests
├── pages/                   # Page Object Models
│   ├── LoginPage.js        # Login page object
│   ├── DashboardPage.js    # Dashboard page object
│   └── NodesPage.js        # Nodes page object
├── fixtures/                # Test fixtures and data
│   ├── users.json          # Test user data
│   └── nodes.json          # Test node configurations
├── helpers/                 # Helper utilities
│   ├── auth.js             # Authentication helpers
│   ├── api.js              # API helpers
│   └── assertions.js       # Custom assertions
├── config/                  # Configuration files
│   ├── playwright.config.js # Playwright configuration
│   └── test.env.example    # Environment variables template
└── reports/                 # Test reports (gitignored)
```

## Running Specific Test Suites

### Authentication Tests

```bash
# All auth tests
npx playwright test auth

# Specific test
npx playwright test auth -g "should login successfully"
```

### Dashboard Tests

```bash
npx playwright test dashboard
```

### Node Management Tests

```bash
npx playwright test nodes
```

### Workload Tests

```bash
npx playwright test workloads
```

## Debugging Tests

### Visual Debugging

```bash
# Run in headed mode with slow motion
npx playwright test --headed --slow-mo=1000

# Debug specific test
npx playwright test --debug auth.spec.js
```

### Using Playwright Inspector

```bash
# Open inspector
npx playwright test --debug

# The inspector allows you to:
# - Step through test execution
# - Inspect page elements
# - Record new tests
# - View console logs
```

### Trace Viewer

```bash
# Tests automatically generate traces on failure
# View trace after test run
npx playwright show-trace test-results/path-to-trace.zip

# Force trace generation for all tests
npx playwright test --trace on
```

### Screenshots and Videos

```bash
# Screenshots are automatically captured on failure
# Videos are recorded based on playwright.config.js settings

# View artifacts in test-results/ directory
ls -la test-results/
```

## Writing New Tests

### Basic Test Structure

```javascript
const { test, expect } = require('@playwright/test');
const { LoginPage } = require('../pages/LoginPage');

test.describe('Feature Name', () => {
  test.beforeEach(async ({ page }) => {
    // Setup before each test
    await page.goto('/');
  });

  test('should perform action', async ({ page }) => {
    // Arrange
    const loginPage = new LoginPage(page);

    // Act
    await loginPage.login('user@example.com', 'password');

    // Assert
    await expect(page).toHaveURL('/dashboard');
  });
});
```

### Using Page Objects

```javascript
// Use Page Object Model for maintainability
const { DashboardPage } = require('../pages/DashboardPage');

test('should display user info', async ({ page }) => {
  const dashboard = new DashboardPage(page);
  await dashboard.goto();

  const userName = await dashboard.getUserName();
  expect(userName).toBe('John Doe');
});
```

See [WRITING_TESTS.md](./WRITING_TESTS.md) for detailed guidance.

## Best Practices

### 1. Use Page Object Model

Encapsulate page interactions in Page Objects for maintainability:

```javascript
// Good
const loginPage = new LoginPage(page);
await loginPage.login(email, password);

// Avoid
await page.fill('#email', email);
await page.fill('#password', password);
await page.click('button[type="submit"]');
```

### 2. Leverage Auto-Waiting

Playwright auto-waits for elements to be actionable:

```javascript
// Playwright automatically waits
await page.click('button'); // Waits for button to be visible and enabled

// No need for manual waits (usually)
// await page.waitForSelector('button'); // Not needed
```

### 3. Use Meaningful Test Data

```javascript
// Good - clear intent
const testUser = {
  email: 'test.user@example.com',
  password: 'SecurePass123!',
  role: 'admin'
};

// Avoid - unclear
const user = { e: 'a@b.com', p: '123' };
```

### 4. Isolate Tests

Each test should be independent:

```javascript
test.beforeEach(async ({ page }) => {
  // Reset state before each test
  await page.goto('/');
  await cleanupTestData();
});

test.afterEach(async ({ page }) => {
  // Cleanup after test
  await logout(page);
});
```

### 5. Use Descriptive Test Names

```javascript
// Good
test('should display error message when login fails with invalid credentials', async ({ page }) => {
  // ...
});

// Avoid
test('login test', async ({ page }) => {
  // ...
});
```

### 6. Group Related Tests

```javascript
test.describe('User Authentication', () => {
  test.describe('Login', () => {
    test('should login with valid credentials', async ({ page }) => {});
    test('should reject invalid credentials', async ({ page }) => {});
  });

  test.describe('Logout', () => {
    test('should logout successfully', async ({ page }) => {});
  });
});
```

### 7. Handle Flaky Tests

```javascript
// Retry flaky tests
test('potentially flaky test', async ({ page }) => {
  // Test code
});

// Configure retries in playwright.config.js
// retries: process.env.CI ? 2 : 0
```

### 8. Use Soft Assertions for Multiple Checks

```javascript
test('should display all user details', async ({ page }) => {
  // Continue test even if assertions fail
  await expect.soft(page.locator('.name')).toHaveText('John Doe');
  await expect.soft(page.locator('.email')).toHaveText('john@example.com');
  await expect.soft(page.locator('.role')).toHaveText('Admin');

  // Report all failures at end of test
});
```

## Performance Tips

### 1. Run Tests in Parallel

```bash
# Tests run in parallel by default
# Configure workers in playwright.config.js
npx playwright test --workers=4
```

### 2. Shard Tests in CI

```bash
# Split tests across multiple machines
npx playwright test --shard=1/3
npx playwright test --shard=2/3
npx playwright test --shard=3/3
```

### 3. Reuse Authentication State

```javascript
// Global setup to save auth state
// tests/e2e/config/global-setup.js
const { chromium } = require('@playwright/test');

module.exports = async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();

  await page.goto('http://localhost:3000/login');
  await page.fill('#email', 'test@example.com');
  await page.fill('#password', 'password');
  await page.click('button[type="submit"]');

  // Save storage state
  await page.context().storageState({ path: 'auth.json' });
  await browser.close();
};

// Use in tests
test.use({ storageState: 'auth.json' });
```

### 4. Skip Unnecessary Waits

```javascript
// Good - rely on auto-waiting
await page.click('button');

// Avoid - unnecessary explicit wait
await page.waitForTimeout(3000);
await page.click('button');
```

## Troubleshooting

### Common Issues

#### Tests Fail Locally but Pass in CI

```bash
# Ensure consistent environment
# 1. Check Node.js version matches CI
node --version

# 2. Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install

# 3. Update browsers
npx playwright install
```

#### Timeouts

```javascript
// Increase timeout for slow operations
test('slow operation', async ({ page }) => {
  test.setTimeout(60000); // 60 seconds

  await page.goto('http://slow-page.com');
});
```

#### Element Not Found

```javascript
// Use debugging tools
await page.pause(); // Opens inspector

// Check element visibility
await expect(page.locator('button')).toBeVisible();

// Wait for specific state
await page.waitForLoadState('networkidle');
```

## CI/CD Integration

Tests run automatically on:
- Pull requests
- Pushes to main branch
- Scheduled nightly runs

See [CI_CD.md](./CI_CD.md) for detailed CI/CD documentation.

## Additional Resources

- [Playwright Documentation](https://playwright.dev/)
- [Setup Guide](./SETUP.md)
- [Writing Tests Guide](./WRITING_TESTS.md)
- [CI/CD Guide](./CI_CD.md)
- [Migration Guide](./MIGRATION_GUIDE.md)
- [API Reference](./API.md)

## Support

- **Issues**: Create an issue in the repository
- **Questions**: Ask in team Slack #testing channel
- **Documentation**: Update these docs when you find gaps

---

**Last Updated**: 2025-01-10
**Playwright Version**: 1.40+
**Node.js Version**: 18+
