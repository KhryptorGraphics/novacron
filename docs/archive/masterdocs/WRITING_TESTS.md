# Writing E2E Tests Guide

This guide covers best practices and patterns for writing effective E2E tests using Playwright.

## Table of Contents

- [Test Structure](#test-structure)
- [Page Object Model](#page-object-model)
- [Locator Strategies](#locator-strategies)
- [Assertions](#assertions)
- [Test Data Management](#test-data-management)
- [Async Operations](#async-operations)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

## Test Structure

### Basic Test Anatomy

```javascript
const { test, expect } = require('@playwright/test');

test.describe('Feature Name', () => {
  // Runs before each test in this describe block
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  // Runs after each test in this describe block
  test.afterEach(async ({ page, context }) => {
    // Cleanup
    await context.clearCookies();
  });

  test('should perform expected action', async ({ page }) => {
    // Arrange - Set up test data and state
    const username = 'testuser@example.com';
    const password = 'SecurePass123!';

    // Act - Perform the action being tested
    await page.fill('#email', username);
    await page.fill('#password', password);
    await page.click('button[type="submit"]');

    // Assert - Verify expected outcomes
    await expect(page).toHaveURL('/dashboard');
    await expect(page.locator('.welcome-message')).toBeVisible();
  });
});
```

### Nested Describe Blocks

```javascript
test.describe('User Management', () => {
  test.describe('Authentication', () => {
    test.describe('Login', () => {
      test('should login with valid credentials', async ({ page }) => {
        // Test implementation
      });

      test('should reject invalid credentials', async ({ page }) => {
        // Test implementation
      });

      test('should handle network errors gracefully', async ({ page }) => {
        // Test implementation
      });
    });

    test.describe('Logout', () => {
      test.beforeEach(async ({ page }) => {
        // Login before each logout test
        await loginAsTestUser(page);
      });

      test('should logout successfully', async ({ page }) => {
        // Test implementation
      });
    });
  });
});
```

### Test Hooks

```javascript
test.describe('Dashboard', () => {
  // Runs once before all tests
  test.beforeAll(async ({ browser }) => {
    // Setup test database
    await setupTestDatabase();
  });

  // Runs once after all tests
  test.afterAll(async ({ browser }) => {
    // Cleanup test database
    await cleanupTestDatabase();
  });

  // Runs before each test
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
  });

  // Runs after each test
  test.afterEach(async ({ page }) => {
    // Clear localStorage
    await page.evaluate(() => localStorage.clear());
  });
});
```

## Page Object Model

### Creating a Page Object

**pages/LoginPage.js:**

```javascript
class LoginPage {
  constructor(page) {
    this.page = page;

    // Define locators
    this.emailInput = page.locator('#email');
    this.passwordInput = page.locator('#password');
    this.submitButton = page.locator('button[type="submit"]');
    this.errorMessage = page.locator('.error-message');
    this.rememberMeCheckbox = page.locator('#remember-me');
  }

  // Navigation methods
  async goto() {
    await this.page.goto('/login');
  }

  // Action methods
  async login(email, password, rememberMe = false) {
    await this.emailInput.fill(email);
    await this.passwordInput.fill(password);

    if (rememberMe) {
      await this.rememberMeCheckbox.check();
    }

    await this.submitButton.click();
  }

  async loginAsAdmin() {
    await this.login(
      process.env.TEST_ADMIN_EMAIL,
      process.env.TEST_ADMIN_PASSWORD
    );
  }

  // Assertion methods
  async expectLoginSuccess() {
    await expect(this.page).toHaveURL('/dashboard');
  }

  async expectLoginError(message) {
    await expect(this.errorMessage).toBeVisible();
    await expect(this.errorMessage).toHaveText(message);
  }

  // Query methods
  async getErrorMessage() {
    return await this.errorMessage.textContent();
  }

  async isRememberMeChecked() {
    return await this.rememberMeCheckbox.isChecked();
  }
}

module.exports = { LoginPage };
```

### Using Page Objects in Tests

```javascript
const { test, expect } = require('@playwright/test');
const { LoginPage } = require('../pages/LoginPage');

test.describe('Login', () => {
  test('should login successfully', async ({ page }) => {
    const loginPage = new LoginPage(page);

    await loginPage.goto();
    await loginPage.login('user@example.com', 'password123');
    await loginPage.expectLoginSuccess();
  });

  test('should display error for invalid credentials', async ({ page }) => {
    const loginPage = new LoginPage(page);

    await loginPage.goto();
    await loginPage.login('invalid@example.com', 'wrongpass');
    await loginPage.expectLoginError('Invalid email or password');
  });
});
```

### Advanced Page Object Pattern

**pages/DashboardPage.js:**

```javascript
class DashboardPage {
  constructor(page) {
    this.page = page;

    // Component locators
    this.navigation = {
      home: page.locator('nav a[href="/dashboard"]'),
      nodes: page.locator('nav a[href="/nodes"]'),
      workloads: page.locator('nav a[href="/workloads"]'),
      settings: page.locator('nav a[href="/settings"]'),
    };

    this.user = {
      avatar: page.locator('.user-avatar'),
      name: page.locator('.user-name'),
      dropdown: page.locator('.user-dropdown'),
      logoutButton: page.locator('[data-testid="logout"]'),
    };

    this.stats = {
      totalNodes: page.locator('[data-stat="total-nodes"]'),
      activeWorkloads: page.locator('[data-stat="active-workloads"]'),
      cpuUsage: page.locator('[data-stat="cpu-usage"]'),
      memoryUsage: page.locator('[data-stat="memory-usage"]'),
    };
  }

  async goto() {
    await this.page.goto('/dashboard');
    await this.page.waitForLoadState('networkidle');
  }

  async navigateTo(section) {
    const navItem = this.navigation[section];
    if (!navItem) {
      throw new Error(`Unknown section: ${section}`);
    }
    await navItem.click();
  }

  async logout() {
    await this.user.avatar.click();
    await this.user.logoutButton.click();
  }

  async getStats() {
    return {
      totalNodes: await this.stats.totalNodes.textContent(),
      activeWorkloads: await this.stats.activeWorkloads.textContent(),
      cpuUsage: await this.stats.cpuUsage.textContent(),
      memoryUsage: await this.stats.memoryUsage.textContent(),
    };
  }

  async expectStatsVisible() {
    await expect(this.stats.totalNodes).toBeVisible();
    await expect(this.stats.activeWorkloads).toBeVisible();
    await expect(this.stats.cpuUsage).toBeVisible();
    await expect(this.stats.memoryUsage).toBeVisible();
  }
}

module.exports = { DashboardPage };
```

## Locator Strategies

### Priority of Locators

1. **User-facing attributes** (role, text, label)
2. **Test IDs** (data-testid)
3. **CSS selectors** (as last resort)

### Recommended Locators

```javascript
// 1. By Role (Best - accessibility-focused)
await page.getByRole('button', { name: 'Submit' }).click();
await page.getByRole('textbox', { name: 'Email' }).fill('user@example.com');
await page.getByRole('link', { name: 'Home' }).click();

// 2. By Label (Good for form inputs)
await page.getByLabel('Email address').fill('user@example.com');
await page.getByLabel('Password').fill('password123');

// 3. By Placeholder
await page.getByPlaceholder('Enter your email').fill('user@example.com');

// 4. By Text
await page.getByText('Welcome back!').waitFor();
await page.getByText(/sign in/i).click(); // Case-insensitive regex

// 5. By Test ID (Recommended for dynamic content)
await page.getByTestId('submit-button').click();
await page.getByTestId('user-profile').click();

// 6. By Alt Text (for images)
await page.getByAltText('Company logo').click();

// 7. By Title
await page.getByTitle('Close dialog').click();
```

### CSS Selectors (Use Sparingly)

```javascript
// Avoid when possible, but acceptable when necessary
await page.locator('.btn-primary').click();
await page.locator('#user-menu').click();
await page.locator('[data-user-id="123"]').click();

// Chaining locators
await page.locator('.modal').locator('.close-button').click();

// Filtering
await page.locator('.item').filter({ hasText: 'Active' }).click();

// nth element
await page.locator('.item').nth(0).click(); // First item
await page.locator('.item').last().click(); // Last item
```

### XPath (Avoid)

```javascript
// Use XPath only when absolutely necessary
await page.locator('xpath=//button[contains(text(), "Submit")]').click();

// Playwright provides better alternatives
await page.getByRole('button', { name: /submit/i }).click();
```

## Assertions

### Basic Assertions

```javascript
// Element visibility
await expect(page.locator('.welcome')).toBeVisible();
await expect(page.locator('.loading')).toBeHidden();

// Element state
await expect(page.locator('button')).toBeEnabled();
await expect(page.locator('button')).toBeDisabled();
await expect(page.locator('input')).toBeFocused();
await expect(page.locator('checkbox')).toBeChecked();

// Text content
await expect(page.locator('h1')).toHaveText('Welcome');
await expect(page.locator('h1')).toContainText('Wel');
await expect(page.locator('.items')).toHaveText(['Item 1', 'Item 2']);

// Attributes
await expect(page.locator('a')).toHaveAttribute('href', '/home');
await expect(page.locator('img')).toHaveAttribute('src', /logo\.png$/);

// CSS Classes
await expect(page.locator('div')).toHaveClass('active');
await expect(page.locator('div')).toHaveClass(/active/);

// Count
await expect(page.locator('.item')).toHaveCount(5);

// Input values
await expect(page.locator('input')).toHaveValue('example@email.com');
await expect(page.locator('input')).toHaveValue(/example/);

// URL
await expect(page).toHaveURL('/dashboard');
await expect(page).toHaveURL(/dashboard/);

// Title
await expect(page).toHaveTitle('Dashboard - NovaCron');
```

### Soft Assertions

```javascript
test('should validate all fields', async ({ page }) => {
  // Continue test even if assertions fail
  await expect.soft(page.locator('.name')).toHaveText('John Doe');
  await expect.soft(page.locator('.email')).toHaveText('john@example.com');
  await expect.soft(page.locator('.role')).toHaveText('Admin');

  // All failures will be reported at the end
});
```

### Custom Assertions

```javascript
// helpers/assertions.js
async function expectElementToBeVisibleAndEnabled(page, selector) {
  await expect(page.locator(selector)).toBeVisible();
  await expect(page.locator(selector)).toBeEnabled();
}

async function expectFormValidation(page, selector, errorMessage) {
  const input = page.locator(selector);
  await expect(input).toHaveAttribute('aria-invalid', 'true');
  await expect(page.locator(`${selector} ~ .error`)).toHaveText(errorMessage);
}

module.exports = {
  expectElementToBeVisibleAndEnabled,
  expectFormValidation,
};

// Usage in tests
const { expectFormValidation } = require('../helpers/assertions');

test('should validate email', async ({ page }) => {
  await page.fill('#email', 'invalid');
  await page.click('#submit');
  await expectFormValidation(page, '#email', 'Invalid email address');
});
```

### Negative Assertions

```javascript
// Use .not for negative assertions
await expect(page.locator('.error')).not.toBeVisible();
await expect(page.locator('button')).not.toBeDisabled();
await expect(page).not.toHaveURL('/login');
```

## Test Data Management

### Using Fixtures

**fixtures/users.json:**

```json
{
  "validUser": {
    "email": "test.user@example.com",
    "password": "SecurePass123!",
    "name": "Test User",
    "role": "user"
  },
  "adminUser": {
    "email": "admin@example.com",
    "password": "AdminPass123!",
    "name": "Admin User",
    "role": "admin"
  },
  "invalidUser": {
    "email": "invalid@example.com",
    "password": "wrongpassword"
  }
}
```

**Using fixtures in tests:**

```javascript
const { test, expect } = require('@playwright/test');
const users = require('../fixtures/users.json');

test('should login with valid user', async ({ page }) => {
  await page.goto('/login');
  await page.fill('#email', users.validUser.email);
  await page.fill('#password', users.validUser.password);
  await page.click('button[type="submit"]');

  await expect(page).toHaveURL('/dashboard');
});
```

### Test Data Factories

**helpers/factories.js:**

```javascript
const { faker } = require('@faker-js/faker');

function createUser(overrides = {}) {
  return {
    email: faker.internet.email(),
    password: faker.internet.password({ length: 12 }),
    firstName: faker.person.firstName(),
    lastName: faker.person.lastName(),
    role: 'user',
    ...overrides,
  };
}

function createNode(overrides = {}) {
  return {
    name: `node-${faker.string.alphanumeric(8)}`,
    ip: faker.internet.ipv4(),
    port: faker.number.int({ min: 3000, max: 9999 }),
    region: faker.location.city(),
    capacity: {
      cpu: faker.number.int({ min: 1, max: 16 }),
      memory: faker.number.int({ min: 2, max: 64 }),
      storage: faker.number.int({ min: 10, max: 1000 }),
    },
    ...overrides,
  };
}

module.exports = { createUser, createNode };
```

**Using factories:**

```javascript
const { createUser, createNode } = require('../helpers/factories');

test('should create user', async ({ page }) => {
  const user = createUser({ role: 'admin' });

  await page.goto('/users/new');
  await page.fill('#email', user.email);
  await page.fill('#firstName', user.firstName);
  await page.fill('#lastName', user.lastName);
  await page.selectOption('#role', user.role);
  await page.click('button[type="submit"]');

  await expect(page.locator('.success')).toContainText(user.email);
});
```

### Database Fixtures

**helpers/database.js:**

```javascript
const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

async function seedTestUser(userData) {
  return await prisma.user.create({
    data: {
      email: userData.email,
      password: await hashPassword(userData.password),
      name: userData.name,
      role: userData.role,
    },
  });
}

async function cleanupTestData() {
  await prisma.node.deleteMany({
    where: { name: { startsWith: 'test-' } },
  });
  await prisma.user.deleteMany({
    where: { email: { endsWith: '@test.example.com' } },
  });
}

module.exports = { seedTestUser, cleanupTestData };
```

## Async Operations

### Waiting Strategies

```javascript
// Auto-waiting (preferred)
await page.click('button'); // Waits for button to be actionable

// Wait for element
await page.waitForSelector('.modal', { state: 'visible' });
await page.waitForSelector('.loading', { state: 'hidden' });

// Wait for navigation
await Promise.all([
  page.waitForNavigation(),
  page.click('a[href="/next-page"]'),
]);

// Wait for load state
await page.waitForLoadState('load');
await page.waitForLoadState('domcontentloaded');
await page.waitForLoadState('networkidle');

// Wait for URL
await page.waitForURL('/dashboard');
await page.waitForURL(/dashboard/);

// Wait for response
const response = await page.waitForResponse(
  (response) => response.url().includes('/api/users') && response.status() === 200
);

// Wait for request
await page.waitForRequest((request) =>
  request.url().includes('/api/auth') && request.method() === 'POST'
);

// Wait for function
await page.waitForFunction(() => document.title === 'Dashboard');
await page.waitForFunction(() => window.dataLoaded === true);
```

### Handling Race Conditions

```javascript
// Good - wait for specific condition
test('should load user data', async ({ page }) => {
  await page.goto('/users/123');

  // Wait for API call to complete
  await page.waitForResponse((resp) =>
    resp.url().includes('/api/users/123')
  );

  await expect(page.locator('.user-name')).toBeVisible();
});

// Better - use auto-waiting
test('should load user data', async ({ page }) => {
  await page.goto('/users/123');

  // Playwright auto-waits for element to be visible
  await expect(page.locator('.user-name')).toBeVisible();
});
```

### Timeouts

```javascript
// Test timeout
test('slow operation', async ({ page }) => {
  test.setTimeout(60000); // 60 seconds
  await performSlowOperation(page);
});

// Assertion timeout
await expect(page.locator('.data')).toBeVisible({ timeout: 10000 });

// Action timeout
await page.click('button', { timeout: 5000 });

// Navigation timeout
await page.goto('/slow-page', { timeout: 30000 });
```

## Error Handling

### Try-Catch for Expected Errors

```javascript
test('should handle network error gracefully', async ({ page }) => {
  await page.route('**/api/data', (route) => route.abort('failed'));

  await page.goto('/dashboard');

  try {
    await page.waitForResponse((resp) => resp.url().includes('/api/data'), {
      timeout: 5000
    });
  } catch (error) {
    // Expected - verify error message is shown
    await expect(page.locator('.error-message')).toBeVisible();
    await expect(page.locator('.error-message')).toContainText('Failed to load data');
  }
});
```

### Test Steps for Better Error Context

```javascript
test('complex workflow', async ({ page }) => {
  await test.step('Navigate to login', async () => {
    await page.goto('/login');
  });

  await test.step('Enter credentials', async () => {
    await page.fill('#email', 'user@example.com');
    await page.fill('#password', 'password123');
  });

  await test.step('Submit form', async () => {
    await page.click('button[type="submit"]');
  });

  await test.step('Verify dashboard loaded', async () => {
    await expect(page).toHaveURL('/dashboard');
    await expect(page.locator('.welcome')).toBeVisible();
  });
});
```

### Retry Logic

```javascript
// Retry specific assertions
async function waitForElement(page, selector, retries = 3) {
  for (let i = 0; i < retries; i++) {
    try {
      await expect(page.locator(selector)).toBeVisible({ timeout: 5000 });
      return;
    } catch (error) {
      if (i === retries - 1) throw error;
      await page.reload();
    }
  }
}

test('flaky element test', async ({ page }) => {
  await page.goto('/dashboard');
  await waitForElement(page, '.data-table');
});
```

## Best Practices

### 1. Descriptive Test Names

```javascript
// Good
test('should display validation error when email is invalid', async ({ page }) => {});
test('should redirect to dashboard after successful login', async ({ page }) => {});
test('should disable submit button while request is in progress', async ({ page }) => {});

// Avoid
test('login test', async ({ page }) => {});
test('test validation', async ({ page }) => {});
```

### 2. One Assertion Per Test (Usually)

```javascript
// Good - focused test
test('should display error for invalid email', async ({ page }) => {
  await page.goto('/login');
  await page.fill('#email', 'invalid');
  await page.click('button[type="submit"]');

  await expect(page.locator('.error')).toContainText('Invalid email');
});

// Acceptable - related assertions
test('should validate login form', async ({ page }) => {
  await page.goto('/login');
  await page.click('button[type="submit"]');

  await expect(page.locator('#email-error')).toContainText('Email is required');
  await expect(page.locator('#password-error')).toContainText('Password is required');
});
```

### 3. Test Independence

```javascript
// Good - independent tests
test.describe('User Management', () => {
  test.beforeEach(async ({ page }) => {
    await seedTestUser();
    await page.goto('/users');
  });

  test.afterEach(async () => {
    await cleanupTestUser();
  });

  test('should display user list', async ({ page }) => {
    // Test implementation
  });
});

// Avoid - dependent tests
test('create user', async ({ page }) => {
  // Creates user with ID 123
});

test('edit user', async ({ page }) => {
  // Assumes user 123 exists from previous test - BAD!
});
```

### 4. Use Data Attributes for Testing

**HTML:**

```html
<button
  class="btn btn-primary"
  data-testid="submit-button"
  type="submit"
>
  Submit
</button>

<div
  class="user-card"
  data-testid="user-123"
  data-user-role="admin"
>
  User Name
</div>
```

**Tests:**

```javascript
// Prefer data-testid
await page.getByTestId('submit-button').click();

// Or use data attributes for filtering
await page.locator('[data-user-role="admin"]').click();
```

### 5. Avoid Hard-Coded Waits

```javascript
// Bad
await page.waitForTimeout(3000);
await page.click('button');

// Good - use auto-waiting
await page.click('button');

// Good - wait for specific condition
await page.waitForSelector('.modal', { state: 'visible' });
await page.click('.modal button');
```

### 6. Clean Up Test Data

```javascript
test.afterEach(async () => {
  // Clean up created resources
  await deleteTestNodes();
  await deleteTestUsers();
  await clearTestDatabase();
});

test.afterAll(async () => {
  // Final cleanup
  await resetDatabase();
});
```

### 7. Use POM for Reusability

```javascript
// Good - reusable page object
const loginPage = new LoginPage(page);
await loginPage.login(user.email, user.password);

// Avoid - repeated code
await page.fill('#email', user.email);
await page.fill('#password', user.password);
await page.click('button[type="submit"]');
```

### 8. Handle Dynamic Content

```javascript
// Wait for dynamic content
await page.waitForSelector('[data-testid="data-loaded"]');

// Or use auto-waiting with explicit expect
await expect(page.locator('.data')).toBeVisible();

// For lists with unknown count
const items = await page.locator('.item').all();
expect(items.length).toBeGreaterThan(0);
```

---

**Last Updated**: 2025-01-10
**Playwright Version**: 1.40+
