# E2E Testing Best Practices for NovaCron

## Table of Contents

1. [General Principles](#general-principles)
2. [Test Design](#test-design)
3. [Locator Strategies](#locator-strategies)
4. [Page Object Model](#page-object-model)
5. [Test Data Management](#test-data-management)
6. [Assertions](#assertions)
7. [Error Handling](#error-handling)
8. [Performance Optimization](#performance-optimization)
9. [Debugging](#debugging)
10. [Accessibility Testing](#accessibility-testing)

---

## General Principles

### 1. Write Independent Tests

Each test should be able to run in isolation without depending on other tests.

```typescript
// ❌ Bad - Tests depend on execution order
test('create VM', async ({ page }) => {
  // Creates VM with ID 'test-vm-1'
});

test('update VM', async ({ page }) => {
  // Assumes 'test-vm-1' exists
});

// ✅ Good - Each test is independent
test('create VM', async ({ vmFactory }) => {
  const vm = await vmFactory.create();
  // Test with this specific VM
});

test('update VM', async ({ vmFactory }) => {
  const vm = await vmFactory.create();
  // Test with this specific VM
});
```

### 2. Follow AAA Pattern

Structure tests using Arrange-Act-Assert pattern:

```typescript
test('should create VM successfully', async ({ page, vmFactory }) => {
  // Arrange - Set up test data and page
  const vmData = {
    name: 'test-vm',
    cpu: 2,
    memory: 4096,
  };
  const vmCreationPage = new VMCreationPage(page);
  await vmCreationPage.navigate();

  // Act - Perform the action
  await vmCreationPage.createVM(vmData);

  // Assert - Verify the outcome
  const vms = await vmFactory.apiClient.get('/api/v1/vms');
  expect(vms).toContainEqual(expect.objectContaining({ name: vmData.name }));
});
```

### 3. Test User Journeys, Not Implementation

Focus on what users do, not how the application works internally.

```typescript
// ❌ Bad - Testing implementation details
test('should update Redux store when VM is created', async ({ page }) => {
  // Testing internal state management
});

// ✅ Good - Testing user-visible behavior
test('should display newly created VM in the list', async ({ page, vmFactory }) => {
  const vm = await vmFactory.create();
  await page.goto('/vms');
  await expect(page.locator(`[data-testid="vm-${vm.id}"]`)).toBeVisible();
});
```

---

## Test Design

### 1. Keep Tests Small and Focused

Each test should verify one specific behavior.

```typescript
// ❌ Bad - Test does too much
test('VM management workflow', async ({ page }) => {
  await loginPage.login();
  await vmPage.createVM();
  await vmPage.startVM();
  await vmPage.stopVM();
  await vmPage.deleteVM();
  await logoutPage.logout();
});

// ✅ Good - Separate focused tests
test('should create VM', async ({ page }) => { /* ... */ });
test('should start VM', async ({ page }) => { /* ... */ });
test('should stop VM', async ({ page }) => { /* ... */ });
test('should delete VM', async ({ page }) => { /* ... */ });
```

### 2. Use Descriptive Test Names

Test names should clearly describe what is being tested and the expected outcome.

```typescript
// ❌ Bad - Vague test name
test('VM test', async ({ page }) => { /* ... */ });

// ✅ Good - Descriptive test name
test('should display error message when creating VM with invalid CPU count', async ({ page }) => {
  /* ... */
});
```

### 3. Group Related Tests

Use `test.describe` to group related tests:

```typescript
test.describe('VM Creation', () => {
  test('should create VM with minimal configuration', async ({ page }) => { /* ... */ });
  test('should create VM with advanced options', async ({ page }) => { /* ... */ });
  test('should validate required fields', async ({ page }) => { /* ... */ });
});

test.describe('VM Lifecycle', () => {
  test('should start stopped VM', async ({ page }) => { /* ... */ });
  test('should stop running VM', async ({ page }) => { /* ... */ });
  test('should restart running VM', async ({ page }) => { /* ... */ });
});
```

---

## Locator Strategies

### Priority Order for Locators

1. **User-facing attributes** (data-testid, ARIA roles)
2. **User-visible text**
3. **CSS selectors** (last resort)

```typescript
// ✅ Best - data-testid
page.locator('[data-testid="submit-button"]')

// ✅ Good - ARIA role with accessible name
page.getByRole('button', { name: 'Submit' })

// ✅ Good - User-visible text
page.getByText('Submit')

// ⚠️ Acceptable - Placeholder text
page.getByPlaceholder('Enter VM name')

// ❌ Avoid - CSS class names
page.locator('.btn.btn-primary.submit')

// ❌ Avoid - Complex CSS selectors
page.locator('div > div > button:nth-child(3)')
```

### Use Specific Locators

Prefer specific locators over generic ones:

```typescript
// ❌ Bad - Too generic
await page.locator('button').click();

// ✅ Good - Specific
await page.locator('[data-testid="create-vm-button"]').click();
```

### Chain Locators for Scoped Selection

```typescript
// Scope searches within parent elements
const vmList = page.locator('[data-testid="vm-list"]');
const firstVM = vmList.locator('[data-testid="vm-item"]').first();
const vmName = firstVM.locator('[data-testid="vm-name"]');

await expect(vmName).toHaveText('test-vm-1');
```

---

## Page Object Model

### 1. Encapsulate Page Logic

Keep all page-specific logic in page objects:

```typescript
export class VMCreationPage extends BasePage {
  private readonly nameInput = this.page.locator('[data-testid="vm-name-input"]');
  private readonly submitButton = this.page.locator('[data-testid="submit-button"]');

  async createVM(name: string): Promise<void> {
    await this.nameInput.fill(name);
    await this.submitButton.click();
    await this.waitForSuccess();
  }

  private async waitForSuccess(): Promise<void> {
    await this.page.waitForSelector('[data-testid="success-notification"]');
  }
}
```

### 2. Return Page Objects for Navigation

Methods that cause navigation should return the new page object:

```typescript
export class LoginPage extends BasePage {
  async login(email: string, password: string): Promise<DashboardPage> {
    await this.emailInput.fill(email);
    await this.passwordInput.fill(password);
    await this.submitButton.click();

    // Wait for navigation
    await this.page.waitForURL(/\/dashboard/);

    return new DashboardPage(this.page);
  }
}

// Usage
test('should navigate to dashboard after login', async ({ page }) => {
  const loginPage = new LoginPage(page);
  const dashboardPage = await loginPage.login('user@example.com', 'password');

  await expect(dashboardPage.welcomeMessage).toBeVisible();
});
```

### 3. Use Composition for Reusable Components

```typescript
export class BasePage {
  protected navigation: NavigationComponent;
  protected notifications: NotificationComponent;

  constructor(protected page: Page) {
    this.navigation = new NavigationComponent(page);
    this.notifications = new NotificationComponent(page);
  }
}

export class VMListPage extends BasePage {
  async navigateToCreateVM(): Promise<void> {
    await this.navigation.clickMenuItem('Create VM');
  }

  async expectSuccessNotification(): Promise<void> {
    await this.notifications.expectSuccess('VM created successfully');
  }
}
```

---

## Test Data Management

### 1. Use Factories for Test Data

```typescript
export class VMFactory {
  async create(template: Partial<VMTemplate> = {}): Promise<VM> {
    const vmData = {
      name: template.name || `test-vm-${faker.string.uuid()}`,
      cpu: template.cpu || 2,
      memory: template.memory || 4096,
      // ... other fields with defaults
    };

    return this.apiClient.post('/api/v1/vms', vmData);
  }
}

// Usage
test('should display VM in list', async ({ vmFactory }) => {
  const vm = await vmFactory.create({ name: 'my-specific-vm' });
  // Test logic
});
```

### 2. Clean Up Test Data

Always clean up test data, even if tests fail:

```typescript
// ✅ Good - Using fixtures for automatic cleanup
export const test = base.extend<VMFixtures>({
  vmFactory: async ({}, use) => {
    const factory = new VMFactory(apiClient);
    await use(factory);
    await factory.cleanup(); // Automatic cleanup
  },
});

// ✅ Good - Manual cleanup in try-finally
test('manual cleanup example', async ({ page }) => {
  const vmIds: string[] = [];

  try {
    const vm = await apiClient.post('/api/v1/vms', { /* ... */ });
    vmIds.push(vm.id);
    // Test logic
  } finally {
    for (const id of vmIds) {
      await apiClient.delete(`/api/v1/vms/${id}`).catch(() => {});
    }
  }
});
```

### 3. Isolate Test Data

Each test should use unique data to avoid conflicts:

```typescript
// ✅ Good - Unique names
const vmName = `test-vm-${Date.now()}-${faker.string.uuid().substring(0, 8)}`;

// ❌ Bad - Hardcoded names (can cause conflicts)
const vmName = 'test-vm';
```

---

## Assertions

### 1. Use Playwright's Auto-Waiting Assertions

Playwright assertions auto-wait for conditions to be met:

```typescript
// ✅ Good - Auto-waiting assertion
await expect(page.locator('[data-testid="vm-list"]')).toBeVisible();

// ❌ Bad - Manual wait
await page.waitForSelector('[data-testid="vm-list"]');
expect(await page.locator('[data-testid="vm-list"]').isVisible()).toBe(true);
```

### 2. Be Specific with Assertions

Use the most specific assertion available:

```typescript
// ✅ Good - Specific assertions
await expect(vmName).toHaveText('test-vm');
await expect(submitButton).toBeEnabled();
await expect(errorMessage).toContainText('Invalid input');

// ❌ Bad - Generic assertions
expect(await vmName.textContent()).toBe('test-vm');
expect(await submitButton.isEnabled()).toBe(true);
```

### 3. Assert on User-Visible Behavior

```typescript
// ✅ Good - Testing visible behavior
await expect(page.locator('[data-testid="success-message"]')).toBeVisible();
await expect(page.locator('[data-testid="vm-count"]')).toHaveText('5 VMs');

// ❌ Bad - Testing implementation details
const storeState = await page.evaluate(() => window.__REDUX_STORE__.getState());
expect(storeState.vms.length).toBe(5);
```

---

## Error Handling

### 1. Provide Helpful Error Messages

```typescript
// ✅ Good - Custom error message
await expect(vmList, 'VM list should be visible after navigation').toBeVisible();

// ❌ Bad - No context
await expect(vmList).toBeVisible();
```

### 2. Capture Debug Information on Failure

```typescript
test('should create VM', async ({ page }) => {
  try {
    const vmCreationPage = new VMCreationPage(page);
    await vmCreationPage.createVM({ name: 'test-vm' });
  } catch (error) {
    // Capture screenshot
    await page.screenshot({
      path: `test-failures/${test.info().title}.png`,
      fullPage: true
    });

    // Capture trace
    await page.context().tracing.stop({
      path: `test-failures/${test.info().title}-trace.zip`
    });

    throw error;
  }
});
```

### 3. Handle Expected Errors Gracefully

```typescript
// ✅ Good - Handle expected errors
test('should show error for invalid VM configuration', async ({ page }) => {
  const vmCreationPage = new VMCreationPage(page);

  await vmCreationPage.fillForm({ cpu: -1 }); // Invalid
  await vmCreationPage.submit();

  await expect(page.locator('[data-testid="cpu-error"]'))
    .toHaveText('CPU count must be positive');
});
```

---

## Performance Optimization

### 1. Reuse Browser Contexts

```typescript
// Use storageState to reuse authentication
test.use({
  storageState: 'auth/user.json',
});

test('authenticated test', async ({ page }) => {
  // Already authenticated
  await page.goto('/dashboard');
});
```

### 2. Parallelize Independent Tests

```typescript
// Run tests in parallel (default)
test.describe.configure({ mode: 'parallel' });

test('test 1', async ({ page }) => { /* ... */ });
test('test 2', async ({ page }) => { /* ... */ });
test('test 3', async ({ page }) => { /* ... */ });
```

### 3. Mock Slow or External Services

```typescript
test('should handle slow API', async ({ page }) => {
  // Mock slow API endpoint
  await page.route('/api/v1/slow-endpoint', (route) => {
    route.fulfill({
      status: 200,
      body: JSON.stringify({ data: 'mocked response' }),
    });
  });

  await page.goto('/page-using-slow-endpoint');
  // Test continues without waiting for slow API
});
```

### 4. Skip Unnecessary Waits

```typescript
// ✅ Good - Wait for specific condition
await page.waitForLoadState('domcontentloaded');

// ❌ Bad - Wait for everything
await page.waitForLoadState('networkidle');

// ❌ Bad - Hard-coded timeout
await page.waitForTimeout(5000);
```

---

## Debugging

### 1. Use Playwright Inspector

```bash
# Run tests in debug mode
npm run test:e2e -- --debug

# Debug specific test
npm run test:e2e -- --debug -g "should create VM"
```

### 2. Use Console Logging Strategically

```typescript
test('debug example', async ({ page }) => {
  console.log('Navigating to VM creation page');
  await page.goto('/vms/create');

  console.log('Filling form');
  await page.fill('[data-testid="vm-name"]', 'test-vm');

  console.log('Submitting form');
  await page.click('[data-testid="submit"]');

  console.log('Waiting for success notification');
  await page.waitForSelector('[data-testid="success"]');
});
```

### 3. Take Screenshots for Debugging

```typescript
test('should create VM', async ({ page }) => {
  await page.goto('/vms/create');
  await page.screenshot({ path: 'debug/1-initial-page.png' });

  await page.fill('[data-testid="vm-name"]', 'test-vm');
  await page.screenshot({ path: 'debug/2-form-filled.png' });

  await page.click('[data-testid="submit"]');
  await page.screenshot({ path: 'debug/3-after-submit.png' });
});
```

### 4. Use Page.pause() for Interactive Debugging

```typescript
test('interactive debug', async ({ page }) => {
  await page.goto('/vms/create');

  // Pauses execution and opens Playwright Inspector
  await page.pause();

  await page.fill('[data-testid="vm-name"]', 'test-vm');
});
```

---

## Accessibility Testing

### 1. Use ARIA Roles for Locators

```typescript
// ✅ Good - Using ARIA roles
await page.getByRole('button', { name: 'Submit' }).click();
await page.getByRole('textbox', { name: 'VM Name' }).fill('test-vm');
await page.getByRole('heading', { name: 'Create VM' }).waitFor();
```

### 2. Test Keyboard Navigation

```typescript
test('should support keyboard navigation', async ({ page }) => {
  await page.goto('/vms/create');

  // Tab through form fields
  await page.keyboard.press('Tab');
  await page.keyboard.type('test-vm');

  await page.keyboard.press('Tab');
  await page.keyboard.type('4');

  // Submit with Enter
  await page.keyboard.press('Enter');

  await expect(page.locator('[data-testid="success"]')).toBeVisible();
});
```

### 3. Verify Color Contrast

```typescript
import { injectAxe, checkA11y } from 'axe-playwright';

test('should have no accessibility violations', async ({ page }) => {
  await page.goto('/vms/create');
  await injectAxe(page);

  await checkA11y(page, null, {
    detailedReport: true,
    detailedReportOptions: {
      html: true,
    },
  });
});
```

---

## Code Review Checklist

When reviewing E2E test PRs, check for:

- [ ] Tests are independent and can run in any order
- [ ] Test names clearly describe what is being tested
- [ ] Page objects are used for UI interactions
- [ ] Test data is created and cleaned up properly
- [ ] Assertions use Playwright's auto-waiting features
- [ ] No hard-coded waits (`waitForTimeout`)
- [ ] Locators use data-testid or ARIA roles
- [ ] Error messages provide helpful context
- [ ] Tests follow AAA pattern (Arrange-Act-Assert)
- [ ] No implementation details are tested

---

## Common Anti-Patterns to Avoid

### 1. Excessive Mocking

```typescript
// ❌ Bad - Mocking everything
test('should display VMs', async ({ page }) => {
  await page.route('/api/**', (route) => route.fulfill({ body: '{}' }));
  // Not testing real integration
});

// ✅ Good - Test real integration
test('should display VMs', async ({ page, vmFactory }) => {
  await vmFactory.create();
  await page.goto('/vms');
  // Tests actual backend integration
});
```

### 2. Testing Multiple Things in One Test

```typescript
// ❌ Bad
test('VM workflow', async ({ page }) => {
  await loginPage.login();
  await vmPage.createVM();
  await vmPage.updateVM();
  await vmPage.deleteVM();
});

// ✅ Good - Separate tests
test('should create VM', async ({ page }) => { /* ... */ });
test('should update VM', async ({ page }) => { /* ... */ });
test('should delete VM', async ({ page }) => { /* ... */ });
```

### 3. Hardcoded Sleeps

```typescript
// ❌ Bad
await page.click('[data-testid="submit"]');
await page.waitForTimeout(5000);

// ✅ Good
await page.click('[data-testid="submit"]');
await page.waitForSelector('[data-testid="success"]');
```

---

## Resources

- [Playwright Best Practices](https://playwright.dev/docs/best-practices)
- [Testing Library Guiding Principles](https://testing-library.com/docs/guiding-principles)
- [Web Accessibility Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)

