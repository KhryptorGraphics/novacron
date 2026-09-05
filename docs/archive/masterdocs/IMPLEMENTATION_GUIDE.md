# NovaCron Playwright E2E Testing - Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing the E2E testing architecture with Playwright for NovaCron.

---

## Phase 1: Initial Setup (Week 1)

### 1.1 Install Dependencies

```bash
cd /home/kp/novacron

# Install Playwright and dependencies
npm install -D @playwright/test
npm install -D @faker-js/faker
npm install -D msw

# Install Playwright browsers
npx playwright install --with-deps
```

### 1.2 Create Directory Structure

```bash
# Create E2E test directory structure
mkdir -p tests/e2e/{tests,page-objects,fixtures,helpers,config,mocks,reports}

# Create subdirectories
mkdir -p tests/e2e/tests/{auth,vm-management,cluster,dwcp,monitoring,smoke}
mkdir -p tests/e2e/page-objects/{base,auth,vm,cluster,monitoring,components}
mkdir -p tests/e2e/helpers/{test-data,api,assertions,utilities}
mkdir -p tests/e2e/config/environments
mkdir -p tests/e2e/mocks/{api,recordings,handlers}
```

### 1.3 Initialize Playwright Configuration

Create `tests/e2e/config/playwright.config.ts`:

```typescript
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: '../tests',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 4 : undefined,
  reporter: [
    ['html', { outputFolder: '../reports/html' }],
    ['json', { outputFile: '../reports/test-results.json' }],
    ['junit', { outputFile: '../reports/junit.xml' }],
  ],

  use: {
    baseURL: process.env.BASE_URL || 'http://localhost:8092',
    trace: 'on-first-retry',
    video: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  webServer: {
    command: 'cd ../../frontend && npm run dev',
    url: 'http://localhost:8092',
    reuseExistingServer: !process.env.CI,
    timeout: 120000,
  },
});
```

### 1.4 Add npm Scripts

Update `package.json`:

```json
{
  "scripts": {
    "test:e2e": "playwright test --config=tests/e2e/config/playwright.config.ts",
    "test:e2e:headed": "npm run test:e2e -- --headed",
    "test:e2e:debug": "npm run test:e2e -- --debug",
    "test:e2e:ui": "npm run test:e2e -- --ui",
    "test:e2e:smoke": "npm run test:e2e -- --grep @smoke",
    "test:e2e:chromium": "npm run test:e2e -- --project=chromium",
    "test:e2e:firefox": "npm run test:e2e -- --project=firefox",
    "test:e2e:webkit": "npm run test:e2e -- --project=webkit",
    "test:e2e:report": "playwright show-report tests/e2e/reports/html"
  }
}
```

---

## Phase 2: Base Infrastructure (Week 1-2)

### 2.1 Create Base Page Class

Create `tests/e2e/page-objects/base/BasePage.ts`:

```typescript
import { Page, Locator, expect } from '@playwright/test';

export abstract class BasePage {
  constructor(
    protected page: Page,
    protected baseURL: string = process.env.BASE_URL || 'http://localhost:8092'
  ) {}

  abstract getPath(): string;

  async navigate(): Promise<void> {
    await this.page.goto(`${this.baseURL}${this.getPath()}`);
    await this.waitForPageLoad();
  }

  async waitForPageLoad(): Promise<void> {
    await this.page.waitForLoadState('domcontentloaded');
  }

  protected async clickElement(locator: Locator): Promise<void> {
    await locator.waitFor({ state: 'visible' });
    await locator.click();
  }

  protected async fillInput(locator: Locator, value: string): Promise<void> {
    await locator.waitFor({ state: 'visible' });
    await locator.clear();
    await locator.fill(value);
  }

  protected async selectOption(locator: Locator, value: string): Promise<void> {
    await locator.waitFor({ state: 'visible' });
    await locator.selectOption(value);
  }

  async expectElementVisible(locator: Locator): Promise<void> {
    await expect(locator).toBeVisible();
  }

  async expectElementHidden(locator: Locator): Promise<void> {
    await expect(locator).toBeHidden();
  }

  async expectElementText(locator: Locator, text: string | RegExp): Promise<void> {
    await expect(locator).toHaveText(text);
  }
}
```

### 2.2 Create API Client

Create `tests/e2e/helpers/api/APIClient.ts`:

```typescript
import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';

export interface APIClientConfig {
  baseURL: string;
  timeout?: number;
  headers?: Record<string, string>;
}

export class APIClient {
  private client: AxiosInstance;

  constructor(config: APIClientConfig) {
    this.client = axios.create({
      baseURL: config.baseURL,
      timeout: config.timeout || 30000,
      headers: {
        'Content-Type': 'application/json',
        ...config.headers,
      },
    });
  }

  async get<T = any>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.get(url, config);
    return response.data;
  }

  async post<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.post(url, data, config);
    return response.data;
  }

  async put<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.put(url, data, config);
    return response.data;
  }

  async patch<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.patch(url, data, config);
    return response.data;
  }

  async delete<T = any>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.delete(url, config);
    return response.data;
  }

  setAuthToken(token: string): void {
    this.client.defaults.headers.common['Authorization'] = `Bearer ${token}`;
  }

  clearAuthToken(): void {
    delete this.client.defaults.headers.common['Authorization'];
  }
}
```

### 2.3 Create Test Data Factory

Create `tests/e2e/helpers/test-data/VMFactory.ts`:

```typescript
import { APIClient } from '../api/APIClient';
import { faker } from '@faker-js/faker';

export interface VMTemplate {
  name?: string;
  cpu?: number;
  memory?: number;
  disk?: number;
  image?: string;
  state?: 'running' | 'stopped' | 'pending';
}

export class VMFactory {
  private createdVMs: string[] = [];

  constructor(private apiClient: APIClient) {}

  async create(template: VMTemplate = {}): Promise<any> {
    const vmData = {
      name: template.name || `test-vm-${faker.string.uuid().substring(0, 8)}`,
      cpu: template.cpu || 2,
      memory: template.memory || 4096,
      disk: template.disk || 20,
      image: template.image || 'ubuntu-20.04',
      state: template.state || 'stopped',
      tags: {
        environment: 'test',
        createdBy: 'e2e-tests',
        timestamp: new Date().toISOString(),
      },
    };

    const vm = await this.apiClient.post('/api/v1/vms', vmData);
    this.createdVMs.push(vm.id);

    return vm;
  }

  async createBatch(count: number, template: VMTemplate = {}): Promise<any[]> {
    const promises = Array.from({ length: count }, () => this.create(template));
    return Promise.all(promises);
  }

  async cleanup(): Promise<void> {
    const deletePromises = this.createdVMs.map((id) =>
      this.apiClient.delete(`/api/v1/vms/${id}`).catch(() => {
        // Ignore errors during cleanup
        console.warn(`Failed to delete VM ${id}`);
      })
    );

    await Promise.all(deletePromises);
    this.createdVMs = [];
  }

  static templates = {
    minimal: { cpu: 1, memory: 1024, disk: 10, image: 'alpine-latest' },
    standard: { cpu: 2, memory: 4096, disk: 20, image: 'ubuntu-20.04' },
    highPerformance: { cpu: 8, memory: 16384, disk: 100, image: 'ubuntu-22.04' },
  };
}
```

---

## Phase 3: Page Objects (Week 2-3)

### 3.1 Create Login Page Object

Create `tests/e2e/page-objects/auth/LoginPage.ts`:

```typescript
import { Page, Locator } from '@playwright/test';
import { BasePage } from '../base/BasePage';

export class LoginPage extends BasePage {
  private readonly emailInput: Locator;
  private readonly passwordInput: Locator;
  private readonly submitButton: Locator;
  private readonly errorMessage: Locator;
  private readonly forgotPasswordLink: Locator;

  constructor(page: Page) {
    super(page);

    this.emailInput = page.locator('[data-testid="email-input"], input[type="email"]');
    this.passwordInput = page.locator('[data-testid="password-input"], input[type="password"]');
    this.submitButton = page.locator('[data-testid="login-submit"], button[type="submit"]');
    this.errorMessage = page.locator('[data-testid="error-message"], [role="alert"]');
    this.forgotPasswordLink = page.locator('[data-testid="forgot-password"], a:has-text("Forgot")');
  }

  getPath(): string {
    return '/auth/login';
  }

  async login(email: string, password: string): Promise<void> {
    await this.fillInput(this.emailInput, email);
    await this.fillInput(this.passwordInput, password);
    await this.clickElement(this.submitButton);
  }

  async waitForRedirect(): Promise<void> {
    await this.page.waitForURL(/\/dashboard/, { timeout: 10000 });
  }

  async getErrorMessage(): Promise<string> {
    await this.errorMessage.waitFor({ state: 'visible' });
    return this.errorMessage.textContent() || '';
  }

  async clickForgotPassword(): Promise<void> {
    await this.clickElement(this.forgotPasswordLink);
  }
}
```

### 3.2 Create VM Creation Page Object

Create `tests/e2e/page-objects/vm/VMCreationPage.ts`:

```typescript
import { Page, Locator } from '@playwright/test';
import { BasePage } from '../base/BasePage';

export interface VMCreationData {
  name: string;
  cpu: number;
  memory: number;
  disk: number;
  image: string;
}

export class VMCreationPage extends BasePage {
  private readonly nameInput: Locator;
  private readonly cpuSelect: Locator;
  private readonly memoryInput: Locator;
  private readonly diskInput: Locator;
  private readonly imageSelect: Locator;
  private readonly submitButton: Locator;
  private readonly successNotification: Locator;

  constructor(page: Page) {
    super(page);

    this.nameInput = page.locator('[data-testid="vm-name-input"]');
    this.cpuSelect = page.locator('[data-testid="vm-cpu-select"]');
    this.memoryInput = page.locator('[data-testid="vm-memory-input"]');
    this.diskInput = page.locator('[data-testid="vm-disk-input"]');
    this.imageSelect = page.locator('[data-testid="vm-image-select"]');
    this.submitButton = page.locator('[data-testid="vm-create-submit"]');
    this.successNotification = page.locator('[data-testid="success-notification"]');
  }

  getPath(): string {
    return '/vms/create';
  }

  async createVM(data: VMCreationData): Promise<void> {
    await this.fillInput(this.nameInput, data.name);
    await this.selectOption(this.cpuSelect, data.cpu.toString());
    await this.fillInput(this.memoryInput, data.memory.toString());
    await this.fillInput(this.diskInput, data.disk.toString());
    await this.selectOption(this.imageSelect, data.image);
    await this.clickElement(this.submitButton);
    await this.waitForSuccess();
  }

  async waitForSuccess(): Promise<void> {
    await this.successNotification.waitFor({ state: 'visible', timeout: 15000 });
  }
}
```

---

## Phase 4: Test Fixtures (Week 3)

### 4.1 Create VM Fixture

Create `tests/e2e/fixtures/vm.fixture.ts`:

```typescript
import { test as base } from '@playwright/test';
import { VMFactory } from '../helpers/test-data/VMFactory';
import { APIClient } from '../helpers/api/APIClient';

type VMFixtures = {
  vmFactory: VMFactory;
  apiClient: APIClient;
};

export const test = base.extend<VMFixtures>({
  apiClient: async ({}, use) => {
    const client = new APIClient({
      baseURL: process.env.API_URL || 'http://localhost:8090',
    });

    await use(client);
  },

  vmFactory: async ({ apiClient }, use) => {
    const factory = new VMFactory(apiClient);
    await use(factory);
    await factory.cleanup();
  },
});

export { expect } from '@playwright/test';
```

### 4.2 Create Auth Fixture

Create `tests/e2e/fixtures/auth.fixture.ts`:

```typescript
import { test as base } from '@playwright/test';
import { LoginPage } from '../page-objects/auth/LoginPage';

type AuthFixtures = {
  authenticatedPage: Page;
};

export const test = base.extend<AuthFixtures>({
  authenticatedPage: async ({ page }, use) => {
    // Login via API for faster setup
    const response = await page.request.post(`${process.env.API_URL}/api/v1/auth/login`, {
      data: {
        email: process.env.TEST_USER_EMAIL || 'test@example.com',
        password: process.env.TEST_USER_PASSWORD || 'password123',
      },
    });

    const { token } = await response.json();

    // Set auth token in browser context
    await page.context().addCookies([
      {
        name: 'auth-token',
        value: token,
        domain: new URL(process.env.BASE_URL!).hostname,
        path: '/',
      },
    ]);

    await use(page);
  },
});

export { expect } from '@playwright/test';
```

---

## Phase 5: Write Tests (Week 4-6)

### 5.1 Smoke Test Example

Create `tests/e2e/tests/smoke/critical-paths.spec.ts`:

```typescript
import { test, expect } from '../../fixtures/auth.fixture';
import { LoginPage } from '../../page-objects/auth/LoginPage';

test.describe('Smoke Tests - Critical Paths @smoke', () => {
  test('should load login page successfully', async ({ page }) => {
    const loginPage = new LoginPage(page);
    await loginPage.navigate();

    await expect(page).toHaveTitle(/NovaCron|Login/);
    await loginPage.expectElementVisible(page.locator('input[type="email"]'));
  });

  test('should login successfully with valid credentials', async ({ page }) => {
    const loginPage = new LoginPage(page);
    await loginPage.navigate();

    await loginPage.login(
      process.env.TEST_USER_EMAIL || 'test@example.com',
      process.env.TEST_USER_PASSWORD || 'password123'
    );

    await loginPage.waitForRedirect();
    await expect(page).toHaveURL(/\/dashboard/);
  });

  test('should load dashboard after authentication', async ({ authenticatedPage }) => {
    await authenticatedPage.goto('/dashboard');
    await expect(authenticatedPage).toHaveURL(/\/dashboard/);

    // Check critical dashboard elements
    await expect(authenticatedPage.locator('[data-testid="vm-stats"]')).toBeVisible();
  });
});
```

### 5.2 VM Lifecycle Test Example

Create `tests/e2e/tests/vm-management/vm-lifecycle.spec.ts`:

```typescript
import { test, expect } from '../../fixtures/vm.fixture';
import { VMCreationPage } from '../../page-objects/vm/VMCreationPage';
import { VMFactory } from '../../helpers/test-data/VMFactory';

test.describe('VM Lifecycle Management @regression', () => {
  test('should create a new VM successfully', async ({ page, vmFactory }) => {
    const vmCreationPage = new VMCreationPage(page);
    await vmCreationPage.navigate();

    const vmData = {
      name: `e2e-test-vm-${Date.now()}`,
      cpu: 2,
      memory: 4096,
      disk: 20,
      image: 'ubuntu-20.04',
    };

    await vmCreationPage.createVM(vmData);

    // Verify VM was created via API
    const vms = await vmFactory.apiClient.get('/api/v1/vms');
    const createdVM = vms.find((vm: any) => vm.name === vmData.name);

    expect(createdVM).toBeDefined();
    expect(createdVM.cpu).toBe(vmData.cpu);
  });

  test('should display VM in list after creation', async ({ page, vmFactory }) => {
    // Create test VM via API
    const vm = await vmFactory.create(VMFactory.templates.standard);

    // Navigate to VM list
    await page.goto('/vms');

    // Verify VM appears in list
    const vmRow = page.locator(`[data-testid="vm-${vm.id}"]`);
    await expect(vmRow).toBeVisible();

    const vmName = vmRow.locator('[data-testid="vm-name"]');
    await expect(vmName).toHaveText(vm.name);
  });
});
```

---

## Phase 6: CI/CD Integration (Week 6)

### 6.1 Create GitHub Actions Workflow

Create `.github/workflows/e2e-tests.yml`:

```yaml
name: E2E Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  e2e-tests:
    timeout-minutes: 60
    runs-on: ubuntu-latest

    strategy:
      fail-fast: false
      matrix:
        browser: [chromium]
        shard: [1/4, 2/4, 3/4, 4/4]

    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Install Playwright
        run: npx playwright install --with-deps ${{ matrix.browser }}

      - name: Run E2E tests
        run: npm run test:e2e -- --project=${{ matrix.browser }} --shard=${{ matrix.shard }}
        env:
          BASE_URL: http://localhost:8092
          API_URL: http://localhost:8090

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: playwright-report-${{ matrix.browser }}-${{ matrix.shard }}
          path: tests/e2e/reports/
          retention-days: 30
```

---

## Phase 7: Migration from Puppeteer (Week 7)

### 7.1 Migration Checklist

- [ ] Install Playwright dependencies
- [ ] Create directory structure
- [ ] Implement base page objects
- [ ] Create test fixtures
- [ ] Convert 1-2 existing Puppeteer tests to Playwright
- [ ] Run both test suites in parallel
- [ ] Gradually migrate remaining tests
- [ ] Update CI/CD pipelines
- [ ] Deprecate Puppeteer tests

### 7.2 Side-by-Side Comparison

**Puppeteer:**
```javascript
const page = await global.puppeteerUtils.createPage();
await page.goto('http://localhost:8092/login');
await page.type('input[name="email"]', 'test@example.com');
await page.click('button[type="submit"]');
```

**Playwright:**
```typescript
const loginPage = new LoginPage(page);
await loginPage.navigate();
await loginPage.login('test@example.com', 'password123');
```

---

## Testing Best Practices

### 1. Use Test Data Attributes

Add `data-testid` attributes to critical UI elements:

```html
<button data-testid="submit-button">Submit</button>
<div data-testid="vm-list">...</div>
```

### 2. Avoid Hard Waits

```typescript
// ❌ Bad
await page.waitForTimeout(5000);

// ✅ Good
await page.waitForSelector('[data-testid="vm-list"]');
await page.waitForResponse(/\/api\/v1\/vms/);
```

### 3. Clean Up Test Data

Always use fixtures for automatic cleanup:

```typescript
test('should create VM', async ({ vmFactory }) => {
  const vm = await vmFactory.create();
  // Test logic
  // Cleanup happens automatically
});
```

### 4. Use Page Objects

```typescript
// ❌ Bad - Direct page interaction in tests
await page.locator('[data-testid="email"]').fill('test@example.com');
await page.locator('[data-testid="submit"]').click();

// ✅ Good - Page object abstraction
const loginPage = new LoginPage(page);
await loginPage.login('test@example.com', 'password123');
```

---

## Troubleshooting

### Common Issues

**Issue: Tests timeout waiting for elements**
```typescript
// Solution: Increase timeout for slow operations
await page.waitForSelector('[data-testid="vm-list"]', { timeout: 30000 });
```

**Issue: Flaky tests due to animations**
```typescript
// Solution: Wait for animations to complete
await page.locator('.modal').waitFor({ state: 'visible' });
await page.waitForTimeout(300); // Wait for animation
```

**Issue: API calls fail in CI**
```typescript
// Solution: Add retry logic
await test.step('Create VM with retry', async () => {
  let attempts = 0;
  while (attempts < 3) {
    try {
      await vmFactory.create();
      break;
    } catch (error) {
      attempts++;
      if (attempts === 3) throw error;
      await page.waitForTimeout(1000);
    }
  }
});
```

---

## Next Steps

1. **Week 1-2**: Set up infrastructure and base components
2. **Week 3**: Implement page objects for critical features
3. **Week 4-6**: Write comprehensive test suites
4. **Week 7**: Migrate from Puppeteer
5. **Ongoing**: Maintain and expand test coverage

---

## Support and Resources

- **Internal Documentation**: `/tests/e2e/docs/`
- **Playwright Docs**: https://playwright.dev
- **Team Slack**: #e2e-testing
- **Questions**: Contact the QA team

