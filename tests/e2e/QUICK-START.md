# Playwright E2E Testing - Quick Start Guide

## Installation (One-Time Setup)

```bash
# 1. Install dependencies
npm install

# 2. Install Playwright browsers
npm run test:e2e:install

# 3. Create environment file
cp tests/e2e/.env.example tests/e2e/.env
```

## Running Tests

### Basic Commands

```bash
# Run all tests
npm run test:e2e:playwright

# Run with browser visible
npm run test:e2e:headed

# Run in debug mode (step through)
npm run test:e2e:debug

# Run UI mode (interactive)
npm run test:e2e:ui
```

### Browser-Specific

```bash
# Chrome only
npm run test:e2e:chromium

# Firefox only
npm run test:e2e:firefox

# Safari (WebKit) only
npm run test:e2e:webkit

# Mobile browsers
npm run test:e2e:mobile
```

### Reports

```bash
# Show HTML report
npm run test:e2e:report

# Run tests for CI (Chrome + Firefox)
npm run test:e2e:ci
```

## Writing Your First Test

Create `tests/e2e/my-feature.spec.ts`:

```typescript
import { test, expect } from '@playwright/test';

test('my first test', async ({ page }) => {
  await page.goto('https://playwright.dev/');
  await expect(page).toHaveTitle(/Playwright/);
});
```

## Using Helper Functions

```typescript
import { test } from '@playwright/test';
import {
  navigateAndWait,
  fillInput,
  clickWithRetry
} from './utils/playwright-helpers';
import { generateUser } from './utils/data-generators';

test('user registration', async ({ page }) => {
  const user = generateUser();

  await navigateAndWait(page, '/register');
  await fillInput(page.locator('#email'), user.email);
  await fillInput(page.locator('#password'), user.password);
  await clickWithRetry(page.locator('button[type="submit"]'));
});
```

## Common Patterns

### Wait for Elements

```typescript
// Wait for visible
await expect(page.locator('.success')).toBeVisible();

// Wait for text
await expect(page.locator('h1')).toHaveText('Welcome');

// Wait for URL
await page.waitForURL('/dashboard');
```

### Interact with Elements

```typescript
// Click
await page.click('button');

// Type
await page.fill('input[name="username"]', 'john');

// Select dropdown
await page.selectOption('select', 'option-value');

// Upload file
await page.setInputFiles('input[type="file"]', 'path/to/file.pdf');
```

### Assertions

```typescript
// Visibility
await expect(element).toBeVisible();
await expect(element).toBeHidden();

// Text content
await expect(element).toHaveText('Expected text');
await expect(element).toContainText('partial text');

// Attributes
await expect(element).toHaveAttribute('href', '/link');
await expect(element).toHaveClass(/active/);

// Count
await expect(page.locator('.item')).toHaveCount(5);
```

## Test Data Generation

```typescript
import {
  generateUser,
  generateVM,
  generateEmail,
  generatePassword
} from './utils/data-generators';

// Generate complete user
const user = generateUser();

// Generate with overrides
const admin = generateUser({ role: 'admin' });

// Generate individual fields
const email = generateEmail('test');
const password = generatePassword(16);

// Generate VM data
const vm = generateVM({ cpu: 8, memory: 16 });
```

## Debugging

### Visual Debugging

```bash
# Run with browser visible
npm run test:e2e:headed

# Debug mode (pause execution)
npm run test:e2e:debug

# Interactive UI
npm run test:e2e:ui
```

### Code Generation

```bash
# Record interactions to generate test code
npm run test:e2e:codegen
```

### Screenshots

```typescript
// Take screenshot
await page.screenshot({ path: 'screenshot.png' });

// Full page screenshot
await page.screenshot({ path: 'full.png', fullPage: true });

// Element screenshot
await page.locator('.chart').screenshot({ path: 'chart.png' });
```

### Traces

```typescript
// Traces are automatically captured on failure
// View them with:
npx playwright show-trace tests/e2e/reports/traces/trace.zip
```

## Environment Configuration

Edit `tests/e2e/.env`:

```bash
# Base URL
PLAYWRIGHT_BASE_URL=http://localhost:3000

# Test credentials
TEST_USER=test@novacron.com
TEST_PASSWORD=testpassword123

# Skip authentication
PLAYWRIGHT_SKIP_AUTH=false

# Performance
PLAYWRIGHT_TIMEOUT=30000
PLAYWRIGHT_WORKERS=4
```

## Useful Selectors

```typescript
// By test ID (recommended)
page.locator('[data-testid="login-button"]')

// By role
page.getByRole('button', { name: 'Login' })

// By text
page.getByText('Submit')

// By placeholder
page.getByPlaceholder('Enter email')

// By label
page.getByLabel('Username')

// CSS selector
page.locator('.btn-primary')

// XPath
page.locator('xpath=//button[@type="submit"]')
```

## Best Practices

1. Use `data-testid` attributes for stable selectors
2. Generate test data dynamically to avoid conflicts
3. Keep tests independent (don't rely on test order)
4. Use meaningful test names that describe behavior
5. Group related tests with `test.describe()`
6. Clean up test data after tests
7. Mock external dependencies when appropriate
8. Use helper functions for common operations

## Troubleshooting

### Test times out
- Increase timeout in `playwright.config.ts`
- Add explicit waits: `await page.waitForLoadState('networkidle')`
- Check if server is running

### Element not found
- Verify selector is correct
- Add wait: `await element.waitFor({ state: 'visible' })`
- Check if element is in iframe: `page.frameLocator('iframe')`

### Flaky tests
- Enable retries: Set `retries: 2` in config
- Use `waitForLoadState('networkidle')`
- Add explicit waits for elements
- Use `toBeVisible()` instead of checking visibility

## Resources

- Full docs: `tests/e2e/README.md`
- Example tests: `tests/e2e/example.spec.ts`
- Helper functions: `tests/e2e/utils/`
- Playwright docs: https://playwright.dev

## Need Help?

1. Check `tests/e2e/README.md` for detailed documentation
2. Review `tests/e2e/example.spec.ts` for examples
3. Use `npm run test:e2e:codegen` to generate test code
4. Enable debug mode: `npm run test:e2e:debug`
