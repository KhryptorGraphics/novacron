# E2E Testing Quick Reference

## Essential Commands

```bash
# Install and setup
npm install -D @playwright/test @faker-js/faker msw
npx playwright install --with-deps

# Run tests
npm run test:e2e                    # All tests
npm run test:e2e -- --headed        # With browser UI
npm run test:e2e -- --debug         # Debug mode
npm run test:e2e -- --ui            # UI mode
npm run test:e2e -- --grep @smoke   # Smoke tests only

# Specific browsers
npm run test:e2e -- --project=chromium
npm run test:e2e -- --project=firefox
npm run test:e2e -- --project=webkit

# Reports
npm run test:e2e:report             # View HTML report
playwright show-trace trace.zip     # View trace file
```

## Common Patterns

### Test Structure

```typescript
test('should do something', async ({ page }) => {
  // Arrange
  const page = new SomePage(page);
  await page.navigate();

  // Act
  await page.performAction();

  // Assert
  await expect(page.result).toBeVisible();
});
```

### Using Fixtures

```typescript
import { test, expect } from '../fixtures/vm.fixture';

test('with fixtures', async ({ vmFactory, authenticatedPage }) => {
  const vm = await vmFactory.create();
  await authenticatedPage.goto('/vms');
  // Test logic
  // Cleanup automatic via fixtures
});
```

### Page Objects

```typescript
export class LoginPage extends BasePage {
  private readonly emailInput = this.page.locator('[data-testid="email"]');

  getPath(): string {
    return '/login';
  }

  async login(email: string, password: string): Promise<void> {
    await this.fillInput(this.emailInput, email);
    await this.submit();
  }
}
```

## Locator Priority

1. `page.locator('[data-testid="element"]')` - Best
2. `page.getByRole('button', { name: 'Submit' })` - Good
3. `page.getByText('Submit')` - Good
4. `page.locator('.css-class')` - Avoid

## Common Assertions

```typescript
// Visibility
await expect(element).toBeVisible();
await expect(element).toBeHidden();

// Text
await expect(element).toHaveText('exact text');
await expect(element).toContainText('partial');

// State
await expect(element).toBeEnabled();
await expect(element).toBeDisabled();
await expect(element).toBeChecked();

// Count
await expect(elements).toHaveCount(5);

// URL
await expect(page).toHaveURL(/dashboard/);
await expect(page).toHaveTitle(/NovaCron/);
```

## Waiting Strategies

```typescript
// Wait for element
await page.waitForSelector('[data-testid="element"]');
await element.waitFor({ state: 'visible' });

// Wait for API
await page.waitForResponse('/api/v1/vms');

// Wait for navigation
await page.waitForURL('/dashboard');

// Wait for load state
await page.waitForLoadState('domcontentloaded');
await page.waitForLoadState('networkidle'); // Slower

// Custom condition
await page.waitForFunction(() => window.ready === true);
```

## Test Categories

```typescript
test.describe('Feature @smoke', () => {
  // Smoke tests - critical paths
});

test.describe('Feature @regression', () => {
  // Regression tests - full coverage
});

test.describe('Feature @integration', () => {
  // Integration tests - cross-feature
});

test.describe('Feature @performance', () => {
  // Performance tests
});
```

## Environment Variables

```bash
# .env.test
BASE_URL=http://localhost:8092
API_URL=http://localhost:8090
TEST_USER_EMAIL=test@example.com
TEST_USER_PASSWORD=password123
CI=false
HEADLESS=true
```

## Directory Structure

```
tests/e2e/
├── tests/               # Test specs
│   ├── smoke/
│   ├── auth/
│   ├── vm-management/
│   └── ...
├── page-objects/        # Page object models
│   ├── base/
│   ├── auth/
│   ├── vm/
│   └── ...
├── fixtures/            # Test fixtures
├── helpers/             # Utilities
│   ├── test-data/       # Factories
│   ├── api/             # API clients
│   └── assertions/      # Custom matchers
├── config/              # Playwright config
├── mocks/               # API mocks
└── reports/             # Test reports
```

## Debugging

```typescript
// Pause execution
await page.pause();

// Screenshot
await page.screenshot({ path: 'debug.png' });

// Console log
page.on('console', msg => console.log(msg.text()));

// Network requests
page.on('request', req => console.log(req.url()));
page.on('response', res => console.log(res.url(), res.status()));
```

## CI/CD Integration

```yaml
# .github/workflows/e2e-tests.yml
- name: Run E2E tests
  run: npm run test:e2e
  env:
    BASE_URL: ${{ secrets.E2E_BASE_URL }}
    API_URL: ${{ secrets.E2E_API_URL }}

- name: Upload results
  if: always()
  uses: actions/upload-artifact@v3
  with:
    name: playwright-report
    path: tests/e2e/reports/
```

## Common Gotchas

### 1. Auto-waiting doesn't work with .all()

```typescript
// ❌ Bad - No auto-waiting
const items = await page.locator('.item').all();

// ✅ Good - Use count() first
const count = await page.locator('.item').count();
expect(count).toBe(5);
```

### 2. Parallel tests need isolation

```typescript
// Each test gets its own browser context
test.use({ storageState: { cookies: [], origins: [] } });
```

### 3. waitForTimeout is flaky

```typescript
// ❌ Bad
await page.waitForTimeout(5000);

// ✅ Good
await page.waitForSelector('[data-testid="element"]');
```

## Migration from Puppeteer

| Puppeteer | Playwright |
|-----------|------------|
| `page.click()` | `page.click()` (same) |
| `page.type()` | `page.fill()` |
| `page.$()` | `page.locator()` |
| `page.$$()` | `page.locator().all()` |
| `page.$eval()` | `page.locator().evaluate()` |
| `page.waitForSelector()` | `page.waitForSelector()` (same) |
| `page.goto()` | `page.goto()` (same) |

## Performance Tips

1. Reuse authentication state
2. Use API for test setup
3. Mock slow endpoints
4. Run tests in parallel
5. Use `domcontentloaded` instead of `networkidle`
6. Disable images/CSS when not needed

## Accessibility Testing

```typescript
import { injectAxe, checkA11y } from 'axe-playwright';

test('a11y', async ({ page }) => {
  await page.goto('/vms');
  await injectAxe(page);
  await checkA11y(page);
});
```

## Visual Regression

```typescript
test('visual', async ({ page }) => {
  await page.goto('/dashboard');
  await expect(page).toHaveScreenshot('dashboard.png', {
    maxDiffPixels: 100,
  });
});
```

## Resources

- **Architecture**: `tests/e2e/docs/ARCHITECTURE.md`
- **Implementation**: `tests/e2e/docs/IMPLEMENTATION_GUIDE.md`
- **Best Practices**: `tests/e2e/docs/BEST_PRACTICES.md`
- **Playwright Docs**: https://playwright.dev

