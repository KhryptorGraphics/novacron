# Migration Guide: Puppeteer to Playwright

This guide helps you migrate existing Puppeteer E2E tests to Playwright.

## Table of Contents

- [Why Migrate](#why-migrate)
- [Key Differences](#key-differences)
- [API Mapping](#api-mapping)
- [Migration Steps](#migration-steps)
- [Migration Checklist](#migration-checklist)
- [Common Pitfalls](#common-pitfalls)

## Why Migrate

### Benefits of Playwright

1. **Multi-Browser Support**
   - Chromium, Firefox, WebKit (Safari)
   - Consistent API across all browsers

2. **Better Testing Experience**
   - Auto-waiting for elements
   - Built-in test runner
   - Better debugging tools (trace viewer, inspector)

3. **Modern Features**
   - Network interception
   - Mobile emulation
   - Video recording
   - Screenshot comparison

4. **Improved Reliability**
   - Fewer flaky tests due to auto-waiting
   - Better error messages
   - Retry mechanisms

5. **Active Development**
   - Regular updates from Microsoft
   - Growing community
   - Better documentation

## Key Differences

### 1. Browser Context

**Puppeteer:**
```javascript
const browser = await puppeteer.launch();
const page = await browser.newPage();
```

**Playwright:**
```javascript
// Option 1: Using fixtures (recommended)
test('my test', async ({ page }) => {
  // page is automatically created
});

// Option 2: Manual
const browser = await chromium.launch();
const context = await browser.newContext();
const page = await context.newPage();
```

### 2. Auto-Waiting

**Puppeteer:**
```javascript
// Manual waiting required
await page.waitForSelector('button');
await page.click('button');
```

**Playwright:**
```javascript
// Auto-waits for element to be actionable
await page.click('button');
```

### 3. Test Runner

**Puppeteer:**
```javascript
// Use Jest or Mocha
describe('Tests', () => {
  let browser, page;

  beforeAll(async () => {
    browser = await puppeteer.launch();
  });

  beforeEach(async () => {
    page = await browser.newPage();
  });

  afterAll(async () => {
    await browser.close();
  });

  it('should test', async () => {
    await page.goto('http://localhost:3000');
  });
});
```

**Playwright:**
```javascript
// Built-in test runner with fixtures
const { test, expect } = require('@playwright/test');

test.describe('Tests', () => {
  test('should test', async ({ page }) => {
    await page.goto('http://localhost:3000');
  });
});
```

### 4. Assertions

**Puppeteer:**
```javascript
const text = await page.$eval('h1', el => el.textContent);
expect(text).toBe('Welcome');

const isVisible = await page.$('button') !== null;
expect(isVisible).toBe(true);
```

**Playwright:**
```javascript
// Web-first assertions with auto-retry
await expect(page.locator('h1')).toHaveText('Welcome');
await expect(page.locator('button')).toBeVisible();
```

## API Mapping

### Navigation

| Puppeteer | Playwright |
|-----------|------------|
| `page.goto(url)` | `page.goto(url)` |
| `page.goBack()` | `page.goBack()` |
| `page.goForward()` | `page.goForward()` |
| `page.reload()` | `page.reload()` |

### Selectors

| Puppeteer | Playwright |
|-----------|------------|
| `page.$('button')` | `page.locator('button')` |
| `page.$$('button')` | `page.locator('button').all()` |
| `page.$eval('h1', el => el.textContent)` | `page.locator('h1').textContent()` |
| `page.$$eval('.item', els => els.length)` | `page.locator('.item').count()` |
| `page.waitForSelector('button')` | `page.locator('button').waitFor()` |

### Actions

| Puppeteer | Playwright |
|-----------|------------|
| `page.click('button')` | `page.click('button')` |
| `page.type('input', 'text')` | `page.fill('input', 'text')` |
| `page.select('select', 'value')` | `page.selectOption('select', 'value')` |
| `page.hover('button')` | `page.hover('button')` |
| `page.focus('input')` | `page.focus('input')` |
| `page.keyboard.press('Enter')` | `page.keyboard.press('Enter')` |

### Wait Functions

| Puppeteer | Playwright |
|-----------|------------|
| `page.waitForSelector('.item')` | `page.locator('.item').waitFor()` |
| `page.waitForNavigation()` | `page.waitForNavigation()` |
| `page.waitForTimeout(1000)` | `page.waitForTimeout(1000)` |
| `page.waitForFunction(() => ...)` | `page.waitForFunction(() => ...)` |
| `page.waitForNetworkIdle()` | `page.waitForLoadState('networkidle')` |

### Evaluation

| Puppeteer | Playwright |
|-----------|------------|
| `page.evaluate(() => ...)` | `page.evaluate(() => ...)` |
| `page.$eval('h1', el => ...)` | `page.locator('h1').evaluate(el => ...)` |
| `page.$$eval('.item', els => ...)` | `page.locator('.item').evaluateAll(els => ...)` |

### Screenshots & PDFs

| Puppeteer | Playwright |
|-----------|------------|
| `page.screenshot({ path: 'screenshot.png' })` | `page.screenshot({ path: 'screenshot.png' })` |
| `page.pdf({ path: 'page.pdf' })` | `page.pdf({ path: 'page.pdf' })` |

### Network

| Puppeteer | Playwright |
|-----------|------------|
| `page.setRequestInterception(true)` | `page.route('**/*', route => ...)` |
| `request.continue()` | `route.continue()` |
| `request.respond({ body: '...' })` | `route.fulfill({ body: '...' })` |
| `request.abort()` | `route.abort()` |

### Cookies

| Puppeteer | Playwright |
|-----------|------------|
| `page.cookies()` | `context.cookies()` |
| `page.setCookie(cookie)` | `context.addCookies([cookie])` |
| `page.deleteCookie(cookie)` | `context.clearCookies()` |

## Migration Steps

### Step 1: Install Playwright

```bash
# Install Playwright
npm install -D @playwright/test

# Install browsers
npx playwright install

# Remove Puppeteer (optional)
npm uninstall puppeteer
```

### Step 2: Update Configuration

**Before (Puppeteer with Jest):**

```javascript
// jest.config.js
module.exports = {
  preset: 'jest-puppeteer',
  testMatch: ['**/tests/**/*.test.js'],
};
```

**After (Playwright):**

```javascript
// playwright.config.js
const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './tests/e2e/specs',
  use: {
    baseURL: 'http://localhost:3000',
  },
});
```

### Step 3: Convert Test Files

**Before (Puppeteer):**

```javascript
describe('Login', () => {
  let browser, page;

  beforeAll(async () => {
    browser = await puppeteer.launch();
  });

  afterAll(async () => {
    await browser.close();
  });

  beforeEach(async () => {
    page = await browser.newPage();
    await page.goto('http://localhost:3000');
  });

  afterEach(async () => {
    await page.close();
  });

  it('should login successfully', async () => {
    await page.waitForSelector('#email');
    await page.type('#email', 'user@example.com');
    await page.type('#password', 'password123');
    await page.click('button[type="submit"]');

    await page.waitForNavigation();

    const url = page.url();
    expect(url).toContain('/dashboard');

    const heading = await page.$eval('h1', el => el.textContent);
    expect(heading).toBe('Dashboard');
  });
});
```

**After (Playwright):**

```javascript
const { test, expect } = require('@playwright/test');

test.describe('Login', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should login successfully', async ({ page }) => {
    await page.fill('#email', 'user@example.com');
    await page.fill('#password', 'password123');
    await page.click('button[type="submit"]');

    // Auto-waits for navigation
    await expect(page).toHaveURL(/.*dashboard/);

    // Web-first assertions
    await expect(page.locator('h1')).toHaveText('Dashboard');
  });
});
```

### Step 4: Update Selectors

**Before (Puppeteer):**

```javascript
const button = await page.$('button');
await button.click();

const buttons = await page.$$('button');
for (const button of buttons) {
  await button.click();
}

const text = await page.$eval('h1', el => el.textContent);
```

**After (Playwright):**

```javascript
await page.locator('button').click();

const buttons = await page.locator('button').all();
for (const button of buttons) {
  await button.click();
}

const text = await page.locator('h1').textContent();
```

### Step 5: Update Waiting Logic

**Before (Puppeteer):**

```javascript
await page.waitForSelector('button');
await page.click('button');

await page.waitForSelector('.modal', { visible: true });

await page.waitForFunction(() => document.querySelector('.data') !== null);
```

**After (Playwright):**

```javascript
// Auto-waits
await page.click('button');

// Explicit wait if needed
await page.locator('.modal').waitFor({ state: 'visible' });

await page.waitForFunction(() => document.querySelector('.data') !== null);
```

### Step 6: Update Network Interception

**Before (Puppeteer):**

```javascript
await page.setRequestInterception(true);

page.on('request', request => {
  if (request.url().includes('/api/')) {
    request.continue();
  } else {
    request.abort();
  }
});
```

**After (Playwright):**

```javascript
await page.route('**/api/**', route => route.continue());
await page.route('**/*', route => route.abort());
```

### Step 7: Update Scripts

**Before (package.json):**

```json
{
  "scripts": {
    "test:e2e": "jest --config jest.config.js"
  }
}
```

**After (package.json):**

```json
{
  "scripts": {
    "test:e2e": "playwright test",
    "test:e2e:headed": "playwright test --headed",
    "test:e2e:debug": "playwright test --debug",
    "test:e2e:report": "playwright show-report"
  }
}
```

## Migration Checklist

### Pre-Migration

- [ ] Review existing Puppeteer tests
- [ ] Identify custom helpers and utilities
- [ ] Document test data and fixtures
- [ ] List all browser configurations
- [ ] Note CI/CD integration points

### Installation

- [ ] Install `@playwright/test`
- [ ] Install Playwright browsers
- [ ] Create `playwright.config.js`
- [ ] Set up test directory structure
- [ ] Configure environment variables

### Code Migration

- [ ] Convert test runner (Jest/Mocha → Playwright Test)
- [ ] Update import statements
- [ ] Replace `$` with `locator`
- [ ] Remove explicit waits where auto-wait applies
- [ ] Update assertions to web-first assertions
- [ ] Convert page object models
- [ ] Migrate helper functions
- [ ] Update network interception
- [ ] Convert screenshot/video logic

### Testing

- [ ] Run migrated tests locally
- [ ] Verify all tests pass
- [ ] Check test execution time
- [ ] Test in multiple browsers
- [ ] Validate CI/CD integration
- [ ] Review test reports

### Post-Migration

- [ ] Update documentation
- [ ] Train team on Playwright
- [ ] Remove Puppeteer dependencies
- [ ] Update README and guides
- [ ] Archive old test code

### Optimization

- [ ] Enable parallel execution
- [ ] Implement test sharding
- [ ] Set up trace viewer
- [ ] Configure video recording
- [ ] Optimize CI/CD pipeline

## Common Pitfalls

### 1. Not Leveraging Auto-Waiting

**Avoid:**
```javascript
await page.waitForSelector('button');
await page.click('button');
```

**Prefer:**
```javascript
await page.click('button'); // Auto-waits
```

### 2. Using `$` Instead of `locator`

**Avoid:**
```javascript
const button = await page.$('button');
await button.click();
```

**Prefer:**
```javascript
await page.locator('button').click();
```

### 3. Manual Element Queries

**Avoid:**
```javascript
const text = await page.$eval('h1', el => el.textContent);
expect(text).toBe('Welcome');
```

**Prefer:**
```javascript
await expect(page.locator('h1')).toHaveText('Welcome');
```

### 4. Not Using Test Fixtures

**Avoid:**
```javascript
let page;

beforeEach(async () => {
  page = await browser.newPage();
});

test('test', async () => {
  await page.goto('/');
});
```

**Prefer:**
```javascript
test('test', async ({ page }) => {
  await page.goto('/');
});
```

### 5. Hardcoded Timeouts

**Avoid:**
```javascript
await page.waitForTimeout(3000);
await page.click('button');
```

**Prefer:**
```javascript
await page.click('button'); // Auto-waits
// Or
await page.locator('button').waitFor();
```

### 6. Not Using Page Objects

**Avoid:**
```javascript
test('login', async ({ page }) => {
  await page.fill('#email', 'user@example.com');
  await page.fill('#password', 'password');
  await page.click('button');
});

test('logout', async ({ page }) => {
  await page.fill('#email', 'user@example.com'); // Duplicated
  await page.fill('#password', 'password'); // Duplicated
  await page.click('button'); // Duplicated
  // ... logout logic
});
```

**Prefer:**
```javascript
const { LoginPage } = require('./pages/LoginPage');

test('login', async ({ page }) => {
  const loginPage = new LoginPage(page);
  await loginPage.login('user@example.com', 'password');
});

test('logout', async ({ page }) => {
  const loginPage = new LoginPage(page);
  await loginPage.login('user@example.com', 'password');
  // ... logout logic
});
```

### 7. Ignoring Browser Context

**Avoid:**
```javascript
// Creating multiple browser instances
const browser1 = await chromium.launch();
const browser2 = await chromium.launch();
```

**Prefer:**
```javascript
// Use contexts for isolation
const browser = await chromium.launch();
const context1 = await browser.newContext();
const context2 = await browser.newContext();
```

## Gradual Migration Strategy

### Phase 1: Parallel Setup (Week 1-2)

1. Install Playwright alongside Puppeteer
2. Create Playwright config
3. Set up directory structure
4. Migrate 1-2 simple tests as proof of concept

### Phase 2: Critical Path (Week 3-4)

1. Migrate smoke tests
2. Migrate authentication tests
3. Migrate critical user flows
4. Update CI/CD to run both Puppeteer and Playwright

### Phase 3: Bulk Migration (Week 5-8)

1. Migrate remaining test suites
2. Update page objects
3. Migrate helpers and utilities
4. Run Playwright tests in CI alongside Puppeteer

### Phase 4: Cleanup (Week 9-10)

1. Remove Puppeteer tests
2. Remove Puppeteer dependencies
3. Update documentation
4. Final CI/CD optimization

## Additional Resources

- [Playwright Documentation](https://playwright.dev/)
- [Playwright Migration Guide](https://playwright.dev/docs/puppeteer)
- [Playwright API Reference](https://playwright.dev/docs/api/class-playwright)
- [Writing Tests Guide](./WRITING_TESTS.md)
- [Setup Guide](./SETUP.md)

---

**Last Updated**: 2025-01-10
**From**: Puppeteer
**To**: Playwright 1.40+
