# Testing Quick Reference - NovaCron

**Last Updated:** 2025-11-11

## Quick Stats

| Metric | Value |
|--------|-------|
| **Total Test Files** | 310+ |
| **Go Tests** | 266 files |
| **JS/TS Tests** | 44 files |
| **E2E Specs** | 52 files |
| **Test Frameworks** | Jest, Playwright, Go, pytest |
| **Coverage Reports** | ✅ Active |
| **CI/CD** | ✅ GitHub Actions |
| **Overall Score** | 78/100 (Good) |

---

## Running Tests

### Unit Tests
```bash
# All unit tests
npm run test

# Unit tests with coverage
npm run test -- --coverage

# Specific unit tests
npm run test:unit

# Watch mode
npm run test -- --watch
```

### Integration Tests
```bash
# All integration tests
npm run test:integration

# By category
npm run test:integration:system
npm run test:integration:mle-star
npm run test:integration:performance

# With coverage
npm run test:integration:coverage

# Parallel execution
npm run test:integration:parallel

# Debug mode
npm run test:integration:debug

# Docker environment
npm run test:integration:docker
```

### E2E Tests (Playwright)
```bash
# All E2E tests
npm run test:e2e:playwright

# Headed mode (see browser)
npm run test:e2e:headed

# Debug mode
npm run test:e2e:debug

# UI mode
npm run test:e2e:ui

# Specific browser
npm run test:e2e:chromium
npm run test:e2e:firefox
npm run test:e2e:webkit

# Mobile
npm run test:e2e:mobile

# CI mode
npm run test:e2e:ci

# View report
npm run test:e2e:report
```

### Go Tests (Backend)
```bash
# All Go tests
cd backend/core && go test ./...

# With coverage
go test -cover ./...

# With race detection
go test -race ./...

# Verbose
go test -v ./...

# Specific package
go test ./network/dwcp/...
```

### Combined
```bash
# All tests (unit + integration)
npm run test:all

# CI tests (with coverage)
npm run test:ci
```

---

## Test Structure

```
tests/
├── unit/               # Fast, isolated tests
├── integration/        # Component interaction tests
├── e2e/               # Full user journey tests
├── performance/       # Load and stress tests
├── chaos/            # Chaos engineering
├── security/         # Security tests
├── compliance/       # Compliance tests
└── setup.js          # Global test setup
```

---

## Key Files

| File | Purpose |
|------|---------|
| `tests/setup.js` | Global Jest setup |
| `playwright.config.ts` | Playwright configuration |
| `tests/e2e/global-setup.ts` | E2E test setup |
| `tests/e2e/global-teardown.ts` | E2E test cleanup |
| `tests/integration/runners/test-runner.js` | Custom integration test runner |
| `tests/utils/initialization-helpers.js` | Test utilities and mocks |
| `.github/workflows/ci.yml` | CI/CD pipeline |

---

## Coverage Reports

### Viewing Coverage
```bash
# Generate coverage
npm run test -- --coverage

# View HTML report
open coverage/index.html

# View LCOV report
cat coverage/lcov.info
```

### Coverage Locations
- HTML: `coverage/index.html`
- LCOV: `coverage/lcov.info`
- LCOV HTML: `coverage/lcov-report/`

---

## Test Utilities

### Global Test Utils (tests/setup.js)
```javascript
// Sleep
await global.testUtils.sleep(1000);

// Generate UUID
const id = global.testUtils.generateUUID();

// Mock date
const restore = global.testUtils.mockDate('2024-01-01');
// ... test code ...
restore();

// Wait for condition
await global.testUtils.expectEventually(
  () => expect(value).toBe(expected),
  5000,  // timeout
  100    // interval
);
```

### Test Helpers (tests/utils/initialization-helpers.js)
```javascript
const {
  createMockDatabase,
  createMockCache,
  createMockLogger,
  createMockVault,
  waitForCondition,
  measureTime,
  measureMemory
} = require('../utils/initialization-helpers');

// Create mocks
const db = createMockDatabase();
const cache = createMockCache();

// Measure performance
const { result, duration } = await measureTime(async () => {
  // ... test code ...
});
```

---

## Page Object Model (E2E)

### Using Page Objects
```typescript
import { test, expect } from '@playwright/test';
import { LoginPage } from './pages/auth/login-page';
import { DashboardPage } from './pages/dashboard-page';

test('user login', async ({ page }) => {
  const loginPage = new LoginPage(page);
  await loginPage.goto();
  await loginPage.login('user@example.com', 'password123');

  const dashboard = new DashboardPage(page);
  await expect(dashboard.welcomeMessage).toBeVisible();
});
```

### Available Page Objects
- `tests/e2e/pages/auth/` - Authentication pages
- `tests/e2e/pages/vms/` - VM management pages
- `tests/e2e/pages/cluster/` - Cluster pages
- `tests/e2e/pages/monitoring/` - Monitoring pages
- `tests/e2e/pages/migration/` - Migration pages

---

## Environment Variables

### Test Configuration
```bash
# Test mode
NODE_ENV=test
NOVACRON_TEST=true
NOVACRON_LOG_LEVEL=error

# Integration tests
NOVACRON_TEST_COVERAGE=true
NOVACRON_TEST_PARALLEL=true
NOVACRON_TEST_DEBUG=true
NOVACRON_USE_DOCKER=true
NOVACRON_TEST_MODE=integration

# E2E tests
PLAYWRIGHT_BASE_URL=http://localhost:3000
PLAYWRIGHT_SKIP_AUTH=false
TEST_ENV=local
TEST_USER=test@novacron.com
TEST_PASSWORD=testpassword123
UPDATE_SNAPSHOTS=false
ARCHIVE_ARTIFACTS=false
ARTIFACT_RETENTION_DAYS=7
```

---

## CI/CD Pipeline

### GitHub Actions Jobs
1. **go-backend** - Go tests with race detection
2. **python-services** - pytest with coverage
3. **frontend** - npm test, lint, build
4. **docker-build** - Build Docker images
5. **security-scan** - Trivy vulnerability scanning

### Coverage Upload
- Codecov integration for all platforms
- Coverage flags: `backend-go`, `backend-python`, `frontend`

---

## Test Patterns

### Unit Test Pattern
```javascript
describe('Component', () => {
  let service;
  let mockDependency;

  beforeEach(() => {
    mockDependency = createMockDependency();
    service = new Service(mockDependency);
  });

  it('should do something', () => {
    const result = service.doSomething();
    expect(result).toBeDefined();
  });
});
```

### Integration Test Pattern
```javascript
describe('Integration', () => {
  let testEnv;

  beforeAll(async () => {
    testEnv = new TestEnvironment();
    await testEnv.setup();
  });

  afterAll(async () => {
    await testEnv.cleanup();
  });

  it('should integrate components', async () => {
    // ... test code ...
  });
});
```

### E2E Test Pattern
```typescript
test.describe('Feature', () => {
  test.beforeEach(async ({ page }) => {
    // Setup
  });

  test('should complete user journey', async ({ page }) => {
    // ... test code ...
  });
});
```

---

## Debugging Tests

### Jest
```bash
# Debug specific test
node --inspect-brk node_modules/.bin/jest tests/unit/example.test.js

# Run single test
npm run test -- -t "test name"

# Run with debug logs
DEBUG=true npm run test
```

### Playwright
```bash
# Debug mode (step through)
npm run test:e2e:debug

# UI mode (visual debugger)
npm run test:e2e:ui

# Headed mode (see browser)
npm run test:e2e:headed

# Trace viewer
npx playwright show-trace tests/e2e/reports/traces/trace.zip
```

### Go
```bash
# Verbose output
go test -v ./...

# Run specific test
go test -run TestName ./...

# Debug with delve
dlv test ./... -- -test.run TestName
```

---

## Best Practices

### ✅ Do
- Write descriptive test names
- Use beforeEach/afterEach for setup/cleanup
- Mock external dependencies
- Test edge cases and error conditions
- Keep tests fast and focused
- Use Page Object Model for E2E tests
- Check coverage reports regularly

### ❌ Don't
- Don't test implementation details
- Don't share state between tests
- Don't use hardcoded delays (use waitFor*)
- Don't skip cleanup
- Don't commit .only() or .skip()
- Don't test third-party libraries

---

## Common Issues

### Test Timeout
```javascript
// Increase timeout for specific test
jest.setTimeout(60000);

// Or in test
test('slow test', async () => {
  // ...
}, 60000);
```

### Flaky Tests
```javascript
// Use retries (Playwright)
test.describe.configure({ retries: 2 });

// Use proper waits
await page.waitForSelector('[data-testid="element"]');
await page.waitForLoadState('networkidle');
```

### Memory Leaks
```javascript
// Cleanup after tests
afterEach(() => {
  jest.clearAllMocks();
  jest.clearAllTimers();
});
```

---

## Resources

### Documentation
- [Full Assessment](/home/kp/novacron/docs/testing/TEST-INFRASTRUCTURE-ASSESSMENT.md)
- [Integration Test Guide](/home/kp/novacron/tests/integration/INTEGRATION_TEST_GUIDE.md)
- [Testing Guide](/home/kp/novacron/tests/integration/TESTING_GUIDE.md)

### External
- [Jest Documentation](https://jestjs.io/docs/getting-started)
- [Playwright Documentation](https://playwright.dev/)
- [Go Testing Package](https://pkg.go.dev/testing)

---

**Need Help?** Refer to the full assessment document for detailed analysis and recommendations.
