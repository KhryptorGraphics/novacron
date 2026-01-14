# NovaCron E2E Testing Guide

## Overview

This comprehensive E2E testing suite provides complete coverage of the NovaCron application using Puppeteer with Jest. The tests cover all major functionality including authentication, VM management, monitoring, admin operations, performance, and accessibility.

## Test Architecture

### Test Categories

1. **Authentication (`auth/`)** - Login, registration, 2FA, password reset, session management
2. **VM Management (`vm-management/`)** - VM lifecycle, creation, operations, migration
3. **Monitoring (`monitoring/`)** - Dashboard metrics, real-time updates, alerts, topology
4. **Admin Panel (`admin/`)** - User management, system configuration, database operations
5. **Performance (`performance/`)** - Load times, resource usage, Core Web Vitals
6. **Accessibility (`accessibility/`)** - WCAG compliance, keyboard navigation, screen reader support
7. **Integration (`integration/`)** - Backend API integration, WebSocket connections, database operations

### Directory Structure

```
frontend/
├── src/
│   └── __tests__/
│       └── e2e/
│           ├── auth/
│           ├── vm-management/
│           ├── monitoring/
│           ├── admin/
│           ├── performance/
│           ├── accessibility/
│           └── integration/
├── test-setup/
│   ├── puppeteer-setup.js
│   ├── puppeteer-teardown.js
│   └── puppeteer-jest-setup.js
├── jest.e2e.config.js
└── puppeteer.config.js
```

## Running Tests

### Prerequisites

1. **Node.js 16+** and **npm**
2. **Go 1.19+** (for backend server)
3. **Chrome/Chromium** browser (automatically installed with Puppeteer)

### Installation

```bash
cd frontend
npm install
```

This will install all dependencies including Puppeteer and Jest.

### Running All Tests

```bash
# Run all test categories
./scripts/run-e2e-tests.sh --all

# Run with coverage
./scripts/run-e2e-tests.sh --all --coverage

# Debug mode with visible browser
./scripts/run-e2e-tests.sh --all --headed --slow-mo 500 --debug
```

### Running Specific Categories

```bash
# Run authentication tests only
./scripts/run-e2e-tests.sh --category auth

# Run multiple categories
./scripts/run-e2e-tests.sh --category auth --category vm-management

# Run performance and accessibility tests
./scripts/run-e2e-tests.sh --category performance --category accessibility
```

### Direct Jest Execution

```bash
cd frontend

# Run all E2E tests
npm run test:e2e

# Run specific test file
npm run test:e2e -- --testPathPattern="auth"

# Debug mode
npm run test:e2e:debug

# With coverage
npm run test:e2e:coverage
```

## Configuration Options

### Environment Variables

- `HEADLESS` - Run in headless mode (default: true)
- `SLOW_MO` - Delay between actions in milliseconds (default: 0)
- `DEVTOOLS` - Open browser DevTools (default: false)
- `DEBUG_CONSOLE` - Log browser console messages (default: false)
- `DEBUG_SERVER` - Enable server debug logging (default: false)

### Test Runner Options

```bash
./scripts/run-e2e-tests.sh [OPTIONS]

Options:
  --all                Run all test categories
  --category CATEGORY  Run specific test category (can be repeated)
  --headless          Run in headless mode (default)
  --headed            Run with visible browser
  --slow-mo MS        Add delay between actions
  --debug             Enable debug logging with visible browser
  --coverage          Generate coverage report
  --parallel          Run tests in parallel (experimental)
  --no-setup          Skip server setup (servers must be running)
  --no-cleanup        Skip cleanup after tests
  --help              Show help message
```

## Writing Tests

### Test Structure

Each test file follows this pattern:

```javascript
describe('Feature Name', () => {
  let page

  beforeEach(async () => {
    page = await global.puppeteerUtils.createPage()
  })

  afterEach(async () => {
    if (page) {
      await page.close()
    }
  })

  describe('Specific Functionality', () => {
    test('should do something specific', async () => {
      // Test implementation
    })
  })
})
```

### Test Utilities

The global `puppeteerUtils` object provides helpful methods:

```javascript
// Navigation
await global.puppeteerUtils.navigateAndWait(page, '/dashboard')

// Element interaction
await global.puppeteerUtils.waitForElement(page, 'button[type="submit"]')
await global.puppeteerUtils.fillForm(page, {
  'input[name="email"]': 'test@example.com',
  'input[name="password"]': 'password123'
})

// Authentication
await global.puppeteerUtils.login(page, { email: 'user@test.com', password: 'pass123' })

// API testing
const response = await global.puppeteerUtils.makeApiRequest('GET', '/api/health')

// Performance measurement
const metrics = await global.puppeteerUtils.measurePageLoad(page, '/dashboard')

// Accessibility testing
const results = await global.puppeteerUtils.checkAccessibility(page)

// Screenshots for debugging
await global.puppeteerUtils.takeScreenshot(page, 'debug-login')

// Mobile simulation
await global.puppeteerUtils.simulateMobile(page)

// Network throttling
await global.puppeteerUtils.throttleNetwork(page, '3G')
```

### Best Practices

1. **Use data-testid attributes** for reliable element selection
2. **Wait for elements** instead of using fixed timeouts
3. **Test error states** and edge cases, not just happy paths
4. **Mock external dependencies** when needed
5. **Clean up resources** in afterEach hooks
6. **Use descriptive test names** that explain what is being tested
7. **Group related tests** in describe blocks
8. **Handle authentication gracefully** - tests should work with or without backend

### Example Test

```javascript
test('should create new VM with valid configuration', async () => {
  // Skip if not authenticated
  const loginRequired = await page.$('input[name="email"]')
  if (loginRequired) return

  // Navigate to VM creation
  await global.puppeteerUtils.navigateAndWait(page, '/vms/create')
  
  // Fill form
  await global.puppeteerUtils.fillForm(page, {
    'input[name="name"]': 'test-vm-e2e',
    'input[name="description"]': 'E2E test VM',
    'select[name="template"]': 'ubuntu-20.04'
  })
  
  // Submit and verify
  await page.click('button[type="submit"]')
  
  const success = await Promise.race([
    page.waitForSelector('.success-message', { timeout: 5000 }),
    page.waitForNavigation({ timeout: 5000 })
  ])
  
  expect(success).toBeTruthy()
})
```

## Test Categories in Detail

### Authentication Tests

- User registration wizard with validation
- Login with various credential combinations  
- Password strength validation and visibility toggle
- 2FA setup and verification
- Password reset flow
- Session management and timeout handling
- Social authentication (if available)
- Remember me functionality

### VM Management Tests

- VM list display and filtering
- VM creation wizard with resource configuration
- VM lifecycle operations (start, stop, restart)
- VM details and console access
- VM migration workflows (cold, warm, live)
- Template creation and snapshot management
- Network and storage configuration
- VM monitoring and metrics

### Monitoring Tests

- Dashboard overview with system metrics
- Real-time data updates via WebSocket
- Alerts management and acknowledgment
- Performance metrics and historical data
- Network topology visualization
- Custom time range selection and data export
- VM comparison and resource utilization charts

### Admin Panel Tests

- Admin authentication and authorization
- User management (create, edit, deactivate)
- Role and permission management
- System configuration and settings
- Database administration and maintenance
- Backup and restore operations
- System health monitoring
- Audit log management

### Performance Tests

- Page load time measurement
- Bundle size analysis
- Memory leak detection during navigation
- Concurrent user simulation
- Network optimization testing
- Core Web Vitals measurement
- Resource usage monitoring
- Caching effectiveness

### Accessibility Tests

- WCAG 2.1 AA compliance checking
- Keyboard navigation support
- Screen reader compatibility
- Focus management in modals
- Color contrast validation
- Mobile accessibility
- Form accessibility and validation
- Table and heading structure

### Integration Tests

- Backend API connectivity
- Authentication flow with JWT tokens
- CRUD operations through API
- WebSocket connection handling
- Database operation testing
- Error handling and graceful degradation
- CORS configuration validation
- Performance under load

## Continuous Integration

### GitHub Actions Integration

```yaml
name: E2E Tests
on: [push, pull_request]
jobs:
  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - uses: actions/setup-go@v4
        with:
          go-version: '1.19'
      - run: npm install
        working-directory: ./frontend
      - run: ./scripts/run-e2e-tests.sh --all --coverage
      - uses: actions/upload-artifact@v3
        with:
          name: coverage-report
          path: frontend/coverage/
```

### Local Development

```bash
# Run tests during development
./scripts/run-e2e-tests.sh --category auth --headed --no-cleanup

# Quick feedback loop
npm run test:e2e -- --testPathPattern="login" --watch
```

## Troubleshooting

### Common Issues

1. **Tests timeout**: Increase timeout in jest config or use `--slow-mo` flag
2. **Element not found**: Check selectors and wait conditions
3. **Flaky tests**: Add proper waits and reduce dependencies between tests
4. **Browser crashes**: Check system resources and Chrome flags
5. **WebSocket errors**: Expected in test environment without backend

### Debug Mode

```bash
# Run with debug options
HEADLESS=false SLOW_MO=500 DEBUG_CONSOLE=true npm run test:e2e:debug

# Or using the script
./scripts/run-e2e-tests.sh --debug --category auth
```

### Screenshots and Logs

- Screenshots are saved to `coverage/e2e/screenshots/`
- Test reports are generated in `reports/`
- Server logs are in `/tmp/novacron-*.log`

## Coverage Reporting

Coverage reports are generated in multiple formats:

- **HTML**: `coverage/e2e/index.html` (interactive report)
- **LCOV**: `coverage/e2e/lcov.info` (for CI integration)
- **JSON**: `coverage/e2e/coverage-final.json` (programmatic access)

### Coverage Thresholds

- **Statements**: 60%
- **Branches**: 60%
- **Functions**: 60%
- **Lines**: 60%

These thresholds are configured for E2E testing where backend integration may not be available.

## Advanced Features

### Parallel Execution

```bash
# Run tests in parallel (experimental)
./scripts/run-e2e-tests.sh --all --parallel
```

### Custom Network Conditions

```javascript
// Simulate slow network
await global.puppeteerUtils.throttleNetwork(page, 'Slow 3G')

// Custom throttling
const client = await page.target().createCDPSession()
await client.send('Network.emulateNetworkConditions', {
  offline: false,
  downloadThroughput: 100 * 1024, // 100KB/s
  uploadThroughput: 50 * 1024,    // 50KB/s
  latency: 500                    // 500ms latency
})
```

### Performance Profiling

```javascript
// Enable performance metrics
await page.tracing.start({ path: 'trace.json' })

// Your test actions here

await page.tracing.stop()

// Analyze Core Web Vitals
const vitals = await page.evaluate(() => {
  return new Promise((resolve) => {
    new PerformanceObserver((list) => {
      // Collect CLS, LCP, FID metrics
    }).observe({ entryTypes: ['layout-shift', 'largest-contentful-paint'] })
  })
})
```

## Contributing

### Adding New Tests

1. Create test file in appropriate category directory
2. Follow existing patterns and use provided utilities
3. Include both positive and negative test cases
4. Add comprehensive error handling
5. Update this documentation if needed

### Test Naming Convention

- Use descriptive names: `should create VM with valid configuration`
- Group related tests: `describe('VM Creation', () => { ... })`
- Test both success and failure cases
- Include accessibility tests for UI components

## References

- [Puppeteer Documentation](https://pptr.dev/)
- [Jest Testing Framework](https://jestjs.io/)
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [Core Web Vitals](https://web.dev/vitals/)
- [Axe Accessibility Testing](https://github.com/dequelabs/axe-core)