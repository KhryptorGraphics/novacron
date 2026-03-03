# CI/CD Documentation for E2E Tests

This guide explains how E2E tests integrate with continuous integration and deployment pipelines.

## Table of Contents

- [GitHub Actions Workflow](#github-actions-workflow)
- [Running Tests in CI](#running-tests-in-ci)
- [Viewing Test Results](#viewing-test-results)
- [Debugging CI Failures](#debugging-ci-failures)
- [Performance Optimization](#performance-optimization)
- [Best Practices](#best-practices)

## GitHub Actions Workflow

### Workflow File

**.github/workflows/e2e-tests.yml:**

```yaml
name: E2E Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  schedule:
    # Run nightly at 2 AM UTC
    - cron: '0 2 * * *'
  workflow_dispatch:
    inputs:
      browser:
        description: 'Browser to test'
        required: false
        default: 'all'
        type: choice
        options:
          - all
          - chromium
          - firefox
          - webkit

env:
  NODE_VERSION: '18'
  PLAYWRIGHT_VERSION: '1.40.0'

jobs:
  test:
    name: E2E Tests (${{ matrix.browser }})
    runs-on: ubuntu-latest
    timeout-minutes: 60

    strategy:
      fail-fast: false
      matrix:
        browser: [chromium, firefox, webkit]
        shard: [1, 2, 3, 4]

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Install Playwright browsers
        run: npx playwright install --with-deps ${{ matrix.browser }}

      - name: Run E2E tests
        run: |
          npx playwright test \
            --project=${{ matrix.browser }} \
            --shard=${{ matrix.shard }}/${{ strategy.job-total }}
        env:
          CI: true
          BASE_URL: http://localhost:3000
          API_URL: http://localhost:8080/api

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: playwright-report-${{ matrix.browser }}-${{ matrix.shard }}
          path: |
            playwright-report/
            test-results/
          retention-days: 30

      - name: Upload traces
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: playwright-traces-${{ matrix.browser }}-${{ matrix.shard }}
          path: test-results/**/trace.zip
          retention-days: 7

      - name: Publish test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results-${{ matrix.browser }}-${{ matrix.shard }}
          path: test-results/results.json

  merge-reports:
    name: Merge Test Reports
    needs: test
    runs-on: ubuntu-latest
    if: always()

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}

      - name: Download all reports
        uses: actions/download-artifact@v4
        with:
          path: all-reports
          pattern: playwright-report-*

      - name: Merge reports
        run: |
          npx playwright merge-reports --reporter html all-reports

      - name: Upload merged report
        uses: actions/upload-artifact@v4
        with:
          name: playwright-report-merged
          path: playwright-report/
          retention-days: 30

      - name: Deploy report to GitHub Pages
        if: github.ref == 'refs/heads/main'
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./playwright-report
          destination_dir: e2e-reports/${{ github.run_number }}

  notify:
    name: Notify Results
    needs: [test, merge-reports]
    runs-on: ubuntu-latest
    if: always()

    steps:
      - name: Send Slack notification
        if: failure()
        uses: slackapi/slack-github-action@v1
        with:
          webhook-url: ${{ secrets.SLACK_WEBHOOK }}
          payload: |
            {
              "text": "E2E Tests Failed",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "❌ E2E tests failed on `${{ github.ref }}`\n<${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}|View Run>"
                  }
                }
              ]
            }
```

### Workflow for Pull Requests

**.github/workflows/pr-e2e.yml:**

```yaml
name: PR E2E Tests

on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  changed-files:
    name: Detect Changed Files
    runs-on: ubuntu-latest
    outputs:
      run-e2e: ${{ steps.changes.outputs.run-e2e }}
    steps:
      - uses: actions/checkout@v4
      - uses: dorny/paths-filter@v2
        id: changes
        with:
          filters: |
            run-e2e:
              - 'frontend/**'
              - 'backend/**'
              - 'tests/e2e/**'
              - 'package.json'
              - 'playwright.config.js'

  e2e-smoke:
    name: E2E Smoke Tests
    needs: changed-files
    if: needs.changed-files.outputs.run-e2e == 'true'
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Install Playwright
        run: npx playwright install --with-deps chromium

      - name: Run smoke tests
        run: npx playwright test --grep @smoke --project=chromium

      - name: Comment PR
        if: always()
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse(
              fs.readFileSync('test-results/results.json', 'utf8')
            );

            const body = `## 🧪 E2E Test Results

            - ✅ Passed: ${results.passed}
            - ❌ Failed: ${results.failed}
            - ⏭️ Skipped: ${results.skipped}

            [View Full Report](${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }})
            `;

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: body
            });
```

## Running Tests in CI

### Environment Variables

Configure these secrets in GitHub repository settings:

```bash
# Required
BASE_URL=http://localhost:3000
API_URL=http://localhost:8080/api

# Optional
TEST_USER_EMAIL=test@example.com
TEST_USER_PASSWORD=<stored-in-secrets>
SLACK_WEBHOOK=<stored-in-secrets>
```

### CI-Specific Configuration

**playwright.config.js:**

```javascript
const config = {
  // More retries in CI
  retries: process.env.CI ? 2 : 0,

  // Fewer workers in CI (resource-constrained)
  workers: process.env.CI ? 2 : undefined,

  // Different reporter in CI
  reporter: process.env.CI
    ? [
        ['html'],
        ['junit', { outputFile: 'test-results/junit.xml' }],
        ['json', { outputFile: 'test-results/results.json' }],
      ]
    : [['html'], ['list']],

  // CI-specific settings
  use: {
    baseURL: process.env.BASE_URL || 'http://localhost:3000',
    trace: process.env.CI ? 'on-first-retry' : 'retain-on-failure',
    video: process.env.CI ? 'retain-on-failure' : 'off',
    screenshot: 'only-on-failure',
  },
};

module.exports = config;
```

### Sharding Tests

Split tests across multiple machines for faster execution:

```yaml
strategy:
  matrix:
    shard: [1, 2, 3, 4]

steps:
  - name: Run tests
    run: npx playwright test --shard=${{ matrix.shard }}/4
```

### Parallel Execution

```yaml
strategy:
  matrix:
    browser: [chromium, firefox, webkit]

steps:
  - name: Run tests
    run: npx playwright test --project=${{ matrix.browser }}
```

## Viewing Test Results

### HTML Report

The HTML report is uploaded as an artifact:

1. Go to Actions tab in GitHub
2. Click on workflow run
3. Download `playwright-report-merged` artifact
4. Extract and open `index.html`

### GitHub Pages

Reports are automatically deployed to GitHub Pages:

```
https://your-org.github.io/novacron/e2e-reports/<run-number>
```

### Test Summary

View test summary in Actions UI:

```yaml
- name: Publish test summary
  if: always()
  uses: test-summary/action@v2
  with:
    paths: test-results/junit.xml
```

### Artifacts

Download artifacts from workflow run:

- `playwright-report-*`: HTML reports per shard
- `playwright-traces-*`: Trace files for failed tests
- `test-results-*`: JSON results for analysis

## Debugging CI Failures

### 1. View Trace Files

```bash
# Download trace artifact from GitHub Actions
unzip playwright-traces-chromium-1.zip

# View trace
npx playwright show-trace test-results/path-to-trace.zip
```

### 2. Enable Debug Logging

```yaml
- name: Run tests
  run: DEBUG=pw:api npx playwright test
  env:
    DEBUG: 'pw:api'
```

### 3. SSH into Runner (GitHub Actions)

```yaml
- name: Setup tmate session
  if: failure()
  uses: mxschmitt/action-tmate@v3
  timeout-minutes: 15
```

### 4. Run Tests Locally with CI Config

```bash
# Simulate CI environment
CI=true npx playwright test --project=chromium --shard=1/4

# Use CI reporter
CI=true npx playwright test --reporter=json
```

### 5. Check Screenshots

Screenshots are automatically captured on failure:

```yaml
- name: Upload screenshots
  if: failure()
  uses: actions/upload-artifact@v4
  with:
    name: screenshots
    path: test-results/**/*.png
```

### 6. Review Video Recordings

```yaml
- name: Upload videos
  if: failure()
  uses: actions/upload-artifact@v4
  with:
    name: videos
    path: test-results/**/*.webm
```

### 7. Common CI Issues

**Issue: Tests timeout in CI**

```javascript
// playwright.config.js
timeout: process.env.CI ? 60000 : 30000, // Longer timeout in CI
```

**Issue: Flaky tests in CI**

```javascript
// Increase retries
retries: process.env.CI ? 3 : 0,

// Or use test.retry
test.describe(() => {
  test.describe.configure({ retries: 2 });

  test('potentially flaky test', async ({ page }) => {
    // Test code
  });
});
```

**Issue: Resource constraints**

```yaml
# Reduce parallel workers
workers: process.env.CI ? 2 : 4

# Or run tests sequentially
- name: Run tests
  run: npx playwright test --workers=1
```

## Performance Optimization

### 1. Cache Dependencies

```yaml
- name: Setup Node.js
  uses: actions/setup-node@v4
  with:
    node-version: '18'
    cache: 'npm'

- name: Cache Playwright browsers
  uses: actions/cache@v3
  with:
    path: ~/.cache/ms-playwright
    key: playwright-${{ env.PLAYWRIGHT_VERSION }}
```

### 2. Install Only Required Browsers

```yaml
# Instead of installing all browsers
- run: npx playwright install --with-deps

# Install only what's needed
- run: npx playwright install --with-deps ${{ matrix.browser }}
```

### 3. Use Dependency Caching

```yaml
- name: Cache node_modules
  uses: actions/cache@v3
  with:
    path: node_modules
    key: ${{ runner.os }}-node-${{ hashFiles('package-lock.json') }}
```

### 4. Optimize Test Execution

```javascript
// playwright.config.js
fullyParallel: true, // Run tests in parallel
workers: process.env.CI ? 2 : 4,

// Skip slow tests in PR
grep: process.env.PR ? /@fast/ : undefined,
```

### 5. Smart Test Selection

```yaml
- name: Get changed files
  id: changed
  run: |
    echo "files=$(git diff --name-only HEAD~1 | grep -E '\\.(js|ts|jsx|tsx)$' | xargs)" >> $GITHUB_OUTPUT

- name: Run affected tests
  if: steps.changed.outputs.files
  run: npx playwright test --grep-invert @slow
```

### 6. Use Test Sharding

```yaml
strategy:
  matrix:
    shard: [1, 2, 3, 4] # Split into 4 shards

steps:
  - run: npx playwright test --shard=${{ matrix.shard }}/4
```

### 7. Reuse Authentication State

```javascript
// global-setup.js
module.exports = async () => {
  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();

  // Login once
  await page.goto('http://localhost:3000/login');
  await page.fill('#email', process.env.TEST_USER_EMAIL);
  await page.fill('#password', process.env.TEST_USER_PASSWORD);
  await page.click('button[type="submit"]');

  // Save auth state
  await context.storageState({ path: 'auth.json' });
  await browser.close();
};

// Use in tests
test.use({ storageState: 'auth.json' });
```

## Best Practices

### 1. Fail Fast Strategy

```yaml
strategy:
  fail-fast: false # Don't cancel other jobs on first failure
```

### 2. Conditional Test Execution

```yaml
# Run E2E only if relevant files changed
- name: Check for changes
  uses: dorny/paths-filter@v2
  id: changes
  with:
    filters: |
      e2e:
        - 'frontend/**'
        - 'backend/**'
        - 'tests/e2e/**'

- name: Run E2E tests
  if: steps.changes.outputs.e2e == 'true'
  run: npx playwright test
```

### 3. Matrix Testing

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    node: [18, 20]
    browser: [chromium, firefox]
```

### 4. Scheduled Runs

```yaml
on:
  schedule:
    # Run full suite nightly
    - cron: '0 2 * * *'
  push:
    # Run smoke tests on push
    branches: [main]
```

### 5. Manual Workflow Dispatch

```yaml
on:
  workflow_dispatch:
    inputs:
      test-suite:
        description: 'Test suite to run'
        required: true
        type: choice
        options:
          - all
          - smoke
          - regression
```

### 6. Test Result Notifications

```yaml
- name: Send notification
  if: failure()
  uses: slackapi/slack-github-action@v1
  with:
    webhook-url: ${{ secrets.SLACK_WEBHOOK }}
    payload: |
      {
        "text": "E2E Tests Failed",
        "attachments": [{
          "color": "danger",
          "fields": [
            {"title": "Branch", "value": "${{ github.ref }}", "short": true},
            {"title": "Commit", "value": "${{ github.sha }}", "short": true}
          ]
        }]
      }
```

### 7. Test Coverage Tracking

```yaml
- name: Generate coverage
  run: npx playwright test --reporter=html,coverage

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage/lcov.info
```

### 8. Dependency Updates

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: npm
    directory: "/"
    schedule:
      interval: weekly
    labels:
      - dependencies
      - e2e-tests
```

## Monitoring and Metrics

### Track Test Metrics

```javascript
// Generate metrics report
const results = require('./test-results/results.json');

const metrics = {
  totalTests: results.total,
  passed: results.passed,
  failed: results.failed,
  flaky: results.flaky,
  duration: results.duration,
  avgDuration: results.duration / results.total,
};

console.log(JSON.stringify(metrics));
```

### Grafana Dashboard

Send metrics to Grafana:

```yaml
- name: Send metrics
  if: always()
  run: |
    curl -X POST ${{ secrets.GRAFANA_URL }}/api/metrics \
      -H "Authorization: Bearer ${{ secrets.GRAFANA_TOKEN }}" \
      -d @test-results/metrics.json
```

## Troubleshooting Guide

### Common CI/CD Issues

1. **Tests pass locally but fail in CI**
   - Check environment variables
   - Verify browser versions match
   - Review CI logs for timing differences

2. **Slow test execution in CI**
   - Enable sharding
   - Reduce parallel workers
   - Cache dependencies

3. **Flaky tests in CI**
   - Increase retries
   - Add explicit waits
   - Use test.step for better debugging

4. **Out of memory errors**
   - Reduce workers
   - Limit browser instances
   - Increase runner memory

---

**Last Updated**: 2025-01-10
**GitHub Actions Version**: v4
**Playwright Version**: 1.40+
