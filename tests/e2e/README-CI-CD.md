# E2E Testing CI/CD Integration Guide

This guide explains the CI/CD integration for Playwright E2E tests using GitHub Actions.

## 📋 Table of Contents

- [Overview](#overview)
- [Workflows](#workflows)
- [Setup](#setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

The E2E testing infrastructure includes three main GitHub Actions workflows:

1. **e2e-tests.yml** - Main E2E test workflow (PR and push)
2. **e2e-nightly.yml** - Comprehensive nightly test suite
3. **e2e-visual-regression.yml** - Visual regression testing

## 🔄 Workflows

### 1. Main E2E Tests (`e2e-tests.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

**Features:**
- ✅ Parallel test execution with sharding (4 shards per browser)
- ✅ Multi-browser testing (Chromium, Firefox, WebKit)
- ✅ Automatic retry on failure
- ✅ Test result artifacts (screenshots, videos, traces)
- ✅ PR comments with test results
- ✅ Failure notifications

**Test Matrix:**
- Browsers: Chromium, Firefox, WebKit
- OS: Ubuntu (optional: Windows, macOS)
- Shards: 4 parallel shards per browser
- Total jobs: 12 (3 browsers × 4 shards)

### 2. Nightly Tests (`e2e-nightly.yml`)

**Triggers:**
- Scheduled: Daily at 2 AM UTC
- Manual workflow dispatch

**Features:**
- ✅ Comprehensive test coverage
- ✅ Extended timeout settings (120 minutes)
- ✅ Performance benchmarking
- ✅ Accessibility testing
- ✅ Mobile device testing
- ✅ Monitoring integration (Grafana, Prometheus)
- ✅ Automatic issue creation on failure
- ✅ Long-term report archiving (365 days)

**Test Matrix:**
- Browsers: Chromium, Firefox, WebKit, Mobile Chrome, Mobile Safari
- Extended timeouts for slow tests
- Performance metrics collection

### 3. Visual Regression Tests (`e2e-visual-regression.yml`)

**Triggers:**
- Push to `main` or `develop` (frontend changes)
- Pull requests (frontend changes)
- Manual workflow dispatch

**Features:**
- ✅ Screenshot comparison across viewports
- ✅ Theme testing (light/dark)
- ✅ Baseline management
- ✅ Diff image generation
- ✅ Automatic baseline updates (main branch)
- ✅ Visual report generation

**Test Matrix:**
- Viewports: Desktop, Tablet, Mobile
- Themes: Light, Dark
- Total jobs: 6 (3 viewports × 2 themes)

## 🚀 Setup

### Prerequisites

1. GitHub repository with Actions enabled
2. Required secrets (if any):
   - `GITHUB_TOKEN` (automatically provided)

### Initial Setup

1. **Copy workflow files:**
   ```bash
   # Workflows are already in .github/workflows/
   ls -la .github/workflows/e2e-*.yml
   ```

2. **Create test environment:**
   ```bash
   chmod +x tests/e2e/scripts/*.sh
   ./tests/e2e/scripts/setup-test-env.sh
   ```

3. **Configure Docker environment:**
   ```bash
   cd tests/e2e/docker
   cp .env.example .env
   # Edit .env as needed
   ```

4. **Verify local setup:**
   ```bash
   docker-compose -f tests/e2e/docker/docker-compose.yml up -d
   cd tests/e2e
   npx playwright test --project=chromium
   ```

## ⚙️ Configuration

### Workflow Configuration

#### Main E2E Tests

Edit `.github/workflows/e2e-tests.yml`:

```yaml
env:
  NODE_VERSION: '18'  # Node.js version

strategy:
  matrix:
    browser: [chromium, firefox, webkit]
    shardIndex: [1, 2, 3, 4]  # Adjust for more/fewer shards
    shardTotal: [4]
```

#### Nightly Tests

Edit `.github/workflows/e2e-nightly.yml`:

```yaml
on:
  schedule:
    - cron: '0 2 * * *'  # Adjust schedule

env:
  EXTENDED_TIMEOUT_MINUTES: 120  # Adjust timeout
```

#### Visual Regression

Edit `.github/workflows/e2e-visual-regression.yml`:

```yaml
strategy:
  matrix:
    viewport: ['desktop', 'tablet', 'mobile']
    theme: ['light', 'dark']
```

### Environment Variables

**Docker Environment (tests/e2e/docker/.env):**

```bash
# Application
NODE_ENV=test
FRONTEND_PORT=3000
BACKEND_PORT=8080

# Database
DATABASE_URL=postgresql://test:test@postgres:5432/test_db

# Redis
REDIS_URL=redis://redis:6379

# Playwright
WORKERS=4
TIMEOUT=30000
```

**GitHub Actions Environment:**

Set in workflow files or repository secrets:

```yaml
env:
  TEST_BASE_URL: http://localhost:3000
  API_BASE_URL: http://localhost:8080
  CI: true
```

## 📖 Usage

### Running Tests Locally

**Quick test:**
```bash
cd tests/e2e
npx playwright test
```

**Specific browser:**
```bash
npx playwright test --project=chromium
```

**With UI mode:**
```bash
npx playwright test --ui
```

**Debug mode:**
```bash
npx playwright test --debug
```

### Running in Docker

**Start services:**
```bash
cd tests/e2e/docker
docker-compose up -d
```

**Run tests:**
```bash
docker-compose run playwright npx playwright test
```

**View logs:**
```bash
docker-compose logs -f
```

**Stop services:**
```bash
docker-compose down -v
```

### Triggering CI/CD Workflows

**Automatic triggers:**
- Push to main/develop branches
- Create pull request
- Wait for scheduled nightly run

**Manual triggers:**

1. Go to Actions tab in GitHub
2. Select workflow (e.g., "E2E Tests")
3. Click "Run workflow"
4. Choose branch and options
5. Click "Run workflow" button

### Updating Visual Baselines

**Manual update:**

1. Go to Actions → Visual Regression Tests
2. Click "Run workflow"
3. Check "Update baseline screenshots"
4. Run workflow
5. Baselines will be committed automatically

**Automatic update:**

Visual baselines are automatically updated on push to `main` branch.

## 📊 Viewing Results

### Test Reports

**In GitHub Actions:**

1. Go to Actions tab
2. Click on workflow run
3. Scroll to "Artifacts" section
4. Download:
   - `playwright-report-*` - HTML reports
   - `screenshots-*` - Test screenshots
   - `videos-*` - Test videos
   - `traces-*` - Playwright traces

**Local viewing:**

```bash
cd tests/e2e
npx playwright show-report
```

### PR Comments

Test results are automatically commented on pull requests:

- ✅ Test summary
- 📊 Pass/fail statistics
- 🔗 Link to full report
- ⚠️ Failure notifications

### Status Badges

Add to README.md:

```markdown
## Test Status

![E2E Tests](https://github.com/YOUR_ORG/YOUR_REPO/workflows/E2E%20Tests/badge.svg)
![Nightly Tests](https://github.com/YOUR_ORG/YOUR_REPO/workflows/E2E%20Nightly%20Tests/badge.svg)
![Visual Tests](https://github.com/YOUR_ORG/YOUR_REPO/workflows/Visual%20Regression%20Tests/badge.svg)
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Tests timeout in CI

**Solution:**
```yaml
# Increase timeout in workflow
timeout-minutes: 60

# Or in Playwright config
use: {
  actionTimeout: 15000,
  navigationTimeout: 30000,
}
```

#### 2. Flaky tests

**Solution:**
```typescript
// In test file
test.describe.configure({ retries: 2 });

// Or in workflow
npx playwright test --retries=2
```

#### 3. Services not starting

**Solution:**
```bash
# Check service health
docker-compose ps
docker-compose logs postgres

# Increase wait time in workflow
timeout 60 bash -c 'until curl -f http://localhost:3000; do sleep 2; done'
```

#### 4. Out of memory errors

**Solution:**
```yaml
# In workflow
env:
  NODE_OPTIONS: --max-old-space-size=4096

# Or limit workers
WORKERS: 2
```

#### 5. Screenshot differences

**Solution:**
```bash
# Update baselines locally
npx playwright test --update-snapshots

# Or trigger update workflow in GitHub
```

### Debug Mode

**Enable debug logs in workflow:**

```yaml
env:
  DEBUG: pw:api
  PWDEBUG: 1
```

**Run with debugging:**

```bash
cd tests/e2e
PWDEBUG=1 npx playwright test
```

### Artifacts Not Uploaded

Check workflow configuration:

```yaml
- name: Upload artifacts
  if: always()  # Ensure this runs even on failure
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: tests/e2e/test-results/
    retention-days: 7
```

## 🎓 Best Practices

1. **Keep tests independent** - Each test should be able to run in isolation
2. **Use proper waits** - Wait for elements, not arbitrary timeouts
3. **Clean test data** - Reset state between tests
4. **Descriptive test names** - Make failures easy to understand
5. **Tag tests** - Use tags for selective test execution
6. **Monitor performance** - Track test execution time
7. **Update baselines** - Keep visual baselines up to date

## 📚 Additional Resources

- [Playwright Documentation](https://playwright.dev)
- [GitHub Actions Documentation](https://docs.github.com/actions)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

## 🤝 Support

For issues or questions:

1. Check [Troubleshooting](#troubleshooting) section
2. Review workflow logs in GitHub Actions
3. Check Docker logs: `docker-compose logs`
4. Open an issue in the repository

---

**Generated by NovaCron E2E Testing Infrastructure**
