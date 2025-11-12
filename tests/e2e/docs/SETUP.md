# E2E Testing Setup Guide

This guide walks you through setting up the E2E testing environment for NovaCron.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Configuration](#environment-configuration)
- [Browser Installation](#browser-installation)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Operating System:**
- Linux (Ubuntu 20.04+, Debian 11+, or similar)
- macOS 12+ (Monterey or later)
- Windows 10+ or Windows Server 2019+

**Hardware:**
- CPU: 2+ cores recommended
- RAM: 8GB minimum, 16GB recommended
- Disk: 2GB free space for browsers and dependencies

**Software:**
- Node.js 18.x or higher
- npm 9.x or higher
- Git 2.x or higher

### Check Your Environment

```bash
# Check Node.js version
node --version
# Should output: v18.x.x or higher

# Check npm version
npm --version
# Should output: 9.x.x or higher

# Check Git version
git --version
# Should output: git version 2.x.x or higher
```

### Install/Update Node.js

**Using nvm (Recommended):**

```bash
# Install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash

# Reload shell
source ~/.bashrc  # or ~/.zshrc for zsh

# Install Node.js 18
nvm install 18
nvm use 18
nvm alias default 18
```

**Using Package Manager:**

```bash
# Ubuntu/Debian
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# macOS (Homebrew)
brew install node@18

# Windows (Chocolatey)
choco install nodejs-lts
```

## Installation

### 1. Clone Repository (if needed)

```bash
git clone https://github.com/your-org/novacron.git
cd novacron
```

### 2. Install Dependencies

```bash
# Install all project dependencies
npm install

# This will install:
# - Playwright
# - Test dependencies
# - Project dependencies
```

### 3. Install Playwright Browsers

```bash
# Install all browsers (Chromium, Firefox, WebKit)
npx playwright install

# Or install specific browser
npx playwright install chromium
npx playwright install firefox
npx playwright install webkit
```

### 4. Install System Dependencies (Linux)

On Linux, Playwright requires additional system libraries:

```bash
# Ubuntu/Debian
npx playwright install-deps

# Or manually install dependencies
sudo apt-get install -y \
  libwoff1 \
  libopus0 \
  libwebp6 \
  libwebpdemux2 \
  libenchant1c2a \
  libgudev-1.0-0 \
  libsecret-1-0 \
  libhyphen0 \
  libgdk-pixbuf2.0-0 \
  libegl1 \
  libnotify4 \
  libxslt1.1 \
  libevent-2.1-7 \
  libgles2 \
  libvpx6 \
  libxcomposite1 \
  libatk1.0-0 \
  libatk-bridge2.0-0 \
  libepoxy0 \
  libgtk-3-0 \
  libharfbuzz-icu0
```

**Fedora/RHEL:**

```bash
sudo dnf install -y \
  alsa-lib \
  atk \
  cups-libs \
  gtk3 \
  ipa-gothic-fonts \
  libXcomposite \
  libXcursor \
  libXdamage \
  libXext \
  libXi \
  libXrandr \
  libXScrnSaver \
  libXtst \
  pango \
  xorg-x11-fonts-100dpi \
  xorg-x11-fonts-75dpi \
  xorg-x11-fonts-cyrillic \
  xorg-x11-fonts-misc \
  xorg-x11-fonts-Type1 \
  xorg-x11-utils
```

## Environment Configuration

### 1. Create Environment File

```bash
# Copy example environment file
cp tests/e2e/config/test.env.example tests/e2e/config/test.env

# Edit with your settings
nano tests/e2e/config/test.env
```

### 2. Configure Environment Variables

**tests/e2e/config/test.env:**

```bash
# Application URLs
BASE_URL=http://localhost:3000
API_URL=http://localhost:8080/api

# Test User Credentials
TEST_USER_EMAIL=test.user@example.com
TEST_USER_PASSWORD=SecureTestPassword123!
TEST_ADMIN_EMAIL=admin@example.com
TEST_ADMIN_PASSWORD=SecureAdminPassword123!

# Database (for test data setup)
TEST_DB_HOST=localhost
TEST_DB_PORT=5432
TEST_DB_NAME=novacron_test
TEST_DB_USER=postgres
TEST_DB_PASSWORD=postgres

# Test Configuration
HEADLESS=true
SLOW_MO=0
TIMEOUT=30000
SCREENSHOT_ON_FAILURE=true
VIDEO_ON_FAILURE=true
TRACE_ON_FAILURE=true

# CI Environment
CI=false
```

### 3. Playwright Configuration

Edit **playwright.config.js** to customize test behavior:

```javascript
const { defineConfig, devices } = require('@playwright/test');

module.exports = defineConfig({
  // Test directory
  testDir: './tests/e2e/specs',

  // Test timeout
  timeout: 30000,

  // Expect timeout
  expect: {
    timeout: 5000
  },

  // Run tests in parallel
  fullyParallel: true,

  // Fail build on CI if tests were accidentally marked as skip
  forbidOnly: !!process.env.CI,

  // Retry on CI
  retries: process.env.CI ? 2 : 0,

  // Parallel workers
  workers: process.env.CI ? 1 : undefined,

  // Reporter
  reporter: [
    ['html'],
    ['list'],
    ['json', { outputFile: 'test-results/results.json' }]
  ],

  // Shared settings
  use: {
    baseURL: process.env.BASE_URL || 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },

  // Browser projects
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
  ],

  // Web server configuration
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:3000',
    reuseExistingServer: !process.env.CI,
    timeout: 120000,
  },
});
```

## Browser Installation

### Install All Browsers

```bash
# Install Chromium, Firefox, and WebKit
npx playwright install
```

### Install Specific Browser

```bash
# Chromium only
npx playwright install chromium

# Firefox only
npx playwright install firefox

# WebKit only
npx playwright install webkit
```

### Browser Locations

Browsers are installed in:

**Linux/macOS:**
```
~/.cache/ms-playwright/
```

**Windows:**
```
%USERPROFILE%\AppData\Local\ms-playwright\
```

### Update Browsers

```bash
# Update Playwright and browsers
npm install @playwright/test@latest
npx playwright install
```

### Using System Browsers

You can configure Playwright to use system-installed browsers:

```javascript
// playwright.config.js
use: {
  channel: 'chrome', // Use system Chrome instead of Chromium
}
```

## Verification

### 1. Verify Installation

```bash
# Check Playwright installation
npx playwright --version

# List installed browsers
npx playwright install --dry-run
```

### 2. Run Test Suite

```bash
# Run all tests
npm run test:e2e

# Run tests in headed mode to see the browser
npm run test:e2e:headed

# Run a single test file
npx playwright test tests/e2e/specs/auth.spec.js
```

### 3. Generate Test Report

```bash
# Run tests and open HTML report
npm run test:e2e
npx playwright show-report
```

### 4. Run Codegen (Record Tests)

```bash
# Open codegen tool to record interactions
npx playwright codegen http://localhost:3000
```

## Troubleshooting

### Issue: Browser Download Fails

**Problem:** Browser download fails or times out

**Solutions:**

```bash
# 1. Clear browser cache
rm -rf ~/.cache/ms-playwright

# 2. Reinstall browsers
npx playwright install --force

# 3. Use different download host (China)
PLAYWRIGHT_DOWNLOAD_HOST=https://playwright.azureedge.net npx playwright install

# 4. Increase timeout
PLAYWRIGHT_DOWNLOAD_CONNECTION_TIMEOUT=300000 npx playwright install
```

### Issue: System Dependencies Missing (Linux)

**Problem:** `Error: Host system is missing dependencies`

**Solution:**

```bash
# Install system dependencies
npx playwright install-deps

# Or manually install
sudo apt-get update
sudo apt-get install -y $(npx playwright install-deps --dry-run 2>&1 | grep 'apt-get install' | sed 's/.*apt-get install //')
```

### Issue: Permission Denied

**Problem:** Permission errors when installing browsers

**Solutions:**

```bash
# 1. Fix cache directory permissions
sudo chown -R $USER:$USER ~/.cache/ms-playwright

# 2. Install with sudo (not recommended)
sudo npx playwright install

# 3. Use user-local installation
npm config set prefix ~/.npm-global
export PATH=~/.npm-global/bin:$PATH
```

### Issue: Tests Timeout

**Problem:** Tests timeout waiting for server

**Solutions:**

```bash
# 1. Increase timeout in playwright.config.js
webServer: {
  timeout: 180000, // 3 minutes
}

# 2. Manually start server before tests
npm run dev &
npx playwright test

# 3. Check server logs
npm run dev
# In another terminal
npx playwright test
```

### Issue: Port Already in Use

**Problem:** `Error: Port 3000 is already in use`

**Solutions:**

```bash
# 1. Find and kill process using port
lsof -ti:3000 | xargs kill -9

# 2. Use different port
PORT=3001 npm run dev

# 3. Configure Playwright to use different port
# playwright.config.js
webServer: {
  command: 'PORT=3001 npm run dev',
  url: 'http://localhost:3001',
}
```

### Issue: Browser Crashes on Linux

**Problem:** Browser crashes with "GPU process exited unexpectedly"

**Solutions:**

```bash
# 1. Disable GPU acceleration
npx playwright test --project=chromium

# In playwright.config.js
use: {
  launchOptions: {
    args: ['--disable-gpu']
  }
}

# 2. Run in headless mode
npx playwright test --headed=false

# 3. Install missing graphics libraries
sudo apt-get install -y libnss3 libgbm1 libasound2
```

### Issue: WebKit Fails on Linux

**Problem:** WebKit tests fail on Linux

**Solutions:**

```bash
# 1. Install WebKit dependencies
npx playwright install-deps webkit

# 2. Skip WebKit in CI (Linux)
# playwright.config.js
projects: [
  {
    name: 'chromium',
    use: { ...devices['Desktop Chrome'] },
  },
  {
    name: 'firefox',
    use: { ...devices['Desktop Firefox'] },
  },
  // Skip WebKit on Linux CI
  ...(process.env.CI && process.platform === 'linux' ? [] : [{
    name: 'webkit',
    use: { ...devices['Desktop Safari'] },
  }]),
],
```

### Issue: Slow Test Execution

**Problem:** Tests run slowly

**Solutions:**

```bash
# 1. Increase parallel workers
npx playwright test --workers=4

# 2. Disable video/screenshots
# playwright.config.js
use: {
  video: 'off',
  screenshot: 'off',
}

# 3. Run in headless mode
npx playwright test --headed=false

# 4. Use faster browser (Chromium)
npx playwright test --project=chromium
```

### Issue: Network Timeouts

**Problem:** Tests fail with network timeouts

**Solutions:**

```javascript
// 1. Increase navigation timeout
test.use({
  navigationTimeout: 60000
});

// 2. Wait for network idle
await page.goto('/', { waitUntil: 'networkidle' });

// 3. Retry flaky network operations
await test.step('Load page', async () => {
  await page.goto('/', { timeout: 60000 });
});
```

### Issue: Authentication State Not Persisted

**Problem:** Login state not saved between tests

**Solutions:**

```bash
# 1. Check auth.json is generated
ls -la auth.json

# 2. Verify storage state configuration
# playwright.config.js
use: {
  storageState: 'auth.json'
}

# 3. Run global setup
npx playwright test --global-setup=tests/e2e/config/global-setup.js
```

## Docker Setup (Optional)

Run tests in Docker for consistent environment:

### Dockerfile

```dockerfile
FROM mcr.microsoft.com/playwright:v1.40.0-focal

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

CMD ["npx", "playwright", "test"]
```

### Build and Run

```bash
# Build image
docker build -t novacron-e2e .

# Run tests
docker run --rm -v $(pwd)/test-results:/app/test-results novacron-e2e

# Run with custom command
docker run --rm novacron-e2e npx playwright test --project=chromium
```

## VS Code Integration

### Install Extension

1. Install [Playwright Test for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-playwright.playwright)
2. Reload VS Code

### Features

- Run tests from sidebar
- Debug tests with breakpoints
- View test results inline
- Record new tests
- Pick locators

### Configuration

**.vscode/settings.json:**

```json
{
  "playwright.env": {
    "BASE_URL": "http://localhost:3000"
  },
  "playwright.reuseBrowser": true,
  "playwright.showTrace": true
}
```

## Next Steps

- [Writing Tests Guide](./WRITING_TESTS.md) - Learn how to write effective E2E tests
- [CI/CD Guide](./CI_CD.md) - Set up continuous integration
- [API Reference](./API.md) - Explore test utilities and helpers

## Additional Resources

- [Playwright Installation Guide](https://playwright.dev/docs/intro)
- [System Requirements](https://playwright.dev/docs/library#system-requirements)
- [Browsers](https://playwright.dev/docs/browsers)
- [Docker](https://playwright.dev/docs/docker)

---

**Last Updated**: 2025-01-10
**Playwright Version**: 1.40+
