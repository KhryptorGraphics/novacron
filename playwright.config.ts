import { defineConfig, devices } from '@playwright/test';
import * as dotenv from 'dotenv';
import * as path from 'path';

// Load environment variables from .env file
dotenv.config({ path: path.resolve(__dirname, '.env') });

/**
 * Playwright Configuration for NovaCron E2E Testing
 *
 * Features:
 * - Multi-browser support (Chromium, Firefox, WebKit)
 * - Parallel execution for faster test runs
 * - Automatic retries for flaky tests
 * - Screenshot and video capture on failure
 * - Trace collection for debugging
 * - Multiple reporters (HTML, JSON, JUnit for CI)
 * - Environment-specific configurations
 */
export default defineConfig({
  // Test directory
  testDir: './tests/e2e',

  // Maximum time one test can run
  timeout: 30 * 1000,

  // Maximum time for each assertion
  expect: {
    timeout: 5000,
  },

  // Run tests in files in parallel
  fullyParallel: true,

  // Fail the build on CI if you accidentally left test.only in the source code
  forbidOnly: !!process.env.CI,

  // Retry on CI only
  retries: process.env.CI ? 2 : 1,

  // Opt out of parallel tests on CI
  workers: process.env.CI ? 2 : undefined,

  // Reporter configuration
  reporter: [
    // HTML reporter for local development
    ['html', { outputFolder: 'tests/e2e/reports/html', open: 'never' }],

    // JSON reporter for CI/CD integration
    ['json', { outputFile: 'tests/e2e/reports/json/results.json' }],

    // JUnit reporter for CI systems (Jenkins, GitLab CI, etc.)
    ['junit', { outputFile: 'tests/e2e/reports/junit/results.xml' }],

    // List reporter for terminal output
    ['list'],

    // GitHub Actions reporter when running on GitHub
    process.env.GITHUB_ACTIONS ? ['github'] : ['list'],
  ],

  // Shared settings for all the projects below
  use: {
    // Base URL for navigation
    baseURL: process.env.PLAYWRIGHT_BASE_URL || 'http://localhost:3000',

    // Collect trace when retrying the failed test
    trace: process.env.CI ? 'on-first-retry' : 'retain-on-failure',

    // Screenshot configuration
    screenshot: process.env.CI ? 'only-on-failure' : 'on',

    // Video configuration
    video: process.env.CI ? 'retain-on-failure' : 'on-first-retry',

    // Maximum time for navigation
    navigationTimeout: 10000,

    // Maximum time for actions
    actionTimeout: 10000,

    // Ignore HTTPS errors
    ignoreHTTPSErrors: true,

    // Viewport size
    viewport: { width: 1280, height: 720 },

    // Color scheme
    colorScheme: 'light',

    // Locale and timezone
    locale: 'en-US',
    timezoneId: 'America/New_York',

    // Geolocation
    geolocation: { longitude: -73.935242, latitude: 40.730610 },
    permissions: ['geolocation'],

    // Extra HTTP headers
    extraHTTPHeaders: {
      'Accept-Language': 'en-US,en;q=0.9',
    },

    // Context options
    contextOptions: {
      recordVideo: {
        dir: 'tests/e2e/reports/videos/',
        size: { width: 1280, height: 720 },
      },
    },
  },

  // Global setup and teardown
  globalSetup: require.resolve('./tests/e2e/global-setup.ts'),
  globalTeardown: require.resolve('./tests/e2e/global-teardown.ts'),

  // Configure projects for major browsers
  projects: [
    // Desktop Chrome
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        // Chrome-specific settings
        channel: 'chrome',
        launchOptions: {
          args: [
            '--disable-web-security',
            '--disable-features=IsolateOrigins,site-per-process',
          ],
        },
      },
    },

    // Desktop Firefox
    {
      name: 'firefox',
      use: {
        ...devices['Desktop Firefox'],
        // Firefox-specific settings
        launchOptions: {
          firefoxUserPrefs: {
            'media.navigator.streams.fake': true,
            'media.navigator.permission.disabled': true,
          },
        },
      },
    },

    // Desktop Safari (WebKit)
    {
      name: 'webkit',
      use: {
        ...devices['Desktop Safari'],
      },
    },

    // Mobile Chrome
    {
      name: 'Mobile Chrome',
      use: {
        ...devices['Pixel 5'],
      },
    },

    // Mobile Safari
    {
      name: 'Mobile Safari',
      use: {
        ...devices['iPhone 12'],
      },
    },

    // Tablet
    {
      name: 'iPad',
      use: {
        ...devices['iPad Pro'],
      },
    },

    // Microsoft Edge
    {
      name: 'edge',
      use: {
        ...devices['Desktop Edge'],
        channel: 'msedge',
      },
    },

    // Branded browsers (optional - uncomment if needed)
    // {
    //   name: 'Google Chrome',
    //   use: {
    //     ...devices['Desktop Chrome'],
    //     channel: 'chrome',
    //   },
    // },
    // {
    //   name: 'Microsoft Edge',
    //   use: {
    //     ...devices['Desktop Edge'],
    //     channel: 'msedge',
    //   },
    // },
  ],

  // Web server configuration for local development
  webServer: process.env.CI ? undefined : {
    command: 'npm run start:api',
    url: 'http://localhost:3000',
    timeout: 120 * 1000,
    reuseExistingServer: !process.env.CI,
    stdout: 'pipe',
    stderr: 'pipe',
    env: {
      NODE_ENV: 'test',
    },
  },

  // Output directory for test artifacts
  outputDir: 'tests/e2e/reports/artifacts',

  // Folder for test artifacts such as screenshots, videos, traces
  snapshotDir: 'tests/e2e/snapshots',

  // Update snapshots with actual results
  updateSnapshots: process.env.UPDATE_SNAPSHOTS === 'true' ? 'all' : 'missing',

  // Whether to preserve output between test runs
  preserveOutput: 'always',

  // Metadata
  metadata: {
    project: 'NovaCron',
    environment: process.env.TEST_ENV || 'local',
    version: process.env.npm_package_version || '1.0.0',
  },
});
