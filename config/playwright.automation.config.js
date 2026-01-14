/**
 * NovaCron Playwright Configuration for Claude-Flow Automation
 * Extended Playwright configuration with enterprise testing capabilities
 */
const { defineConfig, devices } = require('@playwright/test');

module.exports = defineConfig({
  // Test Directory Configuration
  testDir: './tests',
  testMatch: [
    '**/tests/**/*.test.{js,ts}',
    '**/e2e/**/*.test.{js,ts}',
    '**/__tests__/**/*.e2e.{js,ts}'
  ],
  
  // Test Ignore Patterns
  testIgnore: [
    '**/node_modules/**',
    '**/coverage/**',
    '**/*.unit.test.{js,ts}',
    '**/*.integration.test.{js,ts}'
  ],
  
  // Global Configuration
  timeout: 60000,                    // 60s test timeout
  expect: { timeout: 10000 },        // 10s assertion timeout
  globalTimeout: 3600000,            // 1 hour global timeout
  
  // Execution Settings
  fullyParallel: true,               // Run tests in parallel
  forbidOnly: !!process.env.CI,     // Forbid test.only in CI
  retries: process.env.CI ? 3 : 2,   // Retry failed tests
  workers: process.env.CI ? 10 : 4,  // Parallel workers
  
  // Reporting Configuration
  reporter: [
    ['html', { 
      outputFolder: 'test-results/playwright-report',
      open: 'never'
    }],
    ['json', { 
      outputFile: 'test-results/playwright-results.json' 
    }],
    ['junit', { 
      outputFile: 'test-results/playwright-junit.xml' 
    }],
    ['line'],
    ['github'], // GitHub Actions integration
    ['./config/custom-reporter.js'] // Custom NovaCron reporter
  ],
  
  // Output Configuration
  outputDir: 'test-results/playwright-artifacts',
  
  // Global Test Configuration
  use: {
    // Base URL for tests
    baseURL: process.env.BASE_URL || 'http://localhost:8092',
    
    // Browser Configuration
    headless: process.env.HEADLESS !== 'false',
    viewport: { width: 1920, height: 1080 },
    ignoreHTTPSErrors: true,
    acceptDownloads: true,
    
    // Tracing and Recording
    trace: 'on-first-retry',
    video: 'retain-on-failure',
    screenshot: 'only-on-failure',
    
    // Performance and Metrics
    actionTimeout: 10000,
    navigationTimeout: 30000,
    
    // HAR Recording for API testing
    recordHar: {
      mode: 'minimal',
      path: 'test-results/har/'
    },
    
    // Custom Context Options
    contextOptions: {
      // Performance monitoring
      recordVideo: {
        dir: 'test-results/videos/',
        size: { width: 1920, height: 1080 }
      }
    }
  },
  
  // Project Configuration - Multi-Browser Testing
  projects: [
    // Desktop Browsers
    {
      name: 'chromium-desktop',
      use: {
        ...devices['Desktop Chrome'],
        channel: 'chrome',
        viewport: { width: 1920, height: 1080 },
        launchOptions: {
          args: [
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-web-security',
            '--allow-running-insecure-content',
            '--enable-features=NetworkService',
            '--disable-features=VizDisplayCompositor'
          ]
        }
      },
      testDir: './tests',
      testMatch: ['**/*.{test,spec}.{js,ts}']
    },
    
    {
      name: 'firefox-desktop',
      use: {
        ...devices['Desktop Firefox'],
        viewport: { width: 1920, height: 1080 }
      },
      testDir: './tests',
      testMatch: ['**/*.{test,spec}.{js,ts}']
    },
    
    {
      name: 'webkit-desktop',
      use: {
        ...devices['Desktop Safari'],
        viewport: { width: 1920, height: 1080 }
      },
      testDir: './tests',
      testMatch: ['**/*.{test,spec}.{js,ts}']
    },
    
    {
      name: 'edge-desktop',
      use: {
        ...devices['Desktop Chrome'],
        channel: 'msedge',
        viewport: { width: 1920, height: 1080 }
      },
      testDir: './tests',
      testMatch: ['**/*.{test,spec}.{js,ts}']
    },
    
    // Mobile Browsers
    {
      name: 'mobile-chrome',
      use: {
        ...devices['Pixel 5'],
        locale: 'en-US',
        timezoneId: 'America/New_York'
      },
      testDir: './tests/mobile',
      testMatch: ['**/*.mobile.{test,spec}.{js,ts}']
    },
    
    {
      name: 'mobile-safari',
      use: {
        ...devices['iPhone 13'],
        locale: 'en-US',
        timezoneId: 'America/New_York'
      },
      testDir: './tests/mobile',
      testMatch: ['**/*.mobile.{test,spec}.{js,ts}']
    },
    
    // Tablet Testing
    {
      name: 'tablet-chrome',
      use: {
        ...devices['iPad Pro'],
        locale: 'en-US'
      },
      testDir: './tests/tablet',
      testMatch: ['**/*.tablet.{test,spec}.{js,ts}']
    },
    
    // 4K Resolution Testing
    {
      name: 'chrome-4k',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 3840, height: 2160 },
        deviceScaleFactor: 2
      },
      testDir: './tests/4k',
      testMatch: ['**/*.4k.{test,spec}.{js,ts}']
    },
    
    // Network Condition Testing
    {
      name: 'chrome-slow-3g',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1920, height: 1080 },
        offline: false,
        downloadThroughput: 500 * 1024 / 8, // 500kbps
        uploadThroughput: 500 * 1024 / 8,
        latency: 400
      },
      testDir: './tests/network',
      testMatch: ['**/*.network.{test,spec}.{js,ts}']
    },
    
    {
      name: 'chrome-offline',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1920, height: 1080 },
        offline: true
      },
      testDir: './tests/offline',
      testMatch: ['**/*.offline.{test,spec}.{js,ts}']
    },
    
    // Performance Testing Project
    {
      name: 'performance-testing',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1920, height: 1080 },
        trace: 'on',
        video: 'on'
      },
      testDir: './tests/performance',
      testMatch: ['**/*.performance.{test,spec}.{js,ts}'],
      timeout: 120000 // 2 minutes for performance tests
    },
    
    // Accessibility Testing Project
    {
      name: 'accessibility-testing',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1920, height: 1080 },
        colorScheme: 'dark' // Test both light and dark modes
      },
      testDir: './tests/accessibility',
      testMatch: ['**/*.a11y.{test,spec}.{js,ts}']
    },
    
    // Visual Regression Testing Project
    {
      name: 'visual-regression',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1920, height: 1080 }
      },
      testDir: './tests/visual',
      testMatch: ['**/*.visual.{test,spec}.{js,ts}'],
      expect: {
        threshold: 0.05, // 5% pixel difference tolerance
        toHaveScreenshot: { 
          mode: 'css',
          animations: 'disabled'
        }
      }
    },
    
    // API Testing Project
    {
      name: 'api-testing',
      use: {
        baseURL: process.env.API_BASE_URL || 'http://localhost:8080/api/v1',
        extraHTTPHeaders: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        }
      },
      testDir: './tests/api',
      testMatch: ['**/*.api.{test,spec}.{js,ts}']
    },
    
    // Security Testing Project
    {
      name: 'security-testing',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1920, height: 1080 },
        ignoreHTTPSErrors: false, // Strict HTTPS for security tests
        recordHar: {
          mode: 'full',
          path: 'test-results/security-har/'
        }
      },
      testDir: './tests/security',
      testMatch: ['**/*.security.{test,spec}.{js,ts}']
    }
  ],
  
  // Web Server Configuration
  webServer: [
    {
      command: 'npm run dev:api',
      url: 'http://localhost:8080/api/health',
      reuseExistingServer: !process.env.CI,
      timeout: 120000
    },
    {
      command: 'npm run dev:frontend',
      url: 'http://localhost:8092',
      reuseExistingServer: !process.env.CI,
      timeout: 120000
    }
  ],
  
  // Global Setup and Teardown
  globalSetup: require.resolve('./tests/global-setup.js'),
  globalTeardown: require.resolve('./tests/global-teardown.js'),
  
  // Metadata
  metadata: {
    project: 'NovaCron',
    version: process.env.npm_package_version || '1.0.0',
    environment: process.env.NODE_ENV || 'test',
    testSuite: 'E2E Automation',
    framework: 'Playwright with Claude-Flow'
  }
});