import { chromium, FullConfig } from '@playwright/test';
import * as dotenv from 'dotenv';
import * as path from 'path';
import * as fs from 'fs';

/**
 * Global Setup for Playwright Tests
 *
 * Runs once before all tests to:
 * - Load environment variables
 * - Create necessary directories
 * - Perform authentication if needed
 * - Set up test database/state
 * - Start external services
 */
async function globalSetup(config: FullConfig): Promise<void> {
  console.log('🚀 Starting Playwright Global Setup...');

  // Load environment-specific configuration
  const environment = process.env.TEST_ENV || 'local';
  const envFile = path.resolve(__dirname, `../../.env.${environment}`);

  if (fs.existsSync(envFile)) {
    console.log(`📄 Loading environment from: ${envFile}`);
    dotenv.config({ path: envFile });
  } else {
    console.log(`📄 Loading default environment from: .env`);
    dotenv.config({ path: path.resolve(__dirname, '../../.env') });
  }

  // Create required directories
  const directories = [
    'tests/e2e/reports',
    'tests/e2e/reports/html',
    'tests/e2e/reports/json',
    'tests/e2e/reports/junit',
    'tests/e2e/reports/videos',
    'tests/e2e/reports/screenshots',
    'tests/e2e/reports/traces',
    'tests/e2e/reports/artifacts',
    'tests/e2e/snapshots',
    'tests/e2e/fixtures',
  ];

  directories.forEach((dir) => {
    const fullPath = path.resolve(__dirname, `../../${dir}`);
    if (!fs.existsSync(fullPath)) {
      fs.mkdirSync(fullPath, { recursive: true });
      console.log(`📁 Created directory: ${dir}`);
    }
  });

  // Perform authentication if required
  if (process.env.PLAYWRIGHT_SKIP_AUTH !== 'true') {
    await performAuthentication(config);
  }

  // Set up test data
  await setupTestData();

  // Wait for services to be ready
  await waitForServices();

  console.log('✅ Global Setup Complete');
}

/**
 * Perform authentication and save state
 */
async function performAuthentication(config: FullConfig): Promise<void> {
  console.log('🔐 Performing authentication...');

  const baseURL = config.use?.baseURL || 'http://localhost:3000';
  const username = process.env.TEST_USER || 'test@novacron.com';
  const password = process.env.TEST_PASSWORD || 'testpassword123';

  try {
    const browser = await chromium.launch();
    const context = await browser.newContext({
      baseURL,
      ignoreHTTPSErrors: true,
    });

    const page = await context.newPage();

    // Navigate to login page
    await page.goto('/login');

    // Fill in credentials
    await page.fill('[data-testid="username-input"]', username);
    await page.fill('[data-testid="password-input"]', password);

    // Submit login form
    await page.click('[data-testid="login-button"]');

    // Wait for navigation to complete
    await page.waitForURL('/dashboard', { timeout: 10000 });

    // Save authenticated state
    const storageStatePath = path.resolve(__dirname, 'fixtures/auth-state.json');
    await context.storageState({ path: storageStatePath });

    console.log(`✅ Authentication successful, state saved to: ${storageStatePath}`);

    await browser.close();
  } catch (error) {
    console.warn('⚠️  Authentication failed (this may be expected if auth is not required):', error);
  }
}

/**
 * Set up test data
 */
async function setupTestData(): Promise<void> {
  console.log('📊 Setting up test data...');

  // Create test data fixtures if they don't exist
  const fixturesPath = path.resolve(__dirname, 'fixtures/test-data.json');

  if (!fs.existsSync(fixturesPath)) {
    const testData = {
      users: [
        { id: 1, email: 'test@novacron.com', role: 'admin' },
        { id: 2, email: 'user@novacron.com', role: 'user' },
      ],
      vms: [
        { id: 1, name: 'test-vm-1', status: 'running' },
        { id: 2, name: 'test-vm-2', status: 'stopped' },
      ],
    };

    fs.writeFileSync(fixturesPath, JSON.stringify(testData, null, 2));
    console.log(`✅ Created test data fixtures at: ${fixturesPath}`);
  }
}

/**
 * Wait for required services to be ready
 */
async function waitForServices(): Promise<void> {
  console.log('⏳ Waiting for services to be ready...');

  const services = [
    { name: 'API Server', url: process.env.PLAYWRIGHT_BASE_URL || 'http://localhost:3000' },
  ];

  for (const service of services) {
    let retries = 30;
    let isReady = false;

    while (retries > 0 && !isReady) {
      try {
        const browser = await chromium.launch({ headless: true });
        const context = await browser.newContext({ ignoreHTTPSErrors: true });
        const page = await context.newPage();

        await page.goto(service.url, { timeout: 5000 });
        await browser.close();

        console.log(`✅ ${service.name} is ready at ${service.url}`);
        isReady = true;
      } catch (error) {
        retries--;
        if (retries === 0) {
          console.error(`❌ ${service.name} failed to become ready at ${service.url}`);
          throw new Error(`Service ${service.name} is not ready`);
        }
        console.log(`⏳ ${service.name} not ready yet, retrying... (${retries} attempts left)`);
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }
    }
  }
}

/**
 * Export cleanup function to be called on process exit
 */
export function cleanup(): void {
  console.log('🧹 Cleaning up global setup resources...');
  // Add any cleanup logic here
}

// Handle process termination
process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);

export default globalSetup;
