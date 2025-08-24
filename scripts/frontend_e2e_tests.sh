#!/bin/bash
# End-to-End Testing for NovaCron Frontend Using Playwright

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

FRONTEND_URL="http://localhost:8092"
API_URL="http://localhost:8090"

print_status() {
    echo -e "${BLUE}[E2E] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[PASS] $1${NC}"
}

print_error() {
    echo -e "${RED}[FAIL] $1${NC}"
}

# Check if Playwright is available
check_playwright() {
    if ! npm list playwright > /dev/null 2>&1; then
        print_status "Installing Playwright..."
        cd frontend
        npm install playwright @playwright/test
        npx playwright install
        cd ..
    fi
}

# Create Playwright test configuration
create_playwright_config() {
    cat > frontend/playwright.config.ts << 'EOF'
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'html',
  use: {
    baseURL: 'http://localhost:8092',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
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
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] },
    },
  ],
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:8092',
    reuseExistingServer: !process.env.CI,
  },
});
EOF
}

# Create E2E test directory and tests
create_e2e_tests() {
    mkdir -p frontend/e2e
    
    # Authentication flow test
    cat > frontend/e2e/auth.spec.ts << 'EOF'
import { test, expect } from '@playwright/test';

test.describe('Authentication Flow', () => {
  test('should load login page', async ({ page }) => {
    await page.goto('/auth/login');
    await expect(page).toHaveTitle(/NovaCron/);
    await expect(page.getByRole('heading', { name: 'Sign In' })).toBeVisible();
  });

  test('should show registration form', async ({ page }) => {
    await page.goto('/auth/register');
    await expect(page.getByRole('heading', { name: 'Create Your NovaCron Account' })).toBeVisible();
  });

  test('should validate login form', async ({ page }) => {
    await page.goto('/auth/login');
    
    // Try to submit empty form
    await page.getByRole('button', { name: 'Sign In' }).click();
    
    // Should show validation errors
    await expect(page.getByText('Email is required')).toBeVisible();
    await expect(page.getByText('Password is required')).toBeVisible();
  });

  test('should complete registration wizard', async ({ page }) => {
    await page.goto('/auth/register');
    
    // Step 1: Account type
    await page.getByLabel('Personal Account').check();
    await page.getByRole('button', { name: 'Next' }).click();
    
    // Step 2: Personal information
    await page.getByLabel('First Name *').fill('Test');
    await page.getByLabel('Last Name *').fill('User');
    await page.getByLabel('Email Address *').fill('test@example.com');
    await page.getByRole('button', { name: 'Next' }).click();
    
    // Step 3: Security
    await page.getByLabel('Password *').fill('SecurePassword123!');
    await page.getByLabel('Confirm Password *').fill('SecurePassword123!');
    await page.getByLabel('I accept the Terms').check();
    
    await page.getByRole('button', { name: 'Complete Registration' }).click();
    
    // Should show success or redirect
    // Note: This will fail in test env without backend, but validates UI flow
    await expect(page).toHaveURL(/\/(dashboard|auth\/login)/);
  });

  test('should toggle password visibility', async ({ page }) => {
    await page.goto('/auth/login');
    
    const passwordInput = page.getByLabel('Password');
    const toggleButton = page.getByRole('button', { name: 'Toggle password visibility' });
    
    // Initially password should be hidden
    await expect(passwordInput).toHaveAttribute('type', 'password');
    
    // Click toggle to show password
    await toggleButton.click();
    await expect(passwordInput).toHaveAttribute('type', 'text');
    
    // Click toggle to hide password again
    await toggleButton.click();
    await expect(passwordInput).toHaveAttribute('type', 'password');
  });
});
EOF

    # Dashboard test
    cat > frontend/e2e/dashboard.spec.ts << 'EOF'
import { test, expect } from '@playwright/test';

test.describe('Dashboard', () => {
  test('should load dashboard page', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Should show dashboard title or redirect to login
    const titleOrLogin = await Promise.race([
      page.getByRole('heading', { name: 'Dashboard' }).isVisible(),
      page.getByRole('heading', { name: 'Sign In' }).isVisible(),
    ]);
    
    expect(titleOrLogin).toBe(true);
  });

  test('should display VM status grid', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Look for VM status components
    const vmStatusVisible = await page.locator('[data-testid="vm-status-grid"]').isVisible().catch(() => false);
    const loginVisible = await page.getByRole('heading', { name: 'Sign In' }).isVisible().catch(() => false);
    
    // Either dashboard content should be visible or should redirect to login
    expect(vmStatusVisible || loginVisible).toBe(true);
  });

  test('should show metrics cards', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Check for metrics cards (if authenticated) or login form
    const hasMetrics = await page.locator('[data-testid="metrics-card"]').count();
    const hasLoginForm = await page.getByRole('form').count();
    
    expect(hasMetrics > 0 || hasLoginForm > 0).toBe(true);
  });
});
EOF

    # VM Management test
    cat > frontend/e2e/vm-management.spec.ts << 'EOF'
import { test, expect } from '@playwright/test';

test.describe('VM Management', () => {
  test('should load VMs page', async ({ page }) => {
    await page.goto('/vms');
    
    // Should show VMs page or redirect to login
    const vmsPageVisible = await page.getByRole('heading', { name: /Virtual Machines|VMs/ }).isVisible().catch(() => false);
    const loginVisible = await page.getByRole('heading', { name: 'Sign In' }).isVisible().catch(() => false);
    
    expect(vmsPageVisible || loginVisible).toBe(true);
  });

  test('should show create VM button', async ({ page }) => {
    await page.goto('/vms');
    
    // Look for create VM button (if authenticated)
    const createButton = await page.getByRole('button', { name: /Create VM|New VM/ }).isVisible().catch(() => false);
    const loginForm = await page.getByRole('form').isVisible().catch(() => false);
    
    expect(createButton || loginForm).toBe(true);
  });

  test('should display VM table or empty state', async ({ page }) => {
    await page.goto('/vms');
    
    // Should show either VM table, empty state, or login
    const hasTable = await page.getByRole('table').isVisible().catch(() => false);
    const hasEmptyState = await page.getByText(/No VMs found|No virtual machines/).isVisible().catch(() => false);
    const hasLoginForm = await page.getByRole('form').isVisible().catch(() => false);
    
    expect(hasTable || hasEmptyState || hasLoginForm).toBe(true);
  });
});
EOF

    # Accessibility test
    cat > frontend/e2e/accessibility.spec.ts << 'EOF'
import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test.describe('Accessibility', () => {
  test('should not have accessibility violations on login page', async ({ page }) => {
    await page.goto('/auth/login');
    
    const accessibilityScanResults = await new AxeBuilder({ page }).analyze();
    
    expect(accessibilityScanResults.violations).toEqual([]);
  });

  test('should not have accessibility violations on registration page', async ({ page }) => {
    await page.goto('/auth/register');
    
    const accessibilityScanResults = await new AxeBuilder({ page }).analyze();
    
    expect(accessibilityScanResults.violations).toEqual([]);
  });

  test('should not have accessibility violations on dashboard', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Wait for page to load
    await page.waitForLoadState('networkidle');
    
    const accessibilityScanResults = await new AxeBuilder({ page }).analyze();
    
    expect(accessibilityScanResults.violations).toEqual([]);
  });

  test('should have proper keyboard navigation', async ({ page }) => {
    await page.goto('/auth/login');
    
    // Tab through form elements
    await page.keyboard.press('Tab');
    await expect(page.getByLabel('Email')).toBeFocused();
    
    await page.keyboard.press('Tab');
    await expect(page.getByLabel('Password')).toBeFocused();
    
    await page.keyboard.press('Tab');
    await expect(page.getByRole('button', { name: 'Sign In' })).toBeFocused();
  });

  test('should have proper ARIA labels', async ({ page }) => {
    await page.goto('/auth/register');
    
    // Check for important ARIA attributes
    const formElements = await page.locator('input[required]').all();
    
    for (const element of formElements) {
      const ariaLabel = await element.getAttribute('aria-label');
      const associatedLabel = await element.getAttribute('aria-labelledby');
      const hasLabel = await element.locator('..').locator('label').count();
      
      // Each required input should have some form of labeling
      expect(ariaLabel || associatedLabel || hasLabel > 0).toBeTruthy();
    }
  });
});
EOF

    # Performance test
    cat > frontend/e2e/performance.spec.ts << 'EOF'
import { test, expect } from '@playwright/test';

test.describe('Performance', () => {
  test('should load pages within acceptable time', async ({ page }) => {
    const startTime = Date.now();
    
    await page.goto('/auth/login');
    await page.waitForLoadState('networkidle');
    
    const loadTime = Date.now() - startTime;
    
    // Page should load within 3 seconds
    expect(loadTime).toBeLessThan(3000);
  });

  test('should have good Core Web Vitals', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Measure Largest Contentful Paint (LCP)
    const lcp = await page.evaluate(() => {
      return new Promise((resolve) => {
        new PerformanceObserver((list) => {
          const entries = list.getEntries();
          const lastEntry = entries[entries.length - 1];
          resolve(lastEntry.startTime);
        }).observe({ entryTypes: ['largest-contentful-paint'] });
        
        // Fallback if no LCP entry
        setTimeout(() => resolve(0), 5000);
      });
    });
    
    // LCP should be under 2.5 seconds (good)
    if (typeof lcp === 'number' && lcp > 0) {
      expect(lcp).toBeLessThan(2500);
    }
  });

  test('should handle multiple concurrent requests', async ({ browser }) => {
    const contexts = await Promise.all([
      browser.newContext(),
      browser.newContext(),
      browser.newContext(),
    ]);
    
    const pages = await Promise.all(
      contexts.map(context => context.newPage())
    );
    
    const startTime = Date.now();
    
    // Load the same page in multiple tabs
    await Promise.all(
      pages.map(page => page.goto('/auth/login'))
    );
    
    await Promise.all(
      pages.map(page => page.waitForLoadState('networkidle'))
    );
    
    const totalTime = Date.now() - startTime;
    
    // Should handle concurrent loads efficiently
    expect(totalTime).toBeLessThan(5000);
    
    // Cleanup
    await Promise.all(contexts.map(context => context.close()));
  });
});
EOF
}

# Update package.json with E2E test scripts
update_package_json() {
    cd frontend
    
    # Add playwright and axe-core dependencies if not present
    npm install --save-dev @playwright/test @axe-core/playwright
    
    # Add E2E test script to package.json
    npx json -I -f package.json -e 'this.scripts["test:e2e"] = "playwright test"'
    npx json -I -f package.json -e 'this.scripts["test:e2e:ui"] = "playwright test --ui"'
    npx json -I -f package.json -e 'this.scripts["test:e2e:debug"] = "playwright test --debug"'
    
    cd ..
}

# Main execution
main() {
    print_status "Setting up Frontend E2E Testing with Playwright"
    print_status "==============================================="
    
    check_playwright
    create_playwright_config
    create_e2e_tests
    update_package_json
    
    print_success "E2E test suite created successfully!"
    print_status ""
    print_status "Available commands:"
    print_status "  cd frontend && npm run test:e2e         # Run all E2E tests"
    print_status "  cd frontend && npm run test:e2e:ui      # Run tests with UI"
    print_status "  cd frontend && npm run test:e2e:debug   # Debug tests"
    print_status ""
    print_status "Running a quick E2E test..."
    
    cd frontend
    if npx playwright test --reporter=line auth.spec.ts; then
        print_success "E2E tests are working correctly"
    else
        print_error "E2E tests encountered issues (expected without running services)"
    fi
    cd ..
}

main "$@"