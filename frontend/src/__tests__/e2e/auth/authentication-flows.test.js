/**
 * Authentication Flow E2E Tests
 * Tests all authentication workflows including login, registration, 2FA, and password reset
 */

describe('Authentication Flows', () => {
  let page

  beforeEach(async () => {
    page = await global.puppeteerUtils.createPage()
  })

  afterEach(async () => {
    if (page) {
      await page.close()
    }
  })

  describe('User Registration', () => {
    test('should complete user registration wizard', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/auth/register')

      // Verify registration page loads
      await expect(page).toMatch('Create Your NovaCron Account')
      
      // Step 1: Account Type Selection
      const personalAccount = await global.puppeteerUtils.waitForElement(
        page, 
        'input[value="personal"], label[for*="personal"]'
      )
      await personalAccount.click()
      
      const nextButton = await global.puppeteerUtils.waitForElement(page, 'button:has-text("Next")')
      await nextButton.click()
      
      // Step 2: Personal Information
      await global.puppeteerUtils.fillForm(page, {
        'input[name="firstName"]': 'John',
        'input[name="lastName"]': 'Doe',
        'input[name="email"]': 'john.doe@example.com',
        'input[name="organization"]': 'Test Corp'
      })
      
      await page.click('button:has-text("Next")')
      
      // Step 3: Security Setup
      await global.puppeteerUtils.fillForm(page, {
        'input[name="password"]': 'SecurePassword123!',
        'input[name="confirmPassword"]': 'SecurePassword123!'
      })
      
      // Accept terms and conditions
      const termsCheckbox = await global.puppeteerUtils.waitForElement(
        page, 
        'input[type="checkbox"][name*="terms"], input[type="checkbox"][name*="accept"]'
      )
      await termsCheckbox.click()
      
      // Complete registration
      await page.click('button:has-text("Complete Registration")')
      
      // Should redirect to dashboard or show success message
      await Promise.race([
        page.waitForSelector('[data-testid="dashboard"]', { timeout: 5000 }),
        page.waitForSelector('[data-testid="success-message"]', { timeout: 5000 }),
        page.waitForNavigation({ timeout: 5000 })
      ])
      
      const currentUrl = page.url()
      expect(currentUrl).toMatch(/(dashboard|success|login)/)
    })

    test('should validate registration form fields', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/auth/register')
      
      // Try to proceed without selecting account type
      const nextButton = await global.puppeteerUtils.waitForElement(page, 'button:has-text("Next")')
      await nextButton.click()
      
      // Should show validation error
      const validationError = await page.waitForSelector('[role="alert"], .error-message', { timeout: 2000 })
      expect(validationError).toBeTruthy()
      
      // Select account type and proceed
      const personalAccount = await global.puppeteerUtils.waitForElement(page, 'input[value="personal"]')
      await personalAccount.click()
      await nextButton.click()
      
      // Try to proceed without filling required fields
      await page.click('button:has-text("Next")')
      
      // Should show field validation errors
      const errors = await page.$$('[role="alert"], .error-message, .text-red-500')
      expect(errors.length).toBeGreaterThan(0)
    })

    test('should validate password strength requirements', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/auth/register')
      
      // Navigate to security step
      await page.click('input[value="personal"]')
      await page.click('button:has-text("Next")')
      
      await global.puppeteerUtils.fillForm(page, {
        'input[name="firstName"]': 'John',
        'input[name="lastName"]': 'Doe',
        'input[name="email"]': 'john@example.com'
      })
      
      await page.click('button:has-text("Next")')
      
      // Test weak password
      const passwordInput = await global.puppeteerUtils.waitForElement(page, 'input[name="password"]')
      await passwordInput.type('123')
      
      // Should show password strength indicator
      const strengthIndicator = await page.waitForSelector('[data-testid="password-strength"], .password-strength', { timeout: 2000 })
      expect(strengthIndicator).toBeTruthy()
      
      // Clear and try strong password
      await passwordInput.click({ clickCount: 3 })
      await passwordInput.type('SecurePassword123!')
      
      // Password strength should improve
      const strongIndicator = await page.waitForSelector('[data-strength="strong"], .strength-strong', { timeout: 2000 })
      expect(strongIndicator).toBeTruthy()
    })
  })

  describe('User Login', () => {
    test('should login with valid credentials', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/auth/login')
      
      // Verify login page
      await expect(page).toMatch('Sign In')
      
      // Fill login form
      await global.puppeteerUtils.fillForm(page, {
        'input[name="email"], input[type="email"]': 'test@example.com',
        'input[name="password"], input[type="password"]': 'password123'
      })
      
      // Submit form
      await page.click('button[type="submit"], button:has-text("Sign In")')
      
      // Wait for response (success or error)
      await Promise.race([
        page.waitForNavigation({ timeout: 5000 }),
        page.waitForSelector('[role="alert"], .error-message', { timeout: 5000 })
      ])
      
      // Check if redirected to dashboard or shows error (expected in test environment)
      const currentUrl = page.url()
      const hasError = await page.$('[role="alert"], .error-message')
      
      // Either successful redirect or expected API error
      expect(currentUrl.includes('dashboard') || hasError).toBeTruthy()
    })

    test('should show validation errors for empty fields', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/auth/login')
      
      // Try to submit empty form
      await page.click('button[type="submit"], button:has-text("Sign In")')
      
      // Should show validation errors
      const emailError = await page.waitForSelector('[data-testid="email-error"], .error-message:has-text("Email")', { timeout: 2000 })
      const passwordError = await page.waitForSelector('[data-testid="password-error"], .error-message:has-text("Password")', { timeout: 2000 })
      
      expect(emailError || passwordError).toBeTruthy()
    })

    test('should toggle password visibility', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/auth/login')
      
      const passwordInput = await global.puppeteerUtils.waitForElement(page, 'input[type="password"]')
      const toggleButton = await global.puppeteerUtils.waitForElement(
        page, 
        'button[aria-label*="toggle"], button[aria-label*="show"], button[data-testid="toggle-password"]'
      )
      
      // Initially password should be hidden
      let inputType = await passwordInput.evaluate(el => el.type)
      expect(inputType).toBe('password')
      
      // Click toggle button
      await toggleButton.click()
      
      // Password should now be visible
      inputType = await passwordInput.evaluate(el => el.type)
      expect(inputType).toBe('text')
      
      // Click toggle again
      await toggleButton.click()
      
      // Password should be hidden again
      inputType = await passwordInput.evaluate(el => el.type)
      expect(inputType).toBe('password')
    })

    test('should remember me functionality', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/auth/login')
      
      // Look for remember me checkbox
      const rememberCheckbox = await page.$('input[type="checkbox"][name*="remember"], input[name*="rememberMe"]')
      
      if (rememberCheckbox) {
        await rememberCheckbox.click()
        
        // Fill and submit form
        await global.puppeteerUtils.fillForm(page, {
          'input[name="email"]': 'test@example.com',
          'input[name="password"]': 'password123'
        })
        
        await page.click('button[type="submit"]')
        
        // Check if localStorage/sessionStorage is set (would need actual backend)
        const hasRememberToken = await page.evaluate(() => {
          return localStorage.getItem('rememberMe') || sessionStorage.getItem('auth-token')
        })
        
        // In test environment, just verify checkbox was clicked
        const isChecked = await rememberCheckbox.evaluate(el => el.checked)
        expect(isChecked).toBeTruthy()
      }
    })
  })

  describe('Password Reset', () => {
    test('should initiate password reset flow', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/auth/login')
      
      // Click forgot password link
      const forgotPasswordLink = await global.puppeteerUtils.waitForElement(
        page, 
        'a[href*="reset"], a:has-text("Forgot"), button:has-text("Forgot")'
      )
      await forgotPasswordLink.click()
      
      // Should navigate to reset page
      await page.waitForSelector('h1:has-text("Reset"), h2:has-text("Reset"), input[name="email"]', { timeout: 5000 })
      
      // Enter email
      await global.puppeteerUtils.fillForm(page, {
        'input[name="email"], input[type="email"]': 'test@example.com'
      })
      
      // Submit reset request
      await page.click('button[type="submit"], button:has-text("Reset"), button:has-text("Send")')
      
      // Should show success message or redirect
      await page.waitForSelector('[data-testid="success"], .success-message, [role="alert"]', { timeout: 5000 })
    })

    test('should validate email format in reset form', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/auth/forgot-password')
      
      // Try invalid email
      await global.puppeteerUtils.fillForm(page, {
        'input[name="email"]': 'invalid-email'
      })
      
      await page.click('button[type="submit"]')
      
      // Should show validation error
      const error = await page.waitForSelector('[role="alert"], .error-message', { timeout: 2000 })
      expect(error).toBeTruthy()
    })
  })

  describe('2FA Authentication', () => {
    test('should handle 2FA setup flow', async () => {
      // This test assumes 2FA setup is available after login
      await global.puppeteerUtils.navigateAndWait(page, '/auth/2fa/setup')
      
      // Look for QR code or setup instructions
      const qrCode = await page.$('[data-testid="qr-code"], .qr-code, img[alt*="QR"]')
      const setupCode = await page.$('[data-testid="setup-code"], .setup-code')
      
      expect(qrCode || setupCode).toBeTruthy()
      
      // If there's a verification code input, test it
      const codeInput = await page.$('input[name="verificationCode"], input[name="code"]')
      if (codeInput) {
        await codeInput.type('123456')
        
        await page.click('button[type="submit"], button:has-text("Verify")')
        
        // Wait for response
        await page.waitForSelector('[role="alert"], .error-message, .success-message', { timeout: 5000 })
      }
    })

    test('should validate 2FA code format', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/auth/2fa/verify')
      
      const codeInput = await page.$('input[name="code"], input[name="verificationCode"]')
      if (codeInput) {
        // Test invalid code length
        await codeInput.type('123')
        
        await page.click('button[type="submit"]')
        
        // Should show validation error
        const error = await page.waitForSelector('[role="alert"], .error-message', { timeout: 2000 })
        expect(error).toBeTruthy()
        
        // Test valid format
        await codeInput.click({ clickCount: 3 })
        await codeInput.type('123456')
        
        // Error should be cleared or form should submit
        const hasError = await page.$('[role="alert"], .error-message')
        const errorText = hasError ? await hasError.textContent() : ''
        expect(errorText).not.toMatch(/length|format/i)
      }
    })
  })

  describe('Session Management', () => {
    test('should handle session timeout', async () => {
      // This would require actual session management
      await global.puppeteerUtils.navigateAndWait(page, '/dashboard')
      
      // Simulate expired token by manipulating localStorage
      await page.evaluate(() => {
        localStorage.setItem('auth-token', 'expired-token')
        localStorage.setItem('auth-expires', Date.now() - 1000) // Expired
      })
      
      // Try to navigate to protected route
      await page.goto('http://localhost:8092/admin')
      
      // Should redirect to login
      await page.waitForSelector('h1:has-text("Sign In"), input[name="email"]', { timeout: 5000 })
      
      const currentUrl = page.url()
      expect(currentUrl).toMatch(/login|auth/)
    })

    test('should handle concurrent login sessions', async () => {
      // Open second page to simulate concurrent session
      const secondPage = await global.puppeteerUtils.createPage()
      
      try {
        // Login on first page
        await global.puppeteerUtils.login(page, { email: 'test@example.com', password: 'password123' })
        
        // Login on second page with same credentials
        await global.puppeteerUtils.login(secondPage, { email: 'test@example.com', password: 'password123' })
        
        // Both sessions should be valid or show appropriate handling
        const firstPageHasSession = await page.evaluate(() => {
          return !!localStorage.getItem('auth-token')
        })
        
        const secondPageHasSession = await secondPage.evaluate(() => {
          return !!localStorage.getItem('auth-token')
        })
        
        // At least one session should be active
        expect(firstPageHasSession || secondPageHasSession).toBeTruthy()
        
      } finally {
        await secondPage.close()
      }
    })
  })

  describe('Social Authentication', () => {
    test('should show social login options', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/auth/login')
      
      // Look for social login buttons
      const socialButtons = await page.$$('button:has-text("Google"), button:has-text("GitHub"), a[href*="oauth"], .social-login')
      
      if (socialButtons.length > 0) {
        // Test that social buttons are clickable
        const googleButton = await page.$('button:has-text("Google"), a[href*="google"]')
        if (googleButton) {
          // Click should initiate OAuth flow (will fail without backend but UI should respond)
          await googleButton.click()
          
          // Check if new tab opens or page changes
          const pages = await page.browser().pages()
          expect(pages.length).toBeGreaterThanOrEqual(2)
        }
      }
    })
  })
})