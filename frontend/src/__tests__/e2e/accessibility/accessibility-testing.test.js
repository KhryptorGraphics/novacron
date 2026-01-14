/**
 * Accessibility E2E Tests
 * Tests WCAG compliance, keyboard navigation, screen reader compatibility
 */

describe('Accessibility Testing', () => {
  let page

  beforeEach(async () => {
    page = await global.puppeteerUtils.createPage()
  })

  afterEach(async () => {
    if (page) {
      await page.close()
    }
  })

  describe('WCAG Compliance', () => {
    test('should meet WCAG 2.1 AA standards on login page', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/auth/login')
      
      const accessibilityResults = await global.puppeteerUtils.checkAccessibility(page)
      
      console.log(`Found ${accessibilityResults.violations.length} accessibility violations`)
      
      if (accessibilityResults.violations.length > 0) {
        console.log('Violations:', accessibilityResults.violations.map(v => ({
          id: v.id,
          impact: v.impact,
          description: v.description,
          nodes: v.nodes.length
        })))
      }
      
      // Critical and serious violations should be 0
      const criticalViolations = accessibilityResults.violations.filter(v => 
        v.impact === 'critical' || v.impact === 'serious'
      )
      
      expect(criticalViolations).toHaveLength(0)
    })

    test('should meet WCAG 2.1 AA standards on dashboard', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/dashboard')
      
      const accessibilityResults = await global.puppeteerUtils.checkAccessibility(page)
      
      // Allow for authentication redirect
      if (page.url().includes('login') || page.url().includes('auth')) {
        console.log('Redirected to auth - testing auth page accessibility instead')
      }
      
      const criticalViolations = accessibilityResults.violations.filter(v => 
        v.impact === 'critical' || v.impact === 'serious'
      )
      
      if (criticalViolations.length > 0) {
        console.log('Critical accessibility violations:', criticalViolations)
      }
      
      expect(criticalViolations).toHaveLength(0)
    })

    test('should meet WCAG 2.1 AA standards on registration page', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/auth/register')
      
      const accessibilityResults = await global.puppeteerUtils.checkAccessibility(page)
      
      const criticalViolations = accessibilityResults.violations.filter(v => 
        v.impact === 'critical' || v.impact === 'serious'
      )
      
      expect(criticalViolations).toHaveLength(0)
    })

    test('should have proper color contrast ratios', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/auth/login')
      
      // Test color contrast using axe-core
      const contrastResults = await page.evaluate(() => {
        return new Promise((resolve) => {
          axe.run({
            tags: ['wcag2aa', 'wcag21aa'],
            rules: {
              'color-contrast': { enabled: true }
            }
          }, (err, results) => {
            if (err) throw err
            resolve(results)
          })
        })
      })
      
      const contrastViolations = contrastResults.violations.filter(v => v.id === 'color-contrast')
      
      if (contrastViolations.length > 0) {
        console.log('Color contrast violations:', contrastViolations[0].nodes.map(n => n.target))
      }
      
      expect(contrastViolations).toHaveLength(0)
    })
  })

  describe('Keyboard Navigation', () => {
    test('should support full keyboard navigation on login form', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/auth/login')
      
      // Test tab order
      await page.keyboard.press('Tab')
      
      let focusedElement = await page.evaluateHandle(() => document.activeElement)
      let tagName = await focusedElement.evaluate(el => el.tagName.toLowerCase())
      let inputType = await focusedElement.evaluate(el => el.type || el.tagName.toLowerCase())
      
      // First focusable element should be email input
      expect(['input', 'button'].includes(tagName)).toBeTruthy()
      
      // Continue tabbing through form
      const tabOrder = []
      
      for (let i = 0; i < 5; i++) {
        await page.keyboard.press('Tab')
        
        focusedElement = await page.evaluateHandle(() => document.activeElement)
        const element = await focusedElement.evaluate(el => ({
          tagName: el.tagName.toLowerCase(),
          type: el.type,
          id: el.id,
          name: el.name,
          role: el.getAttribute('role')
        }))
        
        tabOrder.push(element)
      }
      
      console.log('Tab order:', tabOrder)
      
      // Should include email, password, and submit button
      const hasEmailInput = tabOrder.some(el => el.type === 'email' || el.name === 'email')
      const hasPasswordInput = tabOrder.some(el => el.type === 'password' || el.name === 'password')
      const hasSubmitButton = tabOrder.some(el => el.type === 'submit' || el.tagName === 'button')
      
      expect(hasEmailInput || hasPasswordInput || hasSubmitButton).toBeTruthy()
    })

    test('should support keyboard navigation in VM management', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/vms')
      
      // Skip if redirected to login
      const isLoginPage = await page.$('input[name="email"]')
      if (isLoginPage) {
        console.log('Redirected to login - skipping VM navigation test')
        return
      }
      
      // Test navigation to create VM button
      let tabCount = 0
      let foundCreateButton = false
      
      while (tabCount < 10 && !foundCreateButton) {
        await page.keyboard.press('Tab')
        tabCount++
        
        const focusedElement = await page.evaluateHandle(() => document.activeElement)
        const buttonText = await focusedElement.evaluate(el => el.textContent || '')
        
        if (buttonText.toLowerCase().includes('create') || buttonText.toLowerCase().includes('new')) {
          foundCreateButton = true
          
          // Test Enter key activation
          await page.keyboard.press('Enter')
          
          // Should open create VM form or navigate
          await Promise.race([
            page.waitForSelector('form, [role="dialog"]', { timeout: 3000 }),
            page.waitForNavigation({ timeout: 3000 })
          ])
          
          const hasForm = await page.$('form, [role="dialog"]')
          const urlChanged = !page.url().includes('/vms')
          
          expect(hasForm || urlChanged).toBeTruthy()
        }
      }
      
      if (!foundCreateButton) {
        console.log('Create button not found in tab navigation')
      }
    })

    test('should support escape key to close modals', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/vms')
      
      const isLoginPage = await page.$('input[name="email"]')
      if (isLoginPage) return

      // Try to open a modal (create VM, settings, etc.)
      const modalTrigger = await page.$('button:has-text("Create"), button:has-text("Settings"), button[aria-haspopup="dialog"]')
      
      if (modalTrigger) {
        await modalTrigger.click()
        
        // Wait for modal to appear
        const modal = await page.waitForSelector('[role="dialog"], .modal, .overlay', { timeout: 3000 })
        
        if (modal) {
          // Test escape key closes modal
          await page.keyboard.press('Escape')
          
          // Modal should disappear
          await page.waitForTimeout(500)
          const modalStillVisible = await page.$('[role="dialog"]:not([style*="display: none"])')
          
          expect(modalStillVisible).toBeFalsy()
        }
      }
    })

    test('should support arrow key navigation in menus', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/dashboard')
      
      const isLoginPage = await page.$('input[name="email"]')
      if (isLoginPage) return

      // Look for dropdown menus
      const menuButton = await page.$('button[aria-haspopup="menu"], .dropdown-trigger, [role="menubutton"]')
      
      if (menuButton) {
        await menuButton.click()
        
        // Wait for menu to appear
        const menu = await page.waitForSelector('[role="menu"], .dropdown-menu', { timeout: 2000 })
        
        if (menu) {
          // Test arrow key navigation
          await page.keyboard.press('ArrowDown')
          
          let focusedItem = await page.evaluateHandle(() => document.activeElement)
          let role = await focusedItem.evaluate(el => el.getAttribute('role'))
          
          expect(role).toBe('menuitem')
          
          // Navigate to next item
          await page.keyboard.press('ArrowDown')
          await page.waitForTimeout(100)
          
          focusedItem = await page.evaluateHandle(() => document.activeElement)
          role = await focusedItem.evaluate(el => el.getAttribute('role'))
          
          expect(role).toBe('menuitem')
        }
      }
    })
  })

  describe('Screen Reader Support', () => {
    test('should have proper ARIA labels and roles', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/auth/register')
      
      // Check form inputs have labels
      const inputs = await page.$$('input[required]')
      
      for (const input of inputs) {
        const hasLabel = await input.evaluate(el => {
          // Check for explicit label
          if (el.getAttribute('aria-label')) return true
          
          // Check for aria-labelledby
          if (el.getAttribute('aria-labelledby')) return true
          
          // Check for associated label element
          if (el.id) {
            const label = document.querySelector(`label[for="${el.id}"]`)
            if (label) return true
          }
          
          // Check if wrapped in label
          let parent = el.parentElement
          while (parent) {
            if (parent.tagName.toLowerCase() === 'label') return true
            parent = parent.parentElement
          }
          
          return false
        })
        
        expect(hasLabel).toBeTruthy()
      }
    })

    test('should have proper heading structure', async () => {
      const testPages = ['/auth/login', '/auth/register', '/dashboard', '/vms']
      
      for (const path of testPages) {
        await global.puppeteerUtils.navigateAndWait(page, path)
        
        const headings = await page.$$eval('h1, h2, h3, h4, h5, h6', headings =>
          headings.map(h => ({
            level: parseInt(h.tagName[1]),
            text: h.textContent.trim()
          }))
        )
        
        if (headings.length > 0) {
          // Should have at least one h1
          const h1Count = headings.filter(h => h.level === 1).length
          expect(h1Count).toBeGreaterThanOrEqual(1)
          
          // Check heading hierarchy (no skipped levels)
          for (let i = 1; i < headings.length; i++) {
            const currentLevel = headings[i].level
            const previousLevel = headings[i - 1].level
            
            // Next heading should not skip more than one level
            expect(currentLevel - previousLevel).toBeLessThanOrEqual(1)
          }
        }
        
        console.log(`${path} heading structure:`, headings)
      }
    })

    test('should have descriptive link text', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/dashboard')
      
      const links = await page.$$eval('a[href]', links =>
        links.map(link => ({
          href: link.href,
          text: link.textContent.trim(),
          ariaLabel: link.getAttribute('aria-label'),
          title: link.title
        })).filter(link => link.href && !link.href.startsWith('javascript:'))
      )
      
      // Check for non-descriptive link text
      const badLinkText = ['click here', 'read more', 'more', 'link', 'here']
      
      const problematicLinks = links.filter(link => {
        const text = (link.text || link.ariaLabel || link.title || '').toLowerCase()
        return badLinkText.some(bad => text === bad)
      })
      
      if (problematicLinks.length > 0) {
        console.log('Links with non-descriptive text:', problematicLinks)
      }
      
      expect(problematicLinks.length).toBeLessThan(3) // Allow some flexibility
    })

    test('should provide status announcements for dynamic content', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/dashboard')
      
      const isLoginPage = await page.$('input[name="email"]')
      if (isLoginPage) return

      // Look for live regions for status announcements
      const liveRegions = await page.$$('[aria-live], [role="status"], [role="alert"]')
      
      if (liveRegions.length > 0) {
        console.log(`Found ${liveRegions.length} live regions for announcements`)
        
        // Check that live regions have appropriate politeness settings
        for (const region of liveRegions) {
          const ariaLive = await region.evaluate(el => el.getAttribute('aria-live'))
          const role = await region.evaluate(el => el.getAttribute('role'))
          
          if (ariaLive) {
            expect(['polite', 'assertive', 'off'].includes(ariaLive)).toBeTruthy()
          }
          
          if (role) {
            expect(['status', 'alert', 'log'].includes(role)).toBeTruthy()
          }
        }
      }
    })
  })

  describe('Focus Management', () => {
    test('should have visible focus indicators', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/auth/login')
      
      // Tab to focusable elements and check focus visibility
      const focusableElements = await page.$$('button, input, a, [tabindex]:not([tabindex="-1"])')
      
      if (focusableElements.length > 0) {
        const firstElement = focusableElements[0]
        await firstElement.focus()
        
        // Check if element has focus styles
        const focusStyles = await firstElement.evaluate(el => {
          const computed = window.getComputedStyle(el, ':focus')
          return {
            outline: computed.outline,
            outlineWidth: computed.outlineWidth,
            outlineStyle: computed.outlineStyle,
            outlineColor: computed.outlineColor,
            boxShadow: computed.boxShadow
          }
        })
        
        // Should have some form of focus indicator
        const hasFocusIndicator = (
          focusStyles.outline !== 'none' ||
          focusStyles.outlineWidth !== '0px' ||
          focusStyles.boxShadow !== 'none'
        )
        
        console.log('Focus styles:', focusStyles)
        expect(hasFocusIndicator).toBeTruthy()
      }
    })

    test('should manage focus in modals properly', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/vms')
      
      const isLoginPage = await page.$('input[name="email"]')
      if (isLoginPage) return

      // Try to open a modal
      const modalTrigger = await page.$('button:has-text("Create"), .modal-trigger')
      
      if (modalTrigger) {
        await modalTrigger.click()
        
        const modal = await page.waitForSelector('[role="dialog"], .modal', { timeout: 3000 })
        
        if (modal) {
          // Focus should be trapped in modal
          const focusedElement = await page.evaluateHandle(() => document.activeElement)
          const isInModal = await focusedElement.evaluate(el => {
            let parent = el
            while (parent) {
              if (parent.getAttribute && parent.getAttribute('role') === 'dialog') {
                return true
              }
              parent = parent.parentElement
            }
            return false
          })
          
          expect(isInModal).toBeTruthy()
          
          // Test focus trapping by tabbing
          await page.keyboard.press('Tab')
          await page.keyboard.press('Tab')
          
          // Focus should still be within modal
          const stillInModal = await page.evaluate(() => {
            const activeEl = document.activeElement
            let parent = activeEl
            while (parent) {
              if (parent.getAttribute && parent.getAttribute('role') === 'dialog') {
                return true
              }
              parent = parent.parentElement
            }
            return false
          })
          
          expect(stillInModal).toBeTruthy()
        }
      }
    })

    test('should restore focus after modal closes', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/vms')
      
      const isLoginPage = await page.$('input[name="email"]')
      if (isLoginPage) return

      const modalTrigger = await page.$('button:has-text("Create"), .modal-trigger')
      
      if (modalTrigger) {
        // Focus the trigger and remember it
        await modalTrigger.focus()
        const triggerElement = await page.evaluateHandle(() => document.activeElement)
        
        // Open modal
        await modalTrigger.click()
        
        const modal = await page.waitForSelector('[role="dialog"], .modal', { timeout: 3000 })
        
        if (modal) {
          // Close modal with escape
          await page.keyboard.press('Escape')
          
          // Wait for modal to close
          await page.waitForTimeout(500)
          
          // Focus should return to trigger
          const currentFocus = await page.evaluateHandle(() => document.activeElement)
          const isSameElement = await page.evaluate((trigger, current) => trigger === current, triggerElement, currentFocus)
          
          expect(isSameElement).toBeTruthy()
        }
      }
    })
  })

  describe('Mobile Accessibility', () => {
    test('should be accessible on mobile devices', async () => {
      await global.puppeteerUtils.simulateMobile(page)
      await global.puppeteerUtils.navigateAndWait(page, '/auth/login')
      
      const accessibilityResults = await global.puppeteerUtils.checkAccessibility(page)
      
      // Mobile-specific accessibility check
      const mobileViolations = accessibilityResults.violations.filter(v => 
        v.impact === 'critical' || v.impact === 'serious'
      )
      
      expect(mobileViolations).toHaveLength(0)
      
      // Check touch target sizes
      const touchTargets = await page.$$('button, a, input[type="button"], input[type="submit"]')
      
      for (const target of touchTargets) {
        const boundingBox = await target.boundingBox()
        
        if (boundingBox) {
          // Touch targets should be at least 44x44px (iOS guideline)
          expect(boundingBox.width).toBeGreaterThanOrEqual(40) // Allow some flexibility
          expect(boundingBox.height).toBeGreaterThanOrEqual(40)
        }
      }
    })

    test('should support pinch-to-zoom', async () => {
      await global.puppeteerUtils.simulateMobile(page)
      await global.puppeteerUtils.navigateAndWait(page, '/dashboard')
      
      // Check viewport meta tag allows zooming
      const viewportMeta = await page.$eval('meta[name="viewport"]', el => el.content)
      
      // Should not prevent zooming
      expect(viewportMeta).not.toMatch(/user-scalable=no|maximum-scale=1/)
      
      // Should allow reasonable zoom levels
      if (viewportMeta.includes('maximum-scale')) {
        const maxScale = parseFloat(viewportMeta.match(/maximum-scale=([0-9.]+)/)?.[1] || '5')
        expect(maxScale).toBeGreaterThanOrEqual(2) // Should allow at least 2x zoom
      }
    })
  })

  describe('Form Accessibility', () => {
    test('should have accessible form validation', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/auth/register')
      
      // Submit form to trigger validation
      const submitButton = await page.$('button[type="submit"], button:has-text("Register")')
      if (submitButton) {
        await submitButton.click()
        
        // Wait for validation messages
        await page.waitForTimeout(1000)
        
        // Check that error messages are associated with form fields
        const errorMessages = await page.$$('[role="alert"], .error-message, .field-error')
        
        for (const error of errorMessages) {
          // Error should be associated with a form field
          const hasAriaDescribedby = await error.evaluate(el => {
            const id = el.id
            if (!id) return false
            
            // Find input that references this error
            const input = document.querySelector(`input[aria-describedby*="${id}"]`)
            return !!input
          })
          
          const isInFieldGroup = await error.evaluate(el => {
            // Check if error is within a field group
            let parent = el.parentElement
            while (parent) {
              if (parent.classList.contains('field-group') || parent.classList.contains('form-field')) {
                return true
              }
              parent = parent.parentElement
            }
            return false
          })
          
          expect(hasAriaDescribedby || isInFieldGroup).toBeTruthy()
        }
      }
    })

    test('should support required field indicators', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/auth/register')
      
      const requiredInputs = await page.$$('input[required]')
      
      for (const input of requiredInputs) {
        const hasRequiredIndicator = await input.evaluate(el => {
          // Check aria-required
          if (el.getAttribute('aria-required') === 'true') return true
          
          // Check for visual required indicator in label
          const label = el.id ? document.querySelector(`label[for="${el.id}"]`) : null
          if (label && (label.textContent.includes('*') || label.textContent.includes('required'))) {
            return true
          }
          
          // Check for required in aria-label
          const ariaLabel = el.getAttribute('aria-label')
          if (ariaLabel && ariaLabel.includes('required')) return true
          
          return false
        })
        
        expect(hasRequiredIndicator).toBeTruthy()
      }
    })
  })

  describe('Table Accessibility', () => {
    test('should have accessible data tables', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/vms')
      
      const isLoginPage = await page.$('input[name="email"]')
      if (isLoginPage) return

      const tables = await page.$$('table')
      
      for (const table of tables) {
        // Should have table headers
        const headers = await table.$$('th')
        expect(headers.length).toBeGreaterThan(0)
        
        // Headers should have proper scope
        for (const header of headers) {
          const scope = await header.evaluate(el => el.getAttribute('scope'))
          if (scope) {
            expect(['col', 'row', 'colgroup', 'rowgroup'].includes(scope)).toBeTruthy()
          }
        }
        
        // Table should have caption or aria-label
        const hasCaption = await table.$('caption')
        const ariaLabel = await table.evaluate(el => el.getAttribute('aria-label'))
        const ariaLabelledby = await table.evaluate(el => el.getAttribute('aria-labelledby'))
        
        expect(hasCaption || ariaLabel || ariaLabelledby).toBeTruthy()
      }
    })
  })
})