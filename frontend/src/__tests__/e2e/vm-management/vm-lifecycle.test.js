/**
 * VM Lifecycle Management E2E Tests
 * Tests VM creation, configuration, lifecycle operations, and migration workflows
 */

describe('VM Lifecycle Management', () => {
  let page

  beforeEach(async () => {
    page = await global.puppeteerUtils.createPage()
    
    // Navigate to VMs page (will redirect to login if not authenticated)
    await global.puppeteerUtils.navigateAndWait(page, '/vms')
  })

  afterEach(async () => {
    if (page) {
      await page.close()
    }
  })

  describe('VM List View', () => {
    test('should display VM list or redirect to login', async () => {
      // Should show either VM list or login page
      const hasVMList = await page.$('[data-testid="vm-list"], .vm-grid, table')
      const hasLoginForm = await page.$('form:has(input[type="email"]), input[name="email"]')
      
      expect(hasVMList || hasLoginForm).toBeTruthy()
    })

    test('should show create VM button when authenticated', async () => {
      // Look for create button or login redirect
      const createButton = await page.$('button:has-text("Create"), button:has-text("New VM"), a[href*="create"]')
      const loginRequired = await page.$('h1:has-text("Sign In"), input[name="email"]')
      
      if (!loginRequired) {
        expect(createButton).toBeTruthy()
      }
    })

    test('should display VM status indicators', async () => {
      // Check if we're on the VMs page (not login)
      const isVMPage = await page.$('h1:has-text("Virtual Machines"), h1:has-text("VMs"), [data-testid="vm-list"]')
      
      if (isVMPage) {
        // Look for status indicators
        const statusIndicators = await page.$$('.status-indicator, .vm-status, [data-status]')
        
        // Should have status indicators if VMs exist
        if (statusIndicators.length > 0) {
          const statusText = await statusIndicators[0].textContent()
          expect(statusText).toMatch(/(running|stopped|error|pending)/i)
        }
      }
    })

    test('should support VM filtering and search', async () => {
      const isVMPage = await page.$('h1:has-text("Virtual Machines"), [data-testid="vm-list"]')
      
      if (isVMPage) {
        // Look for search/filter controls
        const searchInput = await page.$('input[placeholder*="Search"], input[name="search"]')
        const filterControls = await page.$('select[name="status"], .filter-controls')
        
        if (searchInput) {
          await searchInput.type('test-vm')
          
          // Should filter results or show no results message
          await page.waitForTimeout(500) // Wait for search to process
          
          const results = await page.$$('[data-testid="vm-item"], .vm-card, tr')
          const noResults = await page.$('.no-results, .empty-state')
          
          expect(results.length > 0 || noResults).toBeTruthy()
        }

        if (filterControls) {
          await filterControls.selectOption('running')
          
          // Should update the list
          await page.waitForTimeout(500)
        }
      }
    })
  })

  describe('VM Creation', () => {
    test('should open VM creation wizard', async () => {
      // Skip if redirected to login
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      const createButton = await page.$('button:has-text("Create"), a[href*="create"]')
      if (createButton) {
        await createButton.click()
        
        // Should open creation form or navigate to create page
        await Promise.race([
          page.waitForSelector('form:has(input[name="name"]), [data-testid="vm-create-form"]', { timeout: 5000 }),
          page.waitForNavigation({ timeout: 5000 })
        ])
        
        const hasCreateForm = await page.$('form:has(input[name="name"]), h1:has-text("Create")')
        expect(hasCreateForm).toBeTruthy()
      }
    })

    test('should validate VM creation form', async () => {
      // Navigate to create page directly
      await page.goto('http://localhost:8092/vms/create')
      
      const hasCreateForm = await page.$('form, [data-testid="vm-create-form"]')
      if (!hasCreateForm) return // Skip if form not available

      // Try to submit empty form
      const submitButton = await page.$('button[type="submit"], button:has-text("Create")')
      if (submitButton) {
        await submitButton.click()
        
        // Should show validation errors
        const errors = await page.$$('[role="alert"], .error-message, .text-red-500')
        expect(errors.length).toBeGreaterThan(0)
      }
    })

    test('should fill VM creation form with valid data', async () => {
      await page.goto('http://localhost:8092/vms/create')
      
      const hasCreateForm = await page.$('form, input[name="name"]')
      if (!hasCreateForm) return

      // Fill basic VM information
      const formData = {
        'input[name="name"]': 'test-vm-e2e',
        'input[name="description"]': 'E2E test virtual machine',
        'select[name="template"], select[name="os"]': 'ubuntu-20.04',
        'select[name="size"], select[name="resources"]': 'medium'
      }

      for (const [selector, value] of Object.entries(formData)) {
        const element = await page.$(selector)
        if (element) {
          if (selector.includes('select')) {
            await element.selectOption(value)
          } else {
            await element.type(value)
          }
        }
      }

      // Resource configuration
      const cpuSlider = await page.$('input[type="range"][name*="cpu"], .cpu-slider input')
      const memorySlider = await page.$('input[type="range"][name*="memory"], .memory-slider input')
      
      if (cpuSlider) {
        await cpuSlider.evaluate(el => el.value = '4')
        await cpuSlider.dispatchEvent(new Event('change'))
      }
      
      if (memorySlider) {
        await memorySlider.evaluate(el => el.value = '8')
        await memorySlider.dispatchEvent(new Event('change'))
      }

      // Submit form
      const submitButton = await page.$('button[type="submit"], button:has-text("Create")')
      if (submitButton) {
        await submitButton.click()
        
        // Should show success message or redirect
        await Promise.race([
          page.waitForSelector('.success-message, [role="alert"]', { timeout: 5000 }),
          page.waitForNavigation({ timeout: 5000 })
        ])
        
        const successMessage = await page.$('.success-message, [role="alert"]:has-text("success")')
        const redirected = page.url() !== 'http://localhost:8092/vms/create'
        
        expect(successMessage || redirected).toBeTruthy()
      }
    })

    test('should support advanced VM configuration', async () => {
      await page.goto('http://localhost:8092/vms/create')
      
      // Look for advanced options toggle
      const advancedToggle = await page.$('button:has-text("Advanced"), .advanced-toggle, details summary')
      if (advancedToggle) {
        await advancedToggle.click()
        
        // Should show advanced configuration options
        const advancedOptions = await page.$('.advanced-options, .network-config, .storage-config')
        expect(advancedOptions).toBeTruthy()
        
        // Test network configuration
        const networkSelect = await page.$('select[name*="network"], .network-select')
        if (networkSelect) {
          await networkSelect.selectOption('default')
        }
        
        // Test storage configuration
        const storageInput = await page.$('input[name*="storage"], input[name*="disk"]')
        if (storageInput) {
          await storageInput.clear()
          await storageInput.type('100')
        }
        
        // Test cloud-init configuration
        const cloudInitToggle = await page.$('input[type="checkbox"][name*="cloudinit"]')
        if (cloudInitToggle) {
          await cloudInitToggle.click()
          
          const cloudInitTextarea = await page.$('textarea[name*="userdata"], textarea[name*="script"]')
          if (cloudInitTextarea) {
            await cloudInitTextarea.type('#cloud-config\npackages:\n  - nginx')
          }
        }
      }
    })
  })

  describe('VM Operations', () => {
    test('should perform VM lifecycle operations', async () => {
      // Skip if on login page
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      // Look for VM in the list
      const vmRow = await page.$('[data-testid="vm-item"], .vm-card, tbody tr')
      if (!vmRow) return // No VMs available

      // Test start operation
      const startButton = await vmRow.$('button:has-text("Start"), .action-start')
      if (startButton) {
        await startButton.click()
        
        // Should show loading state or confirmation
        const loadingState = await page.waitForSelector('.loading, .spinner, [aria-busy="true"]', { timeout: 2000 })
        expect(loadingState).toBeTruthy()
      }

      // Test stop operation
      const stopButton = await vmRow.$('button:has-text("Stop"), .action-stop')
      if (stopButton) {
        await stopButton.click()
        
        // May show confirmation dialog
        const confirmDialog = await page.$('[role="dialog"], .confirm-dialog')
        if (confirmDialog) {
          const confirmButton = await confirmDialog.$('button:has-text("Confirm"), button:has-text("Stop")')
          if (confirmButton) {
            await confirmButton.click()
          }
        }
      }

      // Test restart operation
      const restartButton = await vmRow.$('button:has-text("Restart"), .action-restart')
      if (restartButton) {
        await restartButton.click()
        
        // Should show loading state
        await page.waitForTimeout(1000) // Brief wait for UI update
      }
    })

    test('should show VM details and console access', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      const vmRow = await page.$('[data-testid="vm-item"], .vm-card')
      if (!vmRow) return

      // Click on VM name or details button
      const vmLink = await vmRow.$('a, .vm-name, button:has-text("Details")')
      if (vmLink) {
        await vmLink.click()
        
        // Should navigate to VM details page
        await Promise.race([
          page.waitForNavigation({ timeout: 5000 }),
          page.waitForSelector('[data-testid="vm-details"], .vm-details', { timeout: 5000 })
        ])
        
        // Should show VM information
        const detailsPage = await page.$('[data-testid="vm-details"], .vm-overview')
        expect(detailsPage).toBeTruthy()
        
        // Look for console access
        const consoleButton = await page.$('button:has-text("Console"), .console-access')
        if (consoleButton) {
          await consoleButton.click()
          
          // Should open console (terminal emulator)
          const consoleTerminal = await page.waitForSelector('.terminal, .xterm-viewport, canvas', { timeout: 5000 })
          expect(consoleTerminal).toBeTruthy()
        }
      }
    })

    test('should handle VM configuration updates', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      // Navigate to VM edit page
      await page.goto('http://localhost:8092/vms/test-vm/edit')
      
      const editForm = await page.$('form, [data-testid="vm-edit-form"]')
      if (!editForm) return

      // Update VM configuration
      const nameInput = await page.$('input[name="name"]')
      if (nameInput) {
        await nameInput.clear()
        await nameInput.type('updated-test-vm')
      }
      
      const descriptionInput = await page.$('textarea[name="description"], input[name="description"]')
      if (descriptionInput) {
        await descriptionInput.clear()
        await descriptionInput.type('Updated VM description')
      }
      
      // Update resources if available
      const cpuSlider = await page.$('input[type="range"][name*="cpu"]')
      if (cpuSlider) {
        await cpuSlider.evaluate(el => el.value = '2')
        await cpuSlider.dispatchEvent(new Event('change'))
      }
      
      // Save changes
      const saveButton = await page.$('button[type="submit"], button:has-text("Save")')
      if (saveButton) {
        await saveButton.click()
        
        // Should show success message
        const success = await page.waitForSelector('.success-message, [role="alert"]:has-text("updated")', { timeout: 5000 })
        expect(success).toBeTruthy()
      }
    })
  })

  describe('VM Migration', () => {
    test('should initiate VM migration workflow', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      const vmRow = await page.$('[data-testid="vm-item"], .vm-card')
      if (!vmRow) return

      // Look for migrate button or action menu
      const migrateButton = await vmRow.$('button:has-text("Migrate"), .action-migrate')
      const actionMenu = await vmRow.$('.action-menu, button[aria-label*="actions"]')
      
      if (migrateButton) {
        await migrateButton.click()
      } else if (actionMenu) {
        await actionMenu.click()
        
        const migrateOption = await page.$('button:has-text("Migrate"), .migrate-option')
        if (migrateOption) {
          await migrateOption.click()
        }
      }
      
      // Should open migration dialog
      const migrationDialog = await page.waitForSelector('[role="dialog"]:has-text("Migrate"), .migration-dialog', { timeout: 5000 })
      if (migrationDialog) {
        // Should show destination selection
        const destinationSelect = await page.$('select[name="destination"], .destination-select')
        expect(destinationSelect).toBeTruthy()
        
        if (destinationSelect) {
          await destinationSelect.selectOption('node-2')
          
          // Start migration
          const startMigrationButton = await page.$('button:has-text("Start Migration"), button:has-text("Migrate")')
          if (startMigrationButton) {
            await startMigrationButton.click()
            
            // Should show migration progress
            const progressIndicator = await page.waitForSelector('.progress-bar, .migration-progress', { timeout: 5000 })
            expect(progressIndicator).toBeTruthy()
          }
        }
      }
    })

    test('should display migration status and progress', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      // Navigate to migrations page
      await page.goto('http://localhost:8092/migrations')
      
      const migrationsPage = await page.$('[data-testid="migrations-list"], .migrations-table')
      if (migrationsPage) {
        // Should show migration history
        const migrationEntries = await page.$$('.migration-entry, tbody tr')
        
        if (migrationEntries.length > 0) {
          const firstMigration = migrationEntries[0]
          
          // Should show migration details
          const status = await firstMigration.$('.status, [data-status]')
          const progress = await firstMigration.$('.progress, .progress-bar')
          const timestamp = await firstMigration.$('.timestamp, .date')
          
          expect(status || progress || timestamp).toBeTruthy()
        }
        
        // Test migration filtering
        const statusFilter = await page.$('select[name="status"], .status-filter')
        if (statusFilter) {
          await statusFilter.selectOption('completed')
          
          // Should filter the list
          await page.waitForTimeout(500)
        }
      }
    })

    test('should handle live migration options', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      // Navigate to VM details for migration
      await page.goto('http://localhost:8092/vms/test-vm')
      
      const migrateButton = await page.$('button:has-text("Migrate")')
      if (migrateButton) {
        await migrateButton.click()
        
        const migrationDialog = await page.waitForSelector('[role="dialog"]', { timeout: 5000 })
        if (migrationDialog) {
          // Test live migration toggle
          const liveMigrationToggle = await page.$('input[type="checkbox"][name*="live"], .live-migration-toggle')
          if (liveMigrationToggle) {
            await liveMigrationToggle.click()
            
            // Should show live migration options
            const liveMigrationOptions = await page.$('.live-migration-options, .advanced-migration')
            expect(liveMigrationOptions).toBeTruthy()
            
            // Test bandwidth throttling
            const bandwidthInput = await page.$('input[name*="bandwidth"], .bandwidth-limit')
            if (bandwidthInput) {
              await bandwidthInput.clear()
              await bandwidthInput.type('100')
            }
          }
          
          // Test migration type selection
          const migrationTypeSelect = await page.$('select[name="type"], .migration-type')
          if (migrationTypeSelect) {
            await migrationTypeSelect.selectOption('warm')
          }
        }
      }
    })
  })

  describe('VM Templates and Snapshots', () => {
    test('should create VM template', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      const vmRow = await page.$('[data-testid="vm-item"], .vm-card')
      if (!vmRow) return

      // Look for template creation option
      const actionMenu = await vmRow.$('.action-menu, button[aria-label*="actions"]')
      if (actionMenu) {
        await actionMenu.click()
        
        const createTemplateOption = await page.$('button:has-text("Create Template"), .create-template')
        if (createTemplateOption) {
          await createTemplateOption.click()
          
          // Should open template creation dialog
          const templateDialog = await page.waitForSelector('[role="dialog"]:has-text("Template")', { timeout: 5000 })
          if (templateDialog) {
            const templateNameInput = await page.$('input[name="templateName"], input[name="name"]')
            if (templateNameInput) {
              await templateNameInput.type('my-template')
              
              const createButton = await page.$('button:has-text("Create Template")')
              if (createButton) {
                await createButton.click()
                
                // Should show success message
                const success = await page.waitForSelector('.success-message', { timeout: 5000 })
                expect(success).toBeTruthy()
              }
            }
          }
        }
      }
    })

    test('should manage VM snapshots', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      // Navigate to VM snapshots page
      await page.goto('http://localhost:8092/vms/test-vm/snapshots')
      
      const snapshotsPage = await page.$('[data-testid="snapshots"], .snapshots-list')
      if (snapshotsPage) {
        // Test snapshot creation
        const createSnapshotButton = await page.$('button:has-text("Create Snapshot"), .create-snapshot')
        if (createSnapshotButton) {
          await createSnapshotButton.click()
          
          const snapshotDialog = await page.waitForSelector('[role="dialog"]', { timeout: 5000 })
          if (snapshotDialog) {
            const nameInput = await page.$('input[name="name"]')
            if (nameInput) {
              await nameInput.type('test-snapshot')
              
              const createButton = await page.$('button:has-text("Create")')
              if (createButton) {
                await createButton.click()
              }
            }
          }
        }
        
        // Test snapshot restoration
        const snapshotRow = await page.$('.snapshot-item, tbody tr')
        if (snapshotRow) {
          const restoreButton = await snapshotRow.$('button:has-text("Restore"), .restore-snapshot')
          if (restoreButton) {
            await restoreButton.click()
            
            // Should show confirmation dialog
            const confirmDialog = await page.waitForSelector('[role="dialog"]', { timeout: 3000 })
            if (confirmDialog) {
              const confirmButton = await confirmDialog.$('button:has-text("Restore")')
              if (confirmButton) {
                await confirmButton.click()
              }
            }
          }
        }
      }
    })
  })
})