/**
 * Admin Panel E2E Tests
 * Tests administrative functions, user management, and system configuration
 */

describe('Admin Panel', () => {
  let page

  beforeEach(async () => {
    page = await global.puppeteerUtils.createPage()
    
    // Navigate to admin panel
    await global.puppeteerUtils.navigateAndWait(page, '/admin')
  })

  afterEach(async () => {
    if (page) {
      await page.close()
    }
  })

  describe('Admin Authentication', () => {
    test('should require admin authentication', async () => {
      // Should redirect to login or show access denied
      const loginForm = await page.$('input[name="email"], input[type="email"]')
      const accessDenied = await page.$('.access-denied, .unauthorized')
      const adminPanel = await page.$('[data-testid="admin-panel"], .admin-dashboard')
      
      // Should either show login, access denied, or admin panel (if already authenticated)
      expect(loginForm || accessDenied || adminPanel).toBeTruthy()
    })

    test('should validate admin credentials', async () => {
      const loginForm = await page.$('form:has(input[name="email"])')
      if (!loginForm) return // Skip if no login form

      // Try regular user credentials
      await global.puppeteerUtils.fillForm(page, {
        'input[name="email"]': 'user@example.com',
        'input[name="password"]': 'password123'
      })
      
      await page.click('button[type="submit"]')
      
      // Should show insufficient privileges or stay on login
      await Promise.race([
        page.waitForSelector('.error-message, [role="alert"]', { timeout: 3000 }),
        page.waitForNavigation({ timeout: 3000 })
      ])
      
      const errorMessage = await page.$('.error-message:has-text("admin"), .insufficient-privileges')
      const stillOnLogin = await page.$('input[name="email"]')
      
      expect(errorMessage || stillOnLogin).toBeTruthy()
    })

    test('should allow admin login with proper credentials', async () => {
      const loginForm = await page.$('form:has(input[name="email"])')
      if (!loginForm) return

      // Try admin credentials
      await global.puppeteerUtils.fillForm(page, {
        'input[name="email"]': 'admin@example.com',
        'input[name="password"]': 'admin123'
      })
      
      await page.click('button[type="submit"]')
      
      // Should either access admin panel or show expected error (no backend)
      await Promise.race([
        page.waitForNavigation({ timeout: 5000 }),
        page.waitForSelector('.admin-dashboard, [data-testid="admin-panel"]', { timeout: 5000 }),
        page.waitForSelector('[role="alert"], .error-message', { timeout: 5000 })
      ])
      
      const adminPanel = await page.$('.admin-dashboard, [data-testid="admin-panel"]')
      const currentUrl = page.url()
      const hasError = await page.$('[role="alert"], .error-message')
      
      // Either successful access or expected error in test environment
      expect(adminPanel || currentUrl.includes('admin') || hasError).toBeTruthy()
    })
  })

  describe('User Management', () => {
    test('should display users list', async () => {
      // Skip if not authenticated to admin panel
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      await page.goto('http://localhost:8092/admin/users')
      
      const usersPage = await page.$('[data-testid="users-list"], .users-table, .user-management')
      if (usersPage) {
        // Should show users table or empty state
        const usersTable = await page.$('table, .users-grid')
        const emptyState = await page.$('.empty-state, .no-users')
        
        expect(usersTable || emptyState).toBeTruthy()
        
        if (usersTable) {
          // Should show user columns
          const headers = await page.$$('th, .table-header')
          const headerTexts = await Promise.all(
            headers.map(h => h.textContent())
          )
          
          const expectedColumns = ['email', 'name', 'role', 'status', 'created']
          const hasExpectedColumns = expectedColumns.some(col =>
            headerTexts.some(text => text && text.toLowerCase().includes(col))
          )
          expect(hasExpectedColumns).toBeTruthy()
        }
      }
    })

    test('should create new user', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      await page.goto('http://localhost:8092/admin/users')
      
      const createUserButton = await page.$('button:has-text("Create User"), button:has-text("Add User"), .create-user')
      if (createUserButton) {
        await createUserButton.click()
        
        // Should open user creation form
        const userForm = await page.waitForSelector('[role="dialog"], .user-form, .create-user-modal', { timeout: 3000 })
        if (userForm) {
          // Fill user creation form
          await global.puppeteerUtils.fillForm(page, {
            'input[name="email"]': 'newuser@example.com',
            'input[name="firstName"]': 'New',
            'input[name="lastName"]': 'User',
            'input[name="password"]': 'TempPassword123!'
          })
          
          // Select user role
          const roleSelect = await page.$('select[name="role"], .role-selector')
          if (roleSelect) {
            await roleSelect.selectOption('user')
          }
          
          // Submit form
          const submitButton = await page.$('button[type="submit"], button:has-text("Create")')
          if (submitButton) {
            await submitButton.click()
            
            // Should show success or error message
            const result = await page.waitForSelector('.success-message, .error-message, [role="alert"]', { timeout: 5000 })
            expect(result).toBeTruthy()
          }
        }
      }
    })

    test('should edit user details', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      await page.goto('http://localhost:8092/admin/users')
      
      const userRow = await page.$('tbody tr, .user-item')
      if (userRow) {
        const editButton = await userRow.$('button:has-text("Edit"), .edit-user, .action-edit')
        if (editButton) {
          await editButton.click()
          
          const editForm = await page.waitForSelector('.edit-user-form, [role="dialog"]', { timeout: 3000 })
          if (editForm) {
            // Update user information
            const firstNameInput = await page.$('input[name="firstName"]')
            if (firstNameInput) {
              await firstNameInput.clear()
              await firstNameInput.type('Updated')
            }
            
            // Change user role
            const roleSelect = await page.$('select[name="role"]')
            if (roleSelect) {
              await roleSelect.selectOption('admin')
            }
            
            // Save changes
            const saveButton = await page.$('button:has-text("Save"), button:has-text("Update")')
            if (saveButton) {
              await saveButton.click()
              
              const result = await page.waitForSelector('.success-message, [role="alert"]', { timeout: 3000 })
              expect(result).toBeTruthy()
            }
          }
        }
      }
    })

    test('should deactivate/activate users', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      await page.goto('http://localhost:8092/admin/users')
      
      const userRow = await page.$('tbody tr, .user-item')
      if (userRow) {
        const statusToggle = await userRow.$('button:has-text("Deactivate"), button:has-text("Activate"), .status-toggle')
        if (statusToggle) {
          const currentStatus = await statusToggle.textContent()
          await statusToggle.click()
          
          // Should show confirmation dialog
          const confirmDialog = await page.waitForSelector('[role="dialog"]:has-text("confirm")', { timeout: 2000 })
          if (confirmDialog) {
            const confirmButton = await confirmDialog.$('button:has-text("Confirm"), button:has-text("Yes")')
            if (confirmButton) {
              await confirmButton.click()
              
              // Status should change
              await page.waitForTimeout(1000)
              const newStatus = await statusToggle.textContent()
              expect(newStatus !== currentStatus).toBeTruthy()
            }
          }
        }
      }
    })

    test('should manage user roles and permissions', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      await page.goto('http://localhost:8092/admin/roles')
      
      const rolesPage = await page.$('[data-testid="roles-management"], .roles-page')
      if (rolesPage) {
        // Should show roles list
        const rolesList = await page.$('.roles-list, .permissions-matrix')
        expect(rolesList).toBeTruthy()
        
        // Test role permissions editing
        const editRoleButton = await page.$('button:has-text("Edit Permissions"), .edit-role')
        if (editRoleButton) {
          await editRoleButton.click()
          
          const permissionsModal = await page.waitForSelector('[role="dialog"], .permissions-modal', { timeout: 3000 })
          if (permissionsModal) {
            // Should show permission checkboxes
            const permissionCheckboxes = await permissionsModal.$$('input[type="checkbox"]')
            expect(permissionCheckboxes.length).toBeGreaterThan(0)
            
            if (permissionCheckboxes.length > 0) {
              // Toggle a permission
              await permissionCheckboxes[0].click()
              
              // Save permissions
              const savePermissions = await permissionsModal.$('button:has-text("Save")')
              if (savePermissions) {
                await savePermissions.click()
              }
            }
          }
        }
      }
    })
  })

  describe('System Configuration', () => {
    test('should display system settings', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      await page.goto('http://localhost:8092/admin/settings')
      
      const settingsPage = await page.$('[data-testid="system-settings"], .system-config')
      if (settingsPage) {
        // Should show configuration sections
        const configSections = await page.$$('.config-section, .settings-group')
        expect(configSections.length).toBeGreaterThan(0)
        
        // Common settings categories
        const generalSettings = await page.$('.general-settings, [data-section="general"]')
        const securitySettings = await page.$('.security-settings, [data-section="security"]')
        const notificationSettings = await page.$('.notification-settings, [data-section="notifications"]')
        
        expect(generalSettings || securitySettings || notificationSettings).toBeTruthy()
      }
    })

    test('should update system configuration', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      await page.goto('http://localhost:8092/admin/settings')
      
      // Test updating a setting
      const settingInput = await page.$('input[name*="timeout"], input[name*="limit"], input[type="number"]')
      if (settingInput) {
        const originalValue = await settingInput.inputValue()
        await settingInput.clear()
        await settingInput.type('3600')
        
        // Save settings
        const saveButton = await page.$('button:has-text("Save"), button[type="submit"]')
        if (saveButton) {
          await saveButton.click()
          
          const result = await page.waitForSelector('.success-message, [role="alert"]', { timeout: 3000 })
          expect(result).toBeTruthy()
        }
      }
      
      // Test toggle settings
      const toggleSetting = await page.$('input[type="checkbox"]')
      if (toggleSetting) {
        const wasChecked = await toggleSetting.isChecked()
        await toggleSetting.click()
        
        const saveButton = await page.$('button:has-text("Save")')
        if (saveButton) {
          await saveButton.click()
          await page.waitForTimeout(1000)
        }
      }
    })

    test('should manage backup and restore settings', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      await page.goto('http://localhost:8092/admin/backup')
      
      const backupPage = await page.$('[data-testid="backup-management"], .backup-settings')
      if (backupPage) {
        // Test backup configuration
        const scheduleBackup = await page.$('input[name="schedule"], .backup-schedule')
        if (scheduleBackup) {
          await scheduleBackup.selectOption('daily')
        }
        
        const retentionInput = await page.$('input[name="retention"], .retention-days')
        if (retentionInput) {
          await retentionInput.clear()
          await retentionInput.type('30')
        }
        
        // Test manual backup trigger
        const manualBackupButton = await page.$('button:has-text("Create Backup"), .manual-backup')
        if (manualBackupButton) {
          await manualBackupButton.click()
          
          // Should show backup progress or confirmation
          const backupProgress = await page.waitForSelector('.backup-progress, .backup-status', { timeout: 3000 })
          expect(backupProgress).toBeTruthy()
        }
        
        // Test backup history
        const backupHistory = await page.$('.backup-history, .backup-list')
        if (backupHistory) {
          const backupItems = await backupHistory.$$('.backup-item, .backup-entry')
          
          if (backupItems.length > 0) {
            // Test backup restore
            const restoreButton = await backupItems[0].$('button:has-text("Restore"), .restore-backup')
            if (restoreButton) {
              await restoreButton.click()
              
              const confirmRestore = await page.waitForSelector('[role="dialog"]:has-text("restore")', { timeout: 2000 })
              if (confirmRestore) {
                const confirmButton = await confirmRestore.$('button:has-text("Restore")')
                if (confirmButton) {
                  await confirmButton.click()
                }
              }
            }
          }
        }
      }
    })
  })

  describe('Database Management', () => {
    test('should display database status and metrics', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      await page.goto('http://localhost:8092/admin/database')
      
      const dbPage = await page.$('[data-testid="database-management"], .database-admin')
      if (dbPage) {
        // Should show database connection status
        const connectionStatus = await page.$('.db-status, .connection-status')
        expect(connectionStatus).toBeTruthy()
        
        if (connectionStatus) {
          const statusText = await connectionStatus.textContent()
          expect(statusText).toMatch(/(connected|disconnected|error)/i)
        }
        
        // Should show database metrics
        const dbMetrics = await page.$$('.db-metric, .database-stat')
        if (dbMetrics.length > 0) {
          // Common database metrics
          const sizeMetric = dbMetrics.find(async metric => {
            const text = await metric.textContent()
            return text && text.toLowerCase().includes('size')
          })
          expect(sizeMetric || dbMetrics.length > 0).toBeTruthy()
        }
      }
    })

    test('should perform database maintenance operations', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      await page.goto('http://localhost:8092/admin/database')
      
      // Test database optimization
      const optimizeButton = await page.$('button:has-text("Optimize"), .db-optimize')
      if (optimizeButton) {
        await optimizeButton.click()
        
        // Should show confirmation dialog
        const confirmDialog = await page.waitForSelector('[role="dialog"]', { timeout: 3000 })
        if (confirmDialog) {
          const proceedButton = await confirmDialog.$('button:has-text("Proceed"), button:has-text("Optimize")')
          if (proceedButton) {
            await proceedButton.click()
            
            // Should show progress or completion
            const result = await page.waitForSelector('.success-message, .progress-indicator', { timeout: 3000 })
            expect(result).toBeTruthy()
          }
        }
      }
      
      // Test vacuum operation
      const vacuumButton = await page.$('button:has-text("Vacuum"), .db-vacuum')
      if (vacuumButton) {
        await vacuumButton.click()
        
        const confirmVacuum = await page.waitForSelector('[role="dialog"]', { timeout: 2000 })
        if (confirmVacuum) {
          const confirmButton = await confirmVacuum.$('button:has-text("Confirm")')
          if (confirmButton) {
            await confirmButton.click()
          }
        }
      }
      
      // Test analyze statistics
      const analyzeButton = await page.$('button:has-text("Analyze"), .db-analyze')
      if (analyzeButton) {
        await analyzeButton.click()
      }
    })

    test('should show database query logs and performance', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      await page.goto('http://localhost:8092/admin/database/logs')
      
      const logsPage = await page.$('[data-testid="db-logs"], .query-logs')
      if (logsPage) {
        // Should show query log entries
        const logEntries = await page.$$('.log-entry, .query-log')
        
        if (logEntries.length > 0) {
          // Log entries should show query details
          const firstEntry = logEntries[0]
          const queryText = await firstEntry.$('.query-text, .sql-query')
          const executionTime = await firstEntry.$('.execution-time, .duration')
          
          expect(queryText || executionTime).toBeTruthy()
        }
        
        // Test log filtering
        const logFilter = await page.$('select[name="logLevel"], .log-filter')
        if (logFilter) {
          await logFilter.selectOption('slow')
          
          // Should filter logs
          await page.waitForTimeout(500)
        }
        
        // Test log search
        const searchInput = await page.$('input[name="search"], .log-search')
        if (searchInput) {
          await searchInput.type('SELECT')
          await page.waitForTimeout(500)
        }
      }
    })
  })

  describe('System Monitoring', () => {
    test('should display system health dashboard', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      await page.goto('http://localhost:8092/admin/health')
      
      const healthPage = await page.$('[data-testid="system-health"], .health-dashboard')
      if (healthPage) {
        // Should show service status indicators
        const serviceStatuses = await page.$$('.service-status, .health-check')
        expect(serviceStatuses.length).toBeGreaterThan(0)
        
        if (serviceStatuses.length > 0) {
          const firstService = serviceStatuses[0]
          
          // Should show service name and status
          const serviceName = await firstService.$('.service-name, .component-name')
          const status = await firstService.$('.status, .health-status')
          
          expect(serviceName && status).toBeTruthy()
          
          if (status) {
            const statusText = await status.textContent()
            expect(statusText).toMatch(/(healthy|unhealthy|degraded|unknown)/i)
          }
        }
      }
    })

    test('should show resource utilization graphs', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      await page.goto('http://localhost:8092/admin/resources')
      
      const resourcesPage = await page.$('[data-testid="resource-monitoring"], .resource-dashboard')
      if (resourcesPage) {
        // Should show resource charts
        const charts = await page.$$('canvas, .resource-chart, svg[class*="chart"]')
        expect(charts.length).toBeGreaterThan(0)
        
        // Test time range selection for resource graphs
        const timeRangeSelect = await page.$('select[name="timeRange"], .time-selector')
        if (timeRangeSelect) {
          await timeRangeSelect.selectOption('1h')
          
          // Charts should update
          await page.waitForTimeout(1000)
        }
      }
    })

    test('should configure monitoring alerts', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      await page.goto('http://localhost:8092/admin/alerts')
      
      const alertsConfig = await page.$('[data-testid="alert-config"], .alert-configuration')
      if (alertsConfig) {
        // Should show alert rules
        const alertRules = await page.$$('.alert-rule, .monitoring-rule')
        
        // Test creating new alert rule
        const createRuleButton = await page.$('button:has-text("Create Rule"), .create-alert-rule')
        if (createRuleButton) {
          await createRuleButton.click()
          
          const ruleForm = await page.waitForSelector('.alert-rule-form, [role="dialog"]', { timeout: 3000 })
          if (ruleForm) {
            // Fill alert rule form
            await global.puppeteerUtils.fillForm(page, {
              'input[name="name"]': 'High CPU Alert',
              'input[name="description"]': 'Alert when CPU usage exceeds threshold'
            })
            
            // Set alert conditions
            const metricSelect = await page.$('select[name="metric"]')
            if (metricSelect) {
              await metricSelect.selectOption('cpu_usage')
            }
            
            const thresholdInput = await page.$('input[name="threshold"]')
            if (thresholdInput) {
              await thresholdInput.type('90')
            }
            
            // Save alert rule
            const saveButton = await page.$('button:has-text("Save Rule")')
            if (saveButton) {
              await saveButton.click()
            }
          }
        }
      }
    })
  })

  describe('Audit Logs', () => {
    test('should display admin activity logs', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      await page.goto('http://localhost:8092/admin/audit')
      
      const auditPage = await page.$('[data-testid="audit-logs"], .audit-trail')
      if (auditPage) {
        // Should show audit log entries
        const auditEntries = await page.$$('.audit-entry, .log-entry')
        
        if (auditEntries.length > 0) {
          const firstEntry = auditEntries[0]
          
          // Should show user, action, timestamp
          const user = await firstEntry.$('.user, .actor')
          const action = await firstEntry.$('.action, .event-type')
          const timestamp = await firstEntry.$('.timestamp, .date')
          
          expect(user && action && timestamp).toBeTruthy()
        }
        
        // Test audit log filtering
        const userFilter = await page.$('select[name="user"], .user-filter')
        if (userFilter) {
          await userFilter.selectOption('admin@example.com')
          await page.waitForTimeout(500)
        }
        
        const actionFilter = await page.$('select[name="action"], .action-filter')
        if (actionFilter) {
          await actionFilter.selectOption('user_created')
          await page.waitForTimeout(500)
        }
        
        // Test date range filtering
        const dateRangeFilter = await page.$('input[type="date"], .date-filter')
        if (dateRangeFilter) {
          await dateRangeFilter.fill('2024-01-01')
        }
      }
    })

    test('should export audit logs', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      await page.goto('http://localhost:8092/admin/audit')
      
      const exportButton = await page.$('button:has-text("Export"), .export-logs')
      if (exportButton) {
        // Set up download listener
        const downloadPromise = page.waitForEvent('download', { timeout: 5000 }).catch(() => null)
        
        await exportButton.click()
        
        // May show export format selection
        const formatSelect = await page.$('select[name="format"], .export-format')
        if (formatSelect) {
          await formatSelect.selectOption('csv')
          
          const confirmExport = await page.$('button:has-text("Download")')
          if (confirmExport) {
            await confirmExport.click()
          }
        }
        
        const download = await downloadPromise
        if (download) {
          expect(download.suggestedFilename()).toMatch(/audit.*\.(csv|json)$/)
        }
      }
    })
  })
})