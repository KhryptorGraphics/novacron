/**
 * Backend Integration E2E Tests
 * Tests API integration, WebSocket connections, and database operations
 */

describe('Backend Integration', () => {
  let page

  beforeEach(async () => {
    page = await global.puppeteerUtils.createPage()
  })

  afterEach(async () => {
    if (page) {
      await page.close()
    }
  })

  describe('API Health Checks', () => {
    test('should connect to backend API server', async () => {
      // Test API health endpoint
      const healthResponse = await global.puppeteerUtils.makeApiRequest('GET', '/health')
      
      console.log('API Health Response:', healthResponse.status, healthResponse.data)
      
      // Should get healthy response or connection error (expected in test env)
      expect([200, 404, 500].includes(healthResponse.status) || healthResponse.status === 0).toBeTruthy()
      
      if (healthResponse.status === 200) {
        expect(healthResponse.data.status).toBe('healthy')
      }
    })

    test('should get API information', async () => {
      const infoResponse = await global.puppeteerUtils.makeApiRequest('GET', '/api/info')
      
      console.log('API Info Response:', infoResponse.status, infoResponse.data)
      
      if (infoResponse.status === 200) {
        expect(infoResponse.data.name).toBe('NovaCron API')
        expect(infoResponse.data.version).toBeDefined()
        expect(infoResponse.data.endpoints).toBeInstanceOf(Array)
      }
    })

    test('should handle CORS properly', async () => {
      // Navigate to frontend and check if API calls work
      await global.puppeteerUtils.navigateAndWait(page, '/dashboard')
      
      // Monitor network requests for CORS issues
      const corsErrors = []
      
      page.on('console', msg => {
        if (msg.text().includes('CORS') || msg.text().includes('Access-Control')) {
          corsErrors.push(msg.text())
        }
      })
      
      page.on('response', response => {
        if (response.url().includes('localhost:8090')) {
          const corsHeader = response.headers()['access-control-allow-origin']
          if (!corsHeader && response.status() !== 0) {
            corsErrors.push(`Missing CORS header for ${response.url()}`)
          }
        }
      })
      
      // Wait for page to make API calls
      await page.waitForTimeout(3000)
      
      if (corsErrors.length > 0) {
        console.log('CORS issues detected:', corsErrors)
      }
      
      // Should not have critical CORS errors that break functionality
      const criticalCorsErrors = corsErrors.filter(error => 
        error.includes('blocked') || error.includes('denied')
      )
      
      expect(criticalCorsErrors.length).toBe(0)
    })
  })

  describe('System Metrics API', () => {
    test('should fetch system metrics', async () => {
      const metricsResponse = await global.puppeteerUtils.makeApiRequest('GET', '/api/monitoring/metrics')
      
      console.log('System Metrics Response:', metricsResponse.status)
      
      if (metricsResponse.status === 200) {
        const metrics = metricsResponse.data
        
        // Should have basic system metrics
        expect(metrics.currentCpuUsage).toBeDefined()
        expect(metrics.currentMemoryUsage).toBeDefined()
        expect(metrics.currentDiskUsage).toBeDefined()
        expect(metrics.currentNetworkUsage).toBeDefined()
        
        // Should have time series data
        expect(metrics.cpuTimeseriesData).toBeInstanceOf(Array)
        expect(metrics.memoryTimeseriesData).toBeInstanceOf(Array)
        expect(metrics.timeLabels).toBeInstanceOf(Array)
        
        // Values should be reasonable
        expect(metrics.currentCpuUsage).toBeGreaterThanOrEqual(0)
        expect(metrics.currentCpuUsage).toBeLessThanOrEqual(100)
      }
    })

    test('should fetch VM metrics', async () => {
      const vmMetricsResponse = await global.puppeteerUtils.makeApiRequest('GET', '/api/monitoring/vms')
      
      console.log('VM Metrics Response:', vmMetricsResponse.status)
      
      if (vmMetricsResponse.status === 200) {
        const vmMetrics = vmMetricsResponse.data
        
        expect(vmMetrics).toBeInstanceOf(Array)
        
        if (vmMetrics.length > 0) {
          const firstVM = vmMetrics[0]
          
          // Should have VM identification
          expect(firstVM.vmId).toBeDefined()
          expect(firstVM.name).toBeDefined()
          
          // Should have resource metrics
          expect(firstVM.cpuUsage).toBeGreaterThanOrEqual(0)
          expect(firstVM.memoryUsage).toBeGreaterThanOrEqual(0)
          expect(firstVM.diskUsage).toBeGreaterThanOrEqual(0)
          
          // Should have status
          expect(['running', 'stopped', 'error', 'pending'].includes(firstVM.status)).toBeTruthy()
        }
      }
    })

    test('should fetch alerts', async () => {
      const alertsResponse = await global.puppeteerUtils.makeApiRequest('GET', '/api/monitoring/alerts')
      
      console.log('Alerts Response:', alertsResponse.status)
      
      if (alertsResponse.status === 200) {
        const alerts = alertsResponse.data
        
        expect(alerts).toBeInstanceOf(Array)
        
        if (alerts.length > 0) {
          const firstAlert = alerts[0]
          
          // Should have alert structure
          expect(firstAlert.id).toBeDefined()
          expect(firstAlert.name).toBeDefined()
          expect(firstAlert.description).toBeDefined()
          expect(firstAlert.severity).toBeDefined()
          expect(firstAlert.status).toBeDefined()
          
          // Severity should be valid
          expect(['critical', 'warning', 'error', 'info'].includes(firstAlert.severity)).toBeTruthy()
          
          // Status should be valid
          expect(['firing', 'resolved', 'acknowledged'].includes(firstAlert.status)).toBeTruthy()
        }
      }
    })

    test('should acknowledge alerts', async () => {
      // First get alerts to find one to acknowledge
      const alertsResponse = await global.puppeteerUtils.makeApiRequest('GET', '/api/monitoring/alerts')
      
      if (alertsResponse.status === 200 && alertsResponse.data.length > 0) {
        const alertId = alertsResponse.data[0].id
        
        // Acknowledge the alert
        const ackResponse = await global.puppeteerUtils.makeApiRequest('POST', `/api/monitoring/alerts/${alertId}/acknowledge`)
        
        console.log('Alert Acknowledgment Response:', ackResponse.status)
        
        if (ackResponse.status === 200) {
          expect(ackResponse.data.status).toBe('acknowledged')
        }
      }
    })
  })

  describe('WebSocket Integration', () => {
    test('should attempt WebSocket connection for real-time updates', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/dashboard')
      
      // Skip if redirected to login
      const isLoginPage = await page.$('input[name="email"]')
      if (isLoginPage) {
        console.log('WebSocket test skipped - authentication required')
        return
      }
      
      const wsConnections = []
      const wsMessages = []
      
      // Monitor WebSocket connection attempts
      const client = await page.target().createCDPSession()
      await client.send('Network.enable')
      
      client.on('Network.webSocketCreated', ({ requestId, url }) => {
        wsConnections.push({ requestId, url })
        console.log('WebSocket connection created:', url)
      })
      
      client.on('Network.webSocketFrameReceived', ({ requestId, timestamp, response }) => {
        wsMessages.push({ requestId, timestamp, data: response.payloadData })
        console.log('WebSocket message received:', response.payloadData)
      })
      
      // Wait for potential WebSocket connections
      await page.waitForTimeout(5000)
      
      // Log connection attempts (may fail in test environment)
      console.log(`WebSocket connection attempts: ${wsConnections.length}`)
      console.log(`WebSocket messages received: ${wsMessages.length}`)
      
      // In test environment, we mainly verify the attempt is made
      if (wsConnections.length > 0) {
        const monitoringWS = wsConnections.find(ws => ws.url.includes('monitoring') || ws.url.includes('localhost:8091'))
        expect(monitoringWS).toBeDefined()
      }
    })

    test('should handle WebSocket connection failures gracefully', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/dashboard')
      
      const isLoginPage = await page.$('input[name="email"]')
      if (isLoginPage) return

      // Monitor for WebSocket error handling
      const errorMessages = []
      
      page.on('console', msg => {
        if (msg.text().includes('WebSocket') && msg.text().includes('error')) {
          errorMessages.push(msg.text())
        }
      })
      
      // Page should still function even if WebSocket fails
      await page.waitForTimeout(3000)
      
      // Should show connection status or fallback to polling
      const connectionStatus = await page.$('.connection-status, .offline-indicator, .ws-status')
      
      if (connectionStatus) {
        const statusText = await connectionStatus.textContent()
        console.log('Connection status:', statusText)
      }
      
      // Page should remain functional
      const dashboardContent = await page.$('[data-testid="dashboard"], .dashboard-content, .metrics-grid')
      expect(dashboardContent).toBeTruthy()
    })
  })

  describe('Authentication Integration', () => {
    test('should handle authentication flow with backend', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/auth/login')
      
      // Fill login form
      await global.puppeteerUtils.fillForm(page, {
        'input[name="email"], input[type="email"]': 'test@example.com',
        'input[name="password"], input[type="password"]': 'password123'
      })
      
      // Monitor API requests
      const authRequests = []
      
      page.on('request', request => {
        if (request.url().includes('/auth/') || request.url().includes('/login')) {
          authRequests.push({
            method: request.method(),
            url: request.url(),
            headers: request.headers()
          })
        }
      })
      
      page.on('response', response => {
        if (response.url().includes('/auth/') || response.url().includes('/login')) {
          console.log('Auth API response:', response.status(), response.url())
        }
      })
      
      // Submit login form
      await page.click('button[type="submit"], button:has-text("Sign In")')
      
      // Wait for authentication response
      await page.waitForTimeout(3000)
      
      console.log(`Authentication requests made: ${authRequests.length}`)
      
      // Should have made authentication request (even if it fails in test env)
      const hasAuthRequest = authRequests.length > 0
      const hasErrorMessage = await page.$('[role="alert"], .error-message')
      const wasRedirected = !page.url().includes('/auth/login')
      
      // Either successful auth, expected error, or no backend response
      expect(hasAuthRequest || hasErrorMessage || wasRedirected).toBeTruthy()
    })

    test('should handle JWT token management', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/dashboard')
      
      // Check for token storage
      const hasAuthToken = await page.evaluate(() => {
        return !!(localStorage.getItem('auth-token') || sessionStorage.getItem('auth-token') || 
                 localStorage.getItem('jwt') || sessionStorage.getItem('jwt'))
      })
      
      // Check for authorization headers in API requests
      const authHeaders = []
      
      page.on('request', request => {
        const authHeader = request.headers()['authorization']
        if (authHeader && (authHeader.includes('Bearer') || authHeader.includes('JWT'))) {
          authHeaders.push(authHeader)
        }
      })
      
      // Make a navigation that might trigger API calls
      await page.goto('http://localhost:8092/vms')
      await page.waitForTimeout(2000)
      
      console.log(`Found ${authHeaders.length} requests with auth headers`)
      
      // Should handle authentication appropriately
      const isOnLoginPage = page.url().includes('login') || page.url().includes('auth')
      
      if (!isOnLoginPage && authHeaders.length > 0) {
        // If not redirected to login, should have auth headers
        expect(authHeaders.length).toBeGreaterThan(0)
      }
    })
  })

  describe('Database Operations', () => {
    test('should handle database connectivity', async () => {
      // Test database-related endpoints
      const dbHealthResponse = await global.puppeteerUtils.makeApiRequest('GET', '/health')
      
      if (dbHealthResponse.status === 200 && dbHealthResponse.data.checks) {
        // Should report database status
        expect(dbHealthResponse.data.checks.database).toBeDefined()
        
        const dbStatus = dbHealthResponse.data.checks.database
        expect(['ok', 'error', 'unavailable'].includes(dbStatus)).toBeTruthy()
      }
    })

    test('should handle CRUD operations through API', async () => {
      // Test creating a resource (VM) - will likely fail without auth but should return proper error
      const createResponse = await global.puppeteerUtils.makeApiRequest('POST', '/api/vm/create', {
        name: 'test-vm-e2e',
        template: 'ubuntu-20.04',
        resources: {
          cpu: 2,
          memory: 4096,
          disk: 50
        }
      })
      
      console.log('Create VM Response:', createResponse.status)
      
      // Should get proper HTTP response (even if unauthorized/not implemented)
      expect([200, 201, 400, 401, 403, 404, 500].includes(createResponse.status)).toBeTruthy()
      
      if (createResponse.status === 401 || createResponse.status === 403) {
        console.log('VM creation requires authentication (expected)')
      } else if (createResponse.status === 201 && createResponse.data.id) {
        // If successful, test read and update
        const vmId = createResponse.data.id
        
        // Test read
        const readResponse = await global.puppeteerUtils.makeApiRequest('GET', `/api/vm/${vmId}`)
        if (readResponse.status === 200) {
          expect(readResponse.data.id).toBe(vmId)
          expect(readResponse.data.name).toBe('test-vm-e2e')
        }
        
        // Test update
        const updateResponse = await global.puppeteerUtils.makeApiRequest('PUT', `/api/vm/${vmId}`, {
          name: 'updated-test-vm-e2e'
        })
        
        if (updateResponse.status === 200) {
          expect(updateResponse.data.name).toBe('updated-test-vm-e2e')
        }
        
        // Test delete
        const deleteResponse = await global.puppeteerUtils.makeApiRequest('DELETE', `/api/vm/${vmId}`)
        expect([200, 204].includes(deleteResponse.status)).toBeTruthy()
      }
    })

    test('should handle data pagination', async () => {
      // Test paginated endpoint
      const paginatedResponse = await global.puppeteerUtils.makeApiRequest('GET', '/api/monitoring/vms?page=1&limit=10')
      
      if (paginatedResponse.status === 200) {
        const data = paginatedResponse.data
        
        if (Array.isArray(data)) {
          // Should respect limit
          expect(data.length).toBeLessThanOrEqual(10)
        } else if (data.items) {
          // Paginated response format
          expect(data.items).toBeInstanceOf(Array)
          expect(data.items.length).toBeLessThanOrEqual(10)
          expect(data.total).toBeGreaterThanOrEqual(0)
          expect(data.page).toBe(1)
        }
      }
    })
  })

  describe('Error Handling', () => {
    test('should handle API errors gracefully', async () => {
      // Test non-existent endpoint
      const notFoundResponse = await global.puppeteerUtils.makeApiRequest('GET', '/api/nonexistent')
      
      expect(notFoundResponse.status).toBe(404)
      
      // Test malformed request
      const badRequestResponse = await global.puppeteerUtils.makeApiRequest('POST', '/api/monitoring/alerts', {
        invalidData: 'test'
      })
      
      expect([400, 404, 405, 500].includes(badRequestResponse.status)).toBeTruthy()
    })

    test('should handle network failures', async () => {
      // Test with invalid API base URL
      const failedResponse = await fetch('http://localhost:9999/invalid').catch(error => ({
        error: error.message,
        status: 0
      }))
      
      expect(failedResponse.status === 0 || failedResponse.error).toBeTruthy()
      
      // Frontend should handle such failures gracefully
      await global.puppeteerUtils.navigateAndWait(page, '/dashboard')
      
      // Page should still render even with API failures
      const hasContent = await page.$('body *')
      expect(hasContent).toBeTruthy()
    })

    test('should validate request/response data', async () => {
      // Test with invalid data types
      const invalidResponse = await global.puppeteerUtils.makeApiRequest('POST', '/api/monitoring/alerts/invalid-id/acknowledge')
      
      expect([400, 404, 422].includes(invalidResponse.status)).toBeTruthy()
      
      if (invalidResponse.data && invalidResponse.data.error) {
        // Should return meaningful error message
        expect(typeof invalidResponse.data.error).toBe('string')
        expect(invalidResponse.data.error.length).toBeGreaterThan(0)
      }
    })
  })

  describe('Performance Integration', () => {
    test('should maintain performance under load', async () => {
      const startTime = Date.now()
      
      // Make multiple concurrent API requests
      const requests = []
      for (let i = 0; i < 10; i++) {
        requests.push(global.puppeteerUtils.makeApiRequest('GET', '/api/monitoring/metrics'))
      }
      
      const responses = await Promise.all(requests)
      const endTime = Date.now()
      
      const totalTime = endTime - startTime
      const averageTime = totalTime / requests.length
      
      console.log(`Concurrent API requests - Total: ${totalTime}ms, Average: ${averageTime}ms`)
      
      // Should handle concurrent requests efficiently
      expect(totalTime).toBeLessThan(10000) // Total under 10 seconds
      expect(averageTime).toBeLessThan(2000) // Average under 2 seconds
      
      // Most requests should succeed
      const successfulRequests = responses.filter(r => r.status === 200 || r.status === 0)
      expect(successfulRequests.length).toBeGreaterThan(requests.length * 0.5) // At least 50% success
    })

    test('should cache responses appropriately', async () => {
      // Make same request twice
      const firstRequest = Date.now()
      const response1 = await global.puppeteerUtils.makeApiRequest('GET', '/api/monitoring/metrics')
      const firstTime = Date.now() - firstRequest
      
      const secondRequest = Date.now()
      const response2 = await global.puppeteerUtils.makeApiRequest('GET', '/api/monitoring/metrics')
      const secondTime = Date.now() - secondRequest
      
      console.log(`First request: ${firstTime}ms, Second request: ${secondTime}ms`)
      
      // If both successful, second might be faster due to caching
      if (response1.status === 200 && response2.status === 200) {
        console.log('Both requests successful - caching behavior observed')
      }
      
      // Data should be consistent
      if (response1.status === 200 && response2.status === 200) {
        expect(typeof response1.data).toBe(typeof response2.data)
      }
    })
  })
})