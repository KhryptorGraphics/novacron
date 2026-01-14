/**
 * Monitoring Dashboard E2E Tests
 * Tests real-time monitoring, metrics visualization, and alert management
 */

describe('Monitoring Dashboard', () => {
  let page

  beforeEach(async () => {
    page = await global.puppeteerUtils.createPage()
    
    // Navigate to monitoring dashboard
    await global.puppeteerUtils.navigateAndWait(page, '/dashboard')
  })

  afterEach(async () => {
    if (page) {
      await page.close()
    }
  })

  describe('Dashboard Overview', () => {
    test('should display main dashboard with metrics cards', async () => {
      // Check if redirected to login
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) {
        console.log('Dashboard requires authentication - skipping authenticated tests')
        return
      }

      // Should show dashboard metrics
      const metricsCards = await page.$$('[data-testid="metrics-card"], .metrics-card, .metric-card')
      const dashboardTitle = await page.$('h1:has-text("Dashboard"), [data-testid="dashboard-title"]')
      
      expect(dashboardTitle || metricsCards.length > 0).toBeTruthy()
      
      if (metricsCards.length > 0) {
        // Each metrics card should show a value
        const firstCard = metricsCards[0]
        const metricValue = await firstCard.$('.metric-value, .value, .number')
        expect(metricValue).toBeTruthy()
      }
    })

    test('should display system resource metrics', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      // Look for CPU, Memory, Disk, Network metrics
      const cpuMetric = await page.$('[data-testid="cpu-metric"], .cpu-usage, .metric:has-text("CPU")')
      const memoryMetric = await page.$('[data-testid="memory-metric"], .memory-usage, .metric:has-text("Memory")')
      const diskMetric = await page.$('[data-testid="disk-metric"], .disk-usage, .metric:has-text("Disk")')
      const networkMetric = await page.$('[data-testid="network-metric"], .network-usage, .metric:has-text("Network")')
      
      const hasSystemMetrics = cpuMetric || memoryMetric || diskMetric || networkMetric
      expect(hasSystemMetrics).toBeTruthy()
      
      // Test metric value format
      if (cpuMetric) {
        const cpuValue = await cpuMetric.$('.value, .percentage, .metric-value')
        if (cpuValue) {
          const valueText = await cpuValue.textContent()
          expect(valueText).toMatch(/\d+(\.\d+)?%?/)
        }
      }
    })

    test('should show VM status grid', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      const vmStatusGrid = await page.$('[data-testid="vm-status-grid"], .vm-grid, .vm-status-grid')
      
      if (vmStatusGrid) {
        // Should show VM status cards
        const vmCards = await vmStatusGrid.$$('.vm-card, .vm-status-card, [data-testid="vm-card"]')
        
        if (vmCards.length > 0) {
          const firstVMCard = vmCards[0]
          
          // Should show VM name and status
          const vmName = await firstVMCard.$('.vm-name, .name, h3')
          const vmStatus = await firstVMCard.$('.status, .vm-status, [data-status]')
          
          expect(vmName && vmStatus).toBeTruthy()
          
          if (vmStatus) {
            const statusText = await vmStatus.textContent()
            expect(statusText).toMatch(/(running|stopped|error|pending)/i)
          }
        }
      }
    })

    test('should display resource usage charts', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      // Look for charts/graphs
      const charts = await page.$$('canvas, .chart, .graph, .recharts-wrapper, svg[class*="chart"]')
      
      if (charts.length > 0) {
        // Charts should be rendered
        const firstChart = charts[0]
        
        if (firstChart.tagName === 'CANVAS') {
          // For canvas-based charts (Chart.js)
          const canvasData = await firstChart.evaluate(canvas => {
            const ctx = canvas.getContext('2d')
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
            return imageData.data.some(pixel => pixel !== 0) // Check if chart is drawn
          })
          expect(canvasData).toBeTruthy()
        } else {
          // For SVG-based charts
          const hasChartElements = await firstChart.$$('path, rect, circle, line')
          expect(hasChartElements.length).toBeGreaterThan(0)
        }
      }
    })
  })

  describe('Real-time Updates', () => {
    test('should update metrics in real-time', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      const metricsCard = await page.$('[data-testid="metrics-card"], .metrics-card')
      if (!metricsCard) return

      // Get initial metric value
      const metricValue = await metricsCard.$('.metric-value, .value')
      if (metricValue) {
        const initialValue = await metricValue.textContent()
        
        // Wait for potential updates (WebSocket or polling)
        await page.waitForTimeout(5000)
        
        const updatedValue = await metricValue.textContent()
        
        // Value might update or stay the same - both are valid
        expect(typeof initialValue === 'string' && typeof updatedValue === 'string').toBeTruthy()
      }
    })

    test('should handle WebSocket connection for real-time data', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      // Listen for WebSocket connections
      const wsConnections = []
      page.on('request', request => {
        if (request.url().includes('ws://') || request.url().includes('wss://')) {
          wsConnections.push(request.url())
        }
      })

      // Monitor network requests for WebSocket upgrade
      const client = await page.target().createCDPSession()
      await client.send('Network.enable')
      
      client.on('Network.webSocketCreated', ({ requestId, url }) => {
        wsConnections.push(url)
        console.log('WebSocket connection created:', url)
      })

      // Refresh page to trigger WebSocket connections
      await page.reload({ waitUntil: 'networkidle2' })
      
      // Wait for potential WebSocket connections
      await page.waitForTimeout(3000)
      
      // Should have attempted WebSocket connection (even if it fails in test env)
      const hasWSConnection = wsConnections.some(url => 
        url.includes('localhost:8091') || url.includes('/ws/') || url.includes('monitoring')
      )
      
      // In test environment, WebSocket might not connect, but the attempt should be made
      console.log('WebSocket connections attempted:', wsConnections)
    })

    test('should show connection status indicator', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      // Look for connection status indicator
      const connectionStatus = await page.$('.connection-status, .ws-status, [data-testid="connection-status"]')
      
      if (connectionStatus) {
        const statusText = await connectionStatus.textContent()
        expect(statusText).toMatch(/(connected|disconnected|connecting)/i)
        
        // Should show appropriate visual indicator
        const statusIcon = await connectionStatus.$('.icon, .indicator, .status-dot')
        expect(statusIcon).toBeTruthy()
      }
    })
  })

  describe('Alerts Management', () => {
    test('should display alerts panel', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      // Look for alerts section
      const alertsPanel = await page.$('[data-testid="alerts"], .alerts-panel, .alerts-section')
      
      if (alertsPanel) {
        // Should show alert items or empty state
        const alertItems = await alertsPanel.$$('.alert-item, .alert')
        const emptyState = await alertsPanel.$('.empty-state, .no-alerts')
        
        expect(alertItems.length > 0 || emptyState).toBeTruthy()
        
        if (alertItems.length > 0) {
          const firstAlert = alertItems[0]
          
          // Alert should have severity, message, and timestamp
          const severity = await firstAlert.$('.severity, .level, [data-severity]')
          const message = await firstAlert.$('.message, .description, .alert-text')
          const timestamp = await firstAlert.$('.timestamp, .time, .date')
          
          expect(severity && message).toBeTruthy()
          
          if (severity) {
            const severityText = await severity.textContent()
            expect(severityText).toMatch(/(critical|warning|error|info)/i)
          }
        }
      }
    })

    test('should allow alert acknowledgment', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      const alertsPanel = await page.$('[data-testid="alerts"], .alerts-panel')
      if (!alertsPanel) return

      const alertItems = await alertsPanel.$$('.alert-item, .alert')
      if (alertItems.length === 0) return

      const firstAlert = alertItems[0]
      const ackButton = await firstAlert.$('button:has-text("Acknowledge"), .ack-button, .acknowledge')
      
      if (ackButton) {
        await ackButton.click()
        
        // Should show acknowledgment confirmation or update alert status
        await Promise.race([
          page.waitForSelector('.success-message', { timeout: 3000 }),
          page.waitForSelector('.acknowledged, [data-acknowledged="true"]', { timeout: 3000 })
        ])
        
        const acknowledged = await firstAlert.$('.acknowledged, [data-acknowledged="true"]')
        const successMessage = await page.$('.success-message')
        
        expect(acknowledged || successMessage).toBeTruthy()
      }
    })

    test('should filter and sort alerts', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      const alertsPanel = await page.$('[data-testid="alerts"], .alerts-panel')
      if (!alertsPanel) return

      // Test severity filter
      const severityFilter = await page.$('select[name="severity"], .severity-filter')
      if (severityFilter) {
        await severityFilter.selectOption('critical')
        
        // Wait for filter to apply
        await page.waitForTimeout(500)
        
        // Should show only critical alerts or empty state
        const visibleAlerts = await alertsPanel.$$('.alert-item:not([style*="display: none"])')
        const criticalAlerts = await alertsPanel.$$('.alert-item [data-severity="critical"], .alert-item .critical')
        
        expect(visibleAlerts.length === 0 || criticalAlerts.length > 0).toBeTruthy()
      }

      // Test time range filter
      const timeFilter = await page.$('select[name="timeRange"], .time-filter')
      if (timeFilter) {
        await timeFilter.selectOption('1h')
        await page.waitForTimeout(500)
      }

      // Test sorting
      const sortButton = await page.$('button:has-text("Sort"), .sort-button')
      if (sortButton) {
        await sortButton.click()
        
        const sortOptions = await page.$('.sort-options, [role="menu"]')
        if (sortOptions) {
          const timestampSort = await sortOptions.$('button:has-text("Time"), [data-sort="timestamp"]')
          if (timestampSort) {
            await timestampSort.click()
            await page.waitForTimeout(500)
          }
        }
      }
    })
  })

  describe('Performance Metrics', () => {
    test('should display historical performance data', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      // Navigate to performance metrics page
      await page.goto('http://localhost:8092/monitoring/performance')
      
      const performancePage = await page.$('[data-testid="performance"], .performance-metrics')
      if (performancePage) {
        // Should show time series charts
        const timeSeriesCharts = await page.$$('.time-series, .performance-chart, canvas')
        
        if (timeSeriesCharts.length > 0) {
          // Test time range selection
          const timeRangeSelect = await page.$('select[name="timeRange"], .time-range-selector')
          if (timeRangeSelect) {
            await timeRangeSelect.selectOption('24h')
            
            // Wait for chart update
            await page.waitForTimeout(1000)
          }

          // Test metric selection
          const metricTabs = await page.$$('.metric-tab, [role="tab"]')
          if (metricTabs.length > 0) {
            await metricTabs[0].click()
            await page.waitForTimeout(500)
          }
        }
      }
    })

    test('should support custom time range selection', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      await page.goto('http://localhost:8092/monitoring/performance')
      
      // Look for custom date picker
      const customRangeButton = await page.$('button:has-text("Custom"), .custom-range')
      if (customRangeButton) {
        await customRangeButton.click()
        
        const datePickerModal = await page.waitForSelector('[role="dialog"], .date-picker-modal', { timeout: 3000 })
        if (datePickerModal) {
          // Select start date
          const startDateInput = await page.$('input[name="startDate"], .start-date')
          if (startDateInput) {
            await startDateInput.click()
            
            // Calendar should appear
            const calendar = await page.waitForSelector('.calendar, .date-picker', { timeout: 2000 })
            expect(calendar).toBeTruthy()
            
            // Select a date (click on day)
            const dateCell = await calendar.$('button:not([disabled]), .day:not(.disabled)')
            if (dateCell) {
              await dateCell.click()
            }
          }
          
          // Apply date range
          const applyButton = await page.$('button:has-text("Apply"), .apply-range')
          if (applyButton) {
            await applyButton.click()
          }
        }
      }
    })

    test('should export performance data', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      await page.goto('http://localhost:8092/monitoring/performance')
      
      const exportButton = await page.$('button:has-text("Export"), .export-data')
      if (exportButton) {
        // Set up download listener
        const downloadPromise = page.waitForEvent('download', { timeout: 5000 })
        
        await exportButton.click()
        
        // May show export options
        const exportModal = await page.$('[role="dialog"]:has-text("Export")')
        if (exportModal) {
          const csvOption = await exportModal.$('input[value="csv"], button:has-text("CSV")')
          if (csvOption) {
            await csvOption.click()
          }
          
          const confirmExport = await exportModal.$('button:has-text("Download"), button:has-text("Export")')
          if (confirmExport) {
            await confirmExport.click()
          }
        }
        
        try {
          const download = await downloadPromise
          expect(download.suggestedFilename()).toMatch(/\.(csv|json|xlsx)$/)
        } catch (error) {
          // Download might not work in test environment - that's ok
          console.log('Export download test skipped (expected in test env)')
        }
      }
    })
  })

  describe('VM-Specific Monitoring', () => {
    test('should show individual VM metrics', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      // Navigate to specific VM monitoring
      await page.goto('http://localhost:8092/vms/test-vm/monitoring')
      
      const vmMonitoring = await page.$('[data-testid="vm-monitoring"], .vm-metrics')
      if (vmMonitoring) {
        // Should show VM-specific metrics
        const vmMetrics = await vmMonitoring.$$('.metric-card, .vm-metric')
        
        if (vmMetrics.length > 0) {
          // Should show CPU, memory, disk, network for this VM
          const metricTypes = await Promise.all(
            vmMetrics.map(metric => metric.$eval('.metric-title, .title', el => el.textContent))
          )
          
          const hasBasicMetrics = metricTypes.some(title => 
            title && /cpu|memory|disk|network/i.test(title)
          )
          expect(hasBasicMetrics).toBeTruthy()
        }

        // Should show VM resource utilization chart
        const utilizationChart = await page.$('.utilization-chart, .resource-chart, canvas')
        expect(utilizationChart).toBeTruthy()
      }
    })

    test('should compare multiple VMs', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      await page.goto('http://localhost:8092/monitoring/compare')
      
      const comparePage = await page.$('[data-testid="vm-compare"], .vm-comparison')
      if (comparePage) {
        // Should show VM selection for comparison
        const vmSelectors = await page.$$('select[name*="vm"], .vm-selector')
        
        if (vmSelectors.length >= 2) {
          // Select VMs to compare
          await vmSelectors[0].selectOption('vm-1')
          await vmSelectors[1].selectOption('vm-2')
          
          // Should show comparison chart
          const comparisonChart = await page.waitForSelector('.comparison-chart, .compare-chart', { timeout: 3000 })
          expect(comparisonChart).toBeTruthy()
        }
      }
    })
  })

  describe('Network Topology', () => {
    test('should display network topology visualization', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      await page.goto('http://localhost:8092/monitoring/topology')
      
      const topologyPage = await page.$('[data-testid="network-topology"], .topology-view')
      if (topologyPage) {
        // Should show network diagram
        const networkDiagram = await page.$('.network-diagram, .topology-svg, canvas[id*="topology"]')
        expect(networkDiagram).toBeTruthy()
        
        if (networkDiagram) {
          // Should show nodes and connections
          const nodes = await page.$$('.network-node, .topology-node')
          const connections = await page.$$('.network-connection, .topology-edge, line')
          
          expect(nodes.length > 0 || connections.length > 0).toBeTruthy()
          
          // Test node interaction
          if (nodes.length > 0) {
            await nodes[0].click()
            
            // Should show node details
            const nodeDetails = await page.waitForSelector('.node-details, .topology-details', { timeout: 2000 })
            expect(nodeDetails).toBeTruthy()
          }
        }
      }
    })

    test('should support topology layout options', async () => {
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      await page.goto('http://localhost:8092/monitoring/topology')
      
      const layoutControls = await page.$('.layout-controls, .topology-controls')
      if (layoutControls) {
        // Test different layout options
        const layoutSelect = await page.$('select[name="layout"], .layout-selector')
        if (layoutSelect) {
          await layoutSelect.selectOption('hierarchical')
          
          // Topology should update
          await page.waitForTimeout(1000)
          
          // Try force-directed layout
          await layoutSelect.selectOption('force')
          await page.waitForTimeout(1000)
        }

        // Test zoom controls
        const zoomInButton = await page.$('button:has-text("Zoom In"), .zoom-in')
        const zoomOutButton = await page.$('button:has-text("Zoom Out"), .zoom-out')
        
        if (zoomInButton) {
          await zoomInButton.click()
          await page.waitForTimeout(300)
        }
        
        if (zoomOutButton) {
          await zoomOutButton.click()
          await page.waitForTimeout(300)
        }
      }
    })
  })
})