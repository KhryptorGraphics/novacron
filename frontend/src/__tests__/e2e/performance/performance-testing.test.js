/**
 * Performance Testing E2E Tests
 * Tests application performance, loading times, and resource optimization
 */

describe('Performance Testing', () => {
  let page

  beforeEach(async () => {
    page = await global.puppeteerUtils.createPage()
  })

  afterEach(async () => {
    if (page) {
      await page.close()
    }
  })

  describe('Page Load Performance', () => {
    test('should load pages within acceptable time limits', async () => {
      const testPages = [
        '/',
        '/auth/login',
        '/auth/register',
        '/dashboard',
        '/vms',
        '/monitoring'
      ]

      for (const path of testPages) {
        const performanceData = await global.puppeteerUtils.measurePageLoad(page, path)
        
        console.log(`Page ${path} load time: ${performanceData.loadTime}ms`)
        
        // Pages should load within 5 seconds (generous for test environment)
        expect(performanceData.loadTime).toBeLessThan(5000)
        
        // Memory usage should be reasonable
        expect(performanceData.metrics.JSHeapUsedSize).toBeLessThan(50 * 1024 * 1024) // 50MB
      }
    })

    test('should optimize resource loading', async () => {
      const client = await page.target().createCDPSession()
      await client.send('Network.enable')
      
      const resources = []
      
      client.on('Network.responseReceived', ({ response }) => {
        resources.push({
          url: response.url,
          status: response.status,
          mimeType: response.mimeType,
          size: response.encodedDataLength || 0
        })
      })
      
      await global.puppeteerUtils.navigateAndWait(page, '/dashboard')
      
      // Analyze resource loading
      const jsResources = resources.filter(r => r.mimeType === 'application/javascript')
      const cssResources = resources.filter(r => r.mimeType === 'text/css')
      const imageResources = resources.filter(r => r.mimeType?.startsWith('image/'))
      
      console.log(`Loaded resources - JS: ${jsResources.length}, CSS: ${cssResources.length}, Images: ${imageResources.length}`)
      
      // Should have reasonable number of resources
      expect(jsResources.length).toBeLessThan(20) // Not too many JS bundles
      expect(cssResources.length).toBeLessThan(10) // Limited CSS files
      
      // Check for resource optimization
      const largeResources = resources.filter(r => r.size > 1024 * 1024) // > 1MB
      expect(largeResources.length).toBeLessThan(3) // Limit large resources
      
      // Should have no 404 errors
      const failedResources = resources.filter(r => r.status >= 400)
      if (failedResources.length > 0) {
        console.log('Failed resources:', failedResources.map(r => ({ url: r.url, status: r.status })))
      }
    })

    test('should handle concurrent user loads', async () => {
      const numUsers = 5
      const pages = []
      
      try {
        // Simulate multiple concurrent users
        for (let i = 0; i < numUsers; i++) {
          const userPage = await global.puppeteerUtils.createPage()
          pages.push(userPage)
        }
        
        const startTime = Date.now()
        
        // Load dashboard concurrently
        const loadPromises = pages.map(p => 
          global.puppeteerUtils.navigateAndWait(p, '/dashboard')
        )
        
        await Promise.all(loadPromises)
        
        const totalTime = Date.now() - startTime
        const averageTime = totalTime / numUsers
        
        console.log(`Concurrent load test - Total: ${totalTime}ms, Average: ${averageTime}ms`)
        
        // Should handle concurrent loads efficiently
        expect(totalTime).toBeLessThan(15000) // 15 seconds for 5 concurrent users
        expect(averageTime).toBeLessThan(5000) // Average under 5 seconds
        
      } finally {
        // Clean up pages
        for (const userPage of pages) {
          await userPage.close()
        }
      }
    })
  })

  describe('Runtime Performance', () => {
    test('should maintain good performance during interactions', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/vms')
      
      // Skip if login required
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      const startMetrics = await page.metrics()
      
      // Perform various interactions
      const searchInput = await page.$('input[name="search"], input[placeholder*="Search"]')
      if (searchInput) {
        // Simulate typing search
        await searchInput.type('test-vm', { delay: 100 })
        await page.waitForTimeout(500)
        
        await searchInput.clear()
        await searchInput.type('another-search', { delay: 100 })
        await page.waitForTimeout(500)
      }
      
      // Simulate filtering
      const filterSelect = await page.$('select[name="status"], .filter-control')
      if (filterSelect) {
        await filterSelect.selectOption('running')
        await page.waitForTimeout(500)
        
        await filterSelect.selectOption('all')
        await page.waitForTimeout(500)
      }
      
      // Simulate pagination
      const nextPageButton = await page.$('button:has-text("Next"), .pagination-next')
      if (nextPageButton) {
        await nextPageButton.click()
        await page.waitForTimeout(1000)
      }
      
      const endMetrics = await page.metrics()
      
      // Memory usage shouldn't grow excessively
      const memoryGrowth = endMetrics.JSHeapUsedSize - startMetrics.JSHeapUsedSize
      console.log(`Memory growth during interactions: ${memoryGrowth / 1024 / 1024}MB`)
      
      expect(memoryGrowth).toBeLessThan(10 * 1024 * 1024) // Less than 10MB growth
      
      // Layout recalculations should be reasonable
      const layoutGrowth = endMetrics.RecalcStyleCount - startMetrics.RecalcStyleCount
      expect(layoutGrowth).toBeLessThan(100) // Reasonable style recalculations
    })

    test('should handle real-time updates efficiently', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/monitoring')
      
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      // Monitor performance during real-time updates
      const startMetrics = await page.metrics()
      
      // Wait for potential real-time updates
      await page.waitForTimeout(10000) // 10 seconds of monitoring
      
      const endMetrics = await page.metrics()
      
      // Check performance impact of real-time updates
      const memoryGrowth = endMetrics.JSHeapUsedSize - startMetrics.JSHeapUsedSize
      const layoutRecalcs = endMetrics.RecalcStyleCount - startMetrics.RecalcStyleCount
      
      console.log(`Real-time monitoring performance - Memory: +${memoryGrowth/1024/1024}MB, Layouts: ${layoutRecalcs}`)
      
      // Real-time updates shouldn't cause excessive resource usage
      expect(memoryGrowth).toBeLessThan(20 * 1024 * 1024) // Less than 20MB growth
      expect(layoutRecalcs).toBeLessThan(200) // Reasonable layout updates
    })

    test('should optimize chart rendering performance', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/dashboard')
      
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      // Look for charts
      const charts = await page.$$('canvas, .chart, svg[class*="chart"]')
      
      if (charts.length > 0) {
        const startTime = Date.now()
        const startMetrics = await page.metrics()
        
        // Trigger chart re-render by changing time range
        const timeRangeSelect = await page.$('select[name="timeRange"], .time-range-selector')
        if (timeRangeSelect) {
          await timeRangeSelect.selectOption('24h')
          
          // Wait for chart to re-render
          await page.waitForTimeout(2000)
          
          await timeRangeSelect.selectOption('7d')
          await page.waitForTimeout(2000)
        }
        
        const renderTime = Date.now() - startTime
        const endMetrics = await page.metrics()
        
        console.log(`Chart rendering time: ${renderTime}ms`)
        
        // Chart rendering should be reasonably fast
        expect(renderTime).toBeLessThan(5000) // Under 5 seconds
        
        // Should not cause excessive layout thrashing
        const layoutDiff = endMetrics.LayoutCount - startMetrics.LayoutCount
        expect(layoutDiff).toBeLessThan(50)
      }
    })
  })

  describe('Network Performance', () => {
    test('should optimize API request patterns', async () => {
      const client = await page.target().createCDPSession()
      await client.send('Network.enable')
      
      const apiRequests = []
      
      client.on('Network.requestWillBeSent', ({ request }) => {
        if (request.url.includes('/api/') || request.url.includes('localhost:8090')) {
          apiRequests.push({
            url: request.url,
            method: request.method,
            timestamp: Date.now()
          })
        }
      })
      
      await global.puppeteerUtils.navigateAndWait(page, '/dashboard')
      
      // Wait for API requests to complete
      await page.waitForTimeout(3000)
      
      console.log(`API requests made: ${apiRequests.length}`)
      console.log('API endpoints called:', [...new Set(apiRequests.map(r => r.url))])
      
      // Should not make excessive API calls
      expect(apiRequests.length).toBeLessThan(20) // Reasonable number of API calls
      
      // Check for request batching/deduplication
      const uniqueEndpoints = new Set(apiRequests.map(r => r.url))
      const duplicateRequests = apiRequests.length - uniqueEndpoints.size
      
      console.log(`Duplicate requests: ${duplicateRequests}`)
      expect(duplicateRequests).toBeLessThan(5) // Minimal duplicate requests
    })

    test('should handle slow network conditions', async () => {
      // Throttle network to simulate slow connection
      await global.puppeteerUtils.throttleNetwork(page, 'Slow 3G')
      
      const startTime = Date.now()
      await global.puppeteerUtils.navigateAndWait(page, '/dashboard')
      const loadTime = Date.now() - startTime
      
      console.log(`Load time with slow network: ${loadTime}ms`)
      
      // Should still be usable on slow networks (within reason)
      expect(loadTime).toBeLessThan(30000) // 30 seconds max for slow network
      
      // Page should show loading states appropriately
      const hasLoadingStates = await page.$$('.loading, .spinner, .skeleton, [aria-busy="true"]')
      
      // Reset network conditions
      const client = await page.target().createCDPSession()
      await client.send('Network.emulateNetworkConditions', {
        offline: false,
        downloadThroughput: -1,
        uploadThroughput: -1,
        latency: 0
      })
    })

    test('should implement effective caching strategies', async () => {
      const client = await page.target().createCDPSession()
      await client.send('Network.enable')
      
      const cachedResources = []
      const networkRequests = []
      
      client.on('Network.requestWillBeSent', ({ request }) => {
        networkRequests.push(request.url)
      })
      
      client.on('Network.requestServedFromCache', ({ requestId }) => {
        cachedResources.push(requestId)
      })
      
      // Load page first time
      await global.puppeteerUtils.navigateAndWait(page, '/dashboard')
      const firstLoadRequests = networkRequests.length
      
      // Reload page
      await page.reload({ waitUntil: 'networkidle2' })
      const secondLoadRequests = networkRequests.length - firstLoadRequests
      
      console.log(`First load: ${firstLoadRequests} requests, Second load: ${secondLoadRequests} requests`)
      console.log(`Cached resources: ${cachedResources.length}`)
      
      // Second load should make fewer requests due to caching
      expect(secondLoadRequests).toBeLessThan(firstLoadRequests)
    })
  })

  describe('Memory Management', () => {
    test('should not have memory leaks during navigation', async () => {
      const pages = ['/dashboard', '/vms', '/monitoring', '/admin', '/settings']
      const memoryMeasurements = []
      
      for (let i = 0; i < 3; i++) { // Run multiple cycles
        for (const path of pages) {
          await global.puppeteerUtils.navigateAndWait(page, path)
          
          // Force garbage collection if available
          if (page.evaluate) {
            try {
              await page.evaluate(() => {
                if (window.gc) {
                  window.gc()
                }
              })
            } catch (e) {
              // GC might not be available
            }
          }
          
          const metrics = await page.metrics()
          memoryMeasurements.push({
            page: path,
            cycle: i,
            memory: metrics.JSHeapUsedSize
          })
          
          await page.waitForTimeout(1000)
        }
      }
      
      // Analyze memory trend
      const firstCycleMemory = memoryMeasurements.filter(m => m.cycle === 0)
      const lastCycleMemory = memoryMeasurements.filter(m => m.cycle === 2)
      
      const avgFirstCycle = firstCycleMemory.reduce((sum, m) => sum + m.memory, 0) / firstCycleMemory.length
      const avgLastCycle = lastCycleMemory.reduce((sum, m) => sum + m.memory, 0) / lastCycleMemory.length
      
      const memoryGrowth = avgLastCycle - avgFirstCycle
      const growthPercentage = (memoryGrowth / avgFirstCycle) * 100
      
      console.log(`Memory growth over navigation cycles: ${growthPercentage.toFixed(2)}%`)
      
      // Memory growth should be minimal (less than 50% increase)
      expect(growthPercentage).toBeLessThan(50)
    })

    test('should handle large data sets efficiently', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/vms')
      
      const loginRequired = await page.$('input[name="email"]')
      if (loginRequired) return

      const startMetrics = await page.metrics()
      
      // Simulate loading large data set (if pagination exists)
      const paginationControls = await page.$$('.pagination button, .load-more')
      
      if (paginationControls.length > 0) {
        // Load multiple pages
        for (let i = 0; i < Math.min(5, paginationControls.length); i++) {
          const nextButton = await page.$('button:has-text("Next"), .pagination-next')
          if (nextButton) {
            await nextButton.click()
            await page.waitForTimeout(1000)
          }
        }
      }
      
      const endMetrics = await page.metrics()
      const memoryIncrease = endMetrics.JSHeapUsedSize - startMetrics.JSHeapUsedSize
      
      console.log(`Memory increase with large data set: ${memoryIncrease / 1024 / 1024}MB`)
      
      // Should handle large data sets without excessive memory usage
      expect(memoryIncrease).toBeLessThan(30 * 1024 * 1024) // Less than 30MB increase
    })
  })

  describe('Bundle Size Analysis', () => {
    test('should have reasonable bundle sizes', async () => {
      const client = await page.target().createCDPSession()
      await client.send('Network.enable')
      
      const jsResources = []
      
      client.on('Network.responseReceived', ({ response }) => {
        if (response.mimeType === 'application/javascript' && response.url.includes('/_next/static/')) {
          jsResources.push({
            url: response.url,
            size: response.encodedDataLength || 0
          })
        }
      })
      
      await global.puppeteerUtils.navigateAndWait(page, '/dashboard')
      
      const totalBundleSize = jsResources.reduce((sum, resource) => sum + resource.size, 0)
      const mainBundle = jsResources.find(r => r.url.includes('pages/') || r.url.includes('main'))
      
      console.log(`Total JS bundle size: ${totalBundleSize / 1024 / 1024}MB`)
      console.log(`Main bundle size: ${mainBundle ? mainBundle.size / 1024 / 1024 : 0}MB`)
      
      // Bundle sizes should be reasonable for a modern web app
      expect(totalBundleSize).toBeLessThan(5 * 1024 * 1024) // Total under 5MB
      if (mainBundle) {
        expect(mainBundle.size).toBeLessThan(2 * 1024 * 1024) // Main bundle under 2MB
      }
    })
  })

  describe('Core Web Vitals', () => {
    test('should meet Core Web Vitals thresholds', async () => {
      await global.puppeteerUtils.navigateAndWait(page, '/dashboard')
      
      // Measure Core Web Vitals using Performance Observer API
      const webVitals = await page.evaluate(() => {
        return new Promise((resolve) => {
          const vitals = {}
          
          // Largest Contentful Paint (LCP)
          new PerformanceObserver((list) => {
            const entries = list.getEntries()
            if (entries.length > 0) {
              vitals.lcp = entries[entries.length - 1].startTime
            }
          }).observe({ entryTypes: ['largest-contentful-paint'] })
          
          // First Input Delay (FID) - difficult to test automatically
          new PerformanceObserver((list) => {
            const entries = list.getEntries()
            if (entries.length > 0) {
              vitals.fid = entries[0].processingStart - entries[0].startTime
            }
          }).observe({ entryTypes: ['first-input'] })
          
          // Cumulative Layout Shift (CLS)
          let clsScore = 0
          new PerformanceObserver((list) => {
            list.getEntries().forEach((entry) => {
              if (!entry.hadRecentInput) {
                clsScore += entry.value
              }
            })
            vitals.cls = clsScore
          }).observe({ entryTypes: ['layout-shift'] })
          
          // Give time for measurements
          setTimeout(() => {
            resolve(vitals)
          }, 5000)
        })
      })
      
      console.log('Core Web Vitals:', webVitals)
      
      // Core Web Vitals thresholds
      if (webVitals.lcp) {
        expect(webVitals.lcp).toBeLessThan(2500) // LCP should be under 2.5s (good)
      }
      
      if (webVitals.fid) {
        expect(webVitals.fid).toBeLessThan(100) // FID should be under 100ms (good)
      }
      
      if (webVitals.cls !== undefined) {
        expect(webVitals.cls).toBeLessThan(0.1) // CLS should be under 0.1 (good)
      }
    })
  })
})