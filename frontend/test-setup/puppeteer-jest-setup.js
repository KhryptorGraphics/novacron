const puppeteer = require('puppeteer')
const fs = require('fs')
const path = require('path')

const WS_ENDPOINT_FILE = path.join(__dirname, '..', 'ws-endpoint')

// Test utilities for Puppeteer E2E tests
global.puppeteerUtils = {
  // Test configuration
  config: {
    baseURL: 'http://localhost:8092',
    apiURL: 'http://localhost:8090',
    defaultTimeout: 10000,
    slowMo: process.env.SLOW_MO ? parseInt(process.env.SLOW_MO) : 0
  },
  
  // Create a new browser page for each test
  async createPage() {
    const endpoint = fs.readFileSync(WS_ENDPOINT_FILE, 'utf8')
    const browser = await puppeteer.connect({ browserWSEndpoint: endpoint })
    const page = await browser.newPage()
    
    // Set viewport
    await page.setViewport({ width: 1920, height: 1080 })
    
    // Set default timeout
    page.setDefaultTimeout(this.config.defaultTimeout)
    
    // Add console logging for debugging
    if (process.env.DEBUG_CONSOLE) {
      page.on('console', msg => {
        console.log(`PAGE LOG: ${msg.text()}`)
      })
      
      page.on('pageerror', error => {
        console.log(`PAGE ERROR: ${error.message}`)
      })
    }
    
    return page
  },
  
  // Navigate to a page and wait for it to load
  async navigateAndWait(page, path = '/') {
    const url = `${this.config.baseURL}${path}`
    await page.goto(url, { waitUntil: 'networkidle2' })
    return page
  },
  
  // Wait for element with better error handling
  async waitForElement(page, selector, options = {}) {
    try {
      return await page.waitForSelector(selector, {
        timeout: options.timeout || this.config.defaultTimeout,
        visible: options.visible !== false,
        ...options
      })
    } catch (error) {
      const screenshot = await page.screenshot({ encoding: 'base64' })
      console.log(`Failed to find element ${selector}. Screenshot: data:image/png;base64,${screenshot}`)
      throw error
    }
  },
  
  // Fill form with validation
  async fillForm(page, formData) {
    for (const [selector, value] of Object.entries(formData)) {
      const element = await this.waitForElement(page, selector)
      await element.click({ clickCount: 3 }) // Select all existing text
      await element.type(value)
      
      // Wait a bit for validation to trigger
      await page.waitForTimeout(100)
    }
  },
  
  // Login helper
  async login(page, credentials = { email: 'test@example.com', password: 'password123' }) {
    await this.navigateAndWait(page, '/auth/login')
    
    await this.fillForm(page, {
      'input[type="email"], input[name="email"]': credentials.email,
      'input[type="password"], input[name="password"]': credentials.password
    })
    
    await page.click('button[type="submit"], button:has-text("Sign In")')
    
    // Wait for navigation or error message
    await Promise.race([
      page.waitForNavigation(),
      page.waitForSelector('[role="alert"], .error-message')
    ])
    
    return page
  },
  
  // API request helper
  async makeApiRequest(method, endpoint, data = null) {
    const url = `${this.config.apiURL}${endpoint}`
    const options = {
      method: method.toUpperCase(),
      headers: {
        'Content-Type': 'application/json',
      }
    }
    
    if (data) {
      options.body = JSON.stringify(data)
    }
    
    const response = await fetch(url, options)
    return {
      status: response.status,
      data: await response.json().catch(() => ({}))
    }
  },
  
  // Screenshot helper for debugging
  async takeScreenshot(page, name) {
    const screenshot = await page.screenshot({ fullPage: true })
    const filename = `debug-${name}-${Date.now()}.png`
    fs.writeFileSync(path.join(__dirname, '..', 'coverage', 'e2e', 'screenshots', filename), screenshot)
    console.log(`Screenshot saved: ${filename}`)
    return filename
  },
  
  // Performance measurement
  async measurePageLoad(page, url) {
    const startTime = Date.now()
    await page.goto(`${this.config.baseURL}${url}`, { waitUntil: 'networkidle2' })
    const loadTime = Date.now() - startTime
    
    const metrics = await page.metrics()
    
    return {
      loadTime,
      metrics: {
        JSHeapUsedSize: metrics.JSHeapUsedSize,
        JSHeapTotalSize: metrics.JSHeapTotalSize,
        LayoutCount: metrics.LayoutCount,
        RecalcStyleCount: metrics.RecalcStyleCount
      }
    }
  },
  
  // Accessibility helper
  async checkAccessibility(page) {
    // Inject axe-core for accessibility testing
    await page.addScriptTag({ path: require.resolve('axe-core/axe.min.js') })
    
    const results = await page.evaluate(() => {
      return new Promise((resolve) => {
        axe.run((err, results) => {
          if (err) throw err
          resolve(results)
        })
      })
    })
    
    return results
  },
  
  // Mobile simulation
  async simulateMobile(page) {
    const mobile = puppeteer.devices['iPhone X']
    await page.emulate(mobile)
    return page
  },
  
  // Network throttling
  async throttleNetwork(page, preset = '3G') {
    const presets = {
      '3G': { downloadThroughput: 1.5 * 1024 * 1024 / 8, uploadThroughput: 750 * 1024 / 8, latency: 40 },
      'Slow 3G': { downloadThroughput: 500 * 1024 / 8, uploadThroughput: 500 * 1024 / 8, latency: 400 },
      'Fast 3G': { downloadThroughput: 1.6 * 1024 * 1024 / 8, uploadThroughput: 750 * 1024 / 8, latency: 150 }
    }
    
    const client = await page.target().createCDPSession()
    await client.send('Network.emulateNetworkConditions', presets[preset])
    return page
  }
}

// Create screenshots directory
const screenshotsDir = path.join(__dirname, '..', 'coverage', 'e2e', 'screenshots')
if (!fs.existsSync(screenshotsDir)) {
  fs.mkdirSync(screenshotsDir, { recursive: true })
}

// Global error handler
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason)
})

// Extend Jest timeout for E2E tests
jest.setTimeout(30000)