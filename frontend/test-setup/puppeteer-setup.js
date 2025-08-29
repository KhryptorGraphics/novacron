const { execSync } = require('child_process')
const puppeteer = require('puppeteer')
const path = require('path')
const fs = require('fs')

const DIR = path.join(__dirname, '..')
const WS_ENDPOINT_FILE = path.join(DIR, 'ws-endpoint')

// Global setup for Puppeteer tests
module.exports = async function globalSetup() {
  console.log('🚀 Starting Puppeteer E2E test environment...')
  
  try {
    // Launch browser
    const browser = await puppeteer.launch({
      headless: process.env.HEADLESS !== 'false',
      slowMo: process.env.SLOW_MO ? parseInt(process.env.SLOW_MO) : 0,
      devtools: process.env.DEVTOOLS === 'true',
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-accelerated-2d-canvas',
        '--disable-gpu',
        '--window-size=1920,1080'
      ]
    })
    
    // Save WebSocket endpoint for tests
    fs.writeFileSync(WS_ENDPOINT_FILE, browser.wsEndpoint())
    console.log('✅ Browser launched successfully')
    
    // Start the Next.js development server if not already running
    const isServerRunning = await checkServerHealth()
    if (!isServerRunning) {
      console.log('🔄 Starting Next.js development server...')
      
      // Start server in background
      const serverProcess = require('child_process').spawn('npm', ['run', 'dev'], {
        cwd: DIR,
        stdio: 'pipe',
        detached: true
      })
      
      // Wait for server to be ready
      await waitForServer('http://localhost:8092', 30000)
      console.log('✅ Development server started')
      
      // Store server process ID for cleanup
      fs.writeFileSync(path.join(DIR, 'server-pid'), serverProcess.pid.toString())
    } else {
      console.log('✅ Development server already running')
    }
    
    // Start backend API if not running
    const isApiRunning = await checkApiHealth()
    if (!isApiRunning) {
      console.log('🔄 Starting backend API server...')
      
      // Start API server in background
      const apiProcess = require('child_process').spawn('go', ['run', './backend/cmd/api-server/main.go'], {
        cwd: path.join(DIR, '..'),
        stdio: 'pipe',
        detached: true
      })
      
      // Wait for API to be ready
      await waitForServer('http://localhost:8090', 30000)
      console.log('✅ Backend API server started')
      
      // Store API process ID for cleanup
      fs.writeFileSync(path.join(DIR, 'api-pid'), apiProcess.pid.toString())
    } else {
      console.log('✅ Backend API server already running')
    }
    
  } catch (error) {
    console.error('❌ Failed to setup test environment:', error)
    throw error
  }
}

// Check if server is running
async function checkServerHealth() {
  try {
    const response = await fetch('http://localhost:8092')
    return response.status === 200 || response.status === 404 // 404 is ok for Next.js
  } catch (error) {
    return false
  }
}

// Check if API is running
async function checkApiHealth() {
  try {
    const response = await fetch('http://localhost:8090/health')
    return response.status === 200
  } catch (error) {
    return false
  }
}

// Wait for server to be ready
async function waitForServer(url, timeout = 30000) {
  const start = Date.now()
  
  while (Date.now() - start < timeout) {
    try {
      const response = await fetch(url)
      if (response.status < 500) {
        return true
      }
    } catch (error) {
      // Server not ready yet, continue waiting
    }
    
    await new Promise(resolve => setTimeout(resolve, 1000))
  }
  
  throw new Error(`Server at ${url} did not start within ${timeout}ms`)
}