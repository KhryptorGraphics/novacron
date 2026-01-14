const fs = require('fs')
const path = require('path')
const puppeteer = require('puppeteer')

const DIR = path.join(__dirname, '..')
const WS_ENDPOINT_FILE = path.join(DIR, 'ws-endpoint')
const SERVER_PID_FILE = path.join(DIR, 'server-pid')
const API_PID_FILE = path.join(DIR, 'api-pid')

// Global teardown for Puppeteer tests
module.exports = async function globalTeardown() {
  console.log('🧹 Cleaning up Puppeteer E2E test environment...')
  
  try {
    // Close browser
    if (fs.existsSync(WS_ENDPOINT_FILE)) {
      const endpoint = fs.readFileSync(WS_ENDPOINT_FILE, 'utf8')
      
      try {
        const browser = await puppeteer.connect({ browserWSEndpoint: endpoint })
        await browser.close()
        console.log('✅ Browser closed successfully')
      } catch (error) {
        console.warn('⚠️ Could not close browser:', error.message)
      }
      
      fs.unlinkSync(WS_ENDPOINT_FILE)
    }
    
    // Stop development server if we started it
    if (fs.existsSync(SERVER_PID_FILE)) {
      const serverPid = fs.readFileSync(SERVER_PID_FILE, 'utf8')
      
      try {
        process.kill(parseInt(serverPid), 'SIGTERM')
        console.log('✅ Development server stopped')
      } catch (error) {
        console.warn('⚠️ Could not stop development server:', error.message)
      }
      
      fs.unlinkSync(SERVER_PID_FILE)
    }
    
    // Stop API server if we started it
    if (fs.existsSync(API_PID_FILE)) {
      const apiPid = fs.readFileSync(API_PID_FILE, 'utf8')
      
      try {
        process.kill(parseInt(apiPid), 'SIGTERM')
        console.log('✅ Backend API server stopped')
      } catch (error) {
        console.warn('⚠️ Could not stop API server:', error.message)
      }
      
      fs.unlinkSync(API_PID_FILE)
    }
    
    console.log('✅ Test environment cleanup completed')
    
  } catch (error) {
    console.error('❌ Failed to cleanup test environment:', error)
    // Don't throw error in teardown to avoid masking test failures
  }
}