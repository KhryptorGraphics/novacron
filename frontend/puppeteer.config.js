// Puppeteer Configuration for NovaCron E2E Tests
module.exports = {
  // Launch options for Puppeteer
  launch: {
    headless: process.env.HEADLESS !== 'false',
    slowMo: parseInt(process.env.SLOW_MO || '0'),
    devtools: process.env.DEVTOOLS === 'true',
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-accelerated-2d-canvas',
      '--disable-gpu',
      '--window-size=1920,1080',
      '--disable-extensions',
      '--disable-plugins',
      '--disable-images', // Speed up loading in test environment
      '--disable-javascript-harmony-shipping',
      '--disable-background-timer-throttling',
      '--disable-renderer-backgrounding',
      '--disable-backgrounding-occluded-windows',
      '--disable-ipc-flooding-protection',
      '--mute-audio'
    ]
  },
  
  // Browser context options
  browserContext: {
    viewport: { width: 1920, height: 1080 },
    userAgent: 'Mozilla/5.0 (compatible; NovaCronE2E/1.0; +https://github.com/novacron/e2e-tests)',
    ignoreHTTPSErrors: true
  },
  
  // Server configuration
  server: {
    command: 'npm run dev',
    port: 8092,
    launchTimeout: 60000,
    debug: process.env.DEBUG_SERVER === 'true'
  },
  
  // Test configuration
  testTimeout: 30000,
  expect: {
    timeout: 10000
  }
}