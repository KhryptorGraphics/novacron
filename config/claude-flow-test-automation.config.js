/**
 * NovaCron Claude-Flow Automation Test Configuration
 * Comprehensive test automation with 14 test types, multi-browser support, and enterprise features
 */
const { defineConfig } = require('claude-flow/automation');

module.exports = defineConfig({
  // Core Framework Configuration
  testFramework: 'playwright',
  
  // Comprehensive Test Types (14 categories)
  testTypes: {
    // Core Testing
    unit: {
      enabled: true,
      framework: 'jest',
      testMatch: ['**/__tests__/**/*.test.{js,ts,tsx}', '!**/e2e/**'],
      coverageThreshold: 90,
      parallel: true,
      workers: 4
    },
    
    integration: {
      enabled: true,
      framework: 'jest',
      testMatch: ['**/tests/integration/**/*.test.{js,ts,go}'],
      coverageThreshold: 85,
      parallel: true,
      workers: 3,
      timeout: 30000
    },
    
    e2e: {
      enabled: true,
      framework: 'playwright',
      testMatch: ['**/tests/e2e/**/*.test.{js,ts}', '**/e2e/**/*.test.{js,ts}'],
      coverageThreshold: 75,
      parallel: true,
      workers: 2,
      timeout: 60000,
      retries: 3
    },
    
    // Performance Testing
    performance: {
      enabled: true,
      framework: 'playwright',
      testMatch: ['**/tests/performance/**/*.test.{js,ts}'],
      metrics: {
        loadTime: 3000,        // 3s max load time
        firstContentfulPaint: 1500, // 1.5s FCP
        largestContentfulPaint: 2500, // 2.5s LCP
        cumulativeLayoutShift: 0.1,   // CLS < 0.1
        firstInputDelay: 100,         // FID < 100ms
        timeToInteractive: 5000       // TTI < 5s
      },
      parallel: false, // Performance tests run serially for accuracy
      workers: 1
    },
    
    // Security Testing
    security: {
      enabled: true,
      framework: 'custom',
      testMatch: ['**/tests/security/**/*.test.{js,ts}'],
      features: {
        securityHeaders: true,
        sslVerification: true,
        xssProtection: true,
        csrfProtection: true,
        authenticationFlows: true,
        authorizationChecks: true,
        dataValidation: true,
        sqlInjectionPrevention: true
      },
      parallel: true,
      workers: 2
    },
    
    // Accessibility Testing
    accessibility: {
      enabled: true,
      framework: 'playwright',
      testMatch: ['**/tests/accessibility/**/*.test.{js,ts}'],
      standards: 'WCAG21AA',
      features: {
        axeCore: true,
        colorContrast: true,
        keyboardNavigation: true,
        screenReaderCompatibility: true,
        ariaLabels: true,
        focusManagement: true
      },
      parallel: true,
      workers: 2
    },
    
    // Visual Regression Testing
    visualRegression: {
      enabled: true,
      framework: 'playwright',
      testMatch: ['**/tests/visual/**/*.test.{js,ts}'],
      features: {
        pixelComparison: true,
        layoutComparison: true,
        responsiveBreakpoints: true,
        crossBrowserConsistency: true,
        threshold: 0.05 // 5% pixel difference tolerance
      },
      parallel: true,
      workers: 3
    },
    
    // API Testing
    api: {
      enabled: true,
      framework: 'playwright',
      testMatch: ['**/tests/api/**/*.test.{js,ts}'],
      features: {
        contractTesting: true,
        schemaValidation: true,
        responseTimeValidation: true,
        statusCodeValidation: true,
        headerValidation: true,
        payloadValidation: true,
        authenticationTesting: true
      },
      parallel: true,
      workers: 4
    },
    
    // Load Testing
    load: {
      enabled: true,
      framework: 'playwright',
      testMatch: ['**/tests/load/**/*.test.{js,ts}'],
      scenarios: {
        normalLoad: { users: 100, duration: '5m' },
        peakLoad: { users: 500, duration: '10m' },
        sustainedLoad: { users: 200, duration: '30m' }
      },
      metrics: {
        responseTime: 2000,    // 2s max response
        throughput: 1000,      // req/s
        errorRate: 0.01        // <1% error rate
      },
      parallel: false,
      workers: 1
    },
    
    // Stress Testing
    stress: {
      enabled: true,
      framework: 'playwright',
      testMatch: ['**/tests/stress/**/*.test.{js,ts}'],
      scenarios: {
        cpuStress: { load: '80%', duration: '15m' },
        memoryStress: { load: '90%', duration: '10m' },
        diskStress: { load: '95%', duration: '5m' },
        networkStress: { bandwidth: '10Mbps', latency: '100ms' }
      },
      parallel: false,
      workers: 1
    },
    
    // Chaos Engineering
    chaos: {
      enabled: true,
      framework: 'custom',
      testMatch: ['**/tests/chaos/**/*.test.{js,ts}'],
      experiments: {
        serviceFailure: true,
        networkPartition: true,
        latencyInjection: true,
        resourceExhaustion: true,
        dataCorruption: true,
        timeSkew: true
      },
      parallel: false,
      workers: 1
    },
    
    // Penetration Testing
    penetration: {
      enabled: true,
      framework: 'custom',
      testMatch: ['**/tests/penetration/**/*.test.{js,ts}'],
      features: {
        vulnerabilityScanning: true,
        authenticationBypass: true,
        privilegeEscalation: true,
        dataExfiltration: true,
        injectionAttacks: true,
        businessLogicFlaws: true
      },
      parallel: false,
      workers: 1
    },
    
    // Compatibility Testing
    compatibility: {
      enabled: true,
      framework: 'playwright',
      testMatch: ['**/tests/compatibility/**/*.test.{js,ts}'],
      features: {
        crossBrowser: true,
        crossPlatform: true,
        backwardCompatibility: true,
        apiVersionCompatibility: true,
        dataFormatCompatibility: true
      },
      parallel: true,
      workers: 6 // One per browser
    },
    
    // Usability Testing
    usability: {
      enabled: true,
      framework: 'playwright',
      testMatch: ['**/tests/usability/**/*.test.{js,ts}'],
      features: {
        userJourneyTesting: true,
        taskCompletion: true,
        navigationTesting: true,
        formUsability: true,
        errorHandling: true,
        responsiveDesign: true
      },
      parallel: true,
      workers: 2
    }
  },
  
  // Multi-Browser Configuration (6 browsers)
  browsers: {
    chrome: {
      enabled: true,
      headless: true,
      viewport: { width: 1920, height: 1080 },
      deviceScaleFactor: 1,
      args: [
        '--no-sandbox',
        '--disable-dev-shm-usage',
        '--disable-web-security',
        '--allow-running-insecure-content'
      ]
    },
    
    firefox: {
      enabled: true,
      headless: true,
      viewport: { width: 1920, height: 1080 },
      deviceScaleFactor: 1
    },
    
    safari: {
      enabled: true,
      headless: false, // Safari doesn't support headless
      viewport: { width: 1920, height: 1080 },
      deviceScaleFactor: 1
    },
    
    edge: {
      enabled: true,
      headless: true,
      viewport: { width: 1920, height: 1080 },
      deviceScaleFactor: 1
    },
    
    mobileChrome: {
      enabled: true,
      headless: true,
      device: 'Pixel 5',
      viewport: { width: 393, height: 851 },
      deviceScaleFactor: 3,
      isMobile: true,
      hasTouch: true
    },
    
    mobileSafari: {
      enabled: true,
      headless: false,
      device: 'iPhone 13',
      viewport: { width: 390, height: 844 },
      deviceScaleFactor: 3,
      isMobile: true,
      hasTouch: true
    }
  },
  
  // Multi-Device Configuration (6 device categories)
  devices: {
    desktop: {
      enabled: true,
      viewport: { width: 1920, height: 1080 },
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    },
    
    tablet: {
      enabled: true,
      viewport: { width: 768, height: 1024 },
      deviceScaleFactor: 2,
      isMobile: true,
      hasTouch: true
    },
    
    mobile: {
      enabled: true,
      viewport: { width: 375, height: 667 },
      deviceScaleFactor: 2,
      isMobile: true,
      hasTouch: true
    },
    
    fourK: {
      enabled: true,
      viewport: { width: 3840, height: 2160 },
      deviceScaleFactor: 2
    },
    
    slow3G: {
      enabled: true,
      networkConditions: {
        offline: false,
        downloadThroughput: 500 * 1024 / 8, // 500kbps
        uploadThroughput: 500 * 1024 / 8,
        latency: 400
      }
    },
    
    offline: {
      enabled: true,
      networkConditions: {
        offline: true,
        downloadThroughput: 0,
        uploadThroughput: 0,
        latency: 0
      }
    }
  },
  
  // Execution Configuration
  execution: {
    coverageThreshold: 100, // Global 100% threshold
    parallelWorkers: 10,
    retryFailed: 3,
    failFast: false,
    timeout: 60000,
    
    // Advanced Features
    screenshots: true,
    videos: true,
    traces: true,
    harRecording: true,
    performanceMetrics: true,
    accessibilityAudit: true,
    securityHeadersCheck: true,
    sslVerification: true,
    apiContractTesting: true,
    mutationTesting: true,
    propertyBasedTesting: true,
    snapshotTesting: true,
    goldenTesting: true
  },
  
  // Reporting Configuration
  reporting: {
    formats: ['html', 'json', 'junit', 'lcov'],
    outputDir: './test-results',
    includeScreenshots: true,
    includeVideos: true,
    includeTraces: true,
    includeMetrics: true,
    
    // Quality Gates
    qualityGates: {
      coverageThreshold: 100,
      performanceThreshold: {
        loadTime: 3000,
        errorRate: 0.01
      },
      securityThreshold: {
        vulnerabilities: 0,
        securityScore: 95
      },
      accessibilityThreshold: {
        wcagLevel: 'AA',
        score: 95
      }
    }
  },
  
  // NovaCron-Specific Configuration
  novacron: {
    // VM Lifecycle Testing
    vmTesting: {
      enabled: true,
      scenarios: ['create', 'start', 'stop', 'migrate', 'snapshot', 'destroy'],
      hypervisors: ['kvm', 'xen', 'vmware'],
      storageTypes: ['local', 'nfs', 'ceph']
    },
    
    // Multi-Cloud Testing
    cloudProviders: {
      enabled: true,
      providers: ['aws', 'azure', 'gcp', 'openstack'],
      regions: ['us-east-1', 'eu-west-1', 'ap-southeast-1']
    },
    
    // ML Engineering Testing
    mlTesting: {
      enabled: true,
      frameworks: ['tensorflow', 'pytorch', 'scikit-learn'],
      models: ['classification', 'regression', 'clustering'],
      datasets: ['training', 'validation', 'test']
    },
    
    // API Endpoints
    apiEndpoints: [
      '/api/v1/vms',
      '/api/v1/templates',
      '/api/v1/networks',
      '/api/v1/storage',
      '/api/v1/users',
      '/api/v1/auth',
      '/api/v1/metrics',
      '/api/v1/alerts'
    ]
  }
});