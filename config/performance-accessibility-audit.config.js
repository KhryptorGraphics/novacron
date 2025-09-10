/**
 * NovaCron Performance and Accessibility Auditing Configuration
 * Comprehensive performance monitoring and accessibility testing integration
 */

module.exports = {
  // Performance Auditing Configuration
  performance: {
    // Core Performance Settings
    enabled: true,
    framework: 'lighthouse',
    outputDir: './test-results/performance',
    
    // Lighthouse Configuration
    lighthouse: {
      enabled: true,
      configPath: './config/lighthouse.config.js',
      outputFormats: ['html', 'json', 'csv'],
      categories: {
        performance: { weight: 30 },
        accessibility: { weight: 25 },
        bestPractices: { weight: 20 },
        seo: { weight: 15 },
        pwa: { weight: 10 }
      },
      
      // Lighthouse Settings
      settings: {
        formFactor: 'desktop',
        throttling: {
          rttMs: 40,
          throughputKbps: 10240,
          requestLatencyMs: 0,
          downloadThroughputKbps: 0,
          uploadThroughputKbps: 0,
          cpuSlowdownMultiplier: 1
        },
        screenEmulation: {
          mobile: false,
          width: 1920,
          height: 1080,
          deviceScaleFactor: 1,
          disabled: false
        },
        emulatedUserAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        maxWaitForLoad: 45000,
        pauseAfterLoadMs: 1000,
        networkQuietThresholdMs: 1000,
        cpuQuietThresholdMs: 1000,
        skipAudits: null,
        onlyAudits: null
      }
    },
    
    // Web Vitals Configuration
    webVitals: {
      enabled: true,
      metrics: {
        // Core Web Vitals
        largestContentfulPaint: {
          enabled: true,
          threshold: {
            good: 2500,
            needsImprovement: 4000,
            poor: 4001
          }
        },
        firstInputDelay: {
          enabled: true,
          threshold: {
            good: 100,
            needsImprovement: 300,
            poor: 301
          }
        },
        cumulativeLayoutShift: {
          enabled: true,
          threshold: {
            good: 0.1,
            needsImprovement: 0.25,
            poor: 0.26
          }
        },
        
        // Additional Performance Metrics
        firstContentfulPaint: {
          enabled: true,
          threshold: {
            good: 1800,
            needsImprovement: 3000,
            poor: 3001
          }
        },
        timeToInteractive: {
          enabled: true,
          threshold: {
            good: 3800,
            needsImprovement: 7300,
            poor: 7301
          }
        },
        totalBlockingTime: {
          enabled: true,
          threshold: {
            good: 200,
            needsImprovement: 600,
            poor: 601
          }
        },
        speedIndex: {
          enabled: true,
          threshold: {
            good: 3400,
            needsImprovement: 5800,
            poor: 5801
          }
        }
      },
      
      // Real User Monitoring (RUM)
      rum: {
        enabled: true,
        endpoint: '/api/v1/metrics/vitals',
        sampleRate: 0.1,
        reportAllChanges: false
      }
    },
    
    // Resource Performance
    resources: {
      enabled: true,
      thresholds: {
        totalSize: 2048, // 2MB
        totalRequests: 100,
        imageSize: 512, // 512KB per image
        jsSize: 1024,   // 1MB total JS
        cssSize: 256,   // 256KB total CSS
        fontSize: 128   // 128KB total fonts
      },
      
      // Resource Optimization
      optimization: {
        compression: true,
        caching: true,
        cdn: true,
        lazyLoading: true,
        criticalResources: true
      }
    },
    
    // Performance Budgets
    budgets: {
      enabled: true,
      categories: {
        // Resource budgets
        'total-size': { budget: 2048, tolerance: 10 },
        'script-size': { budget: 1024, tolerance: 5 },
        'style-size': { budget: 256, tolerance: 5 },
        'image-size': { budget: 512, tolerance: 10 },
        'media-size': { budget: 256, tolerance: 10 },
        'font-size': { budget: 128, tolerance: 5 },
        'third-party-size': { budget: 512, tolerance: 10 },
        
        // Timing budgets
        'first-contentful-paint': { budget: 1800, tolerance: 200 },
        'largest-contentful-paint': { budget: 2500, tolerance: 300 },
        'time-to-interactive': { budget: 3800, tolerance: 400 },
        'cumulative-layout-shift': { budget: 0.1, tolerance: 0.02 },
        'total-blocking-time': { budget: 200, tolerance: 50 },
        
        // Count budgets
        'dom-size': { budget: 1500, tolerance: 100 },
        'requests': { budget: 100, tolerance: 10 }
      },
      
      // Budget enforcement
      enforcement: {
        failOnExceed: true,
        reportViolations: true,
        alertThreshold: 90 // Alert at 90% of budget
      }
    },
    
    // Performance Testing Scenarios
    scenarios: {
      // Desktop scenarios
      desktop: {
        enabled: true,
        viewport: { width: 1920, height: 1080 },
        network: 'broadband',
        cpu: 'normal'
      },
      
      // Mobile scenarios
      mobile: {
        enabled: true,
        device: 'Moto G4',
        network: '3g',
        cpu: 'slow-4x'
      },
      
      // Tablet scenarios
      tablet: {
        enabled: true,
        device: 'iPad',
        network: 'wifi',
        cpu: 'normal'
      },
      
      // Slow network scenarios
      slowNetwork: {
        enabled: true,
        viewport: { width: 1920, height: 1080 },
        network: 'slow-3g',
        cpu: 'normal'
      }
    },
    
    // Continuous Performance Monitoring
    monitoring: {
      enabled: true,
      interval: '1h',
      endpoints: [
        '/',
        '/dashboard',
        '/vms',
        '/templates',
        '/networks',
        '/users'
      ],
      
      // Alerting
      alerts: {
        enabled: true,
        thresholds: {
          performanceScore: 90,
          lcp: 2500,
          fid: 100,
          cls: 0.1
        },
        channels: ['slack', 'email']
      }
    }
  },
  
  // Accessibility Auditing Configuration
  accessibility: {
    // Core Accessibility Settings
    enabled: true,
    standard: 'WCAG21AA',
    outputDir: './test-results/accessibility',
    
    // Axe-Core Configuration
    axeCore: {
      enabled: true,
      version: 'latest',
      configPath: './config/axe.config.js',
      
      // Axe Rules Configuration
      rules: {
        // Level A Rules
        'color-contrast': { enabled: true, level: 'AA' },
        'keyboard-navigation': { enabled: true, level: 'A' },
        'focus-management': { enabled: true, level: 'A' },
        'alt-text': { enabled: true, level: 'A' },
        'form-labels': { enabled: true, level: 'A' },
        'heading-structure': { enabled: true, level: 'A' },
        
        // Level AA Rules
        'color-contrast-enhanced': { enabled: true, level: 'AAA' },
        'resize-text': { enabled: true, level: 'AA' },
        'focus-visible': { enabled: true, level: 'AA' },
        'reflow': { enabled: true, level: 'AA' },
        
        // ARIA Rules
        'aria-valid': { enabled: true, level: 'A' },
        'aria-required': { enabled: true, level: 'A' },
        'aria-roles': { enabled: true, level: 'A' },
        'aria-properties': { enabled: true, level: 'A' }
      },
      
      // Tags to include/exclude
      tags: {
        include: ['wcag2a', 'wcag2aa', 'wcag21aa', 'best-practice'],
        exclude: ['experimental']
      },
      
      // Custom Axe Configuration
      options: {
        runOnly: {
          type: 'tag',
          values: ['wcag2a', 'wcag2aa', 'wcag21aa']
        },
        reporter: 'v2',
        resultTypes: ['violations', 'incomplete', 'inapplicable', 'passes'],
        preload: true,
        performanceTimer: true
      }
    },
    
    // Manual Accessibility Testing
    manual: {
      enabled: true,
      
      // Keyboard Navigation Testing
      keyboard: {
        enabled: true,
        tests: [
          'tab-navigation',
          'shift-tab-navigation',
          'enter-activation',
          'space-activation',
          'arrow-navigation',
          'escape-dismissal'
        ]
      },
      
      // Screen Reader Testing
      screenReader: {
        enabled: true,
        simulators: ['nvda', 'jaws', 'voiceover'],
        tests: [
          'content-reading',
          'navigation-landmarks',
          'form-interaction',
          'table-reading',
          'image-descriptions'
        ]
      },
      
      // Color and Contrast Testing
      colorContrast: {
        enabled: true,
        standards: ['AA', 'AAA'],
        tests: [
          'text-background-contrast',
          'interactive-element-contrast',
          'focus-indicator-contrast',
          'colorblind-simulation'
        ]
      }
    },
    
    // Accessibility Testing Scenarios
    scenarios: {
      // Standard desktop testing
      desktop: {
        enabled: true,
        viewport: { width: 1920, height: 1080 },
        zoom: [100, 150, 200], // Test at different zoom levels
        colorSchemes: ['light', 'dark', 'high-contrast']
      },
      
      // Mobile accessibility testing
      mobile: {
        enabled: true,
        devices: ['iPhone', 'Android'],
        orientations: ['portrait', 'landscape'],
        voiceOver: true
      },
      
      // Assistive technology testing
      assistiveTech: {
        enabled: true,
        technologies: [
          'screen-reader',
          'voice-control',
          'switch-navigation',
          'eye-tracking'
        ]
      }
    },
    
    // Accessibility Compliance
    compliance: {
      // WCAG 2.1 Compliance
      wcag21: {
        enabled: true,
        level: 'AA',
        successCriteria: {
          // Level A
          '1.1.1': 'non-text-content',
          '1.2.1': 'audio-only-and-video-only',
          '1.3.1': 'info-and-relationships',
          '1.3.2': 'meaningful-sequence',
          '1.3.3': 'sensory-characteristics',
          '1.4.1': 'use-of-color',
          '2.1.1': 'keyboard',
          '2.1.2': 'no-keyboard-trap',
          '2.2.1': 'timing-adjustable',
          '2.2.2': 'pause-stop-hide',
          '2.3.1': 'three-flashes-or-below',
          '2.4.1': 'bypass-blocks',
          '2.4.2': 'page-titled',
          '2.4.3': 'focus-order',
          '2.4.4': 'link-purpose',
          '3.1.1': 'language-of-page',
          '3.2.1': 'on-focus',
          '3.2.2': 'on-input',
          '3.3.1': 'error-identification',
          '3.3.2': 'labels-or-instructions',
          '4.1.1': 'parsing',
          '4.1.2': 'name-role-value',
          
          // Level AA
          '1.2.4': 'captions-live',
          '1.2.5': 'audio-description-prerecorded',
          '1.3.4': 'orientation',
          '1.3.5': 'identify-input-purpose',
          '1.4.3': 'contrast-minimum',
          '1.4.4': 'resize-text',
          '1.4.5': 'images-of-text',
          '1.4.10': 'reflow',
          '1.4.11': 'non-text-contrast',
          '1.4.12': 'text-spacing',
          '1.4.13': 'content-on-hover-or-focus',
          '2.1.4': 'character-key-shortcuts',
          '2.4.5': 'multiple-ways',
          '2.4.6': 'headings-and-labels',
          '2.4.7': 'focus-visible',
          '2.5.1': 'pointer-gestures',
          '2.5.2': 'pointer-cancellation',
          '2.5.3': 'label-in-name',
          '2.5.4': 'motion-actuation',
          '3.1.2': 'language-of-parts',
          '3.2.3': 'consistent-navigation',
          '3.2.4': 'consistent-identification',
          '3.3.3': 'error-suggestion',
          '3.3.4': 'error-prevention-legal',
          '4.1.3': 'status-messages'
        }
      },
      
      // Section 508 Compliance
      section508: {
        enabled: true,
        standards: [
          '1194.21', // Software applications
          '1194.22', // Web-based intranet
          '1194.31'  // Functional performance criteria
        ]
      },
      
      // EN 301 549 Compliance (European)
      en301549: {
        enabled: true,
        version: '3.2.1'
      }
    },
    
    // Accessibility Reporting
    reporting: {
      enabled: true,
      formats: ['html', 'json', 'sarif'],
      
      // Report content
      includeScreenshots: true,
      includeRemediation: true,
      includePriority: true,
      includeImpact: true,
      
      // Severity classification
      severity: {
        critical: 'Blocks users with disabilities',
        high: 'Significantly impacts accessibility',
        medium: 'Moderate accessibility impact',
        low: 'Minor accessibility concern'
      },
      
      // Custom reporting
      customReports: {
        executiveSummary: true,
        technicalDetails: true,
        remediationGuide: true,
        complianceMatrix: true
      }
    },
    
    // Continuous Accessibility Monitoring
    monitoring: {
      enabled: true,
      interval: '6h',
      pages: [
        '/',
        '/dashboard',
        '/vms/create',
        '/users/profile',
        '/settings'
      ],
      
      // Regression detection
      regression: {
        enabled: true,
        baseline: './test-results/accessibility/baseline',
        threshold: 0, // No new violations allowed
        alertOnRegression: true
      }
    }
  },
  
  // Combined Performance + Accessibility Testing
  combined: {
    enabled: true,
    
    // Lighthouse + Axe Integration
    lighthouseAxe: {
      enabled: true,
      runSimultaneously: true,
      correlateResults: true
    },
    
    // Performance impact of accessibility features
    accessibilityPerformance: {
      enabled: true,
      metrics: [
        'aria-processing-time',
        'screen-reader-performance',
        'keyboard-navigation-delay',
        'focus-management-overhead'
      ]
    },
    
    // Quality gates combining both
    qualityGates: {
      performance: {
        score: 90,
        lcp: 2500,
        fid: 100,
        cls: 0.1
      },
      accessibility: {
        violations: 0,
        wcagLevel: 'AA',
        contrastRatio: 4.5
      }
    }
  },
  
  // Integration with Test Frameworks
  integration: {
    // Playwright integration
    playwright: {
      enabled: true,
      beforeEach: 'setup-performance-monitoring',
      afterEach: 'collect-metrics-and-accessibility',
      fixtures: ['performance', 'accessibility']
    },
    
    // Jest integration
    jest: {
      enabled: true,
      setupFiles: ['./config/test-setup-performance.js'],
      teardownFiles: ['./config/test-teardown-metrics.js']
    },
    
    // Custom hooks
    hooks: {
      beforeTest: ['initialize-monitoring'],
      afterTest: ['collect-data', 'generate-reports'],
      beforeSuite: ['setup-baseline'],
      afterSuite: ['cleanup-resources']
    }
  }
};