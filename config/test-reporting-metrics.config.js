/**
 * NovaCron Comprehensive Test Reporting and Metrics Configuration
 * Advanced reporting system for all 14 test types with detailed metrics collection
 */

module.exports = {
  // Global Reporting Configuration
  global: {
    enabled: true,
    outputDir: './test-results',
    timestampFormat: 'YYYY-MM-DD_HH-mm-ss',
    generateSummary: true,
    publishResults: true,
    retentionDays: 30
  },
  
  // Report Formats and Outputs
  formats: {
    // HTML Reports
    html: {
      enabled: true,
      outputPath: 'reports/html',
      templatePath: './config/templates/test-report.html',
      includeScreenshots: true,
      includeVideos: true,
      includeTraces: true,
      generateIndex: true,
      responsive: true
    },
    
    // JSON Reports
    json: {
      enabled: true,
      outputPath: 'reports/json',
      formatted: true,
      includeMetadata: true,
      includeTimestamps: true
    },
    
    // JUnit XML Reports
    junit: {
      enabled: true,
      outputPath: 'reports/junit',
      suiteName: 'NovaCron Test Suite',
      includeProperties: true,
      includeSystemOut: true
    },
    
    // LCOV Coverage Reports
    lcov: {
      enabled: true,
      outputPath: 'reports/coverage',
      includeHtml: true,
      includeBadges: true
    },
    
    // Allure Reports
    allure: {
      enabled: true,
      outputPath: 'reports/allure',
      resultsDir: 'allure-results',
      generateReport: true,
      openReport: false
    },
    
    // Custom NovaCron Report
    novacron: {
      enabled: true,
      outputPath: 'reports/novacron',
      template: 'enterprise',
      includeDashboard: true,
      includeTrends: true
    }
  },
  
  // Test Type Specific Reporting
  testTypeReporting: {
    // Unit Test Reports
    unit: {
      enabled: true,
      framework: 'jest',
      outputPath: 'reports/unit',
      formats: ['html', 'json', 'lcov'],
      metrics: {
        coverage: true,
        performance: true,
        assertions: true,
        failureAnalysis: true
      },
      thresholds: {
        passingRate: 100,
        coverage: 100,
        performance: '< 5s'
      }
    },
    
    // Integration Test Reports
    integration: {
      enabled: true,
      framework: 'jest',
      outputPath: 'reports/integration',
      formats: ['html', 'json', 'junit'],
      metrics: {
        endpointCoverage: true,
        databaseOperations: true,
        serviceIntegrations: true,
        responseTime: true
      },
      thresholds: {
        passingRate: 95,
        responseTime: '< 2s',
        errorRate: '< 1%'
      }
    },
    
    // End-to-End Test Reports
    e2e: {
      enabled: true,
      framework: 'playwright',
      outputPath: 'reports/e2e',
      formats: ['html', 'json', 'allure'],
      artifacts: {
        screenshots: true,
        videos: true,
        traces: true,
        har: true
      },
      metrics: {
        userJourneys: true,
        crossBrowser: true,
        visualRegression: true,
        performance: true
      },
      thresholds: {
        passingRate: 90,
        loadTime: '< 3s',
        visualDifference: '< 5%'
      }
    },
    
    // Performance Test Reports
    performance: {
      enabled: true,
      framework: 'playwright',
      outputPath: 'reports/performance',
      formats: ['html', 'json'],
      metrics: {
        loadTime: true,
        firstContentfulPaint: true,
        largestContentfulPaint: true,
        cumulativeLayoutShift: true,
        firstInputDelay: true,
        timeToInteractive: true,
        totalBlockingTime: true,
        speedIndex: true
      },
      thresholds: {
        loadTime: 3000,
        fcp: 1500,
        lcp: 2500,
        cls: 0.1,
        fid: 100,
        tti: 5000
      },
      trending: {
        enabled: true,
        periods: ['daily', 'weekly', 'monthly'],
        alertOnRegression: true
      }
    },
    
    // Security Test Reports
    security: {
      enabled: true,
      framework: 'custom',
      outputPath: 'reports/security',
      formats: ['html', 'json', 'sarif'],
      metrics: {
        vulnerabilities: true,
        securityHeaders: true,
        sslConfiguration: true,
        authenticationFlows: true,
        authorizationChecks: true,
        inputValidation: true
      },
      categorization: {
        critical: true,
        high: true,
        medium: true,
        low: true,
        info: true
      },
      compliance: {
        owasp: true,
        nist: true,
        iso27001: true,
        pci: true
      }
    },
    
    // Accessibility Test Reports
    accessibility: {
      enabled: true,
      framework: 'playwright',
      outputPath: 'reports/accessibility',
      formats: ['html', 'json'],
      standards: ['WCAG21AA', 'WCAG21AAA', 'Section508'],
      metrics: {
        axeViolations: true,
        colorContrast: true,
        keyboardNavigation: true,
        screenReaderCompatibility: true,
        ariaCompliance: true
      },
      reportDetails: {
        includeRemediation: true,
        prioritizeIssues: true,
        generateActionItems: true
      }
    },
    
    // Visual Regression Test Reports
    visualRegression: {
      enabled: true,
      framework: 'playwright',
      outputPath: 'reports/visual',
      formats: ['html', 'json'],
      artifacts: {
        screenshots: true,
        differences: true,
        comparisons: true
      },
      metrics: {
        pixelDifference: true,
        layoutChanges: true,
        colorDifferences: true,
        missingElements: true
      },
      thresholds: {
        pixelDifference: 0.05,
        overallDifference: 0.02
      }
    },
    
    // API Test Reports
    api: {
      enabled: true,
      framework: 'playwright',
      outputPath: 'reports/api',
      formats: ['html', 'json', 'openapi'],
      metrics: {
        endpointCoverage: true,
        responseTime: true,
        statusCodes: true,
        payloadValidation: true,
        schemaCompliance: true,
        contractTesting: true
      },
      documentation: {
        generateApiDocs: true,
        includeExamples: true,
        validateSchemas: true
      }
    },
    
    // Load Test Reports
    load: {
      enabled: true,
      framework: 'playwright',
      outputPath: 'reports/load',
      formats: ['html', 'json', 'grafana'],
      metrics: {
        responseTime: true,
        throughput: true,
        concurrency: true,
        errorRate: true,
        resourceUtilization: true
      },
      scenarios: {
        normalLoad: true,
        peakLoad: true,
        sustainedLoad: true,
        burstLoad: true
      },
      thresholds: {
        responseTime: 2000,
        throughput: 1000,
        errorRate: 0.01,
        cpuUsage: 80,
        memoryUsage: 85
      }
    },
    
    // Stress Test Reports
    stress: {
      enabled: true,
      framework: 'custom',
      outputPath: 'reports/stress',
      formats: ['html', 'json'],
      metrics: {
        breakingPoint: true,
        recoveryTime: true,
        resourceExhaustion: true,
        systemStability: true
      },
      categories: {
        cpu: true,
        memory: true,
        disk: true,
        network: true,
        database: true
      }
    },
    
    // Chaos Engineering Reports
    chaos: {
      enabled: true,
      framework: 'custom',
      outputPath: 'reports/chaos',
      formats: ['html', 'json'],
      experiments: {
        serviceFailure: true,
        networkPartition: true,
        latencyInjection: true,
        resourceExhaustion: true,
        dataCorruption: true
      },
      metrics: {
        resilienceScore: true,
        recoveryTime: true,
        impactAnalysis: true,
        failurePatterns: true
      }
    },
    
    // Penetration Test Reports
    penetration: {
      enabled: true,
      framework: 'custom',
      outputPath: 'reports/penetration',
      formats: ['html', 'json', 'sarif'],
      categories: {
        authentication: true,
        authorization: true,
        injection: true,
        cryptography: true,
        configuration: true
      },
      severity: {
        critical: true,
        high: true,
        medium: true,
        low: true
      },
      compliance: {
        owasp: true,
        sans: true,
        nist: true
      }
    },
    
    // Compatibility Test Reports
    compatibility: {
      enabled: true,
      framework: 'playwright',
      outputPath: 'reports/compatibility',
      formats: ['html', 'json'],
      dimensions: {
        browsers: ['chrome', 'firefox', 'safari', 'edge'],
        devices: ['desktop', 'tablet', 'mobile'],
        operatingSystems: ['windows', 'macos', 'linux', 'android', 'ios'],
        versions: true
      },
      matrix: {
        generate: true,
        includePassFail: true,
        highlightIssues: true
      }
    },
    
    // Usability Test Reports
    usability: {
      enabled: true,
      framework: 'playwright',
      outputPath: 'reports/usability',
      formats: ['html', 'json'],
      metrics: {
        taskCompletion: true,
        timeOnTask: true,
        errorRate: true,
        userSatisfaction: true,
        learnability: true
      },
      heuristics: {
        visibility: true,
        systemStatus: true,
        userControl: true,
        consistency: true,
        errorPrevention: true,
        recognition: true,
        flexibility: true,
        aesthetics: true,
        errorRecovery: true,
        help: true
      }
    }
  },
  
  // Advanced Metrics Collection
  metrics: {
    // Performance Metrics
    performance: {
      enabled: true,
      collectors: [
        'lighthouse',
        'web-vitals',
        'playwright-metrics',
        'resource-timing',
        'navigation-timing'
      ],
      intervals: {
        realTime: '1s',
        batch: '1m',
        aggregated: '5m'
      },
      storage: {
        influxdb: true,
        prometheus: true,
        elasticsearch: true
      }
    },
    
    // Quality Metrics
    quality: {
      enabled: true,
      metrics: [
        'test-coverage',
        'code-quality',
        'defect-density',
        'test-effectiveness',
        'maintenance-index'
      ],
      calculationMethods: {
        coverage: 'branch-and-line',
        quality: 'sonarqube',
        defects: 'per-kloc',
        effectiveness: 'defect-detection-rate'
      }
    },
    
    // Business Metrics
    business: {
      enabled: true,
      metrics: [
        'feature-adoption',
        'user-satisfaction',
        'conversion-rate',
        'error-impact',
        'availability'
      ],
      kpis: {
        availability: '99.9%',
        errorRate: '< 0.1%',
        userSatisfaction: '> 4.5/5',
        conversionRate: '> 85%'
      }
    },
    
    // Infrastructure Metrics
    infrastructure: {
      enabled: true,
      metrics: [
        'resource-utilization',
        'response-time',
        'throughput',
        'error-rate',
        'availability'
      ],
      monitoring: {
        realTime: true,
        alerting: true,
        dashboards: true
      }
    }
  },
  
  // Reporting Integrations
  integrations: {
    // CI/CD Integration
    cicd: {
      enabled: true,
      platforms: {
        github: {
          enabled: true,
          publishChecks: true,
          commentOnPR: true,
          createIssues: true
        },
        jenkins: {
          enabled: true,
          publishResults: true,
          archiveArtifacts: true,
          sendNotifications: true
        },
        azureDevOps: {
          enabled: true,
          publishTestResults: true,
          updateWorkItems: true
        }
      }
    },
    
    // Monitoring Integration
    monitoring: {
      enabled: true,
      platforms: {
        grafana: {
          enabled: true,
          dashboards: ['test-overview', 'performance', 'quality'],
          alerts: true
        },
        datadog: {
          enabled: true,
          metrics: true,
          logs: true,
          traces: true
        },
        newRelic: {
          enabled: true,
          insights: true,
          alerting: true
        }
      }
    },
    
    // Communication Integration
    communication: {
      enabled: true,
      channels: {
        slack: {
          enabled: true,
          webhook: process.env.SLACK_WEBHOOK,
          channels: ['#testing', '#alerts', '#releases']
        },
        teams: {
          enabled: true,
          webhook: process.env.TEAMS_WEBHOOK
        },
        email: {
          enabled: true,
          recipients: ['team@novacron.com'],
          templates: ['summary', 'failure-alert']
        }
      }
    }
  },
  
  // Report Customization
  customization: {
    // Themes
    themes: {
      default: 'novacron-enterprise',
      options: ['light', 'dark', 'high-contrast'],
      customCss: './config/styles/reports.css'
    },
    
    // Branding
    branding: {
      enabled: true,
      logo: './assets/novacron-logo.png',
      colors: {
        primary: '#2563eb',
        secondary: '#64748b',
        success: '#10b981',
        warning: '#f59e0b',
        error: '#ef4444'
      },
      footer: 'NovaCron Test Automation Suite'
    },
    
    // Custom Sections
    sections: {
      executive: true,
      technical: true,
      trends: true,
      recommendations: true,
      actionItems: true
    }
  },
  
  // Data Management
  dataManagement: {
    // Storage
    storage: {
      type: 'hybrid', // local, cloud, hybrid
      retention: {
        raw: '30d',
        aggregated: '1y',
        summaries: '5y'
      },
      compression: true,
      encryption: true
    },
    
    // Archival
    archival: {
      enabled: true,
      schedule: 'weekly',
      destination: 's3://novacron-test-archives',
      format: 'tar.gz'
    },
    
    // Analytics
    analytics: {
      enabled: true,
      engine: 'elasticsearch',
      indexing: 'daily',
      retention: '2y'
    }
  }
};