/**
 * NovaCron 100% Coverage Threshold Validation Configuration
 * Comprehensive coverage validation across all test types and frameworks
 */

module.exports = {
  // Global Coverage Configuration
  global: {
    coverageThreshold: 100,
    enforceThreshold: true,
    failOnBelowThreshold: true,
    reportMissingCoverage: true
  },
  
  // Framework-Specific Coverage Thresholds
  frameworks: {
    jest: {
      coverageThreshold: {
        global: {
          branches: 100,
          functions: 100,
          lines: 100,
          statements: 100
        },
        // Component-specific thresholds
        'src/components/**/*.{js,jsx,ts,tsx}': {
          branches: 100,
          functions: 100,
          lines: 100,
          statements: 100
        },
        'src/hooks/**/*.{js,ts}': {
          branches: 100,
          functions: 100,
          lines: 100,
          statements: 100
        },
        'src/utils/**/*.{js,ts}': {
          branches: 100,
          functions: 100,
          lines: 100,
          statements: 100
        },
        'src/lib/**/*.{js,ts}': {
          branches: 100,
          functions: 100,
          lines: 100,
          statements: 100
        },
        'backend/**/*.{js,ts}': {
          branches: 100,
          functions: 100,
          lines: 100,
          statements: 100
        }
      },
      
      // Coverage Collection Configuration
      collectCoverageFrom: [
        'src/**/*.{js,jsx,ts,tsx}',
        'backend/**/*.{js,ts}',
        '!src/**/*.d.ts',
        '!src/**/*.stories.{js,jsx,ts,tsx}',
        '!src/**/*.test.{js,jsx,ts,tsx}',
        '!src/**/*.spec.{js,jsx,ts,tsx}',
        '!src/app/globals.css',
        '!src/components/ui/**', // Shadcn components
        '!backend/**/*.test.{js,ts}',
        '!backend/**/*.spec.{js,ts}',
        '!**/node_modules/**',
        '!**/coverage/**'
      ],
      
      // Coverage Reporters
      coverageReporters: [
        'text',
        'lcov',
        'html',
        'json',
        'json-summary',
        'cobertura',
        'text-summary'
      ],
      
      // Coverage Directory
      coverageDirectory: 'coverage/jest'
    },
    
    playwright: {
      // Playwright uses V8 coverage
      use: {
        trace: 'on-first-retry',
        screenshot: 'only-on-failure',
        video: 'retain-on-failure'
      },
      
      // Coverage collection setup
      coverage: {
        // Enable V8 coverage collection
        collectCoverage: true,
        coverageDirectory: 'coverage/playwright',
        coverageReporters: ['html', 'json', 'lcov'],
        
        // Include patterns
        include: [
          'src/**/*.{js,jsx,ts,tsx}',
          'backend/**/*.{js,ts}'
        ],
        
        // Exclude patterns
        exclude: [
          'node_modules/**',
          'coverage/**',
          'test-results/**',
          '**/*.test.{js,jsx,ts,tsx}',
          '**/*.spec.{js,jsx,ts,tsx}',
          '**/*.d.ts'
        ],
        
        // Thresholds
        thresholds: {
          branches: 100,
          functions: 100,
          lines: 100,
          statements: 100
        }
      }
    },
    
    go: {
      // Go test coverage configuration
      coverageThreshold: 100,
      coverageMode: 'atomic', // For accurate coverage in parallel tests
      coverageOutput: 'coverage/go/coverage.out',
      coverageHtml: 'coverage/go/coverage.html',
      
      // Coverage commands
      commands: {
        test: 'go test -race -coverprofile=coverage/go/coverage.out -covermode=atomic ./...',
        coverage: 'go tool cover -html=coverage/go/coverage.out -o coverage/go/coverage.html',
        func: 'go tool cover -func=coverage/go/coverage.out'
      },
      
      // Coverage validation
      validation: {
        minCoverage: 100,
        failOnBelowThreshold: true,
        excludePatterns: [
          '*.pb.go',         // Protocol buffer generated files
          '*_test.go',       // Test files
          'vendor/**',       // Vendor dependencies
          'mocks/**',        // Mock files
          'testutils/**'     // Test utilities
        ]
      }
    }
  },
  
  // Test Type Specific Coverage Requirements
  testTypes: {
    unit: {
      coverageThreshold: 100,
      requiredMetrics: ['branches', 'functions', 'lines', 'statements'],
      reportingLevel: 'detailed'
    },
    
    integration: {
      coverageThreshold: 95,
      requiredMetrics: ['branches', 'functions', 'lines'],
      reportingLevel: 'summary',
      focusAreas: [
        'API endpoints',
        'Database operations',
        'Service integrations',
        'Authentication flows'
      ]
    },
    
    e2e: {
      coverageThreshold: 85,
      requiredMetrics: ['lines', 'functions'],
      reportingLevel: 'summary',
      focusAreas: [
        'User workflows',
        'Critical paths',
        'Error scenarios',
        'Cross-browser compatibility'
      ]
    },
    
    performance: {
      coverageThreshold: 75,
      requiredMetrics: ['functions'],
      reportingLevel: 'basic',
      focusAreas: [
        'Performance bottlenecks',
        'Resource usage',
        'Load handling',
        'Optimization paths'
      ]
    },
    
    security: {
      coverageThreshold: 100,
      requiredMetrics: ['branches', 'functions', 'lines', 'statements'],
      reportingLevel: 'detailed',
      focusAreas: [
        'Authentication mechanisms',
        'Authorization checks',
        'Input validation',
        'Security headers',
        'Encryption/decryption',
        'Session management'
      ]
    },
    
    accessibility: {
      coverageThreshold: 95,
      requiredMetrics: ['functions', 'lines'],
      reportingLevel: 'detailed',
      focusAreas: [
        'ARIA attributes',
        'Keyboard navigation',
        'Screen reader compatibility',
        'Color contrast',
        'Focus management'
      ]
    },
    
    api: {
      coverageThreshold: 100,
      requiredMetrics: ['branches', 'functions', 'lines', 'statements'],
      reportingLevel: 'detailed',
      focusAreas: [
        'Endpoint handlers',
        'Request validation',
        'Response formatting',
        'Error handling',
        'Authentication middleware'
      ]
    }
  },
  
  // Coverage Validation Rules
  validation: {
    // Pre-test validation
    preTest: {
      enabled: true,
      checks: [
        'verify-coverage-tools',
        'validate-configuration',
        'check-baseline-coverage'
      ]
    },
    
    // Post-test validation
    postTest: {
      enabled: true,
      checks: [
        'validate-coverage-thresholds',
        'generate-coverage-reports',
        'compare-with-baseline',
        'update-coverage-badges'
      ]
    },
    
    // Continuous validation
    continuous: {
      enabled: true,
      interval: '5m',
      checks: [
        'monitor-coverage-trends',
        'alert-on-coverage-drops',
        'track-coverage-debt'
      ]
    }
  },
  
  // Coverage Reporting Configuration
  reporting: {
    // Output formats
    formats: [
      'html',      // Interactive HTML reports
      'lcov',      // LCOV format for CI/CD integration
      'json',      // JSON format for programmatic access
      'xml',       // XML format for build systems
      'text',      // Console output
      'cobertura'  // Cobertura format for Jenkins/Azure DevOps
    ],
    
    // Output directories
    outputDirs: {
      html: 'coverage/html',
      lcov: 'coverage/lcov',
      json: 'coverage/json',
      xml: 'coverage/xml',
      cobertura: 'coverage/cobertura'
    },
    
    // Report configuration
    htmlReport: {
      subdir: 'html',
      skipCovered: false,
      skipEmpty: false,
      showUncoveredLines: true,
      showBranchCoverage: true,
      showFunctionCoverage: true
    },
    
    // Badge generation
    badges: {
      enabled: true,
      outputDir: 'coverage/badges',
      formats: ['svg', 'json'],
      thresholds: {
        excellent: 100,
        good: 95,
        acceptable: 90,
        poor: 85
      }
    }
  },
  
  // Coverage Quality Gates
  qualityGates: {
    // Gate definitions
    gates: [
      {
        name: 'minimum-coverage',
        type: 'coverage',
        threshold: 100,
        metric: 'lines',
        failBuild: true
      },
      {
        name: 'branch-coverage',
        type: 'coverage',
        threshold: 100,
        metric: 'branches',
        failBuild: true
      },
      {
        name: 'function-coverage',
        type: 'coverage',
        threshold: 100,
        metric: 'functions',
        failBuild: true
      },
      {
        name: 'statement-coverage',
        type: 'coverage',
        threshold: 100,
        metric: 'statements',
        failBuild: true
      },
      {
        name: 'no-coverage-regression',
        type: 'trend',
        allowedDecrease: 0,
        failBuild: true
      }
    ],
    
    // Gate execution
    execution: {
      runOn: ['commit', 'pull-request', 'merge', 'deployment'],
      parallel: true,
      timeout: '5m'
    }
  },
  
  // Coverage Enforcement
  enforcement: {
    // Build integration
    build: {
      failOnBelowThreshold: true,
      generateReports: true,
      publishResults: true
    },
    
    // CI/CD integration
    cicd: {
      enabled: true,
      platforms: ['github-actions', 'jenkins', 'azure-devops'],
      reportFormats: ['lcov', 'cobertura', 'json'],
      artifactPaths: ['coverage/**/*']
    },
    
    // Git hooks
    gitHooks: {
      preCommit: {
        enabled: true,
        runCoverageCheck: true,
        failOnBelowThreshold: true
      },
      prePush: {
        enabled: true,
        runFullCoverage: true,
        generateReport: true
      }
    }
  },
  
  // Advanced Coverage Features
  advanced: {
    // Differential coverage
    differential: {
      enabled: true,
      baselineBranch: 'main',
      requireFullCoverageForNewCode: true,
      allowedUncoveredLinesInNewCode: 0
    },
    
    // Mutation testing integration
    mutation: {
      enabled: true,
      mutationScore: 95,
      framework: 'stryker',
      configFile: 'stryker.conf.json'
    },
    
    // Property-based testing coverage
    propertyBased: {
      enabled: true,
      framework: 'fast-check',
      minTests: 1000,
      shrinkLimit: 100
    },
    
    // Snapshot testing coverage
    snapshot: {
      enabled: true,
      updateOnFailure: false,
      failOnMismatch: true,
      directory: 'tests/__snapshots__'
    }
  }
};