// Stryker Mutation Testing Configuration for Frontend
module.exports = {
  packageManager: 'npm',
  reporters: ['html', 'clear-text', 'progress', 'json', 'dashboard'],
  testRunner: 'jest',
  coverageAnalysis: 'perTest',
  mutate: [
    'frontend/src/**/*.ts',
    'frontend/src/**/*.tsx',
    '!frontend/src/**/*.test.ts',
    '!frontend/src/**/*.test.tsx',
    '!frontend/src/**/*.spec.ts',
    '!frontend/src/**/*.spec.tsx',
    '!frontend/src/**/*.d.ts',
    '!frontend/src/test-utils/**',
    '!frontend/src/__tests__/**',
    '!frontend/src/setupTests.ts',
  ],
  
  // Test files to run for mutation testing
  testFramework: 'jest',
  jest: {
    projectType: 'custom',
    configFile: 'frontend/jest.config.js',
    enableFindRelatedTests: true,
  },

  // Mutation testing thresholds
  thresholds: {
    high: 80,    // 80% mutation score for high quality
    low: 60,     // 60% minimum mutation score
    break: 50,   // Break build if below 50%
  },

  // Specific mutations to enable/disable
  mutator: {
    name: 'typescript',
    excludedMutations: [
      // Exclude some mutations that may not be meaningful for frontend
      'StringLiteral',  // Avoid changing string literals in UI
      'ArithmeticOperator', // Sometimes arithmetic changes don't affect functionality
    ]
  },

  // Performance settings
  maxConcurrentTestRunners: 4,
  timeoutMS: 60000,
  timeoutFactor: 1.5,
  
  // Incremental mutation testing
  incremental: true,
  incrementalFile: 'reports/mutation/stryker-incremental.json',

  // Dashboard reporter settings (if using Stryker Dashboard)
  dashboard: {
    project: 'github.com/novacron/novacron',
    version: 'main'
  },

  // HTML report configuration
  htmlReporter: {
    baseDir: 'reports/mutation/html'
  },

  // JSON report for CI integration
  jsonReporter: {
    fileName: 'reports/mutation/mutation-report.json'
  },

  // Plugins
  plugins: [
    '@stryker-mutator/core',
    '@stryker-mutator/jest-runner',
    '@stryker-mutator/typescript-checker',
    '@stryker-mutator/html-reporter',
    '@stryker-mutator/json-reporter',
    '@stryker-mutator/dashboard-reporter'
  ],

  // TypeScript checker
  checkers: ['typescript'],
  tsconfigFile: 'frontend/tsconfig.json',

  // Logging
  logLevel: 'info',
  fileLogLevel: 'trace',
  allowConsoleColors: true,

  // Clean temp directories after test run
  cleanTempDir: true,

  // Ignore specific patterns for mutation
  ignorePatterns: [
    // Ignore generated files
    '*.generated.ts',
    '*.generated.tsx',
    
    // Ignore configuration files
    '*.config.*',
    
    // Ignore type definitions
    'types/**',
    
    // Ignore mock files
    '**/__mocks__/**',
  ],

  // Custom file headers to ignore
  ignoreStatic: true,

  // Warning settings
  warnings: {
    unknownOptions: false,
    preprocessorErrors: true,
    unserializableOptions: true
  }
};