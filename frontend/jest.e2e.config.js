const path = require('path')

// Custom Jest configuration for Puppeteer E2E tests
module.exports = {
  // Test environment for Puppeteer
  preset: 'jest-puppeteer',
  testEnvironment: 'jsdom',
  
  // Global setup and teardown
  globalSetup: '<rootDir>/test-setup/puppeteer-setup.js',
  globalTeardown: '<rootDir>/test-setup/puppeteer-teardown.js',
  
  // Setup files
  setupFilesAfterEnv: [
    '<rootDir>/jest.setup.js',
    '<rootDir>/test-setup/puppeteer-jest-setup.js'
  ],
  
  // Test patterns - only run E2E tests
  testMatch: [
    '<rootDir>/src/__tests__/e2e/**/*.test.{js,jsx,ts,tsx}',
    '<rootDir>/e2e/**/*.test.{js,jsx,ts,tsx}',
  ],
  
  // Module name mapping (corrected property name)
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
    '\\.(jpg|jpeg|png|gif|eot|otf|webp|svg|ttf|woff|woff2|mp4|webm|wav|mp3|m4a|aac|oga)$':
      '<rootDir>/__mocks__/fileMock.js',
  },
  
  // Coverage configuration
  collectCoverage: true,
  coverageDirectory: 'coverage/e2e',
  coverageReporters: ['text', 'lcov', 'html', 'json'],
  coveragePathIgnorePatterns: [
    '/node_modules/',
    '/.next/',
    '/coverage/',
    '/test-setup/',
    'jest.config.js',
    'jest.setup.js',
  ],
  
  // Files to include in coverage
  collectCoverageFrom: [
    'src/**/*.{js,jsx,ts,tsx}',
    '!src/**/*.d.ts',
    '!src/app/globals.css',
  ],
  
  // Coverage thresholds for E2E tests
  coverageThreshold: {
    global: {
      branches: 60,
      functions: 60,
      lines: 60,
      statements: 60,
    },
  },
  
  // Module file extensions
  moduleFileExtensions: ['js', 'jsx', 'json', 'ts', 'tsx'],
  
  // Transform configuration
  transform: {
    '^.+\\.(js|jsx|ts|tsx)$': ['babel-jest', { 
      presets: [
        ['@babel/preset-env', { targets: { node: 'current' } }],
        ['@babel/preset-react', { runtime: 'automatic' }],
        '@babel/preset-typescript'
      ] 
    }],
  },
  
  // Files to ignore during transformation
  transformIgnorePatterns: [
    '/node_modules/',
    '^.+\\.module\\.(css|sass|scss)$',
  ],
  
  // Test timeout for E2E tests (longer than unit tests)
  testTimeout: 30000,
  
  // Run tests serially for E2E stability
  maxWorkers: 1,
  
  // Clear mocks between tests
  clearMocks: true,
  
  // Restore mocks between tests
  restoreMocks: true,
  
  // Verbose output for debugging
  verbose: true,
  
  // Error handling
  errorOnDeprecated: true,
  
  // Simplified reporters configuration
  reporters: [
    'default',
    ['<rootDir>/node_modules/jest-html-reporter', {
      pageTitle: 'NovaCron E2E Test Report',
      outputPath: 'coverage/e2e/test-report.html',
      includeFailureMsg: true,
      includeSuiteFailure: true
    }]
  ]
}