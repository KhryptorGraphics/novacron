const nextJest = require('next/jest')

// Create Jest config with Next.js defaults
const createJestConfig = nextJest({
  // Path to your Next.js app to load next.config.js and .env files
  dir: './',
})

// Custom Jest configuration
const customJestConfig = {
  // Test environment
  testEnvironment: 'jest-environment-jsdom',
  
  // Setup files
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  
  // Module name mapping for absolute imports. Static assets (css, images,
  // fonts) are already mocked by next/jest, so only the `@/` alias is needed.
  // NOTE: the Jest option is `moduleNameMapper`; the prior `moduleNameMapping`
  // typo silently disabled the alias, so every `@/`-importing test failed to
  // resolve its modules.
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
  },
  
  // Test patterns
  testMatch: [
    '<rootDir>/src/**/__tests__/**/*.{js,jsx,ts,tsx}',
    '<rootDir>/src/**/*.(test|spec).{js,jsx,ts,tsx}',
    '<rootDir>/tests/**/*.{js,jsx,ts,tsx}',
  ],
  
  // Coverage configuration
  collectCoverage: true,
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html'],
  coveragePathIgnorePatterns: [
    '/node_modules/',
    '/.next/',
    '/coverage/',
    '/cypress/',
    'jest.config.js',
    'jest.setup.js',
  ],
  
  // Files to include in coverage
  collectCoverageFrom: [
    'src/**/*.{js,jsx,ts,tsx}',
    '!src/**/*.d.ts',
    '!src/app/globals.css',
    '!src/components/ui/**', // Exclude shadcn/ui components from coverage
  ],
  
  // Coverage thresholds
  coverageThreshold: {
    global: {
      branches: 70,
      functions: 70,
      lines: 70,
      statements: 70,
    },
  },
  
  // Module file extensions
  moduleFileExtensions: ['js', 'jsx', 'json', 'ts', 'tsx'],
  
  // Transform configuration
  transform: {
    '^.+\\.(js|jsx|ts|tsx)$': ['babel-jest', { presets: ['next/babel'] }],
  },
  
  // Files to ignore during transformation
  transformIgnorePatterns: [
    '/node_modules/',
    '^.+\\.module\\.(css|sass|scss)$',
  ],
  
  // Duplicate moduleNameMapping removed
  
  // Global test setup
  globals: {
    'ts-jest': {
      tsconfig: 'tsconfig.json',
    },
  },
  
  // Verbose output for debugging
  verbose: true,
  
  // Test timeout
  testTimeout: 10000,
  
  // Clear mocks between tests
  clearMocks: true,
  
  // Restore mocks between tests
  restoreMocks: true,
  
  // Error handling
  errorOnDeprecated: true,
  
  // Watch plugins for development (commented out due to optional dependencies)
  // watchPlugins: [
  //   'jest-watch-typeahead/filename',
  //   'jest-watch-typeahead/testname',
  // ],
}

// Export Jest config with Next.js defaults merged
module.exports = createJestConfig(customJestConfig)