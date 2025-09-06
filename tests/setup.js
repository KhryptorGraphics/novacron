/**
 * Global Test Setup Configuration
 * Sets up test environment, mocks, and common utilities
 */

// Configure test environment
process.env.NODE_ENV = 'test';
process.env.NOVACRON_TEST = 'true';
process.env.NOVACRON_LOG_LEVEL = 'error';

// Extend Jest timeout for integration tests
jest.setTimeout(30000);

// Global test utilities
global.testUtils = {
  sleep: (ms) => new Promise(resolve => setTimeout(resolve, ms)),
  
  generateUUID: () => {
    return 'test-' + Math.random().toString(36).substr(2, 9);
  },
  
  mockDate: (date) => {
    const mockDate = new Date(date);
    const originalNow = Date.now;
    Date.now = jest.fn(() => mockDate.getTime());
    return () => { Date.now = originalNow; };
  },
  
  expectEventually: async (assertion, timeout = 5000, interval = 100) => {
    const startTime = Date.now();
    while (Date.now() - startTime < timeout) {
      try {
        await assertion();
        return;
      } catch (error) {
        if (Date.now() - startTime >= timeout) {
          throw error;
        }
        await global.testUtils.sleep(interval);
      }
    }
  }
};

// Setup global mocks
global.console = {
  ...console,
  // Suppress console.log in tests unless DEBUG is set
  log: process.env.DEBUG ? console.log : jest.fn(),
  warn: process.env.DEBUG ? console.warn : jest.fn(),
  error: console.error, // Keep error logging
};

// Mock fetch for tests
global.fetch = jest.fn();

// Setup beforeEach and afterEach hooks
beforeEach(() => {
  // Clear all mocks before each test
  jest.clearAllMocks();
  
  // Reset fetch mock
  if (global.fetch.mockClear) {
    global.fetch.mockClear();
  }
});

afterEach(() => {
  // Cleanup any timers
  jest.clearAllTimers();
});

// Handle unhandled promise rejections in tests
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  throw reason;
});