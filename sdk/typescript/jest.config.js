/**
 * Jest config for the DWCP TypeScript SDK.
 *
 * The fabric tests exercise request construction and response handling over
 * real HTTP against a local mock server bound to 127.0.0.1 (no live api-server
 * and no outbound network).
 */

/** @type {import('jest').Config} */
module.exports = {
  testEnvironment: 'node',
  roots: ['<rootDir>/tests'],
  testMatch: ['**/*.test.ts'],
  transform: {
    '^.+\\.ts$': ['ts-jest', { tsconfig: '<rootDir>/tsconfig.test.json' }],
  },
  clearMocks: true,
  verbose: true,
};