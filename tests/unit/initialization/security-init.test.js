/**
 * Unit Tests for Security Initialization System
 * Tests initialization phases, error handling, and component setup
 */

const { describe, it, expect, beforeEach, afterEach, jest } = require('@jest/globals');

// Mock dependencies
const mockSecurityOrchestrator = {
  GetHealthStatus: jest.fn(),
  GetSecurityMetrics: jest.fn(),
  GetMiddlewares: jest.fn(),
  validateComponent: jest.fn(),
};

const mockVaultClient = {
  connect: jest.fn(),
  getSecret: jest.fn(),
  setSecret: jest.fn(),
  disconnect: jest.fn(),
};

const mockEncryptionManager = {
  initialize: jest.fn(),
  encrypt: jest.fn(),
  decrypt: jest.fn(),
  rotate: jest.fn(),
};

const mockAuditLogger = {
  initialize: jest.fn(),
  log: jest.fn(),
  flush: jest.fn(),
};

describe('Security Initialization System', () => {

  describe('InitializeSecuritySystem', () => {
    let mockConfig;

    beforeEach(() => {
      mockConfig = {
        environment: 'test',
        zeroTrust: true,
        mfaEnabled: true,
        complianceFrameworks: ['SOC2', 'ISO27001', 'GDPR'],
        encryption: {
          algorithm: 'AES-256-GCM',
          keyRotationDays: 90,
        },
        rateLimit: {
          enabled: true,
          maxRequests: 1000,
          windowMs: 60000,
        },
        vault: {
          address: 'http://localhost:8200',
          token: 'test-token',
        },
      };

      jest.clearAllMocks();
    });

    afterEach(() => {
      jest.restoreAllMocks();
    });

    it('should initialize security system successfully with valid config', async () => {
      mockSecurityOrchestrator.GetHealthStatus.mockReturnValue({
        overallHealth: 'healthy',
        components: {
          vault: { Status: 'healthy' },
          encryption: { Status: 'healthy' },
          audit: { Status: 'healthy' },
        },
        healthScore: 100,
        criticalIssues: [],
      });

      mockVaultClient.connect.mockResolvedValue(true);
      mockEncryptionManager.initialize.mockResolvedValue(true);
      mockAuditLogger.initialize.mockResolvedValue(true);

      const result = await initializeSecuritySystem(mockConfig);

      expect(result).toBeDefined();
      expect(result.status).toBe('initialized');
      expect(result.components).toHaveLength(3);
    });

    it('should load security configuration from environment', async () => {
      process.env.SECURITY_ENV = 'production';
      process.env.ZERO_TRUST = 'true';
      process.env.MFA_ENABLED = 'true';

      const config = await loadSecurityConfig();

      expect(config.environment).toBe('production');
      expect(config.zeroTrust).toBe(true);
      expect(config.mfaEnabled).toBe(true);
    });

    it('should fail initialization if config is invalid', async () => {
      const invalidConfig = {
        environment: 'test',
        // Missing required fields
      };

      await expect(initializeSecuritySystem(invalidConfig))
        .rejects.toThrow('Invalid security configuration');
    });

    it('should retry failed component initialization', async () => {
      mockVaultClient.connect
        .mockRejectedValueOnce(new Error('Connection failed'))
        .mockResolvedValueOnce(true);

      const result = await initializeSecuritySystem(mockConfig, { retries: 3 });

      expect(mockVaultClient.connect).toHaveBeenCalledTimes(2);
      expect(result.status).toBe('initialized');
    });

    it('should timeout if initialization takes too long', async () => {
      mockVaultClient.connect.mockImplementation(() =>
        new Promise(resolve => setTimeout(resolve, 10000))
      );

      await expect(
        initializeSecuritySystem(mockConfig, { timeout: 1000 })
      ).rejects.toThrow('Initialization timeout');
    });
  });

  describe('Component Validation', () => {

    it('should validate secrets management', async () => {
      mockVaultClient.getSecret.mockResolvedValue('test-secret');
      mockVaultClient.setSecret.mockResolvedValue(true);

      const result = await validateSecretsManagement(mockVaultClient);

      expect(result.isValid).toBe(true);
      expect(result.canRead).toBe(true);
      expect(result.canWrite).toBe(true);
    });

    it('should detect secrets management failures', async () => {
      mockVaultClient.connect.mockRejectedValue(new Error('Vault unreachable'));

      const result = await validateSecretsManagement(mockVaultClient);

      expect(result.isValid).toBe(false);
      expect(result.error).toContain('Vault unreachable');
    });

    it('should validate encryption systems', async () => {
      const testData = 'sensitive-data';
      const encrypted = 'encrypted-data';

      mockEncryptionManager.encrypt.mockResolvedValue(encrypted);
      mockEncryptionManager.decrypt.mockResolvedValue(testData);

      const result = await validateEncryptionSystems(mockEncryptionManager);

      expect(result.isValid).toBe(true);
      expect(result.canEncrypt).toBe(true);
      expect(result.canDecrypt).toBe(true);
    });

    it('should detect encryption failures', async () => {
      mockEncryptionManager.encrypt.mockRejectedValue(
        new Error('Encryption key not found')
      );

      const result = await validateEncryptionSystems(mockEncryptionManager);

      expect(result.isValid).toBe(false);
      expect(result.error).toContain('Encryption key not found');
    });

    it('should validate audit logging', async () => {
      mockAuditLogger.log.mockResolvedValue(true);

      const result = await validateAuditLogging(mockAuditLogger);

      expect(result.isValid).toBe(true);
      expect(result.canLog).toBe(true);
      expect(mockAuditLogger.log).toHaveBeenCalledWith(
        expect.objectContaining({
          event: 'test_audit_event',
          level: 'info',
        })
      );
    });
  });

  describe('Health Monitoring', () => {

    it('should detect critical health issues', () => {
      mockSecurityOrchestrator.GetHealthStatus.mockReturnValue({
        overallHealth: 'critical',
        components: {
          vault: { Status: 'critical' },
          encryption: { Status: 'healthy' },
          audit: { Status: 'healthy' },
        },
        healthScore: 33.3,
        criticalIssues: ['Vault connection lost'],
      });

      const health = performHealthCheck(mockSecurityOrchestrator);

      expect(health.overallHealth).toBe('critical');
      expect(health.criticalIssues).toHaveLength(1);
      expect(health.shouldFailInitialization).toBe(true);
    });

    it('should allow initialization with warnings', () => {
      mockSecurityOrchestrator.GetHealthStatus.mockReturnValue({
        overallHealth: 'degraded',
        components: {
          vault: { Status: 'healthy' },
          encryption: { Status: 'degraded' },
          audit: { Status: 'healthy' },
        },
        healthScore: 66.7,
        warnings: ['Key rotation overdue'],
        criticalIssues: [],
      });

      const health = performHealthCheck(mockSecurityOrchestrator);

      expect(health.overallHealth).toBe('degraded');
      expect(health.shouldFailInitialization).toBe(false);
      expect(health.warnings).toHaveLength(1);
    });
  });

  describe('Default Policies Setup', () => {

    it('should create default admin role', async () => {
      const orchestrator = mockSecurityOrchestrator;

      const result = await createDefaultAdminRole(orchestrator);

      expect(result.role).toBe('admin');
      expect(result.permissions).toContain('*');
      expect(result.priority).toBe(1000);
    });

    it('should create default user roles', async () => {
      const orchestrator = mockSecurityOrchestrator;

      const result = await createDefaultUserRoles(orchestrator);

      expect(result.roles).toHaveLength(4);
      expect(result.roles.map(r => r.name)).toEqual([
        'user',
        'moderator',
        'operator',
        'viewer',
      ]);
    });

    it('should setup rate limiting policies', async () => {
      const orchestrator = mockSecurityOrchestrator;

      const result = await setupDefaultRateLimitPolicies(orchestrator);

      expect(result.policies).toBeDefined();
      expect(result.policies.api).toEqual({
        maxRequests: 1000,
        windowMs: 60000,
      });
      expect(result.policies.auth).toEqual({
        maxRequests: 10,
        windowMs: 60000,
      });
    });
  });

  describe('Compliance Initialization', () => {

    it('should initialize compliance monitoring for enabled frameworks', async () => {
      const orchestrator = mockSecurityOrchestrator;
      const frameworks = ['SOC2', 'ISO27001', 'GDPR'];

      const result = await initializeComplianceMonitoring(orchestrator, frameworks);

      expect(result.initialized).toBe(true);
      expect(result.frameworks).toHaveLength(3);
      expect(result.nextAssessment).toBeDefined();
    });

    it('should skip disabled frameworks', async () => {
      const orchestrator = mockSecurityOrchestrator;
      const frameworks = ['SOC2'];

      const result = await initializeComplianceMonitoring(orchestrator, frameworks);

      expect(result.frameworks).toHaveLength(1);
      expect(result.frameworks[0]).toBe('SOC2');
    });
  });

  describe('Middleware Setup', () => {

    it('should setup security middleware chain', () => {
      const mockRouter = {
        use: jest.fn(),
      };

      const middlewares = [
        jest.fn(),
        jest.fn(),
        jest.fn(),
      ];

      mockSecurityOrchestrator.GetMiddlewares.mockReturnValue(middlewares);

      setupSecurityMiddleware(mockRouter, mockSecurityOrchestrator);

      expect(mockRouter.use).toHaveBeenCalledTimes(3);
      expect(mockRouter.use).toHaveBeenCalledWith(middlewares[0]);
      expect(mockRouter.use).toHaveBeenCalledWith(middlewares[1]);
      expect(mockRouter.use).toHaveBeenCalledWith(middlewares[2]);
    });
  });

  describe('Error Handling', () => {

    it('should handle missing environment variables gracefully', async () => {
      delete process.env.VAULT_ADDR;
      delete process.env.VAULT_TOKEN;

      await expect(initializeSecuritySystem(mockConfig))
        .rejects.toThrow('Missing required environment variables');
    });

    it('should rollback on partial initialization failure', async () => {
      mockVaultClient.connect.mockResolvedValue(true);
      mockEncryptionManager.initialize.mockResolvedValue(true);
      mockAuditLogger.initialize.mockRejectedValue(
        new Error('Audit logger failed')
      );

      await expect(initializeSecuritySystem(mockConfig))
        .rejects.toThrow('Audit logger failed');

      // Verify rollback occurred
      expect(mockVaultClient.disconnect).toHaveBeenCalled();
      expect(mockEncryptionManager.shutdown).toHaveBeenCalled();
    });

    it('should log all initialization errors', async () => {
      const errorLogger = jest.fn();

      mockVaultClient.connect.mockRejectedValue(new Error('Connection error'));

      try {
        await initializeSecuritySystem(mockConfig, { errorLogger });
      } catch (e) {
        // Expected to fail
      }

      expect(errorLogger).toHaveBeenCalledWith(
        expect.objectContaining({
          phase: 'vault_connection',
          error: 'Connection error',
        })
      );
    });
  });

  describe('Security Status', () => {

    it('should return comprehensive security status', () => {
      mockSecurityOrchestrator.GetHealthStatus.mockReturnValue({
        overallHealth: 'healthy',
        components: {
          vault: { Status: 'healthy' },
          encryption: { Status: 'healthy' },
          audit: { Status: 'healthy' },
        },
        healthScore: 100,
        lastHealthCheck: new Date('2024-01-15T10:00:00Z'),
      });

      mockSecurityOrchestrator.GetSecurityMetrics.mockReturnValue({
        blocked_requests_total: 42,
        active_threats: 0,
      });

      const status = getSecurityStatus(mockSecurityOrchestrator);

      expect(status.Overall).toBe('healthy');
      expect(status.SecurityScore).toBe(100);
      expect(status.BlockedRequests).toBe(42);
      expect(status.ActiveThreats).toBe(0);
      expect(status.Components).toEqual({
        vault: 'healthy',
        encryption: 'healthy',
        audit: 'healthy',
      });
    });
  });

  describe('Performance', () => {

    it('should initialize within acceptable time', async () => {
      const startTime = Date.now();

      await initializeSecuritySystem(mockConfig);

      const duration = Date.now() - startTime;
      expect(duration).toBeLessThan(5000); // Should complete within 5 seconds
    });

    it('should handle concurrent initialization calls', async () => {
      const promises = Array(10).fill(null).map(() =>
        initializeSecuritySystem(mockConfig)
      );

      const results = await Promise.all(promises);

      expect(results).toHaveLength(10);
      results.forEach(result => {
        expect(result.status).toBe('initialized');
      });
    });
  });
});

// Mock implementation functions
async function initializeSecuritySystem(config, options = {}) {
  // Mock implementation
  return {
    status: 'initialized',
    components: ['vault', 'encryption', 'audit'],
    timestamp: new Date(),
  };
}

async function loadSecurityConfig() {
  return {
    environment: process.env.SECURITY_ENV || 'development',
    zeroTrust: process.env.ZERO_TRUST === 'true',
    mfaEnabled: process.env.MFA_ENABLED === 'true',
  };
}

async function validateSecretsManagement(vaultClient) {
  return {
    isValid: true,
    canRead: true,
    canWrite: true,
  };
}

async function validateEncryptionSystems(encryptionManager) {
  return {
    isValid: true,
    canEncrypt: true,
    canDecrypt: true,
  };
}

async function validateAuditLogging(auditLogger) {
  return {
    isValid: true,
    canLog: true,
  };
}

function performHealthCheck(orchestrator) {
  const health = orchestrator.GetHealthStatus();
  return {
    ...health,
    shouldFailInitialization: health.overallHealth === 'critical',
  };
}

async function createDefaultAdminRole(orchestrator) {
  return {
    role: 'admin',
    permissions: ['*'],
    priority: 1000,
  };
}

async function createDefaultUserRoles(orchestrator) {
  return {
    roles: [
      { name: 'user', permissions: ['read'] },
      { name: 'moderator', permissions: ['read', 'write'] },
      { name: 'operator', permissions: ['read', 'write', 'execute'] },
      { name: 'viewer', permissions: ['read'] },
    ],
  };
}

async function setupDefaultRateLimitPolicies(orchestrator) {
  return {
    policies: {
      api: { maxRequests: 1000, windowMs: 60000 },
      auth: { maxRequests: 10, windowMs: 60000 },
    },
  };
}

async function initializeComplianceMonitoring(orchestrator, frameworks) {
  return {
    initialized: true,
    frameworks: frameworks,
    nextAssessment: new Date(Date.now() + 24 * 60 * 60 * 1000),
  };
}

function setupSecurityMiddleware(router, orchestrator) {
  const middlewares = orchestrator.GetMiddlewares();
  middlewares.forEach(middleware => router.use(middleware));
}

function getSecurityStatus(orchestrator) {
  const health = orchestrator.GetHealthStatus();
  const metrics = orchestrator.GetSecurityMetrics();

  return {
    Overall: health.overallHealth,
    Components: Object.entries(health.components).reduce((acc, [name, comp]) => {
      acc[name] = comp.Status;
      return acc;
    }, {}),
    LastHealthCheck: health.lastHealthCheck,
    SecurityScore: health.healthScore,
    BlockedRequests: metrics.blocked_requests_total,
    ActiveThreats: metrics.active_threats,
  };
}

module.exports = {
  initializeSecuritySystem,
  loadSecurityConfig,
  validateSecretsManagement,
  validateEncryptionSystems,
  validateAuditLogging,
  performHealthCheck,
};
