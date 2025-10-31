/**
 * Integration tests for Auto-Spawning System
 */

const AutoSpawningOrchestrator = require('../../src/services/auto-spawning-orchestrator');

describe('Auto-Spawning Integration Tests', () => {
  let orchestrator;

  beforeEach(() => {
    orchestrator = new AutoSpawningOrchestrator({
      maxAgents: 8,
      enableMCP: false // Disable MCP for integration tests
    });
  });

  afterEach(async () => {
    await orchestrator.stop();
  });

  describe('System Lifecycle', () => {
    test('should start and stop orchestrator', async () => {
      expect(orchestrator.isRunning).toBe(false);
      
      await orchestrator.start();
      expect(orchestrator.isRunning).toBe(true);
      
      await orchestrator.stop();
      expect(orchestrator.isRunning).toBe(false);
    });

    test('should not start twice', async () => {
      await orchestrator.start();
      await orchestrator.start(); // Should not throw
      expect(orchestrator.isRunning).toBe(true);
    });
  });

  describe('Task Processing', () => {
    beforeEach(async () => {
      await orchestrator.start();
    });

    test('should process simple task', async () => {
      const result = await orchestrator.processTask({
        description: 'Fix typo in README',
        files: ['README.md']
      });

      expect(result).toHaveProperty('plan');
      expect(result).toHaveProperty('spawnedAgents');
      expect(result.plan.complexity.complexity).toBe('simple');
    });

    test('should process complex task with multiple files', async () => {
      const result = await orchestrator.processTask({
        description: 'Implement OAuth authentication',
        files: [
          'backend/auth/oauth.go',
          'frontend/components/Login.tsx',
          'database/migrations/001_add_oauth.sql'
        ]
      });

      expect(result.plan.complexity.complexity).toBe('very-complex');
      expect(result.plan.agents.length).toBeGreaterThan(3);
    });

    test('should select appropriate topology for task', async () => {
      const simpleResult = await orchestrator.processTask({
        description: 'Update comment',
        files: ['src/utils.js']
      });
      expect(simpleResult.plan.topology).toBe('single');

      const complexResult = await orchestrator.processTask({
        description: 'Implement distributed caching system',
        files: ['backend/cache/redis.go', 'backend/cache/memcached.go']
      });
      expect(['hierarchical', 'adaptive']).toContain(complexResult.plan.topology);
    });

    test('should handle Go backend files correctly', async () => {
      const result = await orchestrator.processTask({
        description: 'Add new VM management feature',
        files: ['backend/core/vm/manager.go']
      });

      expect(result.plan.agents).toContain('coder');
    });

    test('should handle React frontend files correctly', async () => {
      const result = await orchestrator.processTask({
        description: 'Create new dashboard component',
        files: ['frontend/components/Dashboard.tsx']
      });

      expect(result.plan.agents).toContain('coder');
    });
  });

  describe('Dynamic Scaling', () => {
    beforeEach(async () => {
      await orchestrator.start();
    });

    test('should trigger scale-up on high load', async () => {
      // Simulate high load
      orchestrator.monitor.updateMetrics({
        queueDepth: 10,
        activeAgents: 2
      });

      const decision = orchestrator.monitor.checkWorkload();
      expect(decision.action).toBe('scale-up');
    });

    test('should trigger scale-down on low load', async () => {
      // Simulate low load
      orchestrator.monitor.updateMetrics({
        queueDepth: 1,
        activeAgents: 8
      });

      const decision = orchestrator.monitor.checkWorkload();
      expect(decision.action).toBe('scale-down');
    });

    test('should maintain stable load', async () => {
      orchestrator.monitor.updateMetrics({
        queueDepth: 4,
        activeAgents: 4
      });

      const decision = orchestrator.monitor.checkWorkload();
      expect(decision.action).toBe('none');
    });
  });

  describe('Status and Metrics', () => {
    test('should provide comprehensive status', async () => {
      await orchestrator.start();
      
      const status = await orchestrator.getStatus();
      
      expect(status).toHaveProperty('running');
      expect(status).toHaveProperty('spawnerMetrics');
      expect(status).toHaveProperty('monitorStats');
      expect(status).toHaveProperty('config');
      expect(status.running).toBe(true);
    });

    test('should track spawning decisions', async () => {
      await orchestrator.start();
      
      await orchestrator.processTask({
        description: 'Task 1',
        files: ['file1.go']
      });
      
      await orchestrator.processTask({
        description: 'Task 2',
        files: ['file2.tsx']
      });

      const status = await orchestrator.getStatus();
      expect(status.spawnerMetrics.spawningDecisions.length).toBe(2);
    });
  });

  describe('Agent Capabilities', () => {
    test('should provide correct capabilities for each agent type', () => {
      const coderCaps = orchestrator.getAgentCapabilities('coder');
      expect(coderCaps).toContain('code-implementation');

      const architectCaps = orchestrator.getAgentCapabilities('architect');
      expect(architectCaps).toContain('system-design');

      const testerCaps = orchestrator.getAgentCapabilities('tester');
      expect(testerCaps).toContain('test-creation');
    });

    test('should provide default capabilities for unknown agent types', () => {
      const unknownCaps = orchestrator.getAgentCapabilities('unknown-agent');
      expect(unknownCaps).toContain('general-development');
    });
  });

  describe('Event Handling', () => {
    test('should handle spawn-plan events', async () => {
      await orchestrator.start();
      
      const eventPromise = new Promise(resolve => {
        orchestrator.spawner.once('spawn-plan', resolve);
      });

      await orchestrator.processTask({
        description: 'Test task',
        files: ['test.js']
      });

      const plan = await eventPromise;
      expect(plan).toBeDefined();
      expect(plan).toHaveProperty('agents');
    });

    test('should handle scaling-decision events', async () => {
      await orchestrator.start();
      
      const eventPromise = new Promise(resolve => {
        orchestrator.monitor.once('scaling-decision', resolve);
      });

      orchestrator.monitor.updateMetrics({
        queueDepth: 10,
        activeAgents: 2
      });
      
      orchestrator.monitor.checkWorkload();

      const decision = await eventPromise;
      expect(decision).toBeDefined();
      expect(decision).toHaveProperty('action');
    });
  });
});

