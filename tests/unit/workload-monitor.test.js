/**
 * Unit tests for Workload Monitor
 */

const WorkloadMonitor = require('../../src/services/workload-monitor');

describe('WorkloadMonitor', () => {
  let monitor;

  beforeEach(() => {
    monitor = new WorkloadMonitor({
      checkInterval: 100, // Fast interval for testing
      scaleUpThreshold: 0.75,
      scaleDownThreshold: 0.25,
      maxAgents: 8,
      minAgents: 1
    });
  });

  afterEach(() => {
    monitor.stop();
  });

  describe('Utilization Calculation', () => {
    test('should calculate 0% utilization with no queue and no agents', () => {
      monitor.updateMetrics({ queueDepth: 0, activeAgents: 0 });
      const utilization = monitor.calculateUtilization();
      expect(utilization).toBe(0);
    });

    test('should calculate 100% utilization with queue but no agents', () => {
      monitor.updateMetrics({ queueDepth: 5, activeAgents: 0 });
      const utilization = monitor.calculateUtilization();
      expect(utilization).toBe(1.0);
    });

    test('should calculate utilization based on queue-to-agent ratio', () => {
      monitor.updateMetrics({ queueDepth: 4, activeAgents: 2 });
      const utilization = monitor.calculateUtilization();
      expect(utilization).toBe(1.0); // 4/2 = 2, capped at 1.0
    });

    test('should calculate partial utilization', () => {
      monitor.updateMetrics({ queueDepth: 2, activeAgents: 4 });
      const utilization = monitor.calculateUtilization();
      expect(utilization).toBe(0.5); // 2/4 = 0.5
    });
  });

  describe('Scaling Decisions', () => {
    test('should recommend scale-up when utilization is high', () => {
      monitor.updateMetrics({ queueDepth: 8, activeAgents: 2 });
      const decision = monitor.makeScalingDecision(0.8);
      
      expect(decision.action).toBe('scale-up');
      expect(decision.targetAgents).toBeGreaterThan(decision.currentAgents);
      expect(decision.targetAgents).toBeLessThanOrEqual(8); // maxAgents
    });

    test('should recommend scale-down when utilization is low', () => {
      monitor.updateMetrics({ queueDepth: 1, activeAgents: 8 });
      const decision = monitor.makeScalingDecision(0.2);
      
      expect(decision.action).toBe('scale-down');
      expect(decision.targetAgents).toBeLessThan(decision.currentAgents);
      expect(decision.targetAgents).toBeGreaterThanOrEqual(1); // minAgents
    });

    test('should maintain when utilization is stable', () => {
      monitor.updateMetrics({ queueDepth: 4, activeAgents: 4 });
      const decision = monitor.makeScalingDecision(0.5);
      
      expect(decision.action).toBe('none');
      expect(decision.targetAgents).toBe(decision.currentAgents);
    });

    test('should not scale up beyond maxAgents', () => {
      monitor.updateMetrics({ queueDepth: 20, activeAgents: 8 });
      const decision = monitor.makeScalingDecision(0.9);
      
      expect(decision.targetAgents).toBeLessThanOrEqual(8);
    });

    test('should not scale down below minAgents', () => {
      monitor.updateMetrics({ queueDepth: 0, activeAgents: 1 });
      const decision = monitor.makeScalingDecision(0.1);
      
      expect(decision.targetAgents).toBeGreaterThanOrEqual(1);
    });
  });

  describe('Monitoring Loop', () => {
    test('should start and stop monitoring', () => {
      expect(monitor.monitoringInterval).toBeNull();
      
      monitor.start();
      expect(monitor.monitoringInterval).not.toBeNull();
      
      monitor.stop();
      expect(monitor.monitoringInterval).toBeNull();
    });

    test('should emit scaling-decision events', (done) => {
      monitor.updateMetrics({ queueDepth: 10, activeAgents: 2 });
      
      monitor.once('scaling-decision', (decision) => {
        expect(decision).toHaveProperty('action');
        expect(decision).toHaveProperty('reason');
        done();
      });
      
      monitor.start();
      monitor.checkWorkload();
    });

    test('should not start multiple monitoring loops', () => {
      monitor.start();
      const firstInterval = monitor.monitoringInterval;
      
      monitor.start();
      expect(monitor.monitoringInterval).toBe(firstInterval);
    });
  });

  describe('Scaling History', () => {
    test('should record scaling decisions', () => {
      monitor.updateMetrics({ queueDepth: 10, activeAgents: 2 });
      monitor.checkWorkload();
      
      expect(monitor.scalingHistory.length).toBeGreaterThan(0);
    });

    test('should limit history to 100 entries', () => {
      for (let i = 0; i < 150; i++) {
        monitor.recordScalingDecision({
          action: 'scale-up',
          timestamp: new Date().toISOString()
        });
      }
      
      expect(monitor.scalingHistory.length).toBe(100);
    });
  });

  describe('Statistics', () => {
    test('should provide comprehensive statistics', () => {
      monitor.updateMetrics({ queueDepth: 5, activeAgents: 3 });
      monitor.checkWorkload();
      
      const stats = monitor.getStatistics();
      
      expect(stats).toHaveProperty('currentMetrics');
      expect(stats).toHaveProperty('scalingHistory');
      expect(stats).toHaveProperty('averageUtilization');
      expect(stats.currentMetrics.queueDepth).toBe(5);
      expect(stats.currentMetrics.activeAgents).toBe(3);
    });

    test('should calculate average utilization', () => {
      monitor.updateMetrics({ queueDepth: 8, activeAgents: 2 });
      monitor.checkWorkload(); // High utilization
      
      monitor.updateMetrics({ queueDepth: 1, activeAgents: 4 });
      monitor.checkWorkload(); // Low utilization
      
      const stats = monitor.getStatistics();
      expect(stats.averageUtilization).toBeGreaterThan(0);
      expect(stats.averageUtilization).toBeLessThan(1);
    });

    test('should count scaling actions', () => {
      // Trigger scale-up
      monitor.updateMetrics({ queueDepth: 10, activeAgents: 2 });
      monitor.checkWorkload();
      
      // Trigger scale-down
      monitor.updateMetrics({ queueDepth: 1, activeAgents: 8 });
      monitor.checkWorkload();
      
      const stats = monitor.getStatistics();
      expect(stats.scalingHistory.scaleUp).toBeGreaterThan(0);
      expect(stats.scalingHistory.scaleDown).toBeGreaterThan(0);
    });
  });

  describe('Metrics Update', () => {
    test('should update metrics correctly', () => {
      monitor.updateMetrics({
        queueDepth: 5,
        activeAgents: 3,
        completedTasks: 10
      });
      
      expect(monitor.metrics.queueDepth).toBe(5);
      expect(monitor.metrics.activeAgents).toBe(3);
      expect(monitor.metrics.completedTasks).toBe(10);
    });

    test('should merge new metrics with existing', () => {
      monitor.updateMetrics({ queueDepth: 5 });
      monitor.updateMetrics({ activeAgents: 3 });
      
      expect(monitor.metrics.queueDepth).toBe(5);
      expect(monitor.metrics.activeAgents).toBe(3);
    });
  });
});

