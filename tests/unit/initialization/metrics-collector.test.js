/**
 * Unit Tests for MetricsCollector
 * Tests initialization metrics collection and reporting
 */

const { describe, it, expect, beforeEach } = require('@jest/globals');

describe('MetricsCollector Unit Tests', () => {
  let metrics;

  beforeEach(() => {
    metrics = createMetricsCollector();
  });

  describe('Constructor', () => {
    it('should initialize with empty collections', () => {
      expect(metrics.componentInitDurations).toEqual({});
      expect(metrics.componentInitSuccess).toEqual({});
    });

    it('should have all required methods', () => {
      expect(typeof metrics.recordComponentInit).toBe('function');
      expect(typeof metrics.recordComponentShutdown).toBe('function');
      expect(typeof metrics.setComponentStatus).toBe('function');
      expect(typeof metrics.getMetrics).toBe('function');
    });
  });

  describe('RecordComponentInit', () => {
    it('should record successful component initialization', () => {
      metrics.recordComponentInit('database', 1500, true);

      expect(metrics.componentInitDurations['database']).toBe(1500);
      expect(metrics.componentInitSuccess['database']).toBe(true);
    });

    it('should record failed component initialization', () => {
      metrics.recordComponentInit('cache', 500, false);

      expect(metrics.componentInitDurations['cache']).toBe(500);
      expect(metrics.componentInitSuccess['cache']).toBe(false);
    });

    it('should handle multiple component recordings', () => {
      metrics.recordComponentInit('database', 1500, true);
      metrics.recordComponentInit('cache', 300, true);
      metrics.recordComponentInit('api', 2000, true);

      expect(Object.keys(metrics.componentInitDurations)).toHaveLength(3);
      expect(Object.keys(metrics.componentInitSuccess)).toHaveLength(3);
    });

    it('should update existing component metrics', () => {
      metrics.recordComponentInit('database', 1000, false);
      metrics.recordComponentInit('database', 1500, true);

      expect(metrics.componentInitDurations['database']).toBe(1500);
      expect(metrics.componentInitSuccess['database']).toBe(true);
    });

    it('should handle zero duration', () => {
      metrics.recordComponentInit('logger', 0, true);

      expect(metrics.componentInitDurations['logger']).toBe(0);
      expect(metrics.componentInitSuccess['logger']).toBe(true);
    });

    it('should handle very large durations', () => {
      const largeDuration = 999999999;
      metrics.recordComponentInit('slow-component', largeDuration, true);

      expect(metrics.componentInitDurations['slow-component']).toBe(largeDuration);
    });

    it('should handle special characters in component names', () => {
      metrics.recordComponentInit('component-with-dash', 100, true);
      metrics.recordComponentInit('component_with_underscore', 100, true);
      metrics.recordComponentInit('component.with.dot', 100, true);

      expect(metrics.componentInitDurations['component-with-dash']).toBe(100);
      expect(metrics.componentInitDurations['component_with_underscore']).toBe(100);
      expect(metrics.componentInitDurations['component.with.dot']).toBe(100);
    });
  });

  describe('RecordComponentShutdown', () => {
    it('should record successful component shutdown', () => {
      metrics.recordComponentShutdown('database', 500, true);

      expect(metrics.componentShutdownDurations['database']).toBe(500);
      expect(metrics.componentShutdownSuccess['database']).toBe(true);
    });

    it('should record failed component shutdown', () => {
      metrics.recordComponentShutdown('api', 1000, false);

      expect(metrics.componentShutdownDurations['api']).toBe(1000);
      expect(metrics.componentShutdownSuccess['api']).toBe(false);
    });

    it('should track both init and shutdown for same component', () => {
      metrics.recordComponentInit('cache', 300, true);
      metrics.recordComponentShutdown('cache', 200, true);

      expect(metrics.componentInitDurations['cache']).toBe(300);
      expect(metrics.componentShutdownDurations['cache']).toBe(200);
    });
  });

  describe('SetComponentStatus', () => {
    it('should set component status', () => {
      metrics.setComponentStatus('database', 'initializing');

      expect(metrics.componentStatuses['database']).toBe('initializing');
    });

    it('should update component status', () => {
      metrics.setComponentStatus('api', 'starting');
      metrics.setComponentStatus('api', 'ready');

      expect(metrics.componentStatuses['api']).toBe('ready');
    });

    it('should handle multiple component statuses', () => {
      metrics.setComponentStatus('database', 'ready');
      metrics.setComponentStatus('cache', 'ready');
      metrics.setComponentStatus('api', 'starting');

      expect(metrics.componentStatuses['database']).toBe('ready');
      expect(metrics.componentStatuses['cache']).toBe('ready');
      expect(metrics.componentStatuses['api']).toBe('starting');
    });

    it('should handle error status', () => {
      metrics.setComponentStatus('database', 'error');

      expect(metrics.componentStatuses['database']).toBe('error');
    });
  });

  describe('GetMetrics', () => {
    it('should return all metrics', () => {
      metrics.recordComponentInit('database', 1500, true);
      metrics.recordComponentInit('cache', 300, true);

      const result = metrics.getMetrics();

      expect(result).toHaveProperty('init_durations');
      expect(result).toHaveProperty('init_success');
    });

    it('should return init durations', () => {
      metrics.recordComponentInit('database', 1500, true);
      metrics.recordComponentInit('cache', 300, true);

      const result = metrics.getMetrics();

      expect(result.init_durations).toEqual({
        database: 1500,
        cache: 300,
      });
    });

    it('should return init success flags', () => {
      metrics.recordComponentInit('database', 1500, true);
      metrics.recordComponentInit('cache', 300, false);

      const result = metrics.getMetrics();

      expect(result.init_success).toEqual({
        database: true,
        cache: false,
      });
    });

    it('should return empty object when no metrics recorded', () => {
      const result = metrics.getMetrics();

      expect(result.init_durations).toEqual({});
      expect(result.init_success).toEqual({});
    });

    it('should include shutdown metrics if recorded', () => {
      metrics.recordComponentInit('database', 1500, true);
      metrics.recordComponentShutdown('database', 500, true);

      const result = metrics.getMetrics();

      expect(result).toHaveProperty('shutdown_durations');
      expect(result.shutdown_durations['database']).toBe(500);
    });

    it('should not mutate internal state', () => {
      metrics.recordComponentInit('database', 1500, true);

      const result1 = metrics.getMetrics();
      result1.init_durations['database'] = 9999;

      const result2 = metrics.getMetrics();

      expect(result2.init_durations['database']).toBe(1500);
    });
  });

  describe('Statistical Analysis', () => {
    it('should calculate average init duration', () => {
      metrics.recordComponentInit('comp1', 1000, true);
      metrics.recordComponentInit('comp2', 2000, true);
      metrics.recordComponentInit('comp3', 3000, true);

      const avg = metrics.getAverageInitDuration();

      expect(avg).toBe(2000);
    });

    it('should calculate total init duration', () => {
      metrics.recordComponentInit('comp1', 1000, true);
      metrics.recordComponentInit('comp2', 2000, true);
      metrics.recordComponentInit('comp3', 3000, true);

      const total = metrics.getTotalInitDuration();

      expect(total).toBe(6000);
    });

    it('should calculate success rate', () => {
      metrics.recordComponentInit('comp1', 1000, true);
      metrics.recordComponentInit('comp2', 2000, true);
      metrics.recordComponentInit('comp3', 3000, false);

      const rate = metrics.getSuccessRate();

      expect(rate).toBeCloseTo(66.67, 1);
    });

    it('should find slowest component', () => {
      metrics.recordComponentInit('comp1', 1000, true);
      metrics.recordComponentInit('comp2', 5000, true);
      metrics.recordComponentInit('comp3', 2000, true);

      const slowest = metrics.getSlowestComponent();

      expect(slowest).toEqual({
        name: 'comp2',
        duration: 5000,
      });
    });

    it('should find fastest component', () => {
      metrics.recordComponentInit('comp1', 1000, true);
      metrics.recordComponentInit('comp2', 5000, true);
      metrics.recordComponentInit('comp3', 2000, true);

      const fastest = metrics.getFastestComponent();

      expect(fastest).toEqual({
        name: 'comp1',
        duration: 1000,
      });
    });

    it('should list failed components', () => {
      metrics.recordComponentInit('comp1', 1000, true);
      metrics.recordComponentInit('comp2', 2000, false);
      metrics.recordComponentInit('comp3', 3000, false);

      const failed = metrics.getFailedComponents();

      expect(failed).toEqual(['comp2', 'comp3']);
    });
  });

  describe('Export and Reporting', () => {
    it('should export metrics as JSON', () => {
      metrics.recordComponentInit('database', 1500, true);
      metrics.recordComponentInit('cache', 300, true);

      const json = metrics.toJSON();

      expect(typeof json).toBe('string');
      expect(JSON.parse(json)).toHaveProperty('init_durations');
    });

    it('should generate summary report', () => {
      metrics.recordComponentInit('comp1', 1000, true);
      metrics.recordComponentInit('comp2', 2000, false);

      const summary = metrics.getSummary();

      expect(summary).toHaveProperty('totalComponents');
      expect(summary).toHaveProperty('successfulComponents');
      expect(summary).toHaveProperty('failedComponents');
      expect(summary).toHaveProperty('totalDuration');
      expect(summary).toHaveProperty('averageDuration');
      expect(summary).toHaveProperty('successRate');
    });

    it('should format summary for human reading', () => {
      metrics.recordComponentInit('comp1', 1000, true);
      metrics.recordComponentInit('comp2', 2000, true);

      const formatted = metrics.formatSummary();

      expect(typeof formatted).toBe('string');
      expect(formatted).toContain('Total Components');
      expect(formatted).toContain('Success Rate');
    });
  });

  describe('Edge Cases', () => {
    it('should handle negative durations', () => {
      metrics.recordComponentInit('invalid', -100, true);

      expect(metrics.componentInitDurations['invalid']).toBe(0);
    });

    it('should handle null component name', () => {
      expect(() => {
        metrics.recordComponentInit(null, 100, true);
      }).toThrow();
    });

    it('should handle undefined values', () => {
      metrics.recordComponentInit('comp', undefined, undefined);

      expect(metrics.componentInitDurations['comp']).toBe(0);
      expect(metrics.componentInitSuccess['comp']).toBe(false);
    });

    it('should handle very long component names', () => {
      const longName = 'a'.repeat(1000);
      metrics.recordComponentInit(longName, 100, true);

      expect(metrics.componentInitDurations[longName]).toBe(100);
    });
  });
});

// Mock MetricsCollector implementation

function createMetricsCollector() {
  const collector = {
    componentInitDurations: {},
    componentInitSuccess: {},
    componentShutdownDurations: {},
    componentShutdownSuccess: {},
    componentStatuses: {},

    recordComponentInit(name, duration, success) {
      if (name === null || name === undefined) {
        throw new Error('Component name is required');
      }
      this.componentInitDurations[name] = Math.max(0, duration || 0);
      this.componentInitSuccess[name] = success !== undefined ? success : false;
    },

    recordComponentShutdown(name, duration, success) {
      this.componentShutdownDurations[name] = duration;
      this.componentShutdownSuccess[name] = success;
    },

    setComponentStatus(name, status) {
      this.componentStatuses[name] = status;
    },

    getMetrics() {
      const result = {
        init_durations: { ...this.componentInitDurations },
        init_success: { ...this.componentInitSuccess },
      };

      if (Object.keys(this.componentShutdownDurations).length > 0) {
        result.shutdown_durations = { ...this.componentShutdownDurations };
      }

      return result;
    },

    getAverageInitDuration() {
      const durations = Object.values(this.componentInitDurations);
      if (durations.length === 0) return 0;
      return durations.reduce((a, b) => a + b, 0) / durations.length;
    },

    getTotalInitDuration() {
      return Object.values(this.componentInitDurations).reduce((a, b) => a + b, 0);
    },

    getSuccessRate() {
      const total = Object.keys(this.componentInitSuccess).length;
      if (total === 0) return 0;
      const successful = Object.values(this.componentInitSuccess).filter(Boolean).length;
      return (successful / total) * 100;
    },

    getSlowestComponent() {
      const entries = Object.entries(this.componentInitDurations);
      if (entries.length === 0) return null;
      const [name, duration] = entries.reduce((a, b) => (b[1] > a[1] ? b : a));
      return { name, duration };
    },

    getFastestComponent() {
      const entries = Object.entries(this.componentInitDurations);
      if (entries.length === 0) return null;
      const [name, duration] = entries.reduce((a, b) => (b[1] < a[1] ? b : a));
      return { name, duration };
    },

    getFailedComponents() {
      return Object.entries(this.componentInitSuccess)
        .filter(([_, success]) => !success)
        .map(([name]) => name);
    },

    toJSON() {
      return JSON.stringify(this.getMetrics());
    },

    getSummary() {
      const total = Object.keys(this.componentInitSuccess).length;
      const successful = Object.values(this.componentInitSuccess).filter(Boolean).length;
      const failed = total - successful;

      return {
        totalComponents: total,
        successfulComponents: successful,
        failedComponents: failed,
        totalDuration: this.getTotalInitDuration(),
        averageDuration: this.getAverageInitDuration(),
        successRate: this.getSuccessRate(),
      };
    },

    formatSummary() {
      const summary = this.getSummary();
      return `
Initialization Summary:
  Total Components: ${summary.totalComponents}
  Successful: ${summary.successfulComponents}
  Failed: ${summary.failedComponents}
  Total Duration: ${summary.totalDuration}ms
  Average Duration: ${summary.averageDuration.toFixed(2)}ms
  Success Rate: ${summary.successRate.toFixed(2)}%
      `.trim();
    },
  };

  return collector;
}

module.exports = {
  createMetricsCollector,
};
