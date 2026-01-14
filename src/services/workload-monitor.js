/**
 * Workload Monitor for Dynamic Agent Scaling
 * Monitors task queue, resource usage, and triggers agent scaling
 */

const EventEmitter = require('events');

class WorkloadMonitor extends EventEmitter {
  constructor(config = {}) {
    super();
    this.config = {
      checkInterval: config.checkInterval || 5000, // 5 seconds
      scaleUpThreshold: config.scaleUpThreshold || 0.75,
      scaleDownThreshold: config.scaleDownThreshold || 0.25,
      maxAgents: config.maxAgents || 8,
      minAgents: config.minAgents || 1,
      ...config
    };
    
    this.metrics = {
      queueDepth: 0,
      activeAgents: 0,
      completedTasks: 0,
      averageTaskDuration: 0,
      resourceUtilization: 0
    };
    
    this.monitoringInterval = null;
    this.scalingHistory = [];
  }

  /**
   * Start monitoring workload
   */
  start() {
    if (this.monitoringInterval) {
      return;
    }

    console.log('🔍 Workload Monitor started');
    this.monitoringInterval = setInterval(() => {
      this.checkWorkload();
    }, this.config.checkInterval);
  }

  /**
   * Stop monitoring
   */
  stop() {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
      console.log('⏹️  Workload Monitor stopped');
    }
  }

  /**
   * Check current workload and make scaling decisions
   */
  checkWorkload() {
    const utilization = this.calculateUtilization();
    const decision = this.makeScalingDecision(utilization);
    
    if (decision.action !== 'none') {
      this.emit('scaling-decision', decision);
      this.recordScalingDecision(decision);
    }
    
    return decision;
  }

  /**
   * Calculate current resource utilization
   */
  calculateUtilization() {
    const { queueDepth, activeAgents } = this.metrics;
    
    // If no agents, utilization is based on queue
    if (activeAgents === 0) {
      return queueDepth > 0 ? 1.0 : 0.0;
    }
    
    // Calculate utilization as ratio of queue to agents
    const utilization = Math.min(queueDepth / activeAgents, 1.0);
    
    this.metrics.resourceUtilization = utilization;
    return utilization;
  }

  /**
   * Make scaling decision based on utilization
   */
  makeScalingDecision(utilization) {
    const { activeAgents, queueDepth } = this.metrics;
    const { maxAgents, minAgents, scaleUpThreshold, scaleDownThreshold } = this.config;
    
    let action = 'none';
    let reason = '';
    let targetAgents = activeAgents;
    
    // Scale up conditions
    if (utilization >= scaleUpThreshold && activeAgents < maxAgents) {
      action = 'scale-up';
      targetAgents = Math.min(activeAgents + Math.ceil(queueDepth / 2), maxAgents);
      reason = `High utilization (${(utilization * 100).toFixed(1)}%) with ${queueDepth} queued tasks`;
    }
    // Scale down conditions
    else if (utilization <= scaleDownThreshold && activeAgents > minAgents) {
      action = 'scale-down';
      targetAgents = Math.max(Math.ceil(activeAgents / 2), minAgents);
      reason = `Low utilization (${(utilization * 100).toFixed(1)}%) with ${queueDepth} queued tasks`;
    }
    // Maintain current scale
    else {
      reason = `Stable utilization (${(utilization * 100).toFixed(1)}%)`;
    }
    
    return {
      action,
      reason,
      currentAgents: activeAgents,
      targetAgents,
      utilization,
      queueDepth,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Update metrics from external sources
   */
  updateMetrics(metrics) {
    this.metrics = {
      ...this.metrics,
      ...metrics
    };
  }

  /**
   * Record scaling decision for analysis
   */
  recordScalingDecision(decision) {
    this.scalingHistory.push(decision);
    
    // Keep only last 100 decisions
    if (this.scalingHistory.length > 100) {
      this.scalingHistory.shift();
    }
  }

  /**
   * Get scaling statistics
   */
  getStatistics() {
    const scaleUpCount = this.scalingHistory.filter(d => d.action === 'scale-up').length;
    const scaleDownCount = this.scalingHistory.filter(d => d.action === 'scale-down').length;
    
    return {
      currentMetrics: this.metrics,
      scalingHistory: {
        total: this.scalingHistory.length,
        scaleUp: scaleUpCount,
        scaleDown: scaleDownCount
      },
      averageUtilization: this.scalingHistory.reduce((sum, d) => sum + d.utilization, 0) / this.scalingHistory.length || 0
    };
  }
}

module.exports = WorkloadMonitor;

