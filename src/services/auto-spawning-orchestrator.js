/**
 * Auto-Spawning Orchestrator
 * Main coordinator for smart agent auto-spawning system
 */

const SmartAgentSpawner = require('./smart-agent-spawner');
const WorkloadMonitor = require('./workload-monitor');
const MCPIntegration = require('./mcp-integration');
const config = require('../config/auto-spawning-config');

class AutoSpawningOrchestrator {
  constructor(customConfig = {}) {
    this.config = { ...config.global, ...customConfig };
    
    // Initialize components
    this.spawner = new SmartAgentSpawner(this.config);
    this.monitor = new WorkloadMonitor(config.scaling);
    this.mcp = new MCPIntegration({ enabled: this.config.enableMCP });
    
    // State
    this.activeSwarm = null;
    this.isRunning = false;
    
    // Setup event listeners
    this.setupEventListeners();
  }

  /**
   * Setup event listeners between components
   */
  setupEventListeners() {
    // When spawner creates a plan, update monitor
    this.spawner.on('spawn-plan', (plan) => {
      console.log(`📋 Spawn plan created: ${plan.agents.length} agents, ${plan.topology} topology`);
      this.monitor.updateMetrics({
        activeAgents: plan.agents.length
      });
    });

    // When monitor detects scaling need, trigger spawning
    this.monitor.on('scaling-decision', async (decision) => {
      console.log(`⚖️  Scaling decision: ${decision.action} - ${decision.reason}`);
      
      if (decision.action === 'scale-up') {
        await this.handleScaleUp(decision);
      } else if (decision.action === 'scale-down') {
        await this.handleScaleDown(decision);
      }
    });
  }

  /**
   * Start the auto-spawning system
   */
  async start() {
    if (this.isRunning) {
      console.log('⚠️  Auto-spawning orchestrator already running');
      return;
    }

    console.log('🚀 Starting Auto-Spawning Orchestrator');
    
    // Health check MCP connection
    const health = await this.mcp.healthCheck();
    if (!health.healthy && this.config.enableMCP) {
      console.warn('⚠️  MCP health check failed, continuing without MCP integration');
    }
    
    // Start workload monitoring
    this.monitor.start();
    
    this.isRunning = true;
    console.log('✅ Auto-Spawning Orchestrator started successfully');
  }

  /**
   * Stop the auto-spawning system
   */
  async stop() {
    if (!this.isRunning) {
      return;
    }

    console.log('⏹️  Stopping Auto-Spawning Orchestrator');
    
    // Stop monitoring
    this.monitor.stop();
    
    this.isRunning = false;
    console.log('✅ Auto-Spawning Orchestrator stopped');
  }

  /**
   * Process a new task with auto-spawning
   */
  async processTask(taskConfig) {
    const { description, files = [], context = {} } = taskConfig;
    
    console.log(`\n📝 Processing task: ${description}`);
    console.log(`📁 Files involved: ${files.length}`);
    
    // Create spawning plan
    const plan = await this.spawner.autoSpawn({
      task: description,
      files,
      context
    });
    
    // Initialize swarm if needed
    if (!this.activeSwarm && this.config.enableMCP) {
      const swarmResult = await this.mcp.initializeSwarm(
        plan.topology,
        plan.agents.length,
        this.config.defaultStrategy
      );
      
      if (swarmResult.success) {
        this.activeSwarm = swarmResult;
        console.log(`✅ Swarm initialized: ${swarmResult.swarmId}`);
      }
    }
    
    // Spawn agents via MCP
    const spawnedAgents = [];
    if (this.config.enableMCP) {
      for (const agentType of plan.agents) {
        const result = await this.mcp.spawnAgent({
          type: agentType,
          name: `${agentType}-${Date.now()}`,
          capabilities: this.getAgentCapabilities(agentType)
        });
        
        if (result.success) {
          spawnedAgents.push(result.agent);
        }
      }
    }
    
    // Orchestrate task
    if (spawnedAgents.length > 0) {
      await this.mcp.orchestrateTask({
        task: description,
        agents: spawnedAgents.map(a => a.id),
        strategy: 'parallel',
        priority: plan.complexity.score >= 3 ? 'high' : 'medium'
      });
    }
    
    // Update monitor metrics
    this.monitor.updateMetrics({
      queueDepth: this.monitor.metrics.queueDepth + 1,
      activeAgents: spawnedAgents.length
    });
    
    return {
      plan,
      spawnedAgents,
      swarm: this.activeSwarm
    };
  }

  /**
   * Get capabilities for an agent type
   */
  getAgentCapabilities(agentType) {
    const capabilityMap = {
      'coordinator': ['task-coordination', 'agent-management'],
      'coder': ['code-implementation', 'debugging', 'refactoring'],
      'architect': ['system-design', 'architecture-planning', 'technical-decisions'],
      'tester': ['test-creation', 'quality-assurance', 'test-automation'],
      'researcher': ['information-gathering', 'analysis', 'documentation'],
      'security-auditor': ['security-analysis', 'vulnerability-scanning', 'compliance'],
      'performance-optimizer': ['performance-analysis', 'optimization', 'benchmarking'],
      'database-expert': ['database-design', 'sql', 'data-modeling'],
      'api-specialist': ['api-design', 'rest', 'grpc', 'api-documentation']
    };
    
    return capabilityMap[agentType] || ['general-development'];
  }

  /**
   * Handle scale-up decision
   */
  async handleScaleUp(decision) {
    const additionalAgents = decision.targetAgents - decision.currentAgents;
    console.log(`📈 Scaling up: adding ${additionalAgents} agents`);
    
    // Spawn additional general-purpose agents
    for (let i = 0; i < additionalAgents; i++) {
      await this.mcp.spawnAgent({
        type: 'coder',
        name: `auto-scaled-coder-${Date.now()}-${i}`,
        capabilities: ['general-development']
      });
    }
  }

  /**
   * Handle scale-down decision
   */
  async handleScaleDown(decision) {
    const agentsToRemove = decision.currentAgents - decision.targetAgents;
    console.log(`📉 Scaling down: removing ${agentsToRemove} agents`);
    
    // In a real implementation, this would gracefully terminate idle agents
    // For now, we just log the decision
  }

  /**
   * Get system status
   */
  async getStatus() {
    const swarmStatus = this.config.enableMCP ? 
      await this.mcp.getSwarmStatus() : 
      { success: false, reason: 'MCP disabled' };
    
    return {
      running: this.isRunning,
      swarm: this.activeSwarm,
      swarmStatus,
      spawnerMetrics: this.spawner.getMetrics(),
      monitorStats: this.monitor.getStatistics(),
      config: this.config
    };
  }
}

module.exports = AutoSpawningOrchestrator;

