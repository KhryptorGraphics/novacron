/**
 * Real MCP Integration for Claude Flow
 * Replaces simulated MCP calls with actual Claude Flow integration
 */

const { exec } = require('child_process');
const { promisify } = require('util');
const execAsync = promisify(exec);

class RealMCPIntegration {
  constructor(config = {}) {
    this.config = {
      enabled: config.enabled !== false,
      mcpCommand: config.mcpCommand || 'npx claude-flow@alpha',
      timeout: config.timeout || 30000,
      retries: config.retries || 3,
      retryDelay: config.retryDelay || 1000,
      ...config
    };
    
    this.connectionStatus = 'disconnected';
    this.lastHealthCheck = null;
    this.activeSwarms = new Map();
    this.activeAgents = new Map();
  }

  /**
   * Execute MCP command with retry logic
   */
  async executeMCPCommand(command, args = {}, retryCount = 0) {
    if (!this.config.enabled) {
      return { success: false, reason: 'MCP disabled' };
    }

    try {
      const argsStr = Object.entries(args)
        .map(([key, value]) => `--${key}="${JSON.stringify(value).replace(/"/g, '\\"')}"`)
        .join(' ');
      
      const fullCommand = `${this.config.mcpCommand} ${command} ${argsStr}`;
      
      const { stdout, stderr } = await execAsync(fullCommand, {
        timeout: this.config.timeout
      });
      
      if (stderr && !stderr.includes('warning')) {
        throw new Error(stderr);
      }
      
      // Parse JSON output
      try {
        return JSON.parse(stdout);
      } catch {
        return { success: true, output: stdout };
      }
      
    } catch (error) {
      if (retryCount < this.config.retries) {
        await new Promise(resolve => setTimeout(resolve, this.config.retryDelay));
        return this.executeMCPCommand(command, args, retryCount + 1);
      }
      
      console.error(`MCP command failed after ${retryCount + 1} attempts:`, error.message);
      return { success: false, error: error.message };
    }
  }

  /**
   * Initialize swarm with real MCP tools
   */
  async initializeSwarm(topology, maxAgents, strategy = 'auto') {
    console.log(`🔄 Initializing ${topology} swarm with Claude Flow MCP...`);
    
    const result = await this.executeMCPCommand('swarm init', {
      topology,
      maxAgents,
      strategy
    });
    
    if (result.success) {
      const swarmId = result.swarmId || `swarm-${Date.now()}`;
      this.activeSwarms.set(swarmId, {
        id: swarmId,
        topology,
        maxAgents,
        strategy,
        createdAt: new Date().toISOString(),
        agents: []
      });
      
      this.connectionStatus = 'connected';
      this.lastHealthCheck = new Date();
      
      console.log(`✅ Swarm initialized: ${swarmId}`);
      return { success: true, swarmId, ...result };
    }
    
    return result;
  }

  /**
   * Spawn agent via real MCP
   */
  async spawnAgent(agentConfig) {
    const { type, name, capabilities = [], swarmId } = agentConfig;
    
    console.log(`🤖 Spawning ${type} agent via Claude Flow MCP...`);
    
    const result = await this.executeMCPCommand('agent spawn', {
      type,
      name: name || type,
      capabilities,
      swarmId: swarmId || 'current'
    });
    
    if (result.success) {
      const agentId = result.agentId || `agent-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      const agent = {
        id: agentId,
        type,
        name: name || type,
        capabilities,
        swarmId,
        status: 'active',
        spawnedAt: new Date().toISOString()
      };
      
      this.activeAgents.set(agentId, agent);
      
      // Add to swarm if exists
      if (swarmId && this.activeSwarms.has(swarmId)) {
        this.activeSwarms.get(swarmId).agents.push(agentId);
      }
      
      console.log(`✅ Agent spawned: ${agentId}`);
      return { success: true, agent, ...result };
    }
    
    return result;
  }

  /**
   * Orchestrate task via real MCP
   */
  async orchestrateTask(taskConfig) {
    const { task, agents, strategy = 'parallel', priority = 'medium', swarmId } = taskConfig;
    
    console.log(`📋 Orchestrating task via Claude Flow MCP...`);
    
    const result = await this.executeMCPCommand('task orchestrate', {
      task,
      agents,
      strategy,
      priority,
      swarmId: swarmId || 'current'
    });
    
    if (result.success) {
      const taskId = result.taskId || `task-${Date.now()}`;
      console.log(`✅ Task orchestrated: ${taskId}`);
      return { success: true, taskId, ...result };
    }
    
    return result;
  }

  /**
   * Get swarm status via real MCP
   */
  async getSwarmStatus(swarmId = 'current') {
    const result = await this.executeMCPCommand('swarm status', { swarmId });
    
    if (result.success) {
      return {
        success: true,
        swarmId,
        status: this.connectionStatus,
        ...result
      };
    }
    
    return result;
  }

  /**
   * Get agent metrics via real MCP
   */
  async getAgentMetrics(agentId) {
    const result = await this.executeMCPCommand('agent metrics', { agentId });
    
    if (result.success) {
      return {
        success: true,
        agentId,
        ...result
      };
    }
    
    return result;
  }

  /**
   * Health check with real MCP
   */
  async healthCheck() {
    try {
      const result = await this.executeMCPCommand('health', {});
      
      if (result.success) {
        this.lastHealthCheck = new Date();
        this.connectionStatus = 'connected';
        return { healthy: true, timestamp: this.lastHealthCheck, ...result };
      }
      
      this.connectionStatus = 'error';
      return { healthy: false, error: result.error };
      
    } catch (error) {
      this.connectionStatus = 'error';
      return { healthy: false, error: error.message };
    }
  }

  /**
   * Get all active swarms
   */
  getActiveSwarms() {
    return Array.from(this.activeSwarms.values());
  }

  /**
   * Get all active agents
   */
  getActiveAgents() {
    return Array.from(this.activeAgents.values());
  }

  /**
   * Terminate agent
   */
  async terminateAgent(agentId) {
    const result = await this.executeMCPCommand('agent terminate', { agentId });
    
    if (result.success) {
      this.activeAgents.delete(agentId);
      console.log(`🛑 Agent terminated: ${agentId}`);
    }
    
    return result;
  }

  /**
   * Shutdown swarm
   */
  async shutdownSwarm(swarmId) {
    const result = await this.executeMCPCommand('swarm shutdown', { swarmId });
    
    if (result.success) {
      this.activeSwarms.delete(swarmId);
      console.log(`🛑 Swarm shutdown: ${swarmId}`);
    }
    
    return result;
  }
}

module.exports = RealMCPIntegration;

