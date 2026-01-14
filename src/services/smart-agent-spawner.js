/**
 * Smart Agent Auto-Spawning Service
 * Automatically spawns the right agents at the right time based on:
 * - File type detection
 * - Task complexity analysis (ML-powered)
 * - Dynamic workload monitoring
 */

const EventEmitter = require('events');
const TaskClassifier = require('../ml/task-classifier');

class SmartAgentSpawner extends EventEmitter {
  constructor(config = {}) {
    super();
    this.config = {
      maxAgents: config.maxAgents || 8,
      minAgents: config.minAgents || 1,
      strategy: config.strategy || 'auto',
      topology: config.topology || 'mesh',
      enableMCP: config.enableMCP !== false,
      useMLClassifier: config.useMLClassifier !== false,
      ...config
    };

    this.activeAgents = new Map();
    this.taskQueue = [];
    this.metrics = {
      totalSpawned: 0,
      totalCompleted: 0,
      averageComplexity: 0,
      spawningDecisions: [],
      mlPredictions: []
    };

    // Initialize ML classifier
    if (this.config.useMLClassifier) {
      this.mlClassifier = new TaskClassifier();
    }
  }

  /**
   * File Type Detection - Maps file extensions to agent types
   */
  detectAgentFromFileType(filePath) {
    const fileTypeMap = {
      // Backend
      '.go': { type: 'coder', specialization: 'go-backend', capabilities: ['go-lang', 'distributed-systems'] },
      '.py': { type: 'coder', specialization: 'python-backend', capabilities: ['python', 'ai-ml'] },
      '.rs': { type: 'coder', specialization: 'rust-systems', capabilities: ['rust', 'performance'] },
      
      // Frontend
      '.tsx': { type: 'coder', specialization: 'react-frontend', capabilities: ['typescript', 'react', 'nextjs'] },
      '.ts': { type: 'coder', specialization: 'typescript-dev', capabilities: ['typescript', 'nodejs'] },
      '.jsx': { type: 'coder', specialization: 'react-dev', capabilities: ['javascript', 'react'] },
      '.js': { type: 'coder', specialization: 'javascript-dev', capabilities: ['javascript', 'nodejs'] },
      
      // Configuration
      '.yaml': { type: 'analyst', specialization: 'config-manager', capabilities: ['yaml', 'configuration'] },
      '.yml': { type: 'analyst', specialization: 'config-manager', capabilities: ['yaml', 'configuration'] },
      '.json': { type: 'analyst', specialization: 'data-analyst', capabilities: ['json', 'data-structures'] },
      '.toml': { type: 'analyst', specialization: 'config-manager', capabilities: ['toml', 'configuration'] },
      
      // Documentation
      '.md': { type: 'researcher', specialization: 'documentation', capabilities: ['markdown', 'technical-writing'] },
      '.rst': { type: 'researcher', specialization: 'documentation', capabilities: ['restructuredtext', 'documentation'] },
      
      // Database
      '.sql': { type: 'analyst', specialization: 'database-expert', capabilities: ['sql', 'database-design'] },
      
      // Infrastructure
      '.tf': { type: 'coder', specialization: 'infrastructure', capabilities: ['terraform', 'iac'] },
      'Dockerfile': { type: 'coder', specialization: 'devops', capabilities: ['docker', 'containers'] },
      'docker-compose.yml': { type: 'coder', specialization: 'devops', capabilities: ['docker-compose', 'orchestration'] },
      
      // Testing
      '.test.js': { type: 'tester', specialization: 'unit-testing', capabilities: ['jest', 'testing'] },
      '.spec.ts': { type: 'tester', specialization: 'unit-testing', capabilities: ['jest', 'typescript-testing'] },
      '.e2e.js': { type: 'tester', specialization: 'e2e-testing', capabilities: ['playwright', 'e2e'] }
    };

    const ext = filePath.includes('.') ? '.' + filePath.split('.').pop() : filePath;
    return fileTypeMap[ext] || { type: 'coder', specialization: 'general', capabilities: ['general-development'] };
  }

  /**
   * Task Complexity Analysis
   * Analyzes task description to determine complexity and required agents
   * Uses ML classifier if enabled, falls back to rule-based
   */
  analyzeTaskComplexity(taskDescription) {
    // Use ML classifier if enabled
    if (this.config.useMLClassifier && this.mlClassifier) {
      const mlPrediction = this.mlClassifier.predict(taskDescription);

      // Record ML prediction
      this.metrics.mlPredictions.push({
        task: taskDescription,
        prediction: mlPrediction,
        timestamp: new Date().toISOString()
      });

      return {
        complexity: mlPrediction.complexity,
        score: mlPrediction.level,
        recommendedAgents: this.getRecommendedAgents(mlPrediction.complexity, taskDescription.toLowerCase()),
        estimatedDuration: mlPrediction.level * 15,
        parallelizable: mlPrediction.level >= 3,
        mlPowered: true,
        confidence: mlPrediction.confidence,
        reasoning: mlPrediction.reasoning
      };
    }

    // Fallback to rule-based analysis
    const complexityIndicators = {
      simple: ['fix typo', 'update comment', 'rename', 'format', 'style'],
      medium: ['add feature', 'refactor', 'optimize', 'update', 'modify'],
      complex: ['implement', 'design', 'architect', 'migrate', 'integrate'],
      veryComplex: ['oauth', 'authentication', 'distributed', 'microservices', 'real-time', 'scaling']
    };

    const lowerTask = taskDescription.toLowerCase();
    let complexity = 'simple';
    let score = 1;

    if (complexityIndicators.veryComplex.some(keyword => lowerTask.includes(keyword))) {
      complexity = 'very-complex';
      score = 4;
    } else if (complexityIndicators.complex.some(keyword => lowerTask.includes(keyword))) {
      complexity = 'complex';
      score = 3;
    } else if (complexityIndicators.medium.some(keyword => lowerTask.includes(keyword))) {
      complexity = 'medium';
      score = 2;
    }

    return {
      complexity,
      score,
      recommendedAgents: this.getRecommendedAgents(complexity, lowerTask),
      estimatedDuration: score * 15, // minutes
      parallelizable: score >= 3,
      mlPowered: false
    };
  }

  /**
   * Get recommended agents based on complexity and task keywords
   */
  getRecommendedAgents(complexity, taskDescription) {
    const baseAgents = ['coordinator'];

    switch (complexity) {
      case 'simple':
        return baseAgents;

      case 'medium':
        return [...baseAgents, 'coder'];

      case 'complex':
        return [...baseAgents, 'architect', 'coder', 'tester'];

      case 'very-complex':
        const agents = [...baseAgents, 'architect', 'coder', 'tester', 'researcher'];

        // Add specialized agents based on keywords
        if (taskDescription.includes('auth') || taskDescription.includes('security')) {
          agents.push('security-auditor');
        }
        if (taskDescription.includes('database') || taskDescription.includes('sql')) {
          agents.push('database-expert');
        }
        if (taskDescription.includes('api') || taskDescription.includes('rest')) {
          agents.push('api-specialist');
        }
        if (taskDescription.includes('performance') || taskDescription.includes('optimize')) {
          agents.push('performance-optimizer');
        }

        return agents;

      default:
        return baseAgents;
    }
  }

  /**
   * Auto-spawn agents based on file changes and task
   */
  async autoSpawn(options = {}) {
    const { files = [], task = '', context = {} } = options;

    // Analyze task complexity
    const complexityAnalysis = this.analyzeTaskComplexity(task);

    // Detect agents from file types
    const fileAgents = files.map(file => this.detectAgentFromFileType(file));

    // Combine recommendations
    const recommendedAgents = this.combineAgentRecommendations(
      complexityAnalysis.recommendedAgents,
      fileAgents
    );

    // Determine topology based on complexity
    const topology = this.selectTopology(complexityAnalysis, recommendedAgents.length);

    const spawningPlan = {
      agents: recommendedAgents,
      topology,
      complexity: complexityAnalysis,
      timestamp: new Date().toISOString(),
      context
    };

    // Record decision
    this.metrics.spawningDecisions.push(spawningPlan);

    // Emit event for spawning
    this.emit('spawn-plan', spawningPlan);

    // Execute spawning if MCP is enabled
    if (this.config.enableMCP) {
      await this.executeMCPSpawning(spawningPlan);
    }

    return spawningPlan;
  }

  /**
   * Combine agent recommendations from different sources
   */
  combineAgentRecommendations(taskAgents, fileAgents) {
    const agentSet = new Set(taskAgents);

    // Add unique agents from file analysis
    fileAgents.forEach(agent => {
      if (agent.type) {
        agentSet.add(agent.type);
      }
    });

    // Ensure we don't exceed max agents
    const agents = Array.from(agentSet).slice(0, this.config.maxAgents);

    return agents;
  }

  /**
   * Select optimal topology based on task characteristics
   */
  selectTopology(complexityAnalysis, agentCount) {
    if (this.config.topology !== 'auto') {
      return this.config.topology;
    }

    // Simple tasks: single agent (no topology needed)
    if (complexityAnalysis.score === 1 || agentCount === 1) {
      return 'single';
    }

    // Medium complexity: mesh for peer collaboration
    if (complexityAnalysis.score === 2 || agentCount <= 3) {
      return 'mesh';
    }

    // Complex tasks: hierarchical for coordination
    if (complexityAnalysis.score === 3 || agentCount <= 5) {
      return 'hierarchical';
    }

    // Very complex: adaptive for dynamic optimization
    return 'adaptive';
  }

  /**
   * Execute agent spawning via MCP tools
   */
  async executeMCPSpawning(plan) {
    try {
      console.log(`🤖 Spawning ${plan.agents.length} agents with ${plan.topology} topology`);

      // This would integrate with actual MCP tools
      // For now, we'll simulate the spawning
      const spawnedAgents = [];

      for (const agentType of plan.agents) {
        const agent = {
          id: `agent-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          type: agentType,
          status: 'active',
          spawnedAt: new Date().toISOString()
        };

        this.activeAgents.set(agent.id, agent);
        spawnedAgents.push(agent);
        this.metrics.totalSpawned++;
      }

      console.log(`✅ Successfully spawned ${spawnedAgents.length} agents`);
      return spawnedAgents;

    } catch (error) {
      console.error('❌ Failed to spawn agents:', error);
      throw error;
    }
  }

  /**
   * Get current metrics
   */
  getMetrics() {
    return {
      ...this.metrics,
      activeAgents: this.activeAgents.size,
      recentDecisions: this.metrics.spawningDecisions.slice(-10)
    };
  }
}

module.exports = SmartAgentSpawner;

