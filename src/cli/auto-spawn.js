#!/usr/bin/env node
/**
 * CLI tool for Smart Agent Auto-Spawning
 * Usage: node src/cli/auto-spawn.js [command] [options]
 */

const AutoSpawningOrchestrator = require('../services/auto-spawning-orchestrator');

const commands = {
  async start(options) {
    console.log('🚀 Starting Auto-Spawning System...\n');
    
    const orchestrator = new AutoSpawningOrchestrator({
      maxAgents: options.maxAgents || 8,
      enableMCP: options.enableMCP !== false
    });
    
    await orchestrator.start();
    
    console.log('✅ System started successfully');
    console.log('\nPress Ctrl+C to stop\n');
    
    // Keep process alive
    process.on('SIGINT', async () => {
      console.log('\n⏹️  Stopping system...');
      await orchestrator.stop();
      process.exit(0);
    });
    
    // Show status every 10 seconds
    setInterval(async () => {
      const status = await orchestrator.getStatus();
      console.log(`📊 Status: ${status.spawnerMetrics.activeAgents} active agents, ${status.monitorStats.currentMetrics.queueDepth} queued tasks`);
    }, 10000);
  },

  async process(options) {
    console.log('📝 Processing task...\n');
    
    const orchestrator = new AutoSpawningOrchestrator({
      maxAgents: options.maxAgents || 8,
      enableMCP: options.enableMCP !== false
    });
    
    await orchestrator.start();
    
    const result = await orchestrator.processTask({
      description: options.task || 'No task description provided',
      files: options.files || [],
      context: { priority: options.priority || 'medium' }
    });
    
    console.log('\n✅ Task processed successfully\n');
    console.log('📋 Spawn Plan:');
    console.log(`   Complexity: ${result.plan.complexity.complexity} (score: ${result.plan.complexity.score})`);
    console.log(`   Topology: ${result.plan.topology}`);
    console.log(`   Agents: ${result.plan.agents.join(', ')}`);
    console.log(`   Estimated Duration: ${result.plan.complexity.estimatedDuration} minutes`);
    
    await orchestrator.stop();
  },

  async analyze(options) {
    console.log('🔍 Analyzing task complexity...\n');
    
    const orchestrator = new AutoSpawningOrchestrator();
    const analysis = orchestrator.spawner.analyzeTaskComplexity(options.task || '');
    
    console.log('📊 Analysis Results:');
    console.log(`   Complexity: ${analysis.complexity}`);
    console.log(`   Score: ${analysis.score}/4`);
    console.log(`   Recommended Agents: ${analysis.recommendedAgents.join(', ')}`);
    console.log(`   Estimated Duration: ${analysis.estimatedDuration} minutes`);
    console.log(`   Parallelizable: ${analysis.parallelizable ? 'Yes' : 'No'}`);
  },

  async detect(options) {
    console.log('🔍 Detecting agent from file type...\n');
    
    const orchestrator = new AutoSpawningOrchestrator();
    const files = options.files || [];
    
    if (files.length === 0) {
      console.log('⚠️  No files provided');
      return;
    }
    
    console.log('📁 File Analysis:');
    files.forEach(file => {
      const agent = orchestrator.spawner.detectAgentFromFileType(file);
      console.log(`\n   ${file}:`);
      console.log(`   → Type: ${agent.type}`);
      console.log(`   → Specialization: ${agent.specialization}`);
      console.log(`   → Capabilities: ${agent.capabilities.join(', ')}`);
    });
  },

  async status(options) {
    console.log('📊 Getting system status...\n');
    
    const orchestrator = new AutoSpawningOrchestrator({
      enableMCP: options.enableMCP !== false
    });
    
    await orchestrator.start();
    const status = await orchestrator.getStatus();
    
    console.log('System Status:');
    console.log(`   Running: ${status.running ? '✅' : '❌'}`);
    console.log(`   Active Agents: ${status.spawnerMetrics.activeAgents}`);
    console.log(`   Total Spawned: ${status.spawnerMetrics.totalSpawned}`);
    console.log(`   Queue Depth: ${status.monitorStats.currentMetrics.queueDepth}`);
    console.log(`   Average Utilization: ${(status.monitorStats.averageUtilization * 100).toFixed(1)}%`);
    console.log(`   MCP Enabled: ${status.config.enableMCP ? '✅' : '❌'}`);
    
    await orchestrator.stop();
  },

  help() {
    console.log(`
Smart Agent Auto-Spawning CLI

Usage:
  node src/cli/auto-spawn.js <command> [options]

Commands:
  start              Start the auto-spawning system
  process            Process a task with auto-spawning
  analyze            Analyze task complexity
  detect             Detect agent from file types
  status             Show system status
  help               Show this help message

Options:
  --task <desc>      Task description
  --files <files>    Comma-separated list of files
  --max-agents <n>   Maximum number of agents (default: 8)
  --priority <p>     Task priority (low, medium, high)
  --enable-mcp       Enable MCP integration (default: true)
  --no-mcp           Disable MCP integration

Examples:
  # Start the system
  node src/cli/auto-spawn.js start --max-agents 10

  # Process a task
  node src/cli/auto-spawn.js process --task "Implement OAuth" --files "auth.go,login.tsx"

  # Analyze task complexity
  node src/cli/auto-spawn.js analyze --task "Implement distributed caching"

  # Detect agents from files
  node src/cli/auto-spawn.js detect --files "backend/api.go,frontend/App.tsx"

  # Check system status
  node src/cli/auto-spawn.js status
    `);
  }
};

// Parse command line arguments
const args = process.argv.slice(2);
const command = args[0] || 'help';

const options = {};
for (let i = 1; i < args.length; i++) {
  if (args[i].startsWith('--')) {
    const key = args[i].substring(2);
    const value = args[i + 1] && !args[i + 1].startsWith('--') ? args[i + 1] : true;
    
    if (key === 'files' && typeof value === 'string') {
      options[key] = value.split(',').map(f => f.trim());
    } else if (key === 'max-agents') {
      options.maxAgents = parseInt(value);
    } else if (key === 'no-mcp') {
      options.enableMCP = false;
    } else {
      options[key] = value;
    }
    
    if (value !== true) i++;
  }
}

// Execute command
if (commands[command]) {
  commands[command](options).catch(error => {
    console.error('❌ Error:', error.message);
    process.exit(1);
  });
} else {
  console.error(`❌ Unknown command: ${command}`);
  commands.help();
  process.exit(1);
}

