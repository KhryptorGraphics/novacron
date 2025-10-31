# Quick Start Guide - Smart Agent Auto-Spawning

Get started with the Smart Agent Auto-Spawning system in 5 minutes!

---

## 📦 Installation

### Prerequisites
- Node.js >= 18.0.0
- npm >= 9.0.0

### Install Dependencies
```bash
npm install
```

---

## 🚀 Quick Start

### 1. Start the Auto-Spawning System
```bash
npm run auto-spawn:start
```

This will:
- Initialize the orchestrator
- Start workload monitoring
- Enable MCP integration (if configured)
- Begin accepting tasks

### 2. Check System Status
```bash
npm run auto-spawn:status
```

Output:
```
📊 Getting system status...

System Status:
   Running: ✅
   Active Agents: 0
   Total Spawned: 0
   Queue Depth: 0
   Average Utilization: 0.0%
   MCP Enabled: ✅
```

### 3. Process Your First Task
```bash
node src/cli/auto-spawn.js process \
  --task "Implement OAuth authentication" \
  --files "backend/auth/oauth.go,frontend/components/Login.tsx"
```

Output:
```
📝 Processing task...

✅ Task processed successfully

📋 Spawn Plan:
   Complexity: very-complex (score: 4)
   Topology: adaptive
   Agents: coordinator, architect, coder, tester, researcher, security-auditor
   Estimated Duration: 60 minutes
```

---

## 💻 CLI Commands

### Start System
```bash
npm run auto-spawn:start
# or
node src/cli/auto-spawn.js start --max-agents 10
```

### Process Task
```bash
node src/cli/auto-spawn.js process \
  --task "Add new feature to dashboard" \
  --files "frontend/components/Dashboard.tsx" \
  --priority high
```

### Analyze Task Complexity
```bash
node src/cli/auto-spawn.js analyze \
  --task "Implement distributed caching system"
```

Output:
```
🔍 Analyzing task complexity...

📊 Analysis Results:
   Complexity: very-complex
   Score: 4/4
   Recommended Agents: coordinator, architect, coder, tester, researcher
   Estimated Duration: 60 minutes
   Parallelizable: Yes
```

### Detect Agents from Files
```bash
node src/cli/auto-spawn.js detect \
  --files "backend/api.go,frontend/App.tsx,database/schema.sql"
```

Output:
```
🔍 Detecting agent from file type...

📁 File Analysis:

   backend/api.go:
   → Type: coder
   → Specialization: go-backend
   → Capabilities: go-lang, distributed-systems

   frontend/App.tsx:
   → Type: coder
   → Specialization: react-frontend
   → Capabilities: typescript, react, nextjs

   database/schema.sql:
   → Type: analyst
   → Specialization: database-expert
   → Capabilities: sql, database-design
```

### Get Help
```bash
npm run auto-spawn:help
```

---

## 🧪 Running Tests

### Unit Tests
```bash
npm run test:auto-spawn
```

### Integration Tests
```bash
npm run test:auto-spawn:integration
```

### All Tests
```bash
npm test
```

---

## 📝 Programmatic Usage

### Basic Example
```javascript
const AutoSpawningOrchestrator = require('./src/services/auto-spawning-orchestrator');

// Create orchestrator
const orchestrator = new AutoSpawningOrchestrator({
  maxAgents: 8,
  enableMCP: true
});

// Start system
await orchestrator.start();

// Process a task
const result = await orchestrator.processTask({
  description: 'Implement OAuth authentication',
  files: ['backend/auth/oauth.go', 'frontend/components/Login.tsx'],
  context: { priority: 'high' }
});

console.log('Spawned agents:', result.spawnedAgents);
console.log('Topology:', result.plan.topology);

// Get status
const status = await orchestrator.getStatus();
console.log('Active agents:', status.spawnerMetrics.activeAgents);

// Stop system
await orchestrator.stop();
```

### Advanced Example with Events
```javascript
const orchestrator = new AutoSpawningOrchestrator();

// Listen for spawn events
orchestrator.spawner.on('spawn-plan', (plan) => {
  console.log('New spawn plan:', plan);
});

// Listen for scaling events
orchestrator.monitor.on('scaling-decision', (decision) => {
  console.log('Scaling decision:', decision);
});

await orchestrator.start();

// Process multiple tasks
const tasks = [
  { description: 'Task 1', files: ['file1.go'] },
  { description: 'Task 2', files: ['file2.tsx'] },
  { description: 'Task 3', files: ['file3.sql'] }
];

for (const task of tasks) {
  await orchestrator.processTask(task);
}
```

---

## ⚙️ Configuration

### Custom Configuration
```javascript
const orchestrator = new AutoSpawningOrchestrator({
  maxAgents: 12,
  minAgents: 2,
  defaultTopology: 'hierarchical',
  defaultStrategy: 'optimal',
  enableMCP: true
});
```

### Environment Variables
```bash
# Set max agents
export AUTO_SPAWN_MAX_AGENTS=10

# Disable MCP
export AUTO_SPAWN_ENABLE_MCP=false

# Set log level
export AUTO_SPAWN_LOG_LEVEL=debug
```

---

## 🔧 Troubleshooting

### Issue: MCP Connection Failed
**Solution**: Check if Claude Flow is installed and configured
```bash
npx claude-flow@alpha --version
```

### Issue: No Agents Spawned
**Solution**: Check system status and logs
```bash
npm run auto-spawn:status
# Check logs in console output
```

### Issue: High Memory Usage
**Solution**: Reduce max agents or enable scaling
```javascript
const orchestrator = new AutoSpawningOrchestrator({
  maxAgents: 4,  // Reduce from 8
  scaling: {
    scaleDownThreshold: 0.3  // More aggressive scale-down
  }
});
```

---

## 📚 Next Steps

1. **Read the Full Guide**: [Smart Agent Auto-Spawning Guide](./smart-agent-auto-spawning-guide.md)
2. **Explore Configuration**: [Configuration Reference](../src/config/auto-spawning-config.js)
3. **Check Examples**: [Integration Tests](../tests/integration/auto-spawning-integration.test.js)
4. **Review Roadmap**: [Development Roadmap](./development-roadmap.md)

---

## 🆘 Support

- **Documentation**: See [docs/](./smart-agent-auto-spawning-guide.md)
- **Issues**: Report bugs or request features
- **Examples**: Check [tests/integration/](../tests/integration/)

---

**Happy Auto-Spawning! 🤖✨**

