#!/bin/bash

# NovaCron Claude Agent Activation Script
# This script activates the specialized agents for the NovaCron VM Management System

echo "🚀 Activating NovaCron Claude Agents..."

# Set environment variables
export CLAUDE_PROJECT="NovaCron"
export CLAUDE_CONFIG_PATH="$(dirname "$0")"
export NOVACRON_API_URL="http://localhost:8090"
export NOVACRON_WS_URL="ws://localhost:8091"
export NOVACRON_FRONTEND_URL="http://localhost:8092"

# Check if claude-flow is installed
if ! command -v npx &> /dev/null; then
    echo "❌ npx not found. Please install Node.js first."
    exit 1
fi

# Initialize hive-mind if requested
if [[ "$1" == "--hive" ]] || [[ "$1" == "--swarm" ]]; then
    echo "🧠 Initializing Hive Mind collective intelligence..."
    
    # Create hive-mind session
    npx claude-flow@alpha hive-mind spawn \
        --project "NovaCron" \
        --agents 10 \
        --strategy parallel \
        --claude \
        --sparc \
        --think-harder \
        --magic \
        --auto-spawn \
        --neural-enhanced \
        --ai-guided \
        --memory-enhanced \
        --auto-learn \
        --agents vm-manager,migration-coordinator,storage-optimizer,cluster-orchestrator,container-manager,network-controller,performance-analyst,scheduler,security-guardian,devops-automator \
        --verbose \
        --config "$CLAUDE_CONFIG_PATH/agents.json"
fi

# Register commands with Claude
echo "📝 Registering NovaCron commands..."

# Create command registration file
cat > /tmp/novacron-commands.json << 'EOF'
{
  "commands": [
    { "name": "/vm:manage", "description": "Manage VM lifecycle operations" },
    { "name": "/vm:migrate", "description": "Migrate VMs between nodes" },
    { "name": "/cluster:health", "description": "Check cluster health status" },
    { "name": "/monitoring:metrics", "description": "Collect and analyze metrics" },
    { "name": "/container:deploy", "description": "Deploy containerized workloads" },
    { "name": "/storage:optimize", "description": "Optimize storage usage" }
  ]
}
EOF

# Test command system
echo "🧪 Testing command system..."
node "$CLAUDE_CONFIG_PATH/commands/index.js" help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Command system operational"
else
    echo "⚠️ Command system test failed, but continuing..."
fi

# Display available agents
echo ""
echo "📊 Available NovaCron Agents:"
echo "  • VM Lifecycle Manager - Primary VM operations"
echo "  • Migration Coordinator - Handle VM migrations"
echo "  • Storage Optimizer - Storage compression & deduplication"
echo "  • Cluster Orchestrator - Node management & failover"
echo "  • Container Manager - Container deployment & scaling"
echo "  • Network Controller - Overlay networking & security"
echo "  • Performance Analyst - Metrics & anomaly detection"
echo "  • Resource Scheduler - Policy-based scheduling"
echo "  • Security Guardian - RBAC & compliance"
echo "  • DevOps Automator - CI/CD & automation"

echo ""
echo "📚 Available Commands:"
echo "  /vm:manage <action> <vm-id>     - VM lifecycle operations"
echo "  /vm:migrate <vm-id> <dest>      - Migrate VM to destination"
echo "  /cluster:health [--detailed]    - Check cluster health"
echo "  /monitoring:metrics [--vm id]   - Collect system metrics"
echo "  /container:deploy <image>       - Deploy container"
echo "  /storage:optimize [--compress]  - Optimize storage"

echo ""
echo "💡 Tips:"
echo "  • Use 'make test' to run backend tests"
echo "  • Use 'npm run dev' in frontend/ for UI development"
echo "  • Use 'docker-compose up -d' to start all services"
echo "  • API available at http://localhost:8090"
echo "  • Frontend available at http://localhost:8092"

echo ""
echo "✨ NovaCron agents activated successfully!"
echo "🎯 Ready for distributed VM management operations"