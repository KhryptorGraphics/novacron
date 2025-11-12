#!/bin/bash
# MCP Server Configuration Export
# This script installs all MCP servers configured in the source Claude Code instance
# Run this on the new server to replicate the MCP setup

set -e  # Exit on error

echo "🚀 Setting up MCP servers for Claude Code..."
echo ""

# ============================================================================
# REQUIRED: Core Orchestration Server
# ============================================================================
echo "📦 Installing REQUIRED MCP server: claude-flow"
echo "   Purpose: Core swarm coordination, SPARC workflows, agent orchestration"
claude mcp add claude-flow npx claude-flow@alpha mcp start
echo "✅ claude-flow installed"
echo ""

# ============================================================================
# OPTIONAL: Enhanced Coordination
# ============================================================================
echo "📦 Installing OPTIONAL MCP server: ruv-swarm"
echo "   Purpose: Enhanced swarm coordination and multi-agent workflows"
read -p "   Install ruv-swarm? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    claude mcp add ruv-swarm npx ruv-swarm mcp start
    echo "✅ ruv-swarm installed"
else
    echo "⏭️  Skipped ruv-swarm"
fi
echo ""

# ============================================================================
# OPTIONAL: Cloud Platform Features
# ============================================================================
echo "📦 Installing OPTIONAL MCP server: flow-nexus"
echo "   Purpose: Cloud-based sandboxes, neural AI, templates, GitHub integration"
echo "   Note: Requires registration at https://flow-nexus.ruv.io"
read -p "   Install flow-nexus? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    claude mcp add flow-nexus npx flow-nexus@latest mcp start
    echo "✅ flow-nexus installed"
    echo ""
    echo "⚠️  AUTHENTICATION REQUIRED:"
    echo "   Run one of the following to authenticate:"
    echo "   • npx flow-nexus@latest register  (new users)"
    echo "   • npx flow-nexus@latest login     (existing users)"
else
    echo "⏭️  Skipped flow-nexus"
fi
echo ""

# ============================================================================
# OPTIONAL: Issue Tracking
# ============================================================================
echo "📦 Installing OPTIONAL MCP server: beads"
echo "   Purpose: Lightweight issue tracking and project management"
read -p "   Install beads? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Note: Beads plugin is typically auto-installed via Claude marketplace
    # Manual installation if needed:
    # npm install -g @beads/beads-cli
    echo "ℹ️  Beads is available as a Claude Code plugin"
    echo "   Install from Claude marketplace or run: npm install -g @beads/beads-cli"
    echo "   Initialize in project: bd init"
else
    echo "⏭️  Skipped beads"
fi
echo ""

# ============================================================================
# OPTIONAL: Deep Research
# ============================================================================
echo "📦 Installing OPTIONAL MCP server: deep-research"
echo "   Purpose: Advanced web research and documentation generation"
read -p "   Install deep-research? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Note: This MCP server may require separate installation
    echo "ℹ️  Deep research MCP: pinkpixel-dev-deep-research-mcp"
    echo "   Install via: claude mcp add deep-research <installation-command>"
    echo "   (Check package documentation for exact command)"
else
    echo "⏭️  Skipped deep-research"
fi
echo ""

# ============================================================================
# Verification
# ============================================================================
echo "🔍 Verifying MCP server installation..."
echo ""
claude mcp list
echo ""

# ============================================================================
# Post-Installation Instructions
# ============================================================================
echo "✅ MCP server setup complete!"
echo ""
echo "📋 Next Steps:"
echo "   1. If you installed flow-nexus, authenticate now:"
echo "      npx flow-nexus@latest login"
echo ""
echo "   2. Copy project configuration files:"
echo "      • CLAUDE.md (project instructions)"
echo "      • .claude/ directory (custom commands)"
echo "      • package.json (if using npm scripts)"
echo ""
echo "   3. Initialize beads in your project (if installed):"
echo "      cd /path/to/your/project"
echo "      bd init"
echo ""
echo "   4. Test MCP servers:"
echo "      claude mcp status"
echo ""
echo "📖 Documentation:"
echo "   • Claude Flow: https://github.com/ruvnet/claude-flow"
echo "   • Flow Nexus: https://flow-nexus.ruv.io"
echo "   • Beads: Check Claude marketplace"
echo ""
echo "🎯 You're ready to use Claude Code with full MCP coordination!"
