#!/bin/bash
# NovaCron Swarm & Build Monitoring Script
# Monitors swarm status, build progress, and test results every 30 seconds

clear

echo "🚀 Starting NovaCron Swarm Monitor..."
echo "   Press Ctrl+C to stop"
echo ""
sleep 2

watch -n 30 '
clear
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           NOVACRON SWARM & BUILD MONITOR                       ║"
echo "║           Updated: $(date +"%Y-%m-%d %H:%M:%S")                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

echo "📊 === SWARM STATUS ==="
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
npx claude-flow@alpha swarm status 2>&1 || echo "⚠️  No active swarm detected"
echo ""

echo "🔨 === BUILD STATUS (DWCP) ==="
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd /home/kp/repos/novacron
go build ./backend/core/network/dwcp/... 2>&1 | tail -10 || echo "✅ Build successful"
echo ""

echo "🧪 === TEST STATUS (DWCP) ==="
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
go test ./backend/core/network/dwcp/... -v 2>&1 | tail -15 || echo "⚠️  Tests failed or no tests found"
echo ""

echo "💾 === MEMORY & PERFORMANCE ==="
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Go processes: $(pgrep -c go || echo "0")"
echo "Memory usage: $(ps aux | grep "[g]o" | awk "{sum+=\$4} END {print sum}") %"
echo ""

echo "🔄 Refreshing in 30 seconds... (Ctrl+C to stop)"
'
