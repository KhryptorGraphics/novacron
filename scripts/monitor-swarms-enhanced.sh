#!/bin/bash
# Enhanced Swarm Monitoring Script for NovaCron DWCP
# Monitors Claude-Flow swarms, builds, tests, neural models, and system health

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
REFRESH_INTERVAL=${1:-30}  # Default 30 seconds, can override with first argument
LOG_FILE="logs/swarm-monitor-$(date +%Y%m%d-%H%M%S).log"
ALERT_THRESHOLD_ERRORS=10
ALERT_THRESHOLD_WARNINGS=20

# Create logs directory if it doesn't exist
mkdir -p logs

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Header function
print_header() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  ${BLUE}NovaCron DWCP - Enhanced Swarm Monitor${NC}                    ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}  Refresh: ${REFRESH_INTERVAL}s | $(date '+%Y-%m-%d %H:%M:%S')                      ${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo
}

# Section header
section() {
    echo -e "\n${YELLOW}▶ $1${NC}"
    echo -e "${YELLOW}$(printf '─%.0s' {1..60})${NC}"
}

# Check swarm status
check_swarm_status() {
    section "CLAUDE-FLOW SWARM STATUS"

    if npx claude-flow@alpha swarm status 2>/dev/null; then
        echo -e "${GREEN}✓ Swarm active${NC}"

        # Get agent list
        echo -e "\n${CYAN}Active Agents:${NC}"
        npx claude-flow@alpha agent list 2>/dev/null || echo "No agent details available"

        # Get swarm metrics
        echo -e "\n${CYAN}Swarm Metrics:${NC}"
        npx claude-flow@alpha swarm monitor 2>/dev/null || echo "Monitoring data not available"

    else
        echo -e "${RED}✗ No active swarm${NC}"
        log "WARNING: No active swarm detected"
    fi
}

# Check build status
check_build_status() {
    section "BUILD STATUS"

    local build_output=$(go build ./backend/core/network/dwcp/... 2>&1)
    local build_exit=$?

    if [ $build_exit -eq 0 ]; then
        echo -e "${GREEN}✓ Build successful${NC}"
        log "Build successful"
    else
        echo -e "${RED}✗ Build failed${NC}"
        echo "$build_output" | tail -10
        log "ERROR: Build failed"

        # Count errors
        local error_count=$(echo "$build_output" | grep -c "error:" || true)
        if [ "$error_count" -gt "$ALERT_THRESHOLD_ERRORS" ]; then
            echo -e "${RED}⚠ ALERT: $error_count build errors detected!${NC}"
        fi
    fi
}

# Check test status
check_test_status() {
    section "TEST STATUS"

    local test_output=$(go test ./backend/core/network/dwcp/... -v 2>&1)
    local test_exit=$?

    if [ $test_exit -eq 0 ]; then
        echo -e "${GREEN}✓ All tests passed${NC}"

        # Extract test summary
        echo "$test_output" | grep -E "(PASS|FAIL|ok|FAIL)" | tail -10

        # Count test results
        local passed=$(echo "$test_output" | grep -c "PASS" || echo "0")
        local failed=$(echo "$test_output" | grep -c "FAIL" || echo "0")
        echo -e "\n${CYAN}Summary: ${GREEN}$passed passed${NC}, ${RED}$failed failed${NC}"

        log "Tests passed: $passed, failed: $failed"
    else
        echo -e "${RED}✗ Tests failed${NC}"
        echo "$test_output" | tail -15
        log "ERROR: Tests failed"
    fi
}

# Check neural models
check_neural_models() {
    section "NEURAL MODELS"

    local models_found=0

    # Check for ONNX models
    if ls backend/core/network/dwcp/*/models/*.onnx 2>/dev/null | head -5; then
        models_found=1
        echo -e "\n${CYAN}ONNX Models:${NC}"
        ls -lh backend/core/network/dwcp/*/models/*.onnx 2>/dev/null | tail -5
    fi

    # Check for pickle models
    if ls backend/core/network/dwcp/*/models/*.pkl 2>/dev/null | head -5; then
        models_found=1
        echo -e "\n${CYAN}Pickle Models:${NC}"
        ls -lh backend/core/network/dwcp/*/models/*.pkl 2>/dev/null | tail -5
    fi

    # Check for PyTorch models
    if ls backend/core/network/dwcp/*/models/*.pt 2>/dev/null | head -5; then
        models_found=1
        echo -e "\n${CYAN}PyTorch Models:${NC}"
        ls -lh backend/core/network/dwcp/*/models/*.pt 2>/dev/null | tail -5
    fi

    if [ $models_found -eq 0 ]; then
        echo -e "${YELLOW}⚠ No neural models found${NC}"
        log "WARNING: No neural models detected"
    else
        echo -e "\n${GREEN}✓ Neural models detected${NC}"

        # Check model freshness
        local latest_model=$(find backend/core/network/dwcp/*/models/ -type f \( -name "*.onnx" -o -name "*.pkl" -o -name "*.pt" \) -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
        if [ -n "$latest_model" ]; then
            echo -e "${CYAN}Latest model:${NC} $latest_model"
            echo -e "${CYAN}Modified:${NC} $(stat -c %y "$latest_model" 2>/dev/null || echo "unknown")"
        fi
    fi
}

# Check memory usage
check_memory_status() {
    section "MEMORY & COORDINATION"

    # Check Claude-Flow memory
    echo -e "${CYAN}Swarm Memory:${NC}"
    npx claude-flow@alpha memory list --namespace "swarm" 2>/dev/null | head -10 || echo "No swarm memory found"

    # Check session persistence
    echo -e "\n${CYAN}Active Sessions:${NC}"
    npx claude-flow@alpha hooks session-status 2>/dev/null || echo "No active sessions"
}

# Check performance metrics
check_performance() {
    section "PERFORMANCE METRICS"

    # Check Claude-Flow performance
    if [ -f ".claude-flow/metrics/performance.json" ]; then
        echo -e "${CYAN}Latest Performance Data:${NC}"
        cat .claude-flow/metrics/performance.json | jq -r '.latest // empty' 2>/dev/null | head -10 || cat .claude-flow/metrics/performance.json | head -10
    fi

    # Check system metrics
    if [ -f ".claude-flow/metrics/system-metrics.json" ]; then
        echo -e "\n${CYAN}System Metrics:${NC}"
        cat .claude-flow/metrics/system-metrics.json | jq -r '.current // empty' 2>/dev/null | head -5 || cat .claude-flow/metrics/system-metrics.json | head -5
    fi
}

# Check Git status
check_git_status() {
    section "GIT STATUS"

    local modified=$(git status --porcelain | wc -l)
    local branch=$(git branch --show-current)

    echo -e "${CYAN}Branch:${NC} $branch"
    echo -e "${CYAN}Modified files:${NC} $modified"

    if [ $modified -gt 0 ]; then
        echo -e "\n${YELLOW}Recent changes:${NC}"
        git status --short | head -10
    fi
}

# Health check summary
health_summary() {
    section "HEALTH SUMMARY"

    local status="HEALTHY"
    local issues=0

    # Check build
    if ! go build ./backend/core/network/dwcp/... 2>/dev/null; then
        echo -e "${RED}✗ Build: FAILED${NC}"
        status="DEGRADED"
        ((issues++))
    else
        echo -e "${GREEN}✓ Build: OK${NC}"
    fi

    # Check tests
    if ! go test ./backend/core/network/dwcp/... 2>/dev/null; then
        echo -e "${RED}✗ Tests: FAILED${NC}"
        status="DEGRADED"
        ((issues++))
    else
        echo -e "${GREEN}✓ Tests: OK${NC}"
    fi

    # Check swarm
    if npx claude-flow@alpha swarm status 2>/dev/null | grep -q "active"; then
        echo -e "${GREEN}✓ Swarm: ACTIVE${NC}"
    else
        echo -e "${YELLOW}⚠ Swarm: INACTIVE${NC}"
    fi

    # Check models
    if ls backend/core/network/dwcp/*/models/*.{onnx,pkl,pt} 2>/dev/null | grep -q .; then
        echo -e "${GREEN}✓ Neural Models: PRESENT${NC}"
    else
        echo -e "${YELLOW}⚠ Neural Models: MISSING${NC}"
    fi

    echo
    if [ "$status" = "HEALTHY" ]; then
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}   SYSTEM STATUS: HEALTHY ✓${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    else
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${YELLOW}   SYSTEM STATUS: $status ($issues issues)${NC}"
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    fi
}

# Main monitoring loop
main() {
    log "Starting enhanced swarm monitoring (refresh: ${REFRESH_INTERVAL}s)"

    while true; do
        clear
        print_header
        check_swarm_status
        check_build_status
        check_test_status
        check_neural_models
        check_memory_status
        check_performance
        check_git_status
        health_summary

        echo
        echo -e "${CYAN}Press Ctrl+C to stop monitoring${NC}"
        echo -e "${CYAN}Log file: $LOG_FILE${NC}"

        sleep "$REFRESH_INTERVAL"
    done
}

# Run main monitoring
main
