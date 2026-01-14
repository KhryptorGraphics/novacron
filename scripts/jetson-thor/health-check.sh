#!/bin/bash

# NovaCron Health Check for Jetson Thor
# Comprehensive health verification for all components

set -e

# Configuration
POSTGRES_PORT="${POSTGRES_PORT:-15432}"
REDIS_PORT="${REDIS_PORT:-16379}"
QDRANT_PORT="${QDRANT_PORT:-16333}"
API_PORT="${API_PORT:-8090}"
FRONTEND_PORT="${FRONTEND_PORT:-8092}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

HEALTHY=0
UNHEALTHY=0

check_pass() {
    echo -e "  ${GREEN}✓${NC} $1"
    ((HEALTHY++))
}

check_fail() {
    echo -e "  ${RED}✗${NC} $1"
    ((UNHEALTHY++))
}

check_warn() {
    echo -e "  ${YELLOW}⚠${NC} $1"
}

echo "========================================"
echo "  NovaCron Health Check"
echo "========================================"
echo ""

# System checks
echo "${BLUE}System:${NC}"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    GPU_STATUS=$(nvidia-smi --query-gpu=gpu_name,memory.used,memory.total --format=csv,noheader 2>/dev/null)
    check_pass "GPU: $GPU_STATUS"
else
    check_warn "NVIDIA GPU tools not available"
fi

# Check disk space
DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}' | tr -d '%')
if [ "$DISK_USAGE" -lt 90 ]; then
    check_pass "Disk usage: ${DISK_USAGE}%"
else
    check_fail "Disk usage critical: ${DISK_USAGE}%"
fi

# Check memory
MEM_USAGE=$(free | awk 'NR==2 {printf "%.0f", $3/$2*100}')
if [ "$MEM_USAGE" -lt 90 ]; then
    check_pass "Memory usage: ${MEM_USAGE}%"
else
    check_fail "Memory usage critical: ${MEM_USAGE}%"
fi

echo ""

# Docker containers
echo "${BLUE}Docker Containers:${NC}"

# PostgreSQL
if docker ps --filter "name=novacron-postgres" --format "{{.Status}}" | grep -q "Up"; then
    if docker exec novacron-postgres pg_isready -U novacron &> /dev/null; then
        check_pass "PostgreSQL: Running and accepting connections"
    else
        check_fail "PostgreSQL: Running but not accepting connections"
    fi
else
    check_fail "PostgreSQL: Not running"
fi

# Redis
if docker ps --filter "name=novacron-redis" --format "{{.Status}}" | grep -q "Up"; then
    if docker exec novacron-redis redis-cli ping 2>/dev/null | grep -q PONG; then
        check_pass "Redis: Running and responding"
    else
        check_fail "Redis: Running but not responding"
    fi
else
    check_fail "Redis: Not running"
fi

# Qdrant
if docker ps --filter "name=novacron-qdrant" --format "{{.Status}}" | grep -q "Up"; then
    if curl -s "http://localhost:${QDRANT_PORT}/collections" > /dev/null 2>&1; then
        check_pass "Qdrant: Running and responding"
    else
        check_warn "Qdrant: Running but not responding to API"
    fi
else
    check_warn "Qdrant: Not running"
fi

echo ""

# Services
echo "${BLUE}NovaCron Services:${NC}"

# API Health
if curl -s --connect-timeout 2 "http://localhost:${API_PORT}/health" > /dev/null 2>&1; then
    API_STATUS=$(curl -s "http://localhost:${API_PORT}/health" | jq -r '.status // "ok"' 2>/dev/null || echo "ok")
    check_pass "API: Healthy (${API_STATUS})"
else
    check_fail "API: Not responding on port ${API_PORT}"
fi

# WebSocket endpoint
if curl -s --connect-timeout 2 -H "Connection: Upgrade" -H "Upgrade: websocket" "http://localhost:${API_PORT}/ws/metrics" -o /dev/null -w "%{http_code}" | grep -q "101\|400"; then
    check_pass "WebSocket: Endpoint available"
else
    check_warn "WebSocket: May not be available"
fi

# Frontend
if curl -s --connect-timeout 2 "http://localhost:${FRONTEND_PORT}" > /dev/null 2>&1; then
    check_pass "Frontend: Responding on port ${FRONTEND_PORT}"
else
    check_fail "Frontend: Not responding on port ${FRONTEND_PORT}"
fi

echo ""

# Database connectivity
echo "${BLUE}Database Connectivity:${NC}"

# Test PostgreSQL query
if docker exec novacron-postgres psql -U novacron -d novacron -c "SELECT 1;" &> /dev/null; then
    check_pass "PostgreSQL: Query successful"
else
    check_fail "PostgreSQL: Query failed"
fi

# Test Redis operations
if docker exec novacron-redis redis-cli SET health_check "ok" EX 5 &> /dev/null && \
   docker exec novacron-redis redis-cli GET health_check | grep -q "ok"; then
    check_pass "Redis: Read/Write successful"
else
    check_fail "Redis: Read/Write failed"
fi

echo ""

# Summary
echo "========================================"
echo "  Health Check Summary"
echo "========================================"
echo ""
echo -e "  Healthy: ${GREEN}${HEALTHY}${NC}"
echo -e "  Unhealthy: ${RED}${UNHEALTHY}${NC}"
echo ""

if [ $UNHEALTHY -eq 0 ]; then
    echo -e "${GREEN}All systems operational!${NC}"
    exit 0
else
    echo -e "${YELLOW}Some issues detected. Check logs for details.${NC}"
    exit 1
fi
