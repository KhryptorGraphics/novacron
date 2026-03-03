#!/bin/bash

# NovaCron Service Stopper for Jetson Thor
# Gracefully stops all NovaCron services

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

echo "========================================"
echo "  Stopping NovaCron Services"
echo "========================================"
echo ""

# Stop frontend
stop_frontend() {
    log_info "Stopping frontend..."
    if [ -f /tmp/novacron-frontend.pid ]; then
        kill $(cat /tmp/novacron-frontend.pid) 2>/dev/null && log_success "Frontend stopped" || log_warn "Frontend not running"
        rm -f /tmp/novacron-frontend.pid
    else
        # Try to find by port
        pkill -f "next.*start" 2>/dev/null || true
    fi
}

# Stop backend
stop_backend() {
    log_info "Stopping backend..."
    if [ -f /tmp/novacron-api.pid ]; then
        kill $(cat /tmp/novacron-api.pid) 2>/dev/null && log_success "Backend stopped" || log_warn "Backend not running"
        rm -f /tmp/novacron-api.pid
    else
        # Try to find by port
        pkill -f "go run.*api-server" 2>/dev/null || true
    fi
}

# Stop containers
stop_containers() {
    log_info "Stopping Docker containers..."
    docker stop novacron-postgres 2>/dev/null && log_success "PostgreSQL stopped" || log_warn "PostgreSQL not running"
    docker stop novacron-redis 2>/dev/null && log_success "Redis stopped" || log_warn "Redis not running"
    docker stop novacron-qdrant 2>/dev/null && log_success "Qdrant stopped" || log_warn "Qdrant not running"
}

# Main
case "${1:-all}" in
    frontend)
        stop_frontend
        ;;
    backend)
        stop_backend
        ;;
    containers)
        stop_containers
        ;;
    all)
        stop_frontend
        stop_backend
        stop_containers
        ;;
    *)
        echo "Usage: $0 [frontend|backend|containers|all]"
        exit 1
        ;;
esac

echo ""
log_success "Services stopped"
