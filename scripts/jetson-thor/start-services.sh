#!/bin/bash

# NovaCron Service Starter for Jetson Thor
# Starts all NovaCron services in the correct order

set -e

# Configuration
NOVACRON_HOME="${NOVACRON_HOME:-/home/kp/repos/novacron}"
POSTGRES_PORT="${POSTGRES_PORT:-15432}"
REDIS_PORT="${REDIS_PORT:-16379}"
API_PORT="${API_PORT:-8090}"
FRONTEND_PORT="${FRONTEND_PORT:-8092}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "========================================"
echo "  Starting NovaCron Services"
echo "========================================"
echo ""

# Start Docker containers
start_containers() {
    log_info "Starting Docker containers..."

    docker start novacron-postgres 2>/dev/null || log_warn "PostgreSQL container not found"
    docker start novacron-redis 2>/dev/null || log_warn "Redis container not found"
    docker start novacron-qdrant 2>/dev/null || log_warn "Qdrant container not found"

    # Wait for containers to be ready
    log_info "Waiting for databases to be ready..."
    sleep 3

    # Check PostgreSQL
    until docker exec novacron-postgres pg_isready -U novacron 2>/dev/null; do
        log_info "Waiting for PostgreSQL..."
        sleep 1
    done
    log_success "PostgreSQL is ready"

    # Check Redis
    until docker exec novacron-redis redis-cli ping 2>/dev/null | grep -q PONG; do
        log_info "Waiting for Redis..."
        sleep 1
    done
    log_success "Redis is ready"
}

# Start backend
start_backend() {
    log_info "Starting backend API..."

    cd "${NOVACRON_HOME}/backend"

    # Load environment
    if [ -f "${NOVACRON_HOME}/.env.local" ]; then
        set -a
        source "${NOVACRON_HOME}/.env.local"
        set +a
    fi

    # Check if already running
    if lsof -i:${API_PORT} -t &> /dev/null; then
        log_warn "Backend already running on port ${API_PORT}"
    else
        # Start in background
        nohup go run cmd/api-server/main.go > /tmp/novacron-api.log 2>&1 &
        echo $! > /tmp/novacron-api.pid
        log_success "Backend started (PID: $(cat /tmp/novacron-api.pid))"
    fi
}

# Start frontend
start_frontend() {
    log_info "Starting frontend..."

    cd "${NOVACRON_HOME}/frontend"

    # Check if already running
    if lsof -i:${FRONTEND_PORT} -t &> /dev/null; then
        log_warn "Frontend already running on port ${FRONTEND_PORT}"
    else
        # Start in background
        PORT=${FRONTEND_PORT} nohup npm run start > /tmp/novacron-frontend.log 2>&1 &
        echo $! > /tmp/novacron-frontend.pid
        log_success "Frontend started (PID: $(cat /tmp/novacron-frontend.pid))"
    fi
}

# Health check
health_check() {
    log_info "Running health checks..."
    sleep 5

    # Check API
    if curl -s "http://localhost:${API_PORT}/health" > /dev/null 2>&1; then
        log_success "API is healthy"
    else
        log_warn "API not responding yet (may still be starting)"
    fi

    # Check Frontend
    if curl -s "http://localhost:${FRONTEND_PORT}" > /dev/null 2>&1; then
        log_success "Frontend is healthy"
    else
        log_warn "Frontend not responding yet (may still be starting)"
    fi
}

# Print status
print_status() {
    echo ""
    echo "========================================"
    echo "  NovaCron Service Status"
    echo "========================================"
    echo ""
    echo "Docker Containers:"
    docker ps --filter "name=novacron" --format "  {{.Names}}: {{.Status}}"
    echo ""
    echo "Processes:"
    [ -f /tmp/novacron-api.pid ] && echo "  API: PID $(cat /tmp/novacron-api.pid)"
    [ -f /tmp/novacron-frontend.pid ] && echo "  Frontend: PID $(cat /tmp/novacron-frontend.pid)"
    echo ""
    echo "Access:"
    echo "  - API: http://localhost:${API_PORT}"
    echo "  - Frontend: http://localhost:${FRONTEND_PORT}"
    echo ""
    echo "Logs:"
    echo "  - API: tail -f /tmp/novacron-api.log"
    echo "  - Frontend: tail -f /tmp/novacron-frontend.log"
    echo ""
}

# Main
main() {
    start_containers
    start_backend
    start_frontend
    health_check
    print_status
}

case "${1:-all}" in
    containers)
        start_containers
        ;;
    backend)
        start_backend
        ;;
    frontend)
        start_frontend
        ;;
    status)
        print_status
        ;;
    all)
        main
        ;;
    *)
        echo "Usage: $0 [containers|backend|frontend|status|all]"
        exit 1
        ;;
esac
