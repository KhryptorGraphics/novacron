#!/bin/bash
# DWCP Phase 1 Deployment Script
# Safely deploys DWCP to staging environment with validation and rollback capability

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/configs"
BACKEND_DIR="$PROJECT_ROOT/backend"
LOG_FILE="/tmp/dwcp-deploy-$(date +%Y%m%d-%H%M%S).log"

# Environment (staging or production)
ENVIRONMENT="${1:-staging}"
CONFIG_FILE="$CONFIG_DIR/dwcp.$ENVIRONMENT.yaml"
BASE_CONFIG="$CONFIG_DIR/dwcp.yaml"

# Functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $*" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $*" | tee -a "$LOG_FILE"
}

check_prerequisites() {
    log "Checking prerequisites..."

    # Check Go version
    if ! command -v go &> /dev/null; then
        error "Go is not installed"
        exit 1
    fi

    GO_VERSION=$(go version | awk '{print $3}' | sed 's/go//')
    REQUIRED_VERSION="1.21"

    if ! printf '%s\n%s\n' "$REQUIRED_VERSION" "$GO_VERSION" | sort -V -C; then
        error "Go version $GO_VERSION is less than required $REQUIRED_VERSION"
        exit 1
    fi
    log "✓ Go version $GO_VERSION OK"

    # Check configuration files
    if [[ ! -f "$BASE_CONFIG" ]]; then
        error "Base configuration not found: $BASE_CONFIG"
        exit 1
    fi
    log "✓ Base configuration found"

    if [[ ! -f "$CONFIG_FILE" ]]; then
        error "Environment configuration not found: $CONFIG_FILE"
        exit 1
    fi
    log "✓ Environment configuration found"

    # Check RDMA hardware (optional)
    if command -v ibv_devices &> /dev/null; then
        RDMA_DEVICES=$(ibv_devices | grep -c "hca_id" || true)
        if [[ $RDMA_DEVICES -gt 0 ]]; then
            log "✓ RDMA hardware detected: $RDMA_DEVICES device(s)"
        else
            warn "No RDMA hardware detected (optional)"
        fi
    else
        warn "RDMA tools not installed (optional)"
    fi

    # Check Prometheus
    if command -v prometheus &> /dev/null || pgrep -x prometheus &> /dev/null; then
        log "✓ Prometheus available"
    else
        warn "Prometheus not running (metrics will be unavailable)"
    fi

    # Validate YAML syntax
    if command -v yamllint &> /dev/null; then
        if yamllint -d relaxed "$BASE_CONFIG" "$CONFIG_FILE"; then
            log "✓ Configuration syntax valid"
        else
            error "Configuration syntax validation failed"
            exit 1
        fi
    else
        warn "yamllint not installed, skipping syntax validation"
    fi
}

backup_current_state() {
    log "Backing up current state..."

    BACKUP_DIR="/tmp/dwcp-backup-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$BACKUP_DIR"

    # Backup configuration
    if [[ -d "$CONFIG_DIR" ]]; then
        cp -r "$CONFIG_DIR" "$BACKUP_DIR/"
        log "✓ Configuration backed up to $BACKUP_DIR"
    fi

    # Backup binary if exists
    if [[ -f "$BACKEND_DIR/bin/api-server" ]]; then
        cp "$BACKEND_DIR/bin/api-server" "$BACKUP_DIR/"
        log "✓ Binary backed up"
    fi

    echo "$BACKUP_DIR" > /tmp/dwcp-latest-backup
    log "✓ Backup complete: $BACKUP_DIR"
}

build_binaries() {
    log "Building DWCP binaries..."

    cd "$BACKEND_DIR"

    # Clean previous builds
    rm -rf bin/
    mkdir -p bin/

    # Build with optimizations
    log "Building api-server..."
    go build -o bin/api-server \
        -ldflags "-X main.Version=1.0.0-dwcp-phase1 -X main.BuildTime=$(date -u +%Y%m%dT%H%M%S)" \
        ./cmd/api-server/main.go

    if [[ ! -f bin/api-server ]]; then
        error "Build failed: binary not created"
        exit 1
    fi

    log "✓ Build successful"
}

run_tests() {
    log "Running test suite..."

    cd "$BACKEND_DIR"

    # Run unit tests
    log "Running unit tests..."
    if go test -v -race -coverprofile=coverage.out ./...; then
        log "✓ Unit tests passed"
    else
        error "Unit tests failed"
        exit 1
    fi

    # Run DWCP-specific tests
    log "Running DWCP tests..."
    if go test -v -race ./core/network/dwcp/...; then
        log "✓ DWCP tests passed"
    else
        error "DWCP tests failed"
        exit 1
    fi

    # Generate coverage report
    COVERAGE=$(go tool cover -func=coverage.out | grep total | awk '{print $3}')
    log "✓ Test coverage: $COVERAGE"
}

run_benchmarks() {
    log "Running benchmarks..."

    cd "$BACKEND_DIR"

    # Run benchmarks
    go test -bench=. -benchmem -run=^$ ./core/network/dwcp/... | tee /tmp/dwcp-benchmarks.txt

    log "✓ Benchmarks complete (results in /tmp/dwcp-benchmarks.txt)"
}

deploy_to_environment() {
    log "Deploying to $ENVIRONMENT environment..."

    # Copy configuration
    log "Installing configuration..."
    mkdir -p /etc/dwcp/
    cp "$BASE_CONFIG" /etc/dwcp/dwcp.yaml
    cp "$CONFIG_FILE" /etc/dwcp/dwcp.$ENVIRONMENT.yaml

    # Copy binary
    log "Installing binary..."
    mkdir -p /usr/local/bin/
    cp "$BACKEND_DIR/bin/api-server" /usr/local/bin/dwcp-api-server
    chmod +x /usr/local/bin/dwcp-api-server

    log "✓ Deployment complete"
}

health_check() {
    log "Running health checks..."

    # Start service in background
    log "Starting DWCP service..."
    DWCP_CONFIG="/etc/dwcp/dwcp.$ENVIRONMENT.yaml" \
        /usr/local/bin/dwcp-api-server &

    SERVICE_PID=$!
    echo "$SERVICE_PID" > /tmp/dwcp-service.pid

    # Wait for service to start
    sleep 5

    # Check if process is running
    if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
        error "Service failed to start"
        return 1
    fi
    log "✓ Service started (PID: $SERVICE_PID)"

    # Check health endpoint
    log "Checking health endpoint..."
    for i in {1..30}; do
        if curl -sf http://localhost:8080/health > /dev/null; then
            log "✓ Health check passed"
            return 0
        fi
        sleep 1
    done

    error "Health check failed after 30 seconds"
    return 1
}

verify_metrics() {
    log "Verifying metrics collection..."

    # Check Prometheus metrics endpoint
    if curl -sf http://localhost:9090/metrics | grep -q "dwcp_"; then
        log "✓ DWCP metrics available"
    else
        warn "DWCP metrics not found (may take a few moments)"
    fi
}

rollback() {
    error "Deployment failed, initiating rollback..."

    # Stop current service
    if [[ -f /tmp/dwcp-service.pid ]]; then
        SERVICE_PID=$(cat /tmp/dwcp-service.pid)
        if kill -0 "$SERVICE_PID" 2>/dev/null; then
            kill "$SERVICE_PID"
            log "✓ Service stopped"
        fi
    fi

    # Restore from backup
    if [[ -f /tmp/dwcp-latest-backup ]]; then
        BACKUP_DIR=$(cat /tmp/dwcp-latest-backup)
        if [[ -d "$BACKUP_DIR" ]]; then
            log "Restoring from backup: $BACKUP_DIR"

            # Restore configuration
            if [[ -d "$BACKUP_DIR/configs" ]]; then
                cp -r "$BACKUP_DIR/configs/"* "$CONFIG_DIR/"
                log "✓ Configuration restored"
            fi

            # Restore binary
            if [[ -f "$BACKUP_DIR/api-server" ]]; then
                cp "$BACKUP_DIR/api-server" /usr/local/bin/dwcp-api-server
                log "✓ Binary restored"
            fi

            log "✓ Rollback complete"
        else
            error "Backup directory not found: $BACKUP_DIR"
        fi
    else
        error "No backup found"
    fi
}

cleanup() {
    log "Cleaning up temporary files..."

    # Remove old backups (keep last 5)
    find /tmp -maxdepth 1 -name "dwcp-backup-*" -type d | sort -r | tail -n +6 | xargs rm -rf

    log "✓ Cleanup complete"
}

# Main deployment flow
main() {
    log "=== DWCP Phase 1 Deployment Starting ==="
    log "Environment: $ENVIRONMENT"
    log "Log file: $LOG_FILE"

    # Set trap for errors
    trap rollback ERR

    # Deployment steps
    check_prerequisites
    backup_current_state
    build_binaries
    run_tests
    run_benchmarks
    deploy_to_environment

    if health_check; then
        verify_metrics
        cleanup

        log "=== DWCP Phase 1 Deployment Successful ==="
        log "Log file: $LOG_FILE"
        log "Backup: $(cat /tmp/dwcp-latest-backup)"
        log ""
        log "Next steps:"
        log "1. Monitor metrics at http://localhost:9090/metrics"
        log "2. Check logs: tail -f /var/log/dwcp.log"
        log "3. Run validation: $SCRIPT_DIR/validate-dwcp.sh"

        exit 0
    else
        error "Health checks failed"
        exit 1
    fi
}

# Run main
main "$@"
