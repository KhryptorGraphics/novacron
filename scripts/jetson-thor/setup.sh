#!/bin/bash

# NovaCron Jetson Thor Setup Script
# Platform: NVIDIA Jetson Thor (Tegra)
# JetPack: 7.4
# CUDA: 13
# TensorRT: Required

set -e

# Configuration
NOVACRON_HOME="${NOVACRON_HOME:-/home/kp/repos/novacron}"
POSTGRES_PORT="${POSTGRES_PORT:-15432}"
REDIS_PORT="${REDIS_PORT:-16379}"
QDRANT_PORT="${QDRANT_PORT:-16333}"
API_PORT="${API_PORT:-8090}"
WS_PORT="${WS_PORT:-8091}"
FRONTEND_PORT="${FRONTEND_PORT:-8092}"

# Security: Generate secure passwords if not provided
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-$(openssl rand -hex 16)}"
SMTP_PASSWORD="${SMTP_PASSWORD:-}"

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
echo "  NovaCron Jetson Thor Setup"
echo "========================================"
echo ""

# Check if running on Jetson
check_platform() {
    log_info "Checking platform..."

    if [ -f /etc/nv_tegra_release ]; then
        log_success "Running on NVIDIA Tegra platform"
        cat /etc/nv_tegra_release
    else
        log_warn "Not running on Tegra platform - proceeding anyway"
    fi

    # Check CUDA
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        log_success "CUDA found: $CUDA_VERSION"
    else
        log_error "CUDA not found - required for NovaCron AI features"
        log_info "Install JetPack 7.4+ with CUDA support"
    fi

    # Check TensorRT
    if [ -d "/usr/lib/aarch64-linux-gnu/libnvinfer.so" ] || ldconfig -p | grep -q libnvinfer; then
        log_success "TensorRT found"
    else
        log_warn "TensorRT not found - some AI features may not work"
    fi
}

# Check and install dependencies
check_dependencies() {
    log_info "Checking dependencies..."

    # Docker
    if command -v docker &> /dev/null; then
        log_success "Docker found: $(docker --version)"
    else
        log_warn "Docker not found - installing..."
        curl -fsSL https://get.docker.com | sh
        sudo usermod -aG docker $USER
    fi

    # Docker Compose
    if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
        log_success "Docker Compose found"
    else
        log_info "Installing Docker Compose..."
        sudo apt-get update && sudo apt-get install -y docker-compose-plugin
    fi

    # Go
    if command -v go &> /dev/null; then
        GO_VERSION=$(go version | awk '{print $3}')
        log_success "Go found: $GO_VERSION"
    else
        log_warn "Go not found - installing Go 1.24..."
        wget https://go.dev/dl/go1.24.linux-arm64.tar.gz
        sudo tar -C /usr/local -xzf go1.24.linux-arm64.tar.gz
        export PATH=$PATH:/usr/local/go/bin
        echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
    fi

    # Node.js
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        log_success "Node.js found: $NODE_VERSION"
    else
        log_info "Installing Node.js..."
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        sudo apt-get install -y nodejs
    fi

    # Other tools
    sudo apt-get install -y curl wget git jq htop net-tools
}

# Setup Docker containers for dependencies
setup_containers() {
    log_info "Setting up Docker containers..."

    # Check port availability
    check_port() {
        local port=$1
        local service=$2
        if lsof -i:$port -t &> /dev/null; then
            log_warn "Port $port is in use - $service may conflict"
            return 1
        fi
        return 0
    }

    # PostgreSQL
    log_info "Setting up PostgreSQL on port $POSTGRES_PORT..."
    if docker ps -a | grep -q novacron-postgres; then
        log_info "PostgreSQL container exists, starting..."
        docker start novacron-postgres
    else
        check_port $POSTGRES_PORT "PostgreSQL"
        docker run -d --restart=always --name novacron-postgres \
            -p ${POSTGRES_PORT}:5432 \
            -e POSTGRES_USER=novacron \
            -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
            -e POSTGRES_DB=novacron \
            -v novacron_postgres_data:/var/lib/postgresql/data \
            postgres:15-alpine
        log_success "PostgreSQL container created"
    fi

    # Redis
    log_info "Setting up Redis on port $REDIS_PORT..."
    if docker ps -a | grep -q novacron-redis; then
        log_info "Redis container exists, starting..."
        docker start novacron-redis
    else
        check_port $REDIS_PORT "Redis"
        docker run -d --restart=always --name novacron-redis \
            -p ${REDIS_PORT}:6379 \
            -v novacron_redis_data:/data \
            redis:7-alpine redis-server --appendonly yes
        log_success "Redis container created"
    fi

    # Qdrant (vector database for AI)
    log_info "Setting up Qdrant on port $QDRANT_PORT..."
    if docker ps -a | grep -q novacron-qdrant; then
        log_info "Qdrant container exists, starting..."
        docker start novacron-qdrant
    else
        check_port $QDRANT_PORT "Qdrant"
        docker run -d --restart=always --name novacron-qdrant \
            -p ${QDRANT_PORT}:6333 \
            -v novacron_qdrant_data:/qdrant/storage \
            qdrant/qdrant
        log_success "Qdrant container created"
    fi

    # Wait for containers to be ready
    log_info "Waiting for containers to be ready..."
    sleep 5

    # Verify containers
    docker ps --filter "name=novacron" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

# Create environment file
create_env_file() {
    log_info "Creating environment configuration..."

    ENV_FILE="${NOVACRON_HOME}/.env.local"

    cat > $ENV_FILE << EOF
# NovaCron Local Environment - Jetson Thor
# Generated: $(date)

# Database Configuration (Non-standard ports to avoid conflicts)
DATABASE_URL=postgresql://novacron:${POSTGRES_PASSWORD}@localhost:${POSTGRES_PORT}/novacron
POSTGRES_HOST=localhost
POSTGRES_PORT=${POSTGRES_PORT}
POSTGRES_USER=novacron
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
POSTGRES_DB=novacron

# Redis Configuration
REDIS_URL=redis://localhost:${REDIS_PORT}
REDIS_HOST=localhost
REDIS_PORT=${REDIS_PORT}

# Qdrant Configuration
QDRANT_URL=http://localhost:${QDRANT_PORT}
QDRANT_HOST=localhost
QDRANT_PORT=${QDRANT_PORT}

# API Configuration
API_HOST=0.0.0.0
API_PORT=${API_PORT}
WS_PORT=${WS_PORT}
FRONTEND_PORT=${FRONTEND_PORT}

# JWT Configuration
JWT_SECRET=$(openssl rand -hex 32)
JWT_EXPIRY=24h
JWT_REFRESH_EXPIRY=168h

# Security
CORS_ORIGINS=http://localhost:${FRONTEND_PORT},http://127.0.0.1:${FRONTEND_PORT}
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Email Configuration
SMTP_HOST=smtp.giggahost.com
SMTP_PORT=587
SMTP_FROM=notifications@giggahost.com
SMTP_PASSWORD=${SMTP_PASSWORD}

# AI/ML Configuration
CUDA_VISIBLE_DEVICES=0
TF_FORCE_GPU_ALLOW_GROWTH=true
TENSORRT_ENABLED=true

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true

# Feature Flags
ENABLE_2FA=true
ENABLE_OAUTH=true
ENABLE_AUDIT_LOG=true
EOF

    chmod 600 $ENV_FILE
    log_success "Environment file created: $ENV_FILE"
}

# Build backend
build_backend() {
    log_info "Building backend..."

    cd "${NOVACRON_HOME}/backend"

    # Tidy dependencies
    go mod tidy

    # Build core
    log_info "Building core module..."
    cd "${NOVACRON_HOME}/backend/core"
    go build -v ./auth/...

    log_success "Backend built successfully"
}

# Build frontend
build_frontend() {
    log_info "Building frontend..."

    cd "${NOVACRON_HOME}/frontend"

    # Install dependencies
    npm install

    # Build for production
    npm run build

    log_success "Frontend built successfully"
}

# Create systemd services
create_services() {
    log_info "Creating systemd services..."

    # Backend API service
    sudo tee /etc/systemd/system/novacron-api.service > /dev/null << EOF
[Unit]
Description=NovaCron API Server
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=$USER
WorkingDirectory=${NOVACRON_HOME}/backend
Environment=PATH=/usr/local/go/bin:/usr/bin:/bin
EnvironmentFile=${NOVACRON_HOME}/.env.local
ExecStart=/usr/local/go/bin/go run cmd/api-server/main.go
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    # Frontend service
    sudo tee /etc/systemd/system/novacron-frontend.service > /dev/null << EOF
[Unit]
Description=NovaCron Frontend
After=network.target novacron-api.service

[Service]
Type=simple
User=$USER
WorkingDirectory=${NOVACRON_HOME}/frontend
EnvironmentFile=${NOVACRON_HOME}/.env.local
ExecStart=/usr/bin/npm run start
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd
    sudo systemctl daemon-reload

    log_success "Systemd services created"
    log_info "To start: sudo systemctl start novacron-api novacron-frontend"
    log_info "To enable on boot: sudo systemctl enable novacron-api novacron-frontend"
}

# Print summary
print_summary() {
    echo ""
    echo "========================================"
    echo "  NovaCron Setup Complete"
    echo "========================================"
    echo ""
    echo "Service Ports:"
    echo "  - PostgreSQL: localhost:${POSTGRES_PORT}"
    echo "  - Redis: localhost:${REDIS_PORT}"
    echo "  - Qdrant: localhost:${QDRANT_PORT}"
    echo "  - API: localhost:${API_PORT}"
    echo "  - WebSocket: localhost:${WS_PORT}"
    echo "  - Frontend: localhost:${FRONTEND_PORT}"
    echo ""
    echo "Quick Commands:"
    echo "  - Start all: docker start novacron-postgres novacron-redis novacron-qdrant"
    echo "  - Check status: docker ps --filter 'name=novacron'"
    echo "  - View logs: docker logs -f novacron-postgres"
    echo ""
    echo "To start NovaCron:"
    echo "  1. cd ${NOVACRON_HOME}/backend && make core-serve"
    echo "  2. cd ${NOVACRON_HOME}/frontend && npm run dev"
    echo ""
    echo "Or use systemd services:"
    echo "  sudo systemctl start novacron-api novacron-frontend"
    echo ""
}

# Main execution
main() {
    check_platform
    check_dependencies
    setup_containers
    create_env_file
    build_backend
    build_frontend
    create_services
    print_summary
}

# Run with option to skip certain steps
case "${1:-all}" in
    platform)
        check_platform
        ;;
    deps)
        check_dependencies
        ;;
    containers)
        setup_containers
        ;;
    env)
        create_env_file
        ;;
    backend)
        build_backend
        ;;
    frontend)
        build_frontend
        ;;
    services)
        create_services
        ;;
    all)
        main
        ;;
    *)
        echo "Usage: $0 [platform|deps|containers|env|backend|frontend|services|all]"
        exit 1
        ;;
esac
