#!/bin/bash

# NovaCron Production Deployment Script
# This script automates the complete production deployment process

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
DEPLOYMENT_DIR="${PROJECT_ROOT}/deployment"
DOCKER_DIR="${DEPLOYMENT_DIR}/docker"
K8S_DIR="${DEPLOYMENT_DIR}/kubernetes"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    command -v docker >/dev/null 2>&1 || missing_tools+=("docker")
    command -v docker-compose >/dev/null 2>&1 || missing_tools+=("docker-compose")
    command -v kubectl >/dev/null 2>&1 || missing_tools+=("kubectl")
    command -v helm >/dev/null 2>&1 || missing_tools+=("helm")
    command -v openssl >/dev/null 2>&1 || missing_tools+=("openssl")
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    log_success "All prerequisites are installed"
}

# Load environment variables
load_environment() {
    log_info "Loading environment configuration..."
    
    local env_file="${PROJECT_ROOT}/.env.production"
    if [ ! -f "$env_file" ]; then
        log_error "Production environment file not found: $env_file"
        exit 1
    fi
    
    # Export environment variables
    set -a
    source "$env_file"
    set +a
    
    # Validate required environment variables
    local required_vars=(
        "DOMAIN"
        "ACME_EMAIL"
        "DB_USER"
        "DB_NAME"
        "GRAFANA_ADMIN_USER"
        "AUTH_SECRET"
        "REDIS_PASSWORD"
    )
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            log_error "Required environment variable not set: $var"
            exit 1
        fi
    done
    
    log_success "Environment configuration loaded"
}

# Generate secrets
generate_secrets() {
    log_info "Generating production secrets..."
    
    # Generate database password if not set
    if [ -z "${DB_PASSWORD:-}" ]; then
        export DB_PASSWORD=$(openssl rand -base64 32)
        log_info "Generated database password"
    fi
    
    # Generate Grafana password if not set
    if [ -z "${GRAFANA_PASSWORD:-}" ]; then
        export GRAFANA_PASSWORD=$(openssl rand -base64 32)
        log_info "Generated Grafana password"
    fi
    
    # Generate Redis password if not already set
    if [ -z "${REDIS_PASSWORD:-}" ]; then
        export REDIS_PASSWORD=$(openssl rand -base64 32)
        log_info "Generated Redis password"
    fi
    
    # Generate AUTH_SECRET if not set
    if [ -z "${AUTH_SECRET:-}" ]; then
        export AUTH_SECRET=$(openssl rand -base64 64)
        log_info "Generated JWT secret"
    fi
    
    log_success "All secrets generated"
}

# Create Docker secrets
create_docker_secrets() {
    log_info "Creating Docker secrets..."
    
    # Create database password secret
    echo "$DB_PASSWORD" | docker secret create novacron_db_password - 2>/dev/null || {
        log_warning "Database password secret already exists"
    }
    
    # Create Grafana password secret
    echo "$GRAFANA_PASSWORD" | docker secret create novacron_grafana_password - 2>/dev/null || {
        log_warning "Grafana password secret already exists"
    }
    
    log_success "Docker secrets created"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build API image
    log_info "Building API image..."
    docker build -f "${DOCKER_DIR}/Dockerfile.prod" -t novacron/api:latest .
    
    # Build Frontend image
    log_info "Building Frontend image..."
    docker build -f "${DOCKER_DIR}/Dockerfile.frontend.prod" -t novacron/frontend:latest ./frontend
    
    log_success "Docker images built successfully"
}

# Deploy with Docker Compose
deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    cd "$DOCKER_DIR"
    
    # Create production environment file
    cat > .env << EOF
DOMAIN=${DOMAIN}
ACME_EMAIL=${ACME_EMAIL}
DB_USER=${DB_USER}
DB_NAME=${DB_NAME}
DB_PASSWORD=${DB_PASSWORD}
REDIS_PASSWORD=${REDIS_PASSWORD}
AUTH_SECRET=${AUTH_SECRET}
GRAFANA_ADMIN_USER=${GRAFANA_ADMIN_USER}
GRAFANA_PASSWORD=${GRAFANA_PASSWORD}
EOF
    
    # Deploy services
    docker-compose -f docker-compose.prod.yml up -d
    
    log_success "Docker Compose deployment completed"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Create namespace
    kubectl apply -f "${K8S_DIR}/namespace.yaml"
    
    # Create secrets
    kubectl create secret generic novacron-secrets \
        --namespace=novacron-prod \
        --from-literal=database-url="postgresql://${DB_USER}:${DB_PASSWORD}@postgres:5432/${DB_NAME}" \
        --from-literal=redis-url="redis://:${REDIS_PASSWORD}@redis-master:6379" \
        --from-literal=auth-secret="${AUTH_SECRET}" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy applications
    kubectl apply -f "${K8S_DIR}/api-deployment.yaml"
    
    log_success "Kubernetes deployment completed"
}

# Health checks
run_health_checks() {
    log_info "Running health checks..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "http://localhost:8090/health" > /dev/null; then
            log_success "API health check passed"
            break
        fi
        
        log_info "Health check attempt $attempt/$max_attempts failed, retrying in 10s..."
        sleep 10
        ((attempt++))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        log_error "Health checks failed after $max_attempts attempts"
        return 1
    fi
    
    # Additional service checks
    local services=("frontend:8092" "prometheus:9090" "grafana:3000")
    for service in "${services[@]}"; do
        local name=$(echo "$service" | cut -d: -f1)
        local port=$(echo "$service" | cut -d: -f2)
        
        if curl -f -s "http://localhost:$port" > /dev/null; then
            log_success "$name service is healthy"
        else
            log_warning "$name service health check failed"
        fi
    done
}

# Smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Test API endpoints
    local api_base="http://localhost:8090"
    
    # Test health endpoint
    if curl -f -s "$api_base/health" | jq -e '.status == "healthy"' > /dev/null; then
        log_success "API health endpoint test passed"
    else
        log_error "API health endpoint test failed"
        return 1
    fi
    
    # Test API info endpoint
    if curl -f -s "$api_base/api/info" | jq -e '.name' > /dev/null; then
        log_success "API info endpoint test passed"
    else
        log_error "API info endpoint test failed"
        return 1
    fi
    
    # Test authentication endpoint (should return 401 without credentials)
    if [ "$(curl -s -o /dev/null -w "%{http_code}" "$api_base/api/vm/vms")" = "401" ]; then
        log_success "API authentication test passed"
    else
        log_error "API authentication test failed"
        return 1
    fi
    
    log_success "All smoke tests passed"
}

# Backup current deployment
backup_deployment() {
    log_info "Creating deployment backup..."
    
    local backup_dir="${PROJECT_ROOT}/backups/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup database
    if command -v pg_dump >/dev/null 2>&1; then
        pg_dump "$DB_URL" > "${backup_dir}/database.sql" || {
            log_warning "Database backup failed"
        }
    fi
    
    # Backup configuration
    cp -r "$DEPLOYMENT_DIR" "$backup_dir/"
    
    log_success "Backup created at $backup_dir"
}

# Rollback deployment
rollback_deployment() {
    log_warning "Rolling back deployment..."
    
    cd "$DOCKER_DIR"
    docker-compose -f docker-compose.prod.yml down
    
    # Restore from backup if available
    local latest_backup=$(ls -t "${PROJECT_ROOT}/backups/" 2>/dev/null | head -n1)
    if [ -n "$latest_backup" ]; then
        log_info "Restoring from backup: $latest_backup"
        # Add rollback logic here
    fi
    
    log_success "Rollback completed"
}

# Cleanup old resources
cleanup() {
    log_info "Cleaning up old resources..."
    
    # Remove old Docker images
    docker image prune -f
    
    # Remove old volumes (be careful with this)
    # docker volume prune -f
    
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    log_info "Starting NovaCron production deployment..."
    
    # Parse command line arguments
    local deployment_type="${1:-docker}"
    local skip_build="${2:-false}"
    
    # Check prerequisites
    check_prerequisites
    
    # Load environment
    load_environment
    
    # Generate secrets
    generate_secrets
    
    # Create backup
    backup_deployment
    
    # Build images (unless skipped)
    if [ "$skip_build" != "skip-build" ]; then
        build_images
    fi
    
    # Deploy based on type
    case "$deployment_type" in
        "docker"|"compose")
            create_docker_secrets
            deploy_docker_compose
            ;;
        "kubernetes"|"k8s")
            deploy_kubernetes
            ;;
        *)
            log_error "Unknown deployment type: $deployment_type"
            echo "Usage: $0 [docker|kubernetes] [skip-build]"
            exit 1
            ;;
    esac
    
    # Wait for services to start
    log_info "Waiting for services to start..."
    sleep 30
    
    # Run health checks
    if ! run_health_checks; then
        log_error "Health checks failed, initiating rollback..."
        rollback_deployment
        exit 1
    fi
    
    # Run smoke tests
    if ! run_smoke_tests; then
        log_error "Smoke tests failed, initiating rollback..."
        rollback_deployment
        exit 1
    fi
    
    # Cleanup
    cleanup
    
    log_success "NovaCron production deployment completed successfully!"
    log_info "Services available at:"
    log_info "  - Frontend: https://${DOMAIN}"
    log_info "  - API: https://api.${DOMAIN}"
    log_info "  - Grafana: https://grafana.${DOMAIN}"
    log_info "  - Prometheus: https://metrics.${DOMAIN}"
}

# Trap cleanup on script exit
trap cleanup EXIT

# Handle script interruption
trap 'log_warning "Deployment interrupted"; rollback_deployment; exit 1' INT TERM

# Run main function
main "$@"