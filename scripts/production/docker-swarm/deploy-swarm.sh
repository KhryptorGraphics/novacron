#!/bin/bash

# Docker Swarm Production Deployment Script
# Usage: ./deploy-swarm.sh [staging|production]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENVIRONMENT="${1:-staging}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%H:%M:%S')] $1${NC}"; }
error() { echo -e "${RED}[$(date +'%H:%M:%S')] ERROR: $1${NC}"; exit 1; }
success() { echo -e "${GREEN}[$(date +'%H:%M:%S')] SUCCESS: $1${NC}"; }

log "Starting Docker Swarm deployment for $ENVIRONMENT"

# Load environment variables
ENV_FILE="$PROJECT_ROOT/deployment/configs/.env.$ENVIRONMENT"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
    log "Loaded environment configuration from $ENV_FILE"
else
    error "Environment file not found: $ENV_FILE"
fi

# Initialize Docker Swarm if not already done
initialize_swarm() {
    log "Checking Docker Swarm status..."
    
    if ! docker info | grep -q "Swarm: active"; then
        log "Initializing Docker Swarm..."
        docker swarm init --advertise-addr $(hostname -I | cut -d' ' -f1)
        success "Docker Swarm initialized"
    else
        log "Docker Swarm already active"
    fi
    
    # Create overlay networks
    docker network create --driver overlay --attachable novacron-network || true
    docker network create --driver overlay --attachable novacron-backend || true
    docker network create --driver overlay --attachable novacron-monitoring || true
    
    success "Networks configured"
}

# Create Docker secrets
create_secrets() {
    log "Creating Docker secrets..."
    
    # Database credentials
    echo "$POSTGRES_PASSWORD" | docker secret create postgres_password_v$(date +%s) - 2>/dev/null || true
    echo "$POSTGRES_USER" | docker secret create postgres_user_v$(date +%s) - 2>/dev/null || true
    
    # JWT secret
    echo "$JWT_SECRET" | docker secret create jwt_secret_v$(date +%s) - 2>/dev/null || true
    
    # Redis password
    echo "$REDIS_PASSWORD" | docker secret create redis_password_v$(date +%s) - 2>/dev/null || true
    
    # SSL certificates (if exists)
    if [[ -f "$PROJECT_ROOT/deployment/ssl/cert.pem" ]]; then
        docker secret create ssl_cert_v$(date +%s) "$PROJECT_ROOT/deployment/ssl/cert.pem" 2>/dev/null || true
        docker secret create ssl_key_v$(date +%s) "$PROJECT_ROOT/deployment/ssl/key.pem" 2>/dev/null || true
    fi
    
    success "Secrets created"
}

# Deploy stack
deploy_stack() {
    log "Deploying NovaCron stack..."
    
    local stack_file="$PROJECT_ROOT/deployment/docker/docker-compose.swarm.yml"
    if [[ ! -f "$stack_file" ]]; then
        error "Stack file not found: $stack_file"
    fi
    
    # Deploy the stack
    docker stack deploy \
        --compose-file "$stack_file" \
        --with-registry-auth \
        novacron
    
    success "Stack deployed"
}

# Wait for services
wait_for_services() {
    log "Waiting for services to be ready..."
    
    local max_attempts=60
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        local ready_services=0
        local total_services=0
        
        # Count running services
        while IFS= read -r line; do
            if [[ "$line" =~ novacron_ ]]; then
                total_services=$((total_services + 1))
                if [[ "$line" =~ "1/1" ]]; then
                    ready_services=$((ready_services + 1))
                fi
            fi
        done < <(docker service ls --format "table {{.Name}}\t{{.Replicas}}")
        
        if [ $ready_services -eq $total_services ] && [ $total_services -gt 0 ]; then
            success "All services are ready ($ready_services/$total_services)"
            break
        fi
        
        log "Services ready: $ready_services/$total_services (attempt $((attempt + 1))/$max_attempts)"
        sleep 10
        attempt=$((attempt + 1))
    done
    
    if [ $attempt -eq $max_attempts ]; then
        error "Services failed to start within timeout"
    fi
}

# Scale services for production
scale_services() {
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log "Scaling services for production..."
        
        # Scale API servers
        docker service update --replicas 3 novacron_api
        
        # Scale frontend
        docker service update --replicas 2 novacron_frontend
        
        # Scale database for HA (if using external clustering)
        # docker service update --replicas 3 novacron_postgres
        
        success "Services scaled for production"
    fi
}

# Configure load balancing
setup_load_balancer() {
    log "Setting up load balancer..."
    
    # Deploy Traefik as reverse proxy
    local traefik_config="$PROJECT_ROOT/deployment/docker/traefik.yml"
    if [[ -f "$traefik_config" ]]; then
        docker stack deploy --compose-file "$traefik_config" traefik
        success "Load balancer configured"
    else
        warn "Traefik configuration not found, skipping load balancer setup"
    fi
}

# Main execution
main() {
    log "=== Starting Docker Swarm Deployment ==="
    
    initialize_swarm
    create_secrets
    deploy_stack
    wait_for_services
    scale_services
    setup_load_balancer
    
    success "=== Docker Swarm Deployment Completed ==="
    
    # Display service status
    echo ""
    log "Current service status:"
    docker service ls
    
    echo ""
    log "Stack services:"
    docker stack services novacron
    
    # Display access information
    echo ""
    echo "=== Access Information ==="
    if [[ "$ENVIRONMENT" == "production" ]]; then
        echo "Frontend: https://${DOMAIN_NAME:-novacron.local}"
        echo "API: https://${DOMAIN_NAME:-novacron.local}/api"
        echo "Grafana: https://${DOMAIN_NAME:-novacron.local}/grafana"
    else
        echo "Frontend: http://localhost:8092"
        echo "API: http://localhost:8090"
        echo "Grafana: http://localhost:3001"
    fi
}

# Error handling
trap 'error "Docker Swarm deployment failed at line $LINENO"' ERR

# Run main function
main