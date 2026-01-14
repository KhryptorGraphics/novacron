#!/bin/bash

# NovaCron Production Deployment Script
# Automated deployment with blue-green strategy, health checks, and rollback capability

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
readonly DEPLOYMENT_STRATEGY="${DEPLOYMENT_STRATEGY:-blue-green}"
readonly ROLLBACK_ON_FAILURE="${ROLLBACK_ON_FAILURE:-true}"
readonly HEALTH_CHECK_RETRIES="${HEALTH_CHECK_RETRIES:-10}"
readonly HEALTH_CHECK_DELAY="${HEALTH_CHECK_DELAY:-30}"
readonly BACKUP_BEFORE_DEPLOY="${BACKUP_BEFORE_DEPLOY:-true}"
readonly DEPLOYMENT_TIMEOUT="${DEPLOYMENT_TIMEOUT:-1800}"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[0;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

# Trap errors and cleanup
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "Deployment failed with exit code: $exit_code"
        if [ "$ROLLBACK_ON_FAILURE" = "true" ]; then
            log_info "Initiating rollback..."
            rollback_deployment
        fi
    fi
    exit $exit_code
}

trap cleanup EXIT

# Function: Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    local tools=("docker" "docker-compose" "kubectl" "helm" "pg_dump" "redis-cli" "curl" "jq")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Required tool '$tool' is not installed"
            exit 1
        fi
    done

    # Check Kubernetes connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check Docker registry access
    if ! docker pull alpine:latest &> /dev/null; then
        log_error "Cannot access Docker registry"
        exit 1
    fi

    log_success "All prerequisites met"
}

# Function: Load environment configuration
load_environment() {
    log_info "Loading environment configuration..."

    local env_file="${PROJECT_ROOT}/deployment/environments/${DEPLOYMENT_ENV}.env"
    if [ ! -f "$env_file" ]; then
        log_error "Environment file not found: $env_file"
        exit 1
    fi

    # shellcheck disable=SC1090
    source "$env_file"

    # Validate required variables
    local required_vars=("CLUSTER_NAME" "NAMESPACE" "DOCKER_REGISTRY" "DATABASE_URL" "REDIS_URL")
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            log_error "Required environment variable '$var' is not set"
            exit 1
        fi
    done

    log_success "Environment configuration loaded"
}

# Function: Build and push Docker images
build_images() {
    log_info "Building Docker images..."

    local version
    version="$(git describe --tags --always --dirty)"
    export IMAGE_TAG="${IMAGE_TAG:-$version}"

    local services=("api-server" "frontend" "ai-engine" "backup" "monitoring-agent")
    for service in "${services[@]}"; do
        log_info "Building $service:$IMAGE_TAG..."

        docker build \
            --build-arg VERSION="$IMAGE_TAG" \
            --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
            --tag "${DOCKER_REGISTRY}/novacron/${service}:${IMAGE_TAG}" \
            --tag "${DOCKER_REGISTRY}/novacron/${service}:latest" \
            --file "${PROJECT_ROOT}/docker/${service}/Dockerfile" \
            "${PROJECT_ROOT}"

        log_info "Pushing $service:$IMAGE_TAG to registry..."
        docker push "${DOCKER_REGISTRY}/novacron/${service}:${IMAGE_TAG}"
        docker push "${DOCKER_REGISTRY}/novacron/${service}:latest"
    done

    log_success "All images built and pushed successfully"
}

# Function: Run tests
run_tests() {
    log_info "Running deployment tests..."

    # Run unit tests
    log_info "Running unit tests..."
    cd "${PROJECT_ROOT}/backend"
    go test -v -race -coverprofile=coverage.out ./...

    # Run integration tests
    log_info "Running integration tests..."
    cd "${PROJECT_ROOT}/tests/integration"
    go test -v -tags=integration ./...

    # Run contract tests
    log_info "Running contract tests..."
    cd "${PROJECT_ROOT}/tests/contract"
    npm test

    log_success "All tests passed"
}

# Function: Backup database
backup_database() {
    if [ "$BACKUP_BEFORE_DEPLOY" != "true" ]; then
        log_info "Skipping database backup (BACKUP_BEFORE_DEPLOY=false)"
        return
    fi

    log_info "Backing up database..."

    local backup_file="novacron-backup-$(date +%Y%m%d-%H%M%S).sql"
    local backup_path="/tmp/${backup_file}"

    # Create backup
    PGPASSWORD="${DB_PASSWORD}" pg_dump \
        -h "${DB_HOST}" \
        -U "${DB_USER}" \
        -d "${DB_NAME}" \
        -f "${backup_path}"

    # Compress backup
    gzip "${backup_path}"

    # Upload to S3 or other storage
    if [ -n "${BACKUP_S3_BUCKET:-}" ]; then
        aws s3 cp "${backup_path}.gz" "s3://${BACKUP_S3_BUCKET}/database-backups/${backup_file}.gz"
        log_success "Database backup uploaded to S3"
    fi

    # Keep last backup reference for rollback
    echo "${backup_path}.gz" > /tmp/last-backup-reference.txt

    log_success "Database backed up successfully"
}

# Function: Run database migrations
run_migrations() {
    log_info "Running database migrations..."

    # Build migration job
    docker build \
        --tag "novacron-migrations:latest" \
        --file "${PROJECT_ROOT}/migrations/Dockerfile" \
        "${PROJECT_ROOT}/migrations"

    # Run migrations
    docker run \
        --rm \
        --env DATABASE_URL="${DATABASE_URL}" \
        --env MIGRATION_DIR="/migrations" \
        --network host \
        "novacron-migrations:latest" \
        migrate up

    log_success "Database migrations completed"
}

# Function: Deploy with Kubernetes (Blue-Green)
deploy_kubernetes() {
    log_info "Deploying to Kubernetes (${DEPLOYMENT_STRATEGY})..."

    local namespace="${NAMESPACE}"
    local deployment_color

    # Determine current active deployment
    if kubectl get service novacron-active -n "$namespace" &> /dev/null; then
        local current_color
        current_color=$(kubectl get service novacron-active -n "$namespace" -o json | jq -r '.spec.selector.color')
        deployment_color=$([[ "$current_color" == "blue" ]] && echo "green" || echo "blue")
    else
        deployment_color="blue"
    fi

    log_info "Deploying to $deployment_color environment..."

    # Apply configurations
    kubectl apply -f "${PROJECT_ROOT}/deployment/kubernetes/namespaces.yaml"
    kubectl apply -f "${PROJECT_ROOT}/deployment/kubernetes/configmaps.yaml"
    kubectl apply -f "${PROJECT_ROOT}/deployment/kubernetes/secrets.yaml"

    # Deploy new version to inactive color
    helm upgrade --install \
        "novacron-${deployment_color}" \
        "${PROJECT_ROOT}/deployment/helm/novacron" \
        --namespace "$namespace" \
        --set image.tag="${IMAGE_TAG}" \
        --set deployment.color="${deployment_color}" \
        --values "${PROJECT_ROOT}/deployment/helm/values-${DEPLOYMENT_ENV}.yaml" \
        --wait \
        --timeout "${DEPLOYMENT_TIMEOUT}s"

    # Wait for deployment to be ready
    kubectl rollout status deployment "novacron-api-${deployment_color}" -n "$namespace"

    log_success "Deployment to $deployment_color completed"

    # Store deployment info for potential rollback
    echo "$deployment_color" > /tmp/last-deployment-color.txt
}

# Function: Deploy with Docker Compose
deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."

    cd "${PROJECT_ROOT}/deployment/production"

    # Pull latest images
    docker-compose -f docker-compose.production.yml pull

    # Deploy with zero-downtime strategy
    docker-compose -f docker-compose.production.yml up -d --no-deps api-server-1 api-server-2

    # Wait for new containers to be healthy
    sleep "$HEALTH_CHECK_DELAY"

    # Remove old containers
    docker-compose -f docker-compose.production.yml up -d --no-deps --remove-orphans

    log_success "Docker Compose deployment completed"
}

# Function: Health check
health_check() {
    log_info "Performing health checks..."

    local retries=0
    local max_retries="$HEALTH_CHECK_RETRIES"
    local health_endpoint="${HEALTH_CHECK_URL:-http://localhost:8080/health}"

    while [ $retries -lt "$max_retries" ]; do
        if curl -sf "$health_endpoint" > /dev/null; then
            log_success "Health check passed"
            return 0
        fi

        retries=$((retries + 1))
        log_warning "Health check failed (attempt $retries/$max_retries)"
        sleep "$HEALTH_CHECK_DELAY"
    done

    log_error "Health check failed after $max_retries attempts"
    return 1
}

# Function: Smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."

    # Test API endpoints
    local endpoints=(
        "/api/v1/status"
        "/api/v1/vms"
        "/api/v1/clusters"
        "/api/v1/metrics"
    )

    for endpoint in "${endpoints[@]}"; do
        if ! curl -sf "${API_BASE_URL}${endpoint}" > /dev/null; then
            log_error "Smoke test failed for endpoint: $endpoint"
            return 1
        fi
    done

    # Test database connectivity
    if ! docker exec postgres-primary pg_isready -U novacron > /dev/null; then
        log_error "Database connectivity test failed"
        return 1
    fi

    # Test Redis connectivity
    if ! docker exec redis-sentinel redis-cli ping > /dev/null; then
        log_error "Redis connectivity test failed"
        return 1
    fi

    log_success "All smoke tests passed"
}

# Function: Switch traffic (Blue-Green)
switch_traffic() {
    if [ "$DEPLOYMENT_STRATEGY" != "blue-green" ]; then
        return
    fi

    log_info "Switching traffic to new deployment..."

    local deployment_color
    deployment_color=$(cat /tmp/last-deployment-color.txt)

    # Update service selector to point to new deployment
    kubectl patch service novacron-active -n "$NAMESPACE" \
        -p '{"spec":{"selector":{"color":"'"$deployment_color"'"}}}'

    # Wait for traffic switch
    sleep 10

    log_success "Traffic switched to $deployment_color deployment"
}

# Function: Cleanup old deployment
cleanup_old_deployment() {
    if [ "$DEPLOYMENT_STRATEGY" != "blue-green" ]; then
        return
    fi

    log_info "Cleaning up old deployment..."

    local deployment_color
    deployment_color=$(cat /tmp/last-deployment-color.txt)
    local old_color=$([[ "$deployment_color" == "blue" ]] && echo "green" || echo "blue")

    # Wait before cleaning up (safety buffer)
    sleep 60

    # Scale down old deployment
    kubectl scale deployment "novacron-api-${old_color}" -n "$NAMESPACE" --replicas=0

    log_success "Old deployment cleaned up"
}

# Function: Rollback deployment
rollback_deployment() {
    log_error "Initiating rollback procedure..."

    if [ "$DEPLOYMENT_STRATEGY" == "blue-green" ] && [ -f /tmp/last-deployment-color.txt ]; then
        local failed_color
        failed_color=$(cat /tmp/last-deployment-color.txt)
        local previous_color=$([[ "$failed_color" == "blue" ]] && echo "green" || echo "blue")

        log_info "Rolling back to $previous_color deployment..."

        # Switch traffic back
        kubectl patch service novacron-active -n "$NAMESPACE" \
            -p '{"spec":{"selector":{"color":"'"$previous_color"'"}}}'

        # Scale down failed deployment
        kubectl scale deployment "novacron-api-${failed_color}" -n "$NAMESPACE" --replicas=0

        log_success "Rolled back to $previous_color deployment"
    elif [ "$DEPLOYMENT_STRATEGY" == "rolling" ]; then
        log_info "Rolling back Kubernetes deployment..."
        kubectl rollout undo deployment/novacron-api -n "$NAMESPACE"
        kubectl rollout status deployment/novacron-api -n "$NAMESPACE"
        log_success "Kubernetes deployment rolled back"
    else
        log_warning "No rollback strategy available"
    fi

    # Restore database if backup exists
    if [ -f /tmp/last-backup-reference.txt ]; then
        local backup_file
        backup_file=$(cat /tmp/last-backup-reference.txt)

        if [ -f "$backup_file" ]; then
            log_info "Restoring database from backup..."
            gunzip -c "$backup_file" | PGPASSWORD="${DB_PASSWORD}" psql \
                -h "${DB_HOST}" \
                -U "${DB_USER}" \
                -d "${DB_NAME}"
            log_success "Database restored from backup"
        fi
    fi
}

# Function: Send deployment notification
send_notification() {
    local status="$1"
    local message="$2"

    log_info "Sending deployment notification..."

    # Slack notification
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        curl -X POST "${SLACK_WEBHOOK_URL}" \
            -H 'Content-Type: application/json' \
            -d "{
                \"text\": \"Deployment ${status}\",
                \"attachments\": [{
                    \"color\": \"$([ "$status" = "SUCCESS" ] && echo "good" || echo "danger")\",
                    \"fields\": [
                        {\"title\": \"Environment\", \"value\": \"${DEPLOYMENT_ENV}\", \"short\": true},
                        {\"title\": \"Version\", \"value\": \"${IMAGE_TAG}\", \"short\": true},
                        {\"title\": \"Message\", \"value\": \"${message}\"}
                    ]
                }]
            }"
    fi

    # Email notification
    if [ -n "${NOTIFICATION_EMAIL:-}" ]; then
        echo "$message" | mail -s "NovaCron Deployment ${status} - ${DEPLOYMENT_ENV}" "${NOTIFICATION_EMAIL}"
    fi
}

# Function: Generate deployment report
generate_report() {
    log_info "Generating deployment report..."

    local report_file="/tmp/deployment-report-$(date +%Y%m%d-%H%M%S).json"

    cat > "$report_file" <<EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": "${DEPLOYMENT_ENV}",
    "version": "${IMAGE_TAG}",
    "strategy": "${DEPLOYMENT_STRATEGY}",
    "status": "SUCCESS",
    "duration": "$((SECONDS / 60)) minutes",
    "services": {
        "api-server": "running",
        "frontend": "running",
        "ai-engine": "running",
        "postgres-primary": "healthy",
        "redis-sentinel": "healthy"
    },
    "tests": {
        "unit": "passed",
        "integration": "passed",
        "smoke": "passed"
    },
    "metrics": {
        "deployment_time": "${SECONDS}s",
        "downtime": "0s",
        "rollback_required": false
    }
}
EOF

    log_success "Deployment report generated: $report_file"

    # Upload report if configured
    if [ -n "${REPORT_S3_BUCKET:-}" ]; then
        aws s3 cp "$report_file" "s3://${REPORT_S3_BUCKET}/deployment-reports/"
    fi
}

# Main deployment flow
main() {
    log_info "Starting NovaCron production deployment..."
    log_info "Environment: ${DEPLOYMENT_ENV}"
    log_info "Strategy: ${DEPLOYMENT_STRATEGY}"

    # Track deployment time
    SECONDS=0

    # Pre-deployment phase
    check_prerequisites
    load_environment

    # Build phase
    build_images
    run_tests

    # Backup phase
    backup_database

    # Deployment phase
    run_migrations

    if [ "${DEPLOYMENT_TARGET:-kubernetes}" == "kubernetes" ]; then
        deploy_kubernetes
    else
        deploy_docker_compose
    fi

    # Validation phase
    health_check
    run_smoke_tests

    # Traffic switch phase
    switch_traffic

    # Post-deployment phase
    cleanup_old_deployment
    generate_report

    # Notification
    send_notification "SUCCESS" "Deployment completed successfully in $((SECONDS / 60)) minutes"

    log_success "Deployment completed successfully!"
    log_info "Total deployment time: $((SECONDS / 60)) minutes and $((SECONDS % 60)) seconds"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            DEPLOYMENT_ENV="$2"
            shift 2
            ;;
        --strategy)
            DEPLOYMENT_STRATEGY="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS="true"
            shift
            ;;
        --skip-backup)
            BACKUP_BEFORE_DEPLOY="false"
            shift
            ;;
        --rollback)
            rollback_deployment
            exit 0
            ;;
        --help)
            cat <<EOF
Usage: $0 [OPTIONS]

Options:
    --env <environment>      Deployment environment (staging|production)
    --strategy <strategy>    Deployment strategy (blue-green|rolling|canary)
    --skip-tests            Skip running tests
    --skip-backup           Skip database backup
    --rollback              Rollback to previous deployment
    --help                  Show this help message

Environment Variables:
    DEPLOYMENT_ENV          Deployment environment (default: production)
    DEPLOYMENT_STRATEGY     Deployment strategy (default: blue-green)
    ROLLBACK_ON_FAILURE     Auto-rollback on failure (default: true)
    HEALTH_CHECK_RETRIES    Number of health check retries (default: 10)
    BACKUP_BEFORE_DEPLOY    Backup database before deployment (default: true)

Examples:
    # Production deployment with blue-green strategy
    $0 --env production --strategy blue-green

    # Staging deployment without tests
    $0 --env staging --skip-tests

    # Rollback last deployment
    $0 --rollback

EOF
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main deployment
main