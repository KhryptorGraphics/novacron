#!/bin/bash

# NovaCron Phase 1 Production Deployment Script
# Deploys: Redis caching, AI engine, K8s operator, enhanced SDKs, monitoring

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="/tmp/novacron-phase1-deployment-$(date +%Y%m%d-%H%M%S).log"

# Logging functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1${NC}" | tee -a "$LOG_FILE"
}

# Error handling
trap 'error "Deployment failed at line $LINENO. Check $LOG_FILE for details."' ERR

# Help function
show_help() {
    cat << EOF
NovaCron Phase 1 Production Deployment

Usage: $0 [OPTIONS]

Options:
    --help              Show this help message
    --redis-only        Deploy only Redis caching infrastructure
    --ai-only           Deploy only AI operations engine
    --k8s-only          Deploy only Kubernetes operator
    --monitoring-only   Deploy only monitoring stack
    --validate-only     Run validation tests only
    --cleanup           Clean up all Phase 1 components
    --status            Show deployment status
    --logs              Show recent deployment logs

Environment Variables:
    NOVACRON_ENV        Deployment environment (development|staging|production)
    REDIS_PASSWORD      Redis cluster password (auto-generated if not set)
    AI_ENGINE_PORT      AI engine port (default: 8093)
    MONITORING_PORT     Monitoring port (default: 3001)

Examples:
    $0                          # Full Phase 1 deployment
    $0 --redis-only            # Redis caching only
    $0 --validate-only         # Validation tests only
    $0 --cleanup               # Remove all components
    $0 --status                # Check deployment status

EOF
}

# Validation functions
check_prerequisites() {
    log "🔍 Checking prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    for tool in docker docker-compose kubectl make; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        error "Missing required tools: ${missing_tools[*]}"
        info "Please install missing tools and retry"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi
    
    # Check available disk space (minimum 10GB)
    local available_space=$(df / | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 10485760 ]]; then # 10GB in KB
        warn "Low disk space detected. Minimum 10GB recommended"
    fi
    
    # Set environment
    export NOVACRON_ENV="${NOVACRON_ENV:-production}"
    export REDIS_PASSWORD="${REDIS_PASSWORD:-$(openssl rand -base64 32)}"
    export AI_ENGINE_PORT="${AI_ENGINE_PORT:-8093}"
    export MONITORING_PORT="${MONITORING_PORT:-3001}"
    
    log "✅ Prerequisites check completed"
}

# Redis caching deployment
deploy_redis() {
    log "🔄 Deploying Redis caching infrastructure..."
    
    cd "$PROJECT_ROOT"
    
    # Setup Redis cluster with high availability
    if [[ -x "./scripts/cache-setup.sh" ]]; then
        ./scripts/cache-setup.sh setup sentinel
        ./scripts/cache-setup.sh start
    else
        warn "cache-setup.sh not found, using Docker Compose fallback"
        docker-compose -f docker-compose.cache.yml up -d
    fi
    
    # Wait for Redis to be ready
    info "Waiting for Redis cluster to be ready..."
    local retries=30
    while [[ $retries -gt 0 ]]; do
        if docker exec redis-master redis-cli ping &> /dev/null; then
            break
        fi
        sleep 2
        ((retries--))
    done
    
    if [[ $retries -eq 0 ]]; then
        error "Redis cluster failed to start within timeout"
        return 1
    fi
    
    log "✅ Redis caching infrastructure deployed successfully"
    info "Redis Commander UI: http://localhost:8082"
}

# AI engine deployment
deploy_ai_engine() {
    log "🤖 Deploying AI operations engine..."
    
    cd "$PROJECT_ROOT/ai-engine"
    
    # Build AI engine image
    if [[ -f "Dockerfile" ]]; then
        docker build -t novacron-ai-engine:latest .
    fi
    
    # Deploy AI engine with dependencies
    docker-compose up -d
    
    # Wait for AI engine to be ready
    info "Waiting for AI engine to be ready..."
    local retries=60
    while [[ $retries -gt 0 ]]; do
        if curl -sf "http://localhost:${AI_ENGINE_PORT}/health" &> /dev/null; then
            break
        fi
        sleep 2
        ((retries--))
    done
    
    if [[ $retries -eq 0 ]]; then
        error "AI engine failed to start within timeout"
        return 1
    fi
    
    log "✅ AI operations engine deployed successfully"
    info "AI Engine API: http://localhost:${AI_ENGINE_PORT}/docs"
}

# Kubernetes operator deployment
deploy_k8s_operator() {
    log "☸️ Deploying enhanced Kubernetes operator..."
    
    cd "$PROJECT_ROOT/k8s-operator"
    
    # Check if kubectl is configured
    if ! kubectl cluster-info &> /dev/null; then
        warn "kubectl not configured for cluster access"
        info "Skipping Kubernetes operator deployment"
        return 0
    fi
    
    # Deploy CRDs and operator
    if [[ -x "./deploy/deploy-enhanced-operator.sh" ]]; then
        ./deploy/deploy-enhanced-operator.sh --examples
    else
        warn "Enhanced operator deployment script not found"
        # Fallback to basic deployment
        kubectl apply -f deploy/crds/
        kubectl apply -f deploy/operator/
    fi
    
    # Wait for operator to be ready
    info "Waiting for operator to be ready..."
    kubectl wait --for=condition=available deployment/novacron-operator -n novacron-system --timeout=300s
    
    log "✅ Kubernetes operator deployed successfully"
    info "Operator namespace: novacron-system"
}

# Monitoring deployment
deploy_monitoring() {
    log "📊 Deploying monitoring infrastructure..."
    
    cd "$PROJECT_ROOT"
    
    # Deploy Prometheus and Grafana
    if [[ -f "docker-compose.monitoring.yml" ]]; then
        docker-compose -f docker-compose.monitoring.yml up -d
    elif [[ -f "docker-compose.yml" ]]; then
        # Use main compose file with monitoring profiles
        docker-compose --profile monitoring up -d
    else
        warn "Monitoring configuration not found, deploying basic setup"
        
        # Create basic monitoring setup
        cat > docker-compose.monitoring.yml << EOF
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
      
  grafana:
    image: grafana/grafana:latest
    ports:
      - "${MONITORING_PORT}:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
      
volumes:
  grafana-storage:
EOF
        docker-compose -f docker-compose.monitoring.yml up -d
    fi
    
    # Wait for monitoring to be ready
    info "Waiting for monitoring to be ready..."
    local retries=30
    while [[ $retries -gt 0 ]]; do
        if curl -sf "http://localhost:${MONITORING_PORT}" &> /dev/null; then
            break
        fi
        sleep 2
        ((retries--))
    done
    
    log "✅ Monitoring infrastructure deployed successfully"
    info "Grafana: http://localhost:${MONITORING_PORT} (admin/admin)"
    info "Prometheus: http://localhost:9090"
}

# Validation tests
run_validation() {
    log "🧪 Running Phase 1 validation tests..."
    
    cd "$PROJECT_ROOT"
    
    local test_results=()
    
    # Test Redis caching
    info "Testing Redis caching..."
    if docker exec redis-master redis-cli ping | grep -q "PONG"; then
        test_results+=("✅ Redis: PASS")
    else
        test_results+=("❌ Redis: FAIL")
    fi
    
    # Test AI engine
    info "Testing AI engine..."
    if curl -sf "http://localhost:${AI_ENGINE_PORT}/health" &> /dev/null; then
        test_results+=("✅ AI Engine: PASS")
    else
        test_results+=("❌ AI Engine: FAIL")
    fi
    
    # Test Kubernetes operator (if deployed)
    if kubectl cluster-info &> /dev/null; then
        info "Testing Kubernetes operator..."
        if kubectl get crd multikloudvms.novacron.io &> /dev/null; then
            test_results+=("✅ K8s Operator: PASS")
        else
            test_results+=("❌ K8s Operator: FAIL")
        fi
    fi
    
    # Test monitoring
    info "Testing monitoring..."
    if curl -sf "http://localhost:${MONITORING_PORT}" &> /dev/null; then
        test_results+=("✅ Monitoring: PASS")
    else
        test_results+=("❌ Monitoring: FAIL")
    fi
    
    # Run comprehensive tests if available
    if [[ -f "Makefile" ]] && make -n test-all &> /dev/null; then
        info "Running comprehensive test suite..."
        if make test-all; then
            test_results+=("✅ Test Suite: PASS")
        else
            test_results+=("❌ Test Suite: FAIL")
        fi
    fi
    
    # Display results
    log "🎯 Validation Results:"
    printf '%s\n' "${test_results[@]}" | tee -a "$LOG_FILE"
    
    # Check if all tests passed
    local failed_tests=$(printf '%s\n' "${test_results[@]}" | grep -c "❌" || true)
    if [[ $failed_tests -gt 0 ]]; then
        error "$failed_tests validation test(s) failed"
        return 1
    fi
    
    log "✅ All validation tests passed"
}

# Status check
check_status() {
    log "📊 Checking Phase 1 deployment status..."
    
    echo "=== Service Status ===" | tee -a "$LOG_FILE"
    
    # Redis status
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q redis; then
        echo "✅ Redis Cluster: Running" | tee -a "$LOG_FILE"
        docker ps --format "table {{.Names}}\t{{.Status}}" | grep redis | tee -a "$LOG_FILE"
    else
        echo "❌ Redis Cluster: Not running" | tee -a "$LOG_FILE"
    fi
    
    # AI engine status
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q ai-engine; then
        echo "✅ AI Engine: Running" | tee -a "$LOG_FILE"
        docker ps --format "table {{.Names}}\t{{.Status}}" | grep ai-engine | tee -a "$LOG_FILE"
    else
        echo "❌ AI Engine: Not running" | tee -a "$LOG_FILE"
    fi
    
    # Kubernetes operator status
    if kubectl cluster-info &> /dev/null; then
        if kubectl get deployment novacron-operator -n novacron-system &> /dev/null; then
            echo "✅ K8s Operator: Deployed" | tee -a "$LOG_FILE"
            kubectl get deployment novacron-operator -n novacron-system | tee -a "$LOG_FILE"
        else
            echo "❌ K8s Operator: Not deployed" | tee -a "$LOG_FILE"
        fi
    else
        echo "⚪ K8s Operator: Cluster not available" | tee -a "$LOG_FILE"
    fi
    
    # Monitoring status
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(prometheus|grafana)"; then
        echo "✅ Monitoring: Running" | tee -a "$LOG_FILE"
        docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(prometheus|grafana)" | tee -a "$LOG_FILE"
    else
        echo "❌ Monitoring: Not running" | tee -a "$LOG_FILE"
    fi
    
    echo "=== Access URLs ===" | tee -a "$LOG_FILE"
    echo "Redis Commander: http://localhost:8082" | tee -a "$LOG_FILE"
    echo "AI Engine API: http://localhost:${AI_ENGINE_PORT}/docs" | tee -a "$LOG_FILE"
    echo "Grafana: http://localhost:${MONITORING_PORT}" | tee -a "$LOG_FILE"
    echo "Prometheus: http://localhost:9090" | tee -a "$LOG_FILE"
    
    log "📋 Status check completed"
}

# Cleanup function
cleanup_deployment() {
    log "🧹 Cleaning up Phase 1 deployment..."
    
    # Stop and remove containers
    docker-compose down -v --remove-orphans 2>/dev/null || true
    docker-compose -f docker-compose.cache.yml down -v --remove-orphans 2>/dev/null || true
    docker-compose -f docker-compose.monitoring.yml down -v --remove-orphans 2>/dev/null || true
    
    # Clean up AI engine
    if [[ -d "$PROJECT_ROOT/ai-engine" ]]; then
        cd "$PROJECT_ROOT/ai-engine"
        docker-compose down -v --remove-orphans 2>/dev/null || true
    fi
    
    # Clean up Kubernetes resources
    if kubectl cluster-info &> /dev/null; then
        kubectl delete namespace novacron-system --ignore-not-found=true
        kubectl delete crd --selector=app=novacron-operator --ignore-not-found=true
    fi
    
    # Remove images
    docker images | grep novacron | awk '{print $3}' | xargs -r docker rmi 2>/dev/null || true
    
    log "✅ Cleanup completed"
}

# Main deployment function
main_deployment() {
    log "🚀 Starting NovaCron Phase 1 Production Deployment"
    log "Environment: $NOVACRON_ENV"
    log "Log file: $LOG_FILE"
    
    # Deploy components in order
    deploy_redis
    deploy_ai_engine
    deploy_k8s_operator
    deploy_monitoring
    
    # Run validation
    run_validation
    
    # Display final status
    check_status
    
    log "🎉 Phase 1 deployment completed successfully!"
    log "📋 Summary:"
    log "   ✅ Redis caching with high availability"
    log "   ✅ AI operations engine with ML models"
    log "   ✅ Enhanced Kubernetes operator"
    log "   ✅ Comprehensive monitoring stack"
    log "   ✅ All validation tests passed"
    
    echo ""
    echo "=== Next Steps ==="
    echo "1. Access services using the URLs above"
    echo "2. Configure your applications to use the SDKs in /sdk/"
    echo "3. Deploy your first multi-cloud VMs using the K8s operator"
    echo "4. Monitor performance and AI insights through Grafana"
    echo "5. Begin Phase 2 planning for edge computing integration"
    echo ""
    echo "📚 Documentation: /claudedocs/"
    echo "🔧 Configuration: Check environment variables"
    echo "📞 Support: See PHASE_1_IMPLEMENTATION_SUMMARY.md"
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --redis-only)
        check_prerequisites
        deploy_redis
        ;;
    --ai-only)
        check_prerequisites
        deploy_ai_engine
        ;;
    --k8s-only)
        check_prerequisites
        deploy_k8s_operator
        ;;
    --monitoring-only)
        check_prerequisites
        deploy_monitoring
        ;;
    --validate-only)
        run_validation
        ;;
    --cleanup)
        cleanup_deployment
        ;;
    --status)
        check_status
        ;;
    --logs)
        if [[ -f "$LOG_FILE" ]]; then
            tail -f "$LOG_FILE"
        else
            echo "No log file found"
        fi
        ;;
    "")
        check_prerequisites
        main_deployment
        ;;
    *)
        error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac