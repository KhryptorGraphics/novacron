#!/bin/bash

# NovaCron Production Health Check Script
# Usage: ./health-check.sh [staging|production]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENVIRONMENT="${1:-production}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%H:%M:%S')] $1${NC}"; }
error() { echo -e "${RED}[$(date +'%H:%M:%S')] ERROR: $1${NC}"; }
success() { echo -e "${GREEN}[$(date +'%H:%M:%S')] SUCCESS: $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARN: $1${NC}"; }

# Configuration
TIMEOUT=30
RETRIES=3
HEALTH_REPORT="/tmp/novacron-health-$(date +%Y%m%d-%H%M%S).json"

# Load environment configuration
ENV_FILE="$PROJECT_ROOT/deployment/configs/.env.$ENVIRONMENT"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
    log "Loaded $ENVIRONMENT configuration"
else
    warn "Environment file not found: $ENV_FILE"
fi

# Determine base URLs
if [[ "$ENVIRONMENT" == "production" ]]; then
    BASE_URL="https://${DOMAIN_NAME:-novacron.local}"
    API_URL="$BASE_URL/api"
else
    BASE_URL="http://localhost:8092"
    API_URL="http://localhost:8090"
fi

log "Starting health check for $ENVIRONMENT environment"

# Initialize health report
init_report() {
    cat > "$HEALTH_REPORT" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "environment": "$ENVIRONMENT", 
  "checks": {},
  "overall_status": "unknown",
  "summary": {
    "total": 0,
    "passed": 0,
    "failed": 0,
    "warnings": 0
  }
}
EOF
}

# Add check result to report
add_check_result() {
    local check_name="$1"
    local status="$2"
    local message="$3"
    local response_time="${4:-0}"
    
    # Update the JSON report
    local temp_file=$(mktemp)
    jq --arg name "$check_name" \
       --arg status "$status" \
       --arg message "$message" \
       --argjson response_time "$response_time" \
       '.checks[$name] = {
         "status": $status,
         "message": $message,
         "response_time_ms": $response_time,
         "timestamp": now | strftime("%Y-%m-%dT%H:%M:%SZ")
       }' "$HEALTH_REPORT" > "$temp_file"
    mv "$temp_file" "$HEALTH_REPORT"
}

# HTTP health check with retry logic
http_check() {
    local name="$1"
    local url="$2"
    local expected_status="${3:-200}"
    local timeout="${4:-$TIMEOUT}"
    
    log "Checking $name: $url"
    
    local attempt=1
    local success_flag=false
    
    while [[ $attempt -le $RETRIES ]]; do
        local start_time=$(date +%s%3N)
        
        if response=$(curl -s -w "%{http_code}" -o /tmp/response_body --connect-timeout "$timeout" --max-time "$timeout" "$url" 2>/dev/null); then
            local end_time=$(date +%s%3N)
            local response_time=$((end_time - start_time))
            local http_code="${response: -3}"
            
            if [[ "$http_code" == "$expected_status" ]]; then
                success "$name is healthy (${http_code}) - ${response_time}ms"
                add_check_result "$name" "pass" "HTTP $http_code - ${response_time}ms" "$response_time"
                success_flag=true
                break
            else
                warn "$name returned HTTP $http_code (attempt $attempt/$RETRIES)"
                if [[ $attempt -eq $RETRIES ]]; then
                    error "$name health check failed: HTTP $http_code"
                    add_check_result "$name" "fail" "HTTP $http_code after $RETRIES attempts" "$response_time"
                fi
            fi
        else
            warn "$name connection failed (attempt $attempt/$RETRIES)"
            if [[ $attempt -eq $RETRIES ]]; then
                error "$name health check failed: Connection error"
                add_check_result "$name" "fail" "Connection error after $RETRIES attempts" 0
            fi
        fi
        
        ((attempt++))
        sleep 2
    done
    
    return $([[ "$success_flag" == "true" ]] && echo 0 || echo 1)
}

# Database connectivity check
check_database() {
    log "Checking database connectivity..."
    
    local db_check_result=0
    
    # Try to connect to database through API health endpoint
    if http_check "Database" "$API_URL/health/db" 200 10; then
        success "Database is accessible through API"
    else
        error "Database check failed"
        db_check_result=1
    fi
    
    return $db_check_result
}

# Redis connectivity check  
check_redis() {
    log "Checking Redis connectivity..."
    
    # Check Redis through API health endpoint
    if http_check "Redis" "$API_URL/health/redis" 200 10; then
        success "Redis is accessible"
    else
        error "Redis check failed"
        return 1
    fi
}

# API endpoints check
check_api_endpoints() {
    log "Checking API endpoints..."
    
    local endpoints=(
        "API Health:$API_URL/health:200"
        "API Info:$API_URL/info:200"
        "API Version:$API_URL/version:200"
        "API Metrics:$API_URL/metrics:200"
    )
    
    local failed_endpoints=0
    
    for endpoint_info in "${endpoints[@]}"; do
        IFS=':' read -r name url expected_code <<< "$endpoint_info"
        if ! http_check "$name" "$url" "$expected_code"; then
            ((failed_endpoints++))
        fi
    done
    
    if [[ $failed_endpoints -eq 0 ]]; then
        success "All API endpoints are healthy"
        return 0
    else
        error "$failed_endpoints API endpoints failed"
        return 1
    fi
}

# Frontend check
check_frontend() {
    log "Checking frontend..."
    
    if http_check "Frontend" "$BASE_URL" 200; then
        success "Frontend is accessible"
        return 0
    else
        error "Frontend check failed"
        return 1
    fi
}

# Monitoring services check
check_monitoring() {
    log "Checking monitoring services..."
    
    local monitoring_services=(
        "Prometheus:http://localhost:9090/-/healthy:200"
        "Grafana:http://localhost:3001/api/health:200"
    )
    
    local failed_services=0
    
    for service_info in "${monitoring_services[@]}"; do
        IFS=':' read -r name url expected_code <<< "$service_info"
        if ! http_check "$name" "$url" "$expected_code"; then
            ((failed_services++))
        fi
    done
    
    if [[ $failed_services -eq 0 ]]; then
        success "All monitoring services are healthy"
        return 0
    else
        warn "$failed_services monitoring services failed"
        return 1
    fi
}

# SSL certificate check
check_ssl_certificate() {
    if [[ "$ENVIRONMENT" == "production" && -n "${DOMAIN_NAME:-}" ]]; then
        log "Checking SSL certificate..."
        
        local cert_info
        if cert_info=$(echo | openssl s_client -servername "$DOMAIN_NAME" -connect "$DOMAIN_NAME:443" 2>/dev/null | openssl x509 -noout -dates 2>/dev/null); then
            local expiry_date=$(echo "$cert_info" | grep "notAfter" | cut -d= -f2)
            local expiry_epoch=$(date -d "$expiry_date" +%s)
            local current_epoch=$(date +%s)
            local days_until_expiry=$(( (expiry_epoch - current_epoch) / 86400 ))
            
            if [[ $days_until_expiry -lt 0 ]]; then
                error "SSL certificate has expired"
                add_check_result "SSL Certificate" "fail" "Certificate expired"
                return 1
            elif [[ $days_until_expiry -lt 14 ]]; then
                warn "SSL certificate expires in $days_until_expiry days"
                add_check_result "SSL Certificate" "warning" "Expires in $days_until_expiry days"
                return 2
            else
                success "SSL certificate is valid (expires in $days_until_expiry days)"
                add_check_result "SSL Certificate" "pass" "Valid, expires in $days_until_expiry days"
                return 0
            fi
        else
            error "Failed to check SSL certificate"
            add_check_result "SSL Certificate" "fail" "Unable to retrieve certificate"
            return 1
        fi
    else
        log "Skipping SSL check for $ENVIRONMENT environment"
        return 0
    fi
}

# Performance metrics check
check_performance() {
    log "Checking performance metrics..."
    
    local start_time=$(date +%s%3N)
    if http_check "Performance Test" "$API_URL/health" 200 5; then
        local end_time=$(date +%s%3N)
        local response_time=$((end_time - start_time))
        
        if [[ $response_time -lt 1000 ]]; then
            success "API response time is good (${response_time}ms)"
            return 0
        elif [[ $response_time -lt 3000 ]]; then
            warn "API response time is slow (${response_time}ms)"
            return 2
        else
            error "API response time is too slow (${response_time}ms)"
            return 1
        fi
    else
        error "Performance check failed"
        return 1
    fi
}

# Container/Pod health check
check_container_health() {
    log "Checking container/pod health..."
    
    # Detect deployment type
    if docker info | grep -q "Swarm: active"; then
        # Docker Swarm
        local unhealthy_services=$(docker service ls --format "table {{.Name}}\t{{.Replicas}}" | grep -v "1/1" | grep -v "NAME" | wc -l)
        if [[ $unhealthy_services -eq 0 ]]; then
            success "All Docker Swarm services are healthy"
            add_check_result "Container Health" "pass" "All services running"
            return 0
        else
            error "$unhealthy_services Docker Swarm services are unhealthy"
            add_check_result "Container Health" "fail" "$unhealthy_services services unhealthy"
            return 1
        fi
    elif kubectl cluster-info &>/dev/null; then
        # Kubernetes
        local unhealthy_pods=$(kubectl get pods -n novacron --field-selector=status.phase!=Running,status.phase!=Succeeded -o name | wc -l)
        if [[ $unhealthy_pods -eq 0 ]]; then
            success "All Kubernetes pods are healthy"
            add_check_result "Container Health" "pass" "All pods running"
            return 0
        else
            error "$unhealthy_pods Kubernetes pods are unhealthy"
            add_check_result "Container Health" "fail" "$unhealthy_pods pods unhealthy"
            return 1
        fi
    else
        # Docker Compose
        local compose_file="$PROJECT_ROOT/docker-compose.yml"
        if [[ -f "$compose_file" ]]; then
            if docker-compose -f "$compose_file" ps | grep -q "Exit\|unhealthy"; then
                error "Some Docker Compose services are unhealthy"
                add_check_result "Container Health" "fail" "Some services unhealthy"
                return 1
            else
                success "All Docker Compose services are healthy"
                add_check_result "Container Health" "pass" "All services running"
                return 0
            fi
        fi
    fi
    
    warn "Could not determine container health"
    add_check_result "Container Health" "warning" "Unable to determine status"
    return 2
}

# Generate summary
generate_summary() {
    log "Generating health check summary..."
    
    # Count results
    local total=$(jq '.checks | length' "$HEALTH_REPORT")
    local passed=$(jq '[.checks[] | select(.status == "pass")] | length' "$HEALTH_REPORT")
    local failed=$(jq '[.checks[] | select(.status == "fail")] | length' "$HEALTH_REPORT")
    local warnings=$(jq '[.checks[] | select(.status == "warning")] | length' "$HEALTH_REPORT")
    
    # Determine overall status
    local overall_status="healthy"
    if [[ $failed -gt 0 ]]; then
        overall_status="unhealthy"
    elif [[ $warnings -gt 0 ]]; then
        overall_status="degraded"
    fi
    
    # Update report
    local temp_file=$(mktemp)
    jq --arg status "$overall_status" \
       --argjson total "$total" \
       --argjson passed "$passed" \
       --argjson failed "$failed" \
       --argjson warnings "$warnings" \
       '.overall_status = $status |
        .summary.total = $total |
        .summary.passed = $passed |
        .summary.failed = $failed |
        .summary.warnings = $warnings' "$HEALTH_REPORT" > "$temp_file"
    mv "$temp_file" "$HEALTH_REPORT"
    
    echo ""
    echo "=== Health Check Summary ==="
    echo "Overall Status: $overall_status"
    echo "Total Checks: $total"
    echo "Passed: $passed"
    echo "Failed: $failed"
    echo "Warnings: $warnings"
    echo "Report: $HEALTH_REPORT"
    echo ""
    
    return $([[ "$overall_status" == "healthy" ]] && echo 0 || echo 1)
}

# Main execution
main() {
    log "=== NovaCron Health Check Started ==="
    
    init_report
    
    # Run all health checks
    local overall_result=0
    
    check_frontend || ((overall_result++))
    check_api_endpoints || ((overall_result++))
    check_database || ((overall_result++))
    check_redis || ((overall_result++))
    check_monitoring || true  # Don't fail overall check for monitoring
    check_ssl_certificate || true  # Don't fail overall check for SSL warnings
    check_performance || true  # Don't fail overall check for performance warnings
    check_container_health || ((overall_result++))
    
    # Generate final report
    generate_summary
    
    if [[ $overall_result -eq 0 ]]; then
        success "=== All Health Checks Passed ==="
        exit 0
    else
        error "=== $overall_result Health Checks Failed ==="
        exit 1
    fi
}

# Error handling
trap 'error "Health check failed at line $LINENO"' ERR

# Run main function
main