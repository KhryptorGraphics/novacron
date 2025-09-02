#!/bin/bash

# NovaCron Production Health Check Script
# Version: 1.0.0
# Description: Comprehensive health check for all NovaCron components

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVIRONMENT="${1:-production}"
NAMESPACE="${NAMESPACE:-novacron}"
MONITORING_NAMESPACE="${MONITORING_NAMESPACE:-monitoring}"
TIMEOUT="${TIMEOUT:-30}"
VERBOSE="${VERBOSE:-false}"

# Domain configuration based on environment
case $ENVIRONMENT in
    "production")
        BASE_DOMAIN="novacron.local"
        ;;
    "staging")
        BASE_DOMAIN="staging.novacron.local"
        ;;
    "development")
        BASE_DOMAIN="dev.novacron.local"
        ;;
    *)
        BASE_DOMAIN="novacron.local"
        ;;
esac

# Health check results
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")
            echo -e "${GREEN}[INFO]${NC} ${timestamp} - $message"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} ${timestamp} - $message"
            ((WARNING_CHECKS++))
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message"
            ((FAILED_CHECKS++))
            ;;
        "PASS")
            echo -e "${GREEN}[PASS]${NC} ${timestamp} - $message"
            ((PASSED_CHECKS++))
            ;;
        "DEBUG")
            if [ "$VERBOSE" = "true" ]; then
                echo -e "${BLUE}[DEBUG]${NC} ${timestamp} - $message"
            fi
            ;;
    esac
    ((TOTAL_CHECKS++))
}

# Run health check with timeout
check_with_timeout() {
    local description=$1
    local command=$2
    local expected_code=${3:-0}
    
    log "DEBUG" "Running check: $description"
    
    if timeout "$TIMEOUT" bash -c "$command" &>/dev/null; then
        if [ $? -eq $expected_code ]; then
            log "PASS" "$description"
            return 0
        else
            log "ERROR" "$description - Command failed with unexpected exit code"
            return 1
        fi
    else
        log "ERROR" "$description - Command timed out or failed"
        return 1
    fi
}

# Check Kubernetes cluster connectivity
check_kubernetes_connectivity() {
    log "INFO" "Checking Kubernetes cluster connectivity..."
    
    if kubectl cluster-info &>/dev/null; then
        log "PASS" "Kubernetes cluster is accessible"
    else
        log "ERROR" "Cannot connect to Kubernetes cluster"
        return 1
    fi
    
    # Check namespace exists
    if kubectl get namespace "$NAMESPACE" &>/dev/null; then
        log "PASS" "Namespace '$NAMESPACE' exists"
    else
        log "ERROR" "Namespace '$NAMESPACE' does not exist"
        return 1
    fi
}

# Check pod status
check_pod_status() {
    log "INFO" "Checking pod status in namespace '$NAMESPACE'..."
    
    local pods=$(kubectl get pods -n "$NAMESPACE" --no-headers)
    
    if [ -z "$pods" ]; then
        log "ERROR" "No pods found in namespace '$NAMESPACE'"
        return 1
    fi
    
    while IFS= read -r line; do
        local pod_name=$(echo "$line" | awk '{print $1}')
        local ready=$(echo "$line" | awk '{print $2}')
        local status=$(echo "$line" | awk '{print $3}')
        local restarts=$(echo "$line" | awk '{print $4}')
        
        if [ "$status" = "Running" ]; then
            if [[ "$ready" =~ ^[0-9]+/[0-9]+$ ]]; then
                local ready_count=$(echo "$ready" | cut -d'/' -f1)
                local total_count=$(echo "$ready" | cut -d'/' -f2)
                
                if [ "$ready_count" -eq "$total_count" ]; then
                    log "PASS" "Pod '$pod_name' is running and ready ($ready)"
                else
                    log "WARN" "Pod '$pod_name' is running but not all containers are ready ($ready)"
                fi
            else
                log "WARN" "Pod '$pod_name' has unexpected ready format: $ready"
            fi
        else
            log "ERROR" "Pod '$pod_name' is not running (Status: $status, Restarts: $restarts)"
        fi
        
        # Check for high restart count
        if [ "$restarts" -gt 5 ]; then
            log "WARN" "Pod '$pod_name' has high restart count: $restarts"
        fi
    done <<< "$pods"
}

# Check service endpoints
check_service_endpoints() {
    log "INFO" "Checking service endpoints..."
    
    local services=("novacron-api" "novacron-frontend" "novacron-postgres")
    
    for service in "${services[@]}"; do
        if kubectl get service "$service" -n "$NAMESPACE" &>/dev/null; then
            local endpoints=$(kubectl get endpoints "$service" -n "$NAMESPACE" -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null)
            
            if [ -n "$endpoints" ]; then
                local endpoint_count=$(echo "$endpoints" | wc -w)
                log "PASS" "Service '$service' has $endpoint_count endpoint(s)"
            else
                log "ERROR" "Service '$service' has no endpoints"
            fi
        else
            log "ERROR" "Service '$service' not found"
        fi
    done
}

# Check ingress configuration
check_ingress() {
    log "INFO" "Checking ingress configuration..."
    
    local ingresses=$(kubectl get ingress -n "$NAMESPACE" --no-headers 2>/dev/null)
    
    if [ -n "$ingresses" ]; then
        while IFS= read -r line; do
            local ingress_name=$(echo "$line" | awk '{print $1}')
            local hosts=$(kubectl get ingress "$ingress_name" -n "$NAMESPACE" -o jsonpath='{.spec.rules[*].host}' 2>/dev/null)
            
            log "PASS" "Ingress '$ingress_name' configured for hosts: $hosts"
        done <<< "$ingresses"
    else
        log "WARN" "No ingress resources found"
    fi
}

# Check persistent volumes
check_persistent_volumes() {
    log "INFO" "Checking persistent volume claims..."
    
    local pvcs=$(kubectl get pvc -n "$NAMESPACE" --no-headers 2>/dev/null)
    
    if [ -n "$pvcs" ]; then
        while IFS= read -r line; do
            local pvc_name=$(echo "$line" | awk '{print $1}')
            local status=$(echo "$line" | awk '{print $2}')
            local volume=$(echo "$line" | awk '{print $3}')
            local capacity=$(echo "$line" | awk '{print $4}')
            
            if [ "$status" = "Bound" ]; then
                log "PASS" "PVC '$pvc_name' is bound to '$volume' ($capacity)"
            else
                log "ERROR" "PVC '$pvc_name' is not bound (Status: $status)"
            fi
        done <<< "$pvcs"
    else
        log "WARN" "No persistent volume claims found"
    fi
}

# Check database connectivity
check_database() {
    log "INFO" "Checking database connectivity..."
    
    # Port forward to database
    kubectl port-forward svc/novacron-postgres 5432:5432 -n "$NAMESPACE" &
    local port_forward_pid=$!
    
    sleep 3
    
    # Get database credentials from secret
    local db_user=$(kubectl get secret novacron-db-secrets -n "$NAMESPACE" -o jsonpath='{.data.postgres-user}' | base64 -d)
    local db_password=$(kubectl get secret novacron-db-secrets -n "$NAMESPACE" -o jsonpath='{.data.postgres-password}' | base64 -d)
    local db_name=$(kubectl get secret novacron-db-secrets -n "$NAMESPACE" -o jsonpath='{.data.postgres-database}' | base64 -d)
    
    # Test database connection
    if PGPASSWORD="$db_password" psql -h localhost -p 5432 -U "$db_user" -d "$db_name" -c "SELECT version();" &>/dev/null; then
        log "PASS" "Database connection successful"
        
        # Check database tables
        local table_count=$(PGPASSWORD="$db_password" psql -h localhost -p 5432 -U "$db_user" -d "$db_name" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" | tr -d ' ')
        
        if [ "$table_count" -gt 0 ]; then
            log "PASS" "Database has $table_count tables"
        else
            log "ERROR" "Database has no tables - migrations may not have run"
        fi
        
        # Check database size
        local db_size=$(PGPASSWORD="$db_password" psql -h localhost -p 5432 -U "$db_user" -d "$db_name" -t -c "SELECT pg_size_pretty(pg_database_size('$db_name'));" | tr -d ' ')
        log "INFO" "Database size: $db_size"
        
    else
        log "ERROR" "Database connection failed"
    fi
    
    # Clean up port forward
    kill $port_forward_pid 2>/dev/null || true
    sleep 1
}

# Check API health endpoints
check_api_health() {
    log "INFO" "Checking API health endpoints..."
    
    local endpoints=(
        "https://$BASE_DOMAIN/health"
        "https://$BASE_DOMAIN/api/info"
        "https://api.$BASE_DOMAIN/health"
    )
    
    for endpoint in "${endpoints[@]}"; do
        if curl -f -s -k --connect-timeout "$TIMEOUT" "$endpoint" &>/dev/null; then
            log "PASS" "API endpoint '$endpoint' is healthy"
            
            # Check response content
            local response=$(curl -s -k --connect-timeout "$TIMEOUT" "$endpoint" 2>/dev/null)
            if echo "$response" | grep -q "healthy\|ok"; then
                log "PASS" "API endpoint '$endpoint' returns healthy status"
            else
                log "WARN" "API endpoint '$endpoint' returns unexpected response"
            fi
        else
            log "ERROR" "API endpoint '$endpoint' is not accessible"
        fi
    done
}

# Check WebSocket connectivity
check_websocket() {
    log "INFO" "Checking WebSocket connectivity..."
    
    # Simple WebSocket test using curl
    if command -v websocat &> /dev/null; then
        local ws_url="wss://ws.$BASE_DOMAIN/"
        
        if timeout 10 websocat -n1 "$ws_url" <<< "ping" &>/dev/null; then
            log "PASS" "WebSocket endpoint is accessible"
        else
            log "WARN" "WebSocket endpoint test failed (websocat may not be available)"
        fi
    else
        log "WARN" "WebSocket test skipped (websocat not available)"
    fi
}

# Check SSL certificates
check_ssl_certificates() {
    log "INFO" "Checking SSL certificates..."
    
    local domains=("$BASE_DOMAIN" "api.$BASE_DOMAIN" "ws.$BASE_DOMAIN")
    
    for domain in "${domains[@]}"; do
        local cert_info=$(echo | openssl s_client -servername "$domain" -connect "$domain:443" 2>/dev/null | openssl x509 -noout -dates 2>/dev/null)
        
        if [ -n "$cert_info" ]; then
            local not_after=$(echo "$cert_info" | grep "notAfter" | cut -d'=' -f2)
            local expiry_date=$(date -d "$not_after" +%s 2>/dev/null)
            local current_date=$(date +%s)
            local days_until_expiry=$(( (expiry_date - current_date) / 86400 ))
            
            if [ "$days_until_expiry" -gt 30 ]; then
                log "PASS" "SSL certificate for '$domain' is valid (expires in $days_until_expiry days)"
            elif [ "$days_until_expiry" -gt 7 ]; then
                log "WARN" "SSL certificate for '$domain' expires soon ($days_until_expiry days)"
            else
                log "ERROR" "SSL certificate for '$domain' expires very soon ($days_until_expiry days)"
            fi
        else
            log "ERROR" "Could not retrieve SSL certificate information for '$domain'"
        fi
    done
}

# Check monitoring stack
check_monitoring() {
    log "INFO" "Checking monitoring stack..."
    
    # Check if monitoring namespace exists
    if kubectl get namespace "$MONITORING_NAMESPACE" &>/dev/null; then
        log "PASS" "Monitoring namespace '$MONITORING_NAMESPACE' exists"
        
        # Check Prometheus
        if kubectl get pods -n "$MONITORING_NAMESPACE" -l "app.kubernetes.io/name=prometheus" --no-headers | grep -q "Running"; then
            log "PASS" "Prometheus is running"
            
            # Test Prometheus endpoint
            kubectl port-forward -n "$MONITORING_NAMESPACE" svc/kube-prometheus-stack-prometheus 9090:9090 &
            local prom_pid=$!
            sleep 3
            
            if curl -f -s http://localhost:9090/-/ready &>/dev/null; then
                log "PASS" "Prometheus is ready"
            else
                log "WARN" "Prometheus may not be fully ready"
            fi
            
            kill $prom_pid 2>/dev/null || true
        else
            log "ERROR" "Prometheus is not running"
        fi
        
        # Check Grafana
        if kubectl get pods -n "$MONITORING_NAMESPACE" -l "app.kubernetes.io/name=grafana" --no-headers | grep -q "Running"; then
            log "PASS" "Grafana is running"
        else
            log "WARN" "Grafana is not running"
        fi
        
        # Check Alertmanager
        if kubectl get pods -n "$MONITORING_NAMESPACE" -l "app.kubernetes.io/name=alertmanager" --no-headers | grep -q "Running"; then
            log "PASS" "Alertmanager is running"
        else
            log "WARN" "Alertmanager is not running"
        fi
        
    else
        log "WARN" "Monitoring namespace '$MONITORING_NAMESPACE' not found"
    fi
}

# Check resource utilization
check_resource_utilization() {
    log "INFO" "Checking resource utilization..."
    
    # Check node resources
    local node_info=$(kubectl top nodes --no-headers 2>/dev/null)
    if [ -n "$node_info" ]; then
        while IFS= read -r line; do
            local node_name=$(echo "$line" | awk '{print $1}')
            local cpu_usage=$(echo "$line" | awk '{print $2}' | sed 's/m$//')
            local cpu_percent=$(echo "$line" | awk '{print $3}' | sed 's/%$//')
            local memory_usage=$(echo "$line" | awk '{print $4}')
            local memory_percent=$(echo "$line" | awk '{print $5}' | sed 's/%$//')
            
            if [ "$cpu_percent" -gt 80 ]; then
                log "WARN" "Node '$node_name' has high CPU usage: $cpu_percent%"
            else
                log "PASS" "Node '$node_name' CPU usage is normal: $cpu_percent%"
            fi
            
            if [ "$memory_percent" -gt 85 ]; then
                log "WARN" "Node '$node_name' has high memory usage: $memory_percent%"
            else
                log "PASS" "Node '$node_name' memory usage is normal: $memory_percent%"
            fi
        done <<< "$node_info"
    else
        log "WARN" "Could not retrieve node resource information"
    fi
    
    # Check pod resources
    local pod_info=$(kubectl top pods -n "$NAMESPACE" --no-headers 2>/dev/null)
    if [ -n "$pod_info" ]; then
        while IFS= read -r line; do
            local pod_name=$(echo "$line" | awk '{print $1}')
            local cpu_usage=$(echo "$line" | awk '{print $2}' | sed 's/m$//')
            local memory_usage=$(echo "$line" | awk '{print $3}' | sed 's/Mi$//')
            
            # Basic thresholds (adjust based on your requirements)
            if [ "$cpu_usage" -gt 1000 ]; then  # 1 CPU core
                log "WARN" "Pod '$pod_name' has high CPU usage: ${cpu_usage}m"
            fi
            
            if [ "$memory_usage" -gt 2048 ]; then  # 2GB
                log "WARN" "Pod '$pod_name' has high memory usage: ${memory_usage}Mi"
            fi
        done <<< "$pod_info"
    fi
}

# Check backup status
check_backup_status() {
    log "INFO" "Checking backup status..."
    
    # Check if backup CronJob exists
    if kubectl get cronjob novacron-backup -n "$NAMESPACE" &>/dev/null; then
        log "PASS" "Backup CronJob exists"
        
        # Check recent backup jobs
        local recent_jobs=$(kubectl get jobs -n "$NAMESPACE" -l job-name=novacron-backup --sort-by='.metadata.creationTimestamp' --no-headers | tail -3)
        
        if [ -n "$recent_jobs" ]; then
            local success_count=0
            while IFS= read -r line; do
                local job_name=$(echo "$line" | awk '{print $1}')
                local completions=$(echo "$line" | awk '{print $2}')
                
                if [[ "$completions" =~ ^1/1$ ]]; then
                    ((success_count++))
                fi
            done <<< "$recent_jobs"
            
            if [ "$success_count" -gt 0 ]; then
                log "PASS" "Recent backup jobs completed successfully ($success_count of last 3)"
            else
                log "ERROR" "Recent backup jobs failed"
            fi
        else
            log "WARN" "No recent backup jobs found"
        fi
    else
        log "WARN" "Backup CronJob not found"
    fi
}

# Check security configuration
check_security() {
    log "INFO" "Checking security configuration..."
    
    # Check network policies
    local network_policies=$(kubectl get networkpolicy -n "$NAMESPACE" --no-headers 2>/dev/null)
    if [ -n "$network_policies" ]; then
        local policy_count=$(echo "$network_policies" | wc -l)
        log "PASS" "Network policies configured ($policy_count policies)"
    else
        log "WARN" "No network policies found"
    fi
    
    # Check pod security context
    local pods_with_security_context=0
    local total_pods=0
    
    local pods=$(kubectl get pods -n "$NAMESPACE" -o name)
    while IFS= read -r pod; do
        ((total_pods++))
        local security_context=$(kubectl get "$pod" -n "$NAMESPACE" -o jsonpath='{.spec.securityContext}' 2>/dev/null)
        if [ -n "$security_context" ] && [ "$security_context" != "null" ]; then
            ((pods_with_security_context++))
        fi
    done <<< "$pods"
    
    if [ "$pods_with_security_context" -eq "$total_pods" ]; then
        log "PASS" "All pods have security context configured"
    else
        log "WARN" "Some pods missing security context ($pods_with_security_context/$total_pods)"
    fi
    
    # Check for non-root containers
    local non_root_pods=0
    pods=$(kubectl get pods -n "$NAMESPACE" -o name)
    while IFS= read -r pod; do
        local run_as_non_root=$(kubectl get "$pod" -n "$NAMESPACE" -o jsonpath='{.spec.securityContext.runAsNonRoot}' 2>/dev/null)
        if [ "$run_as_non_root" = "true" ]; then
            ((non_root_pods++))
        fi
    done <<< "$pods"
    
    if [ "$non_root_pods" -gt 0 ]; then
        log "PASS" "Some pods configured to run as non-root ($non_root_pods)"
    else
        log "WARN" "No pods explicitly configured to run as non-root"
    fi
}

# Generate health report
generate_health_report() {
    log "INFO" "Generating health check report..."
    
    local report_file="/tmp/novacron-health-report-$(date +%Y%m%d_%H%M%S).json"
    local overall_status="HEALTHY"
    
    if [ "$FAILED_CHECKS" -gt 0 ]; then
        overall_status="UNHEALTHY"
    elif [ "$WARNING_CHECKS" -gt 5 ]; then
        overall_status="DEGRADED"
    fi
    
    cat > "$report_file" << EOF
{
    "health_check_report": {
        "timestamp": "$(date -Iseconds)",
        "environment": "$ENVIRONMENT",
        "namespace": "$NAMESPACE",
        "overall_status": "$overall_status",
        "summary": {
            "total_checks": $TOTAL_CHECKS,
            "passed_checks": $PASSED_CHECKS,
            "failed_checks": $FAILED_CHECKS,
            "warning_checks": $WARNING_CHECKS,
            "success_rate": $(( (PASSED_CHECKS * 100) / TOTAL_CHECKS ))
        },
        "components": {
            "kubernetes": "$([ $(kubectl cluster-info &>/dev/null; echo $?) -eq 0 ] && echo "HEALTHY" || echo "UNHEALTHY")",
            "database": "checked",
            "api": "checked",
            "frontend": "checked",
            "monitoring": "checked",
            "security": "checked"
        }
    }
}
EOF
    
    log "INFO" "Health report generated: $report_file"
    
    # Display summary
    echo
    echo "======================================"
    echo "      HEALTH CHECK SUMMARY"
    echo "======================================"
    echo "Environment: $ENVIRONMENT"
    echo "Total Checks: $TOTAL_CHECKS"
    echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
    echo -e "Warnings: ${YELLOW}$WARNING_CHECKS${NC}"
    echo -e "Failed: ${RED}$FAILED_CHECKS${NC}"
    echo -e "Overall Status: $overall_status"
    echo "Success Rate: $(( (PASSED_CHECKS * 100) / TOTAL_CHECKS ))%"
    echo "======================================"
    
    # Exit with appropriate code
    if [ "$FAILED_CHECKS" -gt 0 ]; then
        return 1
    elif [ "$WARNING_CHECKS" -gt 10 ]; then
        return 2
    else
        return 0
    fi
}

# Display usage information
usage() {
    cat << EOF
NovaCron Production Health Check Script

Usage: $0 [ENVIRONMENT] [OPTIONS]

Environments:
    production      Production environment (default)
    staging         Staging environment
    development     Development environment

Options:
    -v, --verbose       Enable verbose output
    -t, --timeout SEC   Set timeout for checks (default: 30)
    -n, --namespace NS  Kubernetes namespace (default: novacron)
    -h, --help         Show this help message

Examples:
    $0                          # Check production
    $0 staging                  # Check staging
    $0 production -v            # Verbose production check
    $0 -t 60 -n my-namespace    # Custom timeout and namespace

Exit Codes:
    0   All checks passed
    1   One or more checks failed
    2   Many warnings (>10)

EOF
}

# Main execution function
main() {
    log "INFO" "NovaCron Health Check v1.0.0"
    log "INFO" "Environment: $ENVIRONMENT"
    log "INFO" "Namespace: $NAMESPACE"
    log "INFO" "Base Domain: $BASE_DOMAIN"
    echo
    
    # Run all health checks
    check_kubernetes_connectivity
    check_pod_status
    check_service_endpoints
    check_ingress
    check_persistent_volumes
    check_database
    check_api_health
    check_websocket
    check_ssl_certificates
    check_monitoring
    check_resource_utilization
    check_backup_status
    check_security
    
    # Generate final report
    generate_health_report
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        production|staging|development)
            ENVIRONMENT="$1"
            shift
            ;;
        *)
            log "ERROR" "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Execute main function
main "$@"