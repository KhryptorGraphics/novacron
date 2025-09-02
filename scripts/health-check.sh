#!/bin/bash

# NovaCron Production Health Check Script
# Comprehensive system health validation for production environments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Default endpoints
API_ENDPOINT="${API_ENDPOINT:-http://localhost:8090}"
FRONTEND_ENDPOINT="${FRONTEND_ENDPOINT:-http://localhost:8092}"
PROMETHEUS_ENDPOINT="${PROMETHEUS_ENDPOINT:-http://localhost:9090}"
GRAFANA_ENDPOINT="${GRAFANA_ENDPOINT:-http://localhost:3000}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Health check results
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
    ((PASSED_CHECKS++))
    ((TOTAL_CHECKS++))
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
    ((WARNING_CHECKS++))
    ((TOTAL_CHECKS++))
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
    ((FAILED_CHECKS++))
    ((TOTAL_CHECKS++))
}

# Health check functions
check_service_endpoint() {
    local name="$1"
    local url="$2"
    local timeout="${3:-10}"
    local expected_code="${4:-200}"
    
    if curl -f -s --max-time "$timeout" -o /dev/null -w "%{http_code}" "$url" | grep -q "$expected_code"; then
        log_success "$name endpoint is healthy ($url)"
        return 0
    else
        log_error "$name endpoint is unhealthy ($url)"
        return 1
    fi
}

check_api_health() {
    log_info "Checking API health..."
    
    # Basic health check
    if check_service_endpoint "API" "$API_ENDPOINT/health"; then
        # Test API endpoints
        check_service_endpoint "API Info" "$API_ENDPOINT/api/info"
        
        # Test authentication (should return 401)
        if curl -s -o /dev/null -w "%{http_code}" "$API_ENDPOINT/api/vm/vms" | grep -q "401"; then
            log_success "API authentication is working"
        else
            log_error "API authentication issue"
        fi
    fi
}

check_system_resources() {
    log_info "Checking System resources..."
    
    # Check disk space
    local disk_usage
    disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    
    if [ "$disk_usage" -lt 80 ]; then
        log_success "Disk usage is healthy ($disk_usage%)"
    elif [ "$disk_usage" -lt 90 ]; then
        log_warning "Disk usage is elevated ($disk_usage%)"
    else
        log_error "Disk usage is critical ($disk_usage%)"
    fi
}

generate_report() {
    echo ""
    echo "=================================="
    echo "    HEALTH CHECK SUMMARY"
    echo "=================================="
    echo "Total Checks: $TOTAL_CHECKS"
    echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
    echo -e "Warnings: ${YELLOW}$WARNING_CHECKS${NC}"
    echo -e "Failed: ${RED}$FAILED_CHECKS${NC}"
    echo "=================================="
    
    if [ "$FAILED_CHECKS" -eq 0 ] && [ "$WARNING_CHECKS" -eq 0 ]; then
        echo -e "${GREEN}✓ System is healthy${NC}"
        return 0
    elif [ "$FAILED_CHECKS" -eq 0 ]; then
        echo -e "${YELLOW}⚠ System is healthy with warnings${NC}"
        return 1
    else
        echo -e "${RED}✗ System has critical issues${NC}"
        return 2
    fi
}

# Main function
main() {
    echo "NovaCron Production Health Check"
    echo "================================"
    echo ""
    
    # Run health checks
    check_system_resources
    check_api_health
    
    # Generate final report
    generate_report
}

# Run main function
main "$@"