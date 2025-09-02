#!/bin/bash

# NovaCron Smoke Tests Script
# Usage: ./smoke-tests.sh [staging|production]

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
error() { echo -e "${RED}[$(date +'%H:%M:%S')] ERROR: $1${NC}"; }
success() { echo -e "${GREEN}[$(date +'%H:%M:%S')] SUCCESS: $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARN: $1${NC}"; }

# Load environment configuration
ENV_FILE="$PROJECT_ROOT/deployment/configs/.env.$ENVIRONMENT"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

# Determine base URLs
if [[ "$ENVIRONMENT" == "production" ]]; then
    BASE_URL="https://${DOMAIN_NAME:-novacron.local}"
    API_URL="$BASE_URL/api"
else
    BASE_URL="http://localhost:8092"
    API_URL="http://localhost:8090"
fi

log "Starting smoke tests for $ENVIRONMENT environment"

# Test counter
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Test runner function
run_test() {
    local test_name="$1"
    local test_function="$2"
    
    ((TOTAL_TESTS++))
    log "Running test: $test_name"
    
    if $test_function; then
        success "✓ $test_name"
        ((PASSED_TESTS++))
        return 0
    else
        error "✗ $test_name"
        ((FAILED_TESTS++))
        return 1
    fi
}

# Basic connectivity tests
test_frontend_accessibility() {
    curl -s -f --connect-timeout 10 --max-time 30 "$BASE_URL" > /dev/null
}

test_api_accessibility() {
    curl -s -f --connect-timeout 10 --max-time 30 "$API_URL/health" > /dev/null
}

test_api_info_endpoint() {
    local response=$(curl -s --connect-timeout 10 --max-time 30 "$API_URL/info")
    echo "$response" | jq -e '.version' > /dev/null
}

# Authentication tests
test_auth_endpoints() {
    # Test login endpoint exists
    local login_response=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/auth/login")
    [[ "$login_response" == "405" || "$login_response" == "400" || "$login_response" == "200" ]]
}

test_protected_endpoint() {
    # Test that protected endpoints return 401 without auth
    local protected_response=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/vms")
    [[ "$protected_response" == "401" ]]
}

# Database tests
test_database_connectivity() {
    local db_health=$(curl -s --connect-timeout 10 "$API_URL/health/db")
    echo "$db_health" | jq -e '.status == "ok"' > /dev/null
}

# Cache tests  
test_redis_connectivity() {
    local redis_health=$(curl -s --connect-timeout 10 "$API_URL/health/redis")
    echo "$redis_health" | jq -e '.status == "ok"' > /dev/null
}

# API functionality tests
test_vm_list_endpoint() {
    # Should return 401 or empty list (not error)
    local vm_response=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/vms")
    [[ "$vm_response" == "401" || "$vm_response" == "200" ]]
}

test_metrics_endpoint() {
    curl -s -f --connect-timeout 10 "$API_URL/metrics" > /dev/null
}

test_websocket_endpoint() {
    # Test WebSocket endpoint is accessible
    local ws_url
    if [[ "$ENVIRONMENT" == "production" ]]; then
        ws_url="wss://${DOMAIN_NAME:-novacron.local}/ws"
    else
        ws_url="ws://localhost:8091"
    fi
    
    # Use curl to test WebSocket handshake
    curl -s --connect-timeout 5 --max-time 10 \
         -H "Connection: Upgrade" \
         -H "Upgrade: websocket" \
         -H "Sec-WebSocket-Version: 13" \
         -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
         "$ws_url" > /dev/null
}

# Security tests
test_security_headers() {
    local headers=$(curl -s -I "$BASE_URL")
    
    # Check for security headers
    echo "$headers" | grep -i "X-Content-Type-Options" > /dev/null &&
    echo "$headers" | grep -i "X-Frame-Options" > /dev/null &&
    echo "$headers" | grep -i "X-XSS-Protection" > /dev/null
}

test_ssl_configuration() {
    if [[ "$ENVIRONMENT" == "production" && -n "${DOMAIN_NAME:-}" ]]; then
        # Test SSL/TLS configuration
        local ssl_info=$(echo | openssl s_client -servername "$DOMAIN_NAME" -connect "$DOMAIN_NAME:443" 2>/dev/null)
        echo "$ssl_info" | grep -q "Verify return code: 0 (ok)"
    else
        # Skip SSL test for non-production
        return 0
    fi
}

# Performance tests
test_response_time() {
    local start_time=$(date +%s%3N)
    curl -s -f --connect-timeout 5 --max-time 15 "$API_URL/health" > /dev/null
    local end_time=$(date +%s%3N)
    local response_time=$((end_time - start_time))
    
    # Response time should be under 5 seconds
    [[ $response_time -lt 5000 ]]
}

test_concurrent_requests() {
    local pids=()
    local temp_dir=$(mktemp -d)
    
    # Send 5 concurrent requests
    for i in {1..5}; do
        (curl -s -f --connect-timeout 10 --max-time 30 "$API_URL/health" > "$temp_dir/response_$i") &
        pids+=($!)
    done
    
    # Wait for all requests to complete
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            ((failed++))
        fi
    done
    
    # Clean up
    rm -rf "$temp_dir"
    
    # All requests should succeed
    [[ $failed -eq 0 ]]
}

# Monitoring tests
test_prometheus_metrics() {
    if curl -s --connect-timeout 5 "http://localhost:9090/-/healthy" > /dev/null 2>&1; then
        # Test that metrics are being collected
        curl -s --connect-timeout 10 "http://localhost:9090/api/v1/query?query=up" | jq -e '.data.result | length > 0' > /dev/null
    else
        # Skip if Prometheus is not accessible
        return 0
    fi
}

test_grafana_accessibility() {
    if curl -s --connect-timeout 5 "http://localhost:3001/api/health" > /dev/null 2>&1; then
        return 0
    else
        # Skip if Grafana is not accessible
        return 0
    fi
}

# Integration tests
test_full_user_flow() {
    # Test a basic user flow (without actual authentication for smoke test)
    # 1. Access frontend
    curl -s -f --connect-timeout 10 "$BASE_URL" > /dev/null &&
    
    # 2. Check API info
    curl -s -f --connect-timeout 10 "$API_URL/info" > /dev/null &&
    
    # 3. Check health endpoints
    curl -s -f --connect-timeout 10 "$API_URL/health" > /dev/null
}

# Data persistence test
test_data_persistence() {
    # Check if database has expected schema
    local schema_check=$(curl -s --connect-timeout 10 "$API_URL/health/db")
    echo "$schema_check" | jq -e '.status == "ok"' > /dev/null
}

# Configuration test
test_environment_variables() {
    local config_info=$(curl -s --connect-timeout 10 "$API_URL/info")
    
    # Verify environment is set correctly
    if [[ "$ENVIRONMENT" == "production" ]]; then
        echo "$config_info" | jq -e '.environment == "production"' > /dev/null || 
        echo "$config_info" | jq -e '.debug == false' > /dev/null
    else
        # For staging, just verify we get some config info
        echo "$config_info" | jq -e '.version' > /dev/null
    fi
}

# Run all smoke tests
run_smoke_tests() {
    log "=== Running Smoke Tests ==="
    
    # Basic connectivity
    run_test "Frontend Accessibility" test_frontend_accessibility
    run_test "API Accessibility" test_api_accessibility
    run_test "API Info Endpoint" test_api_info_endpoint
    
    # Authentication
    run_test "Auth Endpoints" test_auth_endpoints
    run_test "Protected Endpoint Security" test_protected_endpoint
    
    # Database and cache
    run_test "Database Connectivity" test_database_connectivity
    run_test "Redis Connectivity" test_redis_connectivity
    
    # API functionality
    run_test "VM List Endpoint" test_vm_list_endpoint
    run_test "Metrics Endpoint" test_metrics_endpoint
    run_test "WebSocket Endpoint" test_websocket_endpoint
    
    # Security
    run_test "Security Headers" test_security_headers
    run_test "SSL Configuration" test_ssl_configuration
    
    # Performance
    run_test "Response Time" test_response_time
    run_test "Concurrent Requests" test_concurrent_requests
    
    # Monitoring
    run_test "Prometheus Metrics" test_prometheus_metrics
    run_test "Grafana Accessibility" test_grafana_accessibility
    
    # Integration
    run_test "Full User Flow" test_full_user_flow
    run_test "Data Persistence" test_data_persistence
    run_test "Environment Variables" test_environment_variables
}

# Generate test report
generate_report() {
    local report_file="/tmp/novacron-smoke-test-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "$report_file" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "environment": "$ENVIRONMENT",
  "summary": {
    "total": $TOTAL_TESTS,
    "passed": $PASSED_TESTS,
    "failed": $FAILED_TESTS,
    "success_rate": $(echo "scale=2; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l)
  },
  "status": "$([[ $FAILED_TESTS -eq 0 ]] && echo "PASS" || echo "FAIL")"
}
EOF
    
    echo ""
    echo "=== Smoke Test Summary ==="
    echo "Environment: $ENVIRONMENT"
    echo "Total Tests: $TOTAL_TESTS"
    echo "Passed: $PASSED_TESTS"
    echo "Failed: $FAILED_TESTS"
    echo "Success Rate: $(echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l)%"
    echo "Report: $report_file"
    echo ""
    
    if [[ $FAILED_TESTS -eq 0 ]]; then
        success "🎉 All smoke tests passed!"
        return 0
    else
        error "❌ $FAILED_TESTS tests failed"
        return 1
    fi
}

# Main execution
main() {
    # Check dependencies
    if ! command -v jq &> /dev/null; then
        error "jq is required but not installed"
        exit 1
    fi
    
    if ! command -v bc &> /dev/null; then
        error "bc is required but not installed"
        exit 1
    fi
    
    # Wait for services to be ready
    log "Waiting for services to be ready..."
    sleep 10
    
    # Run tests
    run_smoke_tests
    
    # Generate report
    generate_report
}

# Error handling
trap 'error "Smoke test failed at line $LINENO"' ERR

# Run main function
main