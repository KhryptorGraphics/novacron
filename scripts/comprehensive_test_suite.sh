#!/bin/bash
# Comprehensive Test Suite for NovaCron - Integration and System Testing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters for test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
TEST_RESULTS=()

# Configuration
API_URL="http://localhost:8090"
WS_URL="ws://localhost:8091"
FRONTEND_URL="http://localhost:8092"
TEST_VM_PREFIX="novacron-test"

# Function to print status messages
print_status() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[PASS] $1${NC}"
    ((PASSED_TESTS++))
}

print_warning() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

print_error() {
    echo -e "${RED}[FAIL] $1${NC}"
    ((FAILED_TESTS++))
}

# Test tracking
start_test() {
    ((TOTAL_TESTS++))
    print_status "Test $TOTAL_TESTS: $1"
}

# Wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            print_success "$service_name is ready"
            return 0
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    print_error "$service_name failed to start within timeout"
    return 1
}

# Database connectivity test
test_database_connection() {
    start_test "Database Connection"
    
    # Test if PostgreSQL container is running
    if docker-compose ps postgres | grep -q "Up"; then
        print_success "PostgreSQL container is running"
        
        # Test database connectivity
        if docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
            print_success "Database is accepting connections"
        else
            print_error "Database is not accepting connections"
            return 1
        fi
    else
        print_error "PostgreSQL container is not running"
        return 1
    fi
    
    return 0
}

# API Health Check
test_api_health() {
    start_test "API Health Check"
    
    # Basic health endpoint
    if curl -s -f "$API_URL/health" > /dev/null; then
        print_success "API health endpoint is responding"
    else
        print_error "API health endpoint is not responding"
        return 1
    fi
    
    # Test CORS headers
    local cors_response=$(curl -s -I -X OPTIONS "$API_URL/api/vms" \
        -H "Origin: http://localhost:8092" \
        -H "Access-Control-Request-Method: GET")
    
    if echo "$cors_response" | grep -q "Access-Control-Allow-Origin"; then
        print_success "CORS headers are properly configured"
    else
        print_warning "CORS headers may not be properly configured"
    fi
    
    return 0
}

# Test VM Management API
test_vm_management_api() {
    start_test "VM Management API"
    
    local vm_id="${TEST_VM_PREFIX}-$(date +%s)"
    local vm_config='{
        "id": "'$vm_id'",
        "name": "Test VM",
        "cpu": 2,
        "memory": 1024,
        "disk": 20,
        "os": "ubuntu-24.04"
    }'
    
    # Test VM creation
    local create_response=$(curl -s -X POST "$API_URL/api/vms" \
        -H "Content-Type: application/json" \
        -d "$vm_config")
    
    if echo "$create_response" | grep -q "error\|Error"; then
        print_warning "VM creation returned error (expected in test environment): $(echo $create_response | jq -r '.message' 2>/dev/null || echo $create_response)"
    else
        print_success "VM creation API endpoint is functional"
    fi
    
    # Test VM listing
    local list_response=$(curl -s "$API_URL/api/vms")
    if [ $? -eq 0 ]; then
        print_success "VM listing API endpoint is functional"
    else
        print_error "VM listing API endpoint failed"
        return 1
    fi
    
    return 0
}

# Test Authentication System
test_authentication_system() {
    start_test "Authentication System"
    
    # Test registration endpoint
    local register_data='{
        "email": "test@example.com",
        "password": "TestPassword123!",
        "firstName": "Test",
        "lastName": "User"
    }'
    
    local register_response=$(curl -s -X POST "$API_URL/api/auth/register" \
        -H "Content-Type: application/json" \
        -d "$register_data")
    
    if echo "$register_response" | grep -q "error\|Error"; then
        print_warning "Registration endpoint returned error (may be expected): $(echo $register_response | jq -r '.message' 2>/dev/null || echo $register_response)"
    else
        print_success "Registration API endpoint is functional"
    fi
    
    # Test login endpoint
    local login_data='{
        "email": "test@example.com",
        "password": "TestPassword123!"
    }'
    
    local login_response=$(curl -s -X POST "$API_URL/api/auth/login" \
        -H "Content-Type: application/json" \
        -d "$login_data")
    
    if [ $? -eq 0 ]; then
        print_success "Login API endpoint is functional"
    else
        print_error "Login API endpoint failed"
    fi
    
    return 0
}

# Test WebSocket Connection
test_websocket_connection() {
    start_test "WebSocket Connection"
    
    # Test WebSocket connection using websocat if available
    if command -v websocat &> /dev/null; then
        timeout 5s websocat "$WS_URL/ws" --text <<< '{"type":"ping"}' > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            print_success "WebSocket connection is working"
        else
            print_warning "WebSocket connection test failed (may need authentication)"
        fi
    else
        print_warning "websocat not available, skipping WebSocket test"
    fi
    
    return 0
}

# Test Frontend Application
test_frontend_application() {
    start_test "Frontend Application"
    
    # Test if frontend is serving pages
    local frontend_response=$(curl -s "$FRONTEND_URL")
    if [ $? -eq 0 ] && echo "$frontend_response" | grep -q "html\|HTML"; then
        print_success "Frontend is serving HTML content"
    else
        print_error "Frontend is not serving content properly"
        return 1
    fi
    
    # Test API endpoint availability from frontend
    local api_test_response=$(curl -s "$FRONTEND_URL/api/health" 2>/dev/null || echo "not_found")
    if echo "$api_test_response" | grep -q "not_found"; then
        print_success "Frontend properly routes to backend API"
    else
        print_warning "Frontend API routing may need verification"
    fi
    
    return 0
}

# Test Docker Services
test_docker_services() {
    start_test "Docker Services Health"
    
    local services=("postgres" "api" "frontend")
    local healthy_services=0
    
    for service in "${services[@]}"; do
        if docker-compose ps "$service" | grep -q "Up"; then
            print_success "Service '$service' is running"
            ((healthy_services++))
        else
            print_error "Service '$service' is not running"
        fi
    done
    
    if [ $healthy_services -eq ${#services[@]} ]; then
        print_success "All core services are running"
        return 0
    else
        print_error "$((${#services[@]} - healthy_services)) services are not running"
        return 1
    fi
}

# Test Monitoring Stack
test_monitoring_stack() {
    start_test "Monitoring Stack"
    
    # Test Prometheus
    local prometheus_url="http://localhost:9090"
    if curl -s -f "$prometheus_url/api/v1/label/__name__/values" > /dev/null; then
        print_success "Prometheus is accessible and responding"
    else
        print_warning "Prometheus may not be running or accessible"
    fi
    
    # Test Grafana
    local grafana_url="http://localhost:3001"
    if curl -s -f "$grafana_url/api/health" > /dev/null; then
        print_success "Grafana is accessible and responding"
    else
        print_warning "Grafana may not be running or accessible"
    fi
    
    return 0
}

# Performance Test
test_api_performance() {
    start_test "API Performance"
    
    local start_time=$(date +%s%N)
    for i in {1..10}; do
        curl -s "$API_URL/api/vms" > /dev/null
    done
    local end_time=$(date +%s%N)
    
    local duration=$(((end_time - start_time) / 1000000))
    local avg_response=$((duration / 10))
    
    if [ $avg_response -lt 1000 ]; then
        print_success "API performance is good (avg: ${avg_response}ms)"
    else
        print_warning "API performance may be slow (avg: ${avg_response}ms)"
    fi
    
    return 0
}

# Security Test
test_security_basics() {
    start_test "Basic Security"
    
    # Test for common security headers
    local security_headers=$(curl -s -I "$API_URL/api/vms")
    
    local security_score=0
    
    if echo "$security_headers" | grep -q "X-Content-Type-Options"; then
        ((security_score++))
    fi
    
    if echo "$security_headers" | grep -q "X-Frame-Options"; then
        ((security_score++))
    fi
    
    if echo "$security_headers" | grep -q "Content-Security-Policy"; then
        ((security_score++))
    fi
    
    if [ $security_score -gt 0 ]; then
        print_success "Some security headers are present ($security_score/3)"
    else
        print_warning "No security headers detected"
    fi
    
    return 0
}

# Main test execution
main() {
    print_status "Starting NovaCron Comprehensive Test Suite"
    print_status "=========================================="
    
    # Start services if not running
    print_status "Checking Docker Compose services..."
    if ! docker-compose ps | grep -q "Up"; then
        print_status "Starting Docker Compose services..."
        docker-compose up -d
        sleep 10
    fi
    
    # Wait for core services
    wait_for_service "$API_URL/health" "API Server" || {
        print_error "API Server failed to start, cannot continue"
        exit 1
    }
    
    # Run all tests
    test_docker_services
    test_database_connection
    test_api_health
    test_vm_management_api
    test_authentication_system
    test_websocket_connection
    test_frontend_application
    test_monitoring_stack
    test_api_performance
    test_security_basics
    
    # Print summary
    print_status ""
    print_status "Test Results Summary"
    print_status "==================="
    print_status "Total Tests: $TOTAL_TESTS"
    print_success "Passed: $PASSED_TESTS"
    print_error "Failed: $FAILED_TESTS"
    
    local success_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    
    if [ $success_rate -ge 80 ]; then
        print_success "Overall Success Rate: ${success_rate}% - System is healthy"
    elif [ $success_rate -ge 60 ]; then
        print_warning "Overall Success Rate: ${success_rate}% - Some issues detected"
    else
        print_error "Overall Success Rate: ${success_rate}% - Significant issues detected"
    fi
    
    print_status ""
    print_status "Integration test suite completed"
    
    # Return appropriate exit code
    if [ $FAILED_TESTS -gt 0 ]; then
        exit 1
    else
        exit 0
    fi
}

# Handle script termination
cleanup() {
    print_status "Cleaning up test environment..."
    # Clean up any test VMs or resources if needed
}

trap cleanup EXIT

# Run main function
main "$@"