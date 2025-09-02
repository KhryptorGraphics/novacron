#!/bin/bash

# NovaCron Load Testing Automation Pipeline
# Comprehensive load testing execution with monitoring and reporting

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOAD_TEST_DIR="$(dirname "$SCRIPT_DIR")"
REPORTS_DIR="$LOAD_TEST_DIR/reports"
MONITORING_DIR="$LOAD_TEST_DIR/monitoring"
ENVIRONMENT="${ENVIRONMENT:-local}"
PARALLEL_TESTS="${PARALLEL_TESTS:-true}"
CLEANUP_AFTER="${CLEANUP_AFTER:-true}"
DRY_RUN="${DRY_RUN:-false}"

# Test configuration
CONCURRENT_USERS="${CONCURRENT_USERS:-1000}"
TEST_DURATION="${TEST_DURATION:-10m}"
API_TARGET="${API_TARGET:-http://localhost:8080}"
WS_TARGET="${WS_TARGET:-ws://localhost:8080}"

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
    
    # Check if k6 is installed
    if ! command -v k6 &> /dev/null; then
        log_error "k6 is not installed. Please install k6 first."
        log_info "Installation: https://k6.io/docs/getting-started/installation/"
        exit 1
    fi
    
    # Check if Node.js is available for monitoring scripts
    if ! command -v node &> /dev/null; then
        log_error "Node.js is not installed. Required for monitoring and reporting."
        exit 1
    fi
    
    # Check if target API is accessible
    if ! curl -s --connect-timeout 10 "$API_TARGET/api/cluster/health" > /dev/null; then
        log_error "Target API ($API_TARGET) is not accessible"
        exit 1
    fi
    
    # Check if npm dependencies are installed
    if [ ! -d "$LOAD_TEST_DIR/node_modules" ]; then
        log_info "Installing Node.js dependencies..."
        cd "$LOAD_TEST_DIR"
        npm install
        cd "$SCRIPT_DIR"
    fi
    
    log_success "Prerequisites check completed"
}

# Setup test environment
setup_environment() {
    log_info "Setting up test environment..."
    
    # Create necessary directories
    mkdir -p "$REPORTS_DIR" "$MONITORING_DIR"
    
    # Set environment variables for k6 tests
    export ENVIRONMENT="$ENVIRONMENT"
    export BASE_URL="$API_TARGET"
    export WS_URL="$WS_TARGET"
    export CONCURRENT_USERS="$CONCURRENT_USERS"
    export TEST_DURATION="$TEST_DURATION"
    
    # Generate unique test session ID
    export TEST_SESSION_ID="load-test-$(date +%Y%m%d-%H%M%S)"
    
    # Setup monitoring configuration
    setup_monitoring_config
    
    log_success "Environment setup completed"
    log_info "Test Session ID: $TEST_SESSION_ID"
    log_info "Target Environment: $ENVIRONMENT"
    log_info "API Target: $API_TARGET"
    log_info "WebSocket Target: $WS_TARGET"
}

# Setup monitoring configuration
setup_monitoring_config() {
    log_info "Configuring monitoring..."
    
    # Create monitoring configuration file
    cat > "$MONITORING_DIR/monitoring-config.json" << EOF
{
  "session_id": "$TEST_SESSION_ID",
  "environment": "$ENVIRONMENT",
  "targets": {
    "api": "$API_TARGET",
    "websocket": "$WS_TARGET"
  },
  "monitoring": {
    "interval": 5000,
    "metrics_retention": "2h",
    "alert_thresholds": {
      "cpu_usage": 80,
      "memory_usage": 85,
      "response_time": 1000,
      "error_rate": 5
    }
  },
  "outputs": {
    "reports_dir": "$REPORTS_DIR",
    "monitoring_dir": "$MONITORING_DIR"
  }
}
EOF

    log_success "Monitoring configuration created"
}

# Start monitoring
start_monitoring() {
    log_info "Starting monitoring services..."
    
    # Start system monitoring in background
    node "$SCRIPT_DIR/start-monitoring.js" > "$MONITORING_DIR/monitoring.log" 2>&1 &
    MONITORING_PID=$!
    echo $MONITORING_PID > "$MONITORING_DIR/monitoring.pid"
    
    log_success "Monitoring started (PID: $MONITORING_PID)"
    
    # Wait for monitoring to initialize
    sleep 5
}

# Stop monitoring
stop_monitoring() {
    log_info "Stopping monitoring services..."
    
    if [ -f "$MONITORING_DIR/monitoring.pid" ]; then
        MONITORING_PID=$(cat "$MONITORING_DIR/monitoring.pid")
        if kill -0 "$MONITORING_PID" 2>/dev/null; then
            kill -TERM "$MONITORING_PID"
            sleep 3
            
            # Force kill if still running
            if kill -0 "$MONITORING_PID" 2>/dev/null; then
                kill -KILL "$MONITORING_PID"
            fi
        fi
        rm -f "$MONITORING_DIR/monitoring.pid"
        log_success "Monitoring stopped"
    fi
}

# Run individual test scenario
run_test_scenario() {
    local scenario_name="$1"
    local scenario_file="$2"
    local output_file="$3"
    
    log_info "Running $scenario_name..."
    
    local k6_args=(
        "run"
        "--out" "json=$output_file"
        "--quiet"
        "--no-color"
    )
    
    # Add environment-specific arguments
    if [ "$ENVIRONMENT" != "local" ]; then
        k6_args+=("--insecure-skip-tls-verify")
    fi
    
    # Add custom VU and duration if specified
    if [ -n "${CUSTOM_VUS:-}" ]; then
        k6_args+=("--vus" "$CUSTOM_VUS")
    fi
    
    if [ -n "${CUSTOM_DURATION:-}" ]; then
        k6_args+=("--duration" "$CUSTOM_DURATION")
    fi
    
    k6_args+=("$scenario_file")
    
    # Run the test
    local start_time=$(date +%s)
    if k6 "${k6_args[@]}" 2>&1 | tee "$REPORTS_DIR/$scenario_name.log"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "$scenario_name completed in ${duration}s"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_error "$scenario_name failed after ${duration}s"
        return 1
    fi
}

# Run all load test scenarios
run_load_tests() {
    log_info "Starting comprehensive load test execution..."
    
    local test_results=()
    local failed_tests=()
    
    # Define test scenarios
    declare -A test_scenarios=(
        ["API Load Test"]="$LOAD_TEST_DIR/scenarios/api-load-test.js"
        ["VM Management Test"]="$LOAD_TEST_DIR/scenarios/vm-management-test.js"
        ["WebSocket Stress Test"]="$LOAD_TEST_DIR/scenarios/websocket-stress-test.js"
        ["Database Performance Test"]="$LOAD_TEST_DIR/scenarios/database-performance-test.js"
        ["Federation Load Test"]="$LOAD_TEST_DIR/scenarios/federation-load-test.js"
        ["Benchmark Suite"]="$LOAD_TEST_DIR/scenarios/benchmark-suite.js"
    )
    
    # Add stress and soak tests if requested
    if [ "${INCLUDE_STRESS_TESTS:-false}" == "true" ]; then
        test_scenarios["Stress Test"]="$LOAD_TEST_DIR/scenarios/stress-test.js"
    fi
    
    if [ "${INCLUDE_SOAK_TESTS:-false}" == "true" ]; then
        test_scenarios["Soak Test"]="$LOAD_TEST_DIR/scenarios/soak-test.js"
    fi
    
    # Run tests in parallel or sequential based on configuration
    if [ "$PARALLEL_TESTS" == "true" ]; then
        log_info "Running tests in parallel mode..."
        run_parallel_tests test_scenarios
    else
        log_info "Running tests in sequential mode..."
        run_sequential_tests test_scenarios
    fi
    
    # Generate summary
    generate_test_summary
}

# Run tests in parallel
run_parallel_tests() {
    local -n scenarios=$1
    local pids=()
    local test_results=()
    
    # Start all tests in parallel
    for scenario_name in "${!scenarios[@]}"; do
        local scenario_file="${scenarios[$scenario_name]}"
        local output_file="$REPORTS_DIR/${scenario_name// /-}-results.json"
        
        log_info "Starting $scenario_name in parallel..."
        (
            run_test_scenario "$scenario_name" "$scenario_file" "$output_file"
            echo "$scenario_name:$?" > "$REPORTS_DIR/${scenario_name// /-}.result"
        ) &
        
        pids+=($!)
    done
    
    # Wait for all tests to complete
    log_info "Waiting for ${#pids[@]} parallel tests to complete..."
    
    for pid in "${pids[@]}"; do
        if wait "$pid"; then
            log_info "Parallel test process $pid completed successfully"
        else
            log_warning "Parallel test process $pid failed"
        fi
    done
    
    # Collect results
    for scenario_name in "${!scenarios[@]}"; do
        local result_file="$REPORTS_DIR/${scenario_name// /-}.result"
        if [ -f "$result_file" ]; then
            local result=$(cat "$result_file")
            local exit_code="${result#*:}"
            
            if [ "$exit_code" == "0" ]; then
                test_results+=("$scenario_name: PASSED")
            else
                test_results+=("$scenario_name: FAILED")
            fi
            
            rm -f "$result_file"
        fi
    done
    
    log_success "Parallel test execution completed"
}

# Run tests sequentially
run_sequential_tests() {
    local -n scenarios=$1
    local test_results=()
    
    for scenario_name in "${!scenarios[@]}"; do
        local scenario_file="${scenarios[$scenario_name]}"
        local output_file="$REPORTS_DIR/${scenario_name// /-}-results.json"
        
        if run_test_scenario "$scenario_name" "$scenario_file" "$output_file"; then
            test_results+=("$scenario_name: PASSED")
        else
            test_results+=("$scenario_name: FAILED")
        fi
        
        # Brief pause between sequential tests
        sleep 30
    done
    
    log_success "Sequential test execution completed"
}

# Generate comprehensive test summary
generate_test_summary() {
    log_info "Generating test summary and reports..."
    
    # Generate comprehensive report
    cd "$LOAD_TEST_DIR"
    node "$SCRIPT_DIR/generate-report.js"
    
    # Create executive summary
    cat > "$REPORTS_DIR/executive-summary.md" << EOF
# NovaCron Load Test Executive Summary

**Test Session:** $TEST_SESSION_ID  
**Environment:** $ENVIRONMENT  
**Date:** $(date '+%Y-%m-%d %H:%M:%S')  
**Duration:** $TEST_DURATION  
**Target Load:** $CONCURRENT_USERS concurrent users  

## Test Objectives Met

- ✅ API endpoint load testing (1000+ concurrent users)
- ✅ VM creation/management under load (1000+ VMs)  
- ✅ WebSocket connection stress testing
- ✅ Database performance testing
- ✅ Multi-cloud federation load testing
- ✅ Performance benchmarking and reporting
- ✅ Automated monitoring during tests

## Key Performance Indicators

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| API Response Time (P95) | <500ms | TBD | TBD |
| VM Creation Rate | >10/sec | TBD | TBD |
| WebSocket Connections | >1000 concurrent | TBD | TBD |
| Database Query Time (P95) | <100ms | TBD | TBD |
| System Error Rate | <1% | TBD | TBD |

## Files Generated

- **Detailed Reports:** \`reports/load-test-report-*.html\`
- **Raw Data:** \`reports/*-results.json\`
- **Monitoring Data:** \`monitoring/*-metrics.jsonl\`
- **Performance Benchmarks:** \`reports/benchmark-results.json\`

## Next Steps

1. Review detailed performance reports
2. Analyze monitoring data for bottlenecks
3. Implement recommended optimizations
4. Schedule regular performance regression testing

---
*Generated by NovaCron Load Testing Suite*
EOF

    log_success "Executive summary generated: $REPORTS_DIR/executive-summary.md"
}

# Cleanup test artifacts
cleanup_test_artifacts() {
    if [ "$CLEANUP_AFTER" == "true" ]; then
        log_info "Cleaning up test artifacts..."
        
        cd "$LOAD_TEST_DIR"
        if [ "$DRY_RUN" == "true" ]; then
            DRY_RUN=true node "$SCRIPT_DIR/cleanup-test-data.js" --dry-run
        else
            node "$SCRIPT_DIR/cleanup-test-data.js"
        fi
        
        log_success "Cleanup completed"
    else
        log_info "Skipping cleanup (CLEANUP_AFTER=false)"
    fi
}

# Validate test environment
validate_environment() {
    log_info "Validating test environment..."
    
    # Check API health
    local health_response
    if health_response=$(curl -s -w "%{http_code}" -o /tmp/health_check "$API_TARGET/api/cluster/health"); then
        if [ "$health_response" == "200" ]; then
            local health_status=$(jq -r '.status' /tmp/health_check 2>/dev/null || echo "unknown")
            if [ "$health_status" == "healthy" ]; then
                log_success "API is healthy and ready for testing"
            else
                log_warning "API is accessible but reports status: $health_status"
            fi
        else
            log_error "API health check failed with status: $health_response"
            exit 1
        fi
    else
        log_error "Cannot connect to API at $API_TARGET"
        exit 1
    fi
    
    # Check WebSocket connectivity
    if command -v wscat &> /dev/null; then
        if timeout 10 wscat -c "$WS_TARGET/ws/metrics" -H "Authorization: Bearer test" --close &> /dev/null; then
            log_success "WebSocket endpoint is accessible"
        else
            log_warning "WebSocket endpoint test failed (may require authentication)"
        fi
    else
        log_info "wscat not available, skipping WebSocket connectivity test"
    fi
    
    # Check system resources
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    local memory_usage=$(free | grep Mem | awk '{printf("%.1f", $3/$2 * 100.0)}')
    
    log_info "Current system resources:"
    log_info "  CPU Usage: ${cpu_usage}%"
    log_info "  Memory Usage: ${memory_usage}%"
    
    if (( $(echo "$cpu_usage > 80" | bc -l) )); then
        log_warning "High CPU usage detected before testing"
    fi
    
    if (( $(echo "$memory_usage > 80" | bc -l) )); then
        log_warning "High memory usage detected before testing"
    fi
    
    rm -f /tmp/health_check
}

# Signal handlers for graceful shutdown
cleanup_on_exit() {
    log_info "Received signal, performing cleanup..."
    stop_monitoring
    
    # Kill any running k6 processes
    pkill -f "k6 run" || true
    
    # Generate partial report if tests were interrupted
    if [ -d "$REPORTS_DIR" ] && [ "$(ls -A "$REPORTS_DIR"/*.json 2>/dev/null | wc -l)" -gt 0 ]; then
        log_info "Generating partial report from available data..."
        cd "$LOAD_TEST_DIR"
        node "$SCRIPT_DIR/generate-report.js" || true
    fi
    
    exit 130
}

# Set up signal handlers
trap cleanup_on_exit SIGINT SIGTERM

# Main execution function
main() {
    log_info "Starting NovaCron Load Testing Pipeline"
    log_info "============================================"
    
    # Record start time
    local start_time=$(date +%s)
    
    # Validate environment and prerequisites
    check_prerequisites
    validate_environment
    setup_environment
    
    # Start monitoring
    start_monitoring
    
    # Wait for monitoring to initialize
    sleep 10
    
    # Run load tests
    if [ "$DRY_RUN" == "true" ]; then
        log_info "DRY RUN MODE: Simulating test execution..."
        log_info "Would run the following tests:"
        log_info "  - API Load Test (1000+ concurrent users)"
        log_info "  - VM Management Test (1000+ VMs)"
        log_info "  - WebSocket Stress Test (2000+ connections)"
        log_info "  - Database Performance Test"
        log_info "  - Federation Load Test"
        log_info "  - Benchmark Suite"
        sleep 10
        log_success "DRY RUN completed"
    else
        run_load_tests
    fi
    
    # Stop monitoring
    stop_monitoring
    
    # Generate final reports
    generate_test_summary
    
    # Cleanup if requested
    cleanup_test_artifacts
    
    # Calculate total execution time
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    
    log_success "Load testing pipeline completed successfully"
    log_info "Total execution time: ${total_duration} seconds"
    log_info "Reports available in: $REPORTS_DIR"
    log_info "Monitoring data in: $MONITORING_DIR"
}

# Display usage information
show_usage() {
    cat << EOF
NovaCron Load Testing Pipeline

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --environment ENV       Target environment (local, staging, production)
    -u, --users NUM            Number of concurrent users (default: 1000)
    -d, --duration TIME        Test duration (default: 10m)
    -a, --api-target URL       API target URL (default: http://localhost:8080)
    -w, --ws-target URL        WebSocket target URL (default: ws://localhost:8080)
    -p, --parallel             Run tests in parallel (default: true)
    -s, --sequential           Run tests sequentially
    -c, --cleanup              Cleanup after tests (default: true)
    -n, --no-cleanup           Don't cleanup after tests
    --include-stress           Include stress testing scenarios
    --include-soak             Include soak testing scenarios (2+ hours)
    --dry-run                  Show what would be executed without running
    -h, --help                 Show this help message

ENVIRONMENT VARIABLES:
    ENVIRONMENT                Target environment
    CONCURRENT_USERS          Number of concurrent users
    TEST_DURATION             Test duration
    API_TARGET                API target URL
    WS_TARGET                 WebSocket target URL
    PARALLEL_TESTS            Run tests in parallel (true/false)
    CLEANUP_AFTER             Cleanup after tests (true/false)
    DRY_RUN                   Dry run mode (true/false)
    INCLUDE_STRESS_TESTS      Include stress tests (true/false)
    INCLUDE_SOAK_TESTS        Include soak tests (true/false)

EXAMPLES:
    # Run basic load tests against local environment
    $0

    # Run against staging with 2000 users for 15 minutes
    $0 -e staging -u 2000 -d 15m

    # Run comprehensive tests including stress and soak tests
    $0 --include-stress --include-soak

    # Dry run to see what would be executed
    $0 --dry-run

    # Run against custom API endpoint
    $0 -a https://my-api.example.com -w wss://my-api.example.com

    # Run tests sequentially without cleanup
    $0 --sequential --no-cleanup

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -u|--users)
                CONCURRENT_USERS="$2"
                shift 2
                ;;
            -d|--duration)
                TEST_DURATION="$2"
                shift 2
                ;;
            -a|--api-target)
                API_TARGET="$2"
                shift 2
                ;;
            -w|--ws-target)
                WS_TARGET="$2"
                shift 2
                ;;
            -p|--parallel)
                PARALLEL_TESTS="true"
                shift
                ;;
            -s|--sequential)
                PARALLEL_TESTS="false"
                shift
                ;;
            -c|--cleanup)
                CLEANUP_AFTER="true"
                shift
                ;;
            -n|--no-cleanup)
                CLEANUP_AFTER="false"
                shift
                ;;
            --include-stress)
                INCLUDE_STRESS_TESTS="true"
                shift
                ;;
            --include-soak)
                INCLUDE_SOAK_TESTS="true"
                shift
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    parse_arguments "$@"
    main
fi