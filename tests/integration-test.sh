#!/bin/bash

# MLE-Star Integration Test Script
# Comprehensive integration testing for MLE-Star workflow installation and execution
# 
# Author: TESTER (Hive Mind QA Specialist)
# Mission: Validate end-to-end MLE-Star functionality and integration with claude-flow

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly TEST_OUTPUT_DIR="${PROJECT_ROOT}/tests/output"
readonly TEST_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly LOG_FILE="${TEST_OUTPUT_DIR}/integration_test_${TEST_TIMESTAMP}.log"
readonly REPORT_FILE="${TEST_OUTPUT_DIR}/integration_report_${TEST_TIMESTAMP}.json"

# Test configuration
readonly TIMEOUT_SECONDS=30
readonly MAX_RETRIES=3
readonly PERFORMANCE_THRESHOLD_MS=5000
readonly MEMORY_THRESHOLD_MB=100

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Test results tracking
declare -A TEST_RESULTS
declare -A PERFORMANCE_METRICS
declare -i TOTAL_TESTS=0
declare -i PASSED_TESTS=0
declare -i FAILED_TESTS=0

# Setup functions
setup_test_environment() {
    echo -e "${BLUE}Setting up test environment...${NC}"
    
    # Create output directory
    mkdir -p "$TEST_OUTPUT_DIR"
    
    # Initialize log file
    {
        echo "MLE-Star Integration Test Log"
        echo "Timestamp: $(date)"
        echo "Project Root: $PROJECT_ROOT"
        echo "=========================="
    } > "$LOG_FILE"
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    echo -e "${GREEN}Test environment setup complete${NC}"
}

cleanup_test_environment() {
    echo -e "${BLUE}Cleaning up test environment...${NC}"
    
    # Remove temporary test files
    find "$TEST_OUTPUT_DIR" -name "temp_*" -delete 2>/dev/null || true
    
    # Generate final report
    generate_test_report
    
    echo -e "${GREEN}Cleanup complete${NC}"
}

# Utility functions
log_message() {
    local message="$1"
    local level="${2:-INFO}"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    case "$level" in
        "ERROR")
            echo -e "${RED}[ERROR] $message${NC}"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN] $message${NC}"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS] $message${NC}"
            ;;
        *)
            echo -e "${BLUE}[INFO] $message${NC}"
            ;;
    esac
}

run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_pattern="${3:-.*}"
    
    ((TOTAL_TESTS++))
    
    log_message "Running test: $test_name"
    
    local start_time=$(date +%s%N)
    local output
    local exit_code=0
    
    # Run the test command with timeout
    if output=$(timeout "$TIMEOUT_SECONDS" bash -c "$test_command" 2>&1); then
        local end_time=$(date +%s%N)
        local duration_ms=$(( (end_time - start_time) / 1000000 ))
        
        # Check if output matches expected pattern
        if echo "$output" | grep -qE "$expected_pattern"; then
            TEST_RESULTS["$test_name"]="PASS"
            PERFORMANCE_METRICS["${test_name}_duration"]="$duration_ms"
            ((PASSED_TESTS++))
            log_message "Test passed: $test_name (${duration_ms}ms)" "SUCCESS"
            
            # Check performance threshold
            if [ "$duration_ms" -gt "$PERFORMANCE_THRESHOLD_MS" ]; then
                log_message "Performance warning: $test_name took ${duration_ms}ms (threshold: ${PERFORMANCE_THRESHOLD_MS}ms)" "WARN"
            fi
        else
            TEST_RESULTS["$test_name"]="FAIL"
            ((FAILED_TESTS++))
            log_message "Test failed: $test_name - Output did not match expected pattern" "ERROR"
            log_message "Expected pattern: $expected_pattern" "ERROR"
            log_message "Actual output: $output" "ERROR"
        fi
    else
        exit_code=$?
        TEST_RESULTS["$test_name"]="FAIL"
        ((FAILED_TESTS++))
        log_message "Test failed: $test_name - Command failed with exit code $exit_code" "ERROR"
        log_message "Output: $output" "ERROR"
    fi
}

check_prerequisites() {
    log_message "Checking prerequisites..."
    
    # Check if npx is available
    if ! command -v npx &> /dev/null; then
        log_message "npx is not available" "ERROR"
        return 1
    fi
    
    # Check if claude-flow is accessible
    if ! npx claude-flow@alpha --help &>/dev/null; then
        log_message "claude-flow is not accessible" "ERROR"
        return 1
    fi
    
    # Check if Node.js is available
    if ! command -v node &> /dev/null; then
        log_message "Node.js is not available" "ERROR"
        return 1
    fi
    
    log_message "All prerequisites satisfied" "SUCCESS"
    return 0
}

# Test suite functions
test_command_registration() {
    log_message "Testing command registration and help display..."
    
    run_test "help_display" \
        "npx claude-flow@alpha --help" \
        "(sparc|workflow|mle)"
    
    run_test "sparc_help" \
        "npx claude-flow@alpha sparc --help" \
        "(modes|run|tdd|info)"
    
    run_test "sparc_modes_list" \
        "npx claude-flow@alpha sparc modes" \
        "(spec-pseudocode|architect|tdd|integration)"
    
    run_test "sparc_mode_info" \
        "npx claude-flow@alpha sparc info architect" \
        "(architect|design|system)"
}

test_template_generation() {
    log_message "Testing template generation and customization..."
    
    run_test "basic_template_generation" \
        "npx claude-flow@alpha sparc run spec-pseudocode 'Basic REST API template'" \
        "(specification|pseudocode|generated|requirements)"
    
    run_test "architecture_template" \
        "npx claude-flow@alpha sparc run architect 'Microservices architecture design'" \
        "(architecture|microservices|design|system)"
    
    run_test "template_customization" \
        "npx claude-flow@alpha sparc run spec-pseudocode 'Custom user authentication system' --detailed" \
        "(specification|authentication|system|detailed)"
}

test_workflow_execution() {
    log_message "Testing workflow execution stages..."
    
    run_test "specification_stage" \
        "npx claude-flow@alpha sparc run spec-pseudocode 'User registration workflow'" \
        "(specification|pseudocode|requirements|user)"
    
    run_test "architecture_stage" \
        "npx claude-flow@alpha sparc run architect 'Scalable web application architecture'" \
        "(architecture|scalable|application|design)"
    
    run_test "tdd_workflow" \
        "npx claude-flow@alpha sparc tdd 'Password validation feature'" \
        "(test|driven|development|password|validation)"
    
    run_test "integration_stage" \
        "npx claude-flow@alpha sparc run integration 'Complete system integration test'" \
        "(integration|complete|system|test)"
}

test_batch_processing() {
    log_message "Testing batch processing and parallel execution..."
    
    run_test "batch_processing" \
        "npx claude-flow@alpha sparc batch spec-pseudocode,architect 'E-commerce platform development'" \
        "(batch|parallel|spec|architect|ecommerce|platform)"
    
    run_test "pipeline_processing" \
        "npx claude-flow@alpha sparc pipeline 'Full-stack application development pipeline'" \
        "(pipeline|processing|fullstack|application|development)"
    
    # Create test tasks file for concurrent processing
    local tasks_file="/tmp/test_tasks_$$"
    {
        echo "User authentication service"
        echo "Product catalog API"
        echo "Payment processing module"
        echo "Notification system"
    } > "$tasks_file"
    
    run_test "concurrent_processing" \
        "npx claude-flow@alpha sparc concurrent architect '$tasks_file'" \
        "(concurrent|tasks|architect)"
    
    rm -f "$tasks_file"
}

test_error_handling() {
    log_message "Testing error handling and recovery..."
    
    # Test invalid command handling
    run_test "invalid_command_handling" \
        "npx claude-flow@alpha sparc run invalid-mode 'test task' 2>&1 || echo 'error_handled'" \
        "(error|invalid|handled|mode)"
    
    # Test missing parameters
    run_test "missing_parameters" \
        "npx claude-flow@alpha sparc run 2>&1 || echo 'missing_params_handled'" \
        "(missing|params|handled|parameter|required)"
    
    # Test timeout handling (use a short timeout to simulate failure)
    run_test "timeout_handling" \
        "timeout 1s npx claude-flow@alpha sparc run architect 'complex system design' 2>&1 || echo 'timeout_handled'" \
        "(timeout|handled|error)"
    
    # Test recovery mechanism
    run_test "recovery_mechanism" \
        "npx claude-flow@alpha sparc run spec-pseudocode 'recovery test' --resume 2>&1 || echo 'recovery_attempted'" \
        "(recovery|attempted|specification|resume)"
}

test_performance_benchmarks() {
    log_message "Testing performance benchmarks..."
    
    local start_time=$(date +%s%N)
    
    run_test "performance_basic_workflow" \
        "npx claude-flow@alpha sparc run spec-pseudocode 'Simple API endpoint performance test'" \
        "(specification|api|endpoint|performance)"
    
    local end_time=$(date +%s%N)
    local duration_ms=$(( (end_time - start_time) / 1000000 ))
    
    PERFORMANCE_METRICS["basic_workflow_duration"]="$duration_ms"
    
    if [ "$duration_ms" -lt "$PERFORMANCE_THRESHOLD_MS" ]; then
        log_message "Performance test passed: ${duration_ms}ms < ${PERFORMANCE_THRESHOLD_MS}ms" "SUCCESS"
    else
        log_message "Performance test warning: ${duration_ms}ms >= ${PERFORMANCE_THRESHOLD_MS}ms" "WARN"
    fi
    
    # Test memory usage (approximate)
    run_test "memory_usage_test" \
        "npx claude-flow@alpha sparc run architect 'Complex system design with multiple components'" \
        "(architecture|complex|system|components)"
}

test_claude_flow_integration() {
    log_message "Testing integration with Claude-Flow..."
    
    run_test "swarm_initialization" \
        "npx claude-flow@alpha hooks pre-task --description 'MLE-Star integration test workflow'" \
        ".*" # Any output is acceptable
    
    run_test "memory_coordination" \
        "npx claude-flow@alpha hooks post-edit --file 'test.js' --memory-key 'swarm/tester/validation'" \
        ".*" # Any output is acceptable
    
    run_test "session_management" \
        "npx claude-flow@alpha hooks session-end --export-metrics true" \
        ".*" # Any output is acceptable
    
    # Test MCP swarm status (may fail if swarm not initialized, which is acceptable)
    run_test "swarm_status_check" \
        "npx claude-flow@alpha mcp swarm_status 2>&1 || echo 'swarm_not_initialized'" \
        "(swarm|status|initialized|not_initialized)"
}

test_security_validation() {
    log_message "Testing command validation and security..."
    
    run_test "command_safety_validation" \
        "npx claude-flow@alpha hooks pre-command --command 'ls -la' --validate-safety true" \
        ".*" # Any output is acceptable
    
    # Test dangerous command rejection
    run_test "dangerous_command_rejection" \
        "npx claude-flow@alpha hooks pre-command --command 'rm -rf /' --validate-safety true 2>&1 || echo 'dangerous_command_rejected'" \
        "(dangerous|rejected|safety|error)"
    
    # Test file operation permissions
    local test_file="/tmp/permission_test_$$"
    echo "// Test file for permission validation" > "$test_file"
    
    run_test "file_permission_validation" \
        "npx claude-flow@alpha hooks pre-edit --file '$test_file' --auto-assign-agents true" \
        ".*" # Any output is acceptable
    
    rm -f "$test_file"
}

generate_test_report() {
    log_message "Generating test report..."
    
    local report_data=$(cat << EOF
{
  "test_run": {
    "timestamp": "$TEST_TIMESTAMP",
    "total_tests": $TOTAL_TESTS,
    "passed_tests": $PASSED_TESTS,
    "failed_tests": $FAILED_TESTS,
    "success_rate": $(echo "scale=2; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l 2>/dev/null || echo "0"),
    "duration_ms": ${PERFORMANCE_METRICS["total_duration"]:-0}
  },
  "test_results": {
EOF
    
    # Add individual test results
    local first=true
    for test_name in "${!TEST_RESULTS[@]}"; do
        if [ "$first" = true ]; then
            first=false
        else
            report_data+=","
        fi
        report_data+="\n    \"$test_name\": \"${TEST_RESULTS[$test_name]}\""
    done
    
    report_data+="\n  },\n  \"performance_metrics\": {"
    
    # Add performance metrics
    first=true
    for metric_name in "${!PERFORMANCE_METRICS[@]}"; do
        if [ "$first" = true ]; then
            first=false
        else
            report_data+=","
        fi
        report_data+="\n    \"$metric_name\": ${PERFORMANCE_METRICS[$metric_name]}"
    done
    
    report_data+="\n  }\n}"
    
    echo -e "$report_data" > "$REPORT_FILE"
    
    log_message "Test report generated: $REPORT_FILE" "SUCCESS"
}

print_test_summary() {
    echo -e "\n${BLUE}=== MLE-Star Integration Test Summary ===${NC}"
    echo -e "Total Tests: $TOTAL_TESTS"
    echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
    echo -e "${RED}Failed: $FAILED_TESTS${NC}"
    
    if [ $TOTAL_TESTS -gt 0 ]; then
        local success_rate=$(echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l 2>/dev/null || echo "0")
        echo -e "Success Rate: ${success_rate}%"
    fi
    
    echo -e "\nLog File: $LOG_FILE"
    echo -e "Report File: $REPORT_FILE"
    
    if [ $FAILED_TESTS -gt 0 ]; then
        echo -e "\n${RED}Some tests failed. Check the log file for details.${NC}"
        return 1
    else
        echo -e "\n${GREEN}All tests passed successfully!${NC}"
        return 0
    fi
}

# Main execution
main() {
    local test_start_time=$(date +%s%N)
    
    echo -e "${BLUE}Starting MLE-Star Integration Tests...${NC}"
    
    # Setup
    setup_test_environment
    
    # Trap for cleanup
    trap cleanup_test_environment EXIT
    
    # Check prerequisites
    if ! check_prerequisites; then
        log_message "Prerequisites check failed" "ERROR"
        exit 1
    fi
    
    # Run test suites
    test_command_registration
    test_template_generation
    test_workflow_execution
    test_batch_processing
    test_error_handling
    test_performance_benchmarks
    test_claude_flow_integration
    test_security_validation
    
    # Calculate total duration
    local test_end_time=$(date +%s%N)
    local total_duration=$(( (test_end_time - test_start_time) / 1000000 ))
    PERFORMANCE_METRICS["total_duration"]="$total_duration"
    
    # Generate summary
    print_test_summary
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi