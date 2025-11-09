#!/bin/bash

# DWCP Multi-Datacenter WAN Testing Framework - Test Runner
# This script runs the comprehensive DWCP testing suite

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_DIR="$PROJECT_ROOT/backend/core/network/dwcp/testing"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}DWCP Testing Framework - Test Runner${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if running as root (required for tc commands)
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}Warning: Some tests require root privileges for traffic control.${NC}"
    echo -e "${YELLOW}Consider running with: sudo $0${NC}"
    echo ""
fi

# Function to run tests
run_tests() {
    local test_type=$1
    local test_path=$2

    echo -e "${GREEN}Running $test_type...${NC}"
    cd "$TEST_DIR"

    if go test -v $test_path; then
        echo -e "${GREEN}✓ $test_type PASSED${NC}"
        return 0
    else
        echo -e "${RED}✗ $test_type FAILED${NC}"
        return 1
    fi
    echo ""
}

# Parse command line arguments
TEST_TYPE="${1:-all}"

case "$TEST_TYPE" in
    unit)
        echo "Running unit tests only..."
        run_tests "Unit Tests" "-run=Test"
        ;;

    integration)
        echo "Running integration tests..."
        run_tests "Integration Tests" "-run=TestFullTestingPipeline"
        ;;

    scenarios)
        echo "Running scenario tests..."
        run_tests "Scenario Tests" "./scenarios/..."
        ;;

    benchmarks)
        echo "Running performance benchmarks..."
        cd "$TEST_DIR"
        echo -e "${GREEN}Running benchmarks...${NC}"
        go test -bench=. -benchmem -run=^$ | tee benchmark-results.txt
        echo -e "${GREEN}✓ Benchmarks complete. Results saved to benchmark-results.txt${NC}"
        ;;

    chaos)
        echo "Running chaos engineering tests..."
        if [ "$EUID" -ne 0 ]; then
            echo -e "${RED}Error: Chaos tests require root privileges.${NC}"
            echo -e "${YELLOW}Please run with: sudo $0 chaos${NC}"
            exit 1
        fi
        run_tests "Chaos Engineering Tests" "-run=TestChaosEngineering"
        ;;

    continuous)
        echo "Starting continuous testing pipeline..."
        run_tests "Continuous Testing" "-run=TestContinuousTesting"
        ;;

    quick)
        echo "Running quick test suite (reduced duration)..."
        export DWCP_TEST_QUICK=1
        run_tests "Quick Tests" "-run=Test -timeout=5m"
        ;;

    coverage)
        echo "Running tests with coverage analysis..."
        cd "$TEST_DIR"
        go test -coverprofile=coverage.out ./...
        go tool cover -html=coverage.out -o coverage.html
        echo -e "${GREEN}✓ Coverage report generated: coverage.html${NC}"
        ;;

    all)
        echo "Running complete test suite..."
        echo ""

        # Run all test types
        FAILED=0

        run_tests "Unit Tests" "-run=Test -timeout=10m" || FAILED=$((FAILED+1))
        echo ""

        run_tests "Integration Tests" "-run=TestFullTestingPipeline -timeout=15m" || FAILED=$((FAILED+1))
        echo ""

        run_tests "Scenario Tests" "./scenarios/... -timeout=10m" || FAILED=$((FAILED+1))
        echo ""

        # Summary
        echo -e "${GREEN}========================================${NC}"
        if [ $FAILED -eq 0 ]; then
            echo -e "${GREEN}✓ All test suites PASSED${NC}"
        else
            echo -e "${RED}✗ $FAILED test suite(s) FAILED${NC}"
        fi
        echo -e "${GREEN}========================================${NC}"

        exit $FAILED
        ;;

    *)
        echo -e "${RED}Unknown test type: $TEST_TYPE${NC}"
        echo ""
        echo "Usage: $0 [test-type]"
        echo ""
        echo "Available test types:"
        echo "  all           - Run all tests (default)"
        echo "  unit          - Run unit tests only"
        echo "  integration   - Run integration tests"
        echo "  scenarios     - Run scenario-specific tests"
        echo "  benchmarks    - Run performance benchmarks"
        echo "  chaos         - Run chaos engineering tests (requires root)"
        echo "  continuous    - Run continuous testing pipeline"
        echo "  quick         - Run quick test suite (reduced duration)"
        echo "  coverage      - Run tests with coverage analysis"
        echo ""
        echo "Examples:"
        echo "  $0              # Run all tests"
        echo "  $0 unit         # Run unit tests only"
        echo "  sudo $0 chaos   # Run chaos tests with root"
        echo "  $0 benchmarks   # Run performance benchmarks"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Test execution complete!${NC}"
