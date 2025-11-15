#!/bin/bash
# Comprehensive test execution script for Novacron chaos engineering suite

set -e

echo "========================================="
echo "Novacron Chaos Engineering Test Suite"
echo "Target: 96% Code Coverage"
echo "========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

run_test_suite() {
    local name=$1
    local path=$2
    local args=$3

    echo -e "${YELLOW}Running: $name${NC}"
    if go test $path $args -v; then
        echo -e "${GREEN}✓ $name PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}✗ $name FAILED${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo ""
}

# Create coverage directory
mkdir -p coverage

echo "1. Byzantine Fault Tolerance Tests"
echo "-----------------------------------"
run_test_suite "Byzantine Node Behavior" "./tests/chaos" "-run TestByzantineNodeBehavior -coverprofile=coverage/byzantine.out"
run_test_suite "Byzantine Recovery" "./tests/chaos" "-run TestByzantineRecovery -coverprofile=coverage/byzantine_recovery.out"

echo "2. Network Partition Tests"
echo "---------------------------"
run_test_suite "Network Partition Scenarios" "./tests/chaos" "-run TestNetworkPartition -timeout 2m -coverprofile=coverage/partition.out"
run_test_suite "Partition with Byzantine" "./tests/chaos" "-run TestPartitionWithByzantine -coverprofile=coverage/partition_byzantine.out"

echo "3. Failure Scenario Tests"
echo "-------------------------"
run_test_suite "Node Crash Scenarios" "./tests/chaos" "-run TestNodeCrashScenarios -coverprofile=coverage/crashes.out"
run_test_suite "Memory Exhaustion" "./tests/chaos" "-run TestMemoryExhaustion -timeout 1m -coverprofile=coverage/memory.out"
run_test_suite "Disk Full" "./tests/chaos" "-run TestDiskFullScenario -coverprofile=coverage/disk.out"
run_test_suite "CPU Saturation" "./tests/chaos" "-run TestCPUSaturation -timeout 1m -coverprofile=coverage/cpu.out"
run_test_suite "Network Congestion" "./tests/chaos" "-run TestNetworkCongestion -timeout 2m -coverprofile=coverage/network.out"
run_test_suite "Clock Skew" "./tests/chaos" "-run TestClockSkew -coverprofile=coverage/clock.out"
run_test_suite "Split Brain" "./tests/chaos" "-run TestSplitBrain -timeout 2m -coverprofile=coverage/splitbrain.out"

echo "4. Integration Tests"
echo "--------------------"
run_test_suite "Full Stack Integration" "./tests/integration" "-run TestFullStackIntegration -timeout 3m -coverprofile=coverage/integration.out"

echo "5. Performance Benchmarks"
echo "-------------------------"
echo -e "${YELLOW}Running: Performance Benchmarks${NC}"
if go test ./tests/benchmarks -bench=. -benchmem -benchtime=10s > coverage/benchmarks.txt 2>&1; then
    echo -e "${GREEN}✓ Performance Benchmarks COMPLETED${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${RED}✗ Performance Benchmarks FAILED${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))
echo ""

# Combine coverage reports
echo "6. Coverage Analysis"
echo "--------------------"
echo "Combining coverage reports..."

# Merge coverage files
echo "mode: atomic" > coverage/combined.out
tail -q -n +2 coverage/*.out >> coverage/combined.out 2>/dev/null || true

# Generate coverage report
go tool cover -func=coverage/combined.out > coverage/coverage-summary.txt

# Calculate total coverage
TOTAL_COVERAGE=$(go tool cover -func=coverage/combined.out | grep total | awk '{print $3}' | sed 's/%//')

echo ""
echo "Coverage by Package:"
echo "-------------------"
cat coverage/coverage-summary.txt | grep -E "dwcp|ml|api" || echo "No coverage data available"

echo ""
echo "========================================="
echo "TEST RESULTS SUMMARY"
echo "========================================="
echo -e "Total Tests:  $TOTAL_TESTS"
echo -e "${GREEN}Passed:       $PASSED_TESTS${NC}"
echo -e "${RED}Failed:       $FAILED_TESTS${NC}"
echo ""
echo -e "Code Coverage: ${GREEN}${TOTAL_COVERAGE}%${NC}"
echo "Target:        96%"

if (( $(echo "$TOTAL_COVERAGE >= 96" | bc -l) )); then
    echo -e "${GREEN}✓ Coverage target ACHIEVED${NC}"
else
    echo -e "${YELLOW}⚠ Coverage target NOT met (${TOTAL_COVERAGE}% < 96%)${NC}"
fi
echo ""

# Generate HTML coverage report
echo "Generating HTML coverage report..."
go tool cover -html=coverage/combined.out -o coverage/coverage.html
echo -e "${GREEN}HTML report: coverage/coverage.html${NC}"

# Performance report
echo ""
echo "========================================="
echo "PERFORMANCE METRICS"
echo "========================================="
if [ -f coverage/benchmarks.txt ]; then
    echo "Consensus Protocol Benchmarks:"
    grep -A 2 "Benchmark" coverage/benchmarks.txt | head -20 || echo "No benchmark data"
    echo ""
    echo "Full benchmark results: coverage/benchmarks.txt"
fi

# Exit with failure if any tests failed
if [ $FAILED_TESTS -gt 0 ]; then
    echo ""
    echo -e "${RED}TEST SUITE FAILED${NC}"
    exit 1
else
    echo ""
    echo -e "${GREEN}ALL TESTS PASSED ✓${NC}"
    exit 0
fi
