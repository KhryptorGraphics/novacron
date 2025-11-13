#!/bin/bash

# DWCP v3 Test Execution Script
# Comprehensive test runner with coverage reporting

set -e

PROJECT_ROOT="/home/kp/novacron"
DWCP_V3_ROOT="$PROJECT_ROOT/backend/core/network/dwcp/v3"
OUTPUT_DIR="$DWCP_V3_ROOT/test-results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "DWCP v3 Test Suite Execution"
echo "========================================"
echo "Timestamp: $TIMESTAMP"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run tests for a component
run_component_tests() {
    local component=$1
    local test_pattern=$2

    echo -e "${YELLOW}Running $component tests...${NC}"

    if cd "$DWCP_V3_ROOT/$component" 2>/dev/null; then
        if go test -v -race -coverprofile="$OUTPUT_DIR/${component}_coverage.out" ./ 2>&1 | tee "$OUTPUT_DIR/${component}_test.log"; then
            echo -e "${GREEN}✓ $component tests passed${NC}"

            # Generate coverage stats
            if [ -f "$OUTPUT_DIR/${component}_coverage.out" ]; then
                go tool cover -func="$OUTPUT_DIR/${component}_coverage.out" | tail -1 | tee "$OUTPUT_DIR/${component}_coverage.txt"
            fi
            return 0
        else
            echo -e "${RED}✗ $component tests failed${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠ $component directory not found, skipping${NC}"
        return 0
    fi
}

# Function to run benchmarks
run_benchmarks() {
    local component=$1

    echo -e "${YELLOW}Running $component benchmarks...${NC}"

    if cd "$DWCP_V3_ROOT/$component" 2>/dev/null; then
        if go test -bench=. -benchmem -benchtime=3s ./ 2>&1 | tee "$OUTPUT_DIR/${component}_benchmark.log"; then
            echo -e "${GREEN}✓ $component benchmarks completed${NC}"
            return 0
        else
            echo -e "${RED}✗ $component benchmarks failed${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠ $component directory not found, skipping${NC}"
        return 0
    fi
}

# Array to track results
declare -a RESULTS

echo ""
echo "========================================"
echo "Phase 1: Unit Tests"
echo "========================================"
echo ""

# Run component tests
COMPONENTS=(
    "transport"
    "encoding"
    "prediction"
    "sync"
    "consensus"
    "partition"
    "security"
    "monitoring"
    "optimization"
)

for component in "${COMPONENTS[@]}"; do
    if run_component_tests "$component"; then
        RESULTS+=("$component:PASS")
    else
        RESULTS+=("$component:FAIL")
    fi
    echo ""
done

echo ""
echo "========================================"
echo "Phase 2: Integration Tests"
echo "========================================"
echo ""

cd "$DWCP_V3_ROOT"
if go test -v -race -coverprofile="$OUTPUT_DIR/tests_coverage.out" ./tests/ 2>&1 | tee "$OUTPUT_DIR/integration_test.log"; then
    echo -e "${GREEN}✓ Integration tests passed${NC}"
    RESULTS+=("integration:PASS")
else
    echo -e "${RED}✗ Integration tests failed${NC}"
    RESULTS+=("integration:FAIL")
fi

echo ""
echo "========================================"
echo "Phase 3: Benchmarks"
echo "========================================"
echo ""

cd "$DWCP_V3_ROOT"
if go test -bench=. -benchmem -benchtime=3s ./benchmarks/ 2>&1 | tee "$OUTPUT_DIR/benchmarks.log"; then
    echo -e "${GREEN}✓ Benchmarks completed${NC}"
    RESULTS+=("benchmarks:PASS")
else
    echo -e "${RED}✗ Benchmarks failed${NC}"
    RESULTS+=("benchmarks:FAIL")
fi

echo ""
echo "========================================"
echo "Phase 4: Coverage Analysis"
echo "========================================"
echo ""

# Combine coverage files
cd "$DWCP_V3_ROOT"
echo "mode: atomic" > "$OUTPUT_DIR/combined_coverage.out"

for coverage_file in "$OUTPUT_DIR"/*_coverage.out; do
    if [ -f "$coverage_file" ]; then
        tail -n +2 "$coverage_file" >> "$OUTPUT_DIR/combined_coverage.out" 2>/dev/null || true
    fi
done

# Generate combined coverage report
if [ -f "$OUTPUT_DIR/combined_coverage.out" ]; then
    echo "Generating combined coverage report..."
    go tool cover -func="$OUTPUT_DIR/combined_coverage.out" > "$OUTPUT_DIR/combined_coverage.txt"
    go tool cover -html="$OUTPUT_DIR/combined_coverage.out" -o "$OUTPUT_DIR/combined_coverage.html"

    # Extract overall coverage
    OVERALL_COVERAGE=$(go tool cover -func="$OUTPUT_DIR/combined_coverage.out" | tail -1 | awk '{print $3}')
    echo -e "${GREEN}Overall Coverage: $OVERALL_COVERAGE${NC}"
    echo "Overall Coverage: $OVERALL_COVERAGE" > "$OUTPUT_DIR/overall_coverage.txt"
fi

echo ""
echo "========================================"
echo "Test Results Summary"
echo "========================================"
echo ""

PASS_COUNT=0
FAIL_COUNT=0

for result in "${RESULTS[@]}"; do
    component="${result%%:*}"
    status="${result##*:}"

    if [ "$status" = "PASS" ]; then
        echo -e "${GREEN}✓${NC} $component"
        ((PASS_COUNT++))
    else
        echo -e "${RED}✗${NC} $component"
        ((FAIL_COUNT++))
    fi
done

echo ""
echo "Total: $((PASS_COUNT + FAIL_COUNT)) | Passed: $PASS_COUNT | Failed: $FAIL_COUNT"

if [ -f "$OUTPUT_DIR/overall_coverage.txt" ]; then
    cat "$OUTPUT_DIR/overall_coverage.txt"
fi

echo ""
echo "========================================"
echo "Output Files"
echo "========================================"
echo ""
echo "Test Results:    $OUTPUT_DIR/"
echo "Coverage HTML:   $OUTPUT_DIR/combined_coverage.html"
echo "Coverage Report: $OUTPUT_DIR/combined_coverage.txt"
echo "Benchmark Data:  $OUTPUT_DIR/benchmarks.log"
echo ""

# Exit with appropriate code
if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
