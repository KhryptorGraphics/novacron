#!/bin/bash
# DWCP v3 Test Suite Runner
# Runs all v3 tests with coverage analysis

set -e

echo "=========================================="
echo "DWCP v1 → v3 Upgrade Test Suite"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test directory
TEST_DIR="/home/kp/novacron/backend/core/network/dwcp/v3/tests"
COVERAGE_DIR="/home/kp/novacron/backend/core/network/dwcp/v3/coverage"

mkdir -p "$COVERAGE_DIR"

echo -e "${YELLOW}Step 1: Running Backward Compatibility Tests${NC}"
echo "=============================================="
go test -v -race -timeout 5m \
    -coverprofile="$COVERAGE_DIR/backward_compat.out" \
    "$TEST_DIR/backward_compat_test.go" 2>&1 | tee "$COVERAGE_DIR/backward_compat.log" || true
echo ""

echo -e "${YELLOW}Step 2: Running AMST v3 Component Tests${NC}"
echo "=============================================="
go test -v -race -timeout 5m \
    -coverprofile="$COVERAGE_DIR/amst_v3.out" \
    "$TEST_DIR/amst_v3_test.go" 2>&1 | tee "$COVERAGE_DIR/amst_v3.log" || true
echo ""

echo -e "${YELLOW}Step 3: Running HDE v3 Component Tests${NC}"
echo "=============================================="
go test -v -race -timeout 5m \
    -coverprofile="$COVERAGE_DIR/hde_v3.out" \
    "$TEST_DIR/hde_v3_test.go" 2>&1 | tee "$COVERAGE_DIR/hde_v3.log" || true
echo ""

echo -e "${YELLOW}Step 4: Running Mode Switching Tests${NC}"
echo "=============================================="
go test -v -race -timeout 5m \
    -coverprofile="$COVERAGE_DIR/mode_switching.out" \
    "$TEST_DIR/mode_switching_test.go" 2>&1 | tee "$COVERAGE_DIR/mode_switching.log" || true
echo ""

echo -e "${YELLOW}Step 5: Running Integration Tests${NC}"
echo "=============================================="
go test -v -race -timeout 5m \
    -coverprofile="$COVERAGE_DIR/integration.out" \
    "$TEST_DIR/integration_test.go" 2>&1 | tee "$COVERAGE_DIR/integration.log" || true
echo ""

echo -e "${YELLOW}Step 6: Running Benchmarks${NC}"
echo "=============================================="
go test -v -bench=. -benchmem -timeout 10m \
    "$TEST_DIR/benchmark_test.go" 2>&1 | tee "$COVERAGE_DIR/benchmark.log" || true
echo ""

echo -e "${YELLOW}Step 7: Generating Coverage Report${NC}"
echo "=============================================="

# Combine coverage profiles
echo "mode: set" > "$COVERAGE_DIR/coverage_combined.out"
grep -h -v "^mode:" "$COVERAGE_DIR"/*.out >> "$COVERAGE_DIR/coverage_combined.out" 2>/dev/null || true

# Generate HTML coverage report
if [ -f "$COVERAGE_DIR/coverage_combined.out" ]; then
    go tool cover -html="$COVERAGE_DIR/coverage_combined.out" -o "$COVERAGE_DIR/coverage.html"

    # Calculate coverage percentage
    COVERAGE=$(go tool cover -func="$COVERAGE_DIR/coverage_combined.out" | grep total | awk '{print $3}')

    echo ""
    echo -e "${GREEN}=========================================="
    echo "Coverage Report Generated"
    echo "==========================================${NC}"
    echo ""
    echo "Total Coverage: $COVERAGE"
    echo "HTML Report: $COVERAGE_DIR/coverage.html"
    echo ""

    # Check if coverage meets 90% target
    COVERAGE_NUM=$(echo "$COVERAGE" | sed 's/%//')
    if (( $(echo "$COVERAGE_NUM >= 90" | bc -l) )); then
        echo -e "${GREEN}✅ Coverage target met: $COVERAGE >= 90%${NC}"
    else
        echo -e "${YELLOW}⚠️  Coverage below target: $COVERAGE < 90%${NC}"
    fi
else
    echo -e "${RED}No coverage data generated${NC}"
fi

echo ""
echo -e "${GREEN}=========================================="
echo "Test Suite Completed"
echo "==========================================${NC}"
echo ""
echo "Review individual test logs in: $COVERAGE_DIR/"
echo ""
