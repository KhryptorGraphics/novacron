#!/bin/bash

# Docker-based Test Execution for DWCP v3
# Runs full test suite in isolated container with CGO support

set -e

PROJECT_ROOT="/home/kp/novacron"
RESULTS_DIR="$PROJECT_ROOT/backend/core/network/dwcp/v3/test-results"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}DWCP v3 Docker Test Runner${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not found${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

echo -e "${GREEN}✓ Docker found${NC}"
echo ""

# Build test container
echo -e "${YELLOW}Building test container...${NC}"
cd "$PROJECT_ROOT"

if docker build -f backend/core/network/dwcp/v3/Dockerfile.test -t dwcp-v3-test .; then
    echo -e "${GREEN}✓ Container built successfully${NC}"
else
    echo -e "${RED}✗ Container build failed${NC}"
    exit 1
fi

echo ""

# Run tests
echo -e "${YELLOW}Running tests in container...${NC}"
echo ""

if docker run --rm \
    -v "$RESULTS_DIR:/app/test-results" \
    dwcp-v3-test; then
    echo ""
    echo -e "${GREEN}✓ Tests completed successfully${NC}"
    TEST_EXIT=0
else
    echo ""
    echo -e "${RED}✗ Tests failed${NC}"
    TEST_EXIT=1
fi

echo ""

# Display coverage summary
if [ -f "$RESULTS_DIR/coverage.out" ]; then
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}Coverage Summary${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo ""

    cd "$PROJECT_ROOT/backend/core/network/dwcp/v3"
    go tool cover -func="$RESULTS_DIR/coverage.out" | tail -20

    OVERALL_COVERAGE=$(go tool cover -func="$RESULTS_DIR/coverage.out" | tail -1 | awk '{print $3}')
    echo ""
    echo -e "${GREEN}Overall Coverage: $OVERALL_COVERAGE${NC}"

    # Check if meets 90% threshold
    COVERAGE_NUM=$(echo "$OVERALL_COVERAGE" | sed 's/%//')
    if (( $(echo "$COVERAGE_NUM >= 90.0" | bc -l) )); then
        echo -e "${GREEN}✓ Coverage meets 90% threshold${NC}"
    else
        echo -e "${YELLOW}⚠ Coverage below 90% threshold${NC}"
    fi
fi

echo ""
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Output Files${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo "Coverage Report: $RESULTS_DIR/coverage.out"
echo "Coverage HTML:   $RESULTS_DIR/coverage.html"
echo "Test Results:    $RESULTS_DIR/"
echo ""

if [ $TEST_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
