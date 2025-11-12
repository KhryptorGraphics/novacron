#!/bin/bash
# DWCP Phase 1 Test Validation Script
# Validates that all required tests are present and properly structured

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   DWCP Phase 1 Test Validation                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

cd "$(dirname "$0")"

# Test categories and required tests
declare -A REQUIRED_TESTS=(
    ["AMST"]="TestPhase1_AMSTRDMASupport TestPhase1_AMSTBBRCongestion TestPhase1_AMSTDynamicScaling TestPhase1_AMSTFailover TestPhase1_AMSTMetrics TestPhase1_AMSTPerformance TestPhase1_AMSTConcurrency TestPhase1_AMSTGracefulShutdown"
    ["HDE"]="TestPhase1_HDEDictionaryTraining TestPhase1_HDECompressionRatio TestPhase1_HDEAdaptiveCompression TestPhase1_HDEAdvancedDelta TestPhase1_HDEBaselineSync TestPhase1_HDEMetrics TestPhase1_HDEConcurrency TestPhase1_HDEDictionaryUpdate"
    ["Integration"]="TestPhase1_MigrationIntegration TestPhase1_FederationIntegration TestPhase1_EndToEndPerformance TestPhase1_FailoverScenarios TestPhase1_ConfigurationManagement TestPhase1_BackwardCompatibility TestPhase1_MetricsCollection TestPhase1_MonitoringAlerts"
    ["WAN"]="TestWAN_HighLatency TestWAN_LowBandwidth TestWAN_PacketLoss TestWAN_MultiRegion"
    ["Benchmarks"]="BenchmarkAMSTThroughput BenchmarkHDECompression BenchmarkMigrationSpeed BenchmarkFederationSync BenchmarkConcurrentStreams"
)

# File requirements
declare -A REQUIRED_FILES=(
    ["phase1_amst_test.go"]="AMST tests"
    ["phase1_hde_test.go"]="HDE tests"
    ["phase1_integration_test.go"]="Integration tests"
    ["phase1_wan_test.go"]="WAN simulation tests"
    ["phase1_benchmark_test.go"]="Performance benchmarks"
    ["test_runner.sh"]="Test execution script"
    ["TESTING.md"]="Testing documentation"
    ["testdata/README.md"]="Test data documentation"
)

# Validation counters
PASSED=0
FAILED=0
WARNINGS=0

# Check file existence
echo -e "${YELLOW}Checking required files...${NC}"
for file in "${!REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✅ $file${NC} - ${REQUIRED_FILES[$file]}"
        ((PASSED++))
    else
        echo -e "  ${RED}❌ $file${NC} - MISSING"
        ((FAILED++))
    fi
done
echo ""

# Check test presence
echo -e "${YELLOW}Validating test functions...${NC}"
for category in "${!REQUIRED_TESTS[@]}"; do
    echo -e "${BLUE}$category:${NC}"

    for test in ${REQUIRED_TESTS[$category]}; do
        # Find test in all test files
        if grep -r "func $test" *.go > /dev/null 2>&1; then
            echo -e "  ${GREEN}✅ $test${NC}"
            ((PASSED++))
        else
            echo -e "  ${RED}❌ $test${NC} - NOT FOUND"
            ((FAILED++))
        fi
    done
    echo ""
done

# Check test structure
echo -e "${YELLOW}Validating test structure...${NC}"

# Check for proper test package
if head -1 phase1_amst_test.go | grep -q "package dwcp_test"; then
    echo -e "  ${GREEN}✅ Test package name${NC} (dwcp_test)"
    ((PASSED++))
else
    echo -e "  ${RED}❌ Test package name${NC} - Should be dwcp_test"
    ((FAILED++))
fi

# Check for required imports
if grep -q "github.com/stretchr/testify" phase1_*.go; then
    echo -e "  ${GREEN}✅ Testify imports${NC}"
    ((PASSED++))
else
    echo -e "  ${YELLOW}⚠️  Testify imports${NC} - Consider using testify/assert and testify/require"
    ((WARNINGS++))
fi

# Check for race detector compatibility
if grep -q "sync.Mutex\|sync.RWMutex\|atomic" phase1_*.go; then
    echo -e "  ${GREEN}✅ Thread-safety primitives${NC}"
    ((PASSED++))
else
    echo -e "  ${YELLOW}⚠️  Thread-safety${NC} - No mutex/atomic usage detected"
    ((WARNINGS++))
fi

echo ""

# Check documentation
echo -e "${YELLOW}Validating documentation...${NC}"

# Check TESTING.md content
if [ -f TESTING.md ]; then
    if grep -q "Quick Start" TESTING.md && \
       grep -q "Test Categories" TESTING.md && \
       grep -q "Success Criteria" TESTING.md; then
        echo -e "  ${GREEN}✅ TESTING.md${NC} - Complete documentation"
        ((PASSED++))
    else
        echo -e "  ${YELLOW}⚠️  TESTING.md${NC} - Missing some sections"
        ((WARNINGS++))
    fi
fi

# Check test runner
if [ -x test_runner.sh ]; then
    echo -e "  ${GREEN}✅ test_runner.sh${NC} - Executable"
    ((PASSED++))
else
    echo -e "  ${RED}❌ test_runner.sh${NC} - Not executable"
    ((FAILED++))
fi

echo ""

# Coverage requirements
echo -e "${YELLOW}Checking test coverage targets...${NC}"

TOTAL_TESTS=0
for category in "${!REQUIRED_TESTS[@]}"; do
    count=$(echo ${REQUIRED_TESTS[$category]} | wc -w)
    TOTAL_TESTS=$((TOTAL_TESTS + count))
done

echo -e "  ${BLUE}Total tests required:${NC} $TOTAL_TESTS"
echo -e "  ${BLUE}Test files:${NC} 5"
echo -e "  ${BLUE}Benchmark suites:${NC} 5"
echo ""

# Success criteria validation
echo -e "${YELLOW}Phase 1 Success Criteria Coverage:${NC}"

CRITERIA=(
    "Bandwidth utilization >85%:TestPhase1_AMSTPerformance"
    "Compression ratio >10x:TestPhase1_HDECompressionRatio"
    "Migration speedup 2-3x:TestPhase1_MigrationIntegration"
    "Bandwidth savings 40%:TestPhase1_FederationIntegration"
    "RDMA support:TestPhase1_AMSTRDMASupport"
    "BBR congestion control:TestPhase1_AMSTBBRCongestion"
    "Dynamic scaling:TestPhase1_AMSTDynamicScaling"
    "Failover resilience:TestPhase1_AMSTFailover"
)

for criterion in "${CRITERIA[@]}"; do
    desc="${criterion%%:*}"
    test="${criterion##*:}"

    if grep -r "func $test" *.go > /dev/null 2>&1; then
        echo -e "  ${GREEN}✅ $desc${NC}"
    else
        echo -e "  ${RED}❌ $desc${NC}"
    fi
done

echo ""

# Summary
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Validation Summary${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${GREEN}Passed:${NC} $PASSED"
echo -e "  ${RED}Failed:${NC} $FAILED"
echo -e "  ${YELLOW}Warnings:${NC} $WARNINGS"
echo ""

# Test file statistics
echo -e "${BLUE}Test File Statistics:${NC}"
for file in phase1_*.go; do
    if [ -f "$file" ]; then
        lines=$(wc -l < "$file")
        tests=$(grep -c "^func Test" "$file" || echo 0)
        benches=$(grep -c "^func Benchmark" "$file" || echo 0)
        echo -e "  ${file}: ${lines} lines, ${tests} tests, ${benches} benchmarks"
    fi
done
echo ""

# Final result
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║   ✅ VALIDATION PASSED                                 ║${NC}"
    echo -e "${GREEN}║   All required tests are present and properly         ║${NC}"
    echo -e "${GREEN}║   structured. Ready for execution.                    ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"

    if [ $WARNINGS -gt 0 ]; then
        echo ""
        echo -e "${YELLOW}Note: $WARNINGS warning(s) detected. Review output above.${NC}"
    fi

    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "  1. Run tests: ${GREEN}./test_runner.sh${NC}"
    echo -e "  2. Check coverage: ${GREEN}go test -coverprofile=coverage.out ./...${NC}"
    echo -e "  3. View results: ${GREEN}go tool cover -html=coverage.out${NC}"

    exit 0
else
    echo -e "${RED}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║   ❌ VALIDATION FAILED                                 ║${NC}"
    echo -e "${RED}║   $FAILED check(s) failed. Review output above.        ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi
