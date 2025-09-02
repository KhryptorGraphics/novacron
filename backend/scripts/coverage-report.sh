#!/bin/bash
# NovaCron Integration Test Coverage Report Generator
# Generates comprehensive coverage analysis and reports

set -e

# Configuration
COVERAGE_DIR="/app/coverage"
REPORTS_DIR="/app/test-results"
THRESHOLD_TOTAL=80
THRESHOLD_FUNCTIONS=80
THRESHOLD_LINES=80
THRESHOLD_BRANCHES=75

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

# Create necessary directories
create_directories() {
    log_info "Creating coverage and report directories..."
    mkdir -p "$COVERAGE_DIR"
    mkdir -p "$REPORTS_DIR"
    mkdir -p "/app/coverage-reports"
}

# Generate coverage profile
generate_coverage() {
    log_info "Generating coverage profile..."
    
    cd /app/backend
    
    # Run tests with coverage
    go test -v \
        -race \
        -timeout=30m \
        -coverprofile="${COVERAGE_DIR}/coverage.out" \
        -covermode=atomic \
        -coverpkg=./... \
        ./tests/integration/... | tee "${REPORTS_DIR}/test-output.log"
    
    if [ $? -ne 0 ]; then
        log_error "Test execution failed"
        exit 1
    fi
    
    log_success "Coverage profile generated successfully"
}

# Generate HTML coverage report
generate_html_report() {
    log_info "Generating HTML coverage report..."
    
    go tool cover -html="${COVERAGE_DIR}/coverage.out" -o "${COVERAGE_DIR}/coverage.html"
    
    if [ -f "${COVERAGE_DIR}/coverage.html" ]; then
        log_success "HTML coverage report generated: ${COVERAGE_DIR}/coverage.html"
    else
        log_error "Failed to generate HTML coverage report"
    fi
}

# Generate function coverage report
generate_function_report() {
    log_info "Generating function coverage report..."
    
    go tool cover -func="${COVERAGE_DIR}/coverage.out" > "${COVERAGE_DIR}/coverage-functions.txt"
    
    if [ -f "${COVERAGE_DIR}/coverage-functions.txt" ]; then
        log_success "Function coverage report generated: ${COVERAGE_DIR}/coverage-functions.txt"
    else
        log_error "Failed to generate function coverage report"
    fi
}

# Generate detailed coverage analysis
generate_detailed_analysis() {
    log_info "Generating detailed coverage analysis..."
    
    # Extract overall coverage percentage
    TOTAL_COVERAGE=$(go tool cover -func="${COVERAGE_DIR}/coverage.out" | tail -1 | awk '{print $3}' | sed 's/%//')
    
    # Create detailed analysis report
    cat > "${COVERAGE_DIR}/coverage-analysis.md" << EOF
# NovaCron Integration Test Coverage Analysis

Generated: $(date)
Test Run: Integration Tests

## Overall Coverage Summary

- **Total Coverage**: ${TOTAL_COVERAGE}%
- **Coverage Threshold**: ${THRESHOLD_TOTAL}%
- **Status**: $([ $(echo "$TOTAL_COVERAGE >= $THRESHOLD_TOTAL" | bc -l) -eq 1 ] && echo "✅ PASSED" || echo "❌ FAILED")

## Coverage Breakdown by Package

EOF
    
    # Add package-level coverage breakdown
    go tool cover -func="${COVERAGE_DIR}/coverage.out" | grep -v "total:" | while read line; do
        if [[ $line == *".go"* ]]; then
            package=$(echo $line | awk -F'/' '{print $(NF-1)"/"$NF}' | awk '{print $1}')
            coverage=$(echo $line | awk '{print $3}')
            echo "- **${package}**: ${coverage}" >> "${COVERAGE_DIR}/coverage-analysis.md"
        fi
    done
    
    # Add uncovered functions analysis
    cat >> "${COVERAGE_DIR}/coverage-analysis.md" << EOF

## Uncovered Functions

Functions with 0% coverage that need attention:

EOF
    
    go tool cover -func="${COVERAGE_DIR}/coverage.out" | grep "0.0%" | head -20 | while read line; do
        func=$(echo $line | awk '{print $2}')
        file=$(echo $line | awk '{print $1}')
        echo "- \`${func}\` in \`${file}\`" >> "${COVERAGE_DIR}/coverage-analysis.md"
    done
    
    # Add recommendations
    cat >> "${COVERAGE_DIR}/coverage-analysis.md" << EOF

## Recommendations

### High Priority
- Focus on functions with 0% coverage in critical paths
- Add integration tests for authentication and VM lifecycle functions
- Improve WebSocket event handling test coverage

### Medium Priority
- Increase federation scenario test coverage
- Add more error condition tests
- Improve API endpoint error handling coverage

### Low Priority
- Add tests for utility and helper functions
- Improve documentation for uncovered code paths

## Files by Coverage Level

### High Coverage (>90%)
EOF
    
    # Categorize files by coverage level
    go tool cover -func="${COVERAGE_DIR}/coverage.out" | grep -v "total:" | while read line; do
        if [[ $line == *".go"* ]]; then
            file=$(echo $line | awk '{print $1}')
            coverage_num=$(echo $line | awk '{print $3}' | sed 's/%//')
            
            if [ $(echo "$coverage_num >= 90" | bc -l) -eq 1 ]; then
                echo "- \`${file}\` (${coverage_num}%)" >> "${COVERAGE_DIR}/high-coverage.tmp"
            elif [ $(echo "$coverage_num >= 70" | bc -l) -eq 1 ]; then
                echo "- \`${file}\` (${coverage_num}%)" >> "${COVERAGE_DIR}/medium-coverage.tmp"
            else
                echo "- \`${file}\` (${coverage_num}%)" >> "${COVERAGE_DIR}/low-coverage.tmp"
            fi
        fi
    done
    
    # Add categorized files to report
    if [ -f "${COVERAGE_DIR}/high-coverage.tmp" ]; then
        cat "${COVERAGE_DIR}/high-coverage.tmp" >> "${COVERAGE_DIR}/coverage-analysis.md"
    fi
    
    echo "" >> "${COVERAGE_DIR}/coverage-analysis.md"
    echo "### Medium Coverage (70-90%)" >> "${COVERAGE_DIR}/coverage-analysis.md"
    if [ -f "${COVERAGE_DIR}/medium-coverage.tmp" ]; then
        cat "${COVERAGE_DIR}/medium-coverage.tmp" >> "${COVERAGE_DIR}/coverage-analysis.md"
    fi
    
    echo "" >> "${COVERAGE_DIR}/coverage-analysis.md"
    echo "### Low Coverage (<70%)" >> "${COVERAGE_DIR}/coverage-analysis.md"
    if [ -f "${COVERAGE_DIR}/low-coverage.tmp" ]; then
        cat "${COVERAGE_DIR}/low-coverage.tmp" >> "${COVERAGE_DIR}/coverage-analysis.md"
    fi
    
    # Cleanup temp files
    rm -f "${COVERAGE_DIR}"/*.tmp
    
    log_success "Detailed coverage analysis generated: ${COVERAGE_DIR}/coverage-analysis.md"
}

# Generate JSON coverage report for CI/CD integration
generate_json_report() {
    log_info "Generating JSON coverage report..."
    
    # Convert coverage to JSON format using gocov
    gocov convert "${COVERAGE_DIR}/coverage.out" > "${COVERAGE_DIR}/coverage.json"
    
    # Generate additional JSON formats
    gocov-xml < "${COVERAGE_DIR}/coverage.json" > "${COVERAGE_DIR}/coverage.xml"
    
    # Create summary JSON
    TOTAL_COVERAGE=$(go tool cover -func="${COVERAGE_DIR}/coverage.out" | tail -1 | awk '{print $3}' | sed 's/%//')
    
    cat > "${COVERAGE_DIR}/coverage-summary.json" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "test_type": "integration",
  "total_coverage": ${TOTAL_COVERAGE},
  "threshold": ${THRESHOLD_TOTAL},
  "status": "$([ $(echo "$TOTAL_COVERAGE >= $THRESHOLD_TOTAL" | bc -l) -eq 1 ] && echo "passed" || echo "failed")",
  "reports": {
    "html": "/app/coverage/coverage.html",
    "functions": "/app/coverage/coverage-functions.txt",
    "analysis": "/app/coverage/coverage-analysis.md",
    "json": "/app/coverage/coverage.json",
    "xml": "/app/coverage/coverage.xml"
  },
  "thresholds": {
    "total": ${THRESHOLD_TOTAL},
    "functions": ${THRESHOLD_FUNCTIONS},
    "lines": ${THRESHOLD_LINES},
    "branches": ${THRESHOLD_BRANCHES}
  }
}
EOF
    
    log_success "JSON coverage reports generated"
}

# Generate JUnit XML report for CI integration
generate_junit_report() {
    log_info "Generating JUnit XML report..."
    
    if [ -f "${REPORTS_DIR}/test-output.log" ]; then
        cat "${REPORTS_DIR}/test-output.log" | go-junit-report -set-exit-code > "${REPORTS_DIR}/junit-report.xml"
        log_success "JUnit XML report generated: ${REPORTS_DIR}/junit-report.xml"
    else
        log_warning "No test output log found, skipping JUnit report generation"
    fi
}

# Generate badge-compatible coverage data
generate_badge_data() {
    log_info "Generating badge-compatible coverage data..."
    
    TOTAL_COVERAGE=$(go tool cover -func="${COVERAGE_DIR}/coverage.out" | tail -1 | awk '{print $3}' | sed 's/%//')
    
    # Determine badge color based on coverage
    if [ $(echo "$TOTAL_COVERAGE >= 90" | bc -l) -eq 1 ]; then
        COLOR="brightgreen"
    elif [ $(echo "$TOTAL_COVERAGE >= 80" | bc -l) -eq 1 ]; then
        COLOR="green"
    elif [ $(echo "$TOTAL_COVERAGE >= 70" | bc -l) -eq 1 ]; then
        COLOR="yellow"
    elif [ $(echo "$TOTAL_COVERAGE >= 60" | bc -l) -eq 1 ]; then
        COLOR="orange"
    else
        COLOR="red"
    fi
    
    # Create badge data file
    cat > "${COVERAGE_DIR}/badge.json" << EOF
{
  "schemaVersion": 1,
  "label": "integration coverage",
  "message": "${TOTAL_COVERAGE}%",
  "color": "${COLOR}"
}
EOF
    
    log_success "Badge data generated: ${COVERAGE_DIR}/badge.json"
}

# Validate coverage thresholds
validate_coverage() {
    log_info "Validating coverage against thresholds..."
    
    TOTAL_COVERAGE=$(go tool cover -func="${COVERAGE_DIR}/coverage.out" | tail -1 | awk '{print $3}' | sed 's/%//')
    
    echo "Coverage Validation Results:"
    echo "============================"
    
    # Check total coverage
    if [ $(echo "$TOTAL_COVERAGE >= $THRESHOLD_TOTAL" | bc -l) -eq 1 ]; then
        log_success "Total Coverage: ${TOTAL_COVERAGE}% (✅ >= ${THRESHOLD_TOTAL}%)"
        VALIDATION_PASSED=1
    else
        log_error "Total Coverage: ${TOTAL_COVERAGE}% (❌ < ${THRESHOLD_TOTAL}%)"
        VALIDATION_PASSED=0
    fi
    
    # Count functions with low coverage
    LOW_COVERAGE_FUNCTIONS=$(go tool cover -func="${COVERAGE_DIR}/coverage.out" | grep -c "0.0%" || true)
    TOTAL_FUNCTIONS=$(go tool cover -func="${COVERAGE_DIR}/coverage.out" | grep -c "\.go:" || true)
    
    echo ""
    echo "Function Coverage Analysis:"
    echo "- Total Functions: ${TOTAL_FUNCTIONS}"
    echo "- Functions with 0% Coverage: ${LOW_COVERAGE_FUNCTIONS}"
    echo "- Function Coverage: $(echo "scale=2; 100 - (${LOW_COVERAGE_FUNCTIONS} * 100 / ${TOTAL_FUNCTIONS})" | bc)%"
    
    return $VALIDATION_PASSED
}

# Generate comprehensive test report
generate_comprehensive_report() {
    log_info "Generating comprehensive test report..."
    
    TOTAL_COVERAGE=$(go tool cover -func="${COVERAGE_DIR}/coverage.out" | tail -1 | awk '{print $3}' | sed 's/%//')
    TIMESTAMP=$(date)
    
    cat > "/app/coverage-reports/integration-test-report.md" << EOF
# NovaCron Integration Test Report

**Generated**: ${TIMESTAMP}  
**Test Suite**: Integration Tests  
**Coverage**: ${TOTAL_COVERAGE}%

## Executive Summary

This report provides comprehensive analysis of the NovaCron integration test suite,
including coverage metrics, test results, and recommendations for improvement.

### Quick Stats
- **Total Coverage**: ${TOTAL_COVERAGE}%
- **Test Files**: $(find /app/backend/tests/integration -name "*.go" -not -path "*/vendor/*" | wc -l)
- **Test Environment**: Docker Compose with PostgreSQL, Redis, LocalStack, MinIO

### Test Suites Included
1. **Authentication Integration Tests** - User registration, login, JWT validation
2. **VM Lifecycle Tests** - Create, start, stop, delete operations
3. **API Endpoint Tests** - REST API with database connectivity
4. **WebSocket Tests** - Real-time communication and event broadcasting
5. **Federation Tests** - Multi-cloud storage and compute scenarios

## Coverage Analysis

### Overall Coverage: ${TOTAL_COVERAGE}%

$([ $(echo "$TOTAL_COVERAGE >= $THRESHOLD_TOTAL" | bc -l) -eq 1 ] && echo "✅ **PASSED** - Meets ${THRESHOLD_TOTAL}% threshold" || echo "❌ **FAILED** - Below ${THRESHOLD_TOTAL}% threshold")

### Detailed Coverage Reports
- **HTML Report**: [coverage.html](coverage.html) - Interactive coverage browser
- **Function Report**: [coverage-functions.txt](coverage-functions.txt) - Per-function coverage
- **Analysis Report**: [coverage-analysis.md](coverage-analysis.md) - Detailed analysis with recommendations

## Test Environment

The integration tests run in a containerized environment with the following services:

- **PostgreSQL 15** - Primary database with test schema
- **Redis 7** - Caching and session management
- **LocalStack** - AWS service mocking (S3, EC2, IAM)
- **MinIO** - S3-compatible storage for federation testing
- **Prometheus** - Metrics collection for monitoring tests
- **Jaeger** - Distributed tracing for debugging

## Recommendations

### Immediate Actions Required
EOF

    # Add specific recommendations based on coverage
    if [ $(echo "$TOTAL_COVERAGE < 70" | bc -l) -eq 1 ]; then
        echo "- 🚨 **Critical**: Coverage below 70% - immediate attention required" >> "/app/coverage-reports/integration-test-report.md"
    elif [ $(echo "$TOTAL_COVERAGE < 80" | bc -l) -eq 1 ]; then
        echo "- ⚠️ **Warning**: Coverage below target - focus on critical paths" >> "/app/coverage-reports/integration-test-report.md"
    fi

    cat >> "/app/coverage-reports/integration-test-report.md" << EOF

### Next Steps
1. Review uncovered functions in critical authentication and VM management paths
2. Add integration tests for error scenarios and edge cases
3. Improve WebSocket event handling test coverage
4. Expand federation testing with more cloud provider scenarios

## Files Generated
- \`coverage.html\` - Interactive HTML coverage report
- \`coverage.json\` - Machine-readable coverage data
- \`coverage.xml\` - XML coverage for CI/CD integration
- \`junit-report.xml\` - JUnit test results
- \`badge.json\` - Coverage badge data for documentation

## Running the Tests

### Full Integration Test Suite
\`\`\`bash
make test-integration
\`\`\`

### Individual Test Suites
\`\`\`bash
make test-integration-auth      # Authentication tests
make test-integration-vm        # VM lifecycle tests
make test-integration-api       # API endpoint tests
make test-integration-websocket # WebSocket tests
make test-integration-federation # Multi-cloud federation tests
\`\`\`

### Quick Tests (No Coverage)
\`\`\`bash
make test-integration-quick
\`\`\`

---
*Report generated by NovaCron Integration Test Coverage System*
EOF
    
    log_success "Comprehensive test report generated: /app/coverage-reports/integration-test-report.md"
}

# Main execution
main() {
    echo "=========================================="
    echo "NovaCron Integration Test Coverage Report"
    echo "=========================================="
    
    create_directories
    generate_coverage
    generate_html_report
    generate_function_report
    generate_detailed_analysis
    generate_json_report
    generate_junit_report
    generate_badge_data
    generate_comprehensive_report
    
    echo ""
    echo "Coverage Report Generation Complete!"
    echo "===================================="
    
    # Display summary
    TOTAL_COVERAGE=$(go tool cover -func="${COVERAGE_DIR}/coverage.out" | tail -1 | awk '{print $3}' | sed 's/%//')
    echo "Total Coverage: ${TOTAL_COVERAGE}%"
    echo "Reports Location: ${COVERAGE_DIR}/"
    echo ""
    echo "Available Reports:"
    echo "- HTML: ${COVERAGE_DIR}/coverage.html"
    echo "- Analysis: ${COVERAGE_DIR}/coverage-analysis.md"
    echo "- Functions: ${COVERAGE_DIR}/coverage-functions.txt"
    echo "- JSON: ${COVERAGE_DIR}/coverage.json"
    echo "- XML: ${COVERAGE_DIR}/coverage.xml"
    echo "- JUnit: ${REPORTS_DIR}/junit-report.xml"
    echo "- Comprehensive: /app/coverage-reports/integration-test-report.md"
    
    # Validate coverage
    echo ""
    validate_coverage
    VALIDATION_RESULT=$?
    
    if [ $VALIDATION_RESULT -eq 1 ]; then
        log_success "✅ Coverage validation PASSED"
        exit 0
    else
        log_error "❌ Coverage validation FAILED"
        exit 1
    fi
}

# Run main function
main "$@"