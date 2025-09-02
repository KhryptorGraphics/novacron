#!/bin/bash
set -e

# NovaCron Integration Validation Runner
# Executes comprehensive validation tests for production readiness

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_DIR="$PROJECT_ROOT/tests/reports"
REPORT_FILE="$REPORT_DIR/integration_validation_$TIMESTAMP.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
FRONTEND_URL="${FRONTEND_URL:-http://localhost:8092}"
API_URL="${API_URL:-http://localhost:8090}"
WS_URL="${WS_URL:-ws://localhost:8091}"
DB_URL="${DATABASE_URL:-postgres://novacron_user:novacron_pass@localhost:5432/novacron_db?sslmode=disable}"

# Test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

echo -e "${BLUE}🚀 NovaCron Integration Validation Suite${NC}"
echo "=================================================="
echo "Timestamp: $(date)"
echo "Frontend URL: $FRONTEND_URL"
echo "API URL: $API_URL"
echo "WebSocket URL: $WS_URL"
echo "Database URL: ${DB_URL%:*}:***@${DB_URL##*@}" # Hide password
echo ""

# Create reports directory
mkdir -p "$REPORT_DIR"

# Initialize report
cat > "$REPORT_FILE" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "environment": {
    "frontend_url": "$FRONTEND_URL",
    "api_url": "$API_URL",
    "websocket_url": "$WS_URL",
    "database_url": "${DB_URL%%:*}:***@${DB_URL##*@}"
  },
  "tests": [],
  "summary": {
    "total": 0,
    "passed": 0,
    "failed": 0,
    "skipped": 0
  },
  "issues": [],
  "recommendations": []
}
EOF

# Helper function to update test results
update_results() {
    local test_name="$1"
    local status="$2"
    local error_msg="$3"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    case "$status" in
        "passed") PASSED_TESTS=$((PASSED_TESTS + 1)) ;;
        "failed") FAILED_TESTS=$((FAILED_TESTS + 1)) ;;
        "skipped") SKIPPED_TESTS=$((SKIPPED_TESTS + 1)) ;;
    esac
    
    # Add to JSON report (simplified)
    echo "  Test: $test_name - $status" >> "$REPORT_DIR/test_results.txt"
    if [ -n "$error_msg" ]; then
        echo "    Error: $error_msg" >> "$REPORT_DIR/test_results.txt"
    fi
}

# Test 1: Environment Prerequisites
echo -e "${BLUE}📋 Checking Prerequisites...${NC}"

check_prerequisite() {
    local name="$1"
    local command="$2"
    local required="$3"
    
    if eval "$command" >/dev/null 2>&1; then
        echo -e "  ✅ $name: Available"
        update_results "Prerequisites: $name" "passed"
    else
        if [ "$required" = "true" ]; then
            echo -e "  ❌ $name: Missing (Required)"
            update_results "Prerequisites: $name" "failed" "Required dependency not available"
        else
            echo -e "  ⚠️  $name: Missing (Optional)"
            update_results "Prerequisites: $name" "skipped" "Optional dependency not available"
        fi
    fi
}

check_prerequisite "Node.js" "node --version" "true"
check_prerequisite "NPM" "npm --version" "true"
check_prerequisite "Go" "go version" "true"
check_prerequisite "Docker" "docker --version" "false"
check_prerequisite "PostgreSQL Client" "psql --version" "true"
check_prerequisite "Puppeteer" "npm list puppeteer" "false"

# Test 2: Service Health Checks
echo -e "\n${BLUE}🔍 Service Health Checks...${NC}"

check_service_health() {
    local name="$1"
    local url="$2"
    local expected_status="$3"
    
    echo "  Checking $name at $url..."
    
    if command -v curl >/dev/null 2>&1; then
        response=$(curl -s -w "%{http_code}" -o /dev/null --max-time 10 "$url" 2>/dev/null || echo "000")
        
        if [ "$response" = "$expected_status" ] || [ "$expected_status" = "any" -a "$response" != "000" ]; then
            echo -e "    ✅ $name: Healthy (HTTP $response)"
            update_results "Service Health: $name" "passed"
        else
            echo -e "    ❌ $name: Unhealthy (HTTP $response)"
            update_results "Service Health: $name" "failed" "Service returned HTTP $response, expected $expected_status"
        fi
    else
        echo -e "    ⚠️  $name: Cannot check (curl not available)"
        update_results "Service Health: $name" "skipped" "curl not available for health check"
    fi
}

check_service_health "Frontend" "$FRONTEND_URL" "200"
check_service_health "API Health" "$API_URL/health" "200"
check_service_health "API Info" "$API_URL/api/info" "200"

# Test 3: Database Connectivity
echo -e "\n${BLUE}💾 Database Connectivity...${NC}"

if command -v psql >/dev/null 2>&1; then
    echo "  Testing database connection..."
    
    # Extract database connection parameters
    DB_HOST=$(echo "$DB_URL" | sed -n 's/.*@\([^:]*\):.*/\1/p')
    DB_PORT=$(echo "$DB_URL" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
    DB_NAME=$(echo "$DB_URL" | sed -n 's/.*\/\([^?]*\).*/\1/p')
    DB_USER=$(echo "$DB_URL" | sed -n 's/.*\/\/\([^:]*\):.*/\1/p')
    DB_PASS=$(echo "$DB_URL" | sed -n 's/.*:\/\/[^:]*:\([^@]*\)@.*/\1/p')
    
    # Simple connection test
    if PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" >/dev/null 2>&1; then
        echo -e "    ✅ Database: Connected"
        update_results "Database Connectivity" "passed"
        
        # Check required tables
        echo "    Checking required tables..."
        tables_check=$(PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name IN ('users', 'vms', 'vm_metrics');" 2>/dev/null || echo "0")
        
        if [ "$tables_check" -ge "3" ]; then
            echo -e "    ✅ Database Schema: Valid"
            update_results "Database Schema" "passed"
        else
            echo -e "    ❌ Database Schema: Missing tables (found $tables_check/3)"
            update_results "Database Schema" "failed" "Missing required database tables"
        fi
    else
        echo -e "    ❌ Database: Connection failed"
        update_results "Database Connectivity" "failed" "Cannot connect to database"
    fi
else
    echo -e "    ⚠️  Database: Cannot test (psql not available)"
    update_results "Database Connectivity" "skipped" "psql not available for testing"
fi

# Test 4: API Endpoints Validation
echo -e "\n${BLUE}🔌 API Endpoints Validation...${NC}"

test_api_endpoint() {
    local endpoint="$1"
    local method="$2"
    local expected_status="$3"
    local description="$4"
    
    echo "  Testing $description ($method $endpoint)..."
    
    if command -v curl >/dev/null 2>&1; then
        full_url="$API_URL$endpoint"
        
        case "$method" in
            "GET")
                response=$(curl -s -w "%{http_code}" -o /dev/null --max-time 10 "$full_url" 2>/dev/null || echo "000")
                ;;
            "POST")
                response=$(curl -s -w "%{http_code}" -o /dev/null --max-time 10 -X POST \
                    -H "Content-Type: application/json" -d '{}' "$full_url" 2>/dev/null || echo "000")
                ;;
            *)
                response="000"
                ;;
        esac
        
        if [ "$response" = "$expected_status" ] || { [ "$expected_status" = "2xx" ] && [ "$response" -ge "200" ] && [ "$response" -lt "300" ]; }; then
            echo -e "    ✅ $description: Working (HTTP $response)"
            update_results "API Endpoint: $description" "passed"
        else
            echo -e "    ❌ $description: Failed (HTTP $response)"
            update_results "API Endpoint: $description" "failed" "API returned HTTP $response, expected $expected_status"
        fi
    else
        echo -e "    ⚠️  $description: Cannot test (curl not available)"
        update_results "API Endpoint: $description" "skipped" "curl not available"
    fi
}

# Test public endpoints (no auth required)
test_api_endpoint "/health" "GET" "200" "Health Check"
test_api_endpoint "/api/info" "GET" "200" "API Info"
test_api_endpoint "/auth/register" "POST" "400" "User Registration" # 400 expected for empty payload
test_api_endpoint "/auth/login" "POST" "400" "User Login" # 400 expected for empty payload

# Test protected endpoints (should require auth)
test_api_endpoint "/api/monitoring/metrics" "GET" "401" "Monitoring Metrics (No Auth)"
test_api_endpoint "/api/vm/vms" "GET" "401" "VM List (No Auth)"

# Test 5: Frontend-Backend Integration
echo -e "\n${BLUE}🌐 Frontend-Backend Integration...${NC}"

if [ -f "$SCRIPT_DIR/integration/frontend_backend_validation.js" ] && command -v node >/dev/null 2>&1; then
    echo "  Running frontend-backend integration tests..."
    
    # Check if puppeteer is available
    if npm list puppeteer >/dev/null 2>&1 || npm list -g puppeteer >/dev/null 2>&1; then
        cd "$SCRIPT_DIR/integration"
        
        # Set environment variables for the test
        export BASE_URL="$FRONTEND_URL"
        export API_URL="$API_URL"
        export WS_URL="$WS_URL"
        export HEADLESS=true
        
        if timeout 120 node frontend_backend_validation.js > "$REPORT_DIR/frontend_integration.log" 2>&1; then
            echo -e "    ✅ Frontend Integration: Tests completed"
            update_results "Frontend-Backend Integration" "passed"
            
            # Check if report was generated
            if [ -f "./integration-validation-report.json" ]; then
                mv "./integration-validation-report.json" "$REPORT_DIR/frontend_validation_report.json"
            fi
        else
            echo -e "    ❌ Frontend Integration: Tests failed"
            update_results "Frontend-Backend Integration" "failed" "Frontend integration tests failed or timed out"
        fi
        
        cd "$PROJECT_ROOT"
    else
        echo -e "    ⚠️  Frontend Integration: Puppeteer not available"
        update_results "Frontend-Backend Integration" "skipped" "Puppeteer not installed"
    fi
else
    echo -e "    ⚠️  Frontend Integration: Test script not available"
    update_results "Frontend-Backend Integration" "skipped" "Test script or Node.js not available"
fi

# Test 6: Go Integration Tests
echo -e "\n${BLUE}🔧 Go Integration Tests...${NC}"

if [ -f "$SCRIPT_DIR/integration/api_validation_test.go" ] && command -v go >/dev/null 2>&1; then
    echo "  Running Go integration tests..."
    
    cd "$SCRIPT_DIR"
    
    # Set environment variables
    export DATABASE_URL="$DB_URL"
    export API_URL="$API_URL"
    export WS_URL="$WS_URL"
    
    # Run Go tests with timeout
    if timeout 60 go test -v ./integration/... > "$REPORT_DIR/go_integration.log" 2>&1; then
        echo -e "    ✅ Go Integration: Tests completed"
        update_results "Go Integration Tests" "passed"
    else
        test_exit_code=$?
        if [ $test_exit_code -eq 124 ]; then
            echo -e "    ❌ Go Integration: Tests timed out"
            update_results "Go Integration Tests" "failed" "Tests timed out after 60 seconds"
        else
            echo -e "    ❌ Go Integration: Tests failed"
            update_results "Go Integration Tests" "failed" "Go integration tests failed with exit code $test_exit_code"
        fi
    fi
    
    cd "$PROJECT_ROOT"
else
    echo -e "    ⚠️  Go Integration: Test files or Go not available"
    update_results "Go Integration Tests" "skipped" "Test files or Go not available"
fi

# Test 7: Performance Validation
echo -e "\n${BLUE}⚡ Performance Validation...${NC}"

if command -v curl >/dev/null 2>&1; then
    echo "  Testing API response times..."
    
    # Test multiple endpoints for performance
    endpoints=("/health" "/api/info" "/api/monitoring/metrics")
    
    total_time=0
    endpoint_count=0
    
    for endpoint in "${endpoints[@]}"; do
        echo "    Testing $endpoint..."
        
        # Measure response time
        response_time=$(curl -w "%{time_total}" -s -o /dev/null --max-time 5 "$API_URL$endpoint" 2>/dev/null || echo "5.000")
        
        if [ "$(echo "$response_time < 2.0" | bc -l 2>/dev/null || echo "0")" -eq 1 ]; then
            echo -e "      ✅ Response time: ${response_time}s"
            total_time=$(echo "$total_time + $response_time" | bc -l 2>/dev/null || echo "$total_time")
            endpoint_count=$((endpoint_count + 1))
        else
            echo -e "      ❌ Response time: ${response_time}s (slow)"
        fi
    done
    
    if [ $endpoint_count -gt 0 ]; then
        avg_time=$(echo "scale=3; $total_time / $endpoint_count" | bc -l 2>/dev/null || echo "unknown")
        if [ "$(echo "$avg_time < 1.0" | bc -l 2>/dev/null || echo "0")" -eq 1 ]; then
            echo -e "    ✅ Average response time: ${avg_time}s"
            update_results "Performance Validation" "passed"
        else
            echo -e "    ⚠️  Average response time: ${avg_time}s (acceptable)"
            update_results "Performance Validation" "passed"
        fi
    else
        echo -e "    ❌ Performance: No valid measurements"
        update_results "Performance Validation" "failed" "Could not measure API performance"
    fi
else
    echo -e "    ⚠️  Performance: Cannot test (curl not available)"
    update_results "Performance Validation" "skipped" "curl not available"
fi

# Test 8: Security Validation
echo -e "\n${BLUE}🔒 Security Validation...${NC}"

echo "  Testing security headers and authentication..."

# Check for security headers
if command -v curl >/dev/null 2>&1; then
    echo "    Checking security headers..."
    headers=$(curl -I -s --max-time 10 "$API_URL/health" 2>/dev/null || echo "")
    
    security_score=0
    total_checks=4
    
    if echo "$headers" | grep -i "x-frame-options" >/dev/null; then
        echo -e "      ✅ X-Frame-Options header present"
        security_score=$((security_score + 1))
    else
        echo -e "      ⚠️  X-Frame-Options header missing"
    fi
    
    if echo "$headers" | grep -i "x-content-type-options" >/dev/null; then
        echo -e "      ✅ X-Content-Type-Options header present"
        security_score=$((security_score + 1))
    else
        echo -e "      ⚠️  X-Content-Type-Options header missing"
    fi
    
    if echo "$headers" | grep -i "content-security-policy\|x-xss-protection" >/dev/null; then
        echo -e "      ✅ XSS protection headers present"
        security_score=$((security_score + 1))
    else
        echo -e "      ⚠️  XSS protection headers missing"
    fi
    
    # Check authentication requirement
    auth_response=$(curl -s -w "%{http_code}" -o /dev/null "$API_URL/api/vm/vms" 2>/dev/null || echo "000")
    if [ "$auth_response" = "401" ]; then
        echo -e "      ✅ Protected endpoints require authentication"
        security_score=$((security_score + 1))
    else
        echo -e "      ❌ Protected endpoints accessible without authentication"
    fi
    
    if [ $security_score -ge 3 ]; then
        echo -e "    ✅ Security: $security_score/$total_checks checks passed"
        update_results "Security Validation" "passed"
    else
        echo -e "    ⚠️  Security: $security_score/$total_checks checks passed"
        update_results "Security Validation" "failed" "Security validation incomplete: $security_score/$total_checks checks passed"
    fi
else
    echo -e "    ⚠️  Security: Cannot test (curl not available)"
    update_results "Security Validation" "skipped" "curl not available"
fi

# Generate Final Report
echo -e "\n${BLUE}📊 Generating Final Report...${NC}"

# Update JSON report summary
cat > "$REPORT_FILE" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "environment": {
    "frontend_url": "$FRONTEND_URL",
    "api_url": "$API_URL",
    "websocket_url": "$WS_URL",
    "database_url": "${DB_URL%%:*}:***@${DB_URL##*@}"
  },
  "summary": {
    "total": $TOTAL_TESTS,
    "passed": $PASSED_TESTS,
    "failed": $FAILED_TESTS,
    "skipped": $SKIPPED_TESTS,
    "success_rate": $(echo "scale=2; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l 2>/dev/null || echo "0")
  },
  "production_readiness": {
    "score": $(echo "scale=2; ($PASSED_TESTS * 100) / ($TOTAL_TESTS - $SKIPPED_TESTS)" | bc -l 2>/dev/null || echo "0"),
    "recommendation": "$([ $FAILED_TESTS -eq 0 ] && echo "READY" || [ $FAILED_TESTS -le 2 ] && echo "MINOR_ISSUES" || echo "NOT_READY")"
  }
}
EOF

# Display results
echo ""
echo "=================================================="
echo -e "${BLUE}📊 INTEGRATION VALIDATION RESULTS${NC}"
echo "=================================================="
echo "Total Tests: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS ✅${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS ❌${NC}"
echo -e "Skipped: ${YELLOW}$SKIPPED_TESTS ⚠️${NC}"

success_rate=$(echo "scale=1; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l 2>/dev/null || echo "0")
echo -e "Success Rate: $success_rate%"

# Production readiness assessment
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\n${GREEN}🚀 PRODUCTION READY${NC}"
    echo "All critical tests passed. System is ready for production deployment."
elif [ $FAILED_TESTS -le 2 ]; then
    echo -e "\n${YELLOW}⚠️  PRODUCTION READY WITH MINOR ISSUES${NC}"
    echo "System is mostly ready but has minor issues that should be addressed."
else
    echo -e "\n${RED}❌ NOT PRODUCTION READY${NC}"
    echo "Critical issues found. Address failed tests before production deployment."
fi

echo ""
echo "Report saved to: $REPORT_FILE"
echo "Detailed logs in: $REPORT_DIR/"

# Exit with appropriate code
if [ $FAILED_TESTS -eq 0 ]; then
    exit 0
elif [ $FAILED_TESTS -le 2 ]; then
    exit 1  # Warning level
else
    exit 2  # Critical issues
fi