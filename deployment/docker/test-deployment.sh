#!/bin/bash

# NovaCron Demo Testing Script
set -e

echo "🧪 Starting NovaCron Comprehensive Testing..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results
PASSED=0
FAILED=0
TESTS=()

# Function to run test
run_test() {
    local test_name=$1
    local test_command=$2
    local expected_result=${3:-0}
    
    echo -e "${BLUE}🔍 Testing: $test_name${NC}"
    
    if eval "$test_command" > /dev/null 2>&1; then
        if [ $? -eq $expected_result ]; then
            echo -e "${GREEN}✅ PASS: $test_name${NC}"
            PASSED=$((PASSED + 1))
            TESTS+=("✅ $test_name")
        else
            echo -e "${RED}❌ FAIL: $test_name${NC}"
            FAILED=$((FAILED + 1))
            TESTS+=("❌ $test_name")
        fi
    else
        echo -e "${RED}❌ FAIL: $test_name${NC}"
        FAILED=$((FAILED + 1))
        TESTS+=("❌ $test_name")
    fi
}

# Function to run API test with response check
run_api_test() {
    local test_name=$1
    local url=$2
    local expected_status=${3:-200}
    local auth_header=${4:-""}
    
    echo -e "${BLUE}🔍 Testing API: $test_name${NC}"
    
    local curl_cmd="curl -s -o /dev/null -w '%{http_code}'"
    if [ -n "$auth_header" ]; then
        curl_cmd="$curl_cmd -H '$auth_header'"
    fi
    curl_cmd="$curl_cmd $url"
    
    local response_code=$(eval $curl_cmd)
    
    if [ "$response_code" = "$expected_status" ]; then
        echo -e "${GREEN}✅ PASS: $test_name (Status: $response_code)${NC}"
        PASSED=$((PASSED + 1))
        TESTS+=("✅ $test_name")
    else
        echo -e "${RED}❌ FAIL: $test_name (Expected: $expected_status, Got: $response_code)${NC}"
        FAILED=$((FAILED + 1))
        TESTS+=("❌ $test_name")
    fi
}

echo "🏁 Starting Test Suite..."
echo ""

# 1. Service Availability Tests
echo -e "${YELLOW}📊 Service Availability Tests${NC}"
run_test "PostgreSQL Connection" "docker-compose -f docker-compose.demo.yml exec -T postgres pg_isready -U novacron"
run_test "Redis Connection" "docker-compose -f docker-compose.demo.yml exec -T redis redis-cli ping | grep -q PONG"
run_api_test "API Health Check" "http://localhost:15561/health" 200
run_api_test "Frontend Availability" "http://localhost:15566/" 200
run_api_test "Prometheus Health" "http://localhost:15564/-/healthy" 200
run_api_test "Grafana Health" "http://localhost:15565/api/health" 200

echo ""

# 2. Database Tests
echo -e "${YELLOW}🗄️ Database Tests${NC}"
run_test "Database Schema Exists" "docker-compose -f docker-compose.demo.yml exec -T postgres psql -U novacron -d novacron -c '\\dt' | grep -q users"
run_test "Demo Data Populated" "docker-compose -f docker-compose.demo.yml exec -T postgres psql -U novacron -d novacron -c 'SELECT COUNT(*) FROM users;' | grep -q '[1-9]'"
run_test "VM Data Exists" "docker-compose -f docker-compose.demo.yml exec -T postgres psql -U novacron -d novacron -c 'SELECT COUNT(*) FROM virtual_machines;' | grep -q '[1-9]'"

echo ""

# 3. API Endpoint Tests
echo -e "${YELLOW}🔗 API Endpoint Tests${NC}"

# Get auth token for API tests
echo "🔐 Obtaining authentication token..."
TOKEN_RESPONSE=$(curl -s -X POST http://localhost:15561/auth/login \
    -H "Content-Type: application/json" \
    -d '{"username":"admin","password":"admin"}')

if [ $? -eq 0 ]; then
    TOKEN=$(echo $TOKEN_RESPONSE | grep -o '"token":"[^"]*' | cut -d'"' -f4)
    if [ -n "$TOKEN" ]; then
        echo -e "${GREEN}✅ Authentication token obtained${NC}"
        AUTH_HEADER="Authorization: Bearer $TOKEN"
        
        # Authenticated API tests
        run_api_test "Get VMs List" "http://localhost:15561/api/vms" 200 "$AUTH_HEADER"
        run_api_test "Get Dashboard Stats" "http://localhost:15561/api/dashboard/stats" 200 "$AUTH_HEADER"
        run_api_test "Get System Metrics" "http://localhost:15561/api/metrics/system" 200 "$AUTH_HEADER"
        run_api_test "Get Hypervisors" "http://localhost:15561/api/hypervisors" 200 "$AUTH_HEADER"
        run_api_test "Get Users (Admin)" "http://localhost:15561/api/users" 200 "$AUTH_HEADER"
    else
        echo -e "${RED}❌ Failed to extract token from response${NC}"
        FAILED=$((FAILED + 5))
        TESTS+=("❌ Get VMs List")
        TESTS+=("❌ Get Dashboard Stats")
        TESTS+=("❌ Get System Metrics")
        TESTS+=("❌ Get Hypervisors")
        TESTS+=("❌ Get Users (Admin)")
    fi
else
    echo -e "${RED}❌ Failed to authenticate${NC}"
    FAILED=$((FAILED + 6))
    TESTS+=("❌ Authentication")
    TESTS+=("❌ Get VMs List")
    TESTS+=("❌ Get Dashboard Stats")
    TESTS+=("❌ Get System Metrics")
    TESTS+=("❌ Get Hypervisors")
    TESTS+=("❌ Get Users (Admin)")
fi

# Unauthenticated tests (should fail)
run_api_test "Unauthenticated VM Access" "http://localhost:15561/api/vms" 401

echo ""

# 4. Data Validation Tests
echo -e "${YELLOW}📊 Data Validation Tests${NC}"

if [ -n "$TOKEN" ]; then
    # Test VM data structure
    VM_RESPONSE=$(curl -s -H "$AUTH_HEADER" http://localhost:15561/api/vms)
    if echo "$VM_RESPONSE" | grep -q '"vms"' && echo "$VM_RESPONSE" | grep -q '"total"'; then
        echo -e "${GREEN}✅ PASS: VM Response Structure${NC}"
        PASSED=$((PASSED + 1))
        TESTS+=("✅ VM Response Structure")
    else
        echo -e "${RED}❌ FAIL: VM Response Structure${NC}"
        FAILED=$((FAILED + 1))
        TESTS+=("❌ VM Response Structure")
    fi
    
    # Test dashboard stats structure
    STATS_RESPONSE=$(curl -s -H "$AUTH_HEADER" http://localhost:15561/api/dashboard/stats)
    if echo "$STATS_RESPONSE" | grep -q '"vms"' && echo "$STATS_RESPONSE" | grep -q '"users"' && echo "$STATS_RESPONSE" | grep -q '"nodes"'; then
        echo -e "${GREEN}✅ PASS: Dashboard Stats Structure${NC}"
        PASSED=$((PASSED + 1))
        TESTS+=("✅ Dashboard Stats Structure")
    else
        echo -e "${RED}❌ FAIL: Dashboard Stats Structure${NC}"
        FAILED=$((FAILED + 1))
        TESTS+=("❌ Dashboard Stats Structure")
    fi
else
    echo -e "${YELLOW}⚠️ SKIP: Data validation tests (no token)${NC}"
    TESTS+=("⚠️ VM Response Structure - SKIPPED")
    TESTS+=("⚠️ Dashboard Stats Structure - SKIPPED")
fi

echo ""

# 5. Integration Tests
echo -e "${YELLOW}🔄 Integration Tests${NC}"

if [ -n "$TOKEN" ]; then
    # Test VM creation
    echo "🏗️ Testing VM creation..."
    CREATE_RESPONSE=$(curl -s -X POST http://localhost:15561/api/vms \
        -H "Content-Type: application/json" \
        -H "$AUTH_HEADER" \
        -d '{
            "name": "test-vm-'$(date +%s)'",
            "cpu_cores": 2,
            "memory_mb": 2048,
            "disk_gb": 40,
            "os_type": "Ubuntu 22.04 LTS"
        }')
    
    if echo "$CREATE_RESPONSE" | grep -q '"name"' && echo "$CREATE_RESPONSE" | grep -q '"id"'; then
        VM_ID=$(echo "$CREATE_RESPONSE" | grep -o '"id":"[^"]*' | cut -d'"' -f4)
        echo -e "${GREEN}✅ PASS: VM Creation (ID: $VM_ID)${NC}"
        PASSED=$((PASSED + 1))
        TESTS+=("✅ VM Creation")
        
        # Test VM retrieval
        if curl -s -H "$AUTH_HEADER" "http://localhost:15561/api/vms/$VM_ID" | grep -q '"name"'; then
            echo -e "${GREEN}✅ PASS: VM Retrieval${NC}"
            PASSED=$((PASSED + 1))
            TESTS+=("✅ VM Retrieval")
        else
            echo -e "${RED}❌ FAIL: VM Retrieval${NC}"
            FAILED=$((FAILED + 1))
            TESTS+=("❌ VM Retrieval")
        fi
        
        # Test VM deletion
        sleep 2
        if curl -s -X DELETE -H "$AUTH_HEADER" "http://localhost:15561/api/vms/$VM_ID" -w '%{http_code}' | grep -q '204'; then
            echo -e "${GREEN}✅ PASS: VM Deletion${NC}"
            PASSED=$((PASSED + 1))
            TESTS+=("✅ VM Deletion")
        else
            echo -e "${RED}❌ FAIL: VM Deletion${NC}"
            FAILED=$((FAILED + 1))
            TESTS+=("❌ VM Deletion")
        fi
    else
        echo -e "${RED}❌ FAIL: VM Creation${NC}"
        FAILED=$((FAILED + 3))
        TESTS+=("❌ VM Creation")
        TESTS+=("❌ VM Retrieval")
        TESTS+=("❌ VM Deletion")
    fi
else
    echo -e "${YELLOW}⚠️ SKIP: Integration tests (no token)${NC}"
    TESTS+=("⚠️ VM Creation - SKIPPED")
    TESTS+=("⚠️ VM Retrieval - SKIPPED")
    TESTS+=("⚠️ VM Deletion - SKIPPED")
fi

echo ""

# 6. Performance Tests
echo -e "${YELLOW}⚡ Performance Tests${NC}"

# Test response times
if [ -n "$TOKEN" ]; then
    RESPONSE_TIME=$(curl -s -o /dev/null -w '%{time_total}' -H "$AUTH_HEADER" http://localhost:15561/api/vms)
    if (( $(echo "$RESPONSE_TIME < 2.0" | bc -l) )); then
        echo -e "${GREEN}✅ PASS: API Response Time (${RESPONSE_TIME}s < 2s)${NC}"
        PASSED=$((PASSED + 1))
        TESTS+=("✅ API Response Time")
    else
        echo -e "${RED}❌ FAIL: API Response Time (${RESPONSE_TIME}s >= 2s)${NC}"
        FAILED=$((FAILED + 1))
        TESTS+=("❌ API Response Time")
    fi
else
    echo -e "${YELLOW}⚠️ SKIP: Performance tests (no token)${NC}"
    TESTS+=("⚠️ API Response Time - SKIPPED")
fi

echo ""

# 7. Monitoring Tests
echo -e "${YELLOW}📈 Monitoring Tests${NC}"

# Test Prometheus metrics
PROM_METRICS=$(curl -s http://localhost:15564/metrics)
if echo "$PROM_METRICS" | grep -q 'prometheus_build_info'; then
    echo -e "${GREEN}✅ PASS: Prometheus Metrics Available${NC}"
    PASSED=$((PASSED + 1))
    TESTS+=("✅ Prometheus Metrics Available")
else
    echo -e "${RED}❌ FAIL: Prometheus Metrics Available${NC}"
    FAILED=$((FAILED + 1))
    TESTS+=("❌ Prometheus Metrics Available")
fi

# Test Grafana API
GRAFANA_DATASOURCES=$(curl -s http://admin:admin123@localhost:15565/api/datasources)
if echo "$GRAFANA_DATASOURCES" | grep -q 'Prometheus'; then
    echo -e "${GREEN}✅ PASS: Grafana Prometheus Datasource${NC}"
    PASSED=$((PASSED + 1))
    TESTS+=("✅ Grafana Prometheus Datasource")
else
    echo -e "${RED}❌ FAIL: Grafana Prometheus Datasource${NC}"
    FAILED=$((FAILED + 1))
    TESTS+=("❌ Grafana Prometheus Datasource")
fi

echo ""

# Final Results
echo "🏁 Test Results Summary"
echo "======================="
echo ""

for test in "${TESTS[@]}"; do
    echo "  $test"
done

echo ""
echo -e "${GREEN}✅ Passed: $PASSED${NC}"
echo -e "${RED}❌ Failed: $FAILED${NC}"
echo "📊 Total: $((PASSED + FAILED))"

if [ $FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 All tests passed! NovaCron deployment is fully functional.${NC}"
    exit 0
else
    echo ""
    echo -e "${YELLOW}⚠️ Some tests failed. Please check the deployment.${NC}"
    exit 1
fi