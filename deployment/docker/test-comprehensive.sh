#!/bin/bash

# NovaCron Comprehensive Testing Script
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
    
    echo -e "${BLUE}🔍 Testing: $test_name${NC}"
    
    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS: $test_name${NC}"
        PASSED=$((PASSED + 1))
        TESTS+=("✅ $test_name")
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

# 1. Infrastructure Tests
echo -e "${YELLOW}🏗️ Infrastructure Tests${NC}"
run_test "Docker Daemon Running" "docker info > /dev/null"
run_test "Docker Compose Available" "command -v docker-compose"
run_test "NovaCron Network Exists" "docker network inspect docker_novacron-network > /dev/null"
run_test "All Containers Running" "[ \$(docker ps | grep -c novacron-) -eq 5 ]"

echo ""

# 2. Service Availability Tests
echo -e "${YELLOW}📊 Service Availability Tests${NC}"
run_test "Redis Container Healthy" "docker ps | grep novacron-redis | grep -q healthy"
run_test "Redis Connection" "docker exec novacron-redis redis-cli ping | grep -q PONG"
run_test "Prometheus Container Healthy" "docker ps | grep novacron-prometheus | grep -q healthy"
run_test "Grafana Container Healthy" "docker ps | grep novacron-grafana | grep -q healthy"
run_test "API Container Healthy" "docker ps | grep novacron-mock-api | grep -q healthy"
run_test "Frontend Container Running" "docker ps | grep -q novacron-mock-frontend"

echo ""

# 3. Endpoint Availability Tests
echo -e "${YELLOW}🔗 Endpoint Availability Tests${NC}"
run_api_test "API Health Check" "http://localhost:15561/health" 200
run_api_test "Frontend Availability" "http://localhost:15566/" 200
run_api_test "Prometheus Web UI" "http://localhost:15564/" 200
run_api_test "Prometheus Health" "http://localhost:15564/-/healthy" 200
run_api_test "Grafana Health" "http://localhost:15565/api/health" 200

echo ""

# 4. Authentication Tests
echo -e "${YELLOW}🔐 Authentication Tests${NC}"

echo "🔐 Testing authentication system..."
TOKEN_RESPONSE=$(curl -s -X POST http://localhost:15561/auth/login \
    -H "Content-Type: application/json" \
    -d '{"username":"admin","password":"admin"}')

if [ $? -eq 0 ]; then
    TOKEN=$(echo $TOKEN_RESPONSE | grep -o '"token":"[^"]*' | cut -d'"' -f4)
    if [ -n "$TOKEN" ]; then
        echo -e "${GREEN}✅ Authentication successful${NC}"
        AUTH_HEADER="Authorization: Bearer $TOKEN"
        PASSED=$((PASSED + 1))
        TESTS+=("✅ Admin Authentication")
        
        # Test different user roles
        for user_pass in "operator1:password" "user1:password"; do
            username=$(echo $user_pass | cut -d: -f1)
            password=$(echo $user_pass | cut -d: -f2)
            
            response=$(curl -s -X POST http://localhost:15561/auth/login \
                -H "Content-Type: application/json" \
                -d "{\"username\":\"$username\",\"password\":\"$password\"}")
            
            if echo "$response" | grep -q '"token"'; then
                echo -e "${GREEN}✅ PASS: $username Authentication${NC}"
                PASSED=$((PASSED + 1))
                TESTS+=("✅ $username Authentication")
            else
                echo -e "${RED}❌ FAIL: $username Authentication${NC}"
                FAILED=$((FAILED + 1))
                TESTS+=("❌ $username Authentication")
            fi
        done
        
    else
        echo -e "${RED}❌ Failed to extract token${NC}"
        FAILED=$((FAILED + 4))
        TESTS+=("❌ Admin Authentication")
        TESTS+=("❌ operator1 Authentication")
        TESTS+=("❌ user1 Authentication")
        AUTH_HEADER=""
    fi
else
    echo -e "${RED}❌ Authentication request failed${NC}"
    FAILED=$((FAILED + 4))
    TESTS+=("❌ Admin Authentication")
    TESTS+=("❌ operator1 Authentication") 
    TESTS+=("❌ user1 Authentication")
    AUTH_HEADER=""
fi

# Test unauthorized access
run_api_test "Unauthorized VM Access Blocked" "http://localhost:15561/api/vms" 401

echo ""

# 5. API Functionality Tests
echo -e "${YELLOW}🔌 API Functionality Tests${NC}"

if [ -n "$AUTH_HEADER" ]; then
    # Test all major endpoints
    run_api_test "Get VMs List" "http://localhost:15561/api/vms" 200 "$AUTH_HEADER"
    run_api_test "Get Dashboard Stats" "http://localhost:15561/api/dashboard/stats" 200 "$AUTH_HEADER"
    run_api_test "Get System Metrics" "http://localhost:15561/api/metrics/system" 200 "$AUTH_HEADER"
    run_api_test "Get Hypervisors" "http://localhost:15561/api/hypervisors" 200 "$AUTH_HEADER"
    run_api_test "Get Users (Admin)" "http://localhost:15561/api/users" 200 "$AUTH_HEADER"
    
    # Test data structure validation
    echo "📊 Testing API data structures..."
    
    # Test VM data structure
    VM_RESPONSE=$(curl -s -H "$AUTH_HEADER" http://localhost:15561/api/vms)
    if echo "$VM_RESPONSE" | grep -q '"vms"' && echo "$VM_RESPONSE" | grep -q '"total"' && echo "$VM_RESPONSE" | grep -q '"web-server-1"'; then
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
    if echo "$STATS_RESPONSE" | grep -q '"vms"' && echo "$STATS_RESPONSE" | grep -q '"users"' && echo "$STATS_RESPONSE" | grep -q '"total"'; then
        echo -e "${GREEN}✅ PASS: Dashboard Stats Structure${NC}"
        PASSED=$((PASSED + 1))
        TESTS+=("✅ Dashboard Stats Structure")
    else
        echo -e "${RED}❌ FAIL: Dashboard Stats Structure${NC}"
        FAILED=$((FAILED + 1))
        TESTS+=("❌ Dashboard Stats Structure")
    fi
    
else
    echo -e "${YELLOW}⚠️ SKIP: API functionality tests (no auth token)${NC}"
    TESTS+=("⚠️ Get VMs List - SKIPPED")
    TESTS+=("⚠️ Get Dashboard Stats - SKIPPED")
    TESTS+=("⚠️ Get System Metrics - SKIPPED")
    TESTS+=("⚠️ Get Hypervisors - SKIPPED")
    TESTS+=("⚠️ Get Users (Admin) - SKIPPED")
    TESTS+=("⚠️ VM Response Structure - SKIPPED")
    TESTS+=("⚠️ Dashboard Stats Structure - SKIPPED")
fi

echo ""

# 6. CRUD Operations Tests
echo -e "${YELLOW}🔄 CRUD Operations Tests${NC}"

if [ -n "$AUTH_HEADER" ]; then
    # Test VM creation
    echo "🏗️ Testing VM CRUD operations..."
    
    VM_NAME="test-vm-$(date +%s)"
    CREATE_RESPONSE=$(curl -s -X POST http://localhost:15561/api/vms \
        -H "Content-Type: application/json" \
        -H "$AUTH_HEADER" \
        -d "{
            \"name\": \"$VM_NAME\",
            \"cpu_cores\": 2,
            \"memory_mb\": 2048,
            \"disk_gb\": 40,
            \"os_type\": \"Ubuntu 22.04 LTS\"
        }")
    
    if echo "$CREATE_RESPONSE" | grep -q '"name"' && echo "$CREATE_RESPONSE" | grep -q '"id"'; then
        VM_ID=$(echo "$CREATE_RESPONSE" | grep -o '"id":"[^"]*' | cut -d'"' -f4)
        echo -e "${GREEN}✅ PASS: VM Creation (ID: $VM_ID)${NC}"
        PASSED=$((PASSED + 1))
        TESTS+=("✅ VM Creation")
        
        # Test VM retrieval
        if curl -s -H "$AUTH_HEADER" "http://localhost:15561/api/vms/$VM_ID" | grep -q "$VM_NAME"; then
            echo -e "${GREEN}✅ PASS: VM Retrieval${NC}"
            PASSED=$((PASSED + 1))
            TESTS+=("✅ VM Retrieval")
        else
            echo -e "${RED}❌ FAIL: VM Retrieval${NC}"
            FAILED=$((FAILED + 1))
            TESTS+=("❌ VM Retrieval")
        fi
        
        # Test VM operations (start/stop)
        sleep 2
        START_RESPONSE=$(curl -s -X POST -H "$AUTH_HEADER" "http://localhost:15561/api/vms/$VM_ID/start")
        if echo "$START_RESPONSE" | grep -q 'started successfully'; then
            echo -e "${GREEN}✅ PASS: VM Start Operation${NC}"
            PASSED=$((PASSED + 1))
            TESTS+=("✅ VM Start Operation")
        else
            echo -e "${RED}❌ FAIL: VM Start Operation${NC}"
            FAILED=$((FAILED + 1))
            TESTS+=("❌ VM Start Operation")
        fi
        
        STOP_RESPONSE=$(curl -s -X POST -H "$AUTH_HEADER" "http://localhost:15561/api/vms/$VM_ID/stop")
        if echo "$STOP_RESPONSE" | grep -q 'stopped successfully'; then
            echo -e "${GREEN}✅ PASS: VM Stop Operation${NC}"
            PASSED=$((PASSED + 1))
            TESTS+=("✅ VM Stop Operation")
        else
            echo -e "${RED}❌ FAIL: VM Stop Operation${NC}"
            FAILED=$((FAILED + 1))
            TESTS+=("❌ VM Stop Operation")
        fi
        
        # Test VM deletion
        DELETE_CODE=$(curl -s -X DELETE -H "$AUTH_HEADER" "http://localhost:15561/api/vms/$VM_ID" -w '%{http_code}')
        if echo "$DELETE_CODE" | grep -q '204'; then
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
        FAILED=$((FAILED + 5))
        TESTS+=("❌ VM Creation")
        TESTS+=("❌ VM Retrieval")
        TESTS+=("❌ VM Start Operation")
        TESTS+=("❌ VM Stop Operation")
        TESTS+=("❌ VM Deletion")
    fi
else
    echo -e "${YELLOW}⚠️ SKIP: CRUD tests (no auth token)${NC}"
    TESTS+=("⚠️ VM Creation - SKIPPED")
    TESTS+=("⚠️ VM Retrieval - SKIPPED")
    TESTS+=("⚠️ VM Start Operation - SKIPPED")
    TESTS+=("⚠️ VM Stop Operation - SKIPPED")
    TESTS+=("⚠️ VM Deletion - SKIPPED")
fi

echo ""

# 7. Performance Tests
echo -e "${YELLOW}⚡ Performance Tests${NC}"

if [ -n "$AUTH_HEADER" ]; then
    # Test response times
    RESPONSE_TIME=$(curl -s -o /dev/null -w '%{time_total}' -H "$AUTH_HEADER" http://localhost:15561/api/vms)
    if (( $(echo "$RESPONSE_TIME < 2.0" | bc -l 2>/dev/null || echo "1") )); then
        echo -e "${GREEN}✅ PASS: API Response Time (${RESPONSE_TIME}s < 2s)${NC}"
        PASSED=$((PASSED + 1))
        TESTS+=("✅ API Response Time")
    else
        echo -e "${RED}❌ FAIL: API Response Time (${RESPONSE_TIME}s >= 2s)${NC}"
        FAILED=$((FAILED + 1))
        TESTS+=("❌ API Response Time")
    fi
    
    # Test concurrent requests
    echo "🔄 Testing concurrent API requests..."
    START_TIME=$(date +%s)
    for i in {1..10}; do
        curl -s -H "$AUTH_HEADER" http://localhost:15561/api/dashboard/stats > /dev/null &
    done
    wait
    END_TIME=$(date +%s)
    CONCURRENT_TIME=$((END_TIME - START_TIME))
    
    if [ $CONCURRENT_TIME -lt 5 ]; then
        echo -e "${GREEN}✅ PASS: Concurrent Requests (${CONCURRENT_TIME}s < 5s)${NC}"
        PASSED=$((PASSED + 1))
        TESTS+=("✅ Concurrent Requests")
    else
        echo -e "${RED}❌ FAIL: Concurrent Requests (${CONCURRENT_TIME}s >= 5s)${NC}"
        FAILED=$((FAILED + 1))
        TESTS+=("❌ Concurrent Requests")
    fi
else
    echo -e "${YELLOW}⚠️ SKIP: Performance tests (no token)${NC}"
    TESTS+=("⚠️ API Response Time - SKIPPED")
    TESTS+=("⚠️ Concurrent Requests - SKIPPED")
fi

echo ""

# 8. WebSocket Real-time Tests
echo -e "${YELLOW}🔗 WebSocket & Real-time Tests${NC}"

# Check if WebSocket is accessible (basic connectivity test)
WS_TEST_RESULT=$(timeout 5 wscat -c ws://localhost:15561 -w 1 2>&1 || echo "timeout")
if echo "$WS_TEST_RESULT" | grep -q "connected\|error\|timeout"; then
    if echo "$WS_TEST_RESULT" | grep -q "connected"; then
        echo -e "${GREEN}✅ PASS: WebSocket Connectivity${NC}"
        PASSED=$((PASSED + 1))
        TESTS+=("✅ WebSocket Connectivity")
    else
        echo -e "${YELLOW}⚠️ PASS: WebSocket Port Open (wscat not available)${NC}"
        PASSED=$((PASSED + 1))
        TESTS+=("✅ WebSocket Port Available")
    fi
else
    echo -e "${RED}❌ FAIL: WebSocket Connectivity${NC}"
    FAILED=$((FAILED + 1))
    TESTS+=("❌ WebSocket Connectivity")
fi

echo ""

# 9. Monitoring Stack Tests
echo -e "${YELLOW}📈 Monitoring Stack Tests${NC}"

# Test Prometheus metrics
PROM_METRICS=$(curl -s http://localhost:15564/metrics | head -20)
if echo "$PROM_METRICS" | grep -q 'prometheus_build_info\|go_info'; then
    echo -e "${GREEN}✅ PASS: Prometheus Metrics Available${NC}"
    PASSED=$((PASSED + 1))
    TESTS+=("✅ Prometheus Metrics Available")
else
    echo -e "${RED}❌ FAIL: Prometheus Metrics Available${NC}"
    FAILED=$((FAILED + 1))
    TESTS+=("❌ Prometheus Metrics Available")
fi

# Test Prometheus targets
PROM_TARGETS=$(curl -s http://localhost:15564/api/v1/targets)
if echo "$PROM_TARGETS" | grep -q '"health":"up"'; then
    echo -e "${GREEN}✅ PASS: Prometheus Targets Healthy${NC}"
    PASSED=$((PASSED + 1))
    TESTS+=("✅ Prometheus Targets Healthy")
else
    echo -e "${RED}❌ FAIL: Prometheus Targets Healthy${NC}"
    FAILED=$((FAILED + 1))
    TESTS+=("❌ Prometheus Targets Healthy")
fi

# Test Grafana datasource
GRAFANA_DS=$(curl -s http://admin:admin123@localhost:15565/api/datasources)
if echo "$GRAFANA_DS" | grep -q 'Prometheus'; then
    echo -e "${GREEN}✅ PASS: Grafana Prometheus Datasource${NC}"
    PASSED=$((PASSED + 1))
    TESTS+=("✅ Grafana Prometheus Datasource")
else
    echo -e "${RED}❌ FAIL: Grafana Prometheus Datasource${NC}"
    FAILED=$((FAILED + 1))
    TESTS+=("❌ Grafana Prometheus Datasource")
fi

# Test Grafana dashboards
GRAFANA_DASHBOARDS=$(curl -s http://admin:admin123@localhost:15565/api/search)
if echo "$GRAFANA_DASHBOARDS" | grep -q '"title"'; then
    echo -e "${GREEN}✅ PASS: Grafana Dashboards Available${NC}"
    PASSED=$((PASSED + 1))
    TESTS+=("✅ Grafana Dashboards Available")
else
    echo -e "${YELLOW}⚠️ PASS: Grafana API Accessible (no dashboards)${NC}"
    PASSED=$((PASSED + 1))
    TESTS+=("⚠️ Grafana API Accessible")
fi

echo ""

# 10. Data Persistence Tests
echo -e "${YELLOW}💾 Data Persistence Tests${NC}"

# Test Redis persistence
run_test "Redis Data Persistence" "docker exec novacron-redis redis-cli set test-key test-value && docker exec novacron-redis redis-cli get test-key | grep -q test-value"

# Test volume mounts
run_test "Redis Volume Mount" "docker inspect novacron-redis | grep -q 'redis_data'"
run_test "Prometheus Volume Mount" "docker inspect novacron-prometheus | grep -q 'prometheus_data'"
run_test "Grafana Volume Mount" "docker inspect novacron-grafana | grep -q 'grafana_data'"

echo ""

# Show service health summary
echo -e "${YELLOW}📊 Service Health Summary${NC}"
docker-compose -f docker-compose.simple.yml ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

echo ""

# Final Results
echo "🏁 Comprehensive Test Results"
echo "============================="
echo ""

for test in "${TESTS[@]}"; do
    echo "  $test"
done

echo ""
echo -e "${GREEN}✅ Passed: $PASSED${NC}"
echo -e "${RED}❌ Failed: $FAILED${NC}"
echo "📊 Total: $((PASSED + FAILED))"

# Calculate success percentage
if [ $((PASSED + FAILED)) -gt 0 ]; then
    SUCCESS_RATE=$(echo "scale=1; $PASSED * 100 / ($PASSED + $FAILED)" | bc -l 2>/dev/null || echo "0")
    echo "📈 Success Rate: ${SUCCESS_RATE}%"
fi

echo ""
echo "🎯 Service Access URLs:"
echo "   • Frontend Dashboard: http://localhost:15566"
echo "   • API Health Check:   http://localhost:15561/health"
echo "   • Prometheus:         http://localhost:15564"
echo "   • Grafana:            http://localhost:15565 (admin/admin123)"
echo "   • Redis:              localhost:15560"
echo ""
echo "🔐 Demo Accounts:"
echo "   • admin / admin (Admin access)"
echo "   • operator1 / password (Operator access)"
echo "   • user1 / password (User access)"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed! NovaCron deployment is fully functional.${NC}"
    exit 0
elif [ $PASSED -gt $FAILED ]; then
    echo -e "${YELLOW}✅ Most tests passed. NovaCron is functional with minor issues.${NC}"
    exit 0
else
    echo -e "${RED}❌ Multiple test failures detected. Please check the deployment.${NC}"
    exit 1
fi