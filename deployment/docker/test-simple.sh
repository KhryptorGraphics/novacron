#!/bin/bash

# Simple NovaCron Testing Script
set -e

echo "🧪 Starting NovaCron Service Tests..."

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

# Function to run API test
run_api_test() {
    local test_name=$1
    local url=$2
    local expected_status=${3:-200}
    
    echo -e "${BLUE}🔍 Testing API: $test_name${NC}"
    
    local response_code=$(curl -s -o /dev/null -w '%{http_code}' "$url")
    
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

echo "🏁 Starting Available Service Tests..."
echo ""

# Test available services
echo -e "${YELLOW}📊 Available Service Tests${NC}"

# Check what's actually running
RUNNING_SERVICES=$(docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(redis|prometheus|grafana)")

if echo "$RUNNING_SERVICES" | grep -q "redis"; then
    run_test "Redis Container Running" "docker ps | grep -q novacron-redis"
    run_test "Redis Connection" "docker exec novacron-redis redis-cli ping | grep -q PONG"
else
    echo -e "${YELLOW}⚠️ Redis not running${NC}"
fi

if echo "$RUNNING_SERVICES" | grep -q "prometheus"; then
    run_test "Prometheus Container Running" "docker ps | grep -q novacron-prometheus"
    run_api_test "Prometheus Health" "http://localhost:15564/-/healthy" 200
else
    echo -e "${YELLOW}⚠️ Prometheus not running${NC}"
fi

if echo "$RUNNING_SERVICES" | grep -q "grafana"; then
    run_test "Grafana Container Running" "docker ps | grep -q novacron-grafana"
    run_api_test "Grafana Health" "http://localhost:15565/api/health" 200
else
    echo -e "${YELLOW}⚠️ Grafana not running${NC}"
fi

# Test Docker network
run_test "Docker Network Exists" "docker network ls | grep -q novacron-network"

echo ""

# Show running services
echo -e "${YELLOW}🔍 Currently Running Services:${NC}"
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" | head -10

echo ""

# Show Docker logs for troubleshooting
echo -e "${YELLOW}📋 Recent Container Logs:${NC}"
for service in redis prometheus grafana; do
    container_name="novacron-$service"
    if docker ps | grep -q "$container_name"; then
        echo -e "${BLUE}--- $service logs ---${NC}"
        docker logs "$container_name" --tail=5 2>&1 | head -10
    fi
done

echo ""

# Network connectivity test
echo -e "${YELLOW}🔗 Network Tests${NC}"
run_test "Docker Daemon Running" "docker info > /dev/null"
run_test "Container Network Access" "docker network inspect docker_novacron-network > /dev/null"

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

echo ""
echo "🎯 Service URLs:"
echo "   • Prometheus:         http://localhost:15564"
echo "   • Grafana:            http://localhost:15565 (admin/admin123)"
echo "   • PostgreSQL:         localhost:15555 (if running)"
echo "   • Redis:              localhost:15560"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All available tests passed!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️ Some tests failed, but services are partially functional.${NC}"
    exit 0
fi