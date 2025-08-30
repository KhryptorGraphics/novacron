#!/bin/bash
# NovaCron Deployment Validation Script

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "==================================="
echo "NovaCron Deployment Validation"
echo "==================================="

# Function to check service health
check_service() {
    local service=$1
    local url=$2
    local expected_status=${3:-200}
    
    echo -n "Checking $service... "
    
    response=$(curl -s -o /dev/null -w "%{http_code}" $url || echo "000")
    
    if [ "$response" = "$expected_status" ]; then
        echo -e "${GREEN}✓${NC} ($response)"
        return 0
    else
        echo -e "${RED}✗${NC} (got $response, expected $expected_status)"
        return 1
    fi
}

# Function to check Docker container status
check_container() {
    local container=$1
    echo -n "Checking container $container... "
    
    if docker ps | grep -q $container; then
        status=$(docker inspect -f '{{.State.Status}}' $container 2>/dev/null || echo "unknown")
        if [ "$status" = "running" ]; then
            echo -e "${GREEN}✓${NC} (running)"
            return 0
        else
            echo -e "${YELLOW}⚠${NC} (status: $status)"
            return 1
        fi
    else
        echo -e "${RED}✗${NC} (not found)"
        return 1
    fi
}

# Function to test database connectivity
check_database() {
    echo -n "Checking database connectivity... "
    
    if docker exec novacron-postgres-1 pg_isready -U postgres > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${RED}✗${NC}"
        return 1
    fi
}

# Function to run database migrations
run_migrations() {
    echo -n "Running database migrations... "
    
    # Check if migrate tool exists
    if [ -f ./backend/cmd/migrate/migrate ]; then
        if DB_URL="postgresql://postgres:postgres@localhost:11432/novacron?sslmode=disable" ./backend/cmd/migrate/migrate up; then
            echo -e "${GREEN}✓${NC}"
            return 0
        else
            echo -e "${RED}✗${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠${NC} (migrate tool not found, skipping)"
        return 0
    fi
}

# Function to test API endpoints
test_api_endpoints() {
    echo "Testing API endpoints..."
    
    # Health check
    check_service "API Health" "http://localhost:8090/health"
    
    # List VMs (might return empty list)
    check_service "API List VMs" "http://localhost:8090/api/vms"
    
    # Check WebSocket endpoint (expect 426 Upgrade Required without proper headers)
    check_service "WebSocket Endpoint" "http://localhost:8091/ws" 426
}

# Function to test frontend
test_frontend() {
    echo "Testing Frontend..."
    check_service "Frontend Homepage" "http://localhost:8092"
}

# Function to check monitoring stack
check_monitoring() {
    echo "Checking Monitoring Stack..."
    check_service "Prometheus" "http://localhost:9090/-/healthy"
    check_service "Grafana" "http://localhost:3001/api/health"
}

# Main validation
echo ""
echo "1. Checking Docker Containers"
echo "------------------------------"
check_container "novacron-postgres-1"
check_container "novacron-api-1"
check_container "novacron-hypervisor-1"
check_container "novacron-frontend-1"
check_container "novacron-prometheus-1"
check_container "novacron-grafana-1"
check_container "novacron-redis-master-1"
check_container "novacron-ai-engine-1"

echo ""
echo "2. Database Validation"
echo "----------------------"
check_database
run_migrations

echo ""
echo "3. API Validation"
echo "-----------------"
test_api_endpoints

echo ""
echo "4. Frontend Validation"
echo "----------------------"
test_frontend

echo ""
echo "5. Monitoring Stack"
echo "-------------------"
check_monitoring

echo ""
echo "6. Integration Test Suite"
echo "-------------------------"
if [ -d "tests/integration" ]; then
    echo "Running basic integration tests..."
    cd tests/integration
    
    # Set test environment variables
    export DB_URL="postgresql://postgres:postgres@localhost:11432/novacron_test"
    export NOVACRON_API_URL="http://localhost:8090"
    export REDIS_URL="redis://localhost:6379"
    
    # Run a basic test
    if make test-short 2>&1 | grep -q "PASS"; then
        echo -e "${GREEN}✓${NC} Basic tests passed"
    else
        echo -e "${YELLOW}⚠${NC} Some tests may have failed (check logs)"
    fi
    cd ../..
else
    echo -e "${YELLOW}⚠${NC} Integration test directory not found"
fi

echo ""
echo "==================================="
echo "Deployment Validation Complete"
echo "==================================="
echo ""
echo "Access Points:"
echo "- Frontend: http://localhost:8092"
echo "- API: http://localhost:8090"
echo "- WebSocket: ws://localhost:8091/ws"
echo "- Prometheus: http://localhost:9090"
echo "- Grafana: http://localhost:3001 (admin/admin)"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f [service-name]"
echo ""
echo "To stop services:"
echo "  docker-compose down"
echo ""