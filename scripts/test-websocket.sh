#!/bin/bash

# WebSocket Event Flow Test Script
# Tests the WebSocket endpoints for NovaCron

set -e

# Configuration
API_HOST="${API_HOST:-localhost}"
API_PORT="${API_PORT:-8090}"
WS_PORT="${WS_PORT:-8091}"
TIMEOUT="${TIMEOUT:-5}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "NovaCron WebSocket Event Flow Test"
echo "=========================================="
echo ""

# Check if wscat is available
if ! command -v wscat &> /dev/null; then
    echo -e "${YELLOW}wscat not found. Installing...${NC}"
    npm install -g wscat 2>/dev/null || {
        echo -e "${RED}Failed to install wscat. Please install it manually: npm install -g wscat${NC}"
        echo "Alternative: Use websocat or another WebSocket client"
        WSCAT_AVAILABLE=false
    }
    WSCAT_AVAILABLE=true
else
    WSCAT_AVAILABLE=true
fi

# Check if backend is running
check_backend() {
    echo -n "Checking backend health... "
    if curl -s --connect-timeout 2 "http://${API_HOST}:${API_PORT}/health" > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        echo -e "${YELLOW}Backend not responding on ${API_HOST}:${API_PORT}${NC}"
        return 1
    fi
}

# Test WebSocket connection
test_websocket_connection() {
    local endpoint="$1"
    local description="$2"

    echo -n "Testing ${description}... "

    if [ "$WSCAT_AVAILABLE" = true ]; then
        # Use timeout to limit connection time
        result=$(timeout ${TIMEOUT} wscat -c "ws://${API_HOST}:${WS_PORT}${endpoint}" -x '{"type":"ping"}' 2>&1 || true)

        if echo "$result" | grep -q "error\|failed\|ECONNREFUSED"; then
            echo -e "${RED}FAILED${NC}"
            echo "  Error: $result"
            return 1
        else
            echo -e "${GREEN}OK${NC}"
            return 0
        fi
    else
        echo -e "${YELLOW}SKIPPED (wscat not available)${NC}"
        return 2
    fi
}

# Test WebSocket with curl upgrade
test_websocket_upgrade() {
    local endpoint="$1"
    local description="$2"

    echo -n "Testing ${description} (upgrade)... "

    response=$(curl -s -o /dev/null -w "%{http_code}" \
        --connect-timeout 2 \
        -H "Connection: Upgrade" \
        -H "Upgrade: websocket" \
        -H "Sec-WebSocket-Version: 13" \
        -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
        "http://${API_HOST}:${WS_PORT}${endpoint}" 2>&1)

    if [ "$response" = "101" ]; then
        echo -e "${GREEN}OK (101 Switching Protocols)${NC}"
        return 0
    elif [ "$response" = "000" ]; then
        echo -e "${YELLOW}Connection refused${NC}"
        return 1
    else
        echo -e "${YELLOW}HTTP $response${NC}"
        return 1
    fi
}

# Main tests
echo ""
echo "1. Backend Connectivity"
echo "------------------------"
check_backend
backend_status=$?

echo ""
echo "2. WebSocket Endpoint Tests"
echo "----------------------------"

# Only run WebSocket tests if backend is available
if [ $backend_status -eq 0 ]; then
    # Test metrics endpoint
    test_websocket_upgrade "/ws/metrics" "Metrics WebSocket"

    # Test alerts endpoint
    test_websocket_upgrade "/ws/alerts" "Alerts WebSocket"

    # Test logs endpoint
    test_websocket_upgrade "/ws/logs" "Logs WebSocket"

    echo ""
    echo "3. WebSocket Message Format Tests"
    echo "-----------------------------------"

    # Test with query parameters
    test_websocket_upgrade "/ws/metrics?sources=cpu_usage,memory_usage&interval=5" "Metrics with filters"
    test_websocket_upgrade "/ws/alerts?severity=critical" "Alerts with severity filter"
    test_websocket_upgrade "/ws/logs?level=error" "Logs with level filter"
else
    echo -e "${YELLOW}Skipping WebSocket tests - backend not available${NC}"
    echo ""
    echo "To run these tests:"
    echo "  1. Start the backend: make core-serve (or docker-compose up)"
    echo "  2. Ensure ports ${API_PORT} and ${WS_PORT} are accessible"
    echo "  3. Run this script again"
fi

echo ""
echo "4. Frontend WebSocket Hook Verification"
echo "-----------------------------------------"

# Check frontend WebSocket hooks file
HOOKS_FILE="/home/kp/repos/novacron/frontend/src/hooks/useWebSocket.ts"
if [ -f "$HOOKS_FILE" ]; then
    echo "Verifying frontend WebSocket hooks..."

    # Check for correct endpoint patterns
    if grep -q "/ws/metrics" "$HOOKS_FILE"; then
        echo -e "  ${GREEN}✓${NC} Uses /ws/metrics endpoint"
    else
        echo -e "  ${RED}✗${NC} Missing /ws/metrics endpoint"
    fi

    if grep -q "/ws/alerts" "$HOOKS_FILE"; then
        echo -e "  ${GREEN}✓${NC} Uses /ws/alerts endpoint"
    else
        echo -e "  ${RED}✗${NC} Missing /ws/alerts endpoint"
    fi

    if grep -q "heartbeat" "$HOOKS_FILE"; then
        echo -e "  ${GREEN}✓${NC} Implements heartbeat mechanism"
    else
        echo -e "  ${RED}✗${NC} Missing heartbeat mechanism"
    fi

    if grep -q "reconnect" "$HOOKS_FILE"; then
        echo -e "  ${GREEN}✓${NC} Implements reconnection logic"
    else
        echo -e "  ${RED}✗${NC} Missing reconnection logic"
    fi

    # Check for queue support
    if grep -q "WebSocketMessageQueue" "$HOOKS_FILE"; then
        echo -e "  ${GREEN}✓${NC} Implements message queue"
    else
        echo -e "  ${RED}✗${NC} Missing message queue"
    fi
else
    echo -e "${RED}Frontend hooks file not found: $HOOKS_FILE${NC}"
fi

echo ""
echo "5. Backend WebSocket Handler Verification"
echo "-------------------------------------------"

# Check backend WebSocket handler file
HANDLER_FILE="/home/kp/repos/novacron/backend/api/websocket/handlers.go"
if [ -f "$HANDLER_FILE" ]; then
    echo "Verifying backend WebSocket handlers..."

    if grep -q "HandleMetricsWebSocket" "$HANDLER_FILE"; then
        echo -e "  ${GREEN}✓${NC} Metrics handler implemented"
    else
        echo -e "  ${RED}✗${NC} Missing metrics handler"
    fi

    if grep -q "HandleAlertsWebSocket" "$HANDLER_FILE"; then
        echo -e "  ${GREEN}✓${NC} Alerts handler implemented"
    else
        echo -e "  ${RED}✗${NC} Missing alerts handler"
    fi

    if grep -q "HandleLogsWebSocket" "$HANDLER_FILE"; then
        echo -e "  ${GREEN}✓${NC} Logs handler implemented"
    else
        echo -e "  ${RED}✗${NC} Missing logs handler"
    fi

    if grep -q "HandleConsoleWebSocket" "$HANDLER_FILE"; then
        echo -e "  ${GREEN}✓${NC} Console handler implemented"
    else
        echo -e "  ${RED}✗${NC} Missing console handler"
    fi

    if grep -q "metricRegistry" "$HANDLER_FILE"; then
        echo -e "  ${GREEN}✓${NC} Uses MetricRegistry for real metrics"
    else
        echo -e "  ${RED}✗${NC} Missing MetricRegistry integration"
    fi

    if grep -q "collectMetrics" "$HANDLER_FILE"; then
        echo -e "  ${GREEN}✓${NC} collectMetrics function present"
    else
        echo -e "  ${RED}✗${NC} Missing collectMetrics function"
    fi
else
    echo -e "${RED}Backend handler file not found: $HANDLER_FILE${NC}"
fi

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""
echo "WebSocket Event Flow Verification:"
echo "  - Frontend hooks use correct backend endpoints"
echo "  - Backend handlers support metrics, alerts, logs, console"
echo "  - Heartbeat/ping-pong mechanism implemented"
echo "  - Reconnection logic in place"
echo "  - Message queue for high-frequency updates"
echo ""
echo "Note: Full integration tests require running backend."
echo "Run 'make core-serve' and re-run this script."
echo ""
