#!/bin/bash

# NovaCron Comprehensive Load Testing Suite
# Tests API, DWCP, Database, and WebSocket performance at scale

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_URL=${API_URL:-http://localhost:8080}
WS_URL=${WS_URL:-ws://localhost:8080}
DWCP_WS_URL=${DWCP_WS_URL:-ws://localhost:8080/dwcp/v3}
API_TOKEN=${API_TOKEN:-test-token-123}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  NovaCron Load Testing Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "API URL: ${GREEN}$API_URL${NC}"
echo -e "WebSocket URL: ${GREEN}$WS_URL${NC}"
echo -e "DWCP URL: ${GREEN}$DWCP_WS_URL${NC}"
echo -e "Results Directory: ${GREEN}$RESULTS_DIR/$TIMESTAMP${NC}"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR/$TIMESTAMP"

# Function to run a test
run_test() {
    local test_name=$1
    local test_file=$2
    local scale=$3
    local duration=$4

    echo -e "${YELLOW}Running $test_name ($scale scale)...${NC}"

    k6 run \
        --out json="$RESULTS_DIR/$TIMESTAMP/${test_name}_${scale}.json" \
        --summary-export="$RESULTS_DIR/$TIMESTAMP/${test_name}_${scale}_summary.json" \
        --env API_URL="$API_URL" \
        --env WS_URL="$WS_URL" \
        --env DWCP_WS_URL="$DWCP_WS_URL" \
        --env API_TOKEN="$API_TOKEN" \
        "$SCRIPT_DIR/$test_file" \
        2>&1 | tee "$RESULTS_DIR/$TIMESTAMP/${test_name}_${scale}.log"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $test_name ($scale) completed successfully${NC}\n"
    else
        echo -e "${RED}✗ $test_name ($scale) failed${NC}\n"
    fi
}

# Check if k6 is installed
if ! command -v k6 &> /dev/null; then
    echo -e "${RED}Error: k6 is not installed${NC}"
    echo "Install with: curl https://github.com/grafana/k6/releases/download/v0.48.0/k6-v0.48.0-linux-amd64.tar.gz -L | tar xvz && sudo mv k6-v0.48.0-linux-amd64/k6 /usr/local/bin/"
    exit 1
fi

echo -e "${BLUE}Starting load tests...${NC}\n"

# Test 1: API Load Test
echo -e "${BLUE}=== Phase 1: API Load Testing ===${NC}"
run_test "api_load" "api_load_test.js" "full" "30m"

# Test 2: DWCP Protocol Load Test
echo -e "${BLUE}=== Phase 2: DWCP Protocol Testing ===${NC}"
run_test "dwcp_load" "dwcp_load_test.js" "full" "30m"

# Test 3: WebSocket Load Test
echo -e "${BLUE}=== Phase 3: WebSocket Testing ===${NC}"
run_test "websocket_load" "websocket_load_test.js" "full" "30m"

# Test 4: Database Load Test
echo -e "${BLUE}=== Phase 4: Database Testing ===${NC}"
run_test "database_load" "database_load_test.js" "full" "30m"

# Generate combined report
echo -e "${BLUE}=== Generating Combined Report ===${NC}"

cat > "$RESULTS_DIR/$TIMESTAMP/SUMMARY.md" << EOF
# Load Testing Results - Complete Suite
**Date:** $(date)
**Duration:** Full test suite
**k6 Version:** $(k6 version | head -n1)

## Test Configuration
- API URL: $API_URL
- WebSocket URL: $WS_URL
- DWCP URL: $DWCP_WS_URL

## Test Results

### 1. API Load Test
EOF

# Extract key metrics from each test
if [ -f "$RESULTS_DIR/$TIMESTAMP/api_load_full_summary.json" ]; then
    echo "Processing API test results..."
    python3 << PYTHON_SCRIPT >> "$RESULTS_DIR/$TIMESTAMP/SUMMARY.md"
import json
import sys

try:
    with open('$RESULTS_DIR/$TIMESTAMP/api_load_full_summary.json', 'r') as f:
        data = json.load(f)

    metrics = data.get('metrics', {})

    print("\n#### Performance Metrics")

    if 'http_reqs' in metrics:
        print(f"- Total Requests: {metrics['http_reqs']['values']['count']}")

    if 'http_req_duration' in metrics:
        print(f"- P95 Latency: {metrics['http_req_duration']['values']['p(95)']:.2f}ms")
        print(f"- P99 Latency: {metrics['http_req_duration']['values']['p(99)']:.2f}ms")
        print(f"- Avg Latency: {metrics['http_req_duration']['values']['avg']:.2f}ms")

    if 'http_req_failed' in metrics:
        fail_rate = metrics['http_req_failed']['values']['rate'] * 100
        print(f"- Error Rate: {fail_rate:.2f}%")
        print(f"- Status: {'✅ PASS' if fail_rate < 1 else '❌ FAIL'}")

except Exception as e:
    print(f"\nError processing results: {e}")
    sys.exit(1)
PYTHON_SCRIPT
fi

cat >> "$RESULTS_DIR/$TIMESTAMP/SUMMARY.md" << EOF

### 2. DWCP Protocol Test
EOF

if [ -f "$RESULTS_DIR/$TIMESTAMP/dwcp_load_full_summary.json" ]; then
    echo "Processing DWCP test results..."
    python3 << PYTHON_SCRIPT >> "$RESULTS_DIR/$TIMESTAMP/SUMMARY.md"
import json

try:
    with open('$RESULTS_DIR/$TIMESTAMP/dwcp_load_full_summary.json', 'r') as f:
        data = json.load(f)

    metrics = data.get('metrics', {})

    print("\n#### Migration Performance")

    if 'vm_migrations_total' in metrics:
        print(f"- Total Migrations: {metrics['vm_migrations_total']['values']['count']}")

    if 'migration_duration' in metrics:
        print(f"- P95 Duration: {metrics['migration_duration']['values']['p(95)'] / 1000:.2f}s")
        print(f"- Avg Duration: {metrics['migration_duration']['values']['avg'] / 1000:.2f}s")

    if 'dwcp_errors' in metrics:
        error_rate = metrics['dwcp_errors']['values']['rate'] * 100
        print(f"- Error Rate: {error_rate:.2f}%")
        print(f"- Status: {'✅ PASS' if error_rate < 2 else '❌ FAIL'}")

except Exception as e:
    print(f"\nError processing results: {e}")
PYTHON_SCRIPT
fi

cat >> "$RESULTS_DIR/$TIMESTAMP/SUMMARY.md" << EOF

### 3. WebSocket Test
EOF

if [ -f "$RESULTS_DIR/$TIMESTAMP/websocket_load_full_summary.json" ]; then
    echo "Processing WebSocket test results..."
    python3 << PYTHON_SCRIPT >> "$RESULTS_DIR/$TIMESTAMP/SUMMARY.md"
import json

try:
    with open('$RESULTS_DIR/$TIMESTAMP/websocket_load_full_summary.json', 'r') as f:
        data = json.load(f)

    metrics = data.get('metrics', {})

    print("\n#### Connection Performance")

    if 'ws_connections_established' in metrics:
        print(f"- Connections Established: {metrics['ws_connections_established']['values']['count']}")

    if 'ws_messages_received' in metrics:
        print(f"- Messages Received: {metrics['ws_messages_received']['values']['count']}")

    if 'ws_message_latency' in metrics:
        print(f"- P95 Latency: {metrics['ws_message_latency']['values']['p(95)']:.2f}ms")

    if 'ws_errors' in metrics:
        error_rate = metrics['ws_errors']['values']['rate'] * 100
        print(f"- Error Rate: {error_rate:.2f}%")
        print(f"- Status: {'✅ PASS' if error_rate < 5 else '❌ FAIL'}")

except Exception as e:
    print(f"\nError processing results: {e}")
PYTHON_SCRIPT
fi

cat >> "$RESULTS_DIR/$TIMESTAMP/SUMMARY.md" << EOF

### 4. Database Test
EOF

if [ -f "$RESULTS_DIR/$TIMESTAMP/database_load_full_summary.json" ]; then
    echo "Processing Database test results..."
    python3 << PYTHON_SCRIPT >> "$RESULTS_DIR/$TIMESTAMP/SUMMARY.md"
import json

try:
    with open('$RESULTS_DIR/$TIMESTAMP/database_load_full_summary.json', 'r') as f:
        data = json.load(f)

    metrics = data.get('metrics', {})

    print("\n#### Query Performance")

    if 'db_queries_total' in metrics:
        print(f"- Total Queries: {metrics['db_queries_total']['values']['count']}")

    if 'db_reads_total' in metrics and 'db_writes_total' in metrics:
        reads = metrics['db_reads_total']['values']['count']
        writes = metrics['db_writes_total']['values']['count']
        print(f"- Reads: {reads}, Writes: {writes}")

    if 'db_query_latency' in metrics:
        print(f"- P95 Query Latency: {metrics['db_query_latency']['values']['p(95)']:.2f}ms")

    if 'db_errors' in metrics:
        error_rate = metrics['db_errors']['values']['rate'] * 100
        print(f"- Error Rate: {error_rate:.2f}%")
        print(f"- Status: {'✅ PASS' if error_rate < 1 else '❌ FAIL'}")

except Exception as e:
    print(f"\nError processing results: {e}")
PYTHON_SCRIPT
fi

cat >> "$RESULTS_DIR/$TIMESTAMP/SUMMARY.md" << EOF

## Performance Baselines Established

### API Endpoints
- Throughput: Based on test results above
- P95 Latency: Target < 500ms
- Error Rate: Target < 1%

### DWCP Protocol
- Migration Speed: Target < 30s (P95)
- Concurrent Migrations: Target > 100
- Error Rate: Target < 2%

### WebSocket Connections
- Concurrent Connections: Target > 10,000
- Message Latency: Target < 500ms (P95)
- Connection Success Rate: Target > 95%

### Database Operations
- Query Throughput: Based on results
- Read Latency: Target < 200ms (P95)
- Write Latency: Target < 500ms (P95)
- Error Rate: Target < 1%

## Recommendations

1. **API Optimization**: [Review API test logs for specific recommendations]
2. **DWCP Tuning**: [Review DWCP test logs for migration optimizations]
3. **WebSocket Scaling**: [Review WebSocket test logs for connection handling]
4. **Database Indexing**: [Review database test logs for query optimization]

## Next Steps

- [ ] Review detailed test logs in: \`$RESULTS_DIR/$TIMESTAMP/\`
- [ ] Address any failed tests
- [ ] Implement recommended optimizations
- [ ] Re-run tests to validate improvements
- [ ] Set up continuous performance monitoring

---
Generated by NovaCron Load Testing Suite
EOF

echo -e "${GREEN}✓ Combined report generated: $RESULTS_DIR/$TIMESTAMP/SUMMARY.md${NC}"

# Display summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Load Testing Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Results saved to: ${GREEN}$RESULTS_DIR/$TIMESTAMP/${NC}"
echo ""
echo -e "${YELLOW}View summary:${NC} cat $RESULTS_DIR/$TIMESTAMP/SUMMARY.md"
echo -e "${YELLOW}View detailed logs:${NC} ls -lh $RESULTS_DIR/$TIMESTAMP/"
echo ""
