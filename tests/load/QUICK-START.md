# Load Testing Quick Start Guide

## Prerequisites Check
```bash
# 1. Verify k6 is installed
k6 version
# Expected: k6 v0.48.0 or higher

# 2. Check NovaCron services are running
curl http://localhost:8080/api/v1/health
# Expected: 200 OK

# 3. Navigate to test directory
cd /home/kp/novacron/tests/load
```

## Run All Tests (Recommended)
```bash
./run_all_tests.sh
```

This will execute all four test suites:
1. API Load Test (~30 minutes)
2. DWCP Protocol Test (~30 minutes)
3. WebSocket Test (~30 minutes)
4. Database Test (~30 minutes)

**Total Duration**: ~2 hours

## Run Individual Tests

### 1. API Load Test (REST Endpoints)
```bash
k6 run api_load_test.js
```
Tests: VM CRUD operations, search, filtering
Load: 100 → 1K → 10K users
Duration: ~23 minutes

### 2. DWCP Protocol Test (VM Migrations)
```bash
k6 run dwcp_load_test.js
```
Tests: Live migrations, DWCP v3 protocol
Load: 50 → 100 → 500 → 1K migrations
Duration: ~21 minutes

### 3. WebSocket Test (Real-time Connections)
```bash
k6 run websocket_load_test.js
```
Tests: Concurrent connections, message streaming
Load: 100 → 1K → 10K → 100K connections
Duration: ~27 minutes

### 4. Database Test (Query Performance)
```bash
k6 run database_load_test.js
```
Tests: Reads, writes, complex queries, transactions
Load: 200 → 500 → 2K → 5K queries/sec
Duration: ~19 minutes

## Custom Configuration

Set environment variables before running:
```bash
export API_URL=http://your-api-server:8080
export WS_URL=ws://your-websocket-server:8080
export DWCP_WS_URL=ws://your-dwcp-server:8080/dwcp/v3
export API_TOKEN=your-authentication-token

# Then run tests
./run_all_tests.sh
```

## View Results

Results are saved in `results/TIMESTAMP/`:
```bash
# View combined summary
cat results/*/SUMMARY.md | less

# View individual test logs
ls -lh results/*/

# View JSON metrics
cat results/*/api_load_full_summary.json | jq
```

## Quick Performance Check

After tests complete, check these key metrics:

### API Test
```bash
cat results/*/api_load_full.log | grep "http_req_duration"
```
Target: P95 < 500ms

### DWCP Test
```bash
cat results/*/dwcp_load_full.log | grep "migration_duration"
```
Target: P95 < 30s

### WebSocket Test
```bash
cat results/*/websocket_load_full.log | grep "ws_connections"
```
Target: >10K connections

### Database Test
```bash
cat results/*/database_load_full.log | grep "db_query_latency"
```
Target: P95 < 300ms

## Troubleshooting

### k6 Not Installed
```bash
curl https://github.com/grafana/k6/releases/download/v0.48.0/k6-v0.48.0-linux-amd64.tar.gz -L | tar xvz
sudo mv k6-v0.48.0-linux-amd64/k6 /usr/local/bin/
```

### Services Not Running
```bash
# Check API
curl -I http://localhost:8080/api/v1/health

# Check WebSocket
wscat -c ws://localhost:8080/ws

# Check Database connection
# (Application-specific command)
```

### Connection Refused
- Verify firewall rules
- Check service ports are open
- Validate URL configuration

### Tests Taking Too Long
You can reduce test duration by editing the test files:
```javascript
// Reduce stages in api_load_test.js
export const options = {
  stages: [
    { duration: '1m', target: 100 },  // Shortened
    { duration: '2m', target: 1000 }, // Shortened
    { duration: '1m', target: 0 },    // Shortened
  ],
};
```

## Performance Targets

| Component | Metric | Target |
|-----------|--------|--------|
| API | P95 Latency | < 500ms |
| API | Error Rate | < 1% |
| DWCP | Migration Time | < 30s (P95) |
| DWCP | Error Rate | < 2% |
| WebSocket | Message Latency | < 500ms (P95) |
| WebSocket | Connections | > 10,000 |
| Database | Query Latency | < 300ms (P95) |
| Database | Error Rate | < 1% |

## Next Steps After Testing

1. **Review SUMMARY.md** in results directory
2. **Identify bottlenecks** from test logs
3. **Document baselines** in /home/kp/novacron/docs/LOAD-TEST-RESULTS.md
4. **Implement optimizations** based on findings
5. **Re-run tests** to validate improvements

## Support

For detailed information, see:
- [Full Documentation](README.md)
- [Load Test Results](../../docs/LOAD-TEST-RESULTS.md)
- [Test Files](.)

---
NovaCron Load Testing Suite | k6 v0.48.0
