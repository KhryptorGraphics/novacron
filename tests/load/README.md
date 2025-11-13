# NovaCron Load Testing Suite

Comprehensive performance testing for NovaCron at scale (1K, 10K, 100K VMs).

## Overview

This suite tests four critical components:
1. **API Load Test** - REST API endpoints under concurrent load
2. **DWCP Protocol Test** - VM migration performance with DWCP v3
3. **WebSocket Test** - Real-time connection and message handling
4. **Database Test** - Query performance and connection pooling

## Prerequisites

- k6 v0.48.0+ installed
- NovaCron services running (API, DWCP, WebSocket, Database)
- Python 3 (for report generation)

## Quick Start

### Run All Tests
```bash
cd tests/load
chmod +x run_all_tests.sh
./run_all_tests.sh
```

### Run Individual Tests

#### API Load Test
```bash
k6 run api_load_test.js
```

#### DWCP Protocol Test
```bash
k6 run dwcp_load_test.js
```

#### WebSocket Test
```bash
k6 run websocket_load_test.js
```

#### Database Test
```bash
k6 run database_load_test.js
```

## Configuration

Set environment variables to customize tests:

```bash
export API_URL=http://your-api-url:8080
export WS_URL=ws://your-ws-url:8080
export DWCP_WS_URL=ws://your-dwcp-url:8080/dwcp/v3
export API_TOKEN=your-auth-token
```

## Test Stages

All tests follow a progressive load pattern:

1. **Warm-up** - Gradual ramp to baseline load
2. **Baseline** - Sustain moderate load (100-1K users)
3. **Ramp-up** - Increase to target load (1K-10K)
4. **Sustain** - Hold at target load for 5+ minutes
5. **Stress** - Push to maximum load (10K-100K)
6. **Ramp-down** - Graceful shutdown

## Performance Targets

### API Endpoints
- **Throughput**: 10,000+ req/sec
- **Latency (P95)**: < 500ms
- **Error Rate**: < 1%

### DWCP Protocol
- **Migration Time (P95)**: < 30s
- **Concurrent Migrations**: 1,000+
- **Error Rate**: < 2%

### WebSocket Connections
- **Concurrent Connections**: 100,000+
- **Message Latency (P95)**: < 500ms
- **Connection Success**: > 95%

### Database Operations
- **Query Throughput**: 5,000+ queries/sec
- **Read Latency (P95)**: < 200ms
- **Write Latency (P95)**: < 500ms
- **Error Rate**: < 1%

## Metrics Collected

Each test collects:
- Request/response times (min, avg, p95, p99, max)
- Throughput (requests/second, messages/second)
- Error rates and failure types
- Custom business metrics (migrations, connections, queries)

## Results

Results are saved in `tests/load/results/TIMESTAMP/`:
- `SUMMARY.md` - Combined report with all test results
- `*_summary.json` - Structured metrics in JSON format
- `*.log` - Detailed console output
- `*.json` - Full k6 metrics export

## Interpreting Results

### Success Criteria
✅ **PASS** - All thresholds met, error rate within limits
⚠️ **WARNING** - Some thresholds exceeded but system functional
❌ **FAIL** - Critical failures or excessive error rates

### Common Issues

1. **High Latency**: Check database indexes, add caching
2. **Connection Timeouts**: Increase connection pool size
3. **Memory Pressure**: Optimize query patterns, add pagination
4. **WebSocket Drops**: Tune keepalive settings, add load balancer

## Advanced Usage

### Custom Test Scenarios

Modify test files to add custom scenarios:

```javascript
export const options = {
  scenarios: {
    spike_test: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '10s', target: 100 },
        { duration: '1m', target: 10000 }, // Spike!
        { duration: '10s', target: 0 },
      ],
    },
  },
};
```

### Cloud Testing

Run tests from k6 Cloud for distributed load:

```bash
k6 cloud api_load_test.js
```

### CI/CD Integration

Add to your pipeline:

```yaml
test:
  script:
    - ./tests/load/run_all_tests.sh
  artifacts:
    paths:
      - tests/load/results/
```

## Troubleshooting

### k6 Installation
```bash
curl https://github.com/grafana/k6/releases/download/v0.48.0/k6-v0.48.0-linux-amd64.tar.gz -L | tar xvz
sudo mv k6-v0.48.0-linux-amd64/k6 /usr/local/bin/
k6 version
```

### Connection Refused
- Verify services are running
- Check firewall rules
- Validate API_URL and WS_URL

### Out of Memory
- Reduce concurrent VUs
- Increase system resources
- Enable streaming mode in k6

## References

- [k6 Documentation](https://k6.io/docs/)
- [NovaCron Architecture](../../docs/ARCHITECTURE.md)
- [Performance Tuning Guide](../../docs/PERFORMANCE-TUNING.md)

## Support

For issues or questions:
- Check logs in `tests/load/results/`
- Review [troubleshooting guide](../../docs/TROUBLESHOOTING.md)
- Open an issue with test results attached
