# E2E Test Configuration

## Environment Variables

The E2E tests support the following environment variables for configuration:

### API Configuration
- `E2E_API_URL`: API server URL (default: `http://localhost:8080`)
- `E2E_MONITORING_URL`: Monitoring server URL (default: `http://localhost:9090`)
- `E2E_PROMETHEUS_URL`: Prometheus server URL (default: `http://prometheus:9090`)
- `E2E_API_KEY`: API key for authentication (default: empty)

### TLS Configuration
- `E2E_INSECURE_SKIP_TLS`: Skip TLS certificate verification (default: `false`)

## Examples

### Basic usage with localhost
```bash
go test ./tests/e2e
```

### Using custom endpoints
```bash
export E2E_API_URL="https://api.example.com:8443"
export E2E_MONITORING_URL="https://monitoring.example.com:9090"
export E2E_PROMETHEUS_URL="https://prometheus.example.com:9090"
export E2E_API_KEY="your-api-key"
go test ./tests/e2e
```

### Using TLS with custom certificates
```bash
export E2E_API_URL="https://secure-api.example.com:8443"
export E2E_MONITORING_URL="https://secure-monitoring.example.com:9090"
export E2E_INSECURE_SKIP_TLS="false"
go test ./tests/e2e
```

### Testing with insecure TLS (for development)
```bash
export E2E_API_URL="https://dev-api.example.com:8443"
export E2E_INSECURE_SKIP_TLS="true"
go test ./tests/e2e
```

## Test Coverage

The E2E tests cover:

- **SupercomputeScenarios**: Scientific computing, ML training, GPU workloads, auto-scaling
- **CrossClusterOperations**: Federation, migration, resource management, data consistency

Both test suites automatically configure their clients based on the environment variables above.