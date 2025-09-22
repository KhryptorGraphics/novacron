# NovaCron Integration Test Suite

This directory contains comprehensive integration tests for the NovaCron distributed VM management system.

## Test Structure

```
tests/integration/
├── README.md                 # This file
├── test_config.go           # Test configuration and environment setup
├── helpers/                 # Test utilities and helpers
│   ├── database.go          # Database test utilities
│   ├── api_client.go        # HTTP API test client
│   ├── mock_data.go         # Mock data generators
│   └── environment.go       # Test environment management
├── vm_lifecycle_test.go     # VM creation, start, stop, delete
├── storage_test.go          # Storage tiering and operations
├── consensus_test.go        # Raft consensus leader election
├── auth_test.go            # Authentication and authorization
├── autoscaling_test.go     # Auto-scaling triggers
├── loadbalancer_test.go    # Load balancer functionality  
├── backup_test.go          # Backup and restore operations
├── multitenancy_test.go    # Multi-tenant isolation
├── network_test.go         # Network segmentation
├── quotas_test.go          # Resource quota enforcement
├── monitoring_test.go      # Monitoring and metrics
├── api_test.go             # REST and GraphQL API tests
├── performance_test.go     # Performance benchmarks
├── e2e_test.go            # End-to-end workflow tests
├── network_fixes_integration_test.go  # Network component integration tests
├── end_to_end_network_test.go        # Network E2E integration tests
└── verification_test.go              # Verification comment compliance tests
```

## Running Tests

### Prerequisites

1. **Test Environment**: Start test services
   ```bash
   make test-env-up
   ```

2. **Database Setup**: Create test database
   ```bash
   make db-test-setup
   ```

### Running Individual Test Suites

```bash
# Run all integration tests
make test-integration

# Run specific test suites
go test -v ./tests/integration/vm_lifecycle_test.go
go test -v ./tests/integration/auth_test.go
go test -v ./tests/integration/performance_test.go

# Run verification tests for comment fixes (NEW)
go test -v ./tests/integration/verification_test.go
go test -v ./tests/integration/network_fixes_integration_test.go
go test -v ./tests/integration/end_to_end_network_test.go

# Quick verification of all 13 comment fixes
go test -v -run "TestAllVerificationComments" ./tests/integration/verification_test.go
```

### Running Verification Tests (NEW)

The verification tests validate the implementation of 13 specific verification comments:

```bash
# Run automated test runner for all verification tests
./scripts/run_integration_tests.sh

# Run specific verification comment tests
go test -v -run "TestVerificationComment1" ./tests/integration/verification_test.go
go test -v -run "TestVerificationComment12" ./tests/integration/verification_test.go
```

### Running with Docker

```bash
# Run tests in Docker environment
make test-integration

# Run with coverage
make test-integration-coverage
```

## Test Environment

### Required Services

- **PostgreSQL**: Database (port 5432)
- **Redis**: Cache and session storage (port 6379)
- **API Server**: REST/GraphQL APIs (port 8090)
- **Frontend**: Web UI (port 8092)

### Environment Variables

```bash
DB_URL=postgresql://postgres:postgres@localhost:5432/novacron_test
REDIS_URL=redis://localhost:6379
NOVACRON_API_URL=http://localhost:8090
NOVACRON_UI_URL=http://localhost:8092
AUTH_SECRET=test-secret-key
```

## Test Coverage

### Core Components

- ✅ VM Lifecycle Management
- ✅ Storage Operations
- ✅ Raft Consensus
- ✅ Authentication/Authorization
- ✅ Auto-scaling
- ✅ Load Balancing
- ✅ Backup/Restore
- ✅ Multi-tenancy
- ✅ Network Segmentation
- ✅ Resource Quotas
- ✅ Monitoring/Metrics

### API Coverage

- ✅ REST API endpoints
- ✅ GraphQL queries/mutations
- ✅ WebSocket connections
- ✅ Authentication middleware
- ✅ Error handling
- ✅ Rate limiting

### Performance Tests

- ✅ VM creation/deletion benchmarks
- ✅ Storage I/O performance
- ✅ Consensus algorithm performance
- ✅ API response times
- ✅ Concurrent operations
- ✅ Memory usage profiling

### Verification Tests (NEW)

Validates implementation of 13 specific verification comment fixes:

- ✅ **Comment 1**: Wildcard interface thresholds in bandwidth monitoring
- ✅ **Comment 2**: Alert rate limiting conflation fix
- ✅ **Comment 3**: UDPHolePuncher refactoring for single connection
- ✅ **Comment 4**: NAT traversal connection type labeling
- ✅ **Comment 5**: NATTraversalManager.Stop() proper cleanup
- ✅ **Comment 6**: Relay fallback implementation for NAT traversal
- ✅ **Comment 7**: External endpoint propagation in discovery
- ✅ **Comment 8**: Routing table update race condition fix
- ✅ **Comment 9**: QoS kernel state enforcement implementation
- ✅ **Comment 10**: Configurable root qdisc rate in QoS
- ✅ **Comment 11**: Network constraints validation in scheduler
- ✅ **Comment 12**: RequestPlacement global config mutation fix
- ✅ **Comment 13**: STUN client thread safety improvements

## Contributing

1. **Test Structure**: Follow the existing pattern for new tests
2. **Naming**: Use descriptive test names with `Test` prefix
3. **Setup/Teardown**: Use `setup()` and `teardown()` helpers
4. **Assertions**: Use testify package for assertions
5. **Documentation**: Add comments explaining complex test scenarios

## Troubleshooting

### Common Issues

1. **Database Connection**: Ensure test database is running and accessible
2. **Service Dependencies**: Start test environment before running tests
3. **Port Conflicts**: Check that test ports are not in use
4. **Permissions**: Ensure Docker has necessary permissions for privileged operations

### Debug Mode

```bash
# Run with verbose output
go test -v -args -debug ./tests/integration/...

# Run specific test with debugging
go test -v -run TestVMLifecycle ./tests/integration/vm_lifecycle_test.go -args -debug
```