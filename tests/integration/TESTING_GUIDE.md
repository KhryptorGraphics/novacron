# NovaCron Integration Testing Guide

This document provides comprehensive instructions for running and maintaining the NovaCron integration test suite.

## Overview

The integration test suite provides comprehensive testing for all major NovaCron components:

- **VM Lifecycle Management**: Create, start, stop, pause, resume, delete
- **Authentication & Authorization**: User registration, login, RBAC, multi-tenancy
- **Storage Operations**: Volume management, snapshots, tiering, compression, encryption
- **Raft Consensus**: Leader election, log replication, cluster membership
- **Performance Testing**: API throughput, VM creation benchmarks, concurrent operations
- **End-to-End Workflows**: Complete user scenarios, disaster recovery, auto-scaling

## Quick Start

### Prerequisites

1. **Go 1.21+** (for local testing)
2. **Docker & Docker Compose** (for containerized testing)
3. **PostgreSQL 15+** (for database testing)
4. **Redis 7+** (for caching tests)

### Basic Test Execution

```bash
# Run all integration tests
cd tests/integration
make test

# Run specific test suites
make test-vm-lifecycle
make test-auth
make test-storage
make test-consensus
make test-performance
make test-e2e

# Run tests with coverage
make test-coverage
```

### Using Docker Environment

```bash
# Start test environment
make test-integration-setup

# Run tests in Docker
make test-integration-docker

# Clean up
make test-integration-teardown
```

## Test Suites

### 1. VM Lifecycle Tests (`vm_lifecycle_test.go`)

Tests complete VM management operations:

```bash
make test-vm-lifecycle
```

**Coverage:**
- VM creation with various configurations
- State transitions (created → running → stopped → paused)
- VM updates and metadata management
- Concurrent VM operations
- Error handling and validation
- VM metrics collection

**Key Test Cases:**
- `TestVMCreation`: Various VM creation scenarios
- `TestVMLifecycleStates`: State transition testing
- `TestVMConcurrentOperations`: Parallel operations
- `TestVMMetrics`: Performance monitoring

### 2. Authentication Tests (`auth_test.go`)

Tests security and access control:

```bash
make test-auth
```

**Coverage:**
- User registration and validation
- Login/logout workflows
- JWT token validation and expiration
- Role-based access control (RBAC)
- Multi-tenant isolation
- Password security requirements

**Key Test Cases:**
- `TestUserRegistration`: User creation and validation
- `TestUserLogin`: Authentication workflows
- `TestRoleBasedAccess`: Permission testing
- `TestMultiTenantIsolation`: Tenant separation
- `TestPasswordSecurity`: Security requirements

### 3. Storage Tests (`storage_test.go`)

Tests storage operations and management:

```bash
make test-storage
```

**Coverage:**
- Storage tier management (HDD, SSD, NVMe)
- Volume lifecycle operations
- Snapshot creation and restoration
- Storage metrics and performance
- Automated tiering policies
- Compression and encryption
- Quota management

**Key Test Cases:**
- `TestStorageTierManagement`: Tier CRUD operations
- `TestStorageVolumeOperations`: Volume lifecycle
- `TestStorageSnapshots`: Snapshot management
- `TestStorageTieringPolicies`: Automated tiering
- `TestStorageCompression`: Compression features
- `TestStorageEncryption`: Encryption features

### 4. Consensus Tests (`consensus_test.go`)

Tests Raft consensus implementation:

```bash
make test-consensus
```

**Coverage:**
- Cluster status and health monitoring
- Leader election processes
- Log replication and consistency
- Cluster membership management
- Failure recovery scenarios
- Configuration changes
- Snapshot operations

**Key Test Cases:**
- `TestRaftClusterStatus`: Cluster health monitoring
- `TestLeaderElection`: Election processes
- `TestLogReplication`: Data consistency
- `TestClusterMembership`: Node management
- `TestConsensusFailureRecovery`: Fault tolerance
- `TestConsensusSnapshots`: State snapshots

### 5. Performance Tests (`performance_test.go`)

Tests system performance and scalability:

```bash
make test-performance
```

**Coverage:**
- VM creation benchmarks
- API throughput testing
- Concurrent operation performance
- Memory usage patterns
- Database operation performance
- Response time analysis

**Key Test Cases:**
- `TestBenchmarkVMCreation`: VM creation performance
- `TestBenchmarkAPIThroughput`: API performance
- `TestBenchmarkConcurrentOperations`: Concurrent load
- `TestBenchmarkMemoryUsage`: Memory efficiency
- `TestBenchmarkDatabaseOperations`: DB performance

### 6. End-to-End Tests (`e2e_test.go`)

Tests complete user workflows:

```bash
make test-e2e
```

**Coverage:**
- Complete VM lifecycle workflows
- Multi-tenant scenarios
- Disaster recovery procedures
- Auto-scaling workflows
- Real user interaction patterns

**Key Test Cases:**
- `TestCompleteVMLifecycleWorkflow`: Full VM lifecycle
- `TestMultiTenantWorkflow`: Tenant isolation
- `TestDisasterRecoveryWorkflow`: Backup/restore
- `TestAutoScalingWorkflow`: Scaling operations

## Configuration

### Environment Variables

```bash
# Database configuration
export DB_URL="postgresql://postgres:postgres@localhost:5432/novacron_test"

# API configuration
export NOVACRON_API_URL="http://localhost:8090"
export NOVACRON_WS_URL="ws://localhost:8091/ws"
export NOVACRON_UI_URL="http://localhost:8092"

# Cache configuration
export REDIS_URL="redis://localhost:6379"

# Test options
export SKIP_SLOW_TESTS=false
export DEBUG_MODE=false
export CLEANUP_AFTER_TESTS=true
export TEST_TIMEOUT=30m
```

### Test Configuration File

The test suite uses `test_config.go` for centralized configuration:

```go
config := &TestConfig{
    DatabaseURL:      "postgresql://postgres:postgres@localhost:5432/novacron_test",
    APIBaseURL:       "http://localhost:8090",
    RedisURL:         "redis://localhost:6379",
    DefaultTimeout:   30 * time.Second,
    SkipSlowTests:    false,
    DebugMode:        false,
}
```

## Running Tests in CI/CD

The integration tests are designed to run in CI/CD environments:

### GitHub Actions

```yaml
- name: Run Integration Tests
  working-directory: tests/integration
  env:
    DB_URL: postgresql://postgres:postgres@localhost:5432/novacron_test
    REDIS_URL: redis://localhost:6379
    NOVACRON_API_URL: http://localhost:8090
  run: make test
```

### Docker-based CI

```bash
# Build and run test environment
docker-compose -f docker-compose.test.yml up -d
sleep 30  # Wait for services

# Run tests in Docker
docker run --rm \
  -v $(PWD):/app \
  -w /app/tests/integration \
  --network host \
  -e DB_URL="postgresql://postgres:postgres@localhost:5433/novacron_test" \
  golang:1.21 \
  make test
```

## Test Data and Mocking

### Mock Data Generation

The test suite includes comprehensive mock data generators:

```go
mockGen := helpers.NewMockDataGenerator()

// Generate test VMs
vm := mockGen.GenerateVM("tenant-1", 123)

// Generate test users  
user := mockGen.GenerateUser("tenant-1")

// Generate storage tiers
tier := mockGen.GenerateStorageTier()
```

### Database Test Data

Test data is automatically seeded during setup:

- Test users with different roles
- Storage tiers (standard, premium, ultra)
- Resource quotas for tenants
- Sample VM configurations

## Performance Testing

### Benchmarking

Performance tests measure:

- **VM Creation**: Time to create VMs (target: <2s average)
- **API Throughput**: Requests per second (targets vary by endpoint)
- **Concurrent Operations**: Performance under load
- **Memory Usage**: Resource efficiency

### Performance Targets

| Operation | Target | Measurement |
|-----------|--------|-------------|
| VM Creation | < 2s average | P95 < 5s |
| API List VMs | 50+ req/sec | < 500ms average |
| User Profile | 200+ req/sec | < 200ms average |
| Database Ops | < 200ms | 98% success rate |

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   # Check database is running
   docker-compose -f docker-compose.test.yml logs postgres
   
   # Verify connection
   psql postgresql://postgres:postgres@localhost:5432/novacron_test
   ```

2. **API Server Not Ready**
   ```bash
   # Check API health
   curl http://localhost:8090/health
   
   # View API logs
   docker-compose -f docker-compose.test.yml logs api-test
   ```

3. **Test Timeouts**
   ```bash
   # Increase timeout
   export TEST_TIMEOUT=45m
   
   # Run in short mode
   make test-short
   ```

4. **Port Conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep 8090
   
   # Use different ports in docker-compose.test.yml
   ```

### Debug Mode

Enable detailed debugging:

```bash
export DEBUG_MODE=true
make test-vm-lifecycle
```

This provides:
- Detailed HTTP request/response logs
- Database query logging
- WebSocket message tracing
- Performance timing information

### Log Collection

Test logs are available in multiple formats:

```bash
# Generate test report
make report

# View detailed logs
docker-compose -f docker-compose.test.yml logs

# Check specific test output
go test -v -run TestVMLifecycle 2>&1 | tee test.log
```

## Contributing

### Adding New Tests

1. **Create test file**: Follow naming pattern `*_test.go`
2. **Use test suites**: Extend existing suites or create new ones
3. **Follow patterns**: Use established helpers and mock data
4. **Add documentation**: Update this guide with new test cases

### Test Structure

```go
type MyTestSuite struct {
    suite.Suite
    env     *helpers.TestEnvironment
    mockGen *helpers.MockDataGenerator
}

func (suite *MyTestSuite) SetupSuite() {
    suite.env = helpers.NewTestEnvironment(suite.T())
    suite.env.Setup(suite.T())
    suite.mockGen = helpers.NewMockDataGenerator()
}

func (suite *MyTestSuite) TestMyFeature() {
    suite.T().Run("Test Case", func(t *testing.T) {
        // Test implementation
    })
}

func TestMyTestSuite(t *testing.T) {
    suite.Run(t, new(MyTestSuite))
}
```

### Code Quality

- **Test Coverage**: Aim for >80% coverage
- **Assertions**: Use testify for clear assertions
- **Cleanup**: Always clean up resources in teardown
- **Documentation**: Add comments for complex test scenarios
- **Performance**: Consider test execution time

## Makefile Targets

| Target | Description |
|--------|-------------|
| `test` | Run all integration tests |
| `test-short` | Run tests in short mode |
| `test-coverage` | Run with coverage report |
| `test-vm-lifecycle` | VM management tests |
| `test-auth` | Authentication tests |
| `test-storage` | Storage operation tests |
| `test-consensus` | Raft consensus tests |
| `test-performance` | Performance benchmarks |
| `test-e2e` | End-to-end workflows |
| `setup` | Setup test environment |
| `teardown` | Cleanup test environment |
| `clean` | Remove test artifacts |
| `deps` | Install dependencies |
| `fmt` | Format code |
| `lint` | Run linters |
| `debug` | Show debug information |

## Best Practices

1. **Test Independence**: Each test should be independent and idempotent
2. **Resource Cleanup**: Always clean up resources after tests
3. **Realistic Data**: Use realistic test data and scenarios
4. **Error Testing**: Test both success and failure cases
5. **Performance Awareness**: Consider test execution time
6. **CI/CD Ready**: Tests should run reliably in CI environments
7. **Documentation**: Keep test documentation current
8. **Monitoring**: Track test performance and reliability over time

For more information, see the individual test files and helper documentation.