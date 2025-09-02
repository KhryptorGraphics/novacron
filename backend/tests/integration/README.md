# NovaCron Integration Test Suite

This comprehensive integration test suite validates the end-to-end functionality of the NovaCron platform, including authentication, VM lifecycle management, API endpoints, WebSocket communication, and multi-cloud federation scenarios.

## Quick Start

### Run All Integration Tests
```bash
# From project root
make test-integration
```

### Run Specific Test Suites
```bash
make test-integration-auth       # Authentication flow tests
make test-integration-vm         # VM lifecycle tests
make test-integration-api        # API endpoint tests
make test-integration-websocket  # WebSocket communication tests
make test-integration-federation # Multi-cloud federation tests
```

### Quick Tests (No Coverage)
```bash
make test-integration-quick
```

## Test Architecture

### Test Structure
```
tests/integration/
├── integration_suite_test.go   # Base test suite with shared setup
├── auth/                       # Authentication integration tests
│   └── auth_integration_test.go
├── vm/                         # VM lifecycle tests
│   └── vm_lifecycle_test.go
├── api/                        # API endpoint tests
│   └── api_endpoints_test.go
├── websocket/                  # WebSocket communication tests
│   └── websocket_test.go
├── federation/                 # Multi-cloud federation tests
│   └── federation_test.go
├── fixtures/                   # Test data fixtures
│   └── fixtures.go
└── helpers/                    # Test utility functions
    └── helpers.go
```

### Test Environment

The integration tests run in a containerized environment with the following services:

- **PostgreSQL 15** - Primary database with test schema
- **Redis 7** - Caching and session management  
- **LocalStack** - AWS service mocking (S3, EC2, IAM)
- **MinIO** - S3-compatible storage for federation testing
- **Prometheus** - Metrics collection for monitoring tests
- **Jaeger** - Distributed tracing for debugging

## Test Suites

### 1. Authentication Integration Tests (`auth/`)

Tests the complete authentication flow including:
- User registration with validation
- User login and JWT token generation
- JWT token validation and parsing
- Protected endpoint access control
- Token expiration and refresh
- Role-based access control (RBAC)

**Key Test Cases:**
- `TestUserRegistration` - User account creation
- `TestUserLogin` - Authentication and token generation
- `TestJWTValidation` - Token parsing and validation
- `TestProtectedEndpoints` - Access control enforcement
- `TestTokenExpiration` - Token lifecycle management

### 2. VM Lifecycle Tests (`vm/`)

Tests virtual machine lifecycle operations:
- VM creation with resource allocation
- VM state transitions (start, stop, restart)
- VM configuration updates
- VM deletion and cleanup
- Resource monitoring and metrics

**Key Test Cases:**
- `TestVMCreation` - VM provisioning
- `TestVMStateTransitions` - Start/stop/restart operations
- `TestVMDeletion` - Resource cleanup
- `TestVMMetrics` - Performance monitoring

### 3. API Endpoint Tests (`api/`)

Tests REST API endpoints with database connectivity:
- User management CRUD operations
- VM management endpoints
- System monitoring and health checks
- Error handling and validation
- Request/response serialization

**Key Test Cases:**
- `TestUserCRUD` - User management operations
- `TestVMManagement` - VM API endpoints
- `TestSystemHealth` - Monitoring endpoints
- `TestErrorHandling` - Error response validation

### 4. WebSocket Communication Tests (`websocket/`)

Tests real-time communication features:
- WebSocket connection establishment
- Event broadcasting and subscription
- Real-time VM monitoring
- System event notifications
- Connection handling and recovery

**Key Test Cases:**
- `TestWebSocketConnection` - Connection establishment
- `TestEventBroadcasting` - Message distribution
- `TestVMMonitoring` - Real-time metrics streaming
- `TestConnectionRecovery` - Failover scenarios

### 5. Multi-Cloud Federation Tests (`federation/`)

Tests multi-cloud integration scenarios:
- Cross-cloud storage synchronization
- VM migration between providers
- Network connectivity and routing
- Event propagation across federation
- Data consistency and failover

**Key Test Cases:**
- `TestMultiCloudStorage` - Storage federation
- `TestVMFederation` - Cross-cloud VM management
- `TestCrossCloudNetworking` - Network connectivity
- `TestFederationFailover` - Disaster recovery

## Coverage Reporting

### Generate Coverage Report
```bash
make test-coverage-report
```

This generates comprehensive coverage analysis including:
- HTML interactive coverage report
- Function-level coverage analysis
- Package-level coverage breakdown
- Uncovered code identification
- Improvement recommendations

### Coverage Files Generated
- `coverage.html` - Interactive HTML report
- `coverage-functions.txt` - Per-function coverage
- `coverage-analysis.md` - Detailed analysis
- `coverage.json` - Machine-readable data
- `coverage.xml` - XML format for CI/CD
- `junit-report.xml` - JUnit test results

### View Coverage Summary
```bash
make test-coverage-summary
```

### Validate Coverage Thresholds
```bash
make test-coverage-validate
```

Current thresholds:
- Total Coverage: ≥80%
- Function Coverage: ≥80%
- Line Coverage: ≥80%
- Branch Coverage: ≥75%

## Test Data Management

### Fixtures

The test suite uses comprehensive fixtures for consistent test data:

**Test Users:**
- `admin_test` - Administrative user (admin role)
- `user_test` - Standard user (user role)
- `manager_test` - Management user (manager role)
- `readonly_test` - Read-only user (readonly role)

**Test VMs:**
- `test-vm-001` - Web server (2 CPU, 2GB RAM, running)
- `test-vm-002` - Database server (4 CPU, 8GB RAM, running)
- `test-vm-003` - Application server (1 CPU, 1GB RAM, stopped)
- `test-vm-004` - Load balancer (2 CPU, 4GB RAM, running)

### Data Cleanup

Automatic cleanup ensures test isolation:
```bash
# Clean up test data manually
make test-integration-cleanup
```

## Performance Testing

### Run Integration Benchmarks
```bash
make test-integration-benchmarks
```

### Benchmark Areas
- Authentication performance
- VM lifecycle operation speed
- API endpoint response times
- WebSocket message throughput
- Database query performance

## Environment Management

### Start Test Environment
```bash
make test-integration-setup
```

### Stop Test Environment  
```bash
make test-integration-teardown
```

### View Environment Status
```bash
make test-integration-status
```

### View Logs
```bash
make test-integration-logs
```

### Reset Environment
```bash
make test-integration-reset
```

## Docker Configuration

### Test Services (docker-compose.test.yml)

**Core Services:**
- `postgres-test` - PostgreSQL 15 database
- `redis-test` - Redis 7 cache
- `test-runner` - Go test execution environment

**Mock Services:**
- `mock-aws` - LocalStack for AWS service mocking
- `minio-test` - MinIO for S3-compatible storage

**Monitoring Services:**
- `prometheus-test` - Metrics collection
- `jaeger-test` - Distributed tracing

**Utility Services:**
- `migrate` - Database migration runner
- `test-orchestrator` - Test coordination and reporting

## Troubleshooting

### Common Issues

**1. Database Connection Errors**
```bash
# Check database status
make test-integration-status
# Reset environment
make test-integration-reset
```

**2. Service Startup Timeouts**
```bash
# Increase wait times in Makefile
# Check Docker resources (memory/CPU)
# View service logs
make test-integration-logs
```

**3. Test Data Conflicts**
```bash
# Clean up test data
make test-integration-cleanup
# Reset environment
make test-integration-reset
```

**4. Coverage Report Generation Issues**
```bash
# Check test execution logs
# Ensure all services are healthy
make test-integration-status
```

### Debug Mode

Enable debug logging by setting environment variables:
```bash
# In docker-compose.test.yml
LOG_LEVEL=debug
```

### Manual Test Execution

For debugging specific tests:
```bash
# Start environment
make test-integration-setup

# Connect to test runner
cd backend
docker-compose -f docker-compose.test.yml exec test-runner /bin/bash

# Run specific tests
go test -v ./tests/integration/auth/... -run TestUserLogin
```

## CI/CD Integration

### Integration with CI Systems

The test suite generates CI-friendly reports:
- JUnit XML reports for test results
- Cobertura XML for coverage integration
- JSON metrics for dashboard integration
- Badge data for documentation

### Required Environment Variables

For full federation testing:
```bash
# AWS credentials (for LocalStack)
AWS_ACCESS_KEY_ID=test
AWS_SECRET_ACCESS_KEY=test

# MinIO credentials  
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123

# Database configuration
TEST_DB_URL=postgres://postgres:password@postgres-test:5432/novacron_test?sslmode=disable
```

### Sample CI Configuration

```yaml
# GitHub Actions example
- name: Run Integration Tests
  run: |
    make test-integration-setup
    make test-integration-coverage
    make test-coverage-validate

- name: Upload Coverage Reports
  uses: actions/upload-artifact@v3
  with:
    name: coverage-reports
    path: backend/coverage-integration/
```

## Development Guidelines

### Adding New Tests

1. **Choose the appropriate test suite** based on functionality
2. **Follow naming conventions**: `Test{Feature}{Operation}`
3. **Use fixtures for test data** to ensure consistency
4. **Include cleanup** in test teardown
5. **Add assertions** for both success and failure cases

### Test Best Practices

- **Isolation**: Each test should be independent
- **Clarity**: Test names should describe the scenario
- **Coverage**: Test both happy path and error conditions
- **Performance**: Include performance assertions where relevant
- **Documentation**: Comment complex test scenarios

### Example Test Structure

```go
func (suite *AuthIntegrationTestSuite) TestUserRegistration() {
    // Arrange
    userData := fixtures.NewUserData("test_user")
    
    // Act
    user, err := suite.authManager.CreateUser(userData)
    
    // Assert
    suite.NoError(err)
    suite.NotNil(user)
    suite.Equal(userData.Username, user.Username)
    
    // Cleanup (if needed)
    defer suite.fixtures.CleanupUser(user.ID)
}
```

## Metrics and Monitoring

### Test Execution Metrics

The integration tests collect comprehensive metrics:
- Test execution time per suite
- Database query performance
- Memory usage during tests
- Network request latencies
- Error rates by component

### Dashboard Integration

Generate test dashboard:
```bash
make test-dashboard
```

View current test status, coverage percentages, and available commands.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review test logs: `make test-integration-logs`
3. Validate environment: `make test-integration-status`
4. Reset environment: `make test-integration-reset`

## Contributing

When contributing to the integration test suite:
1. Follow the established patterns and conventions
2. Add tests for new features and bug fixes
3. Maintain or improve coverage percentages
4. Update documentation for new test scenarios
5. Verify all tests pass: `make test-integration`

---

*Integration Test Suite for NovaCron Platform*