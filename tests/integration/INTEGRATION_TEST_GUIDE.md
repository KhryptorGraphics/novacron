# NovaCron Integration Testing Guide

Comprehensive guide for running and maintaining the NovaCron integration test suite.

## Overview

The NovaCron integration testing suite provides comprehensive validation of the entire system including:

- **System Integration**: Backend core components, frontend-backend communication, distributed VM management
- **MLE-Star Workflow**: Complete 7-stage ML engineering workflow with multi-framework support
- **Performance Testing**: Load testing, stress testing, auto-scaling validation, and performance benchmarks

## Quick Start

### Prerequisites

```bash
# Node.js 18+ and npm 9+
node --version  # >= 18.0.0
npm --version   # >= 9.0.0

# Docker and Docker Compose (for containerized testing)
docker --version
docker-compose --version

# PostgreSQL and Redis (for local testing)
psql --version
redis-cli --version
```

### Installation

```bash
# Install dependencies
npm install

# Setup test environment
cp .env.test.example .env.test
```

### Running Tests

```bash
# Run all integration tests
npm run test:integration

# Run specific categories
npm run test:integration:system
npm run test:integration:mle-star
npm run test:integration:performance

# Run with Docker
npm run test:integration:docker

# Run with coverage
npm run test:integration:coverage

# Debug mode
npm run test:integration:debug
```

## Test Structure

```
tests/integration/
├── system/                      # System Integration Tests
│   ├── backend/                 # Backend core component tests
│   │   └── core-components.integration.test.js
│   ├── frontend/                # Frontend-backend communication tests
│   │   └── frontend-backend.integration.test.js
│   ├── distributed/             # Distributed VM management tests
│   └── consensus/               # Consensus protocols and migration tests
├── mle-star/                    # MLE-Star Workflow Tests
│   ├── workflows/               # 7-stage workflow tests
│   │   └── mle-star-workflow.integration.test.js
│   ├── frameworks/              # Multi-framework support tests
│   ├── templates/               # Template generation tests
│   └── samples/                 # Sample ML projects for testing
├── performance/                 # Performance Tests
│   ├── load/                    # Load testing
│   │   └── load-testing.integration.test.js
│   ├── stress/                  # Stress testing
│   └── benchmarks/              # Performance benchmarks
├── fixtures/                    # Test fixtures and data
├── runners/                     # Test runners and orchestration
├── utils/                       # Testing utilities and helpers
└── INTEGRATION_TEST_GUIDE.md    # This guide
```

## Test Categories

### 1. System Integration Tests

Tests the complete backend system and frontend-backend communication:

#### Backend Core Components
- **Autoscaling Service**: CPU/memory threshold triggers, scale-up/down validation
- **Consensus Protocol**: Leader election, data consistency across cluster nodes
- **Federation Management**: Cross-cluster VM creation and replication
- **VM Lifecycle**: Complete create-start-stop-delete cycles with state validation
- **Storage Management**: Multi-tier storage, tier migration, performance validation
- **Network Isolation**: Segment isolation, cross-segment policies, connectivity testing
- **Authentication & Security**: RBAC validation, rate limiting, token management
- **System Health**: Health checks, degradation handling, recovery validation

#### Frontend-Backend Communication
- **REST API Integration**: Real data loading, form submissions, error handling
- **WebSocket Real-time Updates**: State synchronization, connection resilience
- **Authentication Flows**: Login/logout, token expiration, session management
- **State Synchronization**: Multi-tab consistency, real-time updates
- **Performance Validation**: Dashboard load times, large dataset handling

**Example Test Run:**
```bash
npm run test:integration:system

# Expected output:
🚀 Starting Backend Core Components Integration Tests...
✅ Test environment initialized successfully
🧪 Running autoscaling service tests...
✅ Auto-scaling triggered: 2 new VMs created
🧪 Running consensus protocol tests...
✅ New leader elected: node-2
📊 Test Metrics: {
  "autoscaling-trigger-time": 15420,
  "leader-election-time": 8750,
  "vm-creation-success-rate": 0.98
}
```

### 2. MLE-Star Workflow Tests

Validates the complete ML Engineering Star methodology:

#### 7-Stage Workflow Testing
1. **Specification**: Project configuration, requirements validation
2. **Design**: Architecture definition, framework selection
3. **Implementation**: Code generation, template customization
4. **Evaluation**: Model training, metrics validation
5. **Deployment**: Multi-environment deployment, endpoint testing
6. **Monitoring**: Metrics collection, alerting setup
7. **Maintenance**: Retraining schedules, drift detection

#### Multi-Framework Support
- **PyTorch**: Neural networks, custom architectures, GPU support
- **TensorFlow**: Keras integration, distributed training
- **Scikit-learn**: Classical ML, pipeline construction

#### Template Generation
- **Project Types**: Classification, regression, clustering, time series
- **Customizations**: Architecture modifications, pipeline configurations
- **Notebook Generation**: EDA, training, evaluation notebooks

**Example Test Run:**
```bash
npm run test:integration:mle-star

# Expected output:
🚀 Starting MLE-Star Workflow Integration Tests...
🏁 Starting complete MLE-Star workflow for: integration-test-project-1693847520
📋 Stage 1: Specification
🎨 Stage 2: Design
⚙️ Stage 3: Implementation
📊 Stage 4: Evaluation
🚀 Stage 5: Deployment
📈 Stage 6: Monitoring
🔧 Stage 7: Maintenance
✅ Complete MLE-Star workflow executed successfully
```

### 3. Performance Tests

Comprehensive performance validation and load testing:

#### Load Testing
- **Concurrent VM Operations**: 100+ VMs with 20 concurrent requests
- **API Endpoint Load**: 1000+ requests with 50 concurrent connections
- **Sustained Load**: 20 RPS for 60 seconds with resource monitoring
- **WebSocket Scaling**: 100+ concurrent WebSocket connections

#### Auto-scaling Validation
- **Trigger Testing**: CPU/memory threshold validation
- **Scale-up Response**: New instance creation timing
- **Scale-down Detection**: Load decrease response
- **Resource Monitoring**: CPU, memory, network utilization

#### Performance Benchmarks
- **Response Times**: Average, 95th percentile, 99th percentile
- **Throughput**: Requests per second, concurrent capacity
- **Resource Efficiency**: CPU/memory usage under load
- **Scalability Limits**: Maximum concurrent operations

**Example Test Run:**
```bash
npm run test:integration:performance

# Expected output:
🚀 Starting Load Testing Suite...
📊 Load Test Configuration: {
  "concurrent_users": 50,
  "test_duration": 300000,
  "ramp_up_time": 60000
}
🔄 Creating 100 VMs with 20 concurrent requests...
📊 VM Creation Results:
  - Total time: 45234.50ms
  - Success rate: 98.00%
  - Average response time: 2150.75ms
  - Throughput: 2.21 VMs/second
✅ Auto-scaling triggered: 3 new VMs created
```

## Configuration

### Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `NOVACRON_TEST_MODE` | Test execution mode | `integration` | `integration` |
| `NOVACRON_API_URL` | API server URL | `http://localhost:8090` | `http://localhost:8090` |
| `NOVACRON_UI_URL` | Frontend URL | `http://localhost:8092` | `http://localhost:8092` |
| `DB_URL` | Database connection | `postgresql://postgres:postgres@localhost:5432/novacron_test` | - |
| `REDIS_URL` | Redis connection | `redis://localhost:6379` | - |
| `NOVACRON_USE_DOCKER` | Use Docker environment | `false` | `true` |
| `NOVACRON_TEST_PARALLEL` | Run tests in parallel | `true` | `false` |
| `NOVACRON_TEST_COVERAGE` | Enable coverage | `false` | `true` |
| `NOVACRON_TEST_DEBUG` | Debug logging | `false` | `true` |
| `NOVACRON_TEST_TIMEOUT` | Test timeout (ms) | `300000` | `600000` |
| `LOAD_TEST_USERS` | Concurrent users for load tests | `50` | `100` |
| `LOAD_TEST_DURATION` | Load test duration (ms) | `300000` | `600000` |

### Docker Configuration

```yaml
# docker-compose.test.yml
services:
  postgres-test:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: novacron_test
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5433:5432"
    
  redis-test:
    image: redis:7-alpine
    ports:
      - "6380:6379"
    
  novacron-api-test:
    build: .
    environment:
      - NODE_ENV=test
      - DB_URL=postgresql://postgres:postgres@postgres-test:5432/novacron_test
      - REDIS_URL=redis://redis-test:6379
    ports:
      - "8091:8090"
```

## Test Utilities

### TestEnvironment
Manages complete test environment setup and cleanup:

```javascript
const TestEnvironment = require('./utils/test-environment');

const testEnv = new TestEnvironment({
  useDocker: true,
  logLevel: 'debug'
});

await testEnv.setup();
await testEnv.waitForServices(['api-server', 'database']);
// ... run tests
await testEnv.cleanup();
```

### APIClient
Enhanced HTTP client with retry logic and metrics:

```javascript
const APIClient = require('./utils/api-client');

const apiClient = new APIClient({
  baseURL: 'http://localhost:8090',
  timeout: 30000,
  retries: 3
});

const response = await apiClient.post('/api/v1/vms', vmConfig);
const metrics = apiClient.getMetrics();
```

### Load Test Runner
Executes concurrent operations with performance monitoring:

```javascript
const LoadTestRunner = require('./utils/load-test-runner');

const runner = new LoadTestRunner({
  concurrent_users: 50,
  test_duration: 300000
});

const results = await runner.executeConcurrentRequests(
  requests,
  requestHandler,
  { concurrency: 20 }
);
```

## Reporting

### HTML Report
Interactive test results with detailed metrics and drill-down capability.

**Location**: `tests/integration/reports/html/index.html`

**Features**:
- Test suite overview with pass/fail rates
- Detailed test results with execution times
- Performance metrics visualization
- Error details and stack traces
- Environment information

### JSON Report
Machine-readable test results for CI/CD integration.

**Location**: `tests/integration/reports/json/results.json`

**Structure**:
```json
{
  "summary": {
    "suites": { "total": 15, "successful": 14, "failed": 1 },
    "tests": { "total": 127, "passed": 124, "failed": 3 },
    "duration": { "total": 450000, "average": 30000 }
  },
  "results": [...],
  "metrics": {...},
  "environment": {...}
}
```

### JUnit Report
XML format compatible with CI/CD systems like Jenkins, GitLab CI, GitHub Actions.

**Location**: `tests/integration/reports/junit/results.xml`

## CI/CD Integration

### GitHub Actions

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        test-category: [system, mle-star, performance]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        
    - name: Install Dependencies
      run: npm ci
    
    - name: Setup Test Environment
      run: |
        docker-compose -f tests/integration/fixtures/docker-compose.test.yml up -d
        sleep 30
    
    - name: Run Integration Tests
      run: npm run test:integration:${{ matrix.test-category }}
      env:
        NOVACRON_USE_DOCKER: true
        NOVACRON_TEST_COVERAGE: true
      
    - name: Upload Test Reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: integration-test-reports-${{ matrix.test-category }}
        path: tests/integration/reports/
    
    - name: Upload Coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage/lcov.info
        flags: integration
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any
    
    environment {
        NOVACRON_USE_DOCKER = 'true'
        NOVACRON_TEST_COVERAGE = 'true'
    }
    
    stages {
        stage('Setup') {
            steps {
                script {
                    sh 'npm ci'
                    sh 'docker-compose -f tests/integration/fixtures/docker-compose.test.yml up -d'
                    sleep 30
                }
            }
        }
        
        stage('Integration Tests') {
            parallel {
                stage('System Tests') {
                    steps {
                        sh 'npm run test:integration:system'
                    }
                }
                stage('MLE-Star Tests') {
                    steps {
                        sh 'npm run test:integration:mle-star'
                    }
                }
                stage('Performance Tests') {
                    steps {
                        sh 'npm run test:integration:performance'
                    }
                }
            }
        }
    }
    
    post {
        always {
            publishHTML([
                allowMissing: false,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: 'tests/integration/reports/html',
                reportFiles: 'index.html',
                reportName: 'Integration Test Report'
            ])
            
            publishTestResults([
                testResultsPattern: 'tests/integration/reports/junit/*.xml'
            ])
            
            sh 'docker-compose -f tests/integration/fixtures/docker-compose.test.yml down -v'
        }
    }
}
```

## Troubleshooting

### Common Issues

#### 1. Service Startup Failures

**Symptoms**: Tests fail with connection errors, services not ready

**Solutions**:
```bash
# Check service logs
docker-compose -f tests/integration/fixtures/docker-compose.test.yml logs

# Increase startup timeout
export NOVACRON_TEST_TIMEOUT=600000

# Manual service verification
curl http://localhost:8090/api/v1/health
```

#### 2. Database Connection Issues

**Symptoms**: Database connection timeouts, migration failures

**Solutions**:
```bash
# Check database status
docker-compose -f tests/integration/fixtures/docker-compose.test.yml exec postgres-test pg_isready

# Reset database
docker-compose -f tests/integration/fixtures/docker-compose.test.yml down -v postgres-test
docker-compose -f tests/integration/fixtures/docker-compose.test.yml up -d postgres-test

# Manual connection test
psql postgresql://postgres:postgres@localhost:5433/novacron_test
```

#### 3. Test Timeouts

**Symptoms**: Tests timeout during execution, especially performance tests

**Solutions**:
```bash
# Increase test timeout
export NOVACRON_TEST_TIMEOUT=900000  # 15 minutes

# Run tests sequentially
export NOVACRON_TEST_PARALLEL=false

# Reduce test load
export LOAD_TEST_USERS=10
export LOAD_TEST_DURATION=60000
```

#### 4. Memory Issues

**Symptoms**: Out of memory errors, test failures during load testing

**Solutions**:
```bash
# Increase Node.js memory limit
export NODE_OPTIONS="--max-old-space-size=4096"

# Reduce concurrent operations
export NOVACRON_TEST_WORKERS=2

# Skip memory-intensive tests
npm run test:integration:system
```

#### 5. Port Conflicts

**Symptoms**: Port binding errors, service startup failures

**Solutions**:
```bash
# Check port usage
netstat -tulpn | grep :8090

# Use alternative ports
export NOVACRON_API_URL=http://localhost:8091
export NOVACRON_UI_URL=http://localhost:8093

# Stop conflicting services
docker stop $(docker ps -q)
```

### Debug Mode

Enable comprehensive logging:

```bash
export NOVACRON_TEST_DEBUG=true
export NOVACRON_LOG_LEVEL=debug
npm run test:integration:debug
```

Debug output includes:
- Detailed request/response logging
- Service health check results
- Resource usage monitoring
- Test execution timings
- Error stack traces

### Performance Monitoring

Monitor system resources during tests:

```bash
# CPU and memory usage
top -p $(pgrep -f "novacron\|postgres\|redis")

# Network connections
netstat -an | grep :809[0-2]

# Docker resource usage
docker stats
```

## Best Practices

### 1. Test Design
- **Isolation**: Each test should be independent and not rely on other tests
- **Cleanup**: Always clean up created resources after tests
- **Deterministic**: Tests should produce consistent results across runs
- **Fast Feedback**: Critical tests should run quickly for faster feedback loops

### 2. Resource Management
- **Efficient Setup**: Reuse test environment across related tests
- **Proper Cleanup**: Clean up all created resources to prevent resource leaks
- **Resource Limits**: Set appropriate limits to prevent system overload
- **Monitoring**: Monitor resource usage during tests

### 3. Error Handling
- **Graceful Failures**: Tests should fail gracefully with clear error messages
- **Retry Logic**: Implement retry logic for flaky operations
- **Timeout Handling**: Set appropriate timeouts for different types of operations
- **Error Context**: Provide sufficient context for debugging failures

### 4. Maintenance
- **Regular Updates**: Keep test dependencies and fixtures up to date
- **Performance Monitoring**: Monitor test execution times and optimize slow tests
- **Documentation**: Keep documentation current with test changes
- **Review Process**: Review test changes as part of code review process

## Contributing

### Adding New Tests

1. **Choose Appropriate Category**: Place tests in the correct category (system, mle-star, performance)
2. **Follow Naming Convention**: Use `*.integration.test.js` suffix
3. **Use Test Utilities**: Leverage existing utilities for consistency
4. **Include Documentation**: Document test purpose, setup requirements, and expected outcomes
5. **Add Cleanup**: Ensure proper resource cleanup

### Test Structure Template

```javascript
/**
 * Test Suite Description
 * 
 * Brief description of what this test suite validates
 */

const { describe, it, beforeAll, afterAll, expect } = require('@jest/globals');
const TestEnvironment = require('../utils/test-environment');
const APIClient = require('../utils/api-client');

describe('Integration: Test Suite Name', () => {
  let testEnv;
  let apiClient;

  beforeAll(async () => {
    console.log('🚀 Starting Test Suite...');
    
    testEnv = new TestEnvironment();
    await testEnv.setup();
    
    apiClient = new APIClient();
    
    console.log('✅ Test suite initialized');
  });

  afterAll(async () => {
    console.log('🧹 Cleaning up test suite...');
    await testEnv?.cleanup();
    console.log('✅ Test suite cleanup complete');
  });

  describe('Feature Category', () => {
    it('should validate specific behavior', async () => {
      // Test implementation
      expect(result).toBe(expected);
    });
  });
});
```

### Review Checklist

- [ ] Test names clearly describe what is being tested
- [ ] Tests are properly isolated and independent
- [ ] Setup and cleanup are properly implemented
- [ ] Error cases are tested appropriately
- [ ] Performance considerations are addressed
- [ ] Documentation is updated
- [ ] CI/CD integration is verified

## Support

For issues and questions related to integration testing:

1. **Check Documentation**: Review this guide and existing test examples
2. **Search Issues**: Check existing GitHub issues for similar problems
3. **Enable Debug Mode**: Use debug logging to investigate issues
4. **Create Issue**: Create a detailed issue with reproduction steps and logs
5. **Contact Team**: Reach out to the development team for assistance