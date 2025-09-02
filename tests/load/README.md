# NovaCron Load Testing Infrastructure

Comprehensive load testing suite for the NovaCron platform, designed to validate performance, scalability, and reliability under various load conditions.

## Overview

This load testing infrastructure provides:

- **API Endpoint Testing**: 1000+ concurrent users across all REST endpoints
- **VM Management Testing**: Creation and management of 1000+ VMs under load  
- **WebSocket Stress Testing**: 2000+ concurrent real-time connections
- **Database Performance Testing**: Query optimization and connection pooling validation
- **Multi-Cloud Federation Testing**: Cross-cluster synchronization and migration
- **Automated Monitoring**: Real-time performance tracking during tests
- **Comprehensive Reporting**: HTML, JSON, CSV, and Markdown reports

## Quick Start

### Prerequisites

1. **k6 Installation**:
   ```bash
   curl -s https://get.k6.io | bash
   ```

2. **Node.js Dependencies**:
   ```bash
   npm install
   ```

3. **Docker & Docker Compose** (for monitoring stack):
   ```bash
   # Verify installation
   docker --version && docker-compose --version
   ```

### Basic Usage

```bash
# Run all core load tests
make test

# Run specific test scenarios
make test-api          # API load testing
make test-vm           # VM management testing
make test-ws           # WebSocket stress testing
make test-db           # Database performance testing
make test-federation   # Federation load testing

# Run comprehensive benchmark
make benchmark

# Setup monitoring and run full suite
make full
```

### Advanced Usage

```bash
# Test against different environments
make test ENVIRONMENT=staging CONCURRENT_USERS=2000

# Include stress and soak testing (long duration)
make test-all INCLUDE_STRESS_TESTS=true INCLUDE_SOAK_TESTS=true

# Custom target and duration
make test API_TARGET=https://my-api.com TEST_DURATION=30m

# Development testing (reduced load)
make dev-test
```

## Test Scenarios

### 1. API Load Testing (`api-load-test.js`)
- **Target**: 1000+ concurrent users
- **Duration**: 10 minutes
- **Focus**: REST API performance under sustained load
- **Metrics**: Response times, throughput, error rates
- **Endpoints**: All VM, storage, monitoring, and cluster APIs

### 2. VM Management Testing (`vm-management-test.js`)
- **Target**: 1000+ VM operations
- **Focus**: VM lifecycle operations under load
- **Operations**: Create, start, stop, migrate, snapshot, delete
- **Metrics**: VM operation latency, success rates, resource utilization

### 3. WebSocket Stress Testing (`websocket-stress-test.js`)
- **Target**: 2000+ concurrent connections  
- **Focus**: Real-time communication scalability
- **Connections**: Metrics, alerts, logs, console streams
- **Metrics**: Connection stability, message throughput, latency

### 4. Database Performance Testing (`database-performance-test.js`)
- **Focus**: Database query performance and connection pooling
- **Operations**: Complex queries, high-frequency reads, write batches
- **Metrics**: Query latency, transaction throughput, connection usage

### 5. Federation Load Testing (`federation-load-test.js`)
- **Focus**: Multi-cloud federation performance
- **Operations**: Cross-cluster sync, VM migration, health monitoring
- **Metrics**: Sync latency, migration success rate, cluster connectivity

### 6. Stress Testing (`stress-test.js`)
- **Target**: 5000+ peak concurrent users
- **Focus**: System breaking points and recovery
- **Stages**: Progressive load increase to failure point
- **Metrics**: Failure thresholds, recovery time, graceful degradation

### 7. Soak Testing (`soak-test.js`)
- **Duration**: 2+ hours sustained load
- **Focus**: Long-term stability, memory leaks, performance degradation
- **Metrics**: Resource growth, performance consistency, connection stability

## Configuration

### Environment Configuration (`configs/test-config.js`)

```javascript
// Environment endpoints
environments: {
  local: { baseURL: 'http://localhost:8080' },
  staging: { baseURL: 'https://staging.novacron.com' },
  production: { baseURL: 'https://api.novacron.com' }
}

// Performance thresholds
thresholds: {
  http_req_duration: ['p(95)<500', 'p(99)<1000'],
  http_req_failed: ['rate<0.01'],
  vm_creation_duration: ['p(95)<30000'],
  ws_connecting_duration: ['p(95)<1000']
}
```

### Monitoring Configuration

The monitoring stack includes:

- **Prometheus**: Metrics collection and alerting
- **Grafana**: Real-time dashboards and visualization  
- **InfluxDB**: K6 metrics storage and analysis
- **Jaeger**: Distributed tracing during load tests
- **Loki**: Log aggregation and analysis
- **Alertmanager**: Alert routing and notification

## Running Tests

### 1. Setup Monitoring (Recommended)

```bash
# Start monitoring infrastructure
make setup

# Verify services are running
docker-compose -f docker-compose.monitoring.yml ps
```

### 2. Run Load Tests

```bash
# Full test suite with monitoring
make full

# Individual scenarios
make test-api
make test-vm  
make test-ws
make test-db
make test-federation

# Stress and endurance testing
make stress
make soak
```

### 3. Generate Reports

```bash
# Generate comprehensive reports
make report

# View reports
open reports/load-test-report-*.html
```

### 4. Cleanup

```bash
# Cleanup test data
make cleanup

# Cleanup monitoring stack
make teardown

# Force cleanup everything
make cleanup-force
```

## Automation & CI/CD

### GitHub Actions Integration

```yaml
name: Load Testing
on:
  schedule:
    - cron: '0 2 * * 1' # Weekly on Monday 2 AM
  workflow_dispatch:

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install k6
        run: |
          curl -s https://get.k6.io | bash
          sudo mv k6 /usr/local/bin/
      - name: Run load tests
        run: |
          cd tests/load
          make ci-test
      - name: Generate reports
        run: |
          cd tests/load  
          make report
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: load-test-results
          path: tests/load/reports/
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any
    
    parameters {
        choice(name: 'ENVIRONMENT', choices: ['staging', 'production'], description: 'Target environment')
        string(name: 'CONCURRENT_USERS', defaultValue: '1000', description: 'Concurrent users')
        string(name: 'TEST_DURATION', defaultValue: '10m', description: 'Test duration')
    }
    
    stages {
        stage('Setup') {
            steps {
                dir('tests/load') {
                    sh 'make install'
                    sh 'make setup'
                }
            }
        }
        
        stage('Load Tests') {
            parallel {
                stage('API Tests') {
                    steps {
                        sh 'make test-api ENVIRONMENT=${ENVIRONMENT}'
                    }
                }
                stage('VM Tests') {
                    steps {
                        sh 'make test-vm ENVIRONMENT=${ENVIRONMENT}'
                    }
                }
                stage('WebSocket Tests') {
                    steps {
                        sh 'make test-ws ENVIRONMENT=${ENVIRONMENT}'
                    }
                }
            }
        }
        
        stage('Report') {
            steps {
                sh 'make report'
                publishHTML([
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: 'tests/load/reports',
                    reportFiles: 'load-test-report-*.html',
                    reportName: 'Load Test Report'
                ])
            }
        }
        
        stage('Cleanup') {
            steps {
                sh 'make cleanup'
                sh 'make teardown'
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'tests/load/reports/**/*', fingerprint: true
        }
    }
}
```

## Performance Thresholds

### API Performance
- **Response Time**: P95 < 500ms, P99 < 1000ms
- **Error Rate**: < 1%
- **Throughput**: > 50 requests/second

### VM Operations  
- **Creation Time**: P95 < 30 seconds
- **Operation Time**: P95 < 10 seconds
- **Success Rate**: > 95%

### WebSocket Performance
- **Connection Time**: P95 < 1 second
- **Session Duration**: P95 < 30 seconds
- **Message Latency**: P95 < 100ms

### Database Performance
- **Query Time**: P95 < 100ms, P99 < 500ms
- **Connection Time**: P95 < 1 second
- **Transaction Rate**: > 500 TPS

## Monitoring & Alerting

### Real-Time Dashboards

Access monitoring dashboards at:

- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **InfluxDB**: http://localhost:8086

### Alert Thresholds

- **High CPU Usage**: > 80%
- **High Memory Usage**: > 85%
- **High Response Time**: > 1000ms
- **High Error Rate**: > 5%
- **WebSocket Failures**: > 5/second

## Reports & Analysis

### Generated Reports

1. **HTML Report**: Comprehensive visual report with charts
2. **JSON Report**: Machine-readable detailed results
3. **CSV Report**: Time-series data for analysis
4. **Markdown Summary**: Executive summary for stakeholders

### Key Metrics Tracked

- Request rates and response times
- VM operation latencies and success rates
- WebSocket connection stability
- Database query performance
- System resource utilization
- Error rates and failure patterns
- Scalability limits and bottlenecks

## Troubleshooting

### Common Issues

1. **k6 not found**:
   ```bash
   curl -s https://get.k6.io | bash
   ```

2. **Authentication failures**:
   ```bash
   # Check API accessibility
   curl -v http://localhost:8080/api/cluster/health
   ```

3. **Monitoring services not starting**:
   ```bash
   # Check Docker status
   docker-compose -f docker-compose.monitoring.yml logs
   ```

4. **High resource usage**:
   ```bash
   # Reduce concurrent users
   make test CONCURRENT_USERS=100
   ```

### Performance Tuning

- **Increase file descriptor limits**: `ulimit -n 65536`
- **Tune TCP settings**: Optimize for high connection loads
- **Monitor system resources**: Keep CPU < 80%, Memory < 85%
- **Use SSD storage**: For faster I/O during VM operations

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Testing Architecture                │
├─────────────────────────────────────────────────────────────┤
│  K6 Test Scenarios                                         │
│  ├── API Load Test        (1000+ users)                    │
│  ├── VM Management Test   (1000+ VMs)                      │
│  ├── WebSocket Stress     (2000+ connections)              │
│  ├── Database Performance (High query load)                │
│  ├── Federation Load      (Multi-cluster)                  │
│  ├── Benchmark Suite      (Performance baselines)          │
│  ├── Stress Test          (Breaking point analysis)        │
│  └── Soak Test           (Long-term stability)             │
├─────────────────────────────────────────────────────────────┤
│  Monitoring Stack                                          │
│  ├── Prometheus          (Metrics collection)              │
│  ├── Grafana            (Dashboards)                      │
│  ├── InfluxDB           (Time-series storage)             │
│  ├── Jaeger             (Distributed tracing)             │
│  ├── Loki               (Log aggregation)                 │
│  └── Alertmanager       (Alert routing)                   │
├─────────────────────────────────────────────────────────────┤
│  Target System (NovaCron)                                  │
│  ├── API Server         (REST endpoints)                   │
│  ├── WebSocket Server   (Real-time features)              │
│  ├── VM Manager         (Hypervisor operations)           │
│  ├── Database           (PostgreSQL)                       │
│  ├── Cache Layer        (Redis)                           │
│  └── Federation         (Multi-cluster)                   │
└─────────────────────────────────────────────────────────────┘
```

## Contributing

When adding new test scenarios:

1. Create test file in `scenarios/`
2. Add configuration to `configs/test-config.js`
3. Update `package.json` scripts
4. Add Makefile target
5. Update documentation

## Support

- **Documentation**: See individual test scenario files for detailed configuration
- **Issues**: Report performance testing issues with full logs and environment details
- **Monitoring**: Use Grafana dashboards for real-time troubleshooting

---

**Load Testing Suite Version**: 1.0.0  
**Last Updated**: 2025-09-02  
**Maintainer**: NovaCron Performance Team