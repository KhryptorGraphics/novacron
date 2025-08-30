# NovaCron Hypervisor Integration Testing

This directory contains comprehensive testing infrastructure for all hypervisor integrations in NovaCron. The testing suite provides extensive coverage of KVM/QEMU, Docker containers, containerd, VMware vSphere, and multi-hypervisor scenarios.

## Test Architecture

### Core Components

1. **HypervisorTestSuite** (`hypervisor_test_suite.go`)
   - Generic test framework for all hypervisor drivers
   - Capability-based testing that adapts to driver features
   - Comprehensive lifecycle testing (create, start, stop, delete)
   - Edge case and error handling validation

2. **MockHypervisor** (`mock_hypervisor.go`)
   - Full-featured mock hypervisor implementation
   - Configurable latency, failure rates, and capabilities
   - Resource usage simulation and performance testing
   - Isolated testing environment without external dependencies

3. **Integration Test Suites**
   - **KVM Integration** (`kvm_integration_test.go`)
   - **Container Integration** (`container_integration_test.go`) 
   - **VMware vSphere Integration** (`vmware_integration_test.go`)
   - **Multi-Hypervisor Testing** (`multi_hypervisor_test.go`)

4. **Comprehensive Test Runner** (`comprehensive_hypervisor_test_runner.go`)
   - Orchestrates all test suites
   - Generates detailed HTML and JSON reports
   - Performance benchmarking and metrics collection
   - Configurable test execution with timeouts

## Test Categories

### 1. KVM/QEMU Tests

**Coverage:**
- QEMU binary detection and validation
- QMP protocol communication testing
- VM lifecycle management (create, start, stop, delete)
- Disk image creation and management
- Resource allocation (CPU, memory, disk)
- Network configuration
- Snapshot operations
- Performance benchmarks
- Error recovery scenarios

**Requirements:**
- QEMU installed (`/usr/bin/qemu-system-x86_64`)
- KVM kernel module loaded (`/dev/kvm`)
- Sufficient disk space for test images

### 2. Container Tests (Docker)

**Coverage:**
- Docker daemon connectivity
- Container lifecycle management
- Resource constraints (CPU, memory)
- Environment variables and configuration
- Volume mounting
- Network configuration
- Pause/resume operations
- Metrics collection
- Concurrent operations
- Error handling

**Requirements:**
- Docker installed and running
- Docker daemon accessible
- Alpine Linux image available

### 3. Containerd Tests

**Coverage:**
- Containerd daemon connectivity
- Namespace management
- Container operations via containerd API
- Resource management
- Image handling
- Network integration
- Runtime configuration

**Requirements:**
- Containerd installed and running
- Proper socket permissions (`/run/containerd/containerd.sock`)

### 4. VMware vSphere Tests

**Coverage:**
- vCenter/ESXi connectivity
- VM creation with proper specifications
- Power operations (on/off/suspend/resume)
- vMotion migration testing
- Snapshot management
- Resource allocation verification
- Network and storage configuration
- Performance metrics collection

**Requirements:**
- vSphere environment access
- Environment variables:
  - `VSPHERE_URL`: vCenter/ESXi URL
  - `VSPHERE_USERNAME`: Username
  - `VSPHERE_PASSWORD`: Password
  - `VSPHERE_DATACENTER`: Datacenter name (optional)
  - `VSPHERE_DATASTORE`: Datastore name (optional)
  - `VSPHERE_NETWORK`: Network name (optional)
  - `VSPHERE_INSECURE`: Skip TLS verification (optional)

### 5. Multi-Hypervisor Tests

**Coverage:**
- Cross-hypervisor compatibility
- Unified interface validation
- Concurrent operations across hypervisors
- Resource isolation between hypervisors
- Performance comparison
- Error handling consistency
- Feature parity analysis

## Running Tests

### Individual Test Suites

```bash
# Run KVM integration tests
go test -v ./backend/core/vm/ -run TestKVMIntegration

# Run Container integration tests
go test -v ./backend/core/vm/ -run TestDockerIntegration
go test -v ./backend/core/vm/ -run TestContainerdIntegration

# Run VMware integration tests (requires vSphere access)
VSPHERE_URL=https://vcenter.example.com \
VSPHERE_USERNAME=admin@vsphere.local \
VSPHERE_PASSWORD=password123 \
go test -v ./backend/core/vm/ -run TestVMwareIntegration

# Run multi-hypervisor tests
go test -v ./backend/core/vm/ -run TestMultiHypervisorIntegration
```

### Comprehensive Test Suite

```bash
# Run all hypervisor integration tests
go test -v ./backend/core/vm/ -run TestComprehensiveHypervisorIntegration

# Run with custom timeout
go test -timeout 45m -v ./backend/core/vm/ -run TestComprehensiveHypervisorIntegration

# Skip slow tests
go test -short ./backend/core/vm/
```

### Benchmark Tests

```bash
# Run performance benchmarks
go test -bench=. ./backend/core/vm/

# Run specific hypervisor benchmarks
go test -bench=BenchmarkKVM ./backend/core/vm/
go test -bench=BenchmarkContainer ./backend/core/vm/
go test -bench=BenchmarkVSphere ./backend/core/vm/
```

## Test Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VSPHERE_URL` | vCenter/ESXi server URL | - |
| `VSPHERE_USERNAME` | vSphere username | - |
| `VSPHERE_PASSWORD` | vSphere password | - |
| `VSPHERE_DATACENTER` | Datacenter name | "Datacenter" |
| `VSPHERE_DATASTORE` | Datastore name | "datastore1" |
| `VSPHERE_NETWORK` | Network name | "VM Network" |
| `VSPHERE_INSECURE` | Skip TLS verification | "false" |
| `VSPHERE_TARGET_HOST` | Target host for migration tests | - |

### Test Runner Configuration

The comprehensive test runner can be configured via `TestRunnerConfig`:

```go
config := TestRunnerConfig{
    IncludeKVM:             true,  // Enable KVM tests
    IncludeContainer:       true,  // Enable Docker tests
    IncludeContainerd:      true,  // Enable containerd tests
    IncludeVMware:          true,  // Enable vSphere tests
    IncludeMultiHypervisor: true,  // Enable multi-hypervisor tests
    GenerateReports:        true,  // Generate test reports
    ReportDirectory:        "test-reports",  // Report output directory
    Timeout:                30 * time.Minute,  // Per-suite timeout
    Verbose:                true,  // Verbose output
    PerformanceBenchmarks:  true,  // Include benchmarks
}
```

## Test Reports

When `GenerateReports` is enabled, the test runner creates:

### HTML Report (`test-results.html`)
- Interactive test results with drill-down capability
- Performance metrics visualization
- Error details and stack traces
- Environment information
- Test duration analysis

### JSON Report (`test-results.json`)
- Machine-readable test results
- Detailed metrics and timing data
- Test configuration and environment
- Suitable for CI/CD integration

### Summary Report (`test-summary.txt`)
- Concise text summary
- Pass/fail counts per test suite
- Overall success rate
- Key performance indicators

## Mock Hypervisor Configuration

The mock hypervisor can be configured for different testing scenarios:

### High-Performance Mock
```go
mock.Configure(
    MockFailureConfig{},  // No failures
    MockLatencyConfig{
        CreateLatency: 50 * time.Millisecond,
        StartLatency:  500 * time.Millisecond,
        StopLatency:   200 * time.Millisecond,
    },
    MockCapabilities{
        MaxVMs:          1000,
        MaxCPUPerVM:     64,
        MaxMemoryPerVM:  128 * 1024,
    },
)
```

### Unreliable Mock (for error testing)
```go
mock.Configure(
    MockFailureConfig{
        CreateFailureRate:   0.1,  // 10% failure rate
        StartFailureRate:    0.05,
        RandomFailures:      true,
    },
    MockLatencyConfig{
        CreateLatency:  200 * time.Millisecond,
        VariabilityPct: 0.5,  // 50% variation
    },
    MockCapabilities{
        MaxVMs:             50,
        SupportsSnapshot:   false,
        SupportsMigrate:    false,
    },
)
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Hypervisor Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Go
      uses: actions/setup-go@v3
      with:
        go-version: 1.21
    
    - name: Install Docker
      run: |
        sudo apt-get update
        sudo apt-get install -y docker.io
        sudo systemctl start docker
    
    - name: Run Integration Tests
      run: |
        go test -timeout 30m -v ./backend/core/vm/ \
          -run TestComprehensiveHypervisorIntegration
    
    - name: Upload Test Reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-reports
        path: test-reports/
```

### Jenkins Pipeline Example

```groovy
pipeline {
    agent any
    
    environment {
        VSPHERE_URL = credentials('vsphere-url')
        VSPHERE_USERNAME = credentials('vsphere-username')
        VSPHERE_PASSWORD = credentials('vsphere-password')
    }
    
    stages {
        stage('Test') {
            steps {
                sh '''
                    go test -timeout 45m -v ./backend/core/vm/ \
                      -run TestComprehensiveHypervisorIntegration
                '''
            }
            post {
                always {
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'test-reports',
                        reportFiles: 'test-results.html',
                        reportName: 'Hypervisor Integration Test Report'
                    ])
                    archiveArtifacts 'test-reports/**/*'
                }
            }
        }
    }
}
```

## Troubleshooting

### Common Issues

1. **KVM Tests Fail**
   - Check if KVM is installed: `ls /dev/kvm`
   - Verify QEMU installation: `which qemu-system-x86_64`
   - Ensure user has KVM access: `sudo adduser $USER kvm`

2. **Docker Tests Fail**
   - Check Docker daemon: `sudo systemctl status docker`
   - Verify Docker permissions: `docker ps`
   - Pull Alpine image: `docker pull alpine:latest`

3. **vSphere Tests Fail**
   - Verify network connectivity: `ping vcenter.example.com`
   - Check credentials and permissions
   - Validate SSL certificates (or use `VSPHERE_INSECURE=true`)

4. **Test Timeouts**
   - Increase timeout: `go test -timeout 60m`
   - Run specific test suites individually
   - Check system resources and performance

### Debugging

Enable verbose logging:
```bash
export NOVACRON_LOG_LEVEL=debug
go test -v ./backend/core/vm/ -run TestSpecificTest
```

Run individual test methods:
```bash
go test -v ./backend/core/vm/ -run TestKVMIntegration/TestKVMDriverCreation
```

## Contributing

When adding new hypervisor drivers or test cases:

1. Implement the `VMDriver` interface
2. Add integration tests following existing patterns
3. Update the comprehensive test runner
4. Add documentation and configuration examples
5. Include error handling and edge case tests

### Test Naming Conventions

- Integration tests: `TestXXXIntegration`
- Benchmark tests: `BenchmarkXXX`
- Unit tests: `TestXXX`
- Helper functions: `testHelperXXX`

### Mock Implementation Guidelines

- Implement realistic latency and behavior
- Support failure injection for robustness testing
- Provide configurable capabilities
- Include resource usage simulation
- Maintain state consistency