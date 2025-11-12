# DWCP Multi-Datacenter WAN Testing Framework

A comprehensive testing framework for validating Distributed WAN Communication Protocol (DWCP) performance across realistic multi-datacenter network conditions.

## Features

- **Realistic Network Simulation**: Simulate WAN conditions including latency, jitter, packet loss, and bandwidth constraints
- **Multi-Datacenter Topologies**: Test across geographically distributed datacenters with real-world network characteristics
- **Traffic Control Integration**: Use Linux `tc` (Traffic Control) for kernel-level network simulation
- **Workload Generation**: Generate realistic VM memory patterns for migration testing
- **Continuous Testing Pipeline**: Automated testing with scheduling and trend analysis
- **Chaos Engineering**: Inject faults to test system resilience
- **Comprehensive Reporting**: HTML, JSON, CSV reports with Grafana/Prometheus integration
- **Performance Benchmarking**: Benchmark suite for performance regression testing

## Architecture

```
testing/
├── network_simulator.go       # Network simulation engine
├── tc_controller.go           # Linux traffic control integration
├── scenarios.go               # Predefined test scenarios
├── workload_generator.go      # Realistic workload generation
├── test_harness.go            # Test execution engine
├── continuous_testing.go      # Continuous testing pipeline
├── chaos_engineering.go       # Chaos experiments
├── reporter.go                # Test reporting and alerting
├── benchmarks_test.go         # Performance benchmarks
├── integration_test.go        # Integration tests
└── scenarios/                 # Scenario-specific tests
    ├── cross_region_test.go
    ├── high_latency_test.go
    └── packet_loss_test.go
```

## Quick Start

### Running Individual Scenarios

```go
package main

import (
    "fmt"
    dwcptest "github.com/yourusername/novacron/backend/core/network/dwcp/testing"
)

func main() {
    // Create test harness
    harness := dwcptest.NewTestHarness()

    // Run cross-region scenario
    scenario := dwcptest.NewCrossRegionScenario()
    result, err := harness.RunScenario(scenario)

    if err != nil {
        panic(err)
    }

    if result.Passed {
        fmt.Println("✓ Test PASSED")
    } else {
        fmt.Println("✗ Test FAILED")
        for _, reason := range result.FailureReasons {
            fmt.Printf("  - %s\n", reason)
        }
    }
}
```

### Running All Scenarios

```go
scenarios := dwcptest.GetAllScenarios()

for _, scenario := range scenarios {
    result, _ := harness.RunScenario(scenario)
    fmt.Printf("%s: %v\n", scenario.Name, result.Passed)
}
```

### Continuous Testing

```go
// Create continuous testing pipeline
schedule := &dwcptest.TestSchedule{
    Interval:      6 * time.Hour,  // Run every 6 hours
    MaxConcurrent: 3,                // Run 3 tests in parallel
    Enabled:       true,
}

ct := dwcptest.NewContinuousTesting(scenarios, schedule)
ct.Start()

// Get results
latestRun := ct.GetLatestRun()
fmt.Printf("Success rate: %.2f%%\n", latestRun.Summary.SuccessRate*100)
```

### Chaos Engineering

```go
// Create chaos engine
simulator := dwcptest.NewNetworkSimulator(topology)
tcController := dwcptest.NewTrafficController("eth0")
chaosEngine := dwcptest.NewChaosEngine(simulator, tcController)

// Add experiments
experiments := dwcptest.PredefinedExperiments()
for _, exp := range experiments {
    chaosEngine.AddExperiment(exp)
}

// Run experiment
err := chaosEngine.RunExperiment("High Latency")
```

## Predefined Scenarios

### 1. Cross-Region Migration
Tests VM migration across continents with realistic WAN conditions.

- **Datacenters**: US-East, EU-West, AP-South
- **Latency**: 80-200ms
- **Bandwidth**: 5-10 Gbps
- **Expected**: 99% success rate, <5min migration time

### 2. High Latency Migration
Tests performance under extreme latency (300ms+).

- **Latency**: 300ms base, 50ms jitter
- **Bandwidth**: 1 Gbps
- **Expected**: >500 Mbps throughput, 15x+ compression

### 3. Packet Loss Resilience
Tests system resilience under packet loss.

- **Packet Loss**: 5% with bursts
- **Expected**: 95% success rate

### 4. Bandwidth Constrained Migration
Tests with limited bandwidth (100 Mbps).

- **Bandwidth**: 100 Mbps
- **Expected**: 95% bandwidth utilization, 20x+ compression

### 5. Disaster Recovery Replication
Tests continuous replication for DR.

- **Duration**: 24 hours
- **RPO**: <100ms
- **Expected**: 99.99% uptime

## Network Profiles

Predefined profiles for common scenarios:

```go
profiles := dwcptest.PredefinedProfiles()

// Available profiles:
// - "perfect": No impairments
// - "lan": 1ms latency, 1 Gbps
// - "wan-low-latency": 20ms, 1 Gbps
// - "wan-high-latency": 100ms, 100 Mbps
// - "transcontinental": 150ms, 100 Mbps
// - "satellite": 600ms, 10 Mbps
// - "degraded": High latency + loss + jitter

tcController.ApplyProfile(profiles["transcontinental"])
```

## Custom Scenarios

Create custom test scenarios:

```go
customScenario := &dwcptest.TestScenario{
    Name:        "Custom Test",
    Description: "My custom network conditions",
    Topology: &dwcptest.NetworkTopology{
        Datacenters: map[string]*dwcptest.Datacenter{
            "dc1": {
                ID:       "dc1",
                Region:   "us-west",
                Location: dwcptest.GeoLocation{37.7749, -122.4194},
            },
            "dc2": {
                ID:       "dc2",
                Region:   "eu-central",
                Location: dwcptest.GeoLocation{50.1109, 8.6821},
            },
        },
        Links: map[string]*dwcptest.Link{
            "dc1-dc2": {
                Latency: dwcptest.LatencyProfile{
                    BaseLatency:  120 * time.Millisecond,
                    Jitter:       15 * time.Millisecond,
                    Distribution: dwcptest.DistributionNormal,
                },
                Bandwidth: dwcptest.BandwidthProfile{
                    Capacity:    500,  // 500 Mbps
                    Utilization: 0.4,
                },
                PacketLoss: dwcptest.LossProfile{
                    Rate:        0.01,  // 1%
                    BurstLength: 5,
                },
            },
        },
    },
    Workload: &dwcptest.Workload{
        Type:        dwcptest.WorkloadMigration,
        VMs:         10,
        VMSize:      8 * 1024 * 1024 * 1024,  // 8 GB
        Operations:  10,
        Concurrency: 2,
        Pattern:     dwcptest.PatternRealWorld,
    },
    Duration: 15 * time.Minute,
    Assertions: []dwcptest.Assertion{
        {
            Type:      dwcptest.AssertionThroughput,
            Threshold: 300,  // 300 Mbps minimum
            Critical:  true,
        },
        {
            Type:      dwcptest.AssertionSuccessRate,
            Threshold: 0.95,
            Critical:  true,
        },
    },
}
```

## Reporting

### Console Report

```go
reporter := dwcptest.NewTestReporter()
report := reporter.GenerateReport(results)
reporter.PrintReport(report)
```

### HTML Report

```go
html := reporter.GenerateHTMLReport(report)
ioutil.WriteFile("report.html", []byte(html), 0644)
```

### JSON Export

```go
jsonData, _ := reporter.ExportToJSON(report)
ioutil.WriteFile("report.json", jsonData, 0644)
```

### CSV Export

```go
csv := reporter.ExportToCSV(report)
ioutil.WriteFile("report.csv", []byte(csv), 0644)
```

### Grafana Integration

```go
// Automatically publishes to Grafana
reporter.PublishToDashboard(report)
```

## Benchmarking

Run performance benchmarks:

```bash
# Run all benchmarks
go test -bench=. -benchmem ./testing/

# Run specific benchmark
go test -bench=BenchmarkCrossRegionMigration -benchmem

# Run with different latencies
go test -bench=BenchmarkVariousLatencies -benchmem
```

## Requirements

### System Requirements

- Linux kernel 2.6+ (for traffic control)
- Root/sudo access (for tc commands)
- Network interfaces available for testing

### Go Dependencies

```bash
go get github.com/yourusername/novacron/backend/core/network/dwcp/testing
```

## Traffic Control Setup

The framework uses Linux Traffic Control (`tc`) for realistic network simulation. This requires root privileges:

```bash
# Grant capabilities (alternative to running as root)
sudo setcap cap_net_admin+ep /path/to/test/binary

# Or run tests with sudo
sudo go test ./testing/
```

## CI/CD Integration

### GitHub Actions

```yaml
name: DWCP Testing

on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours
  push:
    branches: [ main ]

jobs:
  dwcp-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup Go
        uses: actions/setup-go@v2
        with:
          go-version: 1.21

      - name: Run DWCP Tests
        run: |
          cd backend/core/network/dwcp/testing
          sudo go test -v ./...

      - name: Upload Results
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: backend/core/network/dwcp/testing/results/
```

## Monitoring Integration

### Prometheus Metrics

The framework exposes Prometheus metrics:

```
dwcp_test_total{scenario="cross-region"} 100
dwcp_test_passed{scenario="cross-region"} 98
dwcp_test_failed{scenario="cross-region"} 2
dwcp_test_duration_seconds{scenario="cross-region"} 45.2
dwcp_test_throughput_mbps{scenario="cross-region"} 850
```

### Grafana Dashboard

Import the provided Grafana dashboard:

```bash
curl -X POST http://localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @grafana-dashboard.json
```

## Troubleshooting

### Permission Denied (tc commands)

```bash
# Option 1: Run with sudo
sudo go test ./testing/

# Option 2: Grant capabilities
sudo setcap cap_net_admin+ep $(which go)

# Option 3: Use network namespaces (no root required)
# See: https://man7.org/linux/man-pages/man8/ip-netns.8.html
```

### Tests Timing Out

```go
// Increase scenario duration
scenario.Duration = 30 * time.Minute
```

### High Packet Loss Not Working

```bash
# Check tc is installed
which tc

# Verify kernel modules
lsmod | grep sch_netem
```

## Best Practices

1. **Start Small**: Begin with short-duration tests, then scale up
2. **Isolate Tests**: Run network-intensive tests on dedicated hardware
3. **Monitor Resources**: Watch CPU, memory, and network utilization
4. **Use Realistic Workloads**: `PatternRealWorld` mimics actual VM memory
5. **Validate Assertions**: Ensure thresholds match production requirements
6. **Regular Testing**: Run continuous tests to catch regressions early
7. **Chaos Testing**: Regularly inject faults to verify resilience

## Performance Tips

- Use `PatternConstant` for fastest workload generation
- Reduce `VMSize` for quicker tests during development
- Increase `MaxConcurrent` on multi-core systems
- Use benchmarks to track performance over time

## Contributing

When adding new scenarios:

1. Create scenario in `scenarios.go`
2. Add specific tests in `scenarios/`
3. Document expected behavior
4. Add to `GetAllScenarios()`
5. Update this README

## License

Part of the NovaCron project.

## Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/novacron/issues
- Documentation: https://docs.novacron.io/dwcp-testing
