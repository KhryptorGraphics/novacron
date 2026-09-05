## DWCP Testing Framework - Usage Examples

### Example 1: Basic Test Execution

```go
package main

import (
    "fmt"
    "log"

    dwcptest "github.com/yourusername/novacron/backend/core/network/dwcp/testing"
)

func main() {
    // Create test harness
    harness := dwcptest.NewTestHarness()

    // Run cross-region migration test
    scenario := dwcptest.NewCrossRegionScenario()

    fmt.Printf("Running scenario: %s\n", scenario.Name)
    fmt.Printf("Duration: %v\n", scenario.Duration)

    result, err := harness.RunScenario(scenario)
    if err != nil {
        log.Fatalf("Test failed: %v", err)
    }

    // Print results
    if result.Passed {
        fmt.Println("✓ TEST PASSED")
    } else {
        fmt.Println("✗ TEST FAILED")
        for _, reason := range result.FailureReasons {
            fmt.Printf("  Failure: %s\n", reason)
        }
    }

    // Print metrics
    fmt.Printf("\nMetrics:\n")
    fmt.Printf("  Duration: %v\n", result.Duration)
    fmt.Printf("  Operations: %d\n", len(result.Metrics.OperationResults))
    fmt.Printf("  Total Bytes: %d\n", result.Metrics.TotalBytes)

    if result.Metrics.LatencySamples > 0 {
        avgLatency := float64(result.Metrics.TotalLatency.Milliseconds()) /
                     float64(result.Metrics.LatencySamples)
        fmt.Printf("  Avg Latency: %.2fms\n", avgLatency)
    }
}
```

### Example 2: Running Multiple Scenarios

```go
package main

import (
    "fmt"
    "time"

    dwcptest "github.com/yourusername/novacron/backend/core/network/dwcp/testing"
)

func main() {
    harness := dwcptest.NewTestHarness()
    reporter := dwcptest.NewTestReporter()

    // Get all predefined scenarios
    scenarios := dwcptest.GetAllScenarios()

    // Reduce duration for quick testing
    for _, scenario := range scenarios {
        scenario.Duration = 1 * time.Minute
    }

    fmt.Printf("Running %d scenarios...\n", len(scenarios))

    // Run all scenarios
    results := make([]*dwcptest.TestResult, 0)
    for i, scenario := range scenarios {
        fmt.Printf("\n[%d/%d] Running: %s\n", i+1, len(scenarios), scenario.Name)

        result, err := harness.RunScenario(scenario)
        if err != nil {
            fmt.Printf("  ERROR: %v\n", err)
            continue
        }

        results = append(results, result)

        status := "PASSED"
        if !result.Passed {
            status = "FAILED"
        }
        fmt.Printf("  Status: %s (Duration: %v)\n", status, result.Duration)
    }

    // Generate and print report
    report := reporter.GenerateReport(results)
    reporter.PrintReport(report)

    // Export reports
    html := reporter.GenerateHTMLReport(report)
    fmt.Printf("\nHTML Report Length: %d bytes\n", len(html))

    jsonData, _ := reporter.ExportToJSON(report)
    fmt.Printf("JSON Export Length: %d bytes\n", len(jsonData))
}
```

### Example 3: Custom Scenario

```go
package main

import (
    "fmt"
    "time"

    dwcptest "github.com/yourusername/novacron/backend/core/network/dwcp/testing"
)

func main() {
    // Create custom scenario for US-Asia migration
    customScenario := &dwcptest.TestScenario{
        Name:        "US-Asia High-Bandwidth Migration",
        Description: "Test migration between US and Asia with dedicated 10G link",

        Topology: &dwcptest.NetworkTopology{
            Datacenters: map[string]*dwcptest.Datacenter{
                "us-west": {
                    ID:       "us-west-1",
                    Region:   "us-west",
                    Location: dwcptest.GeoLocation{
                        Latitude:  37.7749,
                        Longitude: -122.4194,  // San Francisco
                    },
                },
                "asia-east": {
                    ID:       "asia-east-1",
                    Region:   "asia-east",
                    Location: dwcptest.GeoLocation{
                        Latitude:  35.6762,
                        Longitude: 139.6503,  // Tokyo
                    },
                },
            },

            Links: map[string]*dwcptest.Link{
                "us-asia": {
                    Source:      "us-west-1",
                    Destination: "asia-east-1",

                    Latency: dwcptest.LatencyProfile{
                        BaseLatency:  100 * time.Millisecond,
                        Jitter:       15 * time.Millisecond,
                        Distribution: dwcptest.DistributionNormal,
                    },

                    Bandwidth: dwcptest.BandwidthProfile{
                        Capacity:    10000,  // 10 Gbps
                        Utilization: 0.2,    // 20% baseline
                        Burstable:   true,
                    },

                    PacketLoss: dwcptest.LossProfile{
                        Rate:         0.002,  // 0.2%
                        BurstLength:  3,
                        Distribution: dwcptest.DistributionUniform,
                    },
                },
            },
        },

        Workload: &dwcptest.Workload{
            Type:        dwcptest.WorkloadMigration,
            VMs:         20,
            VMSize:      16 * 1024 * 1024 * 1024,  // 16 GB VMs
            Operations:  20,
            Concurrency: 4,  // 4 concurrent migrations
            Pattern:     dwcptest.PatternRealWorld,
            ThinkTime:   10 * time.Second,
        },

        Duration: 30 * time.Minute,

        Assertions: []dwcptest.Assertion{
            {
                Type:      dwcptest.AssertionThroughput,
                Threshold: 5000,  // 5 Gbps minimum
                Unit:      "mbps",
                Critical:  true,
            },
            {
                Type:      dwcptest.AssertionMigrationTime,
                Threshold: 600,  // 10 minutes max per VM
                Unit:      "seconds",
                Critical:  true,
            },
            {
                Type:      dwcptest.AssertionCompressionRatio,
                Threshold: 12.0,
                Unit:      "ratio",
                Critical:  false,
            },
            {
                Type:      dwcptest.AssertionSuccessRate,
                Threshold: 0.98,
                Unit:      "ratio",
                Critical:  true,
            },
        },

        Tags: []string{"migration", "us-asia", "high-bandwidth", "production"},
    }

    // Run the custom scenario
    harness := dwcptest.NewTestHarness()
    result, err := harness.RunScenario(customScenario)

    if err != nil {
        fmt.Printf("Test failed: %v\n", err)
        return
    }

    fmt.Printf("\nTest Result: %s\n", customScenario.Name)
    fmt.Printf("Status: %v\n", result.Passed)
    fmt.Printf("Duration: %v\n", result.Duration)

    // Print assertion results
    fmt.Println("\nAssertions:")
    for _, assertion := range result.Assertions {
        status := "✓"
        if !assertion.Passed {
            status = "✗"
        }
        fmt.Printf("  %s %s: %.2f (threshold: %.2f)\n",
            status, assertion.Type, assertion.Actual, assertion.Expected)
    }
}
```

### Example 4: Continuous Testing

```go
package main

import (
    "fmt"
    "os"
    "os/signal"
    "syscall"
    "time"

    dwcptest "github.com/yourusername/novacron/backend/core/network/dwcp/testing"
)

func main() {
    // Select scenarios for continuous testing
    scenarios := []*dwcptest.TestScenario{
        dwcptest.NewCrossRegionScenario(),
        dwcptest.NewHighLatencyScenario(),
        dwcptest.NewPacketLossScenario(),
    }

    // Adjust durations for continuous testing
    for _, scenario := range scenarios {
        scenario.Duration = 5 * time.Minute
    }

    // Configure schedule
    schedule := &dwcptest.TestSchedule{
        Interval:      6 * time.Hour,  // Run every 6 hours
        MaxConcurrent: 2,                // 2 tests in parallel
        Enabled:       true,
    }

    // Create continuous testing pipeline
    ct := dwcptest.NewContinuousTesting(scenarios, schedule)

    // Start continuous testing
    if err := ct.Start(); err != nil {
        fmt.Printf("Failed to start continuous testing: %v\n", err)
        return
    }

    fmt.Println("Continuous testing started")
    fmt.Printf("Schedule: Every %v\n", schedule.Interval)
    fmt.Printf("Scenarios: %d\n", len(scenarios))

    // Wait for interrupt signal
    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

    // Periodically print status
    statusTicker := time.NewTicker(1 * time.Minute)
    defer statusTicker.Stop()

    for {
        select {
        case <-sigChan:
            fmt.Println("\nShutting down...")
            ct.Stop()
            return

        case <-statusTicker.C:
            status := ct.GetStatus()
            fmt.Printf("\nStatus: %+v\n", status)

            latestRun := ct.GetLatestRun()
            if latestRun != nil {
                fmt.Printf("Latest run: %s\n", latestRun.ID)
                fmt.Printf("  Success rate: %.2f%%\n",
                    latestRun.Summary.SuccessRate*100)
            }

            // Get trend analysis
            trend := ct.GetTrendAnalysis()
            if trend != nil {
                fmt.Printf("Trend: Success rate %s, Average %.2f%%\n",
                    trend.SuccessTrend, trend.AverageSuccess*100)
            }
        }
    }
}
```

### Example 5: Chaos Engineering

```go
package main

import (
    "fmt"
    "time"

    dwcptest "github.com/yourusername/novacron/backend/core/network/dwcp/testing"
)

func main() {
    // Setup topology
    topology := dwcptest.NewCrossRegionScenario().Topology
    simulator := dwcptest.NewNetworkSimulator(topology)
    simulator.ApplyTopology(topology)

    // Setup traffic controller (requires root)
    tcController := dwcptest.NewTrafficController("eth0")

    // Create chaos engine
    chaosEngine := dwcptest.NewChaosEngine(simulator, tcController)

    // Define custom chaos experiment
    customExperiment := &dwcptest.ChaosExperiment{
        Name:        "Progressive Network Degradation",
        Description: "Gradually increase latency to test adaptation",
        FaultType:   dwcptest.FaultHighLatency,
        Probability: 0.3,  // 30% chance each second
        Duration:    10 * time.Minute,
        ImpactLevel: dwcptest.ImpactHigh,
        Recovery:    dwcptest.RecoveryGradual,
    }

    chaosEngine.AddExperiment(customExperiment)

    // Add predefined experiments
    for _, exp := range dwcptest.PredefinedExperiments() {
        chaosEngine.AddExperiment(exp)
    }

    // Run specific experiment
    fmt.Println("Starting chaos experiment...")

    err := chaosEngine.RunExperiment("Progressive Network Degradation")
    if err != nil {
        fmt.Printf("Experiment failed: %v\n", err)
        return
    }

    // Get metrics
    metrics, running := chaosEngine.GetExperimentStatus("Progressive Network Degradation")
    if metrics != nil {
        fmt.Printf("\nExperiment Metrics:\n")
        fmt.Printf("  Faults Injected: %d\n", metrics.FaultsInjected)
        fmt.Printf("  Recovery Attempts: %d\n", metrics.RecoveryAttempts)
        fmt.Printf("  Recovery Successes: %d\n", metrics.RecoverySuccesses)
        fmt.Printf("  MTTR: %v\n", metrics.MTTR)
        fmt.Printf("  Still running: %v\n", running)
    }

    // Cleanup
    chaosEngine.Stop()
}
```

### Example 6: Workload Generation

```go
package main

import (
    "fmt"

    dwcptest "github.com/yourusername/novacron/backend/core/network/dwcp/testing"
)

func main() {
    // Generate different workload patterns
    patterns := []dwcptest.WorkloadPattern{
        dwcptest.PatternConstant,
        dwcptest.PatternBursty,
        dwcptest.PatternSinusoidal,
        dwcptest.PatternRealWorld,
    }

    size := int64(1 * 1024 * 1024 * 1024)  // 1 GB

    for _, pattern := range patterns {
        fmt.Printf("\nGenerating %s workload...\n", pattern)

        generator := dwcptest.NewWorkloadGenerator(pattern, size)

        // Generate VM memory snapshot
        snapshot := generator.GenerateMemorySnapshot("test-vm", size)

        fmt.Printf("  Size: %d bytes\n", snapshot.Size)
        fmt.Printf("  Zero pages: %d\n", snapshot.Metadata.ZeroPages)
        fmt.Printf("  Estimated compression ratio: %.2fx\n",
            snapshot.Metadata.CompressRatio)
        fmt.Printf("  Entropy: %.2f bits/byte\n", snapshot.Metadata.Entropy)

        // Generate network traffic pattern
        traffic := generator.GenerateNetworkTraffic(
            1*time.Minute,
            1000,  // 1 Gbps target
        )

        fmt.Printf("  Traffic samples: %d\n", len(traffic))
    }
}
```

### Example 7: Network Simulation

```go
package main

import (
    "fmt"
    "time"

    dwcptest "github.com/yourusername/novacron/backend/core/network/dwcp/testing"
)

func main() {
    // Create topology
    topology := &dwcptest.NetworkTopology{
        Datacenters: map[string]*dwcptest.Datacenter{
            "dc1": {
                ID:       "dc1",
                Region:   "us-east",
                Location: dwcptest.GeoLocation{40.7128, -74.0060},
            },
            "dc2": {
                ID:       "dc2",
                Region:   "eu-west",
                Location: dwcptest.GeoLocation{51.5074, -0.1278},
            },
        },
        Links: map[string]*dwcptest.Link{
            "dc1-dc2": {
                Source:      "dc1",
                Destination: "dc2",
                Latency: dwcptest.LatencyProfile{
                    BaseLatency:  80 * time.Millisecond,
                    Jitter:       10 * time.Millisecond,
                    Distribution: dwcptest.DistributionNormal,
                },
                Bandwidth: dwcptest.BandwidthProfile{
                    Capacity:    10000,
                    Utilization: 0.3,
                },
                PacketLoss: dwcptest.LossProfile{
                    Rate:        0.001,
                    BurstLength: 3,
                },
            },
        },
    }

    // Create simulator
    simulator := dwcptest.NewNetworkSimulator(topology)
    simulator.ApplyTopology(topology)

    // Test latency simulation
    fmt.Println("Latency Simulation (100 samples):")
    latencies := make([]time.Duration, 0)

    for i := 0; i < 100; i++ {
        latency := simulator.SimulateLatency("dc1", "dc2")
        latencies = append(latencies, latency)
    }

    // Calculate statistics
    var total time.Duration
    var min, max time.Duration = latencies[0], latencies[0]

    for _, l := range latencies {
        total += l
        if l < min {
            min = l
        }
        if l > max {
            max = l
        }
    }

    avg := total / time.Duration(len(latencies))

    fmt.Printf("  Min: %v\n", min)
    fmt.Printf("  Max: %v\n", max)
    fmt.Printf("  Avg: %v\n", avg)

    // Test packet loss
    fmt.Println("\nPacket Loss Simulation (10000 packets):")
    lossCount := 0

    for i := 0; i < 10000; i++ {
        if simulator.SimulatePacketLoss("dc1", "dc2") {
            lossCount++
        }
    }

    lossRate := float64(lossCount) / 10000.0
    fmt.Printf("  Loss rate: %.2f%%\n", lossRate*100)

    // Test bandwidth
    fmt.Println("\nBandwidth Simulation:")
    for i := 0; i < 10; i++ {
        bandwidth := simulator.GetAvailableBandwidth("dc1", "dc2")
        fmt.Printf("  Sample %d: %d Mbps\n", i+1, bandwidth)
        time.Sleep(100 * time.Millisecond)
    }

    // Get statistics
    stats := simulator.GetTopologyStats()
    fmt.Printf("\nTopology Statistics: %+v\n", stats)
}
```

### Example 8: Traffic Control

```go
package main

import (
    "fmt"
    "time"

    dwcptest "github.com/yourusername/novacron/backend/core/network/dwcp/testing"
)

func main() {
    // Note: Requires root/sudo privileges

    // Create traffic controller for loopback interface
    tc := dwcptest.NewTrafficController("lo")

    // Apply latency
    fmt.Println("Applying 100ms latency...")
    err := tc.ApplyLatency(100*time.Millisecond, 10*time.Millisecond)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }

    // Check status
    status, _ := tc.GetStatus()
    fmt.Printf("Status: %s\n", status)

    time.Sleep(5 * time.Second)

    // Reset
    fmt.Println("Resetting...")
    tc.Reset()

    // Apply complex network profile
    fmt.Println("\nApplying complex network profile...")

    config := &dwcptest.ComplexNetworkConfig{
        Latency:            50 * time.Millisecond,
        Jitter:             10 * time.Millisecond,
        LatencyDistribution: "normal",
        PacketLoss:         1.0,  // 1%
        PacketCorruption:   0.1,  // 0.1%
        RateLimit:          100,  // 100 Mbps
    }

    err = tc.ApplyComplex(config)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }

    // Get statistics
    stats, _ := tc.GetStatistics()
    fmt.Printf("Statistics:\n%s\n", stats)

    time.Sleep(5 * time.Second)

    // Apply predefined profile
    fmt.Println("\nApplying 'transcontinental' profile...")
    profiles := dwcptest.PredefinedProfiles()
    tc.ApplyProfile(profiles["transcontinental"])

    time.Sleep(5 * time.Second)

    // Cleanup
    fmt.Println("Cleaning up...")
    tc.Reset()
}
```

These examples demonstrate the full capabilities of the DWCP testing framework!
