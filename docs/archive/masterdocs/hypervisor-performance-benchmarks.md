# NovaCron Hypervisor Performance Benchmarking Guide

## Overview
This document provides comprehensive performance benchmarking methodologies and baseline metrics for evaluating hypervisor integration performance within the NovaCron ecosystem.

## Benchmarking Framework

### Test Environment Specifications

#### Hardware Requirements
**Minimum Test Configuration:**
- **CPU**: Intel Xeon Gold 6248R (24 cores) or AMD EPYC 7401P (24 cores)
- **Memory**: 128GB DDR4-2933 ECC
- **Storage**: NVMe SSD (2TB, >5000 IOPS)
- **Network**: 10GbE connectivity
- **Virtualization**: Intel VT-x/VT-d or AMD-V/AMD-Vi enabled

**Recommended Test Configuration:**
- **CPU**: Intel Xeon Platinum 8380 (40 cores) or AMD EPYC 7763 (64 cores)  
- **Memory**: 256GB DDR4-3200 ECC
- **Storage**: High-performance NVMe (4TB, >10000 IOPS)
- **Network**: 25GbE or higher
- **Additional**: GPU for passthrough testing (NVIDIA Tesla/RTX or AMD Instinct)

### Performance Metrics Categories

#### 1. VM Lifecycle Performance
```go
type LifecycleMetrics struct {
    VMCreationTime    time.Duration `json:"vm_creation_time_ms"`
    VMStartupTime     time.Duration `json:"vm_startup_time_ms"`
    VMShutdownTime    time.Duration `json:"vm_shutdown_time_ms"`
    VMDestroyTime     time.Duration `json:"vm_destroy_time_ms"`
    SnapshotTime      time.Duration `json:"snapshot_time_ms"`
    RestoreTime       time.Duration `json:"restore_time_ms"`
}
```

#### 2. Resource Utilization Metrics
```go
type ResourceMetrics struct {
    CPUOverhead       float64 `json:"cpu_overhead_percent"`
    MemoryOverhead    int64   `json:"memory_overhead_mb"`
    StorageOverhead   int64   `json:"storage_overhead_mb"`
    NetworkOverhead   float64 `json:"network_overhead_percent"`
    IOPSPerformance   int64   `json:"iops_native_ratio"`
}
```

#### 3. Migration Performance Metrics
```go
type MigrationMetrics struct {
    MigrationDuration   time.Duration `json:"migration_duration_ms"`
    DowntimeActual      time.Duration `json:"downtime_actual_ms"`
    DataTransferred     int64         `json:"data_transferred_mb"`
    TransferRate        float64       `json:"transfer_rate_mbps"`
    MemoryDirtyRate     float64       `json:"memory_dirty_rate_mbps"`
}
```

### Benchmark Test Suites

#### 1. VM Creation and Startup Benchmarks
```bash
#!/bin/bash
# VM lifecycle performance test

for hypervisor in kvm vsphere hyperv xcp-ng proxmox; do
    echo "Testing $hypervisor VM lifecycle performance"
    
    # Test VM creation time
    for i in {1..10}; do
        start_time=$(date +%s%N)
        novacron vm create --hypervisor $hypervisor --config standard_vm.yaml
        end_time=$(date +%s%N)
        creation_time=$(( ($end_time - $start_time) / 1000000 ))
        echo "$hypervisor,vm_create,$creation_time" >> lifecycle_results.csv
        
        # Cleanup
        novacron vm delete test-vm-$i
    done
done
```

#### 2. CPU Performance Benchmarks
```bash
#!/bin/bash
# CPU performance comparison

# Run CPU-intensive workload in VM vs native
cpu_test_vm() {
    local hypervisor=$1
    novacron vm create --hypervisor $hypervisor --cpu 4 --memory 8192 test-cpu-vm
    novacron vm exec test-cpu-vm "sysbench cpu --cpu-max-prime=20000 run"
    novacron vm delete test-cpu-vm
}

cpu_test_native() {
    sysbench cpu --cpu-max-prime=20000 run
}

for hypervisor in kvm vsphere hyperv xcp-ng proxmox; do
    echo "CPU performance test for $hypervisor"
    cpu_test_vm $hypervisor
done
```

#### 3. Memory Performance Benchmarks
```bash
#!/bin/bash
# Memory performance benchmarks

memory_test() {
    local hypervisor=$1
    local test_size="8G"
    
    novacron vm create --hypervisor $hypervisor --memory 16384 test-memory-vm
    
    # Sequential read/write
    novacron vm exec test-memory-vm "sysbench memory --memory-block-size=1M --memory-total-size=$test_size run"
    
    # Random access
    novacron vm exec test-memory-vm "sysbench memory --memory-oper=read --memory-access-mode=rnd run"
    
    novacron vm delete test-memory-vm
}
```

#### 4. Storage I/O Benchmarks
```bash
#!/bin/bash
# Storage performance benchmarks

storage_test() {
    local hypervisor=$1
    
    novacron vm create --hypervisor $hypervisor --disk 50 test-storage-vm
    
    # Sequential I/O
    novacron vm exec test-storage-vm "fio --name=seq-read --rw=read --bs=1M --size=10G --filename=/tmp/testfile"
    novacron vm exec test-storage-vm "fio --name=seq-write --rw=write --bs=1M --size=10G --filename=/tmp/testfile"
    
    # Random I/O
    novacron vm exec test-storage-vm "fio --name=rand-read --rw=randread --bs=4k --size=10G --filename=/tmp/testfile"
    novacron vm exec test-storage-vm "fio --name=rand-write --rw=randwrite --bs=4k --size=10G --filename=/tmp/testfile"
    
    novacron vm delete test-storage-vm
}
```

#### 5. Network Performance Benchmarks
```bash
#!/bin/bash
# Network performance benchmarks

network_test() {
    local hypervisor=$1
    
    # Create two VMs for network testing
    novacron vm create --hypervisor $hypervisor test-net-vm1
    novacron vm create --hypervisor $hypervisor test-net-vm2
    
    # Get VM IP addresses
    vm1_ip=$(novacron vm info test-net-vm1 --json | jq -r '.ip_address')
    vm2_ip=$(novacron vm info test-net-vm2 --json | jq -r '.ip_address')
    
    # TCP throughput test
    novacron vm exec test-net-vm1 "iperf3 -s -D"
    novacron vm exec test-net-vm2 "iperf3 -c $vm1_ip -t 60 -P 4"
    
    # UDP throughput test
    novacron vm exec test-net-vm2 "iperf3 -c $vm1_ip -u -t 60"
    
    # Latency test
    novacron vm exec test-net-vm2 "ping -c 100 $vm1_ip"
    
    # Cleanup
    novacron vm delete test-net-vm1 test-net-vm2
}
```

#### 6. Migration Performance Benchmarks
```bash
#!/bin/bash
# Live migration performance test

migration_test() {
    local source_hypervisor=$1
    local target_hypervisor=$2
    local vm_memory=$3
    
    # Create VM with specified memory
    novacron vm create --hypervisor $source_hypervisor --memory $vm_memory test-migration-vm
    
    # Start memory-intensive workload
    novacron vm exec test-migration-vm "stress --vm 1 --vm-bytes $(($vm_memory * 1024 * 1024 * 0.8)) --timeout 300s &"
    
    # Perform migration and measure metrics
    start_time=$(date +%s%N)
    migration_log=$(novacron vm migrate test-migration-vm --target-hypervisor $target_hypervisor --verbose)
    end_time=$(date +%s%N)
    
    # Parse migration metrics
    migration_duration=$(( ($end_time - $start_time) / 1000000 ))
    downtime=$(echo "$migration_log" | grep "downtime" | awk '{print $3}')
    transferred=$(echo "$migration_log" | grep "transferred" | awk '{print $3}')
    
    echo "$source_hypervisor,$target_hypervisor,$vm_memory,$migration_duration,$downtime,$transferred" >> migration_results.csv
    
    novacron vm delete test-migration-vm
}

# Test different memory sizes
for memory in 1024 2048 4096 8192 16384; do
    migration_test "kvm" "vsphere" $memory
    migration_test "vsphere" "kvm" $memory
    migration_test "kvm" "hyperv" $memory
done
```

### Performance Baseline Targets

#### VM Lifecycle Performance Targets
| Operation | KVM/QEMU | vSphere | Hyper-V | XCP-ng | Proxmox |
|-----------|----------|---------|---------|---------|---------|
| VM Create | <15s | <20s | <25s | <18s | <15s |
| VM Start | <10s | <12s | <8s | <10s | <8s |
| VM Stop | <5s | <8s | <3s | <5s | <3s |
| Snapshot | <30s | <45s | <15s | <35s | <25s |

#### Resource Overhead Targets
| Metric | Target | Acceptable | Needs Optimization |
|--------|--------|------------|-------------------|
| CPU Overhead | <3% | <5% | >5% |
| Memory Overhead | <5% | <8% | >8% |
| Storage Overhead | <2% | <5% | >5% |
| Network Overhead | <3% | <5% | >5% |

#### Migration Performance Targets
| VM Memory | Max Migration Time | Max Downtime | Min Transfer Rate |
|-----------|-------------------|--------------|-------------------|
| 1GB | 30s | 100ms | 50 MB/s |
| 4GB | 60s | 200ms | 80 MB/s |
| 8GB | 90s | 300ms | 100 MB/s |
| 16GB | 180s | 500ms | 120 MB/s |

### Automated Benchmarking Pipeline

#### Continuous Performance Monitoring
```go
// Performance monitoring service
type PerformanceBenchmark struct {
    hypervisors []string
    tests       []TestSuite
    results     *ResultsDatabase
    alerting    *AlertManager
}

func (pb *PerformanceBenchmark) RunContinuousBenchmarks(ctx context.Context) error {
    ticker := time.NewTicker(1 * time.Hour)
    defer ticker.Stop()
    
    for {
        select {
        case <-ticker.C:
            for _, hypervisor := range pb.hypervisors {
                results, err := pb.runTestSuite(hypervisor)
                if err != nil {
                    log.Printf("Benchmark failed for %s: %v", hypervisor, err)
                    continue
                }
                
                // Store results
                pb.results.Store(hypervisor, results)
                
                // Check for performance regressions
                if pb.detectRegression(hypervisor, results) {
                    pb.alerting.SendAlert(hypervisor, results)
                }
            }
        case <-ctx.Done():
            return ctx.Err()
        }
    }
}
```

#### Performance Regression Detection
```go
func (pb *PerformanceBenchmark) detectRegression(hypervisor string, current *BenchmarkResults) bool {
    baseline := pb.results.GetBaseline(hypervisor)
    if baseline == nil {
        return false
    }
    
    // Check for significant performance degradation (>10%)
    thresholds := map[string]float64{
        "vm_creation_time": 1.1,
        "cpu_performance":  0.9,
        "memory_bandwidth": 0.9,
        "storage_iops":     0.9,
        "network_throughput": 0.9,
    }
    
    for metric, threshold := range thresholds {
        currentValue := current.GetMetric(metric)
        baselineValue := baseline.GetMetric(metric)
        
        ratio := currentValue / baselineValue
        if (metric == "vm_creation_time" && ratio > threshold) ||
           (metric != "vm_creation_time" && ratio < threshold) {
            return true
        }
    }
    
    return false
}
```

### Results Analysis and Reporting

#### Performance Report Generation
```go
func (pb *PerformanceBenchmark) GenerateReport(period time.Duration) (*PerformanceReport, error) {
    report := &PerformanceReport{
        Period:      period,
        Hypervisors: make(map[string]*HypervisorPerformance),
        Summary:     &PerformanceSummary{},
    }
    
    for _, hypervisor := range pb.hypervisors {
        results := pb.results.GetResultsForPeriod(hypervisor, period)
        
        performance := &HypervisorPerformance{
            Hypervisor: hypervisor,
            Metrics:    pb.calculateAverages(results),
            Trends:     pb.calculateTrends(results),
            Ranking:    pb.calculateRanking(hypervisor, results),
        }
        
        report.Hypervisors[hypervisor] = performance
    }
    
    report.Summary = pb.generateSummary(report.Hypervisors)
    return report, nil
}
```

#### Performance Comparison Dashboard
```yaml
# Performance dashboard configuration
dashboard:
  refresh_interval: 30s
  
  panels:
    - title: "VM Creation Performance"
      type: "bar_chart"
      metrics:
        - "vm_creation_time"
      hypervisors: ["kvm", "vsphere", "hyperv", "xcp-ng", "proxmox"]
      
    - title: "CPU Performance Index"
      type: "line_chart"  
      metrics:
        - "cpu_performance_score"
      time_range: "24h"
      
    - title: "Migration Performance"
      type: "heatmap"
      x_axis: "source_hypervisor"
      y_axis: "target_hypervisor"
      metric: "migration_success_rate"
      
    - title: "Resource Overhead"
      type: "stacked_bar"
      metrics:
        - "cpu_overhead"
        - "memory_overhead"
        - "storage_overhead"
        - "network_overhead"
```

### Performance Optimization Recommendations

#### KVM/QEMU Optimizations
```yaml
kvm_optimizations:
  cpu:
    - enable_kvm_pv_clock: true
    - cpu_mode: "host-passthrough"
    - cpu_topology: "match_host"
    - isolate_cpu_cores: true
    
  memory:
    - enable_huge_pages: true
    - huge_page_size: "1GB"
    - memory_backing: "file"
    - enable_ksm: false  # Disable for performance-critical workloads
    
  storage:
    - virtio_scsi_multiqueue: true
    - direct_io: true
    - cache_mode: "none"
    - io_threads: true
    
  network:
    - virtio_net_multiqueue: true
    - vhost_net: true
    - rx_queue_size: 1024
    - tx_queue_size: 1024
```

#### VMware vSphere Optimizations  
```yaml
vsphere_optimizations:
  cpu:
    - cpu_reservation: "minimum_guaranteed"
    - cpu_shares: "high"
    - latency_sensitivity: "high"
    - numa_affinity: "strict"
    
  memory:
    - memory_reservation: "100%"
    - memory_shares: "high"
    - disable_ballooning: true  # For performance-critical VMs
    - large_pages: "always"
    
  storage:
    - scsi_controller: "paravirtual"
    - disk_mode: "persistent"
    - disable_disk_uuid: false
    - storage_io_control: true
    
  network:
    - network_adapter: "vmxnet3"
    - rss_queues: "auto"
    - interrupt_mode: "msi"
```

### Integration with NovaCron Monitoring

#### Performance Metrics Collection
```go
// Integration with existing NovaCron monitoring
type HypervisorPerformanceCollector struct {
    metricsStore *monitoring.MetricsStore
    benchmarks   *PerformanceBenchmark
}

func (hpc *HypervisorPerformanceCollector) CollectAndStore(ctx context.Context) {
    for _, hypervisor := range hpc.benchmarks.hypervisors {
        metrics, err := hpc.benchmarks.collectRealTimeMetrics(hypervisor)
        if err != nil {
            continue
        }
        
        // Store in NovaCron metrics store
        hpc.metricsStore.Store(monitoring.Metric{
            Name:        "hypervisor_performance",
            Tags:        map[string]string{"hypervisor": hypervisor},
            Value:       metrics.OverallScore,
            Timestamp:   time.Now(),
            Dimensions:  metrics.DetailedMetrics,
        })
    }
}
```

This comprehensive benchmarking framework provides the foundation for continuous performance monitoring and optimization of NovaCron's hypervisor integration layer, ensuring optimal performance across all supported virtualization platforms.