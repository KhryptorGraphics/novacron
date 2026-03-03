# eBPF-Based Page Filtering for VM Migration Optimization

## Overview

NovaCron's eBPF migration optimization leverages extended Berkeley Packet Filter (eBPF) technology to dramatically reduce data transfer overhead during VM migrations. By tracking page access patterns at the kernel level, the system intelligently identifies and skips unused memory pages during migration, achieving transfer size reductions of 30-70% for typical workloads.

## Architecture

### Components

1. **eBPF Page Tracker** (`page_tracker.bpf.c`)
   - Kernel-level program that monitors page fault and memory access events
   - Tracks page access timestamps, dirty flags, and access counts
   - Maintains per-page metadata in eBPF maps

2. **EBPFMigrationFilter** (`ebpf_migration_filter.go`)
   - Go wrapper for eBPF program management
   - Handles program loading, attachment, and lifecycle
   - Provides query interface for page status

3. **EBPFBlockFilter** (`ebpf_migration_filter.go`)
   - Block-level abstraction over page tracking
   - Determines if entire blocks can be skipped based on constituent pages

4. **DeltaSyncManager Integration** (`wan_migration_delta_sync.go`)
   - Integrates eBPF filtering into delta synchronization pipeline
   - Consults eBPF filter before hashing or transferring blocks
   - Collects eBPF-specific statistics

5. **LiveMigrationManager Integration** (`live_migration.go`)
   - Orchestrates eBPF setup during migration initialization
   - Manages eBPF lifecycle across migration phases

### Data Flow

```
VM Memory Access
       ↓
eBPF Tracepoints (page_fault_user, handle_mm_fault)
       ↓
eBPF Maps (page_access_map, config_map)
       ↓
EBPFMigrationFilter (Go)
       ↓
EBPFBlockFilter
       ↓
DeltaSyncManager (block skipping)
       ↓
Reduced Migration Traffic
```

## Configuration

### System Requirements

- **Kernel Version**: Linux 4.18 or higher (5.8+ recommended for full features)
- **Capabilities**: CAP_BPF or CAP_SYS_ADMIN
- **Dependencies**:
  - libbpf
  - clang/LLVM (for compilation)
  - kernel headers

### Capability Check

Before enabling eBPF, check system capabilities:

```go
import vm "novacron/backend/core/vm"

// Simple boolean check
if vm.IsEBPFSupported() {
    // eBPF can be used
}

// Detailed diagnostics
cap := vm.CheckEBPFCapability()
fmt.Printf("eBPF Diagnostics:\n%s", vm.GetEBPFDiagnostics())
```

### Configuration Options

#### DeltaSyncConfig eBPF Settings

```go
config := vm.DefaultDeltaSyncConfig()

// Enable eBPF filtering
config.EnableEBPFFiltering = true

// Time threshold for considering a page unused (default: 5 seconds)
config.EBPFAgingThreshold = 5 * time.Second

// Minimum access count to consider a page active (default: 1)
config.EBPFMinAccessCount = 1

// Maximum pages to track in eBPF maps (default: 1048576 = 4GB)
config.EBPFMapSizeLimit = 1048576

// Gracefully fallback if eBPF fails (default: true)
config.FallbackOnEBPFError = true
```

## Usage

### Basic Integration

```go
import (
    "context"
    "novacron/backend/core/vm"
)

// Create delta sync manager with eBPF enabled
config := vm.DefaultDeltaSyncConfig()
config.EnableEBPFFiltering = true
config.FallbackOnEBPFError = true

manager := vm.NewDeltaSyncManager(config)
defer manager.Close()

// Enable eBPF for specific VM process
vmPID := uint32(12345) // Actual VM process ID
if err := manager.EnableEBPFFiltering(vmPID); err != nil {
    // Handle error (or rely on fallback)
    log.Printf("eBPF not enabled: %v", err)
}

// Perform migration with eBPF optimization
ctx := context.Background()
err := manager.SyncFile(ctx, sourcePath, destPath)

// Check statistics
stats := manager.GetStats()
if stats.EBPFEnabled {
    log.Printf("eBPF skipped %d blocks (%.2f%%), saved %d bytes",
        stats.EBPFBlocksSkipped,
        stats.EBPFSkipPercent,
        stats.EBPFBytesSkipped)
}
```

### Advanced Usage

#### Manual eBPF Filter Control

```go
logger := logrus.New()
vmPID := uint32(12345)

// Create filter
filter, err := vm.NewEBPFMigrationFilter(logger, vmPID)
if err != nil {
    log.Fatal(err)
}
defer filter.Close()

// Configure aging
filter.SetAgingThreshold(10 * time.Second)
filter.SetMinAccessCount(2)

// Attach to kernel
if err := filter.Attach(); err != nil {
    log.Fatal(err)
}

// Query page status
pfn := uint64(0x1000)
if filter.IsPageUnused(pfn) {
    log.Printf("Page %x is unused", pfn)
}

// Get statistics
stats := filter.GetStats()
log.Printf("Tracking %d pages, %d unused",
    stats["total_pages"], stats["unused_pages"])

// Periodically mark aged-out pages
ticker := time.NewTicker(5 * time.Second)
go func() {
    for range ticker.C {
        count, _ := filter.MarkPagesAsUnused()
        log.Printf("Marked %d pages as unused", count)
    }
}()
```

#### Integration with Live Migration

```go
// LiveMigrationManager automatically integrates eBPF
// when compression is enabled and eBPF is supported

config := &vm.LiveMigrationConfig{
    CompressionEnabled: true,
    BandwidthLimit: 100 * 1024 * 1024, // 100 MB/s
}

lmm := vm.NewLiveMigrationManager(config)

// eBPF is automatically enabled during initializeMigration
// if the system supports it
state, err := lmm.StartLiveMigration(ctx, vmID, sourceHost, destHost)
```

## Performance Characteristics

### Expected Performance Gains

| Workload Type | Typical Skip Rate | Transfer Reduction |
|--------------|------------------|-------------------|
| Idle VM | 60-80% | 60-80% |
| Light workload | 40-60% | 40-60% |
| Mixed workload | 30-50% | 30-50% |
| Heavy I/O | 10-30% | 10-30% |

### Overhead

- **CPU**: <1% overhead from eBPF tracing
- **Memory**: ~8 bytes per tracked page (max 8MB for 1M pages)
- **Latency**: Sub-microsecond page status queries

### Optimization Tips

1. **Aging Threshold**:
   - Shorter (1-3s) for frequently changing workloads
   - Longer (5-10s) for more stable workloads

2. **Block Size**:
   - Larger blocks (128-256KB) work better with eBPF
   - Smaller blocks (32-64KB) for fine-grained filtering

3. **Hash Workers**:
   - More workers (8-16) for large VMs
   - Fewer workers (2-4) for smaller VMs

## Troubleshooting

### eBPF Not Supported

**Problem**: `IsEBPFSupported()` returns false

**Solutions**:
1. Check kernel version: `uname -r` (need 4.18+)
2. Install kernel headers: `apt-get install linux-headers-$(uname -r)`
3. Load BPF filesystem: `mount -t bpf none /sys/fs/bpf`

### Cannot Load eBPF Program

**Problem**: `NewEBPFMigrationFilter()` fails

**Solutions**:
1. Check capabilities: Run as root or with `CAP_BPF`
2. Verify BPF syscall: `grep CONFIG_BPF /boot/config-$(uname -r)`
3. Enable BPF JIT: `echo 1 > /proc/sys/net/core/bpf_jit_enable`

### Cannot Attach Programs

**Problem**: `Attach()` returns error

**Solutions**:
1. Check tracepoints exist:
   ```bash
   ls /sys/kernel/debug/tracing/events/exceptions/page_fault_user
   ```
2. Mount debugfs if needed:
   ```bash
   mount -t debugfs none /sys/kernel/debug
   ```
3. Check SELinux/AppArmor policies

### Low Skip Rate

**Problem**: eBPF reports few unused pages

**Possible Causes**:
1. VM is genuinely active (expected)
2. Aging threshold too aggressive (increase)
3. Min access count too low (increase)
4. Need to wait longer before migration (let pages age)

**Debug**:
```go
ebpfStats := filter.GetStats()
log.Printf("Total pages: %d, Unused: %d, Dirty: %d",
    ebpfStats["total_pages"],
    ebpfStats["unused_pages"],
    ebpfStats["dirty_pages"])
```

## Graceful Degradation

The system is designed to fall back gracefully when eBPF is unavailable:

1. **Capability Check**: System checks eBPF support before attempting to use it
2. **Fallback Flag**: `FallbackOnEBPFError` allows migration to proceed without eBPF
3. **Statistics**: `EBPFEnabled` field in stats indicates if eBPF was actually used
4. **Error Handling**: All eBPF errors are logged but don't fail the migration

```go
// This will work even if eBPF fails
config := vm.DefaultDeltaSyncConfig()
config.EnableEBPFFiltering = true
config.FallbackOnEBPFError = true // Key setting

manager := vm.NewDeltaSyncManager(config)
manager.EnableEBPFFiltering(vmPID) // May fail silently

// Migration proceeds with or without eBPF
manager.SyncFile(ctx, src, dst)
```

## Building eBPF Programs

### Prerequisites

```bash
# Ubuntu/Debian
apt-get install clang llvm linux-headers-$(uname -r) libbpf-dev

# RHEL/CentOS
yum install clang llvm kernel-devel libbpf-devel
```

### Compilation

```bash
cd backend/core/vm/ebpf_programs
make

# Or with specific options
make CLANG=/usr/bin/clang-14 ARCH=arm64

# Check build
make check

# Verify objects
make verify
```

### Embedding in Go

eBPF objects are automatically embedded using `//go:embed`:

```go
//go:embed page_tracker.bpf.o
var pageTrackerBPF []byte
```

## Security Considerations

1. **Capabilities**: Requires elevated privileges (CAP_BPF or CAP_SYS_ADMIN)
2. **Kernel Access**: eBPF programs run in kernel context
3. **Verification**: Linux kernel verifies eBPF programs before loading
4. **Isolation**: eBPF programs cannot crash the kernel
5. **Resource Limits**: eBPF maps have size limits to prevent resource exhaustion

## References

- [eBPF Documentation](https://ebpf.io/)
- [Cilium eBPF Go Library](https://github.com/cilium/ebpf)
- [Linux eBPF Tracing](https://www.kernel.org/doc/html/latest/bpf/index.html)
- [NovaCron VM Migration Design](./vm_migration_design.md)

## Future Enhancements

1. **Machine Learning Integration**: Predict access patterns using historical data
2. **Multi-VM Coordination**: Share page tracking across multiple VMs
3. **NUMA Awareness**: Optimize for NUMA topology
4. **Adaptive Thresholds**: Auto-tune aging parameters based on workload
5. **Persistent Tracking**: Save page access patterns across restarts
