# RDMA Quick Reference Card

## One-Page Setup Guide

### 1. Prerequisites Check (30 seconds)

```bash
# Check hardware
ibv_devices
# Expected: mlx5_0 or similar

# Verify libibverbs
pkg-config --cflags --libs libibverbs
# Expected: -I/usr/include -libverbs
```

### 2. Quick Start (2 minutes)

```bash
# Install dependencies (Ubuntu)
sudo apt-get install -y libibverbs-dev librdmacm-dev rdma-core

# Build NovaCron with RDMA
cd /home/kp/novacron/backend
CGO_ENABLED=1 go build -tags rdma ./cmd/api-server

# Enable RDMA in config
sed -i 's/enable_rdma: false/enable_rdma: true/' ../configs/dwcp.yaml

# Run tests
cd core/network/dwcp/transport/rdma
go test -v -short  # No hardware needed
```

### 3. Verify RDMA Working (1 minute)

```bash
# Start NovaCron
./api-server --config=../configs/dwcp.yaml

# Check logs for RDMA
tail -f /var/log/novacron/dwcp.log | grep "RDMA"
# Expected: "RDMA available and ready" or "RDMA not available, falling back to TCP"

# Check metrics
curl -s http://localhost:9090/metrics | grep rdma
```

## Configuration Cheat Sheet

### Minimal (Auto-detect)
```yaml
transport:
  amst:
    enable_rdma: true
```

### Optimal for Latency (<1μs)
```yaml
transport:
  amst:
    enable_rdma: true
    rdma_use_event_channel: false  # Polling
    rdma_max_inline_data: 256
    rdma_max_send_wr: 512
    rdma_max_recv_wr: 512
```

### Optimal for Throughput (>100 Gbps)
```yaml
transport:
  amst:
    enable_rdma: true
    rdma_mtu: 4096
    rdma_max_send_wr: 2048
    rdma_max_recv_wr: 2048
    rdma_send_buffer_mb: 8
    rdma_recv_buffer_mb: 8
```

## Common Commands

### Hardware Management
```bash
# List devices
ibv_devices

# Device info
ibv_devinfo -d mlx5_0 -v

# Check port state
ibv_devinfo | grep state

# Test loopback
ib_write_lat  # Server
ib_write_lat SERVER_IP  # Client
```

### Kernel Modules
```bash
# Load
sudo modprobe ib_core ib_uverbs rdma_cm mlx5_ib

# Verify
lsmod | grep -E 'ib_|rdma|mlx'

# Auto-load on boot
echo "mlx5_ib" | sudo tee -a /etc/modules
```

### Performance Testing
```bash
# Latency test (<1μs target)
go test -bench=BenchmarkRDMALatency -benchmem

# Throughput test (>100 Gbps target)
go test -bench=BenchmarkRDMAThroughput -benchmem

# All benchmarks
go test -bench=. -benchmem -benchtime=10s
```

### Network Configuration
```bash
# InfiniBand
sudo ip addr add 192.168.100.1/24 dev ib0
sudo ip link set ib0 up
sudo ip link set ib0 mtu 65520

# RoCE (Ethernet)
sudo ip link set eth0 mtu 9000
sudo ethtool -A eth0 rx on tx on
```

## Troubleshooting

### No RDMA devices found
```bash
# Check hardware
lspci | grep -i mellanox

# Reload drivers
sudo modprobe -r mlx5_ib mlx5_core
sudo modprobe mlx5_core mlx5_ib

# Check dmesg
sudo dmesg | grep -i rdma
```

### Permission denied
```bash
# Check permissions
ls -l /dev/infiniband/

# Add user to rdma group
sudo usermod -a -G rdma $USER

# Fix udev rules
sudo sh -c 'echo "KERNEL==\"uverbs*\", MODE=\"0666\"" > /etc/udev/rules.d/90-rdma.rules'
sudo udevadm control --reload-rules
```

### Port not active
```bash
# Check cable connection
ibv_devinfo -d mlx5_0 | grep state

# Check link
sudo ethtool eth0  # For RoCE

# Restart service
sudo systemctl restart rdma
```

### Low performance
```bash
# Check MTU
ibv_devinfo | grep active_mtu

# Check CPU frequency
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Set performance mode
sudo cpupower frequency-set -g performance

# Check interrupts
cat /proc/interrupts | grep mlx
```

## Performance Targets

| Metric | Target | Command to Verify |
|--------|--------|------------------|
| Small message latency | <1μs | `go test -bench=BenchmarkRDMALatency` |
| Large message throughput | >100 Gbps | `go test -bench=BenchmarkRDMAThroughput` |
| Memory registration | <500ns | `go test -bench=BenchmarkMemoryRegistration` |
| Zero-copy benefit | >50% faster | `go test -bench=BenchmarkZeroCopy` |

## API Quick Reference

### Go API
```go
import "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/transport/rdma"

// Check availability
if !rdma.CheckAvailability() {
    log.Fatal("RDMA not available")
}

// Initialize
config := rdma.DefaultConfig()
mgr, err := rdma.NewRDMAManager(config, logger)
defer mgr.Close()

// Send (two-sided)
err = mgr.Send(data)

// Receive (two-sided)
n, err := mgr.Receive(buffer)

// Write (one-sided, zero-copy)
err = mgr.Write(data, remoteAddr, remoteRKey)

// Statistics
stats := mgr.GetStats()
latency := stats["avg_send_latency_us"].(float64)
```

### Configuration Values

| Parameter | Values | Default | Notes |
|-----------|--------|---------|-------|
| `rdma_qp_type` | RC, UD, DCT | RC | RC = reliable |
| `rdma_mtu` | 256, 512, 1024, 2048, 4096 | 4096 | Larger = better throughput |
| `rdma_max_inline_data` | 0-512 | 256 | Bytes, larger = better for small msgs |
| `rdma_use_event_channel` | true, false | false | false = polling (lower latency) |
| `rdma_use_srq` | true, false | false | true = memory efficient |

## Key Files

| File | Purpose |
|------|---------|
| `rdma_native.c/h` | libibverbs wrapper (C) |
| `rdma_cgo.go` | CGo bindings (Go/C interface) |
| `rdma.go` | High-level RDMA manager |
| `rdma_transport.go` | AMST integration |
| `rdma_test.go` | Unit tests |
| `rdma_benchmark_test.go` | Performance tests |
| `dwcp.yaml` | Configuration |

## Support Resources

- **Setup Guide**: `/home/kp/novacron/docs/RDMA_SETUP_GUIDE.md`
- **Completion Summary**: `/home/kp/novacron/docs/PHASE2_RDMA_COMPLETION_SUMMARY.md`
- **Tests**: `go test -v` in `backend/core/network/dwcp/transport/rdma/`
- **Benchmarks**: `go test -bench=. -benchmem`
- **Logs**: `/var/log/novacron/dwcp.log`
- **Metrics**: `http://localhost:9090/metrics`

## Decision Tree

```
Do you have RDMA hardware?
├─ YES → Is libibverbs installed?
│        ├─ YES → Is port ACTIVE?
│        │        ├─ YES → Enable RDMA ✓
│        │        └─ NO → Check cable/link
│        └─ NO → Install: apt-get install libibverbs-dev
└─ NO → Automatic TCP fallback (works seamlessly)
```

## Production Checklist

- [ ] Hardware detected: `ibv_devices`
- [ ] Port active: `ibv_devinfo | grep state`
- [ ] Modules loaded: `lsmod | grep ib_`
- [ ] Config updated: `enable_rdma: true`
- [ ] Build with CGo: `CGO_ENABLED=1 go build`
- [ ] Tests pass: `go test -v`
- [ ] Benchmarks meet targets: `go test -bench=.`
- [ ] Logs show RDMA: `grep RDMA dwcp.log`
- [ ] Metrics exposed: `curl :9090/metrics | grep rdma`
- [ ] Fallback tested: Disable RDMA, verify TCP works

---

**TIP**: On non-RDMA systems, everything still works via automatic TCP fallback!
