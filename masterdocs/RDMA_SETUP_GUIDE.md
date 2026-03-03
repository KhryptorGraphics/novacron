# RDMA Setup and Configuration Guide

## Overview

This guide explains how to set up and configure RDMA (Remote Direct Memory Access) for NovaCron's DWCP transport layer. RDMA provides ultra-low latency (<1μs) and high throughput (>100 Gbps) networking when compatible hardware is available.

## Architecture

### Phase 2 Implementation

Phase 2 delivers production-ready RDMA with full libibverbs integration:

- **Hardware Acceleration**: Direct access to InfiniBand/RoCE NICs via libibverbs
- **Zero-Copy Transfers**: Eliminate CPU overhead with kernel bypass
- **Sub-Microsecond Latency**: <1μs latency for small messages
- **100+ Gbps Throughput**: Full line-rate on capable hardware
- **Automatic Fallback**: Seamless TCP fallback when RDMA unavailable

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    RDMA Transport Layer                      │
│                                                               │
│  ┌────────────────────────────────────────────────────┐    │
│  │          RDMATransport (rdma_transport.go)         │    │
│  │  - Auto-detection and fallback logic               │    │
│  │  - Integration with AMST                           │    │
│  └─────────────┬──────────────────────────────────────┘    │
│                │                                             │
│  ┌─────────────▼──────────────────────────────────────┐    │
│  │         RDMAManager (rdma.go)                       │    │
│  │  - High-level Go API                               │    │
│  │  - Statistics and monitoring                       │    │
│  │  - Connection management                           │    │
│  └─────────────┬──────────────────────────────────────┘    │
│                │                                             │
│  ┌─────────────▼──────────────────────────────────────┐    │
│  │         CGo Bindings (rdma_cgo.go)                  │    │
│  │  - Go/C interface                                   │    │
│  │  - Memory safety                                    │    │
│  └─────────────┬──────────────────────────────────────┘    │
│                │                                             │
│  ┌─────────────▼──────────────────────────────────────┐    │
│  │    libibverbs Wrapper (rdma_native.c/.h)           │    │
│  │  - Device enumeration                              │    │
│  │  - Queue pair management                           │    │
│  │  - Completion handling                             │    │
│  │  - Memory registration                             │    │
│  └─────────────┬──────────────────────────────────────┘    │
│                │                                             │
│  ┌─────────────▼──────────────────────────────────────┐    │
│  │            libibverbs (Hardware Layer)              │    │
│  │  - InfiniBand verbs API                            │    │
│  │  - Hardware abstraction                            │    │
│  └─────────────┬──────────────────────────────────────┘    │
│                │                                             │
│  ┌─────────────▼──────────────────────────────────────┐    │
│  │      RDMA NIC (Mellanox, Intel, Chelsio, etc.)     │    │
│  │  - InfiniBand (IB)                                 │    │
│  │  - RoCE (RDMA over Converged Ethernet)            │    │
│  │  - iWARP (Internet Wide Area RDMA Protocol)       │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

### Hardware Requirements

**Supported RDMA NICs:**
- Mellanox ConnectX-3/4/5/6/7 (InfiniBand, RoCE)
- Intel Omni-Path Adapters
- Chelsio T5/T6 (iWARP)
- Broadcom NetXtreme-E (RoCE)
- Any InfiniBand or RoCE-capable NIC

**Minimum Specifications:**
- PCIe 3.0 x8 or better
- NUMA-aware system recommended
- Linux kernel 3.10+ (5.x+ recommended)

### Software Requirements

**Linux Packages:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    libibverbs-dev \
    librdmacm-dev \
    ibverbs-utils \
    rdma-core \
    perftest

# RHEL/CentOS/Rocky
sudo yum install -y \
    libibverbs-devel \
    librdmacm-devel \
    rdma-core-devel \
    perftest

# Arch Linux
sudo pacman -S rdma-core
```

**Build Tools:**
```bash
# Required for CGo compilation
sudo apt-get install -y gcc make

# Verify installation
pkg-config --cflags --libs libibverbs
```

## Installation

### 1. Verify RDMA Hardware

```bash
# List RDMA devices
ibv_devices

# Expected output:
#     device                 node GUID
#     ------              ----------------
#     mlx5_0              0002c9030086a100

# Check device capabilities
ibv_devinfo -d mlx5_0

# Test loopback latency
ib_write_lat
```

### 2. Configure RDMA Kernel Modules

```bash
# Load RDMA modules
sudo modprobe ib_core
sudo modprobe ib_uverbs
sudo modprobe rdma_cm
sudo modprobe mlx5_core  # For Mellanox NICs
sudo modprobe mlx5_ib

# Make persistent
echo "ib_core" | sudo tee -a /etc/modules
echo "ib_uverbs" | sudo tee -a /etc/modules
echo "rdma_cm" | sudo tee -a /etc/modules
echo "mlx5_ib" | sudo tee -a /etc/modules

# Verify loaded modules
lsmod | grep -E 'ib_|rdma|mlx'
```

### 3. Configure Network

**For InfiniBand:**
```bash
# Set IP on IB interface
sudo ip addr add 192.168.100.1/24 dev ib0
sudo ip link set ib0 up

# Enable connected mode (higher MTU)
sudo sh -c 'echo connected > /sys/class/net/ib0/mode'

# Set MTU
sudo ip link set ib0 mtu 65520
```

**For RoCE (RDMA over Ethernet):**
```bash
# Enable RoCE on Ethernet interface
sudo cma_roce_mode -d mlx5_0 -p 1 -m 2

# Enable flow control
sudo ethtool -A eth0 rx on tx on

# Enable PFC (Priority Flow Control)
sudo mlnx_qos -i eth0 --trust=dscp

# Set MTU (Jumbo frames recommended)
sudo ip link set eth0 mtu 9000
```

### 4. Configure NovaCron

Edit `/home/kp/novacron/configs/dwcp.yaml`:

```yaml
transport:
  amst:
    # RDMA Configuration
    enable_rdma: true
    rdma_device: "mlx5_0"  # Leave empty for auto-detection
    rdma_port: 1
    rdma_gid_index: 0
    rdma_mtu: 4096
    rdma_max_inline_data: 256
    rdma_max_send_wr: 1024
    rdma_max_recv_wr: 1024
    rdma_max_sge: 16
    rdma_qp_type: "RC"  # Reliable Connection
    rdma_use_srq: false
    rdma_use_event_channel: false  # Polling for <1μs latency
    rdma_send_buffer_mb: 4
    rdma_recv_buffer_mb: 4
```

### 5. Build NovaCron with RDMA

```bash
cd /home/kp/novacron/backend

# Build with CGo enabled
CGO_ENABLED=1 go build -tags rdma ./cmd/api-server

# Verify RDMA support
./api-server --version
# Should show: RDMA support: enabled
```

## Configuration Options

### RDMA Settings

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `enable_rdma` | Enable RDMA acceleration | `true` | Auto-falls back to TCP |
| `rdma_device` | RDMA device name | `mlx5_0` | Empty = auto-detect |
| `rdma_port` | Physical port number | `1` | Usually 1 or 2 |
| `rdma_gid_index` | GID table index | `0` | For multi-subnet |
| `rdma_mtu` | Path MTU | `4096` | 256, 512, 1024, 2048, 4096 |
| `rdma_max_inline_data` | Inline data threshold | `256` | Bytes, 0-512 |
| `rdma_max_send_wr` | Max send work requests | `1024` | Queue depth |
| `rdma_max_recv_wr` | Max recv work requests | `1024` | Queue depth |
| `rdma_max_sge` | Max scatter-gather elements | `16` | Per work request |
| `rdma_qp_type` | Queue pair type | `RC` | RC, UD, DCT |
| `rdma_use_srq` | Use shared receive queue | `false` | Memory efficiency |
| `rdma_use_event_channel` | Event-driven I/O | `false` | Polling = lower latency |
| `rdma_send_buffer_mb` | Send buffer size (MB) | `4` | Pre-allocated |
| `rdma_recv_buffer_mb` | Receive buffer size (MB) | `4` | Pre-allocated |

### Queue Pair Types

**RC (Reliable Connection)** - Default
- Point-to-point, reliable, ordered delivery
- Best for WAN communication
- Lowest latency
- Connection-oriented

**UD (Unreliable Datagram)**
- Multicast support
- Lower resource usage
- No reliability guarantees
- Connectionless

**DCT (Dynamic Connection Transport)**
- Scalable to thousands of endpoints
- Shared receive queue
- Lower memory footprint
- Best for large clusters

## Performance Tuning

### Optimal Settings for Low Latency (<1μs)

```yaml
# Ultra-low latency configuration
rdma_max_inline_data: 256      # Inline small messages
rdma_use_event_channel: false  # Use busy polling
rdma_max_send_wr: 512          # Smaller queue = lower latency
rdma_max_recv_wr: 512
```

### Optimal Settings for High Throughput (>100 Gbps)

```yaml
# Maximum throughput configuration
rdma_mtu: 4096                 # Largest MTU
rdma_max_send_wr: 2048         # Larger queues
rdma_max_recv_wr: 2048
rdma_send_buffer_mb: 8         # Larger buffers
rdma_recv_buffer_mb: 8
rdma_use_srq: true             # Shared receive queue
```

### System-Level Tuning

```bash
# Disable CPU frequency scaling
sudo cpupower frequency-set -g performance

# Set IRQ affinity to dedicated cores
sudo sh -c 'echo 1 > /proc/irq/IRQ_NUMBER/smp_affinity'

# Increase locked memory limit
sudo sh -c 'echo "* soft memlock unlimited" >> /etc/security/limits.conf'
sudo sh -c 'echo "* hard memlock unlimited" >> /etc/security/limits.conf'

# Enable huge pages
sudo sh -c 'echo 1024 > /proc/sys/vm/nr_hugepages'

# Disable IOMMU (if not needed for virtualization)
# Add to kernel boot parameters: intel_iommu=off amd_iommu=off
```

## Testing

### Run RDMA Tests

```bash
cd /home/kp/novacron/backend/core/network/dwcp/transport/rdma

# Run all tests
go test -v

# Run specific test
go test -v -run TestRDMAInitialization

# Skip tests if no hardware
go test -v -short
```

### Run Performance Benchmarks

```bash
# Run all benchmarks
go test -bench=. -benchmem -benchtime=10s

# Latency benchmark (target <1μs)
go test -bench=BenchmarkRDMALatency -benchmem

# Throughput benchmark (target >100 Gbps)
go test -bench=BenchmarkRDMAThroughput -benchmem

# Generate CPU profile
go test -bench=BenchmarkRDMALatency -cpuprofile=cpu.prof
go tool pprof cpu.prof
```

### Expected Benchmark Results

**On RDMA Hardware (Mellanox ConnectX-5 100GbE):**
```
BenchmarkRDMALatency-16              2000000    650 ns/op
BenchmarkRDMAThroughput-16             10000  105.2 Gbps
BenchmarkRDMASmallMessage/64-16     50000000     25 ns/op
BenchmarkRDMAMemoryRegistration     10000000    120 ns/op
```

**On Non-RDMA System (TCP Fallback):**
```
BenchmarkRDMALatency-16               100000  12500 ns/op
BenchmarkRDMAThroughput-16              1000   9.8 Gbps
```

## Troubleshooting

### Common Issues

**1. No RDMA devices found**
```bash
# Check hardware detection
lspci | grep -i mellanox

# Check kernel modules
lsmod | grep mlx

# Reload modules
sudo modprobe -r mlx5_ib mlx5_core
sudo modprobe mlx5_core
sudo modprobe mlx5_ib
```

**2. Permission denied**
```bash
# Check device permissions
ls -l /dev/infiniband/

# Add user to rdma group
sudo usermod -a -G rdma $USER

# Update udev rules
sudo sh -c 'echo "KERNEL==\"uverbs*\", MODE=\"0666\"" > /etc/udev/rules.d/90-rdma.rules'
sudo udevadm control --reload-rules
```

**3. Connection failures**
```bash
# Check port state
ibv_devinfo -d mlx5_0 | grep state
# Should show: state: PORT_ACTIVE

# Check connectivity
ibping -S  # On server
ibping -c mlx5_0 SERVER_LID  # On client

# Verify GID
show_gids
```

**4. Low performance**
```bash
# Check for packet drops
ibv_devinfo -d mlx5_0 -v | grep -E 'rx_drops|tx_drops'

# Monitor RDMA counters
rdma stat show

# Check CPU affinity
cat /proc/interrupts | grep mlx

# Verify MTU
ibv_devinfo | grep active_mtu
```

### Debugging

**Enable debug logging:**
```yaml
dwcp:
  log_level: "debug"
```

**Check RDMA status:**
```bash
# Via API
curl http://localhost:8080/api/v1/transport/status

# Via logs
tail -f /var/log/novacron/dwcp.log | grep -i rdma
```

**Capture RDMA traffic:**
```bash
# Use ibdump (like tcpdump for InfiniBand)
sudo ibdump -d mlx5_0 -w rdma.pcap

# Analyze with Wireshark
wireshark rdma.pcap
```

## Monitoring

### Prometheus Metrics

NovaCron exposes RDMA metrics at `/metrics`:

```
# Latency (nanoseconds)
dwcp_rdma_send_latency_ns{quantile="0.5"}
dwcp_rdma_send_latency_ns{quantile="0.99"}

# Throughput (bytes/sec)
dwcp_rdma_bytes_sent_total
dwcp_rdma_bytes_received_total

# Operations
dwcp_rdma_send_operations_total
dwcp_rdma_recv_operations_total
dwcp_rdma_write_operations_total  # One-sided
dwcp_rdma_read_operations_total   # One-sided

# Errors
dwcp_rdma_send_errors_total
dwcp_rdma_recv_errors_total

# Connection state
dwcp_rdma_connected{device="mlx5_0"}
```

### Grafana Dashboard

Import the RDMA dashboard from `configs/grafana/dwcp-rdma-dashboard.json`

## Security Considerations

### Network Isolation

```bash
# Use separate VLAN for RDMA traffic
sudo ip link add link eth0 name eth0.100 type vlan id 100
sudo ip addr add 192.168.100.1/24 dev eth0.100
```

### Firewall Rules

```bash
# Allow RDMA ports
sudo iptables -A INPUT -p tcp --dport 18515 -j ACCEPT  # RDMA-CM
sudo iptables -A INPUT -p udp --dport 4791 -j ACCEPT   # RoCE

# Save rules
sudo iptables-save > /etc/iptables/rules.v4
```

### Access Control

```yaml
security:
  rdma:
    allowed_gids:
      - "fe80:0000:0000:0000:0002:c903:0086:a100"
    require_authentication: true
```

## Advanced Features

### One-Sided RDMA Operations

```go
// RDMA Write (zero-copy remote write)
err := rdmaManager.Write(localBuffer, remoteAddr, remoteRKey)

// RDMA Read (zero-copy remote read)
err := rdmaManager.Read(localBuffer, remoteAddr, remoteRKey)
```

### Shared Receive Queue (SRQ)

Enable for memory efficiency with many connections:
```yaml
rdma_use_srq: true
```

### Adaptive Routing

For fault tolerance on InfiniBand:
```yaml
rdma_adaptive_routing: true
```

## Production Checklist

- [ ] RDMA hardware installed and detected (`ibv_devices`)
- [ ] Kernel modules loaded (`lsmod | grep ib_`)
- [ ] Network configured (MTU, flow control)
- [ ] NovaCron built with CGo enabled
- [ ] Configuration updated (`dwcp.yaml`)
- [ ] Tests passing (`go test`)
- [ ] Benchmarks meeting targets (<1μs latency, >100 Gbps)
- [ ] Monitoring configured (Prometheus, Grafana)
- [ ] Firewall rules applied
- [ ] Automatic fallback tested (disable RDMA, verify TCP works)

## Support

For RDMA-specific issues:
1. Check NovaCron logs: `tail -f /var/log/novacron/dwcp.log`
2. Verify hardware: `ibv_devinfo -v`
3. Test connectivity: `perftest` suite (ib_read_lat, ib_write_bw)
4. Review documentation: https://docs.novacron.io/rdma

For hardware/driver issues:
- Mellanox: https://www.mellanox.com/support
- Intel: https://www.intel.com/content/www/us/en/support/articles/000005688/network-and-i-o/fabric-products.html
- Linux RDMA: https://github.com/linux-rdma/rdma-core

## References

- [RDMA Aware Programming User Manual](https://www.mellanox.com/related-docs/prod_software/RDMA_Aware_Programming_user_manual.pdf)
- [libibverbs Documentation](https://www.rdmamojo.com/category/verbs/)
- [InfiniBand Architecture Specification](https://www.infinibandta.org/)
- [RoCE v2 Specification](https://www.roceinitiative.org/)
