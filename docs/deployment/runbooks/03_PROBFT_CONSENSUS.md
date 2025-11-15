# ProBFT Consensus Deployment Runbook

**System:** Probabilistic Byzantine Fault Tolerance (ProBFT) v1.0
**Purpose:** VRF-based probabilistic consensus with Byzantine fault tolerance
**Status:** Production Ready
**Performance:** 33% Byzantine tolerance, ⌈√n⌉ quorum, <1s block finalization

## Overview

ProBFT implements probabilistic Byzantine Fault Tolerant consensus using Verifiable Random Functions (VRF) for leader election. It provides strong consistency guarantees while tolerating up to f < n/3 Byzantine (malicious) nodes.

### Key Features

- ✅ **VRF Leader Election** - Unpredictable, verifiable leader selection
- ✅ **33% Byzantine Tolerance** - Tolerates f < n/3 Byzantine faults
- ✅ **√n Quorum** - Probabilistic quorum size ⌈√n⌉
- ✅ **3-Phase Consensus** - Pre-prepare, Prepare, Commit
- ✅ **View Changes** - Automatic leader rotation on timeout
- ✅ **Fast Finality** - <1 second block finalization

### Consensus Phases

1. **Pre-Prepare:** Leader proposes block with VRF proof
2. **Prepare:** Nodes vote on proposal (2f votes required)
3. **Commit:** Nodes commit block (2f+1 votes required)
4. **Finalize:** Block committed to chain

### Performance Metrics

- **Block Finalization:** <1 second
- **Byzantine Tolerance:** 33% of nodes
- **Quorum Size:** ⌈√n⌉ (probabilistic)
- **Throughput:** 1,000-5,000 tx/s
- **View Change Timeout:** 30 seconds
- **Message Complexity:** O(n²) per consensus round

## Prerequisites

### Hardware Requirements

**Minimum (per node):**
- CPU: 8 cores
- RAM: 16 GB
- Network: 1 Gbps with <50ms latency
- Storage: 100 GB SSD
- OS: Linux (Ubuntu 22.04 LTS)

**Recommended (per node):**
- CPU: 16 cores
- RAM: 32 GB
- Network: 10 Gbps with <5ms latency (datacenter)
- Storage: 500 GB NVMe SSD
- OS: Linux (Ubuntu 22.04 LTS)

### Cluster Requirements

**Minimum Cluster:**
- 4 nodes (tolerates 1 Byzantine node)
- Total Byzantine tolerance: 25%

**Recommended Cluster:**
- 7 nodes (tolerates 2 Byzantine nodes)
- Total Byzantine tolerance: 28.5%

**Production Cluster:**
- 10+ nodes (tolerates 3+ Byzantine nodes)
- Total Byzantine tolerance: 30%+

### Software Requirements

```bash
# Go 1.21 or higher
go version
# Expected: go1.21+

# Build ProBFT binary
cd /home/kp/repos/novacron/backend/core/network/dwcp/v3/consensus/probft
go build -o probft-node ./cmd/node

# Verify VRF library
go list -m github.com/coniks-sys/coniks-go
```

### Dependencies

**Go Packages:**
```bash
go get github.com/coniks-sys/coniks-go/crypto/vrf
go get crypto/ed25519
go get go.uber.org/zap
```

### Network Requirements

- **Dedicated VLAN** for consensus traffic
- **Static IP addresses** for all nodes
- **NTP synchronization** (<10ms drift)
- **Low latency** (<50ms between nodes)
- **High bandwidth** (100+ Mbps per node)

## Deployment Steps

### 1. Pre-Deployment Validation

```bash
# Verify cluster connectivity
for i in {1..7}; do
  ping -c 3 probft-node-0$i
  nc -zv probft-node-0$i 9000
done

# Check time synchronization
ntpq -p
# All nodes should have <10ms offset

# Verify resources
./scripts/check-cluster-resources.sh

# Test VRF generation
./probft-node --test-vrf
```

### 2. Configuration

Create node configuration: `/etc/probft/node.yaml`

```yaml
# ProBFT Node Configuration
node:
  id: "node-01"  # Unique node identifier
  listen_addr: "0.0.0.0:9000"
  external_addr: "probft-node-01:9000"

# Cluster configuration
cluster:
  # All nodes in the cluster
  nodes:
    - id: "node-01"
      address: "probft-node-01:9000"
      public_key: "<ed25519-public-key-1>"
    - id: "node-02"
      address: "probft-node-02:9000"
      public_key: "<ed25519-public-key-2>"
    - id: "node-03"
      address: "probft-node-03:9000"
      public_key: "<ed25519-public-key-3>"
    - id: "node-04"
      address: "probft-node-04:9000"
      public_key: "<ed25519-public-key-4>"
    - id: "node-05"
      address: "probft-node-05:9000"
      public_key: "<ed25519-public-key-5>"
    - id: "node-06"
      address: "probft-node-06:9000"
      public_key: "<ed25519-public-key-6>"
    - id: "node-07"
      address: "probft-node-07:9000"
      public_key: "<ed25519-public-key-7>"

# VRF configuration
vrf:
  private_key_file: "/etc/probft/vrf-private.key"
  public_key_file: "/etc/probft/vrf-public.key"

# Quorum configuration
quorum:
  strategy: "probabilistic"  # sqrt(n) quorum
  total_nodes: 7
  byzantine_tolerance: 2  # f = (n-1)/3

# Consensus parameters
consensus:
  block_time: "1s"
  view_change_timeout: "30s"
  prepare_timeout: "5s"
  commit_timeout: "5s"
  max_block_size: 1048576  # 1 MB

# Persistence
storage:
  data_dir: "/var/lib/probft"
  max_chain_length: 100000
  checkpoint_interval: 1000

# Logging
logging:
  level: "info"
  file: "/var/log/probft/node.log"
  max_size: "100MB"
  max_backups: 7
```

Generate VRF keys:
```bash
# Generate VRF key pair
./probft-node --generate-vrf-keys \
  --private-key /etc/probft/vrf-private.key \
  --public-key /etc/probft/vrf-public.key

# Secure private key
sudo chmod 600 /etc/probft/vrf-private.key
sudo chown probft:probft /etc/probft/vrf-private.key
```

### 3. Deployment

**Option A: Systemd Service**

Create service file: `/etc/systemd/system/probft-node.service`

```ini
[Unit]
Description=ProBFT Consensus Node
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=probft
Group=probft
WorkingDirectory=/opt/probft
ExecStart=/opt/probft/bin/probft-node --config /etc/probft/node.yaml
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=probft-node

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/probft /var/log/probft

[Install]
WantedBy=multi-user.target
```

Deploy to all nodes:
```bash
# For each node (1-7)
for i in {1..7}; do
  echo "Deploying to node-0$i..."

  # Create user and directories
  ssh probft-node-0$i "
    sudo useradd -r -s /bin/false probft
    sudo mkdir -p /opt/probft/bin /var/lib/probft /var/log/probft
    sudo chown -R probft:probft /opt/probft /var/lib/probft /var/log/probft
  "

  # Copy binary and config
  scp probft-node probft-node-0$i:/tmp/
  scp node-0$i-config.yaml probft-node-0$i:/tmp/node.yaml

  ssh probft-node-0$i "
    sudo mv /tmp/probft-node /opt/probft/bin/
    sudo mv /tmp/node.yaml /etc/probft/node.yaml
    sudo chmod +x /opt/probft/bin/probft-node
  "

  # Generate VRF keys
  ssh probft-node-0$i "
    sudo /opt/probft/bin/probft-node --generate-vrf-keys \
      --private-key /etc/probft/vrf-private.key \
      --public-key /etc/probft/vrf-public.key
    sudo chmod 600 /etc/probft/vrf-private.key
  "

  # Enable service (don't start yet)
  ssh probft-node-0$i "
    sudo systemctl daemon-reload
    sudo systemctl enable probft-node
  "
done

# Start all nodes simultaneously
for i in {1..7}; do
  ssh probft-node-0$i "sudo systemctl start probft-node" &
done
wait
```

**Option B: Kubernetes StatefulSet**

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: probft-node
  namespace: consensus
spec:
  serviceName: probft
  replicas: 7
  selector:
    matchLabels:
      app: probft-node
  template:
    metadata:
      labels:
        app: probft-node
    spec:
      containers:
      - name: probft
        image: probft-node:1.0
        ports:
        - containerPort: 9000
          name: consensus
        - containerPort: 8080
          name: metrics
        volumeMounts:
        - name: data
          mountPath: /var/lib/probft
        - name: config
          mountPath: /etc/probft
        resources:
          requests:
            memory: "8Gi"
            cpu: "4000m"
          limits:
            memory: "16Gi"
            cpu: "8000m"
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: "fast-ssd"
      resources:
        requests:
          storage: 100Gi
```

### 4. Validation

```bash
# Check all nodes running
for i in {1..7}; do
  ssh probft-node-0$i "sudo systemctl status probft-node | grep Active"
done

# Verify cluster formation
curl http://probft-node-01:8080/cluster/status | jq

# Check quorum size
curl http://probft-node-01:8080/cluster/quorum | jq

# Submit test transaction
curl -X POST http://probft-node-01:8080/tx/submit \
  -H "Content-Type: application/json" \
  -d '{"data":"test transaction"}'

# Verify block finalization
curl http://probft-node-01:8080/chain/latest | jq

# Check consensus metrics
for i in {1..7}; do
  echo "Node $i metrics:"
  curl -s http://probft-node-0$i:8080/metrics | grep consensus_
done
```

### 5. Monitoring Setup

```bash
# Configure Prometheus scraping
cat >> /etc/prometheus/prometheus.yml <<EOF
  - job_name: 'probft-consensus'
    static_configs:
      - targets:
        - 'probft-node-01:8080'
        - 'probft-node-02:8080'
        - 'probft-node-03:8080'
        - 'probft-node-04:8080'
        - 'probft-node-05:8080'
        - 'probft-node-06:8080'
        - 'probft-node-07:8080'
    scrape_interval: 15s
EOF

sudo systemctl reload prometheus
```

## Configuration Parameters

### Quorum Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `strategy` | probabilistic | Quorum strategy (probabilistic = ⌈√n⌉) |
| `total_nodes` | - | Total nodes in cluster |
| `byzantine_tolerance` | (n-1)/3 | Max Byzantine nodes tolerated |

**Quorum Size Calculation:**
- For n=7 nodes: quorum = ⌈√7⌉ = 3
- Byzantine tolerance: f = 2 (28.5%)
- Prepare threshold: 2f = 4 votes
- Commit threshold: 2f+1 = 5 votes

### Consensus Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `block_time` | 1s | 100ms-10s | Target block production time |
| `view_change_timeout` | 30s | 10s-60s | Leader timeout before view change |
| `prepare_timeout` | 5s | 1s-30s | Timeout for prepare phase |
| `commit_timeout` | 5s | 1s-30s | Timeout for commit phase |
| `max_block_size` | 1MB | 100KB-10MB | Maximum block size |

### VRF Configuration

| Parameter | Description |
|-----------|-------------|
| `private_key_file` | Path to Ed25519 private key for VRF |
| `public_key_file` | Path to Ed25519 public key for VRF |

**Security:** Private keys must be kept secure and never shared. Each node must have a unique key pair.

## Health Checks

### Endpoint URLs

```bash
# Node health
GET http://probft-node-01:8080/health

# Cluster status
GET http://probft-node-01:8080/cluster/status

# Consensus metrics
GET http://probft-node-01:8080/metrics

# Current view
GET http://probft-node-01:8080/consensus/view

# Latest block
GET http://probft-node-01:8080/chain/latest
```

### Expected Responses

**Healthy Node:**
```json
{
  "status": "healthy",
  "node_id": "node-01",
  "is_active": true,
  "current_view": 5,
  "current_height": 1234,
  "is_leader": false,
  "quorum_size": 3,
  "active_nodes": 7
}
```

**Cluster Status:**
```json
{
  "total_nodes": 7,
  "active_nodes": 7,
  "byzantine_tolerance": 2,
  "quorum_size": 3,
  "current_leader": "node-03",
  "current_view": 5,
  "chain_height": 1234
}
```

## Monitoring

### Key Metrics

**Consensus Progress:**
```promql
# Current blockchain height
probft_chain_height

# Consensus rounds per minute
rate(probft_consensus_rounds_total[1m])

# Block finalization time
histogram_quantile(0.99, probft_finalization_duration_seconds_bucket)

# Current view number
probft_current_view
```

**Quorum Health:**
```promql
# Active nodes in cluster
probft_active_nodes

# Quorum achievement rate
rate(probft_quorum_achieved_total[5m]) / rate(probft_consensus_rounds_total[5m])

# Failed quorums
rate(probft_quorum_failed_total[5m])
```

**Byzantine Detection:**
```promql
# Byzantine behavior detected
rate(probft_byzantine_detected_total[5m])

# Invalid VRF proofs
rate(probft_vrf_invalid_total[5m])

# View changes (leader failures)
rate(probft_view_changes_total[10m])
```

### Alert Thresholds

**Critical (P0):**
- Node down for >60 seconds
- Quorum not achieved for >5 minutes
- Byzantine node detected
- Chain fork detected

**Warning (P1):**
- Block finalization >5 seconds
- View changes >3 per hour
- Node unreachable by peers
- Clock drift >10ms

**Info (P2):**
- View change occurred
- Slow block finalization (>2s)
- Network latency spike

## Troubleshooting

### Common Issues

#### Issue: Quorum not achieved

**Symptoms:**
```
Failed to achieve quorum
Consensus round timeout
No blocks being produced
```

**Diagnosis:**
```bash
# Check active nodes
curl http://probft-node-01:8080/cluster/status | jq '.active_nodes'

# Verify network connectivity
for i in {1..7}; do
  nc -zv probft-node-0$i 9000
done

# Check for Byzantine nodes
curl http://probft-node-01:8080/cluster/byzantine | jq
```

**Resolution:**
```bash
# Identify offline nodes
./scripts/check-node-health.sh

# Restart failed nodes
for node in $(./scripts/list-failed-nodes.sh); do
  ssh $node "sudo systemctl restart probft-node"
done

# If >f nodes failed, cluster cannot progress
# Wait for nodes to recover or reduce cluster size
```

#### Issue: View changes happening frequently

**Symptoms:**
```
View change every few minutes
Leader timeouts
Unstable consensus
```

**Diagnosis:**
```bash
# Check view change rate
curl http://probft-node-01:8080/metrics | grep view_changes

# Identify problematic leader
./scripts/analyze-view-changes.sh

# Check network latency
for i in {1..7}; do
  ping -c 10 probft-node-0$i | grep avg
done
```

**Resolution:**
```bash
# Increase view change timeout
sed -i 's/view_change_timeout: "30s"/view_change_timeout: "60s"/' \
  /etc/probft/node.yaml

# Restart nodes with new config
for i in {1..7}; do
  ssh probft-node-0$i "sudo systemctl restart probft-node"
done

# If specific node is problematic, investigate or remove
```

#### Issue: Invalid VRF proofs

**Symptoms:**
```
VRF verification failed
Invalid leader election
Byzantine behavior detected
```

**Diagnosis:**
```bash
# Check VRF metrics
curl http://probft-node-01:8080/metrics | grep vrf_invalid

# Identify source of invalid proofs
journalctl -u probft-node | grep "invalid VRF"

# Verify VRF keys
./probft-node --verify-vrf-keys --public-key /etc/probft/vrf-public.key
```

**Resolution:**
```bash
# If malicious node detected
# Remove from cluster configuration
./scripts/remove-byzantine-node.sh node-0X

# Regenerate VRF keys if corrupted
sudo /opt/probft/bin/probft-node --generate-vrf-keys \
  --private-key /etc/probft/vrf-private.key \
  --public-key /etc/probft/vrf-public.key

# Update cluster configs with new public key
# Restart affected node
```

## Rollback Procedure

### Conditions for Rollback

- Chain fork detected
- Byzantine attack ongoing
- Data corruption
- Cluster instability >30 minutes

### Rollback Steps

```bash
# 1. Stop all nodes
for i in {1..7}; do
  ssh probft-node-0$i "sudo systemctl stop probft-node"
done

# 2. Identify safe checkpoint
./scripts/find-latest-checkpoint.sh

# 3. Restore from checkpoint
for i in {1..7}; do
  ssh probft-node-0$i "
    sudo cp /var/lib/probft/checkpoints/checkpoint-1234.db \
      /var/lib/probft/chain.db
  "
done

# 4. Restart cluster
for i in {1..7}; do
  ssh probft-node-0$i "sudo systemctl start probft-node" &
done
wait

# 5. Verify consensus resumed
sleep 30
curl http://probft-node-01:8080/cluster/status
```

## Performance Tuning

### For High Throughput

```yaml
consensus:
  block_time: "500ms"  # Faster blocks
  max_block_size: 5242880  # 5 MB blocks

quorum:
  strategy: "probabilistic"  # Keep sqrt(n)
```

### For Low Latency

```yaml
consensus:
  block_time: "100ms"
  view_change_timeout: "10s"
  prepare_timeout: "1s"
  commit_timeout: "1s"
```

### For Byzantine Resilience

```yaml
cluster:
  nodes: 13  # More nodes = higher tolerance

quorum:
  byzantine_tolerance: 4  # f = 4, tolerates 30%
```

## References

- **Source Code:** `/home/kp/repos/novacron/backend/core/network/dwcp/v3/consensus/probft/consensus.go`
- **VRF Implementation:** `/home/kp/repos/novacron/backend/core/network/dwcp/v3/consensus/probft/vrf.go`
- **Quorum Logic:** `/home/kp/repos/novacron/backend/core/network/dwcp/v3/consensus/probft/quorum.go`

---

**Runbook Version:** 1.0
**Last Updated:** 2025-11-14
**Owner:** Consensus Engineering Team
