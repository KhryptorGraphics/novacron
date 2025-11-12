# DWCP v3.0 Operations Team Training Manual
## Internet-Scale Distributed Level 2 Hypervisor

**Version:** 3.0.0
**Last Updated:** 2025-01-10
**Training Duration:** 3-5 Days
**Target Audience:** Operations Engineers, SREs, DevOps Teams
**Prerequisites:** Basic Linux, Networking, Virtualization Knowledge

---

## Table of Contents

1. [Training Overview](#training-overview)
2. [DWCP v3 Architecture Deep Dive](#dwcp-v3-architecture-deep-dive)
3. [Deployment Procedures](#deployment-procedures)
4. [Monitoring and Alerting](#monitoring-and-alerting)
5. [Troubleshooting Common Issues](#troubleshooting-common-issues)
6. [Incident Response Procedures](#incident-response-procedures)
7. [Rollback Procedures](#rollback-procedures)
8. [Performance Optimization](#performance-optimization)
9. [Security Operations](#security-operations)
10. [Hands-On Exercises](#hands-on-exercises)
11. [Certification Assessment](#certification-assessment)

---

## 1. Training Overview

### 1.1 Learning Objectives

By the end of this training, you will be able to:

- **Understand** the DWCP v3.0 architecture and how it differs from v2.0
- **Deploy** DWCP v3.0 in production environments (1,000+ nodes)
- **Monitor** system health using Grafana, Prometheus, and distributed tracing
- **Troubleshoot** performance issues, network failures, and Byzantine attacks
- **Execute** VM migrations in both datacenter and internet modes
- **Respond** to incidents following established procedures
- **Perform** emergency rollbacks with minimal downtime
- **Optimize** system performance for specific workloads

### 1.2 Training Schedule

#### Day 1: Architecture and Fundamentals
- **Morning:** DWCP v3 architecture overview (4 hours)
- **Afternoon:** Deployment procedures and infrastructure (4 hours)
- **Evening:** Lab setup and verification (1 hour)

#### Day 2: Monitoring and Operations
- **Morning:** Monitoring systems and dashboards (3 hours)
- **Afternoon:** Troubleshooting procedures (3 hours)
- **Evening:** Labs 1-2 (2 hours)

#### Day 3: VM Migration and Performance
- **Morning:** VM migration deep dive (3 hours)
- **Afternoon:** Performance optimization (3 hours)
- **Evening:** Labs 3-4 (2 hours)

#### Day 4: Incident Response and Security
- **Morning:** Incident response procedures (3 hours)
- **Afternoon:** Security operations and Byzantine tolerance (3 hours)
- **Evening:** Labs 5-6 (2 hours)

#### Day 5: Advanced Topics and Certification
- **Morning:** Advanced troubleshooting scenarios (3 hours)
- **Afternoon:** Labs 7-8 (3 hours)
- **Evening:** Certification exam (2 hours)

### 1.3 Training Materials

**Required Reading (Before Day 1):**
- `docs/DWCP-V3-QUICK-START.md` - Overview and quick start
- `docs/deployment/DWCP_V3_GO_LIVE_CHECKLIST.md` - Production checklist
- `docs/research/DWCP-INTERNET-SCALE-DISTRIBUTED-HYPERVISOR.md` - Architecture details

**Reference Materials:**
- `docs/deployment/DWCP_V3_API_REFERENCE.md` - Complete API documentation
- `docs/deployment/DWCP_V3_OPERATIONS_GUIDE.md` - Operations guide
- `docs/deployment/DWCP_V3_PERFORMANCE_TUNING.md` - Performance tuning guide

**Lab Materials:**
- `docs/training/labs/` - All hands-on lab exercises
- `scripts/deployment/` - Deployment automation scripts
- `monitoring/grafana/dashboards/` - Grafana dashboard templates

---

## 2. DWCP v3 Architecture Deep Dive

### 2.1 What is DWCP v3.0?

**DWCP (Distributed Wide-Area Computation Protocol)** is a Layer 2 distributed hypervisor that enables VM migration and orchestration across commodity internet computers.

**Key Innovation:** Turn thousands of unreliable internet computers into a unified, Byzantine-tolerant virtualization layer.

#### 2.1.1 DWCP v2.0 vs v3.0 Comparison

| Aspect | v2.0 (Datacenter) | v3.0 (Internet) |
|--------|-------------------|-----------------|
| **Network** | RDMA, NVLink (1+ Pbps) | TCP/IP over internet (100-900 Mbps) |
| **Latency** | <1ms (same datacenter) | 50-500ms (global WAN) |
| **Hardware** | Specialized (DPUs, SmartNICs) | Commodity (x86/ARM) |
| **Nodes** | Reliable (99.999% uptime) | Unreliable (95-99% uptime) |
| **Trust Model** | Trusted datacenter | Byzantine tolerant |
| **Scale** | 100-1,000 nodes | 1,000-100,000 nodes |
| **Cost** | $50K-$500K per node | $500-$5K per node |

**Why v3.0?**
- Scale to **millions of commodity computers** (BOINC proof: 4M+ nodes)
- **10-100x cost reduction** vs datacenter deployment
- **Global distribution** with Byzantine fault tolerance
- **Heterogeneous hardware** support (x86, ARM, mixed specs)

### 2.2 Six Core Components

DWCP v3.0 consists of 6 tightly integrated components:

#### 2.2.1 AMST v3 (Adaptive Multi-Stream Transport)

**Purpose:** Maximize bandwidth utilization over unreliable internet links

**Key Features:**
- **Multi-stream TCP:** 4-16 parallel streams per migration
- **Adaptive stream scaling:** Adjusts based on network conditions
- **Congestion control:** BBR v2 with custom modifications
- **Performance:** 100-900 Mbps effective throughput on gigabit links

**How it Works:**
```
1. Establish multiple TCP connections (default: 8 streams)
2. Probe each stream's capacity using packet pacing
3. Distribute data across streams using weighted round-robin
4. Dynamically add/remove streams based on congestion
5. Reorder packets at receiver using sequence numbers
```

**Configuration:**
```yaml
amst:
  streams:
    min: 4
    max: 16
    default: 8
  congestion_control: bbr2
  reordering_buffer: 16MB
  timeout: 30s
```

**Operations Tips:**
- Monitor stream health: `dwcp-cli amst status --node <node-id>`
- Check for stream imbalance: Look for >20% variance in throughput
- Adjust stream count for WAN: 4-8 streams for <50ms RTT, 8-16 for >100ms RTT

#### 2.2.2 HDE v3 (Hierarchical Data Encoding)

**Purpose:** Reduce bandwidth consumption via compression and deduplication

**Key Features:**
- **Zstandard compression:** 50-70% reduction, adaptive level (1-19)
- **Content-aware deduplication:** 20-40% reduction, 4KB fixed blocks
- **Combined savings:** 70-85% bandwidth reduction
- **Streaming:** Real-time compression during migration

**How it Works:**
```
1. Split VM memory into 4KB blocks
2. Compute SHA-256 hash for each block
3. Check deduplication database (Redis/RocksDB)
4. Compress unique blocks with Zstandard (level 3-5)
5. Transmit compressed unique blocks only
6. Receiver decompresses and reconstructs memory
```

**Configuration:**
```yaml
hde:
  compression:
    algorithm: zstd
    level: 5  # Balance speed/ratio
    dictionary: auto  # Train from first 64MB
  deduplication:
    block_size: 4096
    database: rocksdb
    cache_size: 2GB
```

**Operations Tips:**
- Monitor compression ratio: Target 50-70%, alert if <30%
- Deduplication hit rate: Target 20-40%, depends on workload
- CPU usage: Zstd level 5 uses ~10-20% CPU per migration
- Memory: Dedup cache uses ~2GB per 100GB migrated

#### 2.2.3 PBA v3 (Predictive Bandwidth Allocator)

**Purpose:** Forecast network conditions and optimize scheduling

**Key Features:**
- **LSTM neural network:** Trained on historical bandwidth data
- **70%+ prediction accuracy:** 30-second to 5-minute horizon
- **Adaptive scheduling:** Delay migrations during congestion
- **Multi-metric:** Predicts bandwidth, latency, packet loss

**How it Works:**
```
1. Collect network metrics every 5 seconds (bandwidth, RTT, loss)
2. Feed last 60 samples (5 minutes) into LSTM model
3. Generate prediction for next 30-300 seconds
4. Scheduler uses predictions to:
   - Defer migrations if bandwidth drop predicted
   - Allocate bandwidth quotas per migration
   - Trigger preemptive VM evacuation
5. Retrain model weekly with new data
```

**Configuration:**
```yaml
pba:
  model:
    type: lstm
    layers: [128, 64, 32]
    sequence_length: 60
    prediction_horizon: 30s
  training:
    interval: 7d
    min_samples: 10000
  accuracy_threshold: 0.70
```

**Operations Tips:**
- Monitor prediction accuracy: `dwcp-cli pba metrics`
- Retrain if accuracy drops below 70%
- Check for training data staleness (>7 days)
- CPU usage: ~5-10% during prediction, ~50% during training

#### 2.2.4 ASS v3 (Asynchronous State Synchronizer)

**Purpose:** Maintain eventually consistent state across unreliable nodes

**Key Features:**
- **CRDT-based:** Conflict-free replicated data types
- **5-30 second consistency:** Tunable staleness bounds
- **Partition tolerance:** Continue operating during network splits
- **State types:** VM metadata, placement decisions, health status

**How it Works:**
```
1. Each node maintains local CRDT replica
2. State changes applied locally first (instant)
3. Changes propagated via gossip protocol
4. CRDTs automatically resolve conflicts
5. Global consistency achieved within 5-30 seconds
```

**Configuration:**
```yaml
ass:
  crdt:
    type: lww_map  # Last-Write-Wins Map
    replication_factor: 3
  gossip:
    fanout: 6
    interval: 1s
  consistency:
    max_staleness: 10s
```

**Operations Tips:**
- Monitor replication lag: Target <10s, alert if >30s
- Check gossip health: Ensure each node has 6+ peers
- Partition detection: Alert if cluster splits >60s
- Memory: CRDTs use ~100MB per 10,000 VMs

#### 2.2.5 ITP v3 (Intelligent Task Placement)

**Purpose:** Optimize VM placement on heterogeneous hardware

**Key Features:**
- **Multi-constraint optimization:** CPU, RAM, GPU, network
- **Bin packing:** Minimize nodes while meeting SLAs
- **Affinity/anti-affinity:** User-defined placement rules
- **80%+ resource utilization:** vs 40-60% with naive placement

**How it Works:**
```
1. Collect node resources: CPU cores, RAM, GPU, bandwidth
2. Model as bin packing problem with constraints:
   - VM requirements (CPU, RAM, GPU)
   - Affinity rules (colocation/separation)
   - SLA constraints (latency, bandwidth)
3. Solve using genetic algorithm (95% optimal in <10s)
4. Execute migrations to achieve target placement
```

**Configuration:**
```yaml
itp:
  algorithm: genetic
  optimization_target: resource_utilization
  constraints:
    max_vms_per_node: 100
    min_free_memory: 10%
  rebalance_interval: 5m
```

**Operations Tips:**
- Monitor resource utilization: Target 80-85%
- Check for placement violations: Alert if SLAs missed
- Rebalancing overhead: <5% migrations per hour
- CPU usage: Solver uses ~20% CPU for 1,000 nodes

#### 2.2.6 ACP v3 (Adaptive Consensus Protocol)

**Purpose:** Achieve global consensus despite Byzantine faults

**Key Features:**
- **Hybrid protocol:** Raft (datacenter) + Gossip (internet) + PBFT (Byzantine)
- **1-5 second latency:** Global consensus over WAN
- **33% Byzantine tolerance:** System operates with 1/3 malicious nodes
- **Reputation system:** Blacklist malicious nodes automatically

**How it Works:**
```
1. Partition cluster into regions (based on latency)
2. Within region: Use Raft for fast consensus (<100ms)
3. Cross-region: Use Gossip for eventual consistency (1-5s)
4. Byzantine protection: PBFT voting with 3f+1 replicas
5. Reputation: Track node behavior, quarantine misbehaving nodes
```

**Configuration:**
```yaml
acp:
  mode: hybrid  # raft | gossip | pbft | hybrid
  raft:
    election_timeout: 300ms
    heartbeat: 100ms
  pbft:
    checkpoint_interval: 100
    view_change_timeout: 10s
  reputation:
    decay_rate: 0.01/day
    quarantine_threshold: 0.3
```

**Operations Tips:**
- Monitor consensus latency: Target <5s, alert if >10s
- Check leader election: Should be <500ms
- Reputation scores: Alert if >10% nodes below 0.5
- Byzantine detection: Investigate if >5% nodes quarantined

### 2.3 Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     DWCP v3 Architecture                         │
└─────────────────────────────────────────────────────────────────┘

        VM Migration Request
                 ↓
    ┌────────────────────────┐
    │   ITP v3 (Placement)   │  ← Optimization algorithm
    │  Find optimal target   │
    └────────────────────────┘
                 ↓
    ┌────────────────────────┐
    │   PBA v3 (Prediction)  │  ← LSTM forecasting
    │  Check network health  │
    └────────────────────────┘
                 ↓
    ┌────────────────────────┐
    │   ACP v3 (Consensus)   │  ← Byzantine-tolerant voting
    │ Get approval from quorum│
    └────────────────────────┘
                 ↓
    ┌────────────────────────┐
    │   HDE v3 (Encoding)    │  ← Compression + dedup
    │  Compress VM memory     │
    └────────────────────────┘
                 ↓
    ┌────────────────────────┐
    │   AMST v3 (Transport)  │  ← Multi-stream TCP
    │  Transmit over WAN      │
    └────────────────────────┘
                 ↓
    ┌────────────────────────┐
    │   ASS v3 (State Sync)  │  ← CRDT replication
    │ Update global state     │
    └────────────────────────┘
                 ↓
        VM Running on Target
```

### 2.4 System Architecture

#### 2.4.1 Node Types

**Controller Nodes (3-5 nodes):**
- Run ACP consensus (Raft leaders)
- Host ITP placement solver
- Maintain global VM registry
- Coordinate migrations

**Worker Nodes (1,000+ nodes):**
- Run hypervisor (KVM/QEMU/Xen)
- Host VM instances
- Execute migrations
- Report metrics

**Edge Nodes (Optional):**
- Regional aggregators
- Reduce WAN traffic
- Cache frequently accessed data

#### 2.4.2 Network Topology

```
Internet (Global WAN)
        ↓
┌────────────────────────────────────┐
│   Controller Cluster (3-5 nodes)   │
│   - Raft consensus                 │
│   - ITP placement                  │
│   - Global registry                │
└────────────────────────────────────┘
        ↓
        ├── Region 1 (US-East)
        │   ├── Worker 1-500
        │   └── Edge Node
        │
        ├── Region 2 (EU-West)
        │   ├── Worker 501-1000
        │   └── Edge Node
        │
        └── Region 3 (AP-Southeast)
            ├── Worker 1001-1500
            └── Edge Node
```

#### 2.4.3 Data Flow

**VM Migration Flow:**
1. Controller receives migration request
2. ITP selects optimal target node
3. PBA predicts network conditions
4. ACP achieves consensus (3-5 nodes vote)
5. Source node:
   - Pauses VM (pre-copy phase)
   - HDE compresses/dedups memory
   - AMST transmits data
6. Target node:
   - Receives and decompresses
   - Recreates VM state
   - Starts VM
7. ASS propagates state update

**Typical Latencies:**
- Placement decision: 100-500ms
- Consensus: 500ms-5s
- Memory transfer (2GB): 20-80s
- VM downtime: 5-20s

### 2.5 Performance Characteristics

#### 2.5.1 Throughput

| VM Size | Internet Mode | Datacenter Mode |
|---------|---------------|-----------------|
| 2 GB    | 45-90 sec     | 10-20 sec       |
| 8 GB    | 3-6 min       | 30-60 sec       |
| 32 GB   | 15-30 min     | 2-5 min         |

#### 2.5.2 Resource Usage

**Per Migration:**
- CPU: 30-50% (compression + transport)
- Memory: 2-4GB (buffers + dedup cache)
- Network: 100-900 Mbps (depends on link quality)

**Controller Nodes:**
- CPU: 10-20% (consensus + placement)
- Memory: 4-8GB (global state + CRDT)
- Network: <10 Mbps (control plane only)

#### 2.5.3 Scalability Limits

| Metric | Single Node | 1,000 Nodes | 10,000 Nodes | 100,000 Nodes |
|--------|-------------|-------------|--------------|---------------|
| **VMs** | 100 | 100,000 | 1,000,000 | 10,000,000 |
| **Migrations/sec** | 1 | 100 | 1,000 | 10,000 |
| **Consensus latency** | N/A | 500ms | 2s | 5s |
| **State sync latency** | N/A | 5s | 15s | 30s |

---

## 3. Deployment Procedures

### 3.1 Infrastructure Requirements

#### 3.1.1 Controller Nodes

**Minimum Specs (per node):**
- CPU: 8 cores (16 vCPUs)
- RAM: 32 GB
- Disk: 500 GB SSD (IOPS >10K)
- Network: 1 Gbps (10 Gbps recommended)
- OS: Ubuntu 22.04 LTS / Rocky Linux 9

**Deployment:**
- Deploy 3-5 nodes (odd number for Raft)
- Use dedicated network segment
- Enable NTP for time synchronization
- Configure firewalls (allow ports 8080-8090)

#### 3.1.2 Worker Nodes

**Minimum Specs (per node):**
- CPU: 16 cores (32 vCPUs)
- RAM: 64 GB (128 GB recommended)
- Disk: 1 TB NVMe SSD
- Network: 1 Gbps (gigabit internet)
- GPU (optional): NVIDIA A100 / AMD MI250
- OS: Ubuntu 22.04 LTS / Rocky Linux 9

**Hypervisor:**
- KVM with QEMU 7.0+
- Xen 4.17+ (alternative)
- libvirt 9.0+

#### 3.1.3 Network Requirements

**Bandwidth:**
- Controller ↔ Controller: 1 Gbps minimum
- Controller ↔ Worker: 100 Mbps minimum
- Worker ↔ Worker: 100 Mbps minimum (1 Gbps recommended)

**Latency:**
- Controller ↔ Controller: <50ms (same region preferred)
- Controller ↔ Worker: <200ms
- Worker ↔ Worker: <500ms (internet WAN)

**Ports:**
- 8080: DWCP API (REST)
- 8081: DWCP gRPC
- 8082: Raft consensus
- 8083: Gossip protocol
- 8084: Metrics (Prometheus)
- 8085-8100: AMST data streams

### 3.2 Installation Steps

#### 3.2.1 Install DWCP v3 Binary

```bash
# Download latest release
wget https://github.com/your-org/dwcp-v3/releases/download/v3.0.0/dwcp-v3-linux-amd64.tar.gz

# Extract
tar -xzf dwcp-v3-linux-amd64.tar.gz -C /opt/dwcp

# Add to PATH
export PATH="/opt/dwcp/bin:$PATH"
echo 'export PATH="/opt/dwcp/bin:$PATH"' >> ~/.bashrc

# Verify installation
dwcp-server --version
# Expected: DWCP v3.0.0
```

#### 3.2.2 Configure Controller Node

```bash
# Generate configuration
dwcp-server init --node-type controller \
  --cluster-id prod-cluster-1 \
  --output /etc/dwcp/controller.yaml

# Edit configuration
nano /etc/dwcp/controller.yaml
```

**Sample Configuration (`/etc/dwcp/controller.yaml`):**

```yaml
cluster:
  id: prod-cluster-1
  name: Production Cluster 1

node:
  id: controller-1
  type: controller
  region: us-east-1
  datacenter: dc1

api:
  rest:
    bind: 0.0.0.0:8080
    tls:
      enabled: true
      cert: /etc/dwcp/tls/server.crt
      key: /etc/dwcp/tls/server.key
  grpc:
    bind: 0.0.0.0:8081

consensus:
  protocol: raft
  peers:
    - controller-1.dc1.example.com:8082
    - controller-2.dc1.example.com:8082
    - controller-3.dc1.example.com:8082
  data_dir: /var/lib/dwcp/raft

placement:
  algorithm: genetic
  optimization_target: resource_utilization

monitoring:
  prometheus:
    enabled: true
    bind: 0.0.0.0:8084
  tracing:
    enabled: true
    backend: jaeger
    endpoint: jaeger.example.com:14268
```

#### 3.2.3 Configure Worker Node

```bash
# Generate configuration
dwcp-server init --node-type worker \
  --controller controller-1.dc1.example.com:8080 \
  --output /etc/dwcp/worker.yaml

# Edit configuration
nano /etc/dwcp/worker.yaml
```

**Sample Configuration (`/etc/dwcp/worker.yaml`):**

```yaml
cluster:
  id: prod-cluster-1
  controller: controller-1.dc1.example.com:8080

node:
  id: worker-001
  type: worker
  region: us-east-1
  datacenter: dc1

  resources:
    cpu: 32  # vCPUs
    memory: 128  # GB
    disk: 1024  # GB
    gpu:
      enabled: true
      type: nvidia-a100
      count: 1

hypervisor:
  type: kvm
  qemu_path: /usr/bin/qemu-system-x86_64
  libvirt_uri: qemu:///system

transport:
  amst:
    streams:
      min: 4
      max: 16
      default: 8
    congestion_control: bbr2

encoding:
  hde:
    compression:
      algorithm: zstd
      level: 5
    deduplication:
      enabled: true
      block_size: 4096

monitoring:
  prometheus:
    enabled: true
    bind: 0.0.0.0:8084
```

#### 3.2.4 Start Services

**Controller Node:**
```bash
# Start DWCP controller
sudo systemctl start dwcp-controller

# Enable on boot
sudo systemctl enable dwcp-controller

# Check status
sudo systemctl status dwcp-controller

# View logs
sudo journalctl -u dwcp-controller -f
```

**Worker Node:**
```bash
# Start DWCP worker
sudo systemctl start dwcp-worker

# Enable on boot
sudo systemctl enable dwcp-worker

# Check status
sudo systemctl status dwcp-worker

# View logs
sudo journalctl -u dwcp-worker -f
```

### 3.3 Cluster Initialization

#### 3.3.1 Bootstrap Controller Cluster

```bash
# On controller-1
dwcp-cli cluster bootstrap \
  --cluster-id prod-cluster-1 \
  --controllers controller-1,controller-2,controller-3

# Verify cluster health
dwcp-cli cluster status
# Expected:
# Cluster: prod-cluster-1
# Controllers: 3 (all healthy)
# Leader: controller-1
# Consensus: raft (operational)
```

#### 3.3.2 Join Worker Nodes

```bash
# On each worker node
dwcp-cli node join \
  --controller controller-1.dc1.example.com:8080 \
  --node-id worker-001 \
  --region us-east-1

# Verify node joined
dwcp-cli node list
# Expected: worker-001 listed with status "ready"
```

#### 3.3.3 Verify Cluster Health

```bash
# Check all components
dwcp-cli cluster health

# Expected output:
# ✓ Controllers: 3/3 healthy
# ✓ Workers: 1000/1000 ready
# ✓ Consensus: operational (latency 50ms)
# ✓ State sync: healthy (lag 5s)
# ✓ Network: all links up
```

### 3.4 Configuration Management

#### 3.4.1 Using Terraform

See `docs/cicd/DWCP_V3_IAC_GUIDE.md` for complete Terraform configuration.

**Quick Example:**
```hcl
# main.tf
module "dwcp_cluster" {
  source = "./modules/dwcp-cluster"

  cluster_id = "prod-cluster-1"
  controllers = {
    count = 3
    instance_type = "c5.2xlarge"
    region = "us-east-1"
  }
  workers = {
    count = 1000
    instance_type = "c5.4xlarge"
    regions = ["us-east-1", "eu-west-1", "ap-southeast-1"]
  }
}

# Deploy
terraform init
terraform plan
terraform apply
```

#### 3.4.2 Using Ansible

```bash
# Install Ansible playbooks
git clone https://github.com/your-org/dwcp-v3-ansible.git
cd dwcp-v3-ansible

# Configure inventory
nano inventory/production.yaml

# Deploy controllers
ansible-playbook -i inventory/production.yaml playbooks/deploy-controllers.yaml

# Deploy workers
ansible-playbook -i inventory/production.yaml playbooks/deploy-workers.yaml

# Verify deployment
ansible-playbook -i inventory/production.yaml playbooks/verify-cluster.yaml
```

---

## 4. Monitoring and Alerting

### 4.1 Monitoring Stack

DWCP v3 uses the following monitoring tools:

1. **Prometheus** - Metrics collection
2. **Grafana** - Dashboards and visualization
3. **Jaeger** - Distributed tracing
4. **OpenTelemetry** - Instrumentation
5. **Loki** - Log aggregation
6. **AlertManager** - Alert routing

### 4.2 Key Metrics

#### 4.2.1 System Health Metrics

| Metric | Description | Normal Range | Alert Threshold |
|--------|-------------|--------------|-----------------|
| `dwcp_cluster_controllers_healthy` | Healthy controller count | 3-5 | <3 |
| `dwcp_cluster_workers_ready` | Ready worker count | 1000+ | <95% |
| `dwcp_consensus_latency_seconds` | Consensus latency | 0.5-5s | >10s |
| `dwcp_state_sync_lag_seconds` | State sync lag | 5-30s | >60s |

#### 4.2.2 VM Migration Metrics

| Metric | Description | Normal Range | Alert Threshold |
|--------|-------------|--------------|-----------------|
| `dwcp_migration_duration_seconds` | Migration time | 45-90s (2GB) | >180s |
| `dwcp_migration_downtime_seconds` | VM downtime | 5-20s | >30s |
| `dwcp_migration_success_rate` | Success rate | >95% | <90% |
| `dwcp_migration_bandwidth_mbps` | Effective bandwidth | 100-900 Mbps | <50 Mbps |

#### 4.2.3 Network Metrics

| Metric | Description | Normal Range | Alert Threshold |
|--------|-------------|--------------|-----------------|
| `dwcp_amst_streams_active` | Active AMST streams | 4-16 | <2 |
| `dwcp_amst_throughput_mbps` | AMST throughput | 100-900 Mbps | <50 Mbps |
| `dwcp_hde_compression_ratio` | Compression ratio | 0.5-0.7 | <0.3 |
| `dwcp_hde_dedup_hit_rate` | Dedup hit rate | 0.2-0.4 | <0.1 |

#### 4.2.4 Consensus Metrics

| Metric | Description | Normal Range | Alert Threshold |
|--------|-------------|--------------|-----------------|
| `dwcp_raft_leader_elections_total` | Leader elections | <1/hour | >5/hour |
| `dwcp_raft_commit_latency_seconds` | Commit latency | 10-100ms | >500ms |
| `dwcp_pbft_view_changes_total` | PBFT view changes | <1/hour | >10/hour |
| `dwcp_reputation_quarantined_nodes` | Quarantined nodes | 0-5% | >10% |

### 4.3 Grafana Dashboards

#### 4.3.1 Dashboard: Cluster Overview

**File:** `monitoring/grafana/dashboards/cluster-overview.json`

**Panels:**
1. Cluster health status (green/yellow/red)
2. Controller status (3/3 healthy)
3. Worker node map (geographic distribution)
4. VM count and distribution
5. Active migrations (in-progress count)
6. Network bandwidth (total cluster throughput)

**Key Queries:**
```promql
# Cluster health
up{job="dwcp-controller"} == 1

# Worker count
count(up{job="dwcp-worker"} == 1)

# VM count
sum(dwcp_vm_count_total)

# Active migrations
sum(dwcp_migration_active_total)
```

#### 4.3.2 Dashboard: VM Migration

**File:** `monitoring/grafana/dashboards/vm-migration.json`

**Panels:**
1. Migration rate (migrations/sec)
2. Success rate (% successful)
3. Average duration (seconds)
4. Average downtime (seconds)
5. Bandwidth utilization (Mbps)
6. Compression ratio (%)
7. Top 10 slowest migrations (table)

**Key Queries:**
```promql
# Migration rate
rate(dwcp_migration_completed_total[5m])

# Success rate
rate(dwcp_migration_success_total[5m]) / rate(dwcp_migration_completed_total[5m])

# Average duration
avg(dwcp_migration_duration_seconds)

# Compression ratio
avg(dwcp_hde_compression_ratio)
```

#### 4.3.3 Dashboard: Network Performance

**File:** `monitoring/grafana/dashboards/network-performance.json`

**Panels:**
1. AMST throughput (Mbps per node)
2. Stream health (active streams per migration)
3. Compression ratio over time
4. Deduplication hit rate
5. Network errors (packet loss, retransmissions)
6. Bandwidth prediction accuracy (PBA)

**Key Queries:**
```promql
# AMST throughput
sum(rate(dwcp_amst_bytes_transmitted_total[1m])) * 8 / 1000000

# Stream count
avg(dwcp_amst_streams_active)

# Prediction accuracy
dwcp_pba_prediction_accuracy
```

#### 4.3.4 Dashboard: Consensus and State

**File:** `monitoring/grafana/dashboards/consensus-state.json`

**Panels:**
1. Raft leader status (current leader, term)
2. Consensus latency (p50, p95, p99)
3. State sync lag (per node)
4. CRDT conflict rate
5. Gossip fanout (messages sent/received)
6. Reputation scores (histogram)

**Key Queries:**
```promql
# Consensus latency (p95)
histogram_quantile(0.95, dwcp_consensus_latency_seconds_bucket)

# State sync lag (max)
max(dwcp_state_sync_lag_seconds)

# Reputation scores
dwcp_reputation_score
```

### 4.4 Alerting Rules

#### 4.4.1 Critical Alerts (P0)

**File:** `monitoring/prometheus/alerts/critical.yaml`

```yaml
groups:
  - name: dwcp_critical
    interval: 30s
    rules:
      # Controller cluster failure
      - alert: ControllerClusterDegraded
        expr: count(up{job="dwcp-controller"} == 1) < 3
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Controller cluster degraded"
          description: "Only {{ $value }} controllers healthy (need 3+)"

      # Consensus failure
      - alert: ConsensusFailure
        expr: dwcp_consensus_latency_seconds > 10
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Consensus latency high"
          description: "Consensus taking {{ $value }}s (threshold: 10s)"

      # Mass worker failure
      - alert: MassWorkerFailure
        expr: (count(up{job="dwcp-worker"} == 1) / count(up{job="dwcp-worker"})) < 0.90
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Mass worker failure detected"
          description: "Only {{ $value }}% workers healthy (threshold: 90%)"
```

#### 4.4.2 High Alerts (P1)

```yaml
groups:
  - name: dwcp_high
    interval: 1m
    rules:
      # High migration failure rate
      - alert: HighMigrationFailureRate
        expr: (rate(dwcp_migration_failed_total[5m]) / rate(dwcp_migration_total[5m])) > 0.10
        for: 5m
        labels:
          severity: high
        annotations:
          summary: "High migration failure rate"
          description: "{{ $value }}% migrations failing (threshold: 10%)"

      # Network degradation
      - alert: NetworkDegradation
        expr: avg(dwcp_amst_throughput_mbps) < 50
        for: 10m
        labels:
          severity: high
        annotations:
          summary: "Network throughput degraded"
          description: "Average throughput {{ $value }} Mbps (threshold: 50 Mbps)"
```

#### 4.4.3 Medium Alerts (P2)

```yaml
groups:
  - name: dwcp_medium
    interval: 5m
    rules:
      # State sync lag
      - alert: StateSyncLagHigh
        expr: max(dwcp_state_sync_lag_seconds) > 60
        for: 15m
        labels:
          severity: medium
        annotations:
          summary: "State sync lag high"
          description: "Max lag {{ $value }}s (threshold: 60s)"

      # Low compression ratio
      - alert: LowCompressionRatio
        expr: avg(dwcp_hde_compression_ratio) < 0.30
        for: 30m
        labels:
          severity: medium
        annotations:
          summary: "Compression ratio low"
          description: "Average ratio {{ $value }} (threshold: 0.30)"
```

### 4.5 Distributed Tracing

#### 4.5.1 Jaeger Setup

```bash
# Deploy Jaeger (all-in-one)
docker run -d --name jaeger \
  -p 5775:5775/udp \
  -p 6831:6831/udp \
  -p 6832:6832/udp \
  -p 5778:5778 \
  -p 16686:16686 \
  -p 14250:14250 \
  -p 14268:14268 \
  -p 14269:14269 \
  -p 9411:9411 \
  jaegertracing/all-in-one:latest

# Access UI: http://localhost:16686
```

#### 4.5.2 Trace Analysis

**Sample Trace: VM Migration**

```
Span 1: vm_migration_request (duration: 85s)
  ├─ Span 2: placement_decision (duration: 200ms)
  │   ├─ Span 3: resource_discovery (duration: 50ms)
  │   ├─ Span 4: constraint_solving (duration: 120ms)
  │   └─ Span 5: target_selection (duration: 30ms)
  │
  ├─ Span 6: network_prediction (duration: 100ms)
  │   ├─ Span 7: metrics_collection (duration: 20ms)
  │   └─ Span 8: lstm_inference (duration: 80ms)
  │
  ├─ Span 9: consensus_voting (duration: 2s)
  │   ├─ Span 10: propose_request (duration: 100ms)
  │   ├─ Span 11: vote_collection (duration: 1.8s)
  │   └─ Span 12: commit_decision (duration: 100ms)
  │
  ├─ Span 13: memory_transfer (duration: 80s)
  │   ├─ Span 14: compression (duration: 15s, parallel)
  │   ├─ Span 15: deduplication (duration: 10s, parallel)
  │   ├─ Span 16: amst_transmission (duration: 65s)
  │   └─ Span 17: decompression (duration: 10s, parallel)
  │
  └─ Span 18: state_update (duration: 3s)
      ├─ Span 19: local_update (duration: 10ms)
      └─ Span 20: gossip_propagation (duration: 2.99s)
```

**Analyzing Bottlenecks:**
1. Filter traces by duration >120s (P95 target: 90s)
2. Identify slowest spans (typically `amst_transmission`)
3. Check for outliers (spans >2x median)
4. Correlate with network metrics (bandwidth drops)

---

## 5. Troubleshooting Common Issues

### 5.1 Migration Failures

#### 5.1.1 Symptom: Migration Timeout

**Error Message:**
```
Error: Migration timeout after 300s
VM: vm-12345
Source: worker-042
Target: worker-156
```

**Diagnosis:**
```bash
# Check network connectivity
dwcp-cli network test \
  --source worker-042 \
  --target worker-156

# Check AMST stream health
dwcp-cli amst status \
  --node worker-042 \
  --migration vm-12345

# Check bandwidth utilization
dwcp-cli metrics get \
  --metric amst_throughput_mbps \
  --node worker-042 \
  --duration 5m
```

**Common Causes:**
1. **Network congestion:** Check if throughput <50 Mbps
2. **Packet loss:** Check if loss rate >5%
3. **Firewall blocking:** Verify ports 8085-8100 open
4. **Resource exhaustion:** Check CPU/memory on source node

**Solutions:**
```bash
# Increase timeout for large VMs
dwcp-cli migration set-timeout --vm vm-12345 --timeout 600s

# Retry with fewer AMST streams (less aggressive)
dwcp-cli migration retry --vm vm-12345 --amst-streams 4

# Force target selection (avoid congested path)
dwcp-cli migration retry --vm vm-12345 --target worker-200

# Defer migration until network improves
dwcp-cli migration defer --vm vm-12345 --delay 1h
```

#### 5.1.2 Symptom: High Downtime

**Error Message:**
```
Warning: VM downtime exceeded threshold
VM: vm-67890
Downtime: 45s (threshold: 20s)
```

**Diagnosis:**
```bash
# Check VM memory size (larger = longer downtime)
dwcp-cli vm inspect --vm vm-67890 --field memory_size

# Check compression ratio (low ratio = more data)
dwcp-cli metrics get \
  --metric hde_compression_ratio \
  --migration vm-67890

# Check target node load (overloaded = slow VM start)
dwcp-cli node metrics --node worker-156 --fields cpu,memory
```

**Common Causes:**
1. **Large VM:** 32GB+ VMs have 20-40s downtime
2. **Low compression:** <30% ratio means more data to transfer
3. **Slow target node:** High CPU/memory usage delays VM start
4. **Network jitter:** Packet loss during final sync phase

**Solutions:**
```bash
# Use pre-copy migration (reduce downtime)
dwcp-cli migration set-strategy --vm vm-67890 --strategy pre-copy

# Increase compression level (slower but smaller)
dwcp-cli migration set-compression --vm vm-67890 --level 9

# Defer migration until target node less loaded
dwcp-cli placement avoid-node --node worker-156 --duration 1h

# Enable compression dictionary training (better ratio)
dwcp-cli hde train-dict --vm vm-67890 --samples 100MB
```

### 5.2 Consensus Issues

#### 5.2.1 Symptom: Frequent Leader Elections

**Error Message:**
```
Warning: Raft leader election
Old leader: controller-1
New leader: controller-2
Election time: 450ms
Elections in last hour: 12
```

**Diagnosis:**
```bash
# Check network latency between controllers
dwcp-cli network latency \
  --nodes controller-1,controller-2,controller-3

# Check controller resource usage
dwcp-cli node metrics \
  --nodes controller-1,controller-2,controller-3 \
  --fields cpu,memory,disk_io

# View Raft logs
dwcp-cli raft logs --node controller-1 --tail 100
```

**Common Causes:**
1. **Network latency:** >100ms between controllers
2. **Controller overload:** >80% CPU usage
3. **Disk I/O saturation:** >90% disk utilization
4. **Clock skew:** >500ms time difference

**Solutions:**
```bash
# Increase election timeout (tolerate higher latency)
dwcp-cli raft set-timeout --election 500ms --heartbeat 150ms

# Move controller to same datacenter (reduce latency)
dwcp-cli node relocate --node controller-3 --datacenter dc1

# Increase controller resources (reduce load)
# (Requires infrastructure change)

# Enable NTP synchronization (fix clock skew)
sudo systemctl enable ntp
sudo systemctl start ntp
```

#### 5.2.2 Symptom: Consensus Stall

**Error Message:**
```
Error: Consensus stalled
Last commit: 5 minutes ago
Pending proposals: 27
PBFT view: 15 (stuck)
```

**Diagnosis:**
```bash
# Check quorum status
dwcp-cli consensus quorum-status

# Check Byzantine nodes
dwcp-cli reputation list --threshold 0.5

# Check network partitions
dwcp-cli network partition-detect
```

**Common Causes:**
1. **Lost quorum:** <2/3 controllers healthy
2. **Byzantine attack:** >33% malicious nodes in voting set
3. **Network partition:** Controllers split across regions

**Solutions:**
```bash
# Force view change (PBFT)
dwcp-cli consensus force-view-change --new-view 16

# Quarantine Byzantine nodes
dwcp-cli reputation quarantine --node worker-234 --reason "vote_manipulation"

# Trigger manual failover
dwcp-cli consensus failover --target controller-2

# Emergency: Reset consensus (DATA LOSS POSSIBLE)
dwcp-cli consensus reset --confirm
```

### 5.3 Performance Degradation

#### 5.3.1 Symptom: Low Bandwidth

**Error Message:**
```
Warning: AMST throughput degraded
Node: worker-078
Current: 45 Mbps
Expected: 200+ Mbps
```

**Diagnosis:**
```bash
# Run network speed test
dwcp-cli network speedtest \
  --source worker-078 \
  --target worker-156

# Check stream health
dwcp-cli amst stream-status \
  --node worker-078

# Check for congestion
dwcp-cli network congestion-detect \
  --node worker-078

# Check PBA predictions
dwcp-cli pba predictions --node worker-078
```

**Common Causes:**
1. **ISP throttling:** Bandwidth drops during peak hours
2. **Stream imbalance:** One stream carrying >50% traffic
3. **Congestion collapse:** All streams experiencing loss
4. **Insufficient streams:** Only 1-2 streams active

**Solutions:**
```bash
# Increase stream count
dwcp-cli amst set-streams --node worker-078 --count 16

# Enable BBR v2 congestion control
dwcp-cli amst set-cc --algorithm bbr2

# Defer migrations until off-peak hours
dwcp-cli placement set-schedule \
  --window 01:00-06:00 \
  --timezone UTC

# Use bandwidth reservation (if supported by ISP)
# (External configuration)
```

#### 5.3.2 Symptom: High CPU Usage

**Error Message:**
```
Warning: Controller CPU high
Node: controller-1
CPU: 92% (threshold: 80%)
Component: ITP solver
```

**Diagnosis:**
```bash
# Check CPU by component
dwcp-cli node profile --node controller-1 --duration 60s

# Check ITP solver complexity
dwcp-cli placement solver-stats

# Check VM count
dwcp-cli vm count --node controller-1
```

**Common Causes:**
1. **Large cluster:** >10,000 VMs increases solver complexity
2. **Complex constraints:** Many affinity/anti-affinity rules
3. **Frequent rebalancing:** <5min interval triggers too often

**Solutions:**
```bash
# Reduce solver runtime (sacrifice optimality for speed)
dwcp-cli placement set-solver \
  --max-time 5s \
  --target-optimality 0.90

# Reduce rebalance frequency
dwcp-cli placement set-rebalance-interval --interval 15m

# Add more controller nodes (distribute load)
dwcp-cli cluster add-controller --node controller-4

# Simplify constraints (remove rarely used rules)
dwcp-cli placement remove-constraint --id affinity-rule-123
```

### 5.4 State Synchronization Issues

#### 5.4.1 Symptom: High State Sync Lag

**Error Message:**
```
Warning: State sync lag high
Max lag: 120s (threshold: 60s)
Affected nodes: 45
```

**Diagnosis:**
```bash
# Check per-node lag
dwcp-cli state-sync lag-status --show-all

# Check gossip health
dwcp-cli gossip status

# Check CRDT conflicts
dwcp-cli state-sync conflicts --duration 1h
```

**Common Causes:**
1. **Network partition:** Nodes isolated from cluster
2. **Gossip fanout too low:** <3 peers per node
3. **High conflict rate:** Concurrent updates to same keys
4. **Node overload:** CPU/memory exhaustion delays processing

**Solutions:**
```bash
# Increase gossip fanout (more redundancy)
dwcp-cli gossip set-fanout --fanout 10

# Trigger manual sync
dwcp-cli state-sync force-sync --nodes worker-234,worker-235

# Increase replication factor (faster convergence)
dwcp-cli state-sync set-replication --factor 5

# Partition healing: Force rejoin
dwcp-cli node rejoin --node worker-234
```

---

## 6. Incident Response Procedures

### 6.1 Incident Severity Levels

| Level | Description | Response Time | Escalation |
|-------|-------------|---------------|------------|
| **P0 - Critical** | Complete cluster failure, data loss risk | <15 min | Immediate |
| **P1 - High** | Partial cluster failure, degraded service | <1 hour | After 30min |
| **P2 - Medium** | Performance degradation, no user impact | <4 hours | After 2 hours |
| **P3 - Low** | Minor issues, workarounds available | <24 hours | None |
| **P4 - Info** | Informational, no action required | N/A | None |

### 6.2 Incident Response Workflow

```
1. Detection (Alerts, User Reports)
        ↓
2. Triage (Assess Severity, Assign Responder)
        ↓
3. Investigation (Gather Data, Identify Root Cause)
        ↓
4. Mitigation (Implement Fix, Verify Resolution)
        ↓
5. Communication (Update Stakeholders)
        ↓
6. Post-Incident Review (Document, Improve)
```

### 6.3 P0 Incident: Cluster Failure

#### 6.3.1 Symptoms
- All controllers unreachable
- No VM migrations completing
- Workers unable to join cluster
- Prometheus alerts: `ControllerClusterDegraded`

#### 6.3.2 Immediate Actions (First 15 minutes)

**Step 1: Assess Damage**
```bash
# Check controller health
for controller in controller-{1..3}; do
  echo "=== $controller ==="
  ssh $controller "systemctl status dwcp-controller"
  ssh $controller "dwcp-cli cluster status 2>&1"
done

# Check network connectivity
for controller in controller-{1..3}; do
  ping -c 3 $controller
  nc -zv $controller 8080
done
```

**Step 2: Attempt Automatic Recovery**
```bash
# Restart controllers (in order)
ssh controller-1 "sudo systemctl restart dwcp-controller"
sleep 10
ssh controller-2 "sudo systemctl restart dwcp-controller"
sleep 10
ssh controller-3 "sudo systemctl restart dwcp-controller"

# Wait 30s for cluster formation
sleep 30

# Verify cluster reformed
dwcp-cli cluster status
```

**Step 3: Manual Recovery (if automatic fails)**
```bash
# Emergency bootstrap (LAST RESORT)
ssh controller-1 "dwcp-cli cluster emergency-bootstrap --force"

# Rejoin other controllers
ssh controller-2 "dwcp-cli node rejoin --leader controller-1:8082"
ssh controller-3 "dwcp-cli node rejoin --leader controller-1:8082"

# Verify quorum
dwcp-cli consensus quorum-status
```

#### 6.3.3 Communication Template

**Subject:** [P0] DWCP v3 Cluster Failure - Investigating

**Body:**
```
Incident Start Time: [TIMESTAMP]
Severity: P0 - Critical
Status: Investigating
Impact: All VM migrations blocked, cluster unavailable

Actions Taken:
- [TIMESTAMP]: Incident detected via alert "ControllerClusterDegraded"
- [TIMESTAMP]: Responder [NAME] assigned
- [TIMESTAMP]: Investigation started, checking controller health
- [TIMESTAMP]: Restarted controllers 1-3
- [TIMESTAMP]: [STATUS UPDATE]

Next Update: 15 minutes
On-Call: [NAME] ([PHONE])
```

### 6.4 P1 Incident: Mass Worker Failure

#### 6.4.1 Symptoms
- >10% workers unreachable
- High VM migration failure rate
- Prometheus alerts: `MassWorkerFailure`

#### 6.4.2 Investigation Steps

```bash
# Identify failed workers
dwcp-cli node list --status unreachable > /tmp/failed-workers.txt

# Check common failure patterns
cat /tmp/failed-workers.txt | awk '{print $3}' | sort | uniq -c
# Look for patterns: same region, same ISP, same version

# Check recent events
dwcp-cli events list --duration 1h --type node_failure

# Analyze logs (sample 10 failed nodes)
for node in $(head -10 /tmp/failed-workers.txt | awk '{print $1}'); do
  echo "=== $node ==="
  dwcp-cli node logs --node $node --tail 100 | grep -i error
done
```

#### 6.4.3 Common Causes and Solutions

**Cause 1: Network Partition (Regional ISP Issue)**
```bash
# Identify affected region
dwcp-cli node list --status unreachable | awk '{print $3}' | sort | uniq -c

# Expected: High concentration in one region
# Example: 250 nodes in "us-west-2"

# Solution: Enable partition tolerance mode
dwcp-cli cluster set-mode --partition-tolerance high

# Defer migrations involving affected region
dwcp-cli placement set-blacklist \
  --region us-west-2 \
  --duration 4h \
  --reason "ISP outage"

# Monitor partition healing
watch -n 30 'dwcp-cli node list --region us-west-2 --status ready'
```

**Cause 2: Software Bug (Crash Loop)**
```bash
# Check worker version
dwcp-cli node list --field version | sort | uniq -c

# Expected: Version mismatch or recent upgrade
# Example: 250 nodes on v3.0.1 (recently upgraded)

# Solution: Rollback to previous version
dwcp-cli upgrade rollback --version v3.0.0 --nodes <failed-nodes>

# Or: Apply hotfix
dwcp-cli upgrade apply-patch --patch v3.0.1-hotfix1
```

**Cause 3: Resource Exhaustion (OOM Killer)**
```bash
# Check memory usage before crash
for node in $(head -10 /tmp/failed-workers.txt | awk '{print $1}'); do
  ssh $node "dmesg | grep -i 'out of memory'"
done

# Solution: Reduce VM density or increase node memory
dwcp-cli placement set-max-vms --max 80 --nodes <affected-nodes>

# Or: Trigger emergency VM evacuation
dwcp-cli migration evacuate --nodes <affected-nodes> --reason "OOM"
```

### 6.5 P2 Incident: Performance Degradation

See `docs/runbooks/DWCP_V3_PERFORMANCE_TROUBLESHOOTING.md` for detailed procedures.

---

## 7. Rollback Procedures

### 7.1 Rollback Decision Matrix

| Scenario | Rollback? | Method |
|----------|-----------|--------|
| <5% workers failing | No | Fix forward |
| 5-20% workers failing | Maybe | Evaluate after 1 hour |
| >20% workers failing | Yes | Immediate rollback |
| Data corruption detected | Yes | Immediate rollback + restore |
| Security vulnerability | Yes | Immediate rollback + patch |
| Performance <50% baseline | Maybe | Evaluate after troubleshooting |

### 7.2 Pre-Rollback Checklist

```bash
# 1. Verify backup availability
dwcp-cli backup list --date today

# 2. Check cluster state
dwcp-cli cluster status > /tmp/cluster-state-pre-rollback.txt

# 3. Export critical data
dwcp-cli state-sync export --output /tmp/state-backup.json

# 4. Notify stakeholders
# Send email: "[P1] DWCP v3 Rollback Initiated"

# 5. Create rollback snapshot
dwcp-cli cluster snapshot --name pre-rollback-$(date +%s)
```

### 7.3 Rollback Execution

#### 7.3.1 Controller Rollback

**Blue-Green Deployment (Recommended):**
```bash
# Assuming blue cluster (v3.0.1) running, green cluster (v3.0.0) standby

# 1. Switch DNS to green cluster
dwcp-cli dns switch --target green-cluster

# 2. Wait for worker reconnections (5 minutes)
sleep 300

# 3. Verify green cluster healthy
dwcp-cli cluster status --cluster green-cluster

# 4. Decomission blue cluster
dwcp-cli cluster decomission --cluster blue-cluster --confirm
```

**In-Place Rollback (Faster but riskier):**
```bash
# 1. Stop controllers one-by-one
ssh controller-1 "sudo systemctl stop dwcp-controller"
ssh controller-2 "sudo systemctl stop dwcp-controller"
ssh controller-3 "sudo systemctl stop dwcp-controller"

# 2. Restore previous version binaries
for controller in controller-{1..3}; do
  ssh $controller "
    sudo rm /opt/dwcp/bin/dwcp-server
    sudo cp /opt/dwcp/backup/v3.0.0/dwcp-server /opt/dwcp/bin/
  "
done

# 3. Restore configuration
for controller in controller-{1..3}; do
  ssh $controller "
    sudo cp /etc/dwcp/controller.yaml.backup /etc/dwcp/controller.yaml
  "
done

# 4. Start controllers
ssh controller-1 "sudo systemctl start dwcp-controller" & sleep 10
ssh controller-2 "sudo systemctl start dwcp-controller" & sleep 10
ssh controller-3 "sudo systemctl start dwcp-controller" &

# 5. Wait for cluster reformation (30s)
sleep 30

# 6. Verify cluster health
dwcp-cli cluster status
```

#### 7.3.2 Worker Rollback

**Rolling Rollback (No Downtime):**
```bash
# 1. Identify workers needing rollback
dwcp-cli node list --version v3.0.1 > /tmp/rollback-workers.txt

# 2. Rollback in batches (100 workers at a time)
for batch in {1..10}; do
  echo "=== Batch $batch ==="

  # Select 100 workers
  workers=$(awk "NR > $((($batch-1)*100)) && NR <= $(($batch*100)) {print \$1}" /tmp/rollback-workers.txt)

  # Evacuate VMs (migrate to other workers)
  for worker in $workers; do
    dwcp-cli migration evacuate --node $worker &
  done
  wait

  # Rollback workers
  for worker in $workers; do
    ssh $worker "
      sudo systemctl stop dwcp-worker
      sudo rm /opt/dwcp/bin/dwcp-server
      sudo cp /opt/dwcp/backup/v3.0.0/dwcp-server /opt/dwcp/bin/
      sudo systemctl start dwcp-worker
    " &
  done
  wait

  # Verify workers rejoined
  sleep 60
  dwcp-cli node list --nodes $workers --status ready

  # Continue to next batch
done
```

### 7.4 Post-Rollback Verification

```bash
# 1. Cluster health
dwcp-cli cluster health

# 2. Version verification
dwcp-cli node list --fields id,version | grep -v v3.0.0
# Expected: No output (all nodes on v3.0.0)

# 3. Functional testing
dwcp-cli migration test --source worker-001 --target worker-002 --vm test-vm

# 4. Performance baseline
dwcp-cli metrics compare \
  --before /tmp/metrics-pre-upgrade.json \
  --after /tmp/metrics-post-rollback.json

# 5. Data integrity check
dwcp-cli state-sync verify --checksum
```

### 7.5 Post-Rollback Communication

**Subject:** [RESOLVED] DWCP v3 Rollback Complete - Service Restored

**Body:**
```
Incident End Time: [TIMESTAMP]
Total Duration: [X] hours
Rollback Version: v3.0.0 (from v3.0.1)
Status: Resolved

Timeline:
- [TIMESTAMP]: Incident detected (mass worker failures)
- [TIMESTAMP]: Rollback decision made (>20% workers failing)
- [TIMESTAMP]: Rollback initiated (controllers first)
- [TIMESTAMP]: Worker rollback started (rolling batches)
- [TIMESTAMP]: All workers rolled back
- [TIMESTAMP]: Verification complete, service healthy

Post-Incident Actions:
- Root cause analysis scheduled for [DATE]
- v3.0.1 deployment plan to be revised
- Additional testing added to CI/CD pipeline

Uptime Impact: [X] VMs affected, [Y] migrations failed
Service Restored: 100%

On-Call: [NAME] ([PHONE])
```

---

## 8. Performance Optimization

### 8.1 Optimization Areas

1. **AMST Transport:** Increase throughput via stream tuning
2. **HDE Encoding:** Improve compression ratio and speed
3. **PBA Prediction:** Increase accuracy via model retraining
4. **ASS State Sync:** Reduce lag via gossip optimization
5. **ITP Placement:** Improve resource utilization via constraint tuning
6. **ACP Consensus:** Reduce latency via protocol optimization

### 8.2 AMST Optimization

#### 8.2.1 Stream Count Tuning

**Current Performance:**
```bash
# Check current stream count
dwcp-cli amst status --node worker-001

# Output:
# Streams: 8
# Throughput: 450 Mbps
# Utilization: 45% (of 1 Gbps link)
```

**Optimization:**
```bash
# Increase stream count (test 12, 16)
dwcp-cli amst set-streams --node worker-001 --count 12

# Run benchmark
dwcp-cli network speedtest --source worker-001 --target worker-002 --duration 60s

# Expected improvement:
# Streams: 12
# Throughput: 650 Mbps
# Utilization: 65%

# If improvement plateaus, revert to optimal count
dwcp-cli amst set-streams --node worker-001 --count 12 --persist
```

**Best Practices:**
- **Low latency (<50ms):** 4-8 streams
- **Medium latency (50-200ms):** 8-12 streams
- **High latency (>200ms):** 12-16 streams
- **Congested links:** Reduce to 4 streams (less competition)

#### 8.2.2 Congestion Control Tuning

**BBR v2 Parameters:**
```yaml
# /etc/dwcp/worker.yaml
transport:
  amst:
    congestion_control: bbr2
    bbr2:
      pacing_gain: 2.77  # Default: 2.89
      cwnd_gain: 2.0     # Default: 2.0
      probe_rtt_interval: 10s  # Default: 10s
```

**Tuning for Different Scenarios:**

**High Bandwidth, Low Loss (<1%):**
```yaml
bbr2:
  pacing_gain: 2.89  # Aggressive
  cwnd_gain: 2.5     # Larger window
  probe_rtt_interval: 15s  # Less frequent probing
```

**Low Bandwidth, High Loss (>5%):**
```yaml
bbr2:
  pacing_gain: 2.5   # Conservative
  cwnd_gain: 1.5     # Smaller window
  probe_rtt_interval: 5s  # More frequent probing
```

### 8.3 HDE Optimization

#### 8.3.1 Compression Level Tuning

**Trade-off: Speed vs Ratio**

| Level | Speed | Ratio | CPU Usage | Use Case |
|-------|-------|-------|-----------|----------|
| 1 | 500 MB/s | 1.5x | 10% | Low CPU, fast migrations |
| 3 | 300 MB/s | 2.0x | 15% | Balanced (default) |
| 5 | 150 MB/s | 2.5x | 25% | Better ratio, acceptable CPU |
| 9 | 50 MB/s | 3.0x | 50% | Maximum ratio, slow |
| 19 | 10 MB/s | 3.5x | 90% | Extreme (not recommended) |

**Optimization:**
```bash
# Test different levels
for level in 1 3 5 9; do
  dwcp-cli hde set-compression --level $level --node worker-001
  dwcp-cli migration benchmark --vm test-vm --iterations 5
done

# Analyze results
dwcp-cli metrics analyze --metric migration_duration_seconds --group compression_level

# Expected: Level 5 provides best balance for most workloads
```

#### 8.3.2 Dictionary Training

**Concept:** Train compression dictionary on representative data for better ratio

**Procedure:**
```bash
# Collect sample data (100MB from typical VMs)
dwcp-cli hde collect-samples --vms vm-{1..100} --size 100MB --output /tmp/samples.bin

# Train dictionary
dwcp-cli hde train-dict --input /tmp/samples.bin --size 1MB --output /etc/dwcp/hde-dict.bin

# Deploy dictionary to all workers
for worker in worker-{001..1000}; do
  scp /etc/dwcp/hde-dict.bin $worker:/etc/dwcp/
  ssh $worker "sudo systemctl restart dwcp-worker"
done

# Verify improvement (expect 5-15% better ratio)
dwcp-cli metrics compare \
  --metric hde_compression_ratio \
  --before yesterday \
  --after today
```

### 8.4 PBA Optimization

#### 8.4.1 Model Retraining

**When to Retrain:**
- Accuracy drops below 70%
- Network conditions change (new ISP, new routes)
- Seasonal traffic patterns (monthly/quarterly)

**Procedure:**
```bash
# Check current accuracy
dwcp-cli pba metrics --field accuracy

# Output:
# Accuracy: 68% (threshold: 70%)
# Last training: 14 days ago

# Trigger retraining
dwcp-cli pba retrain \
  --duration 7d \
  --samples 100000 \
  --target-accuracy 0.75

# Expected: 1-2 hours training time, 75%+ accuracy
```

#### 8.4.2 Feature Engineering

**Add New Features:**
```yaml
# /etc/dwcp/controller.yaml
prediction:
  pba:
    features:
      - bandwidth_5m_avg
      - bandwidth_1h_avg
      - latency_5m_avg
      - packet_loss_rate
      - time_of_day  # NEW: Capture daily patterns
      - day_of_week  # NEW: Capture weekly patterns
      - isp_congestion_index  # NEW: External data
```

**Expected Improvement:** 70% → 75-80% accuracy

### 8.5 ITP Optimization

#### 8.5.1 Constraint Simplification

**Identify Expensive Constraints:**
```bash
# Profile ITP solver
dwcp-cli placement profile --duration 5m

# Output:
# Total time: 2.5s
# Constraint evaluation: 1.8s (72%)
#   - Affinity rules: 1.2s (67% of constraints)
#   - Anti-affinity rules: 0.4s (22%)
#   - Resource constraints: 0.2s (11%)
```

**Optimize Affinity Rules:**
```bash
# List all affinity rules
dwcp-cli placement list-constraints --type affinity

# Remove rarely used rules
dwcp-cli placement remove-constraint --id affinity-rule-123 --reason "unused"

# Combine related rules
dwcp-cli placement merge-constraints --ids affinity-rule-45,affinity-rule-46

# Expected: 1.8s → 0.9s (50% reduction)
```

### 8.6 Performance Baselines

**After Optimization:**

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Migration Time (2GB)** | 90s | 60s | 33% |
| **Throughput** | 450 Mbps | 650 Mbps | 44% |
| **Compression Ratio** | 50% | 65% | 30% |
| **PBA Accuracy** | 68% | 78% | 15% |
| **ITP Solver Time** | 2.5s | 1.2s | 52% |
| **Consensus Latency** | 3s | 1.5s | 50% |

---

## 9. Security Operations

### 9.1 Byzantine Tolerance Overview

**Threat Model:**
- **Malicious nodes:** Up to 33% of cluster (PBFT guarantee)
- **Attacks:** Vote manipulation, false metrics, state corruption
- **Detection:** Reputation system, behavior anomaly detection
- **Response:** Automatic quarantine, manual investigation

### 9.2 Monitoring for Byzantine Behavior

#### 9.2.1 Reputation System

**How it Works:**
```
1. Each node starts with reputation score 1.0
2. Correct votes/actions increase score by 0.01
3. Incorrect votes/actions decrease score by 0.10
4. Score decays by 0.01/day (natural decay)
5. Scores below 0.3 trigger quarantine
```

**Monitoring:**
```bash
# View reputation scores
dwcp-cli reputation list --sort score

# Output:
# Node          Score  Votes  Incorrect  Status
# worker-001    0.98   1000   2          healthy
# worker-042    0.35   500    150        quarantined
# worker-234    0.25   800    350        quarantined

# Investigate low-score nodes
dwcp-cli reputation analyze --node worker-042
```

#### 9.2.2 Byzantine Attack Detection

**Common Attack Patterns:**

**Attack 1: Vote Manipulation**
```bash
# Symptom: Node consistently votes against consensus
dwcp-cli consensus analyze --node worker-042

# Output:
# Votes cast: 500
# Votes with consensus: 250 (50%)
# Expected: >95%

# Action: Quarantine node
dwcp-cli reputation quarantine --node worker-042 --reason "vote_manipulation"
```

**Attack 2: False Metrics**
```bash
# Symptom: Node reports impossibly high performance
dwcp-cli metrics analyze --node worker-042 --anomaly-detection

# Output:
# Reported bandwidth: 5 Gbps (physical link: 1 Gbps)
# Anomaly score: 0.95 (threshold: 0.80)

# Action: Quarantine and investigate
dwcp-cli reputation quarantine --node worker-042 --reason "false_metrics"
dwcp-cli node forensics --node worker-042 --collect-logs
```

**Attack 3: State Corruption**
```bash
# Symptom: Node propagates incorrect state via gossip
dwcp-cli state-sync verify --node worker-042

# Output:
# State checksum: abc123 (expected: def456)
# Corrupt keys: 45

# Action: Quarantine, force re-sync
dwcp-cli reputation quarantine --node worker-042 --reason "state_corruption"
dwcp-cli state-sync force-resync --node worker-042
```

### 9.3 Incident Response: Byzantine Attack

**Scenario:** Mass Byzantine attack (10% of nodes compromised)

**Detection:**
```bash
# Alert: HighQuarantineRate
dwcp-cli reputation list --threshold 0.5

# Output: 100 nodes with score <0.5 (10% of cluster)
```

**Immediate Actions:**
```bash
# 1. Identify attack pattern
dwcp-cli reputation analyze-attack --duration 1h

# Output:
# Attack type: vote_manipulation
# Affected nodes: 100
# Common attributes: ISP=ISP-X, region=us-west-2

# 2. Mass quarantine
dwcp-cli reputation quarantine-bulk \
  --threshold 0.5 \
  --reason "byzantine_attack_$(date +%s)"

# 3. Increase PBFT quorum requirement (temporary)
dwcp-cli consensus set-quorum --requirement 0.75  # From 0.67

# 4. Notify security team
# Send alert: "[P1] Byzantine Attack Detected - 100 Nodes"
```

**Investigation:**
```bash
# Collect forensic data
for node in $(dwcp-cli reputation list --threshold 0.5 --format id); do
  dwcp-cli node forensics --node $node --output /tmp/forensics-$node.tar.gz &
done
wait

# Analyze logs for common patterns
for file in /tmp/forensics-*.tar.gz; do
  tar -xzf $file -C /tmp/forensics/
done

grep -r "ERROR" /tmp/forensics/ | awk '{print $5}' | sort | uniq -c | sort -nr
# Look for: common error messages, similar timestamps, coordinated behavior
```

**Remediation:**
```bash
# Option 1: Reinstall compromised nodes (recommended)
for node in $(dwcp-cli reputation list --threshold 0.5 --format id); do
  ssh $node "sudo rm -rf /opt/dwcp && sudo apt-get remove --purge dwcp-worker"
  # Re-provision node (Terraform/Ansible)
done

# Option 2: Temporary ban (faster but less secure)
dwcp-cli reputation ban --nodes <compromised-nodes> --duration 30d

# Option 3: Permanent removal (if nodes owned by attacker)
dwcp-cli node remove --nodes <compromised-nodes> --confirm
```

### 9.4 Security Best Practices

1. **Monitor reputation scores daily** (automated dashboard)
2. **Investigate scores <0.7** (potential early warning)
3. **Quarantine scores <0.3** (automatic)
4. **Rotate TLS certificates quarterly** (prevent compromise)
5. **Enable audit logging** (compliance + forensics)
6. **Conduct Byzantine drills quarterly** (test response procedures)

---

## 10. Hands-On Exercises

See `docs/training/labs/` for detailed lab instructions.

### 10.1 Lab 1: Deploy DWCP v3 in Test Environment

**Duration:** 2 hours

**Objectives:**
- Deploy 3 controller nodes
- Deploy 10 worker nodes
- Verify cluster health

**Deliverables:**
- Cluster status showing all nodes healthy
- Screenshot of Grafana dashboard

### 10.2 Lab 2: Monitor with Grafana Dashboards

**Duration:** 1 hour

**Objectives:**
- Import DWCP v3 dashboards
- Configure Prometheus data source
- Create custom alert rule

**Deliverables:**
- All dashboards displaying metrics
- Custom alert triggered and visible in AlertManager

### 10.3 Lab 3: Execute VM Migration (Datacenter Mode)

**Duration:** 1.5 hours

**Objectives:**
- Create test VM (2GB)
- Migrate VM between workers (low latency)
- Measure migration time and downtime

**Deliverables:**
- Migration completed in <30s
- VM accessible post-migration
- Metrics exported to CSV

### 10.4 Lab 4: Execute VM Migration (Internet Mode)

**Duration:** 1.5 hours

**Objectives:**
- Simulate WAN latency (100ms)
- Migrate VM between workers (high latency)
- Compare with datacenter mode

**Deliverables:**
- Migration completed in <90s
- Compression ratio >50%
- Performance comparison chart

### 10.5 Lab 5: Handle Byzantine Attack Simulation

**Duration:** 2 hours

**Objectives:**
- Inject malicious node (vote manipulation)
- Detect attack via reputation system
- Quarantine malicious node

**Deliverables:**
- Attack detected within 5 minutes
- Node automatically quarantined
- Cluster remains operational

### 10.6 Lab 6: Perform Emergency Rollback

**Duration:** 2 hours

**Objectives:**
- Deploy new version (simulated buggy release)
- Detect failure (high error rate)
- Execute rollback procedure

**Deliverables:**
- Rollback completed in <30 minutes
- Service restored to previous version
- Zero data loss

### 10.7 Lab 7: Investigate Performance Issues

**Duration:** 2 hours

**Objectives:**
- Inject performance degradation (bandwidth throttle)
- Use distributed tracing to identify bottleneck
- Apply optimization

**Deliverables:**
- Root cause identified via Jaeger trace
- Optimization applied (e.g., increase streams)
- Performance restored to baseline

### 10.8 Lab 8: Analyze Distributed Traces

**Duration:** 1.5 hours

**Objectives:**
- Capture migration trace
- Identify slowest span
- Propose optimization

**Deliverables:**
- Trace exported from Jaeger
- Bottleneck identified (e.g., consensus voting)
- Optimization proposal documented

---

## 11. Certification Assessment

### 11.1 Written Exam (50 questions, 90 minutes)

**Topics:**
- DWCP v3 architecture (15 questions)
- Deployment procedures (10 questions)
- Monitoring and alerting (10 questions)
- Troubleshooting (10 questions)
- Incident response (5 questions)

**Passing Score:** 80% (40/50 correct)

**Sample Questions:**

**Q1:** What is the maximum Byzantine fault tolerance of DWCP v3 using PBFT?
- A) 25% malicious nodes
- B) 33% malicious nodes ✓
- C) 50% malicious nodes
- D) 67% malicious nodes

**Q2:** Which component is responsible for VM placement optimization?
- A) AMST
- B) HDE
- C) PBA
- D) ITP ✓

**Q3:** What is the default Zstandard compression level in HDE v3?
- A) 1
- B) 3
- C) 5 ✓
- D) 9

### 11.2 Practical Assessment (8 tasks, 4 hours)

**Tasks:**
1. Deploy 3-node controller cluster (30 min)
2. Deploy 5 worker nodes and verify health (30 min)
3. Execute VM migration and measure performance (30 min)
4. Identify and resolve simulated performance issue (45 min)
5. Respond to simulated Byzantine attack (45 min)
6. Perform emergency rollback (30 min)
7. Create custom Grafana dashboard (30 min)
8. Write incident report for simulated P1 incident (30 min)

**Passing Score:** 7/8 tasks completed successfully

### 11.3 Certification Levels

**Level 1: DWCP v3 Operations Associate**
- Written exam: 80%+
- Practical: 7/8 tasks
- Valid: 1 year

**Level 2: DWCP v3 Operations Professional**
- Level 1 certified
- 6+ months production experience
- Advanced practical exam (10 tasks)
- Valid: 2 years

**Level 3: DWCP v3 Operations Expert**
- Level 2 certified
- 2+ years production experience
- Case study presentation
- Valid: 3 years

---

## 12. Additional Resources

### 12.1 Documentation

- **Architecture:** `docs/research/DWCP-INTERNET-SCALE-DISTRIBUTED-HYPERVISOR.md`
- **API Reference:** `docs/deployment/DWCP_V3_API_REFERENCE.md`
- **Performance Tuning:** `docs/deployment/DWCP_V3_PERFORMANCE_TUNING.md`
- **CI/CD Guide:** `docs/cicd/DWCP_V3_CICD_GUIDE.md`
- **IaC Guide:** `docs/cicd/DWCP_V3_IAC_GUIDE.md`

### 12.2 Runbooks

- **Production Rollout:** `docs/runbooks/DWCP_V3_PRODUCTION_ROLLOUT_DETAILED.md`
- **Incident Response:** `docs/runbooks/DWCP_V3_INCIDENT_RESPONSE.md`
- **Performance Troubleshooting:** `docs/runbooks/DWCP_V3_PERFORMANCE_TROUBLESHOOTING.md`
- **Security Incidents:** `docs/runbooks/DWCP_V3_SECURITY_INCIDENT_RESPONSE.md`

### 12.3 Support Channels

- **Slack:** #dwcp-v3-ops
- **Email:** ops-support@example.com
- **On-Call:** PagerDuty schedule "DWCP v3 Operations"
- **Wiki:** https://wiki.example.com/dwcp-v3

### 12.4 Related Training

- **DWCP v3 Developer Training:** `docs/training/DWCP_V3_DEVELOPER_TRAINING.md`
- **DWCP v3 Security Training:** `docs/training/DWCP_V3_SECURITY_TRAINING.md`
- **Kubernetes on DWCP:** `docs/training/K8S_ON_DWCP_TRAINING.md`

---

## Appendix A: Quick Reference Commands

```bash
# Cluster Management
dwcp-cli cluster status
dwcp-cli cluster health
dwcp-cli node list
dwcp-cli node metrics --node <node-id>

# VM Migration
dwcp-cli migration start --vm <vm-id> --target <node-id>
dwcp-cli migration status --vm <vm-id>
dwcp-cli migration list --status in_progress

# Monitoring
dwcp-cli metrics get --metric <metric-name> --duration <duration>
dwcp-cli metrics export --output metrics.json

# Troubleshooting
dwcp-cli network test --source <node-a> --target <node-b>
dwcp-cli node logs --node <node-id> --tail 100
dwcp-cli amst status --node <node-id>

# Security
dwcp-cli reputation list
dwcp-cli reputation quarantine --node <node-id> --reason "<reason>"
dwcp-cli consensus quorum-status

# Performance
dwcp-cli placement solver-stats
dwcp-cli hde set-compression --level <level>
dwcp-cli amst set-streams --count <count>
```

---

## Appendix B: Troubleshooting Decision Tree

```
Migration Failure
    ├─ Timeout (>300s)?
    │   ├─ Yes → Check network bandwidth (target: >50 Mbps)
    │   │   ├─ Low bandwidth → Increase AMST streams, retry
    │   │   └─ Adequate bandwidth → Check compression ratio
    │   └─ No → Check error logs
    │
    ├─ High downtime (>30s)?
    │   ├─ Large VM (>32GB) → Use pre-copy migration
    │   ├─ Low compression → Train dictionary, retry
    │   └─ Slow target node → Defer to less loaded node
    │
    └─ Consensus failure?
        ├─ Leader election → Check controller latency
        ├─ Quorum loss → Verify controller health
        └─ Byzantine attack → Check reputation scores
```

---

**End of Operations Training Manual**

**Next Steps:**
1. Complete hands-on labs (Labs 1-8)
2. Take certification exam
3. Shadow on-call engineer (1 week)
4. Graduate to Level 1: DWCP v3 Operations Associate

**Questions?** Contact: ops-training@example.com
