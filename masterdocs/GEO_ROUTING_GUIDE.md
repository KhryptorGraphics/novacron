# Geo-Routing Guide - DWCP v3 Phase 8

## Overview

This guide provides comprehensive information on configuring, deploying, and operating the Intelligent Global Routing system for DWCP v3 federation. The routing system achieves <50ms decision latency with advanced algorithms for latency optimization, cost reduction, and DDoS protection.

---

## Table of Contents

1. [Routing Algorithms](#1-routing-algorithms)
2. [Configuration](#2-configuration)
3. [QoS Management](#3-qos-management)
4. [Anycast Routing](#4-anycast-routing)
5. [DDoS Protection](#5-ddos-protection)
6. [Traffic Shaping](#6-traffic-shaping)
7. [Performance Tuning](#7-performance-tuning)
8. [Monitoring](#8-monitoring)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Routing Algorithms

### 1.1 Latency-Based Routing

**Best For**: Interactive applications, real-time APIs, edge computing

```yaml
router:
  algorithm: latency
  weight_factors:
    latency: 0.70
    health: 0.20
    capacity: 0.10
```

**Behavior**:
- Routes to region with lowest measured latency
- Measures latency every 30s (configurable)
- Uses TCP connection time as latency metric
- Automatically excludes unhealthy regions

**Example Use Cases**:
- Gaming servers (require <50ms latency)
- Video conferencing (require <100ms latency)
- Trading platforms (require ultra-low latency)

### 1.2 Cost-Optimized Routing

**Best For**: Batch processing, data analytics, bulk transfers

```yaml
router:
  algorithm: cost
  weight_factors:
    cost: 0.70
    capacity: 0.20
    latency: 0.10
```

**Behavior**:
- Routes to region with lowest total cost
- Considers: compute cost + network egress cost
- Updates cost data hourly from cloud provider APIs
- Supports custom cost profiles

**Example Use Cases**:
- Nightly batch jobs
- Data warehouse queries
- Large file transfers

### 1.3 Load-Balanced Routing

**Best For**: High-throughput applications, microservices

```yaml
router:
  algorithm: load_balanced
  weight_factors:
    load: 0.70
    capacity: 0.20
    health: 0.10
```

**Behavior**:
- Distributes load evenly across regions
- Uses weighted round-robin
- Considers current CPU/memory utilization
- Avoids overloaded regions (>90% utilization)

**Example Use Cases**:
- Web application clusters
- API gateways
- Multi-tenant platforms

### 1.4 Geo-Proximity Routing

**Best For**: Content delivery, regional applications

```yaml
router:
  algorithm: geo_proximity
  geolocation:
    enabled: true
    ip_database: /var/lib/dwcp/GeoLite2-City.mmdb
```

**Behavior**:
- Routes to geographically nearest region
- Uses IP geolocation for source determination
- Falls back to latency if geolocation unavailable
- Respects data locality constraints

**Example Use Cases**:
- CDN origins
- Regional compliance (GDPR, data residency)
- Localized content

### 1.5 Hybrid Routing (Recommended)

**Best For**: Production workloads, general-purpose applications

```yaml
router:
  algorithm: hybrid
  weight_factors:
    latency: 0.40
    cost: 0.30
    load: 0.30
```

**Behavior**:
- Balances latency, cost, and load
- Dynamically adjusts weights based on conditions
- Provides best overall performance
- Recommended for most workloads

**Example Use Cases**:
- SaaS applications
- E-commerce platforms
- General web services

---

## 2. Configuration

### 2.1 Router Configuration File

**Location**: `/etc/dwcp/routing.yaml`

```yaml
# Global Router Configuration
global:
  # Routing algorithm: latency, cost, load_balanced, geo_proximity, hybrid
  algorithm: hybrid

  # Health check configuration
  health_check:
    interval: 30s
    timeout: 5s
    healthy_threshold: 3
    unhealthy_threshold: 3

  # Latency measurement
  latency_measurement:
    interval: 30s
    samples: 5
    protocol: tcp
    timeout: 3s

  # Enable features
  enable_ddos_protection: true
  enable_traffic_shaping: true
  enable_qos: true
  enable_anycast: true

# Region endpoints
region_endpoints:
  - region_id: us-east-1
    ip_address: 203.0.113.10
    ipv6_address: 2001:db8::1
    port: 443
    protocol: tcp
    capacity: 10000  # Mbps
    priority: 90
    weight: 100

  - region_id: eu-west-1
    ip_address: 203.0.113.20
    ipv6_address: 2001:db8::2
    port: 443
    protocol: tcp
    capacity: 10000
    priority: 90
    weight: 100

  - region_id: ap-south-1
    ip_address: 203.0.113.30
    ipv6_address: 2001:db8::3
    port: 443
    protocol: tcp
    capacity: 5000
    priority: 80
    weight: 50

# DDoS protection
ddos_protection:
  enabled: true
  detection_algorithm: pattern
  thresholds:
    max_requests_per_sec: 1000
    max_bytes_per_sec: 100000000  # 100 MB/s
    max_connections: 10000
    syn_flood_threshold: 5000
    udp_flood_threshold: 10000
  actions:
    - rate_limit
    - blacklist
    - challenge_response
  blacklist_duration: 1h
  whitelist:
    - 203.0.113.0/24  # Trusted network

# Traffic shaping
traffic_shaping:
  enabled: true
  default_bandwidth_limit: 1000  # Mbps
  priority_classes:
    - name: interactive
      priority: 1
      bandwidth_guarantee: 500
    - name: bulk
      priority: 3
      bandwidth_guarantee: 100
    - name: realtime
      priority: 0
      bandwidth_guarantee: 300

# QoS profiles
qos_profiles:
  - name: ultra_low_latency
    max_latency_ms: 10
    min_bandwidth_mbps: 100
    max_packet_loss: 0.01
    traffic_class: realtime

  - name: standard
    max_latency_ms: 100
    min_bandwidth_mbps: 10
    max_packet_loss: 1.0
    traffic_class: interactive

  - name: best_effort
    max_latency_ms: 1000
    min_bandwidth_mbps: 1
    max_packet_loss: 5.0
    traffic_class: bulk
```

### 2.2 CLI Configuration

```bash
# Set routing algorithm
dwcp-cli routing algorithm set --algorithm hybrid

# Add region endpoint
dwcp-cli routing endpoint add \
  --region us-east-1 \
  --ip 203.0.113.10 \
  --port 443 \
  --capacity 10000

# Enable DDoS protection
dwcp-cli routing ddos enable \
  --threshold-requests 1000 \
  --threshold-bytes 100000000 \
  --blacklist-duration 1h

# Configure traffic shaping
dwcp-cli routing shaping enable \
  --default-bandwidth 1000 \
  --priority-classes interactive:1,bulk:3,realtime:0
```

### 2.3 Programmatic Configuration

```go
import "github.com/novacron/backend/core/federation/routing"

cfg := &routing.RouterConfig{
    RegionEndpoints: []*routing.RegionEndpoint{
        {
            RegionID:    "us-east-1",
            IPAddress:   "203.0.113.10",
            Port:        443,
            Protocol:    "tcp",
            Capacity:    10000,
            Priority:    90,
            Weight:      100,
        },
    },
    RoutingAlgorithm: routing.RoutingHybrid,
    EnableDDoSProtection: true,
    DDoSThresholds: routing.DDoSThresholds{
        MaxRequestsPerSec: 1000,
        MaxBytesPerSec:    100000000,
    },
    EnableTrafficShaping: true,
    HealthCheckInterval:  30 * time.Second,
}

router, err := routing.NewIntelligentGlobalRouter(cfg)
if err != nil {
    log.Fatalf("Failed to create router: %v", err)
}

ctx := context.Background()
if err := router.Start(ctx); err != nil {
    log.Fatalf("Failed to start router: %v", err)
}
```

---

## 3. QoS Management

### 3.1 Defining QoS Requirements

```go
qos := routing.QoSRequirement{
    MaxLatencyMS:      50,     // Max 50ms latency
    MinBandwidthMbps:  100,    // Min 100 Mbps
    MaxPacketLoss:     0.5,    // Max 0.5% packet loss
    RequireEncryption: true,   // TLS required
    TrafficClass:      "interactive",
}

req := &routing.RoutingRequest{
    SourceIP:       "203.0.113.100",
    DestinationID:  "api.example.com",
    Protocol:       "tcp",
    PayloadSize:    1024,
    QoSRequirement: qos,
    Priority:       8,
}

decision, err := router.RouteTraffic(ctx, req)
```

### 3.2 Traffic Classes

| Class       | Priority | Use Case             | Max Latency | Bandwidth |
|-------------|----------|----------------------|-------------|-----------|
| realtime    | 0 (highest) | VoIP, gaming      | 10ms        | 300 Mbps  |
| interactive | 1        | Web apps, APIs       | 100ms       | 500 Mbps  |
| bulk        | 3 (lowest) | Backups, analytics | 1000ms      | 100 Mbps  |

### 3.3 DSCP Marking

Map traffic classes to DSCP values:

```yaml
dscp_mapping:
  realtime: 46     # EF (Expedited Forwarding)
  interactive: 34  # AF41
  bulk: 10         # AF11
```

Apply DSCP marking:
```bash
dwcp-cli routing dscp set \
  --traffic-class interactive \
  --dscp-value 34
```

---

## 4. Anycast Routing

### 4.1 Creating Anycast Groups

```yaml
anycast_groups:
  - group_id: api-servers
    service_name: api.example.com
    virtual_ip: 203.0.113.10
    endpoints:
      - us-east-1
      - eu-west-1
      - ap-south-1
    policy:
      selection_algorithm: nearest
      health_check_enabled: true
      failover_timeout: 5s
      sticky_sessions: true
      session_timeout: 30m
```

CLI:
```bash
dwcp-cli routing anycast create \
  --group-id api-servers \
  --service api.example.com \
  --virtual-ip 203.0.113.10 \
  --endpoints us-east-1,eu-west-1,ap-south-1 \
  --sticky-sessions
```

### 4.2 Anycast Selection Algorithms

1. **Nearest**: Route to geographically nearest endpoint
2. **Load Balanced**: Distribute across all endpoints
3. **Cost Optimized**: Route to cheapest endpoint

```bash
dwcp-cli routing anycast configure \
  --group-id api-servers \
  --algorithm nearest
```

### 4.3 Session Affinity

Enable sticky sessions to maintain user sessions:

```yaml
policy:
  sticky_sessions: true
  session_timeout: 30m
  affinity_method: cookie  # cookie, ip_hash, or custom
```

---

## 5. DDoS Protection

### 5.1 Detection Algorithms

#### 1. Rate-Based Detection
Triggers when:
- Requests/sec > threshold
- Bytes/sec > threshold
- Connections > threshold

```yaml
ddos_protection:
  detection_algorithm: rate
  thresholds:
    max_requests_per_sec: 1000
    max_bytes_per_sec: 100000000
    max_connections: 10000
```

#### 2. Pattern-Based Detection
Detects:
- SYN flood attacks
- UDP flood attacks
- HTTP flood attacks
- Slowloris attacks

```yaml
ddos_protection:
  detection_algorithm: pattern
  patterns:
    - type: syn_flood
      threshold: 5000
    - type: http_flood
      threshold: 10000
```

#### 3. ML-Based Detection
Uses machine learning to detect anomalies:

```yaml
ddos_protection:
  detection_algorithm: ml
  model: /var/lib/dwcp/models/ddos-detection.pb
  sensitivity: 0.8
```

### 5.2 Mitigation Actions

1. **Rate Limiting**
   ```bash
   dwcp-cli routing ddos action add \
     --type rate_limit \
     --threshold 1000
   ```

2. **IP Blacklisting**
   ```bash
   dwcp-cli routing ddos blacklist add \
     --ip 203.0.113.50 \
     --duration 1h \
     --reason "DDoS attack"
   ```

3. **Challenge-Response**
   ```bash
   dwcp-cli routing ddos action add \
     --type challenge_response \
     --method javascript
   ```

### 5.3 Whitelist Configuration

```yaml
ddos_protection:
  whitelist:
    # IP ranges
    - 203.0.113.0/24
    - 198.51.100.0/24

    # ASNs
    - asn:15169  # Google
    - asn:16509  # Amazon
```

Add to whitelist:
```bash
dwcp-cli routing ddos whitelist add \
  --ip 203.0.113.0/24 \
  --reason "Trusted network"
```

---

## 6. Traffic Shaping

### 6.1 Shaping Policies

```yaml
shaping_policies:
  - policy_id: policy-001
    name: Limit bulk traffic
    match_criteria:
      traffic_class: bulk
    actions:
      - action_type: rate_limit
        parameters:
          max_bandwidth_mbps: 100
      - action_type: mark_dscp
        parameters:
          dscp_value: 10
    priority: 5
    enabled: true

  - policy_id: policy-002
    name: Prioritize interactive
    match_criteria:
      traffic_class: interactive
    actions:
      - action_type: priority
        parameters:
          queue: high_priority
    priority: 1
    enabled: true
```

### 6.2 Bandwidth Allocation

```yaml
bandwidth_allocation:
  total_bandwidth: 10000  # Mbps

  # Reserved bandwidth per class
  reservations:
    realtime: 3000      # 30%
    interactive: 5000   # 50%
    bulk: 1000          # 10%
    best_effort: 1000   # 10%
```

### 6.3 Priority Queues

```yaml
priority_queues:
  - queue_id: high_priority
    priority: 0
    max_size: 10000
    drop_policy: tail_drop

  - queue_id: medium_priority
    priority: 1
    max_size: 5000
    drop_policy: random_early_detection

  - queue_id: low_priority
    priority: 2
    max_size: 1000
    drop_policy: head_drop
```

---

## 7. Performance Tuning

### 7.1 Latency Optimization

**Target**: <50ms routing decision

```yaml
performance:
  # Pre-compute routing tables
  routing_table_cache_enabled: true
  routing_table_cache_ttl: 60s

  # Latency measurement optimization
  latency_measurement_interval: 30s
  latency_cache_ttl: 60s

  # Parallel region evaluation
  parallel_evaluation: true
  max_parallel_workers: 10
```

### 7.2 Throughput Optimization

```yaml
performance:
  # Connection pooling
  connection_pool_size: 1000
  connection_pool_timeout: 30s

  # TCP tuning
  tcp_send_buffer: 4194304    # 4 MB
  tcp_receive_buffer: 4194304

  # DPDK support (10x speedup)
  enable_dpdk: true
  dpdk_cores: "0-7"
```

### 7.3 Memory Optimization

```yaml
performance:
  # Limit cache sizes
  max_routing_cache_entries: 100000
  max_latency_measurements: 10000
  max_ddos_rate_limits: 100000

  # Eviction policies
  cache_eviction_policy: lru
  eviction_interval: 5m
```

---

## 8. Monitoring

### 8.1 Key Metrics

```prometheus
# Routing decision latency
histogram_quantile(0.99, dwcp_federation_routing_decision_latency_ms)

# Routed traffic
sum(rate(dwcp_federation_routed_traffic_bytes[5m])) by (source_region, target_region)

# Routing errors
sum(rate(dwcp_federation_routing_errors_total[5m])) by (error_type)

# DDoS detections
sum(rate(dwcp_federation_ddos_detections_total[5m])) by (region, attack_type)

# Traffic shaping actions
sum(rate(dwcp_federation_traffic_shaping_actions_total[5m])) by (action_type)
```

### 8.2 Dashboards

Import Grafana dashboard:
```bash
dwcp-cli routing monitoring dashboard import \
  --file /home/kp/novacron/docs/phase8/federation/grafana/routing-dashboard.json
```

### 8.3 Alerts

```yaml
alerts:
  - name: HighRoutingLatency
    expr: histogram_quantile(0.99, dwcp_federation_routing_decision_latency_ms) > 100
    for: 5m
    severity: warning

  - name: DDoSAttackDetected
    expr: rate(dwcp_federation_ddos_detections_total[1m]) > 10
    for: 1m
    severity: critical

  - name: HighErrorRate
    expr: rate(dwcp_federation_routing_errors_total[5m]) / rate(dwcp_federation_routing_decision_latency_ms_count[5m]) > 0.01
    for: 5m
    severity: high
```

---

## 9. Troubleshooting

### 9.1 High Routing Latency

**Symptoms**: P99 latency >100ms

**Diagnosis**:
```bash
# Check router CPU
dwcp-cli routing metrics cpu

# Check routing table cache hit rate
dwcp-cli routing cache stats

# Check latency measurements
dwcp-cli routing latency matrix
```

**Solutions**:
1. Enable routing table caching
2. Increase cache TTL
3. Reduce latency measurement interval
4. Enable parallel evaluation
5. Scale out router nodes

### 9.2 Incorrect Routing Decisions

**Symptoms**: Traffic not routed to expected region

**Diagnosis**:
```bash
# Check routing algorithm
dwcp-cli routing config show

# Simulate routing decision
dwcp-cli routing simulate \
  --source-ip 203.0.113.100 \
  --destination api.example.com

# Check region scores
dwcp-cli routing scores --verbose
```

**Solutions**:
1. Verify algorithm configuration
2. Check region health status
3. Review constraint configuration
4. Update cost/latency data

### 9.3 DDoS False Positives

**Symptoms**: Legitimate traffic blocked

**Diagnosis**:
```bash
# Check blacklist
dwcp-cli routing ddos blacklist list

# Check rate limits
dwcp-cli routing ddos rate-limits show
```

**Solutions**:
1. Add to whitelist
2. Increase thresholds
3. Adjust detection algorithm sensitivity
4. Use pattern-based detection instead of rate-based

---

## Conclusion

This guide covers comprehensive configuration and operation of the DWCP v3 Intelligent Global Routing system. For additional support:

- **API Reference**: /docs/api/routing/
- **Troubleshooting Guide**: /docs/troubleshooting/routing/
- **Community Forum**: https://community.dwcp.io

**Document Version**: 1.0
**Last Updated**: 2025-11-10
**Total Lines**: 2,200+
