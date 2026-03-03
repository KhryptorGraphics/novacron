# DWCP v3 Production Metrics Interpretation Guide

**Version**: 1.0.0
**Last Updated**: 2025-11-10
**Audience**: Operations Team, SREs, Engineering Managers
**Classification**: Internal Use

---

## Table of Contents

1. [Overview](#overview)
2. [System Health Metrics](#system-health-metrics)
3. [Application Performance Metrics](#application-performance-metrics)
4. [Infrastructure Metrics](#infrastructure-metrics)
5. [Business Metrics](#business-metrics)
6. [Database Metrics](#database-metrics)
7. [Message Queue Metrics](#message-queue-metrics)
8. [Network Metrics](#network-metrics)
9. [Security Metrics](#security-metrics)
10. [Metric Decision Trees](#metric-decision-trees)
11. [Case Studies](#case-studies)
12. [Alerting Thresholds](#alerting-thresholds)

---

## Overview

### Purpose

This guide provides comprehensive interpretation guidance for all production metrics in the DWCP v3 system. It helps operators understand what metrics mean, identify normal vs abnormal patterns, and make data-driven decisions.

### Metric Categories

```
DWCP v3 Metrics Hierarchy
├── System Health (availability, reliability)
├── Application Performance (latency, throughput)
├── Infrastructure (CPU, memory, disk, network)
├── Business (users, requests, conversions)
├── Database (queries, connections, replication)
├── Message Queue (lag, throughput, errors)
├── Network (latency, packet loss, bandwidth)
└── Security (auth failures, suspicious activity)
```

### Reading This Guide

- **Metric Name**: Official name in Prometheus/Grafana
- **Description**: What the metric measures
- **Normal Range**: Expected values under normal conditions
- **Warning Threshold**: Values that require attention
- **Critical Threshold**: Values requiring immediate action
- **Interpretation**: How to read and understand the metric
- **Action Items**: What to do when threshold is breached

---

## System Health Metrics

### 1. Service Availability

**Metric**: `up{job="dwcp-api"}`

**Description**: Binary metric indicating if service is reachable

**Values**:
- `1` = Service is up and healthy
- `0` = Service is down or unreachable

**Normal**: `1` (100% of targets)

**Warning**: <95% of targets returning `1`

**Critical**: <50% of targets returning `1`

**Interpretation**:
```promql
# Calculate availability percentage
sum(up{job="dwcp-api"}) / count(up{job="dwcp-api"}) * 100

# Examples:
# 100% = All instances healthy
# 90% = 1 out of 10 instances down
# 0% = Complete outage
```

**Decision Tree**:
```
up == 0 for a pod?
├─ YES
│   ├─ Check pod logs: kubectl logs <pod>
│   ├─ Check pod events: kubectl describe pod <pod>
│   ├─ Is pod CrashLooping?
│   │   ├─ YES → Check recent deployments
│   │   └─ NO → Check node health
│   └─ Recent deployment?
│       ├─ YES → Rollback
│       └─ NO → Check infrastructure
└─ NO
    └─ All healthy, no action needed
```

**Case Study**:
```
Scenario: API availability dropped to 80%
Investigation:
- 8 out of 10 pods healthy
- 2 pods showing CrashLoopBackOff
- Logs show "Connection refused" to database
- Database metrics show connection pool exhausted

Root Cause: Database connection pool too small
Resolution: Increased pool size from 50 to 100
Result: Availability restored to 100%
```

### 2. Error Rate

**Metric**: `http_requests_total{status=~"5.."}`

**Description**: Rate of HTTP 5xx server errors

**Normal Range**: <0.1% (1 error per 1000 requests)

**Warning Threshold**: >0.5% (5 errors per 1000 requests)

**Critical Threshold**: >1% (10 errors per 1000 requests)

**Calculation**:
```promql
# Error rate as percentage
sum(rate(http_requests_total{status=~"5.."}[5m]))
/
sum(rate(http_requests_total[5m]))
* 100

# Error count per second
sum(rate(http_requests_total{status=~"5.."}[5m]))
```

**Interpretation by Status Code**:

| Status Code | Meaning | Likely Cause |
|-------------|---------|--------------|
| 500 | Internal Server Error | Application bug, unhandled exception |
| 502 | Bad Gateway | Upstream service down, network issue |
| 503 | Service Unavailable | Service overloaded, pod restarting |
| 504 | Gateway Timeout | Slow downstream service, timeout too short |

**Deep Dive Analysis**:
```bash
# Breakdown by endpoint
rate(http_requests_total{status=~"5.."}[5m]) by (path)

# Breakdown by pod
rate(http_requests_total{status=~"5.."}[5m]) by (pod)

# Breakdown by status code
rate(http_requests_total{status=~"5.."}[5m]) by (status)
```

**Pattern Recognition**:
- **Spike**: Sudden increase → recent deployment or dependency failure
- **Gradual Increase**: Slow climb → memory leak or resource exhaustion
- **Periodic**: Regular pattern → scheduled job causing issues
- **Constant**: Steady high rate → configuration issue or chronic bug

### 3. Request Latency (p50, p95, p99)

**Metric**: `http_request_duration_seconds`

**Description**: Time to process HTTP requests (histogram)

**Percentile Explanation**:
- **p50 (median)**: 50% of requests faster, 50% slower
- **p95**: 95% of requests faster, 5% slower
- **p99**: 99% of requests faster, 1% slower

**Normal Ranges**:
| Percentile | Normal | Warning | Critical |
|------------|--------|---------|----------|
| p50 | <100ms | >200ms | >500ms |
| p95 | <200ms | >500ms | >1000ms |
| p99 | <500ms | >1000ms | >2000ms |

**Calculation**:
```promql
# p95 latency
histogram_quantile(0.95,
  rate(http_request_duration_seconds_bucket[5m])
)

# p50 latency
histogram_quantile(0.50,
  rate(http_request_duration_seconds_bucket[5m])
)

# p99 latency
histogram_quantile(0.99,
  rate(http_request_duration_seconds_bucket[5m])
)
```

**Interpretation**:

**Scenario 1: All percentiles high**
```
p50: 800ms (normal: 100ms)
p95: 2000ms (normal: 200ms)
p99: 5000ms (normal: 500ms)

Interpretation: System-wide slowdown
Likely Causes:
- Database performance degradation
- Resource exhaustion (CPU/memory)
- Network issues
- Downstream service slow
```

**Scenario 2: p99 high, p50/p95 normal**
```
p50: 100ms (normal)
p95: 250ms (slightly elevated)
p99: 3000ms (very high)

Interpretation: Long-tail latency issue
Likely Causes:
- Specific slow queries
- Garbage collection pauses
- Occasional timeouts
- Cache misses
```

**Scenario 3: Gradual increase over time**
```
Hour 1: p95 = 200ms
Hour 2: p95 = 300ms
Hour 3: p95 = 400ms
Hour 4: p95 = 600ms

Interpretation: Resource degradation
Likely Causes:
- Memory leak
- Connection pool exhaustion
- Disk I/O saturation
```

**Latency Budget Analysis**:
```
Total Latency Budget: 200ms (p95)

Breakdown:
- API Gateway: 10ms (5%)
- Authentication: 20ms (10%)
- Business Logic: 50ms (25%)
- Database Query: 100ms (50%)
- Response Serialization: 20ms (10%)

If p95 exceeds 200ms, identify which component is slow:
1. Check per-endpoint latency
2. Enable distributed tracing
3. Analyze slow queries
4. Profile application code
```

### 4. Request Rate (RPS)

**Metric**: `http_requests_total`

**Description**: Total HTTP requests per second

**Normal Range**: Varies by time of day and region

**Baseline Calculation**:
```promql
# Current RPS
sum(rate(http_requests_total[5m]))

# RPS by endpoint
sum(rate(http_requests_total[5m])) by (path)

# RPS by region
sum(rate(http_requests_total[5m])) by (region)
```

**Establishing Baseline**:
```bash
# Daily pattern
0:00-6:00:  Low traffic (1000 RPS)
6:00-9:00:  Morning ramp (3000 RPS)
9:00-17:00: Business hours (8000 RPS)
17:00-20:00: Evening peak (10000 RPS)
20:00-24:00: Evening decline (4000 RPS)

# Weekly pattern
Monday-Thursday: High (8000 RPS average)
Friday: Medium (6000 RPS average)
Saturday-Sunday: Low (3000 RPS average)
```

**Anomaly Detection**:
```promql
# Deviation from weekly average
abs(
  rate(http_requests_total[5m])
  -
  rate(http_requests_total[1w] offset 1w)
)
/
rate(http_requests_total[1w] offset 1w)
* 100

# Alert if >50% deviation
> 50
```

**Traffic Pattern Interpretation**:

**Pattern 1: Sudden Spike (10x normal)**
```
Normal: 5000 RPS
Spike: 50000 RPS
Duration: 5 minutes

Possible Causes:
1. Marketing campaign launch
2. Viral content
3. DDoS attack
4. Automated scraper/bot
5. Retry storm from clients

Actions:
1. Check if traffic is legitimate (user-agents, IPs)
2. Enable rate limiting
3. Scale up if legitimate
4. Block if attack
```

**Pattern 2: Gradual Increase**
```
Hour 1: 5000 RPS
Hour 2: 7000 RPS
Hour 3: 10000 RPS
Hour 4: 15000 RPS

Possible Causes:
1. Organic growth
2. Viral event
3. Misconfigured client retry logic
4. Slow-moving DDoS

Actions:
1. Monitor capacity utilization
2. Prepare to scale
3. Verify request patterns
```

**Pattern 3: Sudden Drop**
```
Normal: 8000 RPS
Drop: 1000 RPS
Duration: Ongoing

Possible Causes:
1. Client-side issue
2. DNS failure
3. CDN issue
4. Service outage (clients not reaching us)
5. Major incident affecting users

Actions:
1. Check external monitoring
2. Verify DNS resolution
3. Check CDN status
4. Review error logs
```

### 5. Active Connections

**Metric**: `dwcp_active_connections`

**Description**: Number of concurrent client connections

**Normal Range**: 100-1000 per pod

**Warning Threshold**: >5000 per pod

**Critical Threshold**: >10000 per pod

**Calculation**:
```promql
# Total active connections
sum(dwcp_active_connections)

# Per pod
dwcp_active_connections by (pod)

# Connection rate
rate(dwcp_connections_total[5m])
```

**Interpretation**:

**High Connections, Normal Latency**
```
Connections: 8000
Latency p95: 150ms
CPU: 40%

Interpretation: Healthy high traffic
Action: Monitor capacity, prepare to scale if continues growing
```

**High Connections, High Latency**
```
Connections: 8000
Latency p95: 2000ms
CPU: 95%

Interpretation: System overloaded
Action: Scale up immediately
```

**Low Connections, High Latency**
```
Connections: 500
Latency p95: 3000ms
CPU: 30%

Interpretation: Per-request slowness
Action: Investigate slow queries/operations
```

**Connections Growing Unbounded**
```
Hour 1: 1000 connections
Hour 2: 2000 connections
Hour 3: 4000 connections
Hour 4: 8000 connections

Interpretation: Connection leak
Action: Check for connection pool issues, missing close() calls
```

---

## Application Performance Metrics

### 6. Worker Utilization

**Metric**: `dwcp_worker_busy / dwcp_worker_total`

**Description**: Percentage of workers actively processing tasks

**Normal Range**: 40-70%

**Warning Threshold**: >80%

**Critical Threshold**: >95%

**Calculation**:
```promql
# Worker utilization percentage
sum(dwcp_worker_busy) / sum(dwcp_worker_total) * 100

# Per worker type
sum(dwcp_worker_busy) by (worker_type)
/
sum(dwcp_worker_total) by (worker_type)
* 100
```

**Interpretation**:

**Utilization Zones**:
```
0-40%:   Under-utilized (consider scaling down)
40-70%:  Optimal (good buffer for spikes)
70-85%:  High utilization (monitor closely)
85-95%:  Critical (scale up soon)
95-100%: Saturated (scale up immediately)
```

**Pattern Analysis**:
```
Pattern 1: Steady at 90%
Interpretation: Consistently overloaded
Action: Increase worker count

Pattern 2: Spikes to 100%, then drops to 20%
Interpretation: Bursty workload
Action: Implement auto-scaling

Pattern 3: Gradually increasing
Interpretation: Growing workload
Action: Plan capacity increase
```

### 7. Queue Depth

**Metric**: `dwcp_message_queue_depth`

**Description**: Number of messages waiting to be processed

**Normal Range**: 0-100 messages

**Warning Threshold**: >1000 messages

**Critical Threshold**: >10000 messages

**Calculation**:
```promql
# Current queue depth
dwcp_message_queue_depth

# Queue depth trend
rate(dwcp_message_queue_depth[30m])

# Messages processed per second
rate(dwcp_messages_processed_total[5m])

# Queue drain time
dwcp_message_queue_depth / rate(dwcp_messages_processed_total[5m])
```

**Health Assessment**:
```
Queue Depth: 5000 messages
Processing Rate: 100 msg/s
Drain Time: 50 seconds

Interpretation:
- Queue will be drained in <1 minute
- Temporary backlog, likely OK
- Monitor for continued growth
```

**Backlog Analysis**:
```
Scenario 1: Queue Growing Linearly
Time 0: 1000 messages
Time 5m: 2000 messages
Time 10m: 3000 messages
Growth Rate: 200 msg/min
Processing Rate: 150 msg/min

Interpretation: Processing slower than incoming rate
Action: Scale up workers immediately (50 msg/min deficit)

Scenario 2: Queue Oscillating
Time 0: 500 messages
Time 5m: 100 messages
Time 10m: 600 messages
Time 15m: 50 messages

Interpretation: Bursty traffic, workers keeping up
Action: No action, normal operation

Scenario 3: Queue Stuck at High Level
Time 0: 10000 messages
Time 30m: 10000 messages
Time 60m: 10000 messages

Interpretation: Workers not processing (possible failure)
Action: Check worker health, restart if necessary
```

### 8. Cache Hit Rate

**Metric**: `dwcp_cache_hits / dwcp_cache_requests`

**Description**: Percentage of requests served from cache

**Normal Range**: 80-95%

**Warning Threshold**: <70%

**Critical Threshold**: <50%

**Calculation**:
```promql
# Cache hit rate
sum(rate(dwcp_cache_hits[5m]))
/
sum(rate(dwcp_cache_requests[5m]))
* 100

# By cache type
sum(rate(dwcp_cache_hits[5m])) by (cache_type)
/
sum(rate(dwcp_cache_requests[5m])) by (cache_type)
* 100
```

**Interpretation**:

**High Hit Rate (90%+)**
```
Hit Rate: 92%
Interpretation: Cache working well
Benefits:
- Lower database load
- Faster response times
- Lower cost
```

**Low Hit Rate (<70%)**
```
Hit Rate: 65%
Possible Causes:
1. Cache expiration too aggressive
2. Cache not warming up properly
3. Traffic pattern changed
4. Cache size too small (evictions)
5. High percentage of unique requests

Investigation:
- Check cache eviction rate
- Review cache TTL settings
- Analyze request patterns
- Verify cache size adequate
```

**Sudden Drop in Hit Rate**
```
Before: 90% hit rate
After: 40% hit rate
Duration: Started 10 minutes ago

Possible Causes:
1. Cache was flushed/restarted
2. Deployment changed cache keys
3. Traffic pattern shift
4. Cache infrastructure issue

Actions:
1. Check cache service health
2. Review recent deployments
3. Verify cache warming
4. Monitor recovery
```

**Cache Performance Metrics**:
```promql
# Cache size
dwcp_cache_size_bytes

# Eviction rate
rate(dwcp_cache_evictions_total[5m])

# Cache operation latency
histogram_quantile(0.95, rate(dwcp_cache_operation_duration_seconds_bucket[5m]))
```

### 9. Concurrent Requests

**Metric**: `dwcp_requests_in_flight`

**Description**: Number of requests currently being processed

**Normal Range**: 10-100 per pod

**Warning Threshold**: >500 per pod

**Critical Threshold**: >1000 per pod

**Calculation**:
```promql
# Total in-flight requests
sum(dwcp_requests_in_flight)

# Per pod
dwcp_requests_in_flight by (pod)

# Trend
rate(dwcp_requests_in_flight[5m])
```

**Relationship to Other Metrics**:
```
High In-Flight + High Latency = Overloaded
High In-Flight + Normal Latency = High traffic, coping well
Low In-Flight + High Latency = Slow operations
Low In-Flight + Low Latency = Normal operation
```

---

## Infrastructure Metrics

### 10. CPU Utilization

**Metric**: `node_cpu_seconds_total`

**Description**: CPU usage across all cores

**Normal Range**: 40-70%

**Warning Threshold**: >80%

**Critical Threshold**: >90%

**Calculation**:
```promql
# CPU utilization by node
100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# CPU utilization by pod
sum(rate(container_cpu_usage_seconds_total[5m])) by (pod) * 100

# CPU throttling
rate(container_cpu_cfs_throttled_seconds_total[5m])
```

**Interpretation**:

**CPU Utilization Patterns**:
```
Pattern 1: Steady High (85-90%)
Interpretation: Consistently CPU-bound
Action: Scale up or optimize code

Pattern 2: Spikes to 100%
Interpretation: Bursty CPU-intensive operations
Action: Review resource limits, optimize hot paths

Pattern 3: Gradual increase
Interpretation: Growing load or efficiency regression
Action: Investigate recent changes, plan scaling

Pattern 4: Low utilization (20-30%)
Interpretation: Over-provisioned or I/O bound
Action: Consider scaling down or investigate I/O
```

**CPU Throttling**:
```promql
# Check if pods are being throttled
sum(rate(container_cpu_cfs_throttled_seconds_total[5m])) by (pod) > 0

# Throttling percentage
rate(container_cpu_cfs_throttled_seconds_total[5m])
/
rate(container_cpu_usage_seconds_total[5m])
* 100
```

**Interpretation**:
```
Throttling > 10% = Increase CPU limits
Throttling > 50% = Severe under-provisioning
```

### 11. Memory Utilization

**Metric**: `node_memory_MemAvailable_bytes`

**Description**: Available memory on nodes

**Normal Range**: 20-40% available (60-80% used)

**Warning Threshold**: <15% available (>85% used)

**Critical Threshold**: <10% available (>90% used)

**Calculation**:
```promql
# Memory utilization percentage
(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100

# Per pod memory usage
container_memory_working_set_bytes by (pod)

# Memory usage trend
rate(container_memory_working_set_bytes[30m])
```

**Interpretation**:

**Memory Patterns**:
```
Pattern 1: Steady at 75%
Interpretation: Normal, healthy usage
Action: None

Pattern 2: Gradually increasing (memory leak)
Hour 1: 50%
Hour 2: 60%
Hour 3: 70%
Hour 4: 80%

Interpretation: Possible memory leak
Action: Investigate for leaks, restart if necessary

Pattern 3: Sawtooth pattern
Peak: 80%
Drop: 40%
Peak: 80%
Drop: 40%

Interpretation: Garbage collection cycles
Action: Normal for garbage-collected languages

Pattern 4: Near 90% constantly
Interpretation: Under-provisioned
Action: Increase memory limits/requests
```

**OOM Kill Detection**:
```promql
# OOMKill events
kube_pod_container_status_last_terminated_reason{reason="OOMKilled"}

# Pods near memory limit
container_memory_working_set_bytes / container_spec_memory_limit_bytes > 0.9
```

### 12. Disk I/O

**Metric**: `node_disk_io_time_seconds_total`

**Description**: Time spent on disk I/O operations

**Normal Range**: <20% I/O wait

**Warning Threshold**: >30% I/O wait

**Critical Threshold**: >50% I/O wait

**Calculation**:
```promql
# I/O wait percentage
rate(node_cpu_seconds_total{mode="iowait"}[5m]) * 100

# Disk utilization
rate(node_disk_io_time_seconds_total[5m]) * 100

# IOPS
rate(node_disk_reads_completed_total[5m]) + rate(node_disk_writes_completed_total[5m])

# Throughput
rate(node_disk_read_bytes_total[5m]) + rate(node_disk_written_bytes_total[5m])
```

**Interpretation**:

**I/O Patterns**:
```
Low I/O Wait + High IOPS = Fast storage, handling load well
High I/O Wait + High IOPS = Storage saturated
Low I/O Wait + Low IOPS = Not I/O bound
High I/O Wait + Low IOPS = Slow storage or large requests
```

### 13. Disk Space

**Metric**: `node_filesystem_avail_bytes`

**Description**: Available disk space

**Normal Range**: >30% free

**Warning Threshold**: <20% free

**Critical Threshold**: <10% free

**Calculation**:
```promql
# Disk space available percentage
(node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100

# Disk space growth rate
rate(node_filesystem_avail_bytes[24h])
```

**Forecasting**:
```bash
# Predict when disk will be full
Current Free: 500GB
Daily Decrease: 50GB
Days Until Full: 500GB / 50GB = 10 days

Action: Plan disk expansion or cleanup within 5 days
```

---

## Database Metrics

### 14. Query Latency

**Metric**: `dwcp_db_query_duration_seconds`

**Description**: Time to execute database queries

**Normal Range**:
- Simple queries: <10ms
- Complex queries: <100ms
- Aggregations: <500ms

**Warning Threshold**: p95 >500ms

**Critical Threshold**: p95 >1000ms

**Calculation**:
```promql
# p95 query latency
histogram_quantile(0.95, rate(dwcp_db_query_duration_seconds_bucket[5m]))

# By query type
histogram_quantile(0.95, rate(dwcp_db_query_duration_seconds_bucket[5m])) by (query_type)

# Slow query count
sum(rate(dwcp_db_slow_queries_total[5m]))
```

**Interpretation**:

**Latency Analysis**:
```
Query: SELECT * FROM users WHERE email = ?
p50: 5ms (good)
p95: 8ms (good)
p99: 15ms (acceptable)

Interpretation: Well-indexed, efficient query

Query: SELECT * FROM orders JOIN products JOIN users
p50: 200ms (slow)
p95: 800ms (very slow)
p99: 2000ms (critical)

Interpretation: Missing indexes or inefficient join
Action: Add indexes, optimize query
```

### 15. Database Connections

**Metric**: `dwcp_db_connections_active`

**Description**: Active database connections

**Normal Range**: 30-70% of pool size

**Warning Threshold**: >80% of pool size

**Critical Threshold**: >95% of pool size

**Calculation**:
```promql
# Connection pool utilization
dwcp_db_connections_active / dwcp_db_connections_max * 100

# Wait time for connections
dwcp_db_connection_wait_duration_seconds
```

**Interpretation**:
```
Pool Size: 100
Active: 40
Utilization: 40%
Wait Time: 0ms

Interpretation: Healthy connection pool

Pool Size: 100
Active: 95
Utilization: 95%
Wait Time: 500ms

Interpretation: Connection pool nearly exhausted
Action: Increase pool size or add read replicas
```

### 16. Replication Lag

**Metric**: `dwcp_db_replication_lag_seconds`

**Description**: Time delay between primary and replica

**Normal Range**: <1 second

**Warning Threshold**: >5 seconds

**Critical Threshold**: >30 seconds

**Calculation**:
```promql
# Replication lag
dwcp_db_replication_lag_seconds by (replica)

# Lag rate (increasing or decreasing)
rate(dwcp_db_replication_lag_seconds[5m])
```

**Interpretation**:
```
Lag: 0.5 seconds
Trend: Stable

Interpretation: Healthy replication

Lag: 15 seconds
Trend: Increasing (was 10s, now 15s)

Interpretation: Replica falling behind
Possible Causes:
- High write volume
- Replica under-resourced
- Network issues
```

---

## Metric Decision Trees

### Decision Tree: High Latency

```
API latency >500ms (p95)?
├─ YES
│   ├─ Database latency high?
│   │   ├─ YES
│   │   │   ├─ Slow queries detected?
│   │   │   │   ├─ YES → Optimize queries, add indexes
│   │   │   │   └─ NO → Check database resources
│   │   │   ├─ Connection pool exhausted?
│   │   │   │   ├─ YES → Increase pool size
│   │   │   │   └─ NO → Check for locks
│   │   │   └─ Replication lag high?
│   │   │       ├─ YES → Scale replica
│   │   │       └─ NO → Investigate further
│   │   └─ NO
│   │       ├─ CPU utilization high (>85%)?
│   │       │   ├─ YES → Scale up pods
│   │       │   └─ NO → Continue
│   │       ├─ Memory near limit?
│   │       │   ├─ YES → Check for memory leak
│   │       │   └─ NO → Continue
│   │       └─ External dependency slow?
│   │           ├─ YES → Enable circuit breaker
│   │           └─ NO → Profile application
└─ NO
    └─ Latency normal, no action
```

### Decision Tree: High Error Rate

```
Error rate >1%?
├─ YES
│   ├─ Recent deployment (< 30 min)?
│   │   ├─ YES → Rollback deployment
│   │   └─ NO → Continue investigation
│   ├─ What error codes?
│   │   ├─ 500 Internal Server Error
│   │   │   └─ Check application logs for exceptions
│   │   ├─ 502 Bad Gateway
│   │   │   └─ Check upstream service health
│   │   ├─ 503 Service Unavailable
│   │   │   ├─ Pods restarting?
│   │   │   │   ├─ YES → Check pod logs
│   │   │   │   └─ NO → Check if overloaded
│   │   │   └─ Overloaded (high CPU/memory)?
│   │   │       ├─ YES → Scale up
│   │   │       └─ NO → Check dependencies
│   │   └─ 504 Gateway Timeout
│   │       └─ Increase timeout or optimize slow operations
│   └─ Dependency failure?
│       ├─ YES → Enable degraded mode
│       └─ NO → Continue debugging
└─ NO
    └─ Error rate normal
```

---

## Case Studies

### Case Study 1: Gradual Performance Degradation

**Symptoms**:
```
Day 1:  p95 latency = 200ms
Day 3:  p95 latency = 400ms
Day 7:  p95 latency = 800ms
Day 10: p95 latency = 1200ms
```

**Metrics Investigated**:
```
CPU: Stable at 50%
Memory: Increasing from 60% to 85%
Database connections: Stable at 40
Queue depth: Stable at 100
```

**Analysis**:
```
Memory increasing steadily = Memory leak suspected
Checked heap dumps over time
Found goroutine count increasing from 1000 to 50000
```

**Root Cause**:
```go
// Bug: Goroutines not being cleaned up
for {
    go func() {
        // Missing context cancellation
        processWork()
    }()
}
```

**Fix**:
```go
// Fixed: Properly manage goroutine lifecycle
ctx, cancel := context.WithCancel(context.Background())
defer cancel()

for {
    go func() {
        select {
        case <-ctx.Done():
            return
        default:
            processWork()
        }
    }()
}
```

**Result**: Latency returned to 200ms, memory stabilized at 60%

### Case Study 2: Intermittent Errors

**Symptoms**:
```
Error rate spikes to 5% every hour for 2-3 minutes
Then returns to 0.1%
```

**Metrics During Spike**:
```
Error rate: 5%
Latency: Normal (200ms)
CPU: Normal (60%)
Memory: Normal (70%)
Database connections: Spike from 40 to 95
```

**Analysis**:
```
Pattern: Hourly spike aligns with cron job schedule
Cron job runs: SELECT * FROM large_table (10M rows)
Creates 55 database connections simultaneously
Connection pool size: 100 (leaving only 5 for API)
API requests fail with "connection pool exhausted"
```

**Root Cause**:
```
Batch job not properly connection-pooled
Opening many connections simultaneously
```

**Fix**:
```python
# Before: Opening many connections
for item in large_dataset:
    conn = create_connection()
    process(conn, item)
    conn.close()

# After: Reusing connection pool
with connection_pool.get() as conn:
    for item in large_dataset:
        process(conn, item)
```

**Result**: Error spikes eliminated, connection usage stable

### Case Study 3: Cache Eviction Storm

**Symptoms**:
```
Cache hit rate dropped from 90% to 20%
Database latency increased from 50ms to 500ms
API latency increased from 200ms to 2000ms
```

**Timeline**:
```
10:00 AM: Cache hit rate normal (90%)
10:05 AM: Cache hit rate drops to 20%
10:10 AM: Database alerts fire (high load)
10:15 AM: API latency SLA breached
```

**Metrics**:
```
Cache eviction rate: 10000/sec (normally 100/sec)
Cache memory usage: Spiking between 80% and 95%
Cache size: 10GB
```

**Investigation**:
```
New code deployed at 10:00 AM
Code change: Added large object caching
Object size: 100MB each
10 objects cached = 1GB
Caused eviction of 10000 small objects (100KB each)
```

**Root Cause**:
```
Large cache objects evicting many small objects
Cache not configured with size limits per object
```

**Fix**:
```yaml
# Cache configuration
cache:
  max_object_size: 10MB
  eviction_policy: least_recently_used
  memory_limit: 10GB
  monitor_eviction_rate: true
```

**Result**: Cache hit rate restored to 90%, latency normalized

---

## Alerting Thresholds

### Tiered Alerting Strategy

```yaml
alerts:
  # SEV-1: Critical (Page immediately)
  critical:
    - error_rate > 5%
    - availability < 50%
    - p95_latency > 5000ms
    - db_connections_utilization > 95%
    - disk_space < 5%

  # SEV-2: High (Page during business hours, ticket after hours)
  high:
    - error_rate > 1%
    - availability < 95%
    - p95_latency > 1000ms
    - cpu_utilization > 90%
    - memory_utilization > 90%
    - db_connections_utilization > 80%
    - replication_lag > 30s
    - disk_space < 10%

  # SEV-3: Warning (Create ticket)
  warning:
    - error_rate > 0.5%
    - availability < 99%
    - p95_latency > 500ms
    - cpu_utilization > 80%
    - memory_utilization > 80%
    - cache_hit_rate < 70%
    - db_connections_utilization > 70%
    - replication_lag > 10s
    - disk_space < 20%

  # SEV-4: Info (Log only)
  info:
    - error_rate > 0.1%
    - p95_latency > 200ms
    - cache_hit_rate < 80%
```

### Alert Fatigue Prevention

**Rules**:
1. Alert on symptoms, not causes
2. Alert on user impact, not internal metrics
3. Use "for" clauses to reduce noise
4. Implement alert grouping
5. Add meaningful context to alerts

**Example**:
```yaml
# Bad: Alerts on everything
- alert: HighCPU
  expr: cpu > 80%
  for: 1m

# Good: Alerts on user impact
- alert: HighLatency
  expr: p95_latency > 1000ms
  for: 5m
  annotations:
    summary: "Users experiencing slow response times"
    runbook: "https://wiki.dwcp.io/runbooks/high-latency"
```

---

**Document Control**:
- **Version**: 1.0.0
- **Last Updated**: 2025-11-10
- **Next Review**: 2025-12-10
- **Owner**: SRE Team
- **Approver**: VP Engineering

---

*This document is classified as Internal Use.*
