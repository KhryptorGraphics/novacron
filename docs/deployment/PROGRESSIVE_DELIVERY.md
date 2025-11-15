# Progressive Delivery Configuration - DWCP Systems

## Overview

This document defines the progressive delivery framework for DWCP production rollouts, including feature flags, user segmentation, ring deployment, blue-green deployment, and shadow traffic strategies.

---

## Progressive Delivery Strategy

### Core Principles

1. **Gradual Rollout:** Incrementally increase user exposure
2. **Targeted Delivery:** Route specific users to new features
3. **Risk Mitigation:** Limit blast radius at each stage
4. **Data-Driven:** Metrics-based promotion decisions
5. **Rapid Rollback:** Instant feature disablement via flags

### Delivery Stages

```yaml
progressive-delivery-stages:
  1-shadow:
    traffic: 0% (observation only)
    duration: 3-7 days
    purpose: validate with real data, zero user impact

  2-internal:
    traffic: internal users only
    duration: 2-3 days
    purpose: dogfooding, early feedback

  3-beta:
    traffic: beta users (opt-in)
    duration: 1 week
    purpose: enthusiast validation

  4-canary-1:
    traffic: 1% random users
    duration: 2-3 days
    purpose: small-scale validation

  5-canary-5:
    traffic: 5% random users
    duration: 4-7 days
    purpose: statistical significance

  6-progressive-25:
    traffic: 25% random users
    duration: 1 week
    purpose: large-scale validation

  7-progressive-50:
    traffic: 50% random users
    duration: 3-5 days
    purpose: majority validation

  8-full-rollout:
    traffic: 100% users
    duration: ongoing
    purpose: complete migration
```

---

## Feature Flags

### LaunchDarkly Integration

#### Configuration

```yaml
# feature-flags.yml
feature-flags:
  dwcp-manager-v2:
    type: multivariate
    variations:
      - name: "baseline"
        value: "v1"
        description: "Current production version"

      - name: "canary"
        value: "v2"
        description: "New canary version"

    targeting:
      rules:
        # Rule 1: Internal users always get canary
        - name: "Internal Users"
          clauses:
            - attribute: "email"
              op: "endsWith"
              values: ["@company.com"]
          variation: "canary"

        # Rule 2: Beta users (opt-in)
        - name: "Beta Users"
          clauses:
            - attribute: "betaOptIn"
              op: "equals"
              values: [true]
          variation: "canary"

        # Rule 3: Geographic rollout (US first)
        - name: "US Users"
          clauses:
            - attribute: "country"
              op: "equals"
              values: ["US"]
          rollout:
            variations:
              - variation: "canary"
                weight: 25000  # 25%
              - variation: "baseline"
                weight: 75000  # 75%

        # Rule 4: Premium tier users (higher risk tolerance)
        - name: "Premium Users"
          clauses:
            - attribute: "tier"
              op: "equals"
              values: ["premium"]
          rollout:
            variations:
              - variation: "canary"
                weight: 10000  # 10%
              - variation: "baseline"
                weight: 90000  # 90%

      # Default rule: standard users
      fallthrough:
        rollout:
          variations:
            - variation: "canary"
              weight: 5000   # 5%
            - variation: "baseline"
              weight: 95000  # 95%

    # Default variation (if targeting fails)
    offVariation: "baseline"

    # Kill switch
    on: true  # Set to false to disable canary immediately

    # Prerequisites (dependencies on other flags)
    prerequisites:
      - key: "dwcp-infrastructure-ready"
        variation: "enabled"

    # Metrics to track
    metrics:
      - key: "error-rate"
        kind: "custom"
      - key: "latency-p99"
        kind: "custom"
      - key: "user-satisfaction"
        kind: "custom"

  # Additional feature flags
  compression-selector-ml:
    type: boolean
    variations:
      - name: "disabled"
        value: false
      - name: "enabled"
        value: true

    targeting:
      rules:
        - name: "High Volume Services"
          clauses:
            - attribute: "service"
              op: "in"
              values: ["api-gateway", "data-pipeline"]
          variation: "enabled"

      fallthrough:
        variation: "disabled"

  probft-consensus:
    type: boolean
    variations:
      - name: "disabled"
        value: false
      - name: "enabled"
        value: true

    targeting:
      rules:
        - name: "Byzantine Protection Required"
          clauses:
            - attribute: "requiresByzantineProtection"
              op: "equals"
              values: [true]
          variation: "enabled"

      fallthrough:
        rollout:
          variations:
            - variation: "enabled"
              weight: 10000  # 10%
            - variation: "disabled"
              weight: 90000  # 90%
```

#### Application Integration (Go)

```go
package main

import (
    "context"
    "log"

    ld "github.com/launchdarkly/go-server-sdk/v6"
    "github.com/launchdarkly/go-server-sdk/v6/ldcomponents"
)

// LaunchDarkly client (singleton)
var ldClient *ld.LDClient

func initFeatureFlags() {
    config := ld.Config{
        Events: ldcomponents.SendEvents(),
    }

    client, err := ld.MakeCustomClient("YOUR_SDK_KEY", config, 5*time.Second)
    if err != nil {
        log.Fatal("Failed to initialize LaunchDarkly:", err)
    }

    ldClient = client
}

// User context for feature flag evaluation
type User struct {
    Email  string
    UserID string
    Tier   string
    BetaOptIn bool
    Country string
}

func (u *User) ToLDContext() ld.Context {
    builder := ldcontext.NewBuilder(u.UserID)
    builder.SetString("email", u.Email)
    builder.SetString("tier", u.Tier)
    builder.SetBool("betaOptIn", u.BetaOptIn)
    builder.SetString("country", u.Country)
    return builder.Build()
}

// Check which DWCP version to use
func getDWCPVersion(user *User) string {
    context := user.ToLDContext()

    version, err := ldClient.StringVariation("dwcp-manager-v2", context, "v1")
    if err != nil {
        log.Printf("Error evaluating flag: %v", err)
        return "v1"  // Safe default
    }

    return version
}

// HTTP handler using feature flag
func handleRequest(w http.ResponseWriter, r *http.Request) {
    // Get user from request
    user := getUserFromRequest(r)

    // Check feature flag
    version := getDWCPVersion(user)

    // Route to appropriate version
    if version == "v2" {
        canaryHandler(w, r)
    } else {
        baselineHandler(w, r)
    }

    // Track metrics
    ldClient.TrackEvent(context.Background(), user.ToLDContext(), "request-handled", nil)
}

// Kill switch (emergency disable)
func isCanaryEnabled() bool {
    // Check if canary is enabled at all
    enabled, _ := ldClient.BoolVariation("dwcp-canary-enabled", ldcontext.New("system"), true)
    return enabled
}
```

#### Feature Flag Dashboard

```yaml
# Grafana dashboard for feature flags
dashboard: feature-flags-monitoring

panels:
  - title: "Flag Variations Distribution"
    type: pie-chart
    query: |
      sum by (variation) (
        feature_flag_evaluation_total{flag="dwcp-manager-v2"}
      )

  - title: "Flag Evaluation Rate"
    type: graph
    query: |
      sum(rate(feature_flag_evaluation_total{flag="dwcp-manager-v2"}[5m])) by (variation)

  - title: "Flag Evaluation Errors"
    type: stat
    query: |
      sum(increase(feature_flag_evaluation_errors_total[1h]))
    threshold:
      - value: 0
        color: green
      - value: 10
        color: red

  - title: "Kill Switch Status"
    type: stat
    query: |
      feature_flag_enabled{flag="dwcp-canary-enabled"}
```

---

## User Segmentation

### Segmentation Strategy

```yaml
user-segments:
  internal-users:
    description: "Company employees"
    criteria:
      - email endsWith "@company.com"
    rollout-strategy: "100% canary (dogfooding)"
    size: ~500 users

  beta-users:
    description: "Opt-in beta testers"
    criteria:
      - betaOptIn == true
    rollout-strategy: "100% canary"
    size: ~5,000 users

  premium-users:
    description: "Premium tier customers"
    criteria:
      - tier == "premium"
    rollout-strategy: "10% canary initially, progressive"
    size: ~50,000 users

  enterprise-users:
    description: "Enterprise customers"
    criteria:
      - tier == "enterprise"
    rollout-strategy: "5% canary (risk-averse)"
    size: ~10,000 users

  standard-users:
    description: "Standard tier customers"
    criteria:
      - tier == "standard"
    rollout-strategy: "5% canary, progressive to 100%"
    size: ~1,000,000 users

  geographic-segments:
    us-users:
      rollout-order: 1
      criteria: country == "US"
      size: ~500,000 users

    eu-users:
      rollout-order: 2
      criteria: country in ["UK", "DE", "FR", "ES", "IT"]
      size: ~300,000 users

    apac-users:
      rollout-order: 3
      criteria: country in ["JP", "AU", "SG", "IN"]
      size: ~200,000 users

    rest-of-world:
      rollout-order: 4
      criteria: country not in above
      size: ~100,000 users
```

### Segment-Based Rollout

```go
// Determine canary percentage based on user segment
func getCanaryPercentage(user *User) float64 {
    // Internal users: 100%
    if strings.HasSuffix(user.Email, "@company.com") {
        return 1.0
    }

    // Beta users: 100%
    if user.BetaOptIn {
        return 1.0
    }

    // Premium users: 10%
    if user.Tier == "premium" {
        return 0.10
    }

    // Enterprise users: 5%
    if user.Tier == "enterprise" {
        return 0.05
    }

    // Standard users: 5%
    return 0.05
}

// Smart routing based on user segment
func routeRequest(user *User) string {
    canaryPct := getCanaryPercentage(user)

    // Consistent hashing (same user always gets same version)
    hash := hashUserID(user.UserID)
    threshold := uint64(float64(math.MaxUint64) * canaryPct)

    if hash < threshold {
        return "canary"
    }
    return "baseline"
}

func hashUserID(userID string) uint64 {
    h := fnv.New64a()
    h.Write([]byte(userID))
    return h.Sum64()
}
```

---

## Ring Deployment

### Ring Structure

```yaml
ring-deployment:
  ring-0-canaries:
    description: "Internal dev team"
    size: 50 users
    duration: 1 day
    criteria:
      - team == "engineering"

  ring-1-early-adopters:
    description: "Beta users + premium tier"
    size: 5,000 users
    duration: 3 days
    criteria:
      - betaOptIn == true OR tier == "premium"

  ring-2-fast-adopters:
    description: "20% of users (random)"
    size: 200,000 users
    duration: 1 week
    criteria:
      - random 20%

  ring-3-majority:
    description: "80% of users"
    size: 800,000 users
    duration: 1 week
    criteria:
      - random 80%

  ring-4-all:
    description: "100% of users"
    size: 1,000,000 users
    duration: ongoing
    criteria:
      - all users
```

### Ring Progression

```yaml
ring-progression:
  criteria:
    - zero critical issues in current ring
    - error rate < 2% in current ring
    - latency p99 within 10% of baseline
    - user satisfaction ≥ baseline

  promotion-schedule:
    ring-0-to-ring-1: automatic after 24h if criteria met
    ring-1-to-ring-2: manual approval required
    ring-2-to-ring-3: automatic after 7d if criteria met
    ring-3-to-ring-4: manual approval required

  rollback-triggers:
    - critical issue in any ring → rollback entire deployment
    - error rate >5% → pause progression
    - user complaints >10 → pause and investigate
```

---

## Blue-Green Deployment

### Architecture

```yaml
blue-green-deployment:
  blue-environment:
    description: "Current production (stable)"
    version: v1
    traffic: 100% initially
    state: active

  green-environment:
    description: "New version (canary)"
    version: v2
    traffic: 0% initially
    state: standby

  load-balancer:
    type: kubernetes-service
    traffic-routing: weighted
    cutover-method: instant-switch

  database:
    strategy: backward-compatible-schema
    migration: before-cutover
```

### Deployment Flow

```yaml
blue-green-flow:
  step-1-prepare-green:
    - deploy v2 to green environment
    - run smoke tests
    - verify health checks
    duration: 30 minutes

  step-2-validate-green:
    - send synthetic traffic to green
    - run integration tests
    - verify database migrations
    duration: 1 hour

  step-3-shadow-traffic:
    - mirror production traffic to green
    - observe metrics (zero user impact)
    duration: 24 hours

  step-4-progressive-cutover:
    - route 1% traffic to green
    - monitor for 1 hour
    - increase to 5%, 25%, 50%, 100%
    duration: 1 week

  step-5-full-cutover:
    - 100% traffic on green
    - blue on standby (warm)
    duration: 1 week

  step-6-decommission-blue:
    - scale down blue to 0
    - archive blue artifacts
    - complete migration
```

### Kubernetes Configuration

```yaml
# Service with weighted routing
apiVersion: v1
kind: Service
metadata:
  name: dwcp-service
spec:
  selector:
    app: dwcp
  ports:
    - port: 80
      targetPort: 8080

---
# Blue Deployment (Current)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dwcp-blue
  labels:
    version: v1
    environment: blue
spec:
  replicas: 10
  selector:
    matchLabels:
      app: dwcp
      version: v1
  template:
    metadata:
      labels:
        app: dwcp
        version: v1
    spec:
      containers:
        - name: dwcp
          image: dwcp:v1
          ports:
            - containerPort: 8080

---
# Green Deployment (Canary)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dwcp-green
  labels:
    version: v2
    environment: green
spec:
  replicas: 10
  selector:
    matchLabels:
      app: dwcp
      version: v2
  template:
    metadata:
      labels:
        app: dwcp
        version: v2
    spec:
      containers:
        - name: dwcp
          image: dwcp:v2
          ports:
            - containerPort: 8080

---
# Istio VirtualService for traffic splitting
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: dwcp-vs
spec:
  hosts:
    - dwcp-service
  http:
    - match:
        - headers:
            x-user-segment:
              exact: "internal"
      route:
        - destination:
            host: dwcp-service
            subset: green
          weight: 100

    - route:
        - destination:
            host: dwcp-service
            subset: blue
          weight: 95
        - destination:
            host: dwcp-service
            subset: green
          weight: 5

---
# DestinationRule for subsets
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: dwcp-dr
spec:
  host: dwcp-service
  subsets:
    - name: blue
      labels:
        version: v1

    - name: green
      labels:
        version: v2
```

### Cutover Script

```bash
#!/bin/bash
# blue-green-cutover.sh - Progressive cutover from blue to green

set -e

PERCENTAGES=(1 5 10 25 50 75 90 100)

for pct in "${PERCENTAGES[@]}"; do
    echo "Cutting over ${pct}% traffic to green..."

    # Update VirtualService weights
    kubectl patch virtualservice dwcp-vs -p "{
      \"spec\": {
        \"http\": [{
          \"route\": [
            {\"destination\": {\"host\": \"dwcp-service\", \"subset\": \"blue\"}, \"weight\": $((100 - pct))},
            {\"destination\": {\"host\": \"dwcp-service\", \"subset\": \"green\"}, \"weight\": ${pct}}
          ]
        }]
      }
    }"

    echo "✓ ${pct}% traffic on green"

    # Monitor for issues
    echo "Monitoring for 10 minutes..."
    for i in {1..10}; do
        sleep 60

        # Check error rate
        error_rate=$(curl -s "http://prometheus:9090/api/v1/query?query=rate(http_requests_total{status=~\"5..\",version=\"v2\"}[5m])" | jq -r '.data.result[0].value[1]')

        if (( $(echo "$error_rate > 0.03" | bc -l) )); then
            echo "ERROR: High error rate detected ($error_rate), rolling back..."
            ./scripts/rollback-to-blue.sh
            exit 1
        fi

        echo "  Minute $i/10 - Error rate: $error_rate (OK)"
    done

    echo "✓ ${pct}% cutover successful\n"
done

echo "Blue-green cutover completed successfully!"
```

---

## Shadow Traffic

### Shadow Testing Strategy

```yaml
shadow-traffic:
  purpose: "Validate new version with zero user impact"

  architecture:
    - production-traffic → baseline (responds to users)
    - production-traffic → canary (shadow, observe only)

  response-handling:
    - baseline responses: returned to users
    - canary responses: logged for analysis, discarded

  metrics-comparison:
    - latency: canary vs baseline
    - error-rate: canary vs baseline
    - resource-usage: canary vs baseline
    - behavior-diff: response body comparison

  duration: 3-7 days

  success-criteria:
    - canary latency within 20% of baseline
    - canary error rate ≤ baseline
    - zero critical errors in canary
    - behavior differences explained
```

### Istio Shadow Traffic Configuration

```yaml
# Istio VirtualService for traffic mirroring
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: dwcp-shadow
spec:
  hosts:
    - dwcp-service
  http:
    - route:
        - destination:
            host: dwcp-baseline
          weight: 100

      mirror:
        host: dwcp-canary
      mirrorPercentage:
        value: 100.0  # Mirror 100% of traffic

---
# Service for baseline (production)
apiVersion: v1
kind: Service
metadata:
  name: dwcp-baseline
spec:
  selector:
    app: dwcp
    version: v1
  ports:
    - port: 80
      targetPort: 8080

---
# Service for canary (shadow)
apiVersion: v1
kind: Service
metadata:
  name: dwcp-canary
spec:
  selector:
    app: dwcp
    version: v2
  ports:
    - port: 80
      targetPort: 8080
```

### Shadow Traffic Analysis

```python
#!/usr/bin/env python3
"""
Shadow Traffic Analyzer
Compares baseline vs canary behavior in shadow mode.
"""

import time
import requests
from collections import defaultdict

class ShadowAnalyzer:
    def __init__(self, prometheus_url):
        self.prometheus_url = prometheus_url

    def query(self, query):
        """Query Prometheus"""
        response = requests.get(
            f"{self.prometheus_url}/api/v1/query",
            params={"query": query}
        )
        data = response.json()
        if data["status"] == "success" and data["data"]["result"]:
            return float(data["data"]["result"][0]["value"][1])
        return 0.0

    def analyze_shadow_traffic(self):
        """Analyze shadow traffic metrics"""
        print("Shadow Traffic Analysis Report")
        print("=" * 60)

        # Latency comparison
        baseline_p99 = self.query(
            'histogram_quantile(0.99, '
            'sum(rate(http_request_duration_seconds_bucket{version="v1"}[5m])) by (le))'
        )
        canary_p99 = self.query(
            'histogram_quantile(0.99, '
            'sum(rate(http_request_duration_seconds_bucket{version="v2"}[5m])) by (le))'
        )

        latency_diff_pct = ((canary_p99 - baseline_p99) / baseline_p99) * 100 if baseline_p99 > 0 else 0

        print(f"\n📊 Latency Analysis:")
        print(f"  Baseline P99: {baseline_p99:.3f}s")
        print(f"  Canary P99:   {canary_p99:.3f}s")
        print(f"  Difference:   {latency_diff_pct:+.1f}%")

        if abs(latency_diff_pct) < 10:
            print(f"  Status: ✅ PASS (within 10%)")
        elif abs(latency_diff_pct) < 20:
            print(f"  Status: ⚠️  WARNING (10-20% difference)")
        else:
            print(f"  Status: ❌ FAIL (>20% difference)")

        # Error rate comparison
        baseline_errors = self.query(
            'sum(rate(http_requests_total{status=~"5..",version="v1"}[5m])) / '
            'sum(rate(http_requests_total{version="v1"}[5m]))'
        )
        canary_errors = self.query(
            'sum(rate(http_requests_total{status=~"5..",version="v2"}[5m])) / '
            'sum(rate(http_requests_total{version="v2"}[5m]))'
        )

        print(f"\n❌ Error Rate Analysis:")
        print(f"  Baseline Errors: {baseline_errors:.2%}")
        print(f"  Canary Errors:   {canary_errors:.2%}")

        if canary_errors <= baseline_errors * 1.1:
            print(f"  Status: ✅ PASS (canary ≤ baseline)")
        else:
            print(f"  Status: ❌ FAIL (canary > baseline)")

        # Resource usage comparison
        baseline_cpu = self.query('avg(rate(container_cpu_usage_seconds_total{version="v1"}[5m]))')
        canary_cpu = self.query('avg(rate(container_cpu_usage_seconds_total{version="v2"}[5m]))')

        print(f"\n💻 Resource Usage Analysis:")
        print(f"  Baseline CPU: {baseline_cpu:.3f} cores")
        print(f"  Canary CPU:   {canary_cpu:.3f} cores")
        print(f"  Difference:   {((canary_cpu - baseline_cpu) / baseline_cpu * 100):+.1f}%")

        # Overall recommendation
        print(f"\n🎯 Recommendation:")
        if latency_diff_pct < 20 and canary_errors <= baseline_errors * 1.1:
            print("  ✅ Canary ready for real traffic rollout")
        else:
            print("  ❌ Canary needs optimization before rollout")

if __name__ == "__main__":
    analyzer = ShadowAnalyzer("http://prometheus:9090")
    analyzer.analyze_shadow_traffic()
```

---

## Progressive Delivery Metrics

### Key Metrics

```yaml
progressive-delivery-metrics:
  adoption-rate:
    description: "Percentage of users on new version"
    query: |
      sum(feature_flag_evaluation_total{variation="canary"}) /
      sum(feature_flag_evaluation_total)

  rollout-velocity:
    description: "Rate of traffic increase over time"
    query: |
      rate(feature_flag_evaluation_total{variation="canary"}[1h])

  segment-coverage:
    description: "Percentage of segments rolled out"
    calculation: manual tracking

  rollback-rate:
    description: "Frequency of rollbacks"
    query: |
      sum(increase(rollback_executed_total[7d]))

  time-to-full-rollout:
    description: "Duration from 0% to 100%"
    calculation: manual tracking
```

---

## Conclusion

This progressive delivery framework provides:

1. ✅ **Feature Flags:** Instant rollback capability
2. ✅ **User Segmentation:** Targeted rollout strategies
3. ✅ **Ring Deployment:** Gradual expansion model
4. ✅ **Blue-Green:** Zero-downtime cutover
5. ✅ **Shadow Traffic:** Risk-free validation
6. ✅ **Metrics-Driven:** Data-based promotion decisions

**Expected Outcome:** Safe, controlled rollout with <0.1% user impact and instant rollback capability.
