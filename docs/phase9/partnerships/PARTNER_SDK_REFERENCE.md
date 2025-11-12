# NovaCron Partner SDK Reference Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Authentication](#authentication)
4. [Core Concepts](#core-concepts)
5. [API Reference](#api-reference)
6. [Webhook Integration](#webhook-integration)
7. [Rate Limiting](#rate-limiting)
8. [Best Practices](#best-practices)
9. [Examples](#examples)

## Introduction

The NovaCron Partner SDK enables third-party developers and technology partners to build integrations with NovaCron DWCP v3. The SDK provides a comprehensive set of APIs for VM management, migration orchestration, and platform monitoring.

### Supported Languages

- Go (native)
- Python (via gRPC)
- Node.js (via REST API)
- Java (via gRPC)

### SDK Features

- Partner authentication and authorization
- VM lifecycle management
- Migration orchestration
- Webhook event subscriptions
- Quota and billing management
- Rate limiting and throttling
- Comprehensive error handling

## Getting Started

### Installation

#### Go

```bash
go get github.com/novacron/sdk/partners
```

#### Python

```bash
pip install novacron-partner-sdk
```

#### Node.js

```bash
npm install @novacron/partner-sdk
```

### Quick Start

```go
package main

import (
    "context"
    "github.com/novacron/sdk/partners"
)

func main() {
    // Initialize SDK
    sdk, err := partners.NewSDK(partners.SDKConfig{
        BaseURL:      "https://api.novacron.io",
        PartnerID:    "partner-12345",
        APIKey:       "your-api-key",
        APISecret:    "your-api-secret",
        RateLimitRPS: 100,
    })
    if err != nil {
        panic(err)
    }

    // Get partner information
    partner, err := sdk.GetPartner(context.Background())
    if err != nil {
        panic(err)
    }

    println("Partner:", partner.Name)
}
```

## Authentication

### API Key Authentication

NovaCron uses HMAC-based API key authentication:

```
X-Partner-ID: partner-12345
X-API-Key: your-api-key
X-Timestamp: 1640995200
X-Signature: hmac-sha256-signature
```

### Signature Generation

```go
func generateSignature(method, path string, timestamp int64, secret string) string {
    message := fmt.Sprintf("%s:%s:%d", method, path, timestamp)
    h := hmac.New(sha256.New, []byte(secret))
    h.Write([]byte(message))
    return hex.EncodeToString(h.Sum(nil))
}
```

### OAuth 2.0

For user-facing integrations, use OAuth 2.0:

```go
oauth := partners.NewOAuthProvider(partners.OAuthConfig{
    ClientID:     "your-client-id",
    ClientSecret: "your-client-secret",
    RedirectURL:  "https://yourapp.com/callback",
    Scopes:       []string{"vm:read", "vm:write", "migration:execute"},
})

// Get authorization URL
authURL := oauth.GetAuthorizationURL()

// Exchange code for token
token, err := oauth.ExchangeCode(code)
```

## Core Concepts

### Partners

A **Partner** represents your organization in the NovaCron ecosystem.

```go
type Partner struct {
    PartnerID     string
    Name          string
    Type          string // technology, reseller, managed_service, oem
    Tier          string // bronze, silver, gold, platinum
    Entitlements  []string
    QuotaLimits   map[string]int64
}
```

### Integrations

An **Integration** represents a specific connection between your product and NovaCron.

```go
type Integration struct {
    IntegrationID string
    PartnerID     string
    Name          string
    Type          string // api, webhook, oauth, saml
    Status        string
    Config        map[string]interface{}
    Endpoints     []Endpoint
}
```

### Quotas

Partners have quota limits for API calls and resource usage:

```go
// Check quota before operation
allowed, err := sdk.CheckQuota(ctx, "vm-create", 10)
if !allowed {
    return errors.New("quota exceeded")
}
```

## API Reference

### Partner Management

#### Get Partner Information

```go
partner, err := sdk.GetPartner(ctx)
```

**Response:**
```json
{
  "partner_id": "partner-12345",
  "name": "Acme Corp",
  "type": "technology",
  "tier": "gold",
  "status": "active",
  "entitlements": [
    "advanced-features",
    "premium-support",
    "white-label"
  ],
  "quota_limits": {
    "api_calls_per_minute": 1000,
    "vm_create_per_day": 1000,
    "migration_concurrent": 50
  }
}
```

### Integration Management

#### Create Integration

```go
integration := partners.Integration{
    Name: "VM Backup Integration",
    Type: "api",
    Config: map[string]interface{}{
        "backup_schedule": "0 2 * * *",
        "retention_days":  30,
    },
    Endpoints: []partners.Endpoint{
        {
            Name:     "backup-webhook",
            URL:      "https://backup.acme.com/webhook",
            Method:   "POST",
            AuthType: "bearer",
        },
    },
}

created, err := sdk.CreateIntegration(ctx, integration)
```

#### List Integrations

```go
integrations, err := sdk.ListIntegrations(ctx)
```

#### Update Integration

```go
err := sdk.UpdateIntegration(ctx, integrationID, updates)
```

#### Delete Integration

```go
err := sdk.DeleteIntegration(ctx, integrationID)
```

### VM Management

#### List VMs

```go
vms, err := sdk.ListVMs(ctx, partners.VMListOptions{
    Status: "running",
    Tags:   map[string]string{"env": "production"},
    Limit:  100,
})
```

#### Get VM Details

```go
vm, err := sdk.GetVM(ctx, vmID)
```

#### Create VM

```go
vm := partners.VMSpec{
    Name:         "web-server-01",
    Template:     "ubuntu-22.04",
    CPUCores:     4,
    MemoryGB:     16,
    DiskSizeGB:   100,
    NetworkID:    "net-12345",
    CloudProvider: "aws",
    Region:       "us-east-1",
    Tags: map[string]string{
        "app": "web",
        "env": "prod",
    },
}

created, err := sdk.CreateVM(ctx, vm)
```

#### Delete VM

```go
err := sdk.DeleteVM(ctx, vmID)
```

### Migration Management

#### Start Migration

```go
migration := partners.MigrationSpec{
    SourceVMID:    "vm-source-123",
    TargetCluster: "cluster-456",
    MigrationType: "live",
    Options: map[string]interface{}{
        "downtime_window_minutes": 5,
        "verify_after_migration":  true,
    },
}

result, err := sdk.StartMigration(ctx, migration)
```

#### Get Migration Status

```go
status, err := sdk.GetMigrationStatus(ctx, migrationID)
```

**Response:**
```json
{
  "migration_id": "mig-789",
  "status": "in_progress",
  "phase": "data_sync",
  "progress_percentage": 67,
  "bytes_transferred": 45678901234,
  "estimated_completion": "2024-01-15T10:30:00Z",
  "metrics": {
    "transfer_rate_mbps": 850,
    "source_vm_load": 0.15,
    "target_vm_load": 0.12
  }
}
```

#### Cancel Migration

```go
err := sdk.CancelMigration(ctx, migrationID)
```

### Metrics & Monitoring

#### Get System Metrics

```go
metrics, err := sdk.GetMetrics(ctx, partners.MetricsQuery{
    Resource:   "vm",
    ResourceID: vmID,
    Metrics:    []string{"cpu_usage", "memory_usage", "disk_io"},
    StartTime:  time.Now().Add(-1 * time.Hour),
    EndTime:    time.Now(),
    Interval:   "1m",
})
```

#### Create Custom Metric

```go
err := sdk.SendMetric(ctx, partners.Metric{
    Name:      "custom.backup.duration",
    Value:     125.5,
    Unit:      "seconds",
    Timestamp: time.Now(),
    Tags: map[string]string{
        "backup_type": "incremental",
        "vm_id":       vmID,
    },
})
```

## Webhook Integration

### Subscribe to Events

```go
subscription, err := sdk.SubscribeWebhook(ctx,
    []string{
        "vm.created",
        "vm.deleted",
        "migration.started",
        "migration.completed",
        "migration.failed",
    },
    "https://yourapp.com/webhooks/novacron",
    "your-webhook-secret",
)
```

### Event Types

| Event Type | Description |
|------------|-------------|
| `vm.created` | New VM created |
| `vm.deleted` | VM deleted |
| `vm.started` | VM started |
| `vm.stopped` | VM stopped |
| `migration.started` | Migration started |
| `migration.completed` | Migration completed successfully |
| `migration.failed` | Migration failed |
| `alert.critical` | Critical alert triggered |
| `quota.exceeded` | Quota limit exceeded |

### Webhook Payload

```json
{
  "event_id": "evt-12345",
  "event_type": "migration.completed",
  "timestamp": "2024-01-15T10:00:00Z",
  "partner_id": "partner-12345",
  "data": {
    "migration_id": "mig-789",
    "source_vm_id": "vm-123",
    "target_cluster_id": "cluster-456",
    "duration_seconds": 180,
    "bytes_transferred": 50000000000,
    "status": "success"
  },
  "signature": "sha256-hmac-signature"
}
```

### Verify Webhook Signature

```go
func verifyWebhook(payload []byte, signature string, secret string) bool {
    h := hmac.New(sha256.New, []byte(secret))
    h.Write(payload)
    expected := hex.EncodeToString(h.Sum(nil))
    return hmac.Equal([]byte(expected), []byte(signature))
}
```

### Webhook Handler Example

```go
http.HandleFunc("/webhooks/novacron", func(w http.ResponseWriter, r *http.Request) {
    payload, _ := ioutil.ReadAll(r.Body)
    signature := r.Header.Get("X-Webhook-Signature")

    if !verifyWebhook(payload, signature, webhookSecret) {
        http.Error(w, "Invalid signature", http.StatusUnauthorized)
        return
    }

    var event partners.WebhookEvent
    json.Unmarshal(payload, &event)

    switch event.EventType {
    case "migration.completed":
        handleMigrationCompleted(event.Data)
    case "vm.created":
        handleVMCreated(event.Data)
    }

    w.WriteHeader(http.StatusOK)
})
```

## Rate Limiting

### Rate Limits

| Tier | Requests/Minute | Burst |
|------|----------------|-------|
| Bronze | 100 | 150 |
| Silver | 500 | 750 |
| Gold | 1000 | 1500 |
| Platinum | 5000 | 7500 |

### Handling Rate Limits

```go
err := sdk.CreateVM(ctx, vm)
if err != nil {
    if errors.Is(err, partners.ErrRateLimitExceeded) {
        // Wait and retry
        time.Sleep(1 * time.Second)
        err = sdk.CreateVM(ctx, vm)
    }
}
```

### Rate Limit Headers

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 750
X-RateLimit-Reset: 1640995200
```

## Best Practices

### 1. Error Handling

```go
vm, err := sdk.GetVM(ctx, vmID)
if err != nil {
    switch {
    case errors.Is(err, partners.ErrNotFound):
        // VM doesn't exist
    case errors.Is(err, partners.ErrUnauthorized):
        // Invalid credentials
    case errors.Is(err, partners.ErrRateLimitExceeded):
        // Rate limit exceeded
    default:
        // Other error
    }
}
```

### 2. Pagination

```go
var allVMs []partners.VM
page := 1
limit := 100

for {
    vms, hasMore, err := sdk.ListVMsPaginated(ctx, page, limit)
    if err != nil {
        return err
    }

    allVMs = append(allVMs, vms...)

    if !hasMore {
        break
    }
    page++
}
```

### 3. Concurrent Operations

```go
var wg sync.WaitGroup
semaphore := make(chan struct{}, 10) // Max 10 concurrent operations

for _, vmID := range vmIDs {
    wg.Add(1)
    semaphore <- struct{}{}

    go func(id string) {
        defer wg.Done()
        defer func() { <-semaphore }()

        vm, err := sdk.GetVM(ctx, id)
        // Process vm...
    }(vmID)
}

wg.Wait()
```

### 4. Timeout Handling

```go
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

vm, err := sdk.GetVM(ctx, vmID)
if err != nil {
    if errors.Is(err, context.DeadlineExceeded) {
        // Timeout occurred
    }
}
```

## Examples

### Complete Integration Example

```go
package main

import (
    "context"
    "fmt"
    "log"
    "time"

    "github.com/novacron/sdk/partners"
)

func main() {
    // Initialize SDK
    sdk, err := partners.NewSDK(partners.SDKConfig{
        BaseURL:      "https://api.novacron.io",
        PartnerID:    "partner-12345",
        APIKey:       "your-api-key",
        APISecret:    "your-api-secret",
        RateLimitRPS: 100,
    })
    if err != nil {
        log.Fatal(err)
    }

    ctx := context.Background()

    // Subscribe to webhooks
    _, err = sdk.SubscribeWebhook(ctx,
        []string{"migration.completed", "migration.failed"},
        "https://yourapp.com/webhooks",
        "webhook-secret",
    )
    if err != nil {
        log.Fatal(err)
    }

    // Start migration
    migration := partners.MigrationSpec{
        SourceVMID:    "vm-123",
        TargetCluster: "cluster-456",
        MigrationType: "live",
    }

    result, err := sdk.StartMigration(ctx, migration)
    if err != nil {
        log.Fatal(err)
    }

    // Monitor migration progress
    for {
        status, err := sdk.GetMigrationStatus(ctx, result.MigrationID)
        if err != nil {
            log.Fatal(err)
        }

        fmt.Printf("Migration %s: %d%% complete\n",
            status.Status, status.ProgressPercentage)

        if status.Status == "completed" || status.Status == "failed" {
            break
        }

        time.Sleep(5 * time.Second)
    }
}
```

## Support

- Documentation: https://docs.novacron.io/partners
- API Reference: https://api.novacron.io/docs
- Support Email: partner-support@novacron.io
- Partner Portal: https://partner.novacron.io

## License

Copyright © 2024 NovaCron Technologies. All rights reserved.
