# NovaCron Quota Management System

A comprehensive, enterprise-grade resource quota and limits system designed for multi-tenant cloud environments. This system provides real-time resource tracking, cost management, policy enforcement, and advanced analytics for optimal resource utilization.

## Architecture Overview

The quota management system is built with a modular architecture consisting of several key components:

- **Quota Manager**: Core quota lifecycle management and enforcement
- **Enforcement Engine**: High-performance quota checking with caching and circuit breakers
- **Analytics Engine**: Advanced usage analytics, forecasting, and anomaly detection
- **Template Manager**: Predefined quota templates for different service tiers
- **Integration Layer**: Seamless integration with NovaCron's VM, storage, and network systems
- **Dashboard API**: RESTful API for management dashboards and UI integration

## Key Features

### 🎯 Multi-Level Quota Management
- **System-level**: Global resource limits across the entire platform
- **Tenant-level**: Organization or customer-specific quotas
- **Project-level**: Application or environment-specific limits
- **User-level**: Individual user resource quotas

### 📊 Resource Types Supported
- **Compute**: CPU cores, memory, GPU units, VM instances
- **Storage**: Disk space, volumes, snapshots, backups
- **Network**: Bandwidth, connections, load balancers, floating IPs
- **API**: Request rates, API calls per period
- **Cost**: Budget limits, cost tracking per resource

### ⚡ Performance & Scalability
- **High-throughput enforcement**: >10,000 checks/second with caching
- **Circuit breakers**: Prevent cascading failures in enforcement
- **Rate limiting**: Token bucket rate limiting for resource requests
- **Batch operations**: Efficient multi-quota operations
- **Minimal overhead**: <1ms quota checks with cache hits

### 🛡️ Advanced Enforcement
- **Hard limits**: Strict resource limits with denial on breach
- **Soft limits**: Warning-based limits with grace periods
- **Burst capacity**: Temporary resource overages for peak loads
- **Auto-scaling**: Automatic quota adjustments based on usage patterns
- **Policy engine**: Rule-based quota assignment and management

### 📈 Analytics & Insights
- **Usage forecasting**: ML-based resource usage predictions
- **Anomaly detection**: Identify unusual usage patterns
- **Cost optimization**: Recommendations for resource efficiency
- **Trend analysis**: Historical usage patterns and projections
- **Compliance reporting**: Regulatory compliance tracking

### 🏗️ Enterprise Integration
- **RBAC Integration**: Role-based quota management
- **Multi-tenant isolation**: Secure tenant resource separation
- **Audit trails**: Comprehensive logging and compliance tracking
- **Monitoring integration**: Prometheus metrics and alerting
- **Dashboard APIs**: Rich REST APIs for management interfaces

## Quick Start

### Basic Setup

```go
package main

import (
    "context"
    "log"
    "time"
    
    "github.com/novacron/backend/core/quotas"
)

func main() {
    // Create quota manager with default configuration
    config := quotas.DefaultManagerConfig()
    manager := quotas.NewManager(config)
    
    // Start the manager
    if err := manager.Start(); err != nil {
        log.Fatal("Failed to start quota manager:", err)
    }
    defer manager.Stop()
    
    ctx := context.Background()
    
    // Create a tenant-level CPU quota
    cpuQuota := &quotas.Quota{
        Name:         "tenant-1-cpu",
        Level:        quotas.QuotaLevelTenant,
        EntityID:     "tenant-1",
        ResourceType: quotas.ResourceTypeCPU,
        LimitType:    quotas.LimitTypeHard,
        Limit:        100, // 100 vCPUs
        BurstLimit:   150, // Allow burst to 150 vCPUs
        Unit:         "cores",
        Status:       quotas.QuotaStatusActive,
    }
    
    // Create the quota
    if err := manager.CreateQuota(ctx, cpuQuota); err != nil {
        log.Fatal("Failed to create quota:", err)
    }
    
    // Check if we can allocate 50 vCPUs
    result, err := manager.CheckQuota(ctx, "tenant-1", quotas.ResourceTypeCPU, 50)
    if err != nil {
        log.Fatal("Failed to check quota:", err)
    }
    
    if result.Allowed {
        log.Printf("Allocation allowed. Available: %d vCPUs", result.Available)
        
        // Record the resource consumption
        usage := &quotas.UsageRecord{
            EntityID:     "tenant-1",
            ResourceType: quotas.ResourceTypeCPU,
            Amount:       50,
            Delta:        50,
            Source:       "vm-creation",
            Timestamp:    time.Now(),
        }
        
        if err := manager.ConsumeResource(ctx, usage); err != nil {
            log.Fatal("Failed to record usage:", err)
        }
        
        log.Println("Resource consumption recorded successfully")
    } else {
        log.Printf("Allocation denied: %s", result.Reason)
    }
}
```

### Template-Based Quota Setup

```go
// Create template manager
templateManager := quotas.NewTemplateManager(manager)

// Apply a built-in service tier template
err := templateManager.ApplyServiceTierTemplate(
    ctx, 
    quotas.ServiceTierStartup, 
    "tenant-1", 
    quotas.QuotaLevelTenant,
)
if err != nil {
    log.Fatal("Failed to apply template:", err)
}

// Get recommended template based on usage patterns
template, reason, err := templateManager.GetRecommendedTemplate(ctx, "tenant-1")
if err != nil {
    log.Fatal("Failed to get recommendation:", err)
}

log.Printf("Recommended template: %s - %s", template.Name, reason)
```

### Analytics and Reporting

```go
// Create analytics engine
analyticsConfig := &quotas.AnalyticsConfig{
    ForecastingEnabled:      true,
    AnomalyDetectionEnabled: true,
    TrendAnalysisEnabled:    true,
}
analyticsEngine := quotas.NewAnalyticsEngine(manager, analyticsConfig)

// Generate comprehensive analytics
timeRange := quotas.TimeRange{
    Start: time.Now().Add(-30 * 24 * time.Hour), // Last 30 days
    End:   time.Now(),
}

analytics, err := analyticsEngine.GenerateAnalytics(
    ctx, 
    "tenant-1", 
    quotas.QuotaLevelTenant, 
    timeRange,
)
if err != nil {
    log.Fatal("Failed to generate analytics:", err)
}

// Print usage summary
log.Printf("Overall utilization: %.2f%%", analytics.UsageAnalytics.EfficiencyScore)
log.Printf("Total cost: $%.2f", analytics.CostAnalytics.TotalCost)
log.Printf("Projected monthly cost: $%.2f", analytics.CostAnalytics.ProjectedMonthlyCost)

// Print recommendations
for _, rec := range analytics.Recommendations {
    log.Printf("Recommendation: %s - %s", rec.Title, rec.Description)
}
```

### System Integration

```go
// Set up complete system integration
integrationConfig := quotas.DefaultIntegrationConfig()
integration := quotas.NewIntegration(
    manager,           // Quota manager
    authManager,       // Auth system
    vmManager,         // VM management
    storageService,    // Storage system
    schedulerService,  // Scheduler
    monitoringService, // Monitoring
    integrationConfig,
)

// Initialize all integrations
if err := integration.InitializeIntegration(ctx); err != nil {
    log.Fatal("Failed to initialize integration:", err)
}

// Integration automatically handles:
// - VM resource tracking on creation/deletion
// - Storage usage updates on volume operations
// - Network bandwidth monitoring
// - Scheduler quota-aware placement
// - Metrics collection and alerting
```

## Service Tiers & Templates

The system includes built-in templates for common service tiers:

### Free Tier
- **CPU**: 2 cores (burst to 4)
- **Memory**: 4 GB (burst to 6 GB)
- **Storage**: 10 GB
- **Instances**: 3
- **Monthly Cost**: $0

### Developer Tier
- **CPU**: 8 cores (burst to 12)
- **Memory**: 16 GB (burst to 24 GB)
- **Storage**: 100 GB
- **Instances**: 10
- **Monthly Cost**: $29.99

### Startup Tier
- **CPU**: 32 cores (burst to 48)
- **Memory**: 64 GB (burst to 96 GB)
- **Storage**: 500 GB
- **Instances**: 25
- **GPU**: 2 units
- **Monthly Cost**: $99.99

### Growth Tier
- **CPU**: 128 cores (burst to 192)
- **Memory**: 256 GB (burst to 384 GB)
- **Storage**: 2 TB
- **Instances**: 100
- **GPU**: 8 units
- **Load Balancers**: 10
- **Monthly Cost**: $299.99

### Enterprise Tier
- **Soft limits**: Enterprise-friendly with high burst capacity
- **CPU**: 1000 cores (burst to 1500)
- **Memory**: 1 TB (burst to 1.5 TB)
- **Storage**: 10 TB
- **Instances**: 500
- **24/7 Support**: Premium support included
- **Monthly Cost**: $999.99

## API Integration

### RESTful Dashboard API

The system provides comprehensive REST APIs for dashboard integration:

```bash
# Get quota overview
GET /api/v1/quotas/overview?entity_id=tenant-1

# List all quotas with filtering
GET /api/v1/quotas?entity_id=tenant-1&status=active

# Get utilization metrics
GET /api/v1/quotas/utilization?entity_id=tenant-1

# Get usage history
GET /api/v1/quotas/usage/tenant-1/history?resource_type=cpu&start=2024-01-01T00:00:00Z

# Generate analytics report
GET /api/v1/quotas/analytics/tenant-1?start=2024-01-01T00:00:00Z&end=2024-01-31T23:59:59Z

# Apply quota template
POST /api/v1/quotas/templates/startup/apply
{
  "entity_id": "tenant-1",
  "level": "tenant"
}

# Create resource reservation
POST /api/v1/quotas/reservations
{
  "entity_id": "tenant-1",
  "resource_type": "cpu",
  "amount": 100,
  "start_time": "2024-02-01T09:00:00Z",
  "end_time": "2024-02-01T17:00:00Z",
  "purpose": "batch processing"
}
```

### WebSocket Real-time Updates

Real-time quota updates for dashboard integration:

```javascript
const ws = new WebSocket('ws://localhost:8091/api/v1/quotas/live?entity_id=tenant-1');

ws.onmessage = function(event) {
    const update = JSON.parse(event.data);
    
    switch(update.type) {
        case 'quota_exceeded':
            showAlert('Quota Exceeded', update.data);
            break;
        case 'usage_updated':
            updateUtilizationCharts(update.data);
            break;
        case 'cost_alert':
            showCostAlert(update.data);
            break;
    }
};
```

## Configuration

### Manager Configuration

```go
config := &quotas.ManagerConfig{
    DefaultQuotas: map[quotas.ResourceType]int64{
        quotas.ResourceTypeCPU:     100,
        quotas.ResourceTypeMemory:  102400, // 100 GB in MB
        quotas.ResourceTypeStorage: 1048576, // 1 TB in MB
    },
    UsageAggregationInterval: 5 * time.Minute,
    UsageRetentionPeriod:     30 * 24 * time.Hour,
    EnableCostTracking:       true,
    EnableAutoScaling:        true,
    ComplianceEnabled:        true,
}
```

### Enforcement Configuration

```go
enforcementConfig := &quotas.EnforcementConfig{
    CacheEnabled: true,
    CacheTTL:     30 * time.Second,
    RateLimitEnabled: true,
    CircuitBreakerEnabled: true,
    MaxConcurrentChecks: 50,
    CheckTimeout: 5 * time.Second,
}

enforcementEngine := quotas.NewEnforcementEngine(manager, enforcementConfig)
```

### Integration Configuration

```go
integrationConfig := &quotas.IntegrationConfig{
    VMQuotaEnforcement:      true,
    VMResourceTracking:      true,
    StorageQuotaEnforcement: true,
    NetworkQuotaEnforcement: true,
    TenantQuotaInheritance:  true,
    RBACIntegration:         true,
    MetricsCollection:       true,
    AlertsIntegration:       true,
    SchedulerQuotaAware:     true,
}
```

## Monitoring & Alerting

### Metrics Collection

The system exposes comprehensive metrics for monitoring:

- `quota_utilization{entity_id, resource_type, level}`: Current quota utilization percentage
- `quota_usage{entity_id, resource_type}`: Current resource usage
- `quota_limit{entity_id, resource_type}`: Configured quota limit
- `quota_violations_total{entity_id, resource_type}`: Total quota violations
- `quota_check_duration_seconds`: Quota check latency
- `quota_cache_hit_ratio`: Cache hit ratio for quota checks

### Alert Configuration

```go
alerts := []quotas.QuotaAlert{
    {
        ID:        "cpu-warning",
        Threshold: 80.0,
        Severity:  quotas.AlertSeverityWarning,
        Channels:  []string{"email", "slack"},
        MessageTemplate: "CPU usage at {utilization}% for {entity_id}",
        Enabled:   true,
    },
    {
        ID:        "cpu-critical",
        Threshold: 95.0,
        Severity:  quotas.AlertSeverityCritical,
        Channels:  []string{"email", "slack", "pagerduty"},
        MessageTemplate: "CRITICAL: CPU usage at {utilization}% for {entity_id}",
        Enabled:   true,
    },
}
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
go test -v ./...

# Run with race detection
go test -race -v ./...

# Run benchmarks
go test -bench=. -benchmem ./...

# Run specific test categories
go test -v -run TestManager_CheckQuota ./...
go test -v -run TestEnforcement ./...
go test -v -run TestIntegration ./...
```

### Performance Benchmarks

The system has been tested and optimized for high performance:

- **Quota checks**: >10,000/second with caching enabled
- **Cache hit ratio**: >95% in typical workloads
- **Memory usage**: <100MB for 10,000 active quotas
- **Storage overhead**: <1KB per quota record
- **API response time**: <10ms for dashboard queries

## Best Practices

### 1. Quota Design
- Use hierarchical quotas (system → tenant → project → user)
- Set burst limits 20-50% above normal limits
- Use soft limits for development environments
- Implement grace periods for soft limit violations

### 2. Template Usage
- Start with built-in templates for common service tiers
- Create custom templates for specific organizational needs
- Regularly review and update templates based on usage patterns
- Use template recommendations for optimal resource allocation

### 3. Monitoring & Alerting
- Set up alerting at 80% and 95% utilization thresholds
- Monitor quota trends to identify growth patterns
- Use cost alerts to prevent budget overruns
- Implement automated responses for critical alerts

### 4. Integration
- Enable all relevant integrations for comprehensive tracking
- Use quota-aware scheduling for optimal resource placement
- Implement proper RBAC for quota management permissions
- Regular audit trails for compliance requirements

### 5. Performance
- Enable caching for high-throughput environments
- Use batch operations for bulk quota updates
- Monitor enforcement engine metrics for bottlenecks
- Implement proper circuit breakers for reliability

## Troubleshooting

### Common Issues

**High quota check latency**
- Enable caching with appropriate TTL
- Check for database connection issues
- Verify enforcement engine configuration
- Monitor circuit breaker status

**Quota violations not enforced**
- Verify quota status is 'active'
- Check enforcement action configuration
- Review integration hook registration
- Validate entity ID mapping

**Inaccurate usage tracking**
- Check integration event handlers
- Verify usage record timestamps
- Review usage aggregation intervals
- Validate delta calculations

**Dashboard API timeouts**
- Enable request timeout configuration
- Use pagination for large result sets
- Implement proper caching for analytics
- Monitor database query performance

### Debug Mode

Enable debug logging for troubleshooting:

```go
config.DebugEnabled = true
config.LogLevel = "debug"

// Enable detailed enforcement logging
enforcementConfig.DebugMode = true
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-quota-type`)
3. Commit your changes (`git commit -am 'Add support for new quota type'`)
4. Push to the branch (`git push origin feature/new-quota-type`)
5. Create a Pull Request

### Development Guidelines

- Follow Go best practices and conventions
- Add comprehensive tests for new features
- Update documentation for API changes
- Ensure backward compatibility
- Include performance benchmarks for critical paths

## License

This quota management system is part of the NovaCron project and is licensed under the same terms.

## Support

For technical support, issues, or feature requests:

- GitHub Issues: [Create an issue](https://github.com/novacron/novacron/issues)
- Documentation: [NovaCron Docs](https://docs.novacron.com)
- Community: [NovaCron Discord](https://discord.gg/novacron)