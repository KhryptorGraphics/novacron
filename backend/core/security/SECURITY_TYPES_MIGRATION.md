# NovaCron Security Types Migration Guide

## Overview

This document provides the migration path for resolving type redeclaration conflicts in NovaCron security modules while maintaining backward compatibility and semantic meaning.

## Conflicts Resolved

### 1. MessageType Conflicts
**Before:**
- `secure_messaging.go`: Used for distributed messaging
- `dating_app_security.go`: Used for user content messages

**After:**
- `DistributedMessageType`: For distributed system communications
- `DatingAppMessageType`: For dating app user messages

### 2. ThreatDetectionConfig Conflicts
**Before:**
- `security_monitoring.go`: Basic monitoring configuration
- `enterprise_security.go`: Enterprise-level configuration

**After:**
- `MonitoringThreatDetectionConfig`: For monitoring systems
- `EnterpriseThreatDetectionConfig`: For enterprise security

### 3. SecurityEvent Conflicts
**Before:**
- `security_monitoring.go`: Monitoring-specific events
- `distributed_security_coordinator.go`: Distributed system events

**After:**
- `MonitoringSecurityEvent`: For monitoring domain events
- `DistributedSecurityEvent`: For distributed system events

### 4. SecurityEventType Conflicts
**Before:**
- `security_monitoring.go`: Monitoring event types
- `distributed_security_coordinator.go`: Distributed event types

**After:**
- `MonitoringSecurityEventType`: For monitoring events
- `DistributedSecurityEventType`: For distributed events

### 5. IntegrationConfig Conflicts
**Before:**
- `security_monitoring.go`: Monitoring integrations
- `security_config.go`: General system integrations

**After:**
- `MonitoringIntegrationConfig`: For monitoring system integrations
- `GeneralIntegrationConfig`: For general system integrations

## Migration Steps

### Step 1: Update Type References

#### 1.1 Secure Messaging Module
```go
// Before
type MessageType string

// After
type DistributedMessageType string

// Update constants
const (
    MessageTypeClusterSync     MessageType = "cluster_sync"     // OLD
    DistributedMessageClusterSync DistributedMessageType = "cluster_sync" // NEW
)
```

#### 1.2 Dating App Security Module
```go
// Before
type MessageType string

// After
type DatingAppMessageType string

// Update constants
const (
    MessageText   MessageType = "text"        // OLD
    DatingAppMessageText DatingAppMessageType = "text" // NEW
)
```

#### 1.3 Security Monitoring Module
```go
// Before
type SecurityEvent struct { ... }
type SecurityEventType string
type ThreatDetectionConfig struct { ... }
type IntegrationConfig struct { ... }

// After
type MonitoringSecurityEvent struct { ... }
type MonitoringSecurityEventType string
type MonitoringThreatDetectionConfig struct { ... }
type MonitoringIntegrationConfig struct { ... }
```

#### 1.4 Distributed Security Coordinator
```go
// Before
type SecurityEvent struct { ... }
type SecurityEventType string

// After
type DistributedSecurityEvent struct { ... }
type DistributedSecurityEventType string
```

#### 1.5 Enterprise Security Module
```go
// Before
type ThreatDetectionConfig struct { ... }

// After
type EnterpriseThreatDetectionConfig struct { ... }
```

#### 1.6 Security Config Module
```go
// Before
type IntegrationConfig struct { ... }

// After
type GeneralIntegrationConfig struct { ... }
```

### Step 2: Update Function Signatures

#### 2.1 Event Handler Updates
```go
// Before
func ProcessSecurityEvent(event SecurityEvent) error

// After - using interface for type safety
func ProcessSecurityEvent(event interface{}) error {
    switch e := event.(type) {
    case MonitoringSecurityEvent:
        return processMonitoringEvent(e)
    case DistributedSecurityEvent:
        return processDistributedEvent(e)
    default:
        return fmt.Errorf("unsupported event type: %T", event)
    }
}
```

#### 2.2 Message Handler Updates
```go
// Before
func SendMessage(messageType MessageType, payload []byte) error

// After
func SendDistributedMessage(messageType DistributedMessageType, payload []byte) error
func SendDatingAppMessage(messageType DatingAppMessageType, content string) error
```

### Step 3: Update Configuration Loading

#### 3.1 Config Structure Updates
```go
// Before
type SecurityConfig struct {
    ThreatDetection ThreatDetectionConfig `json:"threat_detection"`
    Integration     IntegrationConfig     `json:"integration"`
}

// After
type SecurityConfig struct {
    MonitoringThreatDetection MonitoringThreatDetectionConfig `json:"monitoring_threat_detection"`
    EnterpriseThreatDetection EnterpriseThreatDetectionConfig `json:"enterprise_threat_detection"`
    MonitoringIntegration     MonitoringIntegrationConfig     `json:"monitoring_integration"`
    GeneralIntegration        GeneralIntegrationConfig        `json:"general_integration"`
}
```

### Step 4: Implement Type Adapters

#### 4.1 Cross-Domain Event Conversion
```go
// SecurityEventConverter provides conversion between domain-specific event types
type SecurityEventConverter struct {
    registry *SecurityEventTypeRegistry
}

func (c *SecurityEventConverter) ConvertToMonitoring(event DistributedSecurityEvent) MonitoringSecurityEvent {
    monitoringType, exists := c.registry.MapDistributedToMonitoring(event.Type)
    if !exists {
        monitoringType = MonitoringEventSuspiciousActivity // Default fallback
    }

    return MonitoringSecurityEvent{
        SecurityEventBase: event.SecurityEventBase,
        Type:             monitoringType,
        UserID:           extractUserID(event.Data),
        IP:               extractIP(event.Data),
        Details:          event.Data,
        RiskScore:        calculateRiskScore(event),
    }
}

func (c *SecurityEventConverter) ConvertToDistributed(event MonitoringSecurityEvent) DistributedSecurityEvent {
    distributedType, exists := c.registry.MapMonitoringToDistributed(event.Type)
    if !exists {
        distributedType = DistributedEventSuspiciousActivity // Default fallback
    }

    return DistributedSecurityEvent{
        SecurityEventBase: event.SecurityEventBase,
        Type:             distributedType,
        Data:             combineEventData(event.Details, event.UserID, event.IP),
        ClusterID:        extractClusterID(event.Metadata),
        NodeID:           extractNodeID(event.Metadata),
    }
}
```

#### 4.2 Configuration Migration Helper
```go
// ConfigMigrator helps migrate old configuration to new format
type ConfigMigrator struct{}

func (m *ConfigMigrator) MigrateSecurityConfig(oldConfig map[string]interface{}) (*SecurityConfig, error) {
    newConfig := &SecurityConfig{}

    // Migrate threat detection config
    if threatDetection, exists := oldConfig["threat_detection"]; exists {
        if td, ok := threatDetection.(map[string]interface{}); ok {
            newConfig.MonitoringThreatDetection = m.migrateMonitoringThreatDetection(td)
            newConfig.EnterpriseThreatDetection = m.migrateEnterpriseThreatDetection(td)
        }
    }

    // Migrate integration config
    if integration, exists := oldConfig["integration"]; exists {
        if integ, ok := integration.(map[string]interface{}); ok {
            newConfig.MonitoringIntegration = m.migrateMonitoringIntegration(integ)
            newConfig.GeneralIntegration = m.migrateGeneralIntegration(integ)
        }
    }

    return newConfig, nil
}
```

### Step 5: Update Tests

#### 5.1 Test Data Updates
```go
// Before
func TestSecurityEvent(t *testing.T) {
    event := SecurityEvent{
        Type: EventAuthFailure,
        // ...
    }
}

// After
func TestMonitoringSecurityEvent(t *testing.T) {
    event := MonitoringSecurityEvent{
        SecurityEventBase: SecurityEventBase{
            ID: "test-event-1",
            Timestamp: time.Now(),
            Source: "test",
            Severity: SeverityMedium,
        },
        Type: MonitoringEventAuthFailure,
        // ...
    }
}

func TestDistributedSecurityEvent(t *testing.T) {
    event := DistributedSecurityEvent{
        SecurityEventBase: SecurityEventBase{
            ID: "test-event-2",
            Timestamp: time.Now(),
            Source: "test-node",
            Severity: SeverityHigh,
        },
        Type: DistributedEventAuthFailure,
        ClusterID: "test-cluster",
        NodeID: "test-node-1",
        // ...
    }
}
```

#### 5.2 Interface Testing
```go
func TestSecurityEventHandler(t *testing.T) {
    handler := &UnifiedSecurityEventHandler{}

    // Test monitoring event
    monEvent := MonitoringSecurityEvent{/*...*/}
    assert.True(t, handler.CanHandle(monEvent))
    assert.NoError(t, handler.HandleEvent(monEvent))

    // Test distributed event
    distEvent := DistributedSecurityEvent{/*...*/}
    assert.True(t, handler.CanHandle(distEvent))
    assert.NoError(t, handler.HandleEvent(distEvent))

    // Test unsupported event
    unsupportedEvent := struct{}{}
    assert.False(t, handler.CanHandle(unsupportedEvent))
    assert.Error(t, handler.HandleEvent(unsupportedEvent))
}
```

## Backward Compatibility

### Phase 1: Dual Support (Recommended)
During the migration period, maintain dual support:

```go
// Maintain old types as aliases for backward compatibility
// TODO: Remove in version 2.0
type MessageType = DistributedMessageType
type SecurityEvent = MonitoringSecurityEvent
type SecurityEventType = MonitoringSecurityEventType
type ThreatDetectionConfig = MonitoringThreatDetectionConfig
type IntegrationConfig = GeneralIntegrationConfig

// Add deprecation warnings
// Deprecated: Use DistributedMessageType instead
const (
    MessageTypeClusterSync = DistributedMessageClusterSync
)
```

### Phase 2: Deprecation Warnings
Add compile-time deprecation warnings:

```go
// Add build tags for deprecation warnings
//go:build deprecated
// +build deprecated

// Deprecated types with compiler warnings
type MessageType = DistributedMessageType // Deprecated: Use DistributedMessageType
```

### Phase 3: Remove Legacy Types
In the next major version, remove all legacy type aliases and update all references.

## Performance Considerations

### 1. Zero Runtime Overhead
- Type aliases have no runtime cost
- Interface-based handlers use type switches (compile-time optimized)
- No additional memory allocations for type conversions

### 2. Compile-Time Safety
- All type conflicts resolved at compile time
- Interface contracts prevent runtime type errors
- Generic handlers maintain type safety

### 3. Memory Efficiency
- Shared base types reduce memory footprint
- Common fields in SecurityEventBase eliminate duplication
- Registry uses maps for O(1) type lookups

## Testing Strategy

### 1. Unit Tests
- Test each domain's types independently
- Test type conversion functions
- Test interface implementations

### 2. Integration Tests
- Test cross-domain event forwarding
- Test configuration migration
- Test backward compatibility

### 3. Performance Tests
- Benchmark type conversion overhead
- Benchmark interface dispatch performance
- Memory allocation profiling

## Future Extensibility

### 1. Adding New Domains
To add a new security domain (e.g., "blockchain"):

```go
// 1. Register the domain
RegisterSecurityDomain("blockchain", DomainTypeNamespace{
    Domain:           "blockchain",
    Prefix:          "Blockchain",
    EventTypesSuffix: "SecurityEventType",
    EventSuffix:     "SecurityEvent",
})

// 2. Define domain-specific types
type BlockchainSecurityEventType string
type BlockchainSecurityEvent struct {
    SecurityEventBase
    Type         BlockchainSecurityEventType `json:"type"`
    BlockHash    string                      `json:"block_hash"`
    TransactionID string                     `json:"transaction_id"`
}

// 3. Add type mappings to registry
registry.AddBlockchainMappings(...)

// 4. Implement handlers
type BlockchainSecurityHandler struct {}
func (h *BlockchainSecurityHandler) HandleEvent(event interface{}) error { ... }
```

### 2. Interface Evolution
Interfaces can be extended without breaking existing implementations:

```go
// SecurityEventHandlerV2 extends the original interface
type SecurityEventHandlerV2 interface {
    SecurityEventHandler
    // New methods for enhanced functionality
    GetHandlerMetrics() HandlerMetrics
    SetHandlerConfig(config interface{}) error
}
```

## Summary

This migration resolves all type conflicts while:
- ✅ Preserving semantic meaning of each domain's types
- ✅ Enabling type safety and compile-time checking
- ✅ Minimizing breaking changes through backward compatibility
- ✅ Supporting future extensibility
- ✅ Maintaining zero runtime performance overhead
- ✅ Providing clear upgrade paths for all modules

The unified type system establishes a solid foundation for NovaCron's security architecture while allowing each domain to maintain its specialized functionality.