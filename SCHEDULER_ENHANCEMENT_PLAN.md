# Resource-Aware Scheduler Enhancement Plan

This document outlines the planned enhancements for the Resource-Aware Scheduler component of NovaCron, as identified in the Project Masterplan.

## Current Status

The Resource-Aware Scheduler component has been successfully integrated with the core system architecture, with the following implemented:

- Basic scheduling architecture with node resource tracking
- Initial workload profiling capabilities through WorkloadAnalyzer
- Migration cost estimation via MigrationCostEstimator
- Placement constraints support for affinity/anti-affinity requirements
- Basic placement scoring and selection logic

## Priority Enhancements

Based on the Project Masterplan, the following enhancements are prioritized for implementation:

### 1. VM Workload Analysis System

The current workload analysis system could be enhanced with:

- **Advanced Pattern Recognition**: Implement time-series analysis algorithms to detect cyclical patterns in resource usage
- **Prediction Model**: Develop a more sophisticated prediction model based on historical usage data
- **Workload Signature Classification**: Create a machine learning model to classify workload signatures
- **Multi-dimensional Analysis**: Consider correlations between different resource types

```go
// Example enhancement for workload analyzer

// WorkloadPattern represents a recognized pattern in resource usage
type WorkloadPattern struct {
    // PatternType identifies the kind of pattern (diurnal, weekly, etc.)
    PatternType string
    
    // CycleDuration is the duration of one pattern cycle
    CycleDuration time.Duration
    
    // ConfidenceScore indicates how confident we are in this pattern (0-1)
    ConfidenceScore float64
    
    // PeakTimestamps contains timestamps when peaks are expected
    PeakTimestamps []time.Time
    
    // TroughTimestamps contains timestamps when troughs are expected
    TroughTimestamps []time.Time
}

// Enhanced WorkloadProfile with pattern recognition
type WorkloadProfile struct {
    // Existing fields...
    
    // RecognizedPatterns are patterns detected in the workload
    RecognizedPatterns []WorkloadPattern
    
    // PredictionModel contains parameters for usage prediction
    PredictionModel map[string]interface{}
    
    // WorkloadStability indicates how stable this workload is (0-1)
    WorkloadStability float64
}
```

### 2. Migration-Aware Placement Logic

Enhance migration cost integration with placement decisions:

- **Proactive Migration Planning**: Plan migrations ahead of resource constraints
- **Migration Window Scheduling**: Schedule migrations during optimal time windows
- **Resource Reservation System**: Reserve resources for planned migrations
- **Migration Grouping Optimizer**: Group related VMs for migration to reduce total cost

```go
// Example enhancement for migration-aware placement

// MigrationWindow represents an optimal time window for migration
type MigrationWindow struct {
    // StartTime is when the window starts
    StartTime time.Time
    
    // EndTime is when the window ends
    EndTime time.Time
    
    // Quality indicates the suitability of this window (0-1)
    Quality float64
    
    // Reason explains why this window was selected
    Reason string
}

// MigrationPlan represents a planned migration
type MigrationPlan struct {
    // VMID is the VM to migrate
    VMID string
    
    // SourceNodeID is the current node
    SourceNodeID string
    
    // DestNodeID is the target node
    DestNodeID string
    
    // ScheduledWindow is when the migration should occur
    ScheduledWindow MigrationWindow
    
    // EstimatedCost is the projected migration cost
    EstimatedCost *MigrationCost
    
    // Status tracks the execution status
    Status string
}
```

### 3. Advanced Placement Constraints

Enhance the constraint system with:

- **Complex Constraint Expression Language**: Allow for more sophisticated constraint definitions
- **Constraint Solver Optimization**: Improve solver algorithms for multi-dimensional constraints
- **Soft Constraint Support**: Add support for preferences vs. requirements
- **Dynamic Constraint Adjustment**: Adjust constraints based on system conditions

```go
// Example enhancement for placement constraints

// ConstraintOperator defines operations for constraint expressions
type ConstraintOperator string

const (
    OperatorEqual     ConstraintOperator = "eq"
    OperatorNotEqual  ConstraintOperator = "neq"
    OperatorLessThan  ConstraintOperator = "lt"
    OperatorGreaterThan ConstraintOperator = "gt"
    OperatorIn        ConstraintOperator = "in"
    OperatorNotIn     ConstraintOperator = "not_in"
    OperatorExists    ConstraintOperator = "exists"
)

// ConstraintExpression represents a complex constraint expression
type ConstraintExpression struct {
    // Left is the left side of the expression (usually a property name)
    Left string
    
    // Operator is the comparison operator
    Operator ConstraintOperator
    
    // Right is the right side of the expression (usually a value)
    Right interface{}
    
    // SubExpressions for compound expressions (AND, OR, etc.)
    SubExpressions []ConstraintExpression
    
    // LogicalOperator for combining sub-expressions (AND, OR)
    LogicalOperator string
}

// EnhancedPlacementConstraint with expressions
type EnhancedPlacementConstraint struct {
    // ID is a unique identifier
    ID string
    
    // Expression is the constraint expression
    Expression ConstraintExpression
    
    // IsSoft indicates if this is a preference rather than a requirement
    IsSoft bool
    
    // Priority for soft constraints (higher is more important)
    Priority int
    
    // Weight for the constraint (0-1)
    Weight float64
}
```

### 4. Network Topology Awareness

Add network topology considerations to placement decisions:

- **Network Topology Model**: Model the physical network structure
- **Bandwidth and Latency Maps**: Create maps of bandwidth and latency between nodes
- **Traffic Flow Analysis**: Analyze VM communication patterns
- **Placement Optimization**: Optimize placement to minimize network latency for communicating VMs

```go
// Example enhancement for network topology awareness

// NetworkNode represents a node in the network topology
type NetworkNode struct {
    // ID is the node identifier
    ID string
    
    // Type is the type of node (switch, router, host, etc.)
    Type string
    
    // Capacity is the maximum throughput in Mbps
    Capacity float64
    
    // CurrentLoad is the current load in Mbps
    CurrentLoad float64
}

// NetworkLink represents a link in the network topology
type NetworkLink struct {
    // SourceID is the source node ID
    SourceID string
    
    // DestID is the destination node ID
    DestID string
    
    // Bandwidth is the link bandwidth in Mbps
    Bandwidth float64
    
    // Latency is the link latency in ms
    Latency float64
    
    // CurrentUtilization is the current utilization (0-1)
    CurrentUtilization float64
}

// NetworkTopology represents the complete network topology
type NetworkTopology struct {
    // Nodes in the network
    Nodes map[string]*NetworkNode
    
    // Links between nodes
    Links map[string]*NetworkLink
    
    // VMCommunicationMatrix maps VM pairs to their communication intensity
    VMCommunicationMatrix map[string]float64
}
```

### 5. Scheduler API Extension

Extend the scheduler API to expose advanced functionality:

- **Policy Management**: API for defining and managing scheduling policies
- **Constraint Management**: Enhanced API for working with constraints
- **Resource Reservation**: API for reserving resources
- **Migration Planning**: API for planning and scheduling migrations
- **Analysis and Reporting**: API for accessing analytics and recommendations

```go
// Example API extensions

// SchedulerAPI provides access to the scheduler functionality
type SchedulerAPI struct {
    // Scheduler is the underlying scheduler
    Scheduler *ResourceAwareScheduler
}

// CreateSchedulingPolicy creates a new scheduling policy
func (api *SchedulerAPI) CreateSchedulingPolicy(name string, description string, parameters map[string]interface{}) (string, error) {
    // Implementation...
}

// CreateConstraint creates a new placement constraint
func (api *SchedulerAPI) CreateConstraint(expression string, isSoft bool, weight float64) (string, error) {
    // Implementation...
}

// ReserveResources reserves resources for future use
func (api *SchedulerAPI) ReserveResources(nodeID string, resources map[string]float64, duration time.Duration) (string, error) {
    // Implementation...
}

// PlanMigration plans a VM migration for a future time
func (api *SchedulerAPI) PlanMigration(vmID string, destNodeID string, scheduledTime time.Time) (string, error) {
    // Implementation...
}

// GetPlacementRecommendations gets recommendations for VM placement
func (api *SchedulerAPI) GetPlacementRecommendations(vmID string) ([]PlacementRecommendation, error) {
    // Implementation...
}
```

## Implementation Timeline

| Enhancement | Estimated Effort | Priority | Dependencies |
|-------------|------------------|----------|--------------|
| VM Workload Analysis System | 2 weeks | High | None |
| Migration-Aware Placement Logic | 2 weeks | High | Workload Analysis |
| Advanced Placement Constraints | 2 weeks | Medium | None |
| Network Topology Awareness | 1 week | Medium | None |
| Scheduler API Extension | 1 week | Low | All other enhancements |

## Next Steps

1. Implement the VM Workload Analysis enhancements first, as they provide the foundation for other improvements
2. Follow with Migration-Aware Placement Logic to better integrate migration costs into placement decisions
3. Implement Advanced Placement Constraints to support more complex scheduling requirements
4. Add Network Topology Awareness to optimize placement based on network characteristics
5. Finally, extend the Scheduler API to expose all new functionality
