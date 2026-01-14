package scheduler

import (
	"fmt"
	"log"

	"github.com/khryptorgraphics/novacron/backend/core/scheduler/migration"
	"github.com/khryptorgraphics/novacron/backend/core/scheduler/network"
	"github.com/khryptorgraphics/novacron/backend/core/scheduler/workload"
)

// SchedulerType defines the type of scheduler to use
type SchedulerType string

const (
	// SchedulerTypeBasic is the basic scheduler without advanced features
	SchedulerTypeBasic SchedulerType = "basic"

	// SchedulerTypeResourceAware is the resource-aware scheduler
	SchedulerTypeResourceAware SchedulerType = "resource-aware"

	// SchedulerTypeNetworkAware is the network-aware scheduler
	SchedulerTypeNetworkAware SchedulerType = "network-aware"
)

// SchedulerFactoryConfig contains configuration for the scheduler factory
type SchedulerFactoryConfig struct {
	// SchedulerType is the type of scheduler to create
	SchedulerType SchedulerType

	// ResourceAwareConfig is the configuration for resource-aware scheduler
	ResourceAwareConfig ResourceAwareSchedulerConfig

	// NetworkAwareConfig is the configuration for network-aware scheduler
	NetworkAwareConfig NetworkAwareSchedulerConfig

	// EnableWorkloadAnalysis enables workload analysis
	EnableWorkloadAnalysis bool

	// EnableMigrationCostEstimation enables migration cost estimation
	EnableMigrationCostEstimation bool

	// EnableNetworkTopology enables network topology awareness
	EnableNetworkTopology bool
}

// DefaultSchedulerFactoryConfig returns a default configuration
func DefaultSchedulerFactoryConfig() SchedulerFactoryConfig {
	return SchedulerFactoryConfig{
		SchedulerType:                 SchedulerTypeNetworkAware,
		ResourceAwareConfig:           DefaultResourceAwareSchedulerConfig(),
		NetworkAwareConfig:            DefaultNetworkAwareSchedulerConfig(),
		EnableWorkloadAnalysis:        true,
		EnableMigrationCostEstimation: true,
		EnableNetworkTopology:         true,
	}
}

// SchedulerInterface defines common interface for all scheduler types
type SchedulerInterface interface {
	Start() error
	Stop() error
	UpdateNodeResources(nodeID string, resources map[ResourceType]*Resource) error
	RequestPlacement(vmID string, policy PlacementPolicy, constraints []PlacementConstraint, resources map[string]float64, priority int) (string, error)
	GetPlacementResult(requestID string) (*PlacementResult, error)
}

// SchedulerFactory creates schedulers based on configuration
type SchedulerFactory struct {
	config SchedulerFactoryConfig

	// workloadAnalyzer provides VM workload analysis
	workloadAnalyzer *workload.WorkloadAnalyzer

	// migrationCostEstimator provides migration cost estimation
	migrationCostEstimator *migration.MigrationCostEstimator

	// networkTopology provides network topology awareness
	networkTopology *network.NetworkTopology
}

// NewSchedulerFactory creates a new scheduler factory
func NewSchedulerFactory(config SchedulerFactoryConfig) *SchedulerFactory {
	factory := &SchedulerFactory{
		config: config,
	}

	// Initialize components based on configuration
	if config.EnableWorkloadAnalysis {
		factory.workloadAnalyzer = workload.NewWorkloadAnalyzer(workload.DefaultWorkloadAnalyzerConfig())
		log.Println("Initialized workload analyzer")
	}

	if config.EnableMigrationCostEstimation {
		factory.migrationCostEstimator = migration.NewMigrationCostEstimator(
			migration.DefaultMigrationCostEstimatorConfig(),
			factory.workloadAnalyzer,
		)
		log.Println("Initialized migration cost estimator")
	}

	if config.EnableNetworkTopology {
		factory.networkTopology = network.NewNetworkTopology()
		log.Println("Initialized network topology")
	}

	return factory
}

// CreateScheduler creates a scheduler based on the configured type
func (f *SchedulerFactory) CreateScheduler() (interface{}, error) {
	log.Printf("Creating scheduler of type %s", f.config.SchedulerType)

	switch f.config.SchedulerType {
	case SchedulerTypeBasic:
		scheduler := NewScheduler(DefaultSchedulerConfig())
		return scheduler, nil

	case SchedulerTypeResourceAware:
		// Create the base scheduler first
		baseScheduler := NewScheduler(DefaultSchedulerConfig())

		// Create the resource-aware scheduler
		resourceScheduler := NewResourceAwareScheduler(
			f.config.ResourceAwareConfig,
			baseScheduler,
			f.workloadAnalyzer,
			f.migrationCostEstimator,
		)

		return resourceScheduler, nil

	case SchedulerTypeNetworkAware:
		// Create the base scheduler first
		baseScheduler := NewScheduler(DefaultSchedulerConfig())

		// Ensure required components are available
		if f.workloadAnalyzer == nil {
			f.workloadAnalyzer = workload.NewWorkloadAnalyzer(workload.DefaultWorkloadAnalyzerConfig())
			log.Println("Created workload analyzer on demand")
		}

		if f.networkTopology == nil {
			f.networkTopology = network.NewNetworkTopology()
			log.Println("Created network topology on demand")
		}

		// Create the network-aware scheduler
		networkScheduler := NewNetworkAwareScheduler(
			f.config.NetworkAwareConfig,
			baseScheduler,
			f.workloadAnalyzer,
			f.networkTopology,
		)

		return networkScheduler, nil

	default:
		return nil, fmt.Errorf("unknown scheduler type: %s", f.config.SchedulerType)
	}
}

// GetWorkloadAnalyzer returns the workload analyzer
func (f *SchedulerFactory) GetWorkloadAnalyzer() *workload.WorkloadAnalyzer {
	return f.workloadAnalyzer
}

// GetMigrationCostEstimator returns the migration cost estimator
func (f *SchedulerFactory) GetMigrationCostEstimator() *migration.MigrationCostEstimator {
	return f.migrationCostEstimator
}

// GetNetworkTopology returns the network topology
func (f *SchedulerFactory) GetNetworkTopology() *network.NetworkTopology {
	return f.networkTopology
}
