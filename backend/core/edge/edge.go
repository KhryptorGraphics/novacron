package edge

import (
	"context"
	"fmt"
)

// EdgeComputing is the main integration point for edge computing
type EdgeComputing struct {
	Config          *EdgeConfig
	Discovery       *EdgeDiscovery
	Placement       *PlacementEngine
	Coordinator     *EdgeCloudCoordinator
	MECIntegration  *MECIntegration
	IoTGateway      *IoTGatewayManager
	NetworkManager  *EdgeNetworkManager
	Monitoring      *EdgeMonitoring
	PolicyManager   *EdgePolicyManager
	VMLifecycle     *EdgeVMLifecycle
}

// NewEdgeComputing creates a new edge computing instance
func NewEdgeComputing(config *EdgeConfig) (*EdgeComputing, error) {
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	// Create discovery
	discovery := NewEdgeDiscovery(config)

	// Create placement engine
	placement := NewPlacementEngine(config, discovery)

	// Create coordinator
	coordinator := NewEdgeCloudCoordinator(config, discovery, placement)

	// Create MEC integration
	mecIntegration := NewMECIntegration(config, "", "")

	// Create IoT gateway manager
	iotGateway := NewIoTGatewayManager(config)

	// Create network manager
	networkManager := NewEdgeNetworkManager(config, discovery)

	// Create monitoring
	monitoring := NewEdgeMonitoring(config, discovery)

	// Create policy manager
	policyManager := NewEdgePolicyManager(config)
	policyManager.InitializeDefaultPolicies()

	// Create VM lifecycle manager
	vmLifecycle := NewEdgeVMLifecycle(config, discovery, placement, coordinator)

	return &EdgeComputing{
		Config:          config,
		Discovery:       discovery,
		Placement:       placement,
		Coordinator:     coordinator,
		MECIntegration:  mecIntegration,
		IoTGateway:      iotGateway,
		NetworkManager:  networkManager,
		Monitoring:      monitoring,
		PolicyManager:   policyManager,
		VMLifecycle:     vmLifecycle,
	}, nil
}

// Start starts all edge computing services
func (ec *EdgeComputing) Start(ctx context.Context) error {
	// Start discovery
	if err := ec.Discovery.Start(ctx); err != nil {
		return fmt.Errorf("failed to start discovery: %w", err)
	}

	// Start monitoring
	if err := ec.Monitoring.Start(ctx); err != nil {
		return fmt.Errorf("failed to start monitoring: %w", err)
	}

	// Setup mesh network if enabled
	if ec.Config.EdgeMeshEnabled {
		if err := ec.NetworkManager.SetupMeshNetwork(ctx); err != nil {
			// Log error but don't fail startup
		}
	}

	return nil
}

// Stop stops all edge computing services
func (ec *EdgeComputing) Stop() error {
	if err := ec.Discovery.Stop(); err != nil {
		return err
	}

	if err := ec.Monitoring.Stop(); err != nil {
		return err
	}

	return nil
}

// DeployVM deploys a VM to the optimal edge location
func (ec *EdgeComputing) DeployVM(ctx context.Context, req *ProvisionRequest) (*EdgeVM, error) {
	// Validate policies
	// (Would implement policy validation here)

	// Provision VM
	return ec.VMLifecycle.ProvisionVM(ctx, req)
}

// GetStatus retrieves comprehensive edge computing status
func (ec *EdgeComputing) GetStatus(ctx context.Context) (*EdgeStatus, error) {
	dashboardMetrics, err := ec.Monitoring.GetDashboardMetrics(ctx)
	if err != nil {
		return nil, err
	}

	networkHealth, err := ec.NetworkManager.MonitorNetworkHealth(ctx)
	if err != nil {
		return nil, err
	}

	lifecycleMetrics, err := ec.VMLifecycle.GetLifecycleMetrics()
	if err != nil {
		return nil, err
	}

	return &EdgeStatus{
		Dashboard:       dashboardMetrics,
		NetworkHealth:   networkHealth,
		Lifecycle:       lifecycleMetrics,
		ActiveAlerts:    len(ec.Monitoring.GetActiveAlerts()),
	}, nil
}

// EdgeStatus represents overall edge computing status
type EdgeStatus struct {
	Dashboard     *DashboardMetrics  `json:"dashboard"`
	NetworkHealth *NetworkHealth     `json:"network_health"`
	Lifecycle     *LifecycleMetrics  `json:"lifecycle"`
	ActiveAlerts  int                `json:"active_alerts"`
}
