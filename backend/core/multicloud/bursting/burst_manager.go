package bursting

import (
	"context"
	"fmt"
	"sync"
	"time"

	"novacron/backend/core/multicloud/abstraction"
)

// BurstManager manages cloud bursting operations
type BurstManager struct {
	providers         map[string]abstraction.CloudProvider
	config            *BurstConfig
	activeWorkloads   map[string]*BurstWorkload
	metrics           *BurstMetrics
	mu                sync.RWMutex
	thresholdMonitor  *ThresholdMonitor
	costCalculator    *CostCalculator
	shutdownCh        chan struct{}
}

// BurstConfig defines cloud bursting configuration
type BurstConfig struct {
	Enabled              bool                  `json:"enabled"`
	CPUThreshold         float64               `json:"cpu_threshold"`
	MemoryThreshold      float64               `json:"memory_threshold"`
	QueueDepthThreshold  int                   `json:"queue_depth_threshold"`
	MonitorInterval      time.Duration         `json:"monitor_interval"`
	ScaleBackThreshold   float64               `json:"scale_back_threshold"`
	CooldownPeriod       time.Duration         `json:"cooldown_period"`
	PreferredProviders   []string              `json:"preferred_providers"`
	MaxBurstVMs          int                   `json:"max_burst_vms"`
	CostOptimized        bool                  `json:"cost_optimized"`
	ProviderPriority     map[string]int        `json:"provider_priority"`
	WorkloadAffinity     map[string]string     `json:"workload_affinity"`
}

// BurstWorkload represents a workload that has been burst to cloud
type BurstWorkload struct {
	ID               string                    `json:"id"`
	Name             string                    `json:"name"`
	Provider         string                    `json:"provider"`
	VMID             string                    `json:"vm_id"`
	VMSpec           abstraction.VMSpec        `json:"vm_spec"`
	State            string                    `json:"state"`
	CreatedAt        time.Time                 `json:"created_at"`
	LastHealthCheck  time.Time                 `json:"last_health_check"`
	Cost             float64                   `json:"cost"`
	Metrics          WorkloadMetrics           `json:"metrics"`
	ScaleBackEligible bool                     `json:"scale_back_eligible"`
}

// WorkloadMetrics contains metrics for a burst workload
type WorkloadMetrics struct {
	CPUUtilization    float64   `json:"cpu_utilization"`
	MemoryUtilization float64   `json:"memory_utilization"`
	NetworkIn         int64     `json:"network_in"`
	NetworkOut        int64     `json:"network_out"`
	RequestsPerSecond float64   `json:"requests_per_second"`
	LastUpdated       time.Time `json:"last_updated"`
}

// BurstMetrics tracks bursting metrics
type BurstMetrics struct {
	TotalBurstEvents     int64     `json:"total_burst_events"`
	ActiveBurstWorkloads int       `json:"active_burst_workloads"`
	TotalBurstCost       float64   `json:"total_burst_cost"`
	CostSavings          float64   `json:"cost_savings"`
	AverageBurstDuration time.Duration `json:"average_burst_duration"`
	LastBurstTime        time.Time `json:"last_burst_time"`
	ProviderDistribution map[string]int `json:"provider_distribution"`
	mu                   sync.RWMutex
}

// ThresholdMonitor monitors resource thresholds
type ThresholdMonitor struct {
	config            *BurstConfig
	resourceProvider  ResourceProvider
	alertChan         chan *ThresholdAlert
	mu                sync.RWMutex
}

// ThresholdAlert represents a threshold breach
type ThresholdAlert struct {
	Type      string    `json:"type"`
	Metric    string    `json:"metric"`
	Current   float64   `json:"current"`
	Threshold float64   `json:"threshold"`
	Duration  time.Duration `json:"duration"`
	Timestamp time.Time `json:"timestamp"`
}

// ResourceProvider provides on-premise resource metrics
type ResourceProvider interface {
	GetCPUUtilization() (float64, error)
	GetMemoryUtilization() (float64, error)
	GetQueueDepth() (int, error)
	GetAvailableCapacity() (*CapacityInfo, error)
}

// CapacityInfo contains capacity information
type CapacityInfo struct {
	TotalCPUs      int     `json:"total_cpus"`
	AvailableCPUs  int     `json:"available_cpus"`
	TotalMemoryGB  int     `json:"total_memory_gb"`
	AvailableMemoryGB int  `json:"available_memory_gb"`
	CPUUtilization float64 `json:"cpu_utilization"`
	MemoryUtilization float64 `json:"memory_utilization"`
}

// CostCalculator calculates cloud costs
type CostCalculator struct {
	providers map[string]abstraction.CloudProvider
	pricing   map[string]*ProviderPricing
	mu        sync.RWMutex
}

// ProviderPricing contains pricing information
type ProviderPricing struct {
	Provider          string             `json:"provider"`
	ComputePricing    map[string]float64 `json:"compute_pricing"`
	StoragePricing    map[string]float64 `json:"storage_pricing"`
	NetworkPricing    map[string]float64 `json:"network_pricing"`
	SpotPricing       map[string]float64 `json:"spot_pricing"`
	LastUpdated       time.Time          `json:"last_updated"`
}

// NewBurstManager creates a new burst manager
func NewBurstManager(providers map[string]abstraction.CloudProvider, config *BurstConfig, resourceProvider ResourceProvider) *BurstManager {
	bm := &BurstManager{
		providers:       providers,
		config:          config,
		activeWorkloads: make(map[string]*BurstWorkload),
		metrics:         &BurstMetrics{
			ProviderDistribution: make(map[string]int),
		},
		shutdownCh:      make(chan struct{}),
	}

	// Initialize threshold monitor
	bm.thresholdMonitor = &ThresholdMonitor{
		config:           config,
		resourceProvider: resourceProvider,
		alertChan:        make(chan *ThresholdAlert, 100),
	}

	// Initialize cost calculator
	bm.costCalculator = &CostCalculator{
		providers: providers,
		pricing:   make(map[string]*ProviderPricing),
	}

	return bm
}

// Start starts the burst manager
func (bm *BurstManager) Start(ctx context.Context) error {
	if !bm.config.Enabled {
		return fmt.Errorf("burst manager is disabled")
	}

	// Start threshold monitoring
	go bm.monitorThresholds(ctx)

	// Start alert handler
	go bm.handleAlerts(ctx)

	// Start workload health monitoring
	go bm.monitorWorkloads(ctx)

	// Start scale-back detector
	go bm.detectScaleBackOpportunities(ctx)

	return nil
}

// Stop stops the burst manager
func (bm *BurstManager) Stop() {
	close(bm.shutdownCh)
}

// monitorThresholds continuously monitors resource thresholds
func (bm *BurstManager) monitorThresholds(ctx context.Context) {
	ticker := time.NewTicker(bm.config.MonitorInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-bm.shutdownCh:
			return
		case <-ticker.C:
			if err := bm.checkThresholds(); err != nil {
				fmt.Printf("Error checking thresholds: %v\n", err)
			}
		}
	}
}

// checkThresholds checks if resource thresholds are breached
func (bm *BurstManager) checkThresholds() error {
	// Check CPU utilization
	cpuUtil, err := bm.thresholdMonitor.resourceProvider.GetCPUUtilization()
	if err != nil {
		return fmt.Errorf("failed to get CPU utilization: %w", err)
	}

	if cpuUtil > bm.config.CPUThreshold {
		bm.thresholdMonitor.alertChan <- &ThresholdAlert{
			Type:      "cpu",
			Metric:    "cpu_utilization",
			Current:   cpuUtil,
			Threshold: bm.config.CPUThreshold,
			Timestamp: time.Now(),
		}
	}

	// Check memory utilization
	memUtil, err := bm.thresholdMonitor.resourceProvider.GetMemoryUtilization()
	if err != nil {
		return fmt.Errorf("failed to get memory utilization: %w", err)
	}

	if memUtil > bm.config.MemoryThreshold {
		bm.thresholdMonitor.alertChan <- &ThresholdAlert{
			Type:      "memory",
			Metric:    "memory_utilization",
			Current:   memUtil,
			Threshold: bm.config.MemoryThreshold,
			Timestamp: time.Now(),
		}
	}

	// Check queue depth
	queueDepth, err := bm.thresholdMonitor.resourceProvider.GetQueueDepth()
	if err != nil {
		return fmt.Errorf("failed to get queue depth: %w", err)
	}

	if queueDepth > bm.config.QueueDepthThreshold {
		bm.thresholdMonitor.alertChan <- &ThresholdAlert{
			Type:      "queue",
			Metric:    "queue_depth",
			Current:   float64(queueDepth),
			Threshold: float64(bm.config.QueueDepthThreshold),
			Timestamp: time.Now(),
		}
	}

	return nil
}

// handleAlerts handles threshold alerts and triggers bursting
func (bm *BurstManager) handleAlerts(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-bm.shutdownCh:
			return
		case alert := <-bm.thresholdMonitor.alertChan:
			if err := bm.triggerBurst(ctx, alert); err != nil {
				fmt.Printf("Failed to trigger burst: %v\n", err)
			}
		}
	}
}

// triggerBurst triggers cloud bursting
func (bm *BurstManager) triggerBurst(ctx context.Context, alert *ThresholdAlert) error {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	// Check if we're in cooldown period
	if time.Since(bm.metrics.LastBurstTime) < bm.config.CooldownPeriod {
		return fmt.Errorf("in cooldown period")
	}

	// Check if we've reached max burst VMs
	if len(bm.activeWorkloads) >= bm.config.MaxBurstVMs {
		return fmt.Errorf("max burst VMs reached")
	}

	// Select best provider based on cost and availability
	provider, err := bm.selectBestProvider(ctx)
	if err != nil {
		return fmt.Errorf("failed to select provider: %w", err)
	}

	// Create VM spec for burst workload
	vmSpec := bm.createBurstVMSpec(alert)

	// Launch VM in cloud
	vm, err := provider.CreateVM(ctx, vmSpec)
	if err != nil {
		return fmt.Errorf("failed to create burst VM: %w", err)
	}

	// Track burst workload
	workload := &BurstWorkload{
		ID:        fmt.Sprintf("burst-%d", time.Now().Unix()),
		Name:      vmSpec.Name,
		Provider:  provider.GetProviderName(),
		VMID:      vm.ID,
		VMSpec:    vmSpec,
		State:     "running",
		CreatedAt: time.Now(),
	}

	bm.activeWorkloads[workload.ID] = workload

	// Update metrics
	bm.metrics.mu.Lock()
	bm.metrics.TotalBurstEvents++
	bm.metrics.ActiveBurstWorkloads++
	bm.metrics.LastBurstTime = time.Now()
	bm.metrics.ProviderDistribution[provider.GetProviderName()]++
	bm.metrics.mu.Unlock()

	fmt.Printf("Burst triggered: %s on %s (VM: %s)\n", workload.ID, provider.GetProviderName(), vm.ID)

	return nil
}

// selectBestProvider selects the best cloud provider for bursting
func (bm *BurstManager) selectBestProvider(ctx context.Context) (abstraction.CloudProvider, error) {
	if !bm.config.CostOptimized {
		// Use preferred provider
		for _, providerName := range bm.config.PreferredProviders {
			if provider, ok := bm.providers[providerName]; ok {
				return provider, nil
			}
		}
	}

	// Find cheapest provider
	var bestProvider abstraction.CloudProvider
	var lowestCost float64 = -1

	for providerName, provider := range bm.providers {
		cost := bm.costCalculator.EstimateCost(providerName, "t3.medium", 1*time.Hour)
		if lowestCost < 0 || cost < lowestCost {
			lowestCost = cost
			bestProvider = provider
		}
	}

	if bestProvider == nil {
		return nil, fmt.Errorf("no available providers")
	}

	return bestProvider, nil
}

// createBurstVMSpec creates a VM spec for burst workload
func (bm *BurstManager) createBurstVMSpec(alert *ThresholdAlert) abstraction.VMSpec {
	return abstraction.VMSpec{
		Name: fmt.Sprintf("burst-%s-%d", alert.Type, time.Now().Unix()),
		Size: abstraction.VMSize{
			CPUs:     4,
			MemoryGB: 8,
			Type:     "t3.medium",
		},
		Image:        "ami-0c55b159cbfafe1f0", // Ubuntu 20.04
		VolumeSize:   50,
		VolumeType:   "gp3",
		PublicIP:     true,
		SpotInstance: bm.config.CostOptimized,
		Tags: map[string]string{
			"burst":      "true",
			"alert-type": alert.Type,
			"created-by": "burst-manager",
		},
	}
}

// monitorWorkloads monitors active burst workloads
func (bm *BurstManager) monitorWorkloads(ctx context.Context) {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-bm.shutdownCh:
			return
		case <-ticker.C:
			bm.updateWorkloadMetrics(ctx)
		}
	}
}

// updateWorkloadMetrics updates metrics for all workloads
func (bm *BurstManager) updateWorkloadMetrics(ctx context.Context) {
	bm.mu.RLock()
	workloads := make([]*BurstWorkload, 0, len(bm.activeWorkloads))
	for _, wl := range bm.activeWorkloads {
		workloads = append(workloads, wl)
	}
	bm.mu.RUnlock()

	for _, workload := range workloads {
		provider, ok := bm.providers[workload.Provider]
		if !ok {
			continue
		}

		// Get VM metrics
		metrics, err := provider.GetMetrics(ctx, workload.VMID, "cpu", abstraction.TimeRange{
			Start: time.Now().Add(-5 * time.Minute),
			End:   time.Now(),
		})
		if err != nil {
			fmt.Printf("Failed to get metrics for %s: %v\n", workload.VMID, err)
			continue
		}

		if len(metrics) > 0 {
			workload.Metrics.CPUUtilization = metrics[len(metrics)-1].Value
			workload.Metrics.LastUpdated = time.Now()
		}

		// Update cost
		duration := time.Since(workload.CreatedAt)
		workload.Cost = bm.costCalculator.EstimateCost(workload.Provider, workload.VMSpec.Size.Type, duration)
	}
}

// detectScaleBackOpportunities detects when workloads can be scaled back
func (bm *BurstManager) detectScaleBackOpportunities(ctx context.Context) {
	ticker := time.NewTicker(2 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-bm.shutdownCh:
			return
		case <-ticker.C:
			bm.scaleBackUnderutilized(ctx)
		}
	}
}

// scaleBackUnderutilized scales back underutilized workloads
func (bm *BurstManager) scaleBackUnderutilized(ctx context.Context) {
	// Check on-premise capacity
	capacity, err := bm.thresholdMonitor.resourceProvider.GetAvailableCapacity()
	if err != nil {
		fmt.Printf("Failed to get available capacity: %v\n", err)
		return
	}

	// Only scale back if on-premise has capacity
	if capacity.CPUUtilization > bm.config.ScaleBackThreshold {
		return
	}

	bm.mu.Lock()
	defer bm.mu.Unlock()

	for id, workload := range bm.activeWorkloads {
		// Check if workload is underutilized
		if workload.Metrics.CPUUtilization < bm.config.ScaleBackThreshold {
			if err := bm.scaleBackWorkload(ctx, workload); err != nil {
				fmt.Printf("Failed to scale back workload %s: %v\n", id, err)
			} else {
				delete(bm.activeWorkloads, id)
				bm.metrics.mu.Lock()
				bm.metrics.ActiveBurstWorkloads--
				bm.metrics.mu.Unlock()
			}
		}
	}
}

// scaleBackWorkload scales back a single workload
func (bm *BurstManager) scaleBackWorkload(ctx context.Context, workload *BurstWorkload) error {
	provider, ok := bm.providers[workload.Provider]
	if !ok {
		return fmt.Errorf("provider not found: %s", workload.Provider)
	}

	// Stop and delete the VM
	if err := provider.StopVM(ctx, workload.VMID); err != nil {
		return fmt.Errorf("failed to stop VM: %w", err)
	}

	if err := provider.DeleteVM(ctx, workload.VMID); err != nil {
		return fmt.Errorf("failed to delete VM: %w", err)
	}

	fmt.Printf("Scaled back workload %s (VM: %s)\n", workload.ID, workload.VMID)

	return nil
}

// GetMetrics returns current burst metrics
func (bm *BurstManager) GetMetrics() *BurstMetrics {
	bm.metrics.mu.RLock()
	defer bm.metrics.mu.RUnlock()

	// Calculate total cost
	totalCost := 0.0
	bm.mu.RLock()
	for _, wl := range bm.activeWorkloads {
		totalCost += wl.Cost
	}
	bm.mu.RUnlock()
	bm.metrics.TotalBurstCost = totalCost

	return bm.metrics
}

// GetActiveWorkloads returns all active burst workloads
func (bm *BurstManager) GetActiveWorkloads() []*BurstWorkload {
	bm.mu.RLock()
	defer bm.mu.RUnlock()

	workloads := make([]*BurstWorkload, 0, len(bm.activeWorkloads))
	for _, wl := range bm.activeWorkloads {
		workloads = append(workloads, wl)
	}

	return workloads
}

// EstimateCost estimates the cost for a workload
func (cc *CostCalculator) EstimateCost(provider string, instanceType string, duration time.Duration) float64 {
	cc.mu.RLock()
	defer cc.mu.RUnlock()

	pricing, ok := cc.pricing[provider]
	if !ok {
		return 0.0
	}

	hourlyRate, ok := pricing.ComputePricing[instanceType]
	if !ok {
		hourlyRate = 0.05 // Default rate
	}

	hours := duration.Hours()
	return hourlyRate * hours
}
