package cache

import (
	"context"
	"fmt"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/vm"
	"github.com/sirupsen/logrus"
)

// CachedVMManager wraps VM manager with caching functionality
type CachedVMManager struct {
	vmManager    vm.VMManagerInterface
	cacheManager *VMCacheManager
	cache        Cache
	logger       *logrus.Logger
	config       *VMIntegrationConfig
}

// VMIntegrationConfig holds VM-cache integration configuration
type VMIntegrationConfig struct {
	EnableCaching      bool          `json:"enable_caching"`
	CacheHitLogging    bool          `json:"cache_hit_logging"`
	PreloadCriticalVMs bool          `json:"preload_critical_vms"`
	CriticalVMList     []string      `json:"critical_vm_list"`
	AsyncUpdates       bool          `json:"async_updates"`
	BatchSize          int           `json:"batch_size"`
	RefreshInterval    time.Duration `json:"refresh_interval"`
}

// DefaultVMIntegrationConfig returns default integration configuration
func DefaultVMIntegrationConfig() *VMIntegrationConfig {
	return &VMIntegrationConfig{
		EnableCaching:      true,
		CacheHitLogging:    false,
		PreloadCriticalVMs: true,
		CriticalVMList:     []string{},
		AsyncUpdates:       true,
		BatchSize:          10,
		RefreshInterval:    5 * time.Minute,
	}
}

// NewCachedVMManager creates a new cached VM manager
func NewCachedVMManager(
	vmManager vm.VMManagerInterface,
	cache Cache,
	config *VMIntegrationConfig,
	logger *logrus.Logger,
) (*CachedVMManager, error) {
	if config == nil {
		config = DefaultVMIntegrationConfig()
	}

	if logger == nil {
		logger = logrus.New()
	}

	// Create VM cache manager
	vmCacheConfig := DefaultVMCacheConfig()
	vmCacheConfig.CriticalVMList = config.CriticalVMList
	cacheManager := NewVMCacheManager(cache, vmCacheConfig, logger)

	cvm := &CachedVMManager{
		vmManager:    vmManager,
		cacheManager: cacheManager,
		cache:        cache,
		logger:       logger,
		config:       config,
	}

	// Preload critical VMs if enabled
	if config.PreloadCriticalVMs {
		go cvm.preloadCriticalVMs()
	}

	// Start refresh loop
	go cvm.startRefreshLoop()

	logger.Info("Cached VM manager initialized")
	return cvm, nil
}

// GetVM retrieves VM information with caching
func (cvm *CachedVMManager) GetVM(vmID string) (*vm.VM, error) {
	ctx := context.Background()
	
	if !cvm.config.EnableCaching {
		return cvm.vmManager.GetVM(vmID)
	}

	// Try to get VM state from cache first
	if _, err := cvm.cacheManager.GetVMState(ctx, vmID); err == nil {
		if cvm.config.CacheHitLogging {
			cvm.logger.WithField("vm_id", vmID).Debug("Cache hit for VM state")
		}
		
		// We can't reconstruct VM objects from cache due to private fields
		// Fall back to VM manager for full VM object
	}

	// Cache miss, get from VM manager
	vmObj, err := cvm.vmManager.GetVM(vmID)
	if err != nil {
		return nil, err
	}

	// Cache the VM state asynchronously
	if cvm.config.AsyncUpdates {
		go cvm.cacheVMState(vmObj)
	} else {
		cvm.cacheVMState(vmObj)
	}

	return vmObj, nil
}

// GetVMState retrieves VM state with caching
func (cvm *CachedVMManager) GetVMState(vmID string) (vm.State, error) {
	ctx := context.Background()

	if !cvm.config.EnableCaching {
		if vmObj, err := cvm.vmManager.GetVM(vmID); err == nil {
			return vmObj.State(), nil
		} else {
			return vm.StateUnknown, err
		}
	}

	// Try cache first
	if vmState, err := cvm.cacheManager.GetVMState(ctx, vmID); err == nil {
		if state, ok := vmState["state"].(string); ok {
			if cvm.config.CacheHitLogging {
				cvm.logger.WithFields(logrus.Fields{
					"vm_id": vmID,
					"state": state,
				}).Debug("Cache hit for VM state")
			}
			return vm.State(state), nil
		}
	}

	// Cache miss, get from VM manager
	vmObj, err := cvm.vmManager.GetVM(vmID)
	if err != nil {
		return vm.StateUnknown, err
	}

	// Cache the state
	stateData := map[string]interface{}{
		"state":      string(vmObj.State()),
		"updated_at": time.Now().Unix(),
	}

	if cvm.config.AsyncUpdates {
		go cvm.cacheManager.SetVMState(context.Background(), vmID, stateData)
	} else {
		cvm.cacheManager.SetVMState(ctx, vmID, stateData)
	}

	return vmObj.State(), nil
}

// GetVMMetrics retrieves VM metrics with caching
func (cvm *CachedVMManager) GetVMMetrics(vmID string) (map[string]interface{}, error) {
	ctx := context.Background()

	if !cvm.config.EnableCaching {
		// Get metrics directly from VM manager - this would need to be implemented
		return cvm.getVMMetricsFromManager(vmID)
	}

	// Try cache first
	if metrics, err := cvm.cacheManager.GetVMMetrics(ctx, vmID); err == nil {
		if cvm.config.CacheHitLogging {
			cvm.logger.WithField("vm_id", vmID).Debug("Cache hit for VM metrics")
		}
		return metrics, nil
	}

	// Cache miss, get from VM manager
	metrics, err := cvm.getVMMetricsFromManager(vmID)
	if err != nil {
		return nil, err
	}

	// Cache the metrics
	if cvm.config.AsyncUpdates {
		go cvm.cacheManager.SetVMMetrics(context.Background(), vmID, metrics)
	} else {
		cvm.cacheManager.SetVMMetrics(ctx, vmID, metrics)
	}

	return metrics, nil
}

// GetVMResources retrieves VM resource allocation with caching
func (cvm *CachedVMManager) GetVMResources(vmID string) (map[string]interface{}, error) {
	ctx := context.Background()

	if !cvm.config.EnableCaching {
		return cvm.getVMResourcesFromManager(vmID)
	}

	// Try cache first
	if resources, err := cvm.cacheManager.GetVMResources(ctx, vmID); err == nil {
		if cvm.config.CacheHitLogging {
			cvm.logger.WithField("vm_id", vmID).Debug("Cache hit for VM resources")
		}
		return resources, nil
	}

	// Cache miss, get from VM manager
	resources, err := cvm.getVMResourcesFromManager(vmID)
	if err != nil {
		return nil, err
	}

	// Cache the resources
	if cvm.config.AsyncUpdates {
		go cvm.cacheManager.SetVMResources(context.Background(), vmID, resources)
	} else {
		cvm.cacheManager.SetVMResources(ctx, vmID, resources)
	}

	return resources, nil
}

// GetVMMigrationStatus retrieves VM migration status with caching
func (cvm *CachedVMManager) GetVMMigrationStatus(vmID string) (map[string]interface{}, error) {
	ctx := context.Background()

	if !cvm.config.EnableCaching {
		return cvm.getVMMigrationStatusFromManager(vmID)
	}

	// Try cache first
	if status, err := cvm.cacheManager.GetVMMigrationStatus(ctx, vmID); err == nil {
		if cvm.config.CacheHitLogging {
			cvm.logger.WithField("vm_id", vmID).Debug("Cache hit for VM migration status")
		}
		return status, nil
	}

	// Cache miss, get from VM manager
	status, err := cvm.getVMMigrationStatusFromManager(vmID)
	if err != nil {
		return nil, err
	}

	// Cache the migration status
	if cvm.config.AsyncUpdates {
		go cvm.cacheManager.SetVMMigrationStatus(context.Background(), vmID, status)
	} else {
		cvm.cacheManager.SetVMMigrationStatus(ctx, vmID, status)
	}

	return status, nil
}

// GetMultipleVMStates retrieves multiple VM states efficiently
func (cvm *CachedVMManager) GetMultipleVMStates(vmIDs []string) (map[string]vm.State, error) {
	if !cvm.config.EnableCaching {
		return cvm.getMultipleVMStatesFromManager(vmIDs)
	}

	ctx := context.Background()
	result := make(map[string]vm.State)
	
	// Try to get from cache in batch
	cachedStates, _ := cvm.cacheManager.GetMultipleVMStates(ctx, vmIDs)
	
	var uncachedVMIDs []string
	for _, vmID := range vmIDs {
		if stateData, exists := cachedStates[vmID]; exists {
			if state, ok := stateData["state"].(string); ok {
				result[vmID] = vm.State(state)
			} else {
				uncachedVMIDs = append(uncachedVMIDs, vmID)
			}
		} else {
			uncachedVMIDs = append(uncachedVMIDs, vmID)
		}
	}

	// Get uncached VMs from manager
	if len(uncachedVMIDs) > 0 {
		uncachedStates, err := cvm.getMultipleVMStatesFromManager(uncachedVMIDs)
		if err != nil {
			return result, err // Return partial results
		}

		// Add to result and cache
		for vmID, state := range uncachedStates {
			result[vmID] = state
			
			stateData := map[string]interface{}{
				"state":      string(state),
				"updated_at": time.Now().Unix(),
			}

			if cvm.config.AsyncUpdates {
				go cvm.cacheManager.SetVMState(context.Background(), vmID, stateData)
			} else {
				cvm.cacheManager.SetVMState(ctx, vmID, stateData)
			}
		}
	}

	if cvm.config.CacheHitLogging {
		cacheHits := len(vmIDs) - len(uncachedVMIDs)
		cvm.logger.WithFields(logrus.Fields{
			"total_vms":   len(vmIDs),
			"cache_hits":  cacheHits,
			"cache_rate":  float64(cacheHits) / float64(len(vmIDs)),
		}).Debug("Batch VM state cache performance")
	}

	return result, nil
}

// InvalidateVM invalidates all cached data for a VM
func (cvm *CachedVMManager) InvalidateVM(vmID string) error {
	ctx := context.Background()
	return cvm.cacheManager.InvalidateVM(ctx, vmID)
}

// InvalidateVMPattern invalidates cached data for VMs matching a pattern
func (cvm *CachedVMManager) InvalidateVMPattern(pattern string) error {
	// This would require pattern matching in the cache implementation
	// For now, we'll implement a simple prefix-based invalidation
	_ = context.Background() // Would be used for pattern-based deletion
	
	// Get all cache keys (this is expensive and should be optimized)
	// In a real implementation, you'd want pattern-based deletion support
	cvm.logger.WithField("pattern", pattern).Warn("Pattern-based cache invalidation not fully implemented")
	
	return nil
}

// OnVMStateChange handles VM state change events
func (cvm *CachedVMManager) OnVMStateChange(vmID string, oldState, newState vm.State) {
	if !cvm.config.EnableCaching {
		return
	}

	cvm.cacheManager.OnVMStateChange(context.Background(), vmID, string(oldState), string(newState))
	
	cvm.logger.WithFields(logrus.Fields{
		"vm_id":     vmID,
		"old_state": oldState,
		"new_state": newState,
	}).Debug("VM state change, updating cache")
}

// OnVMCreated handles VM creation events
func (cvm *CachedVMManager) OnVMCreated(vmID string, nodeID string) {
	if !cvm.config.EnableCaching {
		return
	}

	cvm.cacheManager.OnVMCreated(context.Background(), vmID, nodeID)
	
	cvm.logger.WithFields(logrus.Fields{
		"vm_id":   vmID,
		"node_id": nodeID,
	}).Debug("VM created, updating cache")
}

// OnVMDeleted handles VM deletion events
func (cvm *CachedVMManager) OnVMDeleted(vmID string, nodeID string) {
	if !cvm.config.EnableCaching {
		return
	}

	cvm.cacheManager.OnVMDeleted(context.Background(), vmID, nodeID)
	
	cvm.logger.WithFields(logrus.Fields{
		"vm_id":   vmID,
		"node_id": nodeID,
	}).Debug("VM deleted, updating cache")
}

// GetCacheStats returns cache performance statistics
func (cvm *CachedVMManager) GetCacheStats() CacheStats {
	return cvm.cacheManager.GetCacheStats()
}

// GetCacheMetrics returns detailed cache metrics
func (cvm *CachedVMManager) GetCacheMetrics() map[string]interface{} {
	stats := cvm.GetCacheStats()
	
	return map[string]interface{}{
		"hit_rate":           stats.HitRate,
		"total_hits":         stats.Hits,
		"total_misses":       stats.Misses,
		"total_sets":         stats.Sets,
		"total_deletes":      stats.Deletes,
		"total_errors":       stats.Errors,
		"l1_hits":            stats.L1Hits,
		"l2_hits":            stats.L2Hits,
		"l3_hits":            stats.L3Hits,
		"avg_response_time":  fmt.Sprintf("%.2fμs", float64(stats.AvgResponseTimeNs)/1000),
		"last_updated":       stats.LastUpdated.Format(time.RFC3339),
		"cache_enabled":      cvm.config.EnableCaching,
	}
}

// preloadCriticalVMs preloads critical VM data into cache
func (cvm *CachedVMManager) preloadCriticalVMs() {
	if len(cvm.config.CriticalVMList) == 0 {
		return
	}

	cvm.logger.WithField("count", len(cvm.config.CriticalVMList)).Info("Preloading critical VMs")

	for _, vmID := range cvm.config.CriticalVMList {
		// Preload VM state
		if vmObj, err := cvm.vmManager.GetVM(vmID); err == nil {
			cvm.cacheVMState(vmObj)
			
			// Also preload metrics and resources
			if metrics, err := cvm.getVMMetricsFromManager(vmID); err == nil {
				cvm.cacheManager.SetVMMetrics(context.Background(), vmID, metrics)
			}
			
			if resources, err := cvm.getVMResourcesFromManager(vmID); err == nil {
				cvm.cacheManager.SetVMResources(context.Background(), vmID, resources)
			}
		} else {
			cvm.logger.WithError(err).WithField("vm_id", vmID).Warn("Failed to preload critical VM")
		}
	}
}

// startRefreshLoop starts periodic cache refresh
func (cvm *CachedVMManager) startRefreshLoop() {
	ticker := time.NewTicker(cvm.config.RefreshInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			cvm.refreshCriticalVMs()
		}
	}
}

// refreshCriticalVMs refreshes critical VM data in cache
func (cvm *CachedVMManager) refreshCriticalVMs() {
	if len(cvm.config.CriticalVMList) == 0 {
		return
	}

	cvm.logger.Debug("Refreshing critical VMs cache")

	for _, vmID := range cvm.config.CriticalVMList {
		go func(id string) {
			if vmObj, err := cvm.vmManager.GetVM(id); err == nil {
				cvm.cacheVMState(vmObj)
			}
		}(vmID)
	}
}

// Helper methods

// cacheVMState caches VM state information
func (cvm *CachedVMManager) cacheVMState(vmObj *vm.VM) {
	if vmObj == nil {
		return
	}

	stateData := map[string]interface{}{
		"id":         vmObj.ID(),
		"name":       "vm-name", // VM struct doesn't expose name directly
		"state":      string(vmObj.State()),
		"type":       "kvm", // Default type
		"node_id":    vmObj.NodeID(),
		"created_at": vmObj.CreatedAt().Unix(),
		"updated_at": time.Now().Unix(),
	}

	cvm.cacheManager.SetVMState(context.Background(), vmObj.ID(), stateData)
}

// stateToVM converts cached state back to VM object
// Note: This is a simplified implementation since VM struct uses private fields
// In a real implementation, you would need VM constructor methods
func (cvm *CachedVMManager) stateToVM(vmID string, stateData map[string]interface{}) (*vm.VM, error) {
	// For now, we'll return an error since we can't construct VM objects directly
	// This would need to be implemented with proper VM factory methods
	return nil, fmt.Errorf("VM construction from cache not implemented - use VM manager directly")
}

// Placeholder methods that would need to be implemented based on actual VM manager interface
func (cvm *CachedVMManager) getVMMetricsFromManager(vmID string) (map[string]interface{}, error) {
	// This would call the actual VM manager's metrics endpoint
	return map[string]interface{}{
		"cpu_usage":    0.0,
		"memory_usage": 0.0,
		"disk_usage":   0.0,
		"network_io":   map[string]interface{}{
			"rx_bytes": 0,
			"tx_bytes": 0,
		},
		"last_updated": time.Now().Unix(),
	}, nil
}

func (cvm *CachedVMManager) getVMResourcesFromManager(vmID string) (map[string]interface{}, error) {
	// This would call the actual VM manager's resource allocation endpoint
	return map[string]interface{}{
		"cpu_cores":    1,
		"memory_mb":    1024,
		"disk_gb":      20,
		"network_mbps": 100,
		"last_updated": time.Now().Unix(),
	}, nil
}

func (cvm *CachedVMManager) getVMMigrationStatusFromManager(vmID string) (map[string]interface{}, error) {
	// This would call the actual VM manager's migration status endpoint
	return map[string]interface{}{
		"status":           "none",
		"progress_percent": 0,
		"source_node":      "",
		"target_node":      "",
		"last_updated":     time.Now().Unix(),
	}, nil
}

func (cvm *CachedVMManager) getMultipleVMStatesFromManager(vmIDs []string) (map[string]vm.State, error) {
	result := make(map[string]vm.State)
	
	for _, vmID := range vmIDs {
		if vmObj, err := cvm.vmManager.GetVM(vmID); err == nil {
			result[vmID] = vmObj.State()
		}
	}
	
	return result, nil
}