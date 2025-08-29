package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

// VMCacheManager provides VM-specific caching functionality
type VMCacheManager struct {
	cache  Cache
	logger *logrus.Logger
	config *VMCacheConfig
}

// VMCacheConfig holds VM cache configuration
type VMCacheConfig struct {
	// Cache keys TTL settings
	VMStateTTL         time.Duration `json:"vm_state_ttl"`          // 30 seconds
	VMResourcesTTL     time.Duration `json:"vm_resources_ttl"`      // 2 minutes
	VMMigrationTTL     time.Duration `json:"vm_migration_ttl"`      // 1 minute
	VMMetricsTTL       time.Duration `json:"vm_metrics_ttl"`        // 15 seconds
	VMConfigTTL        time.Duration `json:"vm_config_ttl"`         // 15 minutes
	VMListTTL          time.Duration `json:"vm_list_ttl"`           // 1 minute
	NodeResourcesTTL   time.Duration `json:"node_resources_ttl"`    // 30 seconds
	ClusterStateTTL    time.Duration `json:"cluster_state_ttl"`     // 45 seconds
	
	// Cache warming settings
	EnableWarmup       bool          `json:"enable_warmup"`
	WarmupInterval     time.Duration `json:"warmup_interval"`
	CriticalVMList     []string      `json:"critical_vm_list"`      // VMs to keep warm
	
	// Cache invalidation patterns
	InvalidationDelay  time.Duration `json:"invalidation_delay"`    // 100ms
	BatchInvalidation  bool          `json:"batch_invalidation"`
}

// DefaultVMCacheConfig returns default VM cache configuration
func DefaultVMCacheConfig() *VMCacheConfig {
	return &VMCacheConfig{
		VMStateTTL:         30 * time.Second,
		VMResourcesTTL:     2 * time.Minute,
		VMMigrationTTL:     1 * time.Minute,
		VMMetricsTTL:       15 * time.Second,
		VMConfigTTL:        15 * time.Minute,
		VMListTTL:          1 * time.Minute,
		NodeResourcesTTL:   30 * time.Second,
		ClusterStateTTL:    45 * time.Second,
		EnableWarmup:       true,
		WarmupInterval:     5 * time.Minute,
		CriticalVMList:     []string{},
		InvalidationDelay:  100 * time.Millisecond,
		BatchInvalidation:  true,
	}
}

// NewVMCacheManager creates a new VM cache manager
func NewVMCacheManager(cache Cache, config *VMCacheConfig, logger *logrus.Logger) *VMCacheManager {
	if config == nil {
		config = DefaultVMCacheConfig()
	}
	
	if logger == nil {
		logger = logrus.New()
	}

	vcm := &VMCacheManager{
		cache:  cache,
		logger: logger,
		config: config,
	}

	// Start cache warming if enabled
	if config.EnableWarmup {
		go vcm.startWarmupLoop()
	}

	logger.Info("VM cache manager initialized")
	return vcm
}

// Cache key generators
func (vcm *VMCacheManager) vmStateKey(vmID string) string {
	return fmt.Sprintf("vm:state:%s", vmID)
}

func (vcm *VMCacheManager) vmResourcesKey(vmID string) string {
	return fmt.Sprintf("vm:resources:%s", vmID)
}

func (vcm *VMCacheManager) vmMigrationKey(vmID string) string {
	return fmt.Sprintf("vm:migration:%s", vmID)
}

func (vcm *VMCacheManager) vmMetricsKey(vmID string) string {
	return fmt.Sprintf("vm:metrics:%s", vmID)
}

func (vcm *VMCacheManager) vmConfigKey(vmID string) string {
	return fmt.Sprintf("vm:config:%s", vmID)
}

func (vcm *VMCacheManager) vmListKey(nodeID string) string {
	return fmt.Sprintf("vm:list:%s", nodeID)
}

func (vcm *VMCacheManager) nodeResourcesKey(nodeID string) string {
	return fmt.Sprintf("node:resources:%s", nodeID)
}

func (vcm *VMCacheManager) clusterStateKey() string {
	return "cluster:state"
}

// VM State caching
func (vcm *VMCacheManager) GetVMState(ctx context.Context, vmID string) (map[string]interface{}, error) {
	key := vcm.vmStateKey(vmID)
	data, err := vcm.cache.Get(ctx, key)
	if err != nil {
		return nil, err
	}

	var state map[string]interface{}
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("failed to unmarshal VM state: %w", err)
	}

	return state, nil
}

func (vcm *VMCacheManager) SetVMState(ctx context.Context, vmID string, state map[string]interface{}) error {
	key := vcm.vmStateKey(vmID)
	data, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("failed to marshal VM state: %w", err)
	}

	return vcm.cache.Set(ctx, key, data, vcm.config.VMStateTTL)
}

// VM Resources caching
func (vcm *VMCacheManager) GetVMResources(ctx context.Context, vmID string) (map[string]interface{}, error) {
	key := vcm.vmResourcesKey(vmID)
	data, err := vcm.cache.Get(ctx, key)
	if err != nil {
		return nil, err
	}

	var resources map[string]interface{}
	if err := json.Unmarshal(data, &resources); err != nil {
		return nil, fmt.Errorf("failed to unmarshal VM resources: %w", err)
	}

	return resources, nil
}

func (vcm *VMCacheManager) SetVMResources(ctx context.Context, vmID string, resources map[string]interface{}) error {
	key := vcm.vmResourcesKey(vmID)
	data, err := json.Marshal(resources)
	if err != nil {
		return fmt.Errorf("failed to marshal VM resources: %w", err)
	}

	return vcm.cache.Set(ctx, key, data, vcm.config.VMResourcesTTL)
}

// VM Migration status caching
func (vcm *VMCacheManager) GetVMMigrationStatus(ctx context.Context, vmID string) (map[string]interface{}, error) {
	key := vcm.vmMigrationKey(vmID)
	data, err := vcm.cache.Get(ctx, key)
	if err != nil {
		return nil, err
	}

	var status map[string]interface{}
	if err := json.Unmarshal(data, &status); err != nil {
		return nil, fmt.Errorf("failed to unmarshal VM migration status: %w", err)
	}

	return status, nil
}

func (vcm *VMCacheManager) SetVMMigrationStatus(ctx context.Context, vmID string, status map[string]interface{}) error {
	key := vcm.vmMigrationKey(vmID)
	data, err := json.Marshal(status)
	if err != nil {
		return fmt.Errorf("failed to marshal VM migration status: %w", err)
	}

	return vcm.cache.Set(ctx, key, data, vcm.config.VMMigrationTTL)
}

// VM Metrics caching
func (vcm *VMCacheManager) GetVMMetrics(ctx context.Context, vmID string) (map[string]interface{}, error) {
	key := vcm.vmMetricsKey(vmID)
	data, err := vcm.cache.Get(ctx, key)
	if err != nil {
		return nil, err
	}

	var metrics map[string]interface{}
	if err := json.Unmarshal(data, &metrics); err != nil {
		return nil, fmt.Errorf("failed to unmarshal VM metrics: %w", err)
	}

	return metrics, nil
}

func (vcm *VMCacheManager) SetVMMetrics(ctx context.Context, vmID string, metrics map[string]interface{}) error {
	key := vcm.vmMetricsKey(vmID)
	data, err := json.Marshal(metrics)
	if err != nil {
		return fmt.Errorf("failed to marshal VM metrics: %w", err)
	}

	return vcm.cache.Set(ctx, key, data, vcm.config.VMMetricsTTL)
}

// VM Config caching
func (vcm *VMCacheManager) GetVMConfig(ctx context.Context, vmID string) (map[string]interface{}, error) {
	key := vcm.vmConfigKey(vmID)
	data, err := vcm.cache.Get(ctx, key)
	if err != nil {
		return nil, err
	}

	var config map[string]interface{}
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal VM config: %w", err)
	}

	return config, nil
}

func (vcm *VMCacheManager) SetVMConfig(ctx context.Context, vmID string, config map[string]interface{}) error {
	key := vcm.vmConfigKey(vmID)
	data, err := json.Marshal(config)
	if err != nil {
		return fmt.Errorf("failed to marshal VM config: %w", err)
	}

	return vcm.cache.Set(ctx, key, data, vcm.config.VMConfigTTL)
}

// VM List caching
func (vcm *VMCacheManager) GetVMList(ctx context.Context, nodeID string) ([]string, error) {
	key := vcm.vmListKey(nodeID)
	data, err := vcm.cache.Get(ctx, key)
	if err != nil {
		return nil, err
	}

	var vmList []string
	if err := json.Unmarshal(data, &vmList); err != nil {
		return nil, fmt.Errorf("failed to unmarshal VM list: %w", err)
	}

	return vmList, nil
}

func (vcm *VMCacheManager) SetVMList(ctx context.Context, nodeID string, vmList []string) error {
	key := vcm.vmListKey(nodeID)
	data, err := json.Marshal(vmList)
	if err != nil {
		return fmt.Errorf("failed to marshal VM list: %w", err)
	}

	return vcm.cache.Set(ctx, key, data, vcm.config.VMListTTL)
}

// Node Resources caching
func (vcm *VMCacheManager) GetNodeResources(ctx context.Context, nodeID string) (map[string]interface{}, error) {
	key := vcm.nodeResourcesKey(nodeID)
	data, err := vcm.cache.Get(ctx, key)
	if err != nil {
		return nil, err
	}

	var resources map[string]interface{}
	if err := json.Unmarshal(data, &resources); err != nil {
		return nil, fmt.Errorf("failed to unmarshal node resources: %w", err)
	}

	return resources, nil
}

func (vcm *VMCacheManager) SetNodeResources(ctx context.Context, nodeID string, resources map[string]interface{}) error {
	key := vcm.nodeResourcesKey(nodeID)
	data, err := json.Marshal(resources)
	if err != nil {
		return fmt.Errorf("failed to marshal node resources: %w", err)
	}

	return vcm.cache.Set(ctx, key, data, vcm.config.NodeResourcesTTL)
}

// Cluster State caching
func (vcm *VMCacheManager) GetClusterState(ctx context.Context) (map[string]interface{}, error) {
	key := vcm.clusterStateKey()
	data, err := vcm.cache.Get(ctx, key)
	if err != nil {
		return nil, err
	}

	var state map[string]interface{}
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("failed to unmarshal cluster state: %w", err)
	}

	return state, nil
}

func (vcm *VMCacheManager) SetClusterState(ctx context.Context, state map[string]interface{}) error {
	key := vcm.clusterStateKey()
	data, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("failed to marshal cluster state: %w", err)
	}

	return vcm.cache.Set(ctx, key, data, vcm.config.ClusterStateTTL)
}

// Cache invalidation methods
func (vcm *VMCacheManager) InvalidateVM(ctx context.Context, vmID string) error {
	keys := []string{
		vcm.vmStateKey(vmID),
		vcm.vmResourcesKey(vmID),
		vcm.vmMigrationKey(vmID),
		vcm.vmMetricsKey(vmID),
		vcm.vmConfigKey(vmID),
	}

	if vcm.config.BatchInvalidation {
		return vcm.cache.DeleteMulti(ctx, keys)
	}

	for _, key := range keys {
		if err := vcm.cache.Delete(ctx, key); err != nil {
			vcm.logger.WithError(err).WithField("key", key).Error("Failed to invalidate cache key")
		}
	}

	return nil
}

func (vcm *VMCacheManager) InvalidateNode(ctx context.Context, nodeID string) error {
	keys := []string{
		vcm.vmListKey(nodeID),
		vcm.nodeResourcesKey(nodeID),
	}

	if vcm.config.BatchInvalidation {
		return vcm.cache.DeleteMulti(ctx, keys)
	}

	for _, key := range keys {
		if err := vcm.cache.Delete(ctx, key); err != nil {
			vcm.logger.WithError(err).WithField("key", key).Error("Failed to invalidate cache key")
		}
	}

	return nil
}

func (vcm *VMCacheManager) InvalidateCluster(ctx context.Context) error {
	return vcm.cache.Delete(ctx, vcm.clusterStateKey())
}

// Smart invalidation based on VM state changes
func (vcm *VMCacheManager) OnVMStateChange(ctx context.Context, vmID string, oldState, newState string) {
	// Delay invalidation to allow for state consistency
	go func() {
		time.Sleep(vcm.config.InvalidationDelay)
		
		// Always invalidate state cache
		vcm.cache.Delete(context.Background(), vcm.vmStateKey(vmID))
		
		// Invalidate metrics if VM stopped/started
		if (oldState == "running" && newState != "running") ||
		   (oldState != "running" && newState == "running") {
			vcm.cache.Delete(context.Background(), vcm.vmMetricsKey(vmID))
		}
		
		// Invalidate migration status if VM started migrating or completed migration
		if strings.Contains(newState, "migrat") || strings.Contains(oldState, "migrat") {
			vcm.cache.Delete(context.Background(), vcm.vmMigrationKey(vmID))
		}
	}()
}

func (vcm *VMCacheManager) OnVMResourceChange(ctx context.Context, vmID string) {
	go func() {
		time.Sleep(vcm.config.InvalidationDelay)
		vcm.cache.Delete(context.Background(), vcm.vmResourcesKey(vmID))
		vcm.cache.Delete(context.Background(), vcm.vmMetricsKey(vmID))
	}()
}

func (vcm *VMCacheManager) OnVMCreated(ctx context.Context, vmID string, nodeID string) {
	go func() {
		time.Sleep(vcm.config.InvalidationDelay)
		vcm.cache.Delete(context.Background(), vcm.vmListKey(nodeID))
		vcm.cache.Delete(context.Background(), vcm.nodeResourcesKey(nodeID))
		vcm.cache.Delete(context.Background(), vcm.clusterStateKey())
	}()
}

func (vcm *VMCacheManager) OnVMDeleted(ctx context.Context, vmID string, nodeID string) {
	go func() {
		time.Sleep(vcm.config.InvalidationDelay)
		
		// Remove all VM-specific caches
		vcm.InvalidateVM(context.Background(), vmID)
		
		// Update node and cluster caches
		vcm.cache.Delete(context.Background(), vcm.vmListKey(nodeID))
		vcm.cache.Delete(context.Background(), vcm.nodeResourcesKey(nodeID))
		vcm.cache.Delete(context.Background(), vcm.clusterStateKey())
	}()
}

// Cache warming
func (vcm *VMCacheManager) startWarmupLoop() {
	ticker := time.NewTicker(vcm.config.WarmupInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			vcm.warmupCriticalData()
		}
	}
}

func (vcm *VMCacheManager) warmupCriticalData() {
	ctx := context.Background()
	
	// Warm up critical VM data
	for _, vmID := range vcm.config.CriticalVMList {
		// Check if data exists in cache, if not, it will be loaded on next access
		exists, _ := vcm.cache.Exists(ctx, vcm.vmStateKey(vmID))
		if !exists {
			vcm.logger.WithField("vm_id", vmID).Debug("VM state not in cache, will be loaded on next access")
		}
	}
}

// Batch operations for improved performance
func (vcm *VMCacheManager) GetMultipleVMStates(ctx context.Context, vmIDs []string) (map[string]map[string]interface{}, error) {
	keys := make([]string, len(vmIDs))
	for i, vmID := range vmIDs {
		keys[i] = vcm.vmStateKey(vmID)
	}

	results, err := vcm.cache.GetMulti(ctx, keys)
	if err != nil {
		return nil, err
	}

	states := make(map[string]map[string]interface{})
	for i, vmID := range vmIDs {
		key := keys[i]
		if data, exists := results[key]; exists {
			var state map[string]interface{}
			if err := json.Unmarshal(data, &state); err == nil {
				states[vmID] = state
			}
		}
	}

	return states, nil
}

func (vcm *VMCacheManager) GetMultipleVMMetrics(ctx context.Context, vmIDs []string) (map[string]map[string]interface{}, error) {
	keys := make([]string, len(vmIDs))
	for i, vmID := range vmIDs {
		keys[i] = vcm.vmMetricsKey(vmID)
	}

	results, err := vcm.cache.GetMulti(ctx, keys)
	if err != nil {
		return nil, err
	}

	metrics := make(map[string]map[string]interface{})
	for i, vmID := range vmIDs {
		key := keys[i]
		if data, exists := results[key]; exists {
			var metric map[string]interface{}
			if err := json.Unmarshal(data, &metric); err == nil {
				metrics[vmID] = metric
			}
		}
	}

	return metrics, nil
}

// Utility methods
func (vcm *VMCacheManager) GetCacheStats() CacheStats {
	return vcm.cache.GetStats()
}

func (vcm *VMCacheManager) ClearAllVMCache(ctx context.Context) error {
	// This would clear all VM-related cache entries
	// In a production environment, you might want to implement pattern-based deletion
	return vcm.cache.Clear(ctx)
}