package storage

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/plugins/storage/ceph"
	// TODO: Re-enable tiering after resolving import cycles
	// "github.com/khryptorgraphics/novacron/backend/core/storage/tiering"
)

// cephDriverAdapter adapts the Ceph driver to the StorageDriver interface
type cephDriverAdapter struct {
	driver *cephstorage.CephStorageDriver
}

func (c *cephDriverAdapter) Initialize() error {
	return c.driver.Initialize()
}

func (c *cephDriverAdapter) Shutdown() error {
	return c.driver.Shutdown()
}

func (c *cephDriverAdapter) CreateVolume(ctx context.Context, volumeID string, sizeBytes int64) error {
	return c.driver.CreateVolume(ctx, volumeID, sizeBytes)
}

func (c *cephDriverAdapter) DeleteVolume(ctx context.Context, volumeID string) error {
	return c.driver.DeleteVolume(ctx, volumeID)
}

func (c *cephDriverAdapter) AttachVolume(ctx context.Context, volumeID, nodeID string) error {
	return c.driver.AttachVolume(ctx, volumeID, nodeID)
}

func (c *cephDriverAdapter) DetachVolume(ctx context.Context, volumeID, nodeID string) error {
	return c.driver.DetachVolume(ctx, volumeID, nodeID)
}

func (c *cephDriverAdapter) ReadVolume(ctx context.Context, volumeID string, offset int64, size int) ([]byte, error) {
	return c.driver.ReadVolume(ctx, volumeID, offset, size)
}

func (c *cephDriverAdapter) WriteVolume(ctx context.Context, volumeID string, offset int64, data []byte) error {
	return c.driver.WriteVolume(ctx, volumeID, offset, data)
}

func (c *cephDriverAdapter) GetVolumeInfo(ctx context.Context, volumeID string) (*VolumeInfo, error) {
	cephVol, err := c.driver.GetVolumeInfo(ctx, volumeID)
	if err != nil {
		return nil, err
	}
	// Convert from Ceph VolumeInfo to storage VolumeInfo
	return &VolumeInfo{
		ID:                cephVol.ID,
		Name:              cephVol.Name,
		Type:              VolumeType(cephVol.Type),
		State:             VolumeState(cephVol.State),
		Size:              cephVol.Size,
		Path:              cephVol.Path,
		Format:            VolumeFormat(cephVol.Format),
		CreatedAt:         cephVol.CreatedAt,
		UpdatedAt:         cephVol.UpdatedAt,
		AttachedToVM:      cephVol.AttachedToVM,
		Metadata:          cephVol.Metadata,
		Bootable:          cephVol.Bootable,
		Encrypted:         cephVol.Encrypted,
		ReplicationFactor: cephVol.ReplicationFactor,
	}, nil
}

func (c *cephDriverAdapter) ListVolumes(ctx context.Context) ([]string, error) {
	return c.driver.ListVolumes(ctx)
}

func (c *cephDriverAdapter) GetCapabilities() DriverCapabilities {
	cephCaps := c.driver.GetCapabilities()
	// Convert from Ceph DriverCapabilities to storage DriverCapabilities
	return DriverCapabilities{
		SupportsSnapshots:     cephCaps.SupportsSnapshots,
		SupportsReplication:   cephCaps.SupportsReplication,
		SupportsEncryption:    cephCaps.SupportsEncryption,
		SupportsCompression:   cephCaps.SupportsCompression,
		SupportsDeduplication: cephCaps.SupportsDeduplication,
		MaxVolumeSize:         cephCaps.MaxVolumeSize,
		MinVolumeSize:         cephCaps.MinVolumeSize,
	}
}

func (c *cephDriverAdapter) CreateSnapshot(ctx context.Context, volumeID, snapshotID string) error {
	return c.driver.CreateSnapshot(ctx, volumeID, snapshotID)
}

func (c *cephDriverAdapter) DeleteSnapshot(ctx context.Context, volumeID, snapshotID string) error {
	return c.driver.DeleteSnapshot(ctx, volumeID, snapshotID)
}

func (c *cephDriverAdapter) RestoreSnapshot(ctx context.Context, volumeID, snapshotID string) error {
	return c.driver.RestoreSnapshot(ctx, volumeID, snapshotID)
}

// IntegratedStorageManager manages multiple storage backends and provides unified storage operations
type IntegratedStorageManager struct {
	// Map of driver names to storage drivers
	drivers map[string]StorageDriver

	// TODO: Re-enable tier manager after resolving import cycles
	// tierManager *tiering.TierManager

	// Volume metadata tracking
	volumes map[string]*VolumeInfo
	volumesMu sync.RWMutex

	// Storage pools
	pools map[string]*StoragePool
	poolsMu sync.RWMutex

	// Event listeners
	eventListeners []VolumeEventListener
	eventMu        sync.RWMutex

	// Context for background operations
	ctx    context.Context
	cancel context.CancelFunc

	// Configuration
	config StorageConfig

	// Metrics
	metrics *StorageMetrics
}

// StoragePool represents a storage pool
type StoragePool struct {
	ID         string            `json:"id"`
	Name       string            `json:"name"`
	DriverName string            `json:"driver_name"`
	Type       VolumeType        `json:"type"`
	TotalSize  int64            `json:"total_size"`
	UsedSize   int64            `json:"used_size"`
	Status     string            `json:"status"`
	CreatedAt  time.Time         `json:"created_at"`
	UpdatedAt  time.Time         `json:"updated_at"`
	Tags       []string          `json:"tags,omitempty"`
	Metadata   map[string]string `json:"metadata,omitempty"`
	Config     map[string]interface{} `json:"config,omitempty"`
}

// StorageMetrics contains storage performance metrics
type StorageMetrics struct {
	TotalCapacityBytes     int64            `json:"total_capacity_bytes"`
	UsedCapacityBytes      int64            `json:"used_capacity_bytes"`
	AvailableCapacityBytes int64            `json:"available_capacity_bytes"`
	TotalVolumes           int              `json:"total_volumes"`
	ActiveVolumes          int              `json:"active_volumes"`
	PoolMetrics            map[string]interface{} `json:"pool_metrics"`
	LastUpdated            time.Time        `json:"last_updated"`
}

// NewIntegratedStorageManager creates a new integrated storage manager
func NewIntegratedStorageManager(config StorageConfig) *IntegratedStorageManager {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &IntegratedStorageManager{
		drivers:        make(map[string]StorageDriver),
		// TODO: Re-enable tier manager
		// tierManager:    tiering.NewTierManager(),
		volumes:        make(map[string]*VolumeInfo),
		pools:          make(map[string]*StoragePool),
		eventListeners: make([]VolumeEventListener, 0),
		ctx:            ctx,
		cancel:         cancel,
		config:         config,
		metrics:        &StorageMetrics{
			PoolMetrics: make(map[string]interface{}),
		},
	}
}

// Start starts the integrated storage manager
func (ism *IntegratedStorageManager) Start() error {
	log.Printf("Starting integrated storage manager")

	// Initialize default drivers
	if err := ism.initializeDefaultDrivers(); err != nil {
		return fmt.Errorf("failed to initialize default drivers: %v", err)
	}

	// TODO: Re-enable tier manager setup
	// if err := ism.setupTierManager(); err != nil {
	// 	return fmt.Errorf("failed to setup tier manager: %v", err)
	// }

	// Start background metrics collection
	go ism.metricsCollectionLoop()

	// TODO: Re-enable background tier management
	// if err := ism.tierManager.StartBackgroundWorker(5 * time.Minute); err != nil {
	// 	log.Printf("Warning: Failed to start tier manager background worker: %v", err)
	// }

	log.Printf("Integrated storage manager started successfully")
	return nil
}

// Stop stops the integrated storage manager
func (ism *IntegratedStorageManager) Stop() error {
	log.Printf("Stopping integrated storage manager")
	
	// Stop background workers
	ism.cancel()
	// TODO: Re-enable tier manager
	// ism.tierManager.StopBackgroundWorker()

	// Shutdown all drivers
	for name, driver := range ism.drivers {
		if err := driver.Shutdown(); err != nil {
			log.Printf("Warning: Failed to shutdown driver %s: %v", name, err)
		}
	}

	return nil
}

// initializeDefaultDrivers initializes the default storage drivers
func (ism *IntegratedStorageManager) initializeDefaultDrivers() error {
	// Initialize local driver
	localDriver := NewLocalStorageDriver()
	if err := localDriver.Initialize(); err != nil {
		log.Printf("Warning: Failed to initialize local driver: %v", err)
	} else {
		ism.drivers["local"] = localDriver
		log.Printf("Initialized local storage driver")
	}

	// Initialize Ceph driver if available
	cephConfig := cephstorage.DefaultCephConfig()
	cephDriver := cephstorage.NewCephStorageDriver(cephConfig)
	if err := cephDriver.Initialize(); err != nil {
		log.Printf("Warning: Failed to initialize Ceph driver: %v", err)
	} else {
		// Use adapter to bridge interface differences
		adapter := &cephDriverAdapter{driver: cephDriver}
		ism.drivers["ceph"] = adapter
		log.Printf("Initialized Ceph storage driver")
	}

	// Create default storage pools
	if err := ism.createDefaultPools(); err != nil {
		return fmt.Errorf("failed to create default pools: %v", err)
	}

	return nil
}

// TODO: Re-enable after resolving import cycles
/*
// setupTierManager sets up the tier manager with available drivers
func (ism *IntegratedStorageManager) setupTierManager() error {
	// Add tiers based on available drivers
	if driver, exists := ism.drivers["local"]; exists {
		if err := ism.tierManager.AddTier(tiering.TierHot, driver, "Local SSD", 0.10, 1000); err != nil {
			return fmt.Errorf("failed to add hot tier: %v", err)
		}
	}

	if driver, exists := ism.drivers["ceph"]; exists {
		if err := ism.tierManager.AddTier(tiering.TierWarm, driver, "Ceph RBD", 0.05, 10000); err != nil {
			return fmt.Errorf("failed to add warm tier: %v", err)
		}
		if err := ism.tierManager.AddTier(tiering.TierCold, driver, "Ceph Archive", 0.02, 100000); err != nil {
			return fmt.Errorf("failed to add cold tier: %v", err)
		}
	}

	// Initialize tier manager
	if err := ism.tierManager.Initialize(); err != nil {
		return fmt.Errorf("failed to initialize tier manager: %v", err)
	}

	// Set up default tiering policies
	ism.tierManager.CreateDefaultAgingPolicy()
	ism.tierManager.CreateCostOptimizationPolicy()

	return nil
}
*/

// createDefaultPools creates default storage pools for each driver
func (ism *IntegratedStorageManager) createDefaultPools() error {
	ism.poolsMu.Lock()
	defer ism.poolsMu.Unlock()

	// Create default pools for each driver
	for name, driver := range ism.drivers {
		pool := &StoragePool{
			ID:         fmt.Sprintf("pool-%s-default", name),
			Name:       fmt.Sprintf("Default %s Pool", name),
			DriverName: name,
			Type:       VolumeTypeLocal, // Will be updated based on driver type
			TotalSize:  1000 * 1024 * 1024 * 1024, // 1TB default
			UsedSize:   0,
			Status:     "healthy",
			CreatedAt:  time.Now(),
			UpdatedAt:  time.Now(),
			Tags:       []string{"default"},
			Metadata:   make(map[string]string),
			Config:     make(map[string]interface{}),
		}

		// Set type based on driver capabilities
		caps := driver.GetCapabilities()
		if caps.SupportsReplication {
			pool.Type = VolumeTypeDistributed
		}

		ism.pools[pool.ID] = pool
		log.Printf("Created default pool %s for driver %s", pool.ID, name)
	}

	return nil
}

// CreateVolume creates a new volume using the specified options
func (ism *IntegratedStorageManager) CreateVolume(ctx context.Context, opts VolumeCreateOptions) (*VolumeInfo, error) {
	ism.volumesMu.Lock()
	defer ism.volumesMu.Unlock()

	// Select the best driver for this volume type
	driverName := ism.selectBestDriver(opts.Type)
	driver, exists := ism.drivers[driverName]
	if !exists {
		return nil, fmt.Errorf("no suitable driver found for volume type %s", opts.Type)
	}

	// Create the volume using the selected driver
	volumeID := generateVolumeID()
	if err := driver.CreateVolume(ctx, volumeID, opts.Size); err != nil {
		return nil, fmt.Errorf("failed to create volume with driver %s: %v", driverName, err)
	}

	// Create volume info
	volume := &VolumeInfo{
		ID:                volumeID,
		Name:              opts.Name,
		Type:              opts.Type,
		State:             VolumeStateCreating,
		Size:              opts.Size,
		CreatedAt:         time.Now(),
		UpdatedAt:         time.Now(),
		Metadata:          opts.Metadata,
		Bootable:          opts.Bootable,
		Encrypted:         opts.Encrypted,
		ReplicationFactor: opts.ReplicationFactor,
		NodeIDs:           opts.NodeIDs,
	}

	// Store volume info
	ism.volumes[volumeID] = volume

	// Record access in tier manager
	// TODO: Re-enable tier manager
	// ism.tierManager.RecordVolumeAccess(volumeID)

	// Emit event
	ism.emitEvent(VolumeEvent{
		Type:       VolumeEventCreated,
		VolumeID:   volumeID,
		VolumeName: opts.Name,
		Timestamp:  time.Now(),
		Data:       volume,
	})

	// Asynchronously update state to available
	go func() {
		time.Sleep(2 * time.Second) // Simulate creation time
		ism.volumesMu.Lock()
		if vol, exists := ism.volumes[volumeID]; exists {
			vol.State = VolumeStateAvailable
			vol.UpdatedAt = time.Now()
		}
		ism.volumesMu.Unlock()
	}()

	return volume, nil
}

// DeleteVolume deletes a volume
func (ism *IntegratedStorageManager) DeleteVolume(ctx context.Context, volumeID string) error {
	ism.volumesMu.Lock()
	defer ism.volumesMu.Unlock()

	volume, exists := ism.volumes[volumeID]
	if !exists {
		return ErrVolumeNotFound
	}

	// Check if volume is in use
	if volume.AttachedToVM != "" {
		return ErrVolumeInUse
	}

	// Find the driver that manages this volume
	driverName := ism.getDriverForVolume(volume.Type)
	driver, exists := ism.drivers[driverName]
	if !exists {
		return fmt.Errorf("driver %s not found for volume %s", driverName, volumeID)
	}

	// Delete the volume using the driver
	if err := driver.DeleteVolume(ctx, volumeID); err != nil {
		return fmt.Errorf("failed to delete volume with driver %s: %v", driverName, err)
	}

	// Remove from memory
	delete(ism.volumes, volumeID)

	// Emit event
	ism.emitEvent(VolumeEvent{
		Type:       VolumeEventDeleted,
		VolumeID:   volumeID,
		VolumeName: volume.Name,
		Timestamp:  time.Now(),
	})

	return nil
}

// AttachVolume attaches a volume to a VM
func (ism *IntegratedStorageManager) AttachVolume(ctx context.Context, volumeID string, opts VolumeAttachOptions) error {
	ism.volumesMu.Lock()
	defer ism.volumesMu.Unlock()

	volume, exists := ism.volumes[volumeID]
	if !exists {
		return ErrVolumeNotFound
	}

	if volume.State != VolumeStateAvailable {
		return ErrInvalidOperation
	}

	if volume.AttachedToVM != "" {
		return ErrVolumeInUse
	}

	// Find the driver that manages this volume
	driverName := ism.getDriverForVolume(volume.Type)
	driver, exists := ism.drivers[driverName]
	if !exists {
		return fmt.Errorf("driver %s not found for volume %s", driverName, volumeID)
	}

	// Attach using the driver
	if err := driver.AttachVolume(ctx, volumeID, opts.VMID); err != nil {
		return fmt.Errorf("failed to attach volume with driver %s: %v", driverName, err)
	}

	// Update volume state
	volume.State = VolumeStateAttached
	volume.AttachedToVM = opts.VMID
	volume.UpdatedAt = time.Now()

	// Record access in tier manager
	// TODO: Re-enable tier manager
	// ism.tierManager.RecordVolumeAccess(volumeID)

	// Emit event
	ism.emitEvent(VolumeEvent{
		Type:       VolumeEventAttached,
		VolumeID:   volumeID,
		VolumeName: volume.Name,
		Timestamp:  time.Now(),
		Data:       opts,
	})

	return nil
}

// DetachVolume detaches a volume from a VM
func (ism *IntegratedStorageManager) DetachVolume(ctx context.Context, volumeID string, opts VolumeDetachOptions) error {
	ism.volumesMu.Lock()
	defer ism.volumesMu.Unlock()

	volume, exists := ism.volumes[volumeID]
	if !exists {
		return ErrVolumeNotFound
	}

	if volume.State != VolumeStateAttached {
		return ErrInvalidOperation
	}

	// Find the driver that manages this volume
	driverName := ism.getDriverForVolume(volume.Type)
	driver, exists := ism.drivers[driverName]
	if !exists {
		return fmt.Errorf("driver %s not found for volume %s", driverName, volumeID)
	}

	// Detach using the driver
	if err := driver.DetachVolume(ctx, volumeID, volume.AttachedToVM); err != nil {
		return fmt.Errorf("failed to detach volume with driver %s: %v", driverName, err)
	}

	// Update volume state
	volume.State = VolumeStateAvailable
	volume.AttachedToVM = ""
	volume.UpdatedAt = time.Now()

	// Emit event
	ism.emitEvent(VolumeEvent{
		Type:       VolumeEventDetached,
		VolumeID:   volumeID,
		VolumeName: volume.Name,
		Timestamp:  time.Now(),
		Data:       opts,
	})

	return nil
}

// GetVolume returns information about a volume
func (ism *IntegratedStorageManager) GetVolume(ctx context.Context, volumeID string) (*VolumeInfo, error) {
	ism.volumesMu.RLock()
	defer ism.volumesMu.RUnlock()

	volume, exists := ism.volumes[volumeID]
	if !exists {
		return nil, ErrVolumeNotFound
	}

	// Return a copy to avoid race conditions
	volumeCopy := *volume
	return &volumeCopy, nil
}

// ListVolumes lists all volumes
func (ism *IntegratedStorageManager) ListVolumes(ctx context.Context) ([]VolumeInfo, error) {
	ism.volumesMu.RLock()
	defer ism.volumesMu.RUnlock()

	volumes := make([]VolumeInfo, 0, len(ism.volumes))
	for _, volume := range ism.volumes {
		volumes = append(volumes, *volume)
	}

	return volumes, nil
}

// GetStorageMetrics returns current storage metrics
func (ism *IntegratedStorageManager) GetStorageMetrics(ctx context.Context) (*StorageMetrics, error) {
	return ism.metrics, nil
}

// selectBestDriver selects the best driver for a given volume type
func (ism *IntegratedStorageManager) selectBestDriver(volumeType VolumeType) string {
	switch volumeType {
	case VolumeTypeLocal, VolumeTypeEphemeral:
		if _, exists := ism.drivers["local"]; exists {
			return "local"
		}
	case VolumeTypeDistributed, VolumeTypeCeph:
		if _, exists := ism.drivers["ceph"]; exists {
			return "ceph"
		}
	}

	// Fallback to first available driver
	for name := range ism.drivers {
		return name
	}

	return ""
}

// getDriverForVolume returns the driver name for a given volume type
func (ism *IntegratedStorageManager) getDriverForVolume(volumeType VolumeType) string {
	return ism.selectBestDriver(volumeType)
}

// emitEvent emits a volume event to all listeners
func (ism *IntegratedStorageManager) emitEvent(event VolumeEvent) {
	ism.eventMu.RLock()
	listeners := make([]VolumeEventListener, len(ism.eventListeners))
	copy(listeners, ism.eventListeners)
	ism.eventMu.RUnlock()

	for _, listener := range listeners {
		go listener(event)
	}
}

// AddVolumeEventListener adds a volume event listener
func (ism *IntegratedStorageManager) AddVolumeEventListener(listener VolumeEventListener) {
	ism.eventMu.Lock()
	defer ism.eventMu.Unlock()

	ism.eventListeners = append(ism.eventListeners, listener)
}

// metricsCollectionLoop runs the metrics collection loop
func (ism *IntegratedStorageManager) metricsCollectionLoop() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			ism.updateMetrics()
		case <-ism.ctx.Done():
			return
		}
	}
}

// updateMetrics updates the storage metrics
func (ism *IntegratedStorageManager) updateMetrics() {
	ism.volumesMu.RLock()
	defer ism.volumesMu.RUnlock()

	var totalCapacity, usedCapacity int64
	activeVolumes := 0

	// Calculate metrics from volumes
	for _, volume := range ism.volumes {
		totalCapacity += volume.Size
		if volume.State == VolumeStateAttached {
			activeVolumes++
		}
		// Simplified used calculation
		usedCapacity += volume.Size / 2 // Assume 50% usage
	}

	ism.metrics.TotalCapacityBytes = totalCapacity
	ism.metrics.UsedCapacityBytes = usedCapacity
	ism.metrics.AvailableCapacityBytes = totalCapacity - usedCapacity
	ism.metrics.TotalVolumes = len(ism.volumes)
	ism.metrics.ActiveVolumes = activeVolumes
	ism.metrics.LastUpdated = time.Now()
}