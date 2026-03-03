package iot

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
)

// DeviceType represents different types of IoT devices
type DeviceType string

const (
	DeviceTypeSensor     DeviceType = "sensor"
	DeviceTypeActuator   DeviceType = "actuator"
	DeviceTypeCamera     DeviceType = "camera"
	DeviceTypeGateway    DeviceType = "gateway"
	DeviceTypeController DeviceType = "controller"
	DeviceTypeDisplay    DeviceType = "display"
	DeviceTypeBeacon     DeviceType = "beacon"
	DeviceTypeWearable   DeviceType = "wearable"
)

// DeviceStatus represents the status of an IoT device
type DeviceStatus string

const (
	DeviceStatusOnline      DeviceStatus = "online"
	DeviceStatusOffline     DeviceStatus = "offline"
	DeviceStatusSleeping    DeviceStatus = "sleeping"
	DeviceStatusError       DeviceStatus = "error"
	DeviceStatusMaintenance DeviceStatus = "maintenance"
	DeviceStatusProvisioning DeviceStatus = "provisioning"
)

// Protocol represents communication protocols supported by devices
type Protocol string

const (
	ProtocolMQTT    Protocol = "mqtt"
	ProtocolHTTP    Protocol = "http"
	ProtocolCoAP    Protocol = "coap"
	ProtocolLoRaWAN Protocol = "lorawan"
	ProtocolZigbee  Protocol = "zigbee"
	ProtocolBLE     Protocol = "ble"
	ProtocolWiFi    Protocol = "wifi"
	ProtocolSigfox  Protocol = "sigfox"
)

// DeviceRequirements defines requirements for device placement
type DeviceRequirements struct {
	MinBandwidth       int64    `json:"min_bandwidth"`
	MaxLatency         int      `json:"max_latency_ms"`
	RequiresEdgeCompute bool     `json:"requires_edge_compute"`
	RequiresAI         bool     `json:"requires_ai"`
	PreferredRegion    string   `json:"preferred_region"`
	SecurityLevel      string   `json:"security_level"`
	Protocols          []Protocol `json:"protocols"`
}

// DeviceSecurity defines security configuration for devices
type DeviceSecurity struct {
	EncryptionEnabled bool              `json:"encryption_enabled"`
	AuthMethod        string            `json:"auth_method"`
	Certificates      map[string]string `json:"certificates"`
	Keys              map[string]string `json:"keys"`
	SecurityLevel     string            `json:"security_level"`
}

// DeviceConfiguration holds device-specific configuration
type DeviceConfiguration struct {
	SampleRate      int                    `json:"sample_rate"`
	ReportingInterval time.Duration        `json:"reporting_interval"`
	BufferSize      int                    `json:"buffer_size"`
	CompressionEnabled bool                `json:"compression_enabled"`
	DataFormat      string                 `json:"data_format"`
	Filters         []string               `json:"filters"`
	CustomConfig    map[string]interface{} `json:"custom_config"`
}

// DeviceMetrics contains device performance metrics
type DeviceMetrics struct {
	MessagesReceived  int64         `json:"messages_received"`
	MessagesProcessed int64         `json:"messages_processed"`
	MessagesSent      int64         `json:"messages_sent"`
	ErrorCount        int64         `json:"error_count"`
	AverageLatency    time.Duration `json:"average_latency"`
	BatteryLevel      float64       `json:"battery_level"`
	SignalStrength    float64       `json:"signal_strength"`
	UptimePercent     float64       `json:"uptime_percent"`
	LastUpdate        time.Time     `json:"last_update"`
}

// IoTDevice represents an IoT device
type IoTDevice struct {
	ID            string                  `json:"id"`
	Name          string                  `json:"name"`
	Type          DeviceType              `json:"type"`
	Status        DeviceStatus            `json:"status"`
	GatewayID     string                  `json:"gateway_id"`
	Protocol      Protocol                `json:"protocol"`
	Manufacturer  string                  `json:"manufacturer"`
	Model         string                  `json:"model"`
	Version       string                  `json:"version"`
	SerialNumber  string                  `json:"serial_number"`
	MACAddress    string                  `json:"mac_address"`
	IPAddress     string                  `json:"ip_address,omitempty"`
	Location      DeviceLocation          `json:"location"`
	Configuration DeviceConfiguration     `json:"configuration"`
	Security      DeviceSecurity          `json:"security"`
	Requirements  *DeviceRequirements     `json:"requirements,omitempty"`
	Metrics       *DeviceMetrics          `json:"metrics,omitempty"`
	LastSeen      time.Time               `json:"last_seen"`
	CreatedAt     time.Time               `json:"created_at"`
	UpdatedAt     time.Time               `json:"updated_at"`
	Metadata      map[string]interface{}  `json:"metadata"`
}

// DeviceLocation represents device physical location
type DeviceLocation struct {
	Latitude    float64 `json:"latitude"`
	Longitude   float64 `json:"longitude"`
	Altitude    float64 `json:"altitude"`
	Building    string  `json:"building"`
	Floor       string  `json:"floor"`
	Room        string  `json:"room"`
	Zone        string  `json:"zone"`
	Description string  `json:"description"`
}

// DeviceDiscoveryConfig defines discovery configuration
type DeviceDiscoveryConfig struct {
	Enabled           bool          `json:"enabled"`
	DiscoveryInterval time.Duration `json:"discovery_interval"`
	AutoProvisioning  bool          `json:"auto_provisioning"`
	Protocols         []Protocol    `json:"protocols"`
	NetworkScanRanges []string      `json:"network_scan_ranges"`
	MaxDevicesPerScan int           `json:"max_devices_per_scan"`
}

// DeviceManager manages IoT devices and their lifecycle
type DeviceManager struct {
	devices         map[string]*IoTDevice
	devicesByGateway map[string][]*IoTDevice
	discovery       *DeviceDiscovery
	mu              sync.RWMutex
	logger          logger.Logger
	ctx             context.Context
	cancel          context.CancelFunc
	
	// Event channels
	deviceEvents    chan DeviceEvent
	
	// Configuration
	config          *DeviceManagerConfig
}

// DeviceManagerConfig defines device manager configuration
type DeviceManagerConfig struct {
	HeartbeatInterval    time.Duration         `json:"heartbeat_interval"`
	OfflineThreshold     time.Duration         `json:"offline_threshold"`
	MetricsRetention     time.Duration         `json:"metrics_retention"`
	DiscoveryConfig      DeviceDiscoveryConfig `json:"discovery_config"`
	AutoMigration        bool                  `json:"auto_migration"`
	LoadBalancing        bool                  `json:"load_balancing"`
}

// NewDeviceManager creates a new device manager
func NewDeviceManager() *DeviceManager {
	ctx, cancel := context.WithCancel(context.Background())
	
	config := &DeviceManagerConfig{
		HeartbeatInterval: 30 * time.Second,
		OfflineThreshold:  5 * time.Minute,
		MetricsRetention:  24 * time.Hour,
		DiscoveryConfig: DeviceDiscoveryConfig{
			Enabled:           true,
			DiscoveryInterval: 5 * time.Minute,
			AutoProvisioning:  false,
			Protocols:         []Protocol{ProtocolMQTT, ProtocolHTTP, ProtocolBLE},
			MaxDevicesPerScan: 100,
		},
		AutoMigration: true,
		LoadBalancing: true,
	}
	
	return &DeviceManager{
		devices:          make(map[string]*IoTDevice),
		devicesByGateway: make(map[string][]*IoTDevice),
		discovery:        NewDeviceDiscovery(&config.DiscoveryConfig),
		logger:          logger.GlobalLogger,
		ctx:             ctx,
		cancel:          cancel,
		deviceEvents:    make(chan DeviceEvent, 1000),
		config:          config,
	}
}

// RegisterDevice registers a new IoT device
func (dm *DeviceManager) RegisterDevice(device *IoTDevice) error {
	dm.mu.Lock()
	defer dm.mu.Unlock()
	
	if device.ID == "" {
		device.ID = uuid.New().String()
	}
	
	device.CreatedAt = time.Now()
	device.UpdatedAt = time.Now()
	device.LastSeen = time.Now()
	device.Status = DeviceStatusOnline
	
	// Initialize metrics if not provided
	if device.Metrics == nil {
		device.Metrics = &DeviceMetrics{
			LastUpdate: time.Now(),
		}
	}
	
	dm.devices[device.ID] = device
	
	// Add to gateway mapping
	if device.GatewayID != "" {
		dm.devicesByGateway[device.GatewayID] = append(
			dm.devicesByGateway[device.GatewayID], device)
	}
	
	dm.logger.Info("IoT device registered",
		"device_id", device.ID,
		"device_name", device.Name,
		"device_type", device.Type,
		"gateway_id", device.GatewayID,
	)
	
	// Send registration event
	dm.sendDeviceEvent("device.registered", device.ID, device.GatewayID, map[string]interface{}{
		"name": device.Name,
		"type": device.Type,
	})
	
	return nil
}

// UnregisterDevice removes a device from management
func (dm *DeviceManager) UnregisterDevice(deviceID string) error {
	dm.mu.Lock()
	defer dm.mu.Unlock()
	
	device, exists := dm.devices[deviceID]
	if !exists {
		return fmt.Errorf("device not found: %s", deviceID)
	}
	
	// Remove from gateway mapping
	if device.GatewayID != "" {
		devices := dm.devicesByGateway[device.GatewayID]
		for i, d := range devices {
			if d.ID == deviceID {
				dm.devicesByGateway[device.GatewayID] = append(devices[:i], devices[i+1:]...)
				break
			}
		}
	}
	
	delete(dm.devices, deviceID)
	
	dm.logger.Info("IoT device unregistered", "device_id", deviceID)
	
	// Send unregistration event
	dm.sendDeviceEvent("device.unregistered", deviceID, device.GatewayID, map[string]interface{}{
		"name": device.Name,
	})
	
	return nil
}

// UpdateDeviceStatus updates device status and metrics
func (dm *DeviceManager) UpdateDeviceStatus(deviceID string, status DeviceStatus, metrics *DeviceMetrics) error {
	dm.mu.Lock()
	defer dm.mu.Unlock()
	
	device, exists := dm.devices[deviceID]
	if !exists {
		return fmt.Errorf("device not found: %s", deviceID)
	}
	
	oldStatus := device.Status
	device.Status = status
	device.UpdatedAt = time.Now()
	device.LastSeen = time.Now()
	
	if metrics != nil {
		device.Metrics = metrics
		device.Metrics.LastUpdate = time.Now()
	}
	
	// Send status change event if status changed
	if oldStatus != status {
		dm.sendDeviceEvent("device.status_changed", deviceID, device.GatewayID, map[string]interface{}{
			"old_status": oldStatus,
			"new_status": status,
		})
	}
	
	return nil
}

// GetDevice retrieves a device by ID
func (dm *DeviceManager) GetDevice(deviceID string) (*IoTDevice, error) {
	dm.mu.RLock()
	defer dm.mu.RUnlock()
	
	device, exists := dm.devices[deviceID]
	if !exists {
		return nil, fmt.Errorf("device not found: %s", deviceID)
	}
	
	return device, nil
}

// ListDevices returns all registered devices
func (dm *DeviceManager) ListDevices() []*IoTDevice {
	dm.mu.RLock()
	defer dm.mu.RUnlock()
	
	devices := make([]*IoTDevice, 0, len(dm.devices))
	for _, device := range dm.devices {
		devices = append(devices, device)
	}
	
	return devices
}

// GetDevicesByGateway returns devices connected to a specific gateway
func (dm *DeviceManager) GetDevicesByGateway(gatewayID string) ([]*IoTDevice, error) {
	dm.mu.RLock()
	defer dm.mu.RUnlock()
	
	devices, exists := dm.devicesByGateway[gatewayID]
	if !exists {
		return []*IoTDevice{}, nil
	}
	
	return devices, nil
}

// GetDevicesByType returns devices of a specific type
func (dm *DeviceManager) GetDevicesByType(deviceType DeviceType) []*IoTDevice {
	dm.mu.RLock()
	defer dm.mu.RUnlock()
	
	var devices []*IoTDevice
	for _, device := range dm.devices {
		if device.Type == deviceType {
			devices = append(devices, device)
		}
	}
	
	return devices
}

// GetDevicesByStatus returns devices with a specific status
func (dm *DeviceManager) GetDevicesByStatus(status DeviceStatus) []*IoTDevice {
	dm.mu.RLock()
	defer dm.mu.RUnlock()
	
	var devices []*IoTDevice
	for _, device := range dm.devices {
		if device.Status == status {
			devices = append(devices, device)
		}
	}
	
	return devices
}

// MigrateDevice moves a device from one gateway to another
func (dm *DeviceManager) MigrateDevice(deviceID, newGatewayID string) error {
	dm.mu.Lock()
	defer dm.mu.Unlock()
	
	device, exists := dm.devices[deviceID]
	if !exists {
		return fmt.Errorf("device not found: %s", deviceID)
	}
	
	oldGatewayID := device.GatewayID
	
	// Remove from old gateway
	if oldGatewayID != "" {
		devices := dm.devicesByGateway[oldGatewayID]
		for i, d := range devices {
			if d.ID == deviceID {
				dm.devicesByGateway[oldGatewayID] = append(devices[:i], devices[i+1:]...)
				break
			}
		}
	}
	
	// Add to new gateway
	device.GatewayID = newGatewayID
	device.UpdatedAt = time.Now()
	dm.devicesByGateway[newGatewayID] = append(dm.devicesByGateway[newGatewayID], device)
	
	dm.logger.Info("Device migrated",
		"device_id", deviceID,
		"from_gateway", oldGatewayID,
		"to_gateway", newGatewayID,
	)
	
	// Send migration event
	dm.sendDeviceEvent("device.migrated", deviceID, newGatewayID, map[string]interface{}{
		"old_gateway": oldGatewayID,
		"new_gateway": newGatewayID,
	})
	
	return nil
}

// DisconnectGatewayDevices disconnects all devices from a gateway
func (dm *DeviceManager) DisconnectGatewayDevices(gatewayID string) error {
	dm.mu.Lock()
	defer dm.mu.Unlock()
	
	devices := dm.devicesByGateway[gatewayID]
	for _, device := range devices {
		device.GatewayID = ""
		device.Status = DeviceStatusOffline
		device.UpdatedAt = time.Now()
		
		dm.sendDeviceEvent("device.disconnected", device.ID, gatewayID, nil)
	}
	
	delete(dm.devicesByGateway, gatewayID)
	
	dm.logger.Info("Disconnected all devices from gateway", "gateway_id", gatewayID, "count", len(devices))
	
	return nil
}

// Start starts the device manager
func (dm *DeviceManager) Start() error {
	dm.logger.Info("Starting Device Manager")
	
	// Start monitoring goroutines
	go dm.monitorDevices()
	go dm.processEvents()
	
	// Start device discovery if enabled
	if dm.config.DiscoveryConfig.Enabled {
		go dm.startDeviceDiscovery()
	}
	
	dm.logger.Info("Device Manager started successfully")
	return nil
}

// Stop stops the device manager
func (dm *DeviceManager) Stop() error {
	dm.logger.Info("Stopping Device Manager")
	
	dm.cancel()
	
	// Stop discovery
	if err := dm.discovery.Stop(); err != nil {
		dm.logger.Error("Error stopping device discovery", "error", err)
	}
	
	// Close channels
	close(dm.deviceEvents)
	
	dm.logger.Info("Device Manager stopped")
	return nil
}

// monitorDevices monitors device health and status
func (dm *DeviceManager) monitorDevices() {
	ticker := time.NewTicker(dm.config.HeartbeatInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-dm.ctx.Done():
			return
		case <-ticker.C:
			dm.checkDeviceHealth()
		}
	}
}

// checkDeviceHealth checks the health of all devices
func (dm *DeviceManager) checkDeviceHealth() {
	dm.mu.Lock()
	defer dm.mu.Unlock()
	
	threshold := time.Now().Add(-dm.config.OfflineThreshold)
	
	for id, device := range dm.devices {
		if device.Status == DeviceStatusOnline && device.LastSeen.Before(threshold) {
			device.Status = DeviceStatusOffline
			device.UpdatedAt = time.Now()
			
			dm.logger.Warn("Device marked offline", "device_id", id)
			dm.sendDeviceEvent("device.offline", id, device.GatewayID, map[string]interface{}{
				"last_seen": device.LastSeen,
			})
		}
	}
}

// startDeviceDiscovery starts automatic device discovery
func (dm *DeviceManager) startDeviceDiscovery() {
	ticker := time.NewTicker(dm.config.DiscoveryConfig.DiscoveryInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-dm.ctx.Done():
			return
		case <-ticker.C:
			dm.discoverDevices()
		}
	}
}

// discoverDevices performs device discovery
func (dm *DeviceManager) discoverDevices() {
	dm.logger.Debug("Starting device discovery scan")
	
	discoveredDevices, err := dm.discovery.ScanForDevices()
	if err != nil {
		dm.logger.Error("Device discovery failed", "error", err)
		return
	}
	
	for _, device := range discoveredDevices {
		// Check if device is already registered
		if _, exists := dm.devices[device.ID]; exists {
			continue
		}
		
		// Auto-provision if enabled
		if dm.config.DiscoveryConfig.AutoProvisioning {
			if err := dm.RegisterDevice(device); err != nil {
				dm.logger.Error("Failed to auto-register discovered device",
					"device_id", device.ID,
					"error", err,
				)
			} else {
				dm.logger.Info("Auto-registered discovered device",
					"device_id", device.ID,
					"device_name", device.Name,
				)
			}
		} else {
			dm.logger.Info("Discovered new device (not auto-provisioned)",
				"device_id", device.ID,
				"device_name", device.Name,
			)
		}
	}
	
	dm.logger.Debug("Device discovery scan completed", "discovered", len(discoveredDevices))
}

// processEvents processes device events
func (dm *DeviceManager) processEvents() {
	for {
		select {
		case <-dm.ctx.Done():
			return
		case event := <-dm.deviceEvents:
			dm.handleDeviceEvent(event)
		}
	}
}

// sendDeviceEvent sends a device event
func (dm *DeviceManager) sendDeviceEvent(eventType, deviceID, gatewayID string, data map[string]interface{}) {
	event := DeviceEvent{
		Type:      eventType,
		DeviceID:  deviceID,
		GatewayID: gatewayID,
		Timestamp: time.Now(),
		Data:      data,
	}
	
	select {
	case dm.deviceEvents <- event:
	default:
		dm.logger.Warn("Device event channel full, dropping event", "type", eventType)
	}
}

// handleDeviceEvent handles device events
func (dm *DeviceManager) handleDeviceEvent(event DeviceEvent) {
	dm.logger.Debug("Handling device event",
		"type", event.Type,
		"device_id", event.DeviceID,
		"gateway_id", event.GatewayID,
	)
	
	// Process event based on type
	switch event.Type {
	case "device.offline":
		dm.handleDeviceOffline(event.DeviceID)
	case "device.error":
		dm.handleDeviceError(event.DeviceID, event.Data)
	}
}

// handleDeviceOffline handles device going offline
func (dm *DeviceManager) handleDeviceOffline(deviceID string) {
	// Trigger migration if auto-migration is enabled
	if dm.config.AutoMigration {
		device, err := dm.GetDevice(deviceID)
		if err != nil {
			dm.logger.Error("Failed to get device for offline handling", "device_id", deviceID, "error", err)
			return
		}
		
		// TODO: Implement intelligent device migration
		dm.logger.Debug("Device offline migration not implemented", "device_id", deviceID)
	}
}

// handleDeviceError handles device errors
func (dm *DeviceManager) handleDeviceError(deviceID string, data map[string]interface{}) {
	dm.logger.Warn("Device error reported", "device_id", deviceID, "data", data)
	
	// Update device metrics to reflect error
	dm.mu.Lock()
	device, exists := dm.devices[deviceID]
	if exists && device.Metrics != nil {
		device.Metrics.ErrorCount++
		device.Metrics.LastUpdate = time.Now()
	}
	dm.mu.Unlock()
}

// GetMetrics returns device manager metrics
func (dm *DeviceManager) GetMetrics() map[string]interface{} {
	dm.mu.RLock()
	defer dm.mu.RUnlock()
	
	totalDevices := len(dm.devices)
	onlineDevices := 0
	devicesByType := make(map[string]int)
	devicesByStatus := make(map[string]int)
	
	for _, device := range dm.devices {
		if device.Status == DeviceStatusOnline {
			onlineDevices++
		}
		
		devicesByType[string(device.Type)]++
		devicesByStatus[string(device.Status)]++
	}
	
	return map[string]interface{}{
		"total_devices":     totalDevices,
		"online_devices":    onlineDevices,
		"offline_devices":   totalDevices - onlineDevices,
		"devices_by_type":   devicesByType,
		"devices_by_status": devicesByStatus,
		"discovery_enabled": dm.config.DiscoveryConfig.Enabled,
	}
}