package iot

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
)

// DiscoveryMethod represents different device discovery methods
type DiscoveryMethod string

const (
	DiscoveryMethodmDNS     DiscoveryMethod = "mdns"
	DiscoveryMethodUPnP     DiscoveryMethod = "upnp"
	DiscoveryMethodNetwork  DiscoveryMethod = "network"
	DiscoveryMethodBLE      DiscoveryMethod = "ble"
	DiscoveryMethodZigbee   DiscoveryMethod = "zigbee"
	DiscoveryMethodCustom   DiscoveryMethod = "custom"
)

// DiscoveredDevice represents a device discovered during scanning
type DiscoveredDevice struct {
	ID           string                 `json:"id"`
	Name         string                 `json:"name"`
	Type         DeviceType             `json:"type"`
	Protocol     Protocol               `json:"protocol"`
	MACAddress   string                 `json:"mac_address"`
	IPAddress    string                 `json:"ip_address"`
	Port         int                    `json:"port"`
	Manufacturer string                 `json:"manufacturer"`
	Model        string                 `json:"model"`
	Version      string                 `json:"version"`
	Services     []string               `json:"services"`
	Capabilities map[string]interface{} `json:"capabilities"`
	RSSI         int                    `json:"rssi"`
	DiscoveredAt time.Time              `json:"discovered_at"`
	Method       DiscoveryMethod        `json:"discovery_method"`
	Metadata     map[string]interface{} `json:"metadata"`
}

// DiscoveryPlugin interface for discovery method implementations
type DiscoveryPlugin interface {
	Name() string
	Scan(ctx context.Context, config map[string]interface{}) ([]*DiscoveredDevice, error)
	IsSupported() bool
	Configure(config map[string]interface{}) error
}

// NetworkDiscoveryPlugin implements network-based device discovery
type NetworkDiscoveryPlugin struct {
	logger     logger.Logger
	scanRanges []string
	timeout    time.Duration
}

// NewNetworkDiscoveryPlugin creates a network discovery plugin
func NewNetworkDiscoveryPlugin() *NetworkDiscoveryPlugin {
	return &NetworkDiscoveryPlugin{
		logger:     logger.GlobalLogger,
		scanRanges: []string{"192.168.1.0/24", "10.0.0.0/24"},
		timeout:    30 * time.Second,
	}
}

func (n *NetworkDiscoveryPlugin) Name() string {
	return string(DiscoveryMethodNetwork)
}

func (n *NetworkDiscoveryPlugin) IsSupported() bool {
	return true
}

func (n *NetworkDiscoveryPlugin) Configure(config map[string]interface{}) error {
	if ranges, ok := config["scan_ranges"].([]string); ok {
		n.scanRanges = ranges
	}
	if timeout, ok := config["timeout"].(time.Duration); ok {
		n.timeout = timeout
	}
	return nil
}

func (n *NetworkDiscoveryPlugin) Scan(ctx context.Context, config map[string]interface{}) ([]*DiscoveredDevice, error) {
	var allDevices []*DiscoveredDevice
	var wg sync.WaitGroup
	var mu sync.Mutex
	
	for _, scanRange := range n.scanRanges {
		wg.Add(1)
		go func(cidr string) {
			defer wg.Done()
			
			devices, err := n.scanNetwork(ctx, cidr)
			if err != nil {
				n.logger.Error("Network scan failed", "range", cidr, "error", err)
				return
			}
			
			mu.Lock()
			allDevices = append(allDevices, devices...)
			mu.Unlock()
		}(scanRange)
	}
	
	wg.Wait()
	return allDevices, nil
}

func (n *NetworkDiscoveryPlugin) scanNetwork(ctx context.Context, cidr string) ([]*DiscoveredDevice, error) {
	_, ipNet, err := net.ParseCIDR(cidr)
	if err != nil {
		return nil, fmt.Errorf("invalid CIDR: %w", err)
	}
	
	var devices []*DiscoveredDevice
	var wg sync.WaitGroup
	var mu sync.Mutex
	
	// Generate IP addresses to scan
	for ip := ipNet.IP.Mask(ipNet.Mask); ipNet.Contains(ip); n.incrementIP(ip) {
		if ctx.Err() != nil {
			break
		}
		
		wg.Add(1)
		go func(ipAddr string) {
			defer wg.Done()
			
			device := n.probeHost(ctx, ipAddr)
			if device != nil {
				mu.Lock()
				devices = append(devices, device)
				mu.Unlock()
			}
		}(ip.String())
	}
	
	wg.Wait()
	return devices, nil
}

func (n *NetworkDiscoveryPlugin) probeHost(ctx context.Context, ip string) *DiscoveredDevice {
	// Common IoT device ports to check
	ports := []int{80, 443, 8080, 1883, 5683, 22, 23, 502, 161}
	
	for _, port := range ports {
		conn, err := net.DialTimeout("tcp", fmt.Sprintf("%s:%d", ip, port), 2*time.Second)
		if err != nil {
			continue
		}
		conn.Close()
		
		// Found an open port, create discovered device
		device := &DiscoveredDevice{
			ID:           uuid.New().String(),
			Name:         fmt.Sprintf("Device_%s", ip),
			Type:         DeviceTypeSensor, // Default type
			IPAddress:    ip,
			Port:         port,
			DiscoveredAt: time.Now(),
			Method:       DiscoveryMethodNetwork,
			Metadata: map[string]interface{}{
				"open_port": port,
			},
		}
		
		// Try to identify device type based on port
		switch port {
		case 80, 443, 8080:
			device.Protocol = ProtocolHTTP
		case 1883:
			device.Protocol = ProtocolMQTT
		case 5683:
			device.Protocol = ProtocolCoAP
		default:
			device.Protocol = ProtocolHTTP
		}
		
		return device
	}
	
	return nil
}

func (n *NetworkDiscoveryPlugin) incrementIP(ip net.IP) {
	for j := len(ip) - 1; j >= 0; j-- {
		ip[j]++
		if ip[j] > 0 {
			break
		}
	}
}

// mDNSDiscoveryPlugin implements mDNS-based device discovery
type mDNSDiscoveryPlugin struct {
	logger   logger.Logger
	services []string
}

func NewmDNSDiscoveryPlugin() *mDNSDiscoveryPlugin {
	return &mDNSDiscoveryPlugin{
		logger: logger.GlobalLogger,
		services: []string{
			"_http._tcp.local.",
			"_iot._tcp.local.",
			"_mqtt._tcp.local.",
			"_coap._udp.local.",
		},
	}
}

func (m *mDNSDiscoveryPlugin) Name() string {
	return string(DiscoveryMethodmDNS)
}

func (m *mDNSDiscoveryPlugin) IsSupported() bool {
	// Check if mDNS is available on the system
	return true
}

func (m *mDNSDiscoveryPlugin) Configure(config map[string]interface{}) error {
	if services, ok := config["services"].([]string); ok {
		m.services = services
	}
	return nil
}

func (m *mDNSDiscoveryPlugin) Scan(ctx context.Context, config map[string]interface{}) ([]*DiscoveredDevice, error) {
	// Mock mDNS discovery for now
	// In a real implementation, this would use actual mDNS libraries like github.com/hashicorp/mdns
	
	devices := []*DiscoveredDevice{
		{
			ID:           uuid.New().String(),
			Name:         "Smart Thermostat",
			Type:         DeviceTypeSensor,
			Protocol:     ProtocolHTTP,
			IPAddress:    "192.168.1.100",
			Port:         80,
			Manufacturer: "IoT Corp",
			Model:        "ThermoStat-Pro",
			Version:      "2.1.0",
			Services:     []string{"_http._tcp.local."},
			DiscoveredAt: time.Now(),
			Method:       DiscoveryMethodmDNS,
		},
		{
			ID:           uuid.New().String(),
			Name:         "Security Camera",
			Type:         DeviceTypeCamera,
			Protocol:     ProtocolHTTP,
			IPAddress:    "192.168.1.101",
			Port:         8080,
			Manufacturer: "SecureCam",
			Model:        "Cam-HD-1080",
			Version:      "1.5.2",
			Services:     []string{"_http._tcp.local.", "_rtsp._tcp.local."},
			DiscoveredAt: time.Now(),
			Method:       DiscoveryMethodmDNS,
		},
	}
	
	return devices, nil
}

// BLEDiscoveryPlugin implements Bluetooth Low Energy device discovery
type BLEDiscoveryPlugin struct {
	logger     logger.Logger
	scanTime   time.Duration
	deviceUUIDs []string
}

func NewBLEDiscoveryPlugin() *BLEDiscoveryPlugin {
	return &BLEDiscoveryPlugin{
		logger:   logger.GlobalLogger,
		scanTime: 30 * time.Second,
	}
}

func (b *BLEDiscoveryPlugin) Name() string {
	return string(DiscoveryMethodBLE)
}

func (b *BLEDiscoveryPlugin) IsSupported() bool {
	// Check if BLE is available on the system
	return false // Disabled for mock implementation
}

func (b *BLEDiscoveryPlugin) Configure(config map[string]interface{}) error {
	if scanTime, ok := config["scan_time"].(time.Duration); ok {
		b.scanTime = scanTime
	}
	if uuids, ok := config["device_uuids"].([]string); ok {
		b.deviceUUIDs = uuids
	}
	return nil
}

func (b *BLEDiscoveryPlugin) Scan(ctx context.Context, config map[string]interface{}) ([]*DiscoveredDevice, error) {
	// Mock BLE discovery
	devices := []*DiscoveredDevice{
		{
			ID:           uuid.New().String(),
			Name:         "Fitness Tracker",
			Type:         DeviceTypeWearable,
			Protocol:     ProtocolBLE,
			MACAddress:   "AA:BB:CC:DD:EE:FF",
			Manufacturer: "FitTech",
			Model:        "Tracker-X1",
			Version:      "3.0.1",
			RSSI:         -60,
			DiscoveredAt: time.Now(),
			Method:       DiscoveryMethodBLE,
			Capabilities: map[string]interface{}{
				"heart_rate": true,
				"steps":      true,
				"sleep":      true,
			},
		},
	}
	
	return devices, nil
}

// DeviceDiscovery manages automatic device discovery
type DeviceDiscovery struct {
	plugins    map[string]DiscoveryPlugin
	config     *DeviceDiscoveryConfig
	logger     logger.Logger
	ctx        context.Context
	cancel     context.CancelFunc
	running    bool
	mu         sync.RWMutex
	
	// Discovery cache
	discoveryCache map[string]*DiscoveredDevice
	cacheMu        sync.RWMutex
	cacheExpiry    time.Duration
}

// NewDeviceDiscovery creates a new device discovery manager
func NewDeviceDiscovery(config *DeviceDiscoveryConfig) *DeviceDiscovery {
	ctx, cancel := context.WithCancel(context.Background())
	
	discovery := &DeviceDiscovery{
		plugins:        make(map[string]DiscoveryPlugin),
		config:         config,
		logger:         logger.GlobalLogger,
		ctx:            ctx,
		cancel:         cancel,
		discoveryCache: make(map[string]*DiscoveredDevice),
		cacheExpiry:    10 * time.Minute,
	}
	
	// Register default plugins
	discovery.RegisterPlugin(NewNetworkDiscoveryPlugin())
	discovery.RegisterPlugin(NewmDNSDiscoveryPlugin())
	discovery.RegisterPlugin(NewBLEDiscoveryPlugin())
	
	return discovery
}

// RegisterPlugin registers a discovery plugin
func (dd *DeviceDiscovery) RegisterPlugin(plugin DiscoveryPlugin) error {
	if !plugin.IsSupported() {
		dd.logger.Warn("Discovery plugin not supported on this system", "plugin", plugin.Name())
		return nil
	}
	
	dd.plugins[plugin.Name()] = plugin
	dd.logger.Info("Registered discovery plugin", "plugin", plugin.Name())
	return nil
}

// ScanForDevices performs a comprehensive device discovery scan
func (dd *DeviceDiscovery) ScanForDevices() ([]*IoTDevice, error) {
	dd.logger.Info("Starting device discovery scan")
	
	var allDiscovered []*DiscoveredDevice
	var wg sync.WaitGroup
	var mu sync.Mutex
	
	// Run discovery plugins in parallel
	for name, plugin := range dd.plugins {
		wg.Add(1)
		go func(pluginName string, p DiscoveryPlugin) {
			defer wg.Done()
			
			ctx, cancel := context.WithTimeout(dd.ctx, 60*time.Second)
			defer cancel()
			
			devices, err := p.Scan(ctx, map[string]interface{}{})
			if err != nil {
				dd.logger.Error("Discovery plugin scan failed", 
					"plugin", pluginName, 
					"error", err,
				)
				return
			}
			
			dd.logger.Debug("Discovery plugin scan completed",
				"plugin", pluginName,
				"devices_found", len(devices),
			)
			
			mu.Lock()
			allDiscovered = append(allDiscovered, devices...)
			mu.Unlock()
		}(name, plugin)
	}
	
	wg.Wait()
	
	// Convert discovered devices to IoT devices
	devices := dd.convertToIoTDevices(allDiscovered)
	
	// Update discovery cache
	dd.updateDiscoveryCache(allDiscovered)
	
	dd.logger.Info("Device discovery scan completed",
		"total_discovered", len(allDiscovered),
		"unique_devices", len(devices),
	)
	
	return devices, nil
}

// convertToIoTDevices converts discovered devices to IoT devices
func (dd *DeviceDiscovery) convertToIoTDevices(discovered []*DiscoveredDevice) []*IoTDevice {
	deviceMap := make(map[string]*IoTDevice)
	
	for _, disc := range discovered {
		// Use MAC address or IP as unique identifier
		key := disc.MACAddress
		if key == "" {
			key = disc.IPAddress
		}
		
		// Skip if we've already processed this device
		if _, exists := deviceMap[key]; exists {
			continue
		}
		
		device := &IoTDevice{
			ID:           disc.ID,
			Name:         disc.Name,
			Type:         disc.Type,
			Status:       DeviceStatusProvisioning,
			Protocol:     disc.Protocol,
			Manufacturer: disc.Manufacturer,
			Model:        disc.Model,
			Version:      disc.Version,
			MACAddress:   disc.MACAddress,
			IPAddress:    disc.IPAddress,
			CreatedAt:    disc.DiscoveredAt,
			UpdatedAt:    disc.DiscoveredAt,
			LastSeen:     disc.DiscoveredAt,
			Metadata: map[string]interface{}{
				"discovery_method": disc.Method,
				"rssi":            disc.RSSI,
				"services":        disc.Services,
				"capabilities":    disc.Capabilities,
			},
		}
		
		// Set basic configuration
		device.Configuration = DeviceConfiguration{
			SampleRate:        1000, // 1Hz default
			ReportingInterval: 60 * time.Second,
			BufferSize:        1000,
			DataFormat:        "json",
		}
		
		// Set security defaults
		device.Security = DeviceSecurity{
			EncryptionEnabled: true,
			AuthMethod:        "certificate",
			SecurityLevel:     "medium",
		}
		
		// Set basic requirements based on device type
		device.Requirements = &DeviceRequirements{
			MinBandwidth:    1024,    // 1KB/s
			MaxLatency:      1000,    // 1s
			SecurityLevel:   "medium",
			Protocols:       []Protocol{disc.Protocol},
		}
		
		// Enhanced requirements for specific device types
		switch disc.Type {
		case DeviceTypeCamera:
			device.Requirements.MinBandwidth = 1024 * 1024 // 1MB/s
			device.Requirements.MaxLatency = 100            // 100ms
			device.Requirements.RequiresEdgeCompute = true
		case DeviceTypeActuator:
			device.Requirements.MaxLatency = 50 // 50ms
			device.Requirements.RequiresAI = true
		case DeviceTypeController:
			device.Requirements.RequiresEdgeCompute = true
			device.Requirements.RequiresAI = true
		}
		
		deviceMap[key] = device
	}
	
	// Convert map to slice
	devices := make([]*IoTDevice, 0, len(deviceMap))
	for _, device := range deviceMap {
		devices = append(devices, device)
	}
	
	return devices
}

// updateDiscoveryCache updates the internal discovery cache
func (dd *DeviceDiscovery) updateDiscoveryCache(devices []*DiscoveredDevice) {
	dd.cacheMu.Lock()
	defer dd.cacheMu.Unlock()
	
	now := time.Now()
	expiry := now.Add(-dd.cacheExpiry)
	
	// Remove expired entries
	for id, device := range dd.discoveryCache {
		if device.DiscoveredAt.Before(expiry) {
			delete(dd.discoveryCache, id)
		}
	}
	
	// Add new discoveries
	for _, device := range devices {
		dd.discoveryCache[device.ID] = device
	}
}

// GetCachedDevices returns cached discovered devices
func (dd *DeviceDiscovery) GetCachedDevices() []*DiscoveredDevice {
	dd.cacheMu.RLock()
	defer dd.cacheMu.RUnlock()
	
	devices := make([]*DiscoveredDevice, 0, len(dd.discoveryCache))
	for _, device := range dd.discoveryCache {
		devices = append(devices, device)
	}
	
	return devices
}

// Start starts the device discovery process
func (dd *DeviceDiscovery) Start() error {
	dd.mu.Lock()
	defer dd.mu.Unlock()
	
	if dd.running {
		return fmt.Errorf("device discovery already running")
	}
	
	dd.running = true
	dd.logger.Info("Device discovery started")
	
	return nil
}

// Stop stops the device discovery process
func (dd *DeviceDiscovery) Stop() error {
	dd.mu.Lock()
	defer dd.mu.Unlock()
	
	if !dd.running {
		return nil
	}
	
	dd.cancel()
	dd.running = false
	dd.logger.Info("Device discovery stopped")
	
	return nil
}

// IsRunning returns whether discovery is currently running
func (dd *DeviceDiscovery) IsRunning() bool {
	dd.mu.RLock()
	defer dd.mu.RUnlock()
	return dd.running
}

// GetSupportedMethods returns supported discovery methods
func (dd *DeviceDiscovery) GetSupportedMethods() []string {
	var methods []string
	for name, plugin := range dd.plugins {
		if plugin.IsSupported() {
			methods = append(methods, name)
		}
	}
	return methods
}