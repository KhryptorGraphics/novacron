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

// GatewayStatus represents the status of an IoT gateway
type GatewayStatus string

const (
	GatewayStatusOnline     GatewayStatus = "online"
	GatewayStatusOffline    GatewayStatus = "offline"
	GatewayStatusMaintenance GatewayStatus = "maintenance"
	GatewayStatusError      GatewayStatus = "error"
)

// GatewayCapabilities defines what an IoT gateway can do
type GatewayCapabilities struct {
	MaxDevices       int    `json:"max_devices"`
	EdgeCompute     bool   `json:"edge_compute"`
	AIInference     bool   `json:"ai_inference"`
	LocalStorage    bool   `json:"local_storage"`
	NetworkProtocols []string `json:"network_protocols"`
	SensorTypes     []string `json:"sensor_types"`
}

// GatewayResources represents current resource usage
type GatewayResources struct {
	CPUUsage     float64 `json:"cpu_usage"`
	MemoryUsage  float64 `json:"memory_usage"`
	StorageUsed  int64   `json:"storage_used"`
	StorageTotal int64   `json:"storage_total"`
	NetworkTx    int64   `json:"network_tx"`
	NetworkRx    int64   `json:"network_rx"`
	DeviceCount  int     `json:"device_count"`
	Temperature  float64 `json:"temperature"`
}

// GatewayLocation represents geographical and network location
type GatewayLocation struct {
	Latitude    float64 `json:"latitude"`
	Longitude   float64 `json:"longitude"`
	Altitude    float64 `json:"altitude"`
	Region      string  `json:"region"`
	Zone        string  `json:"zone"`
	DataCenter  string  `json:"data_center"`
	NetworkZone string  `json:"network_zone"`
}

// IoTGateway represents an IoT gateway node
type IoTGateway struct {
	ID           string               `json:"id"`
	Name         string               `json:"name"`
	Status       GatewayStatus        `json:"status"`
	Location     GatewayLocation      `json:"location"`
	Capabilities GatewayCapabilities  `json:"capabilities"`
	Resources    GatewayResources     `json:"resources"`
	LastSeen     time.Time            `json:"last_seen"`
	CreatedAt    time.Time            `json:"created_at"`
	UpdatedAt    time.Time            `json:"updated_at"`
	Metadata     map[string]interface{} `json:"metadata"`
	
	// Edge compute configuration
	EdgeConfig   *EdgeConfig          `json:"edge_config,omitempty"`
	
	// Connected devices
	DeviceCount  int                  `json:"device_count"`
	ConnectedDevices []string         `json:"connected_devices"`
	
	// Performance metrics
	Metrics      *GatewayMetrics      `json:"metrics,omitempty"`
}

// EdgeConfig defines edge computing configuration
type EdgeConfig struct {
	Enabled         bool                   `json:"enabled"`
	ModelRegistry   string                 `json:"model_registry"`
	InferenceModels []string               `json:"inference_models"`
	DataRetention   time.Duration          `json:"data_retention"`
	AnalyticsEnabled bool                  `json:"analytics_enabled"`
	StreamProcessing bool                  `json:"stream_processing"`
	BufferSize      int                    `json:"buffer_size"`
	BatchSize       int                    `json:"batch_size"`
	Config          map[string]interface{} `json:"config"`
}

// GatewayMetrics contains detailed performance metrics
type GatewayMetrics struct {
	MessagesProcessed int64     `json:"messages_processed"`
	MessagesPerSecond float64   `json:"messages_per_second"`
	AverageLatency    time.Duration `json:"average_latency"`
	ErrorCount        int64     `json:"error_count"`
	UptimePercent     float64   `json:"uptime_percent"`
	LastUpdate        time.Time `json:"last_update"`
	
	// Analytics metrics
	AnalyticsProcessed int64   `json:"analytics_processed"`
	AnalyticsLatency   time.Duration `json:"analytics_latency"`
	InferenceCount     int64   `json:"inference_count"`
	InferenceLatency   time.Duration `json:"inference_latency"`
}

// GatewayOrchestrator manages IoT gateways as compute nodes
type GatewayOrchestrator struct {
	gateways      map[string]*IoTGateway
	deviceManager *DeviceManager
	analytics     *EdgeAnalytics
	mu            sync.RWMutex
	logger        logger.Logger
	ctx           context.Context
	cancel        context.CancelFunc
	
	// Event channels
	gatewayEvents chan GatewayEvent
	deviceEvents  chan DeviceEvent
	
	// Configuration
	config *OrchestratorConfig
}

// OrchestratorConfig defines orchestrator configuration
type OrchestratorConfig struct {
	HeartbeatInterval    time.Duration `json:"heartbeat_interval"`
	OfflineThreshold     time.Duration `json:"offline_threshold"`
	MaxDevicesPerGateway int           `json:"max_devices_per_gateway"`
	MetricsRetention     time.Duration `json:"metrics_retention"`
	AnalyticsEnabled     bool          `json:"analytics_enabled"`
	AutoDiscovery        bool          `json:"auto_discovery"`
	LoadBalancing        bool          `json:"load_balancing"`
}

// GatewayEvent represents gateway lifecycle events
type GatewayEvent struct {
	Type      string                 `json:"type"`
	GatewayID string                 `json:"gateway_id"`
	Timestamp time.Time              `json:"timestamp"`
	Data      map[string]interface{} `json:"data"`
}

// DeviceEvent represents device lifecycle events
type DeviceEvent struct {
	Type      string                 `json:"type"`
	DeviceID  string                 `json:"device_id"`
	GatewayID string                 `json:"gateway_id"`
	Timestamp time.Time              `json:"timestamp"`
	Data      map[string]interface{} `json:"data"`
}

// NewGatewayOrchestrator creates a new IoT gateway orchestrator
func NewGatewayOrchestrator(config *OrchestratorConfig) *GatewayOrchestrator {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &GatewayOrchestrator{
		gateways:      make(map[string]*IoTGateway),
		deviceManager: NewDeviceManager(),
		analytics:     NewEdgeAnalytics(),
		logger:        logger.GlobalLogger,
		ctx:           ctx,
		cancel:        cancel,
		gatewayEvents: make(chan GatewayEvent, 1000),
		deviceEvents:  make(chan DeviceEvent, 1000),
		config:        config,
	}
}

// RegisterGateway registers a new IoT gateway
func (g *GatewayOrchestrator) RegisterGateway(gateway *IoTGateway) error {
	g.mu.Lock()
	defer g.mu.Unlock()
	
	if gateway.ID == "" {
		gateway.ID = uuid.New().String()
	}
	
	gateway.CreatedAt = time.Now()
	gateway.UpdatedAt = time.Now()
	gateway.LastSeen = time.Now()
	gateway.Status = GatewayStatusOnline
	
	g.gateways[gateway.ID] = gateway
	
	g.logger.Info("IoT gateway registered",
		"gateway_id", gateway.ID,
		"gateway_name", gateway.Name,
		"location", gateway.Location.Region,
	)
	
	// Send registration event
	g.sendGatewayEvent("gateway.registered", gateway.ID, map[string]interface{}{
		"name":     gateway.Name,
		"location": gateway.Location,
	})
	
	return nil
}

// UnregisterGateway removes a gateway from management
func (g *GatewayOrchestrator) UnregisterGateway(gatewayID string) error {
	g.mu.Lock()
	defer g.mu.Unlock()
	
	gateway, exists := g.gateways[gatewayID]
	if !exists {
		return fmt.Errorf("gateway not found: %s", gatewayID)
	}
	
	// Disconnect all devices from this gateway
	if err := g.deviceManager.DisconnectGatewayDevices(gatewayID); err != nil {
		g.logger.Error("Error disconnecting devices from gateway", "error", err)
	}
	
	delete(g.gateways, gatewayID)
	
	g.logger.Info("IoT gateway unregistered", "gateway_id", gatewayID)
	
	// Send unregistration event
	g.sendGatewayEvent("gateway.unregistered", gatewayID, map[string]interface{}{
		"name": gateway.Name,
	})
	
	return nil
}

// UpdateGatewayStatus updates gateway status and metrics
func (g *GatewayOrchestrator) UpdateGatewayStatus(gatewayID string, resources *GatewayResources) error {
	g.mu.Lock()
	defer g.mu.Unlock()
	
	gateway, exists := g.gateways[gatewayID]
	if !exists {
		return fmt.Errorf("gateway not found: %s", gatewayID)
	}
	
	gateway.Resources = *resources
	gateway.UpdatedAt = time.Now()
	gateway.LastSeen = time.Now()
	
	// Update status based on resources and heartbeat
	if gateway.Status == GatewayStatusOffline {
		gateway.Status = GatewayStatusOnline
		g.sendGatewayEvent("gateway.online", gatewayID, nil)
	}
	
	// Update metrics
	if gateway.Metrics == nil {
		gateway.Metrics = &GatewayMetrics{
			LastUpdate: time.Now(),
		}
	}
	
	gateway.Metrics.LastUpdate = time.Now()
	
	return nil
}

// GetGateway retrieves a gateway by ID
func (g *GatewayOrchestrator) GetGateway(gatewayID string) (*IoTGateway, error) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	
	gateway, exists := g.gateways[gatewayID]
	if !exists {
		return nil, fmt.Errorf("gateway not found: %s", gatewayID)
	}
	
	return gateway, nil
}

// ListGateways returns all registered gateways
func (g *GatewayOrchestrator) ListGateways() []*IoTGateway {
	g.mu.RLock()
	defer g.mu.RUnlock()
	
	gateways := make([]*IoTGateway, 0, len(g.gateways))
	for _, gateway := range g.gateways {
		gateways = append(gateways, gateway)
	}
	
	return gateways
}

// GetGatewaysByRegion returns gateways in a specific region
func (g *GatewayOrchestrator) GetGatewaysByRegion(region string) []*IoTGateway {
	g.mu.RLock()
	defer g.mu.RUnlock()
	
	var gateways []*IoTGateway
	for _, gateway := range g.gateways {
		if gateway.Location.Region == region {
			gateways = append(gateways, gateway)
		}
	}
	
	return gateways
}

// SelectOptimalGateway selects the best gateway for a device based on multiple factors
func (g *GatewayOrchestrator) SelectOptimalGateway(deviceRequirements *DeviceRequirements) (*IoTGateway, error) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	
	var bestGateway *IoTGateway
	var bestScore float64
	
	for _, gateway := range g.gateways {
		if gateway.Status != GatewayStatusOnline {
			continue
		}
		
		score := g.calculateGatewayScore(gateway, deviceRequirements)
		if score > bestScore {
			bestScore = score
			bestGateway = gateway
		}
	}
	
	if bestGateway == nil {
		return nil, fmt.Errorf("no suitable gateway found")
	}
	
	return bestGateway, nil
}

// calculateGatewayScore calculates a score for gateway selection
func (g *GatewayOrchestrator) calculateGatewayScore(gateway *IoTGateway, req *DeviceRequirements) float64 {
	score := 0.0
	
	// Capacity score (40% weight)
	if gateway.Resources.DeviceCount < gateway.Capabilities.MaxDevices {
		capacityRatio := float64(gateway.Resources.DeviceCount) / float64(gateway.Capabilities.MaxDevices)
		score += (1.0 - capacityRatio) * 0.4
	}
	
	// Resource utilization score (30% weight)
	resourceScore := (100.0 - gateway.Resources.CPUUsage) / 100.0 * 0.15
	resourceScore += (100.0 - gateway.Resources.MemoryUsage) / 100.0 * 0.15
	score += resourceScore
	
	// Location score (20% weight) - prefer closer gateways
	if req.PreferredRegion != "" && gateway.Location.Region == req.PreferredRegion {
		score += 0.2
	}
	
	// Capability match score (10% weight)
	capabilityScore := 0.0
	if req.RequiresEdgeCompute && gateway.Capabilities.EdgeCompute {
		capabilityScore += 0.05
	}
	if req.RequiresAI && gateway.Capabilities.AIInference {
		capabilityScore += 0.05
	}
	score += capabilityScore
	
	return score
}

// Start starts the gateway orchestrator
func (g *GatewayOrchestrator) Start() error {
	g.logger.Info("Starting IoT Gateway Orchestrator")
	
	// Start monitoring goroutines
	go g.monitorGateways()
	go g.processEvents()
	
	// Start device manager
	if err := g.deviceManager.Start(); err != nil {
		return fmt.Errorf("failed to start device manager: %w", err)
	}
	
	// Start edge analytics
	if g.config.AnalyticsEnabled {
		if err := g.analytics.Start(); err != nil {
			return fmt.Errorf("failed to start edge analytics: %w", err)
		}
	}
	
	g.logger.Info("IoT Gateway Orchestrator started successfully")
	return nil
}

// Stop stops the gateway orchestrator
func (g *GatewayOrchestrator) Stop() error {
	g.logger.Info("Stopping IoT Gateway Orchestrator")
	
	g.cancel()
	
	// Stop components
	if err := g.deviceManager.Stop(); err != nil {
		g.logger.Error("Error stopping device manager", "error", err)
	}
	
	if err := g.analytics.Stop(); err != nil {
		g.logger.Error("Error stopping edge analytics", "error", err)
	}
	
	// Close channels
	close(g.gatewayEvents)
	close(g.deviceEvents)
	
	g.logger.Info("IoT Gateway Orchestrator stopped")
	return nil
}

// monitorGateways monitors gateway health and status
func (g *GatewayOrchestrator) monitorGateways() {
	ticker := time.NewTicker(g.config.HeartbeatInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-g.ctx.Done():
			return
		case <-ticker.C:
			g.checkGatewayHealth()
		}
	}
}

// checkGatewayHealth checks the health of all gateways
func (g *GatewayOrchestrator) checkGatewayHealth() {
	g.mu.Lock()
	defer g.mu.Unlock()
	
	threshold := time.Now().Add(-g.config.OfflineThreshold)
	
	for id, gateway := range g.gateways {
		if gateway.Status == GatewayStatusOnline && gateway.LastSeen.Before(threshold) {
			gateway.Status = GatewayStatusOffline
			gateway.UpdatedAt = time.Now()
			
			g.logger.Warn("Gateway marked offline", "gateway_id", id)
			g.sendGatewayEvent("gateway.offline", id, map[string]interface{}{
				"last_seen": gateway.LastSeen,
			})
		}
	}
}

// processEvents processes gateway and device events
func (g *GatewayOrchestrator) processEvents() {
	for {
		select {
		case <-g.ctx.Done():
			return
		case event := <-g.gatewayEvents:
			g.handleGatewayEvent(event)
		case event := <-g.deviceEvents:
			g.handleDeviceEvent(event)
		}
	}
}

// sendGatewayEvent sends a gateway event
func (g *GatewayOrchestrator) sendGatewayEvent(eventType, gatewayID string, data map[string]interface{}) {
	event := GatewayEvent{
		Type:      eventType,
		GatewayID: gatewayID,
		Timestamp: time.Now(),
		Data:      data,
	}
	
	select {
	case g.gatewayEvents <- event:
	default:
		g.logger.Warn("Gateway event channel full, dropping event", "type", eventType)
	}
}

// handleGatewayEvent handles gateway events
func (g *GatewayOrchestrator) handleGatewayEvent(event GatewayEvent) {
	g.logger.Debug("Handling gateway event",
		"type", event.Type,
		"gateway_id", event.GatewayID,
	)
	
	// Process event based on type
	switch event.Type {
	case "gateway.offline":
		g.handleGatewayOffline(event.GatewayID)
	case "gateway.online":
		g.handleGatewayOnline(event.GatewayID)
	}
}

// handleDeviceEvent handles device events
func (g *GatewayOrchestrator) handleDeviceEvent(event DeviceEvent) {
	g.logger.Debug("Handling device event",
		"type", event.Type,
		"device_id", event.DeviceID,
		"gateway_id", event.GatewayID,
	)
}

// handleGatewayOffline handles gateway going offline
func (g *GatewayOrchestrator) handleGatewayOffline(gatewayID string) {
	// Trigger failover if needed
	if g.config.LoadBalancing {
		if err := g.triggerFailover(gatewayID); err != nil {
			g.logger.Error("Error triggering failover", "gateway_id", gatewayID, "error", err)
		}
	}
}

// handleGatewayOnline handles gateway coming online
func (g *GatewayOrchestrator) handleGatewayOnline(gatewayID string) {
	// Rebalance devices if needed
	if g.config.LoadBalancing {
		if err := g.rebalanceDevices(); err != nil {
			g.logger.Error("Error rebalancing devices", "error", err)
		}
	}
}

// triggerFailover handles failover when a gateway goes offline
func (g *GatewayOrchestrator) triggerFailover(gatewayID string) error {
	devices, err := g.deviceManager.GetDevicesByGateway(gatewayID)
	if err != nil {
		return fmt.Errorf("failed to get devices for gateway %s: %w", gatewayID, err)
	}
	
	for _, device := range devices {
		// Find alternative gateway
		newGateway, err := g.SelectOptimalGateway(device.Requirements)
		if err != nil {
			g.logger.Error("No alternative gateway found for device", 
				"device_id", device.ID,
				"gateway_id", gatewayID,
			)
			continue
		}
		
		// Migrate device
		if err := g.deviceManager.MigrateDevice(device.ID, newGateway.ID); err != nil {
			g.logger.Error("Failed to migrate device",
				"device_id", device.ID,
				"from_gateway", gatewayID,
				"to_gateway", newGateway.ID,
				"error", err,
			)
		} else {
			g.logger.Info("Device migrated successfully",
				"device_id", device.ID,
				"from_gateway", gatewayID,
				"to_gateway", newGateway.ID,
			)
		}
	}
	
	return nil
}

// rebalanceDevices rebalances devices across gateways
func (g *GatewayOrchestrator) rebalanceDevices() error {
	// TODO: Implement intelligent device rebalancing
	g.logger.Debug("Device rebalancing triggered")
	return nil
}

// GetMetrics returns orchestrator metrics
func (g *GatewayOrchestrator) GetMetrics() map[string]interface{} {
	g.mu.RLock()
	defer g.mu.RUnlock()
	
	totalGateways := len(g.gateways)
	onlineGateways := 0
	totalDevices := 0
	
	for _, gateway := range g.gateways {
		if gateway.Status == GatewayStatusOnline {
			onlineGateways++
		}
		totalDevices += gateway.Resources.DeviceCount
	}
	
	return map[string]interface{}{
		"total_gateways":   totalGateways,
		"online_gateways":  onlineGateways,
		"offline_gateways": totalGateways - onlineGateways,
		"total_devices":    totalDevices,
		"device_manager":   g.deviceManager.GetMetrics(),
		"analytics":        g.analytics.GetMetrics(),
	}
}