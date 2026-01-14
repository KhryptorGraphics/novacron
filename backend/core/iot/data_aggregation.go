package iot

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
)

// DataPoint represents a single data point with metadata
type DataPoint struct {
	DeviceID   string                 `json:"device_id"`
	GatewayID  string                 `json:"gateway_id"`
	Timestamp  time.Time              `json:"timestamp"`
	Field      string                 `json:"field"`
	Value      interface{}            `json:"value"`
	DataType   DataType               `json:"data_type"`
	Quality    float64                `json:"quality"`  // Data quality score 0-1
	Tags       map[string]string      `json:"tags"`
	Metadata   map[string]interface{} `json:"metadata"`
}

// AggregationConfig defines configuration for data aggregation
type AggregationConfig struct {
	BufferSize           int           `json:"buffer_size"`
	FlushInterval        time.Duration `json:"flush_interval"`
	MaxRetentionPeriod   time.Duration `json:"max_retention_period"`
	CompressionEnabled   bool          `json:"compression_enabled"`
	CompressionRatio     float64       `json:"compression_ratio"`
	HierarchicalLevels   []string      `json:"hierarchical_levels"`
	PreAggregationRules  []PreAggRule  `json:"pre_aggregation_rules"`
}

// PreAggRule defines pre-aggregation rules for data optimization
type PreAggRule struct {
	DevicePattern string                 `json:"device_pattern"`
	FieldPattern  string                 `json:"field_pattern"`
	Function      AggregationFunction    `json:"function"`
	Window        time.Duration          `json:"window"`
	Priority      int                    `json:"priority"`
	Config        map[string]interface{} `json:"config"`
}

// AggregationLevel represents different hierarchical levels
type AggregationLevel struct {
	Name        string        `json:"name"`
	Window      time.Duration `json:"window"`
	Retention   time.Duration `json:"retention"`
	Functions   []AggregationFunction `json:"functions"`
	Enabled     bool          `json:"enabled"`
}

// DataBuffer manages hierarchical data buffering and forwarding
type DataBuffer struct {
	deviceData    map[string][]*DataPoint // device_id -> data points
	gatewayData   map[string][]*DataPoint // gateway_id -> aggregated data
	edgeData      []*DataPoint            // edge-level aggregated data
	cloudData     []*DataPoint            // cloud-ready aggregated data
	mu            sync.RWMutex
	logger        logger.Logger
	
	// Configuration
	config        *AggregationConfig
	levels        map[string]*AggregationLevel
	
	// Context for shutdown
	ctx           context.Context
	cancel        context.CancelFunc
	
	// Metrics
	totalIngested    int64
	totalAggregated  int64
	totalForwarded   int64
	totalCompressed  int64
	avgLatency       time.Duration
	bufferUtilization float64
}

// NewDataBuffer creates a new hierarchical data buffer
func NewDataBuffer(config *AggregationConfig) *DataBuffer {
	ctx, cancel := context.WithCancel(context.Background())
	
	// Default levels if not configured
	defaultLevels := map[string]*AggregationLevel{
		"device": {
			Name:      "device",
			Window:    1 * time.Minute,
			Retention: 1 * time.Hour,
			Functions: []AggregationFunction{AggregationAvg, AggregationMin, AggregationMax},
			Enabled:   true,
		},
		"gateway": {
			Name:      "gateway",
			Window:    5 * time.Minute,
			Retention: 6 * time.Hour,
			Functions: []AggregationFunction{AggregationAvg, AggregationCount},
			Enabled:   true,
		},
		"edge": {
			Name:      "edge",
			Window:    15 * time.Minute,
			Retention: 24 * time.Hour,
			Functions: []AggregationFunction{AggregationAvg},
			Enabled:   true,
		},
		"cloud": {
			Name:      "cloud",
			Window:    1 * time.Hour,
			Retention: 30 * 24 * time.Hour, // 30 days
			Functions: []AggregationFunction{AggregationAvg, AggregationSum},
			Enabled:   true,
		},
	}
	
	return &DataBuffer{
		deviceData:  make(map[string][]*DataPoint),
		gatewayData: make(map[string][]*DataPoint),
		edgeData:    make([]*DataPoint, 0),
		cloudData:   make([]*DataPoint, 0),
		logger:      logger.GlobalLogger,
		config:      config,
		levels:      defaultLevels,
		ctx:         ctx,
		cancel:      cancel,
	}
}

// Start starts the data buffer with periodic aggregation and cleanup
func (db *DataBuffer) Start() error {
	db.logger.Info("Starting hierarchical data buffer")
	
	// Start aggregation workers
	go db.deviceAggregationWorker()
	go db.gatewayAggregationWorker()
	go db.edgeAggregationWorker()
	go db.cloudAggregationWorker()
	
	// Start cleanup worker
	go db.cleanupWorker()
	
	// Start metrics collector
	go db.metricsWorker()
	
	db.logger.Info("Hierarchical data buffer started")
	return nil
}

// Stop stops the data buffer
func (db *DataBuffer) Stop() error {
	db.logger.Info("Stopping hierarchical data buffer")
	
	db.cancel()
	
	// Final flush
	db.flushAllLevels()
	
	db.logger.Info("Hierarchical data buffer stopped")
	return nil
}

// IngestData ingests raw data points into the buffer
func (db *DataBuffer) IngestData(points []*DataPoint) error {
	db.mu.Lock()
	defer db.mu.Unlock()
	
	for _, point := range points {
		// Validate data point
		if err := db.validateDataPoint(point); err != nil {
			db.logger.Warn("Invalid data point", "device_id", point.DeviceID, "error", err)
			continue
		}
		
		// Add to device-level buffer
		if _, exists := db.deviceData[point.DeviceID]; !exists {
			db.deviceData[point.DeviceID] = make([]*DataPoint, 0)
		}
		
		// Check buffer limits
		if len(db.deviceData[point.DeviceID]) >= db.config.BufferSize {
			// Remove oldest point to make space
			db.deviceData[point.DeviceID] = db.deviceData[point.DeviceID][1:]
		}
		
		db.deviceData[point.DeviceID] = append(db.deviceData[point.DeviceID], point)
		db.totalIngested++
	}
	
	return nil
}

// validateDataPoint validates a data point
func (db *DataBuffer) validateDataPoint(point *DataPoint) error {
	if point.DeviceID == "" {
		return fmt.Errorf("device_id is required")
	}
	if point.GatewayID == "" {
		return fmt.Errorf("gateway_id is required")
	}
	if point.Field == "" {
		return fmt.Errorf("field is required")
	}
	if point.Value == nil {
		return fmt.Errorf("value is required")
	}
	if point.Timestamp.IsZero() {
		point.Timestamp = time.Now()
	}
	return nil
}

// deviceAggregationWorker aggregates device-level data
func (db *DataBuffer) deviceAggregationWorker() {
	level := db.levels["device"]
	if !level.Enabled {
		return
	}
	
	ticker := time.NewTicker(level.Window)
	defer ticker.Stop()
	
	for {
		select {
		case <-db.ctx.Done():
			return
		case <-ticker.C:
			db.aggregateDeviceLevel()
		}
	}
}

// aggregateDeviceLevel performs device-level aggregation
func (db *DataBuffer) aggregateDeviceLevel() {
	db.mu.Lock()
	defer db.mu.Unlock()
	
	level := db.levels["device"]
	now := time.Now()
	windowStart := now.Add(-level.Window)
	
	// Process each device
	for deviceID, points := range db.deviceData {
		if len(points) == 0 {
			continue
		}
		
		// Get points within the window
		windowPoints := make([]*DataPoint, 0)
		for _, point := range points {
			if point.Timestamp.After(windowStart) {
				windowPoints = append(windowPoints, point)
			}
		}
		
		if len(windowPoints) == 0 {
			continue
		}
		
		// Group by field and aggregate
		fieldGroups := db.groupByField(windowPoints)
		
		for field, fieldPoints := range fieldGroups {
			for _, function := range level.Functions {
				aggregated := db.performAggregation(fieldPoints, function)
				if aggregated != nil {
					// Create aggregated data point
					aggPoint := &DataPoint{
						DeviceID:  deviceID,
						GatewayID: fieldPoints[0].GatewayID,
						Timestamp: now,
						Field:     fmt.Sprintf("%s_%s", field, function),
						Value:     aggregated.Value,
						DataType:  DataTypeNumeric,
						Quality:   db.calculateQuality(fieldPoints),
						Tags: map[string]string{
							"aggregation_level": "device",
							"function":         string(function),
							"original_field":   field,
						},
						Metadata: map[string]interface{}{
							"window_start":  windowStart,
							"window_end":    now,
							"point_count":   len(fieldPoints),
							"original_field": field,
						},
					}
					
					// Add to gateway-level buffer
					gatewayID := fieldPoints[0].GatewayID
					if _, exists := db.gatewayData[gatewayID]; !exists {
						db.gatewayData[gatewayID] = make([]*DataPoint, 0)
					}
					db.gatewayData[gatewayID] = append(db.gatewayData[gatewayID], aggPoint)
					
					db.totalAggregated++
				}
			}
		}
		
		// Keep only recent data points
		retentionCutoff := now.Add(-level.Retention)
		filteredPoints := make([]*DataPoint, 0)
		for _, point := range points {
			if point.Timestamp.After(retentionCutoff) {
				filteredPoints = append(filteredPoints, point)
			}
		}
		db.deviceData[deviceID] = filteredPoints
	}
}

// gatewayAggregationWorker aggregates gateway-level data
func (db *DataBuffer) gatewayAggregationWorker() {
	level := db.levels["gateway"]
	if !level.Enabled {
		return
	}
	
	ticker := time.NewTicker(level.Window)
	defer ticker.Stop()
	
	for {
		select {
		case <-db.ctx.Done():
			return
		case <-ticker.C:
			db.aggregateGatewayLevel()
		}
	}
}

// aggregateGatewayLevel performs gateway-level aggregation
func (db *DataBuffer) aggregateGatewayLevel() {
	db.mu.Lock()
	defer db.mu.Unlock()
	
	level := db.levels["gateway"]
	now := time.Now()
	windowStart := now.Add(-level.Window)
	
	// Process each gateway
	for gatewayID, points := range db.gatewayData {
		if len(points) == 0 {
			continue
		}
		
		// Get points within the window
		windowPoints := make([]*DataPoint, 0)
		for _, point := range points {
			if point.Timestamp.After(windowStart) {
				windowPoints = append(windowPoints, point)
			}
		}
		
		if len(windowPoints) == 0 {
			continue
		}
		
		// Group by original field and aggregate
		fieldGroups := db.groupByOriginalField(windowPoints)
		
		for field, fieldPoints := range fieldGroups {
			for _, function := range level.Functions {
				aggregated := db.performAggregation(fieldPoints, function)
				if aggregated != nil {
					// Create edge-level data point
					edgePoint := &DataPoint{
						DeviceID:  "gateway_" + gatewayID,
						GatewayID: gatewayID,
						Timestamp: now,
						Field:     fmt.Sprintf("%s_%s", field, function),
						Value:     aggregated.Value,
						DataType:  DataTypeNumeric,
						Quality:   db.calculateQuality(fieldPoints),
						Tags: map[string]string{
							"aggregation_level": "gateway",
							"function":         string(function),
							"original_field":   field,
						},
						Metadata: map[string]interface{}{
							"window_start":      windowStart,
							"window_end":        now,
							"point_count":       len(fieldPoints),
							"original_field":    field,
							"source_devices":    db.getUniqueDevices(fieldPoints),
						},
					}
					
					db.edgeData = append(db.edgeData, edgePoint)
					db.totalAggregated++
				}
			}
		}
		
		// Clean up old data
		retentionCutoff := now.Add(-level.Retention)
		filteredPoints := make([]*DataPoint, 0)
		for _, point := range points {
			if point.Timestamp.After(retentionCutoff) {
				filteredPoints = append(filteredPoints, point)
			}
		}
		db.gatewayData[gatewayID] = filteredPoints
	}
}

// edgeAggregationWorker aggregates edge-level data
func (db *DataBuffer) edgeAggregationWorker() {
	level := db.levels["edge"]
	if !level.Enabled {
		return
	}
	
	ticker := time.NewTicker(level.Window)
	defer ticker.Stop()
	
	for {
		select {
		case <-db.ctx.Done():
			return
		case <-ticker.C:
			db.aggregateEdgeLevel()
		}
	}
}

// aggregateEdgeLevel performs edge-level aggregation
func (db *DataBuffer) aggregateEdgeLevel() {
	db.mu.Lock()
	defer db.mu.Unlock()
	
	level := db.levels["edge"]
	now := time.Now()
	windowStart := now.Add(-level.Window)
	
	// Get points within the window
	windowPoints := make([]*DataPoint, 0)
	for _, point := range db.edgeData {
		if point.Timestamp.After(windowStart) {
			windowPoints = append(windowPoints, point)
		}
	}
	
	if len(windowPoints) == 0 {
		return
	}
	
	// Group by original field and aggregate
	fieldGroups := db.groupByOriginalField(windowPoints)
	
	for field, fieldPoints := range fieldGroups {
		for _, function := range level.Functions {
			aggregated := db.performAggregation(fieldPoints, function)
			if aggregated != nil {
				// Create cloud-level data point
				cloudPoint := &DataPoint{
					DeviceID:  "edge",
					GatewayID: "edge",
					Timestamp: now,
					Field:     fmt.Sprintf("%s_%s", field, function),
					Value:     aggregated.Value,
					DataType:  DataTypeNumeric,
					Quality:   db.calculateQuality(fieldPoints),
					Tags: map[string]string{
						"aggregation_level": "edge",
						"function":         string(function),
						"original_field":   field,
					},
					Metadata: map[string]interface{}{
						"window_start":      windowStart,
						"window_end":        now,
						"point_count":       len(fieldPoints),
						"original_field":    field,
						"source_gateways":   db.getUniqueGateways(fieldPoints),
					},
				}
				
				db.cloudData = append(db.cloudData, cloudPoint)
				db.totalAggregated++
			}
		}
	}
	
	// Clean up old edge data
	retentionCutoff := now.Add(-level.Retention)
	filteredPoints := make([]*DataPoint, 0)
	for _, point := range db.edgeData {
		if point.Timestamp.After(retentionCutoff) {
			filteredPoints = append(filteredPoints, point)
		}
	}
	db.edgeData = filteredPoints
}

// cloudAggregationWorker prepares data for cloud forwarding
func (db *DataBuffer) cloudAggregationWorker() {
	level := db.levels["cloud"]
	if !level.Enabled {
		return
	}
	
	ticker := time.NewTicker(level.Window)
	defer ticker.Stop()
	
	for {
		select {
		case <-db.ctx.Done():
			return
		case <-ticker.C:
			db.prepareCloudData()
		}
	}
}

// prepareCloudData prepares aggregated data for cloud forwarding
func (db *DataBuffer) prepareCloudData() {
	db.mu.Lock()
	defer db.mu.Unlock()
	
	level := db.levels["cloud"]
	now := time.Now()
	windowStart := now.Add(-level.Window)
	
	// Get points within the window
	windowPoints := make([]*DataPoint, 0)
	for _, point := range db.cloudData {
		if point.Timestamp.After(windowStart) {
			windowPoints = append(windowPoints, point)
		}
	}
	
	if len(windowPoints) == 0 {
		return
	}
	
	// Apply compression if enabled
	if db.config.CompressionEnabled {
		windowPoints = db.compressData(windowPoints)
	}
	
	// Forward to cloud (mock implementation)
	db.forwardToCloud(windowPoints)
	
	// Clean up old cloud data
	retentionCutoff := now.Add(-level.Retention)
	filteredPoints := make([]*DataPoint, 0)
	for _, point := range db.cloudData {
		if point.Timestamp.After(retentionCutoff) {
			filteredPoints = append(filteredPoints, point)
		}
	}
	db.cloudData = filteredPoints
}

// Helper functions

func (db *DataBuffer) groupByField(points []*DataPoint) map[string][]*DataPoint {
	groups := make(map[string][]*DataPoint)
	for _, point := range points {
		field := point.Field
		if _, exists := groups[field]; !exists {
			groups[field] = make([]*DataPoint, 0)
		}
		groups[field] = append(groups[field], point)
	}
	return groups
}

func (db *DataBuffer) groupByOriginalField(points []*DataPoint) map[string][]*DataPoint {
	groups := make(map[string][]*DataPoint)
	for _, point := range points {
		originalField := "unknown"
		if field, exists := point.Metadata["original_field"].(string); exists {
			originalField = field
		}
		
		if _, exists := groups[originalField]; !exists {
			groups[originalField] = make([]*DataPoint, 0)
		}
		groups[originalField] = append(groups[originalField], point)
	}
	return groups
}

func (db *DataBuffer) performAggregation(points []*DataPoint, function AggregationFunction) *AggregatedData {
	if len(points) == 0 {
		return nil
	}
	
	values := make([]float64, 0, len(points))
	for _, point := range points {
		if val := db.extractNumericValue(point.Value); val != nil {
			values = append(values, *val)
		}
	}
	
	if len(values) == 0 {
		return nil
	}
	
	var result float64
	switch function {
	case AggregationSum:
		for _, v := range values {
			result += v
		}
	case AggregationAvg:
		for _, v := range values {
			result += v
		}
		result /= float64(len(values))
	case AggregationMin:
		result = values[0]
		for _, v := range values {
			if v < result {
				result = v
			}
		}
	case AggregationMax:
		result = values[0]
		for _, v := range values {
			if v > result {
				result = v
			}
		}
	case AggregationCount:
		result = float64(len(values))
	case AggregationMedian:
		sort.Float64s(values)
		n := len(values)
		if n%2 == 0 {
			result = (values[n/2-1] + values[n/2]) / 2
		} else {
			result = values[n/2]
		}
	}
	
	return &AggregatedData{
		Value: result,
		Count: len(values),
	}
}

func (db *DataBuffer) extractNumericValue(value interface{}) *float64 {
	switch v := value.(type) {
	case float64:
		return &v
	case float32:
		f := float64(v)
		return &f
	case int:
		f := float64(v)
		return &f
	case int64:
		f := float64(v)
		return &f
	default:
		return nil
	}
}

func (db *DataBuffer) calculateQuality(points []*DataPoint) float64 {
	if len(points) == 0 {
		return 0.0
	}
	
	totalQuality := 0.0
	for _, point := range points {
		totalQuality += point.Quality
	}
	
	return totalQuality / float64(len(points))
}

func (db *DataBuffer) getUniqueDevices(points []*DataPoint) []string {
	deviceSet := make(map[string]bool)
	for _, point := range points {
		deviceSet[point.DeviceID] = true
	}
	
	devices := make([]string, 0, len(deviceSet))
	for device := range deviceSet {
		devices = append(devices, device)
	}
	
	return devices
}

func (db *DataBuffer) getUniqueGateways(points []*DataPoint) []string {
	gatewaySet := make(map[string]bool)
	for _, point := range points {
		gatewaySet[point.GatewayID] = true
	}
	
	gateways := make([]string, 0, len(gatewaySet))
	for gateway := range gatewaySet {
		gateways = append(gateways, gateway)
	}
	
	return gateways
}

func (db *DataBuffer) compressData(points []*DataPoint) []*DataPoint {
	if len(points) == 0 {
		return points
	}
	
	targetCount := int(float64(len(points)) * db.config.CompressionRatio)
	if targetCount >= len(points) {
		return points
	}
	
	// Simple compression - keep every nth point
	interval := len(points) / targetCount
	if interval <= 1 {
		return points
	}
	
	compressed := make([]*DataPoint, 0, targetCount)
	for i := 0; i < len(points); i += interval {
		compressed = append(compressed, points[i])
	}
	
	db.totalCompressed += int64(len(points) - len(compressed))
	
	return compressed
}

func (db *DataBuffer) forwardToCloud(points []*DataPoint) {
	// Mock cloud forwarding
	db.logger.Debug("Forwarding data to cloud", "points", len(points))
	db.totalForwarded += int64(len(points))
}

func (db *DataBuffer) flushAllLevels() {
	db.mu.Lock()
	defer db.mu.Unlock()
	
	// Perform final aggregations and cleanup
	db.logger.Info("Flushing all aggregation levels")
}

func (db *DataBuffer) cleanupWorker() {
	ticker := time.NewTicker(10 * time.Minute)
	defer ticker.Stop()
	
	for {
		select {
		case <-db.ctx.Done():
			return
		case <-ticker.C:
			db.cleanup()
		}
	}
}

func (db *DataBuffer) cleanup() {
	db.mu.Lock()
	defer db.mu.Unlock()
	
	now := time.Now()
	
	// Clean up device data beyond max retention
	maxRetention := db.config.MaxRetentionPeriod
	cutoff := now.Add(-maxRetention)
	
	for deviceID, points := range db.deviceData {
		filteredPoints := make([]*DataPoint, 0)
		for _, point := range points {
			if point.Timestamp.After(cutoff) {
				filteredPoints = append(filteredPoints, point)
			}
		}
		
		if len(filteredPoints) == 0 {
			delete(db.deviceData, deviceID)
		} else {
			db.deviceData[deviceID] = filteredPoints
		}
	}
}

func (db *DataBuffer) metricsWorker() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()
	
	for {
		select {
		case <-db.ctx.Done():
			return
		case <-ticker.C:
			db.updateMetrics()
		}
	}
}

func (db *DataBuffer) updateMetrics() {
	db.mu.RLock()
	defer db.mu.RUnlock()
	
	// Calculate buffer utilization
	totalPoints := 0
	for _, points := range db.deviceData {
		totalPoints += len(points)
	}
	for _, points := range db.gatewayData {
		totalPoints += len(points)
	}
	totalPoints += len(db.edgeData) + len(db.cloudData)
	
	maxCapacity := db.config.BufferSize * 100 // Rough estimate
	db.bufferUtilization = float64(totalPoints) / float64(maxCapacity)
}

// GetMetrics returns aggregation metrics
func (db *DataBuffer) GetMetrics() map[string]interface{} {
	db.mu.RLock()
	defer db.mu.RUnlock()
	
	deviceBuffers := 0
	totalDevicePoints := 0
	for _, points := range db.deviceData {
		deviceBuffers++
		totalDevicePoints += len(points)
	}
	
	gatewayBuffers := 0
	totalGatewayPoints := 0
	for _, points := range db.gatewayData {
		gatewayBuffers++
		totalGatewayPoints += len(points)
	}
	
	return map[string]interface{}{
		"total_ingested":        db.totalIngested,
		"total_aggregated":      db.totalAggregated,
		"total_forwarded":       db.totalForwarded,
		"total_compressed":      db.totalCompressed,
		"buffer_utilization":    db.bufferUtilization,
		"device_buffers":        deviceBuffers,
		"gateway_buffers":       gatewayBuffers,
		"device_points":         totalDevicePoints,
		"gateway_points":        totalGatewayPoints,
		"edge_points":           len(db.edgeData),
		"cloud_points":          len(db.cloudData),
		"compression_enabled":   db.config.CompressionEnabled,
		"compression_ratio":     db.config.CompressionRatio,
	}
}

// GetLevelData returns data for a specific aggregation level
func (db *DataBuffer) GetLevelData(level string) interface{} {
	db.mu.RLock()
	defer db.mu.RUnlock()
	
	switch level {
	case "device":
		return db.deviceData
	case "gateway":
		return db.gatewayData
	case "edge":
		return db.edgeData
	case "cloud":
		return db.cloudData
	default:
		return nil
	}
}