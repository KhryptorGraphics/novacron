package iot

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
)

// AnalyticsEventType represents different types of analytics events
type AnalyticsEventType string

const (
	EventTypeDeviceData    AnalyticsEventType = "device_data"
	EventTypeAlert         AnalyticsEventType = "alert"
	EventTypeAnomaly       AnalyticsEventType = "anomaly"
	EventTypeAggregation   AnalyticsEventType = "aggregation"
	EventTypeInference     AnalyticsEventType = "inference"
	EventTypeCommand       AnalyticsEventType = "command"
)

// DataType represents different data types for analytics
type DataType string

const (
	DataTypeNumeric     DataType = "numeric"
	DataTypeBoolean     DataType = "boolean"
	DataTypeString      DataType = "string"
	DataTypeJSON        DataType = "json"
	DataTypeBinary      DataType = "binary"
	DataTypeTimeSeries  DataType = "timeseries"
	DataTypeGeolocation DataType = "geolocation"
)

// AnalyticsEvent represents a data event for processing
type AnalyticsEvent struct {
	ID          string                 `json:"id"`
	Type        AnalyticsEventType     `json:"type"`
	DeviceID    string                 `json:"device_id"`
	GatewayID   string                 `json:"gateway_id"`
	Timestamp   time.Time              `json:"timestamp"`
	DataType    DataType               `json:"data_type"`
	Value       interface{}            `json:"value"`
	Metadata    map[string]interface{} `json:"metadata"`
	Tags        []string               `json:"tags"`
	ProcessedAt time.Time              `json:"processed_at"`
	ProcessingLatency time.Duration    `json:"processing_latency"`
}

// AggregationRule defines rules for data aggregation
type AggregationRule struct {
	ID           string                 `json:"id"`
	Name         string                 `json:"name"`
	DeviceFilter string                 `json:"device_filter"`
	DataFilter   string                 `json:"data_filter"`
	Function     AggregationFunction    `json:"function"`
	Window       time.Duration          `json:"window"`
	Interval     time.Duration          `json:"interval"`
	GroupBy      []string               `json:"group_by"`
	Enabled      bool                   `json:"enabled"`
	Config       map[string]interface{} `json:"config"`
}

// AggregationFunction represents aggregation functions
type AggregationFunction string

const (
	AggregationSum       AggregationFunction = "sum"
	AggregationAvg       AggregationFunction = "avg"
	AggregationMin       AggregationFunction = "min"
	AggregationMax       AggregationFunction = "max"
	AggregationCount     AggregationFunction = "count"
	AggregationMedian    AggregationFunction = "median"
	AggregationStdDev    AggregationFunction = "stddev"
	AggregationPercentile AggregationFunction = "percentile"
)

// AggregatedData represents aggregated analytics data
type AggregatedData struct {
	ID        string                 `json:"id"`
	RuleID    string                 `json:"rule_id"`
	Window    TimeWindow             `json:"window"`
	GroupKey  string                 `json:"group_key"`
	Function  AggregationFunction    `json:"function"`
	Value     float64                `json:"value"`
	Count     int                    `json:"count"`
	Metadata  map[string]interface{} `json:"metadata"`
	CreatedAt time.Time              `json:"created_at"`
}

// TimeWindow represents a time window for aggregation
type TimeWindow struct {
	Start time.Time `json:"start"`
	End   time.Time `json:"end"`
}

// AlertRule defines conditions for generating alerts
type AlertRule struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	DeviceFilter string                `json:"device_filter"`
	Condition   AlertCondition         `json:"condition"`
	Severity    AlertSeverity          `json:"severity"`
	Enabled     bool                   `json:"enabled"`
	Cooldown    time.Duration          `json:"cooldown"`
	Actions     []AlertAction          `json:"actions"`
	Config      map[string]interface{} `json:"config"`
}

// AlertCondition defines alert triggering conditions
type AlertCondition struct {
	Field    string      `json:"field"`
	Operator string      `json:"operator"` // >, <, >=, <=, ==, !=, contains
	Value    interface{} `json:"value"`
	Duration time.Duration `json:"duration"` // sustained condition duration
}

// AlertSeverity represents alert severity levels
type AlertSeverity string

const (
	SeverityInfo     AlertSeverity = "info"
	SeverityWarning  AlertSeverity = "warning"
	SeverityCritical AlertSeverity = "critical"
	SeverityError    AlertSeverity = "error"
)

// AlertAction defines actions to take when alert triggers
type AlertAction struct {
	Type   string                 `json:"type"`
	Config map[string]interface{} `json:"config"`
}

// Alert represents a triggered alert
type Alert struct {
	ID          string                 `json:"id"`
	RuleID      string                 `json:"rule_id"`
	DeviceID    string                 `json:"device_id"`
	GatewayID   string                 `json:"gateway_id"`
	Severity    AlertSeverity          `json:"severity"`
	Message     string                 `json:"message"`
	Value       interface{}            `json:"value"`
	Metadata    map[string]interface{} `json:"metadata"`
	CreatedAt   time.Time              `json:"created_at"`
	ResolvedAt  *time.Time             `json:"resolved_at,omitempty"`
	Status      string                 `json:"status"` // active, resolved, acknowledged
}

// EdgeInferenceModel represents an AI model for edge inference
type EdgeInferenceModel struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Version     string                 `json:"version"`
	Type        string                 `json:"type"` // classification, regression, anomaly_detection
	ModelData   []byte                 `json:"model_data,omitempty"`
	ModelPath   string                 `json:"model_path"`
	InputSchema map[string]interface{} `json:"input_schema"`
	OutputSchema map[string]interface{} `json:"output_schema"`
	Metadata    map[string]interface{} `json:"metadata"`
	LoadedAt    time.Time              `json:"loaded_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

// InferenceResult represents the result of edge AI inference
type InferenceResult struct {
	ID         string                 `json:"id"`
	ModelID    string                 `json:"model_id"`
	DeviceID   string                 `json:"device_id"`
	GatewayID  string                 `json:"gateway_id"`
	Input      map[string]interface{} `json:"input"`
	Output     map[string]interface{} `json:"output"`
	Confidence float64                `json:"confidence"`
	Latency    time.Duration          `json:"latency"`
	Timestamp  time.Time              `json:"timestamp"`
	Metadata   map[string]interface{} `json:"metadata"`
}

// StreamProcessor handles real-time data stream processing
type StreamProcessor struct {
	inputBuffer  chan *AnalyticsEvent
	outputBuffer chan *AnalyticsEvent
	rules        map[string]*AggregationRule
	alertRules   map[string]*AlertRule
	models       map[string]*EdgeInferenceModel
	mu           sync.RWMutex
	logger       logger.Logger
	ctx          context.Context
	cancel       context.CancelFunc
	
	// Processing metrics
	eventsProcessed int64
	processingLatency time.Duration
	errorCount      int64
	
	// Data store for windowed aggregations
	dataWindows     map[string][]*AnalyticsEvent
	windowsMu       sync.RWMutex
	
	// Alert state tracking
	alertState      map[string]time.Time
	alertStateMu    sync.RWMutex
}

// NewStreamProcessor creates a new stream processor
func NewStreamProcessor(bufferSize int) *StreamProcessor {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &StreamProcessor{
		inputBuffer:   make(chan *AnalyticsEvent, bufferSize),
		outputBuffer:  make(chan *AnalyticsEvent, bufferSize),
		rules:         make(map[string]*AggregationRule),
		alertRules:    make(map[string]*AlertRule),
		models:        make(map[string]*EdgeInferenceModel),
		logger:        logger.GlobalLogger,
		ctx:           ctx,
		cancel:        cancel,
		dataWindows:   make(map[string][]*AnalyticsEvent),
		alertState:    make(map[string]time.Time),
	}
}

// Start starts the stream processor
func (sp *StreamProcessor) Start() error {
	sp.logger.Info("Starting stream processor")
	
	// Start processing goroutines
	numWorkers := 4
	for i := 0; i < numWorkers; i++ {
		go sp.processEvents()
	}
	
	// Start cleanup routines
	go sp.cleanupWindows()
	go sp.cleanupAlertState()
	
	sp.logger.Info("Stream processor started", "workers", numWorkers)
	return nil
}

// Stop stops the stream processor
func (sp *StreamProcessor) Stop() error {
	sp.logger.Info("Stopping stream processor")
	
	sp.cancel()
	close(sp.inputBuffer)
	close(sp.outputBuffer)
	
	sp.logger.Info("Stream processor stopped")
	return nil
}

// ProcessEvent processes an incoming analytics event
func (sp *StreamProcessor) ProcessEvent(event *AnalyticsEvent) error {
	select {
	case sp.inputBuffer <- event:
		return nil
	default:
		return fmt.Errorf("input buffer full")
	}
}

// GetProcessedEvents returns processed events from output buffer
func (sp *StreamProcessor) GetProcessedEvents() <-chan *AnalyticsEvent {
	return sp.outputBuffer
}

// AddAggregationRule adds a new aggregation rule
func (sp *StreamProcessor) AddAggregationRule(rule *AggregationRule) error {
	sp.mu.Lock()
	defer sp.mu.Unlock()
	
	sp.rules[rule.ID] = rule
	sp.logger.Info("Added aggregation rule", "rule_id", rule.ID, "name", rule.Name)
	return nil
}

// AddAlertRule adds a new alert rule
func (sp *StreamProcessor) AddAlertRule(rule *AlertRule) error {
	sp.mu.Lock()
	defer sp.mu.Unlock()
	
	sp.alertRules[rule.ID] = rule
	sp.logger.Info("Added alert rule", "rule_id", rule.ID, "name", rule.Name)
	return nil
}

// LoadModel loads an AI model for edge inference
func (sp *StreamProcessor) LoadModel(model *EdgeInferenceModel) error {
	sp.mu.Lock()
	defer sp.mu.Unlock()
	
	model.LoadedAt = time.Now()
	sp.models[model.ID] = model
	sp.logger.Info("Loaded inference model", "model_id", model.ID, "name", model.Name)
	return nil
}

// processEvents processes events from the input buffer
func (sp *StreamProcessor) processEvents() {
	for {
		select {
		case <-sp.ctx.Done():
			return
		case event, ok := <-sp.inputBuffer:
			if !ok {
				return
			}
			
			startTime := time.Now()
			sp.processEvent(event)
			sp.processingLatency = time.Since(startTime)
			sp.eventsProcessed++
		}
	}
}

// processEvent processes a single event
func (sp *StreamProcessor) processEvent(event *AnalyticsEvent) {
	event.ProcessedAt = time.Now()
	
	// Apply aggregation rules
	sp.applyAggregationRules(event)
	
	// Check alert conditions
	sp.checkAlertRules(event)
	
	// Perform AI inference if applicable
	sp.performInference(event)
	
	// Send to output buffer
	select {
	case sp.outputBuffer <- event:
	default:
		sp.logger.Warn("Output buffer full, dropping event")
	}
}

// applyAggregationRules applies aggregation rules to the event
func (sp *StreamProcessor) applyAggregationRules(event *AnalyticsEvent) {
	sp.mu.RLock()
	rules := make([]*AggregationRule, 0, len(sp.rules))
	for _, rule := range sp.rules {
		if rule.Enabled {
			rules = append(rules, rule)
		}
	}
	sp.mu.RUnlock()
	
	for _, rule := range rules {
		if sp.matchesFilter(event, rule.DeviceFilter) {
			sp.aggregateData(event, rule)
		}
	}
}

// aggregateData performs data aggregation based on rule
func (sp *StreamProcessor) aggregateData(event *AnalyticsEvent, rule *AggregationRule) {
	sp.windowsMu.Lock()
	defer sp.windowsMu.Unlock()
	
	windowKey := fmt.Sprintf("%s:%s", rule.ID, event.DeviceID)
	
	// Add event to window
	if _, exists := sp.dataWindows[windowKey]; !exists {
		sp.dataWindows[windowKey] = make([]*AnalyticsEvent, 0)
	}
	sp.dataWindows[windowKey] = append(sp.dataWindows[windowKey], event)
	
	// Check if window is full
	windowStart := event.Timestamp.Add(-rule.Window)
	events := sp.dataWindows[windowKey]
	
	// Filter events within window
	var windowEvents []*AnalyticsEvent
	for _, e := range events {
		if e.Timestamp.After(windowStart) {
			windowEvents = append(windowEvents, e)
		}
	}
	sp.dataWindows[windowKey] = windowEvents
	
	// Perform aggregation
	if len(windowEvents) > 0 {
		aggregated := sp.performAggregation(windowEvents, rule)
		if aggregated != nil {
			sp.logger.Debug("Data aggregated",
				"rule_id", rule.ID,
				"device_id", event.DeviceID,
				"function", rule.Function,
				"value", aggregated.Value,
			)
		}
	}
}

// performAggregation performs the actual aggregation calculation
func (sp *StreamProcessor) performAggregation(events []*AnalyticsEvent, rule *AggregationRule) *AggregatedData {
	if len(events) == 0 {
		return nil
	}
	
	values := make([]float64, 0, len(events))
	for _, event := range events {
		if val, ok := sp.extractNumericValue(event.Value); ok {
			values = append(values, val)
		}
	}
	
	if len(values) == 0 {
		return nil
	}
	
	var result float64
	switch rule.Function {
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
	case AggregationStdDev:
		// Calculate mean first
		mean := 0.0
		for _, v := range values {
			mean += v
		}
		mean /= float64(len(values))
		
		// Calculate variance
		variance := 0.0
		for _, v := range values {
			diff := v - mean
			variance += diff * diff
		}
		variance /= float64(len(values))
		result = math.Sqrt(variance)
	}
	
	return &AggregatedData{
		ID:        fmt.Sprintf("agg_%s_%d", rule.ID, time.Now().UnixNano()),
		RuleID:    rule.ID,
		Function:  rule.Function,
		Value:     result,
		Count:     len(values),
		CreatedAt: time.Now(),
		Window: TimeWindow{
			Start: events[0].Timestamp,
			End:   events[len(events)-1].Timestamp,
		},
	}
}

// checkAlertRules checks if event triggers any alerts
func (sp *StreamProcessor) checkAlertRules(event *AnalyticsEvent) {
	sp.mu.RLock()
	alertRules := make([]*AlertRule, 0, len(sp.alertRules))
	for _, rule := range sp.alertRules {
		if rule.Enabled {
			alertRules = append(alertRules, rule)
		}
	}
	sp.mu.RUnlock()
	
	for _, rule := range alertRules {
		if sp.matchesFilter(event, rule.DeviceFilter) && sp.evaluateCondition(event, rule.Condition) {
			// Check cooldown
			sp.alertStateMu.RLock()
			lastAlert, exists := sp.alertState[rule.ID]
			sp.alertStateMu.RUnlock()
			
			if !exists || time.Since(lastAlert) > rule.Cooldown {
				alert := sp.createAlert(event, rule)
				sp.triggerAlert(alert)
				
				sp.alertStateMu.Lock()
				sp.alertState[rule.ID] = time.Now()
				sp.alertStateMu.Unlock()
			}
		}
	}
}

// performInference performs AI inference on event data
func (sp *StreamProcessor) performInference(event *AnalyticsEvent) {
	sp.mu.RLock()
	models := make([]*EdgeInferenceModel, 0, len(sp.models))
	for _, model := range sp.models {
		models = append(models, model)
	}
	sp.mu.RUnlock()
	
	for _, model := range models {
		// Check if model is applicable to this event
		if sp.isModelApplicable(event, model) {
			result := sp.runInference(event, model)
			if result != nil {
				sp.logger.Debug("Inference performed",
					"model_id", model.ID,
					"device_id", event.DeviceID,
					"confidence", result.Confidence,
				)
			}
		}
	}
}

// Helper functions

func (sp *StreamProcessor) matchesFilter(event *AnalyticsEvent, filter string) bool {
	// Simple filter matching - in production, this would be more sophisticated
	if filter == "" || filter == "*" {
		return true
	}
	return event.DeviceID == filter
}

func (sp *StreamProcessor) evaluateCondition(event *AnalyticsEvent, condition AlertCondition) bool {
	value := sp.extractNumericValue(event.Value)
	if value == nil {
		return false
	}
	
	threshold, ok := condition.Value.(float64)
	if !ok {
		return false
	}
	
	switch condition.Operator {
	case ">":
		return *value > threshold
	case "<":
		return *value < threshold
	case ">=":
		return *value >= threshold
	case "<=":
		return *value <= threshold
	case "==":
		return *value == threshold
	case "!=":
		return *value != threshold
	default:
		return false
	}
}

func (sp *StreamProcessor) extractNumericValue(value interface{}) *float64 {
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

func (sp *StreamProcessor) createAlert(event *AnalyticsEvent, rule *AlertRule) *Alert {
	return &Alert{
		ID:        fmt.Sprintf("alert_%s_%d", rule.ID, time.Now().UnixNano()),
		RuleID:    rule.ID,
		DeviceID:  event.DeviceID,
		GatewayID: event.GatewayID,
		Severity:  rule.Severity,
		Message:   fmt.Sprintf("Alert triggered: %s", rule.Name),
		Value:     event.Value,
		CreatedAt: time.Now(),
		Status:    "active",
	}
}

func (sp *StreamProcessor) triggerAlert(alert *Alert) {
	sp.logger.Warn("Alert triggered",
		"alert_id", alert.ID,
		"rule_id", alert.RuleID,
		"device_id", alert.DeviceID,
		"severity", alert.Severity,
	)
	
	// TODO: Implement alert actions (notifications, etc.)
}

func (sp *StreamProcessor) isModelApplicable(event *AnalyticsEvent, model *EdgeInferenceModel) bool {
	// Simple applicability check - in production, this would be more sophisticated
	return event.Type == EventTypeDeviceData
}

func (sp *StreamProcessor) runInference(event *AnalyticsEvent, model *EdgeInferenceModel) *InferenceResult {
	startTime := time.Now()
	
	// Mock inference - in production, this would call actual ML models
	result := &InferenceResult{
		ID:         fmt.Sprintf("inference_%s_%d", model.ID, time.Now().UnixNano()),
		ModelID:    model.ID,
		DeviceID:   event.DeviceID,
		GatewayID:  event.GatewayID,
		Input:      map[string]interface{}{"value": event.Value},
		Output:     map[string]interface{}{"prediction": "normal"},
		Confidence: 0.85,
		Latency:    time.Since(startTime),
		Timestamp:  time.Now(),
	}
	
	return result
}

// cleanupWindows periodically cleans up old window data
func (sp *StreamProcessor) cleanupWindows() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	
	for {
		select {
		case <-sp.ctx.Done():
			return
		case <-ticker.C:
			sp.windowsMu.Lock()
			cutoff := time.Now().Add(-1 * time.Hour) // Keep 1 hour of data
			for key, events := range sp.dataWindows {
				var filteredEvents []*AnalyticsEvent
				for _, event := range events {
					if event.Timestamp.After(cutoff) {
						filteredEvents = append(filteredEvents, event)
					}
				}
				if len(filteredEvents) == 0 {
					delete(sp.dataWindows, key)
				} else {
					sp.dataWindows[key] = filteredEvents
				}
			}
			sp.windowsMu.Unlock()
		}
	}
}

// cleanupAlertState periodically cleans up old alert state
func (sp *StreamProcessor) cleanupAlertState() {
	ticker := time.NewTicker(10 * time.Minute)
	defer ticker.Stop()
	
	for {
		select {
		case <-sp.ctx.Done():
			return
		case <-ticker.C:
			sp.alertStateMu.Lock()
			cutoff := time.Now().Add(-24 * time.Hour) // Keep 24 hours of alert state
			for ruleID, lastAlert := range sp.alertState {
				if lastAlert.Before(cutoff) {
					delete(sp.alertState, ruleID)
				}
			}
			sp.alertStateMu.Unlock()
		}
	}
}

// GetMetrics returns stream processor metrics
func (sp *StreamProcessor) GetMetrics() map[string]interface{} {
	sp.mu.RLock()
	defer sp.mu.RUnlock()
	
	return map[string]interface{}{
		"events_processed":    sp.eventsProcessed,
		"processing_latency":  sp.processingLatency.Milliseconds(),
		"error_count":         sp.errorCount,
		"aggregation_rules":   len(sp.rules),
		"alert_rules":         len(sp.alertRules),
		"loaded_models":       len(sp.models),
		"active_windows":      len(sp.dataWindows),
		"active_alert_states": len(sp.alertState),
	}
}

// EdgeAnalytics is the main analytics engine
type EdgeAnalytics struct {
	processor    *StreamProcessor
	dataBuffer   chan *AnalyticsEvent
	alerts       []*Alert
	aggregations []*AggregatedData
	inferences   []*InferenceResult
	mu           sync.RWMutex
	logger       logger.Logger
	ctx          context.Context
	cancel       context.CancelFunc
}

// NewEdgeAnalytics creates a new edge analytics engine
func NewEdgeAnalytics() *EdgeAnalytics {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &EdgeAnalytics{
		processor:  NewStreamProcessor(10000),
		dataBuffer: make(chan *AnalyticsEvent, 1000),
		logger:     logger.GlobalLogger,
		ctx:        ctx,
		cancel:     cancel,
	}
}

// Start starts the edge analytics engine
func (ea *EdgeAnalytics) Start() error {
	ea.logger.Info("Starting Edge Analytics Engine")
	
	// Start stream processor
	if err := ea.processor.Start(); err != nil {
		return fmt.Errorf("failed to start stream processor: %w", err)
	}
	
	// Start data ingestion
	go ea.ingestData()
	
	ea.logger.Info("Edge Analytics Engine started successfully")
	return nil
}

// Stop stops the edge analytics engine
func (ea *EdgeAnalytics) Stop() error {
	ea.logger.Info("Stopping Edge Analytics Engine")
	
	ea.cancel()
	
	if err := ea.processor.Stop(); err != nil {
		ea.logger.Error("Error stopping stream processor", "error", err)
	}
	
	close(ea.dataBuffer)
	
	ea.logger.Info("Edge Analytics Engine stopped")
	return nil
}

// ingestData ingests data from the buffer and processes it
func (ea *EdgeAnalytics) ingestData() {
	for {
		select {
		case <-ea.ctx.Done():
			return
		case event := <-ea.dataBuffer:
			if err := ea.processor.ProcessEvent(event); err != nil {
				ea.logger.Error("Failed to process event", "error", err)
			}
		}
	}
}

// ProcessDeviceData processes device data for analytics
func (ea *EdgeAnalytics) ProcessDeviceData(deviceID, gatewayID string, data map[string]interface{}) error {
	for field, value := range data {
		event := &AnalyticsEvent{
			ID:        fmt.Sprintf("event_%d", time.Now().UnixNano()),
			Type:      EventTypeDeviceData,
			DeviceID:  deviceID,
			GatewayID: gatewayID,
			Timestamp: time.Now(),
			DataType:  ea.inferDataType(value),
			Value:     value,
			Metadata: map[string]interface{}{
				"field": field,
			},
		}
		
		select {
		case ea.dataBuffer <- event:
		default:
			return fmt.Errorf("data buffer full")
		}
	}
	
	return nil
}

// inferDataType infers the data type from value
func (ea *EdgeAnalytics) inferDataType(value interface{}) DataType {
	switch value.(type) {
	case int, int32, int64, float32, float64:
		return DataTypeNumeric
	case bool:
		return DataTypeBoolean
	case string:
		return DataTypeString
	default:
		return DataTypeJSON
	}
}

// GetMetrics returns analytics engine metrics
func (ea *EdgeAnalytics) GetMetrics() map[string]interface{} {
	ea.mu.RLock()
	defer ea.mu.RUnlock()
	
	metrics := ea.processor.GetMetrics()
	metrics["alerts_count"] = len(ea.alerts)
	metrics["aggregations_count"] = len(ea.aggregations)
	metrics["inferences_count"] = len(ea.inferences)
	
	return metrics
}