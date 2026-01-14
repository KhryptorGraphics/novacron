package autoscaling

import (
	"time"
)

// AutoScaler defines the interface for auto-scaling operations
type AutoScaler interface {
	// StartMonitoring begins monitoring and prediction
	StartMonitoring() error
	
	// StopMonitoring stops all monitoring activities
	StopMonitoring() error
	
	// GetScalingDecision gets a scaling decision based on current metrics
	GetScalingDecision(targetID string) (*ScalingDecision, error)
	
	// GetPrediction gets a prediction for future resource needs
	GetPrediction(targetID string, horizonMinutes int) (*ResourcePrediction, error)
	
	// UpdateMetrics updates the metrics store with new data
	UpdateMetrics(metrics *MetricsData) error
}

// MetricsCollector defines interface for collecting time-series metrics
type MetricsCollector interface {
	// CollectMetrics collects current metrics from the system
	CollectMetrics() (*MetricsData, error)
	
	// GetHistoricalMetrics gets historical metrics for a time range
	GetHistoricalMetrics(start, end time.Time) ([]*MetricsData, error)
	
	// Subscribe subscribes to real-time metrics updates
	Subscribe(handler MetricsHandler) error
	
	// StartCollection starts the metrics collection process
	StartCollection() error
	
	// StopCollection stops the metrics collection process
	StopCollection() error
}

// Predictor defines interface for ML prediction algorithms
type Predictor interface {
	// Train trains the prediction model with historical data
	Train(data []*MetricsData) error
	
	// Predict predicts future values based on current trends
	Predict(current *MetricsData, horizonMinutes int) (*ResourcePrediction, error)
	
	// GetAccuracy returns the current model accuracy
	GetAccuracy() float64
	
	// GetModelInfo returns information about the current model
	GetModelInfo() ModelInfo
}

// ScalingDecisionEngine makes scaling decisions based on predictions and policies
type ScalingDecisionEngine interface {
	// MakeDecision makes a scaling decision
	MakeDecision(prediction *ResourcePrediction, current *MetricsData) (*ScalingDecision, error)
	
	// SetThresholds sets the scaling thresholds
	SetThresholds(thresholds *ScalingThresholds) error
	
	// GetThresholds returns current scaling thresholds
	GetThresholds() *ScalingThresholds
}

// MetricsData represents a snapshot of system metrics
type MetricsData struct {
	Timestamp    time.Time            `json:"timestamp"`
	TargetID     string               `json:"target_id"`
	TargetType   string               `json:"target_type"` // vm, node, cluster
	CPUUsage     float64              `json:"cpu_usage"`
	MemoryUsage  float64              `json:"memory_usage"`
	NetworkIO    float64              `json:"network_io"`
	DiskIO       float64              `json:"disk_io"`
	ActiveVMs    int                  `json:"active_vms,omitempty"`
	QueueLength  int                  `json:"queue_length,omitempty"`
	ResponseTime float64              `json:"response_time,omitempty"`
	CustomMetrics map[string]float64  `json:"custom_metrics,omitempty"`
}

// ResourcePrediction represents a prediction of future resource needs
type ResourcePrediction struct {
	TargetID         string                   `json:"target_id"`
	PredictionTime   time.Time                `json:"prediction_time"`
	HorizonMinutes   int                      `json:"horizon_minutes"`
	PredictedCPU     float64                  `json:"predicted_cpu"`
	PredictedMemory  float64                  `json:"predicted_memory"`
	PredictedLoad    float64                  `json:"predicted_load"`
	Confidence       float64                  `json:"confidence"`
	TrendDirection   TrendDirection           `json:"trend_direction"`
	SeasonalFactor   float64                  `json:"seasonal_factor"`
	AnomalyScore     float64                  `json:"anomaly_score"`
	Metadata         map[string]interface{}   `json:"metadata,omitempty"`
}

// TrendDirection represents the direction of the trend
type TrendDirection string

const (
	TrendIncreasing TrendDirection = "increasing"
	TrendDecreasing TrendDirection = "decreasing"
	TrendStable     TrendDirection = "stable"
	TrendVolatile   TrendDirection = "volatile"
)

// ScalingDecision represents a decision to scale resources
type ScalingDecision struct {
	TargetID       string                 `json:"target_id"`
	DecisionTime   time.Time              `json:"decision_time"`
	Action         ScalingAction          `json:"action"`
	CurrentScale   int                    `json:"current_scale"`
	TargetScale    int                    `json:"target_scale"`
	Reason         string                 `json:"reason"`
	Confidence     float64                `json:"confidence"`
	CooldownUntil  time.Time              `json:"cooldown_until,omitempty"`
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
}

// ScalingAction represents the type of scaling action
type ScalingAction string

const (
	ScalingActionScaleUp   ScalingAction = "scale_up"
	ScalingActionScaleDown ScalingAction = "scale_down"
	ScalingActionNoAction  ScalingAction = "no_action"
)

// ScalingThresholds defines thresholds for scaling decisions
type ScalingThresholds struct {
	CPUScaleUpThreshold     float64       `json:"cpu_scale_up_threshold"`
	CPUScaleDownThreshold   float64       `json:"cpu_scale_down_threshold"`
	MemoryScaleUpThreshold  float64       `json:"memory_scale_up_threshold"`
	MemoryScaleDownThreshold float64      `json:"memory_scale_down_threshold"`
	MinReplicas             int           `json:"min_replicas"`
	MaxReplicas             int           `json:"max_replicas"`
	CooldownPeriod          time.Duration `json:"cooldown_period"`
	ScaleUpStabilization    time.Duration `json:"scale_up_stabilization"`
	ScaleDownStabilization  time.Duration `json:"scale_down_stabilization"`
	PredictionWeight        float64       `json:"prediction_weight"` // Weight of prediction vs current metrics
}

// ModelInfo provides information about a prediction model
type ModelInfo struct {
	ModelType     string                 `json:"model_type"`
	Version       string                 `json:"version"`
	TrainedAt     time.Time              `json:"trained_at"`
	DataPoints    int                    `json:"data_points"`
	Accuracy      float64                `json:"accuracy"`
	Parameters    map[string]interface{} `json:"parameters,omitempty"`
}

// MetricsHandler handles metrics updates
type MetricsHandler interface {
	HandleMetrics(metrics *MetricsData) error
}

// AutoScalingEvent represents an auto-scaling event
type AutoScalingEvent struct {
	Type          EventType              `json:"type"`
	TargetID      string                 `json:"target_id"`
	Decision      *ScalingDecision       `json:"decision,omitempty"`
	Prediction    *ResourcePrediction    `json:"prediction,omitempty"`
	Timestamp     time.Time              `json:"timestamp"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
}

// EventType represents the type of auto-scaling event
type EventType string

const (
	EventTypeScalingDecisionMade EventType = "scaling.decision_made"
	EventTypePredictionGenerated EventType = "scaling.prediction_generated"
	EventTypeMetricsUpdated      EventType = "scaling.metrics_updated"
	EventTypeModelRetrained      EventType = "scaling.model_retrained"
	EventTypeThresholdsBreach    EventType = "scaling.thresholds_breach"
)