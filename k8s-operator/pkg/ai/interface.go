package ai

import (
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// SchedulingEngine represents the AI-powered scheduling engine
type SchedulingEngine interface {
	// Model management
	CreateModel(modelID string, config *ModelConfig) (*Model, error)
	GetModel(modelID string) (*Model, error)
	UpdateModel(modelID string, config *ModelConfig) error
	DeleteModel(modelID string) error

	// Training operations
	StartTraining(modelID string) (*TrainingJob, error)
	GetTrainingStatus(jobID string) (*TrainingJob, error)

	// Prediction and scheduling
	PredictOptimalPlacement(modelID string, request *PlacementRequest) (*PlacementDecision, error)
	ScheduleWorkload(modelID string, workload *WorkloadSpec) (*SchedulingDecision, error)

	// Performance monitoring
	GetAccuracyMetrics(modelID string) (*AccuracyMetrics, error)
	GetRecentDecisions(modelID string, limit int) ([]*SchedulingDecision, error)

	// Data management
	IngestData(modelID string, data *TrainingData) error
	GetDataSummary(modelID string) (*DataSummary, error)
}

// ModelConfig represents AI model configuration
type ModelConfig struct {
	Type            string                 `json:"type"`
	Version         string                 `json:"version,omitempty"`
	Parameters      map[string]interface{} `json:"parameters,omitempty"`
	Objectives      []Objective            `json:"objectives"`
	DataSources     []DataSource           `json:"dataSources,omitempty"`
	TrainingConfig  *TrainingConfig        `json:"trainingConfig,omitempty"`
	LearningConfig  *LearningConfig        `json:"learningConfig,omitempty"`
}

// Objective represents a scheduling objective
type Objective struct {
	Type   string      `json:"type"`
	Weight float64     `json:"weight"`
	Target interface{} `json:"target,omitempty"`
}

// DataSource represents a data source for training
type DataSource struct {
	Type       string            `json:"type"`
	Connection map[string]string `json:"connection"`
	Metrics    []string          `json:"metrics"`
	Interval   string            `json:"interval,omitempty"`
}

// TrainingConfig represents training configuration
type TrainingConfig struct {
	DatasetSize     int64                  `json:"datasetSize,omitempty"`
	ValidationSplit float64                `json:"validationSplit,omitempty"`
	Epochs          int32                  `json:"epochs,omitempty"`
	Features        []string               `json:"features,omitempty"`
	Hyperparameters map[string]interface{} `json:"hyperparameters,omitempty"`
}

// LearningConfig represents learning configuration
type LearningConfig struct {
	OnlineLearning     bool    `json:"onlineLearning,omitempty"`
	LearningRate       float64 `json:"learningRate,omitempty"`
	BatchSize          int32   `json:"batchSize,omitempty"`
	RetrainingInterval string  `json:"retrainingInterval,omitempty"`
}

// Model represents an AI model
type Model struct {
	ID     string       `json:"id"`
	Config *ModelConfig `json:"config"`
	Status *ModelStatus `json:"status"`
}

// ModelStatus represents model status
type ModelStatus struct {
	State            string           `json:"state"`
	Accuracy         float64          `json:"accuracy,omitempty"`
	LastTraining     *metav1.Time     `json:"lastTraining,omitempty"`
	TrainingProgress float64          `json:"trainingProgress,omitempty"`
}

// TrainingJob represents a training job
type TrainingJob struct {
	ID        string    `json:"id"`
	ModelID   string    `json:"modelId"`
	Status    string    `json:"status"`
	Progress  float64   `json:"progress"`
	StartTime time.Time `json:"startTime"`
	EndTime   time.Time `json:"endTime,omitempty"`
}

// TrainingResult represents training results
type TrainingResult struct {
	JobID     string           `json:"jobId"`
	Completed bool             `json:"completed"`
	Progress  LearningProgress `json:"progress"`
	Error     string           `json:"error,omitempty"`
}

// LearningProgress represents learning progress
type LearningProgress struct {
	Iterations         int64   `json:"iterations"`
	Loss              float64 `json:"loss,omitempty"`
	ValidationAccuracy float64 `json:"validationAccuracy,omitempty"`
}

// PlacementRequest represents a placement request
type PlacementRequest struct {
	WorkloadID  string                 `json:"workloadId"`
	Resources   ResourceRequirements   `json:"resources"`
	Constraints []Constraint           `json:"constraints,omitempty"`
	Preferences map[string]interface{} `json:"preferences,omitempty"`
}

// ResourceRequirements represents resource requirements
type ResourceRequirements struct {
	CPU     string `json:"cpu"`
	Memory  string `json:"memory"`
	Storage string `json:"storage"`
	GPU     int    `json:"gpu,omitempty"`
}

// Constraint represents scheduling constraints
type Constraint struct {
	Type     string      `json:"type"`
	Operator string      `json:"operator"`
	Value    interface{} `json:"value"`
	Weight   float64     `json:"weight,omitempty"`
}

// PlacementDecision represents a placement decision
type PlacementDecision struct {
	Target              string                 `json:"target"`
	Provider            string                 `json:"provider,omitempty"`
	Region              string                 `json:"region,omitempty"`
	Resources           ResourceRequirements   `json:"resources"`
	ExpectedPerformance map[string]interface{} `json:"expectedPerformance,omitempty"`
	Confidence          float64                `json:"confidence"`
	Reasoning           string                 `json:"reasoning,omitempty"`
}

// WorkloadSpec represents workload specification
type WorkloadSpec struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	Resources   ResourceRequirements   `json:"resources"`
	QoS         QoSRequirements        `json:"qos,omitempty"`
	Constraints []Constraint           `json:"constraints,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// QoSRequirements represents quality of service requirements
type QoSRequirements struct {
	Latency     string  `json:"latency,omitempty"`
	Throughput  string  `json:"throughput,omitempty"`
	Availability float64 `json:"availability,omitempty"`
	Priority    int     `json:"priority,omitempty"`
}

// SchedulingDecision represents a scheduling decision
type SchedulingDecision struct {
	WorkloadID string            `json:"workloadId"`
	Timestamp  time.Time         `json:"timestamp"`
	Placement  PlacementDecision `json:"placement"`
	Confidence float64           `json:"confidence"`
	Reasoning  string            `json:"reasoning,omitempty"`
}

// AccuracyMetrics represents accuracy metrics
type AccuracyMetrics struct {
	ShortTerm   float64            `json:"shortTerm,omitempty"`
	LongTerm    float64            `json:"longTerm,omitempty"`
	ByObjective map[string]float64 `json:"byObjective,omitempty"`
}

// TrainingData represents training data
type TrainingData struct {
	Timestamp time.Time              `json:"timestamp"`
	Features  map[string]interface{} `json:"features"`
	Target    interface{}            `json:"target"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// DataSummary represents data summary
type DataSummary struct {
	TotalSamples    int64     `json:"totalSamples"`
	LastUpdate      time.Time `json:"lastUpdate"`
	FeatureSummary  map[string]FeatureStats `json:"featureSummary,omitempty"`
	QualityMetrics  QualityMetrics `json:"qualityMetrics,omitempty"`
}

// FeatureStats represents feature statistics
type FeatureStats struct {
	Type     string  `json:"type"`
	Min      float64 `json:"min,omitempty"`
	Max      float64 `json:"max,omitempty"`
	Mean     float64 `json:"mean,omitempty"`
	StdDev   float64 `json:"stdDev,omitempty"`
	Missing  int64   `json:"missing,omitempty"`
}

// QualityMetrics represents data quality metrics
type QualityMetrics struct {
	Completeness float64 `json:"completeness"`
	Consistency  float64 `json:"consistency"`
	Accuracy     float64 `json:"accuracy"`
	Timeliness   float64 `json:"timeliness"`
}