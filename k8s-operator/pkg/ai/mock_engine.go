package ai

import (
	"fmt"
	"math/rand"
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// MockSchedulingEngine implements SchedulingEngine for testing and development
type MockSchedulingEngine struct {
	models        map[string]*Model
	trainingJobs  map[string]*TrainingJob
	decisions     map[string][]*SchedulingDecision
	trainingData  map[string][]*TrainingData
	mutex         sync.RWMutex
}

// NewMockSchedulingEngine creates a new mock scheduling engine
func NewMockSchedulingEngine() SchedulingEngine {
	return &MockSchedulingEngine{
		models:       make(map[string]*Model),
		trainingJobs: make(map[string]*TrainingJob),
		decisions:    make(map[string][]*SchedulingDecision),
		trainingData: make(map[string][]*TrainingData),
	}
}

// CreateModel creates a new AI model
func (e *MockSchedulingEngine) CreateModel(modelID string, config *ModelConfig) (*Model, error) {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	if _, exists := e.models[modelID]; exists {
		return nil, fmt.Errorf("model %s already exists", modelID)
	}

	model := &Model{
		ID:     modelID,
		Config: config,
		Status: &ModelStatus{
			State:            "initializing",
			Accuracy:         0.0,
			TrainingProgress: 0.0,
		},
	}

	e.models[modelID] = model

	// Simulate initialization delay
	go func() {
		time.Sleep(100 * time.Millisecond)
		e.mutex.Lock()
		model.Status.State = "ready"
		model.Status.Accuracy = 0.75 + rand.Float64()*0.2 // 0.75-0.95
		e.mutex.Unlock()
	}()

	return model, nil
}

// GetModel retrieves a model
func (e *MockSchedulingEngine) GetModel(modelID string) (*Model, error) {
	e.mutex.RLock()
	defer e.mutex.RUnlock()

	model, exists := e.models[modelID]
	if !exists {
		return nil, fmt.Errorf("model %s not found", modelID)
	}

	return model, nil
}

// UpdateModel updates a model configuration
func (e *MockSchedulingEngine) UpdateModel(modelID string, config *ModelConfig) error {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	model, exists := e.models[modelID]
	if !exists {
		return fmt.Errorf("model %s not found", modelID)
	}

	model.Config = config
	model.Status.State = "updating"

	// Simulate update delay
	go func() {
		time.Sleep(50 * time.Millisecond)
		e.mutex.Lock()
		model.Status.State = "ready"
		e.mutex.Unlock()
	}()

	return nil
}

// DeleteModel deletes a model
func (e *MockSchedulingEngine) DeleteModel(modelID string) error {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	if _, exists := e.models[modelID]; !exists {
		return fmt.Errorf("model %s not found", modelID)
	}

	delete(e.models, modelID)
	delete(e.decisions, modelID)
	delete(e.trainingData, modelID)

	// Clean up associated training jobs
	for jobID, job := range e.trainingJobs {
		if job.ModelID == modelID {
			delete(e.trainingJobs, jobID)
		}
	}

	return nil
}

// StartTraining starts model training
func (e *MockSchedulingEngine) StartTraining(modelID string) (*TrainingJob, error) {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	model, exists := e.models[modelID]
	if !exists {
		return nil, fmt.Errorf("model %s not found", modelID)
	}

	jobID := fmt.Sprintf("training-%s-%d", modelID, time.Now().Unix())
	job := &TrainingJob{
		ID:        jobID,
		ModelID:   modelID,
		Status:    "running",
		Progress:  0.0,
		StartTime: time.Now(),
	}

	e.trainingJobs[jobID] = job
	model.Status.State = "training"
	model.Status.TrainingProgress = 0.0

	// Simulate training progress
	go e.simulateTraining(jobID, modelID)

	return job, nil
}

func (e *MockSchedulingEngine) simulateTraining(jobID, modelID string) {
	for progress := 0.0; progress <= 100.0; progress += 10.0 {
		time.Sleep(100 * time.Millisecond)
		
		e.mutex.Lock()
		if job, exists := e.trainingJobs[jobID]; exists {
			job.Progress = progress
			if model, exists := e.models[modelID]; exists {
				model.Status.TrainingProgress = progress / 100.0
			}
		}
		e.mutex.Unlock()
	}

	// Complete training
	e.mutex.Lock()
	if job, exists := e.trainingJobs[jobID]; exists {
		job.Status = "completed"
		job.Progress = 100.0
		job.EndTime = time.Now()
	}
	if model, exists := e.models[modelID]; exists {
		model.Status.State = "ready"
		model.Status.Accuracy = 0.8 + rand.Float64()*0.15 // 0.8-0.95
		model.Status.LastTraining = &metav1.Time{Time: time.Now()}
		model.Status.TrainingProgress = 1.0
	}
	e.mutex.Unlock()
}

// GetTrainingStatus gets training job status
func (e *MockSchedulingEngine) GetTrainingStatus(jobID string) (*TrainingJob, error) {
	e.mutex.RLock()
	defer e.mutex.RUnlock()

	job, exists := e.trainingJobs[jobID]
	if !exists {
		return nil, fmt.Errorf("training job %s not found", jobID)
	}

	return job, nil
}

// PredictOptimalPlacement predicts optimal placement
func (e *MockSchedulingEngine) PredictOptimalPlacement(modelID string, request *PlacementRequest) (*PlacementDecision, error) {
	e.mutex.RLock()
	model, exists := e.models[modelID]
	e.mutex.RUnlock()

	if !exists {
		return nil, fmt.Errorf("model %s not found", modelID)
	}

	if model.Status.State != "ready" {
		return nil, fmt.Errorf("model %s is not ready (state: %s)", modelID, model.Status.State)
	}

	// Mock decision based on objectives
	targets := []string{"node-1", "node-2", "node-3", "cluster-a", "cluster-b"}
	providers := []string{"aws", "azure", "gcp", "on-premises"}

	decision := &PlacementDecision{
		Target:   targets[rand.Intn(len(targets))],
		Provider: providers[rand.Intn(len(providers))],
		Region:   "us-east-1",
		Resources: request.Resources,
		ExpectedPerformance: map[string]interface{}{
			"cpu_utilization":    0.6 + rand.Float64()*0.3, // 60-90%
			"memory_utilization": 0.5 + rand.Float64()*0.3, // 50-80%
			"network_latency":    5 + rand.Float64()*10,     // 5-15ms
		},
		Confidence: model.Status.Accuracy * (0.9 + rand.Float64()*0.1), // Apply some variance
		Reasoning:  e.generateReasoning(model.Config.Objectives, request),
	}

	return decision, nil
}

func (e *MockSchedulingEngine) generateReasoning(objectives []Objective, request *PlacementRequest) string {
	primaryObjective := "performance"
	if len(objectives) > 0 {
		// Find highest weight objective
		maxWeight := 0.0
		for _, obj := range objectives {
			if obj.Weight > maxWeight {
				maxWeight = obj.Weight
				primaryObjective = obj.Type
			}
		}
	}

	switch primaryObjective {
	case "cost":
		return "Selected based on cost optimization: lowest hourly cost with spot instance availability"
	case "performance":
		return "Selected based on performance optimization: high CPU/memory availability and low network latency"
	case "availability":
		return "Selected based on availability requirements: multi-AZ deployment with 99.9% uptime SLA"
	case "energy":
		return "Selected based on energy efficiency: renewable energy powered datacenter"
	default:
		return "Selected based on balanced optimization across multiple objectives"
	}
}

// ScheduleWorkload schedules a workload
func (e *MockSchedulingEngine) ScheduleWorkload(modelID string, workload *WorkloadSpec) (*SchedulingDecision, error) {
	placement, err := e.PredictOptimalPlacement(modelID, &PlacementRequest{
		WorkloadID:  workload.ID,
		Resources:   workload.Resources,
		Constraints: workload.Constraints,
	})
	if err != nil {
		return nil, err
	}

	decision := &SchedulingDecision{
		WorkloadID: workload.ID,
		Timestamp:  time.Now(),
		Placement:  *placement,
		Confidence: placement.Confidence,
		Reasoning:  placement.Reasoning,
	}

	// Store decision
	e.mutex.Lock()
	e.decisions[modelID] = append(e.decisions[modelID], decision)
	// Keep only recent decisions (last 100)
	if len(e.decisions[modelID]) > 100 {
		e.decisions[modelID] = e.decisions[modelID][len(e.decisions[modelID])-100:]
	}
	e.mutex.Unlock()

	return decision, nil
}

// GetAccuracyMetrics gets accuracy metrics for a model
func (e *MockSchedulingEngine) GetAccuracyMetrics(modelID string) (*AccuracyMetrics, error) {
	e.mutex.RLock()
	model, exists := e.models[modelID]
	e.mutex.RUnlock()

	if !exists {
		return nil, fmt.Errorf("model %s not found", modelID)
	}

	// Generate mock metrics based on model accuracy
	baseAccuracy := model.Status.Accuracy
	shortTermVariance := (rand.Float64() - 0.5) * 0.1 // ±5% variance
	longTermVariance := (rand.Float64() - 0.5) * 0.05  // ±2.5% variance

	metrics := &AccuracyMetrics{
		ShortTerm: baseAccuracy + shortTermVariance,
		LongTerm:  baseAccuracy + longTermVariance,
		ByObjective: make(map[string]float64),
	}

	// Generate per-objective accuracy
	for _, objective := range model.Config.Objectives {
		objectiveAccuracy := baseAccuracy * (0.95 + rand.Float64()*0.1) // 95-105% of base
		if objectiveAccuracy > 1.0 {
			objectiveAccuracy = 1.0
		}
		metrics.ByObjective[objective.Type] = objectiveAccuracy
	}

	return metrics, nil
}

// GetRecentDecisions gets recent scheduling decisions
func (e *MockSchedulingEngine) GetRecentDecisions(modelID string, limit int) ([]*SchedulingDecision, error) {
	e.mutex.RLock()
	defer e.mutex.RUnlock()

	decisions, exists := e.decisions[modelID]
	if !exists {
		return []*SchedulingDecision{}, nil
	}

	start := 0
	if len(decisions) > limit {
		start = len(decisions) - limit
	}

	result := make([]*SchedulingDecision, len(decisions[start:]))
	copy(result, decisions[start:])

	return result, nil
}

// IngestData ingests training data
func (e *MockSchedulingEngine) IngestData(modelID string, data *TrainingData) error {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	if _, exists := e.models[modelID]; !exists {
		return fmt.Errorf("model %s not found", modelID)
	}

	e.trainingData[modelID] = append(e.trainingData[modelID], data)

	// Keep only recent data (last 10000 samples)
	if len(e.trainingData[modelID]) > 10000 {
		e.trainingData[modelID] = e.trainingData[modelID][len(e.trainingData[modelID])-10000:]
	}

	return nil
}

// GetDataSummary gets data summary for a model
func (e *MockSchedulingEngine) GetDataSummary(modelID string) (*DataSummary, error) {
	e.mutex.RLock()
	defer e.mutex.RUnlock()

	if _, exists := e.models[modelID]; !exists {
		return nil, fmt.Errorf("model %s not found", modelID)
	}

	data := e.trainingData[modelID]
	summary := &DataSummary{
		TotalSamples: int64(len(data)),
		QualityMetrics: QualityMetrics{
			Completeness: 0.95 + rand.Float64()*0.05, // 95-100%
			Consistency:  0.90 + rand.Float64()*0.10, // 90-100%
			Accuracy:     0.85 + rand.Float64()*0.15, // 85-100%
			Timeliness:   0.88 + rand.Float64()*0.12, // 88-100%
		},
	}

	if len(data) > 0 {
		summary.LastUpdate = data[len(data)-1].Timestamp
	} else {
		summary.LastUpdate = time.Now()
	}

	return summary, nil
}