package ml

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/gorilla/mux"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/sirupsen/logrus"
	"github.com/khryptorgraphics/novacron/backend/core/ml"
)

var (
	predictionRequestDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "novacron_ml_prediction_request_duration_seconds",
		Help:    "Duration of ML prediction requests",
		Buckets: prometheus.DefBuckets,
	}, []string{"prediction_type", "model"})

	predictionRequestCount = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "novacron_ml_prediction_requests_total",
		Help: "Total number of ML prediction requests",
	}, []string{"prediction_type", "model", "status"})

	modelTrainingDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "novacron_ml_model_training_duration_seconds",
		Help:    "Duration of ML model training",
		Buckets: prometheus.LinearBuckets(60, 300, 10), // 1min to 30min buckets
	}, []string{"model_type"})

	anomalyDetectionLatency = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "novacron_ml_anomaly_detection_latency_seconds",
		Help:    "Latency of anomaly detection requests",
		Buckets: prometheus.ExponentialBuckets(0.001, 2, 10),
	}, []string{"detector_type"})
)

// PredictionHandler handles ML prediction and anomaly detection endpoints
type PredictionHandler struct {
	predictor        *ml.PredictiveAllocator
	anomalyDetector  *ml.AnomalyDetector
	performancePredictor *ml.PerformancePredictor
	modelManager     *ml.ModelManager
	logger           *logrus.Logger
}

// NewPredictionHandler creates a new ML prediction handler
func NewPredictionHandler(predictor *ml.PredictiveAllocator, anomalyDetector *ml.AnomalyDetector, performancePredictor *ml.PerformancePredictor, modelManager *ml.ModelManager, logger *logrus.Logger) *PredictionHandler {
	return &PredictionHandler{
		predictor:            predictor,
		anomalyDetector:      anomalyDetector,
		performancePredictor: performancePredictor,
		modelManager:         modelManager,
		logger:              logger,
	}
}

// ResourcePrediction represents predicted resource requirements
type ResourcePrediction struct {
	VMID         string                   `json:"vm_id"`
	PredictionID string                   `json:"prediction_id"`
	Timestamp    time.Time                `json:"timestamp"`
	TimeHorizon  time.Duration            `json:"time_horizon"`
	Predictions  ResourceForecast         `json:"predictions"`
	Confidence   PredictionConfidence     `json:"confidence"`
	Recommendations []ResourceRecommendation `json:"recommendations"`
	ModelVersion string                   `json:"model_version"`
	Accuracy     PredictionAccuracy       `json:"accuracy,omitempty"`
}

// ResourceForecast represents predicted resource usage
type ResourceForecast struct {
	CPU     TimeSeriesPrediction `json:"cpu"`
	Memory  TimeSeriesPrediction `json:"memory"`
	Storage TimeSeriesPrediction `json:"storage"`
	Network TimeSeriesPrediction `json:"network"`
}

// TimeSeriesPrediction represents time series prediction data
type TimeSeriesPrediction struct {
	Values      []float64   `json:"values"`
	Timestamps  []time.Time `json:"timestamps"`
	Lower       []float64   `json:"lower_bound"`
	Upper       []float64   `json:"upper_bound"`
	Trend       string      `json:"trend"`        // increasing, decreasing, stable
	Seasonality bool        `json:"seasonality"`
	Anomalies   []int       `json:"anomalies"`    // indices of anomalous points
}

// PredictionConfidence represents confidence metrics
type PredictionConfidence struct {
	Overall    float64            `json:"overall"`     // 0.0 to 1.0
	CPU        float64            `json:"cpu"`
	Memory     float64            `json:"memory"`
	Storage    float64            `json:"storage"`
	Network    float64            `json:"network"`
	Factors    map[string]float64 `json:"factors"`     // factors affecting confidence
}

// ResourceRecommendation represents optimization recommendations
type ResourceRecommendation struct {
	Type        string                 `json:"type"`        // scale_up, scale_down, migrate, optimize
	Resource    string                 `json:"resource"`    // cpu, memory, storage, network
	Current     float64                `json:"current"`
	Recommended float64                `json:"recommended"`
	Reason      string                 `json:"reason"`
	Impact      string                 `json:"impact"`      // high, medium, low
	Priority    int                    `json:"priority"`    // 1-10
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// PredictionAccuracy represents model accuracy metrics
type PredictionAccuracy struct {
	MAE    float64 `json:"mae"`     // Mean Absolute Error
	RMSE   float64 `json:"rmse"`    // Root Mean Square Error
	MAPE   float64 `json:"mape"`    // Mean Absolute Percentage Error
	R2     float64 `json:"r2"`      // R-squared
}

// TrainingRequest represents model training request
type TrainingRequest struct {
	ModelType     string                 `json:"model_type" validate:"required"`     // resource, anomaly, performance
	DataSources   []string               `json:"data_sources,omitempty"`
	StartTime     time.Time              `json:"start_time,omitempty"`
	EndTime       time.Time              `json:"end_time,omitempty"`
	Parameters    map[string]interface{} `json:"parameters,omitempty"`
	Incremental   bool                   `json:"incremental,omitempty"`
	ValidationSet float64                `json:"validation_set,omitempty"`         // 0.0 to 1.0
	CrossFold     int                    `json:"cross_fold,omitempty"`
}

// AnomalyDetectionRequest represents anomaly detection request
type AnomalyDetectionRequest struct {
	Data         []DataPoint            `json:"data" validate:"required"`
	DetectorType string                 `json:"detector_type,omitempty"`           // isolation_forest, lstm, statistical
	Sensitivity  float64                `json:"sensitivity,omitempty"`             // 0.0 to 1.0
	WindowSize   int                    `json:"window_size,omitempty"`
	Threshold    float64                `json:"threshold,omitempty"`
	Features     []string               `json:"features,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// DataPoint represents a single data point for analysis
type DataPoint struct {
	Timestamp time.Time              `json:"timestamp"`
	Values    map[string]float64     `json:"values"`
	Labels    map[string]string      `json:"labels,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// AnomalyResult represents anomaly detection result
type AnomalyResult struct {
	DetectionID   string                 `json:"detection_id"`
	Timestamp     time.Time              `json:"timestamp"`
	AnomaliesFound int                   `json:"anomalies_found"`
	Anomalies     []DetectedAnomaly      `json:"anomalies"`
	ModelUsed     string                 `json:"model_used"`
	Confidence    float64                `json:"confidence"`
	ProcessingTime time.Duration         `json:"processing_time"`
	Recommendations []AnomalyRecommendation `json:"recommendations,omitempty"`
}

// DetectedAnomaly represents a detected anomaly
type DetectedAnomaly struct {
	Index       int                    `json:"index"`
	Timestamp   time.Time              `json:"timestamp"`
	Score       float64                `json:"score"`        // anomaly score
	Severity    string                 `json:"severity"`     // low, medium, high, critical
	Feature     string                 `json:"feature"`      // which feature is anomalous
	Value       float64                `json:"value"`        // actual value
	Expected    float64                `json:"expected"`     // expected value
	Deviation   float64                `json:"deviation"`    // deviation from normal
	Description string                 `json:"description"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// AnomalyRecommendation represents recommended actions for anomalies
type AnomalyRecommendation struct {
	Action      string                 `json:"action"`       // investigate, scale, alert, ignore
	Priority    string                 `json:"priority"`     // low, medium, high, urgent
	Description string                 `json:"description"`
	Automated   bool                   `json:"automated"`    // can be automated
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// ModelInfo represents ML model information
type ModelInfo struct {
	ID            string                 `json:"id"`
	Name          string                 `json:"name"`
	Type          string                 `json:"type"`
	Version       string                 `json:"version"`
	Status        string                 `json:"status"`        // training, ready, deployed, deprecated
	Accuracy      PredictionAccuracy     `json:"accuracy"`
	CreatedAt     time.Time              `json:"created_at"`
	UpdatedAt     time.Time              `json:"updated_at"`
	TrainedAt     *time.Time             `json:"trained_at,omitempty"`
	DataSize      int64                  `json:"data_size"`     // training data size
	Parameters    int64                  `json:"parameters"`    // model parameters
	Features      []string               `json:"features"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
}

// RegisterPredictionRoutes registers ML prediction API routes
func (h *PredictionHandler) RegisterPredictionRoutes(router *mux.Router, require func(string, http.HandlerFunc) http.Handler) {
	mlRouter := router.PathPrefix("/api/v1/ml").Subrouter()

	// Predictions (viewer+ for read, operator+ for write)
	mlRouter.Handle("/predictions/{vmId}", require("viewer", h.GetResourcePredictions)).Methods("GET")
	mlRouter.Handle("/predictions/batch", require("viewer", h.GetBatchPredictions)).Methods("POST")
	
	// Model management (admin only for training, viewer+ for list)
	mlRouter.Handle("/train", require("admin", h.TriggerModelTraining)).Methods("POST")
	mlRouter.Handle("/models", require("viewer", h.ListModels)).Methods("GET")
	mlRouter.Handle("/models/{id}", require("viewer", h.GetModel)).Methods("GET")
	mlRouter.Handle("/models/{id}/retrain", require("admin", h.RetrainModel)).Methods("POST")
	
	// Anomaly detection (operator+)
	mlRouter.Handle("/anomaly/detect", require("operator", h.DetectAnomalies)).Methods("POST")
	mlRouter.Handle("/anomaly/models", require("viewer", h.ListAnomalyModels)).Methods("GET")
}

// GetResourcePredictions handles GET /api/v1/ml/predictions/{vmId}
// @Summary Get resource predictions for VM
// @Description Get predicted resource usage for a specific VM
// @Tags ML
// @Produce json
// @Param vmId path string true "VM ID"
// @Param horizon query string false "Prediction time horizon" default(1h)
// @Param model query string false "Model version to use"
// @Param confidence query float64 false "Minimum confidence threshold" default(0.7)
// @Success 200 {object} ResourcePrediction
// @Failure 400 {object} map[string]interface{} "Invalid request"
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 404 {object} map[string]interface{} "VM not found"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/ml/predictions/{vmId} [get]
func (h *PredictionHandler) GetResourcePredictions(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(predictionRequestDuration.WithLabelValues("resource", "default"))
	defer timer.ObserveDuration()

	vars := mux.Vars(r)
	vmID := vars["vmId"]

	if vmID == "" {
		predictionRequestCount.WithLabelValues("resource", "default", "error").Inc()
		writeError(w, http.StatusBadRequest, "MISSING_VM_ID", "VM ID is required")
		return
	}

	// Parse query parameters
	horizonStr := r.URL.Query().Get("horizon")
	if horizonStr == "" {
		horizonStr = "1h"
	}

	horizon, err := time.ParseDuration(horizonStr)
	if err != nil {
		predictionRequestCount.WithLabelValues("resource", "default", "error").Inc()
		writeError(w, http.StatusBadRequest, "INVALID_HORIZON", "Invalid time horizon format")
		return
	}

	modelVersion := r.URL.Query().Get("model")
	if modelVersion == "" {
		modelVersion = "latest"
	}

	confidenceStr := r.URL.Query().Get("confidence")
	confidence := 0.7
	if confidenceStr != "" {
		if c, err := strconv.ParseFloat(confidenceStr, 64); err == nil && c >= 0 && c <= 1 {
			confidence = c
		}
	}

	// Get predictions from ML service
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	prediction, err := h.predictor.PredictResourceUsage(ctx, ml.PredictionRequest{
		VMID:        vmID,
		Horizon:     horizon,
		ModelVersion: modelVersion,
		MinConfidence: confidence,
	})
	if err != nil {
		if err == ml.ErrVMNotFound {
			predictionRequestCount.WithLabelValues("resource", modelVersion, "not_found").Inc()
			writeError(w, http.StatusNotFound, "VM_NOT_FOUND", "VM not found")
			return
		}
		predictionRequestCount.WithLabelValues("resource", modelVersion, "error").Inc()
		h.logger.WithError(err).Error("Failed to get resource predictions")
		writeError(w, http.StatusInternalServerError, "PREDICTION_ERROR", "Failed to generate resource predictions")
		return
	}

	// Convert to API response format
	apiPrediction := h.convertToAPIPrediction(prediction)

	predictionRequestCount.WithLabelValues("resource", modelVersion, "success").Inc()
	writeJSON(w, http.StatusOK, apiPrediction)
}

// GetBatchPredictions handles POST /api/v1/ml/predictions/batch
// @Summary Get batch resource predictions
// @Description Get predictions for multiple VMs in a single request
// @Tags ML
// @Accept json
// @Produce json
// @Param request body map[string]interface{} true "Batch prediction request"
// @Success 200 {object} map[string]interface{}
// @Failure 400 {object} map[string]interface{} "Invalid request"
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/ml/predictions/batch [post]
func (h *PredictionHandler) GetBatchPredictions(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(predictionRequestDuration.WithLabelValues("batch_resource", "default"))
	defer timer.ObserveDuration()

	var req map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		predictionRequestCount.WithLabelValues("batch_resource", "default", "error").Inc()
		writeError(w, http.StatusBadRequest, "INVALID_JSON", "Invalid JSON format")
		return
	}

	vmIDs, ok := req["vm_ids"].([]interface{})
	if !ok || len(vmIDs) == 0 {
		predictionRequestCount.WithLabelValues("batch_resource", "default", "error").Inc()
		writeError(w, http.StatusBadRequest, "MISSING_VM_IDS", "VM IDs are required")
		return
	}

	if len(vmIDs) > 100 {
		predictionRequestCount.WithLabelValues("batch_resource", "default", "error").Inc()
		writeError(w, http.StatusBadRequest, "TOO_MANY_VMS", "Maximum 100 VMs can be processed in batch")
		return
	}

	// Convert to string slice
	vmIDStrings := make([]string, len(vmIDs))
	for i, vmID := range vmIDs {
		if str, ok := vmID.(string); ok {
			vmIDStrings[i] = str
		} else {
			predictionRequestCount.WithLabelValues("batch_resource", "default", "error").Inc()
			writeError(w, http.StatusBadRequest, "INVALID_VM_ID", fmt.Sprintf("VM ID at index %d is not a string", i))
			return
		}
	}

	// Parse optional parameters
	horizonStr, _ := req["horizon"].(string)
	if horizonStr == "" {
		horizonStr = "1h"
	}

	horizon, err := time.ParseDuration(horizonStr)
	if err != nil {
		predictionRequestCount.WithLabelValues("batch_resource", "default", "error").Inc()
		writeError(w, http.StatusBadRequest, "INVALID_HORIZON", "Invalid time horizon format")
		return
	}

	modelVersion, _ := req["model"].(string)
	if modelVersion == "" {
		modelVersion = "latest"
	}

	// Get batch predictions
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
	defer cancel()

	predictions, err := h.predictor.PredictBatchResourceUsage(ctx, ml.BatchPredictionRequest{
		VMIDs:        vmIDStrings,
		Horizon:      horizon,
		ModelVersion: modelVersion,
	})
	if err != nil {
		predictionRequestCount.WithLabelValues("batch_resource", modelVersion, "error").Inc()
		h.logger.WithError(err).Error("Failed to get batch predictions")
		writeError(w, http.StatusInternalServerError, "BATCH_PREDICTION_ERROR", "Failed to generate batch predictions")
		return
	}

	// Convert to API response format
	apiPredictions := make([]ResourcePrediction, len(predictions))
	for i, prediction := range predictions {
		apiPredictions[i] = h.convertToAPIPrediction(prediction)
	}

	predictionRequestCount.WithLabelValues("batch_resource", modelVersion, "success").Inc()

	response := map[string]interface{}{
		"predictions": apiPredictions,
		"total":       len(apiPredictions),
		"model":       modelVersion,
		"horizon":     horizonStr,
		"generated_at": time.Now(),
	}

	writeJSON(w, http.StatusOK, response)
}

// TriggerModelTraining handles POST /api/v1/ml/train
// @Summary Trigger model training
// @Description Initiate training for ML models with specified parameters
// @Tags ML
// @Accept json
// @Produce json
// @Param request body TrainingRequest true "Training request"
// @Success 202 {object} map[string]interface{}
// @Failure 400 {object} map[string]interface{} "Invalid request"
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/ml/train [post]
func (h *PredictionHandler) TriggerModelTraining(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(modelTrainingDuration.WithLabelValues(""))
	defer timer.ObserveDuration()

	var req TrainingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "INVALID_JSON", "Invalid JSON format")
		return
	}

	// Validate request
	if req.ModelType == "" {
		writeError(w, http.StatusBadRequest, "MISSING_MODEL_TYPE", "Model type is required")
		return
	}

	validTypes := map[string]bool{"resource": true, "anomaly": true, "performance": true}
	if !validTypes[req.ModelType] {
		writeError(w, http.StatusBadRequest, "INVALID_MODEL_TYPE", "Model type must be one of: resource, anomaly, performance")
		return
	}

	// Set defaults
	if req.ValidationSet == 0 {
		req.ValidationSet = 0.2
	}
	if req.CrossFold == 0 {
		req.CrossFold = 5
	}

	// Start training process
	trainingID, err := h.modelManager.StartTraining(r.Context(), ml.TrainingConfig{
		ModelType:     req.ModelType,
		DataSources:   req.DataSources,
		StartTime:     req.StartTime,
		EndTime:       req.EndTime,
		Parameters:    req.Parameters,
		Incremental:   req.Incremental,
		ValidationSet: req.ValidationSet,
		CrossFold:     req.CrossFold,
	})
	if err != nil {
		h.logger.WithError(err).Error("Failed to start model training")
		writeError(w, http.StatusInternalServerError, "TRAINING_ERROR", "Failed to initiate model training")
		return
	}

	writeJSON(w, http.StatusAccepted, map[string]interface{}{
		"training_id": trainingID,
		"model_type":  req.ModelType,
		"status":      "started",
		"message":     "Model training initiated",
		"started_at":  time.Now(),
	})
}

// ListModels handles GET /api/v1/ml/models
// @Summary List ML models
// @Description List all available ML models with their status and performance metrics
// @Tags ML
// @Produce json
// @Param type query string false "Filter by model type"
// @Param status query string false "Filter by status"
// @Success 200 {object} map[string]interface{}
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/ml/models [get]
func (h *PredictionHandler) ListModels(w http.ResponseWriter, r *http.Request) {
	modelType := r.URL.Query().Get("type")
	status := r.URL.Query().Get("status")

	models, err := h.modelManager.ListModels(r.Context(), ml.ModelFilters{
		Type:   modelType,
		Status: status,
	})
	if err != nil {
		h.logger.WithError(err).Error("Failed to list models")
		writeError(w, http.StatusInternalServerError, "LIST_ERROR", "Failed to list models")
		return
	}

	// Convert to API response format
	apiModels := make([]ModelInfo, len(models))
	for i, model := range models {
		apiModels[i] = h.convertToAPIModelInfo(model)
	}

	response := map[string]interface{}{
		"models": apiModels,
		"total":  len(apiModels),
	}

	writeJSON(w, http.StatusOK, response)
}

// GetModel handles GET /api/v1/ml/models/{id}
// @Summary Get model details
// @Description Get detailed information about a specific ML model
// @Tags ML
// @Produce json
// @Param id path string true "Model ID"
// @Success 200 {object} ModelInfo
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 404 {object} map[string]interface{} "Model not found"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/ml/models/{id} [get]
func (h *PredictionHandler) GetModel(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	modelID := vars["id"]

	if modelID == "" {
		writeError(w, http.StatusBadRequest, "MISSING_MODEL_ID", "Model ID is required")
		return
	}

	model, err := h.modelManager.GetModel(r.Context(), modelID)
	if err != nil {
		if err == ml.ErrModelNotFound {
			writeError(w, http.StatusNotFound, "MODEL_NOT_FOUND", "Model not found")
			return
		}
		h.logger.WithError(err).Error("Failed to get model")
		writeError(w, http.StatusInternalServerError, "INTERNAL_ERROR", "Failed to retrieve model information")
		return
	}

	apiModel := h.convertToAPIModelInfo(model)
	writeJSON(w, http.StatusOK, apiModel)
}

// RetrainModel handles POST /api/v1/ml/models/{id}/retrain
// @Summary Retrain existing model
// @Description Retrain an existing model with new data
// @Tags ML
// @Accept json
// @Produce json
// @Param id path string true "Model ID"
// @Param request body map[string]interface{} false "Retrain parameters"
// @Success 202 {object} map[string]interface{}
// @Failure 400 {object} map[string]interface{} "Invalid request"
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 404 {object} map[string]interface{} "Model not found"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/ml/models/{id}/retrain [post]
func (h *PredictionHandler) RetrainModel(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	modelID := vars["id"]

	if modelID == "" {
		writeError(w, http.StatusBadRequest, "MISSING_MODEL_ID", "Model ID is required")
		return
	}

	var req map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		// Allow empty body for retrain
		req = make(map[string]interface{})
	}

	retrainID, err := h.modelManager.RetrainModel(r.Context(), modelID, req)
	if err != nil {
		if err == ml.ErrModelNotFound {
			writeError(w, http.StatusNotFound, "MODEL_NOT_FOUND", "Model not found")
			return
		}
		h.logger.WithError(err).Error("Failed to retrain model")
		writeError(w, http.StatusInternalServerError, "RETRAIN_ERROR", "Failed to initiate model retraining")
		return
	}

	writeJSON(w, http.StatusAccepted, map[string]interface{}{
		"retrain_id": retrainID,
		"model_id":   modelID,
		"status":     "started",
		"message":    "Model retraining initiated",
		"started_at": time.Now(),
	})
}

// DetectAnomalies handles POST /api/v1/ml/anomaly/detect
// @Summary Detect anomalies in data
// @Description Analyze data for anomalies using ML models
// @Tags ML
// @Accept json
// @Produce json
// @Param request body AnomalyDetectionRequest true "Anomaly detection request"
// @Success 200 {object} AnomalyResult
// @Failure 400 {object} map[string]interface{} "Invalid request"
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/ml/anomaly/detect [post]
func (h *PredictionHandler) DetectAnomalies(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(anomalyDetectionLatency.WithLabelValues("default"))
	defer timer.ObserveDuration()

	var req AnomalyDetectionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "INVALID_JSON", "Invalid JSON format")
		return
	}

	// Validate request
	if len(req.Data) == 0 {
		writeError(w, http.StatusBadRequest, "MISSING_DATA", "Data points are required for anomaly detection")
		return
	}

	if len(req.Data) > 10000 {
		writeError(w, http.StatusBadRequest, "TOO_MUCH_DATA", "Maximum 10,000 data points can be processed")
		return
	}

	// Set defaults
	if req.DetectorType == "" {
		req.DetectorType = "isolation_forest"
	}
	if req.Sensitivity == 0 {
		req.Sensitivity = 0.5
	}

	// Convert to ML format
	mlData := make([]ml.DataPoint, len(req.Data))
	for i, dp := range req.Data {
		mlData[i] = ml.DataPoint{
			Timestamp: dp.Timestamp,
			Values:    dp.Values,
			Labels:    dp.Labels,
			Metadata:  dp.Metadata,
		}
	}

	// Perform anomaly detection
	ctx, cancel := context.WithTimeout(r.Context(), 1*time.Minute)
	defer cancel()

	result, err := h.anomalyDetector.DetectAnomalies(ctx, ml.AnomalyDetectionRequest{
		Data:         mlData,
		DetectorType: req.DetectorType,
		Sensitivity:  req.Sensitivity,
		WindowSize:   req.WindowSize,
		Threshold:    req.Threshold,
		Features:     req.Features,
		Metadata:     req.Metadata,
	})
	if err != nil {
		h.logger.WithError(err).Error("Failed to detect anomalies")
		writeError(w, http.StatusInternalServerError, "ANOMALY_ERROR", "Failed to detect anomalies")
		return
	}

	// Convert to API response format
	apiResult := h.convertToAPIAnomalyResult(result)
	writeJSON(w, http.StatusOK, apiResult)
}

// ListAnomalyModels handles GET /api/v1/ml/anomaly/models
// @Summary List anomaly detection models
// @Description List all available anomaly detection models
// @Tags ML
// @Produce json
// @Success 200 {object} map[string]interface{}
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/ml/anomaly/models [get]
func (h *PredictionHandler) ListAnomalyModels(w http.ResponseWriter, r *http.Request) {
	models, err := h.anomalyDetector.ListModels(r.Context())
	if err != nil {
		h.logger.WithError(err).Error("Failed to list anomaly models")
		writeError(w, http.StatusInternalServerError, "LIST_ERROR", "Failed to list anomaly detection models")
		return
	}

	// Convert to API response format
	apiModels := make([]ModelInfo, len(models))
	for i, model := range models {
		apiModels[i] = h.convertToAPIModelInfo(model)
	}

	response := map[string]interface{}{
		"models": apiModels,
		"total":  len(apiModels),
		"types":  []string{"isolation_forest", "lstm", "statistical"},
	}

	writeJSON(w, http.StatusOK, response)
}

// Helper functions

func (h *PredictionHandler) convertToAPIPrediction(prediction ml.ResourcePrediction) ResourcePrediction {
	return ResourcePrediction{
		VMID:         prediction.VMID,
		PredictionID: prediction.PredictionID,
		Timestamp:    prediction.Timestamp,
		TimeHorizon:  prediction.TimeHorizon,
		Predictions: ResourceForecast{
			CPU:     h.convertTimeSeriesPrediction(prediction.Predictions.CPU),
			Memory:  h.convertTimeSeriesPrediction(prediction.Predictions.Memory),
			Storage: h.convertTimeSeriesPrediction(prediction.Predictions.Storage),
			Network: h.convertTimeSeriesPrediction(prediction.Predictions.Network),
		},
		Confidence: PredictionConfidence{
			Overall: prediction.Confidence.Overall,
			CPU:     prediction.Confidence.CPU,
			Memory:  prediction.Confidence.Memory,
			Storage: prediction.Confidence.Storage,
			Network: prediction.Confidence.Network,
			Factors: prediction.Confidence.Factors,
		},
		Recommendations: h.convertRecommendations(prediction.Recommendations),
		ModelVersion:    prediction.ModelVersion,
		Accuracy:        h.convertAccuracy(prediction.Accuracy),
	}
}

func (h *PredictionHandler) convertTimeSeriesPrediction(ts ml.TimeSeriesPrediction) TimeSeriesPrediction {
	return TimeSeriesPrediction{
		Values:      ts.Values,
		Timestamps:  ts.Timestamps,
		Lower:       ts.LowerBound,
		Upper:       ts.UpperBound,
		Trend:       ts.Trend,
		Seasonality: ts.Seasonality,
		Anomalies:   ts.Anomalies,
	}
}

func (h *PredictionHandler) convertRecommendations(recs []ml.ResourceRecommendation) []ResourceRecommendation {
	apiRecs := make([]ResourceRecommendation, len(recs))
	for i, rec := range recs {
		apiRecs[i] = ResourceRecommendation{
			Type:        rec.Type,
			Resource:    rec.Resource,
			Current:     rec.Current,
			Recommended: rec.Recommended,
			Reason:      rec.Reason,
			Impact:      rec.Impact,
			Priority:    rec.Priority,
			Metadata:    rec.Metadata,
		}
	}
	return apiRecs
}

func (h *PredictionHandler) convertAccuracy(acc ml.PredictionAccuracy) PredictionAccuracy {
	return PredictionAccuracy{
		MAE:  acc.MAE,
		RMSE: acc.RMSE,
		MAPE: acc.MAPE,
		R2:   acc.R2,
	}
}

func (h *PredictionHandler) convertToAPIModelInfo(model ml.ModelInfo) ModelInfo {
	return ModelInfo{
		ID:          model.ID,
		Name:        model.Name,
		Type:        model.Type,
		Version:     model.Version,
		Status:      model.Status,
		Accuracy:    h.convertAccuracy(model.Accuracy),
		CreatedAt:   model.CreatedAt,
		UpdatedAt:   model.UpdatedAt,
		TrainedAt:   model.TrainedAt,
		DataSize:    model.DataSize,
		Parameters:  model.Parameters,
		Features:    model.Features,
		Metadata:    model.Metadata,
	}
}

func (h *PredictionHandler) convertToAPIAnomalyResult(result ml.AnomalyResult) AnomalyResult {
	anomalies := make([]DetectedAnomaly, len(result.Anomalies))
	for i, anomaly := range result.Anomalies {
		anomalies[i] = DetectedAnomaly{
			Index:       anomaly.Index,
			Timestamp:   anomaly.Timestamp,
			Score:       anomaly.Score,
			Severity:    anomaly.Severity,
			Feature:     anomaly.Feature,
			Value:       anomaly.Value,
			Expected:    anomaly.Expected,
			Deviation:   anomaly.Deviation,
			Description: anomaly.Description,
			Metadata:    anomaly.Metadata,
		}
	}

	recommendations := make([]AnomalyRecommendation, len(result.Recommendations))
	for i, rec := range result.Recommendations {
		recommendations[i] = AnomalyRecommendation{
			Action:      rec.Action,
			Priority:    rec.Priority,
			Description: rec.Description,
			Automated:   rec.Automated,
			Metadata:    rec.Metadata,
		}
	}

	return AnomalyResult{
		DetectionID:     result.DetectionID,
		Timestamp:       result.Timestamp,
		AnomaliesFound:  result.AnomaliesFound,
		Anomalies:       anomalies,
		ModelUsed:       result.ModelUsed,
		Confidence:      result.Confidence,
		ProcessingTime:  result.ProcessingTime,
		Recommendations: recommendations,
	}
}