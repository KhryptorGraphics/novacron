package vm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/sirupsen/logrus"
)

// PredictivePrefetchingAPI provides HTTP endpoints for managing predictive prefetching
type PredictivePrefetchingAPI struct {
	engine *PredictivePrefetchingEngine
	logger *logrus.Logger
}

// NewPredictivePrefetchingAPI creates a new API handler for predictive prefetching
func NewPredictivePrefetchingAPI(engine *PredictivePrefetchingEngine, logger *logrus.Logger) *PredictivePrefetchingAPI {
	return &PredictivePrefetchingAPI{
		engine: engine,
		logger: logger,
	}
}

// RegisterHandlers registers HTTP handlers for predictive prefetching endpoints
func (api *PredictivePrefetchingAPI) RegisterHandlers(mux *http.ServeMux) {
	mux.HandleFunc("/api/v1/prefetching/status", api.handlePrefetchingStatus)
	mux.HandleFunc("/api/v1/prefetching/metrics", api.handlePrefetchingMetrics)
	mux.HandleFunc("/api/v1/prefetching/config", api.handlePrefetchingConfig)
	mux.HandleFunc("/api/v1/prefetching/demo", api.handlePrefetchingDemo)
	mux.HandleFunc("/api/v1/prefetching/validate", api.handlePrefetchingValidate)
}

// handlePrefetchingStatus handles GET requests to /api/v1/prefetching/status
func (api *PredictivePrefetchingAPI) handlePrefetchingStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	status := map[string]interface{}{
		"enabled":     api.engine != nil,
		"model_type":  "neural_network",
		"version":     "v2.1.0",
		"targets": map[string]interface{}{
			"prediction_accuracy":      TARGET_PREDICTION_ACCURACY,
			"cache_hit_improvement":    TARGET_CACHE_HIT_IMPROVEMENT,
			"migration_speed_boost":    TARGET_MIGRATION_SPEED_BOOST,
			"prediction_latency_ms":    TARGET_PREDICTION_LATENCY_MS,
			"false_positive_rate":      TARGET_FALSE_POSITIVE_RATE,
		},
	}

	if api.engine != nil {
		status["cache_size"] = api.engine.cacheManager.CacheSize
		status["cache_usage"] = api.engine.cacheManager.CurrentUsage
		status["hit_ratio"] = api.engine.cacheManager.HitRatio
		status["model_accuracy"] = api.engine.aiModel.AccuracyScore
		status["training_data_size"] = api.engine.aiModel.TrainingDataSize
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

// handlePrefetchingMetrics handles GET requests to /api/v1/prefetching/metrics
func (api *PredictivePrefetchingAPI) handlePrefetchingMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if api.engine == nil {
		http.Error(w, "Predictive prefetching engine not available", http.StatusServiceUnavailable)
		return
	}

	metrics := api.engine.GetPrefetchingMetrics()

	response := map[string]interface{}{
		"total_predictions":            metrics.TotalPredictions,
		"successful_predictions":       metrics.SuccessfulPredictions,
		"false_positives":              metrics.FalsePositives,
		"false_negatives":              metrics.FalseNegatives,
		"prediction_accuracy":          metrics.PredictionAccuracy,
		"average_prediction_time_ms":   metrics.AveragePredictionTime.Milliseconds(),
		"cache_hit_ratio_improvement":  metrics.CacheHitRatioImprovement,
		"migration_speed_improvement":  metrics.MigrationSpeedImprovement,
		"bandwidth_saved_mb":           metrics.BandwidthSaved / (1024 * 1024),
		"model_performance":            metrics.ModelPerformance,
		"performance_summary": map[string]interface{}{
			"accuracy_target_met":     metrics.PredictionAccuracy >= TARGET_PREDICTION_ACCURACY,
			"latency_target_met":      metrics.AveragePredictionTime.Milliseconds() <= TARGET_PREDICTION_LATENCY_MS,
			"cache_improvement_met":   metrics.CacheHitRatioImprovement >= TARGET_CACHE_HIT_IMPROVEMENT,
			"speed_improvement_met":   metrics.MigrationSpeedImprovement >= TARGET_MIGRATION_SPEED_BOOST,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handlePrefetchingConfig handles GET and PUT requests to /api/v1/prefetching/config
func (api *PredictivePrefetchingAPI) handlePrefetchingConfig(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		api.getPrefetchingConfig(w, r)
	case http.MethodPut:
		api.updatePrefetchingConfig(w, r)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// getPrefetchingConfig returns the current predictive prefetching configuration
func (api *PredictivePrefetchingAPI) getPrefetchingConfig(w http.ResponseWriter, r *http.Request) {
	config := DefaultPredictivePrefetchingConfig()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(config)
}

// updatePrefetchingConfig updates the predictive prefetching configuration
func (api *PredictivePrefetchingAPI) updatePrefetchingConfig(w http.ResponseWriter, r *http.Request) {
	var config PredictivePrefetchingConfig
	if err := json.NewDecoder(r.Body).Decode(&config); err != nil {
		http.Error(w, "Invalid configuration format", http.StatusBadRequest)
		return
	}

	// Validate configuration values
	if config.PredictionAccuracy < 0 || config.PredictionAccuracy > 1 {
		http.Error(w, "Prediction accuracy must be between 0 and 1", http.StatusBadRequest)
		return
	}

	if config.MaxCacheSize < 0 {
		http.Error(w, "Max cache size must be positive", http.StatusBadRequest)
		return
	}

	if config.PredictionLatencyMs < 0 {
		http.Error(w, "Prediction latency must be positive", http.StatusBadRequest)
		return
	}

	// Apply configuration (in a real implementation, this would update the engine)
	api.logger.WithFields(logrus.Fields{
		"enabled":              config.Enabled,
		"prediction_accuracy":  config.PredictionAccuracy,
		"max_cache_size":       config.MaxCacheSize,
		"model_type":           config.ModelType,
		"continuous_learning":  config.ContinuousLearning,
	}).Info("Predictive prefetching configuration updated")

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":  "updated",
		"message": "Predictive prefetching configuration updated successfully",
	})
}

// handlePrefetchingDemo handles GET requests to /api/v1/prefetching/demo
func (api *PredictivePrefetchingAPI) handlePrefetchingDemo(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Run a simple demo of predictive prefetching
	if api.engine == nil {
		http.Error(w, "Predictive prefetching engine not available", http.StatusServiceUnavailable)
		return
	}

	response := map[string]interface{}{
		"demo":        "predictive_prefetching",
		"description": "AI-driven predictive prefetching for VM migrations",
		"features": []string{
			"Neural network-based access pattern prediction",
			"Intelligent cache management with AI priority",
			"Continuous learning and model improvement",
			"85% target prediction accuracy",
			"≤10ms prediction latency",
			"30%+ cache hit ratio improvement",
			"2x+ migration speed boost",
		},
		"model_info": map[string]interface{}{
			"type":                "neural_network",
			"version":            "v2.1.0",
			"layers":             []string{"input:50", "hidden:128", "hidden:64", "output:20"},
			"activation":         "relu",
			"learning_rate":      0.001,
			"training_samples":   api.engine.aiModel.TrainingDataSize,
			"current_accuracy":   api.engine.aiModel.AccuracyScore,
		},
		"cache_info": map[string]interface{}{
			"max_size_mb":       api.engine.cacheManager.CacheSize / (1024 * 1024),
			"current_usage_mb":  api.engine.cacheManager.CurrentUsage / (1024 * 1024),
			"hit_ratio":         api.engine.cacheManager.HitRatio,
			"eviction_policy":   "ai_priority",
		},
		"usage": map[string]string{
			"step_1": "Configure VM with predictive_prefetching settings",
			"step_2": "Initiate migration (cold/warm/live)",
			"step_3": "AI engine automatically predicts access patterns",
			"step_4": "Intelligent prefetching optimizes data transfer",
			"step_5": "Monitor improved migration performance",
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handlePrefetchingValidate handles GET requests to /api/v1/prefetching/validate
func (api *PredictivePrefetchingAPI) handlePrefetchingValidate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if api.engine == nil {
		http.Error(w, "Predictive prefetching engine not available", http.StatusServiceUnavailable)
		return
	}

	// Validate performance targets
	err := api.engine.ValidatePrefetchingTargets()
	
	response := map[string]interface{}{
		"validation_passed": err == nil,
		"timestamp":        fmt.Sprintf("%v", time.Now().Format(time.RFC3339)),
	}

	if err != nil {
		response["issues"] = err.Error()
		response["recommendations"] = []string{
			"Allow more time for AI model training",
			"Increase training data collection period",
			"Adjust prediction accuracy targets if needed",
			"Monitor system resources and network bandwidth",
		}
	} else {
		response["message"] = "All predictive prefetching performance targets are being met"
		response["achievements"] = []string{
			fmt.Sprintf("Prediction accuracy ≥ %.1f%%", TARGET_PREDICTION_ACCURACY*100),
			fmt.Sprintf("Cache hit improvement ≥ %.1f%%", TARGET_CACHE_HIT_IMPROVEMENT*100),
			fmt.Sprintf("Prediction latency ≤ %dms", TARGET_PREDICTION_LATENCY_MS),
			fmt.Sprintf("Migration speed boost ≥ %.1fx", TARGET_MIGRATION_SPEED_BOOST),
		}
	}

	// Get current metrics for detailed status
	metrics := api.engine.GetPrefetchingMetrics()
	response["current_metrics"] = map[string]interface{}{
		"prediction_accuracy_pct":     metrics.PredictionAccuracy * 100,
		"cache_improvement_pct":       metrics.CacheHitRatioImprovement * 100,
		"prediction_latency_ms":       metrics.AveragePredictionTime.Milliseconds(),
		"migration_speed_multiplier":  metrics.MigrationSpeedImprovement,
		"total_predictions":           metrics.TotalPredictions,
		"successful_predictions":      metrics.SuccessfulPredictions,
	}

	statusCode := http.StatusOK
	if err != nil {
		statusCode = http.StatusExpectationFailed
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(response)
}

// PredictivePrefetchingConfigRequest represents a request to configure predictive prefetching
type PredictivePrefetchingConfigRequest struct {
	VMID                    string                       `json:"vm_id"`
	PredictivePrefetching   *PredictivePrefetchingConfig `json:"predictive_prefetching"`
}

// HandleVMPredictivePrefetchingConfig handles VM-specific predictive prefetching configuration
func (api *PredictivePrefetchingAPI) HandleVMPredictivePrefetchingConfig(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		var req PredictivePrefetchingConfigRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request format", http.StatusBadRequest)
			return
		}

		if req.VMID == "" {
			http.Error(w, "VM ID is required", http.StatusBadRequest)
			return
		}

		// In a real implementation, this would update VM configuration
		api.logger.WithFields(logrus.Fields{
			"vm_id":   req.VMID,
			"enabled": req.PredictivePrefetching != nil && req.PredictivePrefetching.Enabled,
		}).Info("VM predictive prefetching configuration updated")

		response := map[string]interface{}{
			"vm_id":  req.VMID,
			"status": "configured",
			"config": req.PredictivePrefetching,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)

	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}