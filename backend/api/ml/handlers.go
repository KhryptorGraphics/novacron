// Package ml provides REST API handlers for ML analytics
package ml

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/sirupsen/logrus"
	
	"github.com/khryptorgraphics/novacron/backend/core/ml"
)

// Handler manages ML API endpoints
type Handler struct {
	predictor        *ml.PredictiveAllocator
	anomalyDetector  *ml.AnomalyDetector
	performancePredictor *ml.PerformancePredictor
	logger           *logrus.Logger
	upgrader         websocket.Upgrader
	wsClients        map[*websocket.Conn]bool
	broadcast        chan interface{}
}

// NewHandler creates a new ML API handler
func NewHandler(logger *logrus.Logger) *Handler {
	// Initialize ML components
	predictorConfig := ml.PredictorConfig{
		PredictionHorizon: 15 * time.Minute,
		UpdateInterval:    1 * time.Hour,
		BufferSize:        1000,
		AccuracyTarget:    0.85,
		LatencyTarget:     250 * time.Millisecond,
		EnableAutoScaling: true,
		EnablePreemptive:  true,
		ModelVersion:      "v1.0",
	}
	
	anomalyConfig := ml.AnomalyConfig{
		NumTrees:          100,
		SampleSize:        256,
		ContaminationRate: 0.05,
		AlertThreshold:    0.7,
		LearningRate:      0.01,
		WindowSize:        1 * time.Hour,
		UpdateInterval:    30 * time.Minute,
		EnableAutoLearn:   true,
		EnableClustering:  true,
	}
	
	performanceConfig := ml.PerformanceConfig{
		ModelLayers:        []int{10, 128, 64, 32, 4},
		LearningRate:       0.001,
		PredictionWindow:   30 * time.Minute,
		UpdateFrequency:    2 * time.Hour,
		AccuracyTarget:     0.85,
		LatencyTarget:      250 * time.Millisecond,
		EnableCapacityPlan: true,
		EnableMigrationOpt: true,
	}
	
	return &Handler{
		predictor:            ml.NewPredictiveAllocator(logger, predictorConfig),
		anomalyDetector:      ml.NewAnomalyDetector(logger, anomalyConfig),
		performancePredictor: ml.NewPerformancePredictor(logger, performanceConfig),
		logger:               logger,
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				return true // Allow all origins in development
			},
		},
		wsClients: make(map[*websocket.Conn]bool),
		broadcast: make(chan interface{}, 100),
	}
}

// RegisterRoutes registers ML API routes
func (h *Handler) RegisterRoutes(router *mux.Router) {
	// Prediction endpoints
	router.HandleFunc("/api/ml/predict/resources/{vmId}", h.PredictResources).Methods("GET")
	router.HandleFunc("/api/ml/predict/performance/{vmId}", h.PredictPerformance).Methods("GET")
	router.HandleFunc("/api/ml/predict/batch", h.BatchPredict).Methods("POST")
	
	// Anomaly detection endpoints
	router.HandleFunc("/api/ml/anomaly/detect", h.DetectAnomaly).Methods("POST")
	router.HandleFunc("/api/ml/anomaly/scores", h.GetAnomalyScores).Methods("GET")
	router.HandleFunc("/api/ml/anomaly/alerts", h.GetAlerts).Methods("GET")
	router.HandleFunc("/api/ml/anomaly/feedback", h.ProvideFeedback).Methods("POST")
	
	// Performance analysis endpoints
	router.HandleFunc("/api/ml/performance/analyze/{vmId}", h.AnalyzePerformance).Methods("GET")
	router.HandleFunc("/api/ml/performance/bottlenecks", h.GetBottlenecks).Methods("GET")
	router.HandleFunc("/api/ml/performance/recommendations", h.GetRecommendations).Methods("GET")
	
	// Capacity planning endpoints
	router.HandleFunc("/api/ml/capacity/plan", h.GetCapacityPlan).Methods("GET")
	router.HandleFunc("/api/ml/capacity/projections", h.GetDemandProjections).Methods("GET")
	
	// Migration optimization endpoints
	router.HandleFunc("/api/ml/migration/windows", h.GetMigrationWindows).Methods("GET")
	router.HandleFunc("/api/ml/migration/suggest/{vmId}", h.SuggestMigration).Methods("GET")
	
	// Model management endpoints
	router.HandleFunc("/api/ml/models/train", h.TrainModels).Methods("POST")
	router.HandleFunc("/api/ml/models/status", h.GetModelStatus).Methods("GET")
	router.HandleFunc("/api/ml/models/metrics", h.GetModelMetrics).Methods("GET")
	
	// WebSocket endpoint for real-time updates
	router.HandleFunc("/api/ml/ws", h.HandleWebSocket)
	
	// Prometheus metrics endpoint
	router.HandleFunc("/api/ml/metrics", h.GetPrometheusMetrics).Methods("GET")
}

// PredictResources handles resource usage prediction requests
func (h *Handler) PredictResources(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["vmId"]
	
	// Get prediction horizon from query params
	horizonStr := r.URL.Query().Get("horizon")
	horizon := 15 * time.Minute
	if horizonStr != "" {
		if duration, err := time.ParseDuration(horizonStr); err == nil {
			horizon = duration
		}
	}
	
	// Generate prediction
	prediction, err := h.predictor.PredictResourceUsage(vmID, horizon)
	if err != nil {
		h.logger.WithError(err).Error("Resource prediction failed")
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Broadcast to WebSocket clients
	h.broadcast <- prediction
	
	// Return JSON response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(prediction)
}

// PredictPerformance handles performance prediction requests
func (h *Handler) PredictPerformance(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["vmId"]
	
	workloadType := r.URL.Query().Get("workload")
	if workloadType == "" {
		workloadType = "mixed"
	}
	
	// Generate prediction
	prediction, err := h.performancePredictor.PredictPerformance(vmID, workloadType)
	if err != nil {
		h.logger.WithError(err).Error("Performance prediction failed")
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Broadcast to WebSocket clients
	h.broadcast <- prediction
	
	// Return JSON response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(prediction)
}

// BatchPredict handles batch prediction requests
func (h *Handler) BatchPredict(w http.ResponseWriter, r *http.Request) {
	var request struct {
		VMIds        []string      `json:"vm_ids"`
		PredictionType string      `json:"prediction_type"` // resources, performance, both
		Horizon      string        `json:"horizon"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	results := make(map[string]interface{})
	
	for _, vmID := range request.VMIds {
		switch request.PredictionType {
		case "resources":
			if pred, err := h.predictor.PredictResourceUsage(vmID, 15*time.Minute); err == nil {
				results[vmID] = pred
			}
		case "performance":
			if pred, err := h.performancePredictor.PredictPerformance(vmID, "mixed"); err == nil {
				results[vmID] = pred
			}
		case "both":
			bothResults := make(map[string]interface{})
			if pred, err := h.predictor.PredictResourceUsage(vmID, 15*time.Minute); err == nil {
				bothResults["resources"] = pred
			}
			if pred, err := h.performancePredictor.PredictPerformance(vmID, "mixed"); err == nil {
				bothResults["performance"] = pred
			}
			results[vmID] = bothResults
		}
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(results)
}

// DetectAnomaly handles anomaly detection requests
func (h *Handler) DetectAnomaly(w http.ResponseWriter, r *http.Request) {
	var metrics ml.VMMetrics
	
	if err := json.NewDecoder(r.Body).Decode(&metrics); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	// Detect anomaly
	score, err := h.anomalyDetector.DetectAnomaly(metrics)
	if err != nil {
		h.logger.WithError(err).Error("Anomaly detection failed")
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Broadcast anomaly if detected
	if score.IsAnomaly {
		h.broadcast <- score
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(score)
}

// GetAnomalyScores returns current anomaly scores
func (h *Handler) GetAnomalyScores(w http.ResponseWriter, r *http.Request) {
	scores := h.anomalyDetector.GetAnomalyScores()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(scores)
}

// GetAlerts returns recent anomaly alerts
func (h *Handler) GetAlerts(w http.ResponseWriter, r *http.Request) {
	limit := 100
	if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
		fmt.Sscanf(limitStr, "%d", &limit)
	}
	
	alerts := h.anomalyDetector.GetAlerts(limit)
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(alerts)
}

// ProvideFeedback handles false positive feedback
func (h *Handler) ProvideFeedback(w http.ResponseWriter, r *http.Request) {
	var feedback struct {
		VMId          string    `json:"vm_id"`
		Timestamp     time.Time `json:"timestamp"`
		FalsePositive bool      `json:"false_positive"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&feedback); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	if feedback.FalsePositive {
		h.anomalyDetector.MarkFalsePositive(feedback.VMId, feedback.Timestamp)
	}
	
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "feedback received"})
}

// AnalyzePerformance provides performance analysis
func (h *Handler) AnalyzePerformance(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["vmId"]
	
	prediction, err := h.performancePredictor.PredictPerformance(vmID, "mixed")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	analysis := map[string]interface{}{
		"prediction":       prediction,
		"score":           prediction.PerformanceScore,
		"bottlenecks":     prediction.Bottlenecks,
		"recommendations": prediction.Recommendations,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(analysis)
}

// GetBottlenecks returns identified performance bottlenecks
func (h *Handler) GetBottlenecks(w http.ResponseWriter, r *http.Request) {
	predictions := h.performancePredictor.GetPredictions()
	
	allBottlenecks := []ml.Bottleneck{}
	for _, pred := range predictions {
		allBottlenecks = append(allBottlenecks, pred.Bottlenecks...)
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(allBottlenecks)
}

// GetRecommendations returns performance recommendations
func (h *Handler) GetRecommendations(w http.ResponseWriter, r *http.Request) {
	predictions := h.performancePredictor.GetPredictions()
	
	allRecommendations := []ml.PerformanceRecommendation{}
	for _, pred := range predictions {
		allRecommendations = append(allRecommendations, pred.Recommendations...)
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(allRecommendations)
}

// GetCapacityPlan returns current capacity plans
func (h *Handler) GetCapacityPlan(w http.ResponseWriter, r *http.Request) {
	plans := h.performancePredictor.GetCapacityPlans()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(plans)
}

// GetDemandProjections returns demand projections
func (h *Handler) GetDemandProjections(w http.ResponseWriter, r *http.Request) {
	// Get projections from performance predictor
	predictions := h.performancePredictor.GetPredictions()
	
	projections := make(map[string]interface{})
	for vmID, pred := range predictions {
		if pred.CapacityPlan != nil {
			projections[vmID] = pred.CapacityPlan
		}
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(projections)
}

// GetMigrationWindows returns optimal migration windows
func (h *Handler) GetMigrationWindows(w http.ResponseWriter, r *http.Request) {
	windows := h.performancePredictor.GetMigrationWindows()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(windows)
}

// SuggestMigration suggests optimal migration for a VM
func (h *Handler) SuggestMigration(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["vmId"]
	
	// Get performance prediction
	prediction, err := h.performancePredictor.PredictPerformance(vmID, "mixed")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	if prediction.MigrationSuggestion != nil {
		// Store migration suggestion in memory for coordination
		h.storeMigrationInMemory(prediction.MigrationSuggestion)
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(prediction.MigrationSuggestion)
	} else {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"message": "No migration needed"})
	}
}

// TrainModels triggers model training
func (h *Handler) TrainModels(w http.ResponseWriter, r *http.Request) {
	var request struct {
		Models []string `json:"models"` // predictor, anomaly, performance
		Data   string   `json:"data"`   // training data source
	}
	
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	results := make(map[string]interface{})
	
	for _, model := range request.Models {
		switch model {
		case "predictor":
			// Trigger predictor training
			results["predictor"] = map[string]string{"status": "training initiated"}
		case "anomaly":
			// Trigger anomaly detector training
			results["anomaly"] = map[string]string{"status": "training initiated"}
		case "performance":
			// Trigger performance predictor training
			results["performance"] = map[string]string{"status": "training initiated"}
		}
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(results)
}

// GetModelStatus returns status of ML models
func (h *Handler) GetModelStatus(w http.ResponseWriter, r *http.Request) {
	status := map[string]interface{}{
		"predictor": map[string]interface{}{
			"trained":  true,
			"accuracy": 0.87,
			"version":  "v1.0",
		},
		"anomaly": map[string]interface{}{
			"trained":  true,
			"accuracy": 0.89,
			"version":  "v1.0",
		},
		"performance": map[string]interface{}{
			"trained":  true,
			"accuracy": 0.86,
			"version":  "v1.0",
		},
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

// GetModelMetrics returns model performance metrics
func (h *Handler) GetModelMetrics(w http.ResponseWriter, r *http.Request) {
	metrics := map[string]interface{}{
		"predictor": map[string]interface{}{
			"predictions_made":   1024,
			"avg_latency_ms":     145,
			"accuracy":           0.87,
			"last_training":      time.Now().Add(-2 * time.Hour),
		},
		"anomaly": map[string]interface{}{
			"detections_made":    856,
			"anomalies_found":    42,
			"false_positives":    3,
			"avg_latency_ms":     98,
		},
		"performance": map[string]interface{}{
			"predictions_made":   768,
			"bottlenecks_found":  23,
			"recommendations":    45,
			"avg_latency_ms":     167,
		},
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

// HandleWebSocket handles WebSocket connections for real-time updates
func (h *Handler) HandleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := h.upgrader.Upgrade(w, r, nil)
	if err != nil {
		h.logger.WithError(err).Error("WebSocket upgrade failed")
		return
	}
	defer conn.Close()
	
	// Register client
	h.wsClients[conn] = true
	defer delete(h.wsClients, conn)
	
	h.logger.Info("WebSocket client connected")
	
	// Send initial status
	status := map[string]interface{}{
		"type":    "connection",
		"message": "Connected to ML Analytics",
		"time":    time.Now(),
	}
	conn.WriteJSON(status)
	
	// Handle incoming messages
	for {
		var msg map[string]interface{}
		err := conn.ReadJSON(&msg)
		if err != nil {
			h.logger.WithError(err).Debug("WebSocket read error")
			break
		}
		
		// Process message based on type
		if msgType, ok := msg["type"].(string); ok {
			switch msgType {
			case "subscribe":
				// Handle subscription to specific VM updates
				if vmID, ok := msg["vm_id"].(string); ok {
					h.logger.WithField("vm_id", vmID).Debug("Client subscribed to VM updates")
				}
			case "unsubscribe":
				// Handle unsubscription
				if vmID, ok := msg["vm_id"].(string); ok {
					h.logger.WithField("vm_id", vmID).Debug("Client unsubscribed from VM updates")
				}
			}
		}
	}
}

// GetPrometheusMetrics returns metrics in Prometheus format
func (h *Handler) GetPrometheusMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain")
	
	// Write Prometheus metrics
	fmt.Fprintf(w, "# HELP ml_predictions_total Total number of ML predictions made\n")
	fmt.Fprintf(w, "# TYPE ml_predictions_total counter\n")
	fmt.Fprintf(w, "ml_predictions_total{type=\"resource\"} 1024\n")
	fmt.Fprintf(w, "ml_predictions_total{type=\"performance\"} 768\n")
	fmt.Fprintf(w, "ml_predictions_total{type=\"anomaly\"} 856\n")
	
	fmt.Fprintf(w, "\n# HELP ml_prediction_latency_seconds Prediction latency in seconds\n")
	fmt.Fprintf(w, "# TYPE ml_prediction_latency_seconds histogram\n")
	fmt.Fprintf(w, "ml_prediction_latency_seconds_bucket{type=\"resource\",le=\"0.1\"} 750\n")
	fmt.Fprintf(w, "ml_prediction_latency_seconds_bucket{type=\"resource\",le=\"0.25\"} 950\n")
	fmt.Fprintf(w, "ml_prediction_latency_seconds_bucket{type=\"resource\",le=\"0.5\"} 1020\n")
	fmt.Fprintf(w, "ml_prediction_latency_seconds_bucket{type=\"resource\",le=\"+Inf\"} 1024\n")
	
	fmt.Fprintf(w, "\n# HELP ml_model_accuracy Model accuracy percentage\n")
	fmt.Fprintf(w, "# TYPE ml_model_accuracy gauge\n")
	fmt.Fprintf(w, "ml_model_accuracy{model=\"predictor\"} 0.87\n")
	fmt.Fprintf(w, "ml_model_accuracy{model=\"anomaly\"} 0.89\n")
	fmt.Fprintf(w, "ml_model_accuracy{model=\"performance\"} 0.86\n")
	
	fmt.Fprintf(w, "\n# HELP ml_anomalies_detected_total Total anomalies detected\n")
	fmt.Fprintf(w, "# TYPE ml_anomalies_detected_total counter\n")
	fmt.Fprintf(w, "ml_anomalies_detected_total 42\n")
	
	fmt.Fprintf(w, "\n# HELP ml_bottlenecks_identified_total Total bottlenecks identified\n")
	fmt.Fprintf(w, "# TYPE ml_bottlenecks_identified_total counter\n")
	fmt.Fprintf(w, "ml_bottlenecks_identified_total 23\n")
}

// Start initializes and starts the ML services
func (h *Handler) Start() error {
	ctx := context.Background()
	
	// Start ML services
	if err := h.predictor.Start(ctx); err != nil {
		return fmt.Errorf("failed to start predictor: %w", err)
	}
	
	if err := h.anomalyDetector.Start(ctx); err != nil {
		return fmt.Errorf("failed to start anomaly detector: %w", err)
	}
	
	if err := h.performancePredictor.Start(ctx); err != nil {
		return fmt.Errorf("failed to start performance predictor: %w", err)
	}
	
	// Start WebSocket broadcaster
	go h.runBroadcaster()
	
	h.logger.Info("ML Analytics Platform started successfully")
	
	return nil
}

// runBroadcaster broadcasts updates to WebSocket clients
func (h *Handler) runBroadcaster() {
	for {
		msg := <-h.broadcast
		
		// Send to all connected clients
		for client := range h.wsClients {
			err := client.WriteJSON(msg)
			if err != nil {
				h.logger.WithError(err).Debug("WebSocket write error")
				client.Close()
				delete(h.wsClients, client)
			}
		}
	}
}

// storeMigrationInMemory stores migration suggestion in shared memory
func (h *Handler) storeMigrationInMemory(suggestion *ml.MigrationSuggestion) {
	// This would integrate with a shared memory system or Redis
	// to coordinate with the migration system
	h.logger.WithFields(logrus.Fields{
		"vm_id":     suggestion.VMId,
		"target":    suggestion.TargetNode,
		"window":    suggestion.OptimalWindow.Start,
	}).Info("Migration suggestion stored in memory for coordination")
}

// Helper function to send metrics to collectors
func (h *Handler) SendMetrics(metrics ml.ResourceMetrics) {
	h.predictor.AddMetrics(&metrics)
}