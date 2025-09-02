//go:build experimental

package orchestration


import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/gorilla/mux"
	"github.com/sirupsen/logrus"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/autoscaling"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/healing"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/policy"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/placement"
)

// OrchestrationAPI provides HTTP handlers for orchestration endpoints
type OrchestrationAPI struct {
	logger            *logrus.Logger
	engine            orchestration.OrchestrationEngine
	autoScaler        autoscaling.AutoScaler
	healingController healing.HealingController
	policyEngine      policy.PolicyEngine
	placementEngine   placement.PlacementEngine
	// WebSocket manager for real-time events (optional)
	webSocketManager  *WebSocketManager
}

// NewOrchestrationAPI creates a new orchestration API handler
func NewOrchestrationAPI(
	logger *logrus.Logger,
	engine orchestration.OrchestrationEngine,
	autoScaler autoscaling.AutoScaler,
	healingController healing.HealingController,
	policyEngine policy.PolicyEngine,
	placementEngine placement.PlacementEngine,
) *OrchestrationAPI {
	return &OrchestrationAPI{
		logger:            logger,
		engine:            engine,
		autoScaler:        autoScaler,
		healingController: healingController,
		policyEngine:      policyEngine,
		placementEngine:   placementEngine,
	}
}

// NewOrchestrationAPIWithWebSocket creates orchestration API with WebSocket support
func NewOrchestrationAPIWithWebSocket(
	logger *logrus.Logger,
	engine orchestration.OrchestrationEngine,
	autoScaler autoscaling.AutoScaler,
	healingController healing.HealingController,
	policyEngine policy.PolicyEngine,
	placementEngine placement.PlacementEngine,
	webSocketManager *WebSocketManager,
) *OrchestrationAPI {
	return &OrchestrationAPI{
		logger:            logger,
		engine:            engine,
		autoScaler:        autoScaler,
		healingController: healingController,
		policyEngine:      policyEngine,
		placementEngine:   placementEngine,
		webSocketManager:  webSocketManager,
	}
}

// RegisterRoutes registers all orchestration API routes
func (api *OrchestrationAPI) RegisterRoutes(router *mux.Router) {
	// Orchestration engine routes
	router.HandleFunc("/orchestration/status", api.GetOrchestrationStatus).Methods("GET")

	// WebSocket routes (if WebSocket manager is available)
	if api.webSocketManager != nil {
		router.HandleFunc("/ws/orchestration", api.HandleWebSocketConnection).Methods("GET")
		router.HandleFunc("/orchestration/websocket/stats", api.GetWebSocketStats).Methods("GET")
	}

	// Placement routes
	router.HandleFunc("/orchestration/placement", api.CreatePlacementRequest).Methods("POST")
	router.HandleFunc("/orchestration/placement/{id}", api.GetPlacementDecision).Methods("GET")

	// Auto-scaling routes
	router.HandleFunc("/orchestration/autoscaling/status", api.GetAutoScalingStatus).Methods("GET")
	router.HandleFunc("/orchestration/autoscaling/targets", api.ListAutoScalingTargets).Methods("GET")
	router.HandleFunc("/orchestration/autoscaling/targets", api.CreateAutoScalingTarget).Methods("POST")
	router.HandleFunc("/orchestration/autoscaling/targets/{id}", api.GetAutoScalingTarget).Methods("GET")
	router.HandleFunc("/orchestration/autoscaling/targets/{id}", api.UpdateAutoScalingTarget).Methods("PUT")
	router.HandleFunc("/orchestration/autoscaling/targets/{id}", api.DeleteAutoScalingTarget).Methods("DELETE")
	router.HandleFunc("/orchestration/autoscaling/targets/{id}/decision", api.GetScalingDecision).Methods("GET")
	router.HandleFunc("/orchestration/autoscaling/targets/{id}/prediction", api.GetScalingPrediction).Methods("GET")

	// Healing routes
	router.HandleFunc("/orchestration/healing/status", api.GetHealingStatus).Methods("GET")
	router.HandleFunc("/orchestration/healing/targets", api.ListHealingTargets).Methods("GET")
	router.HandleFunc("/orchestration/healing/targets", api.CreateHealingTarget).Methods("POST")
	router.HandleFunc("/orchestration/healing/targets/{id}", api.GetHealingTarget).Methods("GET")
	router.HandleFunc("/orchestration/healing/targets/{id}", api.UpdateHealingTarget).Methods("PUT")
	router.HandleFunc("/orchestration/healing/targets/{id}", api.DeleteHealingTarget).Methods("DELETE")
	router.HandleFunc("/orchestration/healing/targets/{id}/health", api.GetTargetHealth).Methods("GET")
	router.HandleFunc("/orchestration/healing/targets/{id}/heal", api.TriggerHealing).Methods("POST")
	router.HandleFunc("/orchestration/healing/targets/{id}/history", api.GetHealingHistory).Methods("GET")

	// Policy routes
	router.HandleFunc("/orchestration/policies", api.ListPolicies).Methods("GET")
	router.HandleFunc("/orchestration/policies", api.CreatePolicy).Methods("POST")
	router.HandleFunc("/orchestration/policies/{id}", api.GetPolicy).Methods("GET")
	router.HandleFunc("/orchestration/policies/{id}", api.UpdatePolicy).Methods("PUT")
	router.HandleFunc("/orchestration/policies/{id}", api.DeletePolicy).Methods("DELETE")
	router.HandleFunc("/orchestration/policies/{id}/evaluate", api.EvaluatePolicy).Methods("POST")
	router.HandleFunc("/orchestration/policies/evaluate-all", api.EvaluateAllPolicies).Methods("POST")
	router.HandleFunc("/orchestration/policies/{id}/validate", api.ValidatePolicy).Methods("POST")
}

// Orchestration Engine Handlers

// GetOrchestrationStatus returns the status of the orchestration engine
func (api *OrchestrationAPI) GetOrchestrationStatus(w http.ResponseWriter, r *http.Request) {
	status := api.engine.GetStatus()
	api.writeJSONResponse(w, http.StatusOK, status)
}

// Placement Handlers

// CreatePlacementRequest creates a new placement request
func (api *OrchestrationAPI) CreatePlacementRequest(w http.ResponseWriter, r *http.Request) {
	var request placement.PlacementRequest
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		api.writeErrorResponse(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	decision, err := api.placementEngine.PlaceVM(r.Context(), &request)
	if err != nil {
		api.writeErrorResponse(w, http.StatusInternalServerError, "Placement failed", err)
		return
	}

	api.writeJSONResponse(w, http.StatusOK, decision)
}

// GetPlacementDecision gets a placement decision by ID
func (api *OrchestrationAPI) GetPlacementDecision(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	// This would typically retrieve from a store
	// For now, return a mock response
	api.writeErrorResponse(w, http.StatusNotImplemented, "Not implemented", nil)
}

// Auto-scaling Handlers

// GetAutoScalingStatus returns the status of the auto-scaler
func (api *OrchestrationAPI) GetAutoScalingStatus(w http.ResponseWriter, r *http.Request) {
	status := api.autoScaler.GetStatus()
	api.writeJSONResponse(w, http.StatusOK, status)
}

// ListAutoScalingTargets lists all auto-scaling targets
func (api *OrchestrationAPI) ListAutoScalingTargets(w http.ResponseWriter, r *http.Request) {
	targets := api.autoScaler.GetTargets()

	// Convert map to slice for JSON response
	targetList := make([]*autoscaling.AutoScalerTarget, 0, len(targets))
	for _, target := range targets {
		targetList = append(targetList, target)
	}

	api.writeJSONResponse(w, http.StatusOK, map[string]interface{}{
		"targets": targetList,
		"count":   len(targetList),
	})
}

// CreateAutoScalingTarget creates a new auto-scaling target
func (api *OrchestrationAPI) CreateAutoScalingTarget(w http.ResponseWriter, r *http.Request) {
	var target autoscaling.AutoScalerTarget
	if err := json.NewDecoder(r.Body).Decode(&target); err != nil {
		api.writeErrorResponse(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	if err := api.autoScaler.AddTarget(&target); err != nil {
		api.writeErrorResponse(w, http.StatusInternalServerError, "Failed to create target", err)
		return
	}

	api.writeJSONResponse(w, http.StatusCreated, target)
}

// GetAutoScalingTarget gets an auto-scaling target by ID
func (api *OrchestrationAPI) GetAutoScalingTarget(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	targets := api.autoScaler.GetTargets()
	target, exists := targets[id]
	if !exists {
		api.writeErrorResponse(w, http.StatusNotFound, "Target not found", nil)
		return
	}

	api.writeJSONResponse(w, http.StatusOK, target)
}

// UpdateAutoScalingTarget updates an auto-scaling target
func (api *OrchestrationAPI) UpdateAutoScalingTarget(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	var target autoscaling.AutoScalerTarget
	if err := json.NewDecoder(r.Body).Decode(&target); err != nil {
		api.writeErrorResponse(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	target.ID = id
	target.UpdatedAt = time.Now()

	// Remove old target and add updated one
	if err := api.autoScaler.RemoveTarget(id); err != nil {
		api.writeErrorResponse(w, http.StatusNotFound, "Target not found", err)
		return
	}

	if err := api.autoScaler.AddTarget(&target); err != nil {
		api.writeErrorResponse(w, http.StatusInternalServerError, "Failed to update target", err)
		return
	}

	api.writeJSONResponse(w, http.StatusOK, target)
}

// DeleteAutoScalingTarget deletes an auto-scaling target
func (api *OrchestrationAPI) DeleteAutoScalingTarget(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if err := api.autoScaler.RemoveTarget(id); err != nil {
		api.writeErrorResponse(w, http.StatusNotFound, "Target not found", err)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

// GetScalingDecision gets a scaling decision for a target
func (api *OrchestrationAPI) GetScalingDecision(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	decision, err := api.autoScaler.GetScalingDecision(id)
	if err != nil {
		api.writeErrorResponse(w, http.StatusInternalServerError, "Failed to get scaling decision", err)
		return
	}

	api.writeJSONResponse(w, http.StatusOK, decision)
}

// GetScalingPrediction gets a scaling prediction for a target
func (api *OrchestrationAPI) GetScalingPrediction(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	horizonStr := r.URL.Query().Get("horizon")
	horizon := 30 // Default 30 minutes
	if horizonStr != "" {
		if h, err := strconv.Atoi(horizonStr); err == nil && h > 0 {
			horizon = h
		}
	}

	prediction, err := api.autoScaler.GetPrediction(id, horizon)
	if err != nil {
		api.writeErrorResponse(w, http.StatusInternalServerError, "Failed to get prediction", err)
		return
	}

	api.writeJSONResponse(w, http.StatusOK, prediction)
}

// Healing Handlers

// GetHealingStatus returns the status of the healing controller
func (api *OrchestrationAPI) GetHealingStatus(w http.ResponseWriter, r *http.Request) {
	status := api.healingController.GetStatus()
	api.writeJSONResponse(w, http.StatusOK, status)
}

// ListHealingTargets lists all healing targets
func (api *OrchestrationAPI) ListHealingTargets(w http.ResponseWriter, r *http.Request) {
	targets := api.healingController.GetTargets()

	// Convert map to slice for JSON response
	targetList := make([]*healing.HealingTarget, 0, len(targets))
	for _, target := range targets {
		targetList = append(targetList, target)
	}

	api.writeJSONResponse(w, http.StatusOK, map[string]interface{}{
		"targets": targetList,
		"count":   len(targetList),
	})
}

// CreateHealingTarget creates a new healing target
func (api *OrchestrationAPI) CreateHealingTarget(w http.ResponseWriter, r *http.Request) {
	var target healing.HealingTarget
	if err := json.NewDecoder(r.Body).Decode(&target); err != nil {
		api.writeErrorResponse(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	if err := api.healingController.RegisterTarget(&target); err != nil {
		api.writeErrorResponse(w, http.StatusInternalServerError, "Failed to create target", err)
		return
	}

	api.writeJSONResponse(w, http.StatusCreated, target)
}

// GetHealingTarget gets a healing target by ID
func (api *OrchestrationAPI) GetHealingTarget(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	targets := api.healingController.GetTargets()
	target, exists := targets[id]
	if !exists {
		api.writeErrorResponse(w, http.StatusNotFound, "Target not found", nil)
		return
	}

	api.writeJSONResponse(w, http.StatusOK, target)
}

// UpdateHealingTarget updates a healing target
func (api *OrchestrationAPI) UpdateHealingTarget(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	var target healing.HealingTarget
	if err := json.NewDecoder(r.Body).Decode(&target); err != nil {
		api.writeErrorResponse(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	target.ID = id
	target.UpdatedAt = time.Now()

	// Unregister old target and register updated one
	if err := api.healingController.UnregisterTarget(id); err != nil {
		api.writeErrorResponse(w, http.StatusNotFound, "Target not found", err)
		return
	}

	if err := api.healingController.RegisterTarget(&target); err != nil {
		api.writeErrorResponse(w, http.StatusInternalServerError, "Failed to update target", err)
		return
	}

	api.writeJSONResponse(w, http.StatusOK, target)
}

// DeleteHealingTarget deletes a healing target
func (api *OrchestrationAPI) DeleteHealingTarget(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if err := api.healingController.UnregisterTarget(id); err != nil {
		api.writeErrorResponse(w, http.StatusNotFound, "Target not found", err)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

// GetTargetHealth gets the health status of a target
func (api *OrchestrationAPI) GetTargetHealth(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	health, err := api.healingController.GetHealthStatus(id)
	if err != nil {
		api.writeErrorResponse(w, http.StatusNotFound, "Target not found", err)
		return
	}

	api.writeJSONResponse(w, http.StatusOK, health)
}

// TriggerHealing manually triggers healing for a target
func (api *OrchestrationAPI) TriggerHealing(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	var request struct {
		Reason string `json:"reason"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		api.writeErrorResponse(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	decision, err := api.healingController.TriggerHealing(id, request.Reason)
	if err != nil {
		api.writeErrorResponse(w, http.StatusInternalServerError, "Failed to trigger healing", err)
		return
	}

	api.writeJSONResponse(w, http.StatusOK, decision)
}

// GetHealingHistory gets the healing history for a target
func (api *OrchestrationAPI) GetHealingHistory(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	limitStr := r.URL.Query().Get("limit")
	limit := 10 // Default limit
	if limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 {
			limit = l
		}
	}

	history, err := api.healingController.GetHealingHistory(id, limit)
	if err != nil {
		api.writeErrorResponse(w, http.StatusNotFound, "Target not found", err)
		return
	}

	api.writeJSONResponse(w, http.StatusOK, map[string]interface{}{
		"history": history,
		"count":   len(history),
	})
}

// Policy Handlers

// ListPolicies lists all orchestration policies
func (api *OrchestrationAPI) ListPolicies(w http.ResponseWriter, r *http.Request) {
	filter := &policy.PolicyFilter{}

	// Parse query parameters
	if namespace := r.URL.Query().Get("namespace"); namespace != "" {
		filter.Namespace = namespace
	}

	if enabledStr := r.URL.Query().Get("enabled"); enabledStr != "" {
		if enabled, err := strconv.ParseBool(enabledStr); err == nil {
			filter.Enabled = &enabled
		}
	}

	if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
		if limit, err := strconv.Atoi(limitStr); err == nil && limit > 0 {
			filter.Limit = limit
		}
	}

	if offsetStr := r.URL.Query().Get("offset"); offsetStr != "" {
		if offset, err := strconv.Atoi(offsetStr); err == nil && offset >= 0 {
			filter.Offset = offset
		}
	}

	policies, err := api.policyEngine.ListPolicies(filter)
	if err != nil {
		api.writeErrorResponse(w, http.StatusInternalServerError, "Failed to list policies", err)
		return
	}

	api.writeJSONResponse(w, http.StatusOK, map[string]interface{}{
		"policies": policies,
		"count":    len(policies),
	})
}

// CreatePolicy creates a new orchestration policy
func (api *OrchestrationAPI) CreatePolicy(w http.ResponseWriter, r *http.Request) {
	var policyObj policy.OrchestrationPolicy
	if err := json.NewDecoder(r.Body).Decode(&policyObj); err != nil {
		api.writeErrorResponse(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	if err := api.policyEngine.CreatePolicy(&policyObj); err != nil {
		api.writeErrorResponse(w, http.StatusInternalServerError, "Failed to create policy", err)
		return
	}

	api.writeJSONResponse(w, http.StatusCreated, policyObj)
}

// GetPolicy gets an orchestration policy by ID
func (api *OrchestrationAPI) GetPolicy(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	policyObj, err := api.policyEngine.GetPolicy(id)
	if err != nil {
		api.writeErrorResponse(w, http.StatusNotFound, "Policy not found", err)
		return
	}

	api.writeJSONResponse(w, http.StatusOK, policyObj)
}

// UpdatePolicy updates an orchestration policy
func (api *OrchestrationAPI) UpdatePolicy(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	var policyObj policy.OrchestrationPolicy
	if err := json.NewDecoder(r.Body).Decode(&policyObj); err != nil {
		api.writeErrorResponse(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	if err := api.policyEngine.UpdatePolicy(id, &policyObj); err != nil {
		api.writeErrorResponse(w, http.StatusInternalServerError, "Failed to update policy", err)
		return
	}

	api.writeJSONResponse(w, http.StatusOK, policyObj)
}

// DeletePolicy deletes an orchestration policy
func (api *OrchestrationAPI) DeletePolicy(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if err := api.policyEngine.DeletePolicy(id); err != nil {
		api.writeErrorResponse(w, http.StatusNotFound, "Policy not found", err)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

// EvaluatePolicy evaluates a policy against a given context
func (api *OrchestrationAPI) EvaluatePolicy(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	var context policy.PolicyEvaluationContext
	if err := json.NewDecoder(r.Body).Decode(&context); err != nil {
		api.writeErrorResponse(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	result, err := api.policyEngine.EvaluatePolicy(id, &context)
	if err != nil {
		api.writeErrorResponse(w, http.StatusInternalServerError, "Policy evaluation failed", err)
		return
	}

	api.writeJSONResponse(w, http.StatusOK, result)
}

// EvaluateAllPolicies evaluates all applicable policies against a given context
func (api *OrchestrationAPI) EvaluateAllPolicies(w http.ResponseWriter, r *http.Request) {
	var context policy.PolicyEvaluationContext
	if err := json.NewDecoder(r.Body).Decode(&context); err != nil {
		api.writeErrorResponse(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	results, err := api.policyEngine.EvaluateAllPolicies(&context)
	if err != nil {
		api.writeErrorResponse(w, http.StatusInternalServerError, "Policy evaluation failed", err)
		return
	}

	api.writeJSONResponse(w, http.StatusOK, map[string]interface{}{
		"results": results,
		"count":   len(results),
	})
}

// ValidatePolicy validates a policy
func (api *OrchestrationAPI) ValidatePolicy(w http.ResponseWriter, r *http.Request) {
	var policyObj policy.OrchestrationPolicy
	if err := json.NewDecoder(r.Body).Decode(&policyObj); err != nil {
		api.writeErrorResponse(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	result, err := api.policyEngine.ValidatePolicy(&policyObj)
	if err != nil {
		api.writeErrorResponse(w, http.StatusInternalServerError, "Policy validation failed", err)
		return
	}

	status := http.StatusOK
	if !result.Valid {
		status = http.StatusBadRequest
	}

	api.writeJSONResponse(w, status, result)
}

// Helper methods

func (api *OrchestrationAPI) writeJSONResponse(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)

	if err := json.NewEncoder(w).Encode(data); err != nil {
		api.logger.WithError(err).Error("Failed to encode JSON response")
	}
}

func (api *OrchestrationAPI) writeErrorResponse(w http.ResponseWriter, status int, message string, err error) {
	api.logger.WithError(err).WithFields(logrus.Fields{
		"status":  status,
		"message": message,
	}).Error("API error response")

	response := map[string]interface{}{
		"error":   message,
		"status":  status,
		"timestamp": time.Now().Format(time.RFC3339),
	}

	if err != nil {
		response["details"] = err.Error()
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(response)
}

// WebSocket Handlers

// HandleWebSocketConnection handles secure WebSocket connections
func (api *OrchestrationAPI) HandleWebSocketConnection(w http.ResponseWriter, r *http.Request) {
	if api.webSocketManager == nil {
		api.writeErrorResponse(w, http.StatusServiceUnavailable, "WebSocket service not available", nil)
		return
	}

	api.webSocketManager.HandleWebSocket(w, r)
}

// GetWebSocketStats returns WebSocket connection statistics
func (api *OrchestrationAPI) GetWebSocketStats(w http.ResponseWriter, r *http.Request) {
	if api.webSocketManager == nil {
		api.writeErrorResponse(w, http.StatusServiceUnavailable, "WebSocket service not available", nil)
		return
	}

	stats := api.webSocketManager.GetStats()
	api.writeJSONResponse(w, http.StatusOK, stats)
}