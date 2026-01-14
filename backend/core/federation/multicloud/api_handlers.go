package multicloud

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/gorilla/mux"
)

// APIHandlers provides HTTP handlers for the multi-cloud API
type APIHandlers struct {
	orchestrator *UnifiedOrchestrator
}

// NewAPIHandlers creates new API handlers
func NewAPIHandlers(orchestrator *UnifiedOrchestrator) *APIHandlers {
	return &APIHandlers{
		orchestrator: orchestrator,
	}
}

// RegisterRoutes registers all multi-cloud API routes
func (h *APIHandlers) RegisterRoutes(router *mux.Router) {
	// Provider management
	router.HandleFunc("/multicloud/providers", h.ListProviders).Methods("GET")
	router.HandleFunc("/multicloud/providers", h.RegisterProvider).Methods("POST")
	router.HandleFunc("/multicloud/providers/{providerId}", h.GetProvider).Methods("GET")
	router.HandleFunc("/multicloud/providers/{providerId}", h.UpdateProvider).Methods("PUT")
	router.HandleFunc("/multicloud/providers/{providerId}", h.UnregisterProvider).Methods("DELETE")
	router.HandleFunc("/multicloud/providers/{providerId}/health", h.GetProviderHealth).Methods("GET")
	router.HandleFunc("/multicloud/providers/{providerId}/metrics", h.GetProviderMetrics).Methods("GET")

	// VM management
	router.HandleFunc("/multicloud/vms", h.ListVMs).Methods("GET")
	router.HandleFunc("/multicloud/vms", h.CreateVM).Methods("POST")
	router.HandleFunc("/multicloud/vms/{vmId}", h.GetVM).Methods("GET")
	router.HandleFunc("/multicloud/vms/{vmId}", h.UpdateVM).Methods("PUT")
	router.HandleFunc("/multicloud/vms/{vmId}", h.DeleteVM).Methods("DELETE")
	router.HandleFunc("/multicloud/vms/{vmId}/start", h.StartVM).Methods("POST")
	router.HandleFunc("/multicloud/vms/{vmId}/stop", h.StopVM).Methods("POST")
	router.HandleFunc("/multicloud/vms/{vmId}/restart", h.RestartVM).Methods("POST")

	// Migration
	router.HandleFunc("/multicloud/migrations", h.ListMigrations).Methods("GET")
	router.HandleFunc("/multicloud/migrations", h.CreateMigration).Methods("POST")
	router.HandleFunc("/multicloud/migrations/{migrationId}", h.GetMigration).Methods("GET")
	router.HandleFunc("/multicloud/migrations/{migrationId}/status", h.GetMigrationStatus).Methods("GET")

	// Cost optimization
	router.HandleFunc("/multicloud/cost/analysis", h.AnalyzeCosts).Methods("POST")
	router.HandleFunc("/multicloud/cost/optimization", h.OptimizeCosts).Methods("POST")
	router.HandleFunc("/multicloud/cost/forecast", h.GetCostForecast).Methods("POST")

	// Compliance
	router.HandleFunc("/multicloud/compliance/report", h.GetComplianceReport).Methods("POST")
	router.HandleFunc("/multicloud/compliance/dashboard", h.GetComplianceDashboard).Methods("GET")
	router.HandleFunc("/multicloud/compliance/policies", h.ListCompliancePolicies).Methods("GET")
	router.HandleFunc("/multicloud/compliance/policies", h.SetCompliancePolicy).Methods("POST")

	// Policies
	router.HandleFunc("/multicloud/policies", h.ListPolicies).Methods("GET")
	router.HandleFunc("/multicloud/policies", h.SetPolicy).Methods("POST")
	router.HandleFunc("/multicloud/policies/{policyId}", h.GetPolicy).Methods("GET")

	// Resource utilization
	router.HandleFunc("/multicloud/resources/utilization", h.GetResourceUtilization).Methods("GET")

	// Dashboard
	router.HandleFunc("/multicloud/dashboard", h.GetDashboard).Methods("GET")
}

// Provider Management Handlers

func (h *APIHandlers) ListProviders(w http.ResponseWriter, r *http.Request) {
	providers := h.orchestrator.registry.ListProviders()
	
	response := make(map[string]interface{})
	for providerID, provider := range providers {
		response[providerID] = map[string]interface{}{
			"type":         provider.GetProviderType(),
			"name":         provider.GetName(),
			"regions":      provider.GetRegions(),
			"capabilities": provider.GetCapabilities(),
		}
	}

	h.writeJSON(w, http.StatusOK, response)
}

func (h *APIHandlers) RegisterProvider(w http.ResponseWriter, r *http.Request) {
	var request struct {
		ProviderID string                 `json:"provider_id"`
		Type       CloudProviderType      `json:"type"`
		Config     CloudProviderConfig    `json:"config"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		h.writeError(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	// Create provider instance based on type
	var provider CloudProvider
	switch request.Type {
	case ProviderAWS:
		// Would create AWS provider
		h.writeError(w, http.StatusNotImplemented, "AWS provider not implemented", nil)
		return
	case ProviderAzure:
		// Would create Azure provider
		h.writeError(w, http.StatusNotImplemented, "Azure provider not implemented", nil)
		return
	case ProviderGCP:
		// Would create GCP provider
		h.writeError(w, http.StatusNotImplemented, "GCP provider not implemented", nil)
		return
	default:
		h.writeError(w, http.StatusBadRequest, fmt.Sprintf("Unsupported provider type: %s", request.Type), nil)
		return
	}

	if err := h.orchestrator.RegisterCloudProvider(request.ProviderID, provider, &request.Config); err != nil {
		h.writeError(w, http.StatusInternalServerError, "Failed to register provider", err)
		return
	}

	h.writeJSON(w, http.StatusCreated, map[string]string{
		"message":     "Provider registered successfully",
		"provider_id": request.ProviderID,
	})
}

func (h *APIHandlers) GetProvider(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	providerID := vars["providerId"]

	provider, err := h.orchestrator.registry.GetProvider(providerID)
	if err != nil {
		h.writeError(w, http.StatusNotFound, "Provider not found", err)
		return
	}

	config, _ := h.orchestrator.registry.GetProviderConfig(providerID)

	response := map[string]interface{}{
		"provider_id":  providerID,
		"type":         provider.GetProviderType(),
		"name":         provider.GetName(),
		"regions":      provider.GetRegions(),
		"capabilities": provider.GetCapabilities(),
		"config":       config,
	}

	h.writeJSON(w, http.StatusOK, response)
}

func (h *APIHandlers) UpdateProvider(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	providerID := vars["providerId"]

	var config CloudProviderConfig
	if err := json.NewDecoder(r.Body).Decode(&config); err != nil {
		h.writeError(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	if err := h.orchestrator.registry.UpdateProviderConfig(providerID, &config); err != nil {
		h.writeError(w, http.StatusInternalServerError, "Failed to update provider", err)
		return
	}

	h.writeJSON(w, http.StatusOK, map[string]string{
		"message": "Provider updated successfully",
	})
}

func (h *APIHandlers) UnregisterProvider(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	providerID := vars["providerId"]

	if err := h.orchestrator.UnregisterCloudProvider(providerID); err != nil {
		h.writeError(w, http.StatusInternalServerError, "Failed to unregister provider", err)
		return
	}

	h.writeJSON(w, http.StatusOK, map[string]string{
		"message": "Provider unregistered successfully",
	})
}

func (h *APIHandlers) GetProviderHealth(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	providerID := vars["providerId"]

	health, err := h.orchestrator.registry.CheckProviderHealth(providerID)
	if err != nil {
		h.writeError(w, http.StatusInternalServerError, "Failed to get provider health", err)
		return
	}

	h.writeJSON(w, http.StatusOK, health)
}

func (h *APIHandlers) GetProviderMetrics(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	providerID := vars["providerId"]

	metrics, err := h.orchestrator.registry.GetProviderMetrics(providerID)
	if err != nil {
		h.writeError(w, http.StatusNotFound, "Provider metrics not found", err)
		return
	}

	h.writeJSON(w, http.StatusOK, metrics)
}

// VM Management Handlers

func (h *APIHandlers) ListVMs(w http.ResponseWriter, r *http.Request) {
	filters := &UnifiedVMFilters{
		ProviderID:   r.URL.Query().Get("provider_id"),
		ProviderType: r.URL.Query().Get("provider_type"),
		Region:       r.URL.Query().Get("region"),
		State:        r.URL.Query().Get("state"),
		NamePattern:  r.URL.Query().Get("name_pattern"),
	}

	vms, err := h.orchestrator.ListVMs(r.Context(), filters)
	if err != nil {
		h.writeError(w, http.StatusInternalServerError, "Failed to list VMs", err)
		return
	}

	h.writeJSON(w, http.StatusOK, map[string]interface{}{
		"vms":   vms,
		"count": len(vms),
	})
}

func (h *APIHandlers) CreateVM(w http.ResponseWriter, r *http.Request) {
	var request UnifiedVMRequest
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		h.writeError(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	vm, err := h.orchestrator.CreateVM(r.Context(), &request)
	if err != nil {
		h.writeError(w, http.StatusInternalServerError, "Failed to create VM", err)
		return
	}

	h.writeJSON(w, http.StatusCreated, vm)
}

func (h *APIHandlers) GetVM(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	vmID := vars["vmId"]

	vm, err := h.orchestrator.GetVM(r.Context(), vmID)
	if err != nil {
		h.writeError(w, http.StatusNotFound, "VM not found", err)
		return
	}

	h.writeJSON(w, http.StatusOK, vm)
}

func (h *APIHandlers) UpdateVM(w http.ResponseWriter, r *http.Request) {
	h.writeError(w, http.StatusNotImplemented, "VM update not implemented", nil)
}

func (h *APIHandlers) DeleteVM(w http.ResponseWriter, r *http.Request) {
	h.writeError(w, http.StatusNotImplemented, "VM deletion not implemented", nil)
}

func (h *APIHandlers) StartVM(w http.ResponseWriter, r *http.Request) {
	h.writeError(w, http.StatusNotImplemented, "VM start not implemented", nil)
}

func (h *APIHandlers) StopVM(w http.ResponseWriter, r *http.Request) {
	h.writeError(w, http.StatusNotImplemented, "VM stop not implemented", nil)
}

func (h *APIHandlers) RestartVM(w http.ResponseWriter, r *http.Request) {
	h.writeError(w, http.StatusNotImplemented, "VM restart not implemented", nil)
}

// Migration Handlers

func (h *APIHandlers) ListMigrations(w http.ResponseWriter, r *http.Request) {
	h.writeJSON(w, http.StatusOK, map[string]interface{}{
		"migrations": []interface{}{},
		"count":      0,
	})
}

func (h *APIHandlers) CreateMigration(w http.ResponseWriter, r *http.Request) {
	var request CrossCloudMigrationRequest
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		h.writeError(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	request.MigrationID = fmt.Sprintf("migration-%d", time.Now().Unix())

	status, err := h.orchestrator.MigrateVM(r.Context(), &request)
	if err != nil {
		h.writeError(w, http.StatusInternalServerError, "Failed to start migration", err)
		return
	}

	h.writeJSON(w, http.StatusCreated, status)
}

func (h *APIHandlers) GetMigration(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	migrationID := vars["migrationId"]

	// Mock response for now
	h.writeJSON(w, http.StatusOK, map[string]interface{}{
		"migration_id": migrationID,
		"status":       "in_progress",
		"progress":     45,
	})
}

func (h *APIHandlers) GetMigrationStatus(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	migrationID := vars["migrationId"]

	// Mock response for now
	h.writeJSON(w, http.StatusOK, map[string]interface{}{
		"migration_id": migrationID,
		"status":       "in_progress",
		"progress":     45,
	})
}

// Cost Optimization Handlers

func (h *APIHandlers) AnalyzeCosts(w http.ResponseWriter, r *http.Request) {
	var request CostAnalysisRequest
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		h.writeError(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	analysis, err := h.orchestrator.GetCostAnalysis(r.Context(), &request)
	if err != nil {
		h.writeError(w, http.StatusInternalServerError, "Failed to analyze costs", err)
		return
	}

	h.writeJSON(w, http.StatusOK, analysis)
}

func (h *APIHandlers) OptimizeCosts(w http.ResponseWriter, r *http.Request) {
	var request CostOptimizationRequest
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		h.writeError(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	plan, err := h.orchestrator.OptimizeCosts(r.Context(), &request)
	if err != nil {
		h.writeError(w, http.StatusInternalServerError, "Failed to generate optimization plan", err)
		return
	}

	h.writeJSON(w, http.StatusOK, plan)
}

func (h *APIHandlers) GetCostForecast(w http.ResponseWriter, r *http.Request) {
	var request CostForecastRequest
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		h.writeError(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	forecast, err := h.orchestrator.costOptimizer.GetCostForecast(r.Context(), &request)
	if err != nil {
		h.writeError(w, http.StatusInternalServerError, "Failed to generate cost forecast", err)
		return
	}

	h.writeJSON(w, http.StatusOK, forecast)
}

// Compliance Handlers

func (h *APIHandlers) GetComplianceReport(w http.ResponseWriter, r *http.Request) {
	var request struct {
		Frameworks []string `json:"frameworks"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		h.writeError(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	report, err := h.orchestrator.GetComplianceReport(r.Context(), request.Frameworks)
	if err != nil {
		h.writeError(w, http.StatusInternalServerError, "Failed to generate compliance report", err)
		return
	}

	h.writeJSON(w, http.StatusOK, report)
}

func (h *APIHandlers) GetComplianceDashboard(w http.ResponseWriter, r *http.Request) {
	dashboard, err := h.orchestrator.complianceEngine.GetComplianceDashboard(r.Context())
	if err != nil {
		h.writeError(w, http.StatusInternalServerError, "Failed to get compliance dashboard", err)
		return
	}

	h.writeJSON(w, http.StatusOK, dashboard)
}

func (h *APIHandlers) ListCompliancePolicies(w http.ResponseWriter, r *http.Request) {
	// Mock response for now
	h.writeJSON(w, http.StatusOK, map[string]interface{}{
		"policies": []interface{}{},
		"count":    0,
	})
}

func (h *APIHandlers) SetCompliancePolicy(w http.ResponseWriter, r *http.Request) {
	var policy CompliancePolicy
	if err := json.NewDecoder(r.Body).Decode(&policy); err != nil {
		h.writeError(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	if err := h.orchestrator.complianceEngine.SetCompliancePolicy(&policy); err != nil {
		h.writeError(w, http.StatusInternalServerError, "Failed to set compliance policy", err)
		return
	}

	h.writeJSON(w, http.StatusCreated, map[string]string{
		"message":   "Compliance policy set successfully",
		"policy_id": policy.ID,
	})
}

// Policy Handlers

func (h *APIHandlers) ListPolicies(w http.ResponseWriter, r *http.Request) {
	policies := h.orchestrator.policyEngine.ListPolicies()
	h.writeJSON(w, http.StatusOK, map[string]interface{}{
		"policies": policies,
		"count":    len(policies),
	})
}

func (h *APIHandlers) SetPolicy(w http.ResponseWriter, r *http.Request) {
	var policy MultiCloudPolicy
	if err := json.NewDecoder(r.Body).Decode(&policy); err != nil {
		h.writeError(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	if err := h.orchestrator.SetMultiCloudPolicy(r.Context(), &policy); err != nil {
		h.writeError(w, http.StatusInternalServerError, "Failed to set policy", err)
		return
	}

	h.writeJSON(w, http.StatusCreated, map[string]string{
		"message":   "Policy set successfully",
		"policy_id": policy.ID,
	})
}

func (h *APIHandlers) GetPolicy(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	policyID := vars["policyId"]

	policy, err := h.orchestrator.policyEngine.GetPolicy(policyID)
	if err != nil {
		h.writeError(w, http.StatusNotFound, "Policy not found", err)
		return
	}

	h.writeJSON(w, http.StatusOK, policy)
}

// Resource Utilization Handlers

func (h *APIHandlers) GetResourceUtilization(w http.ResponseWriter, r *http.Request) {
	utilization, err := h.orchestrator.GetResourceUtilization(r.Context())
	if err != nil {
		h.writeError(w, http.StatusInternalServerError, "Failed to get resource utilization", err)
		return
	}

	h.writeJSON(w, http.StatusOK, utilization)
}

// Dashboard Handler

func (h *APIHandlers) GetDashboard(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	// Get dashboard data in parallel
	type dashboardData struct {
		Providers   map[string]*ProviderHealthStatus    `json:"providers"`
		Metrics     map[string]*ProviderMetrics         `json:"metrics"`
		Utilization *MultiCloudResourceUtilization      `json:"utilization"`
		Compliance  *ComplianceDashboard               `json:"compliance"`
	}

	data := &dashboardData{}
	
	// Get provider health and metrics
	data.Providers = h.orchestrator.GetProviderHealth(ctx)
	data.Metrics = h.orchestrator.GetProviderMetrics(ctx)

	// Get resource utilization
	if utilization, err := h.orchestrator.GetResourceUtilization(ctx); err == nil {
		data.Utilization = utilization
	}

	// Get compliance dashboard
	if compliance, err := h.orchestrator.complianceEngine.GetComplianceDashboard(ctx); err == nil {
		data.Compliance = compliance
	}

	h.writeJSON(w, http.StatusOK, data)
}

// Helper methods

func (h *APIHandlers) writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	
	if err := json.NewEncoder(w).Encode(data); err != nil {
		http.Error(w, "Failed to encode JSON response", http.StatusInternalServerError)
	}
}

func (h *APIHandlers) writeError(w http.ResponseWriter, status int, message string, err error) {
	errorResponse := map[string]interface{}{
		"error":   message,
		"status":  status,
		"timestamp": time.Now().Format(time.RFC3339),
	}

	if err != nil {
		errorResponse["details"] = err.Error()
	}

	h.writeJSON(w, status, errorResponse)
}

func (h *APIHandlers) getQueryParamInt(r *http.Request, param string, defaultValue int) int {
	if value := r.URL.Query().Get(param); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func (h *APIHandlers) getQueryParamBool(r *http.Request, param string, defaultValue bool) bool {
	if value := r.URL.Query().Get(param); value != "" {
		if boolValue, err := strconv.ParseBool(value); err == nil {
			return boolValue
		}
	}
	return defaultValue
}