package quotas

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// Simplified dashboard API without external dependencies

// SimpleDashboardAPI provides basic HTTP endpoints for quota management
type SimpleDashboardAPI struct {
	manager *Manager
}

// NewSimpleDashboardAPI creates a new simple dashboard API
func NewSimpleDashboardAPI(manager *Manager) *SimpleDashboardAPI {
	return &SimpleDashboardAPI{
		manager: manager,
	}
}

// Handle dashboard overview
func (api *SimpleDashboardAPI) HandleDashboard(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	ctx := r.Context()
	
	entityID := r.URL.Query().Get("entity_id")
	if entityID == "" {
		api.writeErrorResponse(w, "entity_id parameter is required", http.StatusBadRequest)
		return
	}
	
	// Generate overview
	overview, err := api.generateOverview(ctx, entityID)
	if err != nil {
		api.writeErrorResponse(w, fmt.Sprintf("Failed to generate overview: %v", err), http.StatusInternalServerError)
		return
	}
	
	// Get utilization
	utilization, err := api.manager.GetQuotaUtilization(ctx, entityID)
	if err != nil {
		api.writeErrorResponse(w, fmt.Sprintf("Failed to get utilization: %v", err), http.StatusInternalServerError)
		return
	}
	
	data := map[string]interface{}{
		"overview":     overview,
		"utilization":  utilization,
		"top_consumers": utilization.TopConsumers,
	}
	
	api.writeSuccessResponse(w, data, &MetaInfo{
		Timestamp:   time.Now(),
		ProcessTime: time.Since(start).String(),
	})
}

// Handle quota listing
func (api *SimpleDashboardAPI) HandleListQuotas(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	ctx := r.Context()
	
	// Parse query parameters
	filter := QuotaFilter{
		EntityID:     r.URL.Query().Get("entity_id"),
		ResourceType: ResourceType(r.URL.Query().Get("resource_type")),
		Status:       QuotaStatus(r.URL.Query().Get("status")),
		Level:        QuotaLevel(r.URL.Query().Get("level")),
	}
	
	quotas, err := api.manager.ListQuotas(ctx, filter)
	if err != nil {
		api.writeErrorResponse(w, fmt.Sprintf("Failed to list quotas: %v", err), http.StatusInternalServerError)
		return
	}
	
	api.writeSuccessResponse(w, quotas, &MetaInfo{
		Timestamp:   time.Now(),
		ProcessTime: time.Since(start).String(),
		Count:       len(quotas),
	})
}

// Handle quota creation
func (api *SimpleDashboardAPI) HandleCreateQuota(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	ctx := r.Context()
	
	var quota Quota
	if err := json.NewDecoder(r.Body).Decode(&quota); err != nil {
		api.writeErrorResponse(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}
	
	if err := api.manager.CreateQuota(ctx, &quota); err != nil {
		api.writeErrorResponse(w, fmt.Sprintf("Failed to create quota: %v", err), http.StatusInternalServerError)
		return
	}
	
	api.writeSuccessResponse(w, quota, &MetaInfo{
		Timestamp:   time.Now(),
		ProcessTime: time.Since(start).String(),
	})
}

// Helper methods

func (api *SimpleDashboardAPI) generateOverview(ctx context.Context, entityID string) (*QuotaOverview, error) {
	quotas, err := api.manager.ListQuotas(ctx, QuotaFilter{EntityID: entityID})
	if err != nil {
		return nil, err
	}
	
	overview := &QuotaOverview{}
	overview.TotalQuotas = len(quotas)
	
	var totalUtilization float64
	var validQuotas int
	
	for _, quota := range quotas {
		switch quota.Status {
		case QuotaStatusActive:
			overview.ActiveQuotas++
		case QuotaStatusExceeded:
			overview.ExceededQuotas++
		case QuotaStatusSuspended:
			overview.SuspendedQuotas++
		}
		
		if quota.Limit > 0 {
			utilization := float64(quota.Used) / float64(quota.Limit) * 100
			totalUtilization += utilization
			validQuotas++
		}
	}
	
	if validQuotas > 0 {
		overview.OverallUtilization = totalUtilization / float64(validQuotas)
	}
	
	// Get cost information
	utilization, err := api.manager.GetQuotaUtilization(ctx, entityID)
	if err == nil {
		overview.TotalCost = utilization.TotalCost
		overview.ProjectedMonthlyCost = utilization.ProjectedMonthlyCost
	}
	
	return overview, nil
}

func (api *SimpleDashboardAPI) writeSuccessResponse(w http.ResponseWriter, data interface{}, meta *MetaInfo) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	
	response := DashboardResponse{
		Success: true,
		Data:    data,
		Meta:    meta,
	}
	
	json.NewEncoder(w).Encode(response)
}

func (api *SimpleDashboardAPI) writeErrorResponse(w http.ResponseWriter, message string, statusCode int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	
	response := DashboardResponse{
		Success: false,
		Error:   message,
		Meta: &MetaInfo{
			Timestamp: time.Now(),
		},
	}
	
	json.NewEncoder(w).Encode(response)
}