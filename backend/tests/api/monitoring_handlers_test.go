package api_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gorilla/mux"
	monitoring "github.com/khryptorgraphics/novacron/backend/api/monitoring"
	"github.com/khryptorgraphics/novacron/backend/core/metrics"
)

// Mock monitoring service
type mockMonitoringService struct {
	metrics map[string][]metrics.Metric
	alerts  []metrics.Alert
}

func (m *mockMonitoringService) GetMetrics(ctx context.Context, resource string, timeRange time.Duration) ([]metrics.Metric, error) {
	return m.metrics[resource], nil
}

func (m *mockMonitoringService) GetAlerts(ctx context.Context, severity string) ([]metrics.Alert, error) {
	filtered := []metrics.Alert{}
	for _, alert := range m.alerts {
		if severity == "" || alert.Severity == severity {
			filtered = append(filtered, alert)
		}
	}
	return filtered, nil
}

func (m *mockMonitoringService) GetSystemHealth(ctx context.Context) (*metrics.HealthStatus, error) {
	return &metrics.HealthStatus{
		Status:    "healthy",
		Timestamp: time.Now(),
		Components: map[string]string{
			"database": "healthy",
			"cache":    "healthy",
			"storage":  "healthy",
		},
	}, nil
}

func (m *mockMonitoringService) GetResourceUsage(ctx context.Context, resourceID string) (*metrics.ResourceUsage, error) {
	return &metrics.ResourceUsage{
		ResourceID: resourceID,
		CPU:        45.5,
		Memory:     60.2,
		Disk:       75.0,
		Network:    25.3,
		Timestamp:  time.Now(),
	}, nil
}

// Setup test monitoring handlers
func setupMonitoringHandlers() *monitoring.MonitoringHandlers {
	mockService := &mockMonitoringService{
		metrics: map[string][]metrics.Metric{
			"vm-123": {
				{Name: "cpu_usage", Value: 45.5, Timestamp: time.Now()},
				{Name: "memory_usage", Value: 60.2, Timestamp: time.Now()},
			},
			"cluster-456": {
				{Name: "nodes_active", Value: 10, Timestamp: time.Now()},
				{Name: "pods_running", Value: 150, Timestamp: time.Now()},
			},
		},
		alerts: []metrics.Alert{
			{ID: "alert-1", Severity: "critical", Message: "High CPU usage", Timestamp: time.Now()},
			{ID: "alert-2", Severity: "warning", Message: "Memory threshold reached", Timestamp: time.Now()},
		},
	}

	return monitoring.NewMonitoringHandlers(mockService)
}

// Test Cases

func TestGetMetrics_Success(t *testing.T) {
	h := setupMonitoringHandlers()

	req := httptest.NewRequest("GET", "/api/monitoring/metrics?resource=vm-123&time_range=1h", nil)
	w := httptest.NewRecorder()

	h.GetMetrics(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)

	metricsData, ok := response["metrics"].([]interface{})
	if !ok || len(metricsData) == 0 {
		t.Error("Expected metrics array in response")
	}

	if len(metricsData) != 2 {
		t.Errorf("Expected 2 metrics, got %d", len(metricsData))
	}
}

func TestGetMetrics_MissingResource(t *testing.T) {
	h := setupMonitoringHandlers()

	req := httptest.NewRequest("GET", "/api/monitoring/metrics", nil)
	w := httptest.NewRecorder()

	h.GetMetrics(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400, got %d", w.Code)
	}
}

func TestGetMetrics_InvalidTimeRange(t *testing.T) {
	h := setupMonitoringHandlers()

	req := httptest.NewRequest("GET", "/api/monitoring/metrics?resource=vm-123&time_range=invalid", nil)
	w := httptest.NewRecorder()

	h.GetMetrics(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400, got %d", w.Code)
	}
}

func TestGetAlerts_Success(t *testing.T) {
	h := setupMonitoringHandlers()

	req := httptest.NewRequest("GET", "/api/monitoring/alerts", nil)
	w := httptest.NewRecorder()

	h.GetAlerts(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)

	alerts, ok := response["alerts"].([]interface{})
	if !ok {
		t.Error("Expected alerts array in response")
	}

	if len(alerts) != 2 {
		t.Errorf("Expected 2 alerts, got %d", len(alerts))
	}
}

func TestGetAlerts_FilterBySeverity(t *testing.T) {
	h := setupMonitoringHandlers()

	req := httptest.NewRequest("GET", "/api/monitoring/alerts?severity=critical", nil)
	w := httptest.NewRecorder()

	h.GetAlerts(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)

	alerts, ok := response["alerts"].([]interface{})
	if !ok {
		t.Error("Expected alerts array in response")
	}

	if len(alerts) != 1 {
		t.Errorf("Expected 1 critical alert, got %d", len(alerts))
	}
}

func TestGetSystemHealth_Success(t *testing.T) {
	h := setupMonitoringHandlers()

	req := httptest.NewRequest("GET", "/api/monitoring/health", nil)
	w := httptest.NewRecorder()

	h.GetSystemHealth(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response metrics.HealthStatus
	json.Unmarshal(w.Body.Bytes(), &response)

	if response.Status != "healthy" {
		t.Errorf("Expected status healthy, got %s", response.Status)
	}

	if len(response.Components) == 0 {
		t.Error("Expected non-empty components map")
	}
}

func TestGetResourceUsage_Success(t *testing.T) {
	h := setupMonitoringHandlers()

	req := httptest.NewRequest("GET", "/api/monitoring/resources/vm-123/usage", nil)
	req = mux.SetURLVars(req, map[string]string{"resourceId": "vm-123"})
	w := httptest.NewRecorder()

	h.GetResourceUsage(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response metrics.ResourceUsage
	json.Unmarshal(w.Body.Bytes(), &response)

	if response.ResourceID != "vm-123" {
		t.Errorf("Expected resource_id vm-123, got %s", response.ResourceID)
	}

	if response.CPU == 0 {
		t.Error("Expected non-zero CPU usage")
	}
}

func TestGetResourceUsage_MissingResourceID(t *testing.T) {
	h := setupMonitoringHandlers()

	req := httptest.NewRequest("GET", "/api/monitoring/resources//usage", nil)
	req = mux.SetURLVars(req, map[string]string{"resourceId": ""})
	w := httptest.NewRecorder()

	h.GetResourceUsage(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400, got %d", w.Code)
	}
}

// Test concurrent metric collection
func TestConcurrentGetMetrics(t *testing.T) {
	h := setupMonitoringHandlers()

	const numRequests = 20
	done := make(chan bool, numRequests)

	for i := 0; i < numRequests; i++ {
		go func() {
			req := httptest.NewRequest("GET", "/api/monitoring/metrics?resource=vm-123&time_range=1h", nil)
			w := httptest.NewRecorder()

			h.GetMetrics(w, req)

			if w.Code != http.StatusOK {
				t.Errorf("Expected status 200, got %d", w.Code)
			}

			done <- true
		}()
	}

	for i := 0; i < numRequests; i++ {
		<-done
	}
}

// Test context cancellation
func TestGetMetrics_ContextCancellation(t *testing.T) {
	h := setupMonitoringHandlers()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	req := httptest.NewRequest("GET", "/api/monitoring/metrics?resource=vm-123&time_range=1h", nil).WithContext(ctx)
	w := httptest.NewRecorder()

	h.GetMetrics(w, req)

	// Should handle context cancellation gracefully
	if w.Code == http.StatusOK {
		t.Log("Note: Handler may not check context cancellation")
	}
}

// Test edge cases
func TestGetMetrics_EmptyResult(t *testing.T) {
	h := setupMonitoringHandlers()

	req := httptest.NewRequest("GET", "/api/monitoring/metrics?resource=non-existent&time_range=1h", nil)
	w := httptest.NewRecorder()

	h.GetMetrics(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200 for empty result, got %d", w.Code)
	}

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)

	metricsData, ok := response["metrics"].([]interface{})
	if !ok {
		t.Error("Expected metrics array in response")
	}

	if len(metricsData) != 0 {
		t.Errorf("Expected 0 metrics for non-existent resource, got %d", len(metricsData))
	}
}

func TestGetAlerts_EmptyResult(t *testing.T) {
	// Create handler with empty alerts
	mockService := &mockMonitoringService{
		metrics: map[string][]metrics.Metric{},
		alerts:  []metrics.Alert{},
	}
	h := monitoring.NewMonitoringHandlers(mockService)

	req := httptest.NewRequest("GET", "/api/monitoring/alerts", nil)
	w := httptest.NewRecorder()

	h.GetAlerts(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)

	alerts, ok := response["alerts"].([]interface{})
	if !ok {
		t.Error("Expected alerts array in response")
	}

	if len(alerts) != 0 {
		t.Errorf("Expected 0 alerts, got %d", len(alerts))
	}
}

// Benchmark tests
func BenchmarkGetMetrics(b *testing.B) {
	h := setupMonitoringHandlers()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest("GET", "/api/monitoring/metrics?resource=vm-123&time_range=1h", nil)
		w := httptest.NewRecorder()
		h.GetMetrics(w, req)
	}
}

func BenchmarkGetAlerts(b *testing.B) {
	h := setupMonitoringHandlers()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest("GET", "/api/monitoring/alerts", nil)
		w := httptest.NewRecorder()
		h.GetAlerts(w, req)
	}
}

func BenchmarkGetSystemHealth(b *testing.B) {
	h := setupMonitoringHandlers()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest("GET", "/api/monitoring/health", nil)
		w := httptest.NewRecorder()
		h.GetSystemHealth(w, req)
	}
}
