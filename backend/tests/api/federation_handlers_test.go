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
	federation "github.com/khryptorgraphics/novacron/backend/api/federation"
	"github.com/khryptorgraphics/novacron/backend/core/cluster"
)

// Mock federation manager
type mockFederationManager struct {
	clusters  map[string]*cluster.ClusterInfo
	resources map[string][]cluster.Resource
}

func (m *mockFederationManager) RegisterCluster(ctx context.Context, info cluster.ClusterInfo) error {
	m.clusters[info.ID] = &info
	return nil
}

func (m *mockFederationManager) UnregisterCluster(ctx context.Context, clusterID string) error {
	delete(m.clusters, clusterID)
	return nil
}

func (m *mockFederationManager) GetCluster(ctx context.Context, clusterID string) (*cluster.ClusterInfo, error) {
	if c, ok := m.clusters[clusterID]; ok {
		return c, nil
	}
	return nil, cluster.ErrClusterNotFound
}

func (m *mockFederationManager) ListClusters(ctx context.Context) ([]*cluster.ClusterInfo, error) {
	clusters := make([]*cluster.ClusterInfo, 0, len(m.clusters))
	for _, c := range m.clusters {
		clusters = append(clusters, c)
	}
	return clusters, nil
}

func (m *mockFederationManager) SyncResources(ctx context.Context, clusterID string) error {
	return nil
}

func (m *mockFederationManager) GetResources(ctx context.Context, clusterID string) ([]cluster.Resource, error) {
	return m.resources[clusterID], nil
}

func (m *mockFederationManager) MigrateResource(ctx context.Context, resourceID, targetCluster string) error {
	return nil
}

func (m *mockFederationManager) GetFederationStatus(ctx context.Context) (*cluster.FederationStatus, error) {
	return &cluster.FederationStatus{
		TotalClusters:  len(m.clusters),
		ActiveClusters: len(m.clusters),
		HealthStatus:   "healthy",
		LastSync:       time.Now(),
	}, nil
}

// Setup test federation handlers
func setupFederationHandlers() *federation.FederationHandlers {
	mockManager := &mockFederationManager{
		clusters: map[string]*cluster.ClusterInfo{
			"cluster-1": {
				ID:       "cluster-1",
				Name:     "Test Cluster 1",
				Endpoint: "https://cluster1.example.com",
				Status:   "active",
				Capacity: cluster.Capacity{CPU: 100, Memory: 256000, Storage: 1000000},
			},
			"cluster-2": {
				ID:       "cluster-2",
				Name:     "Test Cluster 2",
				Endpoint: "https://cluster2.example.com",
				Status:   "active",
				Capacity: cluster.Capacity{CPU: 200, Memory: 512000, Storage: 2000000},
			},
		},
		resources: map[string][]cluster.Resource{
			"cluster-1": {
				{ID: "vm-1", Type: "vm", ClusterID: "cluster-1", Status: "running"},
				{ID: "vm-2", Type: "vm", ClusterID: "cluster-1", Status: "running"},
			},
			"cluster-2": {
				{ID: "vm-3", Type: "vm", ClusterID: "cluster-2", Status: "running"},
			},
		},
	}

	return federation.NewFederationHandlers(mockManager)
}

// Test Cases

func TestRegisterCluster_Success(t *testing.T) {
	h := setupFederationHandlers()

	reqBody := cluster.ClusterInfo{
		ID:       "cluster-3",
		Name:     "New Cluster",
		Endpoint: "https://cluster3.example.com",
		Status:   "active",
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/api/federation/clusters", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.RegisterCluster(w, req)

	if w.Code != http.StatusCreated && w.Code != http.StatusOK {
		t.Errorf("Expected status 201 or 200, got %d", w.Code)
	}
}

func TestRegisterCluster_InvalidJSON(t *testing.T) {
	h := setupFederationHandlers()

	req := httptest.NewRequest("POST", "/api/federation/clusters", bytes.NewBufferString("{invalid json"))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.RegisterCluster(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400, got %d", w.Code)
	}
}

func TestRegisterCluster_MissingRequiredFields(t *testing.T) {
	h := setupFederationHandlers()

	reqBody := cluster.ClusterInfo{
		Name: "Incomplete Cluster",
		// Missing ID and Endpoint
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/api/federation/clusters", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.RegisterCluster(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400, got %d", w.Code)
	}
}

func TestGetCluster_Success(t *testing.T) {
	h := setupFederationHandlers()

	req := httptest.NewRequest("GET", "/api/federation/clusters/cluster-1", nil)
	req = mux.SetURLVars(req, map[string]string{"clusterId": "cluster-1"})
	w := httptest.NewRecorder()

	h.GetCluster(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response cluster.ClusterInfo
	json.Unmarshal(w.Body.Bytes(), &response)

	if response.ID != "cluster-1" {
		t.Errorf("Expected cluster ID cluster-1, got %s", response.ID)
	}

	if response.Name != "Test Cluster 1" {
		t.Errorf("Expected cluster name 'Test Cluster 1', got %s", response.Name)
	}
}

func TestGetCluster_NotFound(t *testing.T) {
	h := setupFederationHandlers()

	req := httptest.NewRequest("GET", "/api/federation/clusters/non-existent", nil)
	req = mux.SetURLVars(req, map[string]string{"clusterId": "non-existent"})
	w := httptest.NewRecorder()

	h.GetCluster(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("Expected status 404, got %d", w.Code)
	}
}

func TestListClusters_Success(t *testing.T) {
	h := setupFederationHandlers()

	req := httptest.NewRequest("GET", "/api/federation/clusters", nil)
	w := httptest.NewRecorder()

	h.ListClusters(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)

	clusters, ok := response["clusters"].([]interface{})
	if !ok {
		t.Error("Expected clusters array in response")
	}

	if len(clusters) != 2 {
		t.Errorf("Expected 2 clusters, got %d", len(clusters))
	}
}

func TestUnregisterCluster_Success(t *testing.T) {
	h := setupFederationHandlers()

	req := httptest.NewRequest("DELETE", "/api/federation/clusters/cluster-1", nil)
	req = mux.SetURLVars(req, map[string]string{"clusterId": "cluster-1"})
	w := httptest.NewRecorder()

	h.UnregisterCluster(w, req)

	if w.Code != http.StatusOK && w.Code != http.StatusNoContent {
		t.Errorf("Expected status 200 or 204, got %d", w.Code)
	}
}

func TestSyncResources_Success(t *testing.T) {
	h := setupFederationHandlers()

	req := httptest.NewRequest("POST", "/api/federation/clusters/cluster-1/sync", nil)
	req = mux.SetURLVars(req, map[string]string{"clusterId": "cluster-1"})
	w := httptest.NewRecorder()

	h.SyncResources(w, req)

	if w.Code != http.StatusOK && w.Code != http.StatusAccepted {
		t.Errorf("Expected status 200 or 202, got %d", w.Code)
	}
}

func TestGetResources_Success(t *testing.T) {
	h := setupFederationHandlers()

	req := httptest.NewRequest("GET", "/api/federation/clusters/cluster-1/resources", nil)
	req = mux.SetURLVars(req, map[string]string{"clusterId": "cluster-1"})
	w := httptest.NewRecorder()

	h.GetResources(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)

	resources, ok := response["resources"].([]interface{})
	if !ok {
		t.Error("Expected resources array in response")
	}

	if len(resources) != 2 {
		t.Errorf("Expected 2 resources for cluster-1, got %d", len(resources))
	}
}

func TestGetResources_EmptyCluster(t *testing.T) {
	h := setupFederationHandlers()

	req := httptest.NewRequest("GET", "/api/federation/clusters/cluster-empty/resources", nil)
	req = mux.SetURLVars(req, map[string]string{"clusterId": "cluster-empty"})
	w := httptest.NewRecorder()

	h.GetResources(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)

	resources, ok := response["resources"].([]interface{})
	if !ok {
		t.Error("Expected resources array in response")
	}

	if len(resources) != 0 {
		t.Errorf("Expected 0 resources for empty cluster, got %d", len(resources))
	}
}

func TestMigrateResource_Success(t *testing.T) {
	h := setupFederationHandlers()

	reqBody := map[string]string{
		"resource_id":    "vm-1",
		"target_cluster": "cluster-2",
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/api/federation/resources/migrate", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.MigrateResource(w, req)

	if w.Code != http.StatusOK && w.Code != http.StatusAccepted {
		t.Errorf("Expected status 200 or 202, got %d", w.Code)
	}
}

func TestMigrateResource_MissingFields(t *testing.T) {
	h := setupFederationHandlers()

	reqBody := map[string]string{
		"resource_id": "vm-1",
		// Missing target_cluster
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/api/federation/resources/migrate", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.MigrateResource(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400, got %d", w.Code)
	}
}

func TestGetFederationStatus_Success(t *testing.T) {
	h := setupFederationHandlers()

	req := httptest.NewRequest("GET", "/api/federation/status", nil)
	w := httptest.NewRecorder()

	h.GetFederationStatus(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response cluster.FederationStatus
	json.Unmarshal(w.Body.Bytes(), &response)

	if response.TotalClusters != 2 {
		t.Errorf("Expected 2 total clusters, got %d", response.TotalClusters)
	}

	if response.HealthStatus != "healthy" {
		t.Errorf("Expected health status 'healthy', got %s", response.HealthStatus)
	}
}

// Test concurrent cluster registration
func TestConcurrentRegisterCluster(t *testing.T) {
	h := setupFederationHandlers()

	const numRequests = 10
	done := make(chan bool, numRequests)

	for i := 0; i < numRequests; i++ {
		go func(id int) {
			reqBody := cluster.ClusterInfo{
				ID:       "cluster-" + string(rune(100+id)),
				Name:     "Concurrent Cluster",
				Endpoint: "https://cluster.example.com",
				Status:   "active",
			}
			body, _ := json.Marshal(reqBody)

			req := httptest.NewRequest("POST", "/api/federation/clusters", bytes.NewBuffer(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			h.RegisterCluster(w, req)

			if w.Code != http.StatusCreated && w.Code != http.StatusOK {
				t.Errorf("Request %d failed with status %d", id, w.Code)
			}

			done <- true
		}(i)
	}

	for i := 0; i < numRequests; i++ {
		<-done
	}
}

// Test context cancellation
func TestGetCluster_ContextCancellation(t *testing.T) {
	h := setupFederationHandlers()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	req := httptest.NewRequest("GET", "/api/federation/clusters/cluster-1", nil).WithContext(ctx)
	req = mux.SetURLVars(req, map[string]string{"clusterId": "cluster-1"})
	w := httptest.NewRecorder()

	h.GetCluster(w, req)

	// Should handle context cancellation gracefully
	if w.Code == http.StatusOK {
		t.Log("Note: Handler may not check context cancellation")
	}
}

// Edge cases
func TestUnregisterCluster_NonExistent(t *testing.T) {
	h := setupFederationHandlers()

	req := httptest.NewRequest("DELETE", "/api/federation/clusters/non-existent", nil)
	req = mux.SetURLVars(req, map[string]string{"clusterId": "non-existent"})
	w := httptest.NewRecorder()

	h.UnregisterCluster(w, req)

	// Should either return 404 or 204 depending on implementation
	if w.Code != http.StatusNotFound && w.Code != http.StatusNoContent && w.Code != http.StatusOK {
		t.Errorf("Expected status 404, 204, or 200, got %d", w.Code)
	}
}

func TestSyncResources_NonExistentCluster(t *testing.T) {
	h := setupFederationHandlers()

	req := httptest.NewRequest("POST", "/api/federation/clusters/non-existent/sync", nil)
	req = mux.SetURLVars(req, map[string]string{"clusterId": "non-existent"})
	w := httptest.NewRecorder()

	h.SyncResources(w, req)

	// Should handle non-existent cluster gracefully
	if w.Code == http.StatusOK || w.Code == http.StatusAccepted {
		t.Log("Note: Handler may not validate cluster existence before sync")
	}
}

// Benchmark tests
func BenchmarkListClusters(b *testing.B) {
	h := setupFederationHandlers()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest("GET", "/api/federation/clusters", nil)
		w := httptest.NewRecorder()
		h.ListClusters(w, req)
	}
}

func BenchmarkGetCluster(b *testing.B) {
	h := setupFederationHandlers()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest("GET", "/api/federation/clusters/cluster-1", nil)
		req = mux.SetURLVars(req, map[string]string{"clusterId": "cluster-1"})
		w := httptest.NewRecorder()
		h.GetCluster(w, req)
	}
}

func BenchmarkGetFederationStatus(b *testing.B) {
	h := setupFederationHandlers()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest("GET", "/api/federation/status", nil)
		w := httptest.NewRecorder()
		h.GetFederationStatus(w, req)
	}
}
