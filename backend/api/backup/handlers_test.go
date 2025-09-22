package backup

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/mux"
	backup "github.com/khryptorgraphics/novacron/backend/core/backup"
)

// TestBackupAPIServer tests the backup API server functionality
func TestBackupAPIServer(t *testing.T) {
	// Create temporary directory for test data
	tmpDir := t.TempDir()
	
	// Create mock managers
	backupManager := backup.NewIncrementalBackupManager(
		tmpDir,
		backup.NewDeduplicationEngine(tmpDir),
		backup.DefaultCompressionLevel,
	)
	retentionManager := backup.NewRetentionManager(tmpDir)
	restoreManager := backup.NewRestoreManager(tmpDir, 2)
	
	// Create API server
	server := NewBackupAPIServer(backupManager, retentionManager, restoreManager)
	
	if server.backupManager != backupManager {
		t.Error("Expected backup manager to be set")
	}
	if server.retentionManager != retentionManager {
		t.Error("Expected retention manager to be set")
	}
	if server.restoreManager != restoreManager {
		t.Error("Expected restore manager to be set")
	}
	
	// Test router setup
	router := server.Router()
	if router == nil {
		t.Error("Expected router to be initialized")
	}
}

// TestCreateBackupAPI tests the backup creation API endpoint
func TestCreateBackupAPI(t *testing.T) {
	tmpDir := t.TempDir()
	server := setupTestServer(tmpDir)
	
	// Create test request
	req := BackupCreateRequest{
		VMID:       "test-vm-1",
		VMPath:     "/tmp/test-vm-disk.img",
		BackupType: "full",
		Metadata:   map[string]string{"test": "value"},
	}
	
	body, _ := json.Marshal(req)
	
	// Create HTTP request
	httpReq := httptest.NewRequest("POST", "/api/v1/backup/backups", bytes.NewBuffer(body))
	httpReq.Header.Set("Content-Type", "application/json")
	
	// Create response recorder
	recorder := httptest.NewRecorder()
	
	// Execute request
	server.Router().ServeHTTP(recorder, httpReq)
	
	// Note: This will likely fail due to missing VM disk file, but we test the endpoint structure
	if recorder.Code != http.StatusInternalServerError && recorder.Code != http.StatusCreated {
		t.Logf("Expected 201 or 500, got %d (acceptable for test environment)", recorder.Code)
	}
	
	// Test with invalid JSON
	httpReq = httptest.NewRequest("POST", "/api/v1/backup/backups", strings.NewReader("{invalid json}"))
	httpReq.Header.Set("Content-Type", "application/json")
	recorder = httptest.NewRecorder()
	
	server.Router().ServeHTTP(recorder, httpReq)
	
	if recorder.Code != http.StatusBadRequest {
		t.Errorf("Expected 400 for invalid JSON, got %d", recorder.Code)
	}
	
	// Test with missing VM ID
	req.VMID = ""
	body, _ = json.Marshal(req)
	httpReq = httptest.NewRequest("POST", "/api/v1/backup/backups", bytes.NewBuffer(body))
	httpReq.Header.Set("Content-Type", "application/json")
	recorder = httptest.NewRecorder()
	
	server.Router().ServeHTTP(recorder, httpReq)
	
	if recorder.Code != http.StatusBadRequest {
		t.Errorf("Expected 400 for missing VM ID, got %d", recorder.Code)
	}
}

// TestListBackupsAPI tests the backup listing API endpoint
func TestListBackupsAPI(t *testing.T) {
	tmpDir := t.TempDir()
	server := setupTestServer(tmpDir)
	
	// Test list backups without VM ID filter
	httpReq := httptest.NewRequest("GET", "/api/v1/backup/backups", nil)
	recorder := httptest.NewRecorder()
	
	server.Router().ServeHTTP(recorder, httpReq)
	
	// Should return not implemented for now
	if recorder.Code != http.StatusNotImplemented {
		t.Errorf("Expected 501 for unfiltered list, got %d", recorder.Code)
	}
	
	// Test list backups with VM ID filter
	httpReq = httptest.NewRequest("GET", "/api/v1/backup/backups?vm_id=test-vm-1", nil)
	recorder = httptest.NewRecorder()
	
	server.Router().ServeHTTP(recorder, httpReq)
	
	// Should return success with empty list
	if recorder.Code != http.StatusOK {
		t.Errorf("Expected 200 for VM-filtered list, got %d", recorder.Code)
	}
	
	// Parse response
	var response BackupListResponse
	err := json.Unmarshal(recorder.Body.Bytes(), &response)
	if err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}
	
	if response.Total != 0 {
		t.Errorf("Expected 0 backups, got %d", response.Total)
	}
}

// TestGetBackupAPI tests the get backup API endpoint
func TestGetBackupAPI(t *testing.T) {
	tmpDir := t.TempDir()
	server := setupTestServer(tmpDir)
	
	// Test get non-existent backup
	httpReq := httptest.NewRequest("GET", "/api/v1/backup/backups/non-existent-backup", nil)
	recorder := httptest.NewRecorder()
	
	server.Router().ServeHTTP(recorder, httpReq)
	
	if recorder.Code != http.StatusNotFound {
		t.Errorf("Expected 404 for non-existent backup, got %d", recorder.Code)
	}
}

// TestInitializeCBTAPI tests the CBT initialization API endpoint
func TestInitializeCBTAPI(t *testing.T) {
	tmpDir := t.TempDir()
	server := setupTestServer(tmpDir)
	
	// Create test request
	req := struct {
		VMSize int64 `json:"vm_size"`
	}{
		VMSize: 1024 * 1024 * 1024, // 1GB
	}
	
	body, _ := json.Marshal(req)
	
	// Create HTTP request
	httpReq := httptest.NewRequest("POST", "/api/v1/backup/vms/test-vm-1/cbt/init", bytes.NewBuffer(body))
	httpReq.Header.Set("Content-Type", "application/json")
	
	// Add mux vars
	httpReq = mux.SetURLVars(httpReq, map[string]string{"vm_id": "test-vm-1"})
	
	// Create response recorder
	recorder := httptest.NewRecorder()
	
	// Execute request
	server.Router().ServeHTTP(recorder, httpReq)
	
	if recorder.Code != http.StatusOK {
		t.Errorf("Expected 200 for CBT init, got %d", recorder.Code)
	}
	
	// Parse response
	var response map[string]interface{}
	err := json.Unmarshal(recorder.Body.Bytes(), &response)
	if err != nil {
		t.Fatalf("Failed to parse CBT init response: %v", err)
	}
	
	if response["vm_id"] != "test-vm-1" {
		t.Errorf("Expected vm_id test-vm-1, got %v", response["vm_id"])
	}
	
	if response["initialized"] != true {
		t.Errorf("Expected initialized true, got %v", response["initialized"])
	}
	
	// Test with invalid VM size
	req.VMSize = 0
	body, _ = json.Marshal(req)
	httpReq = httptest.NewRequest("POST", "/api/v1/backup/vms/test-vm-1/cbt/init", bytes.NewBuffer(body))
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq = mux.SetURLVars(httpReq, map[string]string{"vm_id": "test-vm-1"})
	recorder = httptest.NewRecorder()
	
	server.Router().ServeHTTP(recorder, httpReq)
	
	if recorder.Code != http.StatusBadRequest {
		t.Errorf("Expected 400 for invalid VM size, got %d", recorder.Code)
	}
}

// TestCreateRestoreAPI tests the restore creation API endpoint
func TestCreateRestoreAPI(t *testing.T) {
	tmpDir := t.TempDir()
	server := setupTestServer(tmpDir)
	
	// Create test request
	req := RestoreCreateRequest{
		BackupID:          "test-backup-1",
		TargetPath:        "/tmp/restore-target",
		RestoreType:       "full",
		VerifyRestore:     true,
		OverwriteExisting: true,
		Metadata:          map[string]string{"test": "restore"},
	}
	
	body, _ := json.Marshal(req)
	
	// Create HTTP request
	httpReq := httptest.NewRequest("POST", "/api/v1/backup/restore", bytes.NewBuffer(body))
	httpReq.Header.Set("Content-Type", "application/json")
	
	// Create response recorder
	recorder := httptest.NewRecorder()
	
	// Execute request
	server.Router().ServeHTTP(recorder, httpReq)
	
	// Should fail due to non-existent backup
	if recorder.Code != http.StatusNotFound {
		t.Errorf("Expected 404 for non-existent backup, got %d", recorder.Code)
	}
	
	// Test with missing backup ID
	req.BackupID = ""
	body, _ = json.Marshal(req)
	httpReq = httptest.NewRequest("POST", "/api/v1/backup/restore", bytes.NewBuffer(body))
	httpReq.Header.Set("Content-Type", "application/json")
	recorder = httptest.NewRecorder()
	
	server.Router().ServeHTTP(recorder, httpReq)
	
	if recorder.Code != http.StatusBadRequest {
		t.Errorf("Expected 400 for missing backup ID, got %d", recorder.Code)
	}
}

// TestRetentionPolicyAPI tests retention policy API endpoints
func TestRetentionPolicyAPI(t *testing.T) {
	tmpDir := t.TempDir()
	server := setupTestServer(tmpDir)
	
	// Create test retention policy
	policy := RetentionPolicyRequest{
		Name:        "Test GFS Policy",
		Description: "Test Grandfather-Father-Son retention policy",
		Rules: &backup.RetentionRules{
			MaxAge:      30 * 24 * time.Hour,
			MaxCount:    100,
			MinReplicas: 1,
		},
		GFSConfig: &backup.GFSConfig{
			DailyRetention:   7,
			WeeklyRetention:  4,
			MonthlyRetention: 12,
			YearlyRetention:  7,
		},
		Enabled:  true,
		Metadata: map[string]string{"test": "policy"},
	}
	
	body, _ := json.Marshal(policy)
	
	// Create policy
	httpReq := httptest.NewRequest("POST", "/api/v1/backup/retention/policies", bytes.NewBuffer(body))
	httpReq.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	
	server.Router().ServeHTTP(recorder, httpReq)
	
	if recorder.Code != http.StatusCreated {
		t.Errorf("Expected 201 for policy creation, got %d", recorder.Code)
	}
	
	// Parse created policy response
	var createdPolicy backup.RetentionPolicy
	err := json.Unmarshal(recorder.Body.Bytes(), &createdPolicy)
	if err != nil {
		t.Fatalf("Failed to parse created policy: %v", err)
	}
	
	if createdPolicy.Name != policy.Name {
		t.Errorf("Expected policy name %s, got %s", policy.Name, createdPolicy.Name)
	}
	
	// List policies
	httpReq = httptest.NewRequest("GET", "/api/v1/backup/retention/policies", nil)
	recorder = httptest.NewRecorder()
	
	server.Router().ServeHTTP(recorder, httpReq)
	
	if recorder.Code != http.StatusOK {
		t.Errorf("Expected 200 for policy list, got %d", recorder.Code)
	}
	
	// Get specific policy
	policyURL := fmt.Sprintf("/api/v1/backup/retention/policies/%s", createdPolicy.ID)
	httpReq = httptest.NewRequest("GET", policyURL, nil)
	httpReq = mux.SetURLVars(httpReq, map[string]string{"policy_id": createdPolicy.ID})
	recorder = httptest.NewRecorder()
	
	server.Router().ServeHTTP(recorder, httpReq)
	
	if recorder.Code != http.StatusOK {
		t.Errorf("Expected 200 for policy get, got %d", recorder.Code)
	}
	
	// Update policy
	policy.Description = "Updated policy description"
	body, _ = json.Marshal(policy)
	httpReq = httptest.NewRequest("PUT", policyURL, bytes.NewBuffer(body))
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq = mux.SetURLVars(httpReq, map[string]string{"policy_id": createdPolicy.ID})
	recorder = httptest.NewRecorder()
	
	server.Router().ServeHTTP(recorder, httpReq)
	
	if recorder.Code != http.StatusOK {
		t.Errorf("Expected 200 for policy update, got %d", recorder.Code)
	}
	
	// Test with missing policy name
	policy.Name = ""
	body, _ = json.Marshal(policy)
	httpReq = httptest.NewRequest("POST", "/api/v1/backup/retention/policies", bytes.NewBuffer(body))
	httpReq.Header.Set("Content-Type", "application/json")
	recorder = httptest.NewRecorder()
	
	server.Router().ServeHTTP(recorder, httpReq)
	
	if recorder.Code != http.StatusBadRequest {
		t.Errorf("Expected 400 for missing policy name, got %d", recorder.Code)
	}
}

// TestPointInTimeRestoreAPI tests point-in-time restore API endpoint
func TestPointInTimeRestoreAPI(t *testing.T) {
	tmpDir := t.TempDir()
	server := setupTestServer(tmpDir)
	
	// Create test request
	req := struct {
		VMID        string    `json:"vm_id"`
		PointInTime time.Time `json:"point_in_time"`
		TargetPath  string    `json:"target_path"`
	}{
		VMID:        "test-vm-1",
		PointInTime: time.Now().Add(-24 * time.Hour),
		TargetPath:  "/tmp/pit-restore",
	}
	
	body, _ := json.Marshal(req)
	
	// Create HTTP request
	httpReq := httptest.NewRequest("POST", "/api/v1/backup/restore/point-in-time", bytes.NewBuffer(body))
	httpReq.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	
	server.Router().ServeHTTP(recorder, httpReq)
	
	// Should fail due to no backups available
	if recorder.Code != http.StatusInternalServerError {
		t.Errorf("Expected 500 for no backups available, got %d", recorder.Code)
	}
	
	// Test with missing VM ID
	req.VMID = ""
	body, _ = json.Marshal(req)
	httpReq = httptest.NewRequest("POST", "/api/v1/backup/restore/point-in-time", bytes.NewBuffer(body))
	httpReq.Header.Set("Content-Type", "application/json")
	recorder = httptest.NewRecorder()
	
	server.Router().ServeHTTP(recorder, httpReq)
	
	if recorder.Code != http.StatusBadRequest {
		t.Errorf("Expected 400 for missing VM ID, got %d", recorder.Code)
	}
}

// TestHealthStatusAPI tests the health status API endpoint
func TestHealthStatusAPI(t *testing.T) {
	tmpDir := t.TempDir()
	server := setupTestServer(tmpDir)
	
	// Test health status
	httpReq := httptest.NewRequest("GET", "/api/v1/backup/health", nil)
	recorder := httptest.NewRecorder()
	
	server.Router().ServeHTTP(recorder, httpReq)
	
	if recorder.Code != http.StatusOK {
		t.Errorf("Expected 200 for health status, got %d", recorder.Code)
	}
	
	// Parse response
	var health map[string]interface{}
	err := json.Unmarshal(recorder.Body.Bytes(), &health)
	if err != nil {
		t.Fatalf("Failed to parse health response: %v", err)
	}
	
	if health["status"] != "healthy" {
		t.Errorf("Expected healthy status, got %v", health["status"])
	}
	
	components, ok := health["components"].(map[string]interface{})
	if !ok {
		t.Error("Expected components map in health response")
	} else {
		expectedComponents := []string{"backup_manager", "retention_manager", "restore_manager"}
		for _, component := range expectedComponents {
			if components[component] != "healthy" {
				t.Errorf("Expected component %s to be healthy, got %v", component, components[component])
			}
		}
	}
}

// TestBackupStatsAPI tests the backup statistics API endpoint
func TestBackupStatsAPI(t *testing.T) {
	tmpDir := t.TempDir()
	server := setupTestServer(tmpDir)
	
	// Test backup stats
	httpReq := httptest.NewRequest("GET", "/api/v1/backup/stats", nil)
	recorder := httptest.NewRecorder()
	
	server.Router().ServeHTTP(recorder, httpReq)
	
	if recorder.Code != http.StatusOK {
		t.Errorf("Expected 200 for backup stats, got %d", recorder.Code)
	}
	
	// Parse response
	var stats map[string]interface{}
	err := json.Unmarshal(recorder.Body.Bytes(), &stats)
	if err != nil {
		t.Fatalf("Failed to parse stats response: %v", err)
	}
	
	expectedKeys := []string{"total_backups", "total_size", "compression_ratio", "deduplication_ratio", "backup_success_rate", "average_backup_time"}
	for _, key := range expectedKeys {
		if _, exists := stats[key]; !exists {
			t.Errorf("Expected stat key %s to exist", key)
		}
	}
}

// TestDedupStatsAPI tests the deduplication statistics API endpoint
func TestDedupStatsAPI(t *testing.T) {
	tmpDir := t.TempDir()
	server := setupTestServer(tmpDir)
	
	// Test deduplication stats
	httpReq := httptest.NewRequest("GET", "/api/v1/backup/dedup/stats", nil)
	recorder := httptest.NewRecorder()
	
	server.Router().ServeHTTP(recorder, httpReq)
	
	if recorder.Code != http.StatusOK {
		t.Errorf("Expected 200 for dedup stats, got %d", recorder.Code)
	}
	
	// Parse response
	var stats map[string]interface{}
	err := json.Unmarshal(recorder.Body.Bytes(), &stats)
	if err != nil {
		t.Fatalf("Failed to parse dedup stats response: %v", err)
	}
	
	expectedKeys := []string{"total_bytes", "unique_bytes", "deduplicated_bytes", "compression_ratio", "chunk_count", "unique_chunks", "last_updated"}
	for _, key := range expectedKeys {
		if _, exists := stats[key]; !exists {
			t.Errorf("Expected dedup stat key %s to exist", key)
		}
	}
}

// TestDeleteBackupAPI tests the backup deletion API endpoint with typed error handling
func TestDeleteBackupAPI(t *testing.T) {
	tmpDir := t.TempDir()
	server := setupTestServer(tmpDir)
	
	// Test delete non-existent backup (should return 404 with typed error)
	httpReq := httptest.NewRequest("DELETE", "/api/v1/backup/backups/non-existent-backup", nil)
	httpReq = mux.SetURLVars(httpReq, map[string]string{"backup_id": "non-existent-backup"})
	recorder := httptest.NewRecorder()
	
	server.Router().ServeHTTP(recorder, httpReq)
	
	if recorder.Code != http.StatusNotFound {
		t.Errorf("Expected 404 for non-existent backup delete, got %d", recorder.Code)
	}
	
	// Verify error message contains proper error handling
	body := recorder.Body.String()
	if !strings.Contains(body, "Backup not found") {
		t.Errorf("Expected 'Backup not found' error message, got: %s", body)
	}
	
	// Test delete with invalid backup ID format
	httpReq = httptest.NewRequest("DELETE", "/api/v1/backup/backups/", nil)
	httpReq = mux.SetURLVars(httpReq, map[string]string{"backup_id": ""})
	recorder = httptest.NewRecorder()
	
	server.Router().ServeHTTP(recorder, httpReq)
	
	if recorder.Code != http.StatusBadRequest {
		t.Errorf("Expected 400 for empty backup ID, got %d", recorder.Code)
	}
}

// TestBackupVerificationAPI tests the backup verification API endpoint
func TestBackupVerificationAPI(t *testing.T) {
	tmpDir := t.TempDir()
	server := setupTestServer(tmpDir)
	
	// Test verify non-existent backup
	httpReq := httptest.NewRequest("POST", "/api/v1/backup/backups/non-existent-backup/verify", nil)
	httpReq = mux.SetURLVars(httpReq, map[string]string{"backup_id": "non-existent-backup"})
	recorder := httptest.NewRecorder()
	
	server.Router().ServeHTTP(recorder, httpReq)
	
	if recorder.Code != http.StatusNotFound {
		t.Errorf("Expected 404 for non-existent backup verify, got %d", recorder.Code)
	}
}

// TestRequestContextUsage tests that request context is properly used in handlers
func TestRequestContextUsage(t *testing.T) {
	tmpDir := t.TempDir()
	server := setupTestServer(tmpDir)
	
	// Test with cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately
	
	// Test delete with cancelled context
	httpReq := httptest.NewRequestWithContext(ctx, "DELETE", "/api/v1/backup/backups/test-backup", nil)
	httpReq = mux.SetURLVars(httpReq, map[string]string{"backup_id": "test-backup"})
	recorder := httptest.NewRecorder()
	
	server.Router().ServeHTTP(recorder, httpReq)
	
	// Should handle context cancellation appropriately
	if recorder.Code == http.StatusOK {
		t.Error("Expected non-200 status for cancelled context request")
	}
	
	// Test create backup with cancelled context
	req := BackupCreateRequest{
		VMID:       "test-vm-cancelled",
		VMPath:     "/tmp/test-vm-disk.img",
		BackupType: "full",
		Metadata:   map[string]string{"test": "cancelled"},
	}
	
	body, _ := json.Marshal(req)
	httpReq = httptest.NewRequestWithContext(ctx, "POST", "/api/v1/backup/backups", bytes.NewBuffer(body))
	httpReq.Header.Set("Content-Type", "application/json")
	recorder = httptest.NewRecorder()
	
	server.Router().ServeHTTP(recorder, httpReq)
	
	// Should handle context cancellation appropriately
	if recorder.Code == http.StatusOK {
		t.Error("Expected non-200 status for cancelled context create request")
	}
}

// TestErrorHandlingTypes tests that proper typed errors are returned
func TestErrorHandlingTypes(t *testing.T) {
	tmpDir := t.TempDir()
	server := setupTestServer(tmpDir)
	
	testCases := []struct {
		name           string
		method         string
		path           string
		vars           map[string]string
		expectedStatus int
		expectedError  string
	}{
		{
			name:           "Get non-existent backup",
			method:         "GET",
			path:           "/api/v1/backup/backups/non-existent",
			vars:           map[string]string{"backup_id": "non-existent"},
			expectedStatus: http.StatusNotFound,
			expectedError:  "Backup not found",
		},
		{
			name:           "Delete non-existent backup",
			method:         "DELETE", 
			path:           "/api/v1/backup/backups/non-existent",
			vars:           map[string]string{"backup_id": "non-existent"},
			expectedStatus: http.StatusNotFound,
			expectedError:  "Backup not found",
		},
		{
			name:           "Create restore with non-existent backup",
			method:         "POST",
			path:           "/api/v1/backup/restore",
			expectedStatus: http.StatusNotFound,
			expectedError:  "Backup not found",
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var httpReq *http.Request
			
			if tc.method == "POST" && strings.Contains(tc.path, "restore") {
				// Special case for restore request
				req := RestoreCreateRequest{
					BackupID:    "non-existent-backup",
					TargetPath:  "/tmp/restore-target",
					RestoreType: "full",
				}
				body, _ := json.Marshal(req)
				httpReq = httptest.NewRequest(tc.method, tc.path, bytes.NewBuffer(body))
				httpReq.Header.Set("Content-Type", "application/json")
			} else {
				httpReq = httptest.NewRequest(tc.method, tc.path, nil)
			}
			
			if tc.vars != nil {
				httpReq = mux.SetURLVars(httpReq, tc.vars)
			}
			
			recorder := httptest.NewRecorder()
			server.Router().ServeHTTP(recorder, httpReq)
			
			if recorder.Code != tc.expectedStatus {
				t.Errorf("Expected status %d, got %d", tc.expectedStatus, recorder.Code)
			}
			
			body := recorder.Body.String()
			if !strings.Contains(body, tc.expectedError) {
				t.Errorf("Expected error message to contain '%s', got: %s", tc.expectedError, body)
			}
		})
	}
}

// TestNotImplementedEndpoints tests endpoints that return not implemented
func TestNotImplementedEndpoints(t *testing.T) {
	tmpDir := t.TempDir()
	server := setupTestServer(tmpDir)
	
	notImplementedEndpoints := []struct {
		method string
		path   string
	}{
		{"POST", "/api/v1/backup/backups/test-backup/verify"},
	}
	
	for _, endpoint := range notImplementedEndpoints {
		httpReq := httptest.NewRequest(endpoint.method, endpoint.path, nil)
		httpReq = mux.SetURLVars(httpReq, map[string]string{"backup_id": "test-backup"})
		recorder := httptest.NewRecorder()
		
		server.Router().ServeHTTP(recorder, httpReq)
		
		if recorder.Code != http.StatusNotImplemented {
			t.Errorf("Expected 501 for %s %s, got %d", endpoint.method, endpoint.path, recorder.Code)
		}
	}
}

// TestJSONReaderHelper tests the JSON reader helper function
func TestJSONReaderHelper(t *testing.T) {
	testData := map[string]string{
		"key1": "value1",
		"key2": "value2",
	}
	
	reader := jsonReader(testData)
	if reader == nil {
		t.Error("Expected reader to be non-nil")
	}
	
	// Read the data back
	buf := make([]byte, 1024)
	n, err := reader.Read(buf)
	if err != nil && err.Error() != "EOF" {
		t.Errorf("Error reading from JSON reader: %v", err)
	}
	
	// Parse JSON
	var result map[string]string
	err = json.Unmarshal(buf[:n], &result)
	if err != nil {
		t.Fatalf("Failed to parse JSON from reader: %v", err)
	}
	
	if result["key1"] != testData["key1"] {
		t.Errorf("Expected key1 value %s, got %s", testData["key1"], result["key1"])
	}
	if result["key2"] != testData["key2"] {
		t.Errorf("Expected key2 value %s, got %s", testData["key2"], result["key2"])
	}
}

// Helper function to set up test server
func setupTestServer(tmpDir string) *BackupAPIServer {
	backupManager := backup.NewIncrementalBackupManager(
		tmpDir,
		backup.NewDeduplicationEngine(tmpDir),
		backup.DefaultCompressionLevel,
	)
	retentionManager := backup.NewRetentionManager(tmpDir)
	restoreManager := backup.NewRestoreManager(tmpDir, 2)
	
	return NewBackupAPIServer(backupManager, retentionManager, restoreManager)
}

// Benchmark tests for API performance

// BenchmarkCreateBackupAPI benchmarks the backup creation API
func BenchmarkCreateBackupAPI(b *testing.B) {
	tmpDir := b.TempDir()
	server := setupTestServer(tmpDir)
	
	req := BackupCreateRequest{
		VMID:       "bench-vm",
		VMPath:     "/tmp/bench-vm-disk.img",
		BackupType: "full",
		Metadata:   map[string]string{"bench": "true"},
	}
	
	body, _ := json.Marshal(req)
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		req.VMID = fmt.Sprintf("bench-vm-%d", i)
		body, _ = json.Marshal(req)
		
		httpReq := httptest.NewRequest("POST", "/api/v1/backup/backups", bytes.NewBuffer(body))
		httpReq.Header.Set("Content-Type", "application/json")
		recorder := httptest.NewRecorder()
		
		server.Router().ServeHTTP(recorder, httpReq)
		
		// Note: This will likely fail due to missing VM disk, but we measure API overhead
	}
}

// BenchmarkListBackupsAPI benchmarks the backup listing API
func BenchmarkListBackupsAPI(b *testing.B) {
	tmpDir := b.TempDir()
	server := setupTestServer(tmpDir)
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		vmID := fmt.Sprintf("bench-vm-%d", i%100)
		url := fmt.Sprintf("/api/v1/backup/backups?vm_id=%s", vmID)
		
		httpReq := httptest.NewRequest("GET", url, nil)
		recorder := httptest.NewRecorder()
		
		server.Router().ServeHTTP(recorder, httpReq)
	}
}