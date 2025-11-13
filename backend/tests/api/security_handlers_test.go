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
	handlers "github.com/khryptorgraphics/novacron/backend/api/security"
	"github.com/khryptorgraphics/novacron/backend/core/audit"
	"github.com/khryptorgraphics/novacron/backend/core/auth"
	"github.com/khryptorgraphics/novacron/backend/core/security"
	"novacron/backend/pkg/testutil"
)

// Mock implementations for testing
type mockTwoFactorService struct {
	setupCalled   bool
	setupSecret   string
	verifyCalled  bool
	verifyResult  bool
	enableCalled  bool
	disableCalled bool
}

func (m *mockTwoFactorService) Setup(userID, accountName string) (string, string, error) {
	m.setupCalled = true
	m.setupSecret = "test-secret"
	return m.setupSecret, "otpauth://totp/Test", nil
}

func (m *mockTwoFactorService) Verify(userID, token string) (bool, error) {
	m.verifyCalled = true
	m.verifyResult = (token == "123456")
	return m.verifyResult, nil
}

func (m *mockTwoFactorService) Enable(userID string) error {
	m.enableCalled = true
	return nil
}

func (m *mockTwoFactorService) Disable(userID string) error {
	m.disableCalled = true
	return nil
}

func (m *mockTwoFactorService) GenerateBackupCodes(userID string) ([]string, error) {
	return []string{"BACKUP1", "BACKUP2", "BACKUP3"}, nil
}

func (m *mockTwoFactorService) GetStatus(userID string) (bool, error) {
	return true, nil
}

type mockSecurityCoordinator struct {
	threats       []security.Threat
	vulnerabilities []security.Vulnerability
}

func (m *mockSecurityCoordinator) GetThreats(ctx context.Context) ([]security.Threat, error) {
	return m.threats, nil
}

func (m *mockSecurityCoordinator) GetVulnerabilities(ctx context.Context) ([]security.Vulnerability, error) {
	return m.vulnerabilities, nil
}

func (m *mockSecurityCoordinator) GetComplianceStatus(ctx context.Context) (*security.ComplianceStatus, error) {
	return &security.ComplianceStatus{
		Compliant: true,
		Score:     95,
		LastCheck: time.Now(),
	}, nil
}

type mockVulnerabilityScanner struct {
	scanStarted bool
	scanResults *security.ScanResults
}

func (m *mockVulnerabilityScanner) StartScan(ctx context.Context, target string) (string, error) {
	m.scanStarted = true
	return "scan-123", nil
}

func (m *mockVulnerabilityScanner) GetResults(ctx context.Context, scanID string) (*security.ScanResults, error) {
	if m.scanResults == nil {
		m.scanResults = &security.ScanResults{
			ScanID:    scanID,
			Status:    "completed",
			Findings:  []security.Finding{},
			StartTime: time.Now().Add(-1 * time.Hour),
			EndTime:   time.Now(),
		}
	}
	return m.scanResults, nil
}

type mockAuditLogger struct {
	events []audit.AuditEvent
}

func (m *mockAuditLogger) Log(ctx context.Context, event audit.AuditEvent) error {
	m.events = append(m.events, event)
	return nil
}

func (m *mockAuditLogger) Query(ctx context.Context, filter audit.AuditFilter) ([]audit.AuditEvent, error) {
	return m.events, nil
}

func (m *mockAuditLogger) GetStatistics(ctx context.Context) (*audit.Statistics, error) {
	return &audit.Statistics{
		TotalEvents: len(m.events),
		ByType:      map[string]int{"auth": 5, "access": 10},
	}, nil
}

type mockEncryptionManager struct{}

func (m *mockEncryptionManager) Encrypt(data []byte) ([]byte, error) {
	return data, nil
}

func (m *mockEncryptionManager) Decrypt(data []byte) ([]byte, error) {
	return data, nil
}

// Test Setup

func setupTestHandlers() (*handlers.SecurityHandlers, *mockTwoFactorService) {
	mock2FA := &mockTwoFactorService{}
	mockCoord := &mockSecurityCoordinator{
		threats: []security.Threat{
			{ID: "threat-1", Severity: "high", Description: "Test threat"},
		},
		vulnerabilities: []security.Vulnerability{
			{ID: "vuln-1", Severity: "medium", Description: "Test vulnerability"},
		},
	}
	mockScanner := &mockVulnerabilityScanner{}
	mockAudit := &mockAuditLogger{}
	mockEncryption := &mockEncryptionManager{}

	h := handlers.NewSecurityHandlers(
		mock2FA,
		mockCoord,
		mockScanner,
		mockAudit,
		mockEncryption,
	)

	return h, mock2FA
}

// Test Cases

func TestSetup2FA_Success(t *testing.T) {
	h, mock := setupTestHandlers()

	reqBody := map[string]string{
		"user_id":      "user-123",
		"account_name": testutil.GetTestEmail(),
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/api/auth/2fa/setup", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.Setup2FA(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	if !mock.setupCalled {
		t.Error("Expected Setup to be called on 2FA service")
	}

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)

	if response["secret"] == nil {
		t.Error("Expected secret in response")
	}
}

func TestSetup2FA_MissingUserID(t *testing.T) {
	h, _ := setupTestHandlers()

	reqBody := map[string]string{
		"account_name": testutil.GetTestEmail(),
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/api/auth/2fa/setup", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.Setup2FA(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400, got %d", w.Code)
	}
}

func TestSetup2FA_InvalidJSON(t *testing.T) {
	h, _ := setupTestHandlers()

	req := httptest.NewRequest("POST", "/api/auth/2fa/setup", bytes.NewBufferString("{invalid json"))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.Setup2FA(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400, got %d", w.Code)
	}
}

func TestVerify2FA_Success(t *testing.T) {
	h, mock := setupTestHandlers()

	reqBody := map[string]string{
		"user_id": "user-123",
		"token":   "123456",
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/api/auth/2fa/verify", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.Verify2FA(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	if !mock.verifyCalled {
		t.Error("Expected Verify to be called on 2FA service")
	}

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)

	if response["valid"] != true {
		t.Error("Expected valid to be true")
	}
}

func TestVerify2FA_InvalidToken(t *testing.T) {
	h, _ := setupTestHandlers()

	reqBody := map[string]string{
		"user_id": "user-123",
		"token":   "wrong-token",
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/api/auth/2fa/verify", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.Verify2FA(w, req)

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)

	if response["valid"] == true {
		t.Error("Expected valid to be false for invalid token")
	}
}

func TestEnable2FA_Success(t *testing.T) {
	h, mock := setupTestHandlers()

	reqBody := map[string]string{
		"user_id": "user-123",
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/api/auth/2fa/enable", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.Enable2FA(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	if !mock.enableCalled {
		t.Error("Expected Enable to be called on 2FA service")
	}
}

func TestDisable2FA_Success(t *testing.T) {
	h, mock := setupTestHandlers()

	reqBody := map[string]string{
		"user_id": "user-123",
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/api/auth/2fa/disable", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.Disable2FA(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	if !mock.disableCalled {
		t.Error("Expected Disable to be called on 2FA service")
	}
}

func TestGetBackupCodes_Success(t *testing.T) {
	h, _ := setupTestHandlers()

	req := httptest.NewRequest("GET", "/api/auth/2fa/backup-codes?user_id=user-123", nil)
	w := httptest.NewRecorder()

	h.GetBackupCodes(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)

	codes, ok := response["backup_codes"].([]interface{})
	if !ok || len(codes) == 0 {
		t.Error("Expected backup_codes array in response")
	}
}

func TestGetBackupCodes_MissingUserID(t *testing.T) {
	h, _ := setupTestHandlers()

	req := httptest.NewRequest("GET", "/api/auth/2fa/backup-codes", nil)
	w := httptest.NewRecorder()

	h.GetBackupCodes(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400, got %d", w.Code)
	}
}

func TestGetThreats_Success(t *testing.T) {
	h, _ := setupTestHandlers()

	req := httptest.NewRequest("GET", "/api/security/threats", nil)
	w := httptest.NewRecorder()

	h.GetThreats(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)

	threats, ok := response["threats"].([]interface{})
	if !ok {
		t.Error("Expected threats array in response")
	}

	if len(threats) != 1 {
		t.Errorf("Expected 1 threat, got %d", len(threats))
	}
}

func TestGetVulnerabilities_Success(t *testing.T) {
	h, _ := setupTestHandlers()

	req := httptest.NewRequest("GET", "/api/security/vulnerabilities", nil)
	w := httptest.NewRecorder()

	h.GetVulnerabilities(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)

	vulns, ok := response["vulnerabilities"].([]interface{})
	if !ok {
		t.Error("Expected vulnerabilities array in response")
	}

	if len(vulns) != 1 {
		t.Errorf("Expected 1 vulnerability, got %d", len(vulns))
	}
}

func TestGetComplianceStatus_Success(t *testing.T) {
	h, _ := setupTestHandlers()

	req := httptest.NewRequest("GET", "/api/security/compliance", nil)
	w := httptest.NewRecorder()

	h.GetComplianceStatus(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)

	if response["compliant"] != true {
		t.Error("Expected compliant to be true")
	}

	score, ok := response["score"].(float64)
	if !ok || score != 95 {
		t.Errorf("Expected score 95, got %v", score)
	}
}

func TestStartVulnerabilityScan_Success(t *testing.T) {
	h, _ := setupTestHandlers()

	reqBody := map[string]string{
		"target": "cluster-123",
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/api/security/scan", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.StartVulnerabilityScan(w, req)

	if w.Code != http.StatusOK && w.Code != http.StatusAccepted {
		t.Errorf("Expected status 200 or 202, got %d", w.Code)
	}

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)

	if response["scan_id"] == nil {
		t.Error("Expected scan_id in response")
	}
}

func TestGetScanResults_Success(t *testing.T) {
	h, _ := setupTestHandlers()

	req := httptest.NewRequest("GET", "/api/security/scan/scan-123", nil)
	req = mux.SetURLVars(req, map[string]string{"scanId": "scan-123"})
	w := httptest.NewRecorder()

	h.GetScanResults(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response security.ScanResults
	json.Unmarshal(w.Body.Bytes(), &response)

	if response.ScanID != "scan-123" {
		t.Errorf("Expected scan_id scan-123, got %s", response.ScanID)
	}
}

func TestGetAuditEvents_Success(t *testing.T) {
	h, _ := setupTestHandlers()

	req := httptest.NewRequest("GET", "/api/security/audit/events", nil)
	w := httptest.NewRecorder()

	h.GetAuditEvents(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)

	events, ok := response["events"].([]interface{})
	if !ok {
		t.Error("Expected events array in response")
	}

	if events == nil {
		t.Error("Expected non-nil events array")
	}
}

func TestGetAuditStatistics_Success(t *testing.T) {
	h, _ := setupTestHandlers()

	req := httptest.NewRequest("GET", "/api/security/audit/statistics", nil)
	w := httptest.NewRecorder()

	h.GetAuditStatistics(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response audit.Statistics
	json.Unmarshal(w.Body.Bytes(), &response)

	if response.TotalEvents == 0 {
		t.Error("Expected non-zero total events")
	}

	if len(response.ByType) == 0 {
		t.Error("Expected non-empty ByType map")
	}
}

// Edge cases and error handling

func TestGetThreats_WithContextTimeout(t *testing.T) {
	h, _ := setupTestHandlers()

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Nanosecond)
	defer cancel()

	time.Sleep(10 * time.Millisecond) // Ensure context expires

	req := httptest.NewRequest("GET", "/api/security/threats", nil).WithContext(ctx)
	w := httptest.NewRecorder()

	h.GetThreats(w, req)

	// Should handle context timeout gracefully
	if w.Code == http.StatusOK {
		t.Error("Expected non-200 status for expired context")
	}
}

func TestVerify2FA_EmptyToken(t *testing.T) {
	h, _ := setupTestHandlers()

	reqBody := map[string]string{
		"user_id": "user-123",
		"token":   "",
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/api/auth/2fa/verify", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.Verify2FA(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400, got %d", w.Code)
	}
}

func TestStartVulnerabilityScan_EmptyTarget(t *testing.T) {
	h, _ := setupTestHandlers()

	reqBody := map[string]string{
		"target": "",
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/api/security/scan", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.StartVulnerabilityScan(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400, got %d", w.Code)
	}
}

// Concurrent access tests

func TestConcurrentSetup2FA(t *testing.T) {
	h, _ := setupTestHandlers()

	const numRequests = 10
	done := make(chan bool, numRequests)

	for i := 0; i < numRequests; i++ {
		go func(id int) {
			reqBody := map[string]string{
				"user_id":      "user-" + string(rune(id)),
				"account_name": testutil.GetTestEmail(),
			}
			body, _ := json.Marshal(reqBody)

			req := httptest.NewRequest("POST", "/api/auth/2fa/setup", bytes.NewBuffer(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			h.Setup2FA(w, req)

			if w.Code != http.StatusOK {
				t.Errorf("Request %d failed with status %d", id, w.Code)
			}

			done <- true
		}(i)
	}

	for i := 0; i < numRequests; i++ {
		<-done
	}
}
