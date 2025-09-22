package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/khryptorgraphics/novacron/backend/core/audit"
	"github.com/khryptorgraphics/novacron/backend/core/auth"
	"github.com/khryptorgraphics/novacron/backend/core/security"
)

// SecurityHandlers provides HTTP handlers for security endpoints
type SecurityHandlers struct {
	twoFactorService    *auth.TwoFactorService
	securityCoordinator *security.DistributedSecurityCoordinator
	vulnerabilityScanner *security.VulnerabilityScanner
	auditLogger         audit.AuditLogger
	encryptionManager   *security.EncryptionManager
}

// NewSecurityHandlers creates new security handlers
func NewSecurityHandlers(
	twoFactorService *auth.TwoFactorService,
	securityCoordinator *security.DistributedSecurityCoordinator,
	vulnerabilityScanner *security.VulnerabilityScanner,
	auditLogger audit.AuditLogger,
	encryptionManager *security.EncryptionManager,
) *SecurityHandlers {
	return &SecurityHandlers{
		twoFactorService:    twoFactorService,
		securityCoordinator: securityCoordinator,
		vulnerabilityScanner: vulnerabilityScanner,
		auditLogger:         auditLogger,
		encryptionManager:   encryptionManager,
	}
}

// RegisterRoutes registers all security routes
func (h *SecurityHandlers) RegisterRoutes(router *mux.Router) {
	// 2FA routes
	router.HandleFunc("/api/auth/2fa/setup", h.Setup2FA).Methods("POST")
	router.HandleFunc("/api/auth/2fa/qr", h.GenerateQRCode).Methods("GET")
	router.HandleFunc("/api/auth/2fa/verify", h.Verify2FA).Methods("POST")
	router.HandleFunc("/api/auth/2fa/enable", h.Enable2FA).Methods("POST")
	router.HandleFunc("/api/auth/2fa/disable", h.Disable2FA).Methods("POST")
	router.HandleFunc("/api/auth/2fa/backup-codes", h.GetBackupCodes).Methods("GET")
	router.HandleFunc("/api/auth/2fa/backup-codes", h.RegenerateBackupCodes).Methods("POST")
	router.HandleFunc("/api/auth/2fa/status", h.Get2FAStatus).Methods("GET")

	// Security monitoring routes
	router.HandleFunc("/api/security/threats", h.GetThreats).Methods("GET")
	router.HandleFunc("/api/security/vulnerabilities", h.GetVulnerabilities).Methods("GET")
	router.HandleFunc("/api/security/compliance", h.GetComplianceStatus).Methods("GET")
	router.HandleFunc("/api/security/incidents", h.GetIncidents).Methods("GET")
	router.HandleFunc("/api/security/events", h.GetSecurityEvents).Methods("GET")
	router.HandleFunc("/api/security/scan", h.StartVulnerabilityScan).Methods("POST")
	router.HandleFunc("/api/security/scan/{scanId}", h.GetScanResults).Methods("GET")
	router.HandleFunc("/api/security/cluster/{clusterId}/state", h.GetClusterSecurityState).Methods("GET")

	// Audit routes
	router.HandleFunc("/api/security/audit/events", h.GetAuditEvents).Methods("GET")
	router.HandleFunc("/api/security/audit/export", h.ExportAuditLog).Methods("GET")
	router.HandleFunc("/api/security/audit/statistics", h.GetAuditStatistics).Methods("GET")

	// RBAC routes
	router.HandleFunc("/api/security/rbac/roles", h.GetRoles).Methods("GET")
	router.HandleFunc("/api/security/rbac/permissions", h.GetPermissions).Methods("GET")
	router.HandleFunc("/api/security/rbac/user/{userId}/roles", h.GetUserRoles).Methods("GET")
	router.HandleFunc("/api/security/rbac/user/{userId}/roles", h.AssignUserRoles).Methods("POST")
	router.HandleFunc("/api/security/rbac/user/{userId}/permissions", h.GetUserPermissions).Methods("GET")

	// WebSocket endpoint for real-time security events
	router.HandleFunc("/api/security/events/stream", h.StreamSecurityEvents)
}

// 2FA Handlers

// Setup2FA initiates 2FA setup
func (h *SecurityHandlers) Setup2FA(w http.ResponseWriter, r *http.Request) {
	var req struct {
		UserID      string `json:"user_id"`
		AccountName string `json:"account_name"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if req.UserID == "" || req.AccountName == "" {
		http.Error(w, "user_id and account_name are required", http.StatusBadRequest)
		return
	}

	response, err := h.twoFactorService.SetupTwoFactor(req.UserID, req.AccountName)
	if err != nil {
		log.Printf("2FA setup error: %v", err)
		http.Error(w, "Failed to setup 2FA", http.StatusInternalServerError)
		return
	}

	ctx := r.Context()
	h.auditLogger.LogEvent(ctx, &audit.AuditEvent{
		UserID:   req.UserID,
		Resource: "2fa_setup",
		Action:   audit.ActionUpdate,
		Result:   audit.ResultSuccess,
		Details: map[string]interface{}{
			"description":  "2FA setup initiated",
			"account_name": req.AccountName,
		},
	})

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// GenerateQRCode generates QR code for 2FA setup
func (h *SecurityHandlers) GenerateQRCode(w http.ResponseWriter, r *http.Request) {
	userID := r.URL.Query().Get("user_id")
	if userID == "" {
		http.Error(w, "user_id is required", http.StatusBadRequest)
		return
	}

	qrCode, err := h.twoFactorService.GenerateQRCode(userID)
	if err != nil {
		log.Printf("QR code generation error: %v", err)
		http.Error(w, "Failed to generate QR code", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "image/png")
	w.Write(qrCode)
}

// Verify2FA verifies a 2FA code
func (h *SecurityHandlers) Verify2FA(w http.ResponseWriter, r *http.Request) {
	var req auth.TwoFactorVerifyRequest

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	response, err := h.twoFactorService.VerifyCode(req)
	if err != nil {
		log.Printf("2FA verification error: %v", err)
		http.Error(w, "Verification failed", http.StatusInternalServerError)
		return
	}

	ctx := r.Context()
	result := audit.ResultSuccess
	if !response.Valid {
		result = audit.ResultFailure
	}
	h.auditLogger.LogEvent(ctx, &audit.AuditEvent{
		UserID:   req.UserID,
		Resource: "2fa_verification",
		Action:   audit.ActionRead,
		Result:   result,
		Details: map[string]interface{}{
			"description":    fmt.Sprintf("2FA verification attempt: %t", response.Valid),
			"success":        response.Valid,
			"is_backup_code": req.IsBackupCode,
		},
	})

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// Enable2FA enables 2FA for a user after verification
func (h *SecurityHandlers) Enable2FA(w http.ResponseWriter, r *http.Request) {
	var req struct {
		UserID string `json:"user_id"`
		Code   string `json:"code"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if err := h.twoFactorService.VerifyAndEnable(req.UserID, req.Code); err != nil {
		log.Printf("2FA enable error: %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	ctx := r.Context()
	h.auditLogger.LogEvent(ctx, &audit.AuditEvent{
		UserID:   req.UserID,
		Resource: "2fa_configuration",
		Action:   audit.ActionUpdate,
		Result:   audit.ResultSuccess,
		Details: map[string]interface{}{
			"description": "2FA enabled for user",
		},
	})

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"message": "2FA enabled successfully",
	})
}

// Disable2FA disables 2FA for a user
func (h *SecurityHandlers) Disable2FA(w http.ResponseWriter, r *http.Request) {
	var req struct {
		UserID string `json:"user_id"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if err := h.twoFactorService.DisableTwoFactor(req.UserID); err != nil {
		log.Printf("2FA disable error: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	ctx := r.Context()
	h.auditLogger.LogEvent(ctx, &audit.AuditEvent{
		UserID:   req.UserID,
		Resource: "2fa_configuration",
		Action:   audit.ActionUpdate,
		Result:   audit.ResultSuccess,
		Details: map[string]interface{}{
			"description": "2FA disabled for user",
		},
	})

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"message": "2FA disabled successfully",
	})
}

// GetBackupCodes returns backup codes for a user
func (h *SecurityHandlers) GetBackupCodes(w http.ResponseWriter, r *http.Request) {
	userID := r.URL.Query().Get("user_id")
	if userID == "" {
		http.Error(w, "user_id is required", http.StatusBadRequest)
		return
	}

	codes, err := h.twoFactorService.GetBackupCodes(userID)
	if err != nil {
		log.Printf("Get backup codes error: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"backup_codes": codes,
	})
}

// RegenerateBackupCodes generates new backup codes
func (h *SecurityHandlers) RegenerateBackupCodes(w http.ResponseWriter, r *http.Request) {
	var req struct {
		UserID string `json:"user_id"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	codes, err := h.twoFactorService.RegenerateBackupCodes(req.UserID)
	if err != nil {
		log.Printf("Regenerate backup codes error: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	ctx := r.Context()
	h.auditLogger.LogEvent(ctx, &audit.AuditEvent{
		UserID:   req.UserID,
		Resource: "2fa_backup_codes",
		Action:   audit.ActionUpdate,
		Result:   audit.ResultSuccess,
		Details: map[string]interface{}{
			"description": "2FA backup codes regenerated",
		},
	})

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"backup_codes": codes,
	})
}

// Get2FAStatus returns 2FA status for a user
func (h *SecurityHandlers) Get2FAStatus(w http.ResponseWriter, r *http.Request) {
	userID := r.URL.Query().Get("user_id")
	if userID == "" {
		http.Error(w, "user_id is required", http.StatusBadRequest)
		return
	}

	info, err := h.twoFactorService.GetUserTwoFactorInfo(userID)
	if err != nil {
		// User doesn't have 2FA setup
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"enabled":      false,
			"setup":        false,
		})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"enabled":      info.Enabled,
		"setup":        true,
		"setup_at":     info.SetupAt,
		"last_used":    info.LastUsed,
		"algorithm":    info.Algorithm,
		"digits":       info.Digits,
		"period":       info.Period,
	})
}

// Security Monitoring Handlers

// GetThreats returns current security threats
func (h *SecurityHandlers) GetThreats(w http.ResponseWriter, r *http.Request) {
	states := h.securityCoordinator.GetAllClusterStates()

	var allThreats []security.SecurityEvent
	for _, state := range states {
		allThreats = append(allThreats, state.ActiveThreats...)
	}

	response := map[string]interface{}{
		"threats": allThreats,
		"count":   len(allThreats),
		"updated_at": time.Now(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// GetVulnerabilities returns vulnerability scan results
func (h *SecurityHandlers) GetVulnerabilities(w http.ResponseWriter, r *http.Request) {
	// This would typically come from a database of scan results
	// For now, return mock data structure
	response := map[string]interface{}{
		"vulnerabilities": []map[string]interface{}{},
		"summary": map[string]interface{}{
			"total":    0,
			"critical": 0,
			"high":     0,
			"medium":   0,
			"low":      0,
		},
		"last_scan": nil,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// GetComplianceStatus returns compliance status
func (h *SecurityHandlers) GetComplianceStatus(w http.ResponseWriter, r *http.Request) {
	response := map[string]interface{}{
		"compliance_score": 85.5,
		"frameworks": []map[string]interface{}{
			{
				"name":   "OWASP Top 10",
				"score":  90.0,
				"status": "compliant",
			},
			{
				"name":   "NIST Cybersecurity Framework",
				"score":  82.5,
				"status": "partially_compliant",
			},
		},
		"last_updated": time.Now(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// GetIncidents returns security incidents
func (h *SecurityHandlers) GetIncidents(w http.ResponseWriter, r *http.Request) {
	limit := 50
	if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 && l <= 1000 {
			limit = l
		}
	}

	// This would typically come from a database
	response := map[string]interface{}{
		"incidents": []map[string]interface{}{},
		"total":     0,
		"limit":     limit,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// GetSecurityEvents returns recent security events
func (h *SecurityHandlers) GetSecurityEvents(w http.ResponseWriter, r *http.Request) {
	limit := 100
	if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 && l <= 1000 {
			limit = l
		}
	}

	// This would typically come from a database
	response := map[string]interface{}{
		"events": []map[string]interface{}{},
		"total":  0,
		"limit":  limit,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// StartVulnerabilityScan starts a vulnerability scan
func (h *SecurityHandlers) StartVulnerabilityScan(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Targets   []string                    `json:"targets"`
		ScanTypes []security.ScanType         `json:"scan_types"`
		Config    map[string]interface{}      `json:"config"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if len(req.Targets) == 0 {
		http.Error(w, "At least one target is required", http.StatusBadRequest)
		return
	}

	// Start scan asynchronously
	scanID := fmt.Sprintf("scan-%d", time.Now().UnixNano())

	go func() {
		results, err := h.vulnerabilityScanner.RunComprehensiveScan(r.Context(), req.Targets)
		if err != nil {
			log.Printf("Vulnerability scan failed: %v", err)
		} else {
			log.Printf("Vulnerability scan completed: %s", results.ScanID)
		}
	}()

	response := map[string]interface{}{
		"scan_id":    scanID,
		"status":     "started",
		"targets":    req.Targets,
		"scan_types": req.ScanTypes,
		"started_at": time.Now(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// GetScanResults returns vulnerability scan results
func (h *SecurityHandlers) GetScanResults(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	scanID := vars["scanId"]

	// This would typically fetch from database
	response := map[string]interface{}{
		"scan_id": scanID,
		"status":  "completed",
		"results": map[string]interface{}{
			"findings": []map[string]interface{}{},
			"summary": map[string]interface{}{
				"total":    0,
				"critical": 0,
				"high":     0,
				"medium":   0,
				"low":      0,
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// GetClusterSecurityState returns security state for a specific cluster
func (h *SecurityHandlers) GetClusterSecurityState(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	clusterID := vars["clusterId"]

	state, err := h.securityCoordinator.GetClusterSecurityState(clusterID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(state)
}

// Audit Handlers

// GetAuditEvents returns audit events
func (h *SecurityHandlers) GetAuditEvents(w http.ResponseWriter, r *http.Request) {
	limit := 100
	if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 && l <= 1000 {
			limit = l
		}
	}

	// This would typically come from audit storage
	response := map[string]interface{}{
		"events": []map[string]interface{}{},
		"total":  0,
		"limit":  limit,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// ExportAuditLog exports audit log
func (h *SecurityHandlers) ExportAuditLog(w http.ResponseWriter, r *http.Request) {
	format := r.URL.Query().Get("format")
	if format == "" {
		format = "json"
	}

	// This would typically generate and return audit export
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=audit-log.%s", format))

	switch format {
	case "csv":
		w.Header().Set("Content-Type", "text/csv")
		w.Write([]byte("timestamp,event_type,user_id,description\n"))
	default:
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"events": []map[string]interface{}{},
			"exported_at": time.Now(),
		})
	}
}

// GetAuditStatistics returns audit statistics
func (h *SecurityHandlers) GetAuditStatistics(w http.ResponseWriter, r *http.Request) {
	response := map[string]interface{}{
		"total_events": 0,
		"events_by_type": map[string]int{},
		"events_by_severity": map[string]int{},
		"period": "last_30_days",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// RBAC Handlers

// GetRoles returns available roles
func (h *SecurityHandlers) GetRoles(w http.ResponseWriter, r *http.Request) {
	roles := []map[string]interface{}{
		{
			"id":          "admin",
			"name":        "Administrator",
			"description": "Full system access",
			"permissions": []string{"*"},
		},
		{
			"id":          "user",
			"name":        "User",
			"description": "Standard user access",
			"permissions": []string{"read", "write_own"},
		},
		{
			"id":          "readonly",
			"name":        "Read Only",
			"description": "Read-only access",
			"permissions": []string{"read"},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"roles": roles,
	})
}

// GetPermissions returns available permissions
func (h *SecurityHandlers) GetPermissions(w http.ResponseWriter, r *http.Request) {
	permissions := []map[string]interface{}{
		{
			"id":          "read",
			"name":        "Read",
			"description": "Read access to resources",
		},
		{
			"id":          "write",
			"name":        "Write",
			"description": "Write access to resources",
		},
		{
			"id":          "delete",
			"name":        "Delete",
			"description": "Delete access to resources",
		},
		{
			"id":          "admin",
			"name":        "Admin",
			"description": "Administrative access",
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"permissions": permissions,
	})
}

// GetUserRoles returns roles for a specific user
func (h *SecurityHandlers) GetUserRoles(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	userID := vars["userId"]

	// This would typically come from database
	response := map[string]interface{}{
		"user_id": userID,
		"roles":   []string{"user"},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// AssignUserRoles assigns roles to a user
func (h *SecurityHandlers) AssignUserRoles(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	userID := vars["userId"]

	var req struct {
		Roles []string `json:"roles"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// This would typically update database
	ctx := r.Context()
	h.auditLogger.LogEvent(ctx, &audit.AuditEvent{
		UserID:   userID,
		Resource: "user_roles",
		Action:   audit.ActionUpdate,
		Result:   audit.ResultSuccess,
		Details:   map[string]interface{}{"roles": req.Roles, "description": "User roles assigned"},
	})

	response := map[string]interface{}{
		"user_id": userID,
		"roles":   req.Roles,
		"success": true,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// GetUserPermissions returns effective permissions for a user
func (h *SecurityHandlers) GetUserPermissions(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	userID := vars["userId"]

	// This would typically calculate from roles and return effective permissions
	response := map[string]interface{}{
		"user_id":     userID,
		"permissions": []string{"read", "write_own"},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		// In production, implement proper origin checking
		return true
	},
}

// StreamSecurityEvents handles WebSocket connections for real-time security events
func (h *SecurityHandlers) StreamSecurityEvents(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	// Upgrade connection to WebSocket
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	// Log the connection
	h.auditLogger.LogEvent(ctx, &audit.AuditEvent{
		UserID:   r.Header.Get("X-User-ID"), // In real app, get from auth context
		Resource: "websocket_connection",
		Action:   audit.ActionRead,
		Result:   audit.ResultSuccess,
		Details: map[string]interface{}{
			"description": "Security events WebSocket connection established",
			"remote_addr": r.RemoteAddr,
		},
	})

	// Create a ticker for periodic updates
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// Get current security events
			states := h.securityCoordinator.GetAllClusterStates()

			// Create event summary
			summary := map[string]interface{}{
				"timestamp": time.Now(),
				"clusters":  len(states),
				"events":    []interface{}{},
			}

			totalThreats := 0
			for clusterID, state := range states {
				totalThreats += len(state.ActiveThreats)
				if len(state.ActiveThreats) > 0 {
					summary["events"] = append(summary["events"].([]interface{}), map[string]interface{}{
						"cluster_id":    clusterID,
						"threat_level":  state.ThreatLevel,
						"active_threats": len(state.ActiveThreats),
						"quarantined":   state.Quarantined,
					})
				}
			}

			summary["total_active_threats"] = totalThreats

			// Send the summary
			if err := conn.WriteJSON(summary); err != nil {
				log.Printf("WebSocket write error: %v", err)
				return
			}

		case <-ctx.Done():
			return
		}
	}
}