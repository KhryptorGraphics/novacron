package handlers

import (
	"cmp"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/khryptorgraphics/novacron/backend/core/audit"
	"github.com/khryptorgraphics/novacron/backend/core/auth"
)

// SecurityHandlers provides HTTP handlers for security endpoints
type SecurityHandlers struct {
	twoFactorService *auth.TwoFactorService
	auditLogger      audit.AuditLogger
	rbacStore        UserRoleStore

	scanMu      sync.RWMutex
	scanRecords map[string]*scanRecord
}

type scanRecord struct {
	ScanID      string       `json:"scan_id"`
	Status      string       `json:"status"`
	Targets     []string     `json:"targets"`
	ScanTypes   []ScanType   `json:"scan_types"`
	StartedAt   time.Time    `json:"started_at"`
	CompletedAt *time.Time   `json:"completed_at,omitempty"`
	Error       string       `json:"error,omitempty"`
	Results     *ScanResults `json:"results,omitempty"`
}

// NewSecurityHandlers creates new security handlers
func NewSecurityHandlers(twoFactorService *auth.TwoFactorService, deps ...interface{}) *SecurityHandlers {
	handler := &SecurityHandlers{
		twoFactorService: twoFactorService,
		auditLogger:      audit.NewSimpleAuditLogger(),
		scanRecords:      make(map[string]*scanRecord),
	}

	for _, dependency := range deps {
		switch typed := dependency.(type) {
		case audit.AuditLogger:
			handler.auditLogger = typed
		case UserRoleStore:
			handler.rbacStore = typed
		}
	}

	return handler
}

func (h *SecurityHandlers) WithRBACStore(store UserRoleStore) *SecurityHandlers {
	h.rbacStore = store
	return h
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

func (h *SecurityHandlers) respondJSON(w http.ResponseWriter, status int, payload interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func (h *SecurityHandlers) respondUnsupported(w http.ResponseWriter, r *http.Request, capability string) {
	h.respondJSON(w, http.StatusNotImplemented, map[string]interface{}{
		"error":     "not_supported",
		"message":   capability + " is not supported by the canonical backend yet",
		"path":      r.URL.Path,
		"supported": false,
		"timestamp": time.Now().UTC(),
	})
}

func userIDFromRequest(r *http.Request, explicitUserID string) string {
	if trimmed := strings.TrimSpace(explicitUserID); trimmed != "" {
		return trimmed
	}
	if queryUserID := strings.TrimSpace(r.URL.Query().Get("user_id")); queryUserID != "" {
		return queryUserID
	}
	if userID, ok := r.Context().Value("user_id").(string); ok {
		return strings.TrimSpace(userID)
	}
	return ""
}

func (h *SecurityHandlers) logAudit(ctx context.Context, event *audit.AuditEvent) {
	if h.auditLogger == nil || event == nil {
		return
	}
	if err := h.auditLogger.LogEvent(ctx, event); err != nil {
		log.Printf("audit log error: %v", err)
	}
}

func (h *SecurityHandlers) listAuditEvents(ctx context.Context, filter *audit.AuditFilter) ([]*audit.AuditEvent, error) {
	if h.auditLogger == nil {
		return nil, fmt.Errorf("audit logging is not configured")
	}
	return h.auditLogger.QueryEvents(ctx, filter)
}

func (h *SecurityHandlers) saveScanRecord(record *scanRecord) {
	h.scanMu.Lock()
	defer h.scanMu.Unlock()
	h.scanRecords[record.ScanID] = record
}

func (h *SecurityHandlers) getScanRecord(scanID string) (*scanRecord, bool) {
	h.scanMu.RLock()
	defer h.scanMu.RUnlock()
	record, ok := h.scanRecords[scanID]
	if !ok {
		return nil, false
	}
	copyRecord := *record
	return &copyRecord, true
}

func (h *SecurityHandlers) completedScanRecords() []*scanRecord {
	h.scanMu.RLock()
	defer h.scanMu.RUnlock()

	records := make([]*scanRecord, 0, len(h.scanRecords))
	for _, record := range h.scanRecords {
		if record.Status != "completed" || record.Results == nil {
			continue
		}
		copyRecord := *record
		records = append(records, &copyRecord)
	}

	slices.SortFunc(records, func(a, b *scanRecord) int {
		return cmp.Compare(b.StartedAt.UnixNano(), a.StartedAt.UnixNano())
	})

	return records
}

func (h *SecurityHandlers) buildThreats(limit int) []SecurityThreat {
	threats := make([]SecurityThreat, 0)
	for _, record := range h.completedScanRecords() {
		if record.Results == nil {
			continue
		}
		for _, finding := range record.Results.Findings {
			if finding.Severity != SeverityCritical && finding.Severity != SeverityHigh {
				continue
			}
			threats = append(threats, SecurityThreat{
				ID:          finding.ID,
				Severity:    finding.Severity,
				Title:       finding.Title,
				Description: finding.Description,
				Source:      "vulnerability_scan",
				Target:      finding.Target,
				Timestamp:   finding.DiscoveredAt,
				Metadata: map[string]interface{}{
					"category": finding.Category,
					"scan_id":  record.ScanID,
				},
			})
		}
	}

	slices.SortFunc(threats, func(a, b SecurityThreat) int {
		return cmp.Compare(b.Timestamp.UnixNano(), a.Timestamp.UnixNano())
	})
	if limit > 0 && len(threats) > limit {
		return threats[:limit]
	}
	return threats
}

func (h *SecurityHandlers) buildSecurityEvents(limit int) []SecurityEvent {
	events := make([]SecurityEvent, 0)

	for _, record := range h.completedScanRecords() {
		message := "Security scan completed"
		severity := SeverityInfo
		if record.Results != nil && record.Results.Summary.Critical > 0 {
			severity = SeverityCritical
			message = fmt.Sprintf("Security scan completed with %d critical findings", record.Results.Summary.Critical)
		} else if record.Results != nil && record.Results.Summary.High > 0 {
			severity = SeverityHigh
			message = fmt.Sprintf("Security scan completed with %d high findings", record.Results.Summary.High)
		}

		events = append(events, SecurityEvent{
			ID:        record.ScanID,
			Type:      "scan.completed",
			Severity:  severity,
			Message:   message,
			Source:    "security_scan",
			Timestamp: record.StartedAt,
			Metadata: map[string]interface{}{
				"status":     record.Status,
				"targets":    record.Targets,
				"scan_types": record.ScanTypes,
			},
		})
	}

	auditEvents, err := h.listAuditEvents(context.Background(), &audit.AuditFilter{Limit: limit})
	if err == nil {
		for _, event := range auditEvents {
			severity := SeverityInfo
			if event.Result == audit.ResultFailure {
				severity = SeverityMedium
			}
			events = append(events, SecurityEvent{
				ID:        fmt.Sprintf("audit-%s", event.ID),
				Type:      strings.ToLower(string(event.EventType)),
				Severity:  severity,
				Message:   describeAuditEvent(event),
				Source:    "audit",
				Timestamp: event.Timestamp,
				Metadata: map[string]interface{}{
					"action":   event.Action,
					"result":   event.Result,
					"resource": event.Resource,
					"user_id":  event.UserID,
				},
			})
		}
	}

	slices.SortFunc(events, func(a, b SecurityEvent) int {
		return cmp.Compare(b.Timestamp.UnixNano(), a.Timestamp.UnixNano())
	})
	if limit > 0 && len(events) > limit {
		return events[:limit]
	}
	return events
}

func describeAuditEvent(event *audit.AuditEvent) string {
	if event == nil {
		return "audit event"
	}
	if event.Details != nil {
		if description, ok := event.Details["description"].(string); ok && strings.TrimSpace(description) != "" {
			return description
		}
	}
	return fmt.Sprintf("%s %s on %s", event.UserID, event.Action, event.Resource)
}

func vulnerabilitySummary(records []*scanRecord) map[string]int {
	summary := map[string]int{
		"total":    0,
		"critical": 0,
		"high":     0,
		"medium":   0,
		"low":      0,
		"info":     0,
	}

	for _, record := range records {
		if record == nil || record.Results == nil {
			continue
		}
		for _, finding := range record.Results.Findings {
			summary["total"]++
			severity := strings.ToLower(strings.TrimSpace(string(finding.Severity)))
			if _, ok := summary[severity]; ok {
				summary[severity]++
			}
		}
	}

	return summary
}

func scoreCompliance(records []*scanRecord, threats []SecurityThreat, auditEvents []*audit.AuditEvent) int {
	summary := vulnerabilitySummary(records)
	auditPenalty := 0
	for _, event := range auditEvents {
		if event == nil {
			continue
		}
		switch event.Result {
		case audit.ResultDenied:
			auditPenalty += 4
		case audit.ResultFailure:
			auditPenalty += 2
		}
	}

	score := 100 -
		(summary["critical"] * 14) -
		(summary["high"] * 6) -
		(summary["medium"] * 3) -
		(len(threats) * 8) -
		auditPenalty
	if score < 0 {
		return 0
	}
	if score > 100 {
		return 100
	}
	return score
}

func statusFromScore(score int) string {
	switch {
	case score >= 90:
		return "compliant"
	case score >= 75:
		return "partially_compliant"
	default:
		return "non_compliant"
	}
}

func complianceFrameworks(score int, summary map[string]int, threats []SecurityThreat, auditEvents []*audit.AuditEvent, checkedAt time.Time) []map[string]interface{} {
	accessScore := 100
	denied := 0
	failures := 0
	for _, event := range auditEvents {
		if event == nil {
			continue
		}
		switch event.Result {
		case audit.ResultDenied:
			denied++
			accessScore -= 8
		case audit.ResultFailure:
			failures++
			accessScore -= 4
		}
	}
	if accessScore < 0 {
		accessScore = 0
	}

	vulnerabilityScore := 100 - (summary["critical"] * 18) - (summary["high"] * 8) - (summary["medium"] * 3)
	if vulnerabilityScore < 0 {
		vulnerabilityScore = 0
	}

	incidentScore := 100 - (len(threats) * 12) - (denied * 2)
	if incidentScore < 0 {
		incidentScore = 0
	}

	framework := func(id string, name string, componentScore int, description string) map[string]interface{} {
		return map[string]interface{}{
			"id":           id,
			"name":         name,
			"status":       statusFromScore(componentScore),
			"score":        componentScore,
			"description":  description,
			"last_updated": checkedAt.UTC(),
		}
	}

	return []map[string]interface{}{
		framework(
			"access-control",
			"Access Control",
			accessScore,
			fmt.Sprintf("%d denied and %d failed privileged actions were observed in the audit trail.", denied, failures),
		),
		framework(
			"vulnerability-management",
			"Vulnerability Management",
			vulnerabilityScore,
			fmt.Sprintf("%d critical and %d high findings are currently unresolved.", summary["critical"], summary["high"]),
		),
		framework(
			"incident-response",
			"Incident Response",
			incidentScore,
			fmt.Sprintf("%d active high-severity incidents currently require follow-up.", len(threats)),
		),
		framework(
			"overall-security-posture",
			"Overall Security Posture",
			score,
			"Weighted aggregate of audit, vulnerability, and incident signals from the canonical backend.",
		),
	}
}

func normalizeThreatEvent(threat SecurityThreat) map[string]interface{} {
	status := "active"
	sourceIP := ""
	if threat.Metadata != nil {
		if value, ok := threat.Metadata["status"].(string); ok && strings.TrimSpace(value) != "" {
			status = strings.ToLower(strings.TrimSpace(value))
		}
		if value, ok := threat.Metadata["source_ip"].(string); ok && strings.TrimSpace(value) != "" {
			sourceIP = strings.TrimSpace(value)
		}
	}

	return map[string]interface{}{
		"id":          threat.ID,
		"timestamp":   threat.Timestamp.UTC(),
		"type":        "threat",
		"severity":    strings.ToLower(string(threat.Severity)),
		"source":      threat.Source,
		"resource":    threat.Target,
		"action":      "scan",
		"result":      status,
		"details":     threat.Description,
		"ip":          sourceIP,
		"title":       threat.Title,
		"description": threat.Description,
		"status":      status,
		"source_ip":   sourceIP,
	}
}

func normalizeAuditSeverity(event *audit.AuditEvent) string {
	if event == nil {
		return "low"
	}
	switch event.Result {
	case audit.ResultDenied:
		return "high"
	case audit.ResultFailure:
		return "medium"
	default:
		return "low"
	}
}

func normalizeAuditResult(event *audit.AuditEvent) string {
	if event == nil {
		return "failure"
	}
	switch event.Result {
	case audit.ResultSuccess:
		return "success"
	case audit.ResultDenied:
		return "blocked"
	default:
		return "failure"
	}
}

func normalizeAuditType(event *audit.AuditEvent) string {
	if event == nil {
		return "anomaly"
	}

	resource := strings.ToLower(strings.TrimSpace(event.Resource))
	if resource == "2fa_setup" || resource == "2fa_verification" || strings.Contains(resource, "auth") {
		return "auth"
	}
	if event.Result == audit.ResultDenied {
		return "access"
	}
	switch event.Action {
	case audit.ActionWrite, audit.ActionUpdate, audit.ActionDelete, audit.ActionRotate:
		return "modification"
	default:
		return "anomaly"
	}
}

func normalizeAuditDescription(event *audit.AuditEvent) string {
	if event == nil {
		return "Audit event recorded."
	}
	if event.Details != nil {
		if value, ok := event.Details["description"].(string); ok && strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	if strings.TrimSpace(event.ErrorMsg) != "" {
		return strings.TrimSpace(event.ErrorMsg)
	}

	return fmt.Sprintf("%s %s on %s", strings.ToLower(string(event.Result)), strings.ToLower(string(event.Action)), event.Resource)
}

func normalizeAuditEvent(event *audit.AuditEvent) map[string]interface{} {
	ipAddress := strings.TrimSpace(event.ClientIP)
	if ipAddress == "" {
		ipAddress = strings.TrimSpace(event.IPAddress)
	}

	return map[string]interface{}{
		"id":        event.ID,
		"timestamp": event.Timestamp.UTC(),
		"type":      normalizeAuditType(event),
		"severity":  normalizeAuditSeverity(event),
		"source":    event.Resource,
		"user":      event.Actor,
		"resource":  event.Resource,
		"action":    strings.ToLower(string(event.Action)),
		"result":    normalizeAuditResult(event),
		"details":   normalizeAuditDescription(event),
		"ip":        ipAddress,
	}
}

func incidentSeverityFromAudit(event *audit.AuditEvent) string {
	if event == nil {
		return "medium"
	}
	switch event.Result {
	case audit.ResultDenied:
		return "high"
	case audit.ResultFailure:
		return "medium"
	default:
		return "low"
	}
}

func incidentFromThreat(threat SecurityThreat) map[string]interface{} {
	status := "investigating"
	if threat.Metadata != nil {
		if value, ok := threat.Metadata["status"].(string); ok && strings.TrimSpace(value) != "" {
			status = strings.ToLower(strings.TrimSpace(value))
		}
	}
	if status == "active" {
		status = "investigating"
	}

	sourceIP := ""
	if threat.Metadata != nil {
		if value, ok := threat.Metadata["source_ip"].(string); ok && strings.TrimSpace(value) != "" {
			sourceIP = strings.TrimSpace(value)
		}
	}

	return map[string]interface{}{
		"id":          threat.ID,
		"title":       threat.Title,
		"description": threat.Description,
		"status":      status,
		"severity":    strings.ToLower(string(threat.Severity)),
		"timestamp":   threat.Timestamp.UTC(),
		"createdAt":   threat.Timestamp.UTC(),
		"source":      threat.Source,
		"source_ip":   sourceIP,
		"target":      threat.Target,
	}
}

func incidentFromAudit(event *audit.AuditEvent) map[string]interface{} {
	status := "resolved"
	switch event.Result {
	case audit.ResultDenied:
		status = "blocked"
	case audit.ResultFailure:
		status = "investigating"
	}

	ipAddress := strings.TrimSpace(event.ClientIP)
	if ipAddress == "" {
		ipAddress = strings.TrimSpace(event.IPAddress)
	}

	return map[string]interface{}{
		"id":          fmt.Sprintf("audit-incident-%s", event.ID),
		"title":       strings.Title(strings.ReplaceAll(string(event.EventType), "_", " ")),
		"description": normalizeAuditDescription(event),
		"status":      status,
		"severity":    incidentSeverityFromAudit(event),
		"timestamp":   event.Timestamp.UTC(),
		"createdAt":   event.Timestamp.UTC(),
		"source":      event.Resource,
		"source_ip":   ipAddress,
		"user":        event.Actor,
	}
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

	req.UserID = userIDFromRequest(r, req.UserID)

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
	h.logAudit(ctx, &audit.AuditEvent{
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
	userID := userIDFromRequest(r, "")
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

	req.UserID = userIDFromRequest(r, req.UserID)
	if req.UserID == "" {
		http.Error(w, "user_id is required", http.StatusBadRequest)
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
	h.logAudit(ctx, &audit.AuditEvent{
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

	req.UserID = userIDFromRequest(r, req.UserID)
	if req.UserID == "" {
		http.Error(w, "user_id is required", http.StatusBadRequest)
		return
	}

	if err := h.twoFactorService.VerifyAndEnable(req.UserID, req.Code); err != nil {
		log.Printf("2FA enable error: %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	ctx := r.Context()
	h.logAudit(ctx, &audit.AuditEvent{
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

	req.UserID = userIDFromRequest(r, req.UserID)
	if req.UserID == "" {
		http.Error(w, "user_id is required", http.StatusBadRequest)
		return
	}

	if err := h.twoFactorService.DisableTwoFactor(req.UserID); err != nil {
		log.Printf("2FA disable error: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	ctx := r.Context()
	h.logAudit(ctx, &audit.AuditEvent{
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
	userID := userIDFromRequest(r, "")
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

	req.UserID = userIDFromRequest(r, req.UserID)
	if req.UserID == "" {
		http.Error(w, "user_id is required", http.StatusBadRequest)
		return
	}

	codes, err := h.twoFactorService.RegenerateBackupCodes(req.UserID)
	if err != nil {
		log.Printf("Regenerate backup codes error: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	ctx := r.Context()
	h.logAudit(ctx, &audit.AuditEvent{
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
	userID := userIDFromRequest(r, "")
	if userID == "" {
		http.Error(w, "user_id is required", http.StatusBadRequest)
		return
	}

	info, err := h.twoFactorService.GetUserTwoFactorInfo(userID)
	if err != nil {
		// User doesn't have 2FA setup
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"enabled": false,
			"setup":   false,
		})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"enabled":   info.Enabled,
		"setup":     true,
		"setup_at":  info.SetupAt,
		"last_used": info.LastUsed,
		"algorithm": info.Algorithm,
		"digits":    info.Digits,
		"period":    info.Period,
	})
}

// Security Monitoring Handlers

// GetThreats returns current security threats
func (h *SecurityHandlers) GetThreats(w http.ResponseWriter, r *http.Request) {
	threats := h.buildThreats(100)
	normalized := make([]map[string]interface{}, 0, len(threats))
	for _, threat := range threats {
		normalized = append(normalized, normalizeThreatEvent(threat))
	}

	response := map[string]interface{}{
		"threats":    normalized,
		"count":      len(normalized),
		"updated_at": time.Now(),
	}

	h.respondJSON(w, http.StatusOK, response)
}

// GetVulnerabilities returns vulnerability scan results
func (h *SecurityHandlers) GetVulnerabilities(w http.ResponseWriter, r *http.Request) {
	records := h.completedScanRecords()
	findings := make([]SecurityFinding, 0)
	summary := map[string]int{
		"total":    0,
		"critical": 0,
		"high":     0,
		"medium":   0,
		"low":      0,
		"info":     0,
	}
	var lastScan *time.Time

	for _, record := range records {
		if record.Results == nil {
			continue
		}
		findings = append(findings, record.Results.Findings...)
		if record.CompletedAt != nil {
			if lastScan == nil || record.CompletedAt.After(*lastScan) {
				completedAt := *record.CompletedAt
				lastScan = &completedAt
			}
		}
	}

	slices.SortFunc(findings, func(a, b SecurityFinding) int {
		return cmp.Compare(b.DiscoveredAt.UnixNano(), a.DiscoveredAt.UnixNano())
	})

	for _, finding := range findings {
		summary["total"]++
		severity := strings.ToLower(strings.TrimSpace(string(finding.Severity)))
		if _, ok := summary[severity]; ok {
			summary[severity]++
		}
	}

	response := map[string]interface{}{
		"vulnerabilities": findings,
		"summary":         summary,
		"last_scan":       nil,
	}
	if lastScan != nil {
		response["last_scan"] = lastScan.UTC()
	}

	h.respondJSON(w, http.StatusOK, response)
}

// GetComplianceStatus returns compliance status
func (h *SecurityHandlers) GetComplianceStatus(w http.ResponseWriter, r *http.Request) {
	records := h.completedScanRecords()
	threats := h.buildThreats(200)
	auditEvents, err := h.listAuditEvents(r.Context(), &audit.AuditFilter{Limit: 250})
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	checkedAt := time.Now().UTC()
	summary := vulnerabilitySummary(records)
	score := scoreCompliance(records, threats, auditEvents)

	response := map[string]interface{}{
		"score":            score,
		"compliance_score": score,
		"frameworks":       complianceFrameworks(score, summary, threats, auditEvents, checkedAt),
		"last_updated":     checkedAt,
		"summary": map[string]interface{}{
			"vulnerabilities": summary,
			"active_threats":  len(threats),
			"audit_events":    len(auditEvents),
		},
	}

	h.respondJSON(w, http.StatusOK, response)
}

// GetIncidents returns security incidents
func (h *SecurityHandlers) GetIncidents(w http.ResponseWriter, r *http.Request) {
	limit := 50
	if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 && l <= 1000 {
			limit = l
		}
	}

	threats := h.buildThreats(limit)
	incidents := make([]map[string]interface{}, 0, limit)
	for _, threat := range threats {
		incidents = append(incidents, incidentFromThreat(threat))
	}

	auditEvents, err := h.listAuditEvents(r.Context(), &audit.AuditFilter{Limit: limit})
	if err == nil {
		for _, event := range auditEvents {
			if event == nil || (event.Result != audit.ResultDenied && event.Result != audit.ResultFailure) {
				continue
			}
			incidents = append(incidents, incidentFromAudit(event))
		}
	}

	slices.SortFunc(incidents, func(a, b map[string]interface{}) int {
		left, _ := a["timestamp"].(time.Time)
		right, _ := b["timestamp"].(time.Time)
		return cmp.Compare(right.UnixNano(), left.UnixNano())
	})
	if len(incidents) > limit {
		incidents = incidents[:limit]
	}

	response := map[string]interface{}{
		"incidents":  incidents,
		"total":      len(incidents),
		"limit":      limit,
		"updated_at": time.Now().UTC(),
	}

	h.respondJSON(w, http.StatusOK, response)
}

// GetSecurityEvents returns recent security events
func (h *SecurityHandlers) GetSecurityEvents(w http.ResponseWriter, r *http.Request) {
	limit := 100
	if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 && l <= 1000 {
			limit = l
		}
	}

	events := make([]map[string]interface{}, 0, limit)
	for _, threat := range h.buildThreats(limit) {
		events = append(events, normalizeThreatEvent(threat))
	}

	filter := &audit.AuditFilter{Limit: limit}
	if userID := strings.TrimSpace(r.URL.Query().Get("user_id")); userID != "" {
		filter.UserID = userID
	}
	if userID := strings.TrimSpace(r.URL.Query().Get("user")); userID != "" && filter.UserID == "" {
		filter.UserID = userID
	}
	if action := strings.TrimSpace(r.URL.Query().Get("action")); action != "" {
		filter.Action = action
	}
	if resource := strings.TrimSpace(r.URL.Query().Get("resource")); resource != "" {
		filter.Resource = resource
	}

	auditEvents, err := h.listAuditEvents(r.Context(), filter)
	if err == nil {
		for _, event := range auditEvents {
			events = append(events, normalizeAuditEvent(event))
		}
	}

	if severity := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("severity"))); severity != "" && severity != "all" {
		filtered := events[:0]
		for _, event := range events {
			if eventSeverity, ok := event["severity"].(string); ok && eventSeverity == severity {
				filtered = append(filtered, event)
			}
		}
		events = filtered
	}
	if eventType := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("type"))); eventType != "" && eventType != "all" {
		filtered := events[:0]
		for _, event := range events {
			if normalizedType, ok := event["type"].(string); ok && normalizedType == eventType {
				filtered = append(filtered, event)
			}
		}
		events = filtered
	}

	slices.SortFunc(events, func(a, b map[string]interface{}) int {
		left, _ := a["timestamp"].(time.Time)
		right, _ := b["timestamp"].(time.Time)
		return cmp.Compare(right.UnixNano(), left.UnixNano())
	})
	if len(events) > limit {
		events = events[:limit]
	}

	response := map[string]interface{}{
		"events": events,
		"total":  len(events),
		"limit":  limit,
	}

	h.respondJSON(w, http.StatusOK, response)
}

// StartVulnerabilityScan starts a vulnerability scan
func (h *SecurityHandlers) StartVulnerabilityScan(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Targets   []string               `json:"targets"`
		ScanTypes []ScanType             `json:"scan_types"`
		Config    map[string]interface{} `json:"config"`
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
	record := &scanRecord{
		ScanID:    scanID,
		Status:    "started",
		Targets:   append([]string(nil), req.Targets...),
		ScanTypes: append([]ScanType(nil), normalizeScanTypes(req.ScanTypes)...),
		StartedAt: time.Now().UTC(),
	}
	h.saveScanRecord(record)

	go func() {
		results, err := runLocalScan(context.Background(), scanID, req.Targets, req.ScanTypes, record.StartedAt)
		completedAt := time.Now().UTC()
		record.CompletedAt = &completedAt
		if err != nil {
			log.Printf("Vulnerability scan failed: %v", err)
			record.Status = "failed"
			record.Error = err.Error()
		} else {
			log.Printf("Vulnerability scan completed: %s", results.ScanID)
			results.ScanID = scanID
			record.Status = "completed"
			record.Results = results
		}
		h.saveScanRecord(record)
	}()

	response := map[string]interface{}{
		"scan_id":    scanID,
		"status":     "started",
		"targets":    req.Targets,
		"scan_types": req.ScanTypes,
		"started_at": record.StartedAt,
	}

	h.respondJSON(w, http.StatusAccepted, response)
}

// GetScanResults returns vulnerability scan results
func (h *SecurityHandlers) GetScanResults(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	scanID := vars["scanId"]

	record, ok := h.getScanRecord(scanID)
	if !ok {
		http.Error(w, "scan not found", http.StatusNotFound)
		return
	}

	response := map[string]interface{}{
		"scan_id":      record.ScanID,
		"status":       record.Status,
		"targets":      record.Targets,
		"scan_types":   record.ScanTypes,
		"started_at":   record.StartedAt,
		"completed_at": record.CompletedAt,
		"error":        record.Error,
		"results":      record.Results,
	}

	h.respondJSON(w, http.StatusOK, response)
}

// GetClusterSecurityState returns security state for a specific cluster
func (h *SecurityHandlers) GetClusterSecurityState(w http.ResponseWriter, r *http.Request) {
	h.respondUnsupported(w, r, "distributed cluster security state")
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

	filter := &audit.AuditFilter{
		Limit:    limit,
		UserID:   strings.TrimSpace(r.URL.Query().Get("user_id")),
		Action:   strings.TrimSpace(r.URL.Query().Get("action")),
		Resource: strings.TrimSpace(r.URL.Query().Get("resource")),
	}
	if filter.UserID == "" {
		filter.UserID = strings.TrimSpace(r.URL.Query().Get("user"))
	}
	events, err := h.listAuditEvents(r.Context(), filter)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	slices.SortFunc(events, func(a, b *audit.AuditEvent) int {
		return cmp.Compare(b.Timestamp.UnixNano(), a.Timestamp.UnixNano())
	})
	if len(events) > limit {
		events = events[:limit]
	}

	normalized := make([]map[string]interface{}, 0, len(events))
	for _, event := range events {
		normalized = append(normalized, normalizeAuditEvent(event))
	}

	response := map[string]interface{}{
		"events": normalized,
		"total":  len(normalized),
		"limit":  limit,
	}

	h.respondJSON(w, http.StatusOK, response)
}

// ExportAuditLog exports audit log
func (h *SecurityHandlers) ExportAuditLog(w http.ResponseWriter, r *http.Request) {
	format := r.URL.Query().Get("format")
	if format == "" {
		format = "json"
	}

	events, err := h.listAuditEvents(r.Context(), &audit.AuditFilter{})
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	slices.SortFunc(events, func(a, b *audit.AuditEvent) int {
		return cmp.Compare(a.Timestamp.UnixNano(), b.Timestamp.UnixNano())
	})

	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=audit-log.%s", format))

	switch format {
	case "csv":
		w.Header().Set("Content-Type", "text/csv")
		_, _ = w.Write([]byte("timestamp,event_type,user_id,action,result,resource,description\n"))
		for _, event := range events {
			description := ""
			if event.Details != nil {
				if value, ok := event.Details["description"].(string); ok {
					description = strings.ReplaceAll(value, ",", " ")
				}
			}
			_, _ = w.Write([]byte(fmt.Sprintf("%s,%s,%s,%s,%s,%s,%s\n",
				event.Timestamp.UTC().Format(time.RFC3339),
				event.EventType,
				event.UserID,
				event.Action,
				event.Result,
				event.Resource,
				description,
			)))
		}
	default:
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"events":       events,
			"exported_at":  time.Now().UTC(),
			"total_events": len(events),
		})
	}
}

// GetAuditStatistics returns audit statistics
func (h *SecurityHandlers) GetAuditStatistics(w http.ResponseWriter, r *http.Request) {
	period := r.URL.Query().Get("period")
	if period == "" {
		period = "last_30_days"
	}

	events, err := h.listAuditEvents(r.Context(), &audit.AuditFilter{})
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	eventsByType := make(map[string]int)
	eventsByResult := make(map[string]int)
	eventsByAction := make(map[string]int)
	successCount := 0
	for _, event := range events {
		eventsByType[string(event.EventType)]++
		eventsByResult[string(event.Result)]++
		eventsByAction[string(event.Action)]++
		if event.Result == audit.ResultSuccess {
			successCount++
		}
	}

	overallScore := 100
	if len(events) > 0 {
		overallScore = int(float64(successCount) / float64(len(events)) * 100)
	}

	response := map[string]interface{}{
		"total_events":     len(events),
		"events_by_type":   eventsByType,
		"events_by_result": eventsByResult,
		"events_by_action": eventsByAction,
		"overallScore":     overallScore,
		"period":           period,
	}

	h.respondJSON(w, http.StatusOK, response)
}

// RBAC Handlers

// GetRoles returns available roles
func (h *SecurityHandlers) GetRoles(w http.ResponseWriter, r *http.Request) {
	if h.rbacStore == nil {
		h.respondUnsupported(w, r, "RBAC role enumeration")
		return
	}

	roles, err := h.rbacStore.ListRoles(r.Context())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	h.respondJSON(w, http.StatusOK, map[string]interface{}{"roles": roles})
}

// GetPermissions returns available permissions
func (h *SecurityHandlers) GetPermissions(w http.ResponseWriter, r *http.Request) {
	if h.rbacStore == nil {
		h.respondUnsupported(w, r, "RBAC permission enumeration")
		return
	}

	permissions, err := h.rbacStore.ListPermissions(r.Context())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	h.respondJSON(w, http.StatusOK, map[string]interface{}{"permissions": permissions})
}

// GetUserRoles returns roles for a specific user
func (h *SecurityHandlers) GetUserRoles(w http.ResponseWriter, r *http.Request) {
	if h.rbacStore == nil {
		h.respondUnsupported(w, r, "user role lookup")
		return
	}

	vars := mux.Vars(r)
	userID := vars["userId"]

	roles, err := h.rbacStore.GetUserRoles(r.Context(), userID)
	if err != nil {
		status := http.StatusInternalServerError
		if err == sql.ErrNoRows {
			status = http.StatusNotFound
		}
		http.Error(w, err.Error(), status)
		return
	}

	h.respondJSON(w, http.StatusOK, map[string]interface{}{"user_id": userID, "roles": roles})
}

// AssignUserRoles assigns roles to a user
func (h *SecurityHandlers) AssignUserRoles(w http.ResponseWriter, r *http.Request) {
	if h.rbacStore == nil {
		h.respondUnsupported(w, r, "user role assignment")
		return
	}

	vars := mux.Vars(r)
	userID := vars["userId"]

	var req struct {
		Roles []string `json:"roles"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	roles, err := h.rbacStore.AssignUserRoles(r.Context(), userID, req.Roles)
	if err != nil {
		status := http.StatusInternalServerError
		if err == sql.ErrNoRows {
			status = http.StatusNotFound
		} else if strings.Contains(err.Error(), "at least one role") || strings.Contains(err.Error(), "unsupported role") {
			status = http.StatusBadRequest
		}
		http.Error(w, err.Error(), status)
		return
	}

	ctx := r.Context()
	h.logAudit(ctx, &audit.AuditEvent{
		UserID:   userID,
		Resource: "user_roles",
		Action:   audit.ActionUpdate,
		Result:   audit.ResultSuccess,
		Details:  map[string]interface{}{"roles": req.Roles, "description": "User roles assigned"},
	})

	response := map[string]interface{}{
		"user_id": userID,
		"roles":   roles,
		"success": true,
	}

	h.respondJSON(w, http.StatusOK, response)
}

// GetUserPermissions returns effective permissions for a user
func (h *SecurityHandlers) GetUserPermissions(w http.ResponseWriter, r *http.Request) {
	if h.rbacStore == nil {
		h.respondUnsupported(w, r, "user permission lookup")
		return
	}

	vars := mux.Vars(r)
	userID := vars["userId"]

	permissions, err := h.rbacStore.GetUserPermissions(r.Context(), userID)
	if err != nil {
		status := http.StatusInternalServerError
		if err == sql.ErrNoRows {
			status = http.StatusNotFound
		}
		http.Error(w, err.Error(), status)
		return
	}

	h.respondJSON(w, http.StatusOK, map[string]interface{}{"user_id": userID, "permissions": permissions})
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
	h.logAudit(ctx, &audit.AuditEvent{
		UserID:   userIDFromRequest(r, ""),
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
			events := h.buildSecurityEvents(20)
			threats := h.buildThreats(20)
			summary := map[string]interface{}{
				"timestamp":            time.Now().UTC(),
				"events":               events,
				"threats":              threats,
				"total_events":         len(events),
				"total_active_threats": len(threats),
			}

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
