package admin

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
)

type SecurityHandlers struct {
	db *sql.DB
}

type SecurityAlert struct {
	ID          int       `json:"id"`
	Type        string    `json:"type"`
	Severity    string    `json:"severity"`
	Title       string    `json:"title"`
	Description string    `json:"description"`
	Source      string    `json:"source"`
	IP          string    `json:"ip,omitempty"`
	UserAgent   string    `json:"user_agent,omitempty"`
	Status      string    `json:"status"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

type AuditLogEntry struct {
	ID        int                    `json:"id"`
	UserID    int                    `json:"user_id"`
	Username  string                 `json:"username"`
	Action    string                 `json:"action"`
	Resource  string                 `json:"resource"`
	Details   map[string]interface{} `json:"details"`
	IP        string                 `json:"ip"`
	UserAgent string                 `json:"user_agent"`
	Success   bool                   `json:"success"`
	CreatedAt time.Time              `json:"created_at"`
}

type SecurityMetrics struct {
	TotalAlerts      int              `json:"total_alerts"`
	CriticalAlerts   int              `json:"critical_alerts"`
	FailedLogins     int              `json:"failed_logins_24h"`
	ActiveSessions   int              `json:"active_sessions"`
	LastBreach       *time.Time       `json:"last_breach,omitempty"`
	AlertsByType     map[string]int   `json:"alerts_by_type"`
	AlertsBySeverity map[string]int   `json:"alerts_by_severity"`
	RecentActivities []AuditLogEntry  `json:"recent_activities"`
	TopIPs           []IPStatistic    `json:"top_ips"`
}

type IPStatistic struct {
	IP      string `json:"ip"`
	Count   int    `json:"count"`
	Country string `json:"country,omitempty"`
}

type SecurityPolicy struct {
	ID                  int    `json:"id"`
	Name                string `json:"name"`
	Description         string `json:"description"`
	Enabled             bool   `json:"enabled"`
	MaxLoginAttempts    int    `json:"max_login_attempts"`
	LockoutDuration     int    `json:"lockout_duration_minutes"`
	SessionTimeout      int    `json:"session_timeout_minutes"`
	PasswordMinLength   int    `json:"password_min_length"`
	PasswordRequireSpec bool   `json:"password_require_special"`
	RequireMFA          bool   `json:"require_mfa"`
	AllowedIPs          string `json:"allowed_ips,omitempty"`
	BlockedIPs          string `json:"blocked_ips,omitempty"`
}

func NewSecurityHandlers(db *sql.DB) *SecurityHandlers {
	return &SecurityHandlers{db: db}
}

// GET /api/admin/security/metrics - Get security metrics overview
func (h *SecurityHandlers) GetSecurityMetrics(w http.ResponseWriter, r *http.Request) {
	metrics := SecurityMetrics{
		AlertsByType:     make(map[string]int),
		AlertsBySeverity: make(map[string]int),
	}
	
	// Get total alerts
	err := h.db.QueryRow("SELECT COUNT(*) FROM security_alerts WHERE created_at > NOW() - INTERVAL '30 days'").Scan(&metrics.TotalAlerts)
	if err != nil {
		logger.Error("Failed to get total alerts", "error", err)
	}
	
	// Get critical alerts
	err = h.db.QueryRow("SELECT COUNT(*) FROM security_alerts WHERE severity = 'critical' AND created_at > NOW() - INTERVAL '7 days'").Scan(&metrics.CriticalAlerts)
	if err != nil {
		logger.Error("Failed to get critical alerts", "error", err)
	}
	
	// Get failed logins in last 24 hours
	err = h.db.QueryRow(`
		SELECT COUNT(*) FROM audit_logs 
		WHERE action = 'login' AND success = false 
		AND created_at > NOW() - INTERVAL '24 hours'
	`).Scan(&metrics.FailedLogins)
	if err != nil {
		logger.Error("Failed to get failed logins", "error", err)
	}
	
	// Get active sessions (mock data for now)
	metrics.ActiveSessions = 42
	
	// Get alerts by type
	typeRows, err := h.db.Query(`
		SELECT type, COUNT(*) 
		FROM security_alerts 
		WHERE created_at > NOW() - INTERVAL '30 days'
		GROUP BY type
	`)
	if err == nil {
		defer typeRows.Close()
		for typeRows.Next() {
			var alertType string
			var count int
			if err := typeRows.Scan(&alertType, &count); err == nil {
				metrics.AlertsByType[alertType] = count
			}
		}
	}
	
	// Get alerts by severity
	severityRows, err := h.db.Query(`
		SELECT severity, COUNT(*) 
		FROM security_alerts 
		WHERE created_at > NOW() - INTERVAL '30 days'
		GROUP BY severity
	`)
	if err == nil {
		defer severityRows.Close()
		for severityRows.Next() {
			var severity string
			var count int
			if err := severityRows.Scan(&severity, &count); err == nil {
				metrics.AlertsBySeverity[severity] = count
			}
		}
	}
	
	// Get recent activities
	activityRows, err := h.db.Query(`
		SELECT 
			al.id, al.user_id, u.username, al.action, al.resource,
			al.details, al.ip, al.user_agent, al.success, al.created_at
		FROM audit_logs al
		LEFT JOIN users u ON u.id = al.user_id
		ORDER BY al.created_at DESC
		LIMIT 10
	`)
	if err == nil {
		defer activityRows.Close()
		for activityRows.Next() {
			var activity AuditLogEntry
			var details sql.NullString
			var username sql.NullString
			
			err := activityRows.Scan(&activity.ID, &activity.UserID, &username,
				&activity.Action, &activity.Resource, &details,
				&activity.IP, &activity.UserAgent, &activity.Success, &activity.CreatedAt)
			
			if err == nil {
				if username.Valid {
					activity.Username = username.String
				}
				
				if details.Valid {
					json.Unmarshal([]byte(details.String), &activity.Details)
				} else {
					activity.Details = make(map[string]interface{})
				}
				
				metrics.RecentActivities = append(metrics.RecentActivities, activity)
			}
		}
	}
	
	// Get top IPs by activity
	ipRows, err := h.db.Query(`
		SELECT ip, COUNT(*) as count
		FROM audit_logs
		WHERE created_at > NOW() - INTERVAL '24 hours'
		AND ip IS NOT NULL
		GROUP BY ip
		ORDER BY count DESC
		LIMIT 10
	`)
	if err == nil {
		defer ipRows.Close()
		for ipRows.Next() {
			var ipStat IPStatistic
			if err := ipRows.Scan(&ipStat.IP, &ipStat.Count); err == nil {
				metrics.TopIPs = append(metrics.TopIPs, ipStat)
			}
		}
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

// GET /api/admin/security/alerts - Get security alerts
func (h *SecurityHandlers) GetSecurityAlerts(w http.ResponseWriter, r *http.Request) {
	page, _ := strconv.Atoi(r.URL.Query().Get("page"))
	if page <= 0 {
		page = 1
	}
	
	pageSize, _ := strconv.Atoi(r.URL.Query().Get("page_size"))
	if pageSize <= 0 || pageSize > 100 {
		pageSize = 20
	}
	
	severity := r.URL.Query().Get("severity")
	status := r.URL.Query().Get("status")
	
	offset := (page - 1) * pageSize
	
	// Build query
	query := `
		SELECT id, type, severity, title, description, source, ip, user_agent, status, created_at, updated_at
		FROM security_alerts
		WHERE 1=1
	`
	args := []interface{}{}
	argIndex := 1
	
	if severity != "" {
		query += " AND severity = $" + strconv.Itoa(argIndex)
		args = append(args, severity)
		argIndex++
	}
	
	if status != "" {
		query += " AND status = $" + strconv.Itoa(argIndex)
		args = append(args, status)
		argIndex++
	}
	
	query += " ORDER BY created_at DESC LIMIT $" + strconv.Itoa(argIndex) + " OFFSET $" + strconv.Itoa(argIndex+1)
	args = append(args, pageSize, offset)
	
	rows, err := h.db.Query(query, args...)
	if err != nil {
		logger.Error("Failed to query security alerts", "error", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}
	defer rows.Close()
	
	alerts := []SecurityAlert{}
	for rows.Next() {
		var alert SecurityAlert
		var ip, userAgent sql.NullString
		
		err := rows.Scan(&alert.ID, &alert.Type, &alert.Severity, &alert.Title,
			&alert.Description, &alert.Source, &ip, &userAgent,
			&alert.Status, &alert.CreatedAt, &alert.UpdatedAt)
		
		if err != nil {
			logger.Error("Failed to scan alert", "error", err)
			continue
		}
		
		if ip.Valid {
			alert.IP = ip.String
		}
		if userAgent.Valid {
			alert.UserAgent = userAgent.String
		}
		
		alerts = append(alerts, alert)
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"alerts": alerts,
		"page":   page,
		"count":  len(alerts),
	})
}

// GET /api/admin/security/audit - Get audit logs
func (h *SecurityHandlers) GetAuditLogs(w http.ResponseWriter, r *http.Request) {
	page, _ := strconv.Atoi(r.URL.Query().Get("page"))
	if page <= 0 {
		page = 1
	}
	
	pageSize, _ := strconv.Atoi(r.URL.Query().Get("page_size"))
	if pageSize <= 0 || pageSize > 100 {
		pageSize = 50
	}
	
	action := r.URL.Query().Get("action")
	userID := r.URL.Query().Get("user_id")
	
	offset := (page - 1) * pageSize
	
	query := `
		SELECT 
			al.id, al.user_id, u.username, al.action, al.resource,
			al.details, al.ip, al.user_agent, al.success, al.created_at
		FROM audit_logs al
		LEFT JOIN users u ON u.id = al.user_id
		WHERE 1=1
	`
	args := []interface{}{}
	argIndex := 1
	
	if action != "" {
		query += " AND al.action = $" + strconv.Itoa(argIndex)
		args = append(args, action)
		argIndex++
	}
	
	if userID != "" {
		query += " AND al.user_id = $" + strconv.Itoa(argIndex)
		args = append(args, userID)
		argIndex++
	}
	
	query += " ORDER BY al.created_at DESC LIMIT $" + strconv.Itoa(argIndex) + " OFFSET $" + strconv.Itoa(argIndex+1)
	args = append(args, pageSize, offset)
	
	rows, err := h.db.Query(query, args...)
	if err != nil {
		logger.Error("Failed to query audit logs", "error", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}
	defer rows.Close()
	
	logs := []AuditLogEntry{}
	for rows.Next() {
		var log AuditLogEntry
		var details sql.NullString
		var username sql.NullString
		
		err := rows.Scan(&log.ID, &log.UserID, &username, &log.Action,
			&log.Resource, &details, &log.IP, &log.UserAgent,
			&log.Success, &log.CreatedAt)
		
		if err != nil {
			logger.Error("Failed to scan audit log", "error", err)
			continue
		}
		
		if username.Valid {
			log.Username = username.String
		}
		
		if details.Valid {
			json.Unmarshal([]byte(details.String), &log.Details)
		} else {
			log.Details = make(map[string]interface{})
		}
		
		logs = append(logs, log)
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"logs":  logs,
		"page":  page,
		"count": len(logs),
	})
}

// GET /api/admin/security/policies - Get security policies
func (h *SecurityHandlers) GetSecurityPolicies(w http.ResponseWriter, r *http.Request) {
	// For now, return a default policy
	// In production, these would be stored in the database
	policies := []SecurityPolicy{
		{
			ID:                  1,
			Name:                "Default Security Policy",
			Description:         "Default security settings for the NovaCron system",
			Enabled:             true,
			MaxLoginAttempts:    5,
			LockoutDuration:     30,
			SessionTimeout:      120,
			PasswordMinLength:   8,
			PasswordRequireSpec: true,
			RequireMFA:          false,
			AllowedIPs:          "",
			BlockedIPs:          "",
		},
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"policies": policies,
		"count":    len(policies),
	})
}

// PUT /api/admin/security/policies/{id} - Update security policy
func (h *SecurityHandlers) UpdateSecurityPolicy(w http.ResponseWriter, r *http.Request) {
	var policy SecurityPolicy
	if err := json.NewDecoder(r.Body).Decode(&policy); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}
	
	// For now, just return the updated policy
	// In production, this would update the database
	policy.ID = 1 // Mock ID
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(policy)
}

// Helper function to create a security alert
func (h *SecurityHandlers) CreateSecurityAlert(alertType, severity, title, description, source, ip, userAgent string) error {
	_, err := h.db.Exec(`
		INSERT INTO security_alerts (type, severity, title, description, source, ip, user_agent, status, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, 'open', NOW(), NOW())
	`, alertType, severity, title, description, source, ip, userAgent)
	
	if err != nil {
		logger.Error("Failed to create security alert", "error", err)
		return err
	}
	
	return nil
}

// Helper function to log audit entry
func (h *SecurityHandlers) LogAuditEntry(userID int, action, resource string, details map[string]interface{}, ip, userAgent string, success bool) error {
	detailsJSON, _ := json.Marshal(details)
	
	_, err := h.db.Exec(`
		INSERT INTO audit_logs (user_id, action, resource, details, ip, user_agent, success, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
	`, userID, action, resource, string(detailsJSON), ip, userAgent, success)
	
	if err != nil {
		logger.Error("Failed to create audit log entry", "error", err)
		return err
	}
	
	return nil
}