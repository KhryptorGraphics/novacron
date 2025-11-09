package admin

import (
	"database/sql"

	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
)

// AdminHandlers aggregates all admin API handlers
type AdminHandlers struct {
	UserManagement *UserManagementHandlers
	Security       *SecurityHandlers
	Config         *ConfigHandlers
	Database       *DatabaseHandlers
	Templates      *TemplateHandlers
}

// NewAdminHandlers creates a new admin handlers instance
func NewAdminHandlers(db *sql.DB, configPath string) *AdminHandlers {
	return &AdminHandlers{
		UserManagement: NewUserManagementHandlers(db),
		Security:       NewSecurityHandlers(db),
		Config:         NewConfigHandlers(configPath),
		Database:       NewDatabaseHandlers(db),
		Templates:      NewTemplateHandlers(db),
	}
}

// RegisterRoutes registers all admin API routes
func (h *AdminHandlers) RegisterRoutes(router *mux.Router) {
	logger.Info("Registering admin API routes")

	// Create admin subrouter
	admin := router.PathPrefix("/api/admin").Subrouter()

	// User Management routes
	admin.HandleFunc("/users", h.UserManagement.ListUsers).Methods("GET")
	admin.HandleFunc("/users", h.UserManagement.CreateUser).Methods("POST")
	admin.HandleFunc("/users/{id}", h.UserManagement.GetUser).Methods("GET")
	admin.HandleFunc("/users/{id}", h.UserManagement.UpdateUser).Methods("PUT")
	admin.HandleFunc("/users/{id}", h.UserManagement.DeleteUser).Methods("DELETE")
	admin.HandleFunc("/users/{id}/roles", h.UserManagement.AssignRoles).Methods("POST")
	admin.HandleFunc("/users/bulk", h.UserManagement.BulkOperation).Methods("POST")

	// Security routes
	admin.HandleFunc("/security/metrics", h.Security.GetSecurityMetrics).Methods("GET")
	admin.HandleFunc("/security/alerts", h.Security.ListSecurityAlerts).Methods("GET")
	admin.HandleFunc("/security/alerts/{id}", h.Security.GetSecurityAlert).Methods("GET")
	admin.HandleFunc("/security/alerts/{id}", h.Security.UpdateSecurityAlert).Methods("PUT")
	admin.HandleFunc("/security/audit", h.Security.ListAuditLogs).Methods("GET")
	admin.HandleFunc("/security/audit/{id}", h.Security.GetAuditLog).Methods("GET")
	admin.HandleFunc("/security/policies", h.Security.ListSecurityPolicies).Methods("GET")
	admin.HandleFunc("/security/policies/{id}", h.Security.GetSecurityPolicy).Methods("GET")
	admin.HandleFunc("/security/policies/{id}", h.Security.UpdateSecurityPolicy).Methods("PUT")

	// System Configuration routes
	admin.HandleFunc("/config", h.Config.GetConfig).Methods("GET")
	admin.HandleFunc("/config", h.Config.UpdateConfig).Methods("PUT")
	admin.HandleFunc("/config/validate", h.Config.ValidateConfig).Methods("POST")
	admin.HandleFunc("/config/backup", h.Config.CreateBackup).Methods("POST")
	admin.HandleFunc("/config/backups", h.Config.ListBackups).Methods("GET")
	admin.HandleFunc("/config/restore/{id}", h.Config.RestoreBackup).Methods("POST")

	// Database Administration routes
	admin.HandleFunc("/database/tables", h.Database.ListTables).Methods("GET")
	admin.HandleFunc("/database/tables/{table}", h.Database.GetTableDetails).Methods("GET")
	admin.HandleFunc("/database/query", h.Database.ExecuteQuery).Methods("POST")
	admin.HandleFunc("/database/execute", h.Database.ExecuteStatement).Methods("POST")

	// VM Template routes
	admin.HandleFunc("/templates", h.Templates.ListTemplates).Methods("GET")
	admin.HandleFunc("/templates", h.Templates.CreateTemplate).Methods("POST")
	admin.HandleFunc("/templates/{id}", h.Templates.GetTemplate).Methods("GET")
	admin.HandleFunc("/templates/{id}", h.Templates.UpdateTemplate).Methods("PUT")
	admin.HandleFunc("/templates/{id}", h.Templates.DeleteTemplate).Methods("DELETE")

	logger.Info("Admin API routes registered successfully")
}
