package handlers

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"sort"
	"strings"
)

type RoleDefinition struct {
	ID          string   `json:"id"`
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Permissions []string `json:"permissions"`
}

type PermissionDefinition struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
}

type UserRoleStore interface {
	ListRoles(ctx context.Context) ([]RoleDefinition, error)
	ListPermissions(ctx context.Context) ([]PermissionDefinition, error)
	GetUserRoles(ctx context.Context, userID string) ([]string, error)
	AssignUserRoles(ctx context.Context, userID string, roles []string) ([]string, error)
	GetUserPermissions(ctx context.Context, userID string) ([]string, error)
}

type PostgresRBACStore struct {
	db *sql.DB
}

func NewPostgresRBACStore(db *sql.DB) *PostgresRBACStore {
	return &PostgresRBACStore{db: db}
}

var permissionCatalog = []PermissionDefinition{
	{ID: "read", Name: "Read", Description: "Read access to resources"},
	{ID: "write", Name: "Write", Description: "Write access to resources"},
	{ID: "delete", Name: "Delete", Description: "Delete access to resources"},
	{ID: "admin", Name: "Admin", Description: "Administrative access"},
	{ID: "audit.read", Name: "Audit Read", Description: "Read access to audit events and exports"},
	{ID: "monitoring.read", Name: "Monitoring Read", Description: "Read access to monitoring and metrics streams"},
	{ID: "rbac.manage", Name: "RBAC Manage", Description: "Manage role assignments"},
	{ID: "security.read", Name: "Security Read", Description: "Read access to security dashboards and alerts"},
	{ID: "security.write", Name: "Security Write", Description: "Start scans and mutate security settings"},
	{ID: "vm.console", Name: "VM Console", Description: "Access VM console websocket sessions"},
	{ID: "vm.manage", Name: "VM Manage", Description: "Manage VM lifecycle operations"},
}

var roleCatalog = map[string]RoleDefinition{
	"super-admin": {
		ID:          "super-admin",
		Name:        "Super Admin",
		Description: "Full platform administration",
		Permissions: []string{"*"},
	},
	"admin": {
		ID:          "admin",
		Name:        "Administrator",
		Description: "Administrative access to platform and security surfaces",
		Permissions: []string{"read", "write", "delete", "admin", "audit.read", "monitoring.read", "rbac.manage", "security.read", "security.write", "vm.console", "vm.manage"},
	},
	"operator": {
		ID:          "operator",
		Name:        "Operator",
		Description: "Operational access to consoles and runtime controls",
		Permissions: []string{"read", "monitoring.read", "security.read", "vm.console", "vm.manage"},
	},
	"viewer": {
		ID:          "viewer",
		Name:        "Viewer",
		Description: "Read-only access to monitoring and security telemetry",
		Permissions: []string{"read", "monitoring.read", "security.read"},
	},
	"readonly": {
		ID:          "readonly",
		Name:        "Read Only",
		Description: "Read-only access to platform data",
		Permissions: []string{"read"},
	},
	"user": {
		ID:          "user",
		Name:        "User",
		Description: "Standard user access",
		Permissions: []string{"read", "write"},
	},
}

func (s *PostgresRBACStore) ListRoles(ctx context.Context) ([]RoleDefinition, error) {
	_ = ctx

	roles := make([]RoleDefinition, 0, len(roleCatalog))
	for _, role := range roleCatalog {
		roleCopy := role
		roleCopy.Permissions = append([]string(nil), role.Permissions...)
		roles = append(roles, roleCopy)
	}

	sort.Slice(roles, func(i, j int) bool {
		return roles[i].ID < roles[j].ID
	})

	return roles, nil
}

func (s *PostgresRBACStore) ListPermissions(ctx context.Context) ([]PermissionDefinition, error) {
	_ = ctx

	permissions := append([]PermissionDefinition(nil), permissionCatalog...)
	sort.Slice(permissions, func(i, j int) bool {
		return permissions[i].ID < permissions[j].ID
	})
	return permissions, nil
}

func (s *PostgresRBACStore) GetUserRoles(ctx context.Context, userID string) ([]string, error) {
	role, err := s.getUserRole(ctx, userID)
	if err != nil {
		return nil, err
	}
	return []string{role}, nil
}

func (s *PostgresRBACStore) AssignUserRoles(ctx context.Context, userID string, roles []string) ([]string, error) {
	if len(roles) == 0 {
		return nil, errors.New("at least one role is required")
	}

	role := normalizeRoleName(roles[0])
	if _, ok := roleCatalog[role]; !ok {
		return nil, fmt.Errorf("unsupported role %q", roles[0])
	}

	result, err := s.db.ExecContext(ctx, "UPDATE users SET role = $1, updated_at = NOW() WHERE id = $2", role, userID)
	if err != nil {
		return nil, fmt.Errorf("failed to assign roles: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return nil, fmt.Errorf("failed to confirm role update: %w", err)
	}
	if rowsAffected == 0 {
		return nil, sql.ErrNoRows
	}

	return []string{role}, nil
}

func (s *PostgresRBACStore) GetUserPermissions(ctx context.Context, userID string) ([]string, error) {
	roles, err := s.GetUserRoles(ctx, userID)
	if err != nil {
		return nil, err
	}

	permissionSet := make(map[string]struct{})
	for _, role := range roles {
		definition, ok := roleCatalog[normalizeRoleName(role)]
		if !ok {
			continue
		}
		for _, permission := range definition.Permissions {
			permissionSet[permission] = struct{}{}
		}
	}

	permissions := make([]string, 0, len(permissionSet))
	for permission := range permissionSet {
		permissions = append(permissions, permission)
	}
	sort.Strings(permissions)
	return permissions, nil
}

func (s *PostgresRBACStore) getUserRole(ctx context.Context, userID string) (string, error) {
	if strings.TrimSpace(userID) == "" {
		return "", errors.New("user ID is required")
	}

	var role string
	if err := s.db.QueryRowContext(ctx, "SELECT role FROM users WHERE id = $1", userID).Scan(&role); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return "", sql.ErrNoRows
		}
		return "", fmt.Errorf("failed to fetch user role: %w", err)
	}

	normalized := normalizeRoleName(role)
	if _, ok := roleCatalog[normalized]; !ok {
		return "", fmt.Errorf("user has unsupported role %q", role)
	}

	return normalized, nil
}

func normalizeRoleName(role string) string {
	return strings.ToLower(strings.TrimSpace(role))
}
