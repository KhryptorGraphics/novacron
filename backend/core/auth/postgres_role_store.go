package auth

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
	_ "github.com/lib/pq" // PostgreSQL driver
)

// PostgresRoleStore implements RoleService using PostgreSQL
type PostgresRoleStore struct {
	db *sql.DB
}

// NewPostgresRoleStore creates a new PostgreSQL-backed role store
func NewPostgresRoleStore(db *sql.DB) *PostgresRoleStore {
	return &PostgresRoleStore{db: db}
}

// Create creates a new role
func (s *PostgresRoleStore) Create(role *Role) error {
	if role.ID == "" {
		role.ID = uuid.New().String()
	}

	now := time.Now()
	role.CreatedAt = now
	role.UpdatedAt = now

	// Serialize metadata to JSON
	metadataJSON, err := json.Marshal(role.Metadata)
	if err != nil {
		metadataJSON = []byte("{}")
	}

	query := `
		INSERT INTO roles (id, name, description, is_system, tenant_id, metadata, created_at, updated_at, created_by, updated_by)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
	`

	_, err = s.db.Exec(query,
		role.ID,
		role.Name,
		role.Description,
		role.IsSystem,
		nullableString(role.TenantID),
		metadataJSON,
		role.CreatedAt,
		role.UpdatedAt,
		nullableString(role.CreatedBy),
		nullableString(role.UpdatedBy),
	)

	if err != nil {
		return fmt.Errorf("failed to create role: %w", err)
	}

	// Add permissions
	for _, perm := range role.Permissions {
		if err := s.addPermissionInternal(role.ID, perm); err != nil {
			return fmt.Errorf("failed to add permission: %w", err)
		}
	}

	return nil
}

// Get retrieves a role by ID
func (s *PostgresRoleStore) Get(id string) (*Role, error) {
	query := `
		SELECT id, name, description, is_system, tenant_id, metadata, created_at, updated_at, created_by, updated_by
		FROM roles
		WHERE id = $1
	`

	role := &Role{}
	var tenantID, createdBy, updatedBy sql.NullString
	var metadataJSON []byte

	err := s.db.QueryRow(query, id).Scan(
		&role.ID,
		&role.Name,
		&role.Description,
		&role.IsSystem,
		&tenantID,
		&metadataJSON,
		&role.CreatedAt,
		&role.UpdatedAt,
		&createdBy,
		&updatedBy,
	)

	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("role not found: %s", id)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get role: %w", err)
	}

	if tenantID.Valid {
		role.TenantID = tenantID.String
	}
	if createdBy.Valid {
		role.CreatedBy = createdBy.String
	}
	if updatedBy.Valid {
		role.UpdatedBy = updatedBy.String
	}

	// Parse metadata
	if len(metadataJSON) > 0 {
		_ = json.Unmarshal(metadataJSON, &role.Metadata)
	}

	// Load permissions
	permissions, err := s.getPermissions(id)
	if err != nil {
		return nil, fmt.Errorf("failed to get permissions: %w", err)
	}
	role.Permissions = permissions

	return role, nil
}

// List lists roles with optional filtering
func (s *PostgresRoleStore) List(filter map[string]interface{}) ([]*Role, error) {
	query := `
		SELECT id, name, description, is_system, tenant_id, metadata, created_at, updated_at, created_by, updated_by
		FROM roles
		WHERE 1=1
	`
	args := []interface{}{}
	argIdx := 1

	// Apply filters
	if tenantID, ok := filter["tenant_id"]; ok {
		query += fmt.Sprintf(" AND tenant_id = $%d", argIdx)
		args = append(args, tenantID)
		argIdx++
	}

	if isSystem, ok := filter["is_system"]; ok {
		query += fmt.Sprintf(" AND is_system = $%d", argIdx)
		args = append(args, isSystem)
		argIdx++
	}

	if name, ok := filter["name"]; ok {
		query += fmt.Sprintf(" AND name ILIKE $%d", argIdx)
		args = append(args, "%"+name.(string)+"%")
		argIdx++
	}

	// Add ordering
	query += " ORDER BY name ASC"

	// Apply limit if specified
	if limit, ok := filter["limit"]; ok {
		query += fmt.Sprintf(" LIMIT $%d", argIdx)
		args = append(args, limit)
		argIdx++
	}

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to list roles: %w", err)
	}
	defer rows.Close()

	var roles []*Role
	for rows.Next() {
		role := &Role{}
		var tenantID, createdBy, updatedBy sql.NullString
		var metadataJSON []byte

		err := rows.Scan(
			&role.ID,
			&role.Name,
			&role.Description,
			&role.IsSystem,
			&tenantID,
			&metadataJSON,
			&role.CreatedAt,
			&role.UpdatedAt,
			&createdBy,
			&updatedBy,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan role row: %w", err)
		}

		if tenantID.Valid {
			role.TenantID = tenantID.String
		}
		if createdBy.Valid {
			role.CreatedBy = createdBy.String
		}
		if updatedBy.Valid {
			role.UpdatedBy = updatedBy.String
		}

		if len(metadataJSON) > 0 {
			_ = json.Unmarshal(metadataJSON, &role.Metadata)
		}

		// Load permissions for each role
		permissions, err := s.getPermissions(role.ID)
		if err != nil {
			return nil, fmt.Errorf("failed to get permissions for role %s: %w", role.ID, err)
		}
		role.Permissions = permissions

		roles = append(roles, role)
	}

	return roles, nil
}

// Update updates a role
func (s *PostgresRoleStore) Update(role *Role) error {
	role.UpdatedAt = time.Now()

	metadataJSON, err := json.Marshal(role.Metadata)
	if err != nil {
		metadataJSON = []byte("{}")
	}

	query := `
		UPDATE roles SET
			name = $2,
			description = $3,
			is_system = $4,
			tenant_id = $5,
			metadata = $6,
			updated_at = $7,
			updated_by = $8
		WHERE id = $1
	`

	result, err := s.db.Exec(query,
		role.ID,
		role.Name,
		role.Description,
		role.IsSystem,
		nullableString(role.TenantID),
		metadataJSON,
		role.UpdatedAt,
		nullableString(role.UpdatedBy),
	)

	if err != nil {
		return fmt.Errorf("failed to update role: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("role not found: %s", role.ID)
	}

	return nil
}

// Delete deletes a role
func (s *PostgresRoleStore) Delete(id string) error {
	// Check if it's a system role
	var isSystem bool
	err := s.db.QueryRow("SELECT is_system FROM roles WHERE id = $1", id).Scan(&isSystem)
	if err == sql.ErrNoRows {
		return fmt.Errorf("role not found: %s", id)
	}
	if err != nil {
		return fmt.Errorf("failed to check role: %w", err)
	}

	if isSystem {
		return fmt.Errorf("cannot delete system role: %s", id)
	}

	// Delete role (cascade will handle permissions)
	result, err := s.db.Exec("DELETE FROM roles WHERE id = $1", id)
	if err != nil {
		return fmt.Errorf("failed to delete role: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("role not found: %s", id)
	}

	return nil
}

// AddPermission adds a permission to a role
func (s *PostgresRoleStore) AddPermission(roleID string, permission Permission) error {
	return s.addPermissionInternal(roleID, permission)
}

// addPermissionInternal adds a permission to a role
func (s *PostgresRoleStore) addPermissionInternal(roleID string, permission Permission) error {
	conditionsJSON, err := json.Marshal(permission.Conditions)
	if err != nil {
		conditionsJSON = []byte("{}")
	}

	effect := permission.Effect
	if effect == "" {
		effect = "allow"
	}

	query := `
		INSERT INTO role_permissions (role_id, resource, action, effect, conditions)
		VALUES ($1, $2, $3, $4, $5)
		ON CONFLICT (role_id, resource, action) DO UPDATE SET
			effect = EXCLUDED.effect,
			conditions = EXCLUDED.conditions
	`

	_, err = s.db.Exec(query, roleID, permission.Resource, permission.Action, effect, conditionsJSON)
	if err != nil {
		return fmt.Errorf("failed to add permission: %w", err)
	}

	return nil
}

// RemovePermission removes a permission from a role
func (s *PostgresRoleStore) RemovePermission(roleID string, resource string, action string) error {
	query := `DELETE FROM role_permissions WHERE role_id = $1 AND resource = $2 AND action = $3`

	result, err := s.db.Exec(query, roleID, resource, action)
	if err != nil {
		return fmt.Errorf("failed to remove permission: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("permission not found")
	}

	return nil
}

// HasPermission checks if a role has a specific permission
func (s *PostgresRoleStore) HasPermission(roleID string, resource string, action string) (bool, error) {
	query := `
		SELECT COUNT(*) FROM role_permissions
		WHERE role_id = $1
		AND (resource = $2 OR resource = '*')
		AND (action = $3 OR action = '*')
		AND effect = 'allow'
	`

	var count int
	err := s.db.QueryRow(query, roleID, resource, action).Scan(&count)
	if err != nil {
		return false, fmt.Errorf("failed to check permission: %w", err)
	}

	return count > 0, nil
}

// getPermissions retrieves all permissions for a role
func (s *PostgresRoleStore) getPermissions(roleID string) ([]Permission, error) {
	query := `
		SELECT resource, action, effect, conditions
		FROM role_permissions
		WHERE role_id = $1
	`

	rows, err := s.db.Query(query, roleID)
	if err != nil {
		return nil, fmt.Errorf("failed to get permissions: %w", err)
	}
	defer rows.Close()

	var permissions []Permission
	for rows.Next() {
		var perm Permission
		var conditionsJSON []byte

		err := rows.Scan(&perm.Resource, &perm.Action, &perm.Effect, &conditionsJSON)
		if err != nil {
			return nil, fmt.Errorf("failed to scan permission: %w", err)
		}

		if len(conditionsJSON) > 0 {
			_ = json.Unmarshal(conditionsJSON, &perm.Conditions)
		}

		permissions = append(permissions, perm)
	}

	return permissions, nil
}

// Ensure PostgresRoleStore implements RoleService
var _ RoleService = (*PostgresRoleStore)(nil)
