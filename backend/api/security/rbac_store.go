package handlers

import (
	"context"
	"database/sql"
	"encoding/json"
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

// RoleUpdate carries partial-update fields for UpdateRole. A zero value
// (empty string / nil slice) leaves the corresponding column unchanged,
// mirroring the canonicalAdminUpdateUserRequest convention used elsewhere in
// this codebase (backend/cmd/api-server/main.go).
type RoleUpdate struct {
	Name        string   `json:"name,omitempty"`
	Description string   `json:"description,omitempty"`
	Permissions []string `json:"permissions,omitempty"`
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
	CreateRole(ctx context.Context, role RoleDefinition) (RoleDefinition, error)
	UpdateRole(ctx context.Context, id string, updates RoleUpdate) (RoleDefinition, error)
	DeleteRole(ctx context.Context, id string) error
}

type PostgresRBACStore struct {
	db *sql.DB
}

func NewPostgresRBACStore(db *sql.DB) *PostgresRBACStore {
	return &PostgresRBACStore{db: db}
}

// scanRole reads a role row from either *sql.Row or *sql.Rows (both satisfy
// this minimal Scan-only interface), decoding the JSONB permissions column.
func scanRole(scanner interface{ Scan(dest ...interface{}) error }) (RoleDefinition, error) {
	var role RoleDefinition
	var permissionsJSON []byte
	if err := scanner.Scan(&role.ID, &role.Name, &role.Description, &permissionsJSON); err != nil {
		return RoleDefinition{}, err
	}
	role.Permissions = []string{}
	if len(permissionsJSON) > 0 {
		if err := json.Unmarshal(permissionsJSON, &role.Permissions); err != nil {
			return RoleDefinition{}, fmt.Errorf("failed to decode role permissions: %w", err)
		}
	}
	return role, nil
}

func (s *PostgresRBACStore) ListRoles(ctx context.Context) ([]RoleDefinition, error) {
	rows, err := s.db.QueryContext(ctx, "SELECT id, name, description, permissions FROM roles ORDER BY id")
	if err != nil {
		return nil, fmt.Errorf("failed to list roles: %w", err)
	}
	defer rows.Close()

	roles := make([]RoleDefinition, 0)
	for rows.Next() {
		role, err := scanRole(rows)
		if err != nil {
			return nil, fmt.Errorf("failed to scan role: %w", err)
		}
		roles = append(roles, role)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("failed to list roles: %w", err)
	}

	return roles, nil
}

func (s *PostgresRBACStore) ListPermissions(ctx context.Context) ([]PermissionDefinition, error) {
	rows, err := s.db.QueryContext(ctx, "SELECT id, name, description FROM permissions ORDER BY id")
	if err != nil {
		return nil, fmt.Errorf("failed to list permissions: %w", err)
	}
	defer rows.Close()

	permissions := make([]PermissionDefinition, 0)
	for rows.Next() {
		var permission PermissionDefinition
		if err := rows.Scan(&permission.ID, &permission.Name, &permission.Description); err != nil {
			return nil, fmt.Errorf("failed to scan permission: %w", err)
		}
		permissions = append(permissions, permission)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("failed to list permissions: %w", err)
	}

	return permissions, nil
}

// CreateRole inserts a new role. The caller-supplied ID is normalized
// (lower-cased/trimmed) and used as the primary key, matching how role names
// are normalized everywhere else in this store.
func (s *PostgresRBACStore) CreateRole(ctx context.Context, role RoleDefinition) (RoleDefinition, error) {
	id := normalizeRoleName(role.ID)
	name := strings.TrimSpace(role.Name)
	if id == "" || name == "" {
		return RoleDefinition{}, errors.New("role id and name are required")
	}

	permissions := role.Permissions
	if permissions == nil {
		permissions = []string{}
	}
	permissionsJSON, err := json.Marshal(permissions)
	if err != nil {
		return RoleDefinition{}, fmt.Errorf("failed to encode role permissions: %w", err)
	}

	row := s.db.QueryRowContext(ctx, `
		INSERT INTO roles (id, name, description, permissions, created_at, updated_at)
		VALUES ($1, $2, $3, $4, NOW(), NOW())
		RETURNING id, name, description, permissions
	`, id, name, strings.TrimSpace(role.Description), permissionsJSON)

	created, err := scanRole(row)
	if err != nil {
		if strings.Contains(err.Error(), "duplicate key") {
			return RoleDefinition{}, fmt.Errorf("role %q already exists", id)
		}
		return RoleDefinition{}, fmt.Errorf("failed to create role: %w", err)
	}

	return created, nil
}

// UpdateRole partially updates a role. Only non-zero fields in updates are
// applied; at least one field is required.
func (s *PostgresRBACStore) UpdateRole(ctx context.Context, id string, updates RoleUpdate) (RoleDefinition, error) {
	id = normalizeRoleName(id)
	if id == "" {
		return RoleDefinition{}, errors.New("role id is required")
	}

	setClauses := make([]string, 0, 3)
	args := make([]interface{}, 0, 4)

	if name := strings.TrimSpace(updates.Name); name != "" {
		args = append(args, name)
		setClauses = append(setClauses, fmt.Sprintf("name = $%d", len(args)))
	}
	if description := strings.TrimSpace(updates.Description); description != "" {
		args = append(args, description)
		setClauses = append(setClauses, fmt.Sprintf("description = $%d", len(args)))
	}
	if updates.Permissions != nil {
		permissionsJSON, err := json.Marshal(updates.Permissions)
		if err != nil {
			return RoleDefinition{}, fmt.Errorf("failed to encode role permissions: %w", err)
		}
		args = append(args, permissionsJSON)
		setClauses = append(setClauses, fmt.Sprintf("permissions = $%d", len(args)))
	}
	if len(setClauses) == 0 {
		return RoleDefinition{}, errors.New("no fields to update")
	}
	setClauses = append(setClauses, "updated_at = NOW()")
	args = append(args, id)

	query := fmt.Sprintf(`
		UPDATE roles SET %s
		WHERE id = $%d
		RETURNING id, name, description, permissions
	`, strings.Join(setClauses, ", "), len(args))

	updated, err := scanRole(s.db.QueryRowContext(ctx, query, args...))
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return RoleDefinition{}, sql.ErrNoRows
		}
		return RoleDefinition{}, fmt.Errorf("failed to update role: %w", err)
	}

	return updated, nil
}

func (s *PostgresRBACStore) DeleteRole(ctx context.Context, id string) error {
	id = normalizeRoleName(id)
	if id == "" {
		return errors.New("role id is required")
	}

	result, err := s.db.ExecContext(ctx, "DELETE FROM roles WHERE id = $1", id)
	if err != nil {
		return fmt.Errorf("failed to delete role: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to confirm role deletion: %w", err)
	}
	if rowsAffected == 0 {
		return sql.ErrNoRows
	}

	return nil
}

// getRole fetches a single role by its normalized ID, returning sql.ErrNoRows
// if no such role exists. Used to validate role references (user
// assignments) against the DB-backed catalog instead of a hardcoded map.
func (s *PostgresRBACStore) getRole(ctx context.Context, id string) (RoleDefinition, error) {
	row := s.db.QueryRowContext(ctx, "SELECT id, name, description, permissions FROM roles WHERE id = $1", id)
	role, err := scanRole(row)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return RoleDefinition{}, sql.ErrNoRows
		}
		return RoleDefinition{}, fmt.Errorf("failed to fetch role: %w", err)
	}
	return role, nil
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
	if _, err := s.getRole(ctx, role); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, fmt.Errorf("unsupported role %q", roles[0])
		}
		return nil, err
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
	role, err := s.getUserRoleDefinition(ctx, userID)
	if err != nil {
		return nil, err
	}

	// ponytail: users carry a single role (users.role is one column), so no
	// multi-role permission dedup is needed here. Revisit if/when a user can
	// hold more than one role.
	permissions := append([]string(nil), role.Permissions...)
	sort.Strings(permissions)
	return permissions, nil
}

func (s *PostgresRBACStore) getUserRole(ctx context.Context, userID string) (string, error) {
	role, err := s.getUserRoleDefinition(ctx, userID)
	if err != nil {
		return "", err
	}
	return role.ID, nil
}

// getUserRoleDefinition fetches a user's assigned role and resolves it
// against the DB-backed role catalog in a single pass, so callers needing
// both the role ID and its permissions (GetUserPermissions) don't fetch the
// role twice.
func (s *PostgresRBACStore) getUserRoleDefinition(ctx context.Context, userID string) (RoleDefinition, error) {
	if strings.TrimSpace(userID) == "" {
		return RoleDefinition{}, errors.New("user ID is required")
	}

	var role string
	if err := s.db.QueryRowContext(ctx, "SELECT role FROM users WHERE id = $1", userID).Scan(&role); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return RoleDefinition{}, sql.ErrNoRows
		}
		return RoleDefinition{}, fmt.Errorf("failed to fetch user role: %w", err)
	}

	normalized := normalizeRoleName(role)
	def, err := s.getRole(ctx, normalized)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return RoleDefinition{}, fmt.Errorf("user has unsupported role %q", role)
		}
		return RoleDefinition{}, err
	}

	return def, nil
}

func normalizeRoleName(role string) string {
	return strings.ToLower(strings.TrimSpace(role))
}
