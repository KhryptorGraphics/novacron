package auth

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
	_ "github.com/lib/pq" // PostgreSQL driver
)

// PostgresUserStore implements UserService using PostgreSQL
type PostgresUserStore struct {
	db *sql.DB
}

// NewPostgresUserStore creates a new PostgreSQL-backed user store
func NewPostgresUserStore(db *sql.DB) *PostgresUserStore {
	return &PostgresUserStore{db: db}
}

// Create creates a new user with password
func (s *PostgresUserStore) Create(user *User, password string) error {
	if user.ID == "" {
		user.ID = uuid.New().String()
	}

	// Generate salt and hash password
	salt, err := GenerateRandomSalt()
	if err != nil {
		return fmt.Errorf("failed to generate salt: %w", err)
	}
	passwordHash := HashPassword(password, salt)

	now := time.Now()
	user.PasswordHash = passwordHash
	user.PasswordSalt = salt
	user.CreatedAt = now
	user.UpdatedAt = now
	user.LastPasswordChange = now

	// Map status to is_active
	isActive := user.Status == UserStatusActive || user.Status == ""

	// Get primary role (first role or "user" default)
	role := "user"
	if len(user.RoleIDs) > 0 {
		role = user.RoleIDs[0]
	}

	// Serialize metadata to JSON
	metadataJSON, err := json.Marshal(user.Metadata)
	if err != nil {
		metadataJSON = []byte("{}")
	}

	query := `
		INSERT INTO users (
			id, organization_id, email, username, password_hash,
			first_name, last_name, role, is_active, mfa_enabled,
			last_login_at, failed_attempts, created_at, updated_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
	`

	_, err = s.db.Exec(query,
		user.ID,
		nullableString(user.TenantID),
		user.Email,
		user.Username,
		passwordHash+":"+salt, // Store hash:salt combined
		user.FirstName,
		user.LastName,
		role,
		isActive,
		false,                    // mfa_enabled default
		nullableTime(user.LastLogin),
		user.FailedLoginAttempts,
		user.CreatedAt,
		user.UpdatedAt,
	)

	if err != nil {
		return fmt.Errorf("failed to create user: %w", err)
	}

	// Store extended metadata in separate table or JSON column if needed
	_ = metadataJSON

	return nil
}

// Get retrieves a user by ID
func (s *PostgresUserStore) Get(id string) (*User, error) {
	query := `
		SELECT id, organization_id, email, username, password_hash,
			   first_name, last_name, role, is_active, mfa_enabled,
			   last_login_at, failed_attempts, locked_until, created_at, updated_at
		FROM users
		WHERE id = $1
	`

	user := &User{}
	var orgID sql.NullString
	var lastLogin, lockedUntil sql.NullTime
	var passwordHashCombined string
	var role string
	var isActive, mfaEnabled bool

	err := s.db.QueryRow(query, id).Scan(
		&user.ID,
		&orgID,
		&user.Email,
		&user.Username,
		&passwordHashCombined,
		&user.FirstName,
		&user.LastName,
		&role,
		&isActive,
		&mfaEnabled,
		&lastLogin,
		&user.FailedLoginAttempts,
		&lockedUntil,
		&user.CreatedAt,
		&user.UpdatedAt,
	)

	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("user not found: %s", id)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get user: %w", err)
	}

	// Parse hash:salt format
	s.parsePasswordHash(passwordHashCombined, user)

	// Map database fields to struct
	if orgID.Valid {
		user.TenantID = orgID.String
	}
	if lastLogin.Valid {
		user.LastLogin = lastLogin.Time
	}
	user.RoleIDs = []string{role}
	user.Status = mapIsActiveToStatus(isActive, lockedUntil)

	return user, nil
}

// GetByUsername retrieves a user by username
func (s *PostgresUserStore) GetByUsername(username string) (*User, error) {
	query := `
		SELECT id, organization_id, email, username, password_hash,
			   first_name, last_name, role, is_active, mfa_enabled,
			   last_login_at, failed_attempts, locked_until, created_at, updated_at
		FROM users
		WHERE username = $1
	`

	user := &User{}
	var orgID sql.NullString
	var lastLogin, lockedUntil sql.NullTime
	var passwordHashCombined string
	var role string
	var isActive, mfaEnabled bool

	err := s.db.QueryRow(query, username).Scan(
		&user.ID,
		&orgID,
		&user.Email,
		&user.Username,
		&passwordHashCombined,
		&user.FirstName,
		&user.LastName,
		&role,
		&isActive,
		&mfaEnabled,
		&lastLogin,
		&user.FailedLoginAttempts,
		&lockedUntil,
		&user.CreatedAt,
		&user.UpdatedAt,
	)

	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("user not found: %s", username)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get user by username: %w", err)
	}

	s.parsePasswordHash(passwordHashCombined, user)

	if orgID.Valid {
		user.TenantID = orgID.String
	}
	if lastLogin.Valid {
		user.LastLogin = lastLogin.Time
	}
	user.RoleIDs = []string{role}
	user.Status = mapIsActiveToStatus(isActive, lockedUntil)

	return user, nil
}

// GetByEmail retrieves a user by email
func (s *PostgresUserStore) GetByEmail(email string) (*User, error) {
	query := `
		SELECT id, organization_id, email, username, password_hash,
			   first_name, last_name, role, is_active, mfa_enabled,
			   last_login_at, failed_attempts, locked_until, created_at, updated_at
		FROM users
		WHERE email = $1
	`

	user := &User{}
	var orgID sql.NullString
	var lastLogin, lockedUntil sql.NullTime
	var passwordHashCombined string
	var role string
	var isActive, mfaEnabled bool

	err := s.db.QueryRow(query, email).Scan(
		&user.ID,
		&orgID,
		&user.Email,
		&user.Username,
		&passwordHashCombined,
		&user.FirstName,
		&user.LastName,
		&role,
		&isActive,
		&mfaEnabled,
		&lastLogin,
		&user.FailedLoginAttempts,
		&lockedUntil,
		&user.CreatedAt,
		&user.UpdatedAt,
	)

	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("user not found: %s", email)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get user by email: %w", err)
	}

	s.parsePasswordHash(passwordHashCombined, user)

	if orgID.Valid {
		user.TenantID = orgID.String
	}
	if lastLogin.Valid {
		user.LastLogin = lastLogin.Time
	}
	user.RoleIDs = []string{role}
	user.Status = mapIsActiveToStatus(isActive, lockedUntil)

	return user, nil
}

// List lists users with optional filtering
func (s *PostgresUserStore) List(filter map[string]interface{}) ([]*User, error) {
	query := `
		SELECT id, organization_id, email, username, password_hash,
			   first_name, last_name, role, is_active, mfa_enabled,
			   last_login_at, failed_attempts, locked_until, created_at, updated_at
		FROM users
		WHERE 1=1
	`
	args := []interface{}{}
	argIdx := 1

	// Apply filters
	if status, ok := filter["status"]; ok {
		isActive := status == string(UserStatusActive)
		query += fmt.Sprintf(" AND is_active = $%d", argIdx)
		args = append(args, isActive)
		argIdx++
	}

	if tenantID, ok := filter["tenant_id"]; ok {
		query += fmt.Sprintf(" AND organization_id = $%d", argIdx)
		args = append(args, tenantID)
		argIdx++
	}

	if role, ok := filter["role"]; ok {
		query += fmt.Sprintf(" AND role = $%d", argIdx)
		args = append(args, role)
		argIdx++
	}

	// Add ordering
	query += " ORDER BY created_at DESC"

	// Apply limit if specified
	if limit, ok := filter["limit"]; ok {
		query += fmt.Sprintf(" LIMIT $%d", argIdx)
		args = append(args, limit)
		argIdx++
	}

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to list users: %w", err)
	}
	defer rows.Close()

	var users []*User
	for rows.Next() {
		user := &User{}
		var orgID sql.NullString
		var lastLogin, lockedUntil sql.NullTime
		var passwordHashCombined string
		var role string
		var isActive, mfaEnabled bool

		err := rows.Scan(
			&user.ID,
			&orgID,
			&user.Email,
			&user.Username,
			&passwordHashCombined,
			&user.FirstName,
			&user.LastName,
			&role,
			&isActive,
			&mfaEnabled,
			&lastLogin,
			&user.FailedLoginAttempts,
			&lockedUntil,
			&user.CreatedAt,
			&user.UpdatedAt,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan user row: %w", err)
		}

		s.parsePasswordHash(passwordHashCombined, user)

		if orgID.Valid {
			user.TenantID = orgID.String
		}
		if lastLogin.Valid {
			user.LastLogin = lastLogin.Time
		}
		user.RoleIDs = []string{role}
		user.Status = mapIsActiveToStatus(isActive, lockedUntil)

		users = append(users, user)
	}

	return users, nil
}

// Update updates a user
func (s *PostgresUserStore) Update(user *User) error {
	user.UpdatedAt = time.Now()

	isActive := user.Status == UserStatusActive

	role := "user"
	if len(user.RoleIDs) > 0 {
		role = user.RoleIDs[0]
	}

	query := `
		UPDATE users SET
			email = $2,
			username = $3,
			first_name = $4,
			last_name = $5,
			role = $6,
			is_active = $7,
			organization_id = $8,
			failed_attempts = $9,
			updated_at = $10
		WHERE id = $1
	`

	result, err := s.db.Exec(query,
		user.ID,
		user.Email,
		user.Username,
		user.FirstName,
		user.LastName,
		role,
		isActive,
		nullableString(user.TenantID),
		user.FailedLoginAttempts,
		user.UpdatedAt,
	)

	if err != nil {
		return fmt.Errorf("failed to update user: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("user not found: %s", user.ID)
	}

	return nil
}

// Delete deletes a user
func (s *PostgresUserStore) Delete(id string) error {
	query := `DELETE FROM users WHERE id = $1`

	result, err := s.db.Exec(query, id)
	if err != nil {
		return fmt.Errorf("failed to delete user: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("user not found: %s", id)
	}

	return nil
}

// SetPassword sets a user's password
func (s *PostgresUserStore) SetPassword(id string, password string) error {
	salt, err := GenerateRandomSalt()
	if err != nil {
		return fmt.Errorf("failed to generate salt: %w", err)
	}
	passwordHash := HashPassword(password, salt)

	query := `UPDATE users SET password_hash = $2, updated_at = $3 WHERE id = $1`

	result, err := s.db.Exec(query, id, passwordHash+":"+salt, time.Now())
	if err != nil {
		return fmt.Errorf("failed to set password: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("user not found: %s", id)
	}

	return nil
}

// VerifyPassword verifies a user's password
func (s *PostgresUserStore) VerifyPassword(id string, password string) (bool, error) {
	query := `SELECT password_hash FROM users WHERE id = $1`

	var passwordHashCombined string
	err := s.db.QueryRow(query, id).Scan(&passwordHashCombined)
	if err == sql.ErrNoRows {
		return false, fmt.Errorf("user not found: %s", id)
	}
	if err != nil {
		return false, fmt.Errorf("failed to get password hash: %w", err)
	}

	// Parse hash:salt format
	user := &User{}
	s.parsePasswordHash(passwordHashCombined, user)

	return VerifyHashedPassword(password, user.PasswordSalt, user.PasswordHash), nil
}

// UpdateStatus updates a user's status
func (s *PostgresUserStore) UpdateStatus(id string, status UserStatus) error {
	isActive := status == UserStatusActive
	var lockedUntil *time.Time

	if status == UserStatusLocked {
		lockTime := time.Now().Add(24 * time.Hour)
		lockedUntil = &lockTime
	}

	query := `UPDATE users SET is_active = $2, locked_until = $3, updated_at = $4 WHERE id = $1`

	result, err := s.db.Exec(query, id, isActive, lockedUntil, time.Now())
	if err != nil {
		return fmt.Errorf("failed to update status: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("user not found: %s", id)
	}

	return nil
}

// AddRole adds a role to a user (updates primary role in this schema)
func (s *PostgresUserStore) AddRole(userID string, roleID string) error {
	// In this schema, there's only a single role column
	// We'll update to the new role
	query := `UPDATE users SET role = $2, updated_at = $3 WHERE id = $1`

	result, err := s.db.Exec(query, userID, roleID, time.Now())
	if err != nil {
		return fmt.Errorf("failed to add role: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("user not found: %s", userID)
	}

	return nil
}

// RemoveRole removes a role from a user (sets to default 'user' role)
func (s *PostgresUserStore) RemoveRole(userID string, roleID string) error {
	// In single-role schema, removing role resets to 'user'
	query := `UPDATE users SET role = 'user', updated_at = $2 WHERE id = $1 AND role = $3`

	_, err := s.db.Exec(query, userID, time.Now(), roleID)
	if err != nil {
		return fmt.Errorf("failed to remove role: %w", err)
	}

	return nil
}

// GetRoles gets a user's roles
func (s *PostgresUserStore) GetRoles(userID string) ([]*Role, error) {
	query := `SELECT role FROM users WHERE id = $1`

	var roleName string
	err := s.db.QueryRow(query, userID).Scan(&roleName)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("user not found: %s", userID)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get user role: %w", err)
	}

	// Return single role as a Role object
	role := &Role{
		ID:        roleName,
		Name:      roleName,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	return []*Role{role}, nil
}

// parsePasswordHash parses the combined hash:salt format
func (s *PostgresUserStore) parsePasswordHash(combined string, user *User) {
	// Try to split hash:salt format
	for i := len(combined) - 1; i >= 0; i-- {
		if combined[i] == ':' {
			user.PasswordHash = combined[:i]
			user.PasswordSalt = combined[i+1:]
			return
		}
	}
	// If no separator, assume it's bcrypt format from older data
	user.PasswordHash = combined
	user.PasswordSalt = ""
}

// mapIsActiveToStatus maps database is_active and locked_until to UserStatus
func mapIsActiveToStatus(isActive bool, lockedUntil sql.NullTime) UserStatus {
	if lockedUntil.Valid && lockedUntil.Time.After(time.Now()) {
		return UserStatusLocked
	}
	if isActive {
		return UserStatusActive
	}
	return UserStatusInactive
}

// nullableString returns sql.NullString for optional strings
func nullableString(s string) sql.NullString {
	if s == "" {
		return sql.NullString{Valid: false}
	}
	return sql.NullString{String: s, Valid: true}
}

// nullableTime returns sql.NullTime for optional times
func nullableTime(t time.Time) sql.NullTime {
	if t.IsZero() {
		return sql.NullTime{Valid: false}
	}
	return sql.NullTime{Time: t, Valid: true}
}

// Ensure PostgresUserStore implements UserService
var _ UserService = (*PostgresUserStore)(nil)
