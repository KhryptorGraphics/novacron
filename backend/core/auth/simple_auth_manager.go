package auth

import (
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"golang.org/x/crypto/bcrypt"
)

// SimpleAuthManager provides basic authentication functionality and implements AuthManager interface
type SimpleAuthManager struct {
	db        *sql.DB
	jwtSecret string
}

// NewSimpleAuthManager creates a new simple auth manager
func NewSimpleAuthManager(jwtSecret string, db *sql.DB) *SimpleAuthManager {
	return &SimpleAuthManager{
		db:        db,
		jwtSecret: jwtSecret,
	}
}

// GetJWTSecret returns the JWT secret for middleware
func (m *SimpleAuthManager) GetJWTSecret() string {
	return m.jwtSecret
}

// Authenticate validates username/password and returns user and JWT token
func (m *SimpleAuthManager) Authenticate(username, password string) (*User, string, error) {
	// Get user from database
	user, err := m.getUserByUsername(username)
	if err != nil {
		return nil, "", errors.New("invalid credentials")
	}

	// Verify password
	if err := bcrypt.CompareHashAndPassword([]byte(user.PasswordHash), []byte(password)); err != nil {
		return nil, "", errors.New("invalid credentials")
	}

	// Generate JWT token
	token, err := m.generateJWTToken(user)
	if err != nil {
		return nil, "", fmt.Errorf("failed to generate token: %w", err)
	}

	return user, token, nil
}

// CreateUser creates a new user
func (m *SimpleAuthManager) CreateUser(username, email, password, role, tenantID string) (*User, error) {
	// Hash password
	passwordHash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return nil, fmt.Errorf("failed to hash password: %w", err)
	}

	// Insert user into database
	var userID int
	err = m.db.QueryRow(`
		INSERT INTO users (username, email, password_hash, role, tenant_id)
		VALUES ($1, $2, $3, $4, $5)
		RETURNING id
	`, username, email, string(passwordHash), role, tenantID).Scan(&userID)

	if err != nil {
		return nil, fmt.Errorf("failed to create user: %w", err)
	}

	// Return the created user
	return &User{
		ID:           fmt.Sprintf("%d", userID),
		Username:     username,
		Email:        email,
		PasswordHash: string(passwordHash),
		RoleIDs:      []string{role}, // Use RoleIDs instead of Role
		TenantID:     tenantID,
		Status:       UserStatusActive,
		CreatedAt:    time.Now(),
		UpdatedAt:    time.Now(),
		Roles: []*Role{
			{
				ID:       "1",
				Name:     role,
				TenantID: tenantID,
			},
		},
	}, nil
}

// GetUser gets a user by ID
func (m *SimpleAuthManager) GetUser(userID string) (*User, error) {
	var user User
	var createdAt, updatedAt time.Time
	var role string

	err := m.db.QueryRow(`
		SELECT id, username, email, password_hash, role, tenant_id, created_at, updated_at
		FROM users WHERE id = $1
	`, userID).Scan(
		&user.ID,
		&user.Username,
		&user.Email,
		&user.PasswordHash,
		&role,
		&user.TenantID,
		&createdAt,
		&updatedAt,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, errors.New("user not found")
		}
		return nil, fmt.Errorf("failed to get user: %w", err)
	}

	user.CreatedAt = createdAt
	user.UpdatedAt = updatedAt
	user.Status = UserStatusActive
	user.RoleIDs = []string{role}

	// Load user roles (simplified - just use the role field for now)
	user.Roles = []*Role{
		{
			ID:       "1",
			Name:     role,
			TenantID: user.TenantID,
		},
	}

	return &user, nil
}

// getUserByUsername gets a user by username
func (m *SimpleAuthManager) getUserByUsername(username string) (*User, error) {
	var user User
	var createdAt, updatedAt time.Time
	var role string

	err := m.db.QueryRow(`
		SELECT id, username, email, password_hash, role, tenant_id, created_at, updated_at
		FROM users WHERE username = $1
	`, username).Scan(
		&user.ID,
		&user.Username,
		&user.Email,
		&user.PasswordHash,
		&role,
		&user.TenantID,
		&createdAt,
		&updatedAt,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, errors.New("user not found")
		}
		return nil, fmt.Errorf("failed to get user: %w", err)
	}

	user.CreatedAt = createdAt
	user.UpdatedAt = updatedAt
	user.Status = UserStatusActive
	user.RoleIDs = []string{role}

	// Load user roles (simplified - just use the role field for now)
	user.Roles = []*Role{
		{
			ID:       "1",
			Name:     role,
			TenantID: user.TenantID,
		},
	}

	return &user, nil
}

// generateJWTToken generates a JWT token for a user
func (m *SimpleAuthManager) generateJWTToken(user *User) (string, error) {
	// Get the primary role name for JWT claims
	roleName := "user" // Default role
	if len(user.Roles) > 0 {
		roleName = user.Roles[0].Name
	}

	// Create the claims
	claims := jwt.MapClaims{
		"user_id":   user.ID,
		"username":  user.Username,
		"email":     user.Email,
		"role":      roleName,
		"tenant_id": user.TenantID,
		"exp":       time.Now().Add(24 * time.Hour).Unix(), // Expires in 24 hours
		"iat":       time.Now().Unix(),                     // Issued at
	}

	// Create the token
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)

	// Sign the token
	tokenString, err := token.SignedString([]byte(m.jwtSecret))
	if err != nil {
		return "", fmt.Errorf("failed to sign token: %w", err)
	}

	return tokenString, nil
}