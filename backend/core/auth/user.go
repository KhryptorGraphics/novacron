package auth

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"time"
)

// UserStatus represents the status of a user
type UserStatus string

const (
	// UserStatusActive indicates the user is active
	UserStatusActive UserStatus = "active"

	// UserStatusInactive indicates the user is inactive
	UserStatusInactive UserStatus = "inactive"

	// UserStatusLocked indicates the user is locked
	UserStatusLocked UserStatus = "locked"

	// UserStatusPending indicates the user is pending activation
	UserStatusPending UserStatus = "pending"
)

// User represents a user in the system
type User struct {
	// ID is the unique identifier for the user
	ID string `json:"id"`

	// Username is the username for the user
	Username string `json:"username"`

	// Email is the email address for the user
	Email string `json:"email"`

	// PasswordHash is the hashed password for the user
	PasswordHash string `json:"-"`

	// PasswordSalt is the salt used for password hashing
	PasswordSalt string `json:"-"`

	// FirstName is the first name of the user
	FirstName string `json:"firstName,omitempty"`

	// LastName is the last name of the user
	LastName string `json:"lastName,omitempty"`

	// Status is the status of the user
	Status UserStatus `json:"status"`

	// RoleIDs are the roles assigned to this user
	RoleIDs []string `json:"roleIds"`

	// TenantID is the tenant this user belongs to
	TenantID string `json:"tenantId"`

	// LastLogin is the time of the last login
	LastLogin time.Time `json:"lastLogin,omitempty"`

	// LastPasswordChange is the time of the last password change
	LastPasswordChange time.Time `json:"lastPasswordChange,omitempty"`

	// FailedLoginAttempts is the number of consecutive failed login attempts
	FailedLoginAttempts int `json:"-"`

	// Metadata contains additional metadata
	Metadata map[string]interface{} `json:"metadata,omitempty"`

	// CreatedAt is the time when the user was created
	CreatedAt time.Time `json:"createdAt"`

	// UpdatedAt is the time when the user was last updated
	UpdatedAt time.Time `json:"updatedAt"`

	// CreatedBy is the ID of the user who created this user
	CreatedBy string `json:"createdBy,omitempty"`

	// UpdatedBy is the ID of the user who last updated this user
	UpdatedBy string `json:"updatedBy,omitempty"`
}

// UserService provides operations for managing users
type UserService interface {
	// Create creates a new user
	Create(user *User, password string) error

	// Get gets a user by ID
	Get(id string) (*User, error)

	// GetByUsername gets a user by username
	GetByUsername(username string) (*User, error)

	// GetByEmail gets a user by email
	GetByEmail(email string) (*User, error)

	// List lists users with optional filtering
	List(filter map[string]interface{}) ([]*User, error)

	// Update updates a user
	Update(user *User) error

	// Delete deletes a user
	Delete(id string) error

	// SetPassword sets a user's password
	SetPassword(id string, password string) error

	// VerifyPassword verifies a user's password
	VerifyPassword(id string, password string) (bool, error)

	// UpdateStatus updates a user's status
	UpdateStatus(id string, status UserStatus) error

	// AddRole adds a role to a user
	AddRole(userID string, roleID string) error

	// RemoveRole removes a role from a user
	RemoveRole(userID string, roleID string) error

	// GetRoles gets a user's roles
	GetRoles(userID string) ([]*Role, error)
}

// GenerateRandomSalt generates a random salt for password hashing
func GenerateRandomSalt() (string, error) {
	b := make([]byte, 32)
	_, err := rand.Read(b)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(b), nil
}

// HashPassword hashes a password with a salt
func HashPassword(password string, salt string) string {
	hash := sha256.New()
	hash.Write([]byte(password + salt))
	return hex.EncodeToString(hash.Sum(nil))
}

// VerifyHashedPassword verifies a password against a hash and salt
func VerifyHashedPassword(password string, hash string, salt string) bool {
	return HashPassword(password, salt) == hash
}

// NewUser creates a new user with default values
func NewUser(username, email, tenantID string) *User {
	now := time.Now()
	return &User{
		ID:                 username,
		Username:           username,
		Email:              email,
		Status:             UserStatusPending,
		TenantID:           tenantID,
		CreatedAt:          now,
		UpdatedAt:          now,
		LastPasswordChange: now,
		Metadata:           make(map[string]interface{}),
		RoleIDs:            []string{},
	}
}

// UserMemoryStore is an in-memory implementation of UserService
type UserMemoryStore struct {
	users map[string]*User
}

// NewUserMemoryStore creates a new in-memory user store
func NewUserMemoryStore() *UserMemoryStore {
	return &UserMemoryStore{
		users: make(map[string]*User),
	}
}

// Create creates a new user
func (s *UserMemoryStore) Create(user *User, password string) error {
	if _, exists := s.users[user.ID]; exists {
		return fmt.Errorf("user already exists: %s", user.ID)
	}

	// Check if username is already taken
	for _, existingUser := range s.users {
		if existingUser.Username == user.Username {
			return fmt.Errorf("username already taken: %s", user.Username)
		}
		if existingUser.Email == user.Email {
			return fmt.Errorf("email already in use: %s", user.Email)
		}
	}

	salt, err := GenerateRandomSalt()
	if err != nil {
		return fmt.Errorf("failed to generate salt: %w", err)
	}

	user.PasswordSalt = salt
	user.PasswordHash = HashPassword(password, salt)

	s.users[user.ID] = user
	return nil
}

// Get gets a user by ID
func (s *UserMemoryStore) Get(id string) (*User, error) {
	user, exists := s.users[id]
	if !exists {
		return nil, fmt.Errorf("user not found: %s", id)
	}
	return user, nil
}

// GetByUsername gets a user by username
func (s *UserMemoryStore) GetByUsername(username string) (*User, error) {
	for _, user := range s.users {
		if user.Username == username {
			return user, nil
		}
	}
	return nil, fmt.Errorf("user not found: %s", username)
}

// GetByEmail gets a user by email
func (s *UserMemoryStore) GetByEmail(email string) (*User, error) {
	for _, user := range s.users {
		if user.Email == email {
			return user, nil
		}
	}
	return nil, fmt.Errorf("user not found: %s", email)
}

// List lists users with optional filtering
func (s *UserMemoryStore) List(filter map[string]interface{}) ([]*User, error) {
	users := make([]*User, 0, len(s.users))

	for _, user := range s.users {
		match := true
		for k, v := range filter {
			switch k {
			case "status":
				if user.Status != v.(UserStatus) {
					match = false
				}
			case "tenantId":
				if user.TenantID != v.(string) {
					match = false
				}
			case "roleId":
				roleID := v.(string)
				found := false
				for _, id := range user.RoleIDs {
					if id == roleID {
						found = true
						break
					}
				}
				if !found {
					match = false
				}
			}
		}
		if match {
			users = append(users, user)
		}
	}

	return users, nil
}

// Update updates a user
func (s *UserMemoryStore) Update(user *User) error {
	if _, exists := s.users[user.ID]; !exists {
		return fmt.Errorf("user not found: %s", user.ID)
	}

	// Check if username is already taken by another user
	for id, existingUser := range s.users {
		if id != user.ID && existingUser.Username == user.Username {
			return fmt.Errorf("username already taken: %s", user.Username)
		}
		if id != user.ID && existingUser.Email == user.Email {
			return fmt.Errorf("email already in use: %s", user.Email)
		}
	}

	// Preserve password data
	existingUser := s.users[user.ID]
	user.PasswordHash = existingUser.PasswordHash
	user.PasswordSalt = existingUser.PasswordSalt
	user.LastPasswordChange = existingUser.LastPasswordChange

	user.UpdatedAt = time.Now()
	s.users[user.ID] = user
	return nil
}

// Delete deletes a user
func (s *UserMemoryStore) Delete(id string) error {
	if _, exists := s.users[id]; !exists {
		return fmt.Errorf("user not found: %s", id)
	}
	delete(s.users, id)
	return nil
}

// SetPassword sets a user's password
func (s *UserMemoryStore) SetPassword(id string, password string) error {
	user, exists := s.users[id]
	if !exists {
		return fmt.Errorf("user not found: %s", id)
	}

	salt, err := GenerateRandomSalt()
	if err != nil {
		return fmt.Errorf("failed to generate salt: %w", err)
	}

	user.PasswordSalt = salt
	user.PasswordHash = HashPassword(password, salt)
	user.LastPasswordChange = time.Now()
	user.UpdatedAt = time.Now()

	return nil
}

// VerifyPassword verifies a user's password
func (s *UserMemoryStore) VerifyPassword(id string, password string) (bool, error) {
	user, exists := s.users[id]
	if !exists {
		return false, fmt.Errorf("user not found: %s", id)
	}

	return VerifyHashedPassword(password, user.PasswordHash, user.PasswordSalt), nil
}

// UpdateStatus updates a user's status
func (s *UserMemoryStore) UpdateStatus(id string, status UserStatus) error {
	user, exists := s.users[id]
	if !exists {
		return fmt.Errorf("user not found: %s", id)
	}

	user.Status = status
	user.UpdatedAt = time.Now()
	return nil
}

// AddRole adds a role to a user
func (s *UserMemoryStore) AddRole(userID string, roleID string) error {
	user, exists := s.users[userID]
	if !exists {
		return fmt.Errorf("user not found: %s", userID)
	}

	// Check if the user already has the role
	for _, id := range user.RoleIDs {
		if id == roleID {
			return nil
		}
	}

	user.RoleIDs = append(user.RoleIDs, roleID)
	user.UpdatedAt = time.Now()
	return nil
}

// RemoveRole removes a role from a user
func (s *UserMemoryStore) RemoveRole(userID string, roleID string) error {
	user, exists := s.users[userID]
	if !exists {
		return fmt.Errorf("user not found: %s", userID)
	}

	for i, id := range user.RoleIDs {
		if id == roleID {
			// Remove the role from the slice
			user.RoleIDs = append(user.RoleIDs[:i], user.RoleIDs[i+1:]...)
			user.UpdatedAt = time.Now()
			return nil
		}
	}

	return nil // Role not found, but that's OK
}

// GetRoles gets a user's roles
func (s *UserMemoryStore) GetRoles(userID string) ([]*Role, error) {
	// This is just a stub, in a real implementation we would look up the roles
	// For now, we'll return empty roles with just the IDs filled in
	user, exists := s.users[userID]
	if !exists {
		return nil, fmt.Errorf("user not found: %s", userID)
	}

	roles := make([]*Role, len(user.RoleIDs))
	for i, roleID := range user.RoleIDs {
		roles[i] = &Role{
			ID: roleID,
		}
	}

	return roles, nil
}
