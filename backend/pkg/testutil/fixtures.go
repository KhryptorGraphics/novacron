package testutil

import (
	"fmt"
	"math/rand"
	"time"
)

// GenerateTestEmail generates a unique test email address
func GenerateTestEmail() string {
	timestamp := time.Now().UnixNano()
	return fmt.Sprintf("test-%d@example.com", timestamp)
}

// GenerateTestUsername generates a unique test username
func GenerateTestUsername() string {
	timestamp := time.Now().UnixNano()
	return fmt.Sprintf("test_user_%d", timestamp)
}

// GenerateTestPassword generates a secure test password
func GenerateTestPassword() string {
	const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
	rand.Seed(time.Now().UnixNano())

	password := make([]byte, 16)
	for i := range password {
		password[i] = charset[rand.Intn(len(charset))]
	}
	return string(password)
}

// TestUser represents a test user fixture
type TestUser struct {
	Username string
	Email    string
	Password string
	TenantID string
}

// NewTestUser creates a new test user fixture with unique values
func NewTestUser() *TestUser {
	return &TestUser{
		Username: GenerateTestUsername(),
		Email:    GenerateTestEmail(),
		Password: GenerateTestPassword(),
		TenantID: "test-tenant",
	}
}

// NewTestUserWithDefaults creates a test user with default test values
func NewTestUserWithDefaults() *TestUser {
	return &TestUser{
		Username: DefaultTestUsername,
		Email:    DefaultTestEmail,
		Password: DefaultTestPassword,
		TenantID: "test-tenant",
	}
}
