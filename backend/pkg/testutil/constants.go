package testutil

import (
	"os"
)

// Test user constants
const (
	// DefaultTestEmail is the default email for test users
	DefaultTestEmail = "test@example.com"

	// DefaultTestPassword is the default password for test users
	// This is intentionally weak for testing purposes only
	DefaultTestPassword = "password123"

	// DefaultAdminPassword is the default admin password for tests
	DefaultAdminPassword = "admin123"

	// DefaultTestUsername is the default username for tests
	DefaultTestUsername = "test_user"
)

// Test server constants
const (
	// DefaultTestFrontendURL is the default frontend URL for tests
	DefaultTestFrontendURL = "http://localhost:3000"

	// DefaultTestBackendURL is the default backend URL for tests
	DefaultTestBackendURL = "http://localhost:8080"

	// DefaultTestGrafanaURL is the default Grafana URL for tests
	DefaultTestGrafanaURL = "http://localhost:3000"
)

// GetTestEmail returns test email from environment or default
func GetTestEmail() string {
	if email := os.Getenv("TEST_EMAIL"); email != "" {
		return email
	}
	return DefaultTestEmail
}

// GetTestPassword returns test password from environment or default
func GetTestPassword() string {
	if pwd := os.Getenv("TEST_PASSWORD"); pwd != "" {
		return pwd
	}
	return DefaultTestPassword
}

// GetAdminPassword returns admin password from environment or default
func GetAdminPassword() string {
	if pwd := os.Getenv("TEST_ADMIN_PASSWORD"); pwd != "" {
		return pwd
	}
	return DefaultAdminPassword
}

// GetTestFrontendURL returns test frontend URL from environment or default
func GetTestFrontendURL() string {
	if url := os.Getenv("TEST_FRONTEND_URL"); url != "" {
		return url
	}
	return DefaultTestFrontendURL
}

// GetTestBackendURL returns test backend URL from environment or default
func GetTestBackendURL() string {
	if url := os.Getenv("TEST_BACKEND_URL"); url != "" {
		return url
	}
	return DefaultTestBackendURL
}

// GetTestGrafanaURL returns test Grafana URL from environment or default
func GetTestGrafanaURL() string {
	if url := os.Getenv("TEST_GRAFANA_URL"); url != "" {
		return url
	}
	return DefaultTestGrafanaURL
}
