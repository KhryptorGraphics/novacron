//go:build !experimental

package orchestration

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"

	auth "github.com/khryptorgraphics/novacron/backend/core/auth"
	events "github.com/khryptorgraphics/novacron/backend/core/orchestration/events"
)

// Mock implementations for testing
type MockEventBus struct {
	mock.Mock
}

func (m *MockEventBus) SubscribeToAll(ctx context.Context, handler events.EventHandler) (events.Subscription, error) {
	args := m.Called(ctx, handler)
	return args.Get(0).(events.Subscription), args.Error(1)
}

func (m *MockEventBus) Subscribe(ctx context.Context, eventType events.EventType, handler events.EventHandler) (events.Subscription, error) {
	args := m.Called(ctx, eventType, handler)
	return args.Get(0).(events.Subscription), args.Error(1)
}

func (m *MockEventBus) Publish(ctx context.Context, event *events.OrchestrationEvent) error {
	args := m.Called(ctx, event)
	return args.Error(0)
}

type MockJWTService struct {
	mock.Mock
}

func (m *MockJWTService) ValidateToken(tokenString string) (*auth.JWTClaims, error) {
	args := m.Called(tokenString)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*auth.JWTClaims), args.Error(1)
}

type MockAuthService struct {
	mock.Mock
}

func (m *MockAuthService) ValidateSession(sessionID, token string) (*auth.Session, error) {
	args := m.Called(sessionID, token)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*auth.Session), args.Error(1)
}

func (m *MockAuthService) HasPermissionInTenant(userID, tenantID, resource, action string) (bool, error) {
	args := m.Called(userID, tenantID, resource, action)
	return args.Bool(0), args.Error(1)
}

// Implement other required methods
func (m *MockAuthService) Login(username, password string) (*auth.Session, error) {
	args := m.Called(username, password)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*auth.Session), args.Error(1)
}

func (m *MockAuthService) Logout(sessionID string) error {
	args := m.Called(sessionID)
	return args.Error(0)
}

func (m *MockAuthService) RefreshSession(sessionID, token string) (*auth.Session, error) {
	args := m.Called(sessionID, token)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*auth.Session), args.Error(1)
}

func (m *MockAuthService) HasPermission(userID, resource, action string) (bool, error) {
	args := m.Called(userID, resource, action)
	return args.Bool(0), args.Error(1)
}

func (m *MockAuthService) GetUserRoles(userID string) ([]*auth.Role, error) {
	args := m.Called(userID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).([]*auth.Role), args.Error(1)
}

func (m *MockAuthService) CreateUser(user *auth.User, password string) error {
	args := m.Called(user, password)
	return args.Error(0)
}

func (m *MockAuthService) CreateRole(role *auth.Role) error {
	args := m.Called(role)
	return args.Error(0)
}

func (m *MockAuthService) CreateTenant(tenant *auth.Tenant) error {
	args := m.Called(tenant)
	return args.Error(0)
}

func TestWebSocketSecurity_OriginCheck(t *testing.T) {
	tests := []struct {
		name           string
		origin         string
		allowedOrigins []string
		expected       bool
	}{
		{
			name:           "Valid origin",
			origin:         "https://localhost:3000",
			allowedOrigins: []string{"https://localhost:3000"},
			expected:       true,
		},
		{
			name:           "Invalid origin",
			origin:         "https://evil.com",
			allowedOrigins: []string{"https://localhost:3000"},
			expected:       false,
		},
		{
			name:           "Empty origin (same-origin)",
			origin:         "",
			allowedOrigins: []string{"https://localhost:3000"},
			expected:       true,
		},
		{
			name:           "Wildcard origin",
			origin:         "https://sub.localhost:3000",
			allowedOrigins: []string{"https://*.localhost:3000*"},
			expected:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest("GET", "/ws", nil)
			if tt.origin != "" {
				req.Header.Set("Origin", tt.origin)
			}
			
			result := checkOrigin(req, tt.allowedOrigins)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestWebSocketSecurity_AuthTokenExtraction(t *testing.T) {
	tests := []struct {
		name     string
		setup    func(*http.Request)
		expected string
	}{
		{
			name: "Bearer token in Authorization header",
			setup: func(r *http.Request) {
				r.Header.Set("Authorization", "Bearer test-token-123")
			},
			expected: "test-token-123",
		},
		{
			name: "Token in Sec-WebSocket-Protocol",
			setup: func(r *http.Request) {
				r.Header.Set("Sec-WebSocket-Protocol", "access_token.test-token-456")
			},
			expected: "test-token-456",
		},
		{
			name: "Token in query parameter",
			setup: func(r *http.Request) {
				r.URL.RawQuery = "token=test-token-789"
			},
			expected: "test-token-789",
		},
		{
			name: "No token",
			setup: func(r *http.Request) {
				// No token provided
			},
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest("GET", "/ws", nil)
			tt.setup(req)
			
			token := extractAuthToken(req)
			assert.Equal(t, tt.expected, token)
		})
	}
}

func TestWebSocketSecurity_RateLimiting(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.FatalLevel) // Suppress logs during tests

	mockEventBus := &MockEventBus{}
	mockJWTService := &MockJWTService{}
	mockAuthService := &MockAuthService{}

	config := WebSocketSecurityConfig{
		AllowedOrigins:        []string{"http://localhost:3000"},
		RequireAuthentication: false,
		RateLimitConnections:  2,
		RateLimitMessages:     3,
		MaxConnections:        10,
	}

	wsm := NewWebSocketManager(logger, mockEventBus, mockJWTService, mockAuthService, config)

	// Test connection rate limit
	clientIP := "127.0.0.1"
	
	// First two connections should succeed
	assert.True(t, wsm.checkConnectionRateLimit(clientIP))
	assert.True(t, wsm.checkConnectionRateLimit(clientIP))
	
	// Third connection should fail
	assert.False(t, wsm.checkConnectionRateLimit(clientIP))

	// Test message rate limit
	clientID := "test-client"
	
	// First three messages should succeed
	assert.True(t, wsm.checkMessageRateLimit(clientID))
	assert.True(t, wsm.checkMessageRateLimit(clientID))
	assert.True(t, wsm.checkMessageRateLimit(clientID))
	
	// Fourth message should fail
	assert.False(t, wsm.checkMessageRateLimit(clientID))
}

func TestWebSocketSecurity_Authentication(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.FatalLevel)

	mockEventBus := &MockEventBus{}
	mockJWTService := &MockJWTService{}
	mockAuthService := &MockAuthService{}

	config := WebSocketSecurityConfig{
		AllowedOrigins:        []string{"http://localhost:3000"},
		RequireAuthentication: true,
		RateLimitConnections:  100,
		RateLimitMessages:     100,
		MaxConnections:        10,
		RequirePermissions:    []string{"system:read"},
	}

	wsm := NewWebSocketManager(logger, mockEventBus, mockJWTService, mockAuthService, config)

	// Start the manager
	mockEventBus.On("SubscribeToAll", mock.Anything, mock.Anything).Return(&MockSubscription{}, nil)
	err := wsm.Start()
	assert.NoError(t, err)

	// Test with valid authentication
	validClaims := &auth.JWTClaims{
		UserID:      "test-user",
		TenantID:    "test-tenant",
		SessionID:   "test-session",
		Permissions: []string{"system:read"},
	}
	validSession := &auth.Session{
		ID:       "test-session",
		UserID:   "test-user",
		TenantID: "test-tenant",
	}

	mockJWTService.On("ValidateToken", "valid-token").Return(validClaims, nil)
	mockAuthService.On("ValidateSession", "test-session", "valid-token").Return(validSession, nil)
	mockAuthService.On("HasPermissionInTenant", "test-user", "test-tenant", "system", "read").Return(true, nil)

	// Create test server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		wsm.HandleWebSocket(w, r)
	}))
	defer server.Close()

	// Convert http URL to websocket URL
	wsURL := strings.Replace(server.URL, "http://", "ws://", 1)

	// Test authenticated connection
	header := http.Header{}
	header.Add("Origin", "http://localhost:3000")
	header.Add("Sec-WebSocket-Protocol", "access_token.valid-token")

	conn, _, err := websocket.DefaultDialer.Dial(wsURL, header)
	assert.NoError(t, err)
	if conn != nil {
		conn.Close()
	}

	mockJWTService.AssertExpectations(t)
	mockAuthService.AssertExpectations(t)
}

type MockSubscription struct{}

func (m *MockSubscription) Unsubscribe() error {
	return nil
}

func TestWebSocketSecurity_EventPermissions(t *testing.T) {
	client := &WebSocketClient{
		authenticated: true,
		permissions:   []string{"vm:read", "system:admin"},
	}

	tests := []struct {
		name      string
		eventType events.EventType
		expected  bool
	}{
		{
			name:      "VM event with vm:read permission",
			eventType: events.EventTypeVMCreated,
			expected:  true,
		},
		{
			name:      "System event with system:admin permission",
			eventType: events.EventTypeSystemAlert,
			expected:  true,
		},
		{
			name:      "Node event without node:read permission",
			eventType: events.EventTypeNodeAdded,
			expected:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			event := &events.OrchestrationEvent{
				Type: tt.eventType,
			}
			
			result := client.hasPermissionForEvent(event)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestWebSocketSecurity_ClientIP(t *testing.T) {
	tests := []struct {
		name     string
		setup    func(*http.Request)
		expected string
	}{
		{
			name: "X-Forwarded-For header",
			setup: func(r *http.Request) {
				r.Header.Set("X-Forwarded-For", "192.168.1.1, 10.0.0.1")
			},
			expected: "192.168.1.1",
		},
		{
			name: "X-Real-IP header",
			setup: func(r *http.Request) {
				r.Header.Set("X-Real-IP", "192.168.1.2")
			},
			expected: "192.168.1.2",
		},
		{
			name: "CF-Connecting-IP header",
			setup: func(r *http.Request) {
				r.Header.Set("CF-Connecting-IP", "192.168.1.3")
			},
			expected: "192.168.1.3",
		},
		{
			name: "RemoteAddr fallback",
			setup: func(r *http.Request) {
				r.RemoteAddr = "192.168.1.4:8080"
			},
			expected: "192.168.1.4",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest("GET", "/ws", nil)
			tt.setup(req)
			
			ip := getClientIP(req)
			assert.Equal(t, tt.expected, ip)
		})
	}
}

func TestWebSocketSecurity_Stats(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.FatalLevel)

	mockEventBus := &MockEventBus{}
	mockJWTService := &MockJWTService{}
	mockAuthService := &MockAuthService{}

	config := DefaultWebSocketSecurityConfig()
	wsm := NewWebSocketManager(logger, mockEventBus, mockJWTService, mockAuthService, config)

	stats := wsm.GetStats()
	
	assert.Contains(t, stats, "connected_clients")
	assert.Contains(t, stats, "authenticated_clients")
	assert.Contains(t, stats, "unauthenticated_clients")
	assert.Contains(t, stats, "max_connections")
	assert.Contains(t, stats, "security_config")
	
	securityConfig := stats["security_config"].(map[string]interface{})
	assert.Equal(t, true, securityConfig["require_authentication"])
	assert.Equal(t, 60, securityConfig["rate_limit_connections"])
	assert.Equal(t, 300, securityConfig["rate_limit_messages"])
}