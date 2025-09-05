package backend_test

import (
	"testing"
	"time"
	
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
	
	"github.com/khryptorgraphics/novacron/backend/core/auth"
)

// MockAuthService is a mock implementation of AuthService
type MockAuthService struct {
	mock.Mock
}

func (m *MockAuthService) Login(username, password string) (*auth.Session, error) {
	args := m.Called(username, password)
	return args.Get(0).(*auth.Session), args.Error(1)
}

func (m *MockAuthService) Logout(sessionID string) error {
	args := m.Called(sessionID)
	return args.Error(0)
}

func (m *MockAuthService) ValidateSession(sessionID string) (*auth.Session, error) {
	args := m.Called(sessionID)
	return args.Get(0).(*auth.Session), args.Error(1)
}

func (m *MockAuthService) RefreshSession(sessionID string) (*auth.Session, error) {
	args := m.Called(sessionID)
	return args.Get(0).(*auth.Session), args.Error(1)
}

func (m *MockAuthService) GetUserPermissions(userID string) ([]string, error) {
	args := m.Called(userID)
	return args.Get(0).([]string), args.Error(1)
}

func (m *MockAuthService) HasPermission(userID, permission string) (bool, error) {
	args := m.Called(userID, permission)
	return args.Bool(0), args.Error(1)
}

// AuthServiceTestSuite defines the test suite for auth service
type AuthServiceTestSuite struct {
	suite.Suite
	authService *MockAuthService
}

func (suite *AuthServiceTestSuite) SetupTest() {
	suite.authService = new(MockAuthService)
}

func (suite *AuthServiceTestSuite) TestLogin_Success() {
	// Arrange
	username := "testuser"
	password := "testpass"
	expectedSession := &auth.Session{
		ID:             "test-session-id",
		UserID:         "test-user-id",
		Token:          "test-token",
		ExpiresAt:      time.Now().Add(time.Hour * 24),
		CreatedAt:      time.Now(),
		LastAccessedAt: time.Now(),
	}

	suite.authService.On("Login", username, password).Return(expectedSession, nil)

	// Act
	session, err := suite.authService.Login(username, password)

	// Assert
	assert.NoError(suite.T(), err)
	assert.NotNil(suite.T(), session)
	assert.Equal(suite.T(), expectedSession.ID, session.ID)
	assert.Equal(suite.T(), expectedSession.UserID, session.UserID)
	assert.Equal(suite.T(), expectedSession.Token, session.Token)
	suite.authService.AssertExpectations(suite.T())
}

func (suite *AuthServiceTestSuite) TestLogin_InvalidCredentials() {
	// Arrange
	username := "invalid"
	password := "invalid"

	suite.authService.On("Login", username, password).Return((*auth.Session)(nil), auth.ErrInvalidCredentials)

	// Act
	session, err := suite.authService.Login(username, password)

	// Assert
	assert.Error(suite.T(), err)
	assert.Nil(suite.T(), session)
	assert.Equal(suite.T(), auth.ErrInvalidCredentials, err)
	suite.authService.AssertExpectations(suite.T())
}

func (suite *AuthServiceTestSuite) TestLogin_EmptyCredentials() {
	// Test various empty credential scenarios
	testCases := []struct {
		username string
		password string
	}{
		{"", "password"},
		{"username", ""},
		{"", ""},
	}

	for _, tc := range testCases {
		suite.authService.On("Login", tc.username, tc.password).Return((*auth.Session)(nil), auth.ErrInvalidCredentials)
		
		session, err := suite.authService.Login(tc.username, tc.password)
		
		assert.Error(suite.T(), err)
		assert.Nil(suite.T(), session)
	}
}

func (suite *AuthServiceTestSuite) TestValidateSession_Success() {
	// Arrange
	sessionID := "valid-session-id"
	expectedSession := &auth.Session{
		ID:             sessionID,
		UserID:         "test-user-id",
		Token:          "valid-token",
		ExpiresAt:      time.Now().Add(time.Hour),
		LastAccessedAt: time.Now(),
	}

	suite.authService.On("ValidateSession", sessionID).Return(expectedSession, nil)

	// Act
	session, err := suite.authService.ValidateSession(sessionID)

	// Assert
	assert.NoError(suite.T(), err)
	assert.NotNil(suite.T(), session)
	assert.Equal(suite.T(), sessionID, session.ID)
	suite.authService.AssertExpectations(suite.T())
}

func (suite *AuthServiceTestSuite) TestValidateSession_Expired() {
	// Arrange
	sessionID := "expired-session-id"

	suite.authService.On("ValidateSession", sessionID).Return((*auth.Session)(nil), auth.ErrSessionExpired)

	// Act
	session, err := suite.authService.ValidateSession(sessionID)

	// Assert
	assert.Error(suite.T(), err)
	assert.Nil(suite.T(), session)
	assert.Equal(suite.T(), auth.ErrSessionExpired, err)
	suite.authService.AssertExpectations(suite.T())
}

func (suite *AuthServiceTestSuite) TestValidateSession_NotFound() {
	// Arrange
	sessionID := "nonexistent-session-id"

	suite.authService.On("ValidateSession", sessionID).Return((*auth.Session)(nil), auth.ErrSessionNotFound)

	// Act
	session, err := suite.authService.ValidateSession(sessionID)

	// Assert
	assert.Error(suite.T(), err)
	assert.Nil(suite.T(), session)
	assert.Equal(suite.T(), auth.ErrSessionNotFound, err)
	suite.authService.AssertExpectations(suite.T())
}

func (suite *AuthServiceTestSuite) TestLogout_Success() {
	// Arrange
	sessionID := "valid-session-id"

	suite.authService.On("Logout", sessionID).Return(nil)

	// Act
	err := suite.authService.Logout(sessionID)

	// Assert
	assert.NoError(suite.T(), err)
	suite.authService.AssertExpectations(suite.T())
}

func (suite *AuthServiceTestSuite) TestLogout_SessionNotFound() {
	// Arrange
	sessionID := "nonexistent-session-id"

	suite.authService.On("Logout", sessionID).Return(auth.ErrSessionNotFound)

	// Act
	err := suite.authService.Logout(sessionID)

	// Assert
	assert.Error(suite.T(), err)
	assert.Equal(suite.T(), auth.ErrSessionNotFound, err)
	suite.authService.AssertExpectations(suite.T())
}

func (suite *AuthServiceTestSuite) TestRefreshSession_Success() {
	// Arrange
	sessionID := "valid-session-id"
	refreshedSession := &auth.Session{
		ID:             sessionID,
		UserID:         "test-user-id",
		Token:          "new-token",
		ExpiresAt:      time.Now().Add(time.Hour * 24),
		LastAccessedAt: time.Now(),
	}

	suite.authService.On("RefreshSession", sessionID).Return(refreshedSession, nil)

	// Act
	session, err := suite.authService.RefreshSession(sessionID)

	// Assert
	assert.NoError(suite.T(), err)
	assert.NotNil(suite.T(), session)
	assert.Equal(suite.T(), sessionID, session.ID)
	assert.Equal(suite.T(), "new-token", session.Token)
	suite.authService.AssertExpectations(suite.T())
}

func (suite *AuthServiceTestSuite) TestHasPermission_Success() {
	// Arrange
	userID := "test-user-id"
	permission := "vm:create"

	suite.authService.On("HasPermission", userID, permission).Return(true, nil)

	// Act
	hasPermission, err := suite.authService.HasPermission(userID, permission)

	// Assert
	assert.NoError(suite.T(), err)
	assert.True(suite.T(), hasPermission)
	suite.authService.AssertExpectations(suite.T())
}

func (suite *AuthServiceTestSuite) TestHasPermission_Denied() {
	// Arrange
	userID := "test-user-id"
	permission := "admin:delete"

	suite.authService.On("HasPermission", userID, permission).Return(false, nil)

	// Act
	hasPermission, err := suite.authService.HasPermission(userID, permission)

	// Assert
	assert.NoError(suite.T(), err)
	assert.False(suite.T(), hasPermission)
	suite.authService.AssertExpectations(suite.T())
}

func (suite *AuthServiceTestSuite) TestGetUserPermissions_Success() {
	// Arrange
	userID := "test-user-id"
	expectedPermissions := []string{"vm:create", "vm:read", "vm:update"}

	suite.authService.On("GetUserPermissions", userID).Return(expectedPermissions, nil)

	// Act
	permissions, err := suite.authService.GetUserPermissions(userID)

	// Assert
	assert.NoError(suite.T(), err)
	assert.NotNil(suite.T(), permissions)
	assert.Equal(suite.T(), expectedPermissions, permissions)
	suite.authService.AssertExpectations(suite.T())
}

// Edge case tests
func (suite *AuthServiceTestSuite) TestEdgeCases() {
	// Test with very long username
	longUsername := string(make([]byte, 1000))
	suite.authService.On("Login", longUsername, "password").Return((*auth.Session)(nil), auth.ErrInvalidCredentials)
	
	session, err := suite.authService.Login(longUsername, "password")
	assert.Error(suite.T(), err)
	assert.Nil(suite.T(), session)

	// Test with special characters in password
	specialPassword := "!@#$%^&*()_+-=[]{}|;:,.<>?"
	suite.authService.On("Login", "user", specialPassword).Return(&auth.Session{}, nil)
	
	session, err = suite.authService.Login("user", specialPassword)
	assert.NoError(suite.T(), err)
	assert.NotNil(suite.T(), session)
}

// Run the test suite
func TestAuthServiceTestSuite(t *testing.T) {
	suite.Run(t, new(AuthServiceTestSuite))
}

// Individual unit tests for Session struct
func TestSession_IsExpired(t *testing.T) {
	// Test expired session
	expiredSession := &auth.Session{
		ExpiresAt: time.Now().Add(-time.Hour),
	}
	assert.True(t, expiredSession.IsExpired())

	// Test valid session
	validSession := &auth.Session{
		ExpiresAt: time.Now().Add(time.Hour),
	}
	assert.False(t, validSession.IsExpired())
}

func TestSession_TimeUntilExpiry(t *testing.T) {
	session := &auth.Session{
		ExpiresAt: time.Now().Add(time.Hour),
	}
	
	timeUntil := session.TimeUntilExpiry()
	assert.True(t, timeUntil > 59*time.Minute)
	assert.True(t, timeUntil <= time.Hour)
}

func TestSession_UpdateLastAccess(t *testing.T) {
	session := &auth.Session{
		LastAccessedAt: time.Now().Add(-time.Hour),
	}
	
	oldLastAccess := session.LastAccessedAt
	session.UpdateLastAccess()
	
	assert.True(t, session.LastAccessedAt.After(oldLastAccess))
}

// Benchmark tests
func BenchmarkLogin(b *testing.B) {
	authService := new(MockAuthService)
	session := &auth.Session{ID: "test", UserID: "user"}
	authService.On("Login", "user", "pass").Return(session, nil)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		authService.Login("user", "pass")
	}
}

func BenchmarkValidateSession(b *testing.B) {
	authService := new(MockAuthService)
	session := &auth.Session{ID: "test", UserID: "user"}
	authService.On("ValidateSession", "session-id").Return(session, nil)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		authService.ValidateSession("session-id")
	}
}