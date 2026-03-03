package auth

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"novacron/backend/pkg/testutil"
)

// Test JWT Service
func TestJWTService(t *testing.T) {
	// Generate test keys
	privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("Failed to generate test keys: %v", err)
	}

	config := JWTConfiguration{
		RSAPrivateKey:   privateKey,
		RSAPublicKey:    &privateKey.PublicKey,
		AccessTokenTTL:  15 * time.Minute,
		RefreshTokenTTL: 24 * time.Hour,
		Issuer:          "test-issuer",
		Audience:        "test-audience",
		KeyID:           "test-key",
	}

	jwtService := NewJWTService(config)

	// Test token generation
	userID := "test-user"
	tenantID := "test-tenant"
	roles := []string{"admin", "user"}
	permissions := []string{"read:vm", "write:vm"}
	sessionID := "test-session"
	metadata := map[string]interface{}{"client": "web"}

	tokenPair, err := jwtService.GenerateTokenPair(userID, tenantID, roles, permissions, sessionID, metadata)
	if err != nil {
		t.Fatalf("Failed to generate token pair: %v", err)
	}

	if tokenPair.AccessToken == "" {
		t.Error("Access token is empty")
	}
	if tokenPair.RefreshToken == "" {
		t.Error("Refresh token is empty")
	}
	if tokenPair.TokenType != "Bearer" {
		t.Errorf("Expected token type 'Bearer', got %s", tokenPair.TokenType)
	}

	// Test token validation
	claims, err := jwtService.ValidateToken(tokenPair.AccessToken)
	if err != nil {
		t.Fatalf("Failed to validate token: %v", err)
	}

	if claims.UserID != userID {
		t.Errorf("Expected UserID %s, got %s", userID, claims.UserID)
	}
	if claims.TenantID != tenantID {
		t.Errorf("Expected TenantID %s, got %s", tenantID, claims.TenantID)
	}
	if claims.TokenType != "access" {
		t.Errorf("Expected TokenType 'access', got %s", claims.TokenType)
	}

	// Test token refresh
	newTokenPair, err := jwtService.RefreshToken(tokenPair.RefreshToken)
	if err != nil {
		t.Fatalf("Failed to refresh token: %v", err)
	}

	if newTokenPair.AccessToken == tokenPair.AccessToken {
		t.Error("New access token should be different from old one")
	}

	// Test invalid token
	_, err = jwtService.ValidateToken("invalid-token")
	if err == nil {
		t.Error("Expected error for invalid token")
	}
}

// Test Password Security Service
func TestPasswordSecurityService(t *testing.T) {
	config := DefaultPasswordSecurityConfig()
	passwordService := NewPasswordSecurityService(config)

	// Test password validation
	user := &User{
		Username:  testutil.DefaultTestUsername,
		Email:     testutil.GetTestEmail(),
		FirstName: "Test",
		LastName:  "User",
	}

	// Test weak password
	err := passwordService.ValidatePassword("weak", user)
	if err == nil {
		t.Error("Expected error for weak password")
	}

	// Test strong password
	err = passwordService.ValidatePassword("StrongPassword123!", user)
	if err != nil {
		t.Errorf("Unexpected error for strong password: %v", err)
	}

	// Test password with personal info
	err = passwordService.ValidatePassword("testuserPassword123!", user)
	if err == nil {
		t.Error("Expected error for password containing personal info")
	}

	// Test password hashing
	password := "TestPassword123!"
	hash, err := passwordService.HashPassword(password)
	if err != nil {
		t.Fatalf("Failed to hash password: %v", err)
	}

	if hash.Hash == "" {
		t.Error("Hash is empty")
	}
	if hash.Salt == "" {
		t.Error("Salt is empty")
	}
	if hash.Algorithm == "" {
		t.Error("Algorithm is empty")
	}

	// Test password verification
	valid, err := passwordService.VerifyPassword(password, hash)
	if err != nil {
		t.Fatalf("Failed to verify password: %v", err)
	}
	if !valid {
		t.Error("Password verification failed")
	}

	// Test wrong password
	valid, err = passwordService.VerifyPassword("WrongPassword123!", hash)
	if err != nil {
		t.Fatalf("Failed to verify wrong password: %v", err)
	}
	if valid {
		t.Error("Wrong password should not verify")
	}

	// Test password generation
	generatedPassword, err := passwordService.GenerateSecurePassword()
	if err != nil {
		t.Fatalf("Failed to generate secure password: %v", err)
	}

	if len(generatedPassword) < config.MinLength {
		t.Errorf("Generated password too short: %d < %d", len(generatedPassword), config.MinLength)
	}

	// Generated password should pass validation
	err = passwordService.ValidatePassword(generatedPassword, nil)
	if err != nil {
		t.Errorf("Generated password failed validation: %v", err)
	}
}

// Test Encryption Service
func TestEncryptionService(t *testing.T) {
	config := DefaultEncryptionConfig()
	encryptionService := NewEncryptionService(config)

	// Test key generation
	key, err := encryptionService.GenerateKey("AES-256-GCM")
	if err != nil {
		t.Fatalf("Failed to generate key: %v", err)
	}

	if key.ID == "" {
		t.Error("Key ID is empty")
	}
	if len(key.KeyData) != 32 {
		t.Errorf("Expected key size 32, got %d", len(key.KeyData))
	}
	if key.Algorithm != "AES-256-GCM" {
		t.Errorf("Expected algorithm 'AES-256-GCM', got %s", key.Algorithm)
	}

	// Test data encryption
	plaintext := []byte("This is a secret message")
	encrypted, err := encryptionService.EncryptData(plaintext, key.ID)
	if err != nil {
		t.Fatalf("Failed to encrypt data: %v", err)
	}

	if encrypted.Data == "" {
		t.Error("Encrypted data is empty")
	}
	if encrypted.KeyID != key.ID {
		t.Errorf("Expected KeyID %s, got %s", key.ID, encrypted.KeyID)
	}

	// Test data decryption
	decrypted, err := encryptionService.DecryptData(encrypted)
	if err != nil {
		t.Fatalf("Failed to decrypt data: %v", err)
	}

	if string(decrypted) != string(plaintext) {
		t.Errorf("Decrypted data mismatch. Expected %s, got %s", string(plaintext), string(decrypted))
	}

	// Test string encryption/decryption
	plaintextStr := "Secret string"
	encryptedStr, err := encryptionService.EncryptString(plaintextStr, key.ID)
	if err != nil {
		t.Fatalf("Failed to encrypt string: %v", err)
	}

	decryptedStr, err := encryptionService.DecryptString(encryptedStr)
	if err != nil {
		t.Fatalf("Failed to decrypt string: %v", err)
	}

	if decryptedStr != plaintextStr {
		t.Errorf("String decryption mismatch. Expected %s, got %s", plaintextStr, decryptedStr)
	}

	// Test TLS certificate generation
	domains := []string{"localhost", "127.0.0.1"}
	cert, err := encryptionService.GenerateTLSCertificate(domains, "NovaCron Test")
	if err != nil {
		t.Fatalf("Failed to generate TLS certificate: %v", err)
	}

	if cert.Certificate == nil {
		t.Error("Certificate is nil")
	}
	if len(cert.CertPEM) == 0 {
		t.Error("Certificate PEM is empty")
	}
	if len(cert.KeyPEM) == 0 {
		t.Error("Key PEM is empty")
	}
}

// Test OAuth2 Service
func TestOAuth2Service(t *testing.T) {
	// Create mock JWT service
	privateKey, _ := rsa.GenerateKey(rand.Reader, 2048)
	jwtConfig := JWTConfiguration{
		RSAPrivateKey: privateKey,
		RSAPublicKey:  &privateKey.PublicKey,
	}
	jwtService := NewJWTService(jwtConfig)

	oauth2Config := OAuth2Config{
		ClientID:     "test-client",
		ClientSecret: "test-secret",
		AuthorizeURL: "https://test-oauth.example.test/authorize",
		TokenURL:     "https://test-oauth.example.test/token",
		UserInfoURL:  "https://test-oauth.example.test/userinfo",
		RedirectURL:  "https://app.test/callback",
		Scopes:       []string{"openid", "email", "profile"},
		ProviderName: "test-provider",
		UsePKCE:      true,
	}

	oauth2Service := NewOAuth2Service(oauth2Config, jwtService)

	// Test authorization URL generation
	authorizeURL, state, err := oauth2Service.GetAuthorizationURL("test-tenant", "/dashboard")
	if err != nil {
		t.Fatalf("Failed to generate authorization URL: %v", err)
	}

	if authorizeURL == "" {
		t.Error("Authorization URL is empty")
	}
	if !strings.Contains(authorizeURL, oauth2Config.ClientID) {
		t.Error("Authorization URL missing client ID")
	}
	if !strings.Contains(authorizeURL, "code_challenge") {
		t.Error("Authorization URL missing PKCE challenge")
	}

	if state.State == "" {
		t.Error("State is empty")
	}
	if state.Nonce == "" {
		t.Error("Nonce is empty")
	}
	if state.CodeChallenge == "" {
		t.Error("Code challenge is empty")
	}
	if state.CodeVerifier == "" {
		t.Error("Code verifier is empty")
	}

	// Test user creation from OAuth2
	userInfo := &UserInfo{
		ID:            "oauth-user-123",
		Email:         testutil.GenerateTestEmail(),
		EmailVerified: true,
		Name:          "Test User",
		GivenName:     "Test",
		FamilyName:    "User",
		Provider:      "test-provider",
	}

	user, err := oauth2Service.CreateUserFromOAuth2(userInfo, "test-tenant")
	if err != nil {
		t.Fatalf("Failed to create user from OAuth2: %v", err)
	}

	if user.Email != userInfo.Email {
		t.Errorf("Expected email %s, got %s", userInfo.Email, user.Email)
	}
	if user.FirstName != userInfo.GivenName {
		t.Errorf("Expected first name %s, got %s", userInfo.GivenName, user.FirstName)
	}
}

// Test Security Middleware
func TestSecurityMiddleware(t *testing.T) {
	auditService := NewInMemoryAuditService()
	encryptionService := NewEncryptionService(DefaultEncryptionConfig())
	middleware := NewSecurityMiddleware(DefaultSecurityConfig(), auditService, encryptionService)

	// Create mock auth service
	userStore := NewUserMemoryStore()
	roleStore := NewRoleMemoryStore()
	tenantStore := NewTenantMemoryStore()
	auditLogService := NewInMemoryAuditService()
	authService := NewAuthService(DefaultAuthConfiguration(), userStore, roleStore, tenantStore, auditLogService)

	// Test middleware with valid request
	handler := middleware.Middleware(authService)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	}))

	req := httptest.NewRequest("GET", "/api/test", nil)
	rr := httptest.NewRecorder()

	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", rr.Code)
	}

	// Check security headers
	if rr.Header().Get("Strict-Transport-Security") == "" {
		t.Error("Missing HSTS header")
	}
	if rr.Header().Get("X-Content-Type-Options") == "" {
		t.Error("Missing X-Content-Type-Options header")
	}
	if rr.Header().Get("X-Frame-Options") == "" {
		t.Error("Missing X-Frame-Options header")
	}

	// Test rate limiting
	for i := 0; i < 1100; i++ { // Exceed default rate limit
		req := httptest.NewRequest("GET", "/api/test", nil)
		rr := httptest.NewRecorder()
		handler.ServeHTTP(rr, req)

		if i >= 1000 && rr.Code != http.StatusTooManyRequests {
			t.Errorf("Expected rate limit after 1000 requests, got status %d on request %d", rr.Code, i+1)
			break
		}
	}

	// Test SQL injection detection
	req = httptest.NewRequest("GET", "/api/test?id=1' OR '1'='1", nil)
	rr = httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400 for SQL injection, got %d", rr.Code)
	}

	// Test XSS detection
	req = httptest.NewRequest("GET", "/api/test?name=<script>alert('xss')</script>", nil)
	rr = httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400 for XSS attempt, got %d", rr.Code)
	}
}

// Test Zero Trust Network Service
func TestZeroTrustNetworkService(t *testing.T) {
	auditService := NewInMemoryAuditService()
	encryptionService := NewEncryptionService(DefaultEncryptionConfig())
	ztService := NewZeroTrustNetworkService(auditService, encryptionService)

	// Test policy creation
	policy := &NetworkPolicy{
		ID:       "test-policy",
		Name:     "Test Policy",
		Enabled:  true,
		Priority: 100,
		Source: NetworkPolicySelector{
			TenantIDs: []string{"tenant-1"},
		},
		Destination: NetworkPolicySelector{
			Services: []string{"web-service"},
		},
		Action: NetworkPolicyAllow,
		Protocols: []NetworkProtocol{
			{
				Protocol: "HTTP",
				Ports:    []int{80, 443},
			},
		},
	}

	err := ztService.CreatePolicy(policy)
	if err != nil {
		t.Fatalf("Failed to create policy: %v", err)
	}

	// Test connection evaluation
	connection := &NetworkConnection{
		ID:            "test-conn",
		SourceIP:      "10.0.1.100",
		DestinationIP: "10.0.2.100",
		SourcePort:    45678,
		DestPort:      80,
		Protocol:      "HTTP",
		TenantID:      "tenant-1",
		Service:       "web-service",
		EstablishedAt: time.Now(),
	}

	ctx := context.Background()
	action, matchedPolicy, err := ztService.EvaluateConnection(ctx, connection)
	if err != nil {
		t.Fatalf("Failed to evaluate connection: %v", err)
	}

	if action != NetworkPolicyAllow {
		t.Errorf("Expected action Allow, got %s", action)
	}
	if matchedPolicy.ID != policy.ID {
		t.Errorf("Expected policy %s, got %s", policy.ID, matchedPolicy.ID)
	}

	// Test microsegment creation
	microsegment := &Microsegment{
		ID:        "test-segment",
		Name:      "Test Segment",
		TenantID:  "tenant-1",
		IPRanges:  []string{"10.0.1.0/24"},
		Services:  []string{"web-service"},
		Isolation: IsolationStrict,
	}

	err = ztService.CreateMicrosegment(microsegment)
	if err != nil {
		t.Fatalf("Failed to create microsegment: %v", err)
	}

	// Test device trust
	device := &DeviceTrust{
		DeviceID:   "device-123",
		UserID:     "user-1",
		TenantID:   "tenant-1",
		DeviceType: "laptop",
		OS:         "Windows 10",
		Compliance: DeviceCompliance{
			Antivirus:       true,
			Firewall:        true,
			Encryption:      true,
			OSUpdated:       true,
			ScreenLock:      true,
			Jailbroken:      false,
			ComplianceScore: 95,
			LastCheck:       time.Now(),
		},
	}

	err = ztService.RegisterDevice(device)
	if err != nil {
		t.Fatalf("Failed to register device: %v", err)
	}

	// Test device trust validation
	trusted, deviceInfo, err := ztService.ValidateDeviceTrust(device.DeviceID, 80)
	if err != nil {
		t.Fatalf("Failed to validate device trust: %v", err)
	}

	if !trusted {
		t.Error("Device should be trusted")
	}
	if deviceInfo.DeviceID != device.DeviceID {
		t.Errorf("Expected device ID %s, got %s", device.DeviceID, deviceInfo.DeviceID)
	}
}

// Test Compliance Service
func TestComplianceService(t *testing.T) {
	auditService := NewInMemoryAuditService()
	encryptionService := NewEncryptionService(DefaultEncryptionConfig())
	complianceService := NewComplianceService(auditService, encryptionService)

	// Test assessment creation
	assessment, err := complianceService.CreateAssessment(SOC2, "tenant-1", "assessor-1")
	if err != nil {
		t.Fatalf("Failed to create assessment: %v", err)
	}

	if assessment.ID == "" {
		t.Error("Assessment ID is empty")
	}
	if assessment.Framework != SOC2 {
		t.Errorf("Expected framework SOC2, got %s", assessment.Framework)
	}
	if assessment.Status != NotTested {
		t.Errorf("Expected status NotTested, got %s", assessment.Status)
	}

	// Test automated testing
	ctx := context.Background()
	err = complianceService.RunAutomatedTests(ctx, assessment.ID)
	if err != nil {
		t.Fatalf("Failed to run automated tests: %v", err)
	}

	// Check that tests were run
	updatedAssessment, err := complianceService.GetAssessment(assessment.ID)
	if err != nil {
		t.Fatalf("Failed to get updated assessment: %v", err)
	}

	if updatedAssessment.Status == NotTested {
		t.Error("Assessment should have been tested")
	}

	// Check control results
	testedControls := 0
	for _, result := range updatedAssessment.ControlResults {
		if result.Tested {
			testedControls++
		}
	}

	if testedControls == 0 {
		t.Error("No controls were tested")
	}

	// Test report generation
	report, err := complianceService.GenerateComplianceReport(assessment.ID)
	if err != nil {
		t.Fatalf("Failed to generate compliance report: %v", err)
	}

	if len(report) == 0 {
		t.Error("Compliance report is empty")
	}

	// Test different frameworks
	for _, framework := range []ComplianceFramework{GDPR, HIPAA, PCIDSS} {
		assessment, err := complianceService.CreateAssessment(framework, "tenant-1", "assessor-1")
		if err != nil {
			t.Errorf("Failed to create %s assessment: %v", framework, err)
			continue
		}

		err = complianceService.RunAutomatedTests(ctx, assessment.ID)
		if err != nil {
			t.Errorf("Failed to run %s automated tests: %v", framework, err)
		}
	}
}

// Test Enhanced Auth Service Integration
func TestEnhancedAuthServiceIntegration(t *testing.T) {
	// Create services
	userStore := NewUserMemoryStore()
	roleStore := NewRoleMemoryStore()
	tenantStore := NewTenantMemoryStore()
	auditService := NewInMemoryAuditService()
	encryptionService := NewEncryptionService(DefaultEncryptionConfig())
	passwordService := NewPasswordSecurityService(DefaultPasswordSecurityConfig())

	// Generate JWT keys
	privateKey, _ := rsa.GenerateKey(rand.Reader, 2048)
	jwtConfig := JWTConfiguration{
		RSAPrivateKey: privateKey,
		RSAPublicKey:  &privateKey.PublicKey,
	}
	jwtService := NewJWTService(jwtConfig)

	// Create enhanced auth service
	authService := NewAuthService(DefaultAuthConfiguration(), userStore, roleStore, tenantStore, auditService)

	// Test user creation with secure password
	user := NewUser(testutil.DefaultTestUsername, testutil.GetTestEmail(), "default")
	password := "SecurePassword123!"

	// Validate password first
	err := passwordService.ValidatePassword(password, user)
	if err != nil {
		t.Fatalf("Password validation failed: %v", err)
	}

	// Hash password
	passwordHash, err := passwordService.HashPassword(password)
	if err != nil {
		t.Fatalf("Password hashing failed: %v", err)
	}

	// Create user with secure password hash
	err = authService.CreateUser(user, password)
	if err != nil {
		t.Fatalf("User creation failed: %v", err)
	}

	// Test login with JWT integration
	session, err := authService.Login(user.Username, password)
	if err != nil {
		t.Fatalf("Login failed: %v", err)
	}

	if session.UserID != user.ID {
		t.Errorf("Expected user ID %s, got %s", user.ID, session.UserID)
	}

	// Test session validation
	validatedSession, err := authService.ValidateSession(session.ID, session.Token)
	if err != nil {
		t.Fatalf("Session validation failed: %v", err)
	}

	if validatedSession.UserID != user.ID {
		t.Errorf("Expected user ID %s, got %s", user.ID, validatedSession.UserID)
	}

	// Test permission checking
	hasPermission, err := authService.HasPermission(user.ID, "vm", "read")
	if err != nil {
		t.Fatalf("Permission check failed: %v", err)
	}

	// User should have basic permissions through default role
	if !hasPermission {
		t.Error("User should have read permission on VMs")
	}

	// Test audit logging
	auditEntries, err := auditService.GetUserActions(user.ID, time.Now().Add(-1*time.Hour), time.Now(), 100, 0)
	if err != nil {
		t.Fatalf("Failed to get audit entries: %v", err)
	}

	if len(auditEntries) == 0 {
		t.Error("Expected audit entries for user actions")
	}

	// Test encryption integration
	key, err := encryptionService.GenerateKey("AES-256-GCM")
	if err != nil {
		t.Fatalf("Failed to generate encryption key: %v", err)
	}

	sensitiveData := "Sensitive user data"
	encryptedData, err := encryptionService.EncryptString(sensitiveData, key.ID)
	if err != nil {
		t.Fatalf("Failed to encrypt sensitive data: %v", err)
	}

	decryptedData, err := encryptionService.DecryptString(encryptedData)
	if err != nil {
		t.Fatalf("Failed to decrypt sensitive data: %v", err)
	}

	if decryptedData != sensitiveData {
		t.Errorf("Decrypted data mismatch. Expected %s, got %s", sensitiveData, decryptedData)
	}
}

// Benchmark tests
func BenchmarkJWTGeneration(b *testing.B) {
	privateKey, _ := rsa.GenerateKey(rand.Reader, 2048)
	config := JWTConfiguration{
		RSAPrivateKey: privateKey,
		RSAPublicKey:  &privateKey.PublicKey,
	}
	jwtService := NewJWTService(config)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := jwtService.GenerateTokenPair("user", "tenant", []string{"role"}, []string{"perm"}, "session", nil)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPasswordHashing(b *testing.B) {
	passwordService := NewPasswordSecurityService(DefaultPasswordSecurityConfig())
	password := "BenchmarkPassword123!"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := passwordService.HashPassword(password)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEncryption(b *testing.B) {
	encryptionService := NewEncryptionService(DefaultEncryptionConfig())
	key, _ := encryptionService.GenerateKey("AES-256-GCM")
	data := []byte("This is benchmark data for encryption testing")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := encryptionService.EncryptData(data, key.ID)
		if err != nil {
			b.Fatal(err)
		}
	}
}
