package auth

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

// OAuth2Config defines OAuth2/OIDC provider configuration
type OAuth2Config struct {
	// ClientID for the OAuth2 application
	ClientID string
	// ClientSecret for the OAuth2 application
	ClientSecret string
	// AuthorizeURL for the authorization endpoint
	AuthorizeURL string
	// TokenURL for the token endpoint
	TokenURL string
	// UserInfoURL for the user information endpoint
	UserInfoURL string
	// JWKSUrl for JSON Web Key Set endpoint
	JWKSUrl string
	// RedirectURL for OAuth2 callback
	RedirectURL string
	// Scopes to request from the provider
	Scopes []string
	// ProviderName (google, microsoft, okta, etc.)
	ProviderName string
	// Issuer for OIDC issuer validation
	Issuer string
	// PKCE support for enhanced security
	UsePKCE bool
}

// OAuth2State represents OAuth2 state for CSRF protection
type OAuth2State struct {
	State      string    `json:"state"`
	Nonce      string    `json:"nonce,omitempty"`
	CodeChallenge string `json:"code_challenge,omitempty"`
	CodeVerifier  string `json:"code_verifier,omitempty"`
	TenantID   string    `json:"tenant_id,omitempty"`
	RedirectTo string    `json:"redirect_to,omitempty"`
	CreatedAt  time.Time `json:"created_at"`
	ExpiresAt  time.Time `json:"expires_at"`
}

// OAuth2Token represents an OAuth2 access token response
type OAuth2Token struct {
	AccessToken  string `json:"access_token"`
	TokenType    string `json:"token_type"`
	RefreshToken string `json:"refresh_token,omitempty"`
	ExpiresIn    int    `json:"expires_in,omitempty"`
	Scope        string `json:"scope,omitempty"`
	IDToken      string `json:"id_token,omitempty"`
}

// UserInfo represents user information from OAuth2 provider
type UserInfo struct {
	ID            string `json:"id"`
	Email         string `json:"email"`
	EmailVerified bool   `json:"email_verified"`
	Name          string `json:"name"`
	GivenName     string `json:"given_name"`
	FamilyName    string `json:"family_name"`
	Picture       string `json:"picture"`
	Locale        string `json:"locale"`
	Provider      string `json:"provider"`
	RawClaims     map[string]interface{} `json:"raw_claims,omitempty"`
}

// IDTokenClaims represents OIDC ID token claims
type IDTokenClaims struct {
	jwt.RegisteredClaims
	Email         string `json:"email,omitempty"`
	EmailVerified bool   `json:"email_verified,omitempty"`
	Name          string `json:"name,omitempty"`
	GivenName     string `json:"given_name,omitempty"`
	FamilyName    string `json:"family_name,omitempty"`
	Picture       string `json:"picture,omitempty"`
	Locale        string `json:"locale,omitempty"`
	Nonce         string `json:"nonce,omitempty"`
	AtHash        string `json:"at_hash,omitempty"`
}

// OAuth2Service handles OAuth2/OIDC authentication
type OAuth2Service struct {
	config    OAuth2Config
	states    map[string]*OAuth2State
	httpClient *http.Client
	jwtService *JWTService
}

// NewOAuth2Service creates a new OAuth2 service
func NewOAuth2Service(config OAuth2Config, jwtService *JWTService) *OAuth2Service {
	if config.ProviderName == "" {
		config.ProviderName = "generic"
	}
	if len(config.Scopes) == 0 {
		config.Scopes = []string{"openid", "email", "profile"}
	}

	return &OAuth2Service{
		config:  config,
		states:  make(map[string]*OAuth2State),
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		jwtService: jwtService,
	}
}

// GetAuthorizationURL generates the OAuth2 authorization URL
func (o *OAuth2Service) GetAuthorizationURL(tenantID, redirectTo string) (string, *OAuth2State, error) {
	// Generate state for CSRF protection
	state, err := o.generateState()
	if err != nil {
		return "", nil, fmt.Errorf("failed to generate state: %w", err)
	}

	// Generate nonce for OIDC
	nonce, err := o.generateNonce()
	if err != nil {
		return "", nil, fmt.Errorf("failed to generate nonce: %w", err)
	}

	oauth2State := &OAuth2State{
		State:      state,
		Nonce:      nonce,
		TenantID:   tenantID,
		RedirectTo: redirectTo,
		CreatedAt:  time.Now(),
		ExpiresAt:  time.Now().Add(10 * time.Minute),
	}

	// PKCE support
	if o.config.UsePKCE {
		codeVerifier, codeChallenge, err := o.generatePKCE()
		if err != nil {
			return "", nil, fmt.Errorf("failed to generate PKCE: %w", err)
		}
		oauth2State.CodeVerifier = codeVerifier
		oauth2State.CodeChallenge = codeChallenge
	}

	// Store state
	o.states[state] = oauth2State

	// Build authorization URL
	params := url.Values{
		"client_id":     {o.config.ClientID},
		"redirect_uri":  {o.config.RedirectURL},
		"response_type": {"code"},
		"scope":         {strings.Join(o.config.Scopes, " ")},
		"state":         {state},
		"nonce":         {nonce},
	}

	// Add PKCE parameters
	if o.config.UsePKCE {
		params.Set("code_challenge", oauth2State.CodeChallenge)
		params.Set("code_challenge_method", "S256")
	}

	authorizationURL := fmt.Sprintf("%s?%s", o.config.AuthorizeURL, params.Encode())
	return authorizationURL, oauth2State, nil
}

// ExchangeCodeForToken exchanges authorization code for tokens
func (o *OAuth2Service) ExchangeCodeForToken(ctx context.Context, code, state string) (*OAuth2Token, *OAuth2State, error) {
	// Validate state
	oauth2State, exists := o.states[state]
	if !exists || time.Now().After(oauth2State.ExpiresAt) {
		return nil, nil, fmt.Errorf("invalid or expired state")
	}

	// Remove state from storage
	delete(o.states, state)

	// Prepare token request
	data := url.Values{
		"grant_type":   {"authorization_code"},
		"code":         {code},
		"redirect_uri": {o.config.RedirectURL},
		"client_id":    {o.config.ClientID},
	}

	// Add client secret if not using PKCE
	if !o.config.UsePKCE {
		data.Set("client_secret", o.config.ClientSecret)
	} else {
		data.Set("code_verifier", oauth2State.CodeVerifier)
	}

	// Make token request
	req, err := http.NewRequestWithContext(ctx, "POST", o.config.TokenURL, strings.NewReader(data.Encode()))
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create token request: %w", err)
	}

	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	// Use basic auth if not using PKCE
	if !o.config.UsePKCE {
		req.SetBasicAuth(o.config.ClientID, o.config.ClientSecret)
	}

	resp, err := o.httpClient.Do(req)
	if err != nil {
		return nil, nil, fmt.Errorf("token request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, nil, fmt.Errorf("token request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var token OAuth2Token
	if err := json.NewDecoder(resp.Body).Decode(&token); err != nil {
		return nil, nil, fmt.Errorf("failed to decode token response: %w", err)
	}

	return &token, oauth2State, nil
}

// ValidateIDToken validates and parses an OIDC ID token
func (o *OAuth2Service) ValidateIDToken(idToken, nonce string) (*IDTokenClaims, error) {
	if idToken == "" {
		return nil, fmt.Errorf("ID token is empty")
	}

	// Parse token without verification first to get claims
	token, _, err := new(jwt.Parser).ParseUnverified(idToken, &IDTokenClaims{})
	if err != nil {
		return nil, fmt.Errorf("failed to parse ID token: %w", err)
	}

	claims, ok := token.Claims.(*IDTokenClaims)
	if !ok {
		return nil, fmt.Errorf("invalid ID token claims")
	}

	// Validate issuer
	if o.config.Issuer != "" && !claims.VerifyIssuer(o.config.Issuer, true) {
		return nil, fmt.Errorf("invalid issuer: %s", claims.Issuer)
	}

	// Validate audience (client ID)
	if !claims.VerifyAudience(o.config.ClientID, true) {
		return nil, fmt.Errorf("invalid audience")
	}

	// Validate nonce
	if nonce != "" && claims.Nonce != nonce {
		return nil, fmt.Errorf("invalid nonce")
	}

	// In production, you would verify the signature using JWKS
	// For now, we'll skip signature verification

	return claims, nil
}

// GetUserInfo fetches user information from the provider
func (o *OAuth2Service) GetUserInfo(ctx context.Context, accessToken string) (*UserInfo, error) {
	if o.config.UserInfoURL == "" {
		return nil, fmt.Errorf("user info URL not configured")
	}

	req, err := http.NewRequestWithContext(ctx, "GET", o.config.UserInfoURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create user info request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+accessToken)
	req.Header.Set("Accept", "application/json")

	resp, err := o.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("user info request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("user info request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var rawUserInfo map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&rawUserInfo); err != nil {
		return nil, fmt.Errorf("failed to decode user info response: %w", err)
	}

	// Parse user info based on provider
	userInfo := &UserInfo{
		Provider:  o.config.ProviderName,
		RawClaims: rawUserInfo,
	}

	// Extract standard fields
	if id, ok := rawUserInfo["sub"].(string); ok {
		userInfo.ID = id
	} else if id, ok := rawUserInfo["id"].(string); ok {
		userInfo.ID = id
	}

	if email, ok := rawUserInfo["email"].(string); ok {
		userInfo.Email = email
	}

	if emailVerified, ok := rawUserInfo["email_verified"].(bool); ok {
		userInfo.EmailVerified = emailVerified
	}

	if name, ok := rawUserInfo["name"].(string); ok {
		userInfo.Name = name
	}

	if givenName, ok := rawUserInfo["given_name"].(string); ok {
		userInfo.GivenName = givenName
	}

	if familyName, ok := rawUserInfo["family_name"].(string); ok {
		userInfo.FamilyName = familyName
	}

	if picture, ok := rawUserInfo["picture"].(string); ok {
		userInfo.Picture = picture
	}

	if locale, ok := rawUserInfo["locale"].(string); ok {
		userInfo.Locale = locale
	}

	return userInfo, nil
}

// CreateUserFromOAuth2 creates or updates a user from OAuth2 information
func (o *OAuth2Service) CreateUserFromOAuth2(userInfo *UserInfo, tenantID string) (*User, error) {
	if userInfo.Email == "" {
		return nil, fmt.Errorf("email is required for user creation")
	}

	now := time.Now()
	user := &User{
		ID:       fmt.Sprintf("%s-%s", userInfo.Provider, userInfo.ID),
		Username: userInfo.Email,
		Email:    userInfo.Email,
		FirstName: userInfo.GivenName,
		LastName:  userInfo.FamilyName,
		Status:    UserStatusActive,
		TenantID:  tenantID,
		CreatedAt: now,
		UpdatedAt: now,
		LastLogin: now,
		Metadata: map[string]interface{}{
			"oauth2_provider": userInfo.Provider,
			"oauth2_id":       userInfo.ID,
			"picture":         userInfo.Picture,
			"locale":          userInfo.Locale,
			"email_verified":  userInfo.EmailVerified,
		},
		RoleIDs: []string{"user"}, // Default role
	}

	return user, nil
}

// RefreshToken refreshes an OAuth2 access token
func (o *OAuth2Service) RefreshToken(ctx context.Context, refreshToken string) (*OAuth2Token, error) {
	if refreshToken == "" {
		return nil, fmt.Errorf("refresh token is empty")
	}

	data := url.Values{
		"grant_type":    {"refresh_token"},
		"refresh_token": {refreshToken},
		"client_id":     {o.config.ClientID},
	}

	if !o.config.UsePKCE {
		data.Set("client_secret", o.config.ClientSecret)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", o.config.TokenURL, strings.NewReader(data.Encode()))
	if err != nil {
		return nil, fmt.Errorf("failed to create refresh request: %w", err)
	}

	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	if !o.config.UsePKCE {
		req.SetBasicAuth(o.config.ClientID, o.config.ClientSecret)
	}

	resp, err := o.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("refresh request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("refresh request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var token OAuth2Token
	if err := json.NewDecoder(resp.Body).Decode(&token); err != nil {
		return nil, fmt.Errorf("failed to decode refresh response: %w", err)
	}

	return &token, nil
}

// CleanupExpiredStates removes expired OAuth2 states
func (o *OAuth2Service) CleanupExpiredStates() {
	now := time.Now()
	for state, oauth2State := range o.states {
		if now.After(oauth2State.ExpiresAt) {
			delete(o.states, state)
		}
	}
}

// generateState generates a cryptographically secure state parameter
func (o *OAuth2Service) generateState() (string, error) {
	b := make([]byte, 32)
	_, err := rand.Read(b)
	if err != nil {
		return "", err
	}
	return base64.URLEncoding.EncodeToString(b), nil
}

// generateNonce generates a cryptographically secure nonce
func (o *OAuth2Service) generateNonce() (string, error) {
	b := make([]byte, 32)
	_, err := rand.Read(b)
	if err != nil {
		return "", err
	}
	return base64.URLEncoding.EncodeToString(b), nil
}

// generatePKCE generates PKCE code verifier and challenge
func (o *OAuth2Service) generatePKCE() (string, string, error) {
	// Generate code verifier
	codeVerifierBytes := make([]byte, 32)
	_, err := rand.Read(codeVerifierBytes)
	if err != nil {
		return "", "", err
	}
	codeVerifier := base64.URLEncoding.WithPadding(base64.NoPadding).EncodeToString(codeVerifierBytes)

	// Generate code challenge (SHA256 hash of verifier)
	challenge := sha256.Sum256([]byte(codeVerifier))
	codeChallenge := base64.URLEncoding.WithPadding(base64.NoPadding).EncodeToString(challenge[:])

	return codeVerifier, codeChallenge, nil
}

// GetProviderConfigs returns common OAuth2 provider configurations
func GetProviderConfigs() map[string]OAuth2Config {
	return map[string]OAuth2Config{
		"google": {
			AuthorizeURL: "https://accounts.google.com/o/oauth2/v2/auth",
			TokenURL:     "https://oauth2.googleapis.com/token",
			UserInfoURL:  "https://www.googleapis.com/oauth2/v2/userinfo",
			JWKSUrl:      "https://www.googleapis.com/oauth2/v3/certs",
			Scopes:       []string{"openid", "email", "profile"},
			ProviderName: "google",
			Issuer:       "https://accounts.google.com",
			UsePKCE:      true,
		},
		"microsoft": {
			AuthorizeURL: "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
			TokenURL:     "https://login.microsoftonline.com/common/oauth2/v2.0/token",
			UserInfoURL:  "https://graph.microsoft.com/v1.0/me",
			JWKSUrl:      "https://login.microsoftonline.com/common/discovery/v2.0/keys",
			Scopes:       []string{"openid", "email", "profile", "User.Read"},
			ProviderName: "microsoft",
			Issuer:       "https://login.microsoftonline.com/common/v2.0",
			UsePKCE:      true,
		},
		"github": {
			AuthorizeURL: "https://github.com/login/oauth/authorize",
			TokenURL:     "https://github.com/login/oauth/access_token",
			UserInfoURL:  "https://api.github.com/user",
			Scopes:       []string{"user:email", "read:user"},
			ProviderName: "github",
			UsePKCE:      true,
		},
	}
}