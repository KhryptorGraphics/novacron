package auth

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/mitchellh/go-homedir"
)

// Authenticator applies authentication to requests
type Authenticator interface {
	Apply(req *http.Request) error
	Refresh() error
}

// TokenAuth implements token-based authentication
type TokenAuth struct {
	Token        string    `json:"token"`
	RefreshToken string    `json:"refreshToken,omitempty"`
	ExpiresAt    time.Time `json:"expiresAt"`
}

// Apply applies token authentication to a request
func (t *TokenAuth) Apply(req *http.Request) error {
	if t.Token == "" {
		return fmt.Errorf("no authentication token available")
	}

	if time.Now().After(t.ExpiresAt) {
		if err := t.Refresh(); err != nil {
			return fmt.Errorf("token expired and refresh failed: %w", err)
		}
	}

	req.Header.Set("Authorization", "Bearer "+t.Token)
	return nil
}

// Refresh refreshes the authentication token
func (t *TokenAuth) Refresh() error {
	// TODO: Implement token refresh logic
	return fmt.Errorf("token refresh not implemented")
}

// APIKeyAuth implements API key authentication
type APIKeyAuth struct {
	APIKey string
	Header string
}

// Apply applies API key authentication to a request
func (a *APIKeyAuth) Apply(req *http.Request) error {
	if a.APIKey == "" {
		return fmt.Errorf("no API key available")
	}

	header := a.Header
	if header == "" {
		header = "X-API-Key"
	}

	req.Header.Set(header, a.APIKey)
	return nil
}

// Refresh for API key auth (no-op)
func (a *APIKeyAuth) Refresh() error {
	return nil
}

// TokenStore manages authentication tokens
type TokenStore struct {
	path string
}

// NewTokenStore creates a new token store
func NewTokenStore() (*TokenStore, error) {
	home, err := homedir.Dir()
	if err != nil {
		return nil, err
	}

	tokenPath := filepath.Join(home, ".novacron", "tokens.json")
	return &TokenStore{path: tokenPath}, nil
}

// Save saves tokens to disk
func (s *TokenStore) Save(cluster string, auth *TokenAuth) error {
	// Ensure directory exists
	dir := filepath.Dir(s.path)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return err
	}

	// Load existing tokens
	tokens, err := s.loadAll()
	if err != nil {
		tokens = make(map[string]*TokenAuth)
	}

	// Update token for cluster
	tokens[cluster] = auth

	// Save to file
	data, err := json.MarshalIndent(tokens, "", "  ")
	if err != nil {
		return err
	}

	return ioutil.WriteFile(s.path, data, 0600)
}

// Load loads tokens for a cluster
func (s *TokenStore) Load(cluster string) (*TokenAuth, error) {
	tokens, err := s.loadAll()
	if err != nil {
		return nil, err
	}

	auth, ok := tokens[cluster]
	if !ok {
		return nil, fmt.Errorf("no tokens found for cluster %s", cluster)
	}

	return auth, nil
}

// Delete deletes tokens for a cluster
func (s *TokenStore) Delete(cluster string) error {
	tokens, err := s.loadAll()
	if err != nil {
		return err
	}

	delete(tokens, cluster)

	data, err := json.MarshalIndent(tokens, "", "  ")
	if err != nil {
		return err
	}

	return ioutil.WriteFile(s.path, data, 0600)
}

// loadAll loads all tokens
func (s *TokenStore) loadAll() (map[string]*TokenAuth, error) {
	data, err := ioutil.ReadFile(s.path)
	if err != nil {
		if os.IsNotExist(err) {
			return make(map[string]*TokenAuth), nil
		}
		return nil, err
	}

	var tokens map[string]*TokenAuth
	if err := json.Unmarshal(data, &tokens); err != nil {
		return nil, err
	}

	return tokens, nil
}

// ParseToken parses a JWT token
func ParseToken(tokenString string) (*jwt.Token, error) {
	return jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		// In production, you'd verify the signing key here
		return []byte("secret"), nil
	})
}

// GetTokenExpiry gets the expiry time from a JWT token
func GetTokenExpiry(tokenString string) (time.Time, error) {
	token, err := ParseToken(tokenString)
	if err != nil {
		return time.Time{}, err
	}

	if claims, ok := token.Claims.(jwt.MapClaims); ok {
		if exp, ok := claims["exp"].(float64); ok {
			return time.Unix(int64(exp), 0), nil
		}
	}

	return time.Time{}, fmt.Errorf("no expiry found in token")
}