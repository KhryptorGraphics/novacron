package auth

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestTokenAuthRefreshPostsRefreshTokenAndUpdatesTokens(t *testing.T) {
	expiresAt := time.Now().Add(time.Hour).UTC().Truncate(time.Second)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("expected POST, got %s", r.Method)
		}

		var body map[string]string
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request body: %v", err)
		}
		if body["refreshToken"] != "old-refresh" {
			t.Fatalf("expected refresh token old-refresh, got %q", body["refreshToken"])
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"token":        "new-token",
			"refreshToken": "new-refresh",
			"expiresAt":    expiresAt,
		}); err != nil {
			t.Fatalf("encode response: %v", err)
		}
	}))
	defer server.Close()

	auth := &TokenAuth{
		Token:        "old-token",
		RefreshToken: "old-refresh",
		ExpiresAt:    time.Now().Add(-time.Minute),
		RefreshURL:   server.URL,
	}
	req := httptest.NewRequest(http.MethodGet, "http://example.test/vms", nil)

	if err := auth.Apply(req); err != nil {
		t.Fatalf("apply auth: %v", err)
	}

	if got := req.Header.Get("Authorization"); got != "Bearer new-token" {
		t.Fatalf("expected refreshed authorization header, got %q", got)
	}
	if auth.Token != "new-token" {
		t.Fatalf("expected stored token to update, got %q", auth.Token)
	}
	if auth.RefreshToken != "new-refresh" {
		t.Fatalf("expected stored refresh token to update, got %q", auth.RefreshToken)
	}
	if !auth.ExpiresAt.Equal(expiresAt) {
		t.Fatalf("expected expiry %s, got %s", expiresAt, auth.ExpiresAt)
	}
}

func TestTokenAuthRefreshRequiresRefreshToken(t *testing.T) {
	auth := &TokenAuth{
		Token:      "old-token",
		ExpiresAt:  time.Now().Add(-time.Minute),
		RefreshURL: "http://example.test/api/auth/refresh",
	}
	req := httptest.NewRequest(http.MethodGet, "http://example.test/vms", nil)

	err := auth.Apply(req)
	if err == nil || !strings.Contains(err.Error(), "refresh token is required") {
		t.Fatalf("expected missing refresh token error, got %v", err)
	}
}

func TestTokenAuthRefreshRejectsNonSuccessStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "invalid refresh token", http.StatusUnauthorized)
	}))
	defer server.Close()

	auth := &TokenAuth{
		Token:        "old-token",
		RefreshToken: "old-refresh",
		ExpiresAt:    time.Now().Add(-time.Minute),
		RefreshURL:   server.URL,
	}
	req := httptest.NewRequest(http.MethodGet, "http://example.test/vms", nil)

	err := auth.Apply(req)
	if err == nil || !strings.Contains(err.Error(), "refresh request failed with status 401") {
		t.Fatalf("expected refresh status error, got %v", err)
	}
}

func TestTokenAuthApplyDoesNotRefreshBeforeExpiry(t *testing.T) {
	auth := &TokenAuth{
		Token:     "current-token",
		ExpiresAt: time.Now().Add(time.Hour),
	}
	req := httptest.NewRequest(http.MethodGet, "http://example.test/vms", nil)

	if err := auth.Apply(req); err != nil {
		t.Fatalf("apply auth: %v", err)
	}
	if got := req.Header.Get("Authorization"); got != "Bearer current-token" {
		t.Fatalf("expected current token header, got %q", got)
	}
}
