package commands

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mitchellh/go-homedir"
	"github.com/novacron/cli/pkg/auth"
	"github.com/novacron/cli/pkg/config"
)

func TestLoginCommandAuthenticatesAndPersistsClusterAndToken(t *testing.T) {
	withTempHome(t)

	expiresAt := time.Now().Add(time.Hour).UTC().Truncate(time.Second)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/auth/login" {
			t.Fatalf("expected login path, got %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Fatalf("expected POST, got %s", r.Method)
		}

		var req map[string]string
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode login request: %v", err)
		}
		if req["email"] != "operator@example.com" || req["password"] != "secret" {
			t.Fatalf("unexpected login request: %#v", req)
		}

		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"token":        "access-token",
			"refreshToken": "refresh-token",
			"expiresAt":    expiresAt,
		})
	}))
	defer server.Close()

	cmd := NewLoginCommand()
	cmd.SetOut(&bytes.Buffer{})
	cmd.SetArgs([]string{
		"--cluster", "prod",
		"--server", server.URL,
		"--email", "operator@example.com",
		"--password", "secret",
	})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("login command failed: %v", err)
	}

	manager, err := config.NewManager("")
	if err != nil {
		t.Fatalf("new config manager: %v", err)
	}
	cluster, err := manager.GetCurrentCluster()
	if err != nil {
		t.Fatalf("get current cluster: %v", err)
	}
	if cluster.Name != "prod" || cluster.Server != server.URL || cluster.AuthType != "token" {
		t.Fatalf("unexpected persisted cluster: %#v", cluster)
	}
	if strings.Contains(cluster.AuthData, "access-token") || strings.Contains(cluster.AuthData, "refresh-token") {
		t.Fatalf("cluster auth data should not contain bearer credentials: %#v", cluster)
	}

	store, err := auth.NewTokenStore()
	if err != nil {
		t.Fatalf("new token store: %v", err)
	}
	token, err := store.Load("prod")
	if err != nil {
		t.Fatalf("load token: %v", err)
	}
	if token.Token != "access-token" || token.RefreshToken != "refresh-token" {
		t.Fatalf("unexpected stored token: %#v", token)
	}
	if token.RefreshURL != server.URL+"/api/auth/refresh" {
		t.Fatalf("unexpected refresh URL: %s", token.RefreshURL)
	}
	if !token.ExpiresAt.Equal(expiresAt) {
		t.Fatalf("expected expiry %s, got %s", expiresAt, token.ExpiresAt)
	}
}

func TestNewClusterAPIClientAppliesStoredToken(t *testing.T) {
	withTempHome(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer stored-token" {
			t.Fatalf("expected stored bearer token, got %q", got)
		}
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}))
	defer server.Close()

	store, err := auth.NewTokenStore()
	if err != nil {
		t.Fatalf("new token store: %v", err)
	}
	if err := store.Save("prod", &auth.TokenAuth{
		Token:        "stored-token",
		RefreshToken: "stored-refresh",
		ExpiresAt:    time.Now().Add(time.Hour),
	}); err != nil {
		t.Fatalf("save token: %v", err)
	}

	client, err := newClusterAPIClient(&config.Cluster{
		Name:     "prod",
		Server:   server.URL,
		AuthType: "token",
	})
	if err != nil {
		t.Fatalf("new cluster API client: %v", err)
	}

	var result map[string]string
	if err := client.Get(t.Context(), "/probe", &result); err != nil {
		t.Fatalf("client get: %v", err)
	}
	if result["status"] != "ok" {
		t.Fatalf("unexpected response: %#v", result)
	}
}

func TestLoginCommandUsesExistingClusterServerAndNamespace(t *testing.T) {
	withTempHome(t)

	expiresAt := time.Now().Add(time.Hour).UTC().Truncate(time.Second)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/auth/login" {
			t.Fatalf("expected login path, got %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"token":        "access-token",
			"refreshToken": "refresh-token",
			"expiresAt":    expiresAt,
		})
	}))
	defer server.Close()

	manager, err := config.NewManager("")
	if err != nil {
		t.Fatalf("new config manager: %v", err)
	}
	if err := manager.AddCluster(config.Cluster{
		Name:      "prod",
		Server:    server.URL,
		Namespace: "existing-namespace",
		Insecure:  true,
	}); err != nil {
		t.Fatalf("add cluster: %v", err)
	}

	cmd := NewLoginCommand()
	cmd.SetOut(&bytes.Buffer{})
	cmd.SetArgs([]string{
		"--cluster", "prod",
		"--email", "operator@example.com",
		"--password", "secret",
	})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("login command failed: %v", err)
	}

	manager, err = config.NewManager("")
	if err != nil {
		t.Fatalf("reload config manager: %v", err)
	}
	cluster, err := manager.GetCurrentCluster()
	if err != nil {
		t.Fatalf("get current cluster: %v", err)
	}
	if cluster.Server != server.URL || cluster.Namespace != "existing-namespace" || !cluster.Insecure {
		t.Fatalf("expected existing cluster connection settings to be preserved, got %#v", cluster)
	}
}

func TestNewClusterAPIClientRequiresTokenForTokenCluster(t *testing.T) {
	withTempHome(t)

	_, err := newClusterAPIClient(&config.Cluster{
		Name:     "prod",
		Server:   "http://example.test",
		AuthType: "token",
	})
	if err == nil || !strings.Contains(err.Error(), "load auth token for cluster prod") {
		t.Fatalf("expected missing token error, got %v", err)
	}
}

func withTempHome(t *testing.T) {
	t.Helper()

	tempDir := t.TempDir()
	homedir.DisableCache = true
	t.Setenv("HOME", tempDir)
	t.Setenv("NOVACRON_AUTH_REFRESH_URL", "")
	cfgFile = filepath.Join(tempDir, ".novacron", "config.yaml")
	clusterName = ""
	insecure = false
	t.Cleanup(func() {
		cfgFile = ""
		clusterName = ""
		insecure = false
	})
}
