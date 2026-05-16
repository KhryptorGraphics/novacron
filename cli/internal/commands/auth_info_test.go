package commands

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/novacron/cli/pkg/auth"
	"github.com/novacron/cli/pkg/config"
)

func TestAuthInfoReportsCurrentClusterTokenStatus(t *testing.T) {
	withTempHome(t)

	manager, err := config.NewManager("")
	if err != nil {
		t.Fatalf("new config manager: %v", err)
	}
	if err := manager.AddCluster(config.Cluster{Name: "prod", Server: "https://prod.example.com", AuthType: "token"}); err != nil {
		t.Fatalf("add cluster: %v", err)
	}
	store, err := auth.NewTokenStore()
	if err != nil {
		t.Fatalf("new token store: %v", err)
	}
	if err := store.Save("prod", &auth.TokenAuth{
		Token:        "secret-access-token",
		RefreshToken: "secret-refresh-token",
		ExpiresAt:    time.Now().Add(time.Hour).UTC().Truncate(time.Second),
		RefreshURL:   "https://prod.example.com/api/auth/refresh",
	}); err != nil {
		t.Fatalf("save token: %v", err)
	}

	var output bytes.Buffer
	cmd := NewAuthCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"info"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("auth info failed: %v", err)
	}

	for _, expected := range []string{"CLUSTER", "prod", "AUTH TYPE", "token", "TOKEN STATUS", "valid", "REFRESH TOKEN", "available"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected auth info to contain %q, got:\n%s", expected, output.String())
		}
	}
	for _, secret := range []string{"secret-access-token", "secret-refresh-token"} {
		if strings.Contains(output.String(), secret) {
			t.Fatalf("auth info leaked secret %q in output:\n%s", secret, output.String())
		}
	}
}

func TestAuthInfoReportsExpiredToken(t *testing.T) {
	withTempHome(t)

	manager, err := config.NewManager("")
	if err != nil {
		t.Fatalf("new config manager: %v", err)
	}
	if err := manager.AddCluster(config.Cluster{Name: "prod", Server: "https://prod.example.com", AuthType: "token"}); err != nil {
		t.Fatalf("add cluster: %v", err)
	}
	store, err := auth.NewTokenStore()
	if err != nil {
		t.Fatalf("new token store: %v", err)
	}
	if err := store.Save("prod", &auth.TokenAuth{
		Token:     "expired-token",
		ExpiresAt: time.Now().Add(-time.Minute),
	}); err != nil {
		t.Fatalf("save token: %v", err)
	}

	var output bytes.Buffer
	cmd := NewAuthCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"info"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("auth info failed: %v", err)
	}

	if !strings.Contains(output.String(), "expired") {
		t.Fatalf("expected expired token status, got:\n%s", output.String())
	}
}
