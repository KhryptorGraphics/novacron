package vm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

const migrationCredentialTarget = "dest-node"

func migrationCredentialSenders() map[string]func(context.Context, string, IncomingMigrationRequest) error {
	return map[string]func(context.Context, string, IncomingMigrationRequest) error{
		"shared-storage": func(ctx context.Context, addr string, req IncomingMigrationRequest) error {
			_, err := requestIncomingMigration(ctx, addr, req)
			return err
		},
		"block": func(ctx context.Context, addr string, req IncomingMigrationRequest) error {
			_, _, err := requestIncomingBlockMigration(ctx, addr, req)
			return err
		},
	}
}

func TestIncomingMigrationRequestsUsePerNodeSecretWithoutClusterSecret(t *testing.T) {
	for name, sender := range migrationCredentialSenders() {
		t.Run(name, func(t *testing.T) {
			t.Setenv("NOVACRON_MIGRATION_SECRET", "")
			t.Setenv("NOVACRON_NODE_SECRETS", "other-node=other-secret, dest-node=dest-secret")
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if got := r.Header.Get("X-Migration-Secret"); got != "dest-secret" {
					t.Errorf("X-Migration-Secret = %q, want destination credential", got)
				}
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(IncomingMigrationResponse{Port: 4444, NBDURI: "nbd://127.0.0.1:10809"})
			}))
			defer server.Close()

			err := sender(context.Background(), strings.TrimPrefix(server.URL, "http://"), IncomingMigrationRequest{VMID: "vm-test", TargetNodeID: migrationCredentialTarget})
			if err != nil {
				t.Fatalf("migration request failed: %v", err)
			}
		})
	}
}

func TestIncomingMigrationRequestsAreRejectedWithoutCredential(t *testing.T) {
	for name, sender := range migrationCredentialSenders() {
		t.Run(name, func(t *testing.T) {
			t.Setenv("NOVACRON_MIGRATION_SECRET", "")
			t.Setenv("NOVACRON_NODE_SECRETS", "")
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if got := r.Header.Get("X-Migration-Secret"); got != "" {
					t.Errorf("X-Migration-Secret = %q, want no credential", got)
				}
				http.Error(w, "migration authentication required", http.StatusForbidden)
			}))
			defer server.Close()

			err := sender(context.Background(), strings.TrimPrefix(server.URL, "http://"), IncomingMigrationRequest{VMID: "vm-test", TargetNodeID: migrationCredentialTarget})
			if err == nil || !strings.Contains(err.Error(), "403 Forbidden") {
				t.Fatalf("migration request error = %v, want destination 403", err)
			}
		})
	}
}

func TestIncomingMigrationRequestsPreferSharedSecret(t *testing.T) {
	for name, sender := range migrationCredentialSenders() {
		t.Run(name, func(t *testing.T) {
			t.Setenv("NOVACRON_MIGRATION_SECRET", "shared-secret")
			t.Setenv("NOVACRON_NODE_SECRETS", "dest-node=per-node-secret")
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if got := r.Header.Get("X-Migration-Secret"); got != "shared-secret" {
					t.Errorf("X-Migration-Secret = %q, want shared secret", got)
				}
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(IncomingMigrationResponse{Port: 4444, NBDURI: "nbd://127.0.0.1:10809"})
			}))
			defer server.Close()

			err := sender(context.Background(), strings.TrimPrefix(server.URL, "http://"), IncomingMigrationRequest{VMID: "vm-test", TargetNodeID: migrationCredentialTarget})
			if err != nil {
				t.Fatalf("migration request failed: %v", err)
			}
		})
	}
}
