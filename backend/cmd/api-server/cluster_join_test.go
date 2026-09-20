package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/mux"
)

// joinTestEnv pins the shared secret for the duration of a test.
func joinTestEnv(t *testing.T, secret string) {
	t.Helper()
	old, had := os.LookupEnv("NOVACRON_MIGRATION_SECRET")
	os.Setenv("NOVACRON_MIGRATION_SECRET", secret)
	t.Cleanup(func() {
		if had {
			os.Setenv("NOVACRON_MIGRATION_SECRET", old)
		} else {
			os.Unsetenv("NOVACRON_MIGRATION_SECRET")
		}
	})
}

func joinTestRequest(nodeID, addr string, ts int64) *http.Request {
	body, _ := json.Marshal(joinRequest{NodeID: nodeID, Addr: addr, TS: ts})
	req := httptest.NewRequest(http.MethodPost, "/internal/cluster/join", strings.NewReader(string(body)))
	return req
}

// TestVerifyJoinAuthenticatesGoodSignature: correct HMAC + fresh ts passes.
func TestVerifyJoinAuthenticatesGoodSignature(t *testing.T) {
	joinTestEnv(t, "test-secret")
	now := time.Now().Unix()
	req := joinTestRequest("node2", "10.0.0.2:8090", now)
	req.Header.Set("X-Join-Signature", joinSignature("test-secret", "node2", "10.0.0.2:8090", now))
	if !verifyJoin(req, joinRequest{NodeID: "node2", Addr: "10.0.0.2:8090", TS: now}) {
		t.Fatal("valid signature + fresh timestamp rejected")
	}
}

// TestVerifyJoinRejectsBadSignature: wrong secret, tampered fields, missing
// header, and empty configured secret all fail closed.
func TestVerifyJoinRejectsBadSignature(t *testing.T) {
	now := time.Now().Unix()
	good := joinSignature("test-secret", "node2", "10.0.0.2:8090", now)

	t.Run("wrong secret", func(t *testing.T) {
		joinTestEnv(t, "test-secret")
		req := joinTestRequest("node2", "10.0.0.2:8090", now)
		req.Header.Set("X-Join-Signature", joinSignature("other-secret", "node2", "10.0.0.2:8090", now))
		if verifyJoin(req, joinRequest{NodeID: "node2", Addr: "10.0.0.2:8090", TS: now}) {
			t.Fatal("wrong-secret signature accepted")
		}
	})
	t.Run("tampered addr", func(t *testing.T) {
		joinTestEnv(t, "test-secret")
		req := joinTestRequest("node2", "10.0.0.2:8090", now)
		req.Header.Set("X-Join-Signature", good)
		// signature covers a different addr than the body claims
		if verifyJoin(req, joinRequest{NodeID: "node2", Addr: "10.0.0.9:8090", TS: now}) {
			t.Fatal("tampered addr accepted")
		}
	})
	t.Run("missing header", func(t *testing.T) {
		joinTestEnv(t, "test-secret")
		req := joinTestRequest("node2", "10.0.0.2:8090", now)
		if verifyJoin(req, joinRequest{NodeID: "node2", Addr: "10.0.0.2:8090", TS: now}) {
			t.Fatal("missing signature accepted")
		}
	})
	t.Run("no configured secret", func(t *testing.T) {
		joinTestEnv(t, "")
		req := joinTestRequest("node2", "10.0.0.2:8090", now)
		req.Header.Set("X-Join-Signature", joinSignature("", "node2", "10.0.0.2:8090", now))
		if verifyJoin(req, joinRequest{NodeID: "node2", Addr: "10.0.0.2:8090", TS: now}) {
			t.Fatal("join accepted with no configured secret")
		}
	})
}

// TestVerifyJoinRejectsStaleTimestamp: replay beyond the window fails even
// with a perfectly valid signature.
func TestVerifyJoinRejectsStaleTimestamp(t *testing.T) {
	joinTestEnv(t, "test-secret")
	stale := time.Now().Add(-2 * time.Minute).Unix()
	req := joinTestRequest("node2", "10.0.0.2:8090", stale)
	req.Header.Set("X-Join-Signature", joinSignature("test-secret", "node2", "10.0.0.2:8090", stale))
	if verifyJoin(req, joinRequest{NodeID: "node2", Addr: "10.0.0.2:8090", TS: stale}) {
		t.Fatal("stale join replay accepted")
	}
}

// TestJoinRejectsUnreachableJoiner proves the peer map is not poisoned by a
// bogus addr: the join RPC must reject before registering when the
// callback probe fails, even with a valid signature.
func TestJoinRejectsUnreachableJoiner(t *testing.T) {
	joinTestEnv(t, "test-secret")
	oldNodeID := os.Getenv("NOVACRON_NODE_ID")
	os.Setenv("NOVACRON_NODE_ID", "node1")
	t.Cleanup(func() {
		if oldNodeID != "" {
			os.Setenv("NOVACRON_NODE_ID", oldNodeID)
		} else {
			os.Unsetenv("NOVACRON_NODE_ID")
		}
	})

	router := mux.NewRouter()
	// No DB: upsert is skipped, but the reachability gate still runs.
	registerClusterJoinRoutes(router, router.PathPrefix("/api").Subrouter(), nil, nil, "/tmp")

	now := time.Now().Unix()
	// Port 1 is never a listening api-server.
	req := joinTestRequest("ghost", "127.0.0.1:1", now)
	req.Header.Set("X-Join-Signature", joinSignature("test-secret", "ghost", "127.0.0.1:1", now))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("unreachable joiner: got %d, want 403; body=%s", rec.Code, rec.Body.String())
	}
}

// TestJoinRejectsSelf proves a node cannot join itself (would corrupt its
// own peer map entry) — checked before any signature work.
func TestJoinRejectsSelf(t *testing.T) {
	joinTestEnv(t, "test-secret")
	os.Setenv("NOVACRON_NODE_ID", "node1")
	t.Cleanup(func() { os.Unsetenv("NOVACRON_NODE_ID") })

	router := mux.NewRouter()
	registerClusterJoinRoutes(router, router.PathPrefix("/api").Subrouter(), nil, nil, "/tmp")

	now := time.Now().Unix()
	req := joinTestRequest("node1", "127.0.0.1:8090", now)
	req.Header.Set("X-Join-Signature", joinSignature("test-secret", "node1", "127.0.0.1:8090", now))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("self-join: got %d, want 400", rec.Code)
	}
}

// TestSelfJoinAddrEnvPrecedence: NOVACRON_JOIN_ADDR wins; fallback derives
// from API_HOST/API_PORT.
func TestSelfJoinAddrEnvPrecedence(t *testing.T) {
	for _, kv := range []string{"NOVACRON_JOIN_ADDR", "API_HOST", "API_PORT"} {
		os.Unsetenv(kv)
	}
	t.Setenv("NOVACRON_JOIN_ADDR", "nodeb.example:9443")
	if got := selfJoinAddr(); got != "nodeb.example:9443" {
		t.Fatalf("NOVACRON_JOIN_ADDR ignored: %s", got)
	}
	os.Unsetenv("NOVACRON_JOIN_ADDR")
	t.Setenv("API_HOST", "10.1.2.3")
	t.Setenv("API_PORT", "8123")
	if got := selfJoinAddr(); got != "10.1.2.3:8123" {
		t.Fatalf("API_HOST/API_PORT fallback wrong: %s", got)
	}
}
