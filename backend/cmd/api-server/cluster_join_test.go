package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/gorilla/mux"
)

// nodeCredentialsTestEnv pins both credential knobs for the duration of a test:
// the fabric-wide secret and the NOVACRON_NODE_SECRETS map ("" = none). New
// credential tests go through here so an ambient value can never decide the
// outcome.
func nodeCredentialsTestEnv(t *testing.T, clusterSecret, perNode string) {
	t.Helper()
	t.Setenv("NOVACRON_MIGRATION_SECRET", clusterSecret)
	t.Setenv(nodeSecretsEnv, perNode)
}

// joinTestEnv pins the shared secret for the duration of a test. Per-node
// credentials are cleared: with an ambient NOVACRON_NODE_SECRETS set, a mapped
// node id would be verified against its own entry instead of the secret under
// test.
func joinTestEnv(t *testing.T, secret string) {
	t.Helper()
	nodeCredentialsTestEnv(t, secret, "")
}

// joinTestRequestAddr is a join request body's addr field: httptest gives a
// full URL, the protocol wants the bare host:port the joiner is reachable at.
func joinTestRequestAddr(t *testing.T, serverURL string) string {
	t.Helper()
	addr := strings.TrimPrefix(serverURL, "http://")
	if addr == serverURL || addr == "" {
		t.Fatalf("unexpected httptest server URL %q", serverURL)
	}
	return addr
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

// --- per-node credentials (NOVACRON_NODE_SECRETS) ---------------------------

// TestVerifyJoinPerNodeSecret: a join from a node id listed in
// NOVACRON_NODE_SECRETS is authenticated by THAT NODE's own credential.
func TestVerifyJoinPerNodeSecret(t *testing.T) {
	const addr = "10.0.0.2:8090"
	now := time.Now().Unix()
	nodeCredentialsTestEnv(t, "cluster-wide-secret", "node2=node2-secret,node3=node3-secret")

	req := joinTestRequest("node2", addr, now)
	req.Header.Set("X-Join-Signature", joinSignature("node2-secret", "node2", addr, now))
	if !verifyJoin(req, joinRequest{NodeID: "node2", Addr: addr, TS: now}) {
		t.Fatal("join signed with the node's own NOVACRON_NODE_SECRETS credential rejected")
	}
}

// TestVerifyJoinPerNodeSecretRejectsClusterSecretAndImpersonation is the
// adversarial half of the per-node feature. For a node id present in the map
// the cluster-wide secret no longer signs for it — even though that secret is
// configured and would have matched before this change — and a sibling node's
// credential does not either. An id with no entry keeps the old behaviour (the
// cluster-wide secret), which is what lets a partially-migrated fabric
// converge.
func TestVerifyJoinPerNodeSecretRejectsClusterSecretAndImpersonation(t *testing.T) {
	const addr = "10.0.0.2:8090"
	now := time.Now().Unix()
	nodeCredentialsTestEnv(t, "cluster-wide-secret", "node2=node2-secret,node3=node3-secret")

	cases := []struct {
		name     string
		nodeID   string
		signWith string
		want     bool
	}{
		{"own per-node credential", "node2", "node2-secret", true},
		{"cluster secret cross-signed for a mapped node", "node2", "cluster-wide-secret", false},
		{"sibling node's credential", "node2", "node3-secret", false},
		{"wrong value", "node2", "node2-secret-typo", false},
		{"unmapped id falls back to the cluster secret", "node9", "cluster-wide-secret", true},
		{"unmapped id with a mapped node's credential", "node9", "node2-secret", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := joinTestRequest(tc.nodeID, addr, now)
			req.Header.Set("X-Join-Signature", joinSignature(tc.signWith, tc.nodeID, addr, now))
			if got := verifyJoin(req, joinRequest{NodeID: tc.nodeID, Addr: addr, TS: now}); got != tc.want {
				t.Fatalf("verifyJoin(node=%s signed with %q) = %v, want %v", tc.nodeID, tc.signWith, got, tc.want)
			}
		})
	}
}

// TestVerifyJoinPerNodeSecretWithoutClusterSecret: once every node has an
// entry the shared trust root can be retired outright — per-node credentials
// still authenticate their own node, and an id with no entry (nothing left to
// fall back on) fails closed instead of joining the fabric.
func TestVerifyJoinPerNodeSecretWithoutClusterSecret(t *testing.T) {
	const (
		addr = "10.0.0.2:8090"
		cred = "node2-secret"
	)
	now := time.Now().Unix()
	nodeCredentialsTestEnv(t, "", "node2="+cred)

	req := joinTestRequest("node2", addr, now)
	req.Header.Set("X-Join-Signature", joinSignature(cred, "node2", addr, now))
	if !verifyJoin(req, joinRequest{NodeID: "node2", Addr: addr, TS: now}) {
		t.Fatal("per-node credential rejected with the cluster-wide secret retired")
	}

	unlisted := joinTestRequest("node9", addr, now)
	unlisted.Header.Set("X-Join-Signature", joinSignature(cred, "node9", addr, now))
	if verifyJoin(unlisted, joinRequest{NodeID: "node9", Addr: addr, TS: now}) {
		t.Fatal("unlisted node id accepted with no credential configured for it")
	}
}

// TestVerifyJoinEmptyNodeSecretsKeepsClusterSecret: with NOVACRON_NODE_SECRETS
// unset the fabric behaves exactly as before — the cluster-wide secret signs
// every node id, mapped or not.
func TestVerifyJoinEmptyNodeSecretsKeepsClusterSecret(t *testing.T) {
	const addr = "10.0.0.2:8090"
	now := time.Now().Unix()
	joinTestEnv(t, "cluster-wide-secret") // clears NOVACRON_NODE_SECRETS

	for _, nodeID := range []string{"node2", "node9"} {
		req := joinTestRequest(nodeID, addr, now)
		req.Header.Set("X-Join-Signature", joinSignature("cluster-wide-secret", nodeID, addr, now))
		if !verifyJoin(req, joinRequest{NodeID: nodeID, Addr: addr, TS: now}) {
			t.Fatalf("cluster-wide secret rejected for %s with no per-node map configured", nodeID)
		}
	}
}

// TestParseNodeSecrets pins the map format's edge cases: half-written entries
// are reported and never become a credential for "" (a blank id matching a
// blank secret would authorize the wrong caller), a base64 secret keeps its
// '=' padding, a duplicate id keeps the last value, and whitespace is
// tolerated.
func TestParseNodeSecrets(t *testing.T) {
	creds, problems := parseNodeSecrets(" node1 = s1 ,node2=YWJj==,node1=s1-new, ,broken,=noid,node3=")
	want := map[string]string{"node1": "s1-new", "node2": "YWJj=="}
	if len(creds) != len(want) {
		t.Fatalf("parsed %d credentials, want %d: %v", len(creds), len(want), creds)
	}
	for id, secret := range want {
		if creds[id] != secret {
			t.Fatalf("credential for %q = %q, want %q", id, creds[id], secret)
		}
	}
	for _, dropped := range []string{"", "broken", "noid", "node3"} {
		if _, ok := creds[dropped]; ok {
			t.Fatalf("malformed entry produced a credential for %q", dropped)
		}
	}
	if len(problems) != 4 {
		t.Fatalf("reported %d problems, want 4 (duplicate node1 + three malformed): %v", len(problems), problems)
	}

	for _, raw := range []string{"", "   ", " , "} {
		creds, problems := parseNodeSecrets(raw)
		if len(creds) != 0 || len(problems) != 0 {
			t.Fatalf("parseNodeSecrets(%q) = %v, %v; want empty and no problems", raw, creds, problems)
		}
	}
}

// TestNodeCredentialSelection pins the precedence of the one lookup that feeds
// signature verification, the join callback, heartbeats, dispatch and inbound
// auth: that node's own entry first, the cluster-wide secret as the fallback,
// and NO credential when neither is configured (every caller fails closed).
func TestNodeCredentialSelection(t *testing.T) {
	nodeCredentialsTestEnv(t, "cluster-wide-secret", "node2=node2-secret")

	if got, ok := nodeCredential("node2"); !ok || got != "node2-secret" {
		t.Fatalf("mapped node: got (%q, %v), want the node's own credential", got, ok)
	}
	if got, ok := nodeCredential("node9"); !ok || got != "cluster-wide-secret" {
		t.Fatalf("unmapped node: got (%q, %v), want the cluster-wide secret", got, ok)
	}
	if got, ok := nodeCredential(""); !ok || got != "cluster-wide-secret" {
		t.Fatalf("no node id: got (%q, %v), want the cluster-wide secret", got, ok)
	}

	// Shared trust root retired: a mapped node keeps its own credential, an
	// unmapped one has nothing at all.
	t.Setenv("NOVACRON_MIGRATION_SECRET", "")
	if got, ok := nodeCredential("node2"); !ok || got != "node2-secret" {
		t.Fatalf("mapped node without cluster secret: got (%q, %v)", got, ok)
	}
	if got, ok := nodeCredential("node9"); ok {
		t.Fatalf("unmapped node without cluster secret: got (%q, %v), want no credential", got, ok)
	}
}

// TestMigrationAuthAcceptsNodeCredential: the /internal/migrate/* gate (and
// every other inbound node-to-node RPC through internalSecretOK) accepts THIS
// node's own credential, keeps accepting the fabric-wide secret while it is
// configured, and refuses a credential that belongs to a different node.
func TestMigrationAuthAcceptsNodeCredential(t *testing.T) {
	t.Setenv("NOVACRON_NODE_ID", "node1")
	nodeCredentialsTestEnv(t, "cluster-wide-secret", "node1=node1-secret,node2=node2-secret")

	cases := []struct {
		name   string
		header string
		want   bool
	}{
		{"own per-node credential", "node1-secret", true},
		{"cluster-wide secret (layered rollout)", "cluster-wide-secret", true},
		{"another node's credential", "node2-secret", false},
		{"wrong value", "node1-secret-typo", false},
		{"missing header", "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := httptest.NewRequest(http.MethodPost, "/internal/migrate/incoming", nil)
			if tc.header != "" {
				r.Header.Set("X-Migration-Secret", tc.header)
			}
			if got := internalAuthOK(r); got != tc.want {
				t.Fatalf("internalAuthOK = %v, want %v", got, tc.want)
			}
			// The migrate gate must agree — one implementation, not two.
			if got := migrationAuthOK(r); got != tc.want {
				t.Fatalf("migrationAuthOK = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestMigrationAuthPerNodeOnlyWithoutClusterSecret: with the shared trust root
// retired, this node's own credential is the only way into its internal RPCs.
func TestMigrationAuthPerNodeOnlyWithoutClusterSecret(t *testing.T) {
	t.Setenv("NOVACRON_NODE_ID", "node1")
	nodeCredentialsTestEnv(t, "", "node1=node1-secret,node2=node2-secret")

	own := httptest.NewRequest(http.MethodPost, "/internal/migrate/incoming", nil)
	own.Header.Set("X-Migration-Secret", "node1-secret")
	if !migrationAuthOK(own) {
		t.Fatal("own per-node credential rejected with the cluster-wide secret retired")
	}

	foreign := httptest.NewRequest(http.MethodPost, "/internal/migrate/incoming", nil)
	foreign.Header.Set("X-Migration-Secret", "node2-secret")
	if migrationAuthOK(foreign) {
		t.Fatal("another node's credential accepted")
	}

	none := httptest.NewRequest(http.MethodPost, "/internal/migrate/incoming", nil)
	if migrationAuthOK(none) {
		t.Fatal("request with no credential accepted")
	}
}

// TestClusterJoinRoutePerNodeCredential drives the real join RPC end to end.
// The joiner is a live stub api-server that answers the receiver's callback
// ONLY with the joiner's own credential, so a 200 proves both the signature
// check and the callback used the per-node credential rather than the
// cluster-wide secret. A cluster-secret cross-signature is refused and
// registers nothing.
func TestClusterJoinRoutePerNodeCredential(t *testing.T) {
	t.Setenv("NOVACRON_NODE_ID", "receiver")
	nodeCredentialsTestEnv(t, "cluster-wide-secret", "receiver=receiver-secret,joiner=joiner-secret")

	joiner := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/internal/cluster/capacity" || r.Header.Get("X-Migration-Secret") != "joiner-secret" {
			w.WriteHeader(http.StatusForbidden)
			return
		}
		_ = json.NewEncoder(w).Encode(NodeCapacity{NodeID: "joiner", Cores: 4, MemTotalMB: 4096, Reachable: true})
	}))
	defer joiner.Close()
	joinerAddr := joinTestRequestAddr(t, joiner.URL)

	vmManager := newStubVMManager(t)
	defer vmManager.Stop()
	router := mux.NewRouter()
	registerClusterJoinRoutes(router, router.PathPrefix("/api").Subrouter(), nil, vmManager, t.TempDir())

	now := time.Now().Unix()

	cross := joinTestRequest("joiner", joinerAddr, now)
	cross.Header.Set("X-Join-Signature", joinSignature("cluster-wide-secret", "joiner", joinerAddr, now))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, cross)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("cross-signed join: got %d, want 403; body=%s", rec.Code, rec.Body.String())
	}
	if peers := vmManager.MigrationPeers(); len(peers) != 0 {
		t.Fatalf("rejected join registered peers: %v", peers)
	}

	good := joinTestRequest("joiner", joinerAddr, now)
	good.Header.Set("X-Join-Signature", joinSignature("joiner-secret", "joiner", joinerAddr, now))
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, good)
	if rec.Code != http.StatusOK {
		t.Fatalf("per-node join: got %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if addr := vmManager.MigrationPeers()["joiner"]; addr != joinerAddr {
		t.Fatalf("joiner registered at %q, want %q", addr, joinerAddr)
	}
}

// TestClusterInternalRPCAcceptsNodeCredential proves the inbound gate is wired
// onto a real internal RPC (not just the helper): the capacity endpoint accepts
// this node's own credential and the cluster-wide secret, and refuses another
// node's credential.
func TestClusterInternalRPCAcceptsNodeCredential(t *testing.T) {
	t.Setenv("NOVACRON_NODE_ID", "node1")
	nodeCredentialsTestEnv(t, "cluster-wide-secret", "node1=node1-secret,node2=node2-secret")

	router := mux.NewRouter()
	registerClusterRoutes(router, router.PathPrefix("/api").Subrouter(), nil, nil, t.TempDir())

	cases := []struct {
		name   string
		header string
		want   int
	}{
		{"own credential", "node1-secret", http.StatusOK},
		{"cluster-wide secret", "cluster-wide-secret", http.StatusOK},
		{"another node's credential", "node2-secret", http.StatusForbidden},
		{"no header", "", http.StatusForbidden},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/internal/cluster/capacity", nil)
			if tc.header != "" {
				req.Header.Set("X-Migration-Secret", tc.header)
			}
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)
			if rec.Code != tc.want {
				t.Fatalf("capacity RPC: got %d, want %d; body=%s", rec.Code, tc.want, rec.Body.String())
			}
		})
	}
}

// TestDispatchCreateToPeerUsesPeerCredential: a dispatched create presents the
// PEER's own credential when the map configures one (that peer's gate accepts
// only its own), and the cluster-wide secret for a peer with no entry.
func TestDispatchCreateToPeerUsesPeerCredential(t *testing.T) {
	nodeCredentialsTestEnv(t, "cluster-wide-secret", "peer=peer-secret")

	// peerServer accepts a create only from a caller presenting want.
	peerServer := func(t *testing.T, want string) string {
		t.Helper()
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/internal/vms/create" {
				w.WriteHeader(http.StatusNotFound)
				return
			}
			if r.Header.Get("X-Migration-Secret") != want {
				w.WriteHeader(http.StatusForbidden)
				return
			}
			writeJSON(w, http.StatusCreated, map[string]interface{}{"id": "vm-1", "node_id": "peer", "state": "stopped"})
		}))
		t.Cleanup(srv.Close)
		return joinTestRequestAddr(t, srv.URL)
	}

	t.Run("peer credential", func(t *testing.T) {
		out, err := dispatchCreateToPeerAs("peer", peerServer(t, "peer-secret"), clusterCreateSpec{Name: "vm-1"})
		if err != nil {
			t.Fatalf("dispatch under the peer's own credential failed: %v", err)
		}
		if out["id"] != "vm-1" {
			t.Fatalf("dispatch result = %v, want id vm-1", out)
		}
	})
	t.Run("cluster secret fallback", func(t *testing.T) {
		out, err := dispatchCreateToPeerAs("legacy-peer", peerServer(t, "cluster-wide-secret"), clusterCreateSpec{Name: "vm-1"})
		if err != nil {
			t.Fatalf("dispatch to a peer with no per-node entry failed: %v", err)
		}
		if out["id"] != "vm-1" {
			t.Fatalf("dispatch result = %v, want id vm-1", out)
		}
	})
}

// TestBeatOnceUsesPeerCredential: the heartbeat probes a peer with THAT peer's
// own credential — the stub peer's gate accepts only its own, so a credential
// regression here would leave every per-node peer permanently unreachable in
// the inventory and its link profile (which placement reads) stale.
func TestBeatOnceUsesPeerCredential(t *testing.T) {
	nodeCredentialsTestEnv(t, "cluster-wide-secret", "peer-a=peer-a-secret")
	t.Setenv("NOVACRON_PROBE_BYTES", "1024")

	peer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Migration-Secret") != "peer-a-secret" {
			w.WriteHeader(http.StatusForbidden)
			return
		}
		switch r.URL.Path {
		case "/internal/cluster/capacity":
			_ = json.NewEncoder(w).Encode(NodeCapacity{NodeID: "peer-a", Cores: 2, MemTotalMB: 1024, Reachable: true})
		case "/internal/cluster/probe":
			_, _ = w.Write(make([]byte, 1024))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer peer.Close()

	vmManager := newStubVMManager(t)
	defer vmManager.Stop()
	vmManager.RegisterMigrationPeer("peer-a", joinTestRequestAddr(t, peer.URL))

	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()
	// One upsert per reachable peer; the assertion below fails if the probe
	// never got that far (the stub peer answers 403 to any other credential).
	mock.ExpectExec("INSERT INTO cluster_peers").WillReturnResult(sqlmock.NewResult(0, 1))

	beatOnce(t.Context(), db, vmManager)

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("heartbeat persisted no probe result: %v", err)
	}
}
