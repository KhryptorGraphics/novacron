package main

// Signed cluster-join protocol for the fabric (P1/G1).
//
// A joining node POSTs /internal/cluster/join to any known node with
// {node_id, addr, ts} and an HMAC-SHA256 signature over "node_id|addr|ts"
// keyed by the shared NOVACRON_MIGRATION_SECRET. The receiver:
//   1. verifies the signature (constant-time) and freshness (|now-ts| < 60s),
//   2. calls the joiner back at addr (/internal/cluster/capacity, same
//      secret) — an unreachable joiner is REJECTED so a bogus address can
//      never poison the peer map,
//   3. registers the peer (RegisterMigrationPeer — cluster dispatch and
//      migration resolution work immediately),
//   4. upserts the persisted cluster_peers row (membership survives restarts
//      without env editing), recording the callback's measured RTT, and
//   5. returns its own node id/addr plus its full peer list so the joiner
//      converges on the complete membership (gossip fan-out in one hop).
//
// A background heartbeat loop keeps last_heartbeat/last_rtt_ms fresh for
// every peer; /api/cluster/nodes exposes the live capacity + link profile
// per node.
//
// NOVACRON_PEERS remains a static bootstrap override (loaded first, so an
// operator can always pin a seed); NOVACRON_JOIN_PEERS is the new-node
// convenience: a comma list of known-node addrs to POST the join to at boot.

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gorilla/mux"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

// joinTimestampWindow bounds replay of a captured join request.
const joinTimestampWindow = 60 * time.Second

// heartbeatInterval is how often the heartbeat loop probes every peer.
// Liveness is judged per-use by the live capacity fetch (Reachable=false is
// honest); the loop only refreshes the persisted link profile.
const heartbeatInterval = 30 * time.Second

// joinRequest is the signed join payload.
type joinRequest struct {
	NodeID string `json:"node_id"`
	Addr   string `json:"addr"` // the joiner's own api addr, host:port
	TS     int64  `json:"ts"`   // unix seconds; bounds replay
}

// joinSignature computes the HMAC the joiner sends in X-Join-Signature.
func joinSignature(secret, nodeID, addr string, ts int64) string {
	mac := hmac.New(sha256.New, []byte(secret))
	fmt.Fprintf(mac, "%s|%s|%d", nodeID, addr, ts)
	return hex.EncodeToString(mac.Sum(nil))
}

// verifyJoin authenticates a join request: correct HMAC over the exact
// payload fields, constant-time compare, timestamp inside the freshness
// window. Empty configured secret fails closed (no secret, no joins).
func verifyJoin(r *http.Request, req joinRequest) bool {
	secret := os.Getenv("NOVACRON_MIGRATION_SECRET")
	if secret == "" {
		return false
	}
	got := r.Header.Get("X-Join-Signature")
	if got == "" {
		return false
	}
	want := joinSignature(secret, req.NodeID, req.Addr, req.TS)
	if !hmac.Equal([]byte(got), []byte(want)) {
		return false
	}
	now := time.Now().Unix()
	if now-req.TS > int64(joinTimestampWindow/time.Second) || req.TS-now > int64(joinTimestampWindow/time.Second) {
		return false
	}
	return true
}

// upsertClusterPeer persists (or refreshes) a peer's membership row.
func upsertClusterPeer(ctx context.Context, db *sql.DB, nodeID, addr string, rttMS float64) error {
	link, _ := json.Marshal(map[string]interface{}{"rtt_ms": rttMS})
	_, err := db.ExecContext(ctx, `
		INSERT INTO cluster_peers (node_id, addr, last_heartbeat, last_rtt_ms, link, updated_at)
		VALUES ($1, $2, NOW(), $3, $4, NOW())
		ON CONFLICT (node_id) DO UPDATE SET
			addr = EXCLUDED.addr,
			last_heartbeat = EXCLUDED.last_heartbeat,
			last_rtt_ms = EXCLUDED.last_rtt_ms,
			link = EXCLUDED.link,
			updated_at = NOW()
	`, nodeID, addr, rttMS, link)
	return err
}

// deleteClusterPeer removes a peer's membership row.
func deleteClusterPeer(ctx context.Context, db *sql.DB, nodeID string) error {
	_, err := db.ExecContext(ctx, `DELETE FROM cluster_peers WHERE node_id = $1`, nodeID)
	return err
}

// loadPersistedPeers seeds the peer map from cluster_peers rows at boot.
// A DB hiccup just means this boot starts with the env-configured peers
// only; the next successful join/heartbeat re-persists.
func loadPersistedPeers(vmManager *core_vm.VMManager, db *sql.DB) {
	if vmManager == nil || db == nil {
		return
	}
	rows, err := db.Query(`SELECT node_id, addr FROM cluster_peers`)
	if err != nil {
		log.Printf("cluster_peers load skipped: %v", err)
		return
	}
	defer rows.Close()
	n := 0
	for rows.Next() {
		var id, addr string
		if err := rows.Scan(&id, &addr); err != nil {
			continue
		}
		vmManager.RegisterMigrationPeer(id, addr)
		n++
	}
	if n > 0 {
		log.Printf("loaded %d persisted cluster peer(s)", n)
	}
}

// selfJoinAddr returns this node's advertised api addr for join responses.
// NOVACRON_JOIN_ADDR (host:port) if set; else API_HOST:API_PORT from env.
func selfJoinAddr() string {
	if a := strings.TrimSpace(os.Getenv("NOVACRON_JOIN_ADDR")); a != "" {
		return a
	}
	host := strings.TrimSpace(os.Getenv("API_HOST"))
	if host == "" {
		host = "127.0.0.1"
	}
	port := strings.TrimSpace(os.Getenv("API_PORT"))
	if port == "" {
		port = "8090"
	}
	return host + ":" + port
}

// registerClusterJoinRoutes wires the join/leave node-to-node RPCs and the
// authed /api/cluster/nodes inventory with live link profiles.
func registerClusterJoinRoutes(root *mux.Router, apiRouter *mux.Router, db *sql.DB, vmManager *core_vm.VMManager, storagePath string) {
	// POST /internal/cluster/join — signed node-to-node.
	root.HandleFunc("/internal/cluster/join", func(w http.ResponseWriter, r *http.Request) {
		var req joinRequest
		if err := json.NewDecoder(io.LimitReader(r.Body, 64<<10)).Decode(&req); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}
		req.NodeID = strings.TrimSpace(req.NodeID)
		req.Addr = strings.TrimSpace(req.Addr)
		if req.NodeID == "" || req.Addr == "" {
			writeJSONError(w, http.StatusBadRequest, "node_id and addr are required")
			return
		}
		if req.NodeID == selfNodeID() {
			writeJSONError(w, http.StatusBadRequest, "cannot join yourself")
			return
		}
		if !verifyJoin(r, req) {
			writeJSONError(w, http.StatusForbidden, "invalid or stale join signature")
			return
		}

		// Verify the joiner is actually reachable and answering with the
		// shared secret BEFORE touching the peer map — a bogus addr must
		// never poison cluster dispatch. The measured round trip doubles as
		// the initial link-profile RTT.
		start := time.Now()
		cap, err := fetchPeerCapacityWithSecret(req.Addr, os.Getenv("NOVACRON_MIGRATION_SECRET"))
		if err != nil {
			writeJSONError(w, http.StatusForbidden, fmt.Sprintf("joiner not reachable at %s: %v", req.Addr, err))
			return
		}
		rttMS := float64(time.Since(start).Microseconds()) / 1000.0

		vmManager.RegisterMigrationPeer(req.NodeID, req.Addr)
		if db != nil {
			if err := upsertClusterPeer(r.Context(), db, req.NodeID, req.Addr, rttMS); err != nil {
				log.Printf("cluster_peers upsert failed for %s: %v", req.NodeID, err)
			}
		}

		// Return our own identity plus the full peer list so the joiner can
		// register everyone in one hop.
		peers := vmManager.MigrationPeers()
		peerList := make([]map[string]interface{}, 0, len(peers))
		for id, addr := range peers {
			peerList = append(peerList, map[string]interface{}{"node_id": id, "addr": addr})
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"joined":  true,
			"node_id": selfNodeID(),
			"addr":    selfJoinAddr(),
			"peers":   peerList,
			"joiner_probe": map[string]interface{}{
				"reachable": true, "rtt_ms": rttMS,
				"cores": cap.Cores, "mem_total_mb": cap.MemTotalMB,
			},
		})
	}).Methods(http.MethodPost)

	// POST /internal/cluster/leave — shared-secret node-to-node.
	root.HandleFunc("/internal/cluster/leave", func(w http.ResponseWriter, r *http.Request) {
		if !internalSecretOK(r) {
			writeJSONError(w, http.StatusForbidden, "forbidden")
			return
		}
		var req struct {
			NodeID string `json:"node_id"`
		}
		if err := json.NewDecoder(io.LimitReader(r.Body, 64<<10)).Decode(&req); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}
		req.NodeID = strings.TrimSpace(req.NodeID)
		if req.NodeID == "" {
			writeJSONError(w, http.StatusBadRequest, "node_id is required")
			return
		}
		vmManager.UnregisterMigrationPeer(req.NodeID)
		if db != nil {
			if err := deleteClusterPeer(r.Context(), db, req.NodeID); err != nil {
				log.Printf("cluster_peers delete failed for %s: %v", req.NodeID, err)
			}
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{"left": true, "node_id": req.NodeID})
	}).Methods(http.MethodPost)

	// GET /api/cluster/nodes — authed: live capacity per node + link profile
	// from the persisted heartbeat table.
	apiRouter.HandleFunc("/cluster/nodes", func(w http.ResponseWriter, r *http.Request) {
		nodes := nodeProfiles(vmManager, storagePath, db)
		writeJSON(w, http.StatusOK, map[string]interface{}{"nodes": nodes})
	}).Methods(http.MethodGet)
}

// nodeProfile extends NodeCapacity with the persisted link profile.
type nodeProfile struct {
	NodeCapacity
	Link *linkProfile `json:"link,omitempty"`
}

// linkProfile is the measured link state from the heartbeat loop (nil for a
// peer that has never been probed).
type linkProfile struct {
	RTTMS         float64 `json:"rtt_ms"`
	LastHeartbeat string  `json:"last_heartbeat"`
	Stale         bool    `json:"stale"`
}

// linkProfileStaleness: a profile older than this is marked stale; placement
// (P3) must re-probe before trusting it.
const linkProfileStaleness = 5 * time.Minute

// nodeProfiles assembles live capacities plus persisted link profiles.
// A DB error degrades to capacities without profiles (the fabric keeps
// working; only profile freshness is lost).
func nodeProfiles(vmManager *core_vm.VMManager, storagePath string, db *sql.DB) []nodeProfile {
	caps := allNodeCapacities(vmManager, storagePath)
	profiles := map[string]*linkProfile{}
	if db != nil {
		if rows, err := db.Query(`SELECT node_id, last_rtt_ms, last_heartbeat FROM cluster_peers`); err == nil {
			for rows.Next() {
				var id string
				var rtt sql.NullFloat64
				var beat sql.NullTime
				if err := rows.Scan(&id, &rtt, &beat); err != nil {
					continue
				}
				p := &linkProfile{}
				if rtt.Valid {
					p.RTTMS = rtt.Float64
				}
				if beat.Valid {
					p.LastHeartbeat = beat.Time.UTC().Format(time.RFC3339)
					p.Stale = time.Since(beat.Time) > linkProfileStaleness
				}
				profiles[id] = p
			}
			rows.Close()
		}
	}
	out := make([]nodeProfile, 0, len(caps))
	for _, c := range caps {
		out = append(out, nodeProfile{NodeCapacity: c, Link: profiles[c.NodeID]})
	}
	return out
}

// clusterHeartbeatLoop refreshes last_heartbeat/last_rtt_ms for every peer
// until ctx is done. Best-effort: DB failures log and continue.
func clusterHeartbeatLoop(ctx context.Context, db *sql.DB, vmManager *core_vm.VMManager) {
	t := time.NewTicker(heartbeatInterval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			beatOnce(ctx, db, vmManager)
		}
	}
}

func beatOnce(ctx context.Context, db *sql.DB, vmManager *core_vm.VMManager) {
	if vmManager == nil || db == nil {
		return
	}
	secret := os.Getenv("NOVACRON_MIGRATION_SECRET")
	for id, addr := range vmManager.MigrationPeers() {
		start := time.Now()
		if _, err := fetchPeerCapacityWithSecret(addr, secret); err != nil {
			continue // unreachable peer stays in the map; inventory reports it honestly
		}
		rttMS := float64(time.Since(start).Microseconds()) / 1000.0
		bctx, cancel := context.WithTimeout(ctx, 5*time.Second)
		if err := upsertClusterPeer(bctx, db, id, addr, rttMS); err != nil {
			log.Printf("heartbeat upsert failed for %s: %v", id, err)
		}
		cancel()
	}
}

// joinOutcome is what a successful join to one seed gave back.
type joinOutcome struct {
	seedNodeID string
	seedAddr   string
}

// sendJoin POSTs a signed join to one known node and registers every peer
// it reports back (including that seed node itself). Returns the seed's
// identity on success.
func sendJoin(ctx context.Context, targetAddr, selfID, selfAddr, secret string, vmManager *core_vm.VMManager) (*joinOutcome, error) {
	body := joinRequest{NodeID: selfID, Addr: selfAddr, TS: time.Now().Unix()}
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://"+targetAddr+"/internal/cluster/join", strings.NewReader(string(payload)))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Join-Signature", joinSignature(secret, body.NodeID, body.Addr, body.TS))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("join to %s rejected: %s", targetAddr, resp.Status)
	}
	var out struct {
		NodeID string `json:"node_id"`
		Addr   string `json:"addr"`
		Peers  []struct {
			NodeID string `json:"node_id"`
			Addr   string `json:"addr"`
		} `json:"peers"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&out); err != nil {
		return nil, err
	}
	if out.NodeID != "" && out.Addr != "" && out.NodeID != selfID {
		vmManager.RegisterMigrationPeer(out.NodeID, out.Addr)
	}
	for _, p := range out.Peers {
		if p.NodeID != "" && p.Addr != "" && p.NodeID != selfID {
			vmManager.RegisterMigrationPeer(p.NodeID, p.Addr)
		}
	}
	return &joinOutcome{seedNodeID: out.NodeID, seedAddr: out.Addr}, nil
}

// joinAtBoot POSTs joins to every addr in NOVACRON_JOIN_PEERS (comma list).
// Failures are logged, not fatal — a node with no reachable seed still
// comes up as a single-node fabric and can be joined by others.
func joinAtBoot(ctx context.Context, selfID string, vmManager *core_vm.VMManager, db *sql.DB) {
	raw := strings.TrimSpace(os.Getenv("NOVACRON_JOIN_PEERS"))
	if raw == "" {
		return
	}
	secret := os.Getenv("NOVACRON_MIGRATION_SECRET")
	if secret == "" {
		log.Printf("NOVACRON_JOIN_PEERS set but NOVACRON_MIGRATION_SECRET is not — skipping join (fail closed)")
		return
	}
	selfAddr := selfJoinAddr()
	for _, addr := range strings.Split(raw, ",") {
		addr = strings.TrimSpace(addr)
		if addr == "" {
			continue
		}
		jctx, cancel := context.WithTimeout(ctx, 15*time.Second)
		if _, err := sendJoin(jctx, addr, selfID, selfAddr, secret, vmManager); err != nil {
			log.Printf("join to %s failed: %v", addr, err)
		} else {
			log.Printf("joined fabric via %s", addr)
		}
		cancel()
	}
}
