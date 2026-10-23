package main

// Signed cluster-join protocol for the fabric (P1/G1).
//
// A joining node POSTs /internal/cluster/join to any known node with
// {node_id, addr, ts} and an HMAC-SHA256 signature over "node_id|addr|ts"
// keyed by the CLAIMED NODE ID's credential: that node's own
// NOVACRON_NODE_SECRETS entry when the operator configured one, else the
// fabric-wide NOVACRON_MIGRATION_SECRET (see nodeCredential). The receiver:
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
// every peer and reports peer failure/recovery on the orchestration event bus
// (peerLiveness); /api/cluster/nodes exposes the live capacity + link profile
// per node.
//
// Credentials. NOVACRON_MIGRATION_SECRET is the fabric-wide secret and stays
// the trust root when nothing else is configured. NOVACRON_NODE_SECRETS
// ("node-id=secret,node-id2=secret2") layers a credential PER NODE on top of
// it and is the single source of truth for "what is node N's credential":
//   - a join from a node id listed in the map is verified ONLY against that
//     node's entry — the cluster-wide secret is rejected for it, so a leaked
//     cluster secret can no longer impersonate a configured node (verifyJoin);
//   - an outbound RPC to a peer presents the PEER's entry, so a node can only
//     drive peers whose credential the operator handed it (nodeCredential);
//   - an inbound RPC is accepted when it presents THIS node's entry, plus the
//     cluster-wide secret while the fabric is mid-rollout (internalAuthOK).
// One leaked credential is therefore scoped to one node instead of the whole
// fabric, and a node can be re-keyed or revoked without rotating every other
// node. Give every node an entry and unset NOVACRON_MIGRATION_SECRET to retire
// the shared trust root entirely: a node id with no entry then fails closed on
// both sides instead of falling back to the cluster-wide secret.
//
// NOVACRON_PEERS remains a static bootstrap override (loaded first, so an
// operator can always pin a seed); NOVACRON_JOIN_PEERS is the new-node
// convenience: a comma list of known-node addrs to POST the join to at boot.

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"crypto/subtle"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/events"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

// joinTimestampWindow bounds replay of a captured join request.
const joinTimestampWindow = 60 * time.Second

// heartbeatInterval is how often the heartbeat loop probes every peer.
// Placement still judges reachability per-use by the live capacity fetch
// (Reachable=false is honest); the loop refreshes the persisted link profile
// and feeds peerLiveness.
const heartbeatInterval = 30 * time.Second

// peerFailureThreshold is how many consecutive failed heartbeat probes mark a
// peer failed (3 x heartbeatInterval = 90s): one missed beat is a blip.
const peerFailureThreshold = 3

// nodeSecretsEnv configures per-node credentials, layered on top of the
// fabric-wide NOVACRON_MIGRATION_SECRET. Format:
// "node-id=secret,node-id2=secret2".
const nodeSecretsEnv = "NOVACRON_NODE_SECRETS"

// parseNodeSecrets reads a NOVACRON_NODE_SECRETS value into node id -> secret,
// plus the problems worth logging once at boot. Half-written entries (no id or
// no secret) are DROPPED, never treated as a credential for "": an empty key
// would let a request claiming a blank node id match a blank secret.
// Whitespace around both fields is trimmed; the secret may itself contain '='
// (base64), so only the FIRST '=' separates a pair. A repeated id keeps the
// LAST value (a re-key appended at the end wins) and says so — a silent drop
// would leave an operator believing a node's new credential is in force.
func parseNodeSecrets(raw string) (map[string]string, []string) {
	var problems []string
	if strings.TrimSpace(raw) == "" {
		return nil, nil
	}
	creds := map[string]string{}
	for _, entry := range strings.Split(raw, ",") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		id, secret, hasSep := strings.Cut(entry, "=")
		id, secret = strings.TrimSpace(id), strings.TrimSpace(secret)
		if !hasSep || id == "" || secret == "" {
			problems = append(problems, fmt.Sprintf("ignoring malformed entry %q (want node-id=secret)", entry))
			continue
		}
		if _, dup := creds[id]; dup {
			problems = append(problems, fmt.Sprintf("duplicate entry for %q — the last value wins", id))
		}
		creds[id] = secret
	}
	return creds, problems
}

// nodeSecrets is parseNodeSecrets against the live env. Read per call: env is
// the config source and tests flip it at runtime. Problems are reported once
// at boot by logNodeSecretConfig, never on the request path.
func nodeSecrets() map[string]string {
	creds, _ := parseNodeSecrets(os.Getenv(nodeSecretsEnv))
	return creds
}

// nodeCredential is the credential that belongs to nodeID: its own
// NOVACRON_NODE_SECRETS entry when the operator configured one, else the
// fabric-wide NOVACRON_MIGRATION_SECRET. ok=false means no credential is
// configured for that node at all, and every caller must then fail closed
// rather than send or accept an unauthenticated request. nodeID "" (a call
// site with no known peer identity) resolves to the fabric-wide secret only.
func nodeCredential(nodeID string) (secret string, ok bool) {
	if s, found := nodeSecrets()[strings.TrimSpace(nodeID)]; found {
		return s, true
	}
	if s := os.Getenv("NOVACRON_MIGRATION_SECRET"); s != "" {
		return s, true
	}
	return "", false
}

// clusterSecretOK reports whether the request presents the fabric-wide
// NOVACRON_MIGRATION_SECRET. Fail closed (no configured secret => false) and
// constant time, so a mismatch cannot be timed byte by byte.
func clusterSecretOK(r *http.Request) bool {
	secret := os.Getenv("NOVACRON_MIGRATION_SECRET")
	if secret == "" {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(secret), []byte(r.Header.Get("X-Migration-Secret"))) == 1
}

// internalAuthOK is the fail-closed gate for every inbound node-to-node RPC
// (join/leave, capacity, probe, dispatch, /internal/migrate/*). Accepted
// credentials, both in constant time:
//
//   - THIS node's own credential (NOVACRON_NODE_SECRETS[selfNodeID()]): a peer
//     presents the credential of the node it is talking to, because the header
//     carries no caller identity of its own, and
//   - the fabric-wide NOVACRON_MIGRATION_SECRET, kept so nodes that have no
//     per-node entry yet keep working during rollout.
//
// Neither configured => false. With NOVACRON_NODE_SECRETS unset this is
// exactly the previous single-secret behaviour.
func internalAuthOK(r *http.Request) bool {
	provided := r.Header.Get("X-Migration-Secret")
	if provided == "" {
		return false
	}
	if cred, ok := nodeCredential(selfNodeID()); ok {
		if subtle.ConstantTimeCompare([]byte(cred), []byte(provided)) == 1 {
			return true
		}
	}
	return clusterSecretOK(r)
}

// logNodeSecretConfig records which node ids have a per-node credential and
// whether this node has one of its own — node ids only, NEVER a secret value.
// A node whose own id is missing from NOVACRON_NODE_SECRETS keeps signing with
// the cluster-wide secret and is then rejected by every peer that lists it:
// that asymmetry is the first thing to check when a fabric stops converging,
// so it is called out explicitly.
func logNodeSecretConfig() {
	creds, problems := parseNodeSecrets(os.Getenv(nodeSecretsEnv))
	for _, p := range problems {
		log.Printf("%s: %s", nodeSecretsEnv, p)
	}
	if len(creds) == 0 {
		return
	}
	ids := make([]string, 0, len(creds))
	for id := range creds {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	log.Printf("%s: per-node credentials configured for: %s", nodeSecretsEnv, strings.Join(ids, ","))
	if _, ok := creds[selfNodeID()]; !ok {
		log.Printf("%s: this node (%s) has no per-node entry — it still authenticates with NOVACRON_MIGRATION_SECRET", nodeSecretsEnv, selfNodeID())
	}
}

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

// verifyJoin authenticates a join request: the HMAC over the exact payload
// fields must be keyed by the credential that belongs to the CLAIMED node id
// (nodeCredential) — a node id listed in NOVACRON_NODE_SECRETS is checked
// against its own entry ONLY, so the cluster-wide secret does not authenticate
// it even when that secret is still configured — plus a constant-time compare
// and a timestamp inside the freshness window. No configured credential for
// the claimed node fails closed (no credential, no joins).
func verifyJoin(r *http.Request, req joinRequest) bool {
	got := r.Header.Get("X-Join-Signature")
	if got == "" {
		return false
	}
	secret, ok := nodeCredential(req.NodeID)
	if !ok {
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

// upsertClusterPeer persists (or refreshes) peer membership and any successful
// directional throughput samples. A missing sample never erases the last one.
func upsertClusterPeer(ctx context.Context, db *sql.DB, nodeID, addr string, rttMS float64, inbound, outbound *linkMeasurement) error {
	link := map[string]interface{}{"rtt_ms": rttMS}
	if inbound != nil {
		link["inbound"] = inbound
	}
	if outbound != nil {
		link["outbound"] = outbound
	}
	payload, err := json.Marshal(link)
	if err != nil {
		return err
	}
	_, err = db.ExecContext(ctx, `
		INSERT INTO cluster_peers (node_id, addr, last_heartbeat, last_rtt_ms, link, updated_at)
		VALUES ($1, $2, NOW(), $3, $4, NOW())
		ON CONFLICT (node_id) DO UPDATE SET
			addr = EXCLUDED.addr,
			last_heartbeat = EXCLUDED.last_heartbeat,
			last_rtt_ms = EXCLUDED.last_rtt_ms,
			link = CASE
				WHEN EXCLUDED.link ? 'inbound' OR EXCLUDED.link ? 'outbound'
					THEN COALESCE(cluster_peers.link, '{}'::jsonb) || EXCLUDED.link
				ELSE COALESCE(cluster_peers.link, '{}'::jsonb)
			END,
			updated_at = NOW()
	`, nodeID, addr, rttMS, payload)
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
	logNodeSecretConfig()

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

		// Verify the joiner is actually reachable and answering with ITS OWN
		// credential BEFORE touching the peer map — a bogus addr must never
		// poison cluster dispatch. The callback therefore presents the joiner's
		// credential (the same one its signature was verified against), so a
		// per-node fabric never falls back to the shared secret mid-handshake.
		// The measured round trip doubles as the initial link-profile RTT.
		joinCred, haveCred := nodeCredential(req.NodeID)
		if !haveCred {
			// Unreachable in practice: verifyJoin only passes with a credential.
			// Kept so a future reorder cannot send an unauthenticated callback.
			writeJSONError(w, http.StatusForbidden, "no credential configured for this node")
			return
		}
		start := time.Now()
		cap, err := fetchPeerCapacityWithSecret(req.Addr, joinCred)
		if err != nil {
			writeJSONError(w, http.StatusForbidden, fmt.Sprintf("joiner not reachable at %s: %v", req.Addr, err))
			return
		}
		rttMS := float64(time.Since(start).Microseconds()) / 1000.0

		vmManager.RegisterMigrationPeer(req.NodeID, req.Addr)
		if db != nil {
			if err := upsertClusterPeer(r.Context(), db, req.NodeID, req.Addr, rttMS, nil, nil); err != nil {
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

	// GET /api/cluster/links — directional throughput and measurement ages.
	apiRouter.HandleFunc("/cluster/links", func(w http.ResponseWriter, r *http.Request) {
		nodes := nodeProfiles(vmManager, storagePath, db)
		links := make([]map[string]interface{}, 0, len(nodes))
		for _, n := range nodes {
			entry := map[string]interface{}{"node_id": n.NodeID, "addr": n.Addr, "reachable": n.Reachable}
			if n.Link != nil {
				entry["rtt_ms"] = n.Link.RTTMS
				entry["rtt_stale"] = n.Link.RTTStale
				entry["last_heartbeat"] = n.Link.LastHeartbeat
				entry["stale"] = n.Link.Stale
				entry["inbound"] = n.Link.Inbound
				entry["outbound"] = n.Link.Outbound
			}
			links = append(links, entry)
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{"links": links})
	}).Methods(http.MethodGet)
}

// nodeProfile extends NodeCapacity with the persisted link profile.
type nodeProfile struct {
	NodeCapacity
	Link *linkProfile `json:"link,omitempty"`
}

// linkMeasurement is one successful directional probe, persisted in the
// cluster_peers.link JSONB column.
type linkMeasurement struct {
	ThroughputBps float64 `json:"throughput_bps"`
	ProbeBytes    int     `json:"probe_bytes"`
	MeasuredAt    string  `json:"measured_at"`
}

// linkDirection adds freshness computed from the measurement time, not from a
// later heartbeat.
type linkDirection struct {
	ThroughputBps float64 `json:"throughput_bps"`
	ProbeBytes    int     `json:"probe_bytes"`
	MeasuredAt    string  `json:"measured_at"`
	Stale         bool    `json:"stale"`
}

// linkProfile reports both directions independently. Stale summarizes whether
// either direction is missing or stale; RTT has its own heartbeat freshness.
type linkProfile struct {
	RTTMS         float64        `json:"rtt_ms"`
	Inbound       *linkDirection `json:"inbound,omitempty"`
	Outbound      *linkDirection `json:"outbound,omitempty"`
	LastHeartbeat string         `json:"last_heartbeat"`
	RTTStale      bool           `json:"rtt_stale"`
	Stale         bool           `json:"stale"`
}

// linkProfileStaleness: measured rates and heartbeat RTTs older than this are
// not trusted for placement.
const linkProfileStaleness = 5 * time.Minute

func profileLinkDirection(measurement *linkMeasurement, now time.Time) *linkDirection {
	if measurement == nil {
		return nil
	}
	direction := &linkDirection{
		ThroughputBps: measurement.ThroughputBps,
		ProbeBytes:    measurement.ProbeBytes,
		MeasuredAt:    measurement.MeasuredAt,
		Stale:         measurement.ThroughputBps <= 0 || measurement.ProbeBytes <= 0,
	}
	measuredAt, err := time.Parse(time.RFC3339Nano, measurement.MeasuredAt)
	if err != nil {
		direction.Stale = true
		return direction
	}
	age := now.Sub(measuredAt)
	if age < 0 || age > linkProfileStaleness {
		direction.Stale = true
	}
	return direction
}

// nodeProfiles assembles live capacities plus persisted link profiles.
// A DB error degrades to capacities without profiles (the fabric keeps
// working; only profile freshness is lost).
func nodeProfiles(vmManager *core_vm.VMManager, storagePath string, db *sql.DB) []nodeProfile {
	caps := allNodeCapacities(vmManager, storagePath)
	profiles := map[string]*linkProfile{}
	if db != nil {
		if rows, err := db.Query(`SELECT node_id, last_rtt_ms, last_heartbeat, COALESCE(link, '{}'::jsonb) FROM cluster_peers`); err == nil {
			now := time.Now()
			for rows.Next() {
				var id string
				var rtt sql.NullFloat64
				var beat sql.NullTime
				var raw []byte
				if err := rows.Scan(&id, &rtt, &beat, &raw); err != nil {
					continue
				}
				p := &linkProfile{RTTStale: true, Stale: true}
				if rtt.Valid {
					p.RTTMS = rtt.Float64
				}
				var stored struct {
					Inbound  *linkMeasurement `json:"inbound"`
					Outbound *linkMeasurement `json:"outbound"`
				}
				if err := json.Unmarshal(raw, &stored); err == nil {
					p.Inbound = profileLinkDirection(stored.Inbound, now)
					p.Outbound = profileLinkDirection(stored.Outbound, now)
				}
				if beat.Valid {
					p.LastHeartbeat = beat.Time.UTC().Format(time.RFC3339Nano)
					age := now.Sub(beat.Time)
					p.RTTStale = !rtt.Valid || age < 0 || age > linkProfileStaleness
				}
				p.Stale = p.Inbound == nil || p.Inbound.Stale || p.Outbound == nil || p.Outbound.Stale
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
func clusterHeartbeatLoop(ctx context.Context, db *sql.DB, vmManager *core_vm.VMManager, liveness *peerLiveness) {
	t := time.NewTicker(heartbeatInterval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			beatOnce(ctx, db, vmManager, liveness)
		}
	}
}

func beatOnce(ctx context.Context, db *sql.DB, vmManager *core_vm.VMManager, liveness *peerLiveness) {
	if vmManager == nil || db == nil {
		return
	}
	peers := vmManager.MigrationPeers()
	liveness.forgetExcept(peers)
	for id, addr := range peers {
		secret, ok := nodeCredential(id)
		if !ok {
			continue
		}
		start := time.Now()
		if _, err := fetchPeerCapacityWithSecret(addr, secret); err != nil {
			liveness.observe(ctx, id, err)
			continue
		}
		rttMS := float64(time.Since(start).Microseconds()) / 1000.0
		liveness.observe(ctx, id, nil)

		probeBytes := probeBytesFromEnv()
		var inbound, outbound *linkMeasurement
		if probeBytes > 0 {
			if measurement, err := measurePeerInbound(addr, secret, probeBytes); err == nil {
				inbound = measurement
			} else {
				log.Printf("inbound throughput probe %s failed: %v", id, err)
			}
			if measurement, err := measurePeerOutbound(addr, secret, probeBytes); err == nil {
				outbound = measurement
			} else {
				log.Printf("outbound throughput probe %s failed: %v", id, err)
			}
		}

		bctx, cancel := context.WithTimeout(ctx, 10*time.Second)
		if err := upsertClusterPeer(bctx, db, id, addr, rttMS, inbound, outbound); err != nil {
			log.Printf("heartbeat upsert failed for %s: %v", id, err)
		}
		cancel()
	}
}

// peerLiveness turns heartbeat probe outcomes into orchestration node
// lifecycle events on the shared bus: after peerFailureThreshold consecutive
// failed probes it publishes events.EventTypeNodeFailure once, and on the
// next successful probe of that peer events.EventTypeNodeRecovered. Only the
// heartbeat goroutine touches it.
type peerLiveness struct {
	bus    events.EventBus
	misses map[string]int  // consecutive failed probes per peer
	failed map[string]bool // peers a NodeFailure was published for
}

func newPeerLiveness(bus events.EventBus) *peerLiveness {
	return &peerLiveness{bus: bus, misses: make(map[string]int), failed: make(map[string]bool)}
}

func (p *peerLiveness) observe(ctx context.Context, nodeID string, probeErr error) {
	if probeErr == nil {
		delete(p.misses, nodeID)
		if p.failed[nodeID] {
			delete(p.failed, nodeID)
			p.publish(ctx, events.EventTypeNodeRecovered, events.PriorityNormal, nodeID, map[string]interface{}{"node_id": nodeID})
		}
		return
	}
	p.misses[nodeID]++
	if p.failed[nodeID] || p.misses[nodeID] < peerFailureThreshold {
		return
	}
	p.failed[nodeID] = true
	p.publish(ctx, events.EventTypeNodeFailure, events.PriorityCritical, nodeID, map[string]interface{}{
		"node_id":              nodeID,
		"reason":               probeErr.Error(),
		"consecutive_failures": p.misses[nodeID],
	})
}

// forgetExcept drops state for peers no longer in the peer map.
func (p *peerLiveness) forgetExcept(peers map[string]string) {
	for id := range p.misses {
		if _, ok := peers[id]; !ok {
			delete(p.misses, id)
		}
	}
	for id := range p.failed {
		if _, ok := peers[id]; !ok {
			delete(p.failed, id)
		}
	}
}

func (p *peerLiveness) publish(ctx context.Context, t events.EventType, prio events.EventPriority, nodeID string, data map[string]interface{}) {
	if err := p.bus.Publish(ctx, &events.OrchestrationEvent{
		Type:      t,
		Source:    "cluster-heartbeat",
		Target:    nodeID,
		Timestamp: time.Now(),
		Data:      data,
		Priority:  prio,
	}); err != nil {
		log.Printf("heartbeat: publish %s for %s failed: %v", t, nodeID, err)
	}
}

// probeBytesFromEnv reads NOVACRON_PROBE_BYTES (bytes per direction; 0
// disables). The default and hard cap bound heartbeat traffic per peer.
func probeBytesFromEnv() int {
	raw := strings.TrimSpace(os.Getenv("NOVACRON_PROBE_BYTES"))
	if raw == "" {
		return 1 << 20
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 0 {
		return 1 << 20
	}
	if n > maxProbeBytes {
		return maxProbeBytes
	}
	return n
}

// measurePeerInbound downloads n bytes and times the body transfer after the
// response headers arrive, excluding the request/response RTT.
func measurePeerInbound(addr, secret string, n int) (*linkMeasurement, error) {
	if n <= 0 {
		return nil, fmt.Errorf("probe disabled")
	}
	if n > maxProbeBytes {
		n = maxProbeBytes
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		fmt.Sprintf("http://%s/internal/cluster/probe?bytes=%d", addr, n), nil)
	if err != nil {
		return nil, err
	}
	if secret != "" {
		req.Header.Set("X-Migration-Secret", secret)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("inbound probe %s: %s", addr, resp.Status)
	}
	started := time.Now()
	got, err := io.Copy(io.Discard, resp.Body)
	elapsed := time.Since(started)
	if err != nil {
		return nil, err
	}
	if got != int64(n) || elapsed <= 0 {
		return nil, fmt.Errorf("inbound probe %s: received %d of %d bytes in %s", addr, got, n, elapsed)
	}
	return &linkMeasurement{
		ThroughputBps: float64(got) * 8 / elapsed.Seconds(),
		ProbeBytes:    int(got),
		MeasuredAt:    time.Now().UTC().Format(time.RFC3339Nano),
	}, nil
}

// measurePeerOutbound uploads n bytes and uses the peer's receive duration,
// excluding network RTT and response serialization.
func measurePeerOutbound(addr, secret string, n int) (*linkMeasurement, error) {
	if n <= 0 {
		return nil, fmt.Errorf("probe disabled")
	}
	if n > maxProbeBytes {
		n = maxProbeBytes
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		fmt.Sprintf("http://%s/internal/cluster/probe", addr), bytes.NewReader(probePayload[:n]))
	if err != nil {
		return nil, err
	}
	if secret != "" {
		req.Header.Set("X-Migration-Secret", secret)
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("outbound probe %s: %s", addr, resp.Status)
	}
	var receipt probeReceipt
	if err := json.NewDecoder(io.LimitReader(resp.Body, 64<<10)).Decode(&receipt); err != nil {
		return nil, err
	}
	if receipt.BytesReceived != int64(n) || receipt.ElapsedNS <= 0 {
		return nil, fmt.Errorf("outbound probe %s: peer received %d of %d bytes in %d ns", addr, receipt.BytesReceived, n, receipt.ElapsedNS)
	}
	return &linkMeasurement{
		ThroughputBps: float64(receipt.BytesReceived) * 8 / (float64(receipt.ElapsedNS) / float64(time.Second)),
		ProbeBytes:    int(receipt.BytesReceived),
		MeasuredAt:    time.Now().UTC().Format(time.RFC3339Nano),
	}, nil
}

// joinOutcome is what a successful join to one seed gave back.
type joinOutcome struct {
	seedNodeID string
	seedAddr   string
}

// sendJoin POSTs a signed join to one known node and registers every peer
// it reports back (including that seed node itself). secret must be THIS
// node's own credential (nodeCredential(selfID)) — the seed verifies the
// signature against the credential it holds for the claimed node id. Returns
// the seed's identity on success.
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
	// Join under THIS node's own credential: its NOVACRON_NODE_SECRETS entry
	// when configured, else the cluster-wide secret. Neither configured means
	// we cannot authenticate at all — stay a single-node fabric rather than
	// emit a join every seed will reject (fail closed).
	secret, ok := nodeCredential(selfID)
	if !ok {
		log.Printf("NOVACRON_JOIN_PEERS set but neither %s[%s] nor NOVACRON_MIGRATION_SECRET is configured — skipping join (fail closed)", nodeSecretsEnv, selfID)
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
