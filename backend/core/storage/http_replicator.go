package storage

import (
	"bytes"
	"context"
	"crypto/subtle"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// HTTPReplicator is a production Replicator whose replicas travel over the
// network to a remote storage-agent per node, rather than to an in-process
// path like LoopbackReplicator. Each nodeID is mapped (via SetNode /
// SetNodeFromInfo) to the base URL of a storage-agent running the handler
// returned by NewReplicaHandler; WriteReplica/ReadReplica/DeleteReplica issue
// HTTP requests to that agent, which persists each shard on its own local disk
// root. Because every agent owns a distinct on-disk root on a distinct host,
// replicas placed on different nodes are genuinely independent — losing one
// agent's backend cannot affect another's, and reads fail (never fabricate)
// when an agent is unreachable.
//
// The transport intentionally mirrors the existing node-to-node migration RPC
// (POST /internal/migrate/incoming): plain HTTP gated by a shared secret in a
// header, compared in constant time and failing closed. gRPC is not used
// anywhere in this backend's service code, so introducing it here would add a
// dependency and a pattern nobody else follows.
//
// ponytail: static shared-secret header over HTTP, same trust model as the
// migration RPC. Swap for mTLS if peers aren't mutually trusted.
type HTTPReplicator struct {
	secret string
	client *http.Client

	mu    sync.RWMutex
	nodes map[string]string // nodeID -> base URL, e.g. "http://10.0.0.1:9000"
}

var _ Replicator = (*HTTPReplicator)(nil)

// NewHTTPReplicator creates a network replicator that authenticates to remote
// storage-agents with the given shared secret. The secret must match the one
// the agents were started with (see NewReplicaHandler); an empty secret can
// never authenticate because agents fail closed.
func NewHTTPReplicator(secret string) *HTTPReplicator {
	return &HTTPReplicator{
		secret: secret,
		client: &http.Client{Timeout: 30 * time.Second},
		nodes:  make(map[string]string),
	}
}

// SetNode registers (or updates) the base URL used to reach nodeID's storage
// agent. baseURL is like "http://host:port"; a trailing slash is tolerated.
func (r *HTTPReplicator) SetNode(nodeID, baseURL string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.nodes[nodeID] = strings.TrimRight(baseURL, "/")
}

// SetNodeFromInfo registers nodeID's agent from a NodeInfo, dialing
// http://Address:Port. This is the intended production bridge from the
// service's node table to the transport.
func (r *HTTPReplicator) SetNodeFromInfo(n NodeInfo) {
	r.SetNode(n.ID, fmt.Sprintf("http://%s:%d", n.Address, n.Port))
}

func (r *HTTPReplicator) baseURL(nodeID string) (string, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	u, ok := r.nodes[nodeID]
	return u, ok
}

func (r *HTTPReplicator) shardURL(base, volumeID string, shardIndex int) string {
	return fmt.Sprintf("%s/internal/replica/%s/%d", base, url.PathEscape(volumeID), shardIndex)
}

// WriteReplica implements Replicator.
func (r *HTTPReplicator) WriteReplica(ctx context.Context, nodeID, volumeID string, shardIndex int, data []byte) error {
	base, ok := r.baseURL(nodeID)
	if !ok {
		return fmt.Errorf("httpreplicator: unknown node %s", nodeID)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, r.shardURL(base, volumeID, shardIndex), bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("X-Replica-Secret", r.secret)
	req.Header.Set("Content-Type", "application/octet-stream")
	resp, err := r.client.Do(req)
	if err != nil {
		return fmt.Errorf("httpreplicator: node %s write: %w", nodeID, err)
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	if resp.StatusCode != http.StatusNoContent && resp.StatusCode != http.StatusOK {
		return fmt.Errorf("httpreplicator: node %s write: unexpected status %d", nodeID, resp.StatusCode)
	}
	return nil
}

// ReadReplica implements Replicator. It returns an error (never fabricated
// data) if the node is unreachable or holds no data for the shard.
func (r *HTTPReplicator) ReadReplica(ctx context.Context, nodeID, volumeID string, shardIndex int) ([]byte, error) {
	base, ok := r.baseURL(nodeID)
	if !ok {
		return nil, fmt.Errorf("httpreplicator: unknown node %s", nodeID)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, r.shardURL(base, volumeID, shardIndex), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("X-Replica-Secret", r.secret)
	resp, err := r.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("httpreplicator: node %s read: %w", nodeID, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, resp.Body)
		return nil, fmt.Errorf("httpreplicator: node %s read: unexpected status %d", nodeID, resp.StatusCode)
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("httpreplicator: node %s read body: %w", nodeID, err)
	}
	return data, nil
}

// DeleteReplica implements Replicator. Deleting a replica that does not exist
// is not an error.
func (r *HTTPReplicator) DeleteReplica(ctx context.Context, nodeID, volumeID string, shardIndex int) error {
	base, ok := r.baseURL(nodeID)
	if !ok {
		return fmt.Errorf("httpreplicator: unknown node %s", nodeID)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, r.shardURL(base, volumeID, shardIndex), nil)
	if err != nil {
		return err
	}
	req.Header.Set("X-Replica-Secret", r.secret)
	resp, err := r.client.Do(req)
	if err != nil {
		return fmt.Errorf("httpreplicator: node %s delete: %w", nodeID, err)
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	switch resp.StatusCode {
	case http.StatusNoContent, http.StatusOK, http.StatusNotFound:
		return nil
	default:
		return fmt.Errorf("httpreplicator: node %s delete: unexpected status %d", nodeID, resp.StatusCode)
	}
}

// NodeAvailable implements Replicator by probing the agent's health endpoint
// with a short timeout. An unregistered or unreachable node reports false.
func (r *HTTPReplicator) NodeAvailable(nodeID string) bool {
	base, ok := r.baseURL(nodeID)
	if !ok {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/internal/replica/health", nil)
	if err != nil {
		return false
	}
	req.Header.Set("X-Replica-Secret", r.secret)
	resp, err := r.client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	return resp.StatusCode == http.StatusOK
}

// --- Storage-agent (server) side ------------------------------------------

// NewReplicaHandler returns the HTTP handler a remote storage-agent serves so
// an HTTPReplicator can store/fetch shard replicas on its local rootDir. Every
// request must carry an X-Replica-Secret header matching secret (constant-time
// compare); like the migration RPC it FAILS CLOSED — an empty server secret
// rejects all requests, so a misconfigured agent never silently accepts
// attacker-controlled shard writes to disk.
//
// Routes (all under /internal/replica):
//
//	GET    /internal/replica/health                    -> 200 if authorized
//	PUT    /internal/replica/{volumeID}/{shardIndex}   -> store body, 204
//	GET    /internal/replica/{volumeID}/{shardIndex}   -> 200+body / 404
//	DELETE /internal/replica/{volumeID}/{shardIndex}   -> 204 (404 tolerated)
//
// ponytail: this handler is the reusable transport; a deployable per-node
// storage-agent is a thin main() wrapping http.Server{Handler: this} — added
// when the distributed GA track actually ships multi-host agents, not before.
func NewReplicaHandler(rootDir, secret string) http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /internal/replica/health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	mux.HandleFunc("PUT /internal/replica/{volumeID}/{shardIndex}", func(w http.ResponseWriter, r *http.Request) {
		path, ok := shardPathFromReq(w, rootDir, r)
		if !ok {
			return
		}
		data, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read body", http.StatusBadRequest)
			return
		}
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			http.Error(w, "mkdir", http.StatusInternalServerError)
			return
		}
		if err := os.WriteFile(path, data, 0o644); err != nil {
			http.Error(w, "write", http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	})

	mux.HandleFunc("GET /internal/replica/{volumeID}/{shardIndex}", func(w http.ResponseWriter, r *http.Request) {
		path, ok := shardPathFromReq(w, rootDir, r)
		if !ok {
			return
		}
		data, err := os.ReadFile(path)
		if os.IsNotExist(err) {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		if err != nil {
			http.Error(w, "read", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		_, _ = w.Write(data)
	})

	mux.HandleFunc("DELETE /internal/replica/{volumeID}/{shardIndex}", func(w http.ResponseWriter, r *http.Request) {
		path, ok := shardPathFromReq(w, rootDir, r)
		if !ok {
			return
		}
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			http.Error(w, "delete", http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	})

	return &replicaSecretGate{secret: secret, next: mux}
}

type replicaSecretGate struct {
	secret string
	next   http.Handler
}

func (g *replicaSecretGate) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !replicaAuthOK(g.secret, r) {
		http.Error(w, "forbidden", http.StatusForbidden)
		return
	}
	g.next.ServeHTTP(w, r)
}

// replicaAuthOK mirrors migrationAuthOK: fail closed on an empty server secret,
// constant-time compare otherwise so a mismatch can't be timed byte by byte.
func replicaAuthOK(secret string, r *http.Request) bool {
	if secret == "" {
		return false
	}
	provided := r.Header.Get("X-Replica-Secret")
	return subtle.ConstantTimeCompare([]byte(secret), []byte(provided)) == 1
}

// shardPathFromReq validates the {volumeID}/{shardIndex} path vars and returns
// the on-disk path, guaranteed to stay within rootDir. On any validation
// failure it writes a 400 and returns ok=false.
func shardPathFromReq(w http.ResponseWriter, rootDir string, r *http.Request) (string, bool) {
	volumeID := r.PathValue("volumeID")
	if !safeSegment(volumeID) {
		http.Error(w, "invalid volume id", http.StatusBadRequest)
		return "", false
	}
	idx, err := strconv.Atoi(r.PathValue("shardIndex"))
	if err != nil || idx < 0 {
		http.Error(w, "invalid shard index", http.StatusBadRequest)
		return "", false
	}
	path := filepath.Join(rootDir, volumeID, fmt.Sprintf("shard_%d", idx))
	// Defense in depth: the join must not escape rootDir.
	if !strings.HasPrefix(filepath.Clean(path), filepath.Clean(rootDir)+string(os.PathSeparator)) {
		http.Error(w, "invalid path", http.StatusBadRequest)
		return "", false
	}
	return path, true
}

// safeSegment rejects path segments that could traverse outside a directory.
func safeSegment(s string) bool {
	return s != "" && s != "." && s != ".." && !strings.ContainsAny(s, `/\`)
}
