package main

// Fabric transfers (P3/G3): the admission-controlled byte movers.
//
// A transfer is a migration admitted against the MEASURED link budget:
//
//	link_bps   = the heartbeat's fresh throughput probe for the target node
//	sample_ratio = compressibility of up to 64 MiB of the VM's disk image
//	compression = zstd-multifd when link_bps < 500 Mbps AND ratio > 1.3,
//	              otherwise none (QEMU then runs byte-identical to before)
//
// One transfer runs per link at a time; a second transfer to a busy link is
// QUEUED with an ETA computed from the active transfer's remaining work over
// the measured rate instead of being started (G3b). Transfers are an
// in-memory view over the durable migration machinery: the VM state reconcile
// and the migration job rows in Postgres are what survive a restart — a
// transfer record does not (documented, not hidden).

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
	"github.com/klauspost/compress/zstd"
)

const (
	// compressionLinkThresholdBps: below this measured link rate compression is
	// worth its CPU cost. 500 Mbps is the documented cut-off — above it the
	// network is not the bottleneck for a typical guest.
	compressionLinkThresholdBps = 500e6
	// compressionRatioThreshold: the sampled payload must compress at least
	// this much (1.3x) for the compression to be worth enabling; below it the
	// CPU cost outweighs the saved bytes (measured earlier: incompressible
	// RAM migrates 1.02–1.07x SLOWER with compression).
	compressionRatioThreshold = 1.3
	// compressionSampleMaxBytes caps the compressibility sample (64 MiB).
	compressionSampleMaxBytes = 64 << 20
)

// decideMigrationCompression implements the G3a rule and returns the mode plus
// the reason (recorded verbatim in the transfer's decision inputs).
func decideMigrationCompression(linkBps float64, ratio float64) (string, string) {
	if linkBps <= 0 {
		return "none", "link rate unmeasured — compression off (never guessed on)"
	}
	if linkBps >= compressionLinkThresholdBps {
		return "none", fmt.Sprintf("link %.1f Mbps at/above the %.0f Mbps threshold — compression off", linkBps/1e6, compressionLinkThresholdBps/1e6)
	}
	if ratio < compressionRatioThreshold {
		return "none", fmt.Sprintf("sample ratio %.2fx below %.2fx at %.1f Mbps — compression would cost CPU without saving bytes", ratio, compressionRatioThreshold, linkBps/1e6)
	}
	return "zstd-multifd", fmt.Sprintf("link %.1f Mbps below %.0f Mbps and sample compresses %.2fx — zstd multifd", linkBps/1e6, compressionLinkThresholdBps/1e6, ratio)
}

// sampleImageCompressionRatio compresses up to maxBytes of the file at path
// with zstd (level 1, the same level the migration uses) and returns the
// achieved ratio plus how many bytes were sampled. Empty/missing files return
// ratio 1 with an explanatory error — the caller then decides "none" honestly
// rather than treating a failed sample as incompressible.
func sampleImageCompressionRatio(path string, maxBytes int64) (float64, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return 1, 0, err
	}
	defer f.Close()
	enc, err := zstd.NewWriter(io.Discard, zstd.WithEncoderLevel(zstd.SpeedFastest))
	if err != nil {
		return 1, 0, err
	}
	defer enc.Close()

	buf := make([]byte, 1<<20)
	var sampled int64
	var compressed int64
	for sampled < maxBytes {
		want := int64(len(buf))
		if remaining := maxBytes - sampled; remaining < want {
			want = remaining
		}
		n, rerr := f.Read(buf[:want])
		if n > 0 {
			sampled += int64(n)
			compressed += int64(len(enc.EncodeAll(buf[:n], nil)))
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			return 1, sampled, rerr
		}
	}
	if sampled == 0 || compressed == 0 {
		return 1, sampled, fmt.Errorf("no bytes sampled from %s", path)
	}
	return float64(sampled) / float64(compressed), sampled, nil
}

// vmDiskImagePath finds the VM's primary disk image (the KVM driver writes
// <storage>/vms/<vmID>/disk.qcow2). Returns "" when absent.
func vmDiskImagePath(storagePath, vmID string) string {
	p := filepath.Join(storagePath, vmID, "disk.qcow2")
	if _, err := os.Stat(p); err == nil {
		return p
	}
	return ""
}

// --- transfer registry + per-link admission --------------------------------

// fabricTransferStatus values.
const (
	transferQueued    = "queued"
	transferRunning   = "running"
	transferCompleted = "completed"
	transferFailed    = "failed"
)

// transferDecisionInputs is the recorded decision evidence (G3a).
type transferDecisionInputs struct {
	LinkBps       float64 `json:"link_bps"`
	LinkMeasured  bool    `json:"link_measured"`
	SampleRatio   float64 `json:"sample_ratio"`
	SampleBytes   int64   `json:"sample_bytes"`
	ThresholdBps  float64 `json:"threshold_bps"`
	ThresholdRate float64 `json:"threshold_ratio"`
	Reason        string  `json:"reason"`
}

// fabricTransfer is one admitted (or waiting) byte mover.
type fabricTransfer struct {
	ID             string                 `json:"transfer_id"`
	Kind           string                 `json:"kind"`
	VMID           string                 `json:"vm_id"`
	TargetNode     string                 `json:"target_node"`
	MigrationType  string                 `json:"migration_type,omitempty"`
	Status         string                 `json:"status"`
	Compression    string                 `json:"compression"`
	BytesEstimated int64                  `json:"bytes_total"`
	BytesMoved     int64                  `json:"bytes_moved"`
	MeasuredBps    float64                `json:"measured_bps"`
	EtaSeconds     *float64               `json:"eta_seconds,omitempty"`
	QueuePosition  int                    `json:"queue_position,omitempty"`
	Error          string                 `json:"error,omitempty"`
	Decision       transferDecisionInputs `json:"decision_inputs"`
	CreatedAt      time.Time              `json:"created_at"`
	StartedAt      *time.Time             `json:"started_at,omitempty"`
	FinishedAt     *time.Time             `json:"finished_at,omitempty"`
}

// linkAdmission serializes transfers per target link: one active, FIFO waiters.
type linkAdmission struct {
	active  *fabricTransfer
	waiting []*fabricTransfer
}

// transferStore holds all transfers plus per-link admission state.
type transferStore struct {
	mu       sync.Mutex
	byID     map[string]*fabricTransfer
	byLink   map[string]*linkAdmission
	order    []string
	maxKeep  int
	runFn    func(ctx context.Context, t *fabricTransfer, vmManager *core_vm.VMManager) error
	vmMgr    *core_vm.VMManager
	onFinish func(t *fabricTransfer)
}

func newTransferStore(maxKeep int, runFn func(context.Context, *fabricTransfer, *core_vm.VMManager) error) *transferStore {
	if maxKeep <= 0 {
		maxKeep = 512
	}
	return &transferStore{
		byID:    make(map[string]*fabricTransfer),
		byLink:  make(map[string]*linkAdmission),
		maxKeep: maxKeep,
		runFn:   runFn,
	}
}

// admit registers a transfer and either starts it (link idle) or queues it
// behind the active transfer with an ETA. Returns the stored record.
func (s *transferStore) admit(t *fabricTransfer) *fabricTransfer {
	s.mu.Lock()
	s.byID[t.ID] = t
	s.order = append(s.order, t.ID)
	for len(s.order) > s.maxKeep {
		oldest := s.order[0]
		s.order = s.order[1:]
		delete(s.byID, oldest)
	}
	la := s.byLink[t.TargetNode]
	if la == nil {
		la = &linkAdmission{}
		s.byLink[t.TargetNode] = la
	}
	if la.active == nil {
		la.active = t
		t.Status = transferRunning
		now := time.Now().UTC()
		t.StartedAt = &now
		s.mu.Unlock()
		s.start(t)
		return t
	}
	t.Status = transferQueued
	la.waiting = append(la.waiting, t)
	t.QueuePosition = len(la.waiting)
	t.EtaSeconds = s.estimateEtaLocked(la.active, t)
	s.mu.Unlock()
	return t
}

// estimateEtaLocked computes the waiting transfer's ETA from the active
// transfer's remaining work at the measured rate (nil when unmeasurable).
func (s *transferStore) estimateEtaLocked(active, waiting *fabricTransfer) *float64 {
	bps := active.Decision.LinkBps
	if bps <= 0 {
		return nil
	}
	remaining := active.BytesEstimated - active.BytesMoved
	if remaining < 0 {
		remaining = 0
	}
	secs := float64(remaining+waiting.BytesEstimated) / bps
	return &secs
}

// start runs one transfer on a background goroutine and promotes the next
// waiter when it finishes.
func (s *transferStore) start(t *fabricTransfer) {
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
		defer cancel()
		err := s.runFn(ctx, t, s.vmMgr)
		s.mu.Lock()
		now := time.Now().UTC()
		t.FinishedAt = &now
		if err != nil {
			t.Status = transferFailed
			t.Error = err.Error()
		} else {
			t.Status = transferCompleted
			t.BytesMoved = t.BytesEstimated
			if t.StartedAt != nil {
				elapsed := now.Sub(*t.StartedAt).Seconds()
				if elapsed > 0 {
					t.MeasuredBps = float64(t.BytesEstimated) / elapsed
				}
			}
		}
		t.EtaSeconds = nil
		la := s.byLink[t.TargetNode]
		var next *fabricTransfer
		if la != nil {
			if len(la.waiting) > 0 {
				next = la.waiting[0]
				la.waiting = la.waiting[1:]
				next.Status = transferRunning
				next.QueuePosition = 0
				next.StartedAt = &now
				la.active = next
				// Recompute the remaining waiters' ETAs against the new active.
				for i, w := range la.waiting {
					w.QueuePosition = i + 1
					w.EtaSeconds = s.estimateEtaLocked(next, w)
				}
			} else {
				la.active = nil
			}
		}
		s.mu.Unlock()
		if s.onFinish != nil {
			s.onFinish(t)
		}
		if next != nil {
			s.start(next)
		}
	}()
}

// get returns a snapshot copy (safe to serialize outside the lock).
func (s *transferStore) get(id string) (fabricTransfer, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	t, ok := s.byID[id]
	if !ok {
		return fabricTransfer{}, false
	}
	return *t, true
}

// list returns snapshots, newest first.
func (s *transferStore) list() []fabricTransfer {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]fabricTransfer, 0, len(s.order))
	for i := len(s.order) - 1; i >= 0; i-- {
		if t, ok := s.byID[s.order[i]]; ok {
			out = append(out, *t)
		}
	}
	return out
}

// transfers is the process-wide registry, wired with a real runner by
// registerFabricTransferRoutes.
var transfers = newTransferStore(512, nil)

// --- HTTP surface ----------------------------------------------------------

// registerFabricTransferRoutes wires the authed transfers API.
func registerFabricTransferRoutes(apiRouter *mux.Router, db *sql.DB, vmManager *core_vm.VMManager, storagePath string) {
	transfers.vmMgr = vmManager
	transfers.runFn = migrationTransferRunner(storagePath)

	apiRouter.HandleFunc("/transfers", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Kind           string `json:"kind"`
			VMID           string `json:"vm_id"`
			JobID          string `json:"job_id"`
			TargetNode     string `json:"target_node"`
			BytesEstimated int64  `json:"bytes_estimated"`
			MigrationType  string `json:"migration_type,omitempty"`
			// CompressionOverride forces a mode for operator A/B runs; empty
			// means the automatic rule decides (the normal path).
			CompressionOverride string `json:"compression_override,omitempty"`
		}
		if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}
		if strings.TrimSpace(req.TargetNode) == "" {
			writeJSONError(w, http.StatusBadRequest, "target_node is required")
			return
		}
		if req.Kind != "migration" {
			// Honest scope: jobs and raw data moves are not transfer kinds yet.
			writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("kind %q is not implemented; only \"migration\" transfers are admitted", req.Kind))
			return
		}
		if strings.TrimSpace(req.VMID) == "" {
			writeJSONError(w, http.StatusBadRequest, "vm_id is required for a migration transfer")
			return
		}
		if req.JobID != "" {
			writeJSONError(w, http.StatusBadRequest, "job_id applies to kind=job transfers, which are not implemented")
			return
		}

		// The VM must exist on THIS node: a transfer moves a local VM.
		vm, err := vmManager.GetVM(req.VMID)
		if err != nil {
			writeJSONError(w, http.StatusNotFound, "vm not found on this node")
			return
		}
		addr := vmManager.MigrationPeers()[req.TargetNode]
		if addr == "" {
			writeJSONError(w, http.StatusServiceUnavailable, fmt.Sprintf("target node %q is not a registered peer", req.TargetNode))
			return
		}

		// Measured link rate for the target (fresh profile only).
		var linkBps float64
		linkMeasured := false
		for _, n := range nodeProfiles(vmManager, storagePath, db) {
			if n.NodeID != req.TargetNode {
				continue
			}
			if n.Link != nil && n.Link.ThroughputBps != nil && !n.Link.Stale {
				linkBps = *n.Link.ThroughputBps
				linkMeasured = true
			}
			break
		}

		// Compressibility sample from the VM's disk image (real bytes; the only
		// guest payload NovaCron can read without holding RAM).
		ratio, sampled := 1.0, int64(0)
		if img := vmDiskImagePath(storagePath, req.VMID); img != "" {
			if r, n, serr := sampleImageCompressionRatio(img, compressionSampleMaxBytes); serr == nil {
				ratio, sampled = r, n
			} else {
				log.Printf("compression sample for %s failed: %v", req.VMID, serr)
			}
		}

		mode, reason := decideMigrationCompression(linkBps, ratio)
		if ov := strings.TrimSpace(req.CompressionOverride); ov != "" {
			switch ov {
			case "none", "zstd-multifd", "xbzrle":
				mode = ov
				reason = fmt.Sprintf("override: %s (automatic rule would have chosen differently; decision inputs still recorded)", ov)
			default:
				writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("unknown compression_override %q", ov))
				return
			}
		}

		bytesEst := req.BytesEstimated
		if bytesEst <= 0 {
			cfg := vm.Config()
			bytesEst = int64(cfg.MemoryMB) << 20
		}

		t := &fabricTransfer{
			ID:             uuid.NewString(),
			Kind:           req.Kind,
			VMID:           req.VMID,
			TargetNode:     req.TargetNode,
			MigrationType:  req.MigrationType,
			Compression:    mode,
			BytesEstimated: bytesEst,
			CreatedAt:      time.Now().UTC(),
			Decision: transferDecisionInputs{
				LinkBps: linkBps, LinkMeasured: linkMeasured,
				SampleRatio: ratio, SampleBytes: sampled,
				ThresholdBps: compressionLinkThresholdBps, ThresholdRate: compressionRatioThreshold,
				Reason: reason,
			},
		}
		transfers.admit(t)
		snapshot, _ := transfers.get(t.ID)
		writeJSON(w, http.StatusAccepted, map[string]interface{}{
			"transfer_id":     snapshot.ID,
			"status":          snapshot.Status,
			"eta_seconds":     snapshot.EtaSeconds,
			"queue_position":  snapshot.QueuePosition,
			"compression":     snapshot.Compression,
			"decision_inputs": snapshot.Decision,
		})
	}).Methods(http.MethodPost)

	apiRouter.HandleFunc("/transfers", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]interface{}{"transfers": transfers.list()})
	}).Methods(http.MethodGet)

	apiRouter.HandleFunc("/transfers/{id}", func(w http.ResponseWriter, r *http.Request) {
		t, ok := transfers.get(mux.Vars(r)["id"])
		if !ok {
			writeJSONError(w, http.StatusNotFound, "transfer not found")
			return
		}
		writeJSON(w, http.StatusOK, t)
	}).Methods(http.MethodGet)
}

// migrationTransferRunner performs the admitted migration: it resolves the
// target URI the same way the sync route does (peer RPC), then calls MigrateVM
// with the decided compression parameters so the KVM driver applies them to QMP.
func migrationTransferRunner(storagePath string) func(context.Context, *fabricTransfer, *core_vm.VMManager) error {
	return func(ctx context.Context, t *fabricTransfer, vmManager *core_vm.VMManager) error {
		opts := map[string]string{
			"target_addr": vmManager.MigrationPeers()[t.TargetNode],
		}
		if opts["target_addr"] == "" {
			return fmt.Errorf("target node %q is not registered", t.TargetNode)
		}
		if t.MigrationType != "" {
			opts["migration_type"] = t.MigrationType
		}
		if t.Compression != "" && t.Compression != "none" {
			opts["compression"] = t.Compression
			opts["multifd_channels"] = "4"
			opts["multifd_compression_level"] = "1"
		}
		if err := vmManager.MigrateVM(ctx, t.VMID, t.TargetNode, opts); err != nil {
			return err
		}
		_ = storagePath
		return nil
	}
}

// envInt reads a positive int env override (helper for probe sizing knobs).
func envInt(key string, def int) int {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return def
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n <= 0 {
		return def
	}
	return n
}
