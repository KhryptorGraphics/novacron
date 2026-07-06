//go:build !novacron_enhanced && !novacron_improved && !novacron_multicloud && !novacron_production && !novacron_real_backend && !novacron_secure && !novacron_working && !novacron_simple_api

package main

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/mux"

	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
)

// Async migration: POST /api/vms/{id}/migrate/async returns 202 with a job id and
// runs the migration on a background goroutine, so a large/slow migration can
// outlive the synchronous request's WRITE_TIMEOUT. Job status is polled via
// GET /api/migrate/jobs/{job_id}. Both routes are registered on the same secure
// (JWT) subrouter as the sync route, so they inherit its auth/RBAC unchanged.

const (
	migrateStatusRunning   = "running"
	migrateStatusCompleted = "completed"
	migrateStatusFailed    = "failed"

	// migrateJobStoreCap bounds the in-memory job map; oldest jobs are evicted.
	migrateJobStoreCap = 1024
)

// migrateJob is the externally-visible status of one async migration. Field tags
// match the documented response shape: {id, vm_id, status, error?, started_at,
// finished_at?}.
type migrateJob struct {
	ID         string     `json:"id"`
	VMID       string     `json:"vm_id"`
	Status     string     `json:"status"` // running | completed | failed
	Error      string     `json:"error,omitempty"`
	StartedAt  time.Time  `json:"started_at"`
	FinishedAt *time.Time `json:"finished_at,omitempty"`
}

// migrateJobStore is a process-local registry of async migration jobs.
//
// ponytail: in-memory only — a process restart loses all job state (an in-flight
// migration keeps running via qemu, but its job row disappears). Upgrade path
// when durability matters: persist to a migration_jobs table and reload on boot.
type migrateJobStore struct {
	mu    sync.Mutex
	jobs  map[string]*migrateJob
	order []string // insertion order, for FIFO eviction at capacity
	cap   int
}

func newMigrateJobStore(capacity int) *migrateJobStore {
	if capacity <= 0 {
		capacity = migrateJobStoreCap
	}
	return &migrateJobStore{jobs: make(map[string]*migrateJob), cap: capacity}
}

// migrateJobs is the process-wide store shared by the /api and /api/v1 route
// trees (registerSecureAPIRoutes runs once per tree), so a job created via either
// prefix is visible from both.
var migrateJobs = newMigrateJobStore(migrateJobStoreCap)

// create records a new running job for vmID and returns its id.
//
// ponytail: FIFO eviction ignores status, so under sustained load a very old
// still-running job could be dropped; cap is high enough that terminal jobs age
// out first in practice. Upgrade = DB persistence (see the type comment).
func (s *migrateJobStore) create(vmID string) string {
	id := newMigrateJobID()
	now := time.Now().UTC()
	s.mu.Lock()
	defer s.mu.Unlock()
	for len(s.order) >= s.cap {
		oldest := s.order[0]
		s.order = s.order[1:]
		delete(s.jobs, oldest)
	}
	s.jobs[id] = &migrateJob{ID: id, VMID: vmID, Status: migrateStatusRunning, StartedAt: now}
	s.order = append(s.order, id)
	return id
}

// finish marks a job terminal: completed on a nil error, failed otherwise. It is
// a no-op if the job was already evicted.
func (s *migrateJobStore) finish(id string, err error) {
	now := time.Now().UTC()
	s.mu.Lock()
	defer s.mu.Unlock()
	job, ok := s.jobs[id]
	if !ok {
		return
	}
	job.FinishedAt = &now
	if err != nil {
		job.Status = migrateStatusFailed
		job.Error = err.Error()
		return
	}
	job.Status = migrateStatusCompleted
}

// get returns a value copy of the job (never the stored pointer) so a reader gets
// a consistent snapshot without racing finish().
func (s *migrateJobStore) get(id string) (migrateJob, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	job, ok := s.jobs[id]
	if !ok {
		return migrateJob{}, false
	}
	return *job, true
}

func newMigrateJobID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		// crypto/rand failure is effectively impossible; fall back to a timestamp so
		// we still return a usable id rather than panicking a live request.
		return "mig-" + time.Now().UTC().Format("20060102150405.000000000")
	}
	return "mig-" + hex.EncodeToString(b[:])
}

// migrateRunner runs one migration to completion and returns its terminal error
// (nil on success). Production wires this to VMManager.MigrateVM + source-row
// cleanup (see registerVMMigrateAsyncRoutes); tests inject a deterministic stub.
type migrateRunner func(ctx context.Context, vmID, targetNode string, options map[string]string) error

// newMigrateAsyncHandler builds POST /vms/{id}/migrate/async: it validates the
// same body as the synchronous route, registers a running job, kicks the
// migration off on a background goroutine, and returns 202 immediately.
func newMigrateAsyncHandler(jobs *migrateJobStore, run migrateRunner) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		vmID := mux.Vars(r)["id"]

		// ponytail: mirrors registerVMMigrateRoute's body decode + option building
		// inline, deliberately kept separate so the synchronous route stays
		// byte-for-byte unchanged. Upgrade = a shared parse helper if a third caller
		// appears.
		var req struct {
			TargetNode    string `json:"target_node"`
			MigrationType string `json:"migration_type,omitempty"`
			URI           string `json:"uri,omitempty"`
			TargetAddr    string `json:"target_addr,omitempty"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}
		if strings.TrimSpace(req.TargetNode) == "" {
			writeJSONError(w, http.StatusBadRequest, "target_node is required")
			return
		}
		options := map[string]string{}
		if req.MigrationType != "" {
			options["migration_type"] = req.MigrationType
		}
		if req.URI != "" {
			options["uri"] = req.URI
		}
		if req.TargetAddr != "" {
			options["target_addr"] = req.TargetAddr
		}

		jobID := jobs.create(vmID)

		go func() {
			// context.Background (NOT r.Context): the request returns immediately, so
			// the migration must not be cancelled when its HTTP handler unwinds. The
			// 10-minute ceiling matches the sync route's MigrateVM timeout.
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
			defer cancel()

			err := run(ctx, vmID, req.TargetNode, options)
			jobs.finish(jobID, err)
			if err != nil {
				logger.Warn("async migration failed", "job", jobID, "vm", vmID, "target", req.TargetNode, "error", err)
				return
			}
			logger.Info("async migration completed", "job", jobID, "vm", vmID, "target", req.TargetNode)
		}()

		writeJSON(w, http.StatusAccepted, map[string]interface{}{
			"job_id": jobID,
			"vm_id":  vmID,
			"status": migrateStatusRunning,
		})
	}
}

// newMigrateJobStatusHandler builds GET /migrate/jobs/{job_id}.
func newMigrateJobStatusHandler(jobs *migrateJobStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		jobID := mux.Vars(r)["job_id"]
		job, ok := jobs.get(jobID)
		if !ok {
			writeJSONError(w, http.StatusNotFound, "migration job not found")
			return
		}
		writeJSON(w, http.StatusOK, job)
	}
}

// registerVMMigrateAsyncRoutes wires the async POST + status GET onto the secure
// subrouter, right beside the sync migrate route. The production runner reuses
// the exact same call path as the sync route (VMManager.MigrateVM, then the
// source-row cleanup) — it does not reimplement any migration logic.
func registerVMMigrateAsyncRoutes(router *mux.Router, db *sql.DB, vmManager *core_vm.VMManager) {
	run := func(ctx context.Context, vmID, targetNode string, options map[string]string) error {
		if vmManager == nil {
			return errors.New("vm manager unavailable")
		}
		if err := vmManager.MigrateVM(ctx, vmID, targetNode, options); err != nil {
			return err
		}
		// Mirror the sync route: the guest now runs on targetNode, so drop the source
		// DB row (the destination inserts its own via registerMigratedDest).
		if _, err := db.Exec(`DELETE FROM vms WHERE id = $1`, vmID); err != nil {
			return fmt.Errorf("migration succeeded but source VM row cleanup failed: %w", err)
		}
		return nil
	}

	router.HandleFunc("/vms/{id}/migrate/async", newMigrateAsyncHandler(migrateJobs, run)).Methods(http.MethodPost)
	router.HandleFunc("/migrate/jobs/{job_id}", newMigrateJobStatusHandler(migrateJobs)).Methods(http.MethodGet)
}
