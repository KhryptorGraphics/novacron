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

// migrateJobStore is a registry of async migration jobs.
//
// When a database is configured (s.db != nil) the migration_jobs table is the
// source of truth and job status survives a process restart; the in-memory map
// is a write-through cache in front of it. With s.db == nil (no database, or
// tests) the store is memory-only and a restart loses job state.
//
// ponytail: a DB hit is not written back into the cache, so every status poll
// for an evicted / post-restart job re-runs the SELECT. Upgrade = repopulate the
// cache on read if poll volume for cold jobs ever matters.
type migrateJobStore struct {
	mu    sync.Mutex
	jobs  map[string]*migrateJob
	order []string // insertion order, for FIFO eviction at capacity
	cap   int
	db    *sql.DB // nil = memory-only fallback (see type comment)
}

func newMigrateJobStore(capacity int, db *sql.DB) *migrateJobStore {
	if capacity <= 0 {
		capacity = migrateJobStoreCap
	}
	return &migrateJobStore{jobs: make(map[string]*migrateJob), cap: capacity, db: db}
}

// migrateJobs is the process-wide store shared by the /api and /api/v1 route
// trees (registerSecureAPIRoutes runs once per tree), so a job created via either
// prefix is visible from both. Its db handle is wired in
// registerVMMigrateAsyncRoutes once the server's *sql.DB is available.
var migrateJobs = newMigrateJobStore(migrateJobStoreCap, nil)

// create records a new running job for vmID and returns its id. When a database
// is configured the job is also persisted (INSERT) so its status survives a
// restart.
//
// ponytail: FIFO eviction ignores status, so under sustained load a very old
// still-running job could be dropped from the cache; its DB row survives, so
// get() still finds it via the read-through path.
func (s *migrateJobStore) create(vmID string) string {
	id := newMigrateJobID()
	now := time.Now().UTC()
	s.mu.Lock()
	for len(s.order) >= s.cap {
		oldest := s.order[0]
		s.order = s.order[1:]
		delete(s.jobs, oldest)
	}
	s.jobs[id] = &migrateJob{ID: id, VMID: vmID, Status: migrateStatusRunning, StartedAt: now}
	s.order = append(s.order, id)
	s.mu.Unlock()

	if s.db != nil {
		// INSERT outside the mutex (no lock held during I/O). created_at fills from
		// the column default. ponytail: on failure the job stays in the cache so
		// in-process reads still work — only restart-durability is lost, so this is
		// a warn, not an error that would abort a migration that has not started.
		if _, err := s.db.Exec(
			`INSERT INTO migration_jobs (id, vm_id, status, started_at) VALUES ($1, $2, $3, $4)`,
			id, vmID, migrateStatusRunning, now,
		); err != nil {
			logger.Warn("failed to persist migration job", "job", id, "vm", vmID, "error", err)
		}
	}
	return id
}

// finish marks a job terminal: completed on a nil error, failed otherwise. It
// updates the cache entry (if still present) and, when a database is configured,
// the persisted row. The DB UPDATE runs even if the cache entry was evicted so
// the stored terminal state stays correct.
func (s *migrateJobStore) finish(id string, err error) {
	now := time.Now().UTC()
	status := migrateStatusCompleted
	errMsg := ""
	if err != nil {
		status = migrateStatusFailed
		errMsg = err.Error()
	}

	s.mu.Lock()
	if job, ok := s.jobs[id]; ok {
		job.FinishedAt = &now
		job.Status = status
		job.Error = errMsg
	}
	s.mu.Unlock()

	if s.db != nil {
		errArg := sql.NullString{String: errMsg, Valid: errMsg != ""}
		if _, e := s.db.Exec(
			`UPDATE migration_jobs SET status = $2, error = $3, finished_at = $4 WHERE id = $1`,
			id, status, errArg, now,
		); e != nil {
			logger.Warn("failed to persist migration job completion", "job", id, "status", status, "error", e)
		}
	}
}

// get returns a value copy of the job (never the stored pointer) so a reader gets
// a consistent snapshot without racing finish(). A cache miss falls through to
// the DB (getFromDB), which is what lets status survive a restart.
func (s *migrateJobStore) get(id string) (migrateJob, bool) {
	s.mu.Lock()
	if job, ok := s.jobs[id]; ok {
		cp := *job
		s.mu.Unlock()
		return cp, true
	}
	s.mu.Unlock()

	if s.db == nil {
		return migrateJob{}, false
	}
	return s.getFromDB(id)
}

// getFromDB loads a persisted job by id (the query runs without the mutex held).
// A missing row — or any query error — is reported as not-found so the status
// handler returns 404 rather than surfacing a 500.
func (s *migrateJobStore) getFromDB(id string) (migrateJob, bool) {
	var job migrateJob
	var errStr sql.NullString
	var finished sql.NullTime
	err := s.db.QueryRow(
		`SELECT id, vm_id, status, error, started_at, finished_at FROM migration_jobs WHERE id = $1`,
		id,
	).Scan(&job.ID, &job.VMID, &job.Status, &errStr, &job.StartedAt, &finished)
	if err != nil {
		if !errors.Is(err, sql.ErrNoRows) {
			logger.Warn("failed to load migration job", "job", id, "error", err)
		}
		return migrateJob{}, false
	}
	if errStr.Valid {
		job.Error = errStr.String
	}
	if finished.Valid {
		t := finished.Time
		job.FinishedAt = &t
	}
	return job, true
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
	// Wire the shared store's DB handle here — the single point where db reaches
	// the async routes. ponytail: plain assignment (no lock) is race-free because
	// this runs once per route tree at startup, before serving, with the same db
	// each time; nil db keeps the store on its in-memory fallback.
	migrateJobs.db = db

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
