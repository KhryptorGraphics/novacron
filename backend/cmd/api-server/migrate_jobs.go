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
	// migrateStatusInterrupted marks a job whose owning api-server died before
	// it could finalise the job. It is deliberately NOT "failed": the migration
	// itself is QEMU-native and may well have completed — the row simply cannot
	// know. Callers must inspect the VM's actual node.
	migrateStatusInterrupted = "interrupted"

	// migrateJobStoreCap bounds the in-memory job map; oldest jobs are evicted.
	migrateJobStoreCap = 1024
)

// migrateJob is the externally-visible status of one async migration. The
// organization_id is deliberately internal: it authorizes status reads but is
// not part of the public status response.
type migrateJob struct {
	ID             string     `json:"id"`
	VMID           string     `json:"vm_id"`
	Status         string     `json:"status"` // running | completed | failed
	Error          string     `json:"error,omitempty"`
	StartedAt      time.Time  `json:"started_at"`
	FinishedAt     *time.Time `json:"finished_at,omitempty"`
	OrganizationID string     `json:"-"`
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
// restart. orgID is the normalized organization that owns the source VM; it is
// empty for NULL-org VMs and is stored on the row so the completed job remains
// readable by its tenant after the source VM row is removed.
//
// ponytail: FIFO eviction ignores status, so under sustained load a very old
// still-running job could be dropped from the cache; its DB row survives, so
// get() still finds it via the read-through path.
func (s *migrateJobStore) create(vmID, orgID string) string {
	id := newMigrateJobID()
	now := time.Now().UTC()
	s.mu.Lock()
	for len(s.order) >= s.cap {
		oldest := s.order[0]
		s.order = s.order[1:]
		delete(s.jobs, oldest)
	}
	s.jobs[id] = &migrateJob{ID: id, VMID: vmID, Status: migrateStatusRunning, StartedAt: now, OrganizationID: orgID}
	s.order = append(s.order, id)
	s.mu.Unlock()

	if s.db != nil {
		// INSERT outside the mutex (no lock held during I/O). created_at fills from
		// the column default. ponytail: on failure the job stays in the cache so
		// in-process reads still work — only restart-durability is lost, so this is
		// a warn, not an error that would abort a migration that has not started.
		var orgArg interface{}
		if orgID != "" {
			orgArg = orgID
		}
		if _, err := s.db.Exec(
			`INSERT INTO migration_jobs (id, vm_id, status, started_at, organization_id) VALUES ($1, $2, $3, $4, $5)`,
			id, vmID, migrateStatusRunning, now, orgArg,
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
			`UPDATE migration_jobs
			 SET status = $2, error = $3, finished_at = $4,
			     organization_id = COALESCE(organization_id,
			         (SELECT organization_id FROM vms WHERE id::text = migration_jobs.vm_id))
			 WHERE id = $1`,
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
	var orgID sql.NullString
	err := s.db.QueryRow(
		`SELECT id, vm_id, status, error, started_at, finished_at, organization_id FROM migration_jobs WHERE id = $1`,
		id,
	).Scan(&job.ID, &job.VMID, &job.Status, &errStr, &job.StartedAt, &finished, &orgID)
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
	if orgID.Valid {
		job.OrganizationID = orgID.String
	}
	return job, true
}

// reconcileInterruptedJobs finalises jobs left in "running" by a previous
// process. A job's finishing goroutine dies with the api-server, so without
// this the row reads "running" forever (observed live 2026-09-20: a job still
// reported running long after its migration window had passed). The status is
// "interrupted", not "failed": the migration is QEMU-native and may have
// completed — the row cannot know, and the VM's actual node is the truth.
// Returns how many rows were closed.
func (s *migrateJobStore) reconcileInterruptedJobs() int {
	if s.db == nil {
		return 0
	}
	const note = "api-server restarted while this migration job was in flight; the QEMU migration may have completed — check which node actually runs the VM"
	res, err := s.db.Exec(
		`UPDATE migration_jobs SET status = $1, error = $2, finished_at = NOW()
		 WHERE status = $3 AND finished_at IS NULL`,
		migrateStatusInterrupted, note, migrateStatusRunning,
	)
	if err != nil {
		logger.Warn("migration job reconcile failed", "error", err)
		return 0
	}
	n, _ := res.RowsAffected()
	if n > 0 {
		logger.Info("reconciled interrupted migration jobs", "count", n)
	}
	return int(n)
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

// migrateRunner executes one migration and removes its source row using the
// same organization scope authorized by the request.
type migrateRunner func(ctx context.Context, vmID, targetNode string, options map[string]string, orgID string, isAdmin bool) error

// newMigrateAsyncHandler builds POST /vms/{id}/migrate/async: it validates the
// same body as the synchronous route, authorizes the VM against the caller's
// organization scope, registers a running job with the authorized org, kicks the
// migration off on a background goroutine, and returns 202 immediately.
// Out-of-scope VMs return 404 (hidden existence) and the runner is never invoked.
func newMigrateAsyncHandler(jobs *migrateJobStore, db *sql.DB, run migrateRunner) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		vmID := mux.Vars(r)["id"]

		// Authorize the VM against the caller's organization scope before any
		// side effects. Hidden-existence semantics: out-of-scope returns 404.
		orgID, isAdmin, visible := requireOrgScope(r.Context(), db, vmID)
		if !visible {
			writeJSONError(w, http.StatusNotFound, "vm not found")
			return
		}

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

		// Admins are unscoped, but the persisted job still needs the source VM's
		// organization so its tenant can read it after source-row cleanup.
		if isAdmin && db != nil {
			var sourceOrg sql.NullString
			err := db.QueryRowContext(r.Context(),
				`SELECT organization_id FROM vms WHERE id = $1`, vmID).Scan(&sourceOrg)
			if err != nil {
				writeJSONError(w, http.StatusNotFound, "vm not found")
				return
			}
			if sourceOrg.Valid {
				orgID = sourceOrg.String
			}
		}

		jobID := jobs.create(vmID, orgID)

		go func() {
			// context.Background (NOT r.Context): the request returns immediately, so
			// the migration must not be cancelled when its HTTP handler unwinds. The
			// 10-minute ceiling matches the sync route's MigrateVM timeout.
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
			defer cancel()

			err := run(ctx, vmID, req.TargetNode, options, orgID, isAdmin)
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
// It returns 404 (hidden existence) for jobs outside the caller's org.
func newMigrateJobStatusHandler(jobs *migrateJobStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		job, ok := jobs.get(mux.Vars(r)["job_id"])
		if !ok {
			writeJSONError(w, http.StatusNotFound, "migration job not found")
			return
		}
		scopeOrg, isAdmin, _ := requireOrgScope(r.Context(), nil, "")
		if !isAdmin {
			var rowOrg sql.NullString
			if job.OrganizationID != "" {
				rowOrg = sql.NullString{String: job.OrganizationID, Valid: true}
			}
			if !orgVisible(scopeOrg, rowOrg) {
				writeJSONError(w, http.StatusNotFound, "migration job not found")
				return
			}
		}
		writeJSON(w, http.StatusOK, job)
	}
}
func registerVMMigrateAsyncRoutes(router *mux.Router, db *sql.DB, vmManager *core_vm.VMManager) {
	// Wire the shared store's DB handle here — the single point where db reaches
	// the async routes. ponytail: plain assignment (no lock) is race-free because
	// this runs once per route tree at startup, before serving, with the same db
	// each time; nil db keeps the store on its in-memory fallback.
	migrateJobs.db = db
	// A previous process may have died mid-job; close those rows honestly
	// (status "interrupted") instead of leaving them "running" forever.
	migrateJobs.reconcileInterruptedJobs()

	run := func(ctx context.Context, vmID, targetNode string, options map[string]string, orgID string, isAdmin bool) error {
		if vmManager == nil {
			return errors.New("vm manager unavailable")
		}
		if err := vmManager.MigrateVM(ctx, vmID, targetNode, options); err != nil {
			return err
		}
		// Remove only the source row previously authorized for this caller.
		// Admins are deliberately unscoped; tenant cleanup repeats the scope in
		// the mutation so a changed/mismatched row cannot be deleted cross-org.
		if db == nil {
			return nil
		}
		var (
			result sql.Result
			err    error
		)
		switch {
		case isAdmin:
			result, err = db.ExecContext(ctx, `DELETE FROM vms WHERE id = $1`, vmID)
		case orgID != "":
			result, err = db.ExecContext(ctx, `DELETE FROM vms WHERE id = $1 AND organization_id = $2`, vmID, orgID)
		default:
			result, err = db.ExecContext(ctx, `DELETE FROM vms WHERE id = $1 AND organization_id IS NULL`, vmID)
		}
		if err != nil {
			return fmt.Errorf("migration succeeded but source VM row cleanup failed: %w", err)
		}
		if affected, err := result.RowsAffected(); err == nil && affected == 0 {
			return errors.New("migration succeeded but authorized source VM row was not found")
		}
		return nil
	}

	router.HandleFunc("/vms/{id}/migrate/async", newMigrateAsyncHandler(migrateJobs, db, run)).Methods(http.MethodPost)
	router.HandleFunc("/migrate/jobs/{job_id}", newMigrateJobStatusHandler(migrateJobs)).Methods(http.MethodGet)
}
