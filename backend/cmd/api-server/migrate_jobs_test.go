package main

import (
	"context"
	"database/sql"
	"errors"
	"net/http"
	"net/http/httptest"
	"regexp"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/gorilla/mux"
)

// TestMigrateJobStore exercises the store's terminal-state transitions and
// capacity eviction directly. Discriminating: it fails if create() does not
// start "running", if finish() drops the running->completed/failed transition or
// the error message, or if eviction does not bound the map.
func TestMigrateJobStore(t *testing.T) {
	s := newMigrateJobStore(2, nil)

	id1 := s.create("vm-1")
	if j, ok := s.get(id1); !ok || j.Status != migrateStatusRunning || j.VMID != "vm-1" || j.FinishedAt != nil {
		t.Fatalf("create should record a running job with no finish time, got %#v ok=%v", j, ok)
	}
	if _, ok := s.get("no-such-job"); ok {
		t.Fatalf("unknown id must not be found")
	}

	s.finish(id1, nil)
	j, ok := s.get(id1)
	if !ok || j.Status != migrateStatusCompleted || j.FinishedAt == nil {
		t.Fatalf("finish(nil) should complete the job and stamp finished_at, got %#v", j)
	}
	if j.Error != "" {
		t.Fatalf("completed job must carry no error, got %q", j.Error)
	}

	id2 := s.create("vm-2")
	s.finish(id2, errors.New("kaboom"))
	j2, _ := s.get(id2)
	if j2.Status != migrateStatusFailed || j2.Error != "kaboom" {
		t.Fatalf("finish(err) should fail the job with the message, got %#v", j2)
	}

	// Capacity is 2: adding a third job evicts the oldest (id1).
	id3 := s.create("vm-3")
	if _, ok := s.get(id1); ok {
		t.Fatalf("oldest job should be evicted once at capacity")
	}
	if _, ok := s.get(id3); !ok {
		t.Fatalf("newest job should be present after eviction")
	}
}

// jobStatusRouter mounts the two async handlers against an injected runner so the
// migration outcome is deterministic (no hypervisor).
func jobStatusRouter(store *migrateJobStore, run migrateRunner) *mux.Router {
	router := mux.NewRouter()
	router.HandleFunc("/vms/{id}/migrate/async", newMigrateAsyncHandler(store, run)).Methods(http.MethodPost)
	router.HandleFunc("/migrate/jobs/{job_id}", newMigrateJobStatusHandler(store)).Methods(http.MethodGet)
	return router
}

func getJob(t *testing.T, router http.Handler, jobID string) (int, migrateJob) {
	t.Helper()
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/migrate/jobs/"+jobID, nil))
	if rec.Code != http.StatusOK {
		return rec.Code, migrateJob{}
	}
	var job migrateJob
	decodeJSONBody(t, rec, &job)
	return rec.Code, job
}

func waitForTerminal(t *testing.T, router http.Handler, jobID string) migrateJob {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		code, job := getJob(t, router, jobID)
		if code == http.StatusOK && job.Status != migrateStatusRunning {
			return job
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("job %s never reached a terminal state", jobID)
	return migrateJob{}
}

// TestMigrateAsyncHandlerCompletes proves the full async lifecycle: POST returns
// 202 + a job id with status "running", GET reflects "running" while the
// migration is in flight, and the job transitions to "completed" once the runner
// returns. Discriminating: the runner blocks on a channel, so if create()
// hardcoded a terminal status the "running before release" check fails, and if
// finish() were dropped waitForTerminal times out.
func TestMigrateAsyncHandlerCompletes(t *testing.T) {
	store := newMigrateJobStore(8, nil)
	release := make(chan struct{})
	var gotVM, gotTarget string
	var gotOptions map[string]string
	run := func(ctx context.Context, vmID, targetNode string, options map[string]string) error {
		gotVM, gotTarget, gotOptions = vmID, targetNode, options
		<-release // hold the job in "running" until the test lets it finish
		return nil
	}
	router := jobStatusRouter(store, run)

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, mustJSONRequest(t, http.MethodPost, "/vms/vm-async/migrate/async",
		map[string]string{"target_node": "node-b", "migration_type": "live"}))
	if rec.Code != http.StatusAccepted {
		t.Fatalf("async POST: expected 202, got %d (%s)", rec.Code, rec.Body.String())
	}
	var accepted map[string]string
	decodeJSONBody(t, rec, &accepted)
	jobID := accepted["job_id"]
	if jobID == "" {
		t.Fatalf("async POST: missing job_id, got %#v", accepted)
	}
	if accepted["status"] != migrateStatusRunning || accepted["vm_id"] != "vm-async" {
		t.Fatalf("async POST: unexpected 202 body %#v", accepted)
	}

	// In flight: status must be "running", not yet terminal.
	code, running := getJob(t, router, jobID)
	if code != http.StatusOK || running.Status != migrateStatusRunning {
		t.Fatalf("expected running job before release, got code=%d job=%#v", code, running)
	}
	if running.FinishedAt != nil {
		t.Fatalf("running job must not have finished_at set, got %#v", running)
	}

	close(release) // let the migration finish

	done := waitForTerminal(t, router, jobID)
	if done.Status != migrateStatusCompleted {
		t.Fatalf("expected completed, got %q (err=%q)", done.Status, done.Error)
	}
	if done.FinishedAt == nil || done.Error != "" {
		t.Fatalf("completed job should stamp finished_at and carry no error, got %#v", done)
	}
	if gotVM != "vm-async" || gotTarget != "node-b" || gotOptions["migration_type"] != "live" {
		t.Fatalf("runner did not receive the request fields: vm=%q target=%q opts=%#v", gotVM, gotTarget, gotOptions)
	}
}

// TestMigrateAsyncHandlerFails proves a failing migration lands the job in
// "failed" with the runner's error surfaced, and that an unknown job id is 404.
func TestMigrateAsyncHandlerFails(t *testing.T) {
	store := newMigrateJobStore(8, nil)
	run := func(ctx context.Context, vmID, targetNode string, options map[string]string) error {
		return errors.New("target unreachable")
	}
	router := jobStatusRouter(store, run)

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, mustJSONRequest(t, http.MethodPost, "/vms/vm-x/migrate/async",
		map[string]string{"target_node": "node-b"}))
	if rec.Code != http.StatusAccepted {
		t.Fatalf("async POST: expected 202, got %d (%s)", rec.Code, rec.Body.String())
	}
	var accepted map[string]string
	decodeJSONBody(t, rec, &accepted)

	done := waitForTerminal(t, router, accepted["job_id"])
	if done.Status != migrateStatusFailed || done.Error != "target unreachable" {
		t.Fatalf("expected failed job with error, got %#v", done)
	}

	// Unknown job id -> 404.
	if code, _ := getJob(t, router, "mig-does-not-exist"); code != http.StatusNotFound {
		t.Fatalf("unknown job id: expected 404, got %d", code)
	}
}

// TestMigrateAsyncHandlerRejectsMissingTarget proves the async route enforces the
// same required-field validation as the sync route before creating any job.
func TestMigrateAsyncHandlerRejectsMissingTarget(t *testing.T) {
	store := newMigrateJobStore(8, nil)
	called := false
	run := func(ctx context.Context, vmID, targetNode string, options map[string]string) error {
		called = true
		return nil
	}
	router := jobStatusRouter(store, run)

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, mustJSONRequest(t, http.MethodPost, "/vms/vm-x/migrate/async", map[string]string{}))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("missing target_node: expected 400, got %d (%s)", rec.Code, rec.Body.String())
	}
	// Give any (erroneously spawned) goroutine a chance to run, then assert none did.
	time.Sleep(10 * time.Millisecond)
	if called {
		t.Fatalf("runner must not be invoked when validation fails")
	}
}

// TestRegisterSecureAPIRoutesMountsAsyncMigration proves the async routes are
// actually wired into registerSecureAPIRoutes (guards against a dead handler that
// never receives traffic). router.Match resolves the routes without executing
// the handlers, so no real migration runs.
func TestRegisterSecureAPIRoutesMountsAsyncMigration(t *testing.T) {
	db, _, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	manager := newStubVMManager(t)
	defer manager.Stop()

	router := mux.NewRouter()
	registerSecureAPIRoutes(router, db, manager, t.TempDir())

	cases := []struct{ method, path string }{
		{http.MethodPost, "/vms/vm-1/migrate/async"},
		{http.MethodGet, "/migrate/jobs/job-1"},
	}
	for _, c := range cases {
		req := httptest.NewRequest(c.method, c.path, nil)
		var match mux.RouteMatch
		if !router.Match(req, &match) {
			t.Fatalf("registerSecureAPIRoutes did not mount %s %s", c.method, c.path)
		}
	}
}

const (
	migrateJobInsertSQL = `INSERT INTO migration_jobs (id, vm_id, status, started_at) VALUES ($1, $2, $3, $4)`
	migrateJobUpdateSQL = `UPDATE migration_jobs SET status = $2, error = $3, finished_at = $4 WHERE id = $1`
	migrateJobSelectSQL = `SELECT id, vm_id, status, error, started_at, finished_at FROM migration_jobs WHERE id = $1`
)

// TestMigrateJobStorePersistsCompletedLifecycle proves the DB-backed store writes
// each transition through: create INSERTs a running row (id/started_at generated,
// so AnyArg), and finish(nil) UPDATEs it to completed with a NULL error.
// Discriminating: sqlmock's ordered, argument-checked expectations fail if a
// statement is skipped, carries the wrong status, or writes an error on success,
// and ExpectationsWereMet fails if a transition never reaches the DB.
func TestMigrateJobStorePersistsCompletedLifecycle(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	store := newMigrateJobStore(8, db)

	mock.ExpectExec(regexp.QuoteMeta(migrateJobInsertSQL)).
		WithArgs(sqlmock.AnyArg(), "vm-1", migrateStatusRunning, sqlmock.AnyArg()).
		WillReturnResult(sqlmock.NewResult(1, 1))

	id := store.create("vm-1")
	if id == "" {
		t.Fatal("create returned an empty job id")
	}

	mock.ExpectExec(regexp.QuoteMeta(migrateJobUpdateSQL)).
		WithArgs(id, migrateStatusCompleted, nil, sqlmock.AnyArg()).
		WillReturnResult(sqlmock.NewResult(0, 1))

	store.finish(id, nil)

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations for create+finish(nil): %v", err)
	}
}

// TestMigrateJobStorePersistsFailure proves finish(err) UPDATEs the persisted row
// to failed and stores the runner's error message. Discriminating: the WithArgs
// match fails if the status is not "failed" or the error text is dropped.
func TestMigrateJobStorePersistsFailure(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	store := newMigrateJobStore(8, db)

	mock.ExpectExec(regexp.QuoteMeta(migrateJobInsertSQL)).
		WithArgs(sqlmock.AnyArg(), "vm-x", migrateStatusRunning, sqlmock.AnyArg()).
		WillReturnResult(sqlmock.NewResult(1, 1))

	id := store.create("vm-x")

	mock.ExpectExec(regexp.QuoteMeta(migrateJobUpdateSQL)).
		WithArgs(id, migrateStatusFailed, "target unreachable", sqlmock.AnyArg()).
		WillReturnResult(sqlmock.NewResult(0, 1))

	store.finish(id, errors.New("target unreachable"))

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations for finish(err): %v", err)
	}
}

// TestMigrateJobStoreReadsPersistedRowAfterRestart proves the durability payoff:
// a fresh store (empty cache) models a just-restarted process, and get() falls
// through the cache miss to SELECT the persisted row. Discriminating: if get()
// stopped at the cache it would report not-found, and the field assertions catch
// a botched scan of the nullable error / finished_at columns.
func TestMigrateJobStoreReadsPersistedRowAfterRestart(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	store := newMigrateJobStore(8, db) // empty cache == freshly-restarted process

	startedAt := time.Now().Add(-2 * time.Minute).UTC()
	finishedAt := time.Now().Add(-1 * time.Minute).UTC()
	mock.ExpectQuery(regexp.QuoteMeta(migrateJobSelectSQL)).
		WithArgs("mig-persisted").
		WillReturnRows(sqlmock.NewRows([]string{"id", "vm_id", "status", "error", "started_at", "finished_at"}).
			AddRow("mig-persisted", "vm-7", migrateStatusCompleted, nil, startedAt, finishedAt))

	job, ok := store.get("mig-persisted")
	if !ok {
		t.Fatal("expected the persisted job to be found via the DB after a cache miss")
	}
	if job.ID != "mig-persisted" || job.VMID != "vm-7" || job.Status != migrateStatusCompleted {
		t.Fatalf("persisted job mismatch: %#v", job)
	}
	if job.FinishedAt == nil {
		t.Fatal("persisted completed job should carry finished_at")
	}
	if job.Error != "" {
		t.Fatalf("completed job should have no error, got %q", job.Error)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

// TestMigrateJobStatusHandlerNotFoundWhenAbsentFromDB proves an id absent from
// both the cache and the DB yields 404 (not 500) through the status handler.
func TestMigrateJobStatusHandlerNotFoundWhenAbsentFromDB(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	store := newMigrateJobStore(8, db)
	router := jobStatusRouter(store, func(context.Context, string, string, map[string]string) error { return nil })

	mock.ExpectQuery(regexp.QuoteMeta(migrateJobSelectSQL)).
		WithArgs("mig-nope").
		WillReturnError(sql.ErrNoRows)

	if code, _ := getJob(t, router, "mig-nope"); code != http.StatusNotFound {
		t.Fatalf("expected 404 for an id absent from cache and DB, got %d", code)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}
