package vm

// restart_supervisor.go: crash-restart supervision for compute-provided VM
// drivers (process/KVM). The supervisor watches every vms row whose state is
// 'running', polls the owning driver's GetStatus, and restarts a VM whose
// process died (StateStopped / StateUnknown / status error) using the exact
// VMConfig that created it. Restart behaviour is governed by a per-VM
// RestartPolicy ("no" / "on-failure" / "always") with a fibonacci backoff
// (2s * fib) capped at 8 attempts before the row is marked permanent-failure.
//
// Persistence: every transition is upserted into vm_restart_state (migration
// 000012, ON CONFLICT (vm_id) DO UPDATE) so one row lives per VM and no
// double-kill history accumulates. When constructed with a nil *sql.DB the
// supervisor runs memory-only (used by unit tests).

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"os"
	"reflect"
	"strings"
	"sync"
	"time"
)

// RestartEnvVar overrides the default restart policy for VMs whose config
// does not carry an explicit RestartPolicy.
const RestartEnvVar = "NOVACRON_VM_RESTART_POLICY"

// RestartPolicy controls whether the supervisor restarts a crashed VM.
type RestartPolicy string

// RestartPolicy values (used verbatim in VMConfig.RestartPolicy once the
// integration field lands, in vm_restart_state.policy, and in the JSON API).
const (
	// RestartPolicyNo never auto-restarts.
	RestartPolicyNo RestartPolicy = "no"
	// RestartPolicyOnFailure restarts when the process died unexpectedly.
	// This is the default when NOVACRON_VM_RESTART_POLICY is unset/invalid.
	RestartPolicyOnFailure RestartPolicy = "on-failure"
	// RestartPolicyAlways restarts on any death while the VM is registered
	// as running (an explicit RecordStop still suppresses restarts).
	RestartPolicyAlways RestartPolicy = "always"
)

// Internal row states persisted in vm_restart_state.state.
const (
	restartStateWatching         = "watching"           // healthy / no restart scheduled
	restartStateBackingOff       = "backing-off"        // restart scheduled for next_attempt_at
	restartStateStopped          = "stopped"            // RecordStop'd; never restart until RecordStart
	restartStatePermanentFailure = "permanent-failure"  // 8 attempts exhausted; stops retrying
)

const (
	restartMaxAttempts = 8                  // fib backoff positions before permanent-failure
	restartBackoffBase = 2 * time.Second    // delays: 2s * fib(n) => 2,2,4,6,10,16,26,42 s
	restartStableAfter = 60 * time.Second   // healthy run duration that resets the attempt counter
)

// VMRestartStatus is the inspection payload for a single supervised VM, used
// by the /api/vms/:id restart-status surface.
type VMRestartStatus struct {
	VMID          string        `json:"vm_id"`
	Policy        RestartPolicy `json:"policy"`
	State         string        `json:"state"`
	Attempts      int           `json:"attempts"`
	MaxAttempts   int           `json:"max_attempts"`
	LastState     VMState       `json:"last_state"`
	LastError     string        `json:"last_error,omitempty"`
	NextAttemptAt *time.Time    `json:"next_attempt_at,omitempty"`
	LastRestartAt *time.Time    `json:"last_restart_at,omitempty"`
	UpdatedAt     time.Time     `json:"updated_at"`
}

// vmRestartRecord is the supervisor's live per-VM state.
type vmRestartRecord struct {
	policy        RestartPolicy
	state         string
	attempts      int
	lastState     VMState
	lastError     string
	nextAttemptAt time.Time
	lastRestartAt time.Time
	updatedAt     time.Time
}

// RestartSupervisor periodically reconciles "VM claims to be running" (vms
// table) against "the driver process is actually alive", restarting crashed
// VMs with backoff according to their restart policy.
type RestartSupervisor struct {
	manager *VMManager
	db      *sql.DB // nil => memory-only mode (unit tests)
	tick    time.Duration

	defaultPolicy RestartPolicy

	// lifecycle
	mu      sync.Mutex
	running bool
	cancel  context.CancelFunc
	wg      sync.WaitGroup

	// per-VM records + policy overrides
	recMu       sync.Mutex
	records     map[string]*vmRestartRecord
	policyOf    map[string]RestartPolicy

	// test seams (set before Start)
	now         func() time.Time
	stableAfter time.Duration
}

// NewRestartSupervisor builds a supervisor polling every tick. tick <= 0 is
// clamped to 5s. The default policy comes from NOVACRON_VM_RESTART_POLICY and
// falls back to on-failure.
func NewRestartSupervisor(vmManager *VMManager, db *sql.DB, tick time.Duration) *RestartSupervisor {
	if tick <= 0 {
		tick = 5 * time.Second
	}
	return &RestartSupervisor{
		manager:       vmManager,
		db:            db,
		tick:          tick,
		defaultPolicy: envRestartPolicy(),
		records:       make(map[string]*vmRestartRecord),
		policyOf:      make(map[string]RestartPolicy),
		now:           time.Now,
		stableAfter:   restartStableAfter,
	}
}

func envRestartPolicy() RestartPolicy {
	switch normalizeRestartPolicy(os.Getenv(RestartEnvVar)) {
	case RestartPolicyNo:
		return RestartPolicyNo
	case RestartPolicyAlways:
		return RestartPolicyAlways
	default:
		return RestartPolicyOnFailure
	}
}

func normalizeRestartPolicy(raw string) RestartPolicy {
	switch RestartPolicy(strings.ToLower(strings.TrimSpace(raw))) {
	case RestartPolicyNo:
		return RestartPolicyNo
	case RestartPolicyOnFailure:
		return RestartPolicyOnFailure
	case RestartPolicyAlways:
		return RestartPolicyAlways
	default:
		return ""
	}
}

// configRestartPolicy reads VMConfig.RestartPolicy reflectively: the field is
// added by the api-server integration (auto json tag "restart_policy"); until
// then this returns false and the env/default or SetRestartPolicy value wins.
func configRestartPolicy(cfg VMConfig) (RestartPolicy, bool) {
	f := reflect.ValueOf(cfg).FieldByName("RestartPolicy")
	if !f.IsValid() || f.Kind() != reflect.String {
		return "", false
	}
	if p := normalizeRestartPolicy(f.String()); p != "" {
		return p, true
	}
	return "", false
}

// SetRestartPolicy pins an explicit per-VM policy, winning over the config
// field and the environment default. Used by the caller integration to push
// VMConfig.RestartPolicy values in.
func (s *RestartSupervisor) SetRestartPolicy(vmID string, policy RestartPolicy) {
	if p := normalizeRestartPolicy(string(policy)); p != "" {
		s.recMu.Lock()
		s.policyOf[vmID] = p
		s.recMu.Unlock()
	}
}

// RecordStart registers (or re-arms) a VM with the supervisor. Called by the
// VMManager start path. Resets the crash-spree attempt counter and switches
// the row back to watching. Idempotent.
func (s *RestartSupervisor) RecordStart(ctx context.Context, vmID string) {
	s.recMu.Lock()
	rec := s.recordLocked(vmID, ctx)
	rec.state = restartStateWatching
	rec.attempts = 0
	rec.nextAttemptAt = time.Time{}
	rec.lastError = ""
	rec.updatedAt = s.now()
	upsert := rec
	s.recMu.Unlock()
	s.persist(ctx, vmID, upsert)
}

// RecordStop marks a VM intentionally stopped so the supervisor will not
// restart it even under policy "always". Idempotent.
func (s *RestartSupervisor) RecordStop(ctx context.Context, vmID string) {
	s.recMu.Lock()
	rec := s.recordLocked(vmID, ctx)
	rec.state = restartStateStopped
	rec.nextAttemptAt = time.Time{}
	rec.updatedAt = s.now()
	upsert := rec
	s.recMu.Unlock()
	s.persist(ctx, vmID, upsert)
}

// Inspect returns the supervisor's current view of one VM. The second result
// is false when the VM has never been recorded and no persisted row exists.
func (s *RestartSupervisor) Inspect(ctx context.Context, vmID string) (VMRestartStatus, bool) {
	s.recMu.Lock()
	if rec, ok := s.records[vmID]; ok {
		status := statusOf(vmID, rec)
		s.recMu.Unlock()
		return status, true
	}
	s.recMu.Unlock()
	if rec, ok := s.loadPersistedOne(ctx, vmID); ok {
		return *rec, true
	}
	return VMRestartStatus{}, false
}

// Statuses lists inspection payloads for every tracked VM (stable sort by id
// is the caller's job if needed).
func (s *RestartSupervisor) Statuses(ctx context.Context) []VMRestartStatus {
	s.recMu.Lock()
	out := make([]VMRestartStatus, 0, len(s.records))
	for id, rec := range s.records {
		out = append(out, statusOf(id, rec))
	}
	s.recMu.Unlock()
	return out
}

func statusOf(vmID string, rec *vmRestartRecord) VMRestartStatus {
	st := VMRestartStatus{
		VMID:        vmID,
		Policy:      rec.policy,
		State:       rec.state,
		Attempts:    rec.attempts,
		MaxAttempts: restartMaxAttempts,
		LastState:   rec.lastState,
		LastError:   rec.lastError,
		UpdatedAt:   rec.updatedAt,
	}
	if !rec.nextAttemptAt.IsZero() {
		t := rec.nextAttemptAt
		st.NextAttemptAt = &t
	}
	if !rec.lastRestartAt.IsZero() {
		t := rec.lastRestartAt
		st.LastRestartAt = &t
	}
	return st
}

// Start launches the polling loop. Safe to call repeatedly; a second call on
// a running supervisor is a no-op.
func (s *RestartSupervisor) Start(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.running {
		return nil
	}
	if s.db != nil {
		if err := s.loadPersisted(); err != nil {
			log.Printf("restart-supervisor: failed to load persisted state: %v", err)
		}
	}
	runCtx, cancel := context.WithCancel(ctx)
	s.cancel = cancel
	s.running = true
	s.wg.Add(1)
	go s.run(runCtx)
	return nil
}

// Stop halts the polling loop and waits for the in-flight tick to finish.
// Safe to call repeatedly; stopping a stopped supervisor is a no-op.
func (s *RestartSupervisor) Stop() {
	s.mu.Lock()
	if !s.running {
		s.mu.Unlock()
		return
	}
	s.running = false
	cancel := s.cancel
	s.cancel = nil
	s.mu.Unlock()
	cancel()
	s.wg.Wait()
}

func (s *RestartSupervisor) run(ctx context.Context) {
	defer s.wg.Done()
	ticker := time.NewTicker(s.tick)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.tickAll(ctx)
		}
	}
}

// tickAll runs one supervision pass over every VM whose vms row claims
// 'running' (or, in memory-only mode, every VM the manager reports running).
func (s *RestartSupervisor) tickAll(ctx context.Context) {
	for _, id := range s.runningVMIDs(ctx) {
		if ctx.Err() != nil {
			return
		}
		s.tickVM(ctx, id)
	}
}

func (s *RestartSupervisor) runningVMIDs(ctx context.Context) []string {
	if s.db != nil {
		rows, err := s.db.QueryContext(ctx, `SELECT id::text FROM vms WHERE state = 'running'`)
		if err != nil {
			log.Printf("restart-supervisor: vms query failed: %v", err)
			return nil
		}
		defer rows.Close()
		var ids []string
		for rows.Next() {
			var id string
			if err := rows.Scan(&id); err == nil {
				ids = append(ids, id)
			}
		}
		return ids
	}
	vms := s.manager.ListVMsByState(StateRunning)
	ids := make([]string, 0, len(vms))
	for _, vm := range vms {
		ids = append(ids, vm.ID())
	}
	return ids
}

// tickVM evaluates one VM: observe driver status, schedule restarts for dead
// processes, perform due restarts, reset attempts after a stable run.
func (s *RestartSupervisor) tickVM(ctx context.Context, vmID string) {
	vm, err := s.manager.GetVM(vmID)
	if err != nil {
		return // row not ours (e.g. other node); this node only restarts VMs it manages
	}
	driver, err := s.manager.getDriverForVM(vm)
	if err != nil {
		log.Printf("restart-supervisor: no driver for VM %s: %v", vmID, err)
		return
	}

	now := s.now()

	s.recMu.Lock()
	rec := s.recordLocked(vmID, ctx)
	persist := false

	switch rec.state {
	case restartStatePermanentFailure, restartStateStopped:
		s.recMu.Unlock()
		return
	}

	status, statusErr := driver.GetStatus(ctx, vmID)
	dead := statusErr != nil ||
		status == StateStopped || status == StateUnknown || status == StateFailed

	if !dead {
		if status != rec.lastState {
			rec.lastState = status
			persist = true
		}
		// Healthy streak clears the crash-spree counter.
		if rec.attempts > 0 && rec.state == restartStateWatching &&
			(rec.lastRestartAt.IsZero() || now.Sub(rec.lastRestartAt) >= s.stableAfter) {
			rec.attempts = 0
			rec.lastError = ""
			persist = true
		}
		if persist {
			rec.updatedAt = now
			row := *rec
			s.recMu.Unlock()
			s.persist(ctx, vmID, &row)
			return
		}
		s.recMu.Unlock()
		return
	}

	// Process-dead observation.
	rec.lastState = status
	if statusErr != nil {
		rec.lastState = StateUnknown
		rec.lastError = statusErr.Error()
	}
	switch rec.policy {
	case RestartPolicyNo:
		rec.updatedAt = now
		row := *rec
		s.recMu.Unlock()
		s.persist(ctx, vmID, &row)
		return
	}

	// on-failure / always: schedule if not already scheduled.
	if rec.state != restartStateBackingOff {
		if rec.attempts >= restartMaxAttempts {
			rec.state = restartStatePermanentFailure
			rec.nextAttemptAt = time.Time{}
			rec.updatedAt = now
			row := *rec
			s.recMu.Unlock()
			log.Printf("restart-supervisor: VM %s reached %d restart attempts; marked permanent-failure", vmID, restartMaxAttempts)
			s.persist(ctx, vmID, &row)
			return
		}
		rec.state = restartStateBackingOff
		rec.nextAttemptAt = now.Add(restartBackoff(rec.attempts + 1))
		rec.updatedAt = now
		row := *rec
		s.recMu.Unlock()
		s.persist(ctx, vmID, &row)
		return
	}

	if now.Before(rec.nextAttemptAt) {
		s.recMu.Unlock()
		return
	}

	// Restart is due.
	rec.attempts++
	attempt := rec.attempts
	s.recMu.Unlock()

	restartErr := s.restartVM(ctx, vm, driver)

	now = s.now()
	s.recMu.Lock()
	rec = s.records[vmID] // RecordStart/Stop may have swapped state while we worked
	if rec == nil {
		s.recMu.Unlock()
		return
	}
	if rec.state == restartStateStopped || rec.state == restartStatePermanentFailure {
		s.recMu.Unlock()
		return // caller unregistered the VM mid-restart; respect that
	}
	if restartErr != nil {
		rec.lastError = restartErr.Error()
		if attempt >= restartMaxAttempts {
			rec.state = restartStatePermanentFailure
			rec.nextAttemptAt = time.Time{}
			log.Printf("restart-supervisor: VM %s unrecoverable after %d attempts: %v", vmID, attempt, restartErr)
		} else {
			rec.nextAttemptAt = now.Add(restartBackoff(attempt + 1))
		}
	} else {
		rec.state = restartStateWatching
		rec.lastState = StateRunning
		rec.lastError = ""
		rec.lastRestartAt = now
		rec.nextAttemptAt = time.Time{}
		log.Printf("restart-supervisor: restarted crashed VM %s (attempt %d/%d)", vmID, attempt, restartMaxAttempts)
	}
	rec.updatedAt = now
	row := *rec
	s.recMu.Unlock()
	s.persist(ctx, vmID, &row)
}

// restartVM relaunches a dead VM with the exact same VMConfig that created
// it: best-effort Stop first, then Start; if the driver no longer knows the
// VM at all it is re-Create()d from vm.Config() and then started.
func (s *RestartSupervisor) restartVM(ctx context.Context, vm *VM, driver VMDriver) error {
	vmID := vm.ID()
	// Stop is best-effort: the process is already gone for the case we serve,
	// and some drivers report "not running" as an error.
	_ = driver.Stop(ctx, vmID)
	if err := driver.Start(ctx, vmID); err == nil {
		vm.SetState(StateRunning)
		return nil
	} else {
		if _, cerr := driver.Create(ctx, vm.Config()); cerr != nil {
			return fmt.Errorf("start failed (%v) and recreate failed (%v)", err, cerr)
		}
		if err2 := driver.Start(ctx, vmID); err2 != nil {
			return fmt.Errorf("start failed after recreate: %v", err2)
		}
		vm.SetState(StateRunning)
		return nil
	}
}

// restartBackoff returns 2s * fib(attempt): 2s, 2s, 4s, 6s, 10s, 16s, 26s, 42s.
func restartBackoff(attempt int) time.Duration {
	if attempt < 1 {
		attempt = 1
	}
	a, b := 1, 1
	for i := 1; i < attempt; i++ {
		a, b = b, a+b
	}
	return time.Duration(a) * restartBackoffBase
}

// recordLocked returns (creating if absent) the record for vmID, resolving
// its policy from override > VMConfig.RestartPolicy > env default.
// Callers must hold recMu.
func (s *RestartSupervisor) recordLocked(vmID string, ctx context.Context) *vmRestartRecord {
	if rec, ok := s.records[vmID]; ok {
		return rec
	}
	policy := s.defaultPolicy
	if p, ok := s.policyOf[vmID]; ok {
		policy = p
	} else if vm, err := s.manager.GetVM(vmID); err == nil {
		if p, ok := configRestartPolicy(vm.Config()); ok {
			policy = p
		}
	}
	rec := &vmRestartRecord{
		policy:    policy,
		state:     restartStateWatching,
		updatedAt: s.now(),
	}
	s.records[vmID] = rec
	return rec
}

// persist upserts the record into vm_restart_state (memory-only when db is
// nil). A single ON CONFLICT (vm_id) DO UPDATE row per VM — the same row is
// rewritten on every transition instead of stacking double-kill history rows.
func (s *RestartSupervisor) persist(ctx context.Context, vmID string, rec *vmRestartRecord) {
	if s.db == nil {
		return
	}
	var nextAttempt, lastRestart interface{}
	if !rec.nextAttemptAt.IsZero() {
		nextAttempt = rec.nextAttemptAt
	}
	if !rec.lastRestartAt.IsZero() {
		lastRestart = rec.lastRestartAt
	}
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO vm_restart_state
			(vm_id, state, policy, attempts, last_state, last_error, next_attempt_at, last_attempt_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
		ON CONFLICT (vm_id) DO UPDATE SET
			state           = EXCLUDED.state,
			last_state      = EXCLUDED.last_state,
			policy          = EXCLUDED.policy,
			attempts        = EXCLUDED.attempts,
			last_error      = EXCLUDED.last_error,
			next_attempt_at = EXCLUDED.next_attempt_at,
			last_attempt_at = EXCLUDED.last_attempt_at,
			updated_at      = NOW()`,
		vmID, rec.state, string(rec.policy), rec.attempts, string(rec.lastState), rec.lastError,
		nextAttempt, lastRestart)
	if err != nil {
		log.Printf("restart-supervisor: persist vm=%s: %v", vmID, err)
	}
}

// loadPersisted restores persisted rows so backoff/permanent-failure survive
// an api-server restart.
func (s *RestartSupervisor) loadPersisted() error {
	rows, err := s.db.Query(`
		SELECT vm_id, state, policy, attempts, last_state, last_error, next_attempt_at, last_attempt_at
		FROM vm_restart_state`)
	if err != nil {
		return err
	}
	defer rows.Close()
	now := s.now()
	s.recMu.Lock()
	defer s.recMu.Unlock()
	for rows.Next() {
		var (
			id, state, policy, lastState, lastError string
			attempts                                int
			nextAttempt, lastRestart                sql.NullTime
		)
		if err := rows.Scan(&id, &state, &policy, &attempts, &lastState, &lastError, &nextAttempt, &lastRestart); err != nil {
			continue
		}
		st := normalizeRestartPolicy(policy)
		if st == "" {
			st = s.defaultPolicy
		}
		rec := &vmRestartRecord{
			policy:    st,
			state:     state,
			attempts:  attempts,
			lastState: VMState(lastState),
			lastError: lastError,
			updatedAt: now,
		}
		if nextAttempt.Valid {
			rec.nextAttemptAt = nextAttempt.Time
		}
		if lastRestart.Valid {
			rec.lastRestartAt = lastRestart.Time
		}
		s.records[id] = rec
	}
	return rows.Err()
}

func (s *RestartSupervisor) loadPersistedOne(ctx context.Context, vmID string) (*VMRestartStatus, bool) {
	if s.db == nil {
		return nil, false
	}
	var (
		state, policy, lastState, lastError string
		attempts                            int
		nextAttempt, lastRestart            sql.NullTime
		updatedAt                           time.Time
	)
	err := s.db.QueryRowContext(ctx, `
		SELECT state, policy, attempts, COALESCE(last_state, ''), COALESCE(last_error, ''),
		       next_attempt_at, last_attempt_at, updated_at
		FROM vm_restart_state WHERE vm_id = $1`, vmID).
		Scan(&state, &policy, &attempts, &lastState, &lastError, &nextAttempt, &lastRestart, &updatedAt)
	if err != nil {
		return nil, false
	}
	st := VMRestartStatus{
		VMID:        vmID,
		Policy:      RestartPolicy(policy),
		State:       state,
		Attempts:    attempts,
		MaxAttempts: restartMaxAttempts,
		LastState:   VMState(lastState),
		LastError:   lastError,
		UpdatedAt:   updatedAt,
	}
	if nextAttempt.Valid {
		t := nextAttempt.Time
		st.NextAttemptAt = &t
	}
	if lastRestart.Valid {
		t := lastRestart.Time
		st.LastRestartAt = &t
	}
	return &st, true
}
