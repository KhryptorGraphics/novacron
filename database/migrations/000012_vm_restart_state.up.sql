-- 000012: vm_restart_state — crash-restart supervision state for Compute VM
-- drivers (backend/core/vm/restart_supervisor.go).
--
-- One row per VM, rewritten in place on every transition. All supervisor
-- writes go through exactly one statement shape:
--
--   INSERT INTO vm_restart_state (vm_id, state, policy, attempts, last_state,
--                                 last_error, next_attempt_at, last_attempt_at)
--   VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
--   ON CONFLICT (vm_id) DO UPDATE SET
--       state           = EXCLUDED.state,
--       last_state      = EXCLUDED.last_state,
--       policy          = EXCLUDED.policy,
--       attempts        = EXCLUDED.attempts,
--       last_error      = EXCLUDED.last_error,
--       next_attempt_at = EXCLUDED.next_attempt_at,
--       last_attempt_at = EXCLUDED.last_attempt_at,
--       updated_at      = NOW();
--
-- The ON CONFLICT (vm_id) DO UPDATE upsert rewrites the existing row instead
-- of appending double-kill history rows for a crash-looping VM.
CREATE TABLE IF NOT EXISTS vm_restart_state (
    vm_id           TEXT PRIMARY KEY,
    state           TEXT NOT NULL DEFAULT 'watching'
                    CHECK (state IN ('watching', 'backing-off', 'stopped', 'permanent-failure')),
    policy          TEXT NOT NULL DEFAULT 'on-failure'
                    CHECK (policy IN ('no', 'on-failure', 'always')),
    attempts        INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    last_state      TEXT NOT NULL DEFAULT '',
    last_error      TEXT NOT NULL DEFAULT '',
    next_attempt_at TIMESTAMP WITH TIME ZONE,
    last_attempt_at TIMESTAMP WITH TIME ZONE,
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Fast lookup of rows still owed a restart attempt (backing-off) for the
-- daemon-restart recovery path.
CREATE INDEX IF NOT EXISTS idx_vm_restart_state_due
    ON vm_restart_state (next_attempt_at)
    WHERE state = 'backing-off';
