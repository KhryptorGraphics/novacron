-- 000012 down: drop the crash-restart supervision state table.
DROP INDEX IF EXISTS idx_vm_restart_state_due;
DROP TABLE IF EXISTS vm_restart_state;
