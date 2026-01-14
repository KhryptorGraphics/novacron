/**
 * NovaCron SDK Constants
 */

// VM States
const VMState = {
  UNKNOWN: 'unknown',
  CREATED: 'created',
  CREATING: 'creating',
  PROVISIONING: 'provisioning',
  RUNNING: 'running',
  STOPPED: 'stopped',
  PAUSED: 'paused',
  PAUSING: 'pausing',
  RESUMING: 'resuming',
  RESTARTING: 'restarting',
  DELETING: 'deleting',
  MIGRATING: 'migrating',
  FAILED: 'failed',
};

// Migration Types
const MigrationType = {
  COLD: 'cold',
  WARM: 'warm',
  LIVE: 'live',
};

// Migration Statuses
const MigrationStatus = {
  PENDING: 'pending',
  RUNNING: 'running',
  COMPLETED: 'completed',
  FAILED: 'failed',
  CANCELLED: 'cancelled',
};

// Node Statuses
const NodeStatus = {
  ONLINE: 'online',
  OFFLINE: 'offline',
  MAINTENANCE: 'maintenance',
};

module.exports = {
  VMState,
  MigrationType,
  MigrationStatus,
  NodeStatus,
};