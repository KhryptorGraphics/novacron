-- Migration: cluster_node_identity
-- Created: 2026-09-20
-- Direction: UP
-- Description: Decides ONE coherent cluster-node identity model (novacron-ok7).
--
-- The `nodes` table is UUID-keyed and dead: zero rows, zero INSERT/UPDATE/SELECT
-- anywhere in the canonical api-server binary. The fabric/cluster layer's real,
-- live node identity is the free-form operator-assigned string in
-- cluster_peers.node_id (== NOVACRON_NODE_ID). vms.node_id being a UUID FK to
-- the (always-empty) nodes table meant it could only ever legally be NULL, so
-- neither createVMLocal nor registerMigratedDest ever wrote it -- "which node
-- is this VM on" was not queryable from the schema at all; callers stashed the
-- cluster node id in metadata.cluster_node_id as a workaround instead.
--
-- Decision: cluster node identity is the TEXT string, matching cluster_peers.
-- vms.node_id becomes TEXT (no FK -- cluster membership is dynamic, validated
-- at the API layer against cluster_peers/self, not a static DB FK) and is now
-- actually populated on every local create and every migrated-in registration.
-- migrations.source_node_id/target_node_id get the same treatment for
-- consistency (that table has no live writer yet, but the bead calls out the
-- same class of gap and a future writer should not have to relearn this).
--
-- Owner identity is unchanged in kind (vms.owner_id stays a local UUID FK --
-- local ownership enforcement matters and most VMs are locally owned) but a
-- migrated/cross-node-created VM's real owner, when it does not exist in this
-- node's local users table, was previously stashed in metadata.requested_owner_id
-- JSON with no readers anywhere. Promoted to a real (FK-less, so a foreign UUID
-- from another node's users table can be stored) typed column.

ALTER TABLE vms DROP CONSTRAINT IF EXISTS vms_node_id_fkey;
ALTER TABLE vms ALTER COLUMN node_id TYPE TEXT USING node_id::text;
ALTER TABLE vms ADD COLUMN IF NOT EXISTS requested_owner_id UUID;
COMMENT ON COLUMN vms.node_id IS 'Cluster node id (NOVACRON_NODE_ID / cluster_peers.node_id), not a nodes(id) UUID FK -- see novacron-ok7.';
COMMENT ON COLUMN vms.requested_owner_id IS 'Owner UUID as requested by a cross-node create/migration when it does not exist in this node''s local users table (owner_id is then NULL); preserved for audit/display, never used for local authorization.';

ALTER TABLE migrations DROP CONSTRAINT IF EXISTS migrations_source_node_id_fkey;
ALTER TABLE migrations DROP CONSTRAINT IF EXISTS migrations_target_node_id_fkey;
ALTER TABLE migrations ALTER COLUMN source_node_id TYPE TEXT USING source_node_id::text;
ALTER TABLE migrations ALTER COLUMN target_node_id TYPE TEXT USING target_node_id::text;
