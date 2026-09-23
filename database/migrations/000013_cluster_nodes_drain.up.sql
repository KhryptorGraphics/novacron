-- Migration: cluster_nodes_drain
-- Created: 2026-09-22
-- Direction: UP
-- Description: Node drain lifecycle for /api/nodes/{id}/drain. Every cluster
-- node gets a drain_state ('active' | 'draining' | 'drained') that the drain
-- endpoint drives: 'active' -> 'draining' on POST, 'draining' -> 'drained'
-- once the drain coordinator finds no (running|stopped|migrating) VMs left
-- on the node. 'drained' is terminal until an operator sets it back to
-- 'active' (no API for that yet -- deliberate: coming back online is a human
-- decision, not a load-balanced accident).
--
-- cluster_nodes did not exist before this migration: cluster membership lives
-- in cluster_peers (join protocol, see 000006) and NOVACRON_NODE_ID, both
-- string-keyed. This table is therefore created here with the same TEXT
-- node_id key (001's UUID `nodes` table is dead -- see 000009), holding only
-- the columns the drain API owns.

CREATE TABLE IF NOT EXISTS cluster_nodes (
    node_id    TEXT PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

ALTER TABLE cluster_nodes
    ADD COLUMN IF NOT EXISTS drain_state TEXT NOT NULL DEFAULT 'active'
    CHECK (drain_state IN ('active', 'draining', 'drained'));

COMMENT ON TABLE cluster_nodes IS 'Per-node lifecycle state owned by the node-admin API (drain). Membership facts live in cluster_peers; this table only holds operator-driven lifecycle, keyed by the cluster node id string (NOVACRON_NODE_ID).';
COMMENT ON COLUMN cluster_nodes.drain_state IS 'active | draining | drained. draining = a drain is in flight (409 on duplicate POST); drained = every drainable VM left the node.';
