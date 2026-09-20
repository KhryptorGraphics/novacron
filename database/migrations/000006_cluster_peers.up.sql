-- Migration: cluster_peers
-- Created: 2026-09-20
-- Direction: UP
-- Description: Persisted fabric membership for the signed cluster-join
-- protocol (P1/G1). The canonical nodes table is unsuitable: its PK is a
-- UUID and hostname is UNIQUE while the cluster layer keys peers by the
-- free-form NOVACRON_NODE_ID string, and ip_address INET cannot hold a
-- DNS-name peer address. cluster_peers is keyed by the string node id and
-- carries the peer's RPC address, heartbeat freshness, the measured RTT of
-- the last heartbeat probe, and an extensible link-profile JSON blob.

CREATE TABLE cluster_peers (
    node_id        TEXT PRIMARY KEY,
    addr           TEXT NOT NULL,
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    last_rtt_ms    DOUBLE PRECISION,
    link           JSONB DEFAULT '{}',
    created_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
