-- 000010_usage_events: persisted metering foundation for usage-based billing.
--
-- Product rationale (research/profitability/*, 2026-09-21): the recommended
-- revenue model is usage-metered utility on top of the fabric's EXISTING
-- telemetry — the same telemetry the scheduler uses for placement decisions
-- (bytes moved per transfer, job durations, vCPU allocations) is exactly the
-- telemetry needed to bill for it, at zero additional instrumentation cost.
-- This table is the durable store for those events.
--
-- Design constraints honored here:
--   * No fabricated revenue math — this table records MEASURED resource
--     consumption only. Pricing/rating lives in the API layer as an explicit,
--     operator-configurable rate card, never in the data.
--   * organization_id is NOT NULL with a seeded default org so attribution
--     works from the first event, before full tenant isolation lands.
--   * user_id/vm_id/job_id are soft pointers (no FK on job_id: fabric job ids
--     are not users/vms rows and must survive job-row deletion for audit).
CREATE TABLE IF NOT EXISTS usage_events (
    id              BIGSERIAL PRIMARY KEY,
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE RESTRICT,
    user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
    event_type      TEXT NOT NULL CHECK (event_type IN ('egress_bytes', 'migration', 'job_seconds', 'vcpu_seconds')),
    vm_id           UUID,
    job_id          UUID,
    source_node_id  TEXT,
    target_node_id  TEXT,
    quantity        NUMERIC(24,6) NOT NULL CHECK (quantity >= 0),
    unit            TEXT NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    occurred_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- The hot query is "aggregate by org over a time range" (billing summary).
CREATE INDEX idx_usage_events_org_time ON usage_events(organization_id, occurred_at);
CREATE INDEX idx_usage_events_type ON usage_events(event_type, occurred_at);

-- Default organization: attribution works from the first event, before
-- per-tenant isolation is fully enforced. Non-null org on every event keeps
-- billing queries trivially GROUP BY-able.
INSERT INTO organizations (id, name, slug)
VALUES ('00000000-0000-0000-0000-000000000001', 'Default Organization', 'default')
ON CONFLICT (id) DO NOTHING;
