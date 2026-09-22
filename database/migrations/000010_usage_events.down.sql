-- 000010_usage_events: drop the metering table. Leaves the seeded default
-- organization in place (it may be referenced by users.organization_id rows
-- created after this migration was applied; dropping it could violate FKs).
DROP TABLE IF EXISTS usage_events;
