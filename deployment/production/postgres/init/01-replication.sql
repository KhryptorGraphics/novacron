-- Initialize replication user and slot for PostgreSQL 15
-- This script runs only on the primary during initial setup

DO $$
BEGIN
    -- Create replication role if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'replicator') THEN
        -- Password will be set by the entrypoint script
        CREATE ROLE replicator WITH REPLICATION LOGIN;
        RAISE NOTICE 'Created replicator role';
    ELSE
        RAISE NOTICE 'replicator role already exists';
    END IF;

    -- Grant necessary permissions
    GRANT CONNECT ON DATABASE postgres TO replicator;
    GRANT CONNECT ON DATABASE novacron TO replicator;

    -- Create replication slot for replica if not exists
    IF NOT EXISTS (SELECT 1 FROM pg_replication_slots WHERE slot_name = 'replica_slot') THEN
        PERFORM pg_create_physical_replication_slot('replica_slot');
        RAISE NOTICE 'Created replication slot: replica_slot';
    ELSE
        RAISE NOTICE 'Replication slot replica_slot already exists';
    END IF;

EXCEPTION
    WHEN duplicate_object THEN
        RAISE NOTICE 'Replication objects already configured';
END $$;

-- Create monitoring function for replication status
CREATE OR REPLACE FUNCTION pg_replication_status()
RETURNS TABLE(
    slot_name text,
    active boolean,
    restart_lsn pg_lsn,
    confirmed_flush_lsn pg_lsn,
    wal_status text,
    safe_wal_size bigint
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        s.slot_name,
        s.active,
        s.restart_lsn,
        s.confirmed_flush_lsn,
        s.wal_status,
        s.safe_wal_size
    FROM pg_replication_slots s;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission to monitoring users
GRANT EXECUTE ON FUNCTION pg_replication_status() TO novacron;