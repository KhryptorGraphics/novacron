-- Test-database bootstrap for docker-compose.test.yml (mounted as
-- /docker-entrypoint-initdb.d/init.sql). The compose service already creates
-- the novacron_test database itself; this file only installs the extensions
-- the canonical golang-migrate schema (database/migrations/000001_*) depends
-- on. Schema itself is applied afterwards via scripts/migrate.sh (make
-- db-test-setup), so no tables are created here.
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";