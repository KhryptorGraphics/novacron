# PostgreSQL 15 Streaming Replication Setup

This directory contains the configuration and scripts for PostgreSQL 15 streaming replication using the official `postgres:15-alpine` image.

## Architecture

The setup consists of:
- **Primary Server**: Read-write database server
- **Replica Server**: Read-only standby server with streaming replication
- **Replication Method**: PostgreSQL 15 native streaming replication with `standby.signal`

## Key Features

- ✅ PostgreSQL 15 native replication (no obsolete `recovery.conf`)
- ✅ Uses `standby.signal` file for standby mode
- ✅ Automatic failover capability with replication slots
- ✅ Secure password management via Docker secrets
- ✅ Health checks for both primary and replica
- ✅ WAL archiving enabled for point-in-time recovery
- ✅ Monitoring functions for replication status

## Configuration Files

### Primary Configuration

- **postgresql.conf**: Main PostgreSQL configuration with replication settings
  - `wal_level = replica`: Enables replication
  - `max_wal_senders = 10`: Maximum concurrent replication connections
  - `max_replication_slots = 10`: Maximum replication slots
  - `hot_standby = on`: Allows queries on standby servers

- **pg_hba.conf**: Host-based authentication
  - Allows replication connections from Docker network
  - Secure password authentication for replication user

### Initialization Scripts

1. **01-replication.sql**: Creates replication user and slot
   - Creates `replicator` role with replication privileges
   - Creates physical replication slot `replica_slot`
   - Adds monitoring function `pg_replication_status()`

2. **02-set-replication-password.sh**: Sets replication password from Docker secret
   - Reads password from `/run/secrets/replication_password`
   - Sets password for `replicator` user

3. **init-replica.sh**: Initializes replica server
   - Waits for primary to be ready
   - Performs base backup using `pg_basebackup`
   - Creates `standby.signal` file (PostgreSQL 15 requirement)
   - Configures `primary_conninfo` in `postgresql.auto.conf`

### Health Check Scripts

- **healthcheck-primary.sh**: Verifies primary is not in recovery mode
- **healthcheck-replica.sh**: Verifies replica is in recovery mode and checks replication lag

## Docker Compose Configuration

The `docker-compose.production.yml` has been updated to:
1. Remove unsupported `POSTGRES_REPLICATION_*` environment variables
2. Use proper volume mounts for configuration files
3. Execute initialization scripts through docker-entrypoint-initdb.d
4. Configure health checks to verify replication status

## Deployment Instructions

### Prerequisites

1. Create required directories:
```bash
sudo mkdir -p /data/novacron/postgres-primary
sudo mkdir -p /data/novacron/postgres-replica
sudo mkdir -p /etc/novacron/secrets
```

2. Create secret files:
```bash
# Generate strong passwords
openssl rand -base64 32 | sudo tee /etc/novacron/secrets/postgres_password
openssl rand -base64 32 | sudo tee /etc/novacron/secrets/replication_password

# Set proper permissions
sudo chmod 600 /etc/novacron/secrets/*
```

3. Copy configuration files:
```bash
sudo cp -r deployment/production/postgres /etc/novacron/
```

### Starting the Services

```bash
# Start primary first
docker-compose -f deployment/production/docker-compose.production.yml up -d postgres-primary

# Wait for primary to be healthy
docker-compose -f deployment/production/docker-compose.production.yml ps postgres-primary

# Start replica
docker-compose -f deployment/production/docker-compose.production.yml up -d postgres-replica
```

### Verification

1. Check primary status:
```bash
docker exec postgres-primary psql -U postgres -c "SELECT * FROM pg_stat_replication;"
```

2. Check replica status:
```bash
docker exec postgres-replica psql -U postgres -c "SELECT pg_is_in_recovery();"
```

3. Check replication lag:
```bash
docker exec postgres-replica psql -U postgres -c "
SELECT CASE
    WHEN pg_last_wal_receive_lsn() = pg_last_wal_replay_lsn() THEN 'No lag'
    ELSE pg_size_pretty(pg_wal_lsn_diff(pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn()))
END AS replication_lag;"
```

## Monitoring

### Replication Status on Primary
```sql
-- View connected replicas
SELECT client_addr, state, sync_state, replay_lag
FROM pg_stat_replication;

-- Check replication slots
SELECT slot_name, active, restart_lsn, confirmed_flush_lsn
FROM pg_replication_slots;
```

### Replication Status on Replica
```sql
-- Check if in recovery mode
SELECT pg_is_in_recovery();

-- View last received and replayed WAL positions
SELECT pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn();

-- Calculate replication lag in bytes
SELECT pg_wal_lsn_diff(pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn());
```

## Troubleshooting

### Common Issues

1. **Replica fails to connect**: Check network connectivity and pg_hba.conf
2. **Replication lag growing**: Monitor disk I/O and network bandwidth
3. **Base backup fails**: Ensure replication password is correct and primary is accessible

### Logs

View logs for debugging:
```bash
# Primary logs
docker logs postgres-primary

# Replica logs
docker logs postgres-replica

# PostgreSQL logs inside container
docker exec postgres-primary ls -la /var/lib/postgresql/data/log/
```

## Migration from Bitnami

If migrating from Bitnami PostgreSQL:
1. Backup existing data using `pg_dump`
2. Stop Bitnami containers
3. Start new PostgreSQL 15 containers
4. Restore data using `pg_restore`

## Security Considerations

- Passwords are managed via Docker secrets
- Network isolation via Docker networks
- No direct port exposure in production
- Encrypted replication connections supported (configure SSL in postgresql.conf)
- Regular backup strategy recommended

## Performance Tuning

The configuration is optimized for:
- 4 CPU cores and 4GB RAM (primary)
- 2 CPU cores and 2GB RAM (replica)
- SSD storage (`random_page_cost = 1.1`)
- High concurrent connections (200 max)

Adjust `postgresql.conf` parameters based on your workload.

## Alternative: Bitnami PostgreSQL with Repmgr

If you prefer a simpler setup with automatic failover, consider using Bitnami PostgreSQL with repmgr:

```yaml
postgres-primary:
  image: bitnami/postgresql-repmgr:15
  environment:
    - REPMGR_PARTNER_NODES=postgres-primary,postgres-replica
    - REPMGR_NODE_NAME=postgres-primary
    - REPMGR_NODE_NETWORK_NAME=postgres-primary
    - REPMGR_PRIMARY_HOST=postgres-primary
    - REPMGR_PASSWORD_FILE=/run/secrets/repmgr_password
    - POSTGRESQL_POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password
```

This provides automatic failover and simplified configuration but uses Bitnami's custom image.