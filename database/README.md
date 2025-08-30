# NovaCron Database Migration System

## Overview

NovaCron uses a comprehensive database migration system built on golang-migrate to manage PostgreSQL schema changes. This system provides version control for database schemas, automated migration execution, rollback capabilities, and seed data management for development environments.

## Directory Structure

```
database/
├── migrations/          # SQL migration files (versioned)
│   ├── 000001_init_schema.up.sql
│   ├── 000001_init_schema.down.sql
│   └── ...
├── seeds/              # Development seed data
│   ├── 01_organizations.sql
│   ├── 02_users.sql
│   ├── 03_nodes.sql
│   ├── 04_vms.sql
│   ├── 05_metrics.sql
│   └── clean.sql
├── scripts/            # Migration and seed utilities
│   ├── migrate.sh      # Migration management script
│   └── seed.sh         # Seed data management script
├── migrate.go          # Go migration tool (embedded migrations)
└── README.md          # This file
```

## Prerequisites

- PostgreSQL 14+ installed and running
- Go 1.19+ (for using the Go migration tool)
- golang-migrate CLI tool (automatically installed by scripts if missing)
- psql command-line tool (for database operations)

## Quick Start

### 1. Set Database URL

```bash
export DB_URL="postgres://username:password@localhost:5432/novacron?sslmode=disable"
```

Or use the default Docker Compose database:
```bash
export DB_URL="postgres://postgres:postgres@localhost:5432/novacron?sslmode=disable"
```

### 2. Run Migrations

```bash
# Using Makefile (recommended)
make db-migrate

# Or using the script directly
cd database
./scripts/migrate.sh up
```

### 3. Seed Development Data

```bash
# Using Makefile
make db-seed

# Or using the script directly
cd database
./scripts/seed.sh seed
```

## Makefile Targets

The project includes convenient Makefile targets for all database operations:

| Target | Description |
|--------|-------------|
| `make db-migrate` | Run all pending migrations |
| `make db-rollback` | Rollback the last migration |
| `make db-migrate-create` | Create a new migration |
| `make db-version` | Show current migration version |
| `make db-status` | Show migration status |
| `make db-seed` | Seed database with development data |
| `make db-clean` | Remove all seed data |
| `make db-reset` | Drop, migrate, and seed database |
| `make db-test-setup` | Setup test database |
| `make db-test-clean` | Clean test database |
| `make db-validate` | Validate migration files |
| `make db-console` | Open PostgreSQL console |
| `make db-backup` | Backup database to file |
| `make db-restore` | Restore database from backup |

## Migration Management

### Creating New Migrations

```bash
# Using Makefile
make db-migrate-create
# Enter migration name when prompted

# Using script
./database/scripts/migrate.sh create add_new_feature

# Using Go tool
go run database/migrate.go -create add_new_feature
```

This creates two files:
- `YYYYMMDDHHMMSS_add_new_feature.up.sql` - Forward migration
- `YYYYMMDDHHMMSS_add_new_feature.down.sql` - Rollback migration

### Running Migrations

```bash
# Run all pending migrations
./database/scripts/migrate.sh up

# Run specific number of migrations
./database/scripts/migrate.sh up -n 2

# Rollback last migration
./database/scripts/migrate.sh down

# Rollback specific number of migrations
./database/scripts/migrate.sh down -n 3
```

### Checking Migration Status

```bash
# Show current version
./database/scripts/migrate.sh version

# Show detailed status
./database/scripts/migrate.sh status

# Validate migration files
./database/scripts/migrate.sh validate
```

## Database Schema

The migration system manages the following core tables:

### Core Tables
- **organizations** - Multi-tenant organization management
- **users** - User accounts with RBAC
- **nodes** - Compute nodes for VM hosting
- **vms** - Virtual machine definitions
- **migrations** - VM migration tracking
- **alerts** - System alerts and notifications

### Metrics Tables
- **vm_metrics** - Time-series VM performance data
- **node_metrics** - Time-series node performance data

### Supporting Tables
- **sessions** - User session management
- **audit_logs** - Audit trail for all operations
- **api_keys** - API key management
- **jobs** - Async job queue
- **snapshots** - VM snapshot management
- **storage_volumes** - Storage volume tracking
- **network_interfaces** - Network configuration

## Seed Data

The seed data provides a complete development environment with:

### Organizations (5)
- NovaCron Inc (main organization)
- Acme Corporation
- TechStart
- Enterprise Co
- Dev Team

### Users (9)
- 2 Admin users
- 2 Operator users
- 2 Viewer users
- 3 Test users (various states)

### Infrastructure
- 9 Nodes (production, dev, test, edge)
- 10 VMs (various states and configurations)
- Sample metrics for the last 24 hours
- Network interfaces and storage volumes
- VM snapshots

### Default Credentials
- Admin username: `admin`
- Admin password: `Password123!`
- All test users use password: `Password123!`

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_URL` | PostgreSQL connection string | - |
| `DATABASE_URL` | Alternative to DB_URL | - |
| `MIGRATIONS_DIR` | Path to migrations directory | `./migrations` |
| `SEED_DIR` | Path to seed directory | `./seeds` |

## Testing

### Setup Test Database

```bash
# Create and setup test database
make db-test-setup

# Run tests
make test-integration

# Clean test database
make db-test-clean
```

### Test Database URL

```bash
export DB_TEST_URL="postgres://postgres:postgres@localhost:5432/novacron_test?sslmode=disable"
```

## Production Deployment

### 1. Migration Strategy

For production deployments:

1. **Always backup before migrations**:
   ```bash
   make db-backup
   ```

2. **Test migrations in staging first**:
   ```bash
   # On staging environment
   make db-migrate
   ```

3. **Use transactions for safety**:
   - Migrations are wrapped in transactions by default
   - Failed migrations automatically rollback

4. **Monitor migration execution**:
   ```bash
   make db-status
   ```

### 2. Zero-Downtime Migrations

For zero-downtime deployments:

1. **Make additive changes first**:
   - Add new columns as nullable
   - Add new tables without foreign keys
   - Deploy application code that works with both schemas

2. **Migrate data**:
   - Backfill new columns
   - Update application to use new schema

3. **Clean up old schema**:
   - Remove old columns
   - Add constraints
   - Add foreign keys

### 3. Rollback Procedures

If a migration fails:

```bash
# Check current version
make db-version

# Rollback last migration
make db-rollback

# Or force to specific version
./database/scripts/migrate.sh force <version>
```

## Docker Integration

### Using Docker Compose

```bash
# Start PostgreSQL
docker-compose up -d postgres

# Run migrations
docker-compose run --rm api make db-migrate

# Seed data
docker-compose run --rm api make db-seed
```

### Building Migration Container

```dockerfile
FROM golang:1.19-alpine
RUN apk add --no-cache postgresql-client
RUN go install -tags 'postgres' github.com/golang-migrate/migrate/v4/cmd/migrate@latest
WORKDIR /app
COPY database/ ./database/
CMD ["make", "db-migrate"]
```

## Troubleshooting

### Common Issues

1. **Migration stuck or dirty**:
   ```bash
   # Check version
   make db-version
   # Force to clean version
   ./database/scripts/migrate.sh force <version>
   ```

2. **Connection refused**:
   - Verify PostgreSQL is running
   - Check connection string format
   - Ensure network connectivity

3. **Permission denied**:
   - Check database user permissions
   - Ensure user can create/modify schemas

4. **Migration file validation errors**:
   ```bash
   make db-validate
   ```

### Debug Mode

Enable debug output:
```bash
DEBUG=1 ./database/scripts/migrate.sh up
```

## Best Practices

1. **Always write down migrations**:
   - Every UP migration must have a corresponding DOWN
   - Test both directions

2. **Keep migrations small**:
   - One logical change per migration
   - Easier to debug and rollback

3. **Use transactions**:
   - Wrap DDL operations in transactions
   - Ensure atomicity

4. **Version control**:
   - Commit migration files to git
   - Never modify existing migrations
   - Create new migrations for changes

5. **Test thoroughly**:
   - Test migrations on copy of production data
   - Verify rollback procedures
   - Check performance impact

6. **Document changes**:
   - Add comments in migration files
   - Update this README for major changes
   - Document breaking changes

## Advanced Usage

### Custom Migration Directory

```bash
MIGRATIONS_DIR=/custom/path ./scripts/migrate.sh up
```

### Using Go Migration Tool

The project includes a Go-based migration tool with embedded migrations:

```bash
# Build the tool
go build -o migrate database/migrate.go

# Run migrations
./migrate -db "$DB_URL" -direction up

# Create new migration
./migrate -create feature_name
```

### CI/CD Integration

Example GitHub Actions workflow:

```yaml
- name: Run migrations
  env:
    DB_URL: ${{ secrets.DB_URL }}
  run: |
    make db-migrate
    make db-validate
```

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review migration logs
3. Consult the PostgreSQL documentation
4. Contact the development team

## License

This migration system is part of the NovaCron project and follows the same license terms.