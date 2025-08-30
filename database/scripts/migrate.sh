#!/bin/bash

# NovaCron Database Migration Script
# Usage: ./migrate.sh [command] [options]

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
DB_URL="${DB_URL:-${DATABASE_URL}}"
MIGRATIONS_DIR="${MIGRATIONS_DIR:-./migrations}"
MIGRATE_CMD="migrate"

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if migrate tool is installed
check_migrate_tool() {
    if ! command -v migrate &> /dev/null; then
        print_error "migrate tool is not installed"
        print_info "Installing golang-migrate..."
        go install -tags 'postgres' github.com/golang-migrate/migrate/v4/cmd/migrate@latest
    fi
}

# Function to validate database URL
validate_db_url() {
    if [ -z "$DB_URL" ]; then
        print_error "Database URL not set. Please set DB_URL or DATABASE_URL environment variable"
        print_info "Example: export DB_URL='postgres://user:password@localhost:5432/novacron?sslmode=disable'"
        exit 1
    fi
}

# Function to show usage
show_usage() {
    cat << EOF
NovaCron Database Migration Tool

Usage: $0 [command] [options]

Commands:
    up              Run all pending migrations
    down            Rollback the last migration
    down-all        Rollback all migrations
    create NAME     Create a new migration with the given name
    version         Show current migration version
    force VERSION   Force database to specific version
    drop            Drop everything in the database
    status          Show migration status
    validate        Validate migration files

Options:
    -n, --steps N   Number of migrations to run (for up/down commands)
    -d, --db URL    Database URL (overrides environment variable)
    -h, --help      Show this help message

Environment Variables:
    DB_URL          PostgreSQL connection string
    DATABASE_URL    Alternative to DB_URL

Examples:
    $0 up                     # Run all pending migrations
    $0 up -n 1                # Run next migration
    $0 down                   # Rollback last migration
    $0 down -n 2              # Rollback last 2 migrations
    $0 create add_users       # Create new migration files
    $0 version                # Show current version
    $0 status                 # Show migration status

EOF
}

# Function to run migrations up
migrate_up() {
    local steps="${1:-}"
    print_info "Running migrations up..."
    
    if [ -n "$steps" ]; then
        $MIGRATE_CMD -database "$DB_URL" -path "$MIGRATIONS_DIR" up "$steps"
    else
        $MIGRATE_CMD -database "$DB_URL" -path "$MIGRATIONS_DIR" up
    fi
    
    if [ $? -eq 0 ]; then
        print_info "Migrations completed successfully"
        show_version
    else
        print_error "Migration failed"
        exit 1
    fi
}

# Function to run migrations down
migrate_down() {
    local steps="${1:-1}"
    print_info "Rolling back $steps migration(s)..."
    
    $MIGRATE_CMD -database "$DB_URL" -path "$MIGRATIONS_DIR" down "$steps"
    
    if [ $? -eq 0 ]; then
        print_info "Rollback completed successfully"
        show_version
    else
        print_error "Rollback failed"
        exit 1
    fi
}

# Function to create a new migration
create_migration() {
    local name="$1"
    if [ -z "$name" ]; then
        print_error "Migration name is required"
        exit 1
    fi
    
    # Create timestamp
    timestamp=$(date +%Y%m%d%H%M%S)
    
    # Create migration files
    up_file="$MIGRATIONS_DIR/${timestamp}_${name}.up.sql"
    down_file="$MIGRATIONS_DIR/${timestamp}_${name}.down.sql"
    
    # Create up migration
    cat > "$up_file" << EOF
-- Migration: $name
-- Created: $(date -Iseconds)
-- Direction: UP

-- Add your UP migration SQL here

EOF
    
    # Create down migration
    cat > "$down_file" << EOF
-- Migration: $name
-- Created: $(date -Iseconds)
-- Direction: DOWN

-- Add your DOWN migration SQL here

EOF
    
    print_info "Created migration files:"
    echo "  - $up_file"
    echo "  - $down_file"
}

# Function to show current version
show_version() {
    print_info "Current migration version:"
    $MIGRATE_CMD -database "$DB_URL" -path "$MIGRATIONS_DIR" version 2>/dev/null || echo "No version set"
}

# Function to force version
force_version() {
    local version="$1"
    if [ -z "$version" ]; then
        print_error "Version number is required"
        exit 1
    fi
    
    print_warning "Forcing database to version $version..."
    $MIGRATE_CMD -database "$DB_URL" -path "$MIGRATIONS_DIR" force "$version"
    
    if [ $? -eq 0 ]; then
        print_info "Database forced to version $version"
    else
        print_error "Failed to force version"
        exit 1
    fi
}

# Function to drop database
drop_database() {
    print_warning "This will drop all tables and data in the database!"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "$confirm" = "yes" ]; then
        print_info "Dropping database..."
        $MIGRATE_CMD -database "$DB_URL" -path "$MIGRATIONS_DIR" drop -f
        print_info "Database dropped"
    else
        print_info "Operation cancelled"
    fi
}

# Function to show migration status
show_status() {
    print_info "Migration Status:"
    echo "-------------------"
    
    # Show current version
    current_version=$($MIGRATE_CMD -database "$DB_URL" -path "$MIGRATIONS_DIR" version 2>&1 | grep -oE '[0-9]+' | head -1 || echo "0")
    echo "Current Version: $current_version"
    
    # Count migration files
    total_migrations=$(ls -1 "$MIGRATIONS_DIR"/*.up.sql 2>/dev/null | wc -l)
    echo "Total Migrations: $total_migrations"
    
    # List pending migrations
    echo ""
    print_info "Migration Files:"
    for file in "$MIGRATIONS_DIR"/*.up.sql; do
        if [ -f "$file" ]; then
            basename "$file" | sed 's/.up.sql$//'
        fi
    done | sort -n
}

# Function to validate migration files
validate_migrations() {
    print_info "Validating migration files..."
    
    local errors=0
    
    # Check for matching up/down files
    for up_file in "$MIGRATIONS_DIR"/*.up.sql; do
        if [ -f "$up_file" ]; then
            down_file="${up_file%.up.sql}.down.sql"
            if [ ! -f "$down_file" ]; then
                print_error "Missing down migration for: $(basename "$up_file")"
                errors=$((errors + 1))
            fi
        fi
    done
    
    # Check for orphaned down files
    for down_file in "$MIGRATIONS_DIR"/*.down.sql; do
        if [ -f "$down_file" ]; then
            up_file="${down_file%.down.sql}.up.sql"
            if [ ! -f "$up_file" ]; then
                print_error "Orphaned down migration: $(basename "$down_file")"
                errors=$((errors + 1))
            fi
        fi
    done
    
    # Check file naming convention
    for file in "$MIGRATIONS_DIR"/*.sql; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            if ! echo "$filename" | grep -qE '^[0-9]{6,}_.*\.(up|down)\.sql$'; then
                print_warning "File doesn't follow naming convention: $filename"
            fi
        fi
    done
    
    if [ $errors -eq 0 ]; then
        print_info "All migration files are valid"
    else
        print_error "Found $errors validation error(s)"
        exit 1
    fi
}

# Main script logic
main() {
    # Parse command
    command="${1:-}"
    shift || true
    
    # Parse options
    steps=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--steps)
                steps="$2"
                shift 2
                ;;
            -d|--db)
                DB_URL="$2"
                shift 2
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                if [ "$command" = "create" ] || [ "$command" = "force" ]; then
                    # These commands take an argument
                    arg="$1"
                    shift
                else
                    print_error "Unknown option: $1"
                    show_usage
                    exit 1
                fi
                ;;
        esac
    done
    
    # Execute command
    case "$command" in
        up)
            check_migrate_tool
            validate_db_url
            migrate_up "$steps"
            ;;
        down)
            check_migrate_tool
            validate_db_url
            migrate_down "$steps"
            ;;
        down-all)
            check_migrate_tool
            validate_db_url
            print_warning "This will rollback ALL migrations!"
            read -p "Are you sure? (yes/no): " confirm
            if [ "$confirm" = "yes" ]; then
                $MIGRATE_CMD -database "$DB_URL" -path "$MIGRATIONS_DIR" down -all
                print_info "All migrations rolled back"
            fi
            ;;
        create)
            create_migration "$arg"
            ;;
        version)
            check_migrate_tool
            validate_db_url
            show_version
            ;;
        force)
            check_migrate_tool
            validate_db_url
            force_version "$arg"
            ;;
        drop)
            check_migrate_tool
            validate_db_url
            drop_database
            ;;
        status)
            validate_db_url
            show_status
            ;;
        validate)
            validate_migrations
            ;;
        ""|help)
            show_usage
            ;;
        *)
            print_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"