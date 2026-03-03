#!/bin/bash

# NovaCron Database Seeding Script
# Seeds the database with development/test data

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Database configuration
DB_URL="${DB_URL:-${DATABASE_URL}}"
SEED_DIR="${SEED_DIR:-../seeds}"

# Functions
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Validate database URL
if [ -z "$DB_URL" ]; then
    print_error "Database URL not set. Please set DB_URL or DATABASE_URL environment variable"
    exit 1
fi

# Function to run SQL file
run_sql_file() {
    local file="$1"
    local description="$2"
    
    print_info "Running: $description"
    psql "$DB_URL" -f "$file" > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        print_info "✓ $description completed"
    else
        print_error "✗ $description failed"
        return 1
    fi
}

# Function to check if data exists
check_existing_data() {
    local result=$(psql "$DB_URL" -t -c "SELECT COUNT(*) FROM users WHERE username != 'admin';" 2>/dev/null || echo "0")
    result=$(echo "$result" | tr -d ' ')
    
    if [ "$result" -gt "0" ]; then
        print_warning "Database already contains seed data"
        read -p "Do you want to reset and re-seed? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            print_info "Seeding cancelled"
            exit 0
        fi
        return 0
    fi
    return 1
}

# Main seeding function
seed_database() {
    print_info "Starting database seeding..."
    
    # Check for existing data
    if check_existing_data; then
        print_info "Cleaning existing seed data..."
        run_sql_file "$SEED_DIR/clean.sql" "Clean existing data"
    fi
    
    # Run seed files in order
    print_info "Seeding database with development data..."
    
    # Core data
    run_sql_file "$SEED_DIR/01_organizations.sql" "Organizations"
    run_sql_file "$SEED_DIR/02_users.sql" "Users and authentication"
    run_sql_file "$SEED_DIR/03_nodes.sql" "Compute nodes"
    run_sql_file "$SEED_DIR/04_vms.sql" "Virtual machines"
    run_sql_file "$SEED_DIR/05_metrics.sql" "Sample metrics"
    
    # Optional data
    if [ -f "$SEED_DIR/06_test_data.sql" ]; then
        run_sql_file "$SEED_DIR/06_test_data.sql" "Test data"
    fi
    
    print_info "Database seeding completed successfully!"
    
    # Show summary
    show_summary
}

# Function to show seeded data summary
show_summary() {
    print_info "Seed Data Summary:"
    echo "-------------------"
    
    psql "$DB_URL" -t << EOF
SELECT 'Organizations: ' || COUNT(*) FROM organizations;
SELECT 'Users: ' || COUNT(*) FROM users;
SELECT 'Nodes: ' || COUNT(*) FROM nodes;
SELECT 'VMs: ' || COUNT(*) FROM vms;
SELECT 'Metrics Records: ' || COUNT(*) FROM vm_metrics;
EOF
}

# Show usage
show_usage() {
    cat << EOF
NovaCron Database Seeding Tool

Usage: $0 [command]

Commands:
    seed        Seed the database with development data
    clean       Remove all seed data
    summary     Show current data summary
    help        Show this help message

Environment Variables:
    DB_URL      PostgreSQL connection string

Examples:
    $0 seed     # Seed the database
    $0 clean    # Clean seed data
    $0 summary  # Show data summary

EOF
}

# Parse command
command="${1:-seed}"

case "$command" in
    seed)
        seed_database
        ;;
    clean)
        print_warning "This will remove all seed data!"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            run_sql_file "$SEED_DIR/clean.sql" "Clean seed data"
            print_info "Seed data cleaned"
        fi
        ;;
    summary)
        show_summary
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        print_error "Unknown command: $command"
        show_usage
        exit 1
        ;;
esac