#!/bin/bash

# NovaCron Database Optimization Script
# Applies all performance optimizations to the database

set -e

# Configuration
DB_NAME="${DB_NAME:-novacron}"
DB_USER="${DB_USER:-postgres}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to execute SQL file
execute_sql() {
    local sql_file=$1
    local description=$2
    
    print_status "Applying: $description"
    
    if psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f "$sql_file" > /dev/null 2>&1; then
        print_status "✓ $description completed successfully"
    else
        print_error "✗ Failed to apply $description"
        return 1
    fi
}

# Function to check if extension exists
check_extension() {
    local extension=$1
    result=$(psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -tAc "SELECT 1 FROM pg_extension WHERE extname='$extension'" 2>/dev/null)
    if [ "$result" = "1" ]; then
        return 0
    else
        return 1
    fi
}

# Main execution
main() {
    print_status "Starting NovaCron Database Optimization"
    print_status "Database: $DB_NAME@$DB_HOST:$DB_PORT"
    echo ""

    # Check database connection
    print_status "Testing database connection..."
    if ! psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT 1" > /dev/null 2>&1; then
        print_error "Cannot connect to database. Please check your connection settings."
        exit 1
    fi
    print_status "✓ Database connection successful"
    echo ""

    # Backup warning
    print_warning "This script will modify your database schema and create indexes."
    print_warning "It is recommended to backup your database before proceeding."
    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Operation cancelled."
        exit 0
    fi
    echo ""

    # Create migrations directory if it doesn't exist
    if [ ! -d "migrations" ]; then
        print_error "Migrations directory not found. Please run this script from the database directory."
        exit 1
    fi

    # Step 1: Apply performance indexes
    print_status "Step 1/5: Creating performance indexes..."
    execute_sql "migrations/001_performance_indexes.sql" "Performance indexes"
    echo ""

    # Step 2: Create materialized views
    print_status "Step 2/5: Creating materialized views..."
    execute_sql "migrations/002_materialized_views.sql" "Materialized views"
    
    # Refresh materialized views
    print_status "Refreshing materialized views..."
    psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME <<EOF
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_dashboard_stats;
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_vm_listing;
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_node_capacity;
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_alert_summary;
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_user_activity;
EOF
    print_status "✓ Materialized views refreshed"
    echo ""

    # Step 3: Check for TimescaleDB
    print_status "Step 3/5: Checking TimescaleDB extension..."
    if check_extension "timescaledb"; then
        print_status "TimescaleDB is installed. Applying optimizations..."
        execute_sql "migrations/003_timescaledb_optimization.sql" "TimescaleDB optimization"
        
        # Refresh continuous aggregates
        print_status "Refreshing continuous aggregates..."
        psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME <<EOF
            CALL refresh_continuous_aggregate('vm_metrics_5min', NULL, NULL);
            CALL refresh_continuous_aggregate('vm_metrics_hourly_ts', NULL, NULL);
            CALL refresh_continuous_aggregate('node_metrics_5min', NULL, NULL);
EOF
        print_status "✓ Continuous aggregates refreshed"
    else
        print_warning "TimescaleDB is not installed. Skipping time-series optimizations."
        print_warning "To enable TimescaleDB optimizations, install the extension and re-run this script."
    fi
    echo ""

    # Step 4: Update statistics
    print_status "Step 4/5: Updating table statistics..."
    psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME <<EOF
        ANALYZE vms;
        ANALYZE vm_metrics;
        ANALYZE nodes;
        ANALYZE node_metrics;
        ANALYZE users;
        ANALYZE sessions;
        ANALYZE audit_logs;
        ANALYZE alerts;
        ANALYZE migrations;
        ANALYZE storage_volumes;
        ANALYZE network_interfaces;
        ANALYZE snapshots;
        ANALYZE jobs;
EOF
    print_status "✓ Table statistics updated"
    echo ""

    # Step 5: Verify optimizations
    print_status "Step 5/5: Verifying optimizations..."
    
    # Count indexes
    index_count=$(psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -tAc "SELECT COUNT(*) FROM pg_indexes WHERE schemaname='public'" 2>/dev/null)
    print_status "Total indexes created: $index_count"
    
    # Count materialized views
    mv_count=$(psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -tAc "SELECT COUNT(*) FROM pg_matviews WHERE schemaname='public'" 2>/dev/null)
    print_status "Materialized views created: $mv_count"
    
    # Check for unused indexes
    print_status "Checking for unused indexes..."
    unused=$(psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -tAc "SELECT COUNT(*) FROM index_usage WHERE status='UNUSED'" 2>/dev/null)
    if [ "$unused" -gt "0" ]; then
        print_warning "Found $unused unused indexes. Consider reviewing them."
    else
        print_status "✓ All indexes are being used"
    fi
    echo ""

    # Performance test queries
    print_status "Running performance test queries..."
    
    # Test dashboard query
    start_time=$(date +%s%N)
    psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT * FROM mv_dashboard_stats LIMIT 1" > /dev/null 2>&1
    end_time=$(date +%s%N)
    duration=$((($end_time - $start_time) / 1000000))
    print_status "Dashboard query: ${duration}ms"
    
    # Test VM listing query
    start_time=$(date +%s%N)
    psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT * FROM mv_vm_listing LIMIT 100" > /dev/null 2>&1
    end_time=$(date +%s%N)
    duration=$((($end_time - $start_time) / 1000000))
    print_status "VM listing query: ${duration}ms"
    
    echo ""
    print_status "========================================="
    print_status "Database optimization completed successfully!"
    print_status "========================================="
    echo ""
    
    # Recommendations
    print_status "Recommendations:"
    echo "1. Schedule regular VACUUM ANALYZE to maintain performance"
    echo "2. Monitor slow queries using pg_stat_statements"
    echo "3. Set up automated materialized view refresh (pg_cron)"
    echo "4. Configure connection pooling in your application"
    echo "5. Monitor index usage and remove unused indexes periodically"
    echo ""
    
    # Create cron job script
    cat > refresh_views.sh <<'SCRIPT'
#!/bin/bash
# Refresh materialized views - run via cron every minute

psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME <<EOF
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_dashboard_stats;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_vm_listing;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_node_capacity;
EOF
SCRIPT
    
    chmod +x refresh_views.sh
    print_status "Created refresh_views.sh for cron scheduling"
    print_status "Add to crontab: */1 * * * * /path/to/refresh_views.sh"
}

# Run main function
main