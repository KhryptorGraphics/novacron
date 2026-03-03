#!/bin/bash

# E2E Test Environment Cleanup Script
# This script cleans up the test environment after E2E tests

set -e

echo "🧹 Cleaning up E2E test environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
E2E_DIR="$PROJECT_ROOT/tests/e2e"
DOCKER_DIR="$E2E_DIR/docker"

# Cleanup flags
CLEANUP_DOCKER="${CLEANUP_DOCKER:-true}"
CLEANUP_FILES="${CLEANUP_FILES:-true}"
CLEANUP_CACHE="${CLEANUP_CACHE:-false}"
KEEP_REPORTS="${KEEP_REPORTS:-true}"

# Stop Docker containers
stop_containers() {
    echo "🛑 Stopping Docker containers..."

    cd "$DOCKER_DIR"

    if docker-compose ps | grep -q "Up"; then
        docker-compose stop
        echo -e "${GREEN}✅ Containers stopped${NC}"
    else
        echo -e "${YELLOW}⚠️  No running containers found${NC}"
    fi
}

# Remove Docker containers
remove_containers() {
    echo "🗑️  Removing Docker containers..."

    cd "$DOCKER_DIR"

    docker-compose down -v --remove-orphans

    echo -e "${GREEN}✅ Containers removed${NC}"
}

# Clean Docker volumes
clean_volumes() {
    echo "💾 Cleaning Docker volumes..."

    cd "$DOCKER_DIR"

    # Remove named volumes
    docker-compose down -v

    # Clean up dangling volumes
    if [ "$(docker volume ls -qf dangling=true | wc -l)" -gt 0 ]; then
        docker volume rm $(docker volume ls -qf dangling=true) 2>/dev/null || true
    fi

    echo -e "${GREEN}✅ Volumes cleaned${NC}"
}

# Clean test results
clean_test_results() {
    echo "📊 Cleaning test results..."

    if [ "$KEEP_REPORTS" = "false" ]; then
        rm -rf "$E2E_DIR/test-results/"*
        rm -rf "$E2E_DIR/playwright-report/"*
        echo -e "${GREEN}✅ Test results cleaned${NC}"
    else
        echo -e "${YELLOW}⚠️  Keeping test reports (KEEP_REPORTS=true)${NC}"
    fi
}

# Clean temporary files
clean_temp_files() {
    echo "🗂️  Cleaning temporary files..."

    rm -rf "$E2E_DIR/downloads/"*
    rm -rf "$E2E_DIR/videos/"*
    rm -rf "$E2E_DIR/traces/"*
    rm -rf "$E2E_DIR/.temp/"*

    # Clean screenshots (except baseline)
    rm -rf "$E2E_DIR/visual/actual/"*
    rm -rf "$E2E_DIR/visual/diff/"*

    echo -e "${GREEN}✅ Temporary files cleaned${NC}"
}

# Clean cache
clean_cache() {
    echo "💨 Cleaning cache..."

    rm -rf "$E2E_DIR/.cache"
    rm -rf "$PROJECT_ROOT/.cache"
    rm -rf "$PROJECT_ROOT/.next/cache"
    rm -rf "$PROJECT_ROOT/.turbo"

    echo -e "${GREEN}✅ Cache cleaned${NC}"
}

# Clean database
clean_database() {
    echo "🗄️  Cleaning database..."

    DB_HOST="${DB_HOST:-localhost}"
    DB_PORT="${DB_PORT:-5432}"
    DB_NAME="${POSTGRES_DB:-test_db}"
    DB_USER="${POSTGRES_USER:-test}"
    DB_PASSWORD="${POSTGRES_PASSWORD:-test}"

    # Check if database is accessible
    if PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c '\q' 2>/dev/null; then
        PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME <<-EOSQL
            -- Drop all tables
            DROP SCHEMA public CASCADE;
            CREATE SCHEMA public;
            GRANT ALL ON SCHEMA public TO $DB_USER;
            GRANT ALL ON SCHEMA public TO public;
EOSQL
        echo -e "${GREEN}✅ Database cleaned${NC}"
    else
        echo -e "${YELLOW}⚠️  Database not accessible, skipping${NC}"
    fi
}

# Clean Redis
clean_redis() {
    echo "📦 Cleaning Redis..."

    REDIS_HOST="${REDIS_HOST:-localhost}"
    REDIS_PORT="${REDIS_PORT:-6379}"

    if redis-cli -h $REDIS_HOST -p $REDIS_PORT ping > /dev/null 2>&1; then
        redis-cli -h $REDIS_HOST -p $REDIS_PORT FLUSHALL
        echo -e "${GREEN}✅ Redis cleaned${NC}"
    else
        echo -e "${YELLOW}⚠️  Redis not accessible, skipping${NC}"
    fi
}

# Clean logs
clean_logs() {
    echo "📝 Cleaning logs..."

    rm -rf "$E2E_DIR/logs/"*
    rm -rf "$PROJECT_ROOT/logs/test-"*

    echo -e "${GREEN}✅ Logs cleaned${NC}"
}

# Clean node_modules (optional, aggressive cleanup)
clean_node_modules() {
    echo "📦 Cleaning node_modules..."

    if [ "$AGGRESSIVE_CLEANUP" = "true" ]; then
        rm -rf "$E2E_DIR/node_modules"
        echo -e "${GREEN}✅ node_modules cleaned${NC}"
    else
        echo -e "${YELLOW}⚠️  Skipping node_modules (set AGGRESSIVE_CLEANUP=true to remove)${NC}"
    fi
}

# Generate cleanup report
generate_report() {
    echo "📋 Generating cleanup report..."

    local report_file="$E2E_DIR/cleanup-report-$(date +%Y%m%d-%H%M%S).log"

    cat > "$report_file" << EOF
E2E Test Environment Cleanup Report
====================================
Date: $(date)
User: $(whoami)
Host: $(hostname)

Cleanup Actions:
- Docker containers: ${CLEANUP_DOCKER}
- Test files: ${CLEANUP_FILES}
- Cache: ${CLEANUP_CACHE}
- Reports kept: ${KEEP_REPORTS}

Disk Space:
Before: $DISK_BEFORE
After: $(df -h "$E2E_DIR" | tail -1 | awk '{print $4}')

Status: Complete
EOF

    echo -e "${GREEN}✅ Report generated: $report_file${NC}"
}

# Archive important artifacts before cleanup
archive_artifacts() {
    if [ "$ARCHIVE_BEFORE_CLEANUP" = "true" ]; then
        echo "📦 Archiving artifacts..."

        local archive_dir="$E2E_DIR/archives"
        local archive_name="test-artifacts-$(date +%Y%m%d-%H%M%S).tar.gz"

        mkdir -p "$archive_dir"

        tar -czf "$archive_dir/$archive_name" \
            -C "$E2E_DIR" \
            playwright-report \
            test-results \
            visual \
            2>/dev/null || true

        echo -e "${GREEN}✅ Artifacts archived: $archive_name${NC}"
    fi
}

# Verify cleanup
verify_cleanup() {
    echo "🔍 Verifying cleanup..."

    local issues=0

    # Check if containers are still running
    if docker ps | grep -q "e2e-"; then
        echo -e "${RED}❌ E2E containers still running${NC}"
        issues=$((issues + 1))
    fi

    # Check disk space
    local disk_after=$(df -h "$E2E_DIR" | tail -1 | awk '{print $4}')
    echo "Disk space available: $disk_after"

    if [ $issues -eq 0 ]; then
        echo -e "${GREEN}✅ Cleanup verification passed${NC}"
    else
        echo -e "${YELLOW}⚠️  Found $issues issue(s) during verification${NC}"
    fi
}

# Main execution
main() {
    echo "========================================="
    echo "  E2E Test Environment Cleanup"
    echo "========================================="
    echo ""

    # Save disk space before cleanup
    DISK_BEFORE=$(df -h "$E2E_DIR" | tail -1 | awk '{print $4}')
    echo "Disk space before cleanup: $DISK_BEFORE"
    echo ""

    # Archive artifacts if requested
    archive_artifacts

    # Stop services first
    if [ "$CLEANUP_DOCKER" = "true" ]; then
        stop_containers
        remove_containers
        clean_volumes
    fi

    # Clean database and cache
    if [ "$CLEANUP_FILES" = "true" ]; then
        clean_database
        clean_redis
        clean_test_results
        clean_temp_files
        clean_logs
    fi

    # Clean cache if requested
    if [ "$CLEANUP_CACHE" = "true" ]; then
        clean_cache
    fi

    # Aggressive cleanup
    clean_node_modules

    # Verify and report
    verify_cleanup
    generate_report

    echo ""
    echo "========================================="
    echo -e "${GREEN}✅ E2E test environment cleanup complete!${NC}"
    echo "========================================="
    echo ""
}

# Handle script arguments
case "${1:-}" in
    --docker-only)
        CLEANUP_FILES=false
        ;;
    --files-only)
        CLEANUP_DOCKER=false
        ;;
    --aggressive)
        AGGRESSIVE_CLEANUP=true
        CLEANUP_CACHE=true
        KEEP_REPORTS=false
        ;;
    --archive)
        ARCHIVE_BEFORE_CLEANUP=true
        ;;
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --docker-only    Only cleanup Docker resources"
        echo "  --files-only     Only cleanup files and data"
        echo "  --aggressive     Aggressive cleanup (removes cache, reports, node_modules)"
        echo "  --archive        Archive artifacts before cleanup"
        echo "  --help, -h       Show this help message"
        echo ""
        echo "Environment variables:"
        echo "  CLEANUP_DOCKER=true/false          (default: true)"
        echo "  CLEANUP_FILES=true/false           (default: true)"
        echo "  CLEANUP_CACHE=true/false           (default: false)"
        echo "  KEEP_REPORTS=true/false            (default: true)"
        echo "  AGGRESSIVE_CLEANUP=true/false      (default: false)"
        echo "  ARCHIVE_BEFORE_CLEANUP=true/false  (default: false)"
        exit 0
        ;;
esac

main
