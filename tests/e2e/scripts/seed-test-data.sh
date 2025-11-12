#!/bin/bash

# E2E Test Data Seeding Script
# This script seeds the test database with necessary test data

set -e

echo "🌱 Seeding test data..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
E2E_DIR="$PROJECT_ROOT/tests/e2e"
FIXTURES_DIR="$E2E_DIR/fixtures"

# Database configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${POSTGRES_DB:-test_db}"
DB_USER="${POSTGRES_USER:-test}"
DB_PASSWORD="${POSTGRES_PASSWORD:-test}"

# Load environment variables
if [ -f "$E2E_DIR/docker/.env" ]; then
    export $(cat "$E2E_DIR/docker/.env" | grep -v '^#' | xargs)
fi

# Check database connection
check_database() {
    echo "🔍 Checking database connection..."

    if ! PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c '\q' 2>/dev/null; then
        echo -e "${RED}❌ Cannot connect to database${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ Database connection successful${NC}"
}

# Clear existing test data
clear_test_data() {
    echo "🧹 Clearing existing test data..."

    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME <<-EOSQL
        -- Disable triggers temporarily
        SET session_replication_role = 'replica';

        -- Clear tables in correct order (respecting foreign keys)
        TRUNCATE TABLE sessions CASCADE;
        TRUNCATE TABLE user_roles CASCADE;
        TRUNCATE TABLE permissions CASCADE;
        TRUNCATE TABLE roles CASCADE;
        TRUNCATE TABLE audit_logs CASCADE;
        TRUNCATE TABLE notifications CASCADE;
        TRUNCATE TABLE comments CASCADE;
        TRUNCATE TABLE posts CASCADE;
        TRUNCATE TABLE profiles CASCADE;
        TRUNCATE TABLE users CASCADE;

        -- Reset sequences
        ALTER SEQUENCE users_id_seq RESTART WITH 1;
        ALTER SEQUENCE posts_id_seq RESTART WITH 1;
        ALTER SEQUENCE comments_id_seq RESTART WITH 1;

        -- Re-enable triggers
        SET session_replication_role = 'origin';
EOSQL

    echo -e "${GREEN}✅ Test data cleared${NC}"
}

# Seed users
seed_users() {
    echo "👥 Seeding users..."

    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME <<-EOSQL
        INSERT INTO users (username, email, password_hash, created_at, updated_at) VALUES
        ('admin', 'admin@test.com', '\$2b\$10\$YourHashedPasswordHere', NOW(), NOW()),
        ('testuser1', 'user1@test.com', '\$2b\$10\$YourHashedPasswordHere', NOW(), NOW()),
        ('testuser2', 'user2@test.com', '\$2b\$10\$YourHashedPasswordHere', NOW(), NOW()),
        ('moderator', 'mod@test.com', '\$2b\$10\$YourHashedPasswordHere', NOW(), NOW()),
        ('viewer', 'viewer@test.com', '\$2b\$10\$YourHashedPasswordHere', NOW(), NOW());
EOSQL

    echo -e "${GREEN}✅ Users seeded${NC}"
}

# Seed roles and permissions
seed_roles() {
    echo "🔐 Seeding roles and permissions..."

    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME <<-EOSQL
        -- Insert roles
        INSERT INTO roles (name, description) VALUES
        ('admin', 'Administrator with full access'),
        ('moderator', 'Moderator with limited admin access'),
        ('user', 'Regular user'),
        ('viewer', 'Read-only user');

        -- Insert permissions
        INSERT INTO permissions (name, resource, action) VALUES
        ('user.create', 'user', 'create'),
        ('user.read', 'user', 'read'),
        ('user.update', 'user', 'update'),
        ('user.delete', 'user', 'delete'),
        ('post.create', 'post', 'create'),
        ('post.read', 'post', 'read'),
        ('post.update', 'post', 'update'),
        ('post.delete', 'post', 'delete');

        -- Assign roles to users
        INSERT INTO user_roles (user_id, role_id)
        SELECT u.id, r.id FROM users u, roles r
        WHERE u.username = 'admin' AND r.name = 'admin';

        INSERT INTO user_roles (user_id, role_id)
        SELECT u.id, r.id FROM users u, roles r
        WHERE u.username = 'moderator' AND r.name = 'moderator';

        INSERT INTO user_roles (user_id, role_id)
        SELECT u.id, r.id FROM users u, roles r
        WHERE u.username IN ('testuser1', 'testuser2') AND r.name = 'user';
EOSQL

    echo -e "${GREEN}✅ Roles and permissions seeded${NC}"
}

# Seed posts
seed_posts() {
    echo "📝 Seeding posts..."

    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME <<-EOSQL
        INSERT INTO posts (user_id, title, content, status, created_at, updated_at)
        SELECT
            u.id,
            'Test Post ' || generate_series,
            'This is test content for post ' || generate_series,
            CASE WHEN generate_series % 3 = 0 THEN 'draft' ELSE 'published' END,
            NOW() - (generate_series || ' days')::INTERVAL,
            NOW() - (generate_series || ' days')::INTERVAL
        FROM generate_series(1, 20),
             users u
        WHERE u.username = 'testuser1'
        LIMIT 20;

        -- Add some posts from other users
        INSERT INTO posts (user_id, title, content, status, created_at, updated_at)
        SELECT
            u.id,
            'Admin Post ' || generate_series,
            'Admin content ' || generate_series,
            'published',
            NOW(),
            NOW()
        FROM generate_series(1, 5),
             users u
        WHERE u.username = 'admin'
        LIMIT 5;
EOSQL

    echo -e "${GREEN}✅ Posts seeded${NC}"
}

# Seed comments
seed_comments() {
    echo "💬 Seeding comments..."

    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME <<-EOSQL
        INSERT INTO comments (post_id, user_id, content, created_at)
        SELECT
            p.id,
            u.id,
            'Test comment ' || generate_series || ' on post ' || p.id,
            NOW() - (random() * 100 || ' hours')::INTERVAL
        FROM generate_series(1, 50),
             posts p,
             users u
        WHERE u.username IN ('testuser1', 'testuser2')
        AND p.status = 'published'
        ORDER BY random()
        LIMIT 50;
EOSQL

    echo -e "${GREEN}✅ Comments seeded${NC}"
}

# Seed notifications
seed_notifications() {
    echo "🔔 Seeding notifications..."

    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME <<-EOSQL
        INSERT INTO notifications (user_id, type, title, message, read, created_at)
        SELECT
            u.id,
            CASE (random() * 3)::INT
                WHEN 0 THEN 'comment'
                WHEN 1 THEN 'like'
                WHEN 2 THEN 'mention'
                ELSE 'system'
            END,
            'Test Notification ' || generate_series,
            'This is a test notification message ' || generate_series,
            random() > 0.5,
            NOW() - (random() * 100 || ' hours')::INTERVAL
        FROM generate_series(1, 30),
             users u
        WHERE u.username IN ('testuser1', 'admin')
        LIMIT 30;
EOSQL

    echo -e "${GREEN}✅ Notifications seeded${NC}"
}

# Seed Redis cache data
seed_redis() {
    echo "📦 Seeding Redis cache..."

    REDIS_HOST="${REDIS_HOST:-localhost}"
    REDIS_PORT="${REDIS_PORT:-6379}"

    # Check if Redis is available
    if ! redis-cli -h $REDIS_HOST -p $REDIS_PORT ping > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Redis not available, skipping cache seeding${NC}"
        return
    fi

    # Seed some cache entries
    redis-cli -h $REDIS_HOST -p $REDIS_PORT <<-EOCACHE
        SET session:test:user1 '{"userId":1,"username":"testuser1","role":"user"}'
        EXPIRE session:test:user1 3600

        SET session:test:admin '{"userId":1,"username":"admin","role":"admin"}'
        EXPIRE session:test:admin 3600

        SET cache:posts:popular '[1,2,3,4,5]'
        EXPIRE cache:posts:popular 300

        SET cache:stats:total '{"users":5,"posts":25,"comments":50}'
        EXPIRE cache:stats:total 600
EOCACHE

    echo -e "${GREEN}✅ Redis cache seeded${NC}"
}

# Import fixtures from JSON files
import_fixtures() {
    echo "📥 Importing fixtures..."

    if [ -d "$FIXTURES_DIR" ]; then
        # Import each fixture file
        for fixture in "$FIXTURES_DIR"/*.json; do
            if [ -f "$fixture" ]; then
                echo "Importing $(basename $fixture)..."
                # Your import logic here
                # Example: node scripts/import-fixture.js "$fixture"
            fi
        done
        echo -e "${GREEN}✅ Fixtures imported${NC}"
    else
        echo -e "${YELLOW}⚠️  No fixtures directory found${NC}"
    fi
}

# Verify seeded data
verify_data() {
    echo "🔍 Verifying seeded data..."

    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME <<-EOSQL
        SELECT 'Users' as table_name, COUNT(*) as count FROM users
        UNION ALL
        SELECT 'Posts', COUNT(*) FROM posts
        UNION ALL
        SELECT 'Comments', COUNT(*) FROM comments
        UNION ALL
        SELECT 'Notifications', COUNT(*) FROM notifications
        UNION ALL
        SELECT 'Roles', COUNT(*) FROM roles;
EOSQL

    echo -e "${GREEN}✅ Data verification complete${NC}"
}

# Main execution
main() {
    echo "========================================="
    echo "  E2E Test Data Seeding"
    echo "========================================="
    echo ""

    check_database
    clear_test_data
    seed_users
    seed_roles
    seed_posts
    seed_comments
    seed_notifications
    seed_redis
    import_fixtures
    verify_data

    echo ""
    echo "========================================="
    echo -e "${GREEN}✅ Test data seeding complete!${NC}"
    echo "========================================="
    echo ""
}

# Handle script arguments
case "${1:-}" in
    --clear-only)
        check_database
        clear_test_data
        exit 0
        ;;
    --verify-only)
        check_database
        verify_data
        exit 0
        ;;
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --clear-only   Only clear existing data"
        echo "  --verify-only  Only verify data"
        echo "  --help, -h     Show this help message"
        exit 0
        ;;
esac

main
