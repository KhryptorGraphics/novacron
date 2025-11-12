#!/bin/bash

# E2E Test Environment Setup Script
# This script sets up the test environment for E2E tests

set -e

echo "🚀 Setting up E2E test environment..."

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

# Load environment variables
if [ -f "$DOCKER_DIR/.env" ]; then
    echo "📋 Loading environment variables..."
    export $(cat "$DOCKER_DIR/.env" | grep -v '^#' | xargs)
else
    echo "⚠️  No .env file found. Using defaults..."
    if [ -f "$DOCKER_DIR/.env.example" ]; then
        cp "$DOCKER_DIR/.env.example" "$DOCKER_DIR/.env"
        echo "✅ Created .env from .env.example"
    fi
fi

# Check for required tools
check_requirements() {
    echo "🔍 Checking requirements..."

    local missing_tools=()

    if ! command -v node &> /dev/null; then
        missing_tools+=("node")
    fi

    if ! command -v npm &> /dev/null; then
        missing_tools+=("npm")
    fi

    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi

    if ! command -v docker-compose &> /dev/null; then
        missing_tools+=("docker-compose")
    fi

    if [ ${#missing_tools[@]} -ne 0 ]; then
        echo -e "${RED}❌ Missing required tools: ${missing_tools[*]}${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ All requirements met${NC}"
}

# Create necessary directories
create_directories() {
    echo "📁 Creating directories..."

    mkdir -p "$E2E_DIR/test-results"
    mkdir -p "$E2E_DIR/playwright-report"
    mkdir -p "$E2E_DIR/visual/baseline"
    mkdir -p "$E2E_DIR/visual/actual"
    mkdir -p "$E2E_DIR/visual/diff"
    mkdir -p "$E2E_DIR/reports"
    mkdir -p "$E2E_DIR/downloads"
    mkdir -p "$E2E_DIR/videos"
    mkdir -p "$E2E_DIR/traces"

    echo -e "${GREEN}✅ Directories created${NC}"
}

# Install dependencies
install_dependencies() {
    echo "📦 Installing dependencies..."

    if [ ! -d "$PROJECT_ROOT/node_modules" ]; then
        echo "Installing project dependencies..."
        cd "$PROJECT_ROOT"
        npm ci
    fi

    if [ ! -d "$E2E_DIR/node_modules" ]; then
        echo "Installing E2E test dependencies..."
        cd "$E2E_DIR"
        npm ci
    fi

    echo -e "${GREEN}✅ Dependencies installed${NC}"
}

# Install Playwright browsers
install_browsers() {
    echo "🎭 Installing Playwright browsers..."

    cd "$E2E_DIR"
    npx playwright install --with-deps

    echo -e "${GREEN}✅ Browsers installed${NC}"
}

# Setup Docker environment
setup_docker() {
    echo "🐳 Setting up Docker environment..."

    cd "$DOCKER_DIR"

    # Check if Docker is running
    if ! docker info &> /dev/null; then
        echo -e "${RED}❌ Docker is not running${NC}"
        exit 1
    fi

    # Build Docker images
    echo "Building Docker images..."
    docker-compose build

    echo -e "${GREEN}✅ Docker environment ready${NC}"
}

# Initialize database
init_database() {
    echo "🗄️  Initializing database..."

    cd "$DOCKER_DIR"

    # Start only database services
    docker-compose up -d postgres redis

    # Wait for database to be ready
    echo "Waiting for database to be ready..."
    timeout 60 bash -c 'until docker-compose exec -T postgres pg_isready -U test; do sleep 2; done' || {
        echo -e "${RED}❌ Database failed to start${NC}"
        docker-compose logs postgres
        exit 1
    }

    echo -e "${GREEN}✅ Database initialized${NC}"
}

# Run database migrations
run_migrations() {
    echo "🔄 Running database migrations..."

    cd "$PROJECT_ROOT"

    # Run migrations (adjust command based on your migration tool)
    if [ -f "package.json" ] && grep -q "migrate" package.json; then
        npm run migrate
    else
        echo -e "${YELLOW}⚠️  No migration script found${NC}"
    fi

    echo -e "${GREEN}✅ Migrations completed${NC}"
}

# Verify services
verify_services() {
    echo "🔍 Verifying services..."

    cd "$DOCKER_DIR"

    # Check if services are healthy
    local max_attempts=30
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if docker-compose ps | grep -q "healthy"; then
            echo -e "${GREEN}✅ Services are healthy${NC}"
            return 0
        fi

        echo "Waiting for services to be healthy... (attempt $((attempt+1))/$max_attempts)"
        sleep 2
        attempt=$((attempt+1))
    done

    echo -e "${RED}❌ Services failed to become healthy${NC}"
    docker-compose ps
    docker-compose logs
    exit 1
}

# Clean up old test data
cleanup_old_data() {
    echo "🧹 Cleaning up old test data..."

    # Remove old test results
    rm -rf "$E2E_DIR/test-results/"*
    rm -rf "$E2E_DIR/playwright-report/"*
    rm -rf "$E2E_DIR/videos/"*
    rm -rf "$E2E_DIR/traces/"*
    rm -rf "$E2E_DIR/downloads/"*

    echo -e "${GREEN}✅ Cleanup completed${NC}"
}

# Generate test configuration
generate_config() {
    echo "⚙️  Generating test configuration..."

    cat > "$E2E_DIR/.env.test" << EOF
# Auto-generated test configuration
TEST_BASE_URL=http://localhost:${FRONTEND_PORT:-3000}
API_BASE_URL=http://localhost:${BACKEND_PORT:-8080}
DATABASE_URL=${DATABASE_URL:-postgresql://test:test@localhost:5432/test_db}
REDIS_URL=${REDIS_URL:-redis://localhost:6379}
TEST_ENV=${TEST_ENV:-docker}
CI=${CI:-false}
EOF

    echo -e "${GREEN}✅ Configuration generated${NC}"
}

# Main execution
main() {
    echo "========================================="
    echo "  E2E Test Environment Setup"
    echo "========================================="
    echo ""

    check_requirements
    create_directories
    cleanup_old_data
    install_dependencies

    # Only install browsers if not in CI
    if [ "${CI}" != "true" ]; then
        install_browsers
    fi

    setup_docker
    init_database
    run_migrations
    generate_config

    echo ""
    echo "========================================="
    echo -e "${GREEN}✅ E2E test environment setup complete!${NC}"
    echo "========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Start services: cd tests/e2e/docker && docker-compose up -d"
    echo "  2. Run tests: cd tests/e2e && npx playwright test"
    echo "  3. View report: cd tests/e2e && npx playwright show-report"
    echo ""
}

# Handle script arguments
case "${1:-}" in
    --skip-docker)
        SKIP_DOCKER=true
        ;;
    --skip-browsers)
        SKIP_BROWSERS=true
        ;;
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --skip-docker    Skip Docker setup"
        echo "  --skip-browsers  Skip browser installation"
        echo "  --help, -h       Show this help message"
        exit 0
        ;;
esac

main
