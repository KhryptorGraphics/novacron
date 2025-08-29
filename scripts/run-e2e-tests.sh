#!/bin/bash

# Comprehensive E2E Test Runner for NovaCron
# Runs all Puppeteer E2E tests with proper setup and teardown

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
FRONTEND_URL="http://localhost:8092"
API_URL="http://localhost:8090"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
BACKEND_DIR="$PROJECT_ROOT/backend"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"

# Test categories
declare -A TEST_CATEGORIES=(
    ["auth"]="Authentication and Security Tests"
    ["vm-management"]="VM Lifecycle and Management Tests"
    ["monitoring"]="Dashboard and Monitoring Tests"
    ["admin"]="Admin Panel and System Management Tests"
    ["performance"]="Performance and Optimization Tests"
    ["accessibility"]="Accessibility and WCAG Compliance Tests"
    ["integration"]="Backend Integration and API Tests"
)

# Default settings
RUN_ALL=false
HEADLESS=true
SLOW_MO=0
DEBUG=false
COVERAGE=false
PARALLEL=false
CATEGORIES=()
SETUP_SERVERS=true
CLEANUP=true

print_header() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                    NovaCron E2E Test Suite                   ║${NC}"
    echo -e "${CYAN}║                  Comprehensive Testing Framework            ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_status() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

print_test_category() {
    echo -e "${PURPLE}[CATEGORY] $1${NC}"
}

show_usage() {
    echo -e "${CYAN}Usage: $0 [OPTIONS]${NC}"
    echo ""
    echo -e "${YELLOW}Test Categories:${NC}"
    for category in "${!TEST_CATEGORIES[@]}"; do
        echo -e "  ${GREEN}$category${NC} - ${TEST_CATEGORIES[$category]}"
    done
    echo ""
    echo -e "${YELLOW}Options:${NC}"
    echo -e "  ${GREEN}--all${NC}                Run all test categories"
    echo -e "  ${GREEN}--category CATEGORY${NC}   Run specific test category (can be repeated)"
    echo -e "  ${GREEN}--headless${NC}           Run in headless mode (default: true)"
    echo -e "  ${GREEN}--headed${NC}             Run with visible browser"
    echo -e "  ${GREEN}--slow-mo MS${NC}         Add delay between actions (default: 0)"
    echo -e "  ${GREEN}--debug${NC}              Enable debug logging"
    echo -e "  ${GREEN}--coverage${NC}           Generate coverage report"
    echo -e "  ${GREEN}--parallel${NC}           Run tests in parallel (experimental)"
    echo -e "  ${GREEN}--no-setup${NC}           Skip server setup (servers must be running)"
    echo -e "  ${GREEN}--no-cleanup${NC}         Skip cleanup after tests"
    echo -e "  ${GREEN}--help${NC}               Show this help message"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo -e "  ${GREEN}$0 --all${NC}                                    # Run all tests"
    echo -e "  ${GREEN}$0 --category auth --category vm-management${NC}  # Run specific categories"
    echo -e "  ${GREEN}$0 --headed --slow-mo 500 --debug${NC}           # Debug mode with visible browser"
    echo -e "  ${GREEN}$0 --coverage --parallel${NC}                    # Performance testing with coverage"
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --all)
                RUN_ALL=true
                shift
                ;;
            --category)
                if [[ -n "$2" ]] && [[ "${!TEST_CATEGORIES[@]}" =~ "$2" ]]; then
                    CATEGORIES+=("$2")
                    shift 2
                else
                    print_error "Invalid category: $2"
                    echo "Valid categories: ${!TEST_CATEGORIES[@]}"
                    exit 1
                fi
                ;;
            --headless)
                HEADLESS=true
                shift
                ;;
            --headed)
                HEADLESS=false
                shift
                ;;
            --slow-mo)
                SLOW_MO="$2"
                shift 2
                ;;
            --debug)
                DEBUG=true
                HEADLESS=false
                shift
                ;;
            --coverage)
                COVERAGE=true
                shift
                ;;
            --parallel)
                PARALLEL=true
                shift
                ;;
            --no-setup)
                SETUP_SERVERS=false
                shift
                ;;
            --no-cleanup)
                CLEANUP=false
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # Set default categories if none specified and not running all
    if [[ ${#CATEGORIES[@]} -eq 0 ]] && [[ $RUN_ALL == false ]]; then
        print_warning "No categories specified. Use --all to run all tests or --category to specify categories."
        show_usage
        exit 1
    fi

    # If --all is specified, run all categories
    if [[ $RUN_ALL == true ]]; then
        CATEGORIES=(${!TEST_CATEGORIES[@]})
    fi
}

check_dependencies() {
    print_status "Checking dependencies..."

    # Check Node.js and npm
    if ! command -v node &> /dev/null; then
        print_error "Node.js is required but not installed"
        exit 1
    fi

    if ! command -v npm &> /dev/null; then
        print_error "npm is required but not installed"
        exit 1
    fi

    # Check Go (for backend)
    if ! command -v go &> /dev/null && [[ $SETUP_SERVERS == true ]]; then
        print_warning "Go is not installed - backend server setup will be skipped"
        SETUP_SERVERS=false
    fi

    # Check if we're in the right directory
    if [[ ! -d "$FRONTEND_DIR" ]] || [[ ! -f "$FRONTEND_DIR/package.json" ]]; then
        print_error "Frontend directory not found. Run this script from the project root."
        exit 1
    fi

    print_success "Dependencies check completed"
}

setup_environment() {
    print_status "Setting up test environment..."

    cd "$FRONTEND_DIR"

    # Install dependencies if needed
    if [[ ! -d "node_modules" ]] || [[ ! -d "node_modules/puppeteer" ]]; then
        print_status "Installing frontend dependencies..."
        npm install
    fi

    # Install E2E test dependencies
    if [[ ! -d "node_modules/puppeteer" ]]; then
        print_status "Installing Puppeteer..."
        npm install --save-dev puppeteer jest-puppeteer
    fi

    # Set environment variables
    export HEADLESS=$HEADLESS
    export SLOW_MO=$SLOW_MO
    export DEBUG_CONSOLE=$DEBUG
    export NODE_ENV=test

    if [[ $DEBUG == true ]]; then
        export DEBUG=puppeteer:*
    fi

    print_success "Environment setup completed"
}

start_servers() {
    if [[ $SETUP_SERVERS == false ]]; then
        print_status "Skipping server setup - assuming servers are already running"
        return 0
    fi

    print_status "Starting test servers..."

    # Start backend API server
    if command -v go &> /dev/null; then
        print_status "Starting backend API server..."
        cd "$BACKEND_DIR"
        
        # Check if backend dependencies are available
        if [[ -f "go.mod" ]]; then
            # Start API server in background
            go run cmd/api-server/main.go > /tmp/novacron-api.log 2>&1 &
            API_PID=$!
            echo $API_PID > /tmp/novacron-api.pid
            
            # Wait for API server to start
            for i in {1..30}; do
                if curl -f "$API_URL/health" > /dev/null 2>&1; then
                    print_success "Backend API server started (PID: $API_PID)"
                    break
                fi
                sleep 1
            done
        else
            print_warning "Backend go.mod not found - skipping backend server"
        fi
    fi

    # Start frontend development server
    cd "$FRONTEND_DIR"
    
    # Check if frontend server is already running
    if ! curl -f "$FRONTEND_URL" > /dev/null 2>&1; then
        print_status "Starting frontend development server..."
        
        npm run dev > /tmp/novacron-frontend.log 2>&1 &
        FRONTEND_PID=$!
        echo $FRONTEND_PID > /tmp/novacron-frontend.pid
        
        # Wait for frontend server to start
        for i in {1..60}; do
            if curl -f "$FRONTEND_URL" > /dev/null 2>&1; then
                print_success "Frontend server started (PID: $FRONTEND_PID)"
                break
            fi
            sleep 1
        done
    else
        print_success "Frontend server already running"
    fi
}

run_test_category() {
    local category=$1
    local description=${TEST_CATEGORIES[$category]}
    
    print_test_category "Running $description"
    
    local test_pattern
    if [[ $category == "integration" ]]; then
        test_pattern="src/__tests__/e2e/integration"
    else
        test_pattern="src/__tests__/e2e/$category"
    fi
    
    local jest_args=""
    
    if [[ $COVERAGE == true ]]; then
        jest_args="$jest_args --coverage"
    fi
    
    if [[ $PARALLEL == true ]]; then
        jest_args="$jest_args --maxWorkers=4"
    else
        jest_args="$jest_args --runInBand"
    fi
    
    if [[ $DEBUG == true ]]; then
        jest_args="$jest_args --verbose --no-cache"
    fi
    
    # Run the tests
    local start_time=$(date +%s)
    
    if npm run test:e2e -- --testPathPattern="$test_pattern" $jest_args; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "$description completed in ${duration}s"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_error "$description failed after ${duration}s"
        return 1
    fi
}

generate_report() {
    print_status "Generating test reports..."
    
    cd "$FRONTEND_DIR"
    
    # Create reports directory
    mkdir -p reports
    
    # Copy coverage reports if generated
    if [[ $COVERAGE == true ]] && [[ -d "coverage/e2e" ]]; then
        cp -r coverage/e2e reports/coverage
        print_success "Coverage report generated at reports/coverage/"
    fi
    
    # Generate test summary
    cat > reports/test-summary.md << EOF
# NovaCron E2E Test Report

**Date**: $(date)
**Configuration**: 
- Headless: $HEADLESS
- Debug: $DEBUG
- Coverage: $COVERAGE
- Parallel: $PARALLEL

## Test Categories Executed

EOF

    for category in "${CATEGORIES[@]}"; do
        echo "- **$category**: ${TEST_CATEGORIES[$category]}" >> reports/test-summary.md
    done

    cat >> reports/test-summary.md << EOF

## Results

See individual test output above for detailed results.

## Coverage Report

EOF

    if [[ $COVERAGE == true ]]; then
        echo "Coverage report available at: \`reports/coverage/index.html\`" >> reports/test-summary.md
    else
        echo "Coverage reporting was not enabled for this run." >> reports/test-summary.md
    fi

    print_success "Test summary generated at reports/test-summary.md"
}

cleanup_servers() {
    if [[ $CLEANUP == false ]]; then
        print_status "Skipping cleanup - servers left running"
        return 0
    fi

    print_status "Cleaning up test servers..."
    
    # Stop frontend server
    if [[ -f /tmp/novacron-frontend.pid ]]; then
        local frontend_pid=$(cat /tmp/novacron-frontend.pid)
        if kill -0 $frontend_pid 2>/dev/null; then
            kill $frontend_pid
            print_success "Frontend server stopped"
        fi
        rm -f /tmp/novacron-frontend.pid
    fi
    
    # Stop API server
    if [[ -f /tmp/novacron-api.pid ]]; then
        local api_pid=$(cat /tmp/novacron-api.pid)
        if kill -0 $api_pid 2>/dev/null; then
            kill $api_pid
            print_success "API server stopped"
        fi
        rm -f /tmp/novacron-api.pid
    fi
    
    # Clean up log files
    rm -f /tmp/novacron-*.log
}

run_tests() {
    print_status "Starting E2E test execution..."
    
    local total_categories=${#CATEGORIES[@]}
    local passed_categories=0
    local failed_categories=()
    
    cd "$FRONTEND_DIR"
    
    for category in "${CATEGORIES[@]}"; do
        echo ""
        echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
        
        if run_test_category "$category"; then
            ((passed_categories++))
        else
            failed_categories+=("$category")
        fi
        
        echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    done
    
    # Print final summary
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                         TEST SUMMARY                         ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    print_status "Total categories: $total_categories"
    print_success "Passed categories: $passed_categories"
    
    if [[ ${#failed_categories[@]} -gt 0 ]]; then
        print_error "Failed categories: ${#failed_categories[@]} (${failed_categories[*]})"
        return 1
    else
        print_success "All test categories passed!"
        return 0
    fi
}

main() {
    print_header
    
    parse_arguments "$@"
    check_dependencies
    setup_environment
    
    # Trap cleanup function to ensure servers are stopped on exit
    if [[ $CLEANUP == true ]]; then
        trap cleanup_servers EXIT INT TERM
    fi
    
    start_servers
    
    local test_result=0
    if run_tests; then
        print_success "E2E test suite completed successfully!"
        test_result=0
    else
        print_error "E2E test suite completed with failures!"
        test_result=1
    fi
    
    generate_report
    
    if [[ $CLEANUP == false ]]; then
        echo ""
        print_status "Servers are still running:"
        print_status "  Frontend: $FRONTEND_URL"
        print_status "  Backend API: $API_URL"
        print_status "Run with --cleanup or manually stop servers when done."
    fi
    
    exit $test_result
}

# Run main function with all arguments
main "$@"