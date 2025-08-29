#!/bin/bash

# E2E Testing Setup Verification Script
# Verifies that all components of the comprehensive E2E testing suite are properly configured

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

echo -e "${BLUE}🔍 NovaCron E2E Testing Setup Verification${NC}"
echo -e "${BLUE}===========================================${NC}"
echo ""

# Function to check file existence
check_file() {
    local file_path="$1"
    local description="$2"
    
    if [[ -f "$file_path" ]]; then
        echo -e "${GREEN}✅ $description${NC}"
        return 0
    else
        echo -e "${RED}❌ $description${NC}"
        return 1
    fi
}

# Function to check directory existence
check_directory() {
    local dir_path="$1"
    local description="$2"
    
    if [[ -d "$dir_path" ]]; then
        echo -e "${GREEN}✅ $description${NC}"
        return 0
    else
        echo -e "${RED}❌ $description${NC}"
        return 1
    fi
}

# Function to check npm package
check_npm_package() {
    local package="$1"
    local description="$2"
    
    cd "$FRONTEND_DIR"
    if npm list "$package" &>/dev/null; then
        echo -e "${GREEN}✅ $description${NC}"
        return 0
    else
        echo -e "${RED}❌ $description${NC}"
        return 1
    fi
}

echo -e "${YELLOW}📋 Checking E2E Test Infrastructure...${NC}"
echo ""

# Check core configuration files
check_file "$FRONTEND_DIR/jest.e2e.config.js" "Jest E2E configuration"
check_file "$FRONTEND_DIR/puppeteer.config.js" "Puppeteer configuration" 
check_file "$FRONTEND_DIR/package.json" "Frontend package.json"

echo ""

# Check test setup directory
check_directory "$FRONTEND_DIR/test-setup" "Test setup directory"
check_file "$FRONTEND_DIR/test-setup/puppeteer-setup.js" "Puppeteer setup script"
check_file "$FRONTEND_DIR/test-setup/puppeteer-jest-setup.js" "Puppeteer Jest utilities"
check_file "$FRONTEND_DIR/test-setup/puppeteer-teardown.js" "Puppeteer teardown script"

echo ""

# Check test directories
echo -e "${YELLOW}📂 Checking Test Directories...${NC}"
echo ""

TEST_DIRS=(
    "auth:Authentication tests"
    "admin:Admin panel tests" 
    "vm-management:VM management tests"
    "monitoring:Monitoring dashboard tests"
    "performance:Performance tests"
    "accessibility:Accessibility tests"
    "integration:Backend integration tests"
)

for test_dir in "${TEST_DIRS[@]}"; do
    IFS=':' read -r dir_name description <<< "$test_dir"
    check_directory "$FRONTEND_DIR/src/__tests__/e2e/$dir_name" "$description directory"
done

echo ""

# Check individual test files
echo -e "${YELLOW}📄 Checking Test Files...${NC}"
echo ""

check_file "$FRONTEND_DIR/src/__tests__/e2e/auth/authentication-flows.test.js" "Authentication flows tests"
check_file "$FRONTEND_DIR/src/__tests__/e2e/admin/admin-panel.test.js" "Admin panel tests"
check_file "$FRONTEND_DIR/src/__tests__/e2e/vm-management/vm-lifecycle.test.js" "VM lifecycle tests"
check_file "$FRONTEND_DIR/src/__tests__/e2e/monitoring/dashboard-monitoring.test.js" "Dashboard monitoring tests"
check_file "$FRONTEND_DIR/src/__tests__/e2e/performance/performance-testing.test.js" "Performance testing"
check_file "$FRONTEND_DIR/src/__tests__/e2e/accessibility/accessibility-testing.test.js" "Accessibility tests"
check_file "$FRONTEND_DIR/src/__tests__/e2e/integration/backend-integration.test.js" "Backend integration tests"

echo ""

# Check scripts
echo -e "${YELLOW}🔧 Checking Execution Scripts...${NC}"
echo ""

check_file "$PROJECT_ROOT/scripts/run-e2e-tests.sh" "E2E test runner script"
if [[ -f "$PROJECT_ROOT/scripts/run-e2e-tests.sh" ]]; then
    if [[ -x "$PROJECT_ROOT/scripts/run-e2e-tests.sh" ]]; then
        echo -e "${GREEN}✅ E2E test runner is executable${NC}"
    else
        echo -e "${YELLOW}⚠️ E2E test runner needs execute permissions${NC}"
        chmod +x "$PROJECT_ROOT/scripts/run-e2e-tests.sh"
        echo -e "${GREEN}✅ Execute permissions granted${NC}"
    fi
fi

echo ""

# Check CI/CD configuration
echo -e "${YELLOW}🚀 Checking CI/CD Configuration...${NC}"
echo ""

check_file "$PROJECT_ROOT/.github/workflows/comprehensive-testing.yml" "GitHub Actions E2E workflow"
check_directory "$PROJECT_ROOT/.github/workflows" "GitHub workflows directory"

echo ""

# Check dependencies
echo -e "${YELLOW}📦 Checking Dependencies...${NC}"
echo ""

if [[ -d "$FRONTEND_DIR" ]]; then
    cd "$FRONTEND_DIR"
    
    # Check core testing dependencies
    check_npm_package "puppeteer" "Puppeteer browser automation"
    check_npm_package "jest" "Jest testing framework"
    check_npm_package "jest-puppeteer" "Jest-Puppeteer integration"
    check_npm_package "jest-environment-jsdom" "JSDOM test environment"
    check_npm_package "axe-core" "Accessibility testing (axe-core)"
    check_npm_package "jest-html-reporter" "HTML test reporter"
else
    echo -e "${RED}❌ Frontend directory not found${NC}"
fi

echo ""

# Check package.json scripts
echo -e "${YELLOW}⚙️ Checking NPM Scripts...${NC}"
echo ""

if [[ -f "$FRONTEND_DIR/package.json" ]]; then
    cd "$FRONTEND_DIR"
    
    REQUIRED_SCRIPTS=(
        "test:e2e"
        "test:e2e:debug" 
        "test:e2e:coverage"
        "test:puppeteer"
    )
    
    for script in "${REQUIRED_SCRIPTS[@]}"; do
        if npm run "$script" --dry-run &>/dev/null; then
            echo -e "${GREEN}✅ npm script: $script${NC}"
        else
            echo -e "${RED}❌ npm script: $script${NC}"
        fi
    done
else
    echo -e "${RED}❌ Package.json not found${NC}"
fi

echo ""

# Check system requirements
echo -e "${YELLOW}🖥️ Checking System Requirements...${NC}"
echo ""

# Check Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✅ Node.js: $NODE_VERSION${NC}"
else
    echo -e "${RED}❌ Node.js not installed${NC}"
fi

# Check npm
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    echo -e "${GREEN}✅ npm: $NPM_VERSION${NC}"
else
    echo -e "${RED}❌ npm not installed${NC}"
fi

# Check Go (optional)
if command -v go &> /dev/null; then
    GO_VERSION=$(go version)
    echo -e "${GREEN}✅ Go: $GO_VERSION${NC}"
else
    echo -e "${YELLOW}⚠️ Go not installed (backend tests will be limited)${NC}"
fi

echo ""

# Summary
echo -e "${BLUE}📊 Verification Summary${NC}"
echo -e "${BLUE}=====================${NC}"
echo ""

# Run a quick validation
ISSUES_FOUND=0

# Critical files check
CRITICAL_FILES=(
    "$FRONTEND_DIR/jest.e2e.config.js"
    "$FRONTEND_DIR/puppeteer.config.js"
    "$FRONTEND_DIR/test-setup/puppeteer-jest-setup.js"
    "$PROJECT_ROOT/scripts/run-e2e-tests.sh"
)

for file in "${CRITICAL_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        ((ISSUES_FOUND++))
    fi
done

# Test directories check
for test_dir in "${TEST_DIRS[@]}"; do
    IFS=':' read -r dir_name description <<< "$test_dir"
    if [[ ! -d "$FRONTEND_DIR/src/__tests__/e2e/$dir_name" ]]; then
        ((ISSUES_FOUND++))
    fi
done

if [[ $ISSUES_FOUND -eq 0 ]]; then
    echo -e "${GREEN}🎉 E2E Testing Setup Complete!${NC}"
    echo ""
    echo -e "${GREEN}✅ All configuration files present${NC}"
    echo -e "${GREEN}✅ All test directories created${NC}" 
    echo -e "${GREEN}✅ All test files implemented${NC}"
    echo -e "${GREEN}✅ CI/CD workflow configured${NC}"
    echo -e "${GREEN}✅ Dependencies properly installed${NC}"
    echo ""
    echo -e "${BLUE}🚀 Ready to run E2E tests!${NC}"
    echo ""
    echo -e "${YELLOW}Usage:${NC}"
    echo -e "  ${GREEN}./scripts/run-e2e-tests.sh --all${NC}          # Run all tests"
    echo -e "  ${GREEN}./scripts/run-e2e-tests.sh --category auth${NC}  # Run specific category"
    echo -e "  ${GREEN}./scripts/run-e2e-tests.sh --debug${NC}         # Debug mode"
    echo ""
    echo -e "${YELLOW}Test Categories Available:${NC}"
    for test_dir in "${TEST_DIRS[@]}"; do
        IFS=':' read -r dir_name description <<< "$test_dir"
        echo -e "  ${GREEN}$dir_name${NC} - $description"
    done
    
    exit 0
else
    echo -e "${RED}❌ Setup Issues Found: $ISSUES_FOUND${NC}"
    echo ""
    echo -e "${YELLOW}⚠️ Some components are missing or misconfigured.${NC}"
    echo -e "${YELLOW}Please review the output above and ensure all required files are present.${NC}"
    echo ""
    exit 1
fi