#!/bin/bash

# Quick validation script to test basic functionality without full environment

echo "🔍 NovaCron Quick Validation"
echo "============================"

# Check if running from correct directory
if [[ ! -f "package.json" ]]; then
    echo "❌ Please run from project root directory"
    exit 1
fi

# Test 1: Frontend build
echo "📦 Testing Frontend Build..."
cd frontend
if npm run build > /dev/null 2>&1; then
    echo "  ✅ Frontend builds successfully"
else
    echo "  ❌ Frontend build failed"
fi
cd ..

# Test 2: Backend compilation
echo "🔧 Testing Backend Compilation..."
cd backend
if go build -o /tmp/novacron-test ./cmd/api-server > /dev/null 2>&1; then
    echo "  ✅ Backend compiles successfully"
    rm -f /tmp/novacron-test
else
    echo "  ❌ Backend compilation failed"
fi
cd ..

# Test 3: Test files syntax
echo "🧪 Validating Test Files..."

# Check Go test files
if find tests -name "*.go" -exec go vet {} \; > /dev/null 2>&1; then
    echo "  ✅ Go test files valid"
else
    echo "  ⚠️  Go test files have syntax issues"
fi

# Check JavaScript test files
if command -v node >/dev/null 2>&1; then
    if node -c tests/integration/frontend_backend_validation.js > /dev/null 2>&1; then
        echo "  ✅ JavaScript test files valid"
    else
        echo "  ⚠️  JavaScript test files have syntax issues"
    fi
fi

# Test 4: Configuration validation
echo "⚙️  Validating Configuration..."

config_issues=0

# Check essential config files
if [[ ! -f "frontend/next.config.js" ]]; then
    echo "  ❌ Missing frontend/next.config.js"
    config_issues=$((config_issues + 1))
fi

if [[ ! -f "backend/go.mod" ]]; then
    echo "  ❌ Missing backend/go.mod"
    config_issues=$((config_issues + 1))
fi

if [[ ! -f "docker-compose.yml" ]]; then
    echo "  ⚠️  Missing docker-compose.yml (optional)"
fi

if [[ $config_issues -eq 0 ]]; then
    echo "  ✅ Essential configuration files present"
fi

# Test 5: Database schema validation
echo "📊 Validating Database Schema..."

# Check if schema files exist
schema_files=0
if find . -name "*.sql" -type f | grep -q .; then
    schema_files=$(find . -name "*.sql" -type f | wc -l)
    echo "  ✅ Found $schema_files SQL schema files"
else
    echo "  ⚠️  No SQL schema files found"
fi

# Test 6: API documentation
echo "📚 Checking API Documentation..."

if [[ -f "docs/API.md" ]] || [[ -f "README.md" ]]; then
    echo "  ✅ Documentation files present"
else
    echo "  ⚠️  Consider adding API documentation"
fi

# Summary
echo ""
echo "📋 Quick Validation Summary"
echo "=========================="
echo "This quick validation checks basic project structure and compilation."
echo "For complete integration validation, run:"
echo "  ./tests/run_integration_validation.sh"
echo ""
echo "For detailed production readiness assessment, see:"
echo "  ./docs/PRODUCTION_READINESS_VALIDATION_REPORT.md"

exit 0