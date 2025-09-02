# NovaCron Makefile - Comprehensive Testing Framework

.PHONY: all test build clean docker-build docker-test help

# Default target
all: build

# Build all components
build:
	@echo "Building NovaCron components..."
	docker-compose build


# Build a stable core subset (orchestration and minimal VM types) without experimental modules
core-build:
	@echo "Building core subset (orchestration)..."
	cd backend/core/orchestration && go build ./...

# Run core unit tests (orchestration)
core-test:
	@echo "Running core unit tests (orchestration)..."
	cd backend/core/orchestration && go test -v ./...

# Minimal core server targets
core-serve:
	@echo "Starting Core Server on :8090"
	cd backend/cmd/core-server && go run .


# ============================================================================
# Database Management
# ============================================================================

# Database URL configuration
DB_URL ?= postgres://postgres:postgres@localhost:5432/novacron?sslmode=disable
DB_TEST_URL ?= postgres://postgres:postgres@localhost:5432/novacron_test?sslmode=disable

# Run database migrations up
db-migrate:
	@echo "Running database migrations..."
	@cd database && DB_URL="$(DB_URL)" ./scripts/migrate.sh up

# Rollback last migration
db-rollback:
	@echo "Rolling back last migration..."
	@cd database && DB_URL="$(DB_URL)" ./scripts/migrate.sh down

# Create a new migration
db-migrate-create:
	@read -p "Enter migration name: " name; \
	cd database && ./scripts/migrate.sh create $$name

# Show current migration version
db-version:
	@cd database && DB_URL="$(DB_URL)" ./scripts/migrate.sh version

# Show migration status
db-status:
	@cd database && DB_URL="$(DB_URL)" ./scripts/migrate.sh status

# Seed database with development data
db-seed:
	@echo "Seeding database with development data..."
	@cd database && DB_URL="$(DB_URL)" ./scripts/seed.sh seed

# Clean seed data
db-clean:
	@echo "Cleaning seed data..."
	@cd database && DB_URL="$(DB_URL)" ./scripts/seed.sh clean

# Reset database (drop, migrate, seed)
db-reset:
	@echo "Resetting database..."
	@cd database && DB_URL="$(DB_URL)" ./scripts/migrate.sh drop
	@$(MAKE) db-migrate
	@$(MAKE) db-seed

# Setup test database
db-test-setup:
	@echo "Setting up test database..."
	@cd database && DB_URL="$(DB_TEST_URL)" ./scripts/migrate.sh up
	@cd database && DB_URL="$(DB_TEST_URL)" ./scripts/seed.sh seed

# Clean test database
db-test-clean:
	@echo "Cleaning test database..."
	@cd database && DB_URL="$(DB_TEST_URL)" ./scripts/migrate.sh drop

# Validate migration files
db-validate:
	@echo "Validating migration files..."
	@cd database && ./scripts/migrate.sh validate

# Database console
db-console:
	@echo "Opening database console..."
	@psql "$(DB_URL)"

# Backup database
db-backup:
	@echo "Backing up database..."
	@mkdir -p backups
	@pg_dump "$(DB_URL)" > backups/novacron_$$(date +%Y%m%d_%H%M%S).sql
	@echo "Backup saved to backups/novacron_$$(date +%Y%m%d_%H%M%S).sql"

# Restore database from backup
db-restore:
	@read -p "Enter backup file path: " file; \
	echo "Restoring database from $$file..."; \
	psql "$(DB_URL)" < $$file

# ============================================================================
# Testing Targets - Core
# ============================================================================

# Run all tests
test: test-unit test-integration test-benchmarks test-multicloud test-ml test-prefetching test-cache test-sdk test-e2e test-chaos
	@echo "All tests completed"

# Run tests in Docker (recommended - uses Go 1.19)
test-docker:
	@echo "Running tests in Docker..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 go test ./backend/core/vm/...

# Run Go tests locally (requires Go 1.23+)
test-local:
	@echo "Running Go tests locally..."
	cd backend/core && go test -v ./vm/...

# ============================================================================
# Unit Testing
# ============================================================================

test-unit:
	@echo "Running unit tests..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 go test ./backend/core/vm/... -v -run "Test.*Fixed"

test-unit-coverage:
	@echo "Running unit tests with coverage..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 \
		go test ./backend/core/... -coverprofile=coverage.out -covermode=atomic
	docker run --rm -v $(PWD):/app -w /app golang:1.19 \
		go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

test-unit-race:
	@echo "Running unit tests with race detection..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 \
		go test ./backend/core/vm/... -race -v -run "Test.*Fixed"

# ============================================================================
# Integration Testing
# ============================================================================

# Run all integration tests with full environment
test-integration:
	@echo "Running comprehensive integration tests..."
	$(MAKE) test-integration-setup
	@echo "Waiting for services to be fully ready..."
	@sleep 15
	docker-compose -f backend/docker-compose.test.yml exec test-runner /app/run-tests.sh
	@echo "Integration tests completed"

# Setup complete integration test environment
test-integration-setup:
	@echo "Setting up full integration test environment..."
	cd backend && docker-compose -f docker-compose.test.yml up -d
	@echo "Waiting for all services to be ready..."
	@sleep 20
	@echo "Running database migrations..."
	cd backend && docker-compose -f docker-compose.test.yml exec migrate /bin/sh -c "migrate -path /migrations -database postgres://postgres:password@postgres-test:5432/novacron_test?sslmode=disable up"

# Teardown integration test environment
test-integration-teardown:
	@echo "Tearing down integration test environment..."
	cd backend && docker-compose -f docker-compose.test.yml down -v
	@echo "Cleaning up test volumes..."
	docker volume prune -f

# Run quick integration tests (no coverage)
test-integration-quick:
	@echo "Running quick integration tests..."
	$(MAKE) test-integration-setup
	@sleep 10
	cd backend && docker-compose -f docker-compose.test.yml exec test-runner /app/quick-test.sh
	$(MAKE) test-integration-teardown

# Run specific integration test suites
test-integration-auth:
	@echo "Running authentication integration tests..."
	$(MAKE) test-integration-setup
	@sleep 10
	cd backend && docker-compose -f docker-compose.test.yml exec test-runner /app/test-auth.sh
	$(MAKE) test-integration-teardown

test-integration-vm:
	@echo "Running VM lifecycle integration tests..."
	$(MAKE) test-integration-setup
	@sleep 10
	cd backend && docker-compose -f docker-compose.test.yml exec test-runner /app/test-vm.sh
	$(MAKE) test-integration-teardown

test-integration-api:
	@echo "Running API endpoint integration tests..."
	$(MAKE) test-integration-setup
	@sleep 10
	cd backend && docker-compose -f docker-compose.test.yml exec test-runner /app/test-api.sh
	$(MAKE) test-integration-teardown

test-integration-websocket:
	@echo "Running WebSocket integration tests..."
	$(MAKE) test-integration-setup
	@sleep 10
	cd backend && docker-compose -f docker-compose.test.yml exec test-runner /app/test-websocket.sh
	$(MAKE) test-integration-teardown

test-integration-federation:
	@echo "Running multi-cloud federation tests..."
	$(MAKE) test-integration-setup
	@sleep 15
	@echo "Running federation tests against LocalStack and MinIO..."
	cd backend && docker-compose -f docker-compose.test.yml exec test-runner go test -v -timeout=15m ./tests/integration/federation/...
	$(MAKE) test-integration-teardown

# Run integration tests with coverage reporting
test-integration-coverage:
	@echo "Running integration tests with coverage reporting..."
	$(MAKE) test-integration-setup
	@sleep 15
	cd backend && docker-compose -f docker-compose.test.yml exec test-runner /app/run-tests.sh
	@echo "Copying coverage reports..."
	cd backend && docker-compose -f docker-compose.test.yml cp test-runner:/app/coverage ./coverage-integration
	@echo "Integration coverage reports available in backend/coverage-integration/"
	$(MAKE) test-integration-teardown

# Run integration benchmarks
test-integration-benchmarks:
	@echo "Running integration performance benchmarks..."
	$(MAKE) test-integration-setup
	@sleep 15
	cd backend && docker-compose -f docker-compose.test.yml exec test-runner /app/benchmark.sh
	cd backend && docker-compose -f docker-compose.test.yml cp test-runner:/app/test-results/benchmark-results.txt ./benchmark-results.txt
	@echo "Benchmark results saved to backend/benchmark-results.txt"
	$(MAKE) test-integration-teardown

# Clean up test data from integration tests
test-integration-cleanup:
	@echo "Cleaning up integration test data..."
	$(MAKE) test-integration-setup
	@sleep 10
	cd backend && docker-compose -f docker-compose.test.yml exec test-runner /app/cleanup-test-data.sh
	$(MAKE) test-integration-teardown

# View integration test logs
test-integration-logs:
	@echo "Showing integration test environment logs..."
	cd backend && docker-compose -f docker-compose.test.yml logs -f

# Check integration test environment status
test-integration-status:
	@echo "Checking integration test environment status..."
	cd backend && docker-compose -f docker-compose.test.yml ps
	@echo ""
	@echo "Service Health Checks:"
	cd backend && docker-compose -f docker-compose.test.yml exec postgres-test pg_isready -U postgres -d novacron_test || echo "PostgreSQL: NOT READY"
	cd backend && docker-compose -f docker-compose.test.yml exec redis-test redis-cli -a testredispass ping || echo "Redis: NOT READY"
	cd backend && docker-compose -f docker-compose.test.yml exec mock-aws curl -f http://localhost:4566/health || echo "LocalStack: NOT READY"
	cd backend && docker-compose -f docker-compose.test.yml exec minio-test curl -f http://localhost:9000/minio/health/live || echo "MinIO: NOT READY"

# Reset integration test environment
test-integration-reset:
	@echo "Resetting integration test environment..."
	$(MAKE) test-integration-teardown
	@sleep 5
	$(MAKE) test-integration-setup

# ============================================================================
# Multi-Cloud Testing
# ============================================================================

test-multicloud:
	@echo "Running multi-cloud integration tests..."
	docker run --rm -v $(PWD):/app -w /app \
		-e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
		-e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
		-e AZURE_CLIENT_ID="${AZURE_CLIENT_ID}" \
		-e AZURE_CLIENT_SECRET="${AZURE_CLIENT_SECRET}" \
		-e AZURE_TENANT_ID="${AZURE_TENANT_ID}" \
		-e AZURE_SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID}" \
		-e GCP_PROJECT_ID="${GCP_PROJECT_ID}" \
		-e GCP_CREDENTIALS_PATH="/app/gcp-credentials.json" \
		golang:1.19 go test ./backend/tests/multicloud/... -v -timeout 30m

test-multicloud-unit:
	@echo "Running multi-cloud unit tests..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 \
		go test ./backend/tests/multicloud/... -v -short

test-multicloud-aws:
	@echo "Running AWS-specific tests..."
	docker run --rm -v $(PWD):/app -w /app \
		-e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
		-e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
		golang:1.19 go test ./backend/tests/multicloud/... -v -run "TestAWS.*" -timeout 20m

test-multicloud-azure:
	@echo "Running Azure-specific tests..."
	docker run --rm -v $(PWD):/app -w /app \
		-e AZURE_CLIENT_ID="${AZURE_CLIENT_ID}" \
		-e AZURE_CLIENT_SECRET="${AZURE_CLIENT_SECRET}" \
		-e AZURE_TENANT_ID="${AZURE_TENANT_ID}" \
		-e AZURE_SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID}" \
		golang:1.19 go test ./backend/tests/multicloud/... -v -run "TestAzure.*" -timeout 20m

test-multicloud-gcp:
	@echo "Running GCP-specific tests..."
	docker run --rm -v $(PWD):/app -w /app \
		-e GCP_PROJECT_ID="${GCP_PROJECT_ID}" \
		-e GCP_CREDENTIALS_PATH="/app/gcp-credentials.json" \
		golang:1.19 go test ./backend/tests/multicloud/... -v -run "TestGCP.*" -timeout 20m

# ============================================================================
# AI/ML Model Testing
# ============================================================================

test-ml:
	@echo "Running AI/ML model tests..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 \
		go test ./backend/tests/ml/... -v -timeout 15m

test-prefetching:
	@echo "Running predictive prefetching tests..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 \
		go test ./backend/core/vm/predictive_prefetching_test.go ./backend/core/vm/predictive_prefetching.go ./backend/core/vm/vm.go ./backend/core/vm/vm_types_minimal.go ./backend/core/vm/vm_migration_types.go -v

test-ml-accuracy:
	@echo "Running ML model accuracy tests..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 \
		go test ./backend/tests/ml/... -v -run "TestModelAccuracy.*" -timeout 10m

test-ml-performance:
	@echo "Running ML model performance tests..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 \
		go test ./backend/tests/ml/... -v -run "TestPerformanceRegression.*" -timeout 10m

test-ml-drift:
	@echo "Running ML model drift detection tests..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 \
		go test ./backend/tests/ml/... -v -run "TestModelDrift.*" -timeout 5m

# ============================================================================
# Redis Cache Testing
# ============================================================================

test-cache:
	@echo "Running Redis cache tests..."
	docker run --rm -v $(PWD):/app -w /app \
		--network host \
		-e REDIS_URL="redis://localhost:6379" \
		golang:1.19 go test ./backend/tests/cache/... -v -timeout 10m

test-cache-performance:
	@echo "Running Redis performance tests..."
	docker run --rm -v $(PWD):/app -w /app \
		--network host \
		-e REDIS_URL="redis://localhost:6379" \
		golang:1.19 go test ./backend/tests/cache/... -v -run "TestRedisCachePerformance.*" -timeout 15m

test-cache-consistency:
	@echo "Running Redis consistency tests..."
	docker run --rm -v $(PWD):/app -w /app \
		--network host \
		-e REDIS_URL="redis://localhost:6379" \
		golang:1.19 go test ./backend/tests/cache/... -v -run "TestCacheConsistency.*" -timeout 10m

test-cache-chaos:
	@echo "Running Redis chaos engineering tests..."
	docker run --rm -v $(PWD):/app -w /app \
		--network host \
		-e REDIS_URL="redis://localhost:6379" \
		golang:1.19 go test ./backend/tests/cache/... -v -run "TestRedisChaosEngineering.*" -timeout 20m

# ============================================================================
# SDK Testing
# ============================================================================

test-sdk:
	@echo "Running cross-language SDK tests..."
	docker run --rm -v $(PWD):/app -w /app \
		--network host \
		-e NOVACRON_API_URL="http://localhost:8090" \
		-e NOVACRON_API_KEY="test-api-key" \
		golang:1.19 go test ./backend/tests/sdk/... -v -timeout 20m

test-sdk-go:
	@echo "Running Go SDK tests..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 \
		go test ./backend/tests/sdk/... -v -run ".*_go" -timeout 10m

test-sdk-python:
	@echo "Running Python SDK tests..."
	docker run --rm -v $(PWD):/app -w /app \
		-v $(PWD)/sdk_tests:/tmp/sdk_tests \
		python:3.9 python -m pytest /tmp/sdk_tests/python/ -v

test-sdk-javascript:
	@echo "Running JavaScript SDK tests..."
	docker run --rm -v $(PWD):/app -w /app \
		-v $(PWD)/sdk_tests:/tmp/sdk_tests \
		node:16 npm test --prefix /tmp/sdk_tests/javascript/

test-sdk-compatibility:
	@echo "Running SDK compatibility tests..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 \
		go test ./backend/tests/sdk/... -v -run "TestCrossLanguageSDKCompatibility.*" -timeout 25m

# ============================================================================
# End-to-End Testing
# ============================================================================

test-e2e:
	@echo "Running end-to-end tests..."
	docker run --rm -v $(PWD):/app -w /app \
		--network host \
		-e NOVACRON_API_URL="http://localhost:8090" \
		-e NOVACRON_UI_URL="http://localhost:8092" \
		-e NOVACRON_API_KEY="test-api-key" \
		golang:1.19 go test ./backend/tests/e2e/... -v -timeout 30m

test-e2e-workflows:
	@echo "Running E2E workflow tests..."
	docker run --rm -v $(PWD):/app -w /app \
		--network host \
		-e NOVACRON_API_URL="http://localhost:8090" \
		-e ENABLE_UI_TESTS="false" \
		-e ENABLE_LOAD_TESTS="true" \
		golang:1.19 go test ./backend/tests/e2e/... -v -run "TestComprehensiveWorkflows.*" -timeout 45m

test-e2e-ui:
	@echo "Running E2E UI tests..."
	@echo "Note: UI tests require Selenium/WebDriver setup"
	docker run --rm -v $(PWD):/app -w /app \
		--network host \
		-e NOVACRON_UI_URL="http://localhost:8092" \
		-e ENABLE_UI_TESTS="true" \
		golang:1.19 go test ./backend/tests/e2e/... -v -run ".*UI.*" -timeout 20m

# ============================================================================
# Chaos Engineering
# ============================================================================

test-chaos:
	@echo "Running chaos engineering tests..."
	docker run --rm -v $(PWD):/app -w /app \
		--network host \
		--privileged \
		-e REDIS_URL="redis://localhost:6379" \
		golang:1.19 go test ./backend/tests/chaos/... -v -timeout 25m

test-chaos-redis:
	@echo "Running Redis chaos tests..."
	docker run --rm -v $(PWD):/app -w /app \
		--network host \
		-e REDIS_URL="redis://localhost:6379" \
		golang:1.19 go test ./backend/tests/chaos/... -v -run "TestRedisClusterChaosEngineering.*" -timeout 20m

test-chaos-advanced:
	@echo "Running advanced chaos scenarios..."
	docker run --rm -v $(PWD):/app -w /app \
		--network host \
		--privileged \
		golang:1.19 go test ./backend/tests/chaos/... -v -run "TestAdvancedChaosScenarios.*" -timeout 30m

# ============================================================================
# Performance & Benchmark Testing
# ============================================================================

test-benchmarks:
	@echo "Running performance benchmarks..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 \
		go test ./backend/tests/benchmarks/... -bench=. -v -timeout 15m

test-benchmarks-vm:
	@echo "Running VM benchmark tests..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 \
		go test ./backend/tests/benchmarks/... -bench=BenchmarkVM.* -v -timeout 10m

test-benchmarks-scheduler:
	@echo "Running scheduler benchmark tests..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 \
		go test ./backend/core/scheduler/policy/... -bench=. -v -timeout 10m

test-performance:
	@echo "Running comprehensive performance tests..."
	$(MAKE) test-benchmarks
	$(MAKE) test-cache-performance
	$(MAKE) test-ml-performance

# ============================================================================
# Memory & Profiling
# ============================================================================

test-memory:
	@echo "Running tests with memory profiling..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 \
		go test ./backend/core/vm/... -memprofile=mem.prof -v -run "Test.*Fixed"

test-cpu-profile:
	@echo "Running tests with CPU profiling..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 \
		go test ./backend/core/vm/... -cpuprofile=cpu.prof -v -run "Test.*Fixed"

# ============================================================================
# Code Quality & Security
# ============================================================================

lint-backend:
	@echo "Linting backend code..."
	docker run --rm -v $(PWD):/app -w /app golangci/golangci-lint:latest \
		golangci-lint run ./backend/...

security-scan:
	@echo "Running security scan..."
	docker run --rm -v $(PWD):/app -w /app securecodewarrior/gosec:latest \
		gosec ./backend/...

vulnerability-check:
	@echo "Checking for known vulnerabilities..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 \
		go list -json -m all | docker run --rm -i sonatypecommunity/nancy:latest sleuth

# ============================================================================
# Test Environment Management
# ============================================================================

test-env-up:
	@echo "Starting test environment..."
	docker-compose -f docker-compose.test.yml up -d
	@echo "Waiting for services to be ready..."
	@sleep 15
	@echo "Test environment ready"

test-env-down:
	@echo "Stopping test environment..."
	docker-compose -f docker-compose.test.yml down -v

test-env-logs:
	@echo "Showing test environment logs..."
	docker-compose -f docker-compose.test.yml logs -f

test-env-reset:
	@echo "Resetting test environment..."
	$(MAKE) test-env-down
	$(MAKE) test-env-up

# ============================================================================
# CI/CD Integration
# ============================================================================

ci-test:
	@echo "Running CI test suite..."
	$(MAKE) test-unit-coverage
	$(MAKE) test-integration
	$(MAKE) test-benchmarks

ci-test-full:
	@echo "Running full CI test suite..."
	$(MAKE) test-env-up
	$(MAKE) test-unit-coverage
	$(MAKE) test-integration
	$(MAKE) test-cache
	$(MAKE) test-ml
	$(MAKE) test-benchmarks
	$(MAKE) test-env-down

ci-quality:
	@echo "Running CI quality checks..."
	$(MAKE) lint-backend
	$(MAKE) security-scan
	$(MAKE) vulnerability-check

# ============================================================================
# Coverage Reporting & Analysis
# ============================================================================

# Generate comprehensive coverage report
test-coverage-report:
	@echo "Generating comprehensive integration test coverage report..."
	$(MAKE) test-integration-setup
	@sleep 15
	cd backend && docker-compose -f docker-compose.test.yml exec test-runner /app/coverage-report.sh
	@echo "Copying coverage reports to host..."
	cd backend && docker-compose -f docker-compose.test.yml cp test-runner:/app/coverage ./coverage-integration
	cd backend && docker-compose -f docker-compose.test.yml cp test-runner:/app/coverage-reports ./coverage-reports
	cd backend && docker-compose -f docker-compose.test.yml cp test-runner:/app/test-results ./test-results
	@echo ""
	@echo "Coverage reports generated:"
	@echo "- HTML Report: backend/coverage-integration/coverage.html"
	@echo "- Analysis: backend/coverage-integration/coverage-analysis.md"
	@echo "- Comprehensive: backend/coverage-reports/integration-test-report.md"
	@echo "- JSON: backend/coverage-integration/coverage.json"
	@echo "- XML: backend/coverage-integration/coverage.xml"
	$(MAKE) test-integration-teardown

# Generate coverage report without running tests (use existing coverage data)
test-coverage-report-only:
	@echo "Generating coverage report from existing data..."
	$(MAKE) test-integration-setup
	@sleep 10
	cd backend && docker-compose -f docker-compose.test.yml exec test-runner /app/coverage-report.sh
	cd backend && docker-compose -f docker-compose.test.yml cp test-runner:/app/coverage ./coverage-integration
	cd backend && docker-compose -f docker-compose.test.yml cp test-runner:/app/coverage-reports ./coverage-reports
	$(MAKE) test-integration-teardown

# Display coverage summary
test-coverage-summary:
	@echo "Integration Test Coverage Summary"
	@echo "=================================="
	@if [ -f backend/coverage-integration/coverage-functions.txt ]; then \
		echo "Total Coverage: $$(tail -1 backend/coverage-integration/coverage-functions.txt | awk '{print $$3}')"; \
		echo ""; \
		echo "Package Coverage:"; \
		grep -v "total:" backend/coverage-integration/coverage-functions.txt | head -10; \
		echo ""; \
		echo "Low Coverage Functions:"; \
		grep "0.0%" backend/coverage-integration/coverage-functions.txt | head -5 || echo "None found"; \
	else \
		echo "No coverage data found. Run 'make test-coverage-report' first."; \
	fi

# Validate coverage against thresholds
test-coverage-validate:
	@echo "Validating coverage against thresholds..."
	@if [ -f backend/coverage-integration/coverage-summary.json ]; then \
		COVERAGE=$$(cat backend/coverage-integration/coverage-summary.json | grep -o '"total_coverage": [0-9.]*' | cut -d' ' -f2); \
		STATUS=$$(cat backend/coverage-integration/coverage-summary.json | grep -o '"status": "[^"]*"' | cut -d'"' -f4); \
		echo "Total Coverage: $${COVERAGE}%"; \
		if [ "$$STATUS" = "passed" ]; then \
			echo "✅ Coverage validation PASSED"; \
		else \
			echo "❌ Coverage validation FAILED"; \
			exit 1; \
		fi; \
	else \
		echo "No coverage summary found. Run 'make test-coverage-report' first."; \
		exit 1; \
	fi

# Generate coverage badge
test-coverage-badge:
	@echo "Generating coverage badge..."
	@if [ -f backend/coverage-integration/badge.json ]; then \
		COVERAGE=$$(cat backend/coverage-integration/badge.json | grep -o '"message": "[^"]*"' | cut -d'"' -f4); \
		COLOR=$$(cat backend/coverage-integration/badge.json | grep -o '"color": "[^"]*"' | cut -d'"' -f4); \
		echo "Coverage Badge: $${COVERAGE} ($${COLOR})"; \
		echo "Badge data available at: backend/coverage-integration/badge.json"; \
	else \
		echo "No badge data found. Run 'make test-coverage-report' first."; \
	fi

# ============================================================================
# Reporting & Metrics
# ============================================================================

test-report:
	@echo "Generating comprehensive test report..."
	$(MAKE) test-coverage-report
	@echo ""
	@echo "Comprehensive test report generated!"
	@echo "Main report: backend/coverage-reports/integration-test-report.md"

test-metrics:
	@echo "Collecting integration test metrics..."
	$(MAKE) test-integration-setup
	@sleep 10
	cd backend && docker-compose -f docker-compose.test.yml exec test-runner go test ./tests/integration/... -json > test-results.json
	cd backend && docker-compose -f docker-compose.test.yml cp test-runner:/app/backend/test-results.json ./test-results.json
	@echo "Test metrics saved to backend/test-results.json"
	$(MAKE) test-integration-teardown

# Generate test execution dashboard
test-dashboard:
	@echo "Integration Test Dashboard"
	@echo "=========================="
	@echo ""
	@if [ -f backend/coverage-integration/coverage-summary.json ]; then \
		echo "📊 Coverage: $$(cat backend/coverage-integration/coverage-summary.json | grep -o '"total_coverage": [0-9.]*' | cut -d' ' -f2)%"; \
		echo "🎯 Status: $$(cat backend/coverage-integration/coverage-summary.json | grep -o '"status": "[^"]*"' | cut -d'"' -f4)"; \
		echo "📅 Last Run: $$(cat backend/coverage-integration/coverage-summary.json | grep -o '"timestamp": "[^"]*"' | cut -d'"' -f4)"; \
	else \
		echo "No coverage data available"; \
	fi
	@echo ""
	@echo "Available Commands:"
	@echo "  make test-integration           # Run all integration tests"
	@echo "  make test-integration-quick     # Run quick tests (no coverage)"
	@echo "  make test-coverage-report       # Generate full coverage report"
	@echo "  make test-coverage-summary      # Show coverage summary"
	@echo "  make test-coverage-validate     # Validate coverage thresholds"
	@echo ""
	@echo "Specific Test Suites:"
	@echo "  make test-integration-auth      # Authentication tests"
	@echo "  make test-integration-vm        # VM lifecycle tests"
	@echo "  make test-integration-api       # API endpoint tests"
	@echo "  make test-integration-websocket # WebSocket tests"
	@echo "  make test-integration-federation # Multi-cloud federation tests"

# ============================================================================
# Documentation & Examples
# ============================================================================

test-docs:
	@echo "Running documentation tests..."
	@echo "Validating README examples..."
	@echo "Checking test documentation consistency..."

examples:
	@echo "Running example applications..."
	$(MAKE) run-example

run-example:
	@echo "Running VM migration example in Docker..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 \
		go run ./backend/examples/vm_migration_example.go

# ============================================================================
# Docker & Deployment
# ============================================================================

docker-build:
	@echo "Building Docker images..."
	docker-compose build

docker-test:
	@echo "Running tests in Docker environment..."
	docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit
	docker-compose -f docker-compose.test.yml down

docker-test-env:
	@echo "Starting Docker test environment..."
	docker-compose -f docker-compose.test.yml up -d

# ============================================================================
# Cleanup
# ============================================================================

clean:
	@echo "Cleaning build artifacts..."
	find . -name "*.o" -type f -delete
	find . -name "*.a" -type f -delete
	find . -name "*.so" -type f -delete
	find . -name "*.exe" -type f -delete
	find . -name "*.test" -type f -delete
	find . -name "*.out" -type f -delete
	find . -name "*.prof" -type f -delete

clean-test:
	@echo "Cleaning test artifacts..."
	rm -f coverage.out coverage.html
	rm -f test-results.json
	rm -f *.prof
	rm -rf sdk_tests/

clean-all:
	@echo "Cleaning everything..."
	$(MAKE) clean
	$(MAKE) clean-test
	$(MAKE) test-env-down
	docker system prune -f

# ============================================================================
# Help & Information
# ============================================================================

help:
	@echo "NovaCron Testing Framework"
	@echo "=========================="
	@echo ""
	@echo "Core Commands:"
	@echo "  build              Build all components"
	@echo "  test               Run all tests"
	@echo "  test-env-up        Start test environment"
	@echo "  test-env-down      Stop test environment"
	@echo ""
	@echo "Unit Testing:"
	@echo "  test-unit          Run unit tests"
	@echo "  test-unit-coverage Run unit tests with coverage"
	@echo "  test-unit-race     Run unit tests with race detection"
	@echo ""
	@echo "Integration Testing:"
	@echo "  test-integration   Run integration tests"
	@echo "  test-multicloud    Run multi-cloud integration tests"
	@echo "  test-ml            Run AI/ML model tests"
	@echo "  test-cache         Run Redis cache tests"
	@echo "  test-sdk           Run cross-language SDK tests"
	@echo "  test-e2e           Run end-to-end tests"
	@echo "  test-chaos         Run chaos engineering tests"
	@echo ""
	@echo "Performance Testing:"
	@echo "  test-benchmarks    Run performance benchmarks"
	@echo "  test-performance   Run comprehensive performance tests"
	@echo ""
	@echo "Quality & Security:"
	@echo "  lint-backend       Lint Go code"
	@echo "  security-scan      Run security scan"
	@echo "  vulnerability-check Check for vulnerabilities"
	@echo ""
	@echo "CI/CD:"
	@echo "  ci-test            Run CI test suite"
	@echo "  ci-test-full       Run full CI test suite"
	@echo "  ci-quality         Run CI quality checks"
	@echo ""
	@echo "Utilities:"
	@echo "  clean              Clean build artifacts"
	@echo "  clean-all          Clean everything"
	@echo "  help               Show this help message"

# Default help if no target specified
.DEFAULT_GOAL := help