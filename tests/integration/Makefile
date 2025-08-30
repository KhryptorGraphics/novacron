# Integration Test Makefile for NovaCron
.PHONY: all test clean help setup teardown

# Test configuration
TIMEOUT ?= 30m
VERBOSE ?= -v
TEST_PATTERN ?= ./...
INTEGRATION_TAGS ?= integration

# Environment configuration
DB_URL ?= postgresql://postgres:postgres@localhost:5432/novacron_test
REDIS_URL ?= redis://localhost:6379
NOVACRON_API_URL ?= http://localhost:8090
NOVACRON_WS_URL ?= ws://localhost:8091/ws
NOVACRON_UI_URL ?= http://localhost:8092

# Go test flags
GO_TEST_FLAGS := $(VERBOSE) -timeout $(TIMEOUT) -tags $(INTEGRATION_TAGS)

# Default target
all: test

# Run all integration tests
test: setup
	@echo "Running integration tests..."
	@echo "Configuration:"
	@echo "  Database URL: $(DB_URL)"
	@echo "  API URL: $(NOVACRON_API_URL)"
	@echo "  WebSocket URL: $(NOVACRON_WS_URL)"
	@echo "  UI URL: $(NOVACRON_UI_URL)"
	@echo ""
	DB_URL="$(DB_URL)" \
	REDIS_URL="$(REDIS_URL)" \
	NOVACRON_API_URL="$(NOVACRON_API_URL)" \
	NOVACRON_WS_URL="$(NOVACRON_WS_URL)" \
	NOVACRON_UI_URL="$(NOVACRON_UI_URL)" \
	go test $(GO_TEST_FLAGS) $(TEST_PATTERN)

# Run specific test suites
test-vm-lifecycle:
	@echo "Running VM lifecycle tests..."
	DB_URL="$(DB_URL)" \
	NOVACRON_API_URL="$(NOVACRON_API_URL)" \
	go test $(GO_TEST_FLAGS) -run TestVMLifecycleTestSuite

test-auth:
	@echo "Running authentication tests..."
	DB_URL="$(DB_URL)" \
	NOVACRON_API_URL="$(NOVACRON_API_URL)" \
	go test $(GO_TEST_FLAGS) -run TestAuthTestSuite

test-storage:
	@echo "Running storage tests..."
	DB_URL="$(DB_URL)" \
	NOVACRON_API_URL="$(NOVACRON_API_URL)" \
	go test $(GO_TEST_FLAGS) -run TestStorageTestSuite

test-consensus:
	@echo "Running consensus tests..."
	DB_URL="$(DB_URL)" \
	NOVACRON_API_URL="$(NOVACRON_API_URL)" \
	go test $(GO_TEST_FLAGS) -run TestConsensusTestSuite

test-performance:
	@echo "Running performance tests..."
	DB_URL="$(DB_URL)" \
	NOVACRON_API_URL="$(NOVACRON_API_URL)" \
	go test $(GO_TEST_FLAGS) -run TestPerformanceTestSuite

test-e2e:
	@echo "Running end-to-end tests..."
	DB_URL="$(DB_URL)" \
	NOVACRON_API_URL="$(NOVACRON_API_URL)" \
	NOVACRON_WS_URL="$(NOVACRON_WS_URL)" \
	go test $(GO_TEST_FLAGS) -run TestE2ETestSuite

# Run tests with coverage
test-coverage:
	@echo "Running integration tests with coverage..."
	DB_URL="$(DB_URL)" \
	REDIS_URL="$(REDIS_URL)" \
	NOVACRON_API_URL="$(NOVACRON_API_URL)" \
	NOVACRON_WS_URL="$(NOVACRON_WS_URL)" \
	NOVACRON_UI_URL="$(NOVACRON_UI_URL)" \
	go test $(GO_TEST_FLAGS) -coverprofile=integration-coverage.out $(TEST_PATTERN)
	go tool cover -html=integration-coverage.out -o integration-coverage.html
	@echo "Coverage report generated: integration-coverage.html"

# Run tests in short mode (skip slow tests)
test-short:
	@echo "Running integration tests in short mode..."
	DB_URL="$(DB_URL)" \
	NOVACRON_API_URL="$(NOVACRON_API_URL)" \
	go test $(GO_TEST_FLAGS) -short $(TEST_PATTERN)

# Run specific test by name
test-run:
	@read -p "Enter test name pattern: " pattern; \
	echo "Running tests matching: $$pattern"; \
	DB_URL="$(DB_URL)" \
	NOVACRON_API_URL="$(NOVACRON_API_URL)" \
	go test $(GO_TEST_FLAGS) -run "$$pattern"

# Setup test environment
setup:
	@echo "Setting up integration test environment..."
	@echo "Checking Go modules..."
	@go mod tidy
	@echo "Setup completed"

# Teardown test environment
teardown:
	@echo "Tearing down integration test environment..."
	@echo "Cleaning up temporary files..."
	@rm -f *.out *.html
	@echo "Teardown completed"

# Clean up test artifacts
clean:
	@echo "Cleaning integration test artifacts..."
	@rm -f *.out *.html *.log
	@go clean -testcache
	@echo "Clean completed"

# Install test dependencies
deps:
	@echo "Installing test dependencies..."
	@go mod download
	@go mod verify
	@echo "Dependencies installed"

# Run linting on test code
lint:
	@echo "Linting integration test code..."
	@if command -v golangci-lint >/dev/null 2>&1; then \
		golangci-lint run ./...; \
	else \
		echo "golangci-lint not found, skipping lint"; \
	fi

# Format test code
fmt:
	@echo "Formatting integration test code..."
	@go fmt ./...
	@echo "Formatting completed"

# Run tests in Docker environment
test-docker:
	@echo "Running integration tests in Docker..."
	docker run --rm \
		-v $(PWD):/app \
		-w /app \
		--network host \
		-e DB_URL="$(DB_URL)" \
		-e REDIS_URL="$(REDIS_URL)" \
		-e NOVACRON_API_URL="$(NOVACRON_API_URL)" \
		-e NOVACRON_WS_URL="$(NOVACRON_WS_URL)" \
		-e NOVACRON_UI_URL="$(NOVACRON_UI_URL)" \
		golang:1.21 \
		go test $(GO_TEST_FLAGS) $(TEST_PATTERN)

# Profile test execution
test-profile:
	@echo "Running integration tests with profiling..."
	DB_URL="$(DB_URL)" \
	NOVACRON_API_URL="$(NOVACRON_API_URL)" \
	go test $(GO_TEST_FLAGS) -cpuprofile=cpu.prof -memprofile=mem.prof $(TEST_PATTERN)
	@echo "Profiles generated: cpu.prof, mem.prof"

# Benchmark integration tests
bench:
	@echo "Running integration benchmarks..."
	DB_URL="$(DB_URL)" \
	NOVACRON_API_URL="$(NOVACRON_API_URL)" \
	go test -bench=. -benchmem $(TEST_PATTERN)

# Generate test report
report:
	@echo "Generating integration test report..."
	DB_URL="$(DB_URL)" \
	NOVACRON_API_URL="$(NOVACRON_API_URL)" \
	go test $(GO_TEST_FLAGS) -json $(TEST_PATTERN) > test-report.json
	@echo "Test report generated: test-report.json"

# Watch tests (requires entr)
watch:
	@echo "Watching for changes and running tests..."
	@if command -v entr >/dev/null 2>&1; then \
		find . -name "*.go" | entr -r make test-short; \
	else \
		echo "entr not found. Install with: apt-get install entr"; \
	fi

# Debug test environment
debug:
	@echo "=== Integration Test Debug Information ==="
	@echo "Go version: $$(go version)"
	@echo "Go modules:"
	@go mod list -m all
	@echo ""
	@echo "Environment variables:"
	@echo "  DB_URL: $(DB_URL)"
	@echo "  REDIS_URL: $(REDIS_URL)"
	@echo "  NOVACRON_API_URL: $(NOVACRON_API_URL)"
	@echo "  NOVACRON_WS_URL: $(NOVACRON_WS_URL)"
	@echo "  NOVACRON_UI_URL: $(NOVACRON_UI_URL)"
	@echo ""
	@echo "Test configuration:"
	@echo "  Timeout: $(TIMEOUT)"
	@echo "  Verbose: $(VERBOSE)"
	@echo "  Pattern: $(TEST_PATTERN)"
	@echo "  Tags: $(INTEGRATION_TAGS)"

# Help target
help:
	@echo "NovaCron Integration Test Suite"
	@echo "==============================="
	@echo ""
	@echo "Core Commands:"
	@echo "  test              Run all integration tests"
	@echo "  test-short        Run tests in short mode (skip slow tests)"
	@echo "  test-coverage     Run tests with coverage report"
	@echo "  clean             Clean test artifacts"
	@echo ""
	@echo "Test Suites:"
	@echo "  test-vm-lifecycle Run VM lifecycle tests"
	@echo "  test-auth         Run authentication tests"
	@echo "  test-storage      Run storage tests"
	@echo "  test-consensus    Run consensus tests"
	@echo "  test-performance  Run performance tests"
	@echo "  test-e2e          Run end-to-end tests"
	@echo ""
	@echo "Development:"
	@echo "  setup             Setup test environment"
	@echo "  teardown          Teardown test environment"
	@echo "  deps              Install dependencies"
	@echo "  fmt               Format code"
	@echo "  lint              Run linters"
	@echo "  watch             Watch and run tests on changes"
	@echo ""
	@echo "Advanced:"
	@echo "  test-docker       Run tests in Docker"
	@echo "  test-profile      Run tests with profiling"
	@echo "  bench             Run benchmarks"
	@echo "  report            Generate test report"
	@echo "  debug             Show debug information"
	@echo ""
	@echo "Configuration (environment variables):"
	@echo "  DB_URL            Database connection URL"
	@echo "  REDIS_URL         Redis connection URL"
	@echo "  NOVACRON_API_URL  API server URL"
	@echo "  NOVACRON_WS_URL   WebSocket server URL"
	@echo "  NOVACRON_UI_URL   Frontend server URL"
	@echo "  TIMEOUT           Test timeout (default: 30m)"
	@echo "  VERBOSE           Verbose output (default: -v)"
	@echo "  TEST_PATTERN      Test pattern (default: ./...)"

# Default help target
.DEFAULT_GOAL := help