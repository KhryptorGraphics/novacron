# NovaCron Comprehensive Test Suite Makefile

.PHONY: all test test-storage test-consensus test-vm test-performance test-chaos test-integration
.PHONY: test-quick test-full test-ci test-coverage clean help

# Default target
all: test

# Configuration
GO := go
TEST_TIMEOUT := 180m
COVERAGE_OUTPUT := coverage.out
COVERAGE_HTML := coverage.html
ARTIFACT_DIR := ./test-artifacts
VERBOSE := false

# Test execution targets

## Run all comprehensive tests
test: test-full

## Run quick smoke tests (non-performance)
test-quick:
	@echo "Running quick comprehensive tests..."
	$(GO) run run_comprehensive_tests.go \
		-timeout=30 \
		-short=true \
		-performance=false \
		-chaos=false \
		-verbose=$(VERBOSE) \
		-format=text

## Run full comprehensive test suite
test-full:
	@echo "Running full comprehensive test suite..."
	$(GO) run run_comprehensive_tests.go \
		-timeout=180 \
		-parallel=true \
		-concurrency=4 \
		-artifacts=true \
		-artifact-dir=$(ARTIFACT_DIR) \
		-verbose=$(VERBOSE) \
		-format=text \
		-output=comprehensive-test-report.txt

## Run storage-specific tests
test-storage:
	@echo "Running storage tiering and distributed storage tests..."
	$(GO) run run_comprehensive_tests.go \
		-storage=true \
		-consensus=false \
		-vm=false \
		-performance=false \
		-chaos=false \
		-integration=false \
		-timeout=60 \
		-verbose=$(VERBOSE)

## Run consensus-specific tests  
test-consensus:
	@echo "Running distributed consensus tests..."
	$(GO) run run_comprehensive_tests.go \
		-storage=false \
		-consensus=true \
		-vm=false \
		-performance=false \
		-chaos=false \
		-integration=false \
		-timeout=60 \
		-verbose=$(VERBOSE)

## Run VM lifecycle tests
test-vm:
	@echo "Running VM lifecycle tests..."
	$(GO) run run_comprehensive_tests.go \
		-storage=false \
		-consensus=false \
		-vm=true \
		-performance=false \
		-chaos=false \
		-integration=false \
		-timeout=60 \
		-verbose=$(VERBOSE)

## Run performance benchmarks only
test-performance:
	@echo "Running performance benchmarks..."
	$(GO) run run_comprehensive_tests.go \
		-storage=false \
		-consensus=false \
		-vm=false \
		-performance=true \
		-chaos=false \
		-integration=false \
		-timeout=120 \
		-verbose=$(VERBOSE) \
		-format=json \
		-output=performance-report.json

## Run chaos engineering tests
test-chaos:
	@echo "Running chaos engineering tests..."
	$(GO) run run_comprehensive_tests.go \
		-storage=false \
		-consensus=false \
		-vm=false \
		-performance=false \
		-chaos=true \
		-integration=false \
		-timeout=90 \
		-verbose=$(VERBOSE)

## Run integration tests
test-integration:
	@echo "Running integration tests..."
	$(GO) run run_comprehensive_tests.go \
		-storage=false \
		-consensus=false \
		-vm=false \
		-performance=false \
		-chaos=false \
		-integration=true \
		-timeout=90 \
		-verbose=$(VERBOSE)

## Run tests with coverage analysis
test-coverage:
	@echo "Running tests with coverage analysis..."
	$(GO) test -coverprofile=$(COVERAGE_OUTPUT) -coverpkg=./... ./...
	$(GO) tool cover -html=$(COVERAGE_OUTPUT) -o $(COVERAGE_HTML)
	@echo "Coverage report generated: $(COVERAGE_HTML)"
	$(GO) tool cover -func=$(COVERAGE_OUTPUT) | grep total:

## Run CI-optimized test suite
test-ci:
	@echo "Running CI-optimized test suite..."
	$(GO) run run_comprehensive_tests.go \
		-env=ci \
		-timeout=150 \
		-parallel=true \
		-concurrency=6 \
		-min-success=98.0 \
		-min-coverage=92.0 \
		-max-critical=0 \
		-artifacts=true \
		-artifact-dir=$(ARTIFACT_DIR) \
		-format=json \
		-output=ci-test-report.json \
		-cleanup=true

## Run production readiness tests
test-production:
	@echo "Running production readiness tests..."
	$(GO) run run_comprehensive_tests.go \
		-env=production \
		-real-resources=true \
		-timeout=300 \
		-min-success=99.5 \
		-min-coverage=95.0 \
		-max-critical=0 \
		-artifacts=true \
		-format=xml \
		-output=production-readiness-report.xml

# Development and debugging targets

## Run tests with verbose output
test-verbose:
	$(MAKE) test-full VERBOSE=true

## Run individual test packages
test-unit-storage:
	$(GO) test -v ./storage/...

test-unit-consensus:
	$(GO) test -v ./consensus/...

test-unit-vm:
	$(GO) test -v ./vm_lifecycle/...

test-unit-integration:
	$(GO) test -v ./integration/...

## Benchmark specific components
benchmark-storage:
	$(GO) test -bench=. -benchmem ./storage/...

benchmark-consensus:
	$(GO) test -bench=. -benchmem ./consensus/...

benchmark-vm:
	$(GO) test -bench=. -benchmem ./vm_lifecycle/...

benchmark-integration:
	$(GO) test -bench=. -benchmem ./integration/...

# Utility targets

## Validate test code quality
validate:
	@echo "Validating test code quality..."
	$(GO) vet ./...
	golint ./...
	gofmt -s -l .
	@echo "✅ Test code validation passed"

## Generate test documentation
docs:
	@echo "Generating test documentation..."
	godoc -http=:6060 &
	@echo "📚 Documentation server started at http://localhost:6060"

## Clean test artifacts and temporary files
clean:
	@echo "Cleaning test artifacts..."
	rm -rf $(ARTIFACT_DIR)
	rm -f $(COVERAGE_OUTPUT) $(COVERAGE_HTML)
	rm -f *-test-report.*
	rm -f *.log
	@echo "✅ Test artifacts cleaned"

## Setup test environment
setup:
	@echo "Setting up test environment..."
	mkdir -p $(ARTIFACT_DIR)
	$(GO) mod download
	$(GO) mod tidy
	@echo "✅ Test environment setup complete"

## Display available targets
help:
	@echo "NovaCron Comprehensive Test Suite"
	@echo "================================="
	@echo ""
	@echo "Main test targets:"
	@echo "  test           - Run all comprehensive tests (alias for test-full)"
	@echo "  test-quick     - Run quick smoke tests (30min timeout)"
	@echo "  test-full      - Run complete test suite (180min timeout)"
	@echo ""
	@echo "Component-specific tests:"
	@echo "  test-storage   - Storage tiering and distributed storage tests"
	@echo "  test-consensus - Distributed consensus tests"
	@echo "  test-vm        - VM lifecycle tests"
	@echo "  test-performance - Performance benchmarks"
	@echo "  test-chaos     - Chaos engineering tests"
	@echo "  test-integration - Integration tests"
	@echo ""
	@echo "Environment-specific tests:"
	@echo "  test-ci        - CI/CD optimized test suite"
	@echo "  test-production - Production readiness tests"
	@echo ""
	@echo "Analysis and debugging:"
	@echo "  test-coverage  - Run tests with coverage analysis"
	@echo "  test-verbose   - Run tests with verbose output"
	@echo "  benchmark-*    - Run benchmarks for specific components"
	@echo ""
	@echo "Utilities:"
	@echo "  validate       - Validate test code quality"
	@echo "  clean          - Clean test artifacts"
	@echo "  setup          - Setup test environment"
	@echo "  docs           - Generate test documentation"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Configuration variables:"
	@echo "  VERBOSE=true   - Enable verbose output"
	@echo "  TEST_TIMEOUT   - Test timeout duration (default: 180m)"
	@echo "  ARTIFACT_DIR   - Test artifact directory (default: ./test-artifacts)"
	@echo ""
	@echo "Examples:"
	@echo "  make test-quick"
	@echo "  make test-full VERBOSE=true"
	@echo "  make test-storage"
	@echo "  make test-ci"
	@echo "  make benchmark-storage"

# Quality gates and reporting
.PHONY: quality-gate report

## Check quality gates
quality-gate:
	@echo "Checking quality gates..."
	$(GO) run run_comprehensive_tests.go \
		-timeout=180 \
		-min-success=95.0 \
		-min-coverage=90.0 \
		-max-critical=0 \
		-format=json \
		-output=quality-gate-report.json
	@echo "Quality gate evaluation complete"

## Generate comprehensive report
report:
	@echo "Generating comprehensive test report..."
	$(GO) run run_comprehensive_tests.go \
		-timeout=180 \
		-artifacts=true \
		-format=json \
		-output=comprehensive-report.json
	@echo "📊 Comprehensive report generated: comprehensive-report.json"

# Docker support
.PHONY: test-docker docker-build docker-clean

## Build test Docker image
docker-build:
	docker build -t novacron-comprehensive-tests -f Dockerfile.test .

## Run tests in Docker container
test-docker: docker-build
	docker run --rm -v $(PWD)/test-artifacts:/app/test-artifacts \
		novacron-comprehensive-tests make test-ci

## Clean Docker test resources
docker-clean:
	docker rmi novacron-comprehensive-tests || true
	docker system prune -f