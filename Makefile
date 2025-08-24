# NovaCron Makefile

.PHONY: all test build clean docker-build docker-test

all: build

# Build all components
build:
	@echo "Building NovaCron components..."
	docker-compose build

# Run tests in Docker
test:
	@echo "Running tests in Docker..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 go test ./backend/core/vm/...

# Run Go tests without Docker (requires local Go installation)
test-local:
	@echo "Running Go tests locally..."
	cd backend/core && go test -v ./vm/...

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	find . -name "*.o" -type f -delete
	find . -name "*.a" -type f -delete
	find . -name "*.so" -type f -delete
	find . -name "*.exe" -type f -delete
	find . -name "*.test" -type f -delete
	find . -name "*.out" -type f -delete

# Build Docker images
docker-build:
	@echo "Building Docker images..."
	docker-compose build

# Run the example in Docker
run-example:
	@echo "Running VM migration example in Docker..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 go run ./backend/examples/vm_migration_example.go

# Comprehensive testing targets
test-all: test-unit test-integration test-benchmarks
	@echo "All tests completed"

test-unit:
	@echo "Running unit tests..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 go test ./backend/core/vm/... -v -run "Test.*Fixed"

test-integration:
	@echo "Running integration tests..."
	docker run --rm -v $(PWD):/app -w /app \
		-e DB_URL="postgresql://postgres:postgres@postgres:5432/novacron" \
		golang:1.19 go test ./backend/tests/integration/... -v

test-benchmarks:
	@echo "Running performance benchmarks..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 go test ./backend/tests/benchmarks/... -bench=. -v

test-coverage:
	@echo "Generating test coverage report..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 \
		go test ./backend/core/... -coverprofile=coverage.out -covermode=atomic
	docker run --rm -v $(PWD):/app -w /app golang:1.19 \
		go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

test-race:
	@echo "Running tests with race detection..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 \
		go test ./backend/core/vm/... -race -v -run "Test.*Fixed"

test-memory:
	@echo "Running tests with memory profiling..."
	docker run --rm -v $(PWD):/app -w /app golang:1.19 \
		go test ./backend/core/vm/... -memprofile=mem.prof -v -run "Test.*Fixed"

lint-backend:
	@echo "Linting backend code..."
	docker run --rm -v $(PWD):/app -w /app golangci/golangci-lint:latest golangci-lint run ./backend/...

security-scan:
	@echo "Running security scan..."
	docker run --rm -v $(PWD):/app -w /app securecodewarrior/gosec:latest gosec ./backend/...

