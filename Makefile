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
