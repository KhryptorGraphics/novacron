# NovaCron Development Commands

## Backend Development
```bash
# Run backend API server
cd backend/cmd/api-server
go run main.go

# Build backend
go build -o novacron-api backend/cmd/api-server/main.go

# Test backend
go test ./backend/...
make test-unit
make test-integration
```

## Frontend Development
```bash
# Run frontend dev server
cd frontend
npm run dev

# Build frontend
npm run build

# Test frontend
npm test
npm run test:coverage
```

## Full Stack Development
```bash
# Start both backend and frontend (Windows)
.\start_development.ps1

# Start both backend and frontend (Linux/macOS)
./start_development.sh

# Docker development
docker-compose -f docker-compose.dev.yml up
```

## Testing
```bash
# Run all tests
make test

# Unit tests with coverage
make test-unit-coverage

# Integration tests
make test-integration

# Performance benchmarks
make test-benchmarks

# E2E tests
make test-e2e
```

## Code Quality
```bash
# Lint backend
make lint-backend

# Security scan
make security-scan

# Frontend lint
cd frontend && npm run lint
```

## Database
```bash
# Run migrations
make db-migrate

# Reset database
make db-reset

# Database console
make db-console
```