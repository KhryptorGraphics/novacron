# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NovaCron is a distributed VM management system with advanced migration capabilities. It provides a robust platform for managing virtualized workloads across distributed nodes, with efficient WAN-optimized transfers, multi-driver support (KVM, containers), and resource-aware scheduling.

## Common Development Commands

### Backend Development

```bash
# Run tests in Docker (recommended - uses Go 1.19)
make test

# Run tests locally (requires Go 1.23+)
cd backend/core && go test -v ./...

# Test specific package
go test -v ./backend/core/vm/...

# Run benchmarks
go test -bench=. ./backend/core/scheduler/policy/...

# Run VM migration example
make run-example

# Build all components
make build

# Clean build artifacts
make clean

# Setup project (Linux/macOS)
./scripts/setup.sh

# Setup project (Windows)
./scripts/setup.ps1
```

### Frontend Development

```bash
# Install dependencies
cd frontend && npm install

# Start development server (port 3000)
npm run dev

# Build for production
npm run build

# Run production server
npm start

# Lint code
npm run lint

# Run tests (Jest configured but no tests present)
npm test
```

### Docker Development

```bash
# Build all services
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f [service-name]

# Services available:
# - postgres (database)
# - hypervisor (VM management)
# - api (REST/WebSocket APIs)
# - frontend (Web UI)
# - prometheus (metrics)
# - grafana (visualization)
```

## Architecture Overview

### Backend Structure

The backend is written in Go (1.23+ locally, 1.19 in Docker) and organized into core modules:

- **vm/**: VM lifecycle management, migration execution, and driver implementations (KVM, containers)
- **storage/**: Distributed storage with compression, deduplication, and encryption support
- **scheduler/**: Resource-aware and network-aware scheduling with policy engine
- **monitoring/**: Telemetry collection, analytics, and alerting
- **network/**: Overlay networking and protocol handling
- **auth/**: Authentication, RBAC, and multi-tenancy support

### Frontend Architecture

The frontend uses Next.js 13 with TypeScript and includes:

- App Router for routing
- Tailwind CSS for styling
- Radix UI and shadcn/ui for components
- React Query for data fetching
- WebSocket support for real-time updates
- Advanced visualization components for monitoring

### Key Integration Points

1. **API Communication**: Frontend connects to backend via REST API (port 8080) and WebSocket (port 8081)
2. **VM Drivers**: Abstraction layer supporting multiple virtualization technologies
3. **Storage Plugins**: Extensible storage backend system with registry
4. **Policy Engine**: Flexible constraint-based scheduling system

### Testing Strategy

- **Unit Tests**: Located alongside source files as `*_test.go`
- **Integration Tests**: Test service interactions and end-to-end flows
- **Benchmark Tests**: Performance testing for critical components
- **Mock Objects**: Available for VM managers, expressions, and storage

### Migration Implementation

The system supports three migration types:
1. **Cold Migration**: VM stopped, disk copied, VM started on destination
2. **Warm Migration**: Memory pre-copied while VM runs, brief pause for final sync
3. **Live Migration**: Minimal downtime with iterative memory copy

WAN optimizations include compression, delta sync, and bandwidth throttling for efficient cross-datacenter migrations.

### Important Configuration

- Database: PostgreSQL connection via `DB_URL` environment variable
- Authentication: JWT-based with `AUTH_SECRET` environment variable
- Storage paths: Configured via `STORAGE_PATH` for VM data
- Cluster communication: Nodes register with API service for coordination

### Development Patterns

- Use context.Context for cancellation and timeouts
- Return errors explicitly rather than panic
- Use interfaces for extensibility (drivers, providers, plugins)
- Implement health checks for all services
- Log structured data with appropriate levels (debug, info, warn, error)

### Service Ports

- Frontend: 3000
- API (REST): 8080
- API (WebSocket): 8081
- Prometheus: 9090
- Grafana: 3001 (maps to internal 3000)
- PostgreSQL: 5432 (internal)
- Hypervisor: 9000 (internal)

### Key Dependencies

- **Virtualization**: go-libvirt for KVM, containerd for containers
- **Storage**: klauspost/compress for compression
- **Auth**: golang-jwt/jwt for JWT tokens
- **Frontend**: Next.js, React Query, Tailwind CSS, Chart.js