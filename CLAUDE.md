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

# Start development server (port 8092)
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

# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose logs -f [service-name]

# Services available:
# - postgres (database, port 5432)
# - hypervisor (VM management, port 9000)
# - api (REST/WebSocket APIs, ports 8090/8091)
# - frontend (Web UI, port 8092)
# - prometheus (metrics, port 9090)
# - grafana (visualization, port 3001)
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

1. **API Communication**: Frontend connects to backend via REST API (port 8090) and WebSocket (port 8091)
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

- Frontend: 8092
- API (REST): 8090
- API (WebSocket): 8091
- Prometheus: 9090
- Grafana: 3001 (maps to internal 3000)
- PostgreSQL: 5432 (internal)
- Hypervisor: 9000 (internal)

### Key Dependencies

- **Virtualization**: go-libvirt for KVM, containerd for containers
- **Storage**: klauspost/compress for compression
- **Auth**: golang-jwt/jwt for JWT tokens
- **Frontend**: Next.js, React Query, Tailwind CSS, Chart.js

### Common Issues & Solutions

- **KVM Connection Issues**: The API server can start without KVM for development. Check libvirt daemon status if needed.
- **Docker Compose**: Use `-f docker-compose.dev.yml` for development with hot reloading
- **Go Version**: Core backend uses Go 1.23+, Docker environment uses Go 1.19
- **Frontend Port**: Development server runs on port 8092, not the default 3000

### API Integration Points

The system has two main API servers:
1. **Main API Server** (`backend/cmd/api-server/main.go`): REST endpoints and WebSocket
2. **Core Backend** (`backend/core/`): Business logic modules

Frontend connects via:
- REST API: `http://localhost:8090/api/*`
- WebSocket: `ws://localhost:8091/ws/*`

### Critical File Locations

- **Main API Entry**: `backend/cmd/api-server/main.go`
- **Core VM Logic**: `backend/core/vm/`
- **Frontend Dashboard**: `frontend/src/components/monitoring/MonitoringDashboard.tsx`
- **Docker Configs**: `docker-compose.yml` (production), `docker-compose.dev.yml` (development)
- **Setup Scripts**: `scripts/setup.sh` (Linux/macOS), `scripts/setup.ps1` (Windows)