# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NovaCron is a distributed virtual machine management platform with advanced migration, real-time monitoring, intelligent scheduling, multi-cloud orchestration, and enterprise security. Built on Go 1.24+ backend and Next.js 13.5 frontend.

## Build & Development Commands

### Backend (Go)
```bash
# Build core backend (orchestration subset)
make core-build

# Run core unit tests
make core-test

# Start core server on :8090
make core-serve

# Run all tests (comprehensive)
make test

# Run tests in Docker (recommended for CI consistency)
make test-docker

# Run with coverage
make test-unit-coverage

# Run specific test categories
make test-ml           # AI/ML model tests
make test-cache        # Redis cache tests
make test-multicloud   # Multi-cloud integration
make test-e2e          # End-to-end tests
make test-chaos        # Chaos engineering tests
```

### Frontend (Next.js)
```bash
cd frontend
npm run dev           # Start dev server on :8092
npm run build         # Production build
npm run lint          # ESLint
npm run test          # Jest unit tests
npm run test:watch    # Jest watch mode
npm run test:coverage # With coverage report
npm run test:e2e      # Puppeteer E2E tests
```

### Database
```bash
make db-migrate       # Run migrations
make db-rollback      # Rollback last migration
make db-seed          # Seed dev data
make db-reset         # Drop, migrate, seed
make db-test-setup    # Setup test database
```

### Docker
```bash
docker-compose up -d                    # Start all services
docker-compose -f docker-compose.test.yml up -d  # Test environment
make test-env-up                        # Start test env
make test-env-down                      # Stop test env
```

## Architecture

### Backend Structure (`backend/`)
```
backend/
├── api/                  # HTTP handlers organized by domain
│   ├── vm/              # VM lifecycle, migration, metrics handlers
│   ├── auth/            # Authentication routes & middleware
│   ├── backup/          # Backup/restore handlers
│   ├── federation/      # Multi-cluster federation
│   ├── graphql/         # GraphQL API
│   ├── websocket/       # Real-time WebSocket handlers
│   └── admin/           # Admin management
├── core/                 # Business logic (separate Go module)
│   ├── vm/              # VM types, predictive prefetching
│   ├── auth/            # JWT, OAuth2, 2FA, zero-trust
│   ├── backup/          # Backup manager, CBT, dedup
│   ├── cache/           # Multi-tier cache (L1 memory, L2 Redis)
│   ├── consensus/       # Raft implementation, distributed locks
│   ├── federation/      # Multi-cloud orchestration
│   ├── hypervisor/      # KVM/libvirt integration
│   ├── migration/       # Live migration orchestrator
│   ├── ml/              # ML predictor, anomaly detection
│   ├── monitoring/      # Telemetry, Prometheus, dashboards
│   ├── network/         # SDN, load balancing
│   ├── scheduler/       # Resource scheduling policies
│   └── orchestration/   # Container/VM orchestration
└── cmd/
    ├── api-server/      # Main API entry point
    └── core-server/     # Minimal core server
```

### Frontend Structure (`frontend/`)
```
frontend/src/
├── app/                 # Next.js App Router pages
├── components/          # React components (Radix UI based)
├── lib/                 # Utilities, API clients
├── hooks/               # Custom React hooks
├── contexts/            # React contexts
└── providers/           # Provider wrappers
```

### Key Go Modules (replace directives in go.mod)
- `github.com/khryptorgraphics/novacron/backend/core` → `./backend/core`
- `github.com/khryptorgraphics/novacron/backend/core/orchestration` → `./backend/core/orchestration`
- `github.com/khryptorgraphics/novacron/backend/pkg/logger` → `./backend/pkg/logger`

## Service Ports
| Service | Port |
|---------|------|
| API (REST) | 8090 |
| API (WebSocket) | 8091 |
| Frontend | 8092 |
| AI Engine | 8093 |
| PostgreSQL | 11432 (external) / 5432 (internal) |
| Redis | 6379 |
| Prometheus | 9090 |
| Grafana | 3001 |

## Key Dependencies
- **Backend**: gorilla/mux, gorilla/websocket, lib/pq, redis/go-redis, prometheus/client_golang, go-libvirt, k8s.io/client-go
- **Frontend**: Next.js 13.5, React 18.2, Radix UI, TanStack Query/Table, Chart.js, D3.js, Jotai, react-use-websocket, Tailwind CSS

## Testing Patterns
- Unit tests use `*_test.go` suffix, run with `go test -v -run "Test.*"`
- Integration tests require Docker (`make test-integration-setup` first)
- ML tests: `make test-ml-accuracy`, `make test-ml-drift`
- Cache tests require Redis: `make test-cache-performance`
- Frontend tests use Jest + Testing Library, E2E uses Puppeteer

## File Organization Rules
- **Never save working files to root folder**
- Source code → `/backend`, `/frontend/src`
- Tests → `/backend/tests`, `/frontend/src/__tests__`
- Documentation → `/docs`
- Configuration → `/configs`, `/config`
- Scripts → `/scripts`

## Important Patterns
- Backend uses multi-tier caching (L1 BigCache, L2 Redis)
- Authentication uses JWT with optional OAuth2 and 2FA
- Real-time updates via WebSocket on :8091
- GraphQL available alongside REST API
- Database migrations in `database/` using SQL scripts
- Security-hardened Docker containers (non-root, read-only, cap_drop ALL)

## AI/Claude-Flow Integration

This project supports Claude-Flow orchestration for multi-agent development:

```bash
# SPARC methodology commands
npx claude-flow sparc run <mode> "<task>"
npx claude-flow sparc tdd "<feature>"
```

When using Claude Code's Task tool for agent spawning, batch all operations in single messages. MCP tools coordinate strategy; Claude Code's Task tool executes.
