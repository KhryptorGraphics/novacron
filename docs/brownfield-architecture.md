# NovaCron Brownfield Architecture Document

## Introduction

This document captures the CURRENT STATE of the NovaCron codebase, including technical debt, workarounds, and real-world patterns. It serves as a reference for AI agents working on enhancements.

NovaCron is a sophisticated distributed virtual machine management platform that is approximately 85% complete, with production-ready core functionality and an advanced monitoring dashboard.

### Document Scope
Comprehensive documentation of the entire NovaCron system, focusing on the existing implementation patterns and architecture.

### Change Log
| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2024 | 1.0 | Initial brownfield analysis | BMad Master |

## Quick Reference - Key Files and Entry Points

### Critical Files for Understanding the System

- **Backend Entry**: `backend/cmd/api-server/main.go` - Main API server
- **Configuration**: `backend/configs/`, environment-specific configs
- **Core Business Logic**: `backend/core/` - All core services
- **API Definitions**: `backend/api/` - REST, GraphQL, WebSocket handlers
- **Database Models**: Embedded in service files (no ORM)
- **Key Algorithms**: 
  - `backend/core/scheduler/optimization_expert.go` - Scheduling algorithms
  - `backend/core/migration/live_migration.go` - Migration logic
  - `backend/core/orchestration/engine.go` - Orchestration engine

### Frontend Critical Files
- **Main Entry**: `frontend/src/app/layout.tsx` - Root layout
- **Dashboard**: `frontend/src/app/dashboard/page.tsx` - Main dashboard
- **API Client**: `frontend/src/lib/api/client.ts` - API communication
- **WebSocket**: Real-time updates via `react-use-websocket`

## High Level Architecture

### Technical Summary
Microservices architecture with Go backend and Next.js frontend, using PostgreSQL for persistence and Redis for caching/pub-sub. The system manages VMs across multiple hypervisors and cloud providers.

### Actual Tech Stack (from package.json/go.mod)

| Category | Technology | Version | Notes |
|----------|-----------|---------|-------|
| Backend Runtime | Go | 1.21+ | Native compilation |
| Frontend Runtime | Node.js | 18.x | LTS version |
| Backend Framework | Gorilla/mux | 1.8.0 | HTTP routing |
| Frontend Framework | Next.js | 13.5.6 | App Router enabled |
| UI Library | React | 18.2.0 | With TypeScript |
| Database | PostgreSQL | 15.x | Primary datastore |
| Cache | Redis | 7.x | Sessions & pub/sub |
| State Management | Jotai | 2.6.0 | Atomic state |
| API Client | Axios | 1.6.2 | HTTP client |
| UI Components | Radix UI | Latest | Accessible components |
| Styling | Tailwind CSS | 3.x | Utility-first CSS |
| Charts | Chart.js | 4.4.1 | Data visualization |

### Repository Structure Reality Check
- Type: **Monorepo** - Single repository for backend and frontend
- Package Manager: **npm** for frontend, **go mod** for backend
- Notable: Separate builds but shared deployment scripts

## Source Tree and Module Organization

### Project Structure (Actual)

```text
novacron/
├── backend/                 # Go backend services
│   ├── cmd/                # Entry points for different services
│   │   ├── api-server/     # Main API server (PORT 8080)
│   │   ├── core-server/    # Core services server
│   │   └── workers/        # Background workers
│   ├── api/                # HTTP handlers and routes
│   │   ├── rest/          # REST endpoints
│   │   ├── graphql/       # GraphQL resolvers
│   │   ├── websocket/     # WebSocket handlers
│   │   └── vm/            # VM-specific handlers (LARGE - needs refactoring)
│   ├── core/              # Core business logic
│   │   ├── vm/            # VM management
│   │   ├── migration/     # Migration logic (complex state machine)
│   │   ├── monitoring/    # Metrics collection
│   │   ├── orchestration/ # Orchestration engine
│   │   ├── scheduler/     # Placement algorithms
│   │   ├── storage/       # Storage management
│   │   ├── backup/        # Backup services
│   │   ├── federation/    # Multi-cluster support
│   │   ├── ml/            # ML predictions (experimental)
│   │   └── security/      # Security services
│   ├── pkg/               # Shared packages
│   │   ├── middleware/    # HTTP middleware
│   │   ├── database/      # DB utilities
│   │   └── utils/         # Common utilities
│   └── tests/             # Test files (coverage ~60%)
├── frontend/              # Next.js frontend
│   ├── src/
│   │   ├── app/          # App Router pages
│   │   ├── components/   # React components
│   │   └── lib/          # Utilities and hooks
│   └── .next/            # Build output (DO NOT EDIT)
├── docs/                  # Documentation
├── scripts/               # Deployment and utility scripts
├── docker/                # Docker configurations
└── .claude/               # AI agent configurations
```

### Key Modules and Their Purpose

#### Backend Services
- **VM Manager**: `backend/core/vm/vm_manager.go` - Central VM lifecycle management
- **Migration Engine**: `backend/core/migration/` - Complex live migration with multiple strategies
- **Orchestration Engine**: `backend/core/orchestration/engine.go` - Event-driven orchestration
- **Scheduler**: `backend/core/scheduler/` - Multiple scheduling algorithms (bin packing, spread, etc.)
- **Monitoring**: `backend/core/monitoring/` - Prometheus integration, custom collectors
- **Federation Manager**: `backend/core/federation/` - Multi-cluster coordination (partially implemented)

#### Frontend Components
- **Dashboard**: `frontend/src/app/dashboard/` - Main monitoring interface
- **VM Management**: `frontend/src/app/vms/` - VM CRUD operations
- **Monitoring**: `frontend/src/app/monitoring/` - Real-time metrics display
- **API Client**: `frontend/src/lib/api/` - Centralized API communication

## Data Models and APIs

### Data Models
Database schema defined in raw SQL (no ORM):
- See migration files in `backend/migrations/` (if exists)
- Primary models embedded in service files
- JSONB used extensively for flexible attributes

### API Specifications
- **REST API**: Defined in `backend/api/rest/handlers.go`
- **GraphQL Schema**: `backend/api/graphql/schema.graphql` (if exists)
- **WebSocket Events**: Real-time updates for VM status, metrics
- **No OpenAPI Spec**: API documentation needs to be generated

### Key API Endpoints
```
GET    /api/v1/vms              - List VMs
POST   /api/v1/vms              - Create VM
GET    /api/v1/vms/:id          - Get VM details
PUT    /api/v1/vms/:id          - Update VM
DELETE /api/v1/vms/:id          - Delete VM
POST   /api/v1/vms/:id/migrate  - Initiate migration
GET    /api/v1/metrics          - Get metrics
WS     /api/v1/ws              - WebSocket connection
```

## Technical Debt and Known Issues

### Critical Technical Debt

1. **VM Handlers**: `backend/api/vm/` directory is too large (20+ files), needs splitting
2. **Error Handling**: Inconsistent error handling between services
3. **Database Migrations**: No formal migration system, manual SQL scripts
4. **Federation**: Partially implemented, many TODO comments
5. **ML Integration**: Experimental code in `backend/core/ml/`, not production-ready
6. **Test Coverage**: Only ~60% coverage, integration tests minimal
7. **Frontend State**: Some components use local state instead of Jotai
8. **Type Safety**: Some API responses not fully typed in TypeScript

### Workarounds and Gotchas

- **Port Conflicts**: Backend hardcoded to 8080, frontend to 3000
- **CORS Issues**: Must run both services locally for development
- **Database Connections**: Connection pool size affects migration performance
- **Redis Required**: Even for local development, Redis must be running
- **Build Order**: Must build backend before frontend for type generation
- **WebSocket Reconnection**: Manual reconnection logic needed
- **Metrics Collection**: Prometheus scraping can impact performance

## Integration Points and External Dependencies

### External Services

| Service | Purpose | Integration Type | Key Files |
|---------|---------|-----------------|-----------|
| libvirt | KVM management | CGO bindings | `backend/core/hypervisor/kvm_manager.go` |
| AWS SDK | EC2 management | SDK | `backend/core/cloud/aws/` |
| Prometheus | Metrics | HTTP scraping | `backend/core/monitoring/prometheus/` |
| PostgreSQL | Database | pq driver | Throughout backend |
| Redis | Cache/PubSub | go-redis | `backend/core/cache/` |

### Internal Integration Points
- **Frontend-Backend**: REST API on :8080, WebSocket for real-time
- **Service Communication**: Internal gRPC (partially implemented)
- **Event Bus**: Redis pub/sub for service events
- **Metrics Pipeline**: Prometheus → Grafana (external)

## Development and Deployment

### Local Development Setup

1. **Prerequisites**:
   ```bash
   # Install Go 1.21+
   # Install Node.js 18+
   # Install PostgreSQL 15+
   # Install Redis 7+
   ```

2. **Backend Setup**:
   ```bash
   cd backend
   go mod download
   go run cmd/api-server/main.go
   ```

3. **Frontend Setup**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

4. **Known Setup Issues**:
   - PostgreSQL must have UUID extension
   - Redis must be on default port 6379
   - CORS issues if backend/frontend on different hosts

### Build and Deployment Process

- **Backend Build**: `go build -o novacron cmd/api-server/main.go`
- **Frontend Build**: `npm run build` (Next.js production build)
- **Docker**: Dockerfiles in `docker/` directory
- **Deployment**: Manual via scripts in `scripts/` directory
- **No CI/CD**: GitHub Actions files exist but not fully configured

## Testing Reality

### Current Test Coverage
- **Backend**: ~60% coverage (go test)
- **Frontend**: Minimal tests, mostly component snapshots
- **Integration Tests**: Few, in `backend/tests/`
- **E2E Tests**: None implemented
- **Manual Testing**: Primary QA method

### Running Tests
```bash
# Backend tests
cd backend
go test ./...
go test -cover ./...

# Frontend tests
cd frontend
npm test
npm run test:coverage
```

## Security Considerations

### Authentication & Authorization
- **JWT-based**: Custom implementation in `backend/core/auth/`
- **RBAC**: Role-based access control partially implemented
- **Session Management**: Redis-backed sessions
- **API Keys**: For service-to-service communication

### Known Security Issues
- **Secrets Management**: Currently using environment variables
- **TLS**: Not enforced for internal communication
- **Input Validation**: Inconsistent across endpoints
- **Audit Logging**: Minimal implementation

## Performance Characteristics

### Bottlenecks
- **VM List Endpoint**: Slow with >1000 VMs (no pagination)
- **Migration**: Memory-intensive for large VMs
- **Metrics Collection**: Can impact API performance
- **Frontend Bundle**: Large (~2MB), needs optimization

### Optimization Opportunities
- Implement pagination for list endpoints
- Add caching layer for read-heavy operations
- Optimize database queries (many N+1 issues)
- Enable Next.js ISR for static content

## Appendix - Useful Commands and Scripts

### Frequently Used Commands

```bash
# Development
make dev           # Start both backend and frontend
make backend       # Start backend only
make frontend      # Start frontend only
make test          # Run all tests
make lint          # Run linters

# Database
make db-migrate    # Run migrations
make db-seed       # Seed test data
make db-reset      # Reset database

# Deployment
./scripts/deploy.sh staging  # Deploy to staging
./scripts/deploy.sh prod     # Deploy to production
```

### Debugging and Troubleshooting

- **Backend Logs**: Check console output or `logs/api.log`
- **Frontend Logs**: Browser console and Next.js terminal output
- **Debug Mode**: Set `DEBUG=true` environment variable
- **Common Issues**:
  - Port already in use: Kill existing processes
  - Database connection: Check PostgreSQL is running
  - Redis connection: Ensure Redis is started

## Areas Needing Immediate Attention

1. **Test Coverage**: Increase to >80% before production
2. **Error Handling**: Standardize across all services
3. **Documentation**: Generate OpenAPI spec for REST API
4. **Performance**: Implement pagination and caching
5. **Security**: Implement proper secrets management
6. **Monitoring**: Complete Grafana dashboard setup
7. **CI/CD**: Finish GitHub Actions configuration

## Migration Path for Technical Debt

### Phase 1 (Immediate)
- Standardize error handling
- Implement database migrations
- Increase test coverage

### Phase 2 (Short-term)
- Refactor large VM handlers module
- Complete federation implementation
- Add E2E tests

### Phase 3 (Long-term)
- Implement service mesh
- Move to microservices architecture
- Add ML-based optimization