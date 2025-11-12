# NovaCron Project Structure Map

**Generated:** November 12, 2025
**Project:** NovaCron - Advanced VM Management and ML Engineering Platform
**Status:** Comprehensive exploration complete

---

## Table of Contents

1. [Directory Tree Overview](#directory-tree-overview)
2. [Key Components Map](#key-components-map)
3. [Configuration Files](#configuration-files)
4. [Entry Points](#entry-points)
5. [Dependencies Map](#dependencies-map)
6. [File Organization Patterns](#file-organization-patterns)
7. [API Structure](#api-structure)
8. [Testing Strategy](#testing-strategy)
9. [Integration Points](#integration-points)
10. [Build & Deployment](#build--deployment)

---

## Directory Tree Overview

### Root Level Structure

```
/home/kp/novacron/
├── .augment/                  # Augmentation framework configuration
├── .beads/                    # BEADS issue tracking system
├── .claude/                   # Claude integration configs
├── .claude-flow/              # Claude Flow metrics & coordination
├── .swarm/                    # Swarm memory and state management
├── adapters/                  # Integration adapters
├── advanced/                  # Advanced feature implementations
├── ai-engine/                 # AI engine (Python-based)
├── ai_engine/                 # ML/AI training pipelines
├── backend/                   # Main backend codebase (Go)
├── benchmark-results/         # Performance benchmark outputs
├── cli/                       # Command-line interface
├── cmd/                       # Root level command implementations
├── community/                 # Community management modules
├── config/                    # Configuration files (YAML)
├── configs/                   # Additional configuration directory
├── coverage/                  # Test coverage reports
├── data/                      # Data storage and fixtures
├── database/                  # Database setup and migrations
├── deployment/                # Deployment configurations
├── deployments/               # Advanced deployment specs
├── docker/                    # Docker-related files
├── docs/                      # Documentation (250+ MD files)
├── frontend/                  # Frontend codebase (React/Next.js)
├── k8s/                       # Kubernetes manifests
├── k8s-operator/              # Kubernetes operator implementation
├── marketplace/               # Marketplace integrations
├── memory/                    # Memory store configurations
├── nginx/                     # Nginx configurations
├── node_modules/              # Node.js dependencies
├── notebooks/                 # Jupyter notebooks
├── operator/                  # Custom operator implementation
├── plugins/                   # Plugin ecosystem
├── policies/                  # Policy definitions
├── research/                  # Research implementations
├── scripts/                   # Utility and deployment scripts
├── sdk/                       # Software Development Kits
├── snap/                      # Snapcraft packaging
├── src/                       # Source code (Python & Node)
├── systemd/                   # Systemd service definitions
├── tests/                     # Comprehensive test suite
├── tools/                     # Development tools
├── web-bundles/               # Web asset bundles
├── web_interface/             # Web interface
├── main.go                    # Root Go entry point
├── package.json               # NPM dependencies (root)
├── go.mod                     # Go module definition
├── CLAUDE.md                  # Claude development guidelines
└── README.md                  # Project README
```

---

## Key Components Map

### Backend Structure (`/backend/`)

#### Core Infrastructure (`backend/core/`)

**69 major subsystems organized by domain:**

```
backend/core/ - Major Systems:
├── agents/                    # Agent framework
├── ai/                        # AI/ML implementations (8+ submodules)
├── analytics/                 # Analytics engine (4 components)
├── automation/                # Automation framework (10+ subcomponents)
├── backup/                    # Backup & recovery
├── blockchain/                # Blockchain integration
├── cache/                     # Caching layer
├── chaos/                     # Chaos engineering
├── cicd/                      # CI/CD pipeline
├── cognitive/                 # Cognitive computing
├── compliance/                # Compliance frameworks (4 components)
├── compute/                   # Compute resources
├── consensus/                 # Distributed consensus
├── discovery/                 # Service discovery
├── dr/                        # Disaster recovery (8 components)
├── edge/                      # Edge computing (7 modules)
├── federation/                # Federation system (7 components)
├── global/                    # Global management (8 components)
├── governance/                # Governance engine (15 components)
├── ha/                        # High availability
├── health/                    # Health checks
├── hypervisor/                # Hypervisor management
├── incident/                  # Incident management
├── initialization/            # System initialization (8 components)
├── iot/                       # IoT integration
├── llm/                       # LLM integration
├── metrics/                   # Metrics collection
├── ml/                        # Machine learning (13 components)
├── mlops/                     # MLOps pipeline (8+ components)
├── monitoring/                # Monitoring system (7 components)
├── multicloud/                # Multi-cloud management (6 components)
├── network/                   # Network management & DWCP v3
├── neuromorphic/              # Neuromorphic computing
├── nfv/                       # Network Function Virtualization
├── nlp/                       # Natural Language Processing
├── observability/             # Observability stack
├── orchestration/             # Orchestration engine
├── partnerships/              # Partnership framework
├── performance/               # Performance optimization (8+ modules)
├── photonic/                  # Photonic computing
├── planetary/                 # Planetary scale
├── plugins/                   # Plugin system
├── quantum/                   # Quantum computing
├── quotas/                    # Resource quotas
├── research/                  # Research implementations
├── scheduler/                 # Scheduling system
├── sdn/                       # Software-Defined Networking
├── security/                  # Security framework (4+ modules)
├── shared/                    # Shared utilities
├── snapshot/                  # Snapshot management
├── sre/                       # Site Reliability Engineering
├── storage/                   # Storage management
├── templates/                 # Template system
├── validation/                # Validation framework
├── vm/                        # Virtual machine management
├── v4/ & v5/                  # Future versions (prototypes)
├── zeroops/                   # Zero-ops automation
├── go.mod                     # Go module dependencies
└── go.sum                     # Dependency checksums
```

#### API Layer (`backend/api/`)

```
backend/api/ - 14 API Modules:
├── admin/                     # Admin API
├── auth/                      # Authentication API
├── backup/                    # Backup API
├── compute/                   # Compute resources API
├── federation/                # Federation API
├── gateway/                   # API Gateway
├── graphql/                   # GraphQL API
├── ml/                        # ML/AI API
├── monitoring/                # Monitoring API
├── orchestration/             # Orchestration API
├── rest/                      # REST API
├── security/                  # Security API
├── vm/                        # VM management API
└── websocket/                 # WebSocket API
```

#### Command Entry Points (`backend/cmd/`)

```
backend/cmd/
├── api-server/
│   ├── main.go                # Primary API server (929 lines)
│   ├── main_real_backend.go   # Backend implementation (828 lines)
│   ├── main_enhanced.go       # Enhanced features (569 lines)
│   └── main_improved.go       # Improved version (434 lines)
├── core-server/               # Core service entry point
└── auth-test/                 # Authentication testing
```

#### Business & Enterprise (`backend/`)

```
backend/ - Enterprise Systems:
├── business/                  # Revenue & validation
├── community/                 # Community programs (11 programs)
├── enterprise/                # Enterprise features (6+ modules)
├── operations/                # Operations (8 programs)
├── corporate/                 # Corporate strategy
└── partnerships/              # Strategic partnerships
```

### Frontend Structure (`/frontend/`)

```
frontend/ - React/Next.js:
├── src/
│   ├── app/                   # Next.js app directory
│   ├── components/            # Reusable UI components
│   ├── contexts/              # React contexts
│   ├── hooks/                 # Custom React hooks
│   ├── lib/                   # Utility libraries
│   ├── providers/             # Context providers
│   ├── styles/                # CSS/Tailwind styling
│   └── __tests__/             # Component tests
├── public/                    # Static assets
├── cypress/                   # Cypress E2E tests
├── tests/                     # Test utilities
├── __mocks__/                 # Mock data
├── test-setup/                # Test configuration
└── package.json               # 100+ dependencies
```

**Tech Stack:**
- React 18.2.0, Next.js 13.5.6, TypeScript 5.1.6
- Tailwind CSS 4.0.14, Radix UI, Framer Motion
- TanStack Query, Chart.js, D3.js
- Playwright, Jest, Cypress

### Testing Suite (`/tests/`)

```
tests/ - Comprehensive Testing:
├── e2e/                       # Playwright E2E tests
├── integration/               # Cross-component integration
├── unit/                      # Fast unit tests (Jest/Go)
├── performance/               # Load & stress tests
├── compliance/                # Regulatory compliance
├── security/                  # Security testing
├── chaos/                     # Chaos engineering
├── mlops/                     # ML pipeline tests
└── mle-star-samples/          # ML engineering samples
```

### SDK Layer (`/sdk/`)

```
sdk/ - Client Libraries:
├── go/                        # Go SDK
├── python/                    # Python SDK (with DWCP)
├── typescript/                # TypeScript SDK
├── javascript/                # JavaScript SDK
├── rust/                      # Rust SDK
└── partners/                  # Partner SDK
```

### Kubernetes (`/k8s/` & `/k8s-operator/`)

```
k8s/ - Manifests:
├── novacron-deployment.yaml  # Main deployment
├── novacron-secrets.yaml     # Secrets
├── ingress.yaml              # Ingress
├── redis-deployment.yaml     # Redis
├── mysql-deployment.yaml     # MySQL
├── redis-cluster.yaml        # Redis cluster
├── scheduler-deployment.yaml # Scheduler
└── worker-deployment.yaml    # Worker nodes

k8s-operator/ - Kubernetes Operator:
├── deploy/
│   ├── examples/             # CRD examples
│   ├── rbac/                 # RBAC policies
│   ├── monitoring/           # Monitoring config
│   └── crds/                 # Custom resource definitions
├── pkg/
│   ├── novacron/             # Core resources
│   ├── providers/            # Cloud provider integration
│   ├── controllers/          # Controller logic
│   ├── apis/                 # API definitions
│   └── cache/                # Caching layer
└── cmd/
    └── manager/              # Operator manager entry
```

---

## Configuration Files

### Root Level Configuration

| File | Purpose | Type |
|------|---------|------|
| `package.json` | Node.js dependencies | JSON |
| `go.mod` / `go.sum` | Go dependencies (v1.24.0+) | Text |
| `CLAUDE.md` | Development guidelines | Markdown |
| `README.md` | Project overview | Markdown |
| `docker-compose.yml` | Production Docker setup | YAML |
| `docker-compose.dev.yml` | Development environment | YAML |
| `docker-compose.prod.yml` | Production environment | YAML |
| `playwright.config.ts` | E2E test configuration | TypeScript |

### Backend Configuration

```
config/
├── dwcp-v3-datacenter.yaml    # DWCP datacenter config
├── dwcp-v3-hybrid.yaml        # Hybrid mode config
├── dwcp-v3-internet.yaml      # Internet-scale config
└── examples/                  # Configuration templates

configs/
├── config.yaml                # Main configuration
├── dwcp.yaml                  # DWCP settings
├── dwcp.staging.yaml          # Staging specific
├── dwcp.production.yaml       # Production specific
├── network-topology.yaml      # Network definitions
├── security-hardening.yaml    # Security settings
└── monitoring-stack.yml       # Monitoring setup
```

### Database Configuration

```
database/
├── migrations/                # SQL migration files
└── docker-compose.yml         # Database container setup
```

---

## Entry Points

### API Server (Primary)

**Location:** `/home/kp/novacron/backend/cmd/api-server/main.go`

**Startup Sequence:**
1. Load configuration from YAML/ENV
2. Initialize structured logging
3. Connect to PostgreSQL/MySQL database
4. Initialize JWT authentication manager
5. Setup encryption and audit logging
6. Initialize 2FA service
7. Start security coordinator
8. Register all API route handlers
9. Listen on API port (default 8080) and WebSocket port (8081)

**Core Imports:**
- `github.com/gorilla/mux` - HTTP routing
- `github.com/khryptorgraphics/novacron/backend/api/*` - API modules
- `github.com/khryptorgraphics/novacron/backend/core/*` - Core systems
- `github.com/lib/pq` - PostgreSQL driver

### Frontend Development

**Entry:** `/home/kp/novacron/frontend/src/app/`

**Start:** `npm run dev --prefix frontend` (Port 8092)

**Build:** `npm run build --prefix frontend`

### Workers & Services

**CLI Tool:** `/src/cli/auto-spawn.js` - Intelligent agent spawning

**ML Platform:** `/src/mle-star/` - ML Engineering STAR system

---

## Dependencies Map

### Go Ecosystem

**HTTP & Web:**
- `github.com/gorilla/mux` - Router
- `github.com/gorilla/websocket` - WebSockets
- `github.com/gin-gonic/gin` - Web framework

**Authentication & Security:**
- `github.com/golang-jwt/jwt/v5` - JWT
- `github.com/Azure/azure-sdk-for-go` - Azure
- `github.com/aws/aws-sdk-go` - AWS SDK

**Database:**
- `github.com/lib/pq` - PostgreSQL
- `github.com/mattn/go-sqlite3` - SQLite
- `github.com/jmoiron/sqlx` - SQL utilities
- `github.com/golang-migrate/migrate/v4` - Schema migrations

**Kubernetes & Cloud:**
- `k8s.io/client-go` - Kubernetes client
- `google.golang.org/grpc` - gRPC
- `go.etcd.io/etcd/client/v3` - etcd

**Monitoring & Observability:**
- `github.com/prometheus/client_golang` - Prometheus
- `go.opentelemetry.io/otel` - OpenTelemetry tracing
- `github.com/sirupsen/logrus` - Structured logging

**Distributed Systems:**
- `github.com/hashicorp/consul/api` - Consul
- `github.com/redis/go-redis/v9` - Redis client
- `github.com/sony/gobreaker` - Circuit breaker

### Node.js Root Dependencies

**Core Libraries:**
- `axios` - HTTP client
- `pg` - PostgreSQL (Node)
- `redis` - Redis (Node)
- `ws` - WebSocket server
- `@genkit-ai/mcp` - MCP integration

**Frontend Stack:** See `/frontend/package.json`
- React 18, Next.js 13, Tailwind CSS 4
- 100+ dev dependencies (testing, linting, building)

---

## File Organization Patterns

### Backend Naming

**Go Files:**
- `service.go` - Main implementation
- `service_test.go` - Unit tests
- `service_integration_test.go` - Integration tests
- `types.go` - Data structures
- `handler.go` - HTTP handlers

**Examples:**
- `hypervisor_test.go`
- `federation_adapter_v3.go`
- `dwcp_phase5_integration_test.go`

### Frontend Naming

**React/TypeScript:**
- `ComponentName.tsx` - Component
- `useHookName.ts` - Custom hook
- `utils.ts` - Utility functions
- `ComponentName.test.tsx` - Tests
- `types.ts` - Type definitions

### Configuration Files

**YAML Configuration Pattern:**
```yaml
server:
  api_port: 8080
  ws_port: 8081
  
database:
  driver: postgres
  connection_string: ...
  
logging:
  level: info
  format: json
```

**Environment Pattern:**
- `.env` - Shared environment
- `.env.local` - Local development
- `.env.production` - Production secrets
- `.env.test` - Testing configuration

---

## API Structure

### REST API Routes

**Base Path:** `/api/v1/`

**Modules (14 total):**
```
/admin/*              # Admin operations
/auth/*               # Authentication & authorization
/backup/*             # Backup & restore
/compute/*            # Compute resources
/federation/*         # Federation management
/ml/*                 # ML/AI operations
/monitoring/*         # Monitoring & metrics
/orchestration/*      # Workload orchestration
/security/*           # Security operations
/vm/*                 # Virtual machine management
```

### GraphQL API

**Location:** `/backend/api/graphql/`

**Capabilities:**
- Type-safe queries
- Real-time subscriptions
- Schema validation

### WebSocket API

**Location:** `/backend/api/websocket/`

**Features:**
- Event streaming
- Bidirectional communication
- Connection management

### Admin API

**Location:** `/backend/api/admin/`

**Features:**
- System configuration
- User management
- Audit logs
- Performance metrics

---

## Testing Strategy

### Test Layers

```
tests/
├── unit/              # Fast (< 100ms) isolated tests
├── integration/       # Component interaction tests
├── e2e/               # Full user flow tests (Playwright)
├── performance/       # Load, stress, scalability
├── compliance/        # Regulatory compliance
├── security/          # Security/vulnerability tests
└── chaos/             # Failure scenario testing
```

### Test Tools

| Tool | Purpose | Location |
|------|---------|----------|
| Jest | Node.js/React testing | `tests/unit/`, `frontend/` |
| Playwright | E2E browser automation | `tests/e2e/` |
| Cypress | Component/integration | `frontend/cypress/` |
| Go test | Backend unit tests | `**/*_test.go` |
| k6/Locust | Performance testing | `tests/performance/` |

### Coverage Targets

- Unit tests: 80%+ coverage
- Integration: Critical paths covered
- E2E: User journey coverage
- Security: OWASP Top 10

---

## Integration Points

### External Systems

**Cloud Providers:**
- AWS (EC2, RDS, S3, ECS)
- Azure (VMs, SQL, Storage)
- GCP (Compute, Cloud SQL)

**Kubernetes:**
- Custom Resource Definitions (CRDs)
- Operators for Novacron resources
- RBAC and networking

**Monitoring:**
- Prometheus scraping on `/metrics`
- OpenTelemetry tracing
- Structured JSON logging

**Databases:**
- PostgreSQL (recommended)
- MySQL (compatible)
- SQLite (development only)

**Cache:**
- Redis (primary)
- In-memory caching
- Distributed cache patterns

**Authentication:**
- JWT tokens
- OAuth2 (Azure, AWS, GCP)
- SAML (enterprise)
- API keys

---

## Build & Deployment

### Build Commands

```bash
# Backend
go build -o bin/ ./backend/...

# Frontend
npm run build --prefix frontend

# Combined
npm run build
```

### Docker Images

**Production Images:**
- `novacron-backend` - Go API server
- `novacron-frontend` - React Next.js app
- `novacron-k8s-operator` - Kubernetes operator
- `ai-engine` - Python ML service

### Deployment Options

**Option 1: Docker Compose**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

**Option 2: Kubernetes**
```bash
kubectl apply -f k8s/
kubectl apply -f k8s-operator/deploy/crds/
```

**Option 3: Operator**
```bash
kubectl apply -f k8s-operator/deploy/
```

### Environment Setup

**Development:**
```
docker-compose.dev.yml
+ frontend dev server (port 8092)
+ Backend on port 8080/8081
+ PostgreSQL on port 5432
```

**Staging:**
```
docker-compose.yml
+ Staging credentials
+ Health checks enabled
+ Monitoring setup
```

**Production:**
```
docker-compose.production.yml
+ Secret management
+ TLS/SSL enabled
+ High availability
+ Monitoring & alerting
```

---

## Key Statistics

### Codebase Metrics

**Size:**
- Backend (Go): ~500K+ lines across 1000+ files
- Frontend (TypeScript/React): ~100K+ lines
- Tests: ~200K+ lines across 800+ files
- Documentation: 250+ markdown files

**Systems:**
- 69 core domain modules
- 14 API modules
- 5+ SDK languages
- 8+ major subsystems (cloud, k8s, ml, etc.)

**Dependencies:**
- Go: 100+ direct, 500+ transitive
- Node: 50+ direct, 500+ transitive

### Performance Targets

- API response latency: <100ms p99
- UI interaction: <50ms
- System startup: <30s
- Horizontal scaling: 10K+ VMs/cluster
- Database: 10K+ queries/sec

---

## Memory & Coordination

### Swarm State

**Location:** `/home/kp/novacron/.swarm/memory.db`

**Coordination Metrics:**
- `/backend/.claude-flow/metrics/performance.json`
- `/backend/.claude-flow/metrics/system-metrics.json`
- `/backend/.claude-flow/metrics/task-metrics.json`

---

## Development Workflow

### SPARC Methodology

NovaCron uses SPARC (Specification, Pseudocode, Architecture, Refinement, Completion) with Claude Flow orchestration.

**Commands:**
```bash
npx claude-flow sparc tdd "<feature>"      # Full TDD workflow
npx claude-flow sparc run spec "<task>"    # Specification
npx claude-flow sparc batch <modes> "<task>"  # Parallel execution
```

### Guidelines

See `/home/kp/novacron/CLAUDE.md`:
- **1 MESSAGE = ALL OPERATIONS** (concurrent pattern)
- Use TodoWrite for batching todos
- Use Task tool for agent spawning
- Organize files in appropriate subdirectories
- Never save work files to root

---

## Summary

**NovaCron** is an enterprise-grade, cloud-native platform with:

- **69+ core systems** (VM management, ML/AI, monitoring, security, etc.)
- **Multi-language:** Go (backend), TypeScript/React (frontend), Python (ML)
- **Cloud-native:** Kubernetes operators, multi-cloud federation, edge computing
- **Enterprise-ready:** Security, compliance, governance, disaster recovery
- **Scalable:** Designed for 10K+ VMs, distributed systems, planetary scale
- **Well-tested:** Unit, integration, E2E, performance, compliance, chaos testing
- **Documented:** 250+ markdown files, comprehensive API docs
- **Production-grade:** High availability, monitoring, alerting, auto-scaling

**Total Scope:** 500K+ lines of production code with complete infrastructure.

---

**Documentation Generated:** November 12, 2025
**Repository:** https://github.com/novacron/novacron.git
**License:** MIT
**Status:** Active development, enterprise-ready
