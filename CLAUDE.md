# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NovaCron is a distributed virtual machine management platform with live migration, real-time monitoring, intelligent scheduling, multi-cloud orchestration, and enterprise security. Built on Go 1.24+ backend and Next.js 13.5 frontend.

## Build & Development Commands

### Backend (Go)

**Two server modes exist — prefer `core-server` for stability:**

```bash
# STABLE: Build and run the minimal core server (orchestration + VM only)
make core-build     # builds backend/core/orchestration
make core-serve     # starts minimal core server on :8090

# FULL: Build and run the full API server (may have experimental code)
cd backend/cmd/api-server && go run .

# Run a specific package's tests locally (no Docker required)
cd backend/core && go test -v -run "TestFoo" ./vm/...
cd backend/core && go test -v ./consensus/...
cd backend/core && go test -v -run "Test.*" ./cache/...

# Run core tests (orchestration subset)
make core-test

# Run all tests via Docker (CI-consistent)
make test

# Run specific test categories
make test-ml           # AI/ML model tests
make test-cache        # Redis cache tests (requires Redis)
make test-multicloud   # Multi-cloud integration
make test-e2e          # End-to-end tests
make test-chaos        # Chaos engineering tests
make test-unit-coverage  # Unit tests with coverage report

# Quality checks
make lint-backend       # golangci-lint via Docker
make security-scan      # gosec security scan via Docker
make ci-quality         # lint + security + vulnerability check
```

### Frontend (Next.js)
```bash
cd frontend
npm run dev             # Start dev server on :8092
npm run build           # Production build
npm run lint            # ESLint
npm run test            # Jest unit tests
npm run test:watch      # Jest watch mode
npm run test:coverage   # With coverage report
npm run test:e2e        # Puppeteer E2E tests
npm run test:unit       # Unit tests only
npm run test:components # Component tests only
```

### Database
```bash
make db-migrate         # Run migrations
make db-rollback        # Rollback last migration
make db-reset           # Drop, migrate, seed
make db-seed            # Seed dev data
make db-test-setup      # Setup test database
make db-migrate-create  # Create new migration (interactive)
```

### Docker
```bash
docker-compose up -d                    # Start all services
docker-compose -f docker-compose.test.yml up -d  # Test environment
make test-env-up        # Start test env (postgres + redis)
make test-env-down      # Stop test env
```

## Go Module Structure

The backend is **not a single Go module** — it has multiple separate modules, each with their own `go.mod`:

| Module path | Directory |
|-------------|-----------|
| `github.com/khryptorgraphics/novacron/backend/core` | `backend/core/` |
| `github.com/khryptorgraphics/novacron/backend/core/orchestration` | `backend/core/orchestration/` |
| `github.com/khryptorgraphics/novacron/backend/pkg/logger` | `backend/pkg/logger/` |
| `github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3` | `backend/core/network/dwcp/` |

The main `api-server` and `core-server` reference these via `replace` directives in their go.mod files. When adding imports, always `cd` into the specific module directory first.

**Note:** Files ending in `.go.disabled` are experimental/WIP features intentionally excluded from compilation. Do not rename them to `.go` without verifying they compile.

## Architecture

### Backend Structure (`backend/`)
```
backend/
├── api/                  # HTTP handlers organized by domain
│   ├── vm/              # VM CRUD, migration, snapshot, metrics, cluster handlers
│   ├── auth/            # Authentication routes & JWT middleware
│   ├── backup/          # Backup/restore handlers
│   ├── federation/      # Multi-cluster federation API
│   ├── graphql/         # GraphQL API (gqlgen)
│   ├── websocket/       # Real-time WebSocket handlers
│   ├── orchestration/   # Orchestration engine API
│   └── admin/           # Admin management
├── core/                 # Business logic (primary Go module)
│   ├── vm/              # VM types, state machine, driver abstraction, scheduler,
│   │                    #   live migration, predictive prefetching
│   ├── auth/            # JWT, OAuth2, 2FA, zero-trust, RBAC, tenant management
│   ├── backup/          # CBT-based incremental backup, deduplication, replication
│   ├── cache/           # Multi-tier cache (L1 BigCache, L2 Redis)
│   ├── consensus/       # Raft implementation, distributed locks, split-brain detection
│   ├── federation/      # Multi-cluster federation, multicloud orchestration
│   ├── hypervisor/      # KVM/libvirt integration
│   ├── migration/       # Live migration orchestrator, WAN optimizer, rollback
│   ├── ml/              # ML predictor, anomaly detection, gradient compression
│   ├── monitoring/      # Telemetry, Prometheus, OpenTelemetry tracing, dashboards
│   ├── network/         # SDN, load balancing, DWCP WAN protocol, QoS
│   ├── scheduler/       # Multi-policy resource scheduling (LARS, NILAS, LAVA, GBDT)
│   └── orchestration/   # Event-driven orchestration engine (NATS-backed)
├── pkg/                  # Shared utilities (separate module)
│   ├── config/          # Environment-variable-based config
│   ├── logger/          # Structured logging (logrus-based)
│   ├── middleware/       # Auth/RBAC middleware
│   └── database/        # DB helpers
└── cmd/
    ├── api-server/      # Full production API server (may require libvirt, etc.)
    └── core-server/     # Minimal stable server (safe subset: VM + orchestration)
```

### Frontend Structure (`frontend/src/`)
```
frontend/src/
├── app/                 # Next.js App Router pages
│   ├── vms/            # VM management UI
│   ├── monitoring/     # Metrics dashboards
│   ├── network/        # SDN/network topology
│   ├── storage/        # Storage management
│   ├── security/       # Security & compliance
│   └── admin/          # Admin panel
├── components/          # React components (Radix UI based)
│   ├── ui/             # Base design system components
│   ├── vm/             # VM-specific components
│   ├── monitoring/     # Monitoring charts/widgets
│   └── flows/          # React Flow diagrams
├── lib/                 # API clients, utilities
├── hooks/               # Custom React hooks
└── contexts/            # React contexts (auth, theme)
```

### Key Architectural Patterns

**VM Driver Abstraction:** `backend/core/vm/driver_factory.go` provides a factory that creates drivers based on `VMType` (KVM, Container, Containerd, Process). Drivers implement the `VMDriver` interface. KVM is the primary production driver; Container/Containerd drivers exist for dev environments.

**RBAC Pattern:** API routes use a `require(role, handler)` wrapper with 3 levels: `viewer` (read-only), `operator` (CRUD + VM actions), `admin` (all operations). See `backend/api/vm/routes.go`.

**Config:** All configuration via environment variables, loaded in `backend/pkg/config/config.go`. Key vars: `API_PORT` (8090), `WS_PORT` (8091), `DB_URL`, `AUTH_SECRET`, `STORAGE_PATH`, `HYPERVISOR_ADDRS`.

**Orchestration Engine:** Event-driven, NATS-backed, in `backend/core/orchestration/`. Orchestrates VM placement, evacuation, and healing. Injected into servers via `OrchestrationAdapters`.

**Migration:** `LiveMigrationOrchestrator` in `backend/core/migration/` handles pre-copy/post-copy live migration with WAN optimization, AI-assisted scheduling, and automatic rollback.

**Monitoring:** `NovaCronMonitoringSystem` integrates Prometheus exporters, OpenTelemetry tracing, ML-based anomaly detection, and dashboard widgets.

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

## Testing Patterns
- Go unit tests: `cd backend/core && go test -v -run "TestName" ./package/...`
- Integration tests require Docker: `make test-integration-setup` first, then `make test-integration`
- Cache tests require running Redis: `make test-cache-performance`
- Frontend tests use Jest + Testing Library; E2E uses Puppeteer
- ML model accuracy tests: `make test-ml-accuracy`

## File Organization Rules
- Source code → `backend/`, `frontend/src/`
- Documentation → `docs/`
- Configuration → `configs/`, `config/`
- Database migrations → `database/migrations/`
- **Never save working files to repo root**

## AI/Claude-Flow Integration

```bash
# SPARC methodology commands
npx claude-flow sparc run <mode> "<task>"
npx claude-flow sparc tdd "<feature>"
```

When using Claude Code's Task tool for agent spawning, batch all operations in single messages.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **novacron** (233836 symbols, 427391 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/novacron/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/novacron/context` | Codebase overview, check index freshness |
| `gitnexus://repo/novacron/clusters` | All functional areas |
| `gitnexus://repo/novacron/processes` | All execution flows |
| `gitnexus://repo/novacron/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyze to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyze without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
