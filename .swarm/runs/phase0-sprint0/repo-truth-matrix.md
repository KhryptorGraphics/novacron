# Repo Truth Matrix

This matrix reflects the repository as inspected on 2026-04-22. It is the authoritative Sprint 0 classification baseline for convergence work.

## Product Line Truth

| Surface | Current truth | Convergence interpretation |
| --- | --- | --- |
| `backend/core/cmd/novacron/main.go` | Real daemon-style composition root that wires storage, VM manager, scheduler, migration, network, hypervisor, and API startup | Canonical runtime target |
| `backend/cmd/api-server/main.go` | Default no-build-tag API runtime and current root `npm run start:api` target | Active but competing runtime |
| `backend/cmd/api-server/main_*.go` | Parallel build-tag variants for enhanced, improved, multicloud, production, real-backend, secure, working, and simple API modes | Feature-gated wrappers or archives after convergence |
| `cmd/novacron/main.go` | Legacy launcher with direct VM wiring and static frontend serving | Competing monolith, not canonical |
| `main.go` | Prints legacy completion text only | Archive placeholder, not runtime |

## Top-Level Area Classification

### Runtime

| Area | Class | Notes |
| --- | --- | --- |
| `backend/` | runtime | Core backend, control plane, API, federation, storage, VM, network, and tests |
| `frontend/` | runtime | Primary operator UI and frontend tests |
| `cli/` | runtime | Standalone CLI surface |
| `sdk/` | runtime | Consumer SDKs for Go, JavaScript, and TypeScript |
| `adapters/` | runtime | Adapter package surface |
| `plugins/` | runtime | Runtime extension surfaces such as Terraform plugin code |
| `database/` | runtime | Database service code and migrations |
| `src/` | runtime | Root Node-based runtime helpers such as auto-spawn flows |

### Support

| Area | Class | Notes |
| --- | --- | --- |
| `.beads/` | support | Issue tracking state; currently degraded because the local rig is not Dolt-ready |
| `.codemap/`, `.gitnexus/`, `.swarm/` | support | Local intelligence, code graph, and run-memory state |
| `.augment/`, `.bmad-core/`, `.claude/`, `.claude-flow/`, `.gemini/`, `.hive-mind/`, `.paul/`, `.qwen/`, `.serena/` | support | Agent and workflow support directories |
| `.github/` | support | CI and automation definitions |
| `docs/`, `claudedocs/` | support | Human documentation; authoritative only when tied to active build or sprint artifacts |
| `config/`, `configs/` | support | Configuration and test/config assets |
| `data/`, `memory/`, `checkpoints/` | support | Shared data and stateful support assets |
| `tools/` | support | Internal tooling, indexer, and build helpers |
| `AGENTS.md`, `CLAUDE.md`, `README.md`, `SETUP.md`, `LICENSE` | support | Current operator and contributor guidance |
| `go.mod`, `go.sum`, `package.json`, `package-lock.json`, `pyproject.toml`, `requirements.txt`, `uv.lock`, `Makefile`, `playwright.config.ts` | support | Active build and test control files |

### Research

| Area | Class | Notes |
| --- | --- | --- |
| `ai-engine/`, `ai_engine/` | research | Duplicate or parallel ML service trees |
| `ml/` | research | Root ML experimentation and assets |
| `research/` | research | Research notes and exploratory work |
| `notebooks/` | research | Notebook-based exploration |
| `advanced/` | research | Advanced or experimental surfaces |
| `benchmark-results/` | research | Benchmark output, not runtime control |

### Ops

| Area | Class | Notes |
| --- | --- | --- |
| `deployment/`, `deployments/` | ops | Parallel deployment trees with overlapping Docker, K8s, and monitoring assets |
| `docker/`, `docker-compose*.yml`, `qdrant-docker-compose.yml` | ops | Container build and local orchestration assets |
| `k8s/`, `backend/deployments/k8s/` | ops | Kubernetes manifests and charts |
| `k8s-operator/`, `operator/` | ops | Operator implementation and deployment surfaces |
| `systemd/` | ops | Systemd unit definitions |
| `terraform/` | ops | Terraform infrastructure definition |
| `nginx/`, `apparmor/`, `policies/` | ops | Runtime policy and edge infrastructure assets |
| `deployment-automation.sh`, `integration-validation.sh`, `healthcheck.js`, `test-auth.sh` | ops | Operational scripts and health tooling |

### Archive

| Area | Class | Notes |
| --- | --- | --- |
| `masterdocs/` | archive | Large legacy documentation tree; not authoritative for runtime |
| `temp_main_files/` | archive | Temporary main files and snapshots |
| `web-bundles/`, `web_interface/` | archive | Parallel web surfaces outside the primary frontend |
| `community/`, `marketplace/` | archive | Out-of-scope for core-platform-first convergence unless blocking |
| `logs/`, `output/`, `coverage/`, `.pytest_cache/`, `.venv/`, `node_modules/` | archive | Generated or local-only artifacts |
| `api-server-production`, `novacron-api-enhanced`, `platform-tools-latest-linux.zip` | archive | Built binaries and downloaded artifacts |
| Root status and celebration docs such as `*_SUMMARY.md`, `*_REPORT.md`, `*_COMPLETE*.md`, `FINAL*.md`, `PHASE*.md`, `PROJECT_*`, `PROMPT.md`, `prompt.txt`, `EXECUTE_THIS_PROMPT.txt`, `WEEKS_*.md` | archive | Non-authoritative status residue unless cited by an active sprint artifact |

## Module And Toolchain Inventory

### Go Modules

- 31 Go modules are present in the repository.
- High-impact modules include the root module, `backend/core`, `cli`, `database`, `k8s-operator`, `operator`, `sdk/go`, `tools/indexer`, and dedicated test modules under `tests/`.
- Example nested modules that complicate default `go test ./...` behavior:
  - `backend/core/backup`
  - `backend/core/orchestration`
  - `backend/core/initialization`
  - `backend/examples/*`
  - `tests/automation`
  - `tests/integration`
  - `tests/verification`

### Toolchain Drift

| Surface | Current truth | Risk |
| --- | --- | --- |
| Root Go toolchain | `go 1.24.0` with `toolchain go1.24.6` in `go.mod` | Does not match CI workflows pinned to Go 1.21 |
| Root Node toolchain | `package.json` requires Node `>=18` and npm `>=9` | Too loose for reproducible CI and may not satisfy Next 16 expectations |
| Frontend Node toolchain | `frontend/package.json` uses Next 16.0.2 and React 18.2.0 | Likely expects Node 20+, while some CI still runs Node 18 |
| Package manager paths | Root uses npm, frontend uses npm, `tools/indexer` includes `yarn.lock` and pnpm metadata | Three parallel package-manager signals |
| Python toolchain | `requirements.txt`, `pyproject.toml`, and `uv.lock` all present | Multiple Python environment stories without one default path |

## Entrypoint Inventory

### Product-Line And Operational Entrypoints

| Entrypoint | Current role | Status |
| --- | --- | --- |
| `backend/core/cmd/novacron/main.go` | Full daemon composition root | Canonical target |
| `backend/cmd/api-server/main.go` | Default API runtime | Active and competing |
| `backend/cmd/api-server/main_enhanced.go` | `novacron_enhanced` variant | Feature-gated |
| `backend/cmd/api-server/main_improved.go` | `novacron_improved` variant | Feature-gated |
| `backend/cmd/api-server/main_multicloud.go` | `novacron_multicloud` variant | Feature-gated |
| `backend/cmd/api-server/main_production.go` | `novacron_production` variant | Feature-gated |
| `backend/cmd/api-server/main_real_backend.go` | `novacron_real_backend` variant | Feature-gated |
| `backend/cmd/api-server/main_secure.go` | `novacron_secure` variant | Feature-gated |
| `backend/cmd/api-server/main_working.go` | `novacron_working` variant | Feature-gated |
| `backend/cmd/api-server/simple-api.go` | `novacron_simple_api` variant | Feature-gated |
| `cmd/novacron/main.go` | Legacy direct VM/server launcher | Archive candidate |
| `backend/cmd/core-server/main.go` | Auxiliary backend entrypoint | Support surface |
| `backend/core/edge/agent/main.go` | Edge agent entrypoint | Support surface |
| `cli/cmd/novacron/main.go` | CLI binary entrypoint | Runtime support |
| `k8s-operator/cmd/manager/main.go` | Operator manager entrypoint | Ops surface |
| `tools/indexer/main.go` | Indexer entrypoint | Support tooling |
| `main.go` | Legacy completion message binary | Archive placeholder |

### Build Tags That Create Parallel Runtime Shapes

- API server tags: `novacron_enhanced`, `novacron_improved`, `novacron_multicloud`, `novacron_production`, `novacron_real_backend`, `novacron_secure`, `novacron_working`, `novacron_simple_api`
- Feature tags that materially alter behavior elsewhere: `experimental`, `novacron_security_*`, `novacron_vm_*`, `novacron_zero_trust`, `cgo`, `linux`, `amd64`

## CI Workflow Inventory

| Workflow | Current role | Status |
| --- | --- | --- |
| `.github/workflows/ci.yml` | Canonical verification candidate for backend and frontend | Candidate canonical |
| `.github/workflows/ci-cd.yml` | Generic pipeline with test, build, and deploy stages | Overlapping legacy |
| `.github/workflows/ci-cd-production.yml` | Production-oriented CI/CD lane | Overlapping legacy |
| `.github/workflows/comprehensive-testing.yml` | Expanded test coverage lane | Specialized |
| `.github/workflows/integration-tests.yml` | Integration-specific lane | Specialized |
| `.github/workflows/e2e-tests.yml` | E2E test lane | Specialized |
| `.github/workflows/e2e-nightly.yml` | Nightly E2E lane | Specialized |
| `.github/workflows/e2e-visual-regression.yml` | Visual regression lane | Specialized |
| `.github/workflows/deploy-production.yml` | Production deploy lane | Specialized |
| `.github/workflows/dwcp-phase1-deploy.yml` | DWCP-specific deploy path | Legacy experimental |
| `.github/workflows/dwcp-v3-ci.yml` | DWCP v3 CI path | Legacy experimental |
| `.github/workflows/dwcp-v3-cd.yml` | DWCP v3 CD path | Legacy experimental |
| `.github/workflows/onboarding-system-ci.yml` | Onboarding-specific lane | Legacy support |
| `.github/workflows/update-code-memory.yml` | Memory/index maintenance | Support automation |

## Deployment Inventory

| Surface | Current role | Status |
| --- | --- | --- |
| `deployment/` | Production-leaning deployment tree with Docker, Kubernetes, monitoring, rollback, and scripts | Active but overlapping |
| `deployments/` | Parallel deployment tree with DWCP, K8s, monitoring, Docker, and onboarding assets | Active but overlapping |
| `docker/` | Additional Dockerfiles and entrypoints | Active but overlapping |
| Root `docker-compose*.yml` files | Multiple local and environment-specific compose paths | Active but overlapping |
| `k8s/` | Root Kubernetes manifests | Active but overlapping |
| `backend/deployments/k8s/` | Backend-specific charts, DR, networking, service mesh, and tests | Active but overlapping |
| `k8s-operator/` and `operator/` | Operator implementations and assets | Active but parallel |
| `systemd/` | Systemd service units for API, hypervisor, network, storage, and LLM engine | Active support |
| `terraform/` | Infrastructure as code | Active support |

## Test Harness Inventory

| Surface | Current truth | Status |
| --- | --- | --- |
| Root `package.json` | Jest unit tests, integration runners, Playwright E2E, backend build wrapper | Active but broad |
| `frontend/package.json` | Next build, Jest unit/component/hooks tests, E2E Jest/Puppeteer lane | Active |
| `frontend/cypress/` | Cypress E2E assets | Parallel E2E path |
| `backend/tests/` | Extensive backend API, benchmark, chaos, comprehensive, E2E, multicloud, ML, and phase tests | Active but fragmented |
| Root `tests/` | Cross-stack integration, chaos, compliance, performance, security, verification, and unit suites | Active but fragmented |
| `tests/automation`, `tests/coverage`, `tests/integration`, `tests/verification` | Dedicated Go submodules for specialized test lanes | Active but fragmented |
| `backend/tests/comprehensive/Makefile` and `tests/integration/Makefile` | Extra test orchestration layers | Specialized |

## Documentation Authority

### Authoritative Now

- Code under `backend/`, `frontend/`, `cli/`, `sdk/`, and active ops surfaces
- Active build files and workflows
- `.swarm/validation-matrix.yaml`
- `.swarm/runs/phase0-sprint0/*`

### Non-Authoritative Until Revalidated

- Root completion, summary, and status documents
- `masterdocs/`
- `temp_main_files/`
- Standalone built binaries and downloaded archives
- Any generated output under `.next`, `coverage`, `logs`, `output`, or `node_modules`

## Sprint 1 Starting Decisions

1. Pick one default Go toolchain and update CI to match it.
2. Pick one Node version and one package-manager path for shipping work.
3. Collapse entrypoints so the canonical daemon owns startup and shutdown sequencing.
4. Collapse CI and deployment assets to one canonical hierarchy with clearly marked specialized lanes.
