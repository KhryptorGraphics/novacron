# NovaCron — Development Continuation Prompt

**Last Updated:** 2026-02-28
**Platform:** Jetson Thor (aarch64, CUDA 13.0, SM 11.0)

---

## Current Status

### What Is Complete (Do Not Re-Do)

| Layer | Status |
|-------|--------|
| Auth / JWT / OAuth2 / RBAC / 2FA | ✅ COMPLETE |
| PostgreSQL + Redis persistence | ✅ COMPLETE |
| Security hardening (rate limiting, CORS, headers) | ✅ COMPLETE |
| Jetson Thor deployment scripts | ✅ COMPLETE |
| E2E auth flows tested | ✅ COMPLETE |
| Core API server (`make core-serve`) | ✅ STABLE on :8090 |
| Frontend (Next.js) | ✅ STABLE on :8092 |

### What Is Broken (Your Job)

The **DWCP module** (`backend/core/network/dwcp/`) has **P0 blockers** that prevent compilation.
All further DWCP development is blocked until these are resolved.

**Root cause chain:**
```
rdma_cgo.go has no build tag
  → requires libibverbs (not on Jetson aarch64)
    → entire dwcp package fails to compile
      → 5 critical P0 issues in dwcp_manager.go cannot be fixed/tested
        → Phases 2-5 cannot proceed
```

---

## Open Beads Tasks (Priority Order)

```
bd list --status=open   # to verify this list
```

| ID | Priority | Description |
|----|----------|-------------|
| `novacron-7q6.1` | P0 | Fix 5 P0 Critical Issues in DWCP ← START HERE |
| `novacron-7q6.2` | P0 | Phase 2: Neural Training Pipeline (98% Accuracy) |
| `novacron-7q6` | P0 | [epic] Distributed Computing Enhancement |
| `novacron-92v` | P1 | Phase 2: Intelligence — PBA + ITP |
| `novacron-9tm` | P1 | Phase 3: Synchronization — ASS + ACP |
| `novacron-7q6.3` | P1 | Phase 3: ProBFT Probabilistic Consensus |
| `novacron-7q6.4` | P1 | Phase 4: MADDPG Multi-Agent DRL |
| `novacron-aca` | P1 | Phase 5: Production Validation |
| `novacron-ttc` | P1 | Phase 4: Production Optimization |
| `novacron-38p` | P1 | [in_progress] Run Phase 0 benchmarks |
| `novacron-ahm` | P1 | [epic] DWCP Integration into NovaCron |
| `novacron-7pt` | P1 | [epic] DWCP v5 GA |
| `NC-2hk` | P2 | Fix import cycle: monitoring→vm→federation→vm |
| `novacron-9wq` | P2 | [in_progress] Production deployment pipeline |
| `novacron-7q6.5` | P2 | Phase 5: TCS-FEEL Federated Learning |

---

## IMMEDIATE ACTION: Fix DWCP Compilation

### Step 0 (Prerequisite): Fix RDMA CGO Build Constraint

**Problem:** `rdma_cgo.go` has no build tag so it always tries to link `libibverbs`
(unavailable on Jetson Thor aarch64), breaking all DWCP compilation.

**Files:**
- `backend/core/network/dwcp/transport/rdma/rdma_cgo.go` — CGO types, no build tag
- `backend/core/network/dwcp/transport/rdma/rdma.go` — uses `Context`, `ConnInfo`,
  `CheckAvailability`, `Initialize` defined only in `rdma_cgo.go`

**Fix — Two steps:**

**Step 0a:** Add a build tag to `rdma_cgo.go` so it only compiles with CGO on Linux:

```go
//go:build cgo && linux

package rdma
// ... rest of file unchanged
```

**Step 0b:** Create `backend/core/network/dwcp/transport/rdma/rdma_stub.go` with
stub types for non-CGO / non-Linux builds (Jetson Thor needs this):

```go
//go:build !cgo || !linux

package rdma

import "fmt"

// Stub types — no libibverbs required on this platform

type Context struct{}
type ConnInfo struct {
    LID   uint16
    QPNum uint32
    PSN   uint32
    GID   [16]byte
}
type DeviceInfo struct {
    Name string
}
type Stats struct{}

func CheckAvailability() bool { return false }

func Initialize(deviceName string, port int, useEventChannel bool) (*Context, error) {
    return nil, fmt.Errorf("RDMA not supported on this platform (no CGO or not Linux)")
}

func (c *Context) Close() {}
func (c *Context) IsConnected() bool { return false }
func (c *Context) RegisterMemory(buf []byte) error { return fmt.Errorf("RDMA not supported") }
func (c *Context) UnregisterMemory() {}
func (c *Context) GetConnInfo() (ConnInfo, error) { return ConnInfo{}, fmt.Errorf("RDMA not supported") }
func (c *Context) Connect(remote ConnInfo) error { return fmt.Errorf("RDMA not supported") }
func (c *Context) PostSend(buf []byte, wrID uint64) error { return fmt.Errorf("RDMA not supported") }
func (c *Context) PostRecv(buf []byte, wrID uint64) error { return fmt.Errorf("RDMA not supported") }
func (c *Context) PostWrite(buf []byte, remoteAddr uint64, rkey uint32, wrID uint64) error {
    return fmt.Errorf("RDMA not supported")
}
func (c *Context) PostRead(buf []byte, remoteAddr uint64, rkey uint32, wrID uint64) error {
    return fmt.Errorf("RDMA not supported")
}
func (c *Context) PollCompletion(send bool) (bool, uint64, int, error) {
    return false, 0, 0, fmt.Errorf("RDMA not supported")
}
```

**Verify Step 0:**
```bash
cd backend/core/network/dwcp
go build ./...
# Must produce NO output (clean build)
```

---

## Priority 1 (P0): Fix 5 Critical Issues in DWCP

**Beads task:** `novacron-7q6.1`
**Claim it first:** `bd update novacron-7q6.1 --status in_progress`

All fixes are copy-paste ready in:
- `backend/core/network/dwcp/QUICK_FIX_GUIDE.md` — full replacement code
- `backend/core/network/dwcp/CRITICAL_ISSUES_TRACKER.md` — detailed analysis

### Recommended Fix Order (based on dependencies)

```
#1 Race Condition     → foundation, fix first
#5 Stack Escape       → quick win (1-2h), independent
#3 Config Validation  → quick win (2-3h), independent
#2 Component Lifecycle → depends on #1 (4-6h)
#4 Error Recovery     → depends on #2 (6-8h)
```

---

### Issue #1: Race Condition in Metrics Collection

**File:** `backend/core/network/dwcp/dwcp_manager.go:225-248`

**Bug:** `m.enabled` (protected by `m.mu`) is read inside `m.metricsMutex.Lock()` — wrong lock.

**Fix:**
```go
func (m *Manager) collectMetrics() {
    // Read enabled state under its correct lock
    m.mu.RLock()
    enabled := m.enabled
    m.mu.RUnlock()

    // Update metrics under metrics lock
    m.metricsMutex.Lock()
    defer m.metricsMutex.Unlock()
    m.metrics.Enabled = enabled
    m.metrics.Version = DWCPVersion
}
```

**Verify:** `go test -race ./...` — must show no DATA RACE warnings.

---

### Issue #5: Unsafe Config Copy (Stack Escape)

**File:** `backend/core/network/dwcp/dwcp_manager.go:175-183`

**Bug:** `GetConfig()` returns `&configCopy` where `configCopy` is a stack variable —
caller gets a dangling pointer after the function returns.

**Fix:**
```go
func (m *Manager) GetConfig() *Config {
    m.mu.RLock()
    defer m.mu.RUnlock()

    configCopy := new(Config)  // heap allocation
    *configCopy = *m.config
    return configCopy
}
```

---

### Issue #3: Config Validation Bypass

**File:** `backend/core/network/dwcp/config.go:174-197`

**Bug:** Current `Validate()` returns `nil` immediately when `Enabled=false`,
allowing invalid config to be stored silently.

**Fix:** Always validate structural fields; only skip component-level checks when disabled.

See `QUICK_FIX_GUIDE.md` Issue #3 for the full replacement `Validate()` method (~50 lines).
The key change: move the structural field checks **before** the `if !c.Enabled { return nil }` guard.

---

### Issue #2: Incomplete Component Lifecycle

**File:** `backend/core/network/dwcp/dwcp_manager.go:17-23, 90-109, 138-143`

**Bug:** Components `transport`, `compression`, `prediction` are typed as `interface{}`
(no type safety, never initialized, never shut down).

**Fix — 4 steps (see QUICK_FIX_GUIDE.md Issue #2):**

1. Add concrete interfaces: `TransportLayer`, `CompressionLayer`, `PredictionEngine`,
   `SyncLayer`, `ConsensusLayer`
2. Update `Manager` struct to use typed fields (not `interface{}`)
3. Replace `Start()` to initialize components with proper error handling
4. Replace `Stop()` to shut down components in reverse order with `m.wg.Wait()`

---

### Issue #4: No Error Recovery / Circuit Breaker

**File:** `backend/core/network/dwcp/dwcp_manager.go` (whole file)

**Bug:** `Start()` always returns `nil` even when component initialization fails.
No health monitoring, no restart logic, no circuit breaker.

**Fix — skeleton provided in QUICK_FIX_GUIDE.md Issue #4:**
- Add `healthMonitoringLoop()` goroutine started in `Start()`
- Add `checkComponentHealth()` that calls `IsHealthy()` on each component
- Add exponential backoff retry in `Start()` (3 attempts, 2^n second delays)

---

### Verify All 5 Fixes

```bash
cd backend/core/network/dwcp

# 1. Build must pass
go build ./...

# 2. Race detector must show no races
go test -race ./...

# 3. All tests must pass
go test ./...

# 4. Vet must be clean
go vet ./...

# 5. Quick summary
echo "=== Build ===" && go build ./... && echo "OK"
echo "=== Race ===" && go test -race ./... && echo "OK"
```

**When all pass:** `bd close novacron-7q6.1 --reason "All 5 P0 issues fixed and tests pass"`

---

## Priority 2 (P1): DWCP Phase 2 — Intelligence

**Beads task:** `novacron-92v`
**Prerequisite:** `novacron-7q6.1` closed

Implement PBA (Predictive Bandwidth Allocation) + ITP (Intelligent Task Partitioning).

- **PBA:** LSTM model in `backend/core/network/dwcp/prediction/` — predicts bandwidth
  needs based on historical traffic patterns, enables proactive throttling/scaling
- **ITP:** Deep RL (DQN) for task partitioning decisions across WAN links

Target: 29/29 tests passing in the DWCP prediction test suite.
The Phase 2 foundation is already ~25K lines — review existing files before writing new code.

```bash
cd backend/core/network/dwcp
ls prediction/   # Check what already exists
go test ./prediction/... -v  # See which tests pass/fail
```

---

## Priority 3 (P1): DWCP Phase 3 — Synchronization

**Beads task:** `novacron-9tm`
**Prerequisite:** `novacron-92v` closed

Implement:
- **ASS** (Adaptive State Synchronization): multi-region state sync with conflict resolution
- **ACP** (Adaptive Consensus Protocol): consensus for distributed state changes

Target directories:
- `backend/core/network/dwcp/sync/`
- `backend/core/network/dwcp/consensus/`

---

## Priority 4 (P2): Fix Import Cycle

**Beads task:** `NC-2hk`
**Can be done in parallel with DWCP work (independent)**

**Cycle:** `monitoring → vm → federation → vm` (circular)

**Affected files:**
- `backend/core/monitoring/kvm_vm_manager.go`
- `backend/core/federation/cross_cluster_components.go`

**Fix strategy:** Extract the shared types (VM state, metrics types) that both packages
need into a new `backend/core/types/` package. Both `monitoring` and `federation`
can then import from `types` without creating a cycle.

```bash
# Verify the cycle exists
cd backend/core
go build ./... 2>&1 | grep "import cycle"
```

---

## Priority 5 (P2): Production Deployment Pipeline

**Beads task:** `novacron-9wq` (already in_progress)

Docs exist in `docs/deployment/`. Implement the actual GitHub Actions CI/CD workflows
described there as `.github/workflows/` YAML files.

---

## Key File Reference

| File | Purpose |
|------|---------|
| `backend/core/network/dwcp/dwcp_manager.go` | P0 issues #1, #2, #4, #5 |
| `backend/core/network/dwcp/config.go` | P0 issue #3 |
| `backend/core/network/dwcp/transport/rdma/rdma_cgo.go` | CGO types (add build tag) |
| `backend/core/network/dwcp/transport/rdma/rdma.go` | Uses CGO types (no change needed) |
| `backend/core/network/dwcp/QUICK_FIX_GUIDE.md` | Copy-paste fixes for all 5 issues |
| `backend/core/network/dwcp/CRITICAL_ISSUES_TRACKER.md` | Detailed analysis + test cases |

---

## Quick Start for This Session

```bash
# 1. Claim the P0 task
bd update novacron-7q6.1 --status in_progress

# 2. Fix RDMA CGO (Step 0)
# Add build tag to rdma_cgo.go + create rdma_stub.go

# 3. Verify compilation
cd backend/core/network/dwcp && go build ./...

# 4. Apply fixes #1, #5, #3 (quick wins first)
# Edit dwcp_manager.go + config.go

# 5. Run tests after each fix
go test -race ./...

# 6. Apply fixes #2, #4 (larger changes)
# Replace Start() / Stop() / add health monitoring

# 7. Final verification
go build ./... && go test -race ./... && go vet ./...

# 8. Close task
bd close novacron-7q6.1 --reason "All 5 P0 issues fixed, tests pass"

# 9. Move to Phase 2
bd update novacron-92v --status in_progress
```

---

## Environment Notes

- **Platform:** Jetson Thor, aarch64, Linux 6.8.12-tegra
- **Go:** 1.24+
- **No libibverbs available** — RDMA is stub-only on this machine
- **Services:** `docker start novacron-postgres novacron-redis novacron-qdrant`
- **API:** `make core-serve` → :8090 | Frontend: `cd frontend && npm run start` → :8092
- **Ports:** PostgreSQL 15432, Redis 16379, Qdrant 16333 (non-standard to avoid conflicts)

---

## Session Close Protocol

Before ending any session:

```bash
git status                          # Check what changed
git add <specific files>            # Stage code changes
bd sync                             # Commit beads changes
git commit -m "fix: ..."            # Commit code
bd sync                             # Commit any new beads changes
git push                            # Push to remote
```

**Never skip this.** Work is not done until pushed.
