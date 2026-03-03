# Claude Code + Claude-Flow: DWCP Critical Fixes Swarm (Massively Parallel)

## 🎯 Mission: Phase 1 Critical Fixes (P0 Issues)

You are leading a **massively parallel swarm** to fix 5 critical P0 issues in the DWCP codebase. This runs **concurrently** with the neural training team.

**Epic**: `novacron-7q6` (Beads)  
**Phase**: Phase 1 - Critical Fixes (P0 Issues)  
**Target**: Fix all 5 issues with zero regressions, maintain green build/test status

---

## 🚨 DWCP SANITY CHECKLIST (DO NOT RE-OPEN THESE)

**These are ALREADY FIXED and GREEN - DO NOT touch them:**

1. ✅ Resilience manager API - External code uses `SetDegradationLevelChangeCallback()`, `ForEachErrorBudget()`, `GetAllMetrics()`
2. ✅ Network tier detection - Exactly 3 tiers (Tier1, Tier2, Tier3), NO Tier4
3. ✅ Circuit breaker error code - `ErrCodeCircuitOpen = "CIRCUIT_OPEN"` is defined
4. ✅ Offset parsing - Wire-format compatibility maintained
5. ✅ Federation adapter v3 - Uses `&ClusterConnectionV3`, not v1 types
6. ✅ Partition integration - `Manager.AddTaskPartitioner()` and `PartitionTask()` are wired and tested
7. ✅ Build & tests - `go build ./backend/core/network/dwcp/...` and `go test ./backend/core/network/dwcp/...` are GREEN

**Invariant**: Any change that breaks build or tests is a regression and must be treated as a bug.

---

## 🔧 5 Critical Issues to Fix

### Issue 1: Race Condition in Metrics Collection
**File**: `backend/core/network/dwcp/dwcp_manager.go:225-248`  
**Problem**: Data race when accessing metrics without proper lock ordering  
**Fix Strategy**:
- Acquire locks in consistent order across all methods
- Use local variables to bridge mutex boundaries
- Never hold multiple locks simultaneously if avoidable
- Add race detector tests: `go test -race ./backend/core/network/dwcp/...`

**Acceptance**:
- No race conditions detected by `go test -race`
- Metrics collection maintains accuracy
- No performance degradation

---

### Issue 2: Component Lifecycle Management
**Files**: `backend/core/network/dwcp/interfaces.go`, all component files  
**Problem**: No standardized initialization/shutdown interfaces  
**Fix Strategy**:
- Define `Lifecycle` interface with `Start(ctx)`, `Stop()`, `IsRunning()` methods
- Implement for all components: AMST, HDE, PBA, ASS, ACP, ITP, Resilience, CircuitBreaker
- Manager coordinates lifecycle: start in dependency order, stop in reverse
- Add lifecycle state tracking and validation

**Acceptance**:
- All components implement `Lifecycle` interface
- Manager starts/stops components correctly
- Tests verify proper initialization and cleanup
- No resource leaks (goroutines, connections, file handles)

---

### Issue 3: Configuration Validation
**File**: `backend/core/network/dwcp/config.go`  
**Problem**: Config validation skipped when `Enabled=false`, allowing invalid configs  
**Fix Strategy**:
- Always validate config structure regardless of `Enabled` flag
- Separate validation into: structural validation (always) + runtime validation (when enabled)
- Add comprehensive validation tests for all config fields
- Return detailed validation errors with field paths

**Acceptance**:
- Invalid configs rejected even when `Enabled=false`
- Clear error messages for all validation failures
- 100% test coverage for validation logic

---

### Issue 4: Error Recovery & Circuit Breaker
**Files**: `backend/core/network/dwcp/circuit_breaker.go`, `resilience/circuit_breaker.go`  
**Problem**: No health monitoring, circuit breaker not integrated with error recovery  
**Fix Strategy**:
- Add health monitoring to all critical paths (transport, compression, consensus)
- Integrate circuit breaker with resilience manager
- Implement exponential backoff retry with jitter
- Add circuit breaker state transitions: Closed → Open → HalfOpen → Closed
- Track error budgets and trigger degradation levels

**Acceptance**:
- Circuit breaker prevents cascading failures
- Health checks detect and isolate failing components
- Automatic recovery when health improves
- Metrics track circuit breaker state transitions

---

### Issue 5: Unsafe Config Copy
**File**: `backend/core/network/dwcp/config.go` (likely in a `Copy()` or `Clone()` method)  
**Problem**: Returning pointer to stack variable instead of heap allocation  
**Fix Strategy**:
- Allocate config copy on heap: `newCfg := new(Config); *newCfg = *c`
- Deep copy all pointer fields (slices, maps, nested structs)
- Add tests to verify copy independence (modify original, verify copy unchanged)
- Use `go vet` to catch similar issues

**Acceptance**:
- Config copies are independent (no shared memory)
- `go vet` passes with no warnings
- Tests verify deep copy semantics

---

## 🏗️ Swarm Architecture (Massively Parallel)

### Coordination Layer (1 agent)
```javascript
Task("Swarm Coordinator", `
  Initialize Claude-Flow mesh topology for 5 parallel fix teams.
  Coordinate via hooks: pre-task, post-edit, notify, session-restore.
  Track progress, resolve conflicts, ensure no team breaks DWCP sanity checklist.
  Report status every 15 minutes.
`, "hierarchical-coordinator")
```

### Fix Teams (5 parallel teams, 3 agents each = 15 agents)

**Team 1: Race Condition Hunters**
```javascript
Task("Race Analyzer", "Analyze dwcp_manager.go for all race conditions using go test -race", "code-analyzer")
Task("Race Fixer", "Fix race conditions with proper lock ordering and local variables", "coder")
Task("Race Tester", "Write comprehensive race detector tests", "tester")
```

**Team 2: Lifecycle Engineers**
```javascript
Task("Lifecycle Architect", "Design Lifecycle interface and component integration strategy", "system-architect")
Task("Lifecycle Implementer", "Implement Lifecycle for all 8+ DWCP components", "backend-dev")
Task("Lifecycle Validator", "Test initialization/shutdown sequences and resource cleanup", "tester")
```

**Team 3: Config Validators**
```javascript
Task("Config Analyzer", "Audit all config validation logic and identify gaps", "code-analyzer")
Task("Config Hardener", "Implement comprehensive validation with detailed error messages", "coder")
Task("Config Tester", "Achieve 100% test coverage for validation logic", "tester")
```

**Team 4: Resilience Squad**
```javascript
Task("Health Monitor Designer", "Design health monitoring for transport/compression/consensus", "system-architect")
Task("Circuit Breaker Integrator", "Integrate circuit breaker with resilience manager and error recovery", "backend-dev")
Task("Resilience Tester", "Test circuit breaker state transitions and automatic recovery", "tester")
```

**Team 5: Memory Safety Team**
```javascript
Task("Memory Analyzer", "Find all unsafe config copies and pointer issues with go vet", "code-analyzer")
Task("Memory Fixer", "Fix unsafe copies with proper heap allocation and deep copy", "coder")
Task("Memory Tester", "Verify copy independence and go vet compliance", "tester")
```

### Quality Assurance Layer (3 agents)
```javascript
Task("Integration Tester", "Run full DWCP test suite after each fix, ensure no regressions", "tester")
Task("Performance Validator", "Benchmark critical paths, ensure no performance degradation", "performance-benchmarker")
Task("Security Auditor", "Review all changes for security implications", "reviewer")
```

---

## 🔄 Execution Protocol

### Phase 1: Initialization (Single Message)
```bash
# Claude-Flow coordination setup
npx claude-flow@alpha hooks pre-task --description "DWCP Critical Fixes Swarm"
npx claude-flow@alpha swarm init --topology mesh --max-agents 20

# Spawn ALL 19 agents in ONE message (coordinator + 15 fix agents + 3 QA agents)
```

### Phase 2: Parallel Execution
Each team works independently with coordination via hooks:
```bash
# Before work
npx claude-flow@alpha hooks session-restore --session-id "swarm-critical-fixes"

# During work
npx claude-flow@alpha hooks post-edit --file "<file>" --memory-key "swarm/team<N>/progress"
npx claude-flow@alpha hooks notify --message "Team <N> completed <task>"

# After work
npx claude-flow@alpha hooks post-task --task-id "<task>"
```

### Phase 3: Integration & Validation
```bash
# After each fix
go test -race ./backend/core/network/dwcp/...
go vet ./backend/core/network/dwcp/...
go build ./backend/core/network/dwcp/...

# Full validation
go test -v -cover ./backend/core/network/dwcp/...
```

---

## 📊 Success Criteria

**Per-Issue**:
- [ ] Fix implemented and tested
- [ ] No regressions in existing tests
- [ ] Race detector passes (`go test -race`)
- [ ] Vet passes (`go vet`)
- [ ] Code review approved

**Overall**:
- [ ] All 5 issues fixed
- [ ] Build green: `go build ./backend/core/network/dwcp/...`
- [ ] Tests green: `go test ./backend/core/network/dwcp/...`
- [ ] No race conditions: `go test -race ./backend/core/network/dwcp/...`
- [ ] No vet warnings: `go vet ./backend/core/network/dwcp/...`
- [ ] DWCP sanity checklist still green (no re-opened issues)
- [ ] Performance maintained or improved
- [ ] Beads epic `novacron-7q6` updated with completion status

---

## 🚀 Estimated Timeline

With 19 parallel agents:
- **Initialization**: 5 minutes
- **Parallel Execution**: 30-45 minutes (all 5 issues simultaneously)
- **Integration & Validation**: 15 minutes
- **Total**: ~60 minutes for all 5 critical fixes

---

## 📝 Deliverables

1. **Fixed Code**: All 5 issues resolved in `backend/core/network/dwcp/`
2. **Test Suite**: Comprehensive tests for each fix
3. **Documentation**: Update DWCP_CODE_QUALITY_ANALYSIS.md with fix details
4. **Metrics**: Performance benchmarks before/after
5. **Beads Update**: Mark all 5 subtasks complete in epic `novacron-7q6`

---

## 🎯 PROMPT FOR CLAUDE CODE (COPY THIS)

```
MISSION: Execute DWCP Critical Fixes Swarm (Massively Parallel)

Read execution plan: docs/CLAUDE_CODE_CRITICAL_FIXES_SWARM_PROMPT.md

CRITICAL RULES:
1. Spawn ALL 19 agents in ONE message (1 coordinator + 15 fix agents + 3 QA agents)
2. Use Claude-Flow hooks for coordination (pre-task, post-edit, notify, session-restore)
3. NEVER break DWCP sanity checklist (see doc)
4. Run tests after EVERY change: go test -race ./backend/core/network/dwcp/...

SWARM ARCHITECTURE:
- 1 Coordinator (mesh topology, conflict resolution, progress tracking)
- 5 Fix Teams (3 agents each = 15 agents):
  * Team 1: Race Condition Hunters (analyzer, fixer, tester)
  * Team 2: Lifecycle Engineers (architect, implementer, validator)
  * Team 3: Config Validators (analyzer, hardener, tester)
  * Team 4: Resilience Squad (designer, integrator, tester)
  * Team 5: Memory Safety Team (analyzer, fixer, tester)
- 3 QA Agents (integration tester, performance validator, security auditor)

EXECUTION:
1. Initialize Claude-Flow: npx claude-flow@alpha swarm init --topology mesh --max-agents 20
2. Spawn all 19 agents in SINGLE message with Task tool
3. Each team fixes their issue in parallel
4. QA validates after each fix
5. Coordinator ensures no regressions

TARGET: Fix all 5 P0 issues in ~60 minutes with zero regressions.

START NOW. Spawn the swarm.
```

