# NovaCron → Production: Review · Plan · Execute (ruflo swarm)

> **Paste this whole file as your first message in a fresh PLAN-MODE session at repo root
> (`~/thordrive/novacron`).** It drives the project to an *honest*, production-ready state
> using ruflo swarms, `bd` (beads) tracking, and GitNexus impact analysis. Do not start
> editing code — produce and get approval on a plan first, then execute it.
>
> **Refreshed 2026-07-11** (this session): closed `novacron-ffc` (the reentrant-lock
> deadlock this file's author likely hadn't seen fixed yet), then `novacron-2ub`
> (mode-switch latch) and `novacron-9yb` (stub TLS) — both listed below as "real hardening
> (do these)" are now DONE. Also found: `bd`'s own priorities are inverted by the same
> fabrication pattern this file warns about (8 of 10 open P0/P1 issues are the aspirational
> epics flagged for Phase 0 triage, not real hardening — see §1). §3's ruflo tool names
> corrected against a live ToolSearch; `coordination_orchestrate` and bare
> `memory_store`/`memory_search` did not resolve by those names this session. Nothing else
> here was found stale.

---

## 0. MISSION (read this twice)

Bring NovaCron to **production-ready completion for its real purpose**: a multi-arch
(arm64/Jetson + x86_64) KVM VM manager with live + block migration, auth, Raft consensus,
distributed storage, and monitoring — served by `backend/cmd/api-server` and the Next.js
frontend, deployed via `docker-compose.yml` + `docker/api.Dockerfile`.

**"Production-ready" here means:** the real subsystems are hardened, every tracked bug is
fixed or consciously deferred, the fabricated/aspirational cruft is triaged (kept-and-built,
deferred-and-labeled, or deleted), all claims are true, and the whole thing passes its
**canonical CI gate** and a real deploy smoke test. It does **NOT** mean building out the
$1B-ARR / 98%-accuracy-neural / quantum epics. Those are the exact fantasies that produced
the mess `STATUS.md` is still cleaning up.

### Prime directives (non-negotiable — this repo's history is a cautionary tale)

1. **Evidence over assertion.** "Done" = the canonical CI gate passes for the change, plus a
   discrimination-proven test (the test fails when the fix is reverted). Nothing is "complete"
   because an agent said so. Read `STATUS.md` — it is the *only* trustworthy status doc; the
   files in `docs/archive/` overstate completion and must not be trusted.
2. **Never fabricate.** No invented benchmark numbers, no "achieved" perf claims without a
   reproducible measurement, no fake fallbacks, no stubs relabeled as features. `bd` issue
   `novacron-2vo` exists *because* 64 markdown docs cited fabricated DWCP numbers as ACHIEVED.
   Do not add to that debt; retire it.
3. **No new moonshots.** Do not create speculative scaffolding, new "enterprise/fortune500/
   ipo/community/quantum" trees, or aspirational abstractions. 96 such packages (~344K LOC)
   were already deleted. Consolidate; don't expand surface area.
4. **Impact before edit.** Before modifying any exported symbol, run GitNexus
   `impact({target, direction:"upstream"})` and report the blast radius. Warn on HIGH/CRITICAL.
   Run `lsp references` before changing any exported Go/TS symbol — missed callsites are bugs.
5. **Track everything in `bd`.** One `bd` issue per unit of work; claim before starting; close
   with the verifying command in the comment. `bd` is the swarm's shared work queue.
6. **Clean cutover.** Migrate every caller; leave no shims, dead aliases, or `.disabled`
   files that a real fix would remove. If you quarantine, file a `bd` issue with the re-enable
   criteria (match the existing `.go.disabled` convention).
7. **Honesty in docs.** Every doc you touch must describe what the code *actually does today*.
   Update `STATUS.md` at the end; do not resurrect the archived completion reports.

---

## 1. GROUND TRUTH (verify, don't trust this summary)

**Authoritative sources — read them in Phase 1:**
- `STATUS.md` — the honest state: what's *verified real* vs *simulated/in-progress*.
- `.github/workflows/ci.yml` — the **canonical gate** (the definition of "it works"):
  - Backend: `go build ./backend/cmd/api-server` then
    `go test ./backend/cmd/api-server ./backend/api/graphql ./backend/api/security ./backend/api/websocket ./backend/pkg/config`
  - backend/core (separate module): `cd backend/core && go test -short ./vm/` (non-recursive)
  - Frontend: `cd frontend && npm ci` → 14 curated jest suites (see ci.yml) → `npm run build`
  - x86 KVM boot smoke test (non-gating, `continue-on-error`).
- `bd ready` / `bd list --status open` — the live backlog (24 open, 3 in-progress, 43 closed
  as of 2026-07-11). **Caveat:** `bd` priority is not trustworthy signal here — see the
  priority-inversion note below; triage by the KEEP/DEFER/DELETE split, not by P-number.
- `output/root-build-errors.txt` — snapshot of the **off-path rot**: `go build ./...` at repo
  root fails across `backend/community/*`, `backend/ipo/*`, `backend/enterprise/fortune500`,
  `backend/operations/{six_nines,runbooks,support,onboarding,command}`, `backend/chaos`,
  `backend/deployment`, `backend/scaling`, `adapters/pkg/azure`, `research/*`,
  `marketplace/server`, `sdk/examples`, `temp_main_files`, `config/performance`. These are on
  **no production path** and are excluded from the gate — but they are the single biggest
  "is this production-ready?" liability and MUST be dispositioned (delete or fix), not ignored.
  Same disposition applies to the **root-level doc/prompt sprawl**: 31 `*.md`/`*.txt` files
  at repo root (this file included) plus per-tool state dirs (`.bmad-core/`, `.augment/`,
  `.gemini/`, `.qwen/`, `.hive-mind/`, `.claude-flow/`, …) — residue of many different agent
  frameworks having been pointed at this repo. Fold into Phase 1's off-path-rot scout: which
  docs are current (`STATUS.md`, `AGENTS.md`, `CLAUDE.md`, this file), which are superseded
  history (archive), which are dead prompts nobody will paste again (delete).

**Known state (verify each claim before acting on it):**
- ✅ Real & CI-green: single-node KVM lifecycle, live migration (QMP) + block migration
  (NBD drive-mirror) proven cross-node, auth (JWT/TOTP/OAuth2/RBAC), Raft consensus,
  distributed storage w/ dedup, Prometheus/OTel monitoring, frontend strict-mode type-clean.
- ⚠️ Simulated / partial / off-path: federation cross-region data plane (mechanism proven,
  not wired live), multicloud abstraction (behind `//go:build novacron_multicloud`, hollow
  providers), `network/dwcp` tree (experimental WAN protocol, broadly red off-CI tests).
- ❌ Aspirational-only in `bd` (do NOT chase without explicit human keep-decision in Phase 0):
  `novacron-7q6*` (neural training → "98% accuracy", ProBFT, MADDPG, TCS-FEEL),
  `novacron-7pt` ("Phase 13: DWCP v5 GA, $1B ARR & Industry Dominance"), `novacron-ahm`.

**Priority inversion (found 2026-07-11):** `bd`'s own P-numbers cannot be trusted to route
work. Of the 10 open P0/P1 issues, 8 ARE the aspirational epics above (`novacron-7q6`,
`-7q6.2..7q6.5`, `-7pt`, `-ahm`, `-aca`, `-ttc`, `-9tm`, `-92v` — all P0/P1), while every
confirmed-real hardening item (`-1h2`, `-2vo`, `-y45`, `-94l`, `-3cd`, `-2hk`, …) sits at
P2/P3. Re-priority in Phase 2 by the KEEP/DEFER/DELETE split below, not by the number
already on the issue — the number is itself a symptom of the fabrication problem.

**The backlog cleanly bifurcates — use this split in Phase 0:**
- *Real hardening — DONE:* ~~`novacron-9yb`~~ (TLS stub → real ECDSA P-256 self-signed cert,
  commit `b6fa91ab`), ~~`novacron-2ub`~~ (mode-switch latch → bidirectional auto-recovery,
  commit `b6fa91ab`), ~~`novacron-ffc`~~ (the deadlock underlying both, commit `e7da3963`).
- *Real hardening (still open):* `novacron-1h2` (quarantine cap), `novacron-y45` (unbounded
  vmBaselines map), `novacron-94l` (compression-none aliasing), `novacron-2hk` (import cycle),
  `novacron-2vo` (fabricated numbers, 64 files — filed 2026-07-11), `novacron-hpa`,
  `novacron-v4y`, `novacron-3cd`, `novacron-i7r`, `novacron-e50`, `novacron-77u`,
  `novacron-976`, `novacron-113`.
- *Aspirational (triage first):* the `7q6*` / `7pt` / `ahm` epics above.

---

## 2. WORKFLOW — four phases, human-gated between plan and execution

### PHASE 0 — Scope decision gate (do this first, present to human)
Produce a one-screen **triage table** of the aspirational epics (`7q6*`, `7pt`, `ahm`, DWCP
phase tasks) with a recommendation for each: **KEEP** (on the production path, build it right),
**DEFER** (real but out of scope for GA — label honestly, leave working stubs that error
honestly), or **DELETE** (fantasy scaffolding — remove, recoverable via git). Default
recommendation, per the prime directives: DEFER or DELETE unless the epic serves the real VM
platform. **Ask the human to confirm the KEEP/DEFER/DELETE set before planning further.**

### PHASE 1 — Honest production-readiness review (read-only scout swarm)
Spin a **read-only** review swarm (scouts — no edits) to audit each real subsystem against a
production bar and produce a **gap register**. Fan out one scout per subsystem, in parallel:

- VM lifecycle & drivers (`backend/core/vm`) — incl. the quarantined `.go.disabled` set
- Live/block migration (QMP/NBD, cross-node path)
- Auth / RBAC / tenants / secrets handling
- Consensus (Raft) & distributed locks
- Storage (replication, dedup, leaks)
- Monitoring / telemetry / anomaly detection
- Networking (L4/L7 LB, overlay) + the `network/dwcp` off-path tree
- Frontend (Next.js) — the ~33 off-gate `tsc` test errors + `auth-accessibility` a11y finding
- Security posture (the `9yb` TLS stub, reputation system, apparmor, `policies/*.rego`)
- Deploy/ops (docker-compose, k8s, systemd, terraform) — does it actually stand up?
- **Off-path rot disposition** (`output/root-build-errors.txt` trees) — delete vs fix, per file

Each scout reports: what's real, what's stubbed/simulated, concrete gaps to production, and a
mapping to existing-or-new `bd` issues. Use GitNexus `query`/`context`/`impact` and the
codegraph/serena MCP tools instead of blind grepping. Consolidate into one **gap register**
(store in ruflo memory + write to `local://novacron-gap-register.md`).

### PHASE 2 — Work plan (synthesize, file issues, get approval)
From the gap register:
1. Reconcile with `bd`: update/close stale issues, file new ones for every real gap. Set
   priorities (P0 blockers → P3 nice-to-have) and dependency links (`bd dep`).
2. Group issues into **dependency-ordered waves** (e.g. Wave 1 = build-repair/deletion +
   P1 security; Wave 2 = migration/storage/consensus hardening; Wave 3 = docs-honesty +
   deploy smoke + observability). Prerequisites that everything depends on (e.g. resolving
   the import cycle `novacron-2hk`, deleting the off-path rot) run **first and inline**, not
   in parallel.
3. Define **acceptance criteria per wave**, anchored to the canonical gate + a discrimination
   test per fix.
4. **Present the full plan (waves, issues, agent assignment, acceptance) for human approval.
   This is the exit from plan mode.** Do not execute until approved.

### PHASE 3 — Execute with a ruflo swarm (after approval)
- Initialize the swarm (topology sized to real wave width, not vanity numbers):
  `mcp__ruflo_core_ruflo_swarm_init({ topology:"hierarchical", maxAgents:8, strategy:"specialized" })`
  (use `hierarchical-mesh` + higher maxAgents only if a wave genuinely has >8 independent slices;
  cap at real independence, never pad).
- Spawn **specialist** agents from the project's own library in `.claude/agents/` — match the
  agent to the subsystem, e.g.:
  - `vm-migration-architect`, `hypervisor-integration-specialist`, `ha-fault-tolerance-engineer`
  - `storage-volume-engineer`, `network-sdn-controller`, `scheduler-optimization-expert`
  - `security-compliance-automation`, `performance-telemetry-architect`, `database-state-engineer`
  - `core/{coder,tester,reviewer}`, swarm coordinators in `swarm/`
  Spawn via `mcp__ruflo_core_ruflo_agent_spawn({ agentType, task, model:"sonnet"|"opus", memoryBase })`
  or the native `task` tool with one self-contained assignment per agent.
- **One `bd` issue per agent.** Each agent: `bd update <id> --status in_progress` → GitNexus
  `impact` on target symbols → implement the real fix (no stubs) → add a discrimination-proven
  test → run the exact canonical gate command for its area → `detect_changes()` to confirm scope
  → `bd close <id>` with the verifying command in the comment.
- **Parallel EXECUTION of independent slices only**; serialize a slice B behind A only if B
  strictly needs A's output. Give each parallel agent an isolated git worktree
  (`.claude/worktrees/`) to avoid collisions. Agents coordinate via ruflo memory + IRC, not by
  reading each other's files.
- Coordinate a wave with `mcp__ruflo_core_ruflo_coordination_orchestrate({ agents, strategy:"parallel", task })`;
  persist shared decisions with `memory_store` / retrieve with `memory_search`; checkpoint with
  `mcp__ruflo_core_ruflo_session_save` between waves.
- The orchestrator (you) owns cross-slice contracts, reviews each agent's diff, and NEVER lets
  a wave close on unverified claims.

### PHASE 4 — Verify, honest-docs, land the plane
1. **Full canonical gate must pass** (all commands in §1). Run the exact CI command set locally
   — a subset is not the gate (this repo has a documented red-streak lesson from doing exactly
   that).
2. **Purge fabrication** (`novacron-2vo`): scrub the 64 docs of ACHIEVED-but-fabricated numbers;
   replace with real measurements or explicit "target, not measured" labels.
3. **Rewrite `STATUS.md`** to the true post-work state. Delete/verify any `docs/archive/` claim
   that's now false.
4. Optional: a real deploy smoke test — `docker compose up`, hit the api-server, boot a VM
   through the HTTP API, screenshot the frontend.
5. **Land the plane** (per `AGENTS.md`): file follow-up issues → run quality gates →
   `git pull --rebase && bd sync && git push` → confirm `git status` shows up-to-date.
   Work is NOT complete until `git push` succeeds.

---

## 3. TOOL REFERENCE (use the real ones; don't reinvent)

- **Issue tracking:** `bd ready`, `bd show <id>`, `bd list --status open`, `bd update <id> --status in_progress`, `bd close <id>`, `bd dep`, `bd sync`.
- **Code intelligence (before edits):** GitNexus MCP — `impact`, `context`, `query`, `detect_changes({scope:"compare", base_ref:"main"})`, `rename` (never find-and-replace symbols); codegraph `codegraph_explore`; serena `initial_instructions` first.
- **Swarm (ruflo):** `mcp__ruflo_core_ruflo_swarm_init`, `mcp__ruflo_core_ruflo_agent_spawn`, `mcp__ruflo_core_ruflo_hive_mind_{init,spawn,join,status,memory,broadcast,consensus}`, `mcp__ruflo_core_ruflo_coordination_topology` (get/set/optimize — topology only, not task dispatch), `mcp__ruflo_core_ruflo_session_{save,list,info,restore}`; memory via `mcp__ruflo_core_ruflo_agentdb_{hierarchical_store,pattern_store,batch}` (write) / `memory_retrieve` + `embeddings_search` (read); routing via `hooks_route`/`hooks_model_route`. **Verify exact names via `search_tool_bm25` before use** — `coordination_orchestrate` and bare `memory_store`/`memory_search` did NOT resolve in this session's discoverable set (~900 tools; results vary by query).
- **Specialist agents confirmed present** in `.claude/agents/`: `vm-migration-architect`, `database-state-engineer`, `core/{coder,planner,researcher,reviewer,tester}`, `github/{code-review-swarm,multi-repo-swarm,pr-manager,release-manager,issue-tracker,project-board-sync}`, `testing/validation/production-validator`, `devops/ci-cd/ops-cicd-github`, `v3/{security-architect,performance-engineer,swarm-memory-manager,collective-intelligence-coordinator,v3-integration-architect}`, `sublinear/*`, `templates/sparc-coordinator` — 100+ total (`glob .claude/agents/**/*.md` for the full roster). Some names cited elsewhere (e.g. `hypervisor-integration-specialist`) were NOT directly confirmed this session — verify with `glob` before assigning an agent by name.
- **Skills to load when relevant:** `swarm-init`, `sparc-methodology`, `systematic-debugging`, `verification-before-completion`, `intended-vs-implemented`, `shipping-artifacts`, `ship-gate`, `incident-commander`, `test-driven-development`, `adversarial-reviewer`. For the honesty audit specifically: `intended-vs-implemented` and `verification-before-completion` are load-bearing.
- **Canonical gate commands:** see §1 (this is the definition of done).

## 4. DEFINITION OF DONE (measurable, honesty-anchored)

- [ ] Canonical CI gate green locally on the exact command set (backend build+tests, `vm/` -short, frontend 14 suites + build).
- [ ] Every `bd` "real hardening" issue closed with a discrimination-proven test, or explicitly deferred with a filed issue + rationale.
- [ ] Off-path rot in `output/root-build-errors.txt` dispositioned per file (deleted or building), and `go build ./...` from repo root either passes or fails ONLY on documented, tracked, off-path exclusions.
- [ ] Zero fabricated metrics in docs; `novacron-2vo` closed; `STATUS.md` reflects reality.
- [ ] KEEP/DEFER/DELETE decisions from Phase 0 executed exactly as the human approved.
- [ ] Real deploy smoke test passes (VM created/booted through the API), if in scope.
- [ ] `git push` succeeded; `bd sync` done; `git status` up-to-date with origin.

**Anti-goals (automatic failure):** any new fabricated number; any stub presented as a feature;
any "done" without a passing gate; any new moonshot/enterprise-fantasy tree; any cross-file
symbol rename done by text replace; any epic built without an explicit Phase 0 KEEP decision.
