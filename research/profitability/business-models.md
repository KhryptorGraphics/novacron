# Business Model Comparison: Open-core vs SaaS vs Marketplace

**Evidence synthesis from this session's research:** competitive-landscape.md (14 competitors), monetization-models.md (6 monetization models), technical-audit.md (what's real vs stub/fabricated in codebase), ai-gpu-angle.md (GPU market + honest gap analysis), market-demand.md (decentralized compute market sizing + sovereign premium).

**Summary framework:** Four monetization levers for NovaCron, ordered from easiest to harder bootstrapping given its current standing (zero paying customers, zero external node operators, zero external liquidity). Each lever is evaluated against NovaCron's actual repo state, not aspirational roadmap.

---

## 1. Open-core (zero feature-gating, support-as-product)

**Proxmox precedent** (read in competitive-landscape.md): AGPLv3 base, €120–1,100/yr per-socket support subscription across 4 tiers. Revenue comes entirely from paid support; zero feature gating.

### Real-fit factors
- **One-time cost to add**: ★★★★☆. The repo already has an AGPLv3-licensed KVM fabric (cluster_join.go, fabric_jobs.go, fabric_transfers.go + driver_kvm_migrate.go — confirmed real, compiled into canonical binary per technical-audit.md). No engineering work required to "open-source the base."
- **Liquidity requirement**: ★★★★☆. Zero external supply/demand needed. First customer pays for support, not capacity aggregation.
- **Bootstrap difficulty**: ★★★★☆. First customer (or first org to self-host) pays for 1-year support at €120–1,100. That's real revenue from day 1 if you have a prospect who already runs KVM and trusts the AGPL.
- **Conversion rate**: Proxmox's own figure is effectively 100% of paying customers come from the installed base (they start with the free AGPL download, upgrade when they need SLA/ceph support). Not an arbitrary assumption — it's their documented trajectory.
- **Cross-compatible with fabric differentiators**: Yes. The bandwidth-aware placement, compression decisions, and signed cluster join are features that *could* be gated behind the paid tier, as Proxmox gates Ceph, BGP/EVPN, and its dedicated backup server.

### What changes vs today
- Must NOT copy the $100M+ ARR / "$1B ARR" marketing from `backend/business/revenue/acceleration_engine.go` — that entire cluster is [fabricated] per technical-audit.md, deliberately isolated from the canonical import graph, and has zero callers in the running binary.
- Gate **support + policy features**, not engineering features: e.g., Ceph integration, BGP/EVPN, backup server, priority migration queue, SLA reporting.
- Keep the base `go build ./cmd/api-server` green with zero feature gating. The `//go:build` tags for `novacron_enhanced`, `novacron_improved`, etc. remain separate alternate binaries (they already exist and the canonical build ignores them).

### Verdict for NovaCron today
**Recommended entry price**: €120–1,100/socket/yr (Proxmox's exact tier), scaled to a socket-equivalent unit. NovaCron counts sockets, not VMs, because live migration across nodes scales with socket count, not instance count.

**Fit score**: 4.5 / 5 for immediate bootstrapping; deduct 0.5 because conversion requires a prospect who already runs self-hosted KVM AND values an SLA — a narrower initial TAM than Proxmox's data-center audience.

---

## 2. SaaS / Hosted Control Plane

**Paying customer brings metal; NovaCron runs the control plane + watches jobs dispatch + handles migration admission.**

### Real-fit factors
- **Liquidity requirement**: ★☆☆☆☆. Critical problem: NovaCron has zero external node operators today. A SaaS control plane is useless without nodes to orchestrate. Every prior SaaS play (Fly.io, Railway, Render) owned the metal or had a waitlist of hundreds of early adopters before launch. NovaCron has neither.
- **Bootstrap difficulty**: ★☆☆☆☆. Must bootstrap both (a) the control plane and (b) a supply of nodes. Two-sided marketplace problem, proven deadly for early-stage infra tools (see Akash Network's token-based model — also high-friction, as market-demand.md documents: "tiny transacted base," GPU portion <20% even of Akash's own run-rate). Without nodes, the control plane is a toy; with nodes, it's a two-sided market.
- **Conversion rate**: N/A — no installed base to convert from.
- **Cross-compatible**: Yes. The fabric's existing telemetry (bandwidth per transfer, live migration with compression) is natural SaaS metrics.

### What changes vs today
- Must not launch without solving the node supply problem first. The `scripts/fabric/two-node-fabric-test.sh` harness is a start but requires the operator to provision and netns both nodes manually — not a SaaS onboarding flow.
- Could document a "bring-your-own-node" SaaS path (customer installs api-server on their own metal, pays for the hosted control-plane that watches + charges for cross-node migration). This avoids the two-sided market bootstrapping problem but also reduces the value proposition (the customer could just run the binary themselves).

### Verdict for NovaCron today
**NOT recommended as a first model**. The two-sided market problem (control plane + nodes) has killed many infra SaaS attempts. Without existing node operators, this is a chicken-and-egg dead end.

**Fit score**: 1.5 / 5. High risk of vaporware perception; deduct further if the two-node harness isn't polished into a SaaS onboarding flow first.

---

## 3. Marketplace / Take-rate

**Aggregate capacity from external node operators, take a % of compute sold.**

### Real-fit factors
- **Liquidity requirement**: ★★★★☆. The decisive factor: what density of supply/demand is needed before a marketplace works? market-demand.md documents this harshly: "tiny transacted base," "GPU portion <20% even of Akash's own run-rate," and Akash's March 2026 upgrade that "eliminated explicit take-rates entirely in favor of token-level burn/mint seigniorage ('providers earn competitive revenue without take rates')." The report also flags "DePIN supply [that] is unverified" and "provider-quality uncertainty" for io.net.
- **Bootstrap difficulty**: ★★★★☆. Requires a critical mass of external node operators *before* any revenue flows. NovaCron has zero external node operators today. Zero. The two-node harness is operator-controlled, not public. There is no "list of ready-to-join peers" to onboard.
- **Conversion rate**: N/A — no liquidity yet.
- **Cross-compatible**: Yes. Bandwidth-aware placement + compression decisions are natural marketplace features (you're selling capacity across heterogeneous links).

### What changes vs today
- Must NOT copy Akash's token-based model ("Burn-Mint-Equilibrium mainnet upgrade") unless NovaCron already has a native settlement token and oracle infrastructure — it does not. Akash's move was a rebranding, not a feature add-on.
- Could launch with a "fixed take-rate on confirmed cross-node migrations only" (not on idle capacity), since NovaCron already measures and admits transfers. Revenue per confirmed migration; zero revenue on idle fabric.

### Verdict for NovaCron today
**NOT recommended as a first model** without solving node supply first. market-demand.md explicitly flags this risk: the GPU portion is <20% even of Akash's run-rate, and the decentralized marketplace's liquidity is "worst at the high end."

**Fit score**: 1 / 5 for a standing-start repo with zero external node operators.

---

## 4. Usage-metered / Utility (the natural fit)

**Charge per actual resource consumption: vCPU-hours, GB-hours (storage), and GB of egress/bandwidth moved.**

### Real-fit factors
- **Liquidity requirement**: ★★★★★. Zero external supply needed. The telemetry already exists — market-demand.md confirms NovaCron's fabric measures per-transfer bandwidth (`fabric_transfers.go` computes ETA from measured link budget; `fabric_transfers.go` compression decision applies QMP `migrate-set-capabilities`/`migrate-set-parameters`). technical-audit.md confirms backup/DR and monitoring/alerting are real as libraries (though unwired). And `GET /api/cluster/links` exposes `{rtt_ms, throughput_bps, measured_at, stale}` per STATUS.md.
- **Bootstrap difficulty**: ★★★★☆. First customer pays for what they use. If they run 10 cross-node migrations of 2GB each over 50Mbit links, they pay for the bytes moved — not for "a seat" or "a subscription." This is a consumption model, not a seat model.
- **Conversion rate**: Naturally high for customers who already run workloads across nodes. Every migration they perform generates a billable event.
- **Cross-compatible**: Yes. This is NovaCron's genuine differentiator — no named competitor (Proxmox, Harvester, OpenNebula, KubeVirt, XCP-ng/XO, Fly.io, Railway, Render, Vast.ai, RunPod, Lambda Labs, Akash, Salad, io.net) offers bandwidth-aware placement, per-link admission-controlled transfers with measured budgets, and an adaptive compression engine for migration traffic, per technical-audit.md's comparison and competitive-landscape.md's summary.

### What changes vs today
- Wire `organization_id` into the VM query paths (the schema already has it — `users.organization_id` and `vms` needs it per technical-audit.md's STUB finding). This is the engineering gate: enable per-tenant attribution.
- Wire `vCPU-hours` and `GB-egress` into the transfer admission logic (already partially there via `fabric_transfers.go`'s measured link budget). Extend to a persisted usage table rather than the in-memory-only counters that exist today.
- Replace the in-memory-only `enterprise/billing/advanced_billing.go` (STUB, unwired, "$100M+ ARR" header) with a real, persisted billing model. The metadata model already exists in the Postgres schema (organizations, users.organization_id); extend it with `v_cpu_hours`, `gb_egress`, `migrations`, `created_at`.
- Expose a `GET /api/billing/invoice` endpoint that tallies per-tenant usage from the usage table and renders a PDF or CSV.
- Keep pricing simple: $0.03–0.05/GB egress (between Fly's cheapest band and Render's most expensive, per monetization-models.md §2) + $0.05/vCPU-hr for migration orchestration.

### Verdict for NovaCron today
**Recommended as the primary entry lever**. It leverages existing telemetry, requires zero external supply, and provides immediate billable events from the first customer's first migration. It also naturally dovetails with the open-core support model (pay-as-you-go metering + optional SLA support tier).

**Fit score**: 4.5 / 5. Highest of all four models for a standing start. The only deduction (0.5) is that the first customer must accept usage-based billing rather than a flat subscription — but this is a far lower barrier than bootstrapping a two-sided marketplace or a SaaS control plane without nodes.

---

## Final Recommendation: Four-Layer Hybrid Model (lowest risk, immediate revenue)

Following the evidence chain from this session's research, the optimal path for NovaCron from a standing start (zero customers, zero node operators) is a four-layer hybrid that builds from the easiest bootstrapping lever to the harder ones, exactly as Monetization-2's report §6 recommended — but now grounded in the actual repo state, not aspirational roadmap:

### Layer 1 — Open core with zero feature gating, support as product
- Base: confirmed AGPLv3-licensed KVM fabric (already compiled and running)
- Gate: support tier (Ceph integration, BGP/EVPN, backup server, SLA reporting, priority migration queue)
- Price: €120–1,100/socket/yr (Proxmox's exact published tiers, per competitive-landscape.md)
- Revenue timeline: Month 1 if you have a KVM-running prospect who values an SLA

### Layer 2 — Usage-metered utility on top of existing telemetry
- Metrics already partially present: per-transfer bandwidth, live migration with compression, link RTT/throughput probes
- Extend: persisted usage table (v_cpu_hours, gb_egress, migrations per organization_id)
- Price: $0.03–0.05/GB egress + $0.05/vCPU-hr for migration orchestration (per monetization-models.md §2, market-demand.md §5)
- Revenue timeline: Month 1 from Layer 1's first customer, once the usage table is persisted

### Layer 3 — Per-node enterprise support with multi-year discounts
- Pattern: Proxmox + Vates (€120–1,100/yr support × 1–3 year commitment discount)
- Anchor: existing Layer 1 customers who want SLA-backed production support
- Revenue timeline: Month 3 from Layer 1 customers upgrading

### Layer 4 — Marketplace take-rate (deferred until external node operators exist)
- Take rate: 10–15% of confirmed migration value, undercutting AWS Marketplace's 20% (per monetization-models.md §3)
- Trigger: only after Layer 1–3 have a proven customer base AND a public list of ready-to-join node operators
- Revenue timeline: Year 2+, conditional on solving the node-supply problem

### Why not start with any other model
- SaaS control plane: chicken-and-egg — no nodes = no value. Fit score 1.5/5.
- Marketplace take-rate: same problem, worse — no liquidity = no revenue. Fit score 1/5.
- Open-core alone: works but leaves the highest-value monetization lever (usage-metered) on the table. Can be added as Layer 2.

### Summary of fit scores
| Model | Fit score (1-5) | Key enabler | Key blocker |
|---|---|---|---|
| Open-core (support) | 4.5 | AGPL base already compiled | Prospect who values SLA over free |
| Usage-metered utility | 4.5 | Existing telemetry (bandwidth per transfer, migration ETA) | First customer accepts usage billing |
| SaaS control plane | 1.5 | Control-plane code already real | Zero external node operators |
| Marketplace take-rate | 1 | Bandwidth-aware placement is natural marketplace feature | Zero external node operators + liquidity risk |

**The four-layer hybrid** starts with the two 4.5/5 levers (open core + usage-metered) that have zero external liquidity requirement, then adds the discount+commitment layer (3), and defers the marketplace (4) until a node-supply problem is solved — exactly matching the evidence from this session's research and the original Monetization-2 recommendation, now grounded in NovaCron's actual repo state.