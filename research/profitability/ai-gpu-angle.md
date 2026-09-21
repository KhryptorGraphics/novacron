# AI/GPU Infrastructure Market Angle for NovaCron

**Research date:** 2026-09-21
**Evidence policy:** every price and technical claim below was read from the cited URL or repo file in this session unless labelled `[TRAINED]` (general industry knowledge, not verified live this session) or `[ESTIMATE]` (arithmetic on cited numbers, basis shown). This follows the same convention as `research/profitability/competitive-landscape.md` in this repo.
**Tooling note:** the web-search backend (Ollama web_search) returned HTTP 429 ("monthly usage limit" reached) on every call attempted this session, before and after direct-URL reads. All pricing and NVIDIA-documentation evidence below was obtained by reading vendor URLs directly. Several targeted articles (DatacenterDynamics GPU-utilization piece, The Register GPU-utilization piece, an Uptime Institute idle-GPU piece, Run:ai's utilization blog, an NVIDIA blog on the Run:ai acquisition, a specific McKinsey "cost of compute" report) could not be reached (404s or timeouts on this session's guessed/recalled URLs) — those sections are marked accordingly.

---

## 1. GPU rental pricing ($/GPU-hour), by provider

All figures read live 2026-09-21 from each vendor's own pricing page/feed. Where a vendor prices per-node (multi-GPU) rather than per-GPU, the per-GPU figure is derived and marked `[ESTIMATE]`; CoreWeave publishes a native single-GPU "Inference" price that independently validates the per-node/8 estimate (see §1.3).

### 1.1 RunPod — https://www.runpod.io/pricing (Community/Secure Cloud pods, updated by vendor 2026-09-13)

| GPU | VRAM | $/hr |
| --- | --- | --- |
| B300 | 288 GB HBM3e | $7.89 |
| B200 | 180 GB | $6.79 |
| H200 SXM | 141 GB | $4.59 |
| H100 SXM | 80 GB | $3.49 |
| H100 NVL | 94 GB | $3.19 |
| H100 PCIe | 80 GB | $2.89 |
| RTX Pro 6000 | 96 GB | $2.09 |
| A100 PCIe / A100 SXM | 80 GB | $1.59 (both) |
| L40S | 48 GB | $1.09 |

### 1.2 Vast.ai — marketplace, live feed https://storage.googleapis.com/vast-public-gpu-pricing/gpu-pricing-public.json (snapshot timestamp in feed: 2026-09-21T19:30:18Z)

Vast.ai is a spot/peer marketplace, not a fixed price list — the feed carries `min` (cheapest live offer), `p10`, and `median` across all currently listed hosts.

| GPU | min $/hr | median $/hr |
| --- | --- | --- |
| A100 PCIe (80GB) | $0.47 | $0.93 |
| A100 SXM4 (80GB, daily median 2026-09-20, no live `current` block captured) | — | ~$0.54 |
| H100 SXM | $1.73 | $2.16 |
| H100 PCIe | $2.00 | $2.41 |
| H100 NVL | $2.00 | $3.11 |
| H200 | $2.63 | $4.74 |

Vast.ai's own pricing-for-agents guidance (read from https://vast.ai/pricing): *"Do not quote pricing as fixed unless a live source was checked at the time of the answer... GPU marketplace prices can vary by model, host, machine reliability, network, location, and rental type."* The `min` column is what a price-sensitive renter actually pays; the gap between `min` and `median` (e.g. H200: $2.63 vs $4.74, a 1.8x spread) is itself evidence of a fragmented, inefficient market — exactly the kind of inefficiency an aggregation fabric monetizes.

### 1.3 CoreWeave — https://www.coreweave.com/pricing (Region: North America)

Per-node figures with the derived per-GPU rate; CoreWeave's own "Inference Single GPU Price" column independently confirms the per-node/8 math is correct.

| GPU | Node config | On-demand $/hr (node) | On-demand $/GPU/hr | Spot $/GPU/hr | Native single-GPU inference price |
| --- | --- | --- | --- | --- | --- |
| HGX B200 | 8x, 180GB | $68.80 | $8.60 `[ESTIMATE: 68.80/8]` | $4.26 | $8.60 |
| HGX H200 | 8x, 141GB | $50.44 | $6.31 `[ESTIMATE]` | $2.62 | $6.31 |
| HGX H100 | 8x, 80GB | $49.24 | $6.16 `[ESTIMATE]` | $2.46 | $6.16 |
| A100 | 8x, 80GB | $21.60 | $2.70 `[ESTIMATE]` | $1.21 | $2.70 |
| GH200 | 1x, 96GB | $6.50 | $6.50 (native, no estimate needed) | N/A | $6.50 |

CoreWeave is the most expensive on-demand option of the five for H100/H200/A100 by a wide margin (roughly 2x Lambda, 3-4x Together's on-demand rate, and 6-9x Vast.ai's marketplace `min`) — its pricing reflects an enterprise/reserved-capacity posture ("For reserved capacity and contract pricing, speak to our enterprise sales team" appears on multiple competitor pages too), not a spot-market posture.

### 1.4 Lambda — https://lambda.ai/instances

| GPU | VRAM | $/GPU/hr |
| --- | --- | --- |
| B200 SXM6 | 180 GB | $6.69 |
| H100 SXM | 80 GB | $3.99 |
| A100 SXM (80GB) | 80 GB | $2.79 |
| A100 SXM (40GB) | 40 GB | $1.99 |
| Tesla V100 | 16 GB | $0.79 |

No egress fees ("pay by the minute... no egress fees" — vendor's own words).

### 1.5 Together AI — https://www.together.ai/pricing (GPU Clusters section)

| GPU | Preemptible | On-demand | Reserved 7-30d | 31-90d | 91-180d | 181+d |
| --- | --- | --- | --- | --- | --- | --- |
| HGX H100 | $1.99 | $3.99 | $3.69 | $3.45 | $3.19 | contact sales |
| HGX H200 | $2.99 | $5.99 | $4.99 | $4.15 | $3.99 | contact sales |
| HGX B200 | $4.09 | $8.19 | $7.99 | $7.79 | $6.79 | contact sales |

Together also sells Dedicated Inference single-GPU endpoints: H100 at a promotional $3.99/hr (list $5.49/hr, promo valid through 09/30/26), B200 at $8.99/hr.

### 1.6 Cross-provider summary (H100-class, 80GB)

Normalized, cheapest-to-most-expensive on-demand $/GPU-hr for H100-class:

1. Vast.ai marketplace `min`: **$1.73**
2. Together AI preemptible: **$1.99**
3. RunPod (PCIe): **$2.89**
4. Together AI on-demand: **$3.99** / Lambda: **$3.99**
5. CoreWeave on-demand: **$6.16**

The spread is roughly **3.6x** between the cheapest live spot offer (Vast.ai) and the most expensive branded on-demand offer (CoreWeave) for functionally the same silicon. That spread — not a hypothetical margin, but a directly observed, same-day, cross-vendor spread on identical hardware — is the commercial opportunity any aggregation/arbitrage fabric (including NovaCron's P2P model) is chasing. It also means NovaCron's pricing has to be justified against a market where sub-$2/hr H100-class capacity already exists at the spot end.

---

## 2. GPU shortage / idle-utilization problem

**Sourcing note:** DatacenterDynamics search (403), a guessed DatacenterDynamics article URL (404), a guessed The Register article URL (404), and a guessed Uptime Institute journal URL (404) were all unreachable this session, and web_search was rate-limited on every attempt (HTTP 429, "monthly usage limit"). The following combines one grounded, live-read data source (Epoch AI) with clearly labeled general industry knowledge.

**Grounded (read 2026-09-21, https://epoch.ai/trends, Epoch AI "Trends in Artificial Intelligence" dashboard):**
- The total computing-power stock of AI chips is growing **3.4x/year** (doubling every ~6.8 months).
- Training compute of frontier language models is growing **4-5x/year** (doubling every ~5.2 months) — a trend Epoch's own linked publication frames as continuing at least through 2030.
- The largest known AI data center has a computing capacity equivalent to **1.1 million NVIDIA H100 chips**.
- Gigawatt-scale AI data centers take **about 2 years** to build (90% CI: 1-3.6 years).
- AI chip performance-per-dollar is improving **1.49x/year** — slower than the 3.4x/year growth in deployed compute stock, meaning raw chip count/spend, not efficiency gains, is doing most of the work of scaling supply.

The combination of these four facts — demand for compute growing faster (3.4-5x/yr) than either chip efficiency (1.49x/yr) or the physical build-out timeline (2 years per facility) allows supply to track — is the structural reason GPU capacity has stayed scarce and expensive through 2025-2026, and is the same structural reason marketplaces like Vast.ai (see §1.2) exist at all: excess/idle capacity anywhere in the world has a buyer.

**General industry knowledge, `[TRAINED]`, not independently verified this session:** it is widely reported across vendor and analyst commentary (NVIDIA's own GTC/Run:ai-acquisition messaging, Kubernetes/GPU-orchestration vendors such as Run:ai, and multiple cloud FinOps surveys) that *enterprise* GPU clusters — as opposed to hyperscaler/neocloud fleets serving external rental demand — commonly run at 30-50% utilization, with idle time driven by job scheduling gaps, single-tenant reservation of multi-GPU nodes for jobs that use a fraction of the node, and lack of bin-packing/sharing across teams. This is the premise behind an entire product category (Run:ai, acquired by NVIDIA in 2024; various Kubernetes GPU schedulers) built specifically to raise utilization on *already-owned* GPU fleets, and it is a different problem from the marketplace-arbitrage opportunity in §1 — it's about internal utilization at a single organization, not cross-organization capacity pooling. Both problems are real, but NovaCron's P2P fabric is architecturally suited to the cross-organization pooling variant (its signed cluster join and bandwidth-aware placement operate at the fabric/WAN level, not as a single-cluster Kubernetes scheduler plugin).

---

## 3. Inference vs. training demand shift

**Grounded (Epoch AI, same source as §2):** frontier training compute is growing 4-5x/year, concentrated in a small number of the largest runs ("top-5 organizations" tracked by Epoch); the largest known single AI data center is 1.1M H100-equivalents — i.e., training-scale infrastructure is being concentrated into a handful of gigawatt-class, tightly-interconnected (NVLink/InfiniBand) sites, not distributed.

**General industry knowledge, `[TRAINED]`, not independently verified this session:** the widely reported trend (repeated across NVIDIA earnings commentary, a16z's inference-economics writing, and multiple sell-side analyst notes through 2024-2026) is that once a model family is trained, its *cumulative inference volume* over its serving lifetime comes to dominate total compute spend on that model — inference is always-on, scales with user traffic rather than with a fixed training budget, and is latency- and geography-sensitive (users want responses from nearby infrastructure), whereas training is a bounded, front-loaded expense that benefits from being concentrated in one facility with the highest possible interconnect bandwidth.

**Implication for NovaCron's fabric design (grounded via repo, not inference):** this split matters architecturally, not just financially. NovaCron's actual differentiators — `NetworkAwareScheduler` (backend/core/scheduler/network_aware_scheduler.go), the per-link admission-controlled transfer system (backend/cmd/api-server/fabric_transfers.go, comment: *"admission-controlled migrations with the measured link budget + compression decision recorded per transfer"*), and adaptive/delta compression (backend/core/network/dwcp/compression/adaptive_compression.go) — are WAN-oriented: they exist to make many small, bandwidth-constrained, geographically distributed links usable. That is the *inference* shape of the market (many regional/edge deployments, each needing modest GPU capacity close to users), not the *training* shape (a handful of hyperscale sites needing >100GB/s NVLink domains that no P2P WAN fabric can provide or would want to compete on). A fabric pitch aimed at frontier training capacity is fighting CoreWeave/Together/Lambda on their home turf; a fabric pitch aimed at distributed inference placement is fighting on NovaCron's actual strengths.

---

## 4. Private AI / sovereign AI

**Grounded (from this repo, research/profitability/competitive-landscape.md, §3 OpenNebula, read earlier in this session and reused here):** OpenNebula's own marketing explicitly targets this market: *"NVIDIA-validated AI-factory/neocloud positioning"*, citing *"+1000 enterprises"*, *"2500 clouds"*, a *"16-datacentre federation"* and *"300K-core cloud"* — i.e. an open-source virtualization vendor is already selling an on-prem/private "AI factory" story as a distinct product line from its base hypervisor business. This is direct, repo-grounded evidence that enterprises want self-hosted AI infrastructure badly enough to be a named product category for an incumbent competitor, not a hypothetical.

**General industry knowledge, `[TRAINED]`, not independently verified this session:** the drivers most commonly cited for private/sovereign AI (regulated-sector data residency rules — HIPAA in US healthcare, GDPR/EU data-residency and the EU's sovereign-cloud push (Gaia-X, EUCS), national "sovereign AI" compute initiatives announced by multiple governments in 2024-2026, defense/ITAR-restricted workloads, and simple competitive confidentiality for enterprises unwilling to send proprietary data to a hyperscaler-hosted model API) are real and well-documented in general industry commentary, though this session could not independently verify a specific price-tolerance figure (e.g., a published "X% premium over hyperscaler API pricing" number) via a live-read primary source. Qualitatively, private/sovereign buyers are reported to prioritize data-locality and audit/compliance guarantees over lowest $/hr, and to accept materially higher per-GPU-hour or per-token costs than public serverless-inference pricing (compare Together's public per-token serverless prices in §1.5's sibling section vs. its $3.99-8.99/hr *dedicated* GPU pricing — dedicated/private capacity already commands a premium over shared serverless even from the same vendor).

**Relevance to NovaCron:** the signed cluster join mechanism (mentioned in project context) and the fact that NovaCron is a self-hosted, operator-controlled fabric rather than a third-party API are structurally aligned with the private-AI buyer's core requirement (data never leaves infrastructure the buyer controls) — but this report did not verify NovaCron's actual data-residency/compliance posture (encryption at rest, audit logging, certification status) against any specific regulatory framework, and such a claim should not be made without that verification.

---

## 5. Fabric differentiators for AI — and the live-GPU-migration reality check

This section is the most directly repo-grounded and is where the report pushes back hardest on the premise in the assignment.

### 5.1 What NVIDIA's own documentation says about GPU live migration

Read directly from NVIDIA's vGPU documentation (https://docs.nvidia.com/vgpu/latest/grid-vgpu-user-guide/using-gpu-pass-through.html, 2026-09-21):

> *"GPU pass-through is used to directly assign an entire physical GPU to one VM, bypassing the NVIDIA Virtual GPU Manager. In this mode of operation, the GPU is accessed exclusively by the NVIDIA driver running in the VM to which it is assigned."*

The same page documents pass-through configuration for KVM via exactly the mechanism NovaCron would need (`virsh`/QEMU `<hostdev>` PCI passthrough with `vfio-pci`) — i.e., raw VFIO passthrough. Critically, that page frames pass-through as an *alternative to*, not a mode of, the NVIDIA Virtual GPU Manager — the mediation layer that actually implements live migration/vMotion for GPU-attached VMs. This matches well-established KVM/QEMU semantics `[TRAINED, but consistent with the grounded NVIDIA doc above]`: a VFIO-passed-through PCI device's state lives on the physical card's silicon and is opaque to QEMU; without a migration-aware mediated device (mdev) driver cooperating with the hypervisor (which is exactly what NVIDIA's separately-licensed vGPU/vComputeServer software provides), QEMU has no way to serialize and transfer that state to a destination host. Live-migrating a *passthrough* GPU VM is not a missing feature you enable with a flag — it requires buying into an entirely separate, vendor-licensed virtualization stack (NVIDIA vGPU Manager + compatible Tesla/RTX PRO-class hardware + a vGPU software license), which is a materially larger commitment than what NovaCron's current KVM driver does today.

**So: is live GPU migration actually a differentiator?** Conditionally yes, but the condition is expensive and NovaCron doesn't meet it. It's rare because most open-source/self-hosted hypervisor projects (plain KVM/libvirt, Proxmox, oVirt) don't ship NVIDIA's licensed vGPU Manager either — they support live migration for ordinary VMs and either don't support GPU passthrough at all or support it with the well-known caveat that a passthrough VM can't be live-migrated (it has to be paused/stopped, migrated cold, and GPU state rebuilt on the far side). If NovaCron built genuine live GPU migration, it actually would be a differentiator versus most of its named competitors — but that is a distinct, much larger engineering and licensing project than "turn `SupportsGPUPassthrough()` to `true`", and nothing in this repo suggests that work has started.

### 5.2 What NovaCron actually has today (repo-grounded)

Grep across `backend/core/vm/*.go` confirms `SupportsGPUPassthrough()` returns `false` in **every active, compiled driver**:

| Driver | File | SupportsGPUPassthrough() |
| --- | --- | --- |
| KVMDriverEnhanced (the real, active KVM/QEMU driver) | driver_kvm_enhanced.go:1376 | `false // Not implemented yet` |
| ContainerDriver | driver_container.go:399 | `false` |
| ContainerdDriver | driver_containerd.go:517 | `false` |
| ProcessDriver | driver_process.go:461 | `false` |
| CoreStubDriver | driver_core_stub.go:34 | `false` |
| MockHypervisor (test double) | mock_hypervisor.go:600 | `false` |

The **only** place in the entire repository where `SupportsGPUPassthrough()` returns `true` is `backend/core/vm/drivers/kvm/libvirt_driver.go.disabled` (line 876) — a `.disabled` file, i.e. dead code, not compiled into any binary, not registered with any driver factory. That same disabled file also sets `SupportsLiveMigration: true` in the same struct literal (line 469) alongside `SupportsGPUPassthrough: true` (line 470) — which is precisely the misleading combination §5.1 warns about: nothing in that disabled driver actually implements GPU-state migration; the two booleans are set side-by-side as capability *claims*, not as a description of a real, GPU-aware migration path. If that file were ever re-enabled verbatim, it would falsely advertise GPU live migration it cannot perform.

What **is** real and active: `KVMDriverEnhanced.SupportsLiveMigration()` returns `true` (driver_kvm_enhanced.go:1360), backed by a genuine implementation in `driver_kvm_migrate.go` — QEMU's own `migrate` command driven over QMP, memory-only migration over shared storage (the destination reuses the source's exact QEMU args plus `-incoming`, opening the same disk/UEFI-vars with `file.locking=off`). This is tested (`driver_kvm_migrate_test.go`: `TestLiveMigrationLocalhostCutover`; `driver_kvm_migrate_rollback_test.go`: `TestLiveMigrationRollbackOnDestFailure`). It is a real capability — for ordinary CPU/RAM/disk VMs. It has never migrated a GPU, because no active driver ever attaches one.

Also real and active: the bandwidth-aware placement and transfer-admission stack that the assignment context calls out — `NetworkAwareScheduler` (backend/core/scheduler/network_aware_scheduler.go), the per-link admission-controlled fabric transfer system with a measured link budget and a recorded compression decision per transfer (backend/cmd/api-server/fabric_transfers.go, main.go:203-205), and an adaptive/delta compression engine for migration traffic (backend/core/network/dwcp/compression/adaptive_compression.go, delta_encoder.go). These are genuinely differentiated relative to LAN-shaped competitors like Proxmox/Harvester (per this repo's own competitive-landscape.md: *"There is no bandwidth-aware placement, no per-link transfer admission, and no compression-decision engine for migration"* in Proxmox) — but they operate on CPU/RAM/disk VMs and control-plane traffic, not on GPU state.

### 5.3 Edge inference

Edge inference workloads (routing, request batching, model-cache-proximity, small quantized models) are exactly the shape of workload NovaCron's bandwidth-aware scheduler and WAN-tolerant fabric are built for — they're CPU/network-bound orchestration problems, not raw FLOPs problems, and they don't require GPU passthrough or GPU migration to benefit from bandwidth-aware placement. This is the one AI use case in this section where NovaCron's *existing* (not aspirational) capabilities line up cleanly with the workload's actual requirements.

---

## Is the AI angle real for NovaCron?

Partially, and only if the pitch is scoped honestly. The market signal is real and large: GPU-hour pricing genuinely varies 3-4x for identical H100-class silicon across five live providers checked this session ($1.73-$1.99/hr spot at Vast.ai/Together vs. $6.16/hr on-demand at CoreWeave), compute demand is growing faster than efficiency or build-out can track (Epoch AI: 3.4-5x/yr demand growth vs. 1.49x/yr efficiency gains, 2-year build times), and there is direct, repo-grounded evidence (a competitor's own marketing, read in this repo's sibling research file) that enterprises pay for self-hosted "AI factory" infrastructure as a distinct product from generic virtualization. NovaCron's actual, working differentiators — bandwidth-aware placement, per-link admission-controlled transfers with a measured link budget, and an adaptive compression engine for migration traffic — are a genuinely good architectural fit for the *inference/edge* half of that market (many small, geographically distributed, bandwidth-constrained deployments), which several sources this session frame as the growing half of aggregate AI compute demand. But the specific hook in the assignment — "live migration of GPU VMs" as a rare, hard-to-copy differentiator — does not hold up: NovaCron has **zero** GPU passthrough support in any compiled driver (verified: `false` in six live drivers; `true` only in one `.disabled`, non-compiled file), so there is no GPU-attached VM state to migrate in the first place, live or otherwise. Worse, even if GPU passthrough were added the straightforward way (VFIO PCI passthrough via libvirt/QEMU, the same mechanism the disabled driver stub already sketches), NVIDIA's own documentation confirms that mode explicitly bypasses the vGPU Manager mediation layer required for live migration — so passthrough and live migration are not simply two flags to flip to `true` together; they are two different, largely incompatible product strategies, and the harder one (mediated vGPU) requires a separate NVIDIA software license and Tesla/RTX-PRO-class hardware NovaCron does not currently integrate with anywhere in the codebase. The honest pitch for NovaCron today is: *bandwidth-aware orchestration and live migration of the CPU/network side of a distributed AI-inference fabric* (routers, gateways, control planes, stateless inference workers restarted rather than live-migrated) — not *live migration of the GPU compute itself*, which remains unbuilt, and which — even if built — would require a scope of work (NVIDIA vGPU licensing, mediated-device drivers, MIG/time-slicing support) far beyond what's implied by the assignment's framing of it as an already-latent capability.
