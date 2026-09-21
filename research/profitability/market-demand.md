# Market Size & Demand Analysis — Distributed Compute Fabrics

**Research date:** 2026-09-21
**Scope:** addressable market and demand evidence for NovaCron's capabilities — KVM VM management with a P2P fabric, bandwidth-aware workload placement and migration, live VM migration with compression decisions, and job dispatch across heterogeneous nodes.

**Evidence policy (same convention as the sibling files in this directory):**
- A figure is **unlabelled** when it was read from the cited URL *in this session*, or is a direct restatement of a sibling file in `research/profitability/` that was itself read this session.
- `[ESTIMATE]` — arithmetic performed on figures that *are* grounded; the basis is always stated.
- `[INFERENCE]` — a conclusion drawn from grounded facts, not itself a sourced number.
- `[TRAINED]` — general industry knowledge, **not** independently verified this session. Treat as directional only.
- `UNREACHABLE` — a specific source attempted and not obtainable this session; stated explicitly rather than substituted.

**Tooling note:** the `web_search` backend (Ollama) returned **HTTP 429 on every call attempted this session** ("monthly usage limit reached"), so **no search-engine results are used**. Every number below came from reading a specific vendor page, public API, live JSON feed, or analyst report page directly. Sources that were attempted and failed are named in §7 rather than filled in with remembered numbers.

**Sibling research reused (not re-researched), read this session:**
- `research/profitability/competitive-landscape.md` — 14 competitors + pricing
- `research/profitability/monetization-models.md` — open-core / SaaS / marketplace take-rate / metering benchmarks
- `research/profitability/ai-gpu-angle.md` — GPU pricing + honest gap analysis
- `research/profitability/technical-audit.md` — what is real vs. stub in the codebase

---

## Executive summary

1. **The "distributed compute marketplace" category is real but commercially tiny — and we can now measure it.** Akash Network, the longest-running and most-cited decentralized compute marketplace, has transacted **$6,240,098 in total USD-denominated compute spend across its entire lifetime** and is currently running at **≈$9,134/day (≈$3.33M/year annualised)**, with **58 active providers** and **289 active GPUs** (live API, 2026-09-21). Vast.ai — the largest and most liquid GPU spot marketplace — has **3,824 GPUs currently available**, whose total listed value at current median prices is **$2,596/hr**, i.e. **≈$22.7M/year if every available GPU were rented 24/7 at median price** `[ESTIMATE]` — a ceiling that no marketplace achieves. The entire observable decentralized-compute sector trades in **single-digit to low-double-digit millions of dollars per year**, not billions. Any business plan that models this as a large near-term revenue pool is not supported by the evidence.

2. **The money is in on-prem/private/sovereign infrastructure, not in the decentralized marketplace.** Published, readable market sizes: private cloud **$124.8B (2024) → $190.9B (2029), CAGR 8.9%**; sovereign IaaS **$52.4B (2025) → $275.9B (2032), CAGR 27%**; edge computing **$111.34B (2026) → $317.39B (2031), CAGR 23.3%**; distributed cloud **20.6% CAGR**; software-defined data center **$100.0B (2026) → $265.8B (2031), CAGR 21.6%**. These are the budgets a self-hosted KVM/P2P fabric can actually address.

3. **Repatriation is real but selective, and now quantified.** Flexera's 2026 State of the Cloud (N=753) reports **23% of cloud-based workloads have been repatriated, up 2 percentage points YoY**, while **77% remain in cloud** and public-cloud spend continues to rise. IDC reports only **8–9% of companies plan full repatriation**, while **close to half of cloud buyers overspent in 2023** and 59% expected to overspend in 2024. 37signals/Basecamp states a **~$10M five-year saving, a 50–66% infrastructure cost reduction, achieved without adding staff**. The trend is *selective workload placement*, not a wholesale reversal — which is precisely the shape of demand a placement-and-migration fabric serves.

4. **Willingness-to-pay is highest where sovereignty or multi-site bandwidth is the constraint, and it is explicitly priced.** The sovereign-IaaS report states a **"sovereignty premium (15–40% over standard public cloud)"** — buyers demonstrably pay more for jurisdictional control. That is the clearest, most directly quantified WTP signal found this session, and it maps onto NovaCron's self-hosted, operator-controlled architecture better than onto any cloud-delivered alternative.

5. **The strongest-WTP segments for NovaCron are (A) sovereign/regulated private-cloud operators and (B) bandwidth-constrained multi-site operators.** The decentralized GPU marketplace is explicitly **not** recommended as a primary segment: the transacted revenue is small, the measurable growth is happening in the numerically-smallest part of the market, and `technical-audit.md` shows the canonical NovaCron binary has **no GPU passthrough in any compiling driver** (`SupportsGPUPassthrough()` returns `false` in every active driver; the only `true` lives in a `.go.disabled` file).

---

## 1. Distributed / decentralized compute marketplaces

### 1.1 What these networks actually transact (live measurements)

This section uses **measured network state** rather than vendor marketing, because the marketing numbers and the transacted numbers differ by orders of magnitude.

#### Akash Network — live network state

Read from Akash's public console API, `https://console-api.akash.network/v1/dashboard-data` (snapshot `2026-09-21T21:00:56Z`):

| Metric | Value (2026-09-21) | Prior day (2026-09-20) |
| --- | --- | --- |
| Active leases | **676** | 745 |
| New leases/day | 533 | 440 |
| Lifetime leases | 599,863 | 599,330 |
| Active providers | **58** | 57 |
| Active GPU / total GPU / available GPU | **289 / 458 / 163** | 192 / — / — |
| Active CPU (cores) / total | ≈4,733 / 15,749 | ≈4,420 / — |
| Active memory | 34.3 TB | 28.7 TB |
| Total storage provisioned | 865.8 TB | — |
| **Daily USD spent (network-wide)** | **$9,134.31** | $9,078.95 |
| **Lifetime USD spent (network-wide)** | **$6,240,098.47** | $6,230,964.16 |

`[ESTIMATE]` Derived from the above, basis = the API's own daily and lifetime micro-USD counters:
- Annualised current run-rate: **$9,134.31 × 365 = $3.33M/year**
- Monthly current run-rate: **≈$278K/month**
- Revenue per active lease: **$13.51/day**
- Implied historical average daily spend since public launch (27 Apr 2020, 2,338 days): **$2,669/day** — i.e. **the current run-rate is ~3.4× Akash's lifetime average**, so the network is growing, but from a very small base.

Akash GPU prices are also live and public (`https://console-api.akash.network/v1/gpu-prices`):

| GPU | Total / available | Min $/GPU-hr | Median $/GPU-hr | Avg $/GPU-hr |
| --- | --- | --- | --- | --- |
| A100 80GB SXM4 | 230 / 68 | $1.07 | $1.84 | $1.70 |
| H100 80GB SXM5 | 68 / 19 | $2.04 | $2.55 | $2.58 |
| H200 141GB SXM5 | 40 / 15 | $4.45 | $4.45 | $4.45 |
| RTX 3090 24GB | — | $0.0966 | $0.1681 | — |
| B300 288GB SXM6 | 24 / **0** | *no price* | *no price* | *no price* |

The B300 row is a liquidity tell: 24 cards exist on the network and **zero are available**, and the network publishes no price for them because nothing is bid.

#### Vast.ai — live marketplace state

Read from Vast.ai's own public pricing feed, `https://storage.googleapis.com/vast-public-gpu-pricing/gpu-pricing-public.json` (feed timestamp `2026-09-21T20:30:17Z`; the feed's licence text requires attribution and forbids bulk collection or use in a derivative product without a data licence from `data@vast.ai`).

`[ESTIMATE]` Basis: sum over every model in the feed of `available × current.median`:
- GPUs currently available across the marketplace: **3,824** (72 of 79 listed models carry a live price)
- Aggregate listed value: **$2,596/hr**
- If 100% of available capacity were rented continuously at median price: **≈$1.90M/month, ≈$22.7M/year** — an *upper bound*, since no marketplace runs at 100% utilisation.

Selected live prices (min / median, $/GPU-hr):

| GPU | Available | Min | Median | Min↔median spread |
| --- | --- | --- | --- | --- |
| RTX 5090 32GB | 943 | $0.2674 | $0.6270 | 2.3× |
| RTX 4090 24GB | 493 | $0.1352 | $0.5083 | 3.8× |
| RTX 3090 24GB | 247 | $0.0966 | $0.1681 | 1.7× |
| A100 PCIe 80GB | 32 | $0.4676 | $0.9337 | 2.0× |
| H100 SXM 80GB | 36 | $1.7338 | $2.1605 | 1.2× |
| H100 PCIe 80GB | 21 | $2.0014 | $2.4094 | 1.2× |
| H200 141GB | 26 | $2.6325 | $4.6471 | 1.8× |
| B200 192GB | 13 | $6.2513 | $8.1266 | 1.3× |
| B300 288GB | 52 | $8.6878 | $9.3755 | 1.1× |

The **min↔median spread is the marketplace's real economic signature**: for identical silicon on the same day, the cheapest live offer can be 2–4× below the median. `[INFERENCE]` That dispersion is the inefficiency an aggregation/placement layer exists to exploit — and it is also why any marketplace that quotes a single price is selling a *service*, not a *market*.

#### Nosana — the only decentralized network in this set with a **published, quantified take rate**

Read from the live markets API, `https://api.nosana.com/api/markets/`:

- **Every one of the 46 priced markets carries `network_fee_percentage: 10`.** This is the only explicit, machine-readable percentage take rate found on any decentralized compute network this session.
- Markets are tiered `PREMIUM` (hosts "that have staked and passed benchmarking"), `COMMUNITY` (same GPUs, lower stake requirement, lower price) and `OTHER` (multi-GPU/specialised).
- Billing is **per second, per replica, per market** — the whole GPU is held for the deployment's duration.
- Representatives: `usd_reward_per_hour` **$1.00** for NVIDIA Pro 6000 Server Edition, **$0.3636** for a 4090, **$0.1745** for a 3090, **$0.0436** for a 3060.
- Nosana's own pricing doc snapshots (2026-09-03) a **premium/community price pair for the same silicon**: H100 **$1.3636** premium vs **$1.0227** community; 4090 **$0.2909** vs **$0.2182**; 3060 **$0.0436** vs **$0.0327**. `https://nosana.com/pricing.md`

`[INFERENCE]` The explicit premium↔community price gap is the network's own price for *verified reliability* — roughly **20–30%** on the sampled models. That is a directly usable anchor for a "verified tier" argument.

#### Render Network — burn-and-mint, no published percentage

Read via the Render Network documentation (`https://know.rendernetwork.com/`) and a targeted query against its own docs-Q&A endpoint:

- Settlement model is **Burn-Mint Equilibrium**: creators convert fiat to RENDER, and **the RENDER is burned on job completion**; the burn log is the basis for contributor rewards each epoch.
- The documentation **does not state a network take rate or fee percentage** — the docs query returned: *"I can't find any docs page that states a network 'take rate' or fee percentage charged on each job."* Pricing is expressed via OctaneBench tiers, not a % fee.
- Published emission budgets: **Year 1 = 9,126,804 RENDER; Year 2 = 5,905,580 RENDER**, allocated per epoch (*"typically spanning a week period based on network usage"*).
- The same query confirmed the docs publish **no current totals** for GPU nodes, frames/jobs rendered, or RENDER burned.

#### io.net and Golem — attempted, not measurable this session

- **io.net**: `https://io.net/` serves a JS marketing shell; its only quantified public claim is the meta-description **"30,000+ GPUs … 130+ countries … up to 70% lower cost than AWS"** (a vendor marketing claim, not an audited figure). Its pricing page was already found **404/broken** in the sibling `competitive-landscape.md`. **No revenue, take-rate, or utilisation figure could be obtained.** `UNREACHABLE`.
- **Golem Network**: `https://stats.golem.network/` renders as a client-side dashboard; the two API endpoints tried (`https://api.stats.golem.network/v2/network/online`, `https://stats.golem.network/api/v1/network/online`) returned DNS failure and HTTP 404 respectively. **No numbers quoted.** `UNREACHABLE`.

### 1.2 Market size — published third-party figures

**Important framing:** no third-party analyst report readable this session publishes a market size for "decentralized compute marketplaces" as a standalone category. A Grand View Research page for exactly that was attempted and returned **HTTP 403** `UNREACHABLE`. The honest substitute is (a) the nearest *published* adjacent markets and (b) the *measured* transacted value above.

| Market (as defined by the publisher) | Size | Forecast | CAGR | Source |
| --- | --- | --- | --- | --- |
| **GPU as a Service** (IaaS/PaaS, high/mid/low-end GPUs, public/private/hybrid) | $8.21B (2025) | **$26.62B (2030)** | **26.5%** | `marketsandmarkets.com/Market-Reports/gpu-as-a-service-market-153834402.html` (Mar 2025) |
| **Cloud Services Brokerage** (aggregation, intermediation, arbitrage incl. marketplace/catalog) | $15.36B (2026) | **$36.52B (2031)** | **18.9%** | `marketsandmarkets.com/Market-Reports/cloud-brokerage-market-771.html` (Oct 2026) |
| AI Infrastructure (compute/memory/network/storage/software; on-prem + cloud + hybrid) | $135.81B (2024) | **$394.46B (2030)** | 19.4% | `marketsandmarkets.com/Market-Reports/ai-infrastructure-market-38254348.html` (Nov 2024) |
| Global cloud industry (all cloud) | $1,091.4B (2024) → $1,256.8B (2025) | — | **+15.1% YoY** (2025) | `marketsandmarkets.com/Market-Reports/cloud-industry-outlook-233205216.html` (Jan 2025) |

`[ESTIMATE]` **Measured floor for the decentralized-compute slice**, basis = the live counters in §1.1: Akash's annualised run-rate ($3.33M) is a *measured floor* for one network. The two measurable GPU networks together — Akash's actual spend plus Vast.ai's 100%-utilisation ceiling — bound the observable sector at roughly **$3M–$26M/year gross compute spend**. Networks not measurable this session (Render, io.net, Golem, Salad, and the long tail) sit outside that bound. `[INFERENCE]` Even if the unmeasured networks collectively transacted several times the measurable pair, the sector remains **orders of magnitude below the $8–27B GPUaaS market** it is frequently described as disrupting — because GPUaaS includes hyperscaler and neocloud *owned* capacity, whereas the decentralized networks are re-selling third-party idle capacity.

### 1.3 Business-model viability

| Network | Model | Take rate | Evidence quality |
| --- | --- | --- | --- |
| **Nosana** | Marketplace; per-second, per-replica, per-market | **10% flat on all 46 markets** | **Grounded, machine-readable** (live API field `network_fee_percentage`) |
| **Akash** | Marketplace; provider-set prices, reverse auction | **0% explicit** — value extracted at protocol/token level post-BME (recorded in `monetization-models.md` §3b, read from `akash.network/blog/what-burn-mint-equilibrium-means-for-akash`) | Grounded via sibling file |
| **Render** | Marketplace; burn-and-mint | **Not published**; emissions-funded (Y1 9.13M RENDER, Y2 5.91M RENDER) | Grounded (absence is the finding) |
| **Vast.ai** | Live spot marketplace | Not published; commission on host earnings `[TRAINED]` | Feed terms are the only published commercial text read this session |
| **AWS Marketplace** (reference) | Software marketplace | **≈20% referral fee** `[TRAINED]` | Sibling `monetization-models.md` §3a notes the fee pages did not render; treated as industry-cited, not verified |
| **Snowflake / Databricks Marketplace** | Data/app marketplace | Undisclosed, negotiated | Sibling `monetization-models.md` §3c |

`[INFERENCE]` Three viable extraction designs are observable in the wild: an **explicit flat percentage** (Nosana, 10%), a **structure-free protocol mechanism** (Akash burn/mint, 0% line item), and a **hidden/negotiated share** (Render, Vast, Snowflake). Only Nosana's is verifiable from a public endpoint. For a fabric operator, Nosana's 10% is the cleanest published comparable; it is *lower* than the ~20% AWS software-marketplace referral fee, consistent with the general pattern that infrastructure marketplaces price their take below software marketplaces.

### 1.4 Key constraints — evidence, not assertion

**Liquidity.**
- Akash: **58 active providers**; **163 of 458 GPUs (36%) available**, i.e. unsold; 0 of 24 B300s available and no price published. A network whose marquee hardware is entirely unlisted and whose newest hardware has no bids is liquidity-constrained at the high end.
- Vast.ai: deep at the consumer/prosumer end (943 RTX 5090s, 493 RTX 4090s, 247 RTX 3090s available) but thin at the datacentre end (**32 A100 PCIe, 36 H100 SXM, 26 H200, 13 B200, 52 B300**). `[INFERENCE]` The marketplace's liquidity is inversely correlated with the value of the hardware — the most expensive GPUs have the fewest offers and the widest min↔median spreads.
- `[INFERENCE]` Both facts point to the same conclusion: a fabric competing on *aggregation of scarce, expensive, verified capacity* is competing in the least liquid part of the market, where reliable placement is worth the most.

**Trust.**
- Nosana makes trust a **priced, tiered product**: `PREMIUM` hosts "have staked and passed benchmarking"; the same GPU costs ~20–30% more in the premium tier. This is the only decentralized network in this set that has operationalised (and published a price for) verified supply.
- Vast.ai's feed licence explicitly forbids *"use in any index, benchmark, or derivative product"* without a paid data licence — i.e. the marketplace itself treats its own price data as a licensed asset, a sign that third-party verification/benchmarking is commercially sensitive.
- `[INFERENCE]` Trust is therefore monetisable — and a signed-cluster-join model (HMAC-signed admission with mandatory reachability callback, per `technical-audit.md` §2) is a direct answer to a constraint every one of these networks is currently pricing around.

**Performance.**
- Nosana's per-second, per-replica, whole-GPU-hold billing (`nosana.com/pricing.md`) means a buyer cannot sub-divide a GPU across jobs — a performance/isolation trade-off imposed by the platform, not chosen by the workload.
- Akash's network is Kubernetes-manifest shaped (SDL); Vast.ai's is container-shaped. **None of the five networks in this section offers a full virtual-machine abstraction with cross-node live migration.** `[INFERENCE]` That is the structural gap: they sell *capacity*, not *placement*.

---

## 2. Edge + distributed cloud infrastructure

### 2.1 Market size and growth

| Market | Value | Forecast | CAGR | Source |
| --- | --- | --- | --- | --- |
| **Edge computing** (hardware, software, services) | $87.81B (2025) → **$111.34B (2026)** | **$317.39B (2031)** | **23.3%** | `marketsandmarkets.com/Market-Reports/edge-computing-market-133384090.html` (Jul 2026, 220 tables, 310 pages) |
| — of which **services** | — | — | **26.5%** (fastest segment) | same |
| **Distributed cloud** (data security, storage, networking; edge/CDN/IoT apps) | — | **$11.2B (2027)** | **20.6%** | `marketsandmarkets.com/Market-Reports/distributed-cloud-market-165173185.html` (Aug 2022 — **dated**) |
| **Software-defined data center** (SDDC) | **$100.00B (2026)** | **$265.80B (2031)** | **21.6%** | `marketsandmarkets.com/Market-Reports/software-defined-data-center-sddc-market-1025.html` (Mar 2026) |
| Data center solutions (power, cooling, IT, physical, software) | $535.45B (2026) | $1,336.55B (2031) | 20.1% | `marketsandmarkets.com/Market-Reports/data-center-construction-market-232213604.html` (Jul 2026) |
| Network function virtualisation (NFV) | $39.0B (2025) | $360.0B (2035) | 24.9% | `marketsandmarkets.com/Market-Reports/network-function-virtualization-market-93929190.html` (Jan 2020 — **dated**) |

`[INFERENCE]` The edge-computing figure is the most relevant and the most reliable: it is the newest of the set (Jul 2026) and **23.3% CAGR on a $111B base is the single largest credible growth curve found in this research run**. Note also that MnM's own forecast cites *"increasing data sovereignty requirements"* as a driver of edge demand — the edge and sovereignty themes reinforce each other.

### 2.2 Who buys, and why — as stated by the market report itself

Read verbatim from the edge-computing report page:

| Dimension | Finding |
| --- | --- |
| **Fastest-growing deployment model** | **Regional / Cloud Edge** — *"organizations prioritize distributed, cloud-managed deployment models that combine hyperscale cloud capabilities with localized edge processing. This model reduces latency for time-sensitive workloads, supports **data residency and sovereignty requirements** in regulated industries…"* |
| **Largest deployment model today** | **On-premises edge** — *"driven by enterprise demand for localized processing, data control, low-latency analytics, and operational continuity"*, across manufacturing, healthcare, energy, transportation, government and BFSI |
| **Largest vertical** | Manufacturing (industrial automation, predictive maintenance, AI-enabled production) |
| **Fastest-growing vertical** | Healthcare & life sciences (*"compliance with stringent data privacy rules"*, real-time diagnostics, AI imaging) |
| **Fastest-growing org size** | SMEs — *"adopt localized processing to overcome bandwidth and latency constraints without investing in large-scale data centers"* |
| **Fastest-growing region** | Asia Pacific (smart-city programmes; Digital India, Smart Nation, China New Infrastructure Plan) |
| **Named drivers** | IoT/endpoint scale; low-latency demand; 5G; AI-enabled workloads; *"data sovereignty"* |
| **Named restraints** | *"Economic & policy constraints in emerging markets"*; **"Complex nature of edge computing infrastructure"** |
| **Named challenges** | Data privacy/security; **"Skill gap & operational expertise"** — *"Shortage of skilled professionals capable of managing distributed edge environments remains a major challenge"* |

`[INFERENCE]` Two of those lines are effectively a description of NovaCron's product thesis, written by an analyst firm: **"complex nature of edge computing infrastructure"** as a restraint, and **"skill gap & operational expertise"** in managing distributed edge deployments as a challenge. A fabric that makes many heterogeneous, bandwidth-constrained sites behave like one schedulable pool is an answer to both, and the report explicitly names SMEs — the segment least able to hire the missing distributed-systems expertise — as the fastest-growing buyer cohort.

### 2.3 Key vendors

Named on the edge-computing report's key-player list: AWS, Microsoft, Google, IBM, Dell, HPE, Cisco, NVIDIA, Intel, Huawei, Lenovo, Supermicro, Advantech, Oracle, Akamai, Cloudflare, Fastly, Siemens, Belden, Ericsson, Nokia, Moxa, Vapor IO. **Start-ups/SMEs explicitly listed:** ZEDEDA, Spectro Cloud, Avassa, Axelera AI, EdgeCortix, Vapor IO.

`[INFERENCE]` The SME cohort (ZEDEDA, Spectro Cloud, Avassa) is the realistic competitive set for a fabric *management plane*; the incumbent cohort (AWS/Azure/Google/HPE/Dell) is the set whose budgets and installed base a fabric displaces or plugs into. The report's own evaluation matrix names **AWS as "Star"** (Outposts, Wavelength, Local Zones) and **Huawei as "Emerging Leader"**.

---

## 3. Private / on-prem / sovereign cloud

### 3.1 Market size and growth

| Market | Value | Forecast | CAGR | Source |
| --- | --- | --- | --- | --- |
| **Private cloud** (virtual, on-premises, hosted, managed; IaaS/PaaS/SaaS) | $112.0B (2023), **$124.8B (2024)** | **$190.9B (2029)** | **8.9%** | `marketsandmarkets.com/Market-Reports/private-cloud-market-101816685.html` (Jul 2024) |
| **Sovereign IaaS** | **$52,358M (2025)** | **$275,943M (2032)** | **27.0%** | `marketsandmarkets.com/Market-Reports/sovereign-infrastructure-as-a-service-iaas-market-169448214.html` (Oct 2026) |
| — Europe sovereign IaaS | $13,090M (2025) | ~$73,000M (2032) | **28.0%** | same |
| — North America sovereign IaaS | $18,325M (2025) | ~$90,000M (2032) | 26.0% | same |
| — Asia Pacific sovereign IaaS | $15,707M (2025) | ~$90,943M (2032) | **29.0%** (fastest) | same |
| — Rest of World sovereign IaaS | $5,236M (2025) | ~$22,000M (2032) | 23.0% | same |
| **European sovereign cloud IaaS spend, actual trajectory** | $6.9B (2025) | **$12.6B (2026) → $23.1B (2027)** | **+83% YoY** | same |

`[INFERENCE]` Sovereign IaaS is the fastest-growing *directly relevant* segment found this session (27% CAGR to 2032, with Europe at +83% YoY) and it is **the one segment where the growth is fastest precisely because buyers are rejecting the incumbents' architecture** — the report is explicit that EU SEAL-3 *"by design, excludes infrastructure controlled by US hyperscalers, even when physically located in Europe."* Self-hosted, operator-controlled infrastructure is the only architecture that clears that bar.

### 3.2 Demand drivers

Verbatim, from the sovereign-IaaS report:

- **Regulation, at scale:** *"Over 140 countries now have data sovereignty or data localization laws on the books."*
- **A certification framework that creates procurement demand:** the EU's Cloud Sovereignty Framework establishes three tiers — **SEAL-1** (basic security), **SEAL-2** (operational controls + EU data residency), **SEAL-3** (full-stack: EU-controlled operations, EU-national personnel, EU-based key management, protection from foreign government access orders).
- **Legal extraterritoriality:** *"The US CLOUD Act gives American authorities the power to demand data held by US companies regardless of where the data is physically stored"*; combined with **Schrems II** invalidating the EU–US Privacy Shield.
- **Sovereign AI as the accelerant:** *"Running AI models — training, fine-tuning, and inference — on jurisdiction-controlled compute is a requirement that governments and regulated enterprises are now specifying explicitly… This requirement is pulling GPU-class compute into the sovereign IaaS market for the first time, significantly expanding the per-workload value."*
- **Real, funded procurement:** April 2026, the European Commission awarded the **Cloud III framework — EUR 180M over six years** — to four European-controlled provider consortia.
- **Private cloud drivers** (MnM private-cloud report): *"increasing demand for data security"*; GDPR/HIPAA/CCPA; *"Customization and control over IT infrastructure"*; *"Enhanced performance and reliability"*. Named restraint: *"High initial cost"*. Named challenge: *"Vendor lock-in and interoperability challenges"*.

### 3.3 Who buys self-hosted / on-prem KVM orchestrators

| Buyer cohort | Evidence | Why they buy self-hosted |
| --- | --- | --- |
| **Government & defence** | **"roughly a third of sovereign IaaS demand"**; the strictest requirements; *"much of the time no public cloud — even with sovereign controls — can fully satisfy"* them (private sovereign IaaS) | Jurisdiction control, cleared personnel, no foreign access orders |
| **Financial services** (fastest-growing sovereign end user) | *"driven by DORA, national banking regulations, and central bank mandates that require jurisdiction-controlled infrastructure for systemically important financial data"*; BFSI is also the **largest vertical in private cloud** | Regulatory mandate + systemically-important data |
| **Healthcare & life sciences** | Fastest-growing vertical in edge computing; private-cloud driver list names HIPAA | Data privacy, real-time diagnostics at the point of care |
| **Large enterprises / BFSI** | Private cloud report: BFSI largest vertical; **SMEs fastest-growing org size** | Control, customisation, compliance, avoiding lock-in |
| **Regional service providers / MSPs** | MnM distributed-cloud SME cohort (Platform9, ZEDEDA, Wind River, Vapor IO, PhoenixNAP, SCC) | Cannot buy hyperscaler economics at their scale |

### 3.4 Willingness to pay — the single most valuable number found

> **"The sovereignty premium: Sovereign IaaS costs 15–40% more than equivalent standard public cloud capacity, because the operational constraints (local personnel, dedicated hardware, compliance certification, restricted support channels) add cost that standard cloud regions do not bear. Buyers must justify this premium through compliance necessity, risk avoidance, or regulatory mandate — not raw cost efficiency."**
> — `marketsandmarkets.com/Market-Reports/sovereign-infrastructure-as-a-service-iaas-market-169448214.html`

`[INFERENCE]` This is the strongest WTP datum in the entire research run, because it is (a) quantified, (b) stated as a *cost premium buyers already accept*, and (c) attributed to exactly the operational constraints that a self-hosted fabric imposes and relieves. It means a fabric sold into a sovereign/regulated context does **not** need to win a price war against hyperscalers — it needs to be *credible on control*, and can price 15–40% above commodity cloud while remaining the cheaper option against the incumbent sovereign alternatives.

---

## 4. Cloud repatriation — evidence of workloads moving off hyperscalers

### 4.1 The numbers

| Source | Finding | Evidence class |
| --- | --- | --- |
| **Flexera 2026 State of the Cloud** — Figure 11, N=753, fielded winter 2025 | **23% of cloud-based workloads have been repatriated; 77% remain in cloud.** Up **2 percentage points** YoY. Read directly from the published chart image at `resources.flexera.com/web/eloqua/images/charts/sotc/2026/v2/full/figure-11-full.png` | **Grounded (chart read as image)** |
| Flexera 2026, narrative | *"The percentage of cloud-based workloads and cloud-based data that organizations have repatriated each increased by 2 percentage points year over year, which may indicate that organizations are trying to leverage their hybrid cloud environment by placing workloads where it makes the most sense to run them."* | Grounded |
| Flexera 2026, hybrid architecture | **73% of organizations operate hybrid estates** (+3pp YoY); multi-cloud +2pp; **only 14% run multi-cloud without a private cloud** | Grounded |
| Flexera 2026, cost pressure | **Estimated wasted cloud spend rose to 29%**, *"reversing a five-year downward trend"*; **fewer than half** of organizations use any commitment discount per provider | Grounded |
| **IDC Cloud Pulse 4Q 2023** (via IDC blog, Oct 2024) | *"close to half of cloud buyers spent more on cloud than they expected in 2023, with 59% anticipating similar overruns in 2024"* | Grounded |
| **IDC Server & Storage Workloads Survey** | *"only 8–9% of companies plan full workload repatriation"*; larger organisations lead repatriation; the most-repatriated elements are **production data, backup/DR and compute** | Grounded |
| **37signals / Basecamp** (`basecamp.com/cloud-exit`) | *"Leaving the cloud will save us ~$10 million over five years"*; *"a reduction in our infrastructure costs of between half and two-thirds"*; pulled Basecamp, HEY and five other apps off AWS *"without adding any new staff"* | **Grounded (vendor's own claim)** |

### 4.2 The honest counter-evidence

Repatriation is **not** a wholesale reversal, and the same sources say so:

- **77% of cloud workloads are still in cloud** (Flexera Figure 11).
- IDC: only **8–9%** of companies plan *full* repatriation — most repatriate selected elements.
- Public-cloud spending is still rising: Flexera reports the **top spend tiers grew 3% collectively** while the bottom three fell 4%; **76% of large enterprises spend more than $5M/month** on public cloud. Published cloud-market forecasts agree: US cloud **$485.54B (2025) → $721.30B (2030), CAGR 8.2%**; Europe **$325.92B → $550.42B (2030), CAGR 11.0%**; APAC **$348.75B → $752.78B (2030), CAGR 16.6%** (`marketsandmarkets.com` regional cloud reports, Sep 2026).
- Flexera frames the multi-cloud cause more as **"applications isolated in distinct environments"** than deliberate placement, and lists *"strategic workload placement and workload bursting"* as the **least common** motivations for multi-cloud.

`[INFERENCE]` The correct reading is: **the repatriation opportunity is a placement opportunity, not a migration-away opportunity.** Buyers are not abandoning cloud; they are accumulating heterogeneous estates (73% hybrid) and are increasingly willing to move *individual* workloads, with **23% already having done so and the trend rising 2pp/year**. A tool whose value proposition is "decide where each workload should run, and move it there reliably" addresses the growth part of that curve; a tool whose value proposition is "exit the cloud" addresses the 8–9% tail.

---

## 5. Willingness to pay — comparable products and unit prices

Per the assignment, pricing already gathered in `competitive-landscape.md` and `monetization-models.md` is **reused, not re-researched**. All rows below are read from those files, which were read this session.

### 5.1 The five pricing shapes in the market

| Shape | Vendor | Exact figure | Source |
| --- | --- | --- | --- |
| **Per CPU socket / year** (support, zero feature gating) | Proxmox VE | Community **€120**; Basic **€370**; Standard **€550**; Premium **€1,100** | `proxmox.com/en/proxmox-virtual-environment/pricing` (via competitive-landscape.md §1) |
| **Per host / year** (support; 4 tiers by ticket count, coverage, Sev-1 response, max hosts 3→64) | Vates / XCP-ng / Xen Orchestra | Absolute EUR not published (client-side); **published multi-year discounts −10% @3yr, −15% @5yr** | `vates.tech/en/pricing-and-support/` (via competitive-landscape.md §5, monetization-models.md §4b) |
| **Quote-only enterprise** (feature-gated open-core) | OpenNebula (STANDARD 9×5 / PREMIUM 24×7); SUSE Harvester; Red Hat OpenShift Virtualization | No public list prices | competitive-landscape.md §2–4 |
| **Per vCPU-hour / GB-hour / GB-month, metered** | Railway, Fly.io, Render | Railway containers **≈$20/vCPU-mo, ≈$10/GB-RAM-mo**; Railway Sandboxes (VM) **≈$50/vCPU-mo, ≈$50/GB-mo active use**; Fly machines $1.94–$496/mo **plus ≈$5/GB-RAM/30d**; Render $7–$450/mo flat tiers | `railway.com/pricing`, `fly.io/docs/about/pricing/`, `render.com/pricing` (via competitive-landscape.md §6–8) |
| **Per VM-hour** (NovaCron's own published listing) | NovaCron | **$0.15/VM-hr (Standard, ≤100 VMs); $0.12/VM-hr (Professional, ≤1,000 VMs)**; Enterprise "Custom pricing" | `marketplace/listings/aws/product-description.md` |

`[ESTIMATE]` Normalising NovaCron's own price to a monthly rate at 100% duty cycle: **$0.12 × 730 = $87.60/VM-month; $0.15 × 730 = $109.50/VM-month** (basis: 730 h/month). Comparing that to the metered competitors requires an explicit vCPU assumption and is therefore ambiguous rather than favourable or unfavourable:
- **At one vCPU per VM** ($50/vCPU-month, Railway Sandboxes): NovaCron at $88–$110/VM-month is **~1.8–2.2× dearer**.
- **At four vCPU per VM** (a realistic guest shape): Railway would be **$200/VM-month** (4 × $50), so NovaCron at $88–$110 is **~1.8–2.3× cheaper**.

`[INFERENCE]` The conclusion is that **the published per-VM-hour band is defensible only if NovaCron's VMs are described as multi-vCPU guests, and only if the migration/HA/placement capability the PaaS vendors lack is part of the price**. It is not a price advantage on its own. (The sibling `monetization-models.md` §2a phrases this more favourably — *"undercuts Railway's $50/vCPU-month sandbox rate on a like-for-like managed-VM basis"* — which holds under the multi-vCPU reading but not the single-vCPU one; the assumption is stated here so the reader can decide.)

For the enterprise-support shape, scaling Proxmox's published numbers gives a usable band `[ESTIMATE]`: a **20-socket** estate costs **€2,200/yr at Community, €11,000/yr at Standard, €22,000/yr at Premium**.

### 5.2 Adjacent unit prices a fabric would have to live alongside

| Unit | Observed price band | Sources |
| --- | --- | --- |
| Block/volume storage | **$0.15/GB-month** (Fly volumes $0.15; Railway volumes ≈$0.15; Render disk-adjacent $0.25, Postgres storage $0.30) | competitive-landscape.md §6–8 |
| Object storage | Railway $0.015/GB-mo (free egress) | same |
| **Egress** | **$0.02–$0.15/GB**: CloudFront US/EU **$0.085** stepping to **$0.020 above 5PB**; Fly **$0.02** (NA/EU) to **$0.12** (Africa/India); Railway flat **$0.05**; Render **$0.15** overage | monetization-models.md §5a (from `aws.amazon.com/cloudfront/pricing/pay-as-you-go/`, fly.io, railway.com, render.com) |
| Serverless/CPU-hour | Render Workflows **$0.20/active-CPU-hr**, **$0.05/active-GB-hr** | render.com/pricing |

`[INFERENCE]` Egress is the unit NovaCron already measures natively (its per-link admission and bandwidth-aware placement *require* link-cost telemetry), and it is the unit with the widest cross-vendor dispersion (7.5× between the cheapest and dearest published rate). The sibling analysis recommends an entry price of **$0.03–$0.05/GB** with regional multipliers — this is consistent with everything found here and is not revised.

### 5.3 GPU-hour prices (relevant only if NovaCron ever attaches GPUs)

Cross-provider H100-class, cheapest → dearest on-demand (from `ai-gpu-angle.md`, all read live this session): **Vast.ai marketplace min $1.73 → Together preemptible $1.99 → RunPod PCIe $2.89 → Together on-demand $3.99 = Lambda $3.99 → CoreWeave $6.16** — a **3.6× same-day spread for identical silicon**. Akash's live H100 min of **$2.04** and Nosana's premium H100 of **$1.3636** both sit inside that band.

`[ESTIMATE]` Implied addressable GPU-rental spend at Akash's current scale, basis = 289 active GPUs × 730 h × ~$2.5/GPU-hr ≈ **$0.53M/year** — i.e. well under 20% of Akash's $3.33M total run-rate, implying most Akash revenue is CPU/memory/storage, not GPU. `[INFERENCE]` This is worth stating plainly: **the decentralized GPU rental business that the sector's marketing leads with is a small fraction of an already-small market.**

---

## 6. TAM / SAM / SOM for NovaCron

### 6.1 TAM — the budgets that could pay for this

No readable analyst report isolates "KVM VM management with a distributed/P2P fabric". The honest TAM construction is a union of the published adjacent markets, presented with the overlap caveat stated:

| Included budget pool | Published size | Growth |
| --- | --- | --- |
| Private cloud (on-prem/hosted/managed) | $124.8B (2024) → $190.9B (2029) | 8.9% |
| Edge computing | $111.34B (2026) → $317.39B (2031) | 23.3% |
| Sovereign IaaS | $52.36B (2025) → $275.94B (2032) | **27.0%** |
| SDDC software-defined infrastructure | $100.0B (2026) → $265.8B (2031) | 21.6% |
| Distributed cloud (narrow definition) | → $11.2B (2027) | 20.6% |

**Caveat, stated plainly:** these overlap heavily (sovereign IaaS is largely a subset of private cloud; SDDC includes hardware and storage; edge includes devices and sensors). **Summing them would be wrong.** `[INFERENCE]` The defensible statement is that the *union* of budgets containing a line item for "virtualisation management and workload placement software" is **a nine-figure-to-low-ten-figure-billions-per-year global pool growing at 9–27%/yr**, and that the **narrow "distributed cloud" definition ($11.2B by 2027)** is the closest published analogue to what NovaCron is.

### 6.2 SAM — the part NovaCron can actually serve

`[ESTIMATE]` **Basis and assumptions, all explicit:**
- Addressable unit: **an annual support/management/subscription fee per CPU socket or per node**, because that is the shape every on-prem vendor in the competitive set prices in (Proxmox per socket; Vates per host; OpenNebula quote-based).
- Price band: **€120–€1,100/socket/yr** (Proxmox's published range — the only publicly listed price band in the self-hosted KVM manager market).
- Worldwide installed base of self-managed KVM sockets: **~2,000,000 sockets** — `[TRAINED]`, **not verified this session**; no source read in this run published a socket count for self-hosted KVM. This is the single softest assumption in the report and is flagged as such.
- At Proxmox's **mid (Standard, €550)** price: 2,000,000 × €550 ≈ **€1.1B/year** of support-and-management spend in the self-hosted KVM segment `[ESTIMATE]`.
- Apply NovaCron's actual filter — **multi-site / bandwidth-constrained / sovereign estates**, which is the only place its differentiators apply: if **10–20%** of those estates are genuinely distributed: **SAM ≈ $110M–$220M/year** `[ESTIMATE]`.

**Sensitivity, so the assumption is visible rather than buried:**

| Installed base (sockets) | SAM @ €120/socket | SAM @ €550/socket | SAM @ €1,100/socket | Distributed subset (10–20%) @ €550 |
| --- | --- | --- | --- | --- |
| 500,000 | €60M | €275M | €550M | $28M–$55M |
| **2,000,000** `[TRAINED]` | €240M | **€1.1B** | €2.2B | **$110M–$220M** |
| 5,000,000 | €600M | €2.75B | €5.5B | $275M–$550M |

### 6.3 SOM — bottom-up from NovaCron's own published price

Because NovaCron already publishes a price (`$0.12–0.15/VM-hr`), the SOM can be computed directly rather than assumed. At 100% duty cycle:

| Scenario | VMs | ARR @ $0.12/VM-hr | ARR @ $0.15/VM-hr | Plausibility anchor |
| --- | --- | --- | --- | --- |
| Beachhead: one regional operator / sovereign agency site | 100 | **$87.6K** | **$109.5K** | Flexera: ~25% of respondents run 1–50 VMs at each major provider; this is a small-but-real fleet |
| Regional operator / mid-market enterprise | 1,000 | **$876K** | **$1.095M** | Flexera: 14% of respondents run 101–1,000 instances at AWS; 13% at Azure |
| National operator / large enterprise | 10,000 | **$8.76M** | **$10.95M** | Flexera: 11% of respondents run **>1,000 VMs at AWS alone**; 76% of large enterprises spend >$5M/month on public cloud |

Per-node alternative (Proxmox-shaped support pricing) `[ESTIMATE]`, basis = €500–€1,000/node/yr, a deliberate mid-band choice inside Proxmox's published range omitting the €120 Community floor (no support) and the €1,100 Premium ceiling (unlimited 24/7 tickets):

| Fleet | ARR @ €500/node/yr | ARR @ €1,000/node/yr |
| --- | --- | --- |
| 50 nodes | **€25K** | **€50K** |
| 500 nodes | **€250K** | **€500K** |
| 5,000 nodes | **€2.5M** | **€5.0M** |

`[ESTIMATE]` Combining both shapes: **a realistic 3-year SOM band is $0.5M–$5M ARR** for a focused beachhead of a few hundred operators, and **$5M–$12M ARR** if a single national/sovereign operator-scale customer lands.

**Explicit reality check.** NovaCron today is a pre-revenue project whose canonical binary is a small, actively-maintained server with a genuinely tested 2-node P2P fabric (`technical-audit.md` §2, `STATUS.md`-cited two-node acceptance harness 10/10). Nothing in this section is a forecast of NovaCron's traction; it is a sizing of the market *around* an unproven product. The 1–5% conversion folklore for open-source self-hosted users `[TRAINED]` (recorded as unverified in `monetization-models.md` §1) applies on top.

---

## 7. What this means for NovaCron

### 7.1 The two segments with the strongest willingness to pay

#### Segment A — Sovereign / regulated private-cloud operators **(strongest WTP)**

**Why, in evidence:**
- **A quantified, already-accepted price premium:** sovereign IaaS costs **15–40% more** than equivalent standard public cloud, and buyers *"justify this premium through compliance necessity, risk avoidance, or regulatory mandate — not raw cost efficiency."* This is the single cleanest statement of WTP found in the entire research run.
- **The fastest directly-relevant growth curve:** sovereign IaaS **27% CAGR to 2032** ($52.4B → $275.9B); Europe **+83% YoY** actual spending.
- **The fastest-growing sub-segment structurally excludes the incumbents:** EU **SEAL-3** requires EU-controlled operations, EU-national personnel, EU key management and freedom from foreign access orders — *"by design, excludes infrastructure controlled by US hyperscalers."* Self-hosted, operator-owned infrastructure is the only architecture that trivially clears this.
- **Concentrated, funded buyers:** government & defence ≈ **one third** of sovereign IaaS demand; financial services the fastest-growing (DORA, central-bank mandates); **140+ countries** have data-localisation laws; the EU has already awarded a **EUR 180M, 6-year** sovereign-cloud framework.
- **NovaCron's structural fit:** the product *is* an operator-controlled fabric. `technical-audit.md` rates the signed, reachability-verified cluster admission (HMAC-signed join, clock-skew rejection, constant-time compare, mandatory reachability callback before admission) as one of the three hardest-to-copy capabilities in the repo. That is a **verifiable-supply** primitive — the exact thing Akash has no answer for (58 providers, no admission gate) and Nosana charges a 10% fee plus a price premium to approximate.

**WTP shape:** **per-node / per-socket annual support**, matching Proxmox and Vates, in the **€500–€1,000/node/yr** band `[ESTIMATE]`, plus optional per-VM-hour metering where the buyer wants consumption pricing. Do **not** lead with a 15–40%-premium argument — lead with *control*, and let the premium be the unspoken consequence.

**Blocking prerequisite, stated bluntly:** `technical-audit.md` found the AWS-marketplace listing asserts SOC 2 Type II, ISO 27001, HIPAA and FedRAMP authorisation *"with no supporting evidence anywhere in the repo (no compliance program, no audit artifacts)."* **Sovereign buyers cannot buy an uncertified product** — SEAL-2/3, SecNumCloud, BSI C5 and FedRAMP High are procurement gates, not marketing. Certification is the cost of entry for Segment A, and none of it exists today.

#### Segment B — Bandwidth-constrained multi-site operators (edge, multi-DC, colo, regional service providers) **(second-strongest WTP)**

**Why, in evidence:**
- **Large and fast:** edge computing **$111.34B (2026) → $317.39B (2031), 23.3% CAGR**, with **services growing fastest (26.5%)**; distributed cloud **20.6% CAGR**.
- **The analyst's own restraint and challenge are NovaCron's product thesis:** *"Complex nature of edge computing infrastructure"* is listed as a **restraint**; *"Skill gap & operational expertise"* as a **challenge** — *"Shortage of skilled professionals capable of managing distributed edge environments remains a major challenge."* The **fastest-growing buyer cohort is SMEs** — precisely those least able to hire that expertise.
- **On-premises edge is already the largest deployment model**, and the fastest-growing one (regional/cloud edge) is explicitly motivated by **data residency** — so Segments A and B share a driver and can be sold together.
- **The five on-prem hypervisor competitors all lack the capability.** `competitive-landscape.md` found that across Proxmox, Harvester, OpenNebula, KubeVirt and XCP-ng *"there is no bandwidth-aware placement, no per-link transfer admission, and no compression-aware migration."* `technical-audit.md` ranks NovaCron's **bandwidth-aware transfer admission + adaptive compression decision on measured link state** (decision rule: link <500 Mbps **and** sampled ratio >1.3 ⇒ zstd-multifd) and **cross-node non-shared-storage block live migration with ownership handoff** as its #1 and #2 most-difficult-to-copy capabilities, both backed by a live two-node acceptance harness and measured results (2.17× wall-time and 2.2× wire-byte improvement for a RAM-dominant guest).
- **The economics are demonstrable in the buyer's own units.** Egress prices span **$0.02–$0.15/GB** (7.5×); the sibling analysis recommends $0.03–$0.05/GB. A customer moving large VMs across many thin links is buying exactly the product NovaCron has proven it can measure and act on.

**WTP shape:** per-node support plus **metered egress at $0.03–$0.05/GB** with regional multipliers — the one meter NovaCron gets *for free*, because the same telemetry that decides whether to admit a transfer across a link is the telemetry that bills for it (`monetization-models.md` §5b).

### 7.2 What NOT to build the business on

**The decentralized GPU marketplace, as a revenue segment.** Reasons, all grounded:
1. **The transacted market is small.** Akash — the category's flagship — is at **$9,134/day, $3.33M/yr annualised, $6.24M lifetime**, across **58 providers**. Vast.ai's *entire available inventory* at median price is a **$22.7M/yr ceiling at 100% utilisation** `[ESTIMATE]`. The measurable sector is single-digit to low-double-digit millions per year, not billions.
2. **The GPU portion is smaller still.** `[ESTIMATE]` Akash's 289 active GPUs at ~$2.5/GPU-hr ≈ **$0.53M/yr** — under 20% of its own run-rate.
3. **Liquidity is worst exactly where the money is** — 13 B200s, 26 H200s, 36 H100 SXMs available on the largest marketplace, and **zero** of Akash's 24 B300s available with no price published.
4. **NovaCron cannot attach a GPU today.** `technical-audit.md`: `SupportsGPUPassthrough()` returns `false` in *every* compiling driver (KVMDriverEnhanced, Container, Containerd, Process, CoreStub, Mock); the only `true` is in `libvirt_driver.go.disabled`, quarantined dead code. And `ai-gpu-angle.md` documents that NVIDIA's own vGPU documentation frames passthrough as an *alternative to* the mediation layer that implements live migration — so GPU live migration is a separate, larger engineering plus licensing project.

`[INFERENCE]` The decentralized-compute ecosystem is better understood as **a supply channel and a credibility narrative** than as an addressable revenue pool: it demonstrates that third-party-operated capacity can be aggregated at all, which supports the Segment B pitch, without requiring NovaCron to win a price war against a $1.73/hr H100.

### 7.3 Pricing and go-to-market implications

1. **Price per node/socket/year for the enterprise motion** (Proxmox-shaped, €500–€1,000/node/yr `[ESTIMATE]`), because that is what this buyer cohort already buys and it is the only shape with public comparables. Keep the €120-Community-style floor as a free/self-hosted tier.
2. **Keep the fabric core open with zero feature gating.** `monetization-models.md` §6 reached this conclusion independently; the market evidence here reinforces it — the fabric's value is a network effect (Segment B buyers are *multi-site*, and every additional node makes the mesh more useful to the sites already on it).
3. **Meter egress at $0.03–$0.05/GB** with honest regional multipliers rather than one global rate — every vendor examined prices regionally and none prices flat globally.
4. **Validate the published per-VM-hour price band rather than moving it.** At $0.12–0.15/VM-hr ($88–$110/VM-month) NovaCron already sits inside the Railway/Fly per-vCPU band; the risk is not the price, it is the absence of the migration/placement proof behind it.
5. **Treat certification as the gating cost for Segment A, not as marketing.** The existing marketplace listing's uncorroborated SOC 2 / ISO 27001 / HIPAA / FedRAMP claims should be removed rather than repeated — sovereign procurement will check.
6. **Sell "workload placement", not "cloud exit."** 77% of workloads remain in cloud, only 8–9% of companies plan full repatriation, and public-cloud spend is still growing — but **23% of workloads have already been repatriated and the trend is rising 2pp/year**. The growth is in *deciding where each workload runs*, which is exactly what bandwidth-aware placement and compression-decided migration do.

---

## 8. Limitations of this research

Stated so no reader mistakes a gap for a finding:

1. **No search engine was available.** `web_search` returned HTTP 429 ("monthly usage limit") on every call. All evidence is from directly-read URLs and public APIs.
2. **No credible third-party market size exists (readable) for "decentralized compute marketplaces".** Grand View Research's page on that topic returned **HTTP 403**. The measured-network substitute in §1.1 is used in its place and is labelled as measured, not projected.
3. **Four specific sources were unreachable:** Cloudian's repatriation survey landing pages (**404**); a16z's "Cost of Cloud" article (**404** on both URL variants tried); `grandviewresearch.com` (**403**); io.net's pricing page (**404**, per sibling). None were substituted with remembered figures.
4. **Golem Network publishes no machine-readable stats endpoint that could be found** (DNS failure and 404 on two candidate endpoints); its dashboard is client-side rendered. No Golem numbers are quoted.
5. **The 2,000,000-socket installed-base assumption in §6.2 is `[TRAINED]` and unverified.** It is the softest input in the report; the sensitivity table exists so a reader with a better number can substitute it.
6. **Analyst reports cited here are vendor-published abstracts, not the full studies** ($4,950–$8,150 list price each). Abstract-level figures can differ from the paid methodology. `[INFERENCE]` Where two MnM edge/cloud figures conflict in vintage (e.g. the 2022 distributed-cloud report versus the 2026 edge report), the newer one was preferred and the older one flagged as dated.
7. **Akash's USD counters are the network's own accounting** (micro-USD units in its public API). They are un-audited; they are used here because they are the only live, transparent revenue counter published by any decentralized compute network found this session.
8. **Vast.ai's capacity-value estimate assumes 100% utilisation at median price** and is therefore a ceiling, not an estimate of actual gross bookings. Vast.ai publishes available-count and price, not rentals booked.

---

### Source index (all read this session)

**Live APIs / machine-readable feeds:** `console-api.akash.network/v1/dashboard-data`; `console-api.akash.network/v1/gpu-prices`; `storage.googleapis.com/vast-public-gpu-pricing/gpu-pricing-public.json`; `api.nosana.com/api/markets/`; `nosana.com/pricing.md`; `know.rendernetwork.com/master.md` (+ `?ask=` query endpoint).

**Vendor pages:** `basecamp.com/cloud-exit`; `proxmox.com/en/products/proxmox-virtual-environment/overview`; `proxmox.com/en/about/about-us/company` (vendor-published history only — **no installation/subscriber count is published by Proxmox**); `io.net/` and `explorer.io.net/`; `golem.network/`.

**Analyst / survey:** `marketsandmarkets.com` — edge-computing-market-133384090, sovereign-infrastructure-as-a-service-iaas-market-169448214, private-cloud-market-101816685, distributed-cloud-market-165173185, software-defined-data-center-sddc-market-1025, data-center-construction-market-232213604, gpu-as-a-service-market-153834402, cloud-brokerage-market-771, ai-infrastructure-market-38254348, cloud-industry-outlook-233205216, network-function-virtualization-market-93929190, plus the report-search index for each topic; `info.flexera.com/CM-REPORT-State-of-the-Cloud` (2026 State of the Cloud, N=753) including Figure 11 read as an image; `idc.com/resource-center/blog/storm-clouds-ahead-missed-expectations-in-cloud-computing/` (Oct 2024, citing IDC Cloud Pulse 4Q 2023 and IDC Server & Storage Workloads Survey); `cloudian.com/blog/cloud-repatriation/` (IDC-sourced summary); `epoch.ai/trends` (updated Feb 2026).

**Repo (sibling research, read this session):** `research/profitability/competitive-landscape.md`; `monetization-models.md`; `ai-gpu-angle.md`; `technical-audit.md`; `marketplace/listings/aws/product-description.md` (via the above).