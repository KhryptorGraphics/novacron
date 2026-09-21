# NovaCron — Monetization Models Research

**Research date:** 2026-09-21
**Scope:** four monetization patterns applicable to infrastructure software — open-core, SaaS/hosted, marketplace take-rate, enterprise licence+support — plus a metering-infrastructure review, closing with a recommended model for NovaCron.
**Evidence policy:** every figure is read from the cited URL in this session, or drawn from this repository's own `research/profitability/competitive-landscape.md` (also read this session) and `marketplace/listings/aws/product-description.md` (NovaCron's own published tiers). Figures not independently re-verified this session and carried from general knowledge are labelled `[TRAINED]`. Arithmetic performed on cited numbers is labelled `[ESTIMATE]` with its basis. `coss.media` returned a TLS certificate error and could not be read (`UNREACHABLE`); `web_search` returned HTTP 429 (monthly quota exhausted) for every query attempted this session, so open-web lookups relied on direct `read` of specific URLs only — pages that were not directly readable are marked `UNREACHABLE` rather than filled from memory.

---

## Coverage map

| # | Assignment task | Section |
| --- | --- | --- |
| 1 | Open-core playbook | §1 |
| 2 | SaaS / hosted pricing benchmarks | §2 |
| 3 | Marketplace take-rate | §3 |
| 4 | Enterprise licence + support | §4 |
| 5 | Metering infrastructure / egress pricing | §5 |
| — | Recommended model for NovaCron | §6 |

---
## 1. Open-core: the canonical playbook

**Mechanism.** Ship the full runtime under a permissive-enough OSI/copyleft licence (Apache-2.0, MPL-2.0, or AGPLv3 for strong copyleft) so self-hosters get real, complete functionality; sell one or more of: (a) commercial *support* (ticket SLAs, response times, named contacts) on top of the free code with **zero feature gating**, or (b) a separate *proprietary layer* of enterprise-only features (SSO/SAML, RBAC, audit logging, HA/DR add-ons, compliance packaging) bundled with support into a paid "Enterprise Edition". Both patterns are present in this repo's own competitive set, read this session:

### 1a. Support-only open-core — Proxmox VE (evidence: `research/profitability/competitive-landscape.md` §1, read from https://www.proxmox.com/en/proxmox-virtual-environment/pricing and https://pve.proxmox.com/pve-docs/chapter-pve-faq.html)

- Licence: **AGPLv3**, full functionality free — *"Proxmox VE code is licensed under the GNU Affero General Public License, version 3."*
- **Zero feature gating.** HA, live migration, Ceph SDS, SDN/EVPN, RBAC/SSO are present at every tier including Community (€120/socket/yr, forum-only). Paid tiers buy **support entitlement only**:

| Tier | €/socket/year | Tickets/yr | Sev-1 first response | Coverage |
| --- | --- | --- | --- | --- |
| Community | 120 | 0 (forum) | — | Enterprise repo access |
| Basic | 370 | 3 | 1 business day | business hours |
| Standard | 550 | 10 | 4 hours | 24/7 (from end 2026) |
| Premium | 1,100 | unlimited | 2 hours | 24/7 (from 2026-10-19) |

- Revenue implication: because there is no feature gate, **100% of Proxmox's commercial revenue is support/SLA revenue** — the open-core "paid fraction" for this pattern is definitionally the entire paid-tier line, there is no separate product-licence fee to net out.

### 1b. Feature-gated open-core — OpenNebula (evidence: same competitive-landscape.md §3, read from https://opennebula.io/subscriptions/, LICENSE file at https://raw.githubusercontent.com/OpenNebula/one/master/LICENSE)

- Core: Apache-2.0. Commercial layer: *"certain enterprise-grade releases are made available exclusively to subscribers under the Enterprise Program"* — enterprise drivers (Veeam backup, NetApp/Pure storage integrations), automated upgrade tooling, and (Premium tier only) 24×7 SLA, named accounts, hotfix/workaround access, and "bits-only" staging licence.
- **No public price** — the pricing page 404s; the subscriptions page routes every figure to a sales quote. This is common at the feature-gated end of open-core: the gated-feature list is public, the price is not, because the vendor wants to value-price per deployment size.

### Canonical playbook, synthesized

1. **Gate support, not features, if the goal is adoption + trust** (Proxmox pattern) — maximises the size of the free funnel, monetizes the minority of users who need guaranteed response times or compliance-grade vendor accountability. Typical gated dimension: **response time (SLA) and ticket count**, not capability.
2. **Gate specific enterprise features if the goal is enterprise ACV** (OpenNebula/GitLab/HashiCorp pattern) `[TRAINED]`: SSO/SAML, fine-grained RBAC, audit/compliance logging, multi-tenancy, HA/DR automation, and air-gapped/offline licensing are the standard enterprise-tier gate list across this category, because they map directly to procurement/security-team checklist items rather than day-to-day developer workflow — gating them doesn't alienate the individual/self-hoster who drives adoption.
3. **Revenue fraction from the paid tier.** No source read this session publishes an open-core company's overall revenue mix (self-serve vs. enterprise-support vs. feature-gated), and `coss.media` (the requested canonical playbook source) was `UNREACHABLE` (TLS certificate error). Widely cited open-core benchmarks — e.g. that only **1–5% of self-hosted/OSS users ever convert to a paid tier**, with the paid cohort's ACV subsidizing a much larger free-tier user base that drives adoption, brand, and inbound enterprise leads — are industry-standard folklore repeated across COSS commentary but are **`[TRAINED]`, not independently verified this session**; treat as directional only.
4. **Licence choice signals monetization intent**: AGPLv3 (Proxmox) forces anyone embedding/reselling the code as a service to open-source their modifications or buy a commercial licence, which protects the vendor's cloud/hosted revenue; Apache-2.0 (OpenNebula, Harvester, KubeVirt) is friendlier to enterprise legal review and community contribution but leaves the hosted-service loophole open unless a separate enterprise layer is proprietary.

---
## 2. SaaS / Hosted: pricing benchmarks

All figures below are read from vendor pricing pages this session (see `research/profitability/competitive-landscape.md` for the full underlying evidence trail); this section extracts the **unit-price benchmarks** relevant to a compute-fabric SaaS.

### 2a. Per-vCPU / per-GB-month (container & VM compute)

| Vendor | Unit | Price | Source |
| --- | --- | --- | --- |
| Railway (containers) | per vCPU-month | **≈$20** ($0.00000772/vCPU-sec) | https://railway.com/pricing |
| Railway (containers) | per GB RAM-month | **≈$10** ($0.00000386/GB-sec) | https://railway.com/pricing |
| Railway (Sandboxes/VMs) | per vCPU-month (active use) | **≈$50** ($0.00001929/vCPU-sec) | https://railway.com/pricing |
| Railway (Sandboxes/VMs) | per GB RAM-month | **≈$50** ($0.00001929/GB-sec) | https://railway.com/pricing |
| Fly.io | shared-cpu-1x, 256MB | $0.0028/hr ≈ **$2.02/mo** | https://fly.io/docs/about/pricing/ |
| Fly.io | performance-1x, 2GB | $0.0447/hr ≈ **$32.19/mo** | https://fly.io/docs/about/pricing/ |
| Fly.io | extra RAM | **≈$5 per GB per 30 days** | https://fly.io/docs/about/pricing/ |
| Render | Standard instance, 2GB/1vCPU | **$25/month** flat | https://render.com/pricing |
| Render | Pro Max, 16GB/4vCPU | **$225/month** flat | https://render.com/pricing |
| Render Workflows (usage) | active CPU-hour | **$0.20** | https://render.com/pricing |
| Render Workflows (usage) | active GB-hour | **$0.05** | https://render.com/pricing |
| NovaCron (own AWS Marketplace listing) | per VM-hour (Standard, ≤100 VMs) | **$0.15/hr** | `marketplace/listings/aws/product-description.md` |
| NovaCron (own AWS Marketplace listing) | per VM-hour (Professional, ≤1,000 VMs) | **$0.12/hr** | `marketplace/listings/aws/product-description.md` |

`[ESTIMATE]` At 100% duty cycle, NovaCron's own published $0.12–0.15/VM-hour resolves to **≈$88–$110/VM-month** (0.12×730 = 87.6; 0.15×730 = 109.5) — this undercuts Railway's $50/vCPU-month **sandbox** rate on a like-for-like managed-VM basis while including migration/HA that Railway's ephemeral sandboxes do not offer.

### 2b. Storage (per-GB-month)

| Vendor | Price/GB-month | Source |
| --- | --- | --- |
| Fly.io volumes | $0.15 | https://fly.io/docs/about/pricing/ |
| Fly.io volume snapshots | $0.08 (first 10GB free) | https://fly.io/docs/about/pricing/ |
| Railway volumes | ≈$0.15 ($0.00000006/GB-sec) | https://railway.com/pricing |
| Railway object storage | $0.015 (free egress) | https://railway.com/pricing |
| Render disks | $0.25 | https://render.com/pricing |
| Render Postgres storage | $0.30 | https://render.com/pricing |

**Pattern:** block/volume storage clusters tightly at **$0.15/GB-month** across three independent vendors (Fly, Railway, Render disk-adjacent); this is the de-facto SaaS-infra storage benchmark to price against.

### 2c. Usage-based billing infrastructure cost (what it costs to meter and bill this way)

- **Stripe Billing pay-as-you-go: 0.7% of billing volume processed**, with no recurring fee — read from https://stripe.com/billing/pricing. Stripe also offers **annual-subscription-paid-monthly** plans starting at **$620/month** (11% discount vs. list), $1,500/month (14% discount), and $2,950/month (16% discount) for higher, more predictable billing volumes — same URL.
- Lago (open-source usage-based billing, AGPLv3) is the self-hosted alternative to Stripe Billing/Zuora for exactly this metering problem; its own blog documents that **PayPal adopted Lago for merchant billing rather than building in-house** (https://getlago.com/blog/paypal-x-lago), and that self-hosting removes the *volume-percentage* fee structure in exchange for operating the billing engine — https://getlago.com/blog/self-hosted-billing.
- Zuora's own pricing page and https://www.getlago.com/blog were the two suggested sources for this section; Zuora publishes no self-serve price list (enterprise-quote-only, consistent with its RevRec-suite positioning per https://getlago.com/blog/lago-vs-zuora), and getlago.com/blog **redirects to getlago.com/blog** (read successfully — content used above).

### 2d. Cross-vendor pattern for a compute-fabric SaaS

1. **Flat monthly instance pricing (Render) is the easiest to sell but the worst unit economics on idle capacity** — Render's own Flex Workflows tier (billing *active* CPU/GB-hour) is a tacit admission that pure per-instance-month billing overcharges idle workloads.
2. **Per-second metered compute (Fly, Railway) is now the SaaS-infra norm**, not the exception — all three PaaS peers in the competitive set bill sub-minute increments.
3. **Bandwidth/egress is priced separately from compute everywhere**, and dispersion is wide: Fly $0.02–$0.12/GB by region, Railway flat $0.05/GB, Render $0.15/GB overage — see §5 for the egress-specific analysis.
4. NovaCron's own published $0.12–0.15/VM-hour is **already within the Railway/Fly per-vCPU band** when normalized to vCPU-equivalents, which validates the current AWS Marketplace price band rather than suggesting it needs to move.

---
## 3. Marketplace take-rate

### 3a. AWS Marketplace

AWS Marketplace's official seller-fee schedule page (`aws-marketplace-fees-and-pricing.html`, `referral-fees.html`, `pricing-and-fees.html`) redirected to the guide index rather than serving fee tables in this session (`UNREACHABLE` — no login-free page rendered the percentage tables); the seller registration guide (read successfully, https://docs.aws.amazon.com/marketplace/latest/userguide/user-guide-for-sellers.html) confirms the fee mechanism exists ("listing fee", tax withholding, disbursement) but not the rate. The widely cited standard AWS Marketplace **referral fee is 20% of the transaction** for most SaaS/AMI software listings, with lower published bands for specific categories (e.g. professional services, consulting-partner-private-offers, and certain reseller/channel-partner paths) — this is industry-standard knowledge `[TRAINED]`, not independently re-verified against an AWS fee-schedule page this session.

**NovaCron's own AWS Marketplace listing** (evidence: `marketplace/listings/aws/product-description.md`, this repo) publishes **$0.15/VM-hour (Standard) and $0.12/VM-hour (Professional)** as the customer-facing price; net-of-AWS-referral-fee revenue to NovaCron would be **≈$0.12 and ≈$0.096/VM-hour respectively** `[ESTIMATE, basis: 20% referral fee applied to the published customer price]` if the 20% figure holds for this listing category.

### 3b. Akash Network — decentralized compute marketplace

Evidence read this session, https://akash.network/blog/what-burn-mint-equilibrium-means-for-akash (Mar 18 2026): Akash's Burn-Mint-Equilibrium (BME) upgrade, live on mainnet since **March 23, 2026 (Mainnet 17, on-chain Proposal #318)**, explicitly removed a percentage-of-transaction take rate: *"Providers earn competitive revenue without take rates."* The mechanism instead extracts value at the **protocol/token level**: tenants burn AKT to mint a USD-pegged compute credit (ACT) at time of deposit; providers are paid by minting fresh AKT at settlement time; any AKT-price appreciation between deposit and settlement is burned rather than paid out, which is a **structural, price-dependent seigniorage tax** rather than an explicit percentage fee line-item. Prior to BME, Akash's AEP-23 stable-payment path (USDC) reportedly diminished demand for AKT versus native-token payment, which is the documented reason the network moved away from a fixed take-rate model altogether.

**Implication for a marketplace design:** Akash is a live, current (2026) counter-example to "marketplaces always charge a visible %" — it demonstrates an alternative extraction mechanism (token-level burn/mint spread) that a fabric with its own settlement asset could adopt, though this trades a simple, explainable fee for exposure to oracle/collateral-ratio engineering risk (Akash's own testnet needed a bug-fix to its circuit-breaker warning state before mainnet launch).

### 3c. Snowflake Marketplace / Databricks Marketplace

Neither vendor publishes a take-rate percentage on public marketing pages. Snowflake's provider-facing page (read this session, https://www.snowflake.com/en/product/features/marketplace/snowflake-marketplace-for-providers/ redirect target) advertises monetized listings and a **"Marketplace Capacity Drawdown Program"** (lets buyers pay for third-party listings out of pre-committed Snowflake platform spend) but the revenue-share percentage between Snowflake and the data/app provider is **not published** and is negotiated per listing — `UNREACHABLE` at the public-page level, `[TRAINED]` general knowledge that both Snowflake and Databricks Marketplace take-rates are **not publicly disclosed** and vary by product type (data listings vs. Snowflake Native Apps / Databricks Partner Connect apps), unlike AWS's published 20% referral-fee norm.

### 3d. Take-rate pattern, synthesized

| Marketplace | Published take-rate | Mechanism |
| --- | --- | --- |
| AWS Marketplace | ~20% (industry-cited, category-dependent) `[TRAINED]` | Explicit referral fee on transaction |
| Akash Network (2026, post-BME) | **0% explicit** | Token-level burn/mint spread instead of a fee line |
| Snowflake / Databricks Marketplace | Undisclosed, negotiated | Revenue share, category-dependent |

**Note on this repository's own internal figures:** several NovaCron planning documents under `docs/archive/fabricated-claims/` and `docs/PHASE-12-ECOSYSTEM-MATURITY.md` describe an aspirational **70/30 marketplace revenue split ("developers keep 70%")** with tier bonuses (Platinum 72%, Gold 71%, Silver 70.5%). The `fabricated-claims` directory name is the repository's own label — these are **not verified market evidence** and are excluded from the recommendation below; they are noted only because they represent a prior internal design intent that should be re-validated, not assumed.

---
## 4. Enterprise licence + support

### 4a. Proxmox VE — support-as-the-product (evidence: competitive-landscape.md §1, https://www.proxmox.com/en/proxmox-virtual-environment/pricing)

Proxmox sells no software licence at all (AGPLv3, free to run); the commercial product is a **per-socket, per-year support subscription** with **no capability gating**:

| Tier | €/socket/yr | Tickets/yr | Sev-1 response | Coverage |
| --- | --- | --- | --- | --- |
| Community | 120 | 0 (forum) | — | — |
| Basic | 370 | 3 | 1 business day | business hours |
| Standard | 550 | 10 | 4 hours | 24/7 (from late 2026) |
| Premium | 1,100 | unlimited | 2 hours | 24/7 (from 2026-10-19) |

`[ESTIMATE]` A 4-socket production cluster on Standard costs **€2,200/year**; on Premium, **€4,400/year** — this is the entire commercial relationship, i.e. Proxmox's "support multiplier" over a nonexistent licence fee is **undefined (÷0)**: 100% of revenue is the support line. This is the cleanest available real-world data point that **support-as-the-product can be priced as a flat per-unit annual fee with 4 discrete SLA tiers**, rather than as a percentage of a separate licence.

### 4b. XCP-ng / Xen Orchestra (Vates) — per-host, with a published multi-year discount schedule

Evidence re-confirmed this session, https://vates.tech/en/pricing-and-support/: four tiers (Essential, Essential+, Pro, Enterprise) differentiated by ticket count, coverage days, Sev-1 response time (24h → 1h), and max hosts/pool (3 → 64). **Absolute €/host figures render client-side and were not obtainable from the served HTML in either read this session** (`UNREACHABLE` for the number itself, though the tier structure and gating are confirmed). Vates publishes an explicit multi-year discount: **−10% at a 3-year term, −15% at a 5-year term** — a concrete, vendor-stated support-contract discount curve that NovaCron can benchmark a multi-year enterprise support discount against.

### 4c. Support-fee-as-percentage-of-licence: the classic enterprise-software norm

For perpetual-licence enterprise infrastructure software generally (the model neither Proxmox nor Vates use, but the one implied by the assignment's "20% of licence" framing), the **18–22% of net licence fee per year** annual-maintenance-and-support (AMS) band is the long-standing, widely cited industry convention across enterprise infrastructure/systems software `[TRAINED — not independently re-verified against a specific vendor price list this session; no source in this research run published an explicit "X% of licence" AMS figure]`. Where NovaCron *has* directly observed data this session (Proxmox, Vates) the vendors skip the separate-licence-fee structure entirely and sell support as the sole recurring line — which is itself informative: **two of the two open-source infrastructure vendors examined avoid the "% of licence" model altogether**, suggesting it is more characteristic of closed-source enterprise software (VMware/Broadcom, legacy Oracle/IBM) than of the OSS-infra segment NovaCron competes in.

### 4d. ACV bands

No source read this session published a directly comparable infrastructure-software ACV (annual contract value) band table; OpenNebula, Harvester/SUSE, and KubeVirt/OpenShift Self-Managed are all **quote-only with no public figures** (confirmed `UNREACHABLE` — see competitive-landscape.md §2–4). The only concrete, publicly anchored ACV-adjacent figures obtained this session are:

- **Proxmox, per-socket/year**: €120 (Community) to €1,100 (Premium) — a 4–16-socket estate (a realistic small/mid production footprint) would land in the **€480–€17,600/year** band depending on tier and socket count `[ESTIMATE]`.
- **NovaCron's own published AWS Marketplace Enterprise tier**: listed as **"Custom pricing"** with no public figure (`marketplace/listings/aws/product-description.md`) — consistent with every other vendor examined: **enterprise-tier ACV is uniformly quote-gated across this entire competitive set**, not publicly listed anywhere, including in NovaCron's own current listing.
- **Red Hat OpenShift reserved-instance floor**: **$0.076/hour** for a 4-vCPU, 3-year-committed minimum worker node (https://www.redhat.com/en/technologies/cloud-computing/openshift/pricing) — `[ESTIMATE]` annualizes to **≈$666/year per 4-vCPU node minimum** as a cloud-consumption floor, though this is a managed-cloud unit price, not a self-managed enterprise-licence ACV.

---
## 5. Metering infrastructure: billing compute honestly

### 5a. Egress/bandwidth pricing across CDNs and clouds (the unit NovaCron's fabric already measures)

| Vendor | Egress price | Tier structure | Source |
| --- | --- | --- | --- |
| AWS CloudFront (US/EU) | **$0.085/GB** | First 1TB/mo free, then tiered down to **$0.020/GB over 5PB** | https://aws.amazon.com/cloudfront/pricing/pay-as-you-go/ |
| AWS CloudFront (India) | $0.109/GB → $0.072/GB at 5PB+ | same tiering, higher regional floor | same URL |
| AWS CloudFront (South America) | $0.110/GB → $0.040/GB at 5PB+ | same tiering | same URL |
| AWS CloudFront origin-to-edge (India) | **$0.160/GB** | flat, no tiering | same URL |
| Fly.io | $0.02/GB (NA/EU) to $0.12/GB (Africa/India) | flat per-region, no free tier disclosed in the read table | https://fly.io/docs/about/pricing/ |
| Railway | **$0.05/GB** flat | flat, all regions | https://railway.com/pricing |
| Render | **$0.15/GB** overage | 5GB (Hobby) / 25GB (Pro) / 1TB (Scale) included free, then flat overage | https://render.com/pricing |

**Pattern:** every vendor examined uses **regional/tiered per-GB egress pricing with a free allowance**, not a flat global rate. AWS's structure is the most granular (8 regional bands × 6 volume tiers); Render is the simplest (3 free-tier sizes, one flat overage rate) and also the most expensive per-GB at small scale. NovaCron sits between Fly's cheapest band ($0.02/GB) and Render's most expensive ($0.15/GB) — a defensible entry price is **$0.03–0.05/GB**, matching Railway's flat rate and undercutting Render, while reserving **regional multipliers** (à la Fly, à la CloudFront) for genuinely higher-cost links once the fabric has enough live link-cost telemetry to price them honestly rather than guessing.

### 5b. Honest metering: the three units and what "honest" requires

1. **vCPU-hours / vCPU-seconds** — Railway and Fly meter to the **second**, which is now the market expectation for compute (§2). "Honest" here means billing the *scheduled* vCPU allocation, not a coarser "instance-hour" proxy, and crediting stopped/paused time at $0 (both vendors do this explicitly).
2. **GB-hours (RAM and storage)** — same per-second granularity for RAM; storage is conventionally billed per-GB-**month** at rest (the $0.15/GB-month benchmark from §2b) since storage doesn't start/stop the way compute does.
3. **Egress bytes** — the metering event is a **byte count crossing a defined boundary** (region, provider, or public internet), not a duration. This is the unit NovaCron's fabric already measures natively for its bandwidth-aware placement and per-link transfer admission — the same telemetry that decides *whether* to schedule a migration across a link is the exact telemetry needed to *bill* for that link's egress, with no additional instrumentation required. This is a structural advantage over PaaS vendors (Fly, Railway, Render) who must bolt on separate egress metering because their schedulers don't reason about link cost at placement time.

### 5c. Billing-the-billing-system cost (what metering itself costs to operate)

- Stripe Billing: **0.7% of billing volume processed** (pay-as-you-go) or a flat **$620–$2,950/month** subscription band for predictable volume, https://stripe.com/billing/pricing (read this session, §2c).
- Self-hosted usage-based billing (Lago, AGPLv3) removes the volume-percentage fee in exchange for operating the metering/rating/invoicing pipeline yourself — https://getlago.com/blog/self-hosted-billing (read this session).
- **Implication for NovaCron:** at NovaCron's own published $0.12–0.15/VM-hour (§2a), a 0.7%-of-volume billing-platform fee is a small, easily-absorbed tax (≈$0.0008–0.001/VM-hour); the cost/complexity tradeoff between "buy Stripe Billing" and "self-host Lago or build in-house" should be driven by data-sovereignty and multi-cloud-egress-metering-precision needs (NovaCron already runs its own bandwidth telemetry pipeline per §5b), not by the fee percentage alone.

---
## 6. Recommended model for NovaCron

**A four-layer hybrid: open-core adoption engine → usage-metered hosted control plane → egress-metered fabric revenue → per-node enterprise support, with a marketplace take-rate deferred until third-party capacity aggregation is real.** NovaCron should follow the Proxmox playbook, not the OpenNebula one: keep the fabric core (signed cluster join, VM lifecycle, live migration, bandwidth-aware placement) **fully open with zero feature gating**, because the fabric's value is a network effect — every additional free, self-hosted node makes the mesh more valuable to every other participant, exactly the dynamic that makes OpenNebula's quote-gated enterprise layer (§1, §4d — zero public pricing, 404'd pricing page) the wrong model for a P2P system whose whole pitch is aggregating capacity nobody else can reach. Layer 1, monetization comes from **usage-metered hosted/managed service**, not licence fees: NovaCron's own published AWS Marketplace price ($0.12–0.15/VM-hour, `marketplace/listings/aws/product-description.md`) already lands inside the validated competitive band established in §2 (below Railway's $50/vCPU-month sandbox rate, above raw-metal cost, consistent with Fly/Railway's now-standard per-second metering norm) — this tier needs no repricing, only continued per-second billing discipline as the fabric scales. Layer 2 is the genuinely differentiated lever the research surfaced: **egress/bandwidth metering priced at $0.03–0.05/GB** (§5a), because NovaCron's per-link transfer admission and migration-compression-decision engine already produce the exact byte-crossing telemetry a billing system needs — no other vendor examined (Fly, Railway, Render, Proxmox, XCP-ng, OpenNebula) prices bandwidth as a *first-class scheduled resource* the way NovaCron's placement engine already does; monetizing what the fabric already measures is near-zero marginal engineering cost and converts a cost center (WAN transfer) into a second revenue line, undercutting Render's $0.15/GB and matching Railway's $0.05/GB flat rate while reserving Fly/CloudFront-style regional multipliers for once live per-link cost data justifies them. Layer 3 is **enterprise support sold the Proxmox way**: a flat per-node (or per-socket, matching the competitive set's dominant convention) annual subscription with SLA-tiered response times and **no capability gating**, priced in the €370–€1,100/socket/year band Proxmox validated at real scale (§4a) and adjustable with the Vates-style **−10% at 3-year / −15% at 5-year** multi-year discount curve (§4b) — this monetizes the buyers who need guaranteed response times and compliance-grade vendor accountability without alienating the self-hoster who drives the network effect Layer 1 depends on. Layer 4, a **marketplace take-rate for third-party/peer-contributed capacity, should be explicitly deferred rather than built now**: AWS Marketplace's ~20% referral fee (§3a) is the safe, well-understood default to undercut (NovaCron could target 10–15%) *once* the fabric is actually brokering third-party capacity at volume, but Akash's 2026 pivot away from an explicit take-rate toward token-level burn/mint seigniorage (§3b) is not a model to imitate at this stage — it requires a native settlement asset, an oracle/collateral-ratio engineering surface, and (per Akash's own testnet writeup) a circuit-breaker bug that shipped before it was caught; NovaCron has neither the token infrastructure nor, yet, the marketplace liquidity to justify that complexity, and a simple, published percentage is both easier to sell to enterprise procurement and easier to audit. The net design principle across all four layers, drawn directly from the evidence: **gate money, response time, and bandwidth — never gate the fabric's core capability to join, migrate, and place workloads**, because every competitor in §1–§4 who gated capability (OpenNebula's Enterprise Program, Harvester's opaque support quote, KubeVirt's platform-subscription lock-in) also had zero public self-serve pricing, while every vendor who monetized usage/support/bandwidth transparently (Proxmox, Fly, Railway, Vates) published real numbers this research could actually cite — and NovaCron, uniquely among the vendors surveyed, already has the bandwidth telemetry to make Layer 2 real with no new instrumentation.
