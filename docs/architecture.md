# Architecture

Locked design decisions for the Auctus token-based cloud resource auction protocol. Entries marked **LOCKED** are not subject to revision without invalidating the paper's theoretical claims.

---

## 1. Auction Mechanism

**LOCKED — VCG second-price (Vickrey) for single-unit, single-resource-per-round allocations.**

- Each auction round clears one resource type independently.
- The winner pays the second-highest bid: $p_{i^*} = \max_{j \neq i^*} b_j$.
- **GSP is explicitly rejected.** Generalized Second-Price does not satisfy Dominant Strategy Incentive Compatibility (DSIC). All incentive compatibility claims in this paper depend strictly on VCG. Substituting GSP invalidates Section 3.5 of the problem formulation and any theorem derived from it.
- **Multi-resource combinatorial auctions:** out of scope. Combinatorial VCG introduces exponential winner-determination complexity and is deferred to future work.

---

## 2. Token Economy

**LOCKED — Non-transferable monthly budget.**

- Each node receives $T_0 = 1000$ tokens at the start of each monthly period.
- Tokens are **non-transferable**: no secondary market, no gifting, no lending between nodes.
- **Soft rollover** at period boundary:

  $$\tau_i(0) = T_0 + \min(0.2 \cdot T_0,\ \tau_i(T_{\text{period}}))$$

  Maximum carryover is capped at $0.2 \cdot T_0 = 200$ tokens, bounding compounding advantage to $1.2 \cdot T_0$.

- **Reserve pool:**
  - Fixed at $10\%$ of total capacity per resource type, withheld from the open auction market.
  - Replenished by the $0.30 \cdot p_c$ surplus credited from each ERP event (see §4).
  - Services SFG allocations (§3) and ERP reallocations (§4) only.

---

## 3. Starvation Floor Guarantee (SFG)

**LOCKED — W = 10 consecutive empty rounds (~50 minutes real time).**

- `hunger_i(t)` counts consecutive rounds with no allocation for node $i$.
- **Trigger:** `hunger_i(t) >= W` → node $i$ receives a guaranteed allocation drawn from the reserve pool.
- **SFG price:** charged at the last market clearing price for that resource type, not an auction price:

  $$p_{\text{SFG}}(t, r) = \text{clearing\_price}(t-1, r)$$

- Allocation is drawn **from the reserve pool**, not by preempting any other node. This is a hard constraint; SFG must not cause secondary preemption.

---

## 4. Emergency Reallocation Protocol (ERP)

**LOCKED — Admin-assigned criticality, fixed weight vector, 3 declarations/month cap.**

### 4.1 Criticality Tags

Criticality $\text{criticality}_i \in \{1, 2, 3\}$ is **ADMIN-ASSIGNED at node registration and immutable**.

| Class | Label | Meaning |
|---|---|---|
| 1 | E1 | Critical infrastructure |
| 2 | E2 | Production outage |
| 3 | E3 | SLA breach |

Nodes cannot self-report or update their criticality class. Self-reporting would make the ERP score gameable and invalidate the fairness guarantees.

### 4.2 ERP Score

$$\text{ERP\_score}_i = \underbrace{0.4}_{\,w_1} \cdot \text{SLA\_breach\_prob}_i + \underbrace{0.4}_{\,w_2} \cdot \text{criticality}_i - \underbrace{0.2}_{\,w_3} \cdot \text{emergency\_count}_i$$

Weight vector $(w_1, w_2, w_3) = (0.4, 0.4, 0.2)$, threshold $\theta = 0.7$. These values are **locked for all reported experiments**; sensitivity analysis over $\theta$ is permissible as a supplementary ablation.

### 4.3 Token Accounting

Let $p_c$ = prevailing clearing price at round $t$. An ERP event distributes tokens as follows:

| Party | Delta |
|---|---|
| Emergency requester $i$ | $-1.5 \cdot p_c$ |
| Preempted node $j$ | $+0.70 \cdot p_c$ |
| Reserve pool | $+0.30 \cdot p_c$ |
| **Net** | $0$ |

The accounting is closed: $1.5 = 0.70 + 0.30 + 0.50$, where the remaining $0.50 \cdot p_c$ is the premium retained by the system (split into the two outgoing credits). No tokens are created or destroyed.

### 4.4 Monthly Cap

$$\text{emergency\_count}_i \leq N_e = 3 \quad \text{per monthly period}$$

Excess declarations are rejected regardless of ERP score. This cap prevents criticality-rich nodes from monopolizing the reallocation channel.

---

## 5. Audit Log

- **Structure:** append-only SHA-256 hash chain.

  $$L_t = \text{SHA256}(L_{t-1} \,\|\, \text{event\_type} \,\|\, \text{node\_id} \,\|\, \text{timestamp} \,\|\, \text{payload\_hash})$$

- Each auction round, SFG trigger, ERP declaration, and token deduction produces one log entry.
- **Prototype scope:** centrally administered; the central admin is a trusted party.
- **Acknowledged limitation:** central administration is a single point of trust failure. Distributed hash-chain replication (e.g., via a BFT log) is future work and is explicitly out of scope for this paper.

---

## 6. Baseline Configurations

Six baselines are implemented for comparative evaluation. All baselines run on the same SimPy DES infrastructure and the same workload traces.

| # | Name | Description |
|---|---|---|
| 1 | **FIFO** | First-come-first-served. No pricing, no priority, no compensation. |
| 2 | **Round-Robin** | Cyclic allocation across nodes in a fixed order. |
| 3 | **Priority-only** | Static node priority. Higher-priority nodes preempt lower. No compensation on preemption. |
| 4 | **Kubernetes-analog** | Priority-based preemption mirroring Kubernetes scheduler semantics. No token economy, no refund mechanism, no starvation floor. |
| 5 | **Spot-analog** | Monetary auction with a power-law wealth distribution: top 20% of nodes hold 60% of initial wealth. No monthly budget reset. Models cloud spot-market dynamics. |
| 6 | **Pure-auction Auctus** | Auctus with SFG disabled. Ablation study isolating the contribution of the starvation floor to fairness metrics. |

Baseline 6 (pure-auction) is the primary ablation; comparing it against full Auctus isolates the $\Delta J$ attributable to the SFG mechanism alone.

---

## 7. Scope Boundary

| In Scope | Out of Scope |
|---|---|
| Batch jobs (ML training, HPC, analytics pipelines) | Real-time / interactive workloads |
| Long-running jobs with SLA deadlines | Sub-second latency scheduling |
| SimPy 4.x discrete-event simulation | Real cluster deployment |
| Single-resource-per-round VCG | Multi-resource combinatorial auctions |
| Centralized audit log | Distributed log replication |

Kubernetes Priority and Preemption is the appropriate mechanism for real-time workloads; Auctus is not designed to replace it in that domain.

---

## 8. Simulation Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Simulation horizon | 30 days | One full monthly token budget period |
| Round interval $\Delta t$ | 5 min | Matches typical batch job scheduling granularity; yields 8,640 rounds/month |
| Node counts $n$ | 50 / 100 / 250 / 500 | Covers small-to-large cluster regimes for scalability analysis |
| Token budget $T_0$ | 1,000 tokens/node/month | Baseline; normalized unit for all pricing ratios |
| CPU capacity $C_{\text{CPU}}$ | 100 cores | Reference single-rack cluster size |
| RAM capacity $C_{\text{RAM}}$ | 512 GB | Reference single-rack cluster size |
| Reserve pool fraction | 10% of $C_r$ | Sufficient to service SFG without materially reducing market capacity |
| Starvation window $W$ | 10 rounds (~50 min) | Minimum SLA-safe wait before guaranteed intervention |
| ERP monthly cap $N_e$ | 3 declarations/node | Prevents ERP channel monopolization by high-criticality nodes |
| ERP threshold $\theta$ | 0.7 | Empirically tuned; sensitivity sweep reported in supplementary material |
| Preemption compensation ratio | 70% of clearing price | Balances requester incentive with preempted-node fairness |
| Soft rollover cap | 20% of $T_0$ (200 tokens) | Limits compounding advantage while rewarding resource conservation |
