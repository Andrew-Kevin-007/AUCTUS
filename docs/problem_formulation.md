# Problem Formulation

## 1. System Definition

Let $\mathcal{N} = \{1, 2, \ldots, n\}$ denote the set of participating nodes and $\mathcal{R} = \{\text{CPU}, \text{RAM}, \text{STG}, \text{NET}\}$ the set of tradeable resource types. Each resource type $r \in \mathcal{R}$ has a fixed total cluster capacity $C_r$, with reference values:

$$C_{\text{CPU}} = 100 \text{ cores}, \qquad C_{\text{RAM}} = 512 \text{ GB}$$

Time is discretized into auction rounds of interval $\Delta t = 5$ minutes, yielding

$$|\mathcal{T}| = 8640 \text{ rounds per monthly period}$$

Each node $i \in \mathcal{N}$ is initialized with a token endowment

$$T_0 = 1000 \text{ tokens per node per monthly period}$$

---

## 2. Token State

### 2.1 Holdings

Let $\tau_i(t) \in \mathbb{R}_{\geq 0}$ denote the token balance of node $i$ at the start of round $t$, subject to the hard cap

$$\tau_i(t) \leq T_0 \quad \forall\, i \in \mathcal{N},\; t \in \mathcal{T}$$

### 2.2 Period Boundary — Soft Rollover

At the start of each monthly period, balances reset with a partial carryover:

$$\tau_i(0) = T_0 + \min\!\left(0.2 \cdot T_0,\; \tau_i(T_{\text{period}})\right)$$

### 2.3 Deduction on Win

When node $i$ wins an allocation in round $t$ at price $p_i(t)$:

$$\tau_i(t+1) = \tau_i(t) - p_i(t)$$

### 2.4 Refund on Preemption

If node $i$ is preempted after winning, it receives a partial refund:

$$\tau_i(t+1) = \tau_i(t) + 0.70 \cdot p_i(t)$$

---

## 3. VCG Auction Clearing

### 3.1 Bid Submission

At each round $t$ and for each resource $r \in \mathcal{R}$, node $i$ submits a sealed bid

$$b_i(t, r) \leq \tau_i(t)$$

### 3.2 Winner Determination

$$i^* = \underset{i \in \mathcal{N}}{\arg\max}\; b_i(t, r)$$

### 3.3 VCG Payment (Second-Price)

The winning node pays the second-highest submitted bid:

$$p_{i^*}(t, r) = \max_{j \neq i^*}\; b_j(t, r)$$

### 3.4 Allocation Rule

$$a_i(t, r) = \begin{cases} 1 & \text{if } i = i^* \\ 0 & \text{if } i \neq i^* \end{cases}$$

### 3.5 Dominant Strategy Incentive Compatibility (DSIC)

Under the second-price rule, truthful bidding is a weakly dominant strategy. Let $v_i(t, r)$ be node $i$'s true valuation. For all alternative bids $b_i \neq v_i$ and all bid profiles $b_{-i}$:

$$u_i\!\left(v_i,\, b_{-i}\right) \geq u_i\!\left(b_i,\, b_{-i}\right)$$

where $u_i$ is utility (valuation minus payment). This property holds regardless of the strategies of other nodes.

---

## 4. Starvation Floor Guarantee (SFG)

### 4.1 Hunger Counter

Define $\text{hunger}_i(t)$ as the number of consecutive rounds ending at $t$ in which node $i$ received no allocation:

$$\text{hunger}_i(t) = \max\!\left\{k \geq 0 : a_i(s, r) = 0 \;\forall\, r \in \mathcal{R},\; \forall\, s \in [t-k, t]\right\}$$

### 4.2 Trigger Condition

If $\text{hunger}_i(t) \geq W$ (default $W = 10$), node $i$ receives a guaranteed allocation from the reserve pool, bypassing the open auction for that round.

### 4.3 SFG Price

The SFG allocation is not subject to competitive clearing. The charged price is the previous round's market clearing price:

$$p_{\text{SFG}}(t, r) = \text{clearing\_price}(t - 1, r)$$

### 4.4 Reserve Pool

A fixed fraction of total capacity is withheld from the open auction to service SFG and ERP allocations:

$$C_r^{\text{reserve}} = 0.10 \cdot C_r \quad \forall\, r \in \mathcal{R}$$

The capacity available in the open auction is $C_r^{\text{market}} = 0.90 \cdot C_r$.

---

## 5. Emergency Reallocation Protocol (ERP)

### 5.1 ERP Score

Each node $i$ carries a continuously updated emergency priority score:

$$\text{ERP\_score}_i = 0.4 \cdot \text{SLA\_breach\_prob}_i + 0.4 \cdot \text{criticality}_i - 0.2 \cdot \text{emergency\_count}_i$$

where:

- $\text{SLA\_breach\_prob}_i \in [0, 1]$ — estimated probability of an imminent SLA violation.
- $\text{criticality}_i \in \{1, 2, 3\}$ — admin-assigned at node registration, **immutable**:
  - $1$ — Critical infrastructure (E1)
  - $2$ — Production outage (E2)
  - $3$ — SLA breach (E3)
- $\text{emergency\_count}_i$ — number of ERP declarations made by node $i$ in the current monthly period.

### 5.2 Trigger Condition

An ERP declaration by node $i$ at round $t$ is valid iff both conditions hold simultaneously:

$$\text{ERP\_score}_i > \theta \quad (\text{default } \theta = 0.7)$$

$$\tau_i(t) \geq 1.5 \cdot \text{clearing\_price}(t, r)$$

### 5.3 ERP Pricing and Token Accounting

Let $p_c = \text{clearing\_price}(t, r)$. The ERP transaction distributes tokens as follows:

| Party | Token Flow |
|---|---|
| Emergency requester $i$ | $-1.5 \cdot p_c$ |
| Preempted node $j$ | $+0.70 \cdot p_c$ |
| Reserve pool | $+0.30 \cdot p_c$ |

The $0.30 \cdot p_c$ credited to the reserve pool closes the token accounting loop.

### 5.4 Monthly Cap

$$\text{emergency\_count}_i \leq N_e = 3 \quad \forall\, i \in \mathcal{N},\; \text{per monthly period}$$

Declarations in excess of $N_e$ are rejected regardless of ERP score.

---

## 6. Fairness Metric

System-wide allocation fairness over a simulation period of $T$ rounds is measured by **Jain's Fairness Index**:

$$J(\mathbf{x}) = \frac{\left(\displaystyle\sum_{i=1}^{n} x_i\right)^2}{n \cdot \displaystyle\sum_{i=1}^{n} x_i^2}, \qquad J \in \left[\frac{1}{n},\, 1\right]$$

where $x_i$ is the total allocation received by node $i$ over the simulation period:

$$x_i = \sum_{t=1}^{T} \sum_{r \in \mathcal{R}} a_i(t, r)$$

$J = 1$ indicates perfect fairness; $J = 1/n$ indicates maximum unfairness (one node receives all allocations). The SFG mechanism enforces a lower bound on $J$ by preventing any node from being starved for more than $W$ consecutive rounds.
