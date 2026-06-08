"""
src/core/node.py

Node entity for the Auctus token-based cloud resource auction simulation.
Criticality is ADMIN-ASSIGNED at registration and must not be mutated by
any node-level method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Criticality(IntEnum):
    """Admin-assigned criticality class.  Immutable after registration."""
    E1_CRITICAL        = 1   # Critical infrastructure
    E2_PROD_OUTAGE     = 2   # Production outage
    E3_SLA_BREACH      = 3   # SLA breach


# Per-resource base valuation in tokens per unit.
# CPU: tokens / core,  RAM: tokens / GB
BASE_VALUATION: Dict[str, float] = {
    "CPU": 5.0,
    "RAM": 1.0,
    "STG": 0.2,
    "NET": 0.5,
}

# ERP parameters (locked per architecture.md)
_ERP_W1: float = 0.4
_ERP_W2: float = 0.4
_ERP_W3: float = 0.2
_ERP_THETA: float = 0.7
_ERP_MONTHLY_CAP: int = 3
_ERP_REQUESTER_MULTIPLIER: float = 1.5

# Preemption refund ratio (locked)
_PREEMPTION_REFUND_RATIO: float = 0.70

# Soft-rollover cap fraction (locked)
_ROLLOVER_CAP_FRACTION: float = 0.20


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """
    Represents a single bidding node in the Auctus simulation.

    Parameters
    ----------
    node_id : str
        Unique identifier for this node.
    criticality : int
        Admin-assigned criticality class (1=E1, 2=E2, 3=E3).
        Must not be changed after construction.
    tier : int
        Node tier 1–5.  Affects Poisson job-arrival rate λ in the workload
        generator (higher tier → higher λ).
    T0 : float
        Monthly token budget.  Defaults to 1000.
    sla_deadline : float
        Task deadline expressed in simulation time (minutes from sim start).
    current_task_progress : float
        Progress fraction of the current task in [0.0, 1.0].
    current_sim_time : float
        Current simulation time in minutes.  Updated externally each round.
    rng : np.random.Generator | None
        Optional seeded RNG for reproducible bid generation.
    """

    node_id:               str
    criticality:           int        # Criticality.E1/E2/E3 — ADMIN ONLY
    tier:                  int        # 1–5
    sla_deadline:          float      # sim-time minutes
    current_task_progress: float = 0.0
    current_sim_time:      float = 0.0
    T0:                    float = 1000.0

    # Mutable runtime state
    tokens:           float = field(init=False)
    hunger_counter:   int   = field(init=False, default=0)
    emergency_count:  int   = field(init=False, default=0)

    total_allocated:    Dict[str, float]                   = field(init=False)
    allocation_history: List[Tuple[int, str, float]]       = field(init=False)
    bid_history:        List[Tuple[int, str, float]]       = field(init=False)

    rng: np.random.Generator = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Post-init
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.criticality not in (1, 2, 3):
            raise ValueError(
                f"criticality must be 1, 2, or 3 (admin-assigned); got {self.criticality}"
            )
        if not (1 <= self.tier <= 5):
            raise ValueError(f"tier must be 1–5; got {self.tier}")

        self.tokens = self.T0
        self.total_allocated = {r: 0.0 for r in BASE_VALUATION}
        self.allocation_history = []
        self.bid_history = []

        if self.rng is None:
            self.rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # SLA helpers
    # ------------------------------------------------------------------

    def sla_breach_probability(self) -> float:
        """
        Linear estimate of SLA breach probability.

        If the node's current task progress is below the fraction of the
        deadline that has already elapsed, the shortfall is returned as a
        probability, capped at 1.0.

        Returns 0.0 if the deadline has not started (sim_time == 0) or
        progress is on/ahead of schedule.
        """
        if self.sla_deadline <= 0.0:
            return 0.0

        elapsed_fraction = min(self.current_sim_time / self.sla_deadline, 1.0)
        shortfall = elapsed_fraction - self.current_task_progress
        return float(np.clip(shortfall, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Bidding
    # ------------------------------------------------------------------

    def generate_bid(self, resource_type: str, current_round: int) -> float:
        """
        Generate a sealed bid for *resource_type* in the current auction round.

        Demand urgency is drawn from Beta(2, 5), producing values skewed
        below 1.  An urgency modifier scales the bid upward when an SLA
        breach is imminent.

        Returns 0.0 if the node cannot afford to participate (tokens < 1).

        Parameters
        ----------
        resource_type : str
            One of "CPU", "RAM", "STG", "NET".
        current_round : int
            Current auction round number (used for history logging).

        Returns
        -------
        float
            Bid amount in tokens.
        """
        if self.tokens < 1.0:
            return 0.0

        base = BASE_VALUATION.get(resource_type, 1.0)

        # Demand urgency ~ Beta(2, 5); E[X] ≈ 0.286
        demand_urgency: float = float(self.rng.beta(2, 5))

        # Urgency modifier: scale up when near a deadline
        sla_prob = self.sla_breach_probability()
        urgency_modifier: float = 1.0 + (sla_prob * 2.0) if sla_prob > 0.0 else 1.0

        bid = min(self.tokens, demand_urgency * urgency_modifier * base)

        self.bid_history.append((current_round, resource_type, bid))
        return bid

    # ------------------------------------------------------------------
    # ERP
    # ------------------------------------------------------------------

    def erp_score(self, clearing_price: float) -> float:
        """
        Compute this node's Emergency Reallocation Protocol score.

        ERP_score = 0.4 * sla_breach_prob
                  + 0.4 * (criticality / 3)
                  - 0.2 * (emergency_count / 3)

        Clipped to [0, 1].

        Parameters
        ----------
        clearing_price : float
            Current market clearing price (used by can_declare_emergency).
        """
        score = (
            _ERP_W1 * self.sla_breach_probability()
            + _ERP_W2 * (self.criticality / 3.0)
            - _ERP_W3 * (self.emergency_count / max(1, _ERP_MONTHLY_CAP))
        )
        return float(np.clip(score, 0.0, 1.0))

    def can_declare_emergency(self, clearing_price: float) -> bool:
        """
        Return True iff this node is eligible to invoke the ERP this round.

        All three conditions must hold simultaneously (architecture.md §4):
          1. Monthly cap not exhausted (emergency_count < 3)
          2. Token balance >= 1.5 × clearing_price
          3. ERP score > θ = 0.7
        """
        return (
            self.emergency_count < _ERP_MONTHLY_CAP
            and self.tokens >= _ERP_REQUESTER_MULTIPLIER * clearing_price
            and self.erp_score(clearing_price) > _ERP_THETA
        )

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def record_win(
        self,
        resource_type: str,
        units: float,
        payment: float,
        round_num: int,
    ) -> None:
        """
        Apply the outcome of winning an allocation.

        - Deducts *payment* from token balance.
        - Resets hunger_counter to 0 (allocation received).
        - Accumulates units in total_allocated.
        - Appends entry to allocation_history.
        """
        self.tokens -= payment
        self.hunger_counter = 0
        self.total_allocated[resource_type] = (
            self.total_allocated.get(resource_type, 0.0) + units
        )
        self.allocation_history.append((round_num, resource_type, units))

    def record_loss(self, round_num: int) -> None:
        """
        Apply the outcome of losing an auction round.

        Increments hunger_counter.  No token movement.

        Parameters
        ----------
        round_num : int
            Current round (reserved for future per-round hunger logging).
        """
        self.hunger_counter += 1

    def record_preemption(self, payment_made: float) -> None:
        """
        Apply a preemption event: refund 70 % of the payment already deducted.

        Parameters
        ----------
        payment_made : float
            The payment that was deducted when the node originally won this
            slot (i.e., the VCG clearing price that was charged at win time).
        """
        self.tokens += _PREEMPTION_REFUND_RATIO * payment_made

    # ------------------------------------------------------------------
    # Period boundary
    # ------------------------------------------------------------------

    def period_reset(self) -> None:
        """
        Execute the monthly period-boundary soft rollover.

        New balance = T0 + min(0.2 * T0, current_tokens)

        All per-period counters and histories are cleared.
        """
        carry = min(_ROLLOVER_CAP_FRACTION * self.T0, self.tokens)
        self.tokens = self.T0 + carry

        self.hunger_counter  = 0
        self.emergency_count = 0
        self.total_allocated = {r: 0.0 for r in BASE_VALUATION}
        self.allocation_history.clear()
        self.bid_history.clear()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Node(id={self.node_id!r}, criticality=E{self.criticality}, "
            f"tier={self.tier}, tokens={self.tokens:.2f}, "
            f"hunger={self.hunger_counter}, erp_declarations={self.emergency_count})"
        )
