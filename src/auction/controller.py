"""
src/auction/controller.py

AuctionController: SimPy discrete-event coordinator for the Auctus protocol.

Runs one auction round every ROUND_INTERVAL sim-minutes. Per round, each
resource type is cleared independently via VCG second-price. Token
deduction, resource allocation, SFG starvation checks, and audit logging
are all orchestrated here.

TOKEN OWNERSHIP CONTRACT (must not be violated):
    TokenManager.deduct() is called FIRST.
    node.record_win() is ALWAYS called with payment=0.0 when TokenManager
    is in use. Double-deduction is a protocol violation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import simpy

from src.core.node          import Node
from src.core.resource_pool import ResourcePool, RESOURCE_TYPES
from src.core.token_manager import TokenManager
from src.core.audit_log     import (
    AuditLog,
    AUCTION_WIN,
    AUCTION_LOSS,
    SFG_ALLOCATE,
    PERIOD_RESET,
)
from src.auction.vcg import VCGAuction, AuctionResult


# ---------------------------------------------------------------------------
# Module-level constants (locked per architecture.md section 8)
# ---------------------------------------------------------------------------

ROUND_INTERVAL: int = 5              # sim-minutes per auction round
TASK_DURATION_MEAN: float = 30.0     # mean task duration (exponential), sim-minutes

RESOURCE_UNITS: Dict[str, float] = {
    "CPU": 4.0,
    "RAM": 16.0,
    "STG": 100.0,
    "NET": 1.0,
}

SFG_HUNGER_THRESHOLD: int = 10       # W = 10 consecutive empty rounds
ROUNDS_PER_PERIOD: int = 8640        # 30 days * 288 rounds/day


# ---------------------------------------------------------------------------
# AuctionController
# ---------------------------------------------------------------------------

class AuctionController:
    """
    Central coordinator for a single Auctus simulation run.

    Parameters
    ----------
    env : simpy.Environment
    nodes : list[Node]
    resource_pool : ResourcePool
    token_manager : TokenManager
    audit_log : AuditLog
    rng : numpy.random.Generator, optional
        Seeded RNG for task-duration sampling.
    """

    def __init__(
        self,
        env: simpy.Environment,
        nodes: List[Node],
        resource_pool: ResourcePool,
        token_manager: TokenManager,
        audit_log: AuditLog,
        rng=None,
    ) -> None:
        self.env:           simpy.Environment = env
        self.nodes:         List[Node]        = nodes
        self.resource_pool: ResourcePool      = resource_pool
        self.token_manager: TokenManager      = token_manager
        self.audit_log:     AuditLog          = audit_log
        self.vcg:           VCGAuction        = VCGAuction()
        self.round_num:     int               = 0

        self.stats: Dict = {
            "total_allocations": 0,
            "total_rounds":      0,
            "erp_events":        0,
            "sfg_events":        0,
            "failed_deductions": 0,
        }

        self._node_map: Dict[str, Node] = {n.node_id: n for n in nodes}

        if rng is None:
            import numpy as np
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

    # ------------------------------------------------------------------
    # Main round
    # ------------------------------------------------------------------

    def run_round(self) -> None:
        """Execute one complete auction round across all resource types."""
        self.round_num += 1
        rn = self.round_num

        for resource_type in RESOURCE_TYPES:
            if resource_type not in RESOURCE_UNITS:
                continue

            units = RESOURCE_UNITS[resource_type]

            # Step 1: Collect bids (skip zero bids)
            bids: Dict[str, float] = {}
            for node in self.nodes:
                bid = node.generate_bid(resource_type, rn)
                if bid > 0:
                    bids[node.node_id] = bid

            # Step 2: VCG clearing
            result: AuctionResult = self.vcg.clear(resource_type, bids, rn)

            if result.payment > 0:
                self.resource_pool.last_clearing_price[resource_type] = result.payment

            # Step 3: Apply outcome
            allocated = False

            if result.winner_id is not None:
                winner_id   = result.winner_id
                winner_node = self._node_map[winner_id]

                # Deduct payment via TokenManager
                deducted = self.token_manager.deduct(
                    winner_id, result.payment, reason="auction_win"
                )

                if deducted:
                    alloc_ok = self.resource_pool.allocate(
                        winner_id, resource_type, units
                    )
                    if alloc_ok:
                        # payment=0.0: TokenManager already deducted
                        winner_node.record_win(
                            resource_type, units, payment=0.0, round_num=rn
                        )
                        self.audit_log.append(
                            AUCTION_WIN,
                            winner_id,
                            {
                                "resource": resource_type,
                                "payment":  result.payment,
                                "bid":      result.winning_bid,
                                "round":    rn,
                                "units":    units,
                            },
                        )
                        self.stats["total_allocations"] += 1
                        allocated = True

                        duration = float(self._rng.exponential(TASK_DURATION_MEAN))
                        self.env.process(
                            self.schedule_deallocation(
                                winner_id, resource_type, units, duration
                            )
                        )
                    else:
                        # Pool full: refund and treat as loss
                        self.token_manager.refund(
                            winner_id, result.payment, reason="pool_full_refund"
                        )
                        deducted = False

                if not deducted:
                    winner_node.record_loss(rn)
                    self.stats["failed_deductions"] += 1
                    self.audit_log.append(
                        AUCTION_LOSS,
                        winner_id,
                        {
                            "resource": resource_type,
                            "reason":   "insufficient_tokens_or_pool_full",
                            "round":    rn,
                        },
                    )

            # Step 4: Record losses for bidding non-winners only
            winner_id_safe = result.winner_id if allocated else None
            for node in self.nodes:
                if node.node_id == winner_id_safe:
                    continue
                if node.node_id in bids:
                    node.record_loss(rn)
                    self.audit_log.append(
                        AUCTION_LOSS,
                        node.node_id,
                        {"resource": resource_type, "round": rn},
                    )

            # Step 5: SFG starvation check
            self._check_starvation(resource_type)

        self.stats["total_rounds"] += 1

        # Period boundary
        if self.round_num % ROUNDS_PER_PERIOD == 0:
            self.token_manager.period_reset()
            self.audit_log.append(
                PERIOD_RESET,
                "SYSTEM",
                {"period": self.round_num // ROUNDS_PER_PERIOD, "round": rn},
            )

    # ------------------------------------------------------------------
    # Deallocation process
    # ------------------------------------------------------------------

    def schedule_deallocation(
        self,
        node_id: str,
        resource_type: str,
        units: float,
        duration: float,
    ):
        """SimPy generator: wait duration sim-minutes then deallocate."""
        yield self.env.timeout(duration)
        self.resource_pool.deallocate(node_id, resource_type, units)

    # ------------------------------------------------------------------
    # Starvation floor
    # ------------------------------------------------------------------

    def _check_starvation(self, resource_type: str) -> None:
        """Apply SFG for nodes whose hunger_counter >= W."""
        for node in self.nodes:
            if node.hunger_counter < SFG_HUNGER_THRESHOLD:
                continue

            sfg_price = self.resource_pool.last_clearing_price.get(resource_type, 0.0)
            units     = RESOURCE_UNITS.get(resource_type, 1.0)

            if not self.resource_pool.can_allocate(resource_type, units, from_reserve=True):
                continue
            if not self.token_manager.deduct(node.node_id, sfg_price, reason="sfg_allocation"):
                continue

            alloc_ok = self.resource_pool.allocate(
                node.node_id, resource_type, units, from_reserve=True
            )
            if alloc_ok:
                node.record_win(resource_type, units, payment=0.0, round_num=self.round_num)
                self.audit_log.append(
                    SFG_ALLOCATE,
                    node.node_id,
                    {
                        "resource":  resource_type,
                        "sfg_price": sfg_price,
                        "hunger":    node.hunger_counter,
                        "round":     self.round_num,
                    },
                )
                self.stats["sfg_events"] += 1
                duration = float(self._rng.exponential(TASK_DURATION_MEAN))
                self.env.process(
                    self.schedule_deallocation(
                        node.node_id, resource_type, units, duration
                    )
                )
            else:
                self.token_manager.refund(
                    node.node_id, sfg_price, reason="sfg_allocation_failed_refund"
                )

    # ------------------------------------------------------------------
    # Simulation entry point
    # ------------------------------------------------------------------

    def start(self, sim_duration_minutes: float) -> None:
        """Register the auction loop and run the SimPy environment."""
        self.env.process(self._auction_loop(sim_duration_minutes))
        self.env.run(until=sim_duration_minutes)

    def _auction_loop(self, sim_duration_minutes: float):
        """SimPy generator: fire one round every ROUND_INTERVAL minutes."""
        while True:
            yield self.env.timeout(ROUND_INTERVAL)
            if self.env.now > sim_duration_minutes:
                break
            self.run_round()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return running statistics plus per-node state snapshot."""
        per_node = {
            node.node_id: {
                "tokens":           node.tokens,
                "hunger_counter":   node.hunger_counter,
                "emergency_count":  node.emergency_count,
                "total_allocated":  dict(node.total_allocated),
                "allocation_count": len(node.allocation_history),
            }
            for node in self.nodes
        }
        return {**self.stats, "nodes": per_node}

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AuctionController(round={self.round_num}, "
            f"nodes={len(self.nodes)}, "
            f"allocations={self.stats['total_allocations']})"
        )
