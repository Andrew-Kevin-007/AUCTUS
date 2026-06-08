"""
src/auction/starvation_floor.py

Starvation Floor Guarantee (SFG) manager for the Auctus simulation.

SFG ensures no bidding node goes more than W=10 consecutive rounds without
any allocation.  Triggered allocations are drawn from the reserve pool
(10% of total capacity, held back from the open market) and priced at the
most recent VCG clearing price for the resource type.

Design contract (architecture.md section 3, problem_formulation.md section 4):
  - Trigger  : hunger_counter >= W
  - Price    : last_clearing_price(resource_type), floor 1.0 if no prior clear
  - Source   : reserve pool only — no preemption of existing holders
  - Deduction: TokenManager.deduct() is called; node.record_win() uses payment=0.0
"""

from __future__ import annotations

from typing import Dict, List

from src.core.node          import Node
from src.core.resource_pool import ResourcePool
from src.core.token_manager import TokenManager
from src.core.audit_log     import AuditLog, SFG_ALLOCATE
from src.auction.vcg        import VCGAuction


# ---------------------------------------------------------------------------
# Constants (locked per architecture.md section 3 & 8)
# ---------------------------------------------------------------------------

STARVATION_WINDOW: int = 10    # W: consecutive empty rounds before SFG fires

RESOURCE_UNITS: Dict[str, float] = {
    "CPU": 4.0,
    "RAM": 16.0,
    "STG": 100.0,
    "NET": 1.0,
}

_SFG_PRICE_FLOOR: float = 1.0  # minimum price when market has never cleared


class SFGManager:
    """
    Evaluates and applies Starvation Floor Guarantee allocations.

    Called once per round by the AuctionController after normal VCG clearing.
    Each node whose ``hunger_counter >= STARVATION_WINDOW`` receives one
    guaranteed CPU slot from the reserve pool at the last clearing price.

    Parameters
    ----------
    nodes : dict
        {node_id: Node} — full fleet registry.
    resource_pool : ResourcePool
    token_manager : TokenManager
    audit_log : AuditLog
    vcg : VCGAuction
        Shared VCG instance for clearing-price lookups.
    """

    def __init__(
        self,
        nodes: Dict[str, Node],
        resource_pool: ResourcePool,
        token_manager: TokenManager,
        audit_log: AuditLog,
        vcg: VCGAuction,
    ) -> None:
        self.nodes:          Dict[str, Node] = nodes
        self.resource_pool:  ResourcePool    = resource_pool
        self.token_manager:  TokenManager    = token_manager
        self.audit_log:      AuditLog        = audit_log
        self.vcg:            VCGAuction      = vcg
        self.sfg_events_total: int           = 0

    # ------------------------------------------------------------------
    # Round-level entry point
    # ------------------------------------------------------------------

    def check_and_apply(self, round_num: int) -> None:
        """
        Scan all nodes for SFG eligibility and apply allocations.

        Called once per round, after normal auction clearing for all
        resource types has completed.  Primary resource for SFG is CPU;
        a node is relieved as soon as it receives one successful SFG
        allocation (hunger_counter resets inside record_win).

        Parameters
        ----------
        round_num : int
            Current simulation round number.
        """
        for node in self.nodes.values():
            if node.hunger_counter < STARVATION_WINDOW:
                continue

            # Primary resource: CPU.  Only attempt CPU SFG for now;
            # multi-resource SFG is future work (architecture.md scope table).
            resource_type = "CPU"
            if not self.resource_pool.can_allocate(
                resource_type,
                RESOURCE_UNITS[resource_type],
                from_reserve=True,
            ):
                # Reserve pool exhausted — skip this node this round
                continue

            self.execute_sfg_allocation(node, resource_type, round_num)

    # ------------------------------------------------------------------
    # SFG allocation execution
    # ------------------------------------------------------------------

    def execute_sfg_allocation(
        self,
        node: Node,
        resource_type: str,
        round_num: int,
    ) -> bool:
        """
        Execute one SFG allocation for *node* on *resource_type*.

        Token deduction is handled by TokenManager (maintains the single
        source of truth for all token movements).  node.record_win() is
        called with ``payment=0.0`` to avoid double-deduction.

        Parameters
        ----------
        node : Node
        resource_type : str
        round_num : int

        Returns
        -------
        bool
            True on success.  False if:
            - Node has insufficient tokens to cover the SFG price.
            - Reserve pool does not have capacity for this resource.
        """
        # Step 1: SFG price = last clearing price, with a floor of 1.0
        sfg_price = self.vcg.get_last_clearing_price(resource_type)
        if sfg_price <= 0.0:
            sfg_price = _SFG_PRICE_FLOOR

        # Step 2: node must be able to afford the SFG price
        if node.tokens < sfg_price:
            return False

        # Step 3: reserve pool must have physical capacity
        units = RESOURCE_UNITS[resource_type]
        if not self.resource_pool.can_allocate(resource_type, units, from_reserve=True):
            return False

        # Step 4: execute allocation
        # 4a. Deduct via TokenManager (single token authority)
        deducted = self.token_manager.deduct(
            node.node_id, sfg_price, reason="sfg_allocation"
        )
        if not deducted:
            # TokenManager rejected (safety net — should not occur if step 2 passed)
            return False

        # 4b. Allocate from reserve pool
        alloc_ok = self.resource_pool.allocate(
            node.node_id, resource_type, units, from_reserve=True
        )
        if not alloc_ok:
            # Reserve pool race condition (safety net) — refund and abort
            self.token_manager.refund(
                node.node_id, sfg_price, reason="sfg_reserve_race_refund"
            )
            return False

        # 4c. Update node state — payment=0.0 (TokenManager already deducted)
        #     record_win also resets hunger_counter to 0
        hunger_was = node.hunger_counter
        node.record_win(resource_type, units, payment=0.0, round_num=round_num)

        # Step 5: audit log — capture hunger value before the reset
        self.audit_log.append(
            SFG_ALLOCATE,
            node.node_id,
            {
                "resource":           resource_type,
                "sfg_price":          sfg_price,
                "hunger_counter_was": hunger_was,
                "round":              round_num,
                "units":              units,
            },
        )

        # Step 6: increment event counter
        self.sfg_events_total += 1

        return True

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_starvation_stats(self) -> Dict:
        """
        Return a snapshot of current starvation metrics.

        Returns
        -------
        dict
            - ``sfg_events_total`` : cumulative SFG allocations this simulation
            - ``currently_starving`` : list of node_ids at or above threshold
            - ``max_hunger`` : highest hunger_counter in the fleet (0 if no nodes)
        """
        node_list = list(self.nodes.values())
        return {
            "sfg_events_total": self.sfg_events_total,
            "currently_starving": [
                n.node_id
                for n in node_list
                if n.hunger_counter >= STARVATION_WINDOW
            ],
            "max_hunger": max(
                (n.hunger_counter for n in node_list), default=0
            ),
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SFGManager(nodes={len(self.nodes)}, "
            f"sfg_events_total={self.sfg_events_total})"
        )
