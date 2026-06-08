"""
src/auction/emergency.py

Emergency Reallocation Protocol (ERP) manager for the Auctus simulation.

Token accounting per ERP event (closed):
  Requester pays : -1.5 x clearing_price
  Preempted gets : +0.70 x clearing_price
  Reserve credit : +0.30 x clearing_price  (token-denominated)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.core.node          import Node
from src.core.resource_pool import ResourcePool
from src.core.token_manager import TokenManager
from src.core.audit_log     import AuditLog, ERP_DECLARE, ERP_PREEMPT, RESERVE_TOPUP
from src.auction.vcg        import VCGAuction

RESOURCE_UNITS: Dict[str, float] = {
    "CPU": 4.0,
    "RAM": 16.0,
    "STG": 100.0,
    "NET": 1.0,
}

_ERP_MULTIPLIER:   float = 1.5
_PREEMPTED_REFUND: float = 0.70
_RESERVE_CREDIT:   float = 0.30
_REFERENCE_RESOURCE: str = "CPU"


class ERPManager:
    """
    Evaluates and executes Emergency Reallocation Protocol events.

    Parameters
    ----------
    nodes : dict
        {node_id: Node}
    resource_pool : ResourcePool
    token_manager : TokenManager
    audit_log : AuditLog
    vcg : VCGAuction
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
        self.erp_events_this_period: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Round-level entry point
    # ------------------------------------------------------------------

    def process_emergency_requests(self, round_num: int) -> List[str]:
        """
        Evaluate all nodes for ERP eligibility; execute at most one per round.

        Returns list of node_ids that successfully triggered ERP (at most one).
        """
        clearing_price = self.vcg.get_last_clearing_price(_REFERENCE_RESOURCE)
        triggered: List[str] = []

        candidates: List[tuple] = []
        for node in self.nodes.values():
            if node.can_declare_emergency(clearing_price):
                score = node.erp_score(clearing_price)
                candidates.append((score, node))

        if not candidates:
            return triggered

        # Sort descending by score; ascending node_id for deterministic tie-break
        candidates.sort(key=lambda x: (-x[0], x[1].node_id))
        _, best_node = candidates[0]

        for resource_type in RESOURCE_UNITS:
            if self.execute_erp(best_node, resource_type, round_num):
                triggered.append(best_node.node_id)
                break

        return triggered

    # ------------------------------------------------------------------
    # ERP execution
    # ------------------------------------------------------------------

    def execute_erp(
        self,
        requester: Node,
        resource_type: str,
        round_num: int,
    ) -> bool:
        """
        Execute a full ERP event: preempt the current holder of
        *resource_type* and reallocate it to *requester*.

        Returns True on success, False if:
          - No current holder exists.
          - Requester has insufficient tokens (< 1.5 x clearing price).
          - Requester is the current holder (no self-preemption).
        """
        # Step 1: identify preemption target
        target_id = self.resource_pool.get_current_holder(resource_type)
        if target_id is None:
            return False
        if target_id == requester.node_id:
            return False

        target_node = self.nodes.get(target_id)
        if target_node is None:
            return False

        # Step 2: verify requester can afford ERP price
        clearing_price = self.vcg.get_last_clearing_price(resource_type)
        erp_price      = _ERP_MULTIPLIER * clearing_price

        if requester.tokens < erp_price:
            return False

        # Step 3: token flows
        self.token_manager.deduct(
            requester.node_id, erp_price, reason="erp_emergency"
        )
        preempted_refund = _PREEMPTED_REFUND * clearing_price
        self.token_manager.refund(
            target_node.node_id, preempted_refund, reason="erp_preemption_refund"
        )
        surplus = _RESERVE_CREDIT * clearing_price
        self.resource_pool.reserve_token_balance += surplus

        # Step 4: reallocate physical resource
        units = RESOURCE_UNITS[resource_type]
        self.resource_pool.deallocate(target_id, resource_type, units)
        self.resource_pool.allocate(requester.node_id, resource_type, units)

        # Step 5: update node state
        # Note: target's token refund is already handled by token_manager.refund()
        # above. Do NOT call target_node.record_preemption() here — that method
        # adds tokens directly to node.tokens and would double-count the 70% credit.
        requester.record_win(
            resource_type, units, payment=0.0, round_num=round_num
        )
        requester.emergency_count += 1
        self.erp_events_this_period[requester.node_id] = (
            self.erp_events_this_period.get(requester.node_id, 0) + 1
        )

        # Step 6: audit log
        self.audit_log.append(
            ERP_DECLARE,
            requester.node_id,
            {
                "target":    target_id,
                "resource":  resource_type,
                "erp_price": erp_price,
                "clearing":  clearing_price,
                "round":     round_num,
            },
        )
        self.audit_log.append(
            ERP_PREEMPT,
            target_id,
            {
                "requester": requester.node_id,
                "refund":    preempted_refund,
                "resource":  resource_type,
                "round":     round_num,
            },
        )
        self.audit_log.append(
            RESERVE_TOPUP,
            "SYSTEM",
            {
                "amount":   surplus,
                "source":   "ERP",
                "resource": resource_type,
                "round":    round_num,
            },
        )

        return True

    # ------------------------------------------------------------------
    # Period boundary
    # ------------------------------------------------------------------

    def period_reset(self) -> None:
        """Reset per-period ERP event counters at the monthly boundary."""
        self.erp_events_this_period = {}

    def __repr__(self) -> str:  # pragma: no cover
        total = sum(self.erp_events_this_period.values())
        return f"ERPManager(nodes={len(self.nodes)}, erp_this_period={total})"
