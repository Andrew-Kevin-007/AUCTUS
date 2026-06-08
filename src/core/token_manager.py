"""
src/core/token_manager.py

Manages the token economy for all nodes in the Auctus simulation.

All token movements go through this class so that the audit log receives a
single consistent call site per transaction type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from src.core.node import Node

# ERP token-split ratios (locked per architecture.md §4.3)
_ERP_REQUESTER_MULTIPLIER: float = 1.5
_ERP_PREEMPTED_REFUND_RATIO: float = 0.70
_ERP_RESERVE_CREDIT_RATIO: float = 0.30


class TokenManager:
    """
    Central authority for all token movements in the Auctus simulation.

    Enforces the token accounting rules from architecture.md §2 and §4.3:
      - Deduction on auction win
      - Refund on preemption (70%)
      - ERP three-way split (requester −1.5×, reserve +0.30×)
      - Soft rollover at period boundary

    All movements are appended to ``transaction_log`` for post-hoc analytics.
    For cryptographic audit integrity, callers should additionally pass events
    to an :class:`~src.core.audit_log.AuditLog` instance.

    Parameters
    ----------
    T0 : float
        Monthly token budget per node.  Defaults to 1000.
    """

    def __init__(self, T0: float = 1000.0) -> None:
        self.T0: float = T0
        self.period_number: int = 0
        self.nodes: Dict[str, "Node"] = {}
        self.transaction_log: List[dict] = []

    # ------------------------------------------------------------------
    # Node registry
    # ------------------------------------------------------------------

    def register_node(self, node: "Node") -> None:
        """
        Register a node with this token manager.

        The node's T0 is overridden to match the manager's T0 so that the
        entire fleet operates on a consistent budget.

        Parameters
        ----------
        node : Node
        """
        node.T0 = self.T0
        node.tokens = self.T0
        self.nodes[node.node_id] = node

    # ------------------------------------------------------------------
    # Core token movements
    # ------------------------------------------------------------------

    def deduct(
        self,
        node_id: str,
        amount: float,
        reason: str = "auction_win",
    ) -> bool:
        """
        Deduct *amount* tokens from *node_id*.

        Returns False (no-op) if the node does not have sufficient balance.

        Parameters
        ----------
        node_id : str
        amount : float
        reason : str
            Label stored in the transaction log.

        Returns
        -------
        bool
            True on success, False if insufficient tokens.
        """
        node = self.nodes.get(node_id)
        if node is None:
            return False
        if node.tokens < amount:
            return False

        node.tokens -= amount
        self._log(node_id, "DEDUCT", amount, reason, balance_after=node.tokens)
        return True

    def refund(
        self,
        node_id: str,
        amount: float,
        reason: str = "preemption_refund",
    ) -> None:
        """
        Add *amount* tokens back to *node_id*.

        There is intentionally no upper-bound check here: refunds restore
        tokens that were previously deducted and must not be silently dropped.
        The soft rollover cap is enforced only at period boundary, not mid-period.

        Parameters
        ----------
        node_id : str
        amount : float
        reason : str
        """
        node = self.nodes.get(node_id)
        if node is None:
            return

        node.tokens += amount
        self._log(node_id, "REFUND", amount, reason, balance_after=node.tokens)

    # ------------------------------------------------------------------
    # ERP token accounting
    # ------------------------------------------------------------------

    def add_to_reserve_for_emergency(
        self,
        requester_id: str,
        clearing_price: float,
    ) -> float:
        """
        Execute the ERP token split for an emergency reallocation event.

        Token flows (architecture.md §4.3):
          - Requester pays:  1.5 × clearing_price
          - Reserve pool receives: 0.30 × clearing_price  (returned as float)
          - Preempted node refund: 0.70 × clearing_price  (caller's responsibility)

        The preempted-node refund is **not** applied here because the caller
        (the auctioneer / ERP handler) knows which specific node was preempted.
        The caller must call :meth:`refund` with the preempted node's id and
        ``0.70 * clearing_price``.

        Parameters
        ----------
        requester_id : str
            Node that declared the emergency.
        clearing_price : float
            Current market clearing price for the contested resource.

        Returns
        -------
        float
            The reserve-pool credit (0.30 × clearing_price) that the caller
            must pass to ``ResourcePool.update_reserve_pool()``.

        Raises
        ------
        ValueError
            If the requester does not have sufficient tokens (caller should
            gate on ``can_declare_emergency`` first).
        """
        cost = _ERP_REQUESTER_MULTIPLIER * clearing_price
        node = self.nodes.get(requester_id)
        if node is None or node.tokens < cost:
            raise ValueError(
                f"ERP failed: node '{requester_id}' has insufficient tokens "
                f"({getattr(node, 'tokens', 0):.2f} < {cost:.2f})"
            )

        node.tokens -= cost
        self._log(
            requester_id, "DEDUCT", cost, "erp_requester_payment",
            balance_after=node.tokens,
            meta={"clearing_price": clearing_price},
        )

        reserve_credit = _ERP_RESERVE_CREDIT_RATIO * clearing_price
        self._log(
            requester_id, "RESERVE_CREDIT", reserve_credit, "erp_reserve_topup",
            balance_after=node.tokens,
            meta={"clearing_price": clearing_price},
        )

        # Track emergency declaration
        node.emergency_count += 1

        return reserve_credit

    # ------------------------------------------------------------------
    # Period boundary
    # ------------------------------------------------------------------

    def period_reset(self) -> None:
        """
        Execute the monthly period-boundary soft rollover for all nodes.

        Delegates to each node's own ``period_reset()`` method, which applies
        the carry-forward cap and clears per-period counters.  After all nodes
        are reset, increments ``period_number`` and logs the event.
        """
        for node in self.nodes.values():
            node.period_reset()

        self.period_number += 1
        self.transaction_log.append({
            "event":         "PERIOD_RESET",
            "period_number": self.period_number,
            "node_count":    len(self.nodes),
        })

    # ------------------------------------------------------------------
    # Analytics helpers
    # ------------------------------------------------------------------

    def get_token_distribution(self) -> Dict[str, float]:
        """
        Return the current token balance for every registered node.

        Used by the fairness / metrics layer to compute Jain's index and
        plot token-balance histograms.

        Returns
        -------
        dict
            {node_id: tokens}
        """
        return {nid: node.tokens for nid, node in self.nodes.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(
        self,
        node_id: str,
        direction: str,
        amount: float,
        reason: str,
        balance_after: float,
        meta: Optional[dict] = None,
    ) -> None:
        entry: dict = {
            "node_id":       node_id,
            "direction":     direction,
            "amount":        amount,
            "reason":        reason,
            "balance_after": balance_after,
            "period":        self.period_number,
        }
        if meta:
            entry["meta"] = meta
        self.transaction_log.append(entry)
