"""
src/auction/vcg.py

VCG (Vickrey-Clarke-Groves) second-price sealed-bid clearing mechanism.

Design contract (architecture.md §1, problem_formulation.md §3):
  - Single resource, single unit, per round.
  - Winner  = argmax bid  (ties broken by lexicographically smallest node_id).
  - Payment = second-highest bid  (Vickrey price).
  - Single bidder pays 0.0  (no competing bid exists).
  - DSIC: truthful bidding is weakly dominant because the payment never
    depends on the winner's own bid.

THIS MODULE IS PURE COMPUTATION.
VCGAuction.clear() does not touch tokens, nodes, or the resource pool.
All state mutations are the exclusive responsibility of TokenManager and
ResourcePool, which the auctioneer layer calls after receiving AuctionResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Default clearing prices (architecture.md §8 — reference base valuations)
# ---------------------------------------------------------------------------

_DEFAULT_CLEARING_PRICES: Dict[str, float] = {
    "CPU": 5.0,
    "RAM": 1.0,
    "STG": 0.5,
    "NET": 2.0,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AuctionResult:
    """
    Immutable record of a single VCG clearing outcome.

    Attributes
    ----------
    resource_type : str
        The resource type cleared in this round.
    winner_id : str or None
        Node ID of the winning bidder, or None if no valid bids were received.
    payment : float
        VCG payment (second-highest bid).  0.0 when there is only one bidder
        or no bids at all.
    winning_bid : float
        The highest submitted bid.  0.0 when winner_id is None.
    all_bids : dict
        Full snapshot of ``{node_id: bid_amount}`` submitted this round,
        including losers.  Stored for audit-log payload.
    round_num : int
        Auction round number in which this result was produced.
    """
    resource_type: str
    winner_id:     Optional[str]
    payment:       float
    winning_bid:   float
    all_bids:      Dict[str, float]
    round_num:     int


# ---------------------------------------------------------------------------
# VCGAuction
# ---------------------------------------------------------------------------

class VCGAuction:
    """
    Stateful VCG clearing engine.

    The only mutable state is ``last_clearing_price``, which is updated each
    round whenever the computed VCG payment is strictly positive.  It is used
    by the SFG mechanism to price guaranteed allocations at the most recent
    market rate.

    Parameters
    ----------
    initial_clearing_prices : dict, optional
        Seed values for ``last_clearing_price``.  Defaults to
        ``_DEFAULT_CLEARING_PRICES``.
    """

    def __init__(
        self,
        initial_clearing_prices: Optional[Dict[str, float]] = None,
    ) -> None:
        self.last_clearing_price: Dict[str, float] = (
            dict(_DEFAULT_CLEARING_PRICES)
            if initial_clearing_prices is None
            else dict(initial_clearing_prices)
        )

    # ------------------------------------------------------------------
    # Clearing
    # ------------------------------------------------------------------

    def clear(
        self,
        resource_type: str,
        bids: Dict[str, float],
        round_num: int,
    ) -> AuctionResult:
        """
        Run VCG second-price clearing for *resource_type* in *round_num*.

        Parameters
        ----------
        resource_type : str
            Resource being auctioned (e.g. "CPU", "RAM").
        bids : dict
            ``{node_id: bid_amount}`` — caller must pre-filter to only include
            bids where ``bid_amount > 0``.  Passing zero-bids is not an error
            but those nodes cannot win.
        round_num : int
            Current simulation round number, stored verbatim in AuctionResult.

        Returns
        -------
        AuctionResult
            Full clearing outcome.  ``winner_id`` is None for empty bid sets.

        Notes
        -----
        Tie-breaking rule: when two or more nodes submit the identical highest
        bid, the winner is the lexicographically *smallest* node_id string.
        This rule is deterministic, side-effect-free, and documented here so
        it can be cited in the paper's implementation section.

        ``last_clearing_price`` is updated only when ``payment > 0``, i.e.
        only in a competitive round with at least two valid bidders.
        """
        all_bids_snapshot = dict(bids)   # immutable copy for audit log

        # -- Edge case: no bids
        if not bids:
            return AuctionResult(
                resource_type=resource_type,
                winner_id=None,
                payment=0.0,
                winning_bid=0.0,
                all_bids=all_bids_snapshot,
                round_num=round_num,
            )

        # Sort descending by bid, with lexicographic node_id as tie-breaker
        # (ascending node_id = smaller string wins ties).
        ranked: list[tuple[float, str]] = sorted(
            ((-bid, node_id) for node_id, bid in bids.items()),
            key=lambda x: (x[0], x[1]),   # (-bid asc, node_id asc)
        )

        # Winner: first entry after sort (highest bid, lowest node_id on tie)
        _, winner_id = ranked[0]
        winning_bid  = bids[winner_id]

        # -- Edge case: single bidder
        if len(ranked) == 1:
            return AuctionResult(
                resource_type=resource_type,
                winner_id=winner_id,
                payment=0.0,
                winning_bid=winning_bid,
                all_bids=all_bids_snapshot,
                round_num=round_num,
            )

        # Normal case: VCG payment = second-highest bid
        _, second_node_id = ranked[1]
        payment = bids[second_node_id]

        # Update clearing price only when the round is competitive
        if payment > 0:
            self.last_clearing_price[resource_type] = payment

        return AuctionResult(
            resource_type=resource_type,
            winner_id=winner_id,
            payment=payment,
            winning_bid=winning_bid,
            all_bids=all_bids_snapshot,
            round_num=round_num,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_last_clearing_price(self, resource_type: str) -> float:
        """
        Return the most recent VCG clearing price for *resource_type*.

        Falls back to the default seed price if no competitive round has
        occurred yet for this resource.

        Parameters
        ----------
        resource_type : str

        Returns
        -------
        float
        """
        return self.last_clearing_price.get(
            resource_type,
            _DEFAULT_CLEARING_PRICES.get(resource_type, 0.0),
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        prices = ", ".join(
            f"{r}={p:.2f}" for r, p in self.last_clearing_price.items()
        )
        return f"VCGAuction(clearing_prices=[{prices}])"
