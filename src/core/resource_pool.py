"""
src/core/resource_pool.py

Manages cluster resource capacity for the Auctus auction simulation.

Market pool  = 90% of total capacity — cleared each auction round.
Reserve pool = 10% of total capacity — allocated only via SFG or ERP.
"""

from __future__ import annotations

from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Module-level constants (locked per architecture.md §2, §8)
# ---------------------------------------------------------------------------

RESOURCE_TYPES: list[str] = ["CPU", "RAM", "STG", "NET"]

DEFAULT_CAPACITY: Dict[str, float] = {
    "CPU": 100.0,    # cores
    "RAM": 512.0,    # GB
    "STG": 10240.0,  # GB
    "NET": 100.0,    # Gbps
}

RESERVE_POOL_FRACTION: float = 0.10   # 10% withheld from the open market


# ---------------------------------------------------------------------------
# ResourcePool
# ---------------------------------------------------------------------------

class ResourcePool:
    """
    Tracks cluster capacity, current allocations, and reserve pool balance
    for all resource types across a simulation period.

    Capacity split (locked):
        - market pool  : 90% of total → serviced via VCG auction
        - reserve pool : 10% of total → serviced via SFG / ERP only

    Parameters
    ----------
    capacity : dict, optional
        Total cluster capacity per resource type.  Defaults to
        DEFAULT_CAPACITY when None.
    """

    def __init__(self, capacity: Optional[Dict[str, float]] = None) -> None:
        self.capacity: Dict[str, float] = (
            dict(DEFAULT_CAPACITY) if capacity is None else dict(capacity)
        )

        # Reserve pool — held back from the open market
        self.reserved: Dict[str, float] = {
            r: RESERVE_POOL_FRACTION * c for r, c in self.capacity.items()
        }

        # Reserve pool current balance (starts full)
        self.reserve_pool_balance: Dict[str, float] = dict(self.reserved)

        # Market pool — what the auction can hand out
        self.available: Dict[str, float] = {
            r: self.capacity[r] - self.reserved[r]
            for r in self.capacity
        }

        # node_id -> {resource_type: units_allocated}
        self.allocations: Dict[str, Dict[str, float]] = {}

        # Last VCG clearing price per resource type (seeded to 0; updated by auctioneer)
        self.last_clearing_price: Dict[str, float] = {r: 0.0 for r in self.capacity}

    # ------------------------------------------------------------------
    # Capacity queries
    # ------------------------------------------------------------------

    def can_allocate(
        self,
        resource_type: str,
        units: float,
        from_reserve: bool = False,
    ) -> bool:
        """
        Return True iff *units* of *resource_type* are currently available.

        Parameters
        ----------
        resource_type : str
        units : float
        from_reserve : bool
            If True, check the reserve pool balance; otherwise check the
            market pool (available).
        """
        if resource_type not in self.capacity:
            return False
        if units <= 0:
            return False
        if from_reserve:
            return self.reserve_pool_balance.get(resource_type, 0.0) >= units
        return self.available.get(resource_type, 0.0) >= units

    # ------------------------------------------------------------------
    # Allocation / deallocation
    # ------------------------------------------------------------------

    def allocate(
        self,
        node_id: str,
        resource_type: str,
        units: float,
        from_reserve: bool = False,
    ) -> bool:
        """
        Attempt to allocate *units* of *resource_type* to *node_id*.

        Parameters
        ----------
        node_id : str
        resource_type : str
        units : float
        from_reserve : bool
            If True, draw from the reserve pool (SFG / ERP path).
            If False, draw from the market pool (auction path).

        Returns
        -------
        bool
            True on success, False if insufficient capacity.
        """
        if not self.can_allocate(resource_type, units, from_reserve=from_reserve):
            return False

        if from_reserve:
            self.reserve_pool_balance[resource_type] -= units
        else:
            self.available[resource_type] -= units

        # Record per-node allocation (additive; a node may hold several units)
        node_allocs = self.allocations.setdefault(node_id, {})
        node_allocs[resource_type] = node_allocs.get(resource_type, 0.0) + units

        return True

    def deallocate(
        self,
        node_id: str,
        resource_type: str,
        units: float,
    ) -> None:
        """
        Return *units* of *resource_type* from *node_id* back to the market
        pool.  Silently clamps to the actual held amount to avoid accounting
        drift from floating-point rounding.

        Parameters
        ----------
        node_id : str
        resource_type : str
        units : float
        """
        node_allocs = self.allocations.get(node_id, {})
        held = node_allocs.get(resource_type, 0.0)

        # Clamp: cannot return more than was allocated
        actual_return = min(units, held)
        self.available[resource_type] = (
            self.available.get(resource_type, 0.0) + actual_return
        )

        remaining = held - actual_return
        if remaining > 0:
            node_allocs[resource_type] = remaining
        else:
            node_allocs.pop(resource_type, None)

        if not node_allocs:
            self.allocations.pop(node_id, None)

    # ------------------------------------------------------------------
    # Utilization
    # ------------------------------------------------------------------

    def get_utilization(self) -> Dict[str, float]:
        """
        Return market-pool utilization percentage per resource type.

        Formula:
            allocated_market = market_capacity - available
            utilization %    = allocated_market / market_capacity × 100

        where market_capacity = capacity - reserved (i.e. 90% of total).
        The reserve pool is excluded from this metric; it is tracked
        separately via reserve_pool_balance.

        Returns
        -------
        dict
            {resource_type: utilization_percent}  values in [0, 100].
        """
        result: Dict[str, float] = {}
        for r in self.capacity:
            market_cap = self.capacity[r] - self.reserved[r]   # 90%
            if market_cap <= 0:
                result[r] = 0.0
                continue
            allocated = market_cap - self.available[r]
            result[r] = max(0.0, min(100.0, (allocated / market_cap) * 100.0))
        return result

    # ------------------------------------------------------------------
    # Reserve pool management
    # ------------------------------------------------------------------

    def update_reserve_pool(self, delta: Dict[str, float]) -> None:
        """
        Credit *delta* units into the reserve pool balance (e.g. the 30%
        ERP surplus).  Balance is capped at the pool's maximum (reserved).

        Parameters
        ----------
        delta : dict
            {resource_type: units_to_add}
        """
        for r, amount in delta.items():
            if r not in self.reserve_pool_balance:
                continue
            new_balance = self.reserve_pool_balance[r] + amount
            self.reserve_pool_balance[r] = min(new_balance, self.reserved[r])

    # ------------------------------------------------------------------
    # Preemption support
    # ------------------------------------------------------------------

    def get_current_holder(self, resource_type: str) -> Optional[str]:
        """
        Return the node_id of the current holder of *resource_type* with the
        lowest allocation for that resource (proxy for lowest priority).

        In the single-unit-per-round VCG model there will be at most one
        holder per resource type at any time.  For the multi-unit case this
        returns the node holding the smallest quantity — the most appropriate
        candidate for preemption.

        Returns None if no node currently holds *resource_type*.
        """
        candidates = {
            node_id: allocs[resource_type]
            for node_id, allocs in self.allocations.items()
            if resource_type in allocs and allocs[resource_type] > 0
        }
        if not candidates:
            return None
        # Node with the smallest held quantity — lowest claim on the resource
        return min(candidates, key=candidates.__getitem__)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        parts = ", ".join(
            f"{r}: {self.available.get(r, 0):.1f}/{self.capacity.get(r, 0):.1f}"
            for r in RESOURCE_TYPES
            if r in self.capacity
        )
        return f"ResourcePool(market_available=[{parts}])"
