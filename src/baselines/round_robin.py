"""
src/baselines/round_robin.py

Round-Robin allocation baseline.

Each round, resources cycle through nodes in a fixed rotation — the next
node in the queue gets the slot regardless of demand or priority.  Nodes
with no demand are skipped and the pointer advances.

Used in experiments/run_experiment.py as a fairness-by-construction baseline.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.core.node          import Node
from src.core.resource_pool import ResourcePool, RESOURCE_TYPES


RESOURCE_UNITS: Dict[str, float] = {
    "CPU": 4.0,
    "RAM": 16.0,
    "STG": 100.0,
    "NET": 1.0,
}


class RoundRobinAllocator:
    """
    Round-robin allocation baseline.

    Maintains one pointer per resource type.  Each round the pointer
    advances to the next node that has positive demand and available
    capacity.  If the full cycle finds no eligible node, the resource
    goes unallocated.

    Parameters
    ----------
    nodes : list[Node]
    resource_pool : ResourcePool
    """

    def __init__(self, nodes: List[Node], resource_pool: ResourcePool) -> None:
        self.nodes:          List[Node]   = nodes
        self.resource_pool:  ResourcePool = resource_pool
        # Per-resource rotation pointer (index into self.nodes)
        self._pointer: Dict[str, int] = {r: 0 for r in RESOURCE_TYPES}
        self.round_num:      int = 0
        self.total_allocations: int = 0

    def run_round(self) -> Dict[str, Optional[str]]:
        """Execute one round-robin round across all resource types."""
        self.round_num += 1
        rn      = self.round_num
        winners: Dict[str, Optional[str]] = {}
        n       = len(self.nodes)

        if n == 0:
            return winners

        for resource_type in RESOURCE_TYPES:
            units = RESOURCE_UNITS.get(resource_type)
            if units is None:
                continue

            winner_id  = None
            start      = self._pointer[resource_type]

            # Collect bids first (so bid_history is populated for all nodes)
            bids: Dict[str, float] = {}
            for node in self.nodes:
                bid = node.generate_bid(resource_type, rn)
                if bid > 0:
                    bids[node.node_id] = bid

            # Advance through the rotation from current pointer
            for offset in range(n):
                idx  = (start + offset) % n
                node = self.nodes[idx]

                if node.node_id not in bids:
                    continue   # no demand this round

                if self.resource_pool.can_allocate(resource_type, units):
                    self.resource_pool.allocate(node.node_id, resource_type, units)
                    node.record_win(resource_type, units, payment=0.0, round_num=rn)
                    winner_id = node.node_id
                    # Advance pointer past the winner for next round
                    self._pointer[resource_type] = (idx + 1) % n
                    self.total_allocations += 1
                    break

            # Record losses for non-winners that bid
            for node in self.nodes:
                if node.node_id in bids and node.node_id != winner_id:
                    node.record_loss(rn)

            winners[resource_type] = winner_id

        return winners

    def get_stats(self) -> Dict:
        per_node = {
            node.node_id: {
                "total_allocated": dict(node.total_allocated),
                "hunger_counter":  node.hunger_counter,
            }
            for node in self.nodes
        }
        return {
            "total_rounds":      self.round_num,
            "total_allocations": self.total_allocations,
            "nodes":             per_node,
        }

    def __repr__(self) -> str:  # pragma: no cover
        ptrs = ", ".join(f"{r}={p}" for r, p in self._pointer.items())
        return (
            f"RoundRobinAllocator(round={self.round_num}, "
            f"pointers=[{ptrs}], allocations={self.total_allocations})"
        )
