"""
src/baselines/random_alloc.py

Random allocation baseline.

Each round, for each resource type, one randomly-chosen node with positive
demand wins the allocation. No payment, no auction, no priority.

Used in experiments/run_experiment.py as the lower bound on fairness / efficiency.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.core.node          import Node
from src.core.resource_pool import ResourcePool, RESOURCE_TYPES


RESOURCE_UNITS: Dict[str, float] = {
    "CPU": 4.0,
    "RAM": 16.0,
    "STG": 100.0,
    "NET": 1.0,
}


class RandomAllocator:
    """
    Random allocation baseline.

    Parameters
    ----------
    nodes : list[Node]
    resource_pool : ResourcePool
    rng : np.random.Generator, optional
    """

    def __init__(
        self,
        nodes: List[Node],
        resource_pool: ResourcePool,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.nodes:          List[Node]   = nodes
        self.resource_pool:  ResourcePool = resource_pool
        self._rng = rng if rng is not None else np.random.default_rng()
        self.round_num:      int          = 0
        self.total_allocations: int       = 0

    def run_round(self) -> Dict[str, Optional[str]]:
        """Execute one random-allocation round across all resource types."""
        self.round_num += 1
        rn      = self.round_num
        winners: Dict[str, Optional[str]] = {}

        for resource_type in RESOURCE_TYPES:
            units = RESOURCE_UNITS.get(resource_type)
            if units is None:
                continue

            # Collect nodes with demand
            demanding = []
            for node in self.nodes:
                demand = node.generate_bid(resource_type, rn)
                if demand > 0:
                    demanding.append(node)

            winner_id = None
            if demanding and self.resource_pool.can_allocate(resource_type, units):
                winner = demanding[int(self._rng.integers(len(demanding)))]
                self.resource_pool.allocate(winner.node_id, resource_type, units)
                winner.record_win(resource_type, units, payment=0.0, round_num=rn)
                winner_id = winner.node_id
                self.total_allocations += 1

            for node in demanding:
                if node.node_id != winner_id:
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
        return (
            f"RandomAllocator(round={self.round_num}, "
            f"allocations={self.total_allocations})"
        )
