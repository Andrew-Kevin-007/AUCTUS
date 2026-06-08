"""
src/baselines/fcfs.py

First-Come-First-Served (FCFS) baseline allocator.

FCFS allocates each resource to the node that submitted a request first
(lowest node_id index in the ordered node list, as a proxy for arrival order).
No token deduction — this is a non-market baseline used to contrast fairness
and utilization against the Auctus VCG mechanism.

Used in experiments/run_experiment.py for the baseline comparison table.
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


class FCFSAllocator:
    """
    First-Come-First-Served baseline.

    Allocation order is determined by the position of nodes in the *nodes*
    list (index 0 = highest priority).  In each round, the first node that
    has demand (generate_bid > 0) wins the resource — no auction, no payment.

    Parameters
    ----------
    nodes : list[Node]
    resource_pool : ResourcePool
    """

    def __init__(self, nodes: List[Node], resource_pool: ResourcePool) -> None:
        self.nodes:          List[Node]   = nodes
        self.resource_pool:  ResourcePool = resource_pool
        self.round_num:      int          = 0
        self.total_allocations: int       = 0

    def run_round(self) -> Dict[str, Optional[str]]:
        """
        Execute one FCFS round across all resource types.

        Returns
        -------
        dict
            {resource_type: winner_id | None}
        """
        self.round_num += 1
        rn      = self.round_num
        winners: Dict[str, Optional[str]] = {}

        for resource_type in RESOURCE_TYPES:
            units = RESOURCE_UNITS.get(resource_type)
            if units is None:
                continue

            winner_id = None
            for node in self.nodes:
                # Any positive demand counts; no bid amount comparison
                demand = node.generate_bid(resource_type, rn)
                if demand <= 0:
                    continue
                if self.resource_pool.can_allocate(resource_type, units):
                    self.resource_pool.allocate(node.node_id, resource_type, units)
                    node.record_win(resource_type, units, payment=0.0, round_num=rn)
                    winner_id = node.node_id
                    self.total_allocations += 1
                    break  # first-come wins; skip remaining nodes

            # Record losses for all nodes that bid but didn't win
            for node in self.nodes:
                if node.node_id != winner_id:
                    # Only increment hunger for nodes that had demand
                    if node.bid_history and node.bid_history[-1][0] == rn:
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
            f"FCFSAllocator(round={self.round_num}, "
            f"allocations={self.total_allocations})"
        )
