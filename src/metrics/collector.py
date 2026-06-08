"""
src/metrics/collector.py

Metrics collection for the Auctus simulation.

Captures per-round snapshots of:
  - Resource utilization per type
  - Token distribution (mean, std, min, max, Jain's index)
  - Hunger / starvation state
  - Allocation counts per node

All data is accumulated in-memory as lists of dicts and can be exported
to a pandas DataFrame or CSV for paper figures.

Design notes:
  - Collector is *passive* — it observes state but never mutates it.
  - One Collector instance per simulation run.
  - attach() wires it to a controller via a post-round callback hook.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from src.core.node          import Node
from src.core.resource_pool import ResourcePool
from src.core.token_manager import TokenManager


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Passive observer that records per-round simulation metrics.

    Parameters
    ----------
    nodes : list[Node]
    resource_pool : ResourcePool
    token_manager : TokenManager
    """

    def __init__(
        self,
        nodes: List[Node],
        resource_pool: ResourcePool,
        token_manager: TokenManager,
    ) -> None:
        self.nodes:          List[Node]    = nodes
        self.resource_pool:  ResourcePool  = resource_pool
        self.token_manager:  TokenManager  = token_manager

        # Per-round snapshots
        self.round_records:  List[Dict]   = []
        self._node_ids: List[str] = [n.node_id for n in nodes]

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def record_round(self, round_num: int, sim_time: float) -> None:
        """
        Capture a full snapshot at the end of *round_num*.

        Parameters
        ----------
        round_num : int
        sim_time : float
            Current SimPy environment time (minutes).
        """
        util   = self.resource_pool.get_utilization()
        tokens = [n.tokens for n in self.nodes]

        record = {
            "round":         round_num,
            "sim_time":      sim_time,
            # Utilization (%)
            "util_cpu":      util.get("CPU", 0.0),
            "util_ram":      util.get("RAM", 0.0),
            "util_stg":      util.get("STG", 0.0),
            "util_net":      util.get("NET", 0.0),
            # Token economy
            "token_mean":    _mean(tokens),
            "token_std":     _std(tokens),
            "token_min":     min(tokens) if tokens else 0.0,
            "token_max":     max(tokens) if tokens else 0.0,
            "token_jain":    jain_index(tokens),
            "token_total":   sum(tokens),
            # Starvation
            "hungry_nodes":  sum(1 for n in self.nodes if n.hunger_counter >= 10),
            "max_hunger":    max((n.hunger_counter for n in self.nodes), default=0),
            # Per-node token snapshot (flat columns: token_<node_id>)
            **{f"tok_{n.node_id}": n.tokens for n in self.nodes},
        }
        self.round_records.append(record)

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    def allocation_jain(self) -> float:
        """
        Jain's Fairness Index on cumulative CPU allocations across all nodes.

        Returns 0.0 if no allocations have occurred.
        """
        allocs = [n.total_allocated.get("CPU", 0.0) for n in self.nodes]
        return jain_index(allocs)

    def starvation_rate(self) -> float:
        """
        Fraction of node-rounds that ended with hunger_counter > 0.

        Computed from per-round records: sum(hungry_nodes) / (rounds * n_nodes).
        Returns 0.0 if no rounds recorded yet.
        """
        if not self.round_records or not self.nodes:
            return 0.0
        total_hungry = sum(r["hungry_nodes"] for r in self.round_records)
        return total_hungry / (len(self.round_records) * len(self.nodes))

    def mean_utilization(self, resource: str = "CPU") -> float:
        """Time-averaged utilization for *resource* across all recorded rounds."""
        key = f"util_{resource.lower()}"
        vals = [r[key] for r in self.round_records if key in r]
        return _mean(vals)

    def summary(self) -> Dict:
        """
        Return a flat summary dict suitable for a results table.

        Keys match column names used in experiments/run_experiment.py.
        """
        if not self.round_records:
            return {}

        last = self.round_records[-1]
        return {
            "rounds":            last["round"],
            "util_cpu_mean":     self.mean_utilization("CPU"),
            "util_ram_mean":     self.mean_utilization("RAM"),
            "util_stg_mean":     self.mean_utilization("STG"),
            "util_net_mean":     self.mean_utilization("NET"),
            "token_jain_final":  last["token_jain"],
            "alloc_jain":        self.allocation_jain(),
            "starvation_rate":   self.starvation_rate(),
            "max_hunger_ever":   max(r["max_hunger"] for r in self.round_records),
            "token_total_final": last["token_total"],
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dataframe(self):
        """
        Return all round records as a pandas DataFrame.

        Requires pandas (listed in requirements).

        Returns
        -------
        pd.DataFrame
        """
        import pandas as pd  # lazy import — not needed if caller uses to_csv
        return pd.DataFrame(self.round_records)

    def to_csv(self, path: str) -> None:
        """
        Write all round records to *path* as a CSV file.

        Uses stdlib csv — no pandas dependency.
        """
        import csv
        if not self.round_records:
            return
        fieldnames = list(self.round_records[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.round_records)

    def __len__(self) -> int:
        return len(self.round_records)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MetricsCollector(nodes={len(self.nodes)}, "
            f"rounds_recorded={len(self.round_records)})"
        )


# ---------------------------------------------------------------------------
# Pure metric functions (importable independently)
# ---------------------------------------------------------------------------

def jain_index(values: List[float]) -> float:
    """
    Jain's Fairness Index.

    J(x) = (sum(x_i))^2 / (n * sum(x_i^2))

    Returns 1.0 for perfect equality, 1/n for maximum inequality.
    Returns 0.0 if all values are zero or the list is empty.

    Parameters
    ----------
    values : list[float]

    Returns
    -------
    float in [0, 1]
    """
    n = len(values)
    if n == 0:
        return 0.0
    s  = sum(values)
    s2 = sum(x * x for x in values)
    if s2 == 0.0:
        return 0.0
    return (s * s) / (n * s2)


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m  = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance)
