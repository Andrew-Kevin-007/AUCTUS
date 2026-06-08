"""
experiments/run_experiment.py

Auctus protocol simulation experiment runner.

Runs the full Auctus protocol (VCG + ERP + SFG) and all three baselines
(FCFS, Random, Round-Robin) across node counts N = {50, 100, 250, 500}
over a 30-day simulated horizon (8640 rounds at 5-min intervals).

Outputs
-------
results/raw/auctus_N<n>.csv          -- per-round metrics for Auctus
results/raw/summary_table.csv        -- aggregate comparison table
results/raw/summary_table.txt        -- human-readable table

Usage
-----
    python experiments/run_experiment.py [--node-counts 50 100] [--days 1] [--seed 42]

Progress is printed to stdout so you can tail -f while it runs.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List

import numpy as np
import simpy

# Ensure src/ is on the path when running as a script from any working dir
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.core.node           import Node
from src.core.resource_pool  import ResourcePool
from src.core.token_manager  import TokenManager
from src.core.audit_log      import AuditLog
from src.auction.controller  import AuctionController, ROUND_INTERVAL
from src.auction.emergency   import ERPManager
from src.auction.starvation_floor import SFGManager
from src.baselines.fcfs       import FCFSAllocator
from src.baselines.random_alloc import RandomAllocator
from src.baselines.round_robin  import RoundRobinAllocator
from src.metrics.collector   import MetricsCollector, jain_index


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_NODE_COUNTS = [50, 100, 250, 500]
MINUTES_PER_DAY     = 1440
ROUNDS_PER_DAY      = MINUTES_PER_DAY // ROUND_INTERVAL    # 288
T0                  = 1000.0


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def build_nodes(n: int, rng: np.random.Generator) -> List[Node]:
    """
    Create *n* nodes with a realistic criticality / tier distribution:
      - 10% E1 critical  (tier 1)
      - 30% E2 prod      (tier 2)
      - 60% E3 SLA       (tier 3)
    """
    nodes = []
    for i in range(n):
        if i < max(1, n // 10):
            crit, tier = 1, 1
        elif i < max(2, n * 4 // 10):
            crit, tier = 2, 2
        else:
            crit, tier = 3, 3

        # Stagger SLA deadlines and progress so not all nodes are identical
        deadline  = float(rng.integers(200, 800))
        progress  = float(rng.uniform(0.1, 0.6))
        sim_time  = float(rng.integers(50, 300))

        node = Node(
            node_id=f"node_{i:04d}",
            criticality=crit,
            tier=tier,
            sla_deadline=deadline,
            current_task_progress=progress,
            current_sim_time=sim_time,
            T0=T0,
            rng=np.random.default_rng(int(rng.integers(1 << 31))),
        )
        nodes.append(node)
    return nodes


# ---------------------------------------------------------------------------
# Integrated controller (mirrors test_day2_integration.py approach)
# ---------------------------------------------------------------------------

class IntegratedController(AuctionController):
    """AuctionController extended with external ERP + SFG hooks."""

    def attach(self, erp: ERPManager, sfg: SFGManager,
               collector: MetricsCollector) -> None:
        self._erp       = erp
        self._sfg       = sfg
        self._collector = collector

    def run_round(self) -> None:
        super().run_round()
        rn = self.round_num

        triggered = self._erp.process_emergency_requests(rn)
        self.stats["erp_events"] += len(triggered)

        before = self._sfg.sfg_events_total
        self._sfg.check_and_apply(rn)
        self.stats["sfg_events"] += self._sfg.sfg_events_total - before

        self._collector.record_round(rn, sim_time=self.env.now)


# ---------------------------------------------------------------------------
# Auctus run
# ---------------------------------------------------------------------------

def run_auctus(n: int, sim_minutes: float, seed: int) -> Dict:
    rng   = np.random.default_rng(seed)
    nodes = build_nodes(n, rng)

    pool  = ResourcePool()
    tm    = TokenManager(T0=T0)
    audit = AuditLog(creation_timestamp=float(seed))

    for node in nodes:
        tm.register_node(node)

    env  = simpy.Environment()
    ctrl = IntegratedController(
        env=env, nodes=nodes, resource_pool=pool,
        token_manager=tm, audit_log=audit,
        rng=np.random.default_rng(seed + 1),
    )

    node_map  = {n.node_id: n for n in nodes}
    collector = MetricsCollector(nodes, pool, tm)

    erp = ERPManager(
        nodes=node_map, resource_pool=pool,
        token_manager=tm, audit_log=audit, vcg=ctrl.vcg,
    )
    sfg = SFGManager(
        nodes=node_map, resource_pool=pool,
        token_manager=tm, audit_log=audit, vcg=ctrl.vcg,
    )

    ctrl.attach(erp, sfg, collector)
    ctrl.start(sim_duration_minutes=sim_minutes)

    stats = ctrl.get_stats()
    s     = collector.summary()

    return {
        "mechanism":        "Auctus",
        "n_nodes":          n,
        "rounds":           ctrl.round_num,
        "total_alloc":      stats["total_allocations"],
        "erp_events":       stats.get("erp_events", 0),
        "sfg_events":       stats.get("sfg_events", 0) + sfg.sfg_events_total,
        "failed_deduct":    stats.get("failed_deductions", 0),
        "util_cpu_mean":    s.get("util_cpu_mean", 0.0),
        "alloc_jain":       s.get("alloc_jain", 0.0),
        "token_jain_final": s.get("token_jain_final", 0.0),
        "starvation_rate":  s.get("starvation_rate", 0.0),
        "max_hunger_ever":  s.get("max_hunger_ever", 0),
        "audit_ok":         audit.verify_integrity(),
        "collector":        collector,   # for CSV export
    }


# ---------------------------------------------------------------------------
# Baseline runs (no SimPy — purely synchronous)
# ---------------------------------------------------------------------------

def run_baseline(mechanism: str, n: int, rounds: int, seed: int) -> Dict:
    rng   = np.random.default_rng(seed)
    nodes = build_nodes(n, rng)
    pool  = ResourcePool()

    if mechanism == "FCFS":
        alloc = FCFSAllocator(nodes, pool)
    elif mechanism == "Random":
        alloc = RandomAllocator(nodes, pool, rng=np.random.default_rng(seed + 2))
    elif mechanism == "RoundRobin":
        alloc = RoundRobinAllocator(nodes, pool)
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")

    for _ in range(rounds):
        # Reset pool each round so it never permanently fills
        # (baselines have no task-duration/deallocation model)
        for r in pool.available:
            pool.available[r] = pool.capacity[r] - pool.reserved[r]
        pool.allocations.clear()
        alloc.run_round()

    alloc_vals = [n.total_allocated.get("CPU", 0.0) for n in nodes]

    return {
        "mechanism":        mechanism,
        "n_nodes":          n,
        "rounds":           alloc.round_num,
        "total_alloc":      alloc.total_allocations,
        "erp_events":       0,
        "sfg_events":       0,
        "failed_deduct":    0,
        "util_cpu_mean":    0.0,   # baselines don't track time-series utilization
        "alloc_jain":       jain_index(alloc_vals),
        "token_jain_final": 1.0,   # no token economy in baselines
        "starvation_rate":  0.0,
        "max_hunger_ever":  max((n.hunger_counter for n in nodes), default=0),
        "audit_ok":         True,
        "collector":        None,
    }


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

_SUMMARY_FIELDS = [
    "mechanism", "n_nodes", "rounds", "total_alloc",
    "erp_events", "sfg_events", "failed_deduct",
    "util_cpu_mean", "alloc_jain", "token_jain_final",
    "starvation_rate", "max_hunger_ever", "audit_ok",
]


def write_summary_csv(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_txt(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    col_w = [14, 8, 8, 11, 10, 10, 13, 14, 11, 16, 16, 15, 9]
    headers = ["mechanism", "N", "rounds", "total_alloc",
               "erp_events", "sfg_events", "failed_deduct",
               "util_cpu%", "alloc_jain", "token_jain",
               "starv_rate", "max_hunger", "audit_ok"]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"

    def fmt_row(vals):
        return "|" + "|".join(
            str(v)[:w-1].center(w) for v, w in zip(vals, col_w)
        ) + "|"

    with open(path, "w") as f:
        f.write(sep + "\n")
        f.write(fmt_row(headers) + "\n")
        f.write(sep + "\n")
        for row in rows:
            vals = [
                row["mechanism"],
                row["n_nodes"],
                row["rounds"],
                row["total_alloc"],
                row["erp_events"],
                row["sfg_events"],
                row["failed_deduct"],
                f"{row['util_cpu_mean']:.2f}",
                f"{row['alloc_jain']:.4f}",
                f"{row['token_jain_final']:.4f}",
                f"{row['starvation_rate']:.4f}",
                row["max_hunger_ever"],
                row["audit_ok"],
            ]
            f.write(fmt_row(vals) + "\n")
        f.write(sep + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Auctus experiment runner")
    parser.add_argument("--node-counts", nargs="+", type=int,
                        default=DEFAULT_NODE_COUNTS,
                        metavar="N",
                        help="Node counts to simulate (default: 50 100 250 500)")
    parser.add_argument("--days", type=float, default=30.0,
                        help="Simulated days (default: 30)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base RNG seed (default: 42)")
    parser.add_argument("--out-dir", default="results/raw",
                        help="Output directory (default: results/raw)")
    args = parser.parse_args()

    sim_minutes = args.days * MINUTES_PER_DAY
    rounds      = int(sim_minutes // ROUND_INTERVAL)
    out_dir     = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nAuctus Experiment Runner")
    print(f"  Node counts : {args.node_counts}")
    print(f"  Sim horizon : {args.days:.0f} days  ({sim_minutes:.0f} min, {rounds} rounds)")
    print(f"  Base seed   : {args.seed}")
    print(f"  Output dir  : {out_dir}\n")

    all_rows: List[Dict] = []

    for n in args.node_counts:
        mechanisms = ["Auctus", "FCFS", "Random", "RoundRobin"]
        for mech in mechanisms:
            t0 = time.time()
            print(f"  Running {mech:12s}  N={n:4d} ...", end="", flush=True)

            if mech == "Auctus":
                row = run_auctus(n, sim_minutes, seed=args.seed + n)
                # Export per-round CSV
                col = row.pop("collector")
                csv_path = os.path.join(out_dir, f"auctus_N{n:04d}.csv")
                if col is not None:
                    col.to_csv(csv_path)
            else:
                row = run_baseline(mech, n, rounds, seed=args.seed + n + 1000)
                row.pop("collector", None)

            elapsed = time.time() - t0
            print(
                f"  done  {elapsed:5.1f}s  "
                f"alloc={row['total_alloc']:6d}  "
                f"jain={row['alloc_jain']:.4f}  "
                f"starv={row['starvation_rate']:.4f}"
            )
            all_rows.append(row)

    # Write summary files
    summary_csv = os.path.join(out_dir, "summary_table.csv")
    summary_txt = os.path.join(out_dir, "summary_table.txt")
    write_summary_csv(all_rows, summary_csv)
    write_summary_txt(all_rows, summary_txt)

    print(f"\nResults written to {out_dir}/")
    print(f"  {summary_csv}")
    print(f"  {summary_txt}")

    # Print the table to stdout as well
    print()
    with open(summary_txt) as f:
        print(f.read())


if __name__ == "__main__":
    main()
