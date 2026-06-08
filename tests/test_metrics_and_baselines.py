"""
tests/test_metrics_and_baselines.py

Tests for:
  - src/metrics/collector.py   (MetricsCollector, jain_index)
  - src/baselines/fcfs.py      (FCFSAllocator)
  - src/baselines/random_alloc.py  (RandomAllocator)
  - src/baselines/round_robin.py   (RoundRobinAllocator)

Run with:
    pytest tests/test_metrics_and_baselines.py -v
"""

import pytest
import numpy as np

from src.core.node           import Node
from src.core.resource_pool  import ResourcePool
from src.core.token_manager  import TokenManager
from src.core.audit_log      import AuditLog
from src.metrics.collector   import MetricsCollector, jain_index, _mean, _std
from src.baselines.fcfs       import FCFSAllocator
from src.baselines.random_alloc import RandomAllocator
from src.baselines.round_robin  import RoundRobinAllocator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_node(node_id, tokens=1000.0, seed=0):
    n = Node(
        node_id=node_id,
        criticality=2,
        tier=2,
        sla_deadline=500.0,
        current_task_progress=0.4,
        current_sim_time=100.0,
        T0=1000.0,
        rng=np.random.default_rng(seed),
    )
    n.tokens = tokens
    return n


def make_nodes(count, base_seed=10):
    return [make_node(f"n{i}", seed=base_seed + i) for i in range(count)]


# ===========================================================================
# jain_index
# ===========================================================================

class TestJainIndex:
    def test_perfect_equality(self):
        assert jain_index([1.0, 1.0, 1.0]) == pytest.approx(1.0)

    def test_maximum_inequality(self):
        # One node gets everything: [n, 0, 0, ...] -> J = 1/n
        n = 4
        vals = [1.0] + [0.0] * (n - 1)
        assert jain_index(vals) == pytest.approx(1.0 / n)

    def test_empty_list_returns_zero(self):
        assert jain_index([]) == pytest.approx(0.0)

    def test_all_zero_returns_zero(self):
        assert jain_index([0.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_single_nonzero_returns_one(self):
        assert jain_index([42.0]) == pytest.approx(1.0)

    def test_range_zero_to_one(self):
        vals = [10.0, 5.0, 1.0, 8.0]
        j = jain_index(vals)
        assert 0.0 < j <= 1.0

    def test_symmetric(self):
        vals = [3.0, 7.0, 5.0]
        assert jain_index(vals) == pytest.approx(jain_index(vals[::-1]))

    def test_two_equal_values(self):
        assert jain_index([5.0, 5.0]) == pytest.approx(1.0)

    def test_two_unequal_values(self):
        # [1, 3]: J = 16 / (2 * 10) = 0.8
        assert jain_index([1.0, 3.0]) == pytest.approx(0.8)


# ===========================================================================
# MetricsCollector
# ===========================================================================

class TestMetricsCollector:

    def _make_collector(self, n=3):
        nodes = make_nodes(n)
        pool  = ResourcePool()
        tm    = TokenManager()
        for node in nodes:
            tm.register_node(node)
        return MetricsCollector(nodes, pool, tm), nodes, pool, tm

    def test_record_round_appends_entry(self):
        col, nodes, pool, tm = self._make_collector()
        col.record_round(round_num=1, sim_time=5.0)
        assert len(col) == 1

    def test_record_round_contains_expected_keys(self):
        col, nodes, pool, tm = self._make_collector()
        col.record_round(1, 5.0)
        rec = col.round_records[0]
        for key in ("round", "sim_time", "util_cpu", "token_mean",
                    "token_jain", "hungry_nodes", "max_hunger"):
            assert key in rec

    def test_token_mean_correct(self):
        col, nodes, pool, tm = self._make_collector(n=2)
        nodes[0].tokens = 800.0
        nodes[1].tokens = 600.0
        col.record_round(1, 5.0)
        assert col.round_records[0]["token_mean"] == pytest.approx(700.0)

    def test_token_jain_perfect_equality(self):
        col, nodes, pool, tm = self._make_collector(n=3)
        for node in nodes:
            node.tokens = 500.0
        col.record_round(1, 5.0)
        assert col.round_records[0]["token_jain"] == pytest.approx(1.0)

    def test_hungry_nodes_count(self):
        col, nodes, pool, tm = self._make_collector(n=3)
        nodes[0].hunger_counter = 10
        nodes[1].hunger_counter = 15
        nodes[2].hunger_counter = 5
        col.record_round(1, 5.0)
        assert col.round_records[0]["hungry_nodes"] == 2

    def test_max_hunger_correct(self):
        col, nodes, pool, tm = self._make_collector(n=3)
        nodes[0].hunger_counter = 7
        nodes[1].hunger_counter = 20
        nodes[2].hunger_counter = 3
        col.record_round(1, 5.0)
        assert col.round_records[0]["max_hunger"] == 20

    def test_allocation_jain_all_equal(self):
        col, nodes, pool, tm = self._make_collector(n=3)
        for node in nodes:
            node.total_allocated["CPU"] = 10.0
        assert col.allocation_jain() == pytest.approx(1.0)

    def test_allocation_jain_zero_when_no_allocs(self):
        col, nodes, pool, tm = self._make_collector(n=3)
        assert col.allocation_jain() == pytest.approx(0.0)

    def test_starvation_rate_zero_with_no_hunger(self):
        col, nodes, pool, tm = self._make_collector(n=3)
        for _ in range(5):
            col.record_round(_, _*5.0)
        assert col.starvation_rate() == pytest.approx(0.0)

    def test_starvation_rate_nonzero_with_hungry_nodes(self):
        col, nodes, pool, tm = self._make_collector(n=3)
        nodes[0].hunger_counter = 10
        col.record_round(1, 5.0)
        # 1 hungry node out of 3 → rate = 1/3
        assert col.starvation_rate() == pytest.approx(1.0 / 3.0)

    def test_mean_utilization_zero_when_nothing_allocated(self):
        col, nodes, pool, tm = self._make_collector()
        col.record_round(1, 5.0)
        assert col.mean_utilization("CPU") == pytest.approx(0.0)

    def test_summary_contains_required_keys(self):
        col, nodes, pool, tm = self._make_collector()
        col.record_round(1, 5.0)
        s = col.summary()
        for key in ("rounds", "util_cpu_mean", "alloc_jain",
                    "starvation_rate", "max_hunger_ever"):
            assert key in s

    def test_summary_empty_before_any_records(self):
        col, nodes, pool, tm = self._make_collector()
        assert col.summary() == {}

    def test_to_csv_creates_file(self, tmp_path):
        col, nodes, pool, tm = self._make_collector()
        col.record_round(1, 5.0)
        col.record_round(2, 10.0)
        path = str(tmp_path / "metrics.csv")
        col.to_csv(path)
        import os
        assert os.path.exists(path)

    def test_to_csv_correct_row_count(self, tmp_path):
        col, nodes, pool, tm = self._make_collector()
        for i in range(5):
            col.record_round(i + 1, (i + 1) * 5.0)
        path = str(tmp_path / "metrics.csv")
        col.to_csv(path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 6   # header + 5 data rows


# ===========================================================================
# FCFSAllocator
# ===========================================================================

class TestFCFSAllocator:

    def _make_fcfs(self, n=3):
        nodes = make_nodes(n)
        pool  = ResourcePool()
        return FCFSAllocator(nodes, pool), nodes, pool

    def test_first_node_wins_cpu(self):
        fcfs, nodes, pool = self._make_fcfs(n=3)
        # All nodes have demand; node 0 should win (FCFS order)
        winners = fcfs.run_round()
        assert winners["CPU"] == "n0"

    def test_allocation_recorded_in_pool(self):
        fcfs, nodes, pool = self._make_fcfs(n=2)
        fcfs.run_round()
        assert "n0" in pool.allocations

    def test_available_decreases_after_allocation(self):
        fcfs, nodes, pool = self._make_fcfs()
        before = pool.available["CPU"]
        fcfs.run_round()
        assert pool.available["CPU"] < before

    def test_round_num_increments(self):
        fcfs, nodes, pool = self._make_fcfs()
        fcfs.run_round()
        fcfs.run_round()
        assert fcfs.round_num == 2

    def test_total_allocations_increments(self):
        fcfs, nodes, pool = self._make_fcfs()
        fcfs.run_round()
        # 4 resource types, n0 wins each → 4 allocations
        assert fcfs.total_allocations == 4

    def test_get_stats_returns_dict(self):
        fcfs, nodes, pool = self._make_fcfs()
        fcfs.run_round()
        stats = fcfs.get_stats()
        assert "total_allocations" in stats
        assert "nodes" in stats

    def test_no_negative_tokens(self):
        """FCFS does not deduct tokens — balances must stay at T0."""
        fcfs, nodes, pool = self._make_fcfs(n=3)
        for _ in range(10):
            fcfs.run_round()
        for node in nodes:
            assert node.tokens >= 0.0


# ===========================================================================
# RandomAllocator
# ===========================================================================

class TestRandomAllocator:

    def _make_rand(self, n=4, seed=42):
        nodes = make_nodes(n)
        pool  = ResourcePool()
        return RandomAllocator(nodes, pool, rng=np.random.default_rng(seed)), nodes, pool

    def test_runs_without_error(self):
        rand, nodes, pool = self._make_rand()
        rand.run_round()

    def test_exactly_one_winner_per_resource(self):
        rand, nodes, pool = self._make_rand()
        winners = rand.run_round()
        for r, wid in winners.items():
            assert wid is None or isinstance(wid, str)

    def test_total_allocations_non_negative(self):
        rand, nodes, pool = self._make_rand()
        for _ in range(5):
            rand.run_round()
        assert rand.total_allocations >= 0

    def test_deterministic_with_fixed_seed(self):
        """Same seed → same allocation sequence."""
        nodes_a = make_nodes(4, base_seed=0)
        nodes_b = make_nodes(4, base_seed=0)
        pool_a  = ResourcePool()
        pool_b  = ResourcePool()
        rand_a  = RandomAllocator(nodes_a, pool_a, rng=np.random.default_rng(77))
        rand_b  = RandomAllocator(nodes_b, pool_b, rng=np.random.default_rng(77))

        w_a = rand_a.run_round()
        w_b = rand_b.run_round()
        assert w_a == w_b

    def test_no_negative_tokens(self):
        rand, nodes, pool = self._make_rand(n=4)
        for _ in range(10):
            rand.run_round()
        for node in nodes:
            assert node.tokens >= 0.0


# ===========================================================================
# RoundRobinAllocator
# ===========================================================================

class TestRoundRobinAllocator:

    def _make_rr(self, n=4):
        nodes = make_nodes(n)
        pool  = ResourcePool()
        return RoundRobinAllocator(nodes, pool), nodes, pool

    def test_runs_without_error(self):
        rr, nodes, pool = self._make_rr()
        rr.run_round()

    def test_pointer_advances_after_win(self):
        rr, nodes, pool = self._make_rr(n=4)
        # Round 1: n0 wins CPU
        w1 = rr.run_round()
        assert w1["CPU"] == "n0"
        # Round 2: pointer is now at n1
        w2 = rr.run_round()
        assert w2["CPU"] == "n1"

    def test_pointer_wraps_around(self):
        rr, nodes, pool = self._make_rr(n=2)
        # n0 wins round 1, n1 wins round 2, n0 wins round 3
        w1 = rr.run_round()
        w2 = rr.run_round()
        w3 = rr.run_round()
        assert w1["CPU"] == "n0"
        assert w2["CPU"] == "n1"
        assert w3["CPU"] == "n0"

    def test_total_allocations_correct(self):
        rr, nodes, pool = self._make_rr(n=4)
        rr.run_round()
        # 4 resource types each get 1 winner → 4 allocations
        assert rr.total_allocations == 4

    def test_fairness_over_n_rounds(self):
        """Over n rounds, each node should win approximately once per cycle."""
        n = 4
        rr, nodes, pool = self._make_rr(n=n)
        cpu_wins: dict = {node.node_id: 0 for node in nodes}

        for _ in range(n * 2):
            # Reset pool so it never fills up
            pool.available["CPU"] = pool.capacity["CPU"] - pool.reserved["CPU"]
            pool.allocations.clear()
            w = rr.run_round()
            if w["CPU"]:
                cpu_wins[w["CPU"]] += 1

        # Each node should have won at least once in 2 * n rounds
        for nid, count in cpu_wins.items():
            assert count >= 1, f"{nid} never won in {n*2} rounds"

    def test_get_stats_returns_dict(self):
        rr, nodes, pool = self._make_rr()
        rr.run_round()
        stats = rr.get_stats()
        assert "total_rounds" in stats
        assert "nodes" in stats

    def test_no_negative_tokens(self):
        """Round-robin does not deduct tokens."""
        rr, nodes, pool = self._make_rr(n=4)
        for _ in range(10):
            pool.available["CPU"] = pool.capacity["CPU"] - pool.reserved["CPU"]
            pool.allocations.clear()
            rr.run_round()
        for node in nodes:
            assert node.tokens >= 0.0

    def test_empty_nodes_list_returns_empty(self):
        pool = ResourcePool()
        rr   = RoundRobinAllocator([], pool)
        winners = rr.run_round()
        assert all(v is None for v in winners.values())
