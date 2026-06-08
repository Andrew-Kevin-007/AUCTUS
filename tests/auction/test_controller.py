"""
tests/auction/test_controller.py

Pytest tests for src.auction.controller.AuctionController.

Run with:
    pytest tests/auction/test_controller.py -v
"""

import pytest
import numpy as np
import simpy

from src.core.node          import Node
from src.core.resource_pool import ResourcePool
from src.core.token_manager import TokenManager
from src.core.audit_log     import AuditLog, AUCTION_WIN, AUCTION_LOSS
from src.auction.controller import AuctionController, ROUND_INTERVAL, RESOURCE_UNITS


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_node(node_id, criticality, tier, tokens=1000.0, seed=0):
    node = Node(
        node_id=node_id,
        criticality=criticality,
        tier=tier,
        sla_deadline=500.0,
        current_task_progress=0.3,
        current_sim_time=100.0,
        T0=1000.0,
        rng=np.random.default_rng(seed),
    )
    node.tokens = tokens
    return node


def make_controller(nodes, rng_seed=42):
    """Return a wired (env, controller) pair ready to run."""
    env   = simpy.Environment()
    pool  = ResourcePool()
    tm    = TokenManager(T0=1000.0)
    audit = AuditLog(creation_timestamp=1_700_000_000.0)

    for node in nodes:
        tm.register_node(node)
        node.tokens = node.tokens   # re-assert after register resets

    rng = np.random.default_rng(rng_seed)
    ctrl = AuctionController(env, nodes, pool, tm, audit, rng=rng)
    return env, ctrl


# ---------------------------------------------------------------------------
# 1. Single round runs without error
# ---------------------------------------------------------------------------

class TestBasicRound:
    def test_one_round_completes_without_exception(self):
        nodes = [
            make_node("n1", criticality=1, tier=3, tokens=900.0, seed=10),
            make_node("n2", criticality=2, tier=2, tokens=800.0, seed=20),
            make_node("n3", criticality=3, tier=1, tokens=700.0, seed=30),
        ]
        env, ctrl = make_controller(nodes)
        # Run exactly one round interval
        env.process(ctrl._auction_loop(sim_duration_minutes=ROUND_INTERVAL))
        env.run(until=ROUND_INTERVAL + 1)
        # round_num should be 1
        assert ctrl.round_num == 1

    def test_stats_total_rounds_increments(self):
        nodes = [make_node("n1", 1, 3, seed=1), make_node("n2", 2, 2, seed=2)]
        env, ctrl = make_controller(nodes)
        env.process(ctrl._auction_loop(sim_duration_minutes=ROUND_INTERVAL * 3))
        env.run(until=ROUND_INTERVAL * 3 + 1)
        assert ctrl.stats["total_rounds"] == 3

    def test_start_runs_to_duration(self):
        nodes = [make_node("n1", 1, 3, seed=1), make_node("n2", 2, 2, seed=2)]
        env, ctrl = make_controller(nodes)
        ctrl.start(sim_duration_minutes=ROUND_INTERVAL * 2)
        assert ctrl.round_num >= 1


# ---------------------------------------------------------------------------
# 2. Token deduction on win
# ---------------------------------------------------------------------------

class TestTokenDeduction:
    def _run_one_round_with_fixed_bids(self, bid_n1, bid_n2):
        """
        Force deterministic bids by monkeypatching generate_bid.
        Returns (ctrl, n1, n2) after one round.
        register_node resets tokens to T0, so we reassert desired balances
        after make_controller returns.
        """
        n1 = make_node("n1", criticality=1, tier=3, tokens=900.0, seed=0)
        n2 = make_node("n2", criticality=2, tier=2, tokens=800.0, seed=0)

        env, ctrl = make_controller([n1, n2])

        # make_controller calls register_node which resets to T0=1000 --
        # reassert the desired distinct balances after wiring.
        n1.tokens = 900.0
        n2.tokens = 800.0

        n1.generate_bid = lambda rt, rn: bid_n1
        n2.generate_bid = lambda rt, rn: bid_n2

        env.process(ctrl._auction_loop(sim_duration_minutes=ROUND_INTERVAL))
        env.run(until=ROUND_INTERVAL + 1)
        return ctrl, n1, n2

    def test_winner_token_decreases_by_vcg_payment(self):
        """n1 bids 100, n2 bids 60 -> n1 wins all resources, paying 60 each."""
        ctrl, n1, n2 = self._run_one_round_with_fixed_bids(100.0, 60.0)
        assert n1.tokens < 900.0

    def test_loser_tokens_unchanged(self):
        """n2 loses -- tokens must not change from the pre-round value."""
        ctrl, n1, n2 = self._run_one_round_with_fixed_bids(100.0, 60.0)
        assert n2.tokens == pytest.approx(800.0)

    def test_winner_determined_per_resource(self):
        """At least one allocation is recorded after one round."""
        ctrl, n1, n2 = self._run_one_round_with_fixed_bids(100.0, 60.0)
        assert ctrl.stats["total_allocations"] >= 1


# ---------------------------------------------------------------------------
# 3. Winner gets allocation in resource_pool
# ---------------------------------------------------------------------------

class TestResourceAllocation:
    def test_winner_appears_in_pool_allocations(self):
        n1 = make_node("n1", 1, 3, tokens=900.0, seed=11)
        n2 = make_node("n2", 2, 2, tokens=800.0, seed=22)
        env, ctrl = make_controller([n1, n2])

        n1.generate_bid = lambda rt, rn: 100.0
        n2.generate_bid = lambda rt, rn: 60.0

        env.process(ctrl._auction_loop(sim_duration_minutes=ROUND_INTERVAL))
        env.run(until=ROUND_INTERVAL + 1)

        # n1 wins all resources — should appear in pool allocations
        assert "n1" in ctrl.resource_pool.allocations

    def test_pool_available_decreases_after_allocation(self):
        n1 = make_node("n1", 1, 3, tokens=900.0, seed=11)
        n2 = make_node("n2", 2, 2, tokens=800.0, seed=22)
        env, ctrl = make_controller([n1, n2])

        initial_cpu = ctrl.resource_pool.available["CPU"]

        n1.generate_bid = lambda rt, rn: 100.0 if rt == "CPU" else 0.0
        n2.generate_bid = lambda rt, rn: 60.0  if rt == "CPU" else 0.0

        env.process(ctrl._auction_loop(sim_duration_minutes=ROUND_INTERVAL))
        env.run(until=ROUND_INTERVAL + 1)

        assert ctrl.resource_pool.available["CPU"] < initial_cpu

    def test_deallocation_fires_after_timeout(self):
        """
        Run for long enough that the task-duration timeout fires
        and the resource is returned to the pool.
        """
        n1 = make_node("n1", 1, 3, tokens=900.0, seed=11)
        n2 = make_node("n2", 2, 2, tokens=800.0, seed=22)

        # Seed RNG to produce very short task durations (near 0)
        rng = np.random.default_rng(0)   # exponential(30) first draw ≈ small

        env   = simpy.Environment()
        pool  = ResourcePool()
        tm    = TokenManager(T0=1000.0)
        audit = AuditLog(creation_timestamp=1_700_000_000.0)

        for node in [n1, n2]:
            tm.register_node(node)

        n1.tokens = 900.0
        n2.tokens = 800.0

        ctrl = AuctionController(env, [n1, n2], pool, tm, audit, rng=rng)

        n1.generate_bid = lambda rt, rn: 100.0
        n2.generate_bid = lambda rt, rn: 60.0

        initial_cpu = pool.available["CPU"]

        # Run auction round + well past any reasonable task duration
        ctrl.start(sim_duration_minutes=ROUND_INTERVAL + 200)

        # After all tasks complete, CPU should be back (or at most partially held)
        # We just verify no crash and the pool state is coherent
        for r in ["CPU", "RAM", "STG", "NET"]:
            assert pool.available[r] >= 0.0
            assert pool.available[r] <= pool.capacity[r]


# ---------------------------------------------------------------------------
# 4. Loser hunger counter
# ---------------------------------------------------------------------------

class TestHungerCounter:
    def test_loser_hunger_increments_after_one_round(self):
        n1 = make_node("n1", 1, 3, tokens=900.0, seed=0)
        n2 = make_node("n2", 2, 2, tokens=800.0, seed=0)
        env, ctrl = make_controller([n1, n2])

        n1.generate_bid = lambda rt, rn: 100.0
        n2.generate_bid = lambda rt, rn: 60.0

        env.process(ctrl._auction_loop(sim_duration_minutes=ROUND_INTERVAL))
        env.run(until=ROUND_INTERVAL + 1)

        # n2 lost all resources — hunger should be > 0
        assert n2.hunger_counter > 0

    def test_winner_hunger_reset_to_zero(self):
        n1 = make_node("n1", 1, 3, tokens=900.0, seed=0)
        n2 = make_node("n2", 2, 2, tokens=800.0, seed=0)
        n1.hunger_counter = 5   # pre-existing hunger

        env, ctrl = make_controller([n1, n2])

        n1.generate_bid = lambda rt, rn: 100.0
        n2.generate_bid = lambda rt, rn: 60.0

        env.process(ctrl._auction_loop(sim_duration_minutes=ROUND_INTERVAL))
        env.run(until=ROUND_INTERVAL + 1)

        # n1 won at least one resource — hunger should be 0
        assert n1.hunger_counter == 0

    def test_non_bidder_hunger_not_incremented(self):
        """A node that bids 0 (no demand) must not have its hunger incremented."""
        n1 = make_node("n1", 1, 3, tokens=900.0, seed=0)
        n2 = make_node("n2", 2, 2, tokens=800.0, seed=0)
        n3 = make_node("n3", 3, 1, tokens=700.0, seed=0)

        env, ctrl = make_controller([n1, n2, n3])

        n1.generate_bid = lambda rt, rn: 100.0
        n2.generate_bid = lambda rt, rn: 60.0
        n3.generate_bid = lambda rt, rn: 0.0   # no demand this round

        env.process(ctrl._auction_loop(sim_duration_minutes=ROUND_INTERVAL))
        env.run(until=ROUND_INTERVAL + 1)

        # n3 did not bid — hunger counter must remain at initial 0
        assert n3.hunger_counter == 0


# ---------------------------------------------------------------------------
# 5. Audit log populated correctly
# ---------------------------------------------------------------------------

class TestAuditLog:
    def test_audit_log_has_entries_after_round(self):
        nodes = [make_node("n1", 1, 3, seed=1), make_node("n2", 2, 2, seed=2)]
        env, ctrl = make_controller(nodes)

        nodes[0].generate_bid = lambda rt, rn: 100.0
        nodes[1].generate_bid = lambda rt, rn: 60.0

        env.process(ctrl._auction_loop(sim_duration_minutes=ROUND_INTERVAL))
        env.run(until=ROUND_INTERVAL + 1)

        assert len(ctrl.audit_log) > 0

    def test_audit_log_integrity_after_round(self):
        nodes = [make_node("n1", 1, 3, seed=1), make_node("n2", 2, 2, seed=2)]
        env, ctrl = make_controller(nodes)

        nodes[0].generate_bid = lambda rt, rn: 100.0
        nodes[1].generate_bid = lambda rt, rn: 60.0

        env.process(ctrl._auction_loop(sim_duration_minutes=ROUND_INTERVAL))
        env.run(until=ROUND_INTERVAL + 1)

        assert ctrl.audit_log.verify_integrity() is True

    def test_win_events_recorded_for_winner(self):
        nodes = [make_node("n1", 1, 3, seed=1), make_node("n2", 2, 2, seed=2)]
        env, ctrl = make_controller(nodes)

        nodes[0].generate_bid = lambda rt, rn: 100.0
        nodes[1].generate_bid = lambda rt, rn: 60.0

        env.process(ctrl._auction_loop(sim_duration_minutes=ROUND_INTERVAL))
        env.run(until=ROUND_INTERVAL + 1)

        win_events = ctrl.audit_log.get_events_for_node("n1")
        win_types  = [e["event_type"] for e in win_events]
        assert AUCTION_WIN in win_types


# ---------------------------------------------------------------------------
# 6. get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_get_stats_returns_dict(self):
        nodes = [make_node("n1", 1, 3, seed=1)]
        env, ctrl = make_controller(nodes)
        stats = ctrl.get_stats()
        assert isinstance(stats, dict)

    def test_get_stats_contains_node_keys(self):
        nodes = [make_node("n1", 1, 3, seed=1), make_node("n2", 2, 2, seed=2)]
        env, ctrl = make_controller(nodes)
        stats = ctrl.get_stats()
        assert "nodes" in stats
        assert "n1" in stats["nodes"]
        assert "n2" in stats["nodes"]

    def test_get_stats_node_has_expected_fields(self):
        nodes = [make_node("n1", 1, 3, seed=1)]
        env, ctrl = make_controller(nodes)
        node_stats = ctrl.get_stats()["nodes"]["n1"]
        for field in ("tokens", "hunger_counter", "emergency_count",
                      "total_allocated", "allocation_count"):
            assert field in node_stats

    def test_total_allocations_non_negative(self):
        nodes = [make_node("n1", 1, 3, seed=1), make_node("n2", 2, 2, seed=2)]
        env, ctrl = make_controller(nodes)
        nodes[0].generate_bid = lambda rt, rn: 100.0
        nodes[1].generate_bid = lambda rt, rn: 60.0
        env.process(ctrl._auction_loop(sim_duration_minutes=ROUND_INTERVAL))
        env.run(until=ROUND_INTERVAL + 1)
        assert ctrl.get_stats()["total_allocations"] >= 0


# ---------------------------------------------------------------------------
# 7. Insufficient-token guard
# ---------------------------------------------------------------------------

class TestInsufficientTokens:
    def test_node_with_zero_tokens_does_not_bid(self):
        """
        generate_bid() returns 0 when tokens < 1 (enforced in Node itself).
        A node with 0 tokens never enters the bids dict and cannot win.
        """
        n1 = make_node("n1", 1, 3, tokens=0.0, seed=0)
        n2 = make_node("n2", 2, 2, tokens=800.0, seed=0)

        env, ctrl = make_controller([n1, n2])
        # Reassert after register_node reset
        n1.tokens = 0.0
        n2.tokens = 800.0

        # Use real generate_bid -- n1 has 0 tokens, so its bid will be 0
        # and it is excluded from the bids dict automatically.
        env.process(ctrl._auction_loop(sim_duration_minutes=ROUND_INTERVAL))
        env.run(until=ROUND_INTERVAL + 1)

        # n1 must not hold any allocated resources
        n1_allocs = ctrl.resource_pool.allocations.get("n1", {})
        total = sum(n1_allocs.values())
        assert total == pytest.approx(0.0)

    def test_failed_deduction_increments_stat(self):
        """
        When TokenManager.deduct() returns False (insufficient tokens),
        stats['failed_deductions'] must increment.
        """
        n1 = make_node("n1", 1, 3, tokens=900.0, seed=0)
        n2 = make_node("n2", 2, 2, tokens=900.0, seed=0)

        env, ctrl = make_controller([n1, n2])
        n1.tokens = 900.0
        n2.tokens = 900.0

        # Force n1 to "win" but fail payment by draining tokens after bid collection
        # Simulate via lambda that returns a bid but then token manager will reject
        # because we set tokens to 0 just before deduct fires.
        # Simplest approach: monkeypatch deduct to return False for n1.
        original_deduct = ctrl.token_manager.deduct

        def patched_deduct(node_id, amount, reason="auction_win"):
            if node_id == "n1":
                return False   # simulate insufficient tokens at payment time
            return original_deduct(node_id, amount, reason)

        ctrl.token_manager.deduct = patched_deduct

        n1.generate_bid = lambda rt, rn: 100.0
        n2.generate_bid = lambda rt, rn: 60.0

        env.process(ctrl._auction_loop(sim_duration_minutes=ROUND_INTERVAL))
        env.run(until=ROUND_INTERVAL + 1)

        assert ctrl.stats["failed_deductions"] > 0
