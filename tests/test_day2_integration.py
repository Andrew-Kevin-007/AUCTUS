"""
tests/test_day2_integration.py

Full 10-node, 1-day (288-round) integration test wiring:
  AuctionController + ERPManager + SFGManager

Nodes 6-9 are designed to lose every normal round (tiny bids) so their
hunger_counter builds up and SFG must fire.

Run with:
    pytest tests/test_day2_integration.py -v -s
"""

import math
import pytest
import numpy as np
import simpy

from src.core.node           import Node
from src.core.resource_pool  import ResourcePool
from src.core.token_manager  import TokenManager
from src.core.audit_log      import AuditLog
from src.auction.vcg         import VCGAuction
from src.auction.controller  import AuctionController, ROUND_INTERVAL, RESOURCE_UNITS
from src.auction.emergency   import ERPManager
from src.auction.starvation_floor import SFGManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIM_MINUTES   = 1440          # 1 simulated day
EXPECTED_ROUNDS = SIM_MINUTES // ROUND_INTERVAL   # 288


# ---------------------------------------------------------------------------
# Extended controller that hooks ERP + SFG after each round
# ---------------------------------------------------------------------------

class IntegratedController(AuctionController):
    """
    Subclass of AuctionController that calls ERPManager and SFGManager
    after each round's VCG clearing.
    """

    def attach(self, erp_manager: ERPManager, sfg_manager: SFGManager) -> None:
        self._erp_manager = erp_manager
        self._sfg_manager = sfg_manager

    def run_round(self) -> None:
        # Run normal VCG auction (includes controller's internal SFG check)
        super().run_round()
        rn = self.round_num

        # Hook external ERP manager
        triggered = self._erp_manager.process_emergency_requests(rn)
        self.stats["erp_events"] += len(triggered)

        # Hook external SFG manager (dedicated SFGManager, independent of
        # the controller's built-in _check_starvation which runs per-resource)
        events_before = self._sfg_manager.sfg_events_total
        self._sfg_manager.check_and_apply(rn)
        self.stats["sfg_events"] += (
            self._sfg_manager.sfg_events_total - events_before
        )


# ---------------------------------------------------------------------------
# Fixture: build and run the full simulation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sim_result():
    """
    Build, run, and return the full 1-day simulation.
    Scoped to module so it runs once and all tests share the output.
    """
    rng = np.random.default_rng(2024)

    # ── Build nodes ────────────────────────────────────────────────────
    nodes = []

    # nodes 0-1: criticality=1, tier=1 — aggressive, high SLA urgency
    for i in range(2):
        n = Node(
            node_id=f"node_{i}",
            criticality=1,
            tier=1,
            sla_deadline=200.0,
            current_task_progress=0.1,   # well behind schedule → high urgency
            current_sim_time=150.0,
            T0=1000.0,
            rng=np.random.default_rng(i),
        )
        nodes.append(n)

    # nodes 2-5: criticality=2, tier=2 — moderate
    for i in range(2, 6):
        n = Node(
            node_id=f"node_{i}",
            criticality=2,
            tier=2,
            sla_deadline=500.0,
            current_task_progress=0.4,
            current_sim_time=200.0,
            T0=1000.0,
            rng=np.random.default_rng(i),
        )
        nodes.append(n)

    # nodes 6-9: criticality=3, tier=3 — conservative; override to tiny bids
    for i in range(6, 10):
        n = Node(
            node_id=f"node_{i}",
            criticality=3,
            tier=3,
            sla_deadline=500.0,
            current_task_progress=0.5,
            current_sim_time=100.0,
            T0=1000.0,
            rng=np.random.default_rng(i),
        )
        # Override generate_bid: always bid max(1.0, tokens * 0.05)
        # so they consistently lose competitive rounds and build up hunger.
        # Closure captures node object via default-arg trick.
        def _tiny_bid(resource_type, current_round, _node=n):
            bid = max(1.0, _node.tokens * 0.05)
            _node.bid_history.append((current_round, resource_type, bid))
            return bid

        n.generate_bid = _tiny_bid
        nodes.append(n)

    # ── Infrastructure ─────────────────────────────────────────────────
    pool  = ResourcePool()
    tm    = TokenManager(T0=1000.0)
    audit = AuditLog(creation_timestamp=1_700_000_000.0)

    for node in nodes:
        tm.register_node(node)
        # register_node resets to T0; tokens stay at 1000 — fine for the sim

    # Shared VCG instance (controller creates its own internally; ERP/SFG need
    # a reference price — give them the controller's vcg after construction)
    env = simpy.Environment()

    ctrl = IntegratedController(
        env=env,
        nodes=nodes,
        resource_pool=pool,
        token_manager=tm,
        audit_log=audit,
        rng=np.random.default_rng(999),
    )

    node_map = {n.node_id: n for n in nodes}

    erp = ERPManager(
        nodes=node_map,
        resource_pool=pool,
        token_manager=tm,
        audit_log=audit,
        vcg=ctrl.vcg,          # share the controller's VCG instance
    )

    sfg = SFGManager(
        nodes=node_map,
        resource_pool=pool,
        token_manager=tm,
        audit_log=audit,
        vcg=ctrl.vcg,
    )

    ctrl.attach(erp, sfg)

    # ── Run ────────────────────────────────────────────────────────────
    ctrl.start(sim_duration_minutes=SIM_MINUTES)

    return {
        "ctrl":  ctrl,
        "nodes": nodes,
        "pool":  pool,
        "tm":    tm,
        "audit": audit,
        "erp":   erp,
        "sfg":   sfg,
    }


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

class TestDay2Integration:

    def test_correct_number_of_rounds(self, sim_result):
        ctrl = sim_result["ctrl"]
        assert ctrl.round_num == EXPECTED_ROUNDS, (
            f"Expected {EXPECTED_ROUNDS} rounds, got {ctrl.round_num}"
        )

    def test_at_least_one_allocation_occurred(self, sim_result):
        ctrl = sim_result["ctrl"]
        assert ctrl.stats["total_allocations"] > 0

    def test_no_node_has_negative_token_balance(self, sim_result):
        for node in sim_result["nodes"]:
            assert node.tokens >= 0.0, (
                f"{node.node_id} has negative tokens: {node.tokens:.4f}"
            )

    def test_reserve_pool_token_balance_non_negative(self, sim_result):
        pool = sim_result["pool"]
        assert pool.reserve_token_balance >= 0.0

    def test_reserve_pool_capacity_non_negative(self, sim_result):
        """Physical reserve pool balances must not go below 0."""
        pool = sim_result["pool"]
        for r, bal in pool.reserve_pool_balance.items():
            assert bal >= -1e-9, (
                f"Reserve pool balance for {r} went negative: {bal:.4f}"
            )

    def test_audit_log_integrity(self, sim_result):
        assert sim_result["audit"].verify_integrity() is True

    def test_sfg_fired_for_low_bid_nodes(self, sim_result):
        """
        Nodes 6-9 bid tiny amounts and should lose most rounds.
        After W=10 consecutive losses the SFG must fire at least once.
        We verify at least one SFG event occurred total (nodes 6-9 are the
        likely triggers, but we assert the global counter rather than per-node
        to keep the test robust to stochastic outcomes).
        """
        sfg = sim_result["sfg"]
        ctrl = sim_result["ctrl"]
        # Both the controller's built-in SFG and the external SFGManager run;
        # check at least one fired via either path.
        total_sfg = sfg.sfg_events_total + ctrl.stats.get("sfg_events", 0)
        assert total_sfg > 0, (
            "Expected at least one SFG event for the low-bid nodes (6-9), "
            f"got sfg_manager.sfg_events_total={sfg.sfg_events_total}, "
            f"ctrl.stats['sfg_events']={ctrl.stats.get('sfg_events', 0)}"
        )

    def test_pool_available_not_negative(self, sim_result):
        """Market available pool must never go below 0."""
        pool = sim_result["pool"]
        for r, avail in pool.available.items():
            assert avail >= -1e-9, (
                f"Market pool for {r} went negative: {avail:.4f}"
            )


# ---------------------------------------------------------------------------
# Summary table (captured by -s flag)
# ---------------------------------------------------------------------------

class TestDay2Summary:

    def test_print_summary(self, sim_result):
        ctrl  = sim_result["ctrl"]
        nodes = sim_result["nodes"]
        pool  = sim_result["pool"]
        sfg   = sim_result["sfg"]
        erp   = sim_result["erp"]

        stats = ctrl.get_stats()

        print("\n")
        print("=" * 75)
        print("  AUCTUS DAY-2 INTEGRATION  |  1 simulated day  |  10 nodes")
        print("=" * 75)

        # Per-node table
        header = f"  {'node_id':12s} {'tokens':>10s} {'cpu_alloc':>10s} {'hunger':>8s} {'erp_cnt':>8s}"
        print(header)
        print("  " + "-" * 53)

        allocation_totals = []
        for node in nodes:
            cpu_alloc = node.total_allocated.get("CPU", 0.0)
            allocation_totals.append(cpu_alloc)
            print(
                f"  {node.node_id:12s} "
                f"{node.tokens:>10.2f} "
                f"{cpu_alloc:>10.1f} "
                f"{node.hunger_counter:>8d} "
                f"{node.emergency_count:>8d}"
            )

        print("  " + "-" * 53)

        # Jain's Fairness Index on total CPU allocated
        n  = len(allocation_totals)
        s  = sum(allocation_totals)
        s2 = sum(x * x for x in allocation_totals)
        jain = (s * s) / (n * s2) if s2 > 0 else 0.0

        print(f"\n  Total rounds completed : {ctrl.round_num}")
        print(f"  Total allocations      : {stats['total_allocations']}")
        print(f"  ERP events (ctrl)      : {stats.get('erp_events', 0)}")
        print(f"  SFG events (ctrl)      : {stats.get('sfg_events', 0)}")
        print(f"  SFG events (manager)   : {sfg.sfg_events_total}")
        print(f"  Failed deductions      : {stats.get('failed_deductions', 0)}")
        print(f"  Jain Fairness Index    : {jain:.4f}  (1.0 = perfect)")
        print(f"  Audit log entries      : {len(sim_result['audit'])}")
        print(f"  Reserve token balance  : {pool.reserve_token_balance:.2f}")

        print("\n  Resource utilization at end of sim:")
        util = pool.get_utilization()
        for r, pct in util.items():
            bar = "#" * int(pct / 5)
            print(f"    {r:4s}  {pct:6.2f}%  {bar}")

        print("=" * 75)

        # Jain's index must be computable (no division by zero if any allocs)
        if s > 0:
            assert 0.0 < jain <= 1.0, f"Jain index out of range: {jain}"
