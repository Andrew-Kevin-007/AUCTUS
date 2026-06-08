"""
tests/auction/test_sfg.py

Pytest tests for src.auction.starvation_floor.SFGManager.

Run with:
    pytest tests/auction/test_sfg.py -v
"""

import pytest
import numpy as np

from src.core.node           import Node
from src.core.resource_pool  import ResourcePool
from src.core.token_manager  import TokenManager
from src.core.audit_log      import AuditLog, SFG_ALLOCATE
from src.auction.vcg         import VCGAuction
from src.auction.starvation_floor import (
    SFGManager,
    STARVATION_WINDOW,
    RESOURCE_UNITS,
    _SFG_PRICE_FLOOR,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

CLEARING_PRICE = 50.0
RESOURCE = "CPU"
ROUND = 1


def make_node(node_id, tokens=500.0, hunger=0, seed=42):
    node = Node(
        node_id=node_id,
        criticality=2,
        tier=2,
        sla_deadline=500.0,
        current_task_progress=0.5,
        current_sim_time=100.0,
        T0=1000.0,
        rng=np.random.default_rng(seed),
    )
    node.tokens = tokens
    node.hunger_counter = hunger
    return node


def make_env(nodes_cfg):
    """
    nodes_cfg: list of (node_id, tokens, hunger_counter)
    Returns (sfg, nodes_dict, pool, tm, audit, vcg)
    """
    nodes = {}
    pool  = ResourcePool()
    tm    = TokenManager(T0=1000.0)
    audit = AuditLog(creation_timestamp=1_700_000_000.0)
    vcg   = VCGAuction(initial_clearing_prices={RESOURCE: CLEARING_PRICE})

    for nid, tokens, hunger in nodes_cfg:
        n = make_node(nid, tokens=tokens, hunger=hunger)
        tm.register_node(n)
        n.tokens = tokens           # reassert after register reset
        n.hunger_counter = hunger   # register doesn't touch hunger; explicit anyway
        nodes[nid] = n

    sfg = SFGManager(
        nodes=nodes,
        resource_pool=pool,
        token_manager=tm,
        audit_log=audit,
        vcg=vcg,
    )
    return sfg, nodes, pool, tm, audit, vcg


# ---------------------------------------------------------------------------
# 1. Trigger threshold
# ---------------------------------------------------------------------------

class TestTriggerThreshold:
    def test_hunger_at_W_triggers_sfg(self):
        """hunger_counter == STARVATION_WINDOW (10) must trigger."""
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", 500.0, STARVATION_WINDOW)]
        )
        sfg.check_and_apply(round_num=ROUND)
        assert sfg.sfg_events_total == 1

    def test_hunger_above_W_triggers_sfg(self):
        """hunger_counter > W must also trigger."""
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", 500.0, STARVATION_WINDOW + 5)]
        )
        sfg.check_and_apply(round_num=ROUND)
        assert sfg.sfg_events_total == 1

    def test_hunger_below_W_does_not_trigger(self):
        """hunger_counter == W-1 must NOT trigger SFG."""
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", 500.0, STARVATION_WINDOW - 1)]
        )
        sfg.check_and_apply(round_num=ROUND)
        assert sfg.sfg_events_total == 0

    def test_hunger_zero_does_not_trigger(self):
        sfg, nodes, pool, tm, audit, vcg = make_env([("n1", 500.0, 0)])
        sfg.check_and_apply(round_num=ROUND)
        assert sfg.sfg_events_total == 0

    def test_multiple_starving_nodes_all_get_sfg(self):
        """Each qualifying node should receive one SFG allocation per round."""
        sfg, nodes, pool, tm, audit, vcg = make_env([
            ("n1", 500.0, STARVATION_WINDOW),
            ("n2", 500.0, STARVATION_WINDOW),
            ("n3", 500.0, STARVATION_WINDOW - 1),  # should NOT trigger
        ])
        sfg.check_and_apply(round_num=ROUND)
        assert sfg.sfg_events_total == 2


# ---------------------------------------------------------------------------
# 2. Token deduction
# ---------------------------------------------------------------------------

class TestTokenDeduction:
    def test_sfg_deducts_clearing_price_from_node(self):
        """SFG price = last clearing price, deducted via TokenManager."""
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", 500.0, STARVATION_WINDOW)]
        )
        tokens_before = nodes["n1"].tokens
        sfg.execute_sfg_allocation(nodes["n1"], RESOURCE, ROUND)
        assert nodes["n1"].tokens == pytest.approx(tokens_before - CLEARING_PRICE)

    def test_sfg_deduction_recorded_in_tm_transaction_log(self):
        """TokenManager transaction log must contain the SFG deduct entry."""
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", 500.0, STARVATION_WINDOW)]
        )
        sfg.execute_sfg_allocation(nodes["n1"], RESOURCE, ROUND)
        reasons = [e["reason"] for e in tm.transaction_log]
        assert "sfg_allocation" in reasons

    def test_sfg_uses_floor_price_when_market_never_cleared(self):
        """If last clearing price is 0, SFG must use the floor price of 1.0."""
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", 500.0, STARVATION_WINDOW)]
        )
        # Override VCG price to 0
        vcg.last_clearing_price[RESOURCE] = 0.0
        tokens_before = nodes["n1"].tokens
        sfg.execute_sfg_allocation(nodes["n1"], RESOURCE, ROUND)
        assert nodes["n1"].tokens == pytest.approx(tokens_before - _SFG_PRICE_FLOOR)

    def test_no_double_deduction(self):
        """record_win is called with payment=0.0; only TokenManager deducts."""
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", 500.0, STARVATION_WINDOW)]
        )
        tokens_before = nodes["n1"].tokens
        sfg.execute_sfg_allocation(nodes["n1"], RESOURCE, ROUND)
        # Deduction must be exactly one SFG price, not two
        assert nodes["n1"].tokens == pytest.approx(tokens_before - CLEARING_PRICE)


# ---------------------------------------------------------------------------
# 3. Reserve pool usage
# ---------------------------------------------------------------------------

class TestReservePoolUsage:
    def test_sfg_draws_from_reserve_not_market(self):
        """Reserve pool balance must decrease; market available must be unchanged."""
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", 500.0, STARVATION_WINDOW)]
        )
        market_before  = pool.available[RESOURCE]
        reserve_before = pool.reserve_pool_balance[RESOURCE]

        sfg.execute_sfg_allocation(nodes["n1"], RESOURCE, ROUND)

        assert pool.available[RESOURCE] == pytest.approx(market_before)
        assert pool.reserve_pool_balance[RESOURCE] == pytest.approx(
            reserve_before - RESOURCE_UNITS[RESOURCE]
        )

    def test_node_appears_in_pool_allocations_after_sfg(self):
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", 500.0, STARVATION_WINDOW)]
        )
        sfg.execute_sfg_allocation(nodes["n1"], RESOURCE, ROUND)
        assert "n1" in pool.allocations
        assert pool.allocations["n1"].get(RESOURCE, 0) == pytest.approx(
            RESOURCE_UNITS[RESOURCE]
        )

    def test_sfg_returns_false_when_reserve_exhausted(self):
        """If reserve pool has no capacity, execute_sfg_allocation returns False."""
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", 500.0, STARVATION_WINDOW)]
        )
        # Drain the entire reserve pool
        pool.reserve_pool_balance[RESOURCE] = 0.0

        result = sfg.execute_sfg_allocation(nodes["n1"], RESOURCE, ROUND)
        assert result is False

    def test_check_and_apply_skips_node_when_reserve_exhausted(self):
        """check_and_apply must not crash and must not trigger if reserve is empty."""
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", 500.0, STARVATION_WINDOW)]
        )
        pool.reserve_pool_balance[RESOURCE] = 0.0
        sfg.check_and_apply(round_num=ROUND)
        assert sfg.sfg_events_total == 0


# ---------------------------------------------------------------------------
# 4. Insufficient tokens
# ---------------------------------------------------------------------------

class TestInsufficientTokens:
    def test_returns_false_when_tokens_below_sfg_price(self):
        """Starving node with no tokens must not receive allocation."""
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", 0.0, STARVATION_WINDOW)]    # 0 tokens
        )
        result = sfg.execute_sfg_allocation(nodes["n1"], RESOURCE, ROUND)
        assert result is False

    def test_no_pool_change_when_tokens_insufficient(self):
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", 0.0, STARVATION_WINDOW)]
        )
        reserve_before = pool.reserve_pool_balance[RESOURCE]
        sfg.execute_sfg_allocation(nodes["n1"], RESOURCE, ROUND)
        assert pool.reserve_pool_balance[RESOURCE] == pytest.approx(reserve_before)

    def test_no_audit_entry_on_failure(self):
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", 0.0, STARVATION_WINDOW)]
        )
        sfg.execute_sfg_allocation(nodes["n1"], RESOURCE, ROUND)
        assert len(audit.chain) == 0

    def test_sfg_events_total_unchanged_on_failure(self):
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", 0.0, STARVATION_WINDOW)]
        )
        sfg.execute_sfg_allocation(nodes["n1"], RESOURCE, ROUND)
        assert sfg.sfg_events_total == 0

    def test_exactly_sfg_price_tokens_succeeds(self):
        """Node with exactly sfg_price tokens must succeed."""
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", CLEARING_PRICE, STARVATION_WINDOW)]
        )
        result = sfg.execute_sfg_allocation(nodes["n1"], RESOURCE, ROUND)
        assert result is True


# ---------------------------------------------------------------------------
# 5. Hunger counter reset
# ---------------------------------------------------------------------------

class TestHungerCounterReset:
    def test_hunger_resets_to_zero_after_sfg(self):
        """record_win inside execute_sfg_allocation resets hunger_counter."""
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", 500.0, STARVATION_WINDOW)]
        )
        sfg.execute_sfg_allocation(nodes["n1"], RESOURCE, ROUND)
        assert nodes["n1"].hunger_counter == 0

    def test_hunger_unchanged_on_failed_sfg(self):
        """Failed SFG (no tokens) must not reset hunger_counter."""
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", 0.0, STARVATION_WINDOW)]
        )
        sfg.execute_sfg_allocation(nodes["n1"], RESOURCE, ROUND)
        assert nodes["n1"].hunger_counter == STARVATION_WINDOW

    def test_check_and_apply_resets_hunger_for_qualifying_nodes(self):
        sfg, nodes, pool, tm, audit, vcg = make_env([
            ("n1", 500.0, STARVATION_WINDOW),
            ("n2", 500.0, STARVATION_WINDOW - 1),
        ])
        sfg.check_and_apply(round_num=ROUND)
        assert nodes["n1"].hunger_counter == 0        # triggered, reset
        assert nodes["n2"].hunger_counter == STARVATION_WINDOW - 1  # unchanged


# ---------------------------------------------------------------------------
# 6. sfg_events_total counter
# ---------------------------------------------------------------------------

class TestSfgEventsTotal:
    def test_increments_by_one_per_successful_sfg(self):
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", 500.0, STARVATION_WINDOW)]
        )
        sfg.execute_sfg_allocation(nodes["n1"], RESOURCE, ROUND)
        assert sfg.sfg_events_total == 1

    def test_increments_for_each_starving_node(self):
        sfg, nodes, pool, tm, audit, vcg = make_env([
            ("n1", 500.0, STARVATION_WINDOW),
            ("n2", 500.0, STARVATION_WINDOW),
        ])
        sfg.check_and_apply(round_num=ROUND)
        assert sfg.sfg_events_total == 2

    def test_does_not_increment_on_failure(self):
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", 0.0, STARVATION_WINDOW)]
        )
        sfg.execute_sfg_allocation(nodes["n1"], RESOURCE, ROUND)
        assert sfg.sfg_events_total == 0

    def test_cumulative_across_multiple_rounds(self):
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", 5000.0, STARVATION_WINDOW)]
        )
        for rn in range(3):
            # Re-starve node between rounds for repeated triggering.
            # Also deallocate the prior SFG slot back to the reserve pool so
            # the 10-core reserve (10% of 100) doesn't exhaust across iterations
            # (3 x 4 cores = 12 > 10).
            if rn > 0:
                pool.reserve_pool_balance[RESOURCE] += RESOURCE_UNITS[RESOURCE]
                pool.allocations.get("n1", {}).pop(RESOURCE, None)
                if not pool.allocations.get("n1"):
                    pool.allocations.pop("n1", None)
            nodes["n1"].hunger_counter = STARVATION_WINDOW
            sfg.execute_sfg_allocation(nodes["n1"], RESOURCE, rn + 1)
        assert sfg.sfg_events_total == 3



# ---------------------------------------------------------------------------
# 7. Audit log
# ---------------------------------------------------------------------------

class TestAuditLog:
    def test_sfg_allocate_event_appended(self):
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", 500.0, STARVATION_WINDOW)]
        )
        sfg.execute_sfg_allocation(nodes["n1"], RESOURCE, ROUND)
        assert len(audit.chain) == 1
        assert audit.chain[0]["event_type"] == SFG_ALLOCATE

    def test_audit_payload_contains_hunger_counter_was(self):
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", 500.0, STARVATION_WINDOW)]
        )
        sfg.execute_sfg_allocation(nodes["n1"], RESOURCE, ROUND)
        payload = audit.chain[0]["payload"]
        assert payload["hunger_counter_was"] == STARVATION_WINDOW

    def test_audit_payload_contains_sfg_price(self):
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", 500.0, STARVATION_WINDOW)]
        )
        sfg.execute_sfg_allocation(nodes["n1"], RESOURCE, ROUND)
        payload = audit.chain[0]["payload"]
        assert payload["sfg_price"] == pytest.approx(CLEARING_PRICE)

    def test_audit_integrity_after_sfg(self):
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", 500.0, STARVATION_WINDOW)]
        )
        sfg.execute_sfg_allocation(nodes["n1"], RESOURCE, ROUND)
        assert audit.verify_integrity() is True


# ---------------------------------------------------------------------------
# 8. get_starvation_stats
# ---------------------------------------------------------------------------

class TestGetStarvationStats:
    def test_returns_dict_with_expected_keys(self):
        sfg, nodes, pool, tm, audit, vcg = make_env([("n1", 500.0, 0)])
        stats = sfg.get_starvation_stats()
        for key in ("sfg_events_total", "currently_starving", "max_hunger"):
            assert key in stats

    def test_currently_starving_lists_nodes_at_threshold(self):
        sfg, nodes, pool, tm, audit, vcg = make_env([
            ("n1", 500.0, STARVATION_WINDOW),
            ("n2", 500.0, STARVATION_WINDOW - 1),
        ])
        stats = sfg.get_starvation_stats()
        assert "n1" in stats["currently_starving"]
        assert "n2" not in stats["currently_starving"]

    def test_max_hunger_reflects_highest_counter(self):
        sfg, nodes, pool, tm, audit, vcg = make_env([
            ("n1", 500.0, 3),
            ("n2", 500.0, 15),
            ("n3", 500.0, 7),
        ])
        assert sfg.get_starvation_stats()["max_hunger"] == 15

    def test_max_hunger_zero_when_no_nodes(self):
        sfg = SFGManager(
            nodes={},
            resource_pool=ResourcePool(),
            token_manager=TokenManager(),
            audit_log=AuditLog(creation_timestamp=0.0),
            vcg=VCGAuction(),
        )
        assert sfg.get_starvation_stats()["max_hunger"] == 0

    def test_sfg_events_total_reflects_cumulative_count(self):
        sfg, nodes, pool, tm, audit, vcg = make_env(
            [("n1", 500.0, STARVATION_WINDOW)]
        )
        sfg.execute_sfg_allocation(nodes["n1"], RESOURCE, ROUND)
        assert sfg.get_starvation_stats()["sfg_events_total"] == 1
