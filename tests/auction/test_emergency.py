"""
tests/auction/test_emergency.py

Pytest tests for src.auction.emergency.ERPManager.

Run with:
    pytest tests/auction/test_emergency.py -v
"""

import pytest
import numpy as np

from src.core.node          import Node
from src.core.resource_pool import ResourcePool
from src.core.token_manager import TokenManager
from src.core.audit_log     import AuditLog, ERP_DECLARE, ERP_PREEMPT, RESERVE_TOPUP
from src.auction.vcg        import VCGAuction
from src.auction.emergency  import ERPManager, RESOURCE_UNITS, _ERP_MULTIPLIER, _PREEMPTED_REFUND, _RESERVE_CREDIT


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

CLEARING_PRICE = 100.0   # fixed seed price for deterministic arithmetic
RESOURCE = "CPU"
ROUND = 1


def make_node(node_id, criticality, tokens=1000.0, seed=42,
              sla_breach_prob_override=None):
    """
    Create a Node.  If sla_breach_prob_override is set, monkeypatch
    sla_breach_probability() to return that value so we can force
    ERP scores above/below theta deterministically.
    """
    node = Node(
        node_id=node_id,
        criticality=criticality,
        tier=2,
        sla_deadline=100.0,
        current_task_progress=0.0,    # behind schedule by default
        current_sim_time=90.0,        # 90% elapsed -> sla_prob=0.9
        T0=1000.0,
        rng=np.random.default_rng(seed),
    )
    node.tokens = tokens
    if sla_breach_prob_override is not None:
        _val = sla_breach_prob_override
        node.sla_breach_probability = lambda: _val
    return node


def make_env(requester_tokens=2000.0, target_tokens=500.0):
    """
    Wire up a complete ERP environment.
    Requester: criticality=3, high SLA breach prob -> high ERP score.
    Target:    criticality=1, on-schedule -> low ERP score.
    Target holds CPU in resource pool.
    Returns (erp_manager, requester, target, pool, tm, audit, vcg).
    """
    requester = make_node("req", criticality=3, tokens=requester_tokens,
                          sla_breach_prob_override=0.9)
    target    = make_node("tgt", criticality=1, tokens=target_tokens,
                          sla_breach_prob_override=0.0)

    pool  = ResourcePool()
    tm    = TokenManager(T0=1000.0)
    audit = AuditLog(creation_timestamp=1_700_000_000.0)
    vcg   = VCGAuction(initial_clearing_prices={RESOURCE: CLEARING_PRICE})

    tm.register_node(requester)
    tm.register_node(target)

    # Reassert token balances (register_node resets to T0)
    requester.tokens = requester_tokens
    target.tokens    = target_tokens

    # Give target a CPU allocation to preempt
    pool.allocate("tgt", RESOURCE, RESOURCE_UNITS[RESOURCE])

    erp = ERPManager(
        nodes={"req": requester, "tgt": target},
        resource_pool=pool,
        token_manager=tm,
        audit_log=audit,
        vcg=vcg,
    )
    return erp, requester, target, pool, tm, audit, vcg


# ---------------------------------------------------------------------------
# 1. Token accounting
# ---------------------------------------------------------------------------

class TestTokenAccounting:
    def test_requester_deducted_1_5x_clearing_price(self):
        erp, req, tgt, pool, tm, audit, vcg = make_env(requester_tokens=2000.0)
        tokens_before = req.tokens
        erp.execute_erp(req, RESOURCE, ROUND)
        expected_deduction = _ERP_MULTIPLIER * CLEARING_PRICE   # 1.5 * 100 = 150
        assert req.tokens == pytest.approx(tokens_before - expected_deduction)

    def test_preempted_node_refunded_0_70x(self):
        erp, req, tgt, pool, tm, audit, vcg = make_env(target_tokens=500.0)
        tokens_before = tgt.tokens
        erp.execute_erp(req, RESOURCE, ROUND)
        expected_refund = _PREEMPTED_REFUND * CLEARING_PRICE    # 0.70 * 100 = 70
        assert tgt.tokens == pytest.approx(tokens_before + expected_refund)

    def test_reserve_token_balance_increases_by_0_30x(self):
        erp, req, tgt, pool, tm, audit, vcg = make_env()
        balance_before = pool.reserve_token_balance
        erp.execute_erp(req, RESOURCE, ROUND)
        expected_credit = _RESERVE_CREDIT * CLEARING_PRICE      # 0.30 * 100 = 30
        assert pool.reserve_token_balance == pytest.approx(balance_before + expected_credit)

    def test_token_accounting_closes(self):
        """
        1.5x out from requester == 0.70x refund to preempted + 0.30x reserve.
        Net system token change must equal -1.5x + 0.70x + 0.30x = -0.5x
        (the 0.5x premium is extracted from the economy as the ERP fee).
        """
        erp, req, tgt, pool, tm, audit, vcg = make_env(
            requester_tokens=2000.0, target_tokens=500.0
        )
        total_before = req.tokens + tgt.tokens

        erp.execute_erp(req, RESOURCE, ROUND)

        total_after = req.tokens + tgt.tokens
        net_extracted = total_before - total_after

        # Net extraction = erp_price - refund = 1.5x - 0.70x = 0.80x
        # (0.30x goes to reserve_token_balance, not back to nodes)
        erp_price = _ERP_MULTIPLIER * CLEARING_PRICE
        refund    = _PREEMPTED_REFUND * CLEARING_PRICE
        expected_net = erp_price - refund

        assert abs(net_extracted - expected_net) < 0.001

    def test_1_5x_equals_0_70x_plus_0_30x_plus_0_50x(self):
        """Mathematical closure: 1.5 = 0.70 + 0.30 + 0.50 (premium retained)."""
        assert abs(_ERP_MULTIPLIER - (_PREEMPTED_REFUND + _RESERVE_CREDIT + 0.50)) < 1e-9

    def test_reserve_credit_accumulates_across_events(self):
        """Multiple ERP events accumulate reserve_token_balance."""
        erp, req, tgt, pool, tm, audit, vcg = make_env(requester_tokens=5000.0)

        for i in range(3):
            if i > 0:
                # After each ERP, requester holds the slot.
                # Return it to pool then give it back to target so there is
                # something to preempt in the next iteration.
                pool.deallocate("req", RESOURCE, RESOURCE_UNITS[RESOURCE])
                pool.allocate("tgt", RESOURCE, RESOURCE_UNITS[RESOURCE])
                # Temporarily lower emergency_count to allow re-declaration
                req.emergency_count -= 1
            erp.execute_erp(req, RESOURCE, round_num=i + 1)

        expected = _RESERVE_CREDIT * CLEARING_PRICE * 3
        assert pool.reserve_token_balance == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 2. execute_erp failure paths
# ---------------------------------------------------------------------------

class TestExecuteErpFailures:
    def test_returns_false_when_no_holder(self):
        erp, req, tgt, pool, tm, audit, vcg = make_env()
        # Deallocate the target's holding so nothing to preempt
        pool.deallocate("tgt", RESOURCE, RESOURCE_UNITS[RESOURCE])
        result = erp.execute_erp(req, RESOURCE, ROUND)
        assert result is False

    def test_returns_false_when_requester_insufficient_tokens(self):
        erp, req, tgt, pool, tm, audit, vcg = make_env(requester_tokens=10.0)
        # erp_price = 1.5 * 100 = 150; requester has only 10
        result = erp.execute_erp(req, RESOURCE, ROUND)
        assert result is False

    def test_requester_tokens_unchanged_on_failure_no_holder(self):
        erp, req, tgt, pool, tm, audit, vcg = make_env(requester_tokens=2000.0)
        pool.deallocate("tgt", RESOURCE, RESOURCE_UNITS[RESOURCE])
        tokens_before = req.tokens
        erp.execute_erp(req, RESOURCE, ROUND)
        assert req.tokens == pytest.approx(tokens_before)

    def test_requester_tokens_unchanged_on_failure_insufficient(self):
        erp, req, tgt, pool, tm, audit, vcg = make_env(requester_tokens=10.0)
        tokens_before = req.tokens
        erp.execute_erp(req, RESOURCE, ROUND)
        assert req.tokens == pytest.approx(tokens_before)

    def test_target_tokens_unchanged_on_failure(self):
        erp, req, tgt, pool, tm, audit, vcg = make_env()
        pool.deallocate("tgt", RESOURCE, RESOURCE_UNITS[RESOURCE])
        tokens_before = tgt.tokens
        erp.execute_erp(req, RESOURCE, ROUND)
        assert tgt.tokens == pytest.approx(tokens_before)

    def test_returns_false_self_preemption(self):
        """Requester cannot preempt itself."""
        erp, req, tgt, pool, tm, audit, vcg = make_env(requester_tokens=2000.0)
        # Give requester the CPU slot instead of target
        pool.deallocate("tgt", RESOURCE, RESOURCE_UNITS[RESOURCE])
        pool.allocate("req", RESOURCE, RESOURCE_UNITS[RESOURCE])
        result = erp.execute_erp(req, RESOURCE, ROUND)
        assert result is False


# ---------------------------------------------------------------------------
# 3. Resource reallocation
# ---------------------------------------------------------------------------

class TestResourceReallocation:
    def test_target_loses_resource_after_erp(self):
        erp, req, tgt, pool, tm, audit, vcg = make_env()
        erp.execute_erp(req, RESOURCE, ROUND)
        tgt_allocs = pool.allocations.get("tgt", {})
        assert tgt_allocs.get(RESOURCE, 0.0) == pytest.approx(0.0)

    def test_requester_gains_resource_after_erp(self):
        erp, req, tgt, pool, tm, audit, vcg = make_env()
        erp.execute_erp(req, RESOURCE, ROUND)
        req_allocs = pool.allocations.get("req", {})
        assert req_allocs.get(RESOURCE, 0.0) == pytest.approx(RESOURCE_UNITS[RESOURCE])

    def test_total_pool_units_conserved(self):
        """Preemption is a transfer, not a creation/destruction of capacity."""
        erp, req, tgt, pool, tm, audit, vcg = make_env()
        before_available = pool.available[RESOURCE]
        erp.execute_erp(req, RESOURCE, ROUND)
        # available should be unchanged: same units, just different holder
        assert pool.available[RESOURCE] == pytest.approx(before_available)


# ---------------------------------------------------------------------------
# 4. Node state updates
# ---------------------------------------------------------------------------

class TestNodeStateUpdates:
    def test_requester_emergency_count_increments(self):
        erp, req, tgt, pool, tm, audit, vcg = make_env()
        count_before = req.emergency_count
        erp.execute_erp(req, RESOURCE, ROUND)
        assert req.emergency_count == count_before + 1

    def test_requester_allocation_history_updated(self):
        erp, req, tgt, pool, tm, audit, vcg = make_env()
        erp.execute_erp(req, RESOURCE, ROUND)
        assert len(req.allocation_history) == 1
        assert req.allocation_history[0][1] == RESOURCE

    def test_requester_hunger_counter_reset(self):
        erp, req, tgt, pool, tm, audit, vcg = make_env()
        req.hunger_counter = 7
        erp.execute_erp(req, RESOURCE, ROUND)
        assert req.hunger_counter == 0

    def test_erp_events_this_period_tracked(self):
        erp, req, tgt, pool, tm, audit, vcg = make_env()
        erp.execute_erp(req, RESOURCE, ROUND)
        assert erp.erp_events_this_period.get("req", 0) == 1


# ---------------------------------------------------------------------------
# 5. Monthly cap enforcement
# ---------------------------------------------------------------------------

class TestMonthlyCap:
    def test_fourth_emergency_returns_false_from_can_declare(self):
        """
        After 3 successful declarations, can_declare_emergency must return
        False (cap hit) even if ERP score is above theta.
        """
        erp, req, tgt, pool, tm, audit, vcg = make_env(requester_tokens=50_000.0)

        # Execute 3 successful ERPs
        for i in range(3):
            # Each ERP moves the slot from tgt -> req.
            # Return it to req->pool->tgt so there is a fresh target next round.
            if i > 0:
                pool.deallocate("req", RESOURCE, RESOURCE_UNITS[RESOURCE])
                pool.allocate("tgt", RESOURCE, RESOURCE_UNITS[RESOURCE])
            success = erp.execute_erp(req, RESOURCE, round_num=i + 1)
            assert success is True, f"Expected ERP {i+1} to succeed"

        assert req.emergency_count == 3

        # Fourth attempt: can_declare_emergency must reject (cap = 3)
        clearing_price = vcg.get_last_clearing_price(RESOURCE)
        assert req.can_declare_emergency(clearing_price) is False

    def test_period_reset_clears_erp_events_this_period(self):
        erp, req, tgt, pool, tm, audit, vcg = make_env()
        erp.execute_erp(req, RESOURCE, ROUND)
        assert erp.erp_events_this_period.get("req", 0) == 1
        erp.period_reset()
        assert erp.erp_events_this_period == {}


# ---------------------------------------------------------------------------
# 6. Audit log
# ---------------------------------------------------------------------------

class TestAuditLog:
    def test_erp_declare_appended(self):
        erp, req, tgt, pool, tm, audit, vcg = make_env()
        erp.execute_erp(req, RESOURCE, ROUND)
        events = [e for e in audit.chain if e["event_type"] == ERP_DECLARE]
        assert len(events) == 1
        assert events[0]["node_id"] == "req"

    def test_erp_preempt_appended(self):
        erp, req, tgt, pool, tm, audit, vcg = make_env()
        erp.execute_erp(req, RESOURCE, ROUND)
        events = [e for e in audit.chain if e["event_type"] == ERP_PREEMPT]
        assert len(events) == 1
        assert events[0]["node_id"] == "tgt"

    def test_reserve_topup_appended(self):
        erp, req, tgt, pool, tm, audit, vcg = make_env()
        erp.execute_erp(req, RESOURCE, ROUND)
        events = [e for e in audit.chain if e["event_type"] == RESERVE_TOPUP]
        assert len(events) == 1
        assert events[0]["payload"]["source"] == "ERP"

    def test_audit_chain_integrity_after_erp(self):
        erp, req, tgt, pool, tm, audit, vcg = make_env()
        erp.execute_erp(req, RESOURCE, ROUND)
        assert audit.verify_integrity() is True

    def test_three_entries_per_erp_event(self):
        """Each ERP produces exactly 3 audit entries: DECLARE + PREEMPT + TOPUP."""
        erp, req, tgt, pool, tm, audit, vcg = make_env()
        erp.execute_erp(req, RESOURCE, ROUND)
        assert len(audit.chain) == 3


# ---------------------------------------------------------------------------
# 7. process_emergency_requests
# ---------------------------------------------------------------------------

class TestProcessEmergencyRequests:
    def test_returns_empty_list_when_no_eligible_nodes(self):
        erp, req, tgt, pool, tm, audit, vcg = make_env(requester_tokens=10.0)
        # Requester has too few tokens; no node is eligible
        result = erp.process_emergency_requests(round_num=1)
        assert result == []

    def test_returns_requester_id_on_success(self):
        erp, req, tgt, pool, tm, audit, vcg = make_env(requester_tokens=2000.0)
        result = erp.process_emergency_requests(round_num=1)
        assert "req" in result

    def test_at_most_one_erp_per_round(self):
        """Even with two eligible nodes, only one ERP fires per round."""
        req1 = make_node("req1", criticality=3, tokens=5000.0,
                         sla_breach_prob_override=0.9)
        req2 = make_node("req2", criticality=3, tokens=5000.0,
                         sla_breach_prob_override=0.85)
        tgt  = make_node("tgt",  criticality=1, tokens=500.0,
                         sla_breach_prob_override=0.0)

        pool  = ResourcePool()
        tm    = TokenManager(T0=1000.0)
        audit = AuditLog(creation_timestamp=1_700_000_000.0)
        vcg   = VCGAuction(initial_clearing_prices={"CPU": CLEARING_PRICE})

        for n in [req1, req2, tgt]:
            tm.register_node(n)
            n.tokens = n.tokens   # keep pre-register value (already set)

        req1.tokens = 5000.0
        req2.tokens = 5000.0
        tgt.tokens  = 500.0

        pool.allocate("tgt", RESOURCE, RESOURCE_UNITS[RESOURCE])

        erp = ERPManager(
            nodes={"req1": req1, "req2": req2, "tgt": tgt},
            resource_pool=pool,
            token_manager=tm,
            audit_log=audit,
            vcg=vcg,
        )

        result = erp.process_emergency_requests(round_num=1)
        assert len(result) <= 1
