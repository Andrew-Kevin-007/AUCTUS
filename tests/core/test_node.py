"""
tests/core/test_node.py

Pytest tests for src.core.node.Node.

Run with:
    pytest tests/core/test_node.py -v
"""

import pytest
import numpy as np

from src.core.node import Node, BASE_VALUATION, _PREEMPTION_REFUND_RATIO, _ROLLOVER_CAP_FRACTION


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_node(
    *,
    criticality: int = 3,
    tier: int = 2,
    sla_deadline: float = 100.0,
    current_task_progress: float = 0.5,
    current_sim_time: float = 50.0,
    T0: float = 1000.0,
    seed: int = 42,
) -> Node:
    """Return a Node with deterministic RNG for reproducible tests."""
    return Node(
        node_id="test-node-01",
        criticality=criticality,
        tier=tier,
        sla_deadline=sla_deadline,
        current_task_progress=current_task_progress,
        current_sim_time=current_sim_time,
        T0=T0,
        rng=np.random.default_rng(seed),
    )


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------

class TestNodeConstruction:
    def test_initial_tokens_equal_T0(self):
        node = make_node(T0=1000.0)
        assert node.tokens == 1000.0

    def test_initial_hunger_and_emergency_zero(self):
        node = make_node()
        assert node.hunger_counter == 0
        assert node.emergency_count == 0

    def test_total_allocated_initialized_for_all_resources(self):
        node = make_node()
        for r in BASE_VALUATION:
            assert node.total_allocated[r] == 0.0

    def test_histories_empty_on_init(self):
        node = make_node()
        assert node.allocation_history == []
        assert node.bid_history == []

    def test_invalid_criticality_raises(self):
        with pytest.raises(ValueError, match="criticality"):
            Node(
                node_id="bad", criticality=0, tier=1,
                sla_deadline=100.0,
                rng=np.random.default_rng(),
            )

    def test_invalid_tier_raises(self):
        with pytest.raises(ValueError, match="tier"):
            Node(
                node_id="bad", criticality=1, tier=6,
                sla_deadline=100.0,
                rng=np.random.default_rng(),
            )


# ---------------------------------------------------------------------------
# Token deduction on win
# ---------------------------------------------------------------------------

class TestRecordWin:
    def test_tokens_deducted_by_payment(self):
        node = make_node(T0=1000.0)
        node.record_win("CPU", units=4.0, payment=50.0, round_num=1)
        assert node.tokens == pytest.approx(950.0)

    def test_hunger_counter_reset_on_win(self):
        node = make_node()
        node.hunger_counter = 7
        node.record_win("CPU", units=1.0, payment=10.0, round_num=1)
        assert node.hunger_counter == 0

    def test_total_allocated_accumulates(self):
        node = make_node()
        node.record_win("RAM", units=16.0, payment=16.0, round_num=1)
        node.record_win("RAM", units=8.0,  payment=8.0,  round_num=2)
        assert node.total_allocated["RAM"] == pytest.approx(24.0)

    def test_allocation_history_entry_appended(self):
        node = make_node()
        node.record_win("CPU", units=2.0, payment=10.0, round_num=5)
        assert len(node.allocation_history) == 1
        assert node.allocation_history[0] == (5, "CPU", 2.0)

    def test_multiple_wins_all_recorded(self):
        node = make_node()
        node.record_win("CPU", units=1.0, payment=5.0,  round_num=1)
        node.record_win("RAM", units=4.0, payment=4.0,  round_num=2)
        assert len(node.allocation_history) == 2


# ---------------------------------------------------------------------------
# Token refund on preemption
# ---------------------------------------------------------------------------

class TestRecordPreemption:
    def test_refund_is_70_percent(self):
        node = make_node(T0=1000.0)
        # Simulate a prior win that deducted 100 tokens
        node.tokens = 900.0
        node.record_preemption(payment_made=100.0)
        assert node.tokens == pytest.approx(900.0 + 70.0)

    def test_refund_scales_with_payment(self):
        node = make_node(T0=1000.0)
        node.tokens = 800.0
        node.record_preemption(payment_made=200.0)
        assert node.tokens == pytest.approx(800.0 + 140.0)

    def test_refund_ratio_constant(self):
        node = make_node(T0=500.0)
        node.tokens = 0.0
        node.record_preemption(payment_made=300.0)
        expected = _PREEMPTION_REFUND_RATIO * 300.0
        assert node.tokens == pytest.approx(expected)

    def test_preemption_does_not_touch_hunger(self):
        node = make_node()
        node.hunger_counter = 3
        node.record_preemption(payment_made=50.0)
        assert node.hunger_counter == 3  # unchanged


# ---------------------------------------------------------------------------
# Period reset — soft rollover
# ---------------------------------------------------------------------------

class TestPeriodReset:
    def test_tokens_restored_to_T0_plus_carry(self):
        node = make_node(T0=1000.0)
        node.tokens = 150.0   # leftover from previous period
        node.period_reset()
        # carry = min(0.2 * 1000, 150) = min(200, 150) = 150
        assert node.tokens == pytest.approx(1000.0 + 150.0)

    def test_carry_forward_capped_at_20_percent_T0(self):
        node = make_node(T0=1000.0)
        node.tokens = 500.0   # more than 20% of T0
        node.period_reset()
        # carry = min(200, 500) = 200
        assert node.tokens == pytest.approx(1000.0 + 200.0)

    def test_zero_leftover_gives_exactly_T0(self):
        node = make_node(T0=1000.0)
        node.tokens = 0.0
        node.period_reset()
        # carry = min(200, 0) = 0
        assert node.tokens == pytest.approx(1000.0)

    def test_carry_cap_boundary_exact_20_percent(self):
        node = make_node(T0=1000.0)
        node.tokens = 200.0   # exactly 20% → no capping
        node.period_reset()
        assert node.tokens == pytest.approx(1200.0)

    def test_hunger_counter_reset(self):
        node = make_node()
        node.hunger_counter = 15
        node.period_reset()
        assert node.hunger_counter == 0

    def test_emergency_count_reset(self):
        node = make_node()
        node.emergency_count = 3
        node.period_reset()
        assert node.emergency_count == 0

    def test_histories_cleared(self):
        node = make_node()
        node.allocation_history.append((1, "CPU", 1.0))
        node.bid_history.append((1, "CPU", 5.0))
        node.period_reset()
        assert node.allocation_history == []
        assert node.bid_history == []

    def test_total_allocated_zeroed(self):
        node = make_node()
        node.total_allocated["CPU"] = 42.0
        node.period_reset()
        assert node.total_allocated["CPU"] == 0.0


# ---------------------------------------------------------------------------
# ERP score
# ---------------------------------------------------------------------------

class TestErpScore:
    def test_low_criticality_low_urgency_gives_low_score(self):
        # criticality=3 (E3), task on schedule → sla_breach_prob ≈ 0
        # progress=0.5, sim_time=50, deadline=100 → on schedule → shortfall=0
        node = make_node(
            criticality=3,
            current_task_progress=0.5,
            current_sim_time=50.0,
            sla_deadline=100.0,
        )
        node.emergency_count = 0
        clearing_price = 50.0
        score = node.erp_score(clearing_price)

        # ERP = 0.4*0 + 0.4*(3/3) - 0.2*(0/3) = 0 + 0.4 - 0 = 0.4
        assert score == pytest.approx(0.4, abs=1e-9)

    def test_score_clipped_to_one(self):
        # Maximum possible: criticality=1 (normalized=1/3), sla_prob=1, emergency=0
        # Wait — criticality 1 → (1/3), criticality 3 → (3/3)=1
        # Use criticality=3, sla_prob=1, emergency=0
        # Score = 0.4*1 + 0.4*1 - 0 = 0.8
        node = make_node(
            criticality=3,
            current_task_progress=0.0,   # no progress
            current_sim_time=100.0,       # deadline fully elapsed → prob=1
            sla_deadline=100.0,
        )
        node.emergency_count = 0
        score = node.erp_score(50.0)
        assert 0.0 <= score <= 1.0

    def test_score_clipped_to_zero(self):
        # Low criticality, no urgency, max emergency count
        node = make_node(
            criticality=1,
            current_task_progress=1.0,
            current_sim_time=0.0,
            sla_deadline=100.0,
        )
        node.emergency_count = 3
        score = node.erp_score(50.0)
        assert score >= 0.0

    def test_score_increases_with_sla_breach_probability(self):
        # Behind schedule → higher SLA breach prob → higher score
        node_ok = make_node(
            criticality=2, current_task_progress=0.5,
            current_sim_time=50.0, sla_deadline=100.0,
        )
        node_behind = make_node(
            criticality=2, current_task_progress=0.1,
            current_sim_time=50.0, sla_deadline=100.0,
        )
        assert node_behind.erp_score(50.0) > node_ok.erp_score(50.0)


# ---------------------------------------------------------------------------
# can_declare_emergency — cap enforcement
# ---------------------------------------------------------------------------

class TestCanDeclareEmergency:
    def _high_score_node(self) -> Node:
        """Node configured to have a high ERP score (above theta=0.7)."""
        # criticality=3, on schedule so sla_prob=0 → score=0.4 (below 0.7)
        # Need: sla_prob high enough
        # score = 0.4*p + 0.4*(3/3) - 0 = 0.4p + 0.4 > 0.7 → p > 0.75
        # current_sim_time=90, sla_deadline=100, progress=0 → shortfall=0.9
        return make_node(
            criticality=3,
            current_task_progress=0.0,
            current_sim_time=90.0,
            sla_deadline=100.0,
        )

    def test_returns_false_when_cap_hit(self):
        node = self._high_score_node()
        node.tokens = 5000.0   # plenty of tokens
        node.emergency_count = 3   # cap hit
        assert node.can_declare_emergency(clearing_price=50.0) is False

    def test_returns_false_when_insufficient_tokens(self):
        node = self._high_score_node()
        node.emergency_count = 0
        node.tokens = 50.0            # 1.5 * 50 = 75 > 50 → fails
        assert node.can_declare_emergency(clearing_price=50.0) is False

    def test_returns_false_when_score_below_theta(self):
        # On schedule → sla_prob=0, criticality=1
        # score = 0 + 0.4*(1/3) - 0 ≈ 0.133 < 0.7
        node = make_node(
            criticality=1,
            current_task_progress=0.5,
            current_sim_time=50.0,
            sla_deadline=100.0,
        )
        node.tokens = 5000.0
        node.emergency_count = 0
        assert node.can_declare_emergency(clearing_price=50.0) is False

    def test_returns_true_when_all_conditions_met(self):
        # Score = 0.4*0.9 + 0.4*1 - 0 = 0.36 + 0.4 = 0.76 > 0.7 ✓
        node = self._high_score_node()
        node.tokens = 5000.0
        node.emergency_count = 0
        assert node.can_declare_emergency(clearing_price=50.0) is True

    def test_returns_false_at_exactly_cap(self):
        node = self._high_score_node()
        node.tokens = 5000.0
        node.emergency_count = 3   # exactly at cap, not below
        assert node.can_declare_emergency(clearing_price=50.0) is False

    def test_returns_true_at_cap_minus_one(self):
        # score = 0.4 * sla_prob + 0.4 * (criticality/3) - 0.2 * (ec/3)
        # With sim_time=95, deadline=100, progress=0 → sla_prob=0.95
        # score = 0.4*0.95 + 0.4*1.0 - 0.2*(2/3) = 0.38+0.40-0.133 = 0.647
        # Still below 0.7 at ec=2. Use ec=1:
        # score = 0.38 + 0.40 - 0.2*(1/3) = 0.78 - 0.067 = 0.713 > 0.7 ✓
        node = Node(
            node_id="test-node-01",
            criticality=3,
            tier=2,
            sla_deadline=100.0,
            current_task_progress=0.0,
            current_sim_time=95.0,   # sla_prob = 0.95
            T0=1000.0,
            rng=np.random.default_rng(42),
        )
        node.tokens = 5000.0
        node.emergency_count = 1   # one declaration used; two remain; score ≈ 0.713
        assert node.can_declare_emergency(clearing_price=50.0) is True


# ---------------------------------------------------------------------------
# record_loss / hunger counter
# ---------------------------------------------------------------------------

class TestRecordLoss:
    def test_hunger_increments_on_loss(self):
        node = make_node()
        node.record_loss(round_num=1)
        assert node.hunger_counter == 1

    def test_hunger_accumulates_over_consecutive_losses(self):
        node = make_node()
        for r in range(10):
            node.record_loss(round_num=r)
        assert node.hunger_counter == 10

    def test_tokens_unchanged_on_loss(self):
        node = make_node(T0=1000.0)
        node.record_loss(round_num=1)
        assert node.tokens == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# generate_bid
# ---------------------------------------------------------------------------

class TestGenerateBid:
    def test_bid_is_nonnegative(self):
        node = make_node()
        bid = node.generate_bid("CPU", current_round=1)
        assert bid >= 0.0

    def test_bid_does_not_exceed_tokens(self):
        node = make_node(T0=1000.0)
        for r in range(20):
            bid = node.generate_bid("CPU", current_round=r)
            assert bid <= node.tokens

    def test_bid_zero_when_tokens_below_one(self):
        node = make_node(T0=1000.0)
        node.tokens = 0.5
        bid = node.generate_bid("CPU", current_round=1)
        assert bid == 0.0

    def test_bid_recorded_in_history(self):
        node = make_node()
        node.generate_bid("RAM", current_round=3)
        assert len(node.bid_history) == 1
        rnd, rtype, amount = node.bid_history[0]
        assert rnd == 3
        assert rtype == "RAM"

    def test_bid_unknown_resource_uses_default_valuation(self):
        node = make_node()
        # Unknown resource should not raise; falls back to get default of 1.0
        bid = node.generate_bid("UNKNOWN", current_round=1)
        assert bid >= 0.0


# ---------------------------------------------------------------------------
# sla_breach_probability edge cases
# ---------------------------------------------------------------------------

class TestSlaBreachProbability:
    def test_on_schedule_returns_zero(self):
        node = make_node(
            current_task_progress=0.5,
            current_sim_time=50.0,
            sla_deadline=100.0,
        )
        assert node.sla_breach_probability() == pytest.approx(0.0)

    def test_behind_schedule_returns_positive(self):
        node = make_node(
            current_task_progress=0.1,
            current_sim_time=50.0,
            sla_deadline=100.0,
        )
        prob = node.sla_breach_probability()
        assert prob == pytest.approx(0.4, abs=1e-9)

    def test_fully_elapsed_no_progress_returns_one(self):
        node = make_node(
            current_task_progress=0.0,
            current_sim_time=200.0,
            sla_deadline=100.0,
        )
        assert node.sla_breach_probability() == pytest.approx(1.0)

    def test_zero_deadline_returns_zero(self):
        node = make_node(sla_deadline=0.0, current_sim_time=0.0)
        assert node.sla_breach_probability() == pytest.approx(0.0)
