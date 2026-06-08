"""
tests/core/test_resource_pool.py

Pytest tests for src.core.resource_pool.ResourcePool.

Run with:
    pytest tests/core/test_resource_pool.py -v
"""

import pytest

from src.core.resource_pool import (
    ResourcePool,
    DEFAULT_CAPACITY,
    RESERVE_POOL_FRACTION,
    RESOURCE_TYPES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pool() -> ResourcePool:
    """Default resource pool with locked capacity values."""
    return ResourcePool()


@pytest.fixture
def small_pool() -> ResourcePool:
    """Minimal pool for arithmetic clarity in edge-case tests."""
    return ResourcePool(capacity={"CPU": 100.0, "RAM": 200.0})


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults_applied_when_no_capacity_given(self, pool):
        for r in RESOURCE_TYPES:
            assert pool.capacity[r] == DEFAULT_CAPACITY[r]

    def test_reserved_is_10_percent_of_capacity(self, pool):
        for r in RESOURCE_TYPES:
            assert pool.reserved[r] == pytest.approx(0.10 * DEFAULT_CAPACITY[r])

    def test_available_is_90_percent_of_capacity(self, pool):
        for r in RESOURCE_TYPES:
            assert pool.available[r] == pytest.approx(0.90 * DEFAULT_CAPACITY[r])

    def test_reserve_pool_balance_starts_full(self, pool):
        for r in RESOURCE_TYPES:
            assert pool.reserve_pool_balance[r] == pytest.approx(pool.reserved[r])

    def test_allocations_empty_on_init(self, pool):
        assert pool.allocations == {}

    def test_last_clearing_price_zeroed(self, pool):
        for r in RESOURCE_TYPES:
            assert pool.last_clearing_price[r] == 0.0

    def test_custom_capacity_respected(self):
        p = ResourcePool(capacity={"CPU": 200.0})
        assert p.capacity["CPU"] == 200.0
        assert p.reserved["CPU"] == pytest.approx(20.0)
        assert p.available["CPU"] == pytest.approx(180.0)


# ---------------------------------------------------------------------------
# can_allocate
# ---------------------------------------------------------------------------

class TestCanAllocate:
    def test_returns_true_within_market_capacity(self, pool):
        assert pool.can_allocate("CPU", 10.0) is True

    def test_returns_false_beyond_market_capacity(self, pool):
        market_cap = 0.90 * DEFAULT_CAPACITY["CPU"]
        assert pool.can_allocate("CPU", market_cap + 1.0) is False

    def test_returns_false_for_zero_units(self, pool):
        assert pool.can_allocate("CPU", 0.0) is False

    def test_returns_false_for_negative_units(self, pool):
        assert pool.can_allocate("CPU", -5.0) is False

    def test_returns_false_for_unknown_resource(self, pool):
        assert pool.can_allocate("UNKNOWN", 1.0) is False

    def test_reserve_path_checks_reserve_balance(self, pool):
        reserve = pool.reserve_pool_balance["RAM"]
        assert pool.can_allocate("RAM", reserve, from_reserve=True) is True
        assert pool.can_allocate("RAM", reserve + 1.0, from_reserve=True) is False


# ---------------------------------------------------------------------------
# allocate — reduces available
# ---------------------------------------------------------------------------

class TestAllocate:
    def test_allocate_reduces_available(self, small_pool):
        small_pool.allocate("node-1", "CPU", 10.0)
        # market available = 90.0 initially; should now be 80.0
        assert small_pool.available["CPU"] == pytest.approx(80.0)

    def test_allocate_records_in_allocations(self, small_pool):
        small_pool.allocate("node-1", "CPU", 10.0)
        assert "node-1" in small_pool.allocations
        assert small_pool.allocations["node-1"]["CPU"] == pytest.approx(10.0)

    def test_allocate_accumulates_for_same_node(self, small_pool):
        small_pool.allocate("node-1", "CPU", 5.0)
        small_pool.allocate("node-1", "CPU", 3.0)
        assert small_pool.allocations["node-1"]["CPU"] == pytest.approx(8.0)

    def test_allocate_returns_true_on_success(self, small_pool):
        result = small_pool.allocate("node-1", "CPU", 10.0)
        assert result is True

    def test_allocate_returns_false_on_insufficient_capacity(self, small_pool):
        # Exhaust market pool first
        small_pool.allocate("node-1", "CPU", 90.0)   # 90% of 100
        result = small_pool.allocate("node-2", "CPU", 5.0)
        assert result is False

    def test_allocate_does_not_touch_reserve_pool(self, small_pool):
        original_reserve = small_pool.reserve_pool_balance["CPU"]
        small_pool.allocate("node-1", "CPU", 10.0)
        assert small_pool.reserve_pool_balance["CPU"] == pytest.approx(original_reserve)

    # -- Double-allocate beyond capacity
    def test_double_allocate_beyond_capacity_returns_false(self, small_pool):
        market_cap = 0.90 * 100.0   # 90 cores
        assert small_pool.allocate("node-1", "CPU", market_cap) is True
        assert small_pool.allocate("node-2", "CPU", 1.0) is False   # pool exhausted

    def test_allocation_state_unchanged_after_failed_allocate(self, small_pool):
        small_pool.allocate("node-1", "CPU", 90.0)
        before_available = small_pool.available["CPU"]
        small_pool.allocate("node-2", "CPU", 10.0)
        assert small_pool.available["CPU"] == pytest.approx(before_available)

    # -- Reserve pool allocation
    def test_reserve_allocation_draws_from_reserve_not_market(self, small_pool):
        market_before = small_pool.available["CPU"]
        reserve_before = small_pool.reserve_pool_balance["CPU"]
        small_pool.allocate("node-sfg", "CPU", 5.0, from_reserve=True)
        assert small_pool.available["CPU"] == pytest.approx(market_before)
        assert small_pool.reserve_pool_balance["CPU"] == pytest.approx(reserve_before - 5.0)

    def test_reserve_allocation_fails_beyond_reserve_balance(self, small_pool):
        reserve_cap = small_pool.reserve_pool_balance["CPU"]   # 10.0
        result = small_pool.allocate("node-sfg", "CPU", reserve_cap + 1.0, from_reserve=True)
        assert result is False


# ---------------------------------------------------------------------------
# deallocate — restores available
# ---------------------------------------------------------------------------

class TestDeallocate:
    def test_deallocate_restores_available(self, small_pool):
        small_pool.allocate("node-1", "CPU", 20.0)
        small_pool.deallocate("node-1", "CPU", 20.0)
        assert small_pool.available["CPU"] == pytest.approx(90.0)

    def test_deallocate_removes_node_when_all_returned(self, small_pool):
        small_pool.allocate("node-1", "CPU", 10.0)
        small_pool.deallocate("node-1", "CPU", 10.0)
        assert "node-1" not in small_pool.allocations

    def test_deallocate_partial_leaves_remainder(self, small_pool):
        small_pool.allocate("node-1", "CPU", 20.0)
        small_pool.deallocate("node-1", "CPU", 5.0)
        assert small_pool.allocations["node-1"]["CPU"] == pytest.approx(15.0)
        assert small_pool.available["CPU"] == pytest.approx(90.0 - 15.0)

    def test_deallocate_clamps_to_held_amount(self, small_pool):
        """Returning more than was allocated should not create negative holdings."""
        small_pool.allocate("node-1", "CPU", 10.0)
        small_pool.deallocate("node-1", "CPU", 50.0)   # more than held
        # available should be restored by at most 10 (what was held)
        assert small_pool.available["CPU"] == pytest.approx(90.0)
        assert "node-1" not in small_pool.allocations

    def test_deallocate_noop_for_unknown_node(self, small_pool):
        before = small_pool.available["CPU"]
        small_pool.deallocate("ghost-node", "CPU", 10.0)
        assert small_pool.available["CPU"] == pytest.approx(before)

    def test_deallocate_preserves_other_resource_allocations(self, small_pool):
        small_pool.allocate("node-1", "CPU", 10.0)
        small_pool.allocate("node-1", "RAM", 32.0)
        small_pool.deallocate("node-1", "CPU", 10.0)
        assert "node-1" in small_pool.allocations
        assert small_pool.allocations["node-1"].get("RAM") == pytest.approx(32.0)


# ---------------------------------------------------------------------------
# Utilization
# ---------------------------------------------------------------------------

class TestGetUtilization:
    def test_utilization_zero_when_nothing_allocated(self, small_pool):
        util = small_pool.get_utilization()
        assert util["CPU"] == pytest.approx(0.0)
        assert util["RAM"] == pytest.approx(0.0)

    def test_utilization_100_when_market_fully_used(self, small_pool):
        market_cap = 0.90 * 100.0
        small_pool.allocate("node-1", "CPU", market_cap)
        util = small_pool.get_utilization()
        assert util["CPU"] == pytest.approx(100.0)

    def test_utilization_50_when_half_market_used(self, small_pool):
        market_cap = 0.90 * 100.0   # 90 cores
        small_pool.allocate("node-1", "CPU", market_cap / 2)
        util = small_pool.get_utilization()
        assert util["CPU"] == pytest.approx(50.0)

    def test_utilization_excludes_reserve_pool(self, small_pool):
        """Reserve pool consumption should NOT appear in market utilization."""
        small_pool.allocate("node-sfg", "CPU", 5.0, from_reserve=True)
        util = small_pool.get_utilization()
        # Market pool is untouched → utilization must still be 0%
        assert util["CPU"] == pytest.approx(0.0)

    def test_utilization_returns_all_resource_types(self, pool):
        util = pool.get_utilization()
        for r in RESOURCE_TYPES:
            assert r in util

    def test_utilization_after_deallocate_restores_to_zero(self, small_pool):
        small_pool.allocate("node-1", "CPU", 45.0)
        small_pool.deallocate("node-1", "CPU", 45.0)
        util = small_pool.get_utilization()
        assert util["CPU"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Reserve pool management
# ---------------------------------------------------------------------------

class TestUpdateReservePool:
    def test_credit_increases_balance(self, small_pool):
        small_pool.allocate("node-sfg", "CPU", 5.0, from_reserve=True)
        before = small_pool.reserve_pool_balance["CPU"]
        small_pool.update_reserve_pool({"CPU": 3.0})
        assert small_pool.reserve_pool_balance["CPU"] == pytest.approx(before + 3.0)

    def test_credit_capped_at_reserved_max(self, small_pool):
        # reserve starts full; any credit should not exceed cap
        cap = small_pool.reserved["CPU"]   # 10.0
        small_pool.update_reserve_pool({"CPU": 999.0})
        assert small_pool.reserve_pool_balance["CPU"] == pytest.approx(cap)

    def test_unknown_resource_in_delta_is_ignored(self, small_pool):
        # should not raise
        small_pool.update_reserve_pool({"UNKNOWN": 5.0})

    def test_erp_30_percent_surplus_credited_correctly(self, small_pool):
        clearing_price = 100.0
        erp_surplus_fraction = 0.30
        # Drain reserve first so we can see the delta clearly
        small_pool.allocate("node-sfg", "CPU", 8.0, from_reserve=True)
        before = small_pool.reserve_pool_balance["CPU"]
        small_pool.update_reserve_pool({"CPU": erp_surplus_fraction * clearing_price})
        assert small_pool.reserve_pool_balance["CPU"] == pytest.approx(
            min(before + 30.0, small_pool.reserved["CPU"])
        )


# ---------------------------------------------------------------------------
# get_current_holder
# ---------------------------------------------------------------------------

class TestGetCurrentHolder:
    def test_returns_none_when_no_allocations(self, pool):
        assert pool.get_current_holder("CPU") is None

    def test_returns_only_holder(self, small_pool):
        small_pool.allocate("node-1", "CPU", 10.0)
        assert small_pool.get_current_holder("CPU") == "node-1"

    def test_returns_smallest_holder_among_multiple(self, small_pool):
        # node-2 holds less CPU → it is the lowest-priority candidate
        small_pool.allocate("node-1", "CPU", 20.0)
        small_pool.allocate("node-2", "CPU", 5.0)
        assert small_pool.get_current_holder("CPU") == "node-2"

    def test_returns_none_after_full_deallocate(self, small_pool):
        small_pool.allocate("node-1", "CPU", 10.0)
        small_pool.deallocate("node-1", "CPU", 10.0)
        assert small_pool.get_current_holder("CPU") is None

    def test_does_not_return_holder_of_different_resource(self, small_pool):
        small_pool.allocate("node-1", "RAM", 32.0)
        # CPU has no holder
        assert small_pool.get_current_holder("CPU") is None
