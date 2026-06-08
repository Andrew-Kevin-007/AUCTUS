"""
tests/auction/test_vcg.py

Pytest tests for src.auction.vcg.VCGAuction and AuctionResult.

Run with:
    pytest tests/auction/test_vcg.py -v
"""

import pytest

from src.auction.vcg import VCGAuction, AuctionResult, _DEFAULT_CLEARING_PRICES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vcg() -> VCGAuction:
    """Fresh VCGAuction with default seed prices."""
    return VCGAuction()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_clearing_prices_set(self, vcg):
        for r, price in _DEFAULT_CLEARING_PRICES.items():
            assert vcg.last_clearing_price[r] == price

    def test_custom_initial_prices(self):
        custom = {"CPU": 42.0, "RAM": 7.0}
        vcg = VCGAuction(initial_clearing_prices=custom)
        assert vcg.last_clearing_price["CPU"] == 42.0
        assert vcg.last_clearing_price["RAM"] == 7.0

    def test_custom_prices_do_not_share_reference(self):
        src = {"CPU": 10.0}
        vcg = VCGAuction(initial_clearing_prices=src)
        vcg.last_clearing_price["CPU"] = 99.0
        assert src["CPU"] == 10.0   # mutation must not bleed back


# ---------------------------------------------------------------------------
# clear() — standard 3-bidder case
# ---------------------------------------------------------------------------

class TestClearStandard:
    def test_winner_is_highest_bidder(self, vcg):
        bids = {"node-A": 100.0, "node-B": 80.0, "node-C": 50.0}
        result = vcg.clear("CPU", bids, round_num=1)
        assert result.winner_id == "node-A"

    def test_payment_is_second_highest_bid(self, vcg):
        bids = {"node-A": 100.0, "node-B": 80.0, "node-C": 50.0}
        result = vcg.clear("CPU", bids, round_num=1)
        assert result.payment == pytest.approx(80.0)

    def test_winning_bid_recorded(self, vcg):
        bids = {"node-A": 100.0, "node-B": 80.0, "node-C": 50.0}
        result = vcg.clear("CPU", bids, round_num=1)
        assert result.winning_bid == pytest.approx(100.0)

    def test_resource_type_in_result(self, vcg):
        bids = {"node-A": 100.0, "node-B": 80.0}
        result = vcg.clear("RAM", bids, round_num=5)
        assert result.resource_type == "RAM"

    def test_round_num_in_result(self, vcg):
        bids = {"node-A": 100.0, "node-B": 80.0}
        result = vcg.clear("CPU", bids, round_num=42)
        assert result.round_num == 42

    def test_last_clearing_price_updated_after_competitive_round(self, vcg):
        bids = {"node-A": 100.0, "node-B": 80.0}
        vcg.clear("CPU", bids, round_num=1)
        assert vcg.last_clearing_price["CPU"] == pytest.approx(80.0)

    def test_result_is_auction_result_instance(self, vcg):
        result = vcg.clear("CPU", {"node-A": 10.0, "node-B": 5.0}, round_num=1)
        assert isinstance(result, AuctionResult)

    def test_three_bidders_specific_values(self, vcg):
        """Explicit scenario from the spec: [100, 80, 50] -> winner pays 80."""
        bids = {"node-1": 100.0, "node-2": 80.0, "node-3": 50.0}
        result = vcg.clear("CPU", bids, round_num=1)
        assert result.winner_id  == "node-1"
        assert result.payment    == pytest.approx(80.0)
        assert result.winning_bid == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# clear() — single bidder
# ---------------------------------------------------------------------------

class TestClearSingleBidder:
    def test_single_bidder_wins(self, vcg):
        result = vcg.clear("CPU", {"node-X": 75.0}, round_num=1)
        assert result.winner_id == "node-X"

    def test_single_bidder_pays_zero(self, vcg):
        result = vcg.clear("CPU", {"node-X": 75.0}, round_num=1)
        assert result.payment == pytest.approx(0.0)

    def test_single_bidder_winning_bid_recorded(self, vcg):
        result = vcg.clear("CPU", {"node-X": 75.0}, round_num=1)
        assert result.winning_bid == pytest.approx(75.0)

    def test_last_clearing_price_NOT_updated_for_single_bidder(self, vcg):
        """Single-bidder payment is 0 -- clearing price must not be overwritten."""
        original_price = vcg.last_clearing_price["CPU"]
        vcg.clear("CPU", {"node-X": 75.0}, round_num=1)
        assert vcg.last_clearing_price["CPU"] == pytest.approx(original_price)

    def test_single_bidder_clearing_price_unchanged_after_prior_competitive(self, vcg):
        # Establish a competitive price first
        vcg.clear("CPU", {"node-A": 100.0, "node-B": 60.0}, round_num=1)
        price_after_competitive = vcg.last_clearing_price["CPU"]   # 60.0

        # Single-bidder round must not overwrite it
        vcg.clear("CPU", {"node-X": 200.0}, round_num=2)
        assert vcg.last_clearing_price["CPU"] == pytest.approx(price_after_competitive)


# ---------------------------------------------------------------------------
# clear() — empty bids
# ---------------------------------------------------------------------------

class TestClearEmptyBids:
    def test_empty_bids_winner_is_none(self, vcg):
        result = vcg.clear("CPU", {}, round_num=1)
        assert result.winner_id is None

    def test_empty_bids_payment_zero(self, vcg):
        result = vcg.clear("CPU", {}, round_num=1)
        assert result.payment == pytest.approx(0.0)

    def test_empty_bids_winning_bid_zero(self, vcg):
        result = vcg.clear("CPU", {}, round_num=1)
        assert result.winning_bid == pytest.approx(0.0)

    def test_empty_bids_all_bids_is_empty_dict(self, vcg):
        result = vcg.clear("CPU", {}, round_num=1)
        assert result.all_bids == {}

    def test_empty_bids_clearing_price_unchanged(self, vcg):
        original = vcg.last_clearing_price["CPU"]
        vcg.clear("CPU", {}, round_num=1)
        assert vcg.last_clearing_price["CPU"] == pytest.approx(original)


# ---------------------------------------------------------------------------
# clear() — tie-breaking
# ---------------------------------------------------------------------------

class TestClearTieBreaking:
    def test_tie_broken_by_lexicographic_node_id_ascending(self, vcg):
        """Smaller node_id string wins the tie."""
        bids = {"node-B": 100.0, "node-A": 100.0, "node-C": 100.0}
        result = vcg.clear("CPU", bids, round_num=1)
        assert result.winner_id == "node-A"

    def test_tie_payment_equals_tied_bid(self, vcg):
        """VCG payment when top two bids are equal = that bid amount."""
        bids = {"node-B": 100.0, "node-A": 100.0, "node-C": 50.0}
        result = vcg.clear("CPU", bids, round_num=1)
        assert result.payment == pytest.approx(100.0)

    def test_two_equal_highest_bids_winner_deterministic(self, vcg):
        """Re-running the same auction produces the same winner."""
        bids = {"node-Z": 200.0, "node-A": 200.0}
        r1 = vcg.clear("CPU", bids, round_num=1)
        r2 = vcg.clear("CPU", bids, round_num=2)
        assert r1.winner_id == r2.winner_id

    def test_tie_only_between_non_winners(self, vcg):
        """Tie among losers does not affect winner or payment."""
        bids = {"node-W": 200.0, "node-X": 100.0, "node-Y": 100.0}
        result = vcg.clear("CPU", bids, round_num=1)
        assert result.winner_id == "node-W"
        # payment = second highest = 100 (tied, but value is unambiguous)
        assert result.payment == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# all_bids snapshot
# ---------------------------------------------------------------------------

class TestAllBidsSnapshot:
    def test_all_bids_contains_winner_and_losers(self, vcg):
        bids = {"node-A": 100.0, "node-B": 80.0, "node-C": 50.0}
        result = vcg.clear("CPU", bids, round_num=1)
        assert set(result.all_bids.keys()) == {"node-A", "node-B", "node-C"}

    def test_all_bids_values_match_input(self, vcg):
        bids = {"node-A": 100.0, "node-B": 80.0}
        result = vcg.clear("CPU", bids, round_num=1)
        assert result.all_bids["node-A"] == pytest.approx(100.0)
        assert result.all_bids["node-B"] == pytest.approx(80.0)

    def test_all_bids_is_independent_copy(self, vcg):
        """Mutating the input dict after the call must not alter the snapshot."""
        bids = {"node-A": 100.0, "node-B": 80.0}
        result = vcg.clear("CPU", bids, round_num=1)
        bids["node-A"] = 999.0
        assert result.all_bids["node-A"] == pytest.approx(100.0)

    def test_all_bids_snapshot_mutation_safe(self, vcg):
        """Mutating the returned all_bids dict must not affect VCGAuction state."""
        bids = {"node-A": 100.0, "node-B": 80.0}
        result = vcg.clear("CPU", bids, round_num=1)
        result.all_bids["node-A"] = 0.0   # tamper returned snapshot
        # Re-clear same bids: price should still be based on original values
        result2 = vcg.clear("CPU", {"node-A": 100.0, "node-B": 80.0}, round_num=2)
        assert result2.winner_id == "node-A"


# ---------------------------------------------------------------------------
# last_clearing_price update logic
# ---------------------------------------------------------------------------

class TestClearingPriceUpdates:
    def test_price_updates_to_second_highest_bid(self, vcg):
        vcg.clear("RAM", {"node-A": 50.0, "node-B": 30.0}, round_num=1)
        assert vcg.last_clearing_price["RAM"] == pytest.approx(30.0)

    def test_price_does_not_update_when_payment_zero_single_bidder(self, vcg):
        initial = vcg.last_clearing_price["RAM"]
        vcg.clear("RAM", {"node-A": 50.0}, round_num=1)
        assert vcg.last_clearing_price["RAM"] == pytest.approx(initial)

    def test_price_does_not_update_when_payment_zero_empty(self, vcg):
        initial = vcg.last_clearing_price["STG"]
        vcg.clear("STG", {}, round_num=1)
        assert vcg.last_clearing_price["STG"] == pytest.approx(initial)

    def test_price_tracks_most_recent_competitive_round(self, vcg):
        vcg.clear("NET", {"node-A": 40.0, "node-B": 20.0}, round_num=1)
        assert vcg.last_clearing_price["NET"] == pytest.approx(20.0)
        vcg.clear("NET", {"node-A": 90.0, "node-B": 70.0}, round_num=2)
        assert vcg.last_clearing_price["NET"] == pytest.approx(70.0)

    def test_price_updates_per_resource_independently(self, vcg):
        vcg.clear("CPU", {"node-A": 80.0, "node-B": 60.0}, round_num=1)
        vcg.clear("RAM", {"node-A": 20.0, "node-B": 10.0}, round_num=1)
        assert vcg.last_clearing_price["CPU"] == pytest.approx(60.0)
        assert vcg.last_clearing_price["RAM"] == pytest.approx(10.0)

    def test_single_bidder_after_competitive_preserves_price(self, vcg):
        vcg.clear("CPU", {"node-A": 100.0, "node-B": 75.0}, round_num=1)
        vcg.clear("CPU", {"node-A": 999.0}, round_num=2)
        assert vcg.last_clearing_price["CPU"] == pytest.approx(75.0)

    def test_payment_zero_second_bidder_does_not_update_price(self, vcg):
        """When second bidder bids exactly 0, payment is 0 -- price unchanged."""
        original = vcg.last_clearing_price["CPU"]
        # pre-filter: caller should exclude 0-bids, but test robustness
        vcg.clear("CPU", {"node-A": 100.0, "node-B": 0.0}, round_num=1)
        # payment = second-highest = 0.0; price must not be overwritten with 0
        assert vcg.last_clearing_price["CPU"] == pytest.approx(original)


# ---------------------------------------------------------------------------
# get_last_clearing_price
# ---------------------------------------------------------------------------

class TestGetLastClearingPrice:
    def test_returns_default_for_untouched_resource(self, vcg):
        assert vcg.get_last_clearing_price("CPU") == pytest.approx(
            _DEFAULT_CLEARING_PRICES["CPU"]
        )

    def test_returns_updated_price_after_competitive_round(self, vcg):
        vcg.clear("CPU", {"node-A": 100.0, "node-B": 55.0}, round_num=1)
        assert vcg.get_last_clearing_price("CPU") == pytest.approx(55.0)

    def test_unknown_resource_returns_zero(self, vcg):
        assert vcg.get_last_clearing_price("UNKNOWN") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Separation of concerns — VCG does not mutate external state
# ---------------------------------------------------------------------------

class TestPurity:
    def test_clear_does_not_mutate_input_bids_dict(self, vcg):
        bids = {"node-A": 100.0, "node-B": 80.0}
        original = dict(bids)
        vcg.clear("CPU", bids, round_num=1)
        assert bids == original

    def test_repeated_clear_with_same_bids_gives_same_result(self, vcg):
        bids = {"node-A": 100.0, "node-B": 80.0, "node-C": 60.0}
        r1 = vcg.clear("CPU", bids, round_num=1)
        r2 = vcg.clear("CPU", bids, round_num=2)
        assert r1.winner_id   == r2.winner_id
        assert r1.payment     == pytest.approx(r2.payment)
        assert r1.winning_bid == pytest.approx(r2.winning_bid)
