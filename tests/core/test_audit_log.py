"""
tests/core/test_audit_log.py

Pytest tests for src.core.audit_log.AuditLog.

Run with:
    pytest tests/core/test_audit_log.py -v
"""

import copy
import json
import os
import tempfile

import pytest

from src.core.audit_log import (
    AuditLog,
    AUCTION_WIN,
    AUCTION_LOSS,
    ERP_DECLARE,
    ERP_PREEMPT,
    ERP_REFUND,
    SFG_ALLOCATE,
    TOKEN_DEDUCT,
    TOKEN_REFUND,
    PERIOD_RESET,
    RESERVE_TOPUP,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXED_TS = 1_700_000_000.0   # deterministic genesis for all tests


@pytest.fixture
def log() -> AuditLog:
    """Fresh AuditLog with a fixed creation timestamp for reproducibility."""
    return AuditLog(creation_timestamp=FIXED_TS)


def _append_simple(log: AuditLog, node_id: str = "node-1", event_type: str = AUCTION_WIN) -> str:
    """Helper: append a minimal event and return its hash."""
    return log.append(event_type, node_id, {"round": 1, "resource": "CPU"}, timestamp=FIXED_TS + 1)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_genesis_hash_is_64_hex_chars(self, log):
        assert len(log.genesis_hash) == 64
        assert all(c in "0123456789abcdef" for c in log.genesis_hash)

    def test_chain_empty_on_init(self, log):
        assert log.chain == []
        assert len(log) == 0

    def test_different_timestamps_give_different_genesis(self):
        a = AuditLog(creation_timestamp=1000.0)
        b = AuditLog(creation_timestamp=2000.0)
        assert a.genesis_hash != b.genesis_hash

    def test_same_timestamp_gives_same_genesis(self):
        a = AuditLog(creation_timestamp=FIXED_TS)
        b = AuditLog(creation_timestamp=FIXED_TS)
        assert a.genesis_hash == b.genesis_hash


# ---------------------------------------------------------------------------
# append — basic contract
# ---------------------------------------------------------------------------

class TestAppend:
    def test_returns_hex_string(self, log):
        h = _append_simple(log)
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_chain_grows_by_one(self, log):
        _append_simple(log)
        assert len(log) == 1

    def test_multiple_appends_grow_chain(self, log):
        for i in range(5):
            log.append(AUCTION_WIN, f"node-{i}", {"round": i}, timestamp=FIXED_TS + i)
        assert len(log) == 5

    def test_entry_fields_present(self, log):
        _append_simple(log)
        entry = log.chain[0]
        for field in ("index", "event_type", "node_id", "timestamp",
                      "payload", "payload_hash", "prev_hash", "hash"):
            assert field in entry

    def test_first_entry_prev_hash_equals_genesis(self, log):
        _append_simple(log)
        assert log.chain[0]["prev_hash"] == log.genesis_hash

    def test_second_entry_prev_hash_equals_first_entry_hash(self, log):
        h1 = _append_simple(log)
        log.append(AUCTION_LOSS, "node-1", {"round": 2}, timestamp=FIXED_TS + 2)
        assert log.chain[1]["prev_hash"] == h1

    def test_returned_hash_matches_chain_entry(self, log):
        h = _append_simple(log)
        assert log.chain[0]["hash"] == h

    def test_same_payload_same_timestamp_gives_same_hash(self, log):
        a = AuditLog(creation_timestamp=FIXED_TS)
        b = AuditLog(creation_timestamp=FIXED_TS)
        h_a = a.append(AUCTION_WIN, "node-1", {"round": 1}, timestamp=FIXED_TS + 1)
        h_b = b.append(AUCTION_WIN, "node-1", {"round": 1}, timestamp=FIXED_TS + 1)
        assert h_a == h_b

    def test_different_payload_gives_different_hash(self, log):
        a = AuditLog(creation_timestamp=FIXED_TS)
        b = AuditLog(creation_timestamp=FIXED_TS)
        h_a = a.append(AUCTION_WIN, "node-1", {"round": 1}, timestamp=FIXED_TS + 1)
        h_b = b.append(AUCTION_WIN, "node-1", {"round": 2}, timestamp=FIXED_TS + 1)
        assert h_a != h_b

    def test_invalid_event_type_raises(self, log):
        with pytest.raises(ValueError, match="Unknown event_type"):
            log.append("FAKE_EVENT", "node-1", {})

    def test_all_valid_event_types_accepted(self, log):
        valid = [
            AUCTION_WIN, AUCTION_LOSS, ERP_DECLARE, ERP_PREEMPT, ERP_REFUND,
            SFG_ALLOCATE, TOKEN_DEDUCT, TOKEN_REFUND, PERIOD_RESET, RESERVE_TOPUP,
        ]
        for i, et in enumerate(valid):
            log.append(et, "node-1", {"i": i}, timestamp=FIXED_TS + i)
        assert len(log) == len(valid)


# ---------------------------------------------------------------------------
# verify_integrity — unmodified chain
# ---------------------------------------------------------------------------

class TestVerifyIntegrityClean:
    def test_empty_chain_is_valid(self, log):
        assert log.verify_integrity() is True

    def test_single_entry_chain_is_valid(self, log):
        _append_simple(log)
        assert log.verify_integrity() is True

    def test_multi_entry_chain_is_valid(self, log):
        for i in range(10):
            log.append(AUCTION_WIN, f"node-{i % 3}", {"round": i}, timestamp=FIXED_TS + i)
        assert log.verify_integrity() is True


# ---------------------------------------------------------------------------
# verify_integrity — tampered chain
# ---------------------------------------------------------------------------

class TestVerifyIntegrityTampered:
    def _populated_log(self) -> AuditLog:
        log = AuditLog(creation_timestamp=FIXED_TS)
        for i in range(5):
            log.append(AUCTION_WIN, f"node-{i}", {"round": i}, timestamp=FIXED_TS + i)
        return log

    def test_tampered_payload_detected(self):
        log = self._populated_log()
        log.chain[2]["payload"]["round"] = 9999   # mutate payload silently
        assert log.verify_integrity() is False

    def test_tampered_hash_detected(self):
        log = self._populated_log()
        log.chain[1]["hash"] = "a" * 64   # overwrite hash
        assert log.verify_integrity() is False

    def test_tampered_node_id_detected(self):
        log = self._populated_log()
        log.chain[0]["node_id"] = "EVIL_NODE"
        assert log.verify_integrity() is False

    def test_tampered_event_type_detected(self):
        log = self._populated_log()
        log.chain[3]["event_type"] = AUCTION_LOSS   # was AUCTION_WIN
        assert log.verify_integrity() is False

    def test_tampered_timestamp_detected(self):
        log = self._populated_log()
        log.chain[0]["timestamp"] = 0.0
        assert log.verify_integrity() is False

    def test_tampered_prev_hash_detected(self):
        log = self._populated_log()
        log.chain[2]["prev_hash"] = "0" * 64
        assert log.verify_integrity() is False

    def test_tampered_payload_hash_detected(self):
        """Forged payload_hash without changing payload should be caught."""
        log = self._populated_log()
        log.chain[1]["payload_hash"] = "b" * 64
        assert log.verify_integrity() is False

    def test_inserted_entry_detected(self):
        """Manually inserting a crafted entry breaks the chain."""
        log = self._populated_log()
        fake_entry = dict(log.chain[0])
        fake_entry["index"] = 99
        fake_entry["node_id"] = "injected"
        log.chain.insert(2, fake_entry)
        assert log.verify_integrity() is False


# ---------------------------------------------------------------------------
# get_events_for_node
# ---------------------------------------------------------------------------

class TestGetEventsForNode:
    def test_returns_only_matching_node(self, log):
        log.append(AUCTION_WIN,  "node-A", {"round": 1}, timestamp=FIXED_TS + 1)
        log.append(AUCTION_LOSS, "node-B", {"round": 1}, timestamp=FIXED_TS + 2)
        log.append(TOKEN_DEDUCT, "node-A", {"amount": 10}, timestamp=FIXED_TS + 3)

        events = log.get_events_for_node("node-A")
        assert len(events) == 2
        assert all(e["node_id"] == "node-A" for e in events)

    def test_returns_empty_list_for_unknown_node(self, log):
        _append_simple(log, node_id="node-X")
        assert log.get_events_for_node("node-GHOST") == []

    def test_returns_all_entries_when_single_node(self, log):
        for i in range(4):
            log.append(AUCTION_WIN, "node-1", {"round": i}, timestamp=FIXED_TS + i)
        assert len(log.get_events_for_node("node-1")) == 4

    def test_does_not_mutate_chain(self, log):
        _append_simple(log)
        events = log.get_events_for_node("node-1")
        events.clear()
        assert len(log.chain) == 1   # original chain unaffected


# ---------------------------------------------------------------------------
# export_to_json
# ---------------------------------------------------------------------------

class TestExportToJson:
    def test_creates_file(self, log):
        _append_simple(log)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            path = tmp.name
        try:
            log.export_to_json(path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_exported_json_is_valid(self, log):
        _append_simple(log)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
            path = tmp.name
        try:
            log.export_to_json(path)
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            assert "genesis_hash" in data
            assert "chain" in data
            assert data["entry_count"] == 1
        finally:
            os.unlink(path)

    def test_exported_genesis_matches_log(self, log):
        _append_simple(log)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
            path = tmp.name
        try:
            log.export_to_json(path)
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            assert data["genesis_hash"] == log.genesis_hash
        finally:
            os.unlink(path)
