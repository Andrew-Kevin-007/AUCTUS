"""
src/core/audit_log.py

SHA-256 hash-chain audit log for all Auctus protocol events.

Design (architecture.md §5):
    L_t = SHA256(L_{t-1} || event_type || node_id || str(timestamp) || payload_hash)

The chain is append-only.  The genesis block is seeded with a hash of the
literal string "AUCTUS_GENESIS" concatenated with the creation timestamp,
so no two simulation runs produce the same chain even on identical workloads.

Limitation (acknowledged): this prototype uses a centrally administered,
in-memory list.  Distributed replication is future work.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Event-type constants
# ---------------------------------------------------------------------------

AUCTION_WIN:    str = "AUCTION_WIN"
AUCTION_LOSS:   str = "AUCTION_LOSS"
ERP_DECLARE:    str = "ERP_DECLARE"
ERP_PREEMPT:    str = "ERP_PREEMPT"
ERP_REFUND:     str = "ERP_REFUND"
SFG_ALLOCATE:   str = "SFG_ALLOCATE"
TOKEN_DEDUCT:   str = "TOKEN_DEDUCT"
TOKEN_REFUND:   str = "TOKEN_REFUND"
PERIOD_RESET:   str = "PERIOD_RESET"
RESERVE_TOPUP:  str = "RESERVE_TOPUP"

_VALID_EVENT_TYPES = frozenset({
    AUCTION_WIN, AUCTION_LOSS, ERP_DECLARE, ERP_PREEMPT, ERP_REFUND,
    SFG_ALLOCATE, TOKEN_DEDUCT, TOKEN_REFUND, PERIOD_RESET, RESERVE_TOPUP,
})


# ---------------------------------------------------------------------------
# Internal entry structure
# ---------------------------------------------------------------------------

@dataclass
class _LogEntry:
    index:        int
    event_type:   str
    node_id:      str
    timestamp:    float
    payload:      dict
    payload_hash: str
    prev_hash:    str
    hash:         str


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------

class AuditLog:
    """
    Append-only SHA-256 hash-chain audit log.

    Each entry is linked to the previous entry's hash, forming a tamper-
    evident chain.  ``verify_integrity()`` recomputes every hash from the
    genesis block and returns False on the first mismatch.

    Parameters
    ----------
    creation_timestamp : float, optional
        Unix timestamp used to seed the genesis hash.  Defaults to
        ``time.time()`` at construction.  Pass a fixed value in tests for
        reproducibility.
    """

    def __init__(self, creation_timestamp: Optional[float] = None) -> None:
        ts = creation_timestamp if creation_timestamp is not None else time.time()
        self.genesis_hash: str = self._sha256(f"AUCTUS_GENESIS{ts}")
        self.chain: List[dict] = []

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append(
        self,
        event_type: str,
        node_id: str,
        payload: dict,
        timestamp: Optional[float] = None,
    ) -> str:
        """
        Append a new event to the chain and return its hash.

        Parameters
        ----------
        event_type : str
            One of the module-level event-type constants.
        node_id : str
            ID of the node this event pertains to.
        payload : dict
            Arbitrary event metadata.  Must be JSON-serialisable.
        timestamp : float, optional
            Unix timestamp for the event.  Defaults to ``time.time()``.

        Returns
        -------
        str
            SHA-256 hex digest of the new entry.

        Raises
        ------
        ValueError
            If *event_type* is not one of the recognised constants.
        """
        if event_type not in _VALID_EVENT_TYPES:
            raise ValueError(
                f"Unknown event_type {event_type!r}. "
                f"Valid types: {sorted(_VALID_EVENT_TYPES)}"
            )

        ts = timestamp if timestamp is not None else time.time()

        payload_hash = self._sha256(json.dumps(payload, sort_keys=True))
        prev_hash = self.chain[-1]["hash"] if self.chain else self.genesis_hash

        entry_hash = self._sha256(
            prev_hash + event_type + node_id + str(ts) + payload_hash
        )

        entry = {
            "index":        len(self.chain),
            "event_type":   event_type,
            "node_id":      node_id,
            "timestamp":    ts,
            "payload":      payload,
            "payload_hash": payload_hash,
            "prev_hash":    prev_hash,
            "hash":         entry_hash,
        }
        self.chain.append(entry)
        return entry_hash

    # ------------------------------------------------------------------
    # Integrity verification
    # ------------------------------------------------------------------

    def verify_integrity(self) -> bool:
        """
        Recompute every hash in the chain from the genesis block.

        Returns True iff the chain is unmodified.  Returns False on the
        first detected mismatch (early exit).

        Returns
        -------
        bool
        """
        prev_hash = self.genesis_hash

        for entry in self.chain:
            # Re-derive payload_hash
            expected_payload_hash = self._sha256(
                json.dumps(entry["payload"], sort_keys=True)
            )
            if entry["payload_hash"] != expected_payload_hash:
                return False

            # Re-derive entry hash
            expected_hash = self._sha256(
                prev_hash
                + entry["event_type"]
                + entry["node_id"]
                + str(entry["timestamp"])
                + entry["payload_hash"]
            )
            if entry["hash"] != expected_hash:
                return False

            # Verify back-link
            if entry["prev_hash"] != prev_hash:
                return False

            prev_hash = entry["hash"]

        return True

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_events_for_node(self, node_id: str) -> List[dict]:
        """
        Return all chain entries whose ``node_id`` matches *node_id*.

        Parameters
        ----------
        node_id : str

        Returns
        -------
        list of dict
        """
        return [entry for entry in self.chain if entry["node_id"] == node_id]

    def get_events_by_type(self, event_type: str) -> List[dict]:
        """
        Return all entries of a given event type.

        Parameters
        ----------
        event_type : str

        Returns
        -------
        list of dict
        """
        return [entry for entry in self.chain if entry["event_type"] == event_type]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_to_json(self, filepath: str) -> None:
        """
        Dump the entire chain to a JSON file at *filepath*.

        The output includes the genesis hash so the file is self-contained
        for offline verification.

        Parameters
        ----------
        filepath : str
        """
        export = {
            "genesis_hash": self.genesis_hash,
            "entry_count":  len(self.chain),
            "chain":        self.chain,
        }
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(export, fh, indent=2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sha256(data: str) -> str:
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def __len__(self) -> int:
        return len(self.chain)

    def __repr__(self) -> str:  # pragma: no cover
        tip = self.chain[-1]["hash"][:12] + "..." if self.chain else "(empty)"
        return f"AuditLog(entries={len(self.chain)}, tip={tip})"
