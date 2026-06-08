# encoding: utf-8
"""
tests/smoke_test.py

Manual end-to-end smoke test for the Auctus core layer.
Exercises Node, ResourcePool, TokenManager, and AuditLog together
through a single VCG auction round -- no SimPy required.

Run with:
    python tests/smoke_test.py
or:
    pytest tests/smoke_test.py -v -s
"""

import sys
import numpy as np

from src.core.node          import Node
from src.core.resource_pool import ResourcePool
from src.core.token_manager import TokenManager
from src.core.audit_log     import AuditLog, AUCTION_WIN, AUCTION_LOSS

# ---------------------------------------------------------------------------
# Helpers -- pure ASCII to stay safe on Windows cp1252 consoles
# ---------------------------------------------------------------------------

SEP  = "-" * 60
SEP2 = "=" * 60


def section(title: str) -> None:
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)


def step(msg: str) -> None:
    print(f"  > {msg}")


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


def assert_close(actual, expected, label, tol=1e-9):
    if abs(actual - expected) > tol:
        fail(f"{label}: expected {expected}, got {actual}")
    ok(f"{label} = {actual:.4f}  (expected {expected})")


# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------

section("1. Create nodes, pool, token manager, audit log")

T0 = 1000.0

node1 = Node(
    node_id="node-1",
    criticality=1,          # E1 critical
    tier=3,
    sla_deadline=500.0,
    current_task_progress=0.4,
    current_sim_time=200.0,
    T0=T0,
    rng=np.random.default_rng(11),
)

node2 = Node(
    node_id="node-2",
    criticality=2,          # E2 prod outage
    tier=2,
    sla_deadline=500.0,
    current_task_progress=0.3,
    current_sim_time=200.0,
    T0=T0,
    rng=np.random.default_rng(22),
)

node3 = Node(
    node_id="node-3",
    criticality=3,          # E3 SLA breach
    tier=1,
    sla_deadline=500.0,
    current_task_progress=0.5,
    current_sim_time=200.0,
    T0=T0,
    rng=np.random.default_rng(33),
)

pool  = ResourcePool()          # default: CPU=100 cores, RAM=512 GB, ...
tm    = TokenManager(T0=T0)
audit = AuditLog(creation_timestamp=1_700_000_000.0)

tm.register_node(node1)
tm.register_node(node2)
tm.register_node(node3)

# register_node resets tokens to T0 -- override for distinct starting balances
node1.tokens = 900.0
node2.tokens = 750.0
node3.tokens = 600.0

step("3 nodes created and registered")
step(f"  node-1  criticality=E{node1.criticality}  tokens={node1.tokens:.1f}")
step(f"  node-2  criticality=E{node2.criticality}  tokens={node2.tokens:.1f}")
step(f"  node-3  criticality=E{node3.criticality}  tokens={node3.tokens:.1f}")
step(
    f"ResourcePool  CPU market={pool.available['CPU']:.0f} cores"
    f"  reserve={pool.reserved['CPU']:.0f} cores"
)


# ---------------------------------------------------------------------------
# 2. Bid submission
# ---------------------------------------------------------------------------

section("2. Bid submission (round 1, resource CPU)")

RESOURCE  = "CPU"
ROUND     = 1
UNITS     = 4.0            # cores allocated to winner

BID_NODE1 = 50.0
BID_NODE2 = 80.0
BID_NODE3 = 30.0

step(f"node-1 bids {BID_NODE1:.1f} tokens for {RESOURCE}")
step(f"node-2 bids {BID_NODE2:.1f} tokens for {RESOURCE}")
step(f"node-3 bids {BID_NODE3:.1f} tokens for {RESOURCE}")

node1.bid_history.append((ROUND, RESOURCE, BID_NODE1))
node2.bid_history.append((ROUND, RESOURCE, BID_NODE2))
node3.bid_history.append((ROUND, RESOURCE, BID_NODE3))


# ---------------------------------------------------------------------------
# 3. VCG clearing
# ---------------------------------------------------------------------------

section("3. VCG second-price clearing")

bids = {
    "node-1": BID_NODE1,
    "node-2": BID_NODE2,
    "node-3": BID_NODE3,
}

winner_id   = max(bids, key=bids.__getitem__)
winner_bid  = bids[winner_id]
sorted_bids = sorted(bids.values(), reverse=True)
vcg_payment = sorted_bids[1]                  # second-highest bid
losers      = [nid for nid in bids if nid != winner_id]

step(f"Winner      : {winner_id}  (bid={winner_bid:.1f})")
step(f"VCG payment : {vcg_payment:.1f}  (second-highest bid)")
step(f"Losers      : {losers}")

if winner_id != "node-2":
    fail(f"Expected node-2 to win, got {winner_id}")
if vcg_payment != 50.0:
    fail(f"Expected VCG payment 50.0, got {vcg_payment}")
ok("Winner and VCG payment are correct")


# ---------------------------------------------------------------------------
# 4. Apply auction outcome
# ---------------------------------------------------------------------------

section("4. Apply auction outcome")

balance_before = {nid: tm.nodes[nid].tokens for nid in bids}
step("Balances before:")
for nid, bal in balance_before.items():
    step(f"  {nid}: {bal:.2f}")

# 4a. Deduct VCG payment via TokenManager
deduct_ok = tm.deduct(winner_id, vcg_payment, reason="auction_win")
if not deduct_ok:
    fail(f"tm.deduct failed for {winner_id}")
ok(f"tm.deduct({winner_id}, {vcg_payment}) -- success")

# 4b. Allocate resource in pool
alloc_ok = pool.allocate(winner_id, RESOURCE, UNITS)
if not alloc_ok:
    fail(f"pool.allocate failed for {winner_id}")
ok(f"pool.allocate({winner_id}, {RESOURCE}, {UNITS} cores) -- success")

# 4c. Record win on node object (resets hunger, updates allocation history).
#     Payment passed as 0 here because TokenManager.deduct() already deducted
#     the tokens above -- calling record_win with the full payment would
#     double-count the deduction. record_win's token deduction is only used
#     when the node drives its own accounting (i.e. without TokenManager).
tm.nodes[winner_id].record_win(RESOURCE, UNITS, payment=0.0, round_num=ROUND)

# 4d. Audit log -- win event
win_hash = audit.append(
    AUCTION_WIN,
    winner_id,
    {
        "round":    ROUND,
        "resource": RESOURCE,
        "units":    UNITS,
        "payment":  vcg_payment,
        "bid":      winner_bid,
    },
)
ok(f"AuditLog.append(AUCTION_WIN) -- hash={win_hash[:16]}...")

# 4e. Losers: record loss, log to audit chain
for loser_id in losers:
    tm.nodes[loser_id].record_loss(round_num=ROUND)
    audit.append(
        AUCTION_LOSS,
        loser_id,
        {"round": ROUND, "resource": RESOURCE, "bid": bids[loser_id]},
    )

ok(f"record_loss called for {losers}")
ok(f"AuditLog now has {len(audit)} entries")


# ---------------------------------------------------------------------------
# 5. Token accounting verification
# ---------------------------------------------------------------------------

section("5. Token accounting")

dist = tm.get_token_distribution()
step("Token balances after round:")
for nid, tokens in dist.items():
    step(f"  {nid}: {tokens:.4f}")

assert_close(dist["node-2"], balance_before["node-2"] - vcg_payment,
             "node-2 balance (winner, deducted VCG price)")
assert_close(dist["node-1"], balance_before["node-1"],
             "node-1 balance (loser, unchanged)")
assert_close(dist["node-3"], balance_before["node-3"],
             "node-3 balance (loser, unchanged)")

total_before       = sum(balance_before.values())
total_after        = sum(dist.values())
tokens_left_system = total_before - total_after

step(f"\nTotal tokens before : {total_before:.4f}")
step(f"Total tokens after  : {total_after:.4f}")
step(f"Tokens extracted    : {tokens_left_system:.4f}  (should equal VCG payment)")
assert_close(tokens_left_system, vcg_payment,
             "Tokens extracted = VCG payment (accounting closed)")


# ---------------------------------------------------------------------------
# 6. ResourcePool utilization
# ---------------------------------------------------------------------------

section("6. ResourcePool utilization")

util = pool.get_utilization()
step("Utilization after allocation:")
for r, pct in util.items():
    step(f"  {r}: {pct:.4f}%")

market_cpu = pool.capacity[RESOURCE] - pool.reserved[RESOURCE]   # 90 cores
expected_cpu_util = (UNITS / market_cpu) * 100                   # 4/90 * 100
assert_close(util["CPU"], expected_cpu_util,
             f"CPU utilization ({UNITS}/{market_cpu:.0f} cores)")

for r in ["RAM", "STG", "NET"]:
    assert_close(util[r], 0.0, f"{r} utilization (untouched)")


# ---------------------------------------------------------------------------
# 7. Hunger counters
# ---------------------------------------------------------------------------

section("7. Hunger counters")

step(f"node-1 hunger_counter = {node1.hunger_counter}  (expected 1)")
step(f"node-2 hunger_counter = {node2.hunger_counter}  (expected 0, won this round)")
step(f"node-3 hunger_counter = {node3.hunger_counter}  (expected 1)")

if node1.hunger_counter != 1:
    fail(f"node-1 hunger expected 1, got {node1.hunger_counter}")
if node2.hunger_counter != 0:
    fail(f"node-2 hunger expected 0, got {node2.hunger_counter}")
if node3.hunger_counter != 1:
    fail(f"node-3 hunger expected 1, got {node3.hunger_counter}")
ok("All hunger counters correct")


# ---------------------------------------------------------------------------
# 8. Audit log integrity
# ---------------------------------------------------------------------------

section("8. AuditLog integrity")

step(f"Chain length  : {len(audit)} entries")
step(f"Genesis hash  : {audit.genesis_hash[:32]}...")

integrity_ok = audit.verify_integrity()
if not integrity_ok:
    fail("AuditLog.verify_integrity() returned False on unmodified chain")
ok("verify_integrity() returned True -- chain intact")

node2_events = audit.get_events_for_node("node-2")
step(f"Events for node-2: {len(node2_events)}")
if len(node2_events) != 1:
    fail(f"Expected 1 event for node-2, got {len(node2_events)}")
if node2_events[0]["event_type"] != AUCTION_WIN:
    fail(f"Expected AUCTION_WIN for node-2, got {node2_events[0]['event_type']}")
ok("get_events_for_node('node-2') -- 1 entry, event_type=AUCTION_WIN")


# ---------------------------------------------------------------------------
# 9. TokenManager transaction log
# ---------------------------------------------------------------------------

section("9. TokenManager transaction log")

step(f"Total entries: {len(tm.transaction_log)}")
for entry in tm.transaction_log:
    step(
        f"  {entry['node_id']:10s}  {entry['direction']:15s}"
        f"  amount={entry.get('amount', 0):.2f}  reason={entry['reason']}"
    )


# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

print(f"\n{SEP2}")
print("  ALL SMOKE TEST ASSERTIONS PASSED")
print(SEP2)


# ---------------------------------------------------------------------------
# pytest entry point
# ---------------------------------------------------------------------------

def test_smoke():
    """Pytest-discoverable wrapper -- passes iff the script above completed."""
    pass
