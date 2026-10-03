"""The SSD cap holds two copies of the largest session, above a free-disk floor.

2026-09-29: with 25 GiB free the cap was min(100 GiB, free / 4) = 6.25 GiB,
one ~4 GB entry, so replacing a session's saved copy meant deleting it first.
``disk_budget`` keeps the quarter share as the cache's normal size, raises it
to two copies of the largest session whenever the disk above the floor can
hold them, never exceeds the configured cap, and names the disk state in
plain words for the app. Pure arithmetic, at the incident's real scale.
"""

from __future__ import annotations

import pytest

from mtplx.cache_bank.disk_budget import (
    DISK_FLOOR_BYTES,
    DISK_FULL,
    DISK_LOW,
    DISK_OK,
    disk_budget,
)

GIB = 1024**3


def test_the_09_29_disk_holds_the_replacement_beside_its_predecessor():
    """25 GiB free and a 4 GB session: the old rule gave 6.25 GiB, one copy."""

    session = 4_171_484_944  # the 131,735-token entry's logical bytes
    budget = disk_budget(
        configured_max_bytes=100 * GIB,
        free_bytes=25 * GIB,
        store_bytes=session,
        largest_session_bytes=session,
    )
    assert 25 * GIB // 4 < 2 * session, "the old quarter rule held one copy"
    assert budget.cap_bytes >= 2 * session
    assert budget.room_bytes == 25 * GIB - DISK_FLOOR_BYTES
    assert budget.state == DISK_OK and budget.message() is None
    # The cap never reaches into the floor.
    assert budget.cap_bytes <= session + 25 * GIB - DISK_FLOOR_BYTES


def test_a_big_disk_keeps_the_quarter_share_and_the_configured_cap():
    budget = disk_budget(
        configured_max_bytes=100 * GIB,
        free_bytes=285 * GIB,
        store_bytes=0,
        largest_session_bytes=4 * GIB,
    )
    assert budget.cap_bytes == 285 * GIB // 4
    budget = disk_budget(
        configured_max_bytes=100 * GIB,
        free_bytes=900 * GIB,
        store_bytes=0,
        largest_session_bytes=4 * GIB,
    )
    assert budget.cap_bytes == 100 * GIB and not budget.cap_limited_by_disk


def test_the_configured_cap_is_never_exceeded_for_two_copies():
    budget = disk_budget(
        configured_max_bytes=5 * GIB,
        free_bytes=25 * GIB,
        store_bytes=0,
        largest_session_bytes=4 * GIB,
    )
    assert budget.cap_bytes == 5 * GIB
    assert not budget.cap_limited_by_disk
    assert budget.state == DISK_OK  # the disk is fine; the setting is small


def test_a_disk_that_cannot_hold_two_copies_is_low_and_says_so():
    budget = disk_budget(
        configured_max_bytes=100 * GIB,
        free_bytes=12 * GIB,
        store_bytes=4 * GIB,
        largest_session_bytes=4 * GIB,
    )
    assert budget.state == DISK_LOW and budget.low
    assert budget.cap_bytes == 4 * GIB + 2 * GIB  # store plus room above floor
    assert budget.room_bytes == 2 * GIB
    message = budget.message()
    assert "two copies" in message and "kept" in message


@pytest.mark.parametrize("free_gib", (10, 9, 0))
def test_free_disk_at_or_below_the_floor_stops_writes(free_gib):
    budget = disk_budget(
        configured_max_bytes=100 * GIB,
        free_bytes=free_gib * GIB,
        store_bytes=1 * GIB,
        largest_session_bytes=1 * GIB,
    )
    assert budget.state == DISK_FULL and budget.room_bytes == 0
    assert budget.cap_bytes <= 1 * GIB  # no growth into the floor
    assert "stopped saving" in budget.message()


def test_reservations_reduce_the_room_and_unknown_free_space_uses_the_cap():
    budget = disk_budget(
        configured_max_bytes=100 * GIB,
        free_bytes=25 * GIB,
        store_bytes=0,
        largest_session_bytes=4 * GIB,
        reserved_bytes=6 * GIB,
    )
    assert budget.room_bytes == 25 * GIB - DISK_FLOOR_BYTES - 6 * GIB
    unknown = disk_budget(
        configured_max_bytes=100 * GIB,
        free_bytes=None,
        store_bytes=30 * GIB,
        largest_session_bytes=4 * GIB,
        reserved_bytes=6 * GIB,
    )
    assert unknown.cap_bytes == 100 * GIB and unknown.state == DISK_OK
    assert unknown.room_bytes == 94 * GIB
