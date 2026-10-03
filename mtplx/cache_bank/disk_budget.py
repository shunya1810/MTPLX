"""How much disk the SSD session cache may use, and when the disk is low.

The cache replaces a conversation's saved copy by writing the new copy first
and retiring the old one only after the new one is installed. That needs room
for two copies of a conversation at once, so the cap must hold two copies of
the largest conversation the cache holds. It must also never push the
user's free disk below a floor.

The cap used to be a quarter of the free disk, re-read at every write. On a
disk with 25 GiB free that is 6.25 GiB: room for one 4 GB conversation and
not its replacement, so every write evicted the only copy before writing the
new one (2026-09-29, a 138K request found nothing to restore). The quarter
share stays as the normal size of the cache; it is raised to two copies of
the largest conversation whenever the disk above the floor can hold them.

Pure arithmetic, no IO: the tier reads the free space and the store's size
and asks this module what they allow.
"""

from __future__ import annotations

from dataclasses import dataclass

GIB = 1024**3

#: Free disk the cache never writes into. Below it the cache writes nothing.
DISK_FLOOR_BYTES = 10 * GIB

#: ``DiskBudget.state`` values.
DISK_OK = "ok"
DISK_LOW = "low"
DISK_FULL = "full"


def _gib(value: int) -> str:
    return f"{value / GIB:.1f} GiB"


@dataclass(frozen=True)
class DiskBudget:
    """What the disk allows the cache right now.

    ``cap_bytes``: the most the store may hold once writes in flight settle.
    ``room_bytes``: what one more write may add now without taking free disk
    below the floor, after the bytes other writes in flight have reserved.
    ``state``: ``ok``; ``low`` when the disk above the floor cannot hold two
    copies of the largest conversation, so a new copy may be skipped (the
    one already saved is kept); ``full`` when free disk is at or below the
    floor and the cache writes nothing.
    """

    cap_bytes: int
    room_bytes: int
    free_bytes: int
    floor_bytes: int
    store_bytes: int
    largest_session_bytes: int
    configured_max_bytes: int
    state: str

    @property
    def low(self) -> bool:
        return self.state != DISK_OK

    @property
    def cap_limited_by_disk(self) -> bool:
        """The disk, not the configured maximum, sets the cap."""

        return self.cap_bytes < self.configured_max_bytes

    def message(self) -> str | None:
        """The disk state in plain words for the app; None when all is well."""

        if self.state == DISK_FULL:
            return (
                f"Only {_gib(self.free_bytes)} of disk space is free, and the "
                f"SSD cache always leaves {_gib(self.floor_bytes)} free, so it "
                "has stopped saving conversations. Free up disk space to turn "
                "it back on."
            )
        if self.state == DISK_LOW:
            return (
                f"Free disk space is low ({_gib(self.free_bytes)}). The SSD "
                "cache needs room for two copies of your largest conversation "
                f"({_gib(self.largest_session_bytes)}) while it saves a new one, "
                f"and it always leaves {_gib(self.floor_bytes)} free, so it may "
                "skip saving new progress. Copies it already saved are kept."
            )
        return None


def disk_budget(
    *,
    configured_max_bytes: int,
    free_bytes: int | None,
    store_bytes: int,
    largest_session_bytes: int,
    reserved_bytes: int = 0,
    floor_bytes: int = DISK_FLOOR_BYTES,
) -> DiskBudget:
    """The cap and the room the disk allows.

    ``store_bytes`` is what the cache holds on disk now (part of the space it
    may use), ``largest_session_bytes`` the logical size of the largest
    conversation it holds or is writing, and ``reserved_bytes`` what writes
    already in flight have claimed.

    * usable: free disk above the floor.
    * ceiling: the store plus the usable space, the most it could ever hold.
    * share: a quarter of the space the cache can use (free plus its own
      bytes), the normal size of the cache.
    * cap: the share, raised to two copies of the largest conversation, never
      above the ceiling or the configured maximum.

    ``free_bytes=None`` means the free space could not be read (the tier
    has always fallen back to its configured cap then): the configured cap
    bounds the store, a write may add up to one cap's worth before the store
    settles back under it, and the state stays ``ok``.
    """

    configured = max(1, int(configured_max_bytes))
    floor = max(0, int(floor_bytes))
    store = max(0, int(store_bytes))
    largest = max(0, int(largest_session_bytes))
    reserved = max(0, int(reserved_bytes))
    if free_bytes is None:
        return DiskBudget(
            cap_bytes=configured,
            room_bytes=max(0, configured - reserved),
            free_bytes=-1,
            floor_bytes=floor,
            store_bytes=store,
            largest_session_bytes=largest,
            configured_max_bytes=configured,
            state=DISK_OK,
        )
    free = max(0, int(free_bytes))
    usable = max(0, free - floor)
    room = max(0, usable - reserved)
    ceiling = store + usable
    share = (free + store) // 4
    two_copies = 2 * largest
    cap = min(configured, ceiling, max(share, two_copies))
    if usable <= 0:
        state = DISK_FULL
    elif ceiling < two_copies:
        state = DISK_LOW
    else:
        state = DISK_OK
    return DiskBudget(
        cap_bytes=int(cap),
        room_bytes=int(room),
        free_bytes=free,
        floor_bytes=floor,
        store_bytes=store,
        largest_session_bytes=largest,
        configured_max_bytes=configured,
        state=state,
    )
