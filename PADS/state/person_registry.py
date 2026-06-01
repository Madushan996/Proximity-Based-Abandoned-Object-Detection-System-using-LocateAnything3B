"""
Person Registry — PADS M3/M5/M6/M7
Maintains state for every tracked person in the scene.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Deque, Tuple
import time


class PersonStatus:
    ACTIVE = "active"              # In frame, being tracked normally
    OF_INTEREST = "of_interest"   # Separation detected; monitoring exit
    EXITED = "exited"             # Left the frame; abandonment timer running
    RETURNED = "returned"         # Came back to their object within T_abandon


@dataclass
class PersonState:
    person_id: int
    foot_x: float                              # Corrected 2D ground-plane coordinate
    foot_y: float
    bbox: Tuple[int, int, int, int]            # (x_min, y_min, x_max, y_max) pixel space
    position_history: Deque[Tuple[float, float, float]] = field(
        default_factory=lambda: deque(maxlen=1000)
    )
    # position_history stores (x, y, timestamp) tuples
    in_frame: bool = True
    exit_time: Optional[float] = None
    status: str = PersonStatus.ACTIVE
    associated_objects: List[int] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


class PersonRegistry:
    """In-memory store for all tracked persons, keyed by person_id."""

    def __init__(self):
        self._store: dict[int, PersonState] = {}

    def add(self, person: PersonState) -> None:
        self._store[person.person_id] = person

    def get(self, person_id: int) -> Optional[PersonState]:
        return self._store.get(person_id)

    def update(self, person: PersonState) -> None:
        person.last_updated = time.time()
        self._store[person.person_id] = person

    def remove(self, person_id: int) -> None:
        self._store.pop(person_id, None)

    def prune(self, now: float, ttl: float, keep_ids: set = None) -> int:
        """Drop persons who left the frame more than `ttl` seconds ago.

        Without this the store grows for every ByteTrack ID ever seen — an
        unbounded leak on a 24/7 feed. Persons referenced by `keep_ids` (e.g.
        the owner of an object still being monitored) are always retained so a
        pending alarm never loses its associated person.
        Returns the number of records removed.
        """
        keep_ids = keep_ids or set()
        doomed = [
            pid for pid, p in self._store.items()
            if pid not in keep_ids
            and not p.in_frame
            and p.exit_time is not None
            and (now - p.exit_time) > ttl
        ]
        for pid in doomed:
            del self._store[pid]
        return len(doomed)

    def all(self) -> List[PersonState]:
        return list(self._store.values())

    def in_frame(self) -> List[PersonState]:
        return [p for p in self._store.values() if p.in_frame]

    def exited(self) -> List[PersonState]:
        return [p for p in self._store.values()
                if not p.in_frame and p.status == PersonStatus.EXITED]

    def of_interest(self) -> List[PersonState]:
        return [p for p in self._store.values() if p.status == PersonStatus.OF_INTEREST]

    def get_all_ids(self) -> List[int]:
        return list(self._store.keys())

    def __len__(self) -> int:
        return len(self._store)
