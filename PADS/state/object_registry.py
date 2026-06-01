"""
Object Registry — PADS M2/M5/M6
Maintains state for every foreground object detected in the scene.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Deque, Tuple
import time


# Possible lifecycle states for a tracked object
class ObjectStatus:
    NEW = "new"                    # Just detected, not yet associated
    PENDING = "pending"            # Waiting in pending queue for a person to appear
    ASSOCIATED = "associated"      # Linked to a person, distance being monitored
    OF_INTEREST = "of_interest"    # Separation exceeded; person exited or moving away
    RETRIEVED = "retrieved"        # Someone picked it up — no alarm
    EXPIRED = "expired"            # Pending timeout reached with no association; ignored
    ALARMED = "alarmed"            # Alarm has been dispatched for this object


@dataclass
class ObjectState:
    object_id: int
    centroid_x: float                       # Corrected 2D coordinate (metres or pixels)
    centroid_y: float
    bbox: Tuple[int, int, int, int]         # (x_min, y_min, x_max, y_max) in pixel space
    area_px: float
    first_seen: float                       # Unix timestamp
    label: str = ""                         # open-vocab class name (LocateAnything M2)
    pre_existing: bool = False
    associated_person_id: Optional[int] = None
    association_time: Optional[float] = None
    status: str = ObjectStatus.NEW
    distance_history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=300))
    # distance_history stores (timestamp, distance_m) tuples
    last_updated: float = field(default_factory=time.time)


class ObjectRegistry:
    """In-memory store for all tracked objects, keyed by object_id."""

    def __init__(self):
        self._store: dict[int, ObjectState] = {}
        self._next_id: int = 1

    def next_id(self) -> int:
        oid = self._next_id
        self._next_id += 1
        return oid

    def add(self, obj: ObjectState) -> None:
        self._store[obj.object_id] = obj

    def get(self, object_id: int) -> Optional[ObjectState]:
        return self._store.get(object_id)

    def update(self, obj: ObjectState) -> None:
        obj.last_updated = time.time()
        self._store[obj.object_id] = obj

    def remove(self, object_id: int) -> None:
        self._store.pop(object_id, None)

    def all(self) -> List[ObjectState]:
        return list(self._store.values())

    def active(self) -> List[ObjectState]:
        """Objects that are not pre-existing, expired, or retrieved."""
        return [o for o in self._store.values()
                if not o.pre_existing
                and o.status not in (ObjectStatus.EXPIRED, ObjectStatus.RETRIEVED)]

    def pre_existing(self) -> List[ObjectState]:
        return [o for o in self._store.values() if o.pre_existing]

    def pending(self) -> List[ObjectState]:
        return [o for o in self._store.values() if o.status == ObjectStatus.PENDING]

    def of_interest(self) -> List[ObjectState]:
        return [o for o in self._store.values() if o.status == ObjectStatus.OF_INTEREST]

    def __len__(self) -> int:
        return len(self._store)
