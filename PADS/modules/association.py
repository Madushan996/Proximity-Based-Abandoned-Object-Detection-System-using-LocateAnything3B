"""
M5 — Proximity Association Engine
Links newly appeared foreground objects to the nearest person within radius R.
Unresolvable associations go into a pending queue for T_pending seconds.
"""
import time
import math
from typing import List, Optional, Tuple
from modules.coordinates import CoordinateMapper
from state.object_registry import ObjectRegistry, ObjectState, ObjectStatus
from state.person_registry import PersonRegistry, PersonState
from state.event_log import EventLog, EventType


class AssociationEngine:
    def __init__(
        self,
        coord_mapper: CoordinateMapper,
        object_registry: ObjectRegistry,
        person_registry: PersonRegistry,
        event_log: EventLog,
        cfg: dict,
    ):
        self.mapper = coord_mapper
        self.obj_reg = object_registry
        self.person_reg = person_registry
        self.log = event_log

        assoc_cfg = cfg.get("association", {})
        self.R = assoc_cfg.get("proximity_radius_m", 1.5)
        self.T_pending = assoc_cfg.get("pending_timeout_s", 10)
        # How far back to search a person's trajectory for proximity to the
        # object. Detection of a dropped object lags its placement (the blob must
        # mature and the owner must step off it), by which time the owner may
        # already be >R away. Looking back over their track recovers the owner.
        self.lookback = assoc_cfg.get("ownership_lookback_s", max(self.T_pending, 5))

    def _closest_approach(self, obj: ObjectState, person: PersonState,
                          frame_time: float) -> float:
        """Minimum distance between the object and this person, considering both
        the person's current position and their recent trajectory.

        Searching the trajectory (not just the current foot-point) is what lets
        us link an object to an owner who has already walked away by the time the
        object is confirmed. The window starts shortly before the object was
        first seen so we capture the placement moment.
        """
        best = float("inf")
        if person.in_frame:
            best = self.mapper.distance_m(obj.centroid_x, obj.centroid_y,
                                          person.foot_x, person.foot_y)
        window_start = obj.first_seen - self.lookback
        for hx, hy, ht in person.position_history:
            if ht < window_start or ht > frame_time:
                continue
            d = self.mapper.distance_m(obj.centroid_x, obj.centroid_y, hx, hy)
            if d < best:
                best = d
        return best

    def try_associate(self, obj: ObjectState, frame_time: float) -> bool:
        """
        Attempt to associate obj with the most likely owner within R, where
        "within R" is evaluated against each person's recent trajectory as well
        as their current position. Returns True if an association was locked.
        """
        persons = self.person_reg.all()
        if not persons:
            return False

        # Each candidate: (closest_approach_distance, person)
        candidates: List[Tuple[float, PersonState]] = []
        for person in persons:
            d = self._closest_approach(obj, person, frame_time)
            if d <= self.R:
                candidates.append((d, person))

        if not candidates:
            return False

        # Tiebreak: closest first; prefer in-frame owners; then longer dwell.
        candidates.sort(key=lambda x: (
            round(x[0], 3),
            0 if x[1].in_frame else 1,
            -len(x[1].position_history),
        ))

        _, winner = candidates[0]
        self._lock_association(obj, winner, frame_time)
        return True

    def _lock_association(self, obj: ObjectState, person: PersonState, t: float) -> None:
        obj.associated_person_id = person.person_id
        obj.association_time = t
        obj.status = ObjectStatus.ASSOCIATED
        self.obj_reg.update(obj)

        if obj.object_id not in person.associated_objects:
            person.associated_objects.append(obj.object_id)
        self.person_reg.update(person)

        self.log.log(
            EventType.ASSOCIATION_LOCKED,
            person_id=person.person_id,
            object_id=obj.object_id,
            location=(obj.centroid_x, obj.centroid_y),
            metadata={"distance_m": self.mapper.distance_m(
                obj.centroid_x, obj.centroid_y, person.foot_x, person.foot_y
            )},
        )
        print(f"[M5] Association locked: Object {obj.object_id} → Person {person.person_id}")

    def process_new_objects(self, new_obj_ids: List[int], frame_time: float) -> None:
        """Try to associate each new object; put failures into pending queue."""
        for oid in new_obj_ids:
            obj = self.obj_reg.get(oid)
            if obj is None or obj.pre_existing:
                continue
            success = self.try_associate(obj, frame_time)
            if not success:
                obj.status = ObjectStatus.PENDING
                self.obj_reg.update(obj)
                print(f"[M5] Object {oid} unassociated — entering pending queue")

    def retry_pending(self, frame_time: float) -> None:
        """Retry association for objects in the pending queue. Expire if timed out."""
        for obj in self.obj_reg.pending():
            elapsed = frame_time - obj.first_seen
            if elapsed > self.T_pending:
                obj.status = ObjectStatus.EXPIRED
                self.obj_reg.update(obj)
                print(f"[M5] Object {obj.object_id} pending timeout — expired")
                continue
            self.try_associate(obj, frame_time)
