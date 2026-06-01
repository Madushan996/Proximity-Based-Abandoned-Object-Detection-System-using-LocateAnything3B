"""
M6 — Separation Monitoring
Continuously computes the distance between each associated person and their
object. Triggers a SEPARATION EVENT when the sliding-window average exceeds
threshold D for longer than T_grace seconds.
"""
import time
from collections import deque
from modules.coordinates import CoordinateMapper
from state.object_registry import ObjectRegistry, ObjectState, ObjectStatus
from state.person_registry import PersonRegistry, PersonStatus
from state.event_log import EventLog, EventType


class SeparationMonitor:
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

        sep_cfg = cfg.get("separation", {})
        self.D = sep_cfg.get("separation_threshold_m", 2.0)
        self.T_grace = sep_cfg.get("grace_period_s", 8)
        self.T_window = sep_cfg.get("window_s", 1.5)
        self.R = cfg.get("association", {}).get("proximity_radius_m", 1.5)

        # Per-object tracking of when separation threshold was first exceeded
        # { object_id: timestamp_when_exceeded }
        self._separation_start: dict[int, float] = {}

    def process(self, frame_time: float) -> None:
        """Check all associated objects for separation events."""
        for obj in self.obj_reg.active():
            if obj.status not in (ObjectStatus.ASSOCIATED, ObjectStatus.OF_INTEREST):
                continue
            if obj.associated_person_id is None:
                continue

            person = self.person_reg.get(obj.associated_person_id)
            if person is None:
                continue

            # Use last known position even if person has exited frame
            d = self.mapper.distance_m(
                obj.centroid_x, obj.centroid_y,
                person.foot_x, person.foot_y,
            )

            # Record to distance history
            obj.distance_history.append((frame_time, d))
            self.obj_reg.update(obj)

            # Sliding window average
            cutoff = frame_time - self.T_window
            window = [(t, dist) for t, dist in obj.distance_history if t >= cutoff]
            if not window:
                continue
            avg_d = sum(dist for _, dist in window) / len(window)

            if obj.status == ObjectStatus.ASSOCIATED:
                if avg_d > self.D:
                    # Start grace period countdown
                    if obj.object_id not in self._separation_start:
                        self._separation_start[obj.object_id] = frame_time

                    elapsed = frame_time - self._separation_start[obj.object_id]
                    if elapsed >= self.T_grace:
                        self._trigger_separation(obj, person, avg_d, frame_time)
                else:
                    # Person is close again — cancel any pending separation
                    self._separation_start.pop(obj.object_id, None)

            elif obj.status == ObjectStatus.OF_INTEREST:
                # Already in separation — check if the owner (or anyone) returned
                # to the object. We can't rely on the original track ID: ByteTrack
                # commonly reassigns IDs after an occlusion, so an owner who walks
                # back in gets a *new* person_id. Re-claim on proximity of any
                # in-frame person prevents a false alarm when the owner returns.
                if person.in_frame and avg_d <= self.R:
                    self._cancel_separation(obj, person, frame_time)
                    continue
                reclaimer = self._nearest_person_within(obj, self.R)
                if reclaimer is not None:
                    # Re-associate to whoever is now standing with the object.
                    obj.associated_person_id = reclaimer.person_id
                    if obj.object_id not in reclaimer.associated_objects:
                        reclaimer.associated_objects.append(obj.object_id)
                    self.person_reg.update(reclaimer)
                    self._cancel_separation(obj, reclaimer, frame_time)

    def _nearest_person_within(self, obj: ObjectState, radius: float):
        """Return the closest in-frame person within `radius` of the object, else None."""
        best, best_d = None, float("inf")
        for person in self.person_reg.in_frame():
            d = self.mapper.distance_m(obj.centroid_x, obj.centroid_y,
                                       person.foot_x, person.foot_y)
            if d <= radius and d < best_d:
                best, best_d = person, d
        return best

    def _trigger_separation(self, obj: ObjectState, person, avg_d: float, t: float) -> None:
        obj.status = ObjectStatus.OF_INTEREST
        self.obj_reg.update(obj)

        if person.status == PersonStatus.ACTIVE:
            person.status = PersonStatus.OF_INTEREST
            self.person_reg.update(person)

        self._separation_start.pop(obj.object_id, None)

        self.log.log(
            EventType.SEPARATION_DETECTED,
            person_id=person.person_id,
            object_id=obj.object_id,
            location=(obj.centroid_x, obj.centroid_y),
            metadata={"avg_distance_m": round(avg_d, 3)},
        )
        print(f"[M6] SEPARATION: Object {obj.object_id} from Person {person.person_id} "
              f"(avg dist={avg_d:.2f}m)")

    def _cancel_separation(self, obj: ObjectState, person, t: float) -> None:
        obj.status = ObjectStatus.ASSOCIATED
        self.obj_reg.update(obj)
        person.status = PersonStatus.ACTIVE
        self.person_reg.update(person)
        print(f"[M6] Separation cancelled — Person {person.person_id} returned to Object {obj.object_id}")
