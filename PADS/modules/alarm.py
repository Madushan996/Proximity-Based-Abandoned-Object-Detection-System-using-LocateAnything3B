"""
M7 — Alarm & Event Management
Manages the full abandonment lifecycle: person exit detection, T_abandon
countdown, retrieval detection, and alarm dispatch.
"""
import time
import cv2
import json
import numpy as np
from pathlib import Path
from modules.coordinates import CoordinateMapper
from state.object_registry import ObjectRegistry, ObjectState, ObjectStatus
from state.person_registry import PersonRegistry, PersonStatus
from state.event_log import EventLog, EventType


class AlarmManager:
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

        alarm_cfg = cfg.get("alarm", {})
        self.T_abandon = alarm_cfg.get("abandon_timeout_s", 120)
        self.T_retrieve = alarm_cfg.get("retrieve_window_s", 3)
        self.R = cfg.get("association", {}).get("proximity_radius_m", 1.5)
        # Retrieval requires a person within actual reach of the object, not just
        # inside the (larger) association radius. Without this, anyone walking
        # *past* the bag inside R would cancel a legitimate alarm.
        self.retrieve_radius = alarm_cfg.get("retrieve_radius_m", self.R / 2.0)

        # { object_id: exit_timestamp } — when the associated person exited
        self._exit_timers: dict[int, float] = {}
        # { object_id: alarm_record }
        self._alarms: dict[int, dict] = {}
        self._alarm_counter = 1

        Path("logs").mkdir(exist_ok=True)

    def tick(self, frame_time: float, current_frame: np.ndarray,
             fg_blobs_centroids: list) -> list:
        """
        Called every frame. Returns list of newly dispatched alarms this tick.
        fg_blobs_centroids: list of (cx_px, cy_px) for all current foreground blobs.
        """
        new_alarms = []

        for obj in self.obj_reg.of_interest():
            if obj.associated_person_id is None:
                continue
            person = self.person_reg.get(obj.associated_person_id)
            if person is None:
                continue

            # --- Step 1: Watch for person exit ---
            if not person.in_frame and obj.object_id not in self._exit_timers:
                self._exit_timers[obj.object_id] = frame_time
                self.log.log(
                    EventType.PERSON_EXITED,
                    person_id=person.person_id,
                    object_id=obj.object_id,
                    location=(obj.centroid_x, obj.centroid_y),
                )
                print(f"[M7] Person {person.person_id} exited — T_abandon timer started "
                      f"for Object {obj.object_id}")

            # --- Step 2: Check retrieval (any person near object + object disappears) ---
            if self._check_retrieval(obj, fg_blobs_centroids, frame_time):
                self._cancel_alarm(obj, person, frame_time)
                self._exit_timers.pop(obj.object_id, None)
                continue

            # --- Step 3: Count down T_abandon ---
            if obj.object_id in self._exit_timers:
                elapsed = frame_time - self._exit_timers[obj.object_id]
                remaining = self.T_abandon - elapsed

                if remaining <= 0:
                    alarm = self._dispatch_alarm(obj, person, frame_time, current_frame)
                    new_alarms.append(alarm)
                    self._exit_timers.pop(obj.object_id, None)

        return new_alarms

    def _check_retrieval(self, obj: ObjectState, fg_blobs: list, frame_time: float) -> bool:
        """
        An object is retrieved when:
        1. A person is within actual reach (retrieve_radius) of the object, AND
        2. The object's foreground blob has disappeared.

        Requiring reach-distance (rather than the full association radius) stops
        a passer-by who merely walks near the bag from cancelling the alarm.
        """
        # Check if object still has a foreground blob nearby
        obj_still_present = any(
            self.mapper.distance_m(obj.centroid_x, obj.centroid_y,
                                   *self.mapper.to_world(bx, by)) < self.retrieve_radius
            for bx, by in fg_blobs
        )

        if obj_still_present:
            return False

        # Blob gone AND a person is within reach → genuine pick-up
        for person in self.person_reg.in_frame():
            d = self.mapper.distance_m(obj.centroid_x, obj.centroid_y,
                                       person.foot_x, person.foot_y)
            if d <= self.retrieve_radius:
                return True

        return False

    def _cancel_alarm(self, obj: ObjectState, person, frame_time: float) -> None:
        obj.status = ObjectStatus.RETRIEVED
        self.obj_reg.update(obj)
        self.log.log(
            EventType.OBJECT_RETRIEVED,
            person_id=person.person_id,
            object_id=obj.object_id,
            location=(obj.centroid_x, obj.centroid_y),
        )
        print(f"[M7] Object {obj.object_id} retrieved — alarm cancelled")

    def _dispatch_alarm(self, obj: ObjectState, person, frame_time: float,
                        frame: np.ndarray) -> dict:
        alarm_id = f"ALARM_{self._alarm_counter:04d}"
        self._alarm_counter += 1

        # Save snapshot frame
        snapshot_path = f"logs/{alarm_id}_snapshot.jpg"
        annotated = frame.copy()
        x1, y1, x2, y2 = obj.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(annotated, f"ABANDONED OBJECT {obj.object_id}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(snapshot_path, annotated)

        alarm = {
            "alarm_id": alarm_id,
            "object_id": obj.object_id,
            "object_location": (obj.centroid_x, obj.centroid_y),
            "object_bbox": obj.bbox,
            "associated_person_id": person.person_id,
            "exit_time": person.exit_time,
            "alarm_time": frame_time,
            "snapshot_frame": snapshot_path,
        }

        obj.status = ObjectStatus.ALARMED
        self.obj_reg.update(obj)

        self._alarms[obj.object_id] = alarm

        # Write alarm payload to log
        alarm_log_path = f"logs/{alarm_id}.json"
        with open(alarm_log_path, "w") as f:
            json.dump(alarm, f, indent=2)

        self.log.log(
            EventType.ALARM_DISPATCHED,
            person_id=person.person_id,
            object_id=obj.object_id,
            location=(obj.centroid_x, obj.centroid_y),
            metadata={"alarm_id": alarm_id},
        )

        print(f"\n{'='*60}")
        print(f"[M7] *** ALARM DISPATCHED: {alarm_id} ***")
        print(f"     Object {obj.object_id} at {obj.bbox}")
        print(f"     Left by Person {person.person_id}")
        print(f"     Snapshot: {snapshot_path}")
        print(f"{'='*60}\n")

        return alarm

    def get_alarm_countdown(self, object_id: int, frame_time: float) -> float:
        """Returns remaining seconds in T_abandon countdown, or -1 if not counting."""
        if object_id not in self._exit_timers:
            return -1.0
        elapsed = frame_time - self._exit_timers[object_id]
        return max(0.0, self.T_abandon - elapsed)

    def all_alarms(self) -> list:
        return list(self._alarms.values())
