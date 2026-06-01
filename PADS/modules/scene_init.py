"""
M1 — Scene Initialisation
Runs once at startup. Snapshots all objects already present in the scene
and marks them pre_existing=True so they are never flagged as abandoned.
"""
import time
import numpy as np
from modules.coordinates import CoordinateMapper
from state.object_registry import ObjectRegistry, ObjectState, ObjectStatus
from state.event_log import EventLog, EventType


class SceneInitialiser:
    def __init__(
        self,
        fg_detector,   # ForegroundDetector or YOLOObjectDetector (duck-typed)
        coord_mapper: CoordinateMapper,
        object_registry: ObjectRegistry,
        event_log: EventLog,
        cfg: dict,
    ):
        self.fg = fg_detector
        self.mapper = coord_mapper
        self.obj_reg = object_registry
        self.log = event_log
        self.init_frames = cfg.get("input", {}).get("init_frames", 30)

    def run(self, cap, now: float = None) -> None:
        """
        Read init_frames frames from the capture, warm up the background
        model, then detect all objects in the final stabilised frame and
        mark them pre_existing.

        `now` is the timeline value (video seconds for files, wall-clock for
        live) used as first_seen so pre-existing timestamps match the main loop.
        """
        if now is None:
            now = time.time()
        print(f"[M1] Warming up background model over {self.init_frames} frames...")
        last_frame = None

        for i in range(self.init_frames):
            ret, frame = cap.read()
            if not ret:
                break
            self.fg.update_background_only(frame)
            last_frame = frame

        if last_frame is None:
            print("[M1] Warning: no frames read during init.")
            return

        # Run one full detection pass on the warmed-up background
        blobs, _ = self.fg.process(last_frame)

        for blob in blobs:
            # Use the blob's ground-contact point (bottom-centre), matching M2/M3.
            x1, y1, x2, y2 = blob.bbox
            px, py = (x1 + x2) / 2.0, float(y2)
            wx, wy = self.mapper.to_world(px, py)
            oid = self.obj_reg.next_id()
            obj = ObjectState(
                object_id=oid,
                centroid_x=wx,
                centroid_y=wy,
                bbox=blob.bbox,
                area_px=blob.area_px,
                first_seen=now,
                pre_existing=True,
                status=ObjectStatus.EXPIRED,  # pre-existing objects are effectively ignored
            )
            self.obj_reg.add(obj)
            self.log.log(
                EventType.OBJECT_APPEARED,
                object_id=oid,
                location=(wx, wy),
                metadata={"pre_existing": True},
            )

        print(f"[M1] Scene initialised. {len(blobs)} pre-existing objects marked.")
