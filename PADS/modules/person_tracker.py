"""
M3 — Person Detection & Tracking
Uses YOLOv8n (person class only) + ByteTrack to detect and maintain
persistent person identities across frames.
"""
import time
import numpy as np
from ultralytics import YOLO
from modules.coordinates import CoordinateMapper
from state.person_registry import PersonRegistry, PersonState, PersonStatus


class PersonTracker:
    def __init__(self, cfg: dict, coord_mapper: CoordinateMapper, person_registry: PersonRegistry):
        person_cfg = cfg.get("person", {})
        model_path  = person_cfg.get("model", "yolov8n.pt")
        self.conf   = person_cfg.get("confidence", 0.4)
        self.iou    = person_cfg.get("iou", 0.5)
        self.tracker_cfg = person_cfg.get("tracker", "bytetrack.yaml")
        device      = person_cfg.get("device", "cpu")

        self.half = person_cfg.get("half", False)   # FP16 inference (GPU only)

        print(f"[M3] Loading YOLO model: {model_path} on {device}"
              + (" [FP16]" if self.half else ""))
        self.model = YOLO(model_path)
        self.device = device
        self.mapper = coord_mapper
        self.person_reg = person_registry

        # Track which person IDs were seen in the last frame
        self._prev_ids: set[int] = set()

    def process(self, frame: np.ndarray, frame_time: float) -> list:
        """
        Run person detection + tracking on a frame.
        Updates PersonRegistry in place.
        Returns list of (person_id, foot_x_px, foot_y_px, bbox) for current frame.
        """
        results = self.model.track(
            frame,
            persist=True,
            classes=[0],           # COCO class 0 = person
            conf=self.conf,
            iou=self.iou,
            tracker=self.tracker_cfg,
            device=self.device,
            half=self.half,        # FP16 on GPU: ~2× faster on T4/A100
            verbose=False,
        )

        current_ids: set[int] = set()
        detections = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            if boxes.id is not None:
                for box, track_id in zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy()):
                    pid = int(track_id)
                    x1, y1, x2, y2 = box
                    # Foot point = bottom-centre of bounding box
                    foot_px = (float((x1 + x2) / 2), float(y2))
                    foot_wx, foot_wy = self.mapper.to_world(*foot_px)
                    bbox = (int(x1), int(y1), int(x2), int(y2))

                    current_ids.add(pid)
                    detections.append((pid, foot_px[0], foot_px[1], bbox, foot_wx, foot_wy))

                    existing = self.person_reg.get(pid)
                    if existing is None:
                        # New person entering the scene
                        person = PersonState(
                            person_id=pid,
                            foot_x=foot_wx,
                            foot_y=foot_wy,
                            bbox=bbox,
                            in_frame=True,
                            status=PersonStatus.ACTIVE,
                        )
                        person.position_history.append((foot_wx, foot_wy, frame_time))
                        self.person_reg.add(person)
                    else:
                        # Update existing person
                        if existing.status == PersonStatus.EXITED:
                            existing.status = PersonStatus.RETURNED
                        existing.foot_x = foot_wx
                        existing.foot_y = foot_wy
                        existing.bbox = bbox
                        existing.in_frame = True
                        existing.exit_time = None
                        existing.position_history.append((foot_wx, foot_wy, frame_time))
                        self.person_reg.update(existing)

        # Mark persons not seen this frame as exited
        exited_ids = self._prev_ids - current_ids
        for pid in exited_ids:
            person = self.person_reg.get(pid)
            if person and person.in_frame:
                person.in_frame = False
                person.exit_time = frame_time
                if person.status == PersonStatus.OF_INTEREST:
                    # Keep OF_INTEREST → M7 will handle the timer
                    pass
                elif person.status == PersonStatus.ACTIVE:
                    person.status = PersonStatus.EXITED
                self.person_reg.update(person)

        self._prev_ids = current_ids
        return detections
