"""
Streamable PADS pipeline runner.

This is the existing PADS per-frame loop (see PADS/main.py) refactored into a
generator so it can drive a web UI instead of an OpenCV window. It reuses every
PADS module unchanged — M1 scene init, M3 person tracking, M5 association, M6
separation, M7 alarms — and swaps in the LocateAnything VLM for M2 detection.

run_pipeline() yields message dicts:
    {"type": "meta",  "width", "height", "fps", "source"}
    {"type": "frame", "idx", "t", "jpeg": <bytes JPEG>}
    {"type": "event", "event": {...}}      # alarms / lifecycle events
    {"type": "done",  "summary": {...}}

The PADS package must be importable. On the Modal container it is mounted at
/root/pads (added to sys.path here via the PADS_DIR env var).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Make the PADS package importable (modules/, state/, utils/, main.py).
_PADS_DIR = os.environ.get("PADS_DIR", "/root/pads")
if _PADS_DIR and _PADS_DIR not in sys.path:
    sys.path.insert(0, _PADS_DIR)

from modules.coordinates    import CoordinateMapper          # noqa: E402
from modules.foreground     import create_foreground_detector  # noqa: E402
from modules.scene_init     import SceneInitialiser           # noqa: E402
from modules.person_tracker import PersonTracker              # noqa: E402
from modules.association    import AssociationEngine          # noqa: E402
from modules.separation     import SeparationMonitor          # noqa: E402
from modules.alarm          import AlarmManager               # noqa: E402
from state.object_registry  import ObjectRegistry             # noqa: E402
from state.person_registry  import PersonRegistry, PersonStatus  # noqa: E402
from state.event_log        import EventLog                   # noqa: E402
from utils.visualizer       import Visualizer                 # noqa: E402

# Reuse the loop helpers from main.py rather than copy them (single source).
from main import match_blobs_to_registry, video_clock        # noqa: E402


def _encode_jpeg(frame: np.ndarray, quality: int = 75) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


# ── Object-labeling overlay mode (no PADS person/association/alarm logic) ──────

def _label_color(label: str):
    """Stable BGR color per label so each class keeps one colour across frames."""
    h = abs(hash(label))
    return (37 + h % 180, 50 + (h >> 8) % 180, 60 + (h >> 16) % 180)


def _draw_labeled(frame: np.ndarray, boxes: list) -> np.ndarray:
    out = frame.copy()
    for b in boxes:
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        color = _label_color(b.get("label", "object"))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = b.get("label", "object")
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _iou(a, b) -> float:
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / (area_a + area_b - inter)


def _dedup_boxes(boxes: list, iou_thresh: float = 0.5) -> list:
    """Greedy NMS across categories: drop boxes that overlap a kept one.

    The model gives no confidence score, so we keep the larger box of an
    overlapping pair (tends to be the fuller, less-clipped detection).
    """
    kept: list = []
    for b in sorted(boxes, key=lambda x: (x["x2"] - x["x1"]) * (x["y2"] - x["y1"]),
                    reverse=True):
        if all(_iou(b, k) < iou_thresh for k in kept):
            kept.append(b)
    return kept


def _detect_labeled_frame(frame, worker, categories, resize_long, gen_mode, min_area):
    from PIL import Image
    h, w = frame.shape[:2]
    scale = 1.0
    proc = frame
    if resize_long and max(h, w) > resize_long:
        scale = resize_long / float(max(h, w))
        proc = cv2.resize(frame, (int(w * scale), int(h * scale)))
    rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
    raw = worker.detect_labeled(Image.fromarray(rgb), categories, generation_mode=gen_mode)
    boxes = []
    for b in raw:
        x1, y1 = b["x1"] / scale, b["y1"] / scale
        x2, y2 = b["x2"] / scale, b["y2"] / scale
        x1, x2 = sorted((max(0.0, x1), min(float(w), x2)))
        y1, y2 = sorted((max(0.0, y1), min(float(h), y2)))
        if (x2 - x1) * (y2 - y1) >= min_area:
            boxes.append({"label": b["label"], "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return _dedup_boxes(boxes)


def run_labeling(
    video_path: str,
    categories: list,
    worker,
    infer_every: int = 24,
    resize_long: int = 768,
    gen_mode: str = "hybrid",
    min_area: int = 0,
    jpeg_quality: int = 85,
):
    """Detect-and-label every `infer_every` frames; draw boxes on every frame.

    Yields the same meta/frame/done schema as run_pipeline so the Modal render
    thread can pipe frames to ffmpeg unchanged. No person tracking / association
    / alarms — this is a pure open-vocabulary labeling overlay.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    yield {"type": "meta", "width": width, "height": height, "fps": fps,
           "total_frames": total, "source": Path(video_path).name}

    cached: list = []
    seen_labels: set = set()
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % max(1, infer_every) == 1:
            print(f"[LABEL] detect @frame {frame_count} … ", end="", flush=True)
            _t0 = time.perf_counter()
            cached = _detect_labeled_frame(frame, worker, categories,
                                           resize_long, gen_mode, min_area)
            for b in cached:
                seen_labels.add(b["label"])
            print(f"{len(cached)} box(es) ({time.perf_counter() - _t0:.1f}s)", flush=True)
        annotated = _draw_labeled(frame, cached)
        yield {"type": "frame", "idx": frame_count, "t": round(frame_count / fps, 2),
               "jpeg": _encode_jpeg(annotated, jpeg_quality)}

    cap.release()
    yield {"type": "done", "summary": {
        "frames": frame_count,
        "alarms": 0,
        "labels": sorted(seen_labels),
    }}


def run_pipeline(
    video_path: str,
    cfg: dict,
    worker=None,
    stream_every: int = 1,
    jpeg_quality: int = 75,
):
    """Process `video_path` and yield UI messages. See module docstring."""
    source = video_path
    is_live = False

    # On the GPU container person detection runs on CUDA; force it on.
    device = "cuda" if worker is not None else cfg.get("person", {}).get("device", "cpu")
    cfg.setdefault("person", {})["device"] = device
    cfg["person"]["half"] = device != "cpu"

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 24
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    yield {"type": "meta", "width": width, "height": height,
           "fps": fps_src, "total_frames": total, "source": Path(source).name}

    # ── components (identical wiring to main.py) ──────────────────────────────
    obj_reg    = ObjectRegistry()
    person_reg = PersonRegistry()
    event_log  = EventLog("logs/events.jsonl")
    mapper     = CoordinateMapper(cfg)
    fg_det     = create_foreground_detector(cfg, worker=worker)   # ← M2 = LocateAnything
    visualizer = Visualizer(cfg)

    scene_init = SceneInitialiser(fg_det, mapper, obj_reg, event_log, cfg)
    assoc      = AssociationEngine(mapper, obj_reg, person_reg, event_log, cfg)
    sep_mon    = SeparationMonitor(mapper, obj_reg, person_reg, event_log, cfg)
    alarm_mgr  = AlarmManager(mapper, obj_reg, person_reg, event_log, cfg)
    tracker    = PersonTracker(cfg, mapper, person_reg)

    init_t0 = cfg.get("input", {}).get("init_frames", 30) / fps_src
    scene_init.run(cap, now=init_t0)
    fg_det.freeze()

    min_area = cfg.get("foreground", {}).get("obj_min_area", 800)
    person_ttl = cfg.get("person", {}).get("person_ttl_s", 300)
    prune_every = 300

    frame_count = 0
    frame_time = init_t0
    last_frame = None
    blob_centroids: list = []

    # ── main loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame
        frame_count += 1
        frame_time = video_clock(cap, frame_count, fps_src, is_live)

        tracker.process(frame, frame_time)
        person_boxes = [p.bbox for p in person_reg.in_frame()]
        fg_det.set_person_regions(person_boxes, frame.shape)

        blobs, fg_mask = fg_det.process(frame)
        new_obj_ids, blob_centroids = match_blobs_to_registry(
            blobs, obj_reg, mapper, min_area, person_boxes, frame_time
        )

        if new_obj_ids:
            assoc.process_new_objects(new_obj_ids, frame_time)
        assoc.retry_pending(frame_time)
        sep_mon.process(frame_time)

        new_alarms = alarm_mgr.tick(frame_time, frame, blob_centroids)
        for alarm in new_alarms:
            _obj = obj_reg.get(alarm["object_id"])
            yield {"type": "event", "event": {
                "kind": "ALARM",
                "alarm_id": alarm["alarm_id"],
                "object_id": alarm["object_id"],
                "label": getattr(_obj, "label", "") if _obj else "",
                "person_id": alarm["associated_person_id"],
                "bbox": list(alarm["object_bbox"]),
                "t": round(frame_time, 2),
            }}

        if frame_count % prune_every == 0:
            keep = {o.associated_person_id for o in obj_reg.active()
                    if o.associated_person_id is not None}
            person_reg.prune(frame_time, person_ttl, keep_ids=keep)

        annotated = visualizer.draw(frame, obj_reg, person_reg, alarm_mgr, frame_time)

        if stream_every <= 1 or frame_count % stream_every == 0:
            yield {"type": "frame", "idx": frame_count, "t": round(frame_time, 2),
                   "jpeg": _encode_jpeg(annotated, jpeg_quality)}

    # ── end-of-video flush (mirrors main.py) ──────────────────────────────────
    for person in person_reg.in_frame():
        if person.status in (PersonStatus.OF_INTEREST, PersonStatus.ACTIVE):
            person.in_frame = False
            person.exit_time = frame_time
            if person.status == PersonStatus.ACTIVE:
                person.status = PersonStatus.EXITED
            person_reg.update(person)

    for extra_t in range(1, int(alarm_mgr.T_abandon) + 2):
        flush_alarms = alarm_mgr.tick(frame_time + extra_t, last_frame, blob_centroids)
        for alarm in flush_alarms:
            _obj = obj_reg.get(alarm["object_id"])
            yield {"type": "event", "event": {
                "kind": "ALARM",
                "alarm_id": alarm["alarm_id"],
                "object_id": alarm["object_id"],
                "label": getattr(_obj, "label", "") if _obj else "",
                "person_id": alarm["associated_person_id"],
                "bbox": list(alarm["object_bbox"]),
                "t": round(frame_time + extra_t, 2),
                "note": "end-of-video flush",
            }}
        if flush_alarms:
            break

    cap.release()
    yield {"type": "done", "summary": {
        "frames": frame_count,
        "events": event_log.summary(),
        "alarms": len(alarm_mgr.all_alarms()),
    }}
