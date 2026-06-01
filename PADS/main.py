"""
PADS — Proximity-Based Abandoned Object Detection System
Main pipeline entry point.

Usage:
    python main.py                          # use config.yaml defaults
    python main.py --source SampleVideos/Sample1.mp4
    python main.py --source 0               # webcam
    python main.py --no-display             # headless / server mode
"""
import argparse
import time
import math
import cv2
import yaml
import numpy as np
from pathlib import Path

from modules.coordinates   import CoordinateMapper
from modules.foreground    import ForegroundDetector, create_foreground_detector
from modules.scene_init    import SceneInitialiser
from modules.person_tracker import PersonTracker
from modules.association   import AssociationEngine
from modules.separation    import SeparationMonitor
from modules.alarm         import AlarmManager
from state.object_registry import ObjectRegistry, ObjectState, ObjectStatus
from state.person_registry import PersonRegistry
from state.event_log       import EventLog
from utils.visualizer      import Visualizer


# ── helpers ──────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Pass --config <file> or create config.yaml in the working directory."
        )
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} did not parse to a mapping.")
    return cfg


def resolve_device(requested: str) -> tuple[str, bool]:
    """Resolve the inference device and whether FP16 is usable.

    "auto" → CUDA if a GPU is visible (e.g. Colab T4), else CPU.
    FP16 (half) is only enabled on CUDA — it is a no-op / slower on CPU.
    Returns (device_string, half).
    """
    req = str(requested).lower()
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False

    if req in ("auto", ""):
        device = "cuda" if cuda_available else "cpu"
    else:
        device = req
        if device in ("cuda", "0", "gpu") and not cuda_available:
            print("[PADS] WARNING: device requested GPU but CUDA is unavailable — falling back to CPU.")
            device = "cpu"
    half = device not in ("cpu",)
    return device, half


def gui_available() -> bool:
    """True if OpenCV was built with a working HighGUI backend.

    Colab/headless servers ship a headless OpenCV where cv2.imshow raises.
    Probing once lets us degrade gracefully instead of crashing mid-run.
    """
    try:
        test = np.zeros((1, 1, 3), dtype=np.uint8)
        cv2.namedWindow("__pads_probe__", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("__pads_probe__", test)
        cv2.waitKey(1)
        cv2.destroyWindow("__pads_probe__")
        return True
    except Exception:
        return False


def video_clock(cap, frame_count: int, fps: float, is_live: bool) -> float:
    """Return the timeline clock used for all timers.

    For recorded files this is *video time* (seconds of footage elapsed), so
    T_grace/T_abandon/T_pending measure real footage duration regardless of how
    fast or slow the machine decodes. For a live camera index there is no media
    timestamp, so wall-clock is the correct timeline.
    """
    if is_live:
        return time.time()
    pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    if pos_ms and pos_ms > 0:
        return pos_ms / 1000.0
    return frame_count / fps if fps > 0 else float(frame_count)


def _blob_inside_person(px: float, py: float, person_boxes: list) -> bool:
    """Return True if pixel point (px, py) falls inside any tracked person bbox."""
    for x1, y1, x2, y2 in person_boxes:
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True
    return False


def _blob_foot_px(blob) -> tuple[float, float]:
    """Ground-contact point of a blob = bottom-centre of its bbox.

    The homography (and the flat pixel scale) maps the *ground plane*. A blob's
    geometric centroid floats above the floor, so mapping it gives a world
    position that drifts with object height. Persons already use foot-points
    (M3); objects must too, or person↔object distances are not comparable.
    """
    x1, y1, x2, y2 = blob.bbox
    return (x1 + x2) / 2.0, float(y2)


def match_blobs_to_registry(
    blobs,
    obj_reg: ObjectRegistry,
    mapper: CoordinateMapper,
    min_area: float,
    person_boxes: list,
    frame_time: float,
) -> tuple[list[int], list]:
    """
    Compare current foreground blobs against the object registry.

    Fix 1 — centroid locking: once an object is ASSOCIATED or beyond, its
    centroid/bbox is frozen. MOG2 noise on a stationary bag must not shift the
    stored position, which would break distance calculations.

    Fix 2 — person-overlap filter: blobs whose centroid sits inside a tracked
    person bbox are the person themselves moving, not new objects. Skip them.

    Returns (list of new object_ids, list of (cx_px, cy_px) for all blobs).
    """
    LOCKED_STATUSES = {ObjectStatus.ASSOCIATED, ObjectStatus.OF_INTEREST,
                       ObjectStatus.ALARMED, ObjectStatus.RETRIEVED}

    existing = {o.object_id: o for o in obj_reg.active()}
    matched_ids: set[int] = set()
    new_ids: list[int] = []
    blob_centroids: list[tuple] = []

    for blob in blobs:
        px, py = blob.centroid_px
        # Map the blob's ground-contact point (bottom-centre), not its centroid.
        fx, fy = _blob_foot_px(blob)
        # blob_centroids feeds M7 retrieval, which compares against foot-world
        # object positions — so report foot-points here too for consistency.
        blob_centroids.append((fx, fy))

        # Fix 2: skip blobs that are just a person moving
        if _blob_inside_person(px, py, person_boxes):
            continue

        wx, wy = mapper.to_world(fx, fy)

        # Try to match to nearest existing object
        best_match = None
        best_dist = float("inf")
        for oid, obj in existing.items():
            if oid in matched_ids:
                continue
            d = mapper.distance_m(wx, wy, obj.centroid_x, obj.centroid_y)
            if d < best_dist:
                best_dist = d
                best_match = oid

        # Centroid tolerance: ~50px converted to metres, min 0.5m
        tol = max(50 / mapper.scale, 0.5)

        if best_match is not None and best_dist < tol:
            obj = existing[best_match]
            # Fix 1: only update position for NEW/PENDING objects (not locked ones)
            if obj.status not in LOCKED_STATUSES:
                obj.centroid_x = wx
                obj.centroid_y = wy
                obj.bbox = blob.bbox
                obj.area_px = blob.area_px
                obj_reg.update(obj)
            matched_ids.add(best_match)
        else:
            # New object — register it
            oid = obj_reg.next_id()
            new_obj = ObjectState(
                object_id=oid,
                centroid_x=wx,
                centroid_y=wy,
                bbox=blob.bbox,
                area_px=blob.area_px,
                first_seen=frame_time,
                label=getattr(blob, "label", ""),
                pre_existing=False,
                status=ObjectStatus.NEW,
            )
            obj_reg.add(new_obj)
            new_ids.append(oid)

    return new_ids, blob_centroids


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PADS Abandoned Object Detection")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--source",     default=None, help="Override video source")
    parser.add_argument("--no-display", action="store_true", help="Headless mode")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.source is not None:
        cfg["input"]["source"] = args.source
    display = not args.no_display

    source = cfg["input"]["source"]
    # Try integer (webcam index) else string (file path)
    try:
        source = int(source)
        is_live = True
    except (ValueError, TypeError):
        is_live = False

    # Resolve inference device once (auto-detects Colab/T4 GPU) and push it into
    # the config so M3/M2-YOLO pick it up. FP16 is enabled automatically on GPU.
    device, half = resolve_device(cfg.get("person", {}).get("device", "auto"))
    cfg.setdefault("person", {})["device"] = device
    cfg["person"]["half"] = half
    print(f"[PADS] Inference device: {device}" + (" [FP16]" if half else ""))

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    fps_src   = cap.get(cv2.CAP_PROP_FPS) or 24
    width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[PADS] Source: {source} | {width}x{height} @ {fps_src:.1f}fps")

    # ── Initialise all components ─────────────────────────────────────────
    obj_reg    = ObjectRegistry()
    person_reg = PersonRegistry()
    event_log  = EventLog("logs/events.jsonl")
    mapper     = CoordinateMapper(cfg)
    fg_det     = create_foreground_detector(cfg)   # MOG2, KNN, or YOLO
    visualizer = Visualizer(cfg)

    scene_init = SceneInitialiser(fg_det, mapper, obj_reg, event_log, cfg)
    assoc      = AssociationEngine(mapper, obj_reg, person_reg, event_log, cfg)
    sep_mon    = SeparationMonitor(mapper, obj_reg, person_reg, event_log, cfg)
    alarm_mgr  = AlarmManager(mapper, obj_reg, person_reg, event_log, cfg)
    tracker    = PersonTracker(cfg, mapper, person_reg)

    # ── M1: Scene Initialisation ──────────────────────────────────────────
    # Seed first_seen on the same timeline as the main loop (video seconds for
    # files, wall-clock for live) so pre-existing object timestamps are coherent.
    init_t0 = (cfg.get("input", {}).get("init_frames", 30) / fps_src) if not is_live else time.time()
    scene_init.run(cap, now=init_t0)
    fg_det.freeze()   # freeze background model if freeze_after_init=true in config

    # Capture the first active frame as reference for diff-based detection.
    # This is used when reference_diff_enabled=true to detect objects that
    # MOG2 absorbed during init (e.g. a bag present from frame 0 of the video).
    if cfg.get("foreground", {}).get("reference_diff_enabled", False):
        ret_ref, ref_frame = cap.read()
        if ret_ref:
            fg_det.set_reference_frame(ref_frame)
            # Wind back one frame so the main loop processes it too
            cur_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_pos - 1)

    # ── Video writer (optional) ───────────────────────────────────────────
    out_path = cfg.get("visualizer", {}).get("output_video", "")
    writer = None
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_src, (width, height))
        print(f"[PADS] Writing output to: {out_path}")

    show_fg        = cfg.get("visualizer", {}).get("show_foreground_mask", False)
    show_debug     = cfg.get("visualizer", {}).get("show_debug_display", False)
    min_area       = cfg.get("foreground", {}).get("obj_min_area", 800)
    detection_method = cfg.get("foreground", {}).get("method", "MOG2").upper()

    # Graceful headless fallback: a headless OpenCV (Colab/servers) raises on
    # imshow. Probe once and disable the live window instead of crashing.
    if display and not gui_available():
        print("[PADS] No GUI backend available (headless OpenCV) — running without live display. "
              "Output video is still written if configured.")
        display = False

    # Prune persons who left long ago to bound memory on long/live runs.
    person_ttl = cfg.get("person", {}).get("person_ttl_s", 300)
    prune_every = 300  # frames

    print("[PADS] Pipeline running. Press Q to quit.\n")
    frame_count = 0
    frame_time = init_t0   # timeline clock; seeded so end-of-video flush is valid even for empty video
    last_frame = None   # kept for end-of-video alarm snapshots

    # ── Main per-frame loop ───────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[PADS] End of video.")
            break
        last_frame = frame

        frame_count += 1
        frame_time = video_clock(cap, frame_count, fps_src, is_live)

        # M3 — Person detection & tracking (runs first so person boxes are
        #      available for the person-overlap filter in M2 blob matching)
        tracker.process(frame, frame_time)

        # Collect current person pixel bboxes for overlap filtering
        person_boxes = [p.bbox for p in person_reg.in_frame()]

        # Tell M2 where persons are so their footprint never builds heatmap heat
        fg_det.set_person_regions(person_boxes, frame.shape)

        # M2 — Foreground detection
        blobs, fg_mask = fg_det.process(frame)

        # Match blobs to object registry; get new object IDs
        new_obj_ids, blob_centroids = match_blobs_to_registry(
            blobs, obj_reg, mapper, min_area, person_boxes, frame_time
        )

        # M5 — Associate new objects to persons
        if new_obj_ids:
            assoc.process_new_objects(new_obj_ids, frame_time)

        # M5 — Retry pending queue
        assoc.retry_pending(frame_time)

        # M6 — Separation monitoring
        sep_mon.process(frame_time)

        # M7 — Alarm management
        new_alarms = alarm_mgr.tick(frame_time, frame, blob_centroids)
        for alarm in new_alarms:
            print(f"[PADS] ALARM: {alarm['alarm_id']} | Object {alarm['object_id']}")

        # Bound memory on long/live runs: drop persons who exited long ago and
        # are not referenced by any object still being monitored.
        if frame_count % prune_every == 0:
            keep = {o.associated_person_id for o in obj_reg.active()
                    if o.associated_person_id is not None}
            removed = person_reg.prune(frame_time, person_ttl, keep_ids=keep)
            if removed:
                print(f"[PADS] Pruned {removed} stale person record(s).")

        # ── Visualise ─────────────────────────────────────────────────────
        annotated = visualizer.draw(frame, obj_reg, person_reg, alarm_mgr, frame_time)

        if writer:
            writer.write(annotated)

        if display:
            try:
                cv2.imshow("PADS — Abandoned Object Detection", annotated)
                if show_fg:
                    cv2.imshow("PADS — Foreground Mask", fg_mask)
                if show_debug:
                    debug_tile = visualizer.draw_debug_tile(
                        annotated, fg_mask, blobs, min_area, detection_method
                    )
                    cv2.imshow("PADS — M2 Debug", debug_tile)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("[PADS] User quit.")
                    break
            except cv2.error:
                print("[PADS] GUI display failed mid-run — disabling live window.")
                display = False

    # ── End-of-video flush ────────────────────────────────────────────────
    # Mark all in-frame persons as exited and force-tick M7 so that
    # objects of interest that haven't alarmed yet get a final chance.
    from state.person_registry import PersonStatus
    for person in person_reg.in_frame():
        if person.status in (PersonStatus.OF_INTEREST, PersonStatus.ACTIVE):
            person.in_frame = False
            person.exit_time = frame_time
            if person.status == PersonStatus.ACTIVE:
                person.status = PersonStatus.EXITED
            person_reg.update(person)

    # Tick several times to let T_abandon counters that started on the last
    # real frame elapse (simulating time passing after video ends).
    for extra_t in range(1, int(alarm_mgr.T_abandon) + 2):
        flush_time = frame_time + extra_t
        flush_alarms = alarm_mgr.tick(flush_time, last_frame, blob_centroids)
        for alarm in flush_alarms:
            print(f"[PADS] ALARM (end-of-video flush): {alarm['alarm_id']} | Object {alarm['object_id']}")
        if flush_alarms:
            break

    # ── Cleanup ───────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n[PADS] Session complete. {frame_count} frames processed.")
    print(f"[PADS] Event summary: {event_log.summary()}")
    print(f"[PADS] Total alarms: {len(alarm_mgr.all_alarms())}")
    print(f"[PADS] Log: logs/events.jsonl")


if __name__ == "__main__":
    main()
