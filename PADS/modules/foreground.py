"""
M2 — Foreground Detection & Object Discovery
Uses adaptive background subtraction (MOG2 or KNN) to detect new foreground
objects as generic blobs, without classifying what they are.

Supplementary mode — reference_diff:
  When the subject is in frame from the very first frame (e.g. Sample6), MOG2
  learns the object as background and never detects it. A second pass using
  absolute difference against a saved reference frame catches objects that
  moved position (bag carried → bag placed on floor). Enable via config:
    foreground.reference_diff_enabled: true
    foreground.reference_diff_threshold: 40   # pixel change sensitivity
"""
import cv2
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class DetectedBlob:
    """A raw foreground region detected in a single frame."""
    centroid_px: Tuple[float, float]    # (x, y) in pixel space
    bbox: Tuple[int, int, int, int]     # (x_min, y_min, x_max, y_max)
    area_px: float
    contour: np.ndarray                 # raw contour for optional visualisation
    label: str = ""                     # open-vocab class name (LocateAnything M2)


class ForegroundDetector:
    def __init__(self, cfg: dict):
        fg_cfg = cfg.get("foreground", {})
        method          = fg_cfg.get("method", "MOG2")
        history         = fg_cfg.get("history", 500)
        var_threshold   = fg_cfg.get("var_threshold", 50)
        detect_shadows  = fg_cfg.get("detect_shadows", True)
        self.shadow_thresh = fg_cfg.get("shadow_threshold", 0.5)
        k_size          = fg_cfg.get("morph_kernel_size", 5)
        self.dilate_it  = fg_cfg.get("dilate_iterations", 2)
        self.erode_it   = fg_cfg.get("erode_iterations", 1)
        self.min_area   = fg_cfg.get("obj_min_area", 800)

        # Reference frame diff settings
        self.ref_diff_enabled   = fg_cfg.get("reference_diff_enabled", False)
        self.ref_diff_threshold = fg_cfg.get("reference_diff_threshold", 40)
        self._ref_frame: Optional[np.ndarray] = None

        # ── Persistence (stationary blob) filter ──────────────────────────────
        # Each pixel accumulates "heat" when it is part of a foreground blob.
        # Heat decays when absent. Only regions above the threshold are emitted
        # as confirmed blobs. This kills transient ghost blobs from passing
        # persons/shadows (they disappear before heat builds up) while keeping
        # real stationary objects (constant foreground → heat stays maxed).
        # Person-occupied pixels are excluded from accumulation entirely —
        # when the person moves away, their footprint has zero heat to decay,
        # so no ghost object ever forms.
        self.persist_enabled = fg_cfg.get("persistence_enabled", True)
        self.persist_add     = float(fg_cfg.get("persistence_add",       25))   # heat/frame when detected
        self.persist_decay   = float(fg_cfg.get("persistence_decay",     15))   # heat/frame when absent
        self.persist_thresh  = float(fg_cfg.get("persistence_threshold", 100))  # emit threshold
        self._heatmap: Optional[np.ndarray] = None
        self._person_mask: Optional[np.ndarray] = None   # set each frame by main.py

        # ── Background freeze after scene init ────────────────────────────────
        # When freeze_after_init=true, MOG2 stops adapting after scene_init
        # completes. This prevents stationary objects from being absorbed into
        # the background model over time (the #1 cause of missed detections).
        # WARNING: do NOT use for live 24/7 cameras that see dramatic lighting
        # changes (day/night). Use only for short clips or stable indoor scenes.
        self._freeze_after_init = fg_cfg.get("freeze_after_init", False)
        # When frozen, MOG2 normally stops adapting entirely (rate 0), which makes
        # it brittle to lighting drift on live cameras. A small positive rate lets
        # the background track slow illumination changes while still being slow
        # enough that a stationary object isn't absorbed within T_abandon. Keep 0
        # for short stable clips; set e.g. 0.0005 for live indoor feeds.
        self._frozen_learning_rate = float(fg_cfg.get("frozen_learning_rate", 0.0))
        self._learning_rate = -1  # -1 = MOG2 auto; set on freeze()

        if method == "KNN":
            self.subtractor = cv2.createBackgroundSubtractorKNN(
                history=history, detectShadows=detect_shadows
            )
        else:
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=var_threshold,
                detectShadows=detect_shadows,
            )

        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k_size, k_size)
        )

    def set_person_regions(self, person_boxes: list, frame_shape: tuple) -> None:
        """
        Call once per frame (after person tracking) with the current person bboxes.
        Pixels inside person bboxes are excluded from heatmap accumulation —
        the person's footprint never builds heat, so no ghost object forms
        when they walk away.
        """
        if not person_boxes:
            self._person_mask = None
            return
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for (x1, y1, x2, y2) in person_boxes:
            mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 255
        self._person_mask = mask

    def freeze(self) -> None:
        """
        Freeze the background model (stop adapting).
        Call this once after scene_init completes.
        Only active when freeze_after_init=true in config.
        """
        if self._freeze_after_init:
            self._learning_rate = self._frozen_learning_rate
            if self._frozen_learning_rate <= 0:
                print("[M2] Background model frozen — MOG2 will no longer adapt to scene changes.")
            else:
                print(f"[M2] Background model slow-adapting (rate={self._frozen_learning_rate}) "
                      "— tracks lighting drift without quickly absorbing stationary objects.")

    def set_reference_frame(self, frame: np.ndarray) -> None:
        """
        Store a reference frame for the supplementary diff-based detection.
        Call this with the first active frame (right after scene init).
        The diff will catch objects that have MOVED relative to this frame
        — e.g. a bag carried at waist height that is later placed on the floor.
        """
        self._ref_frame = frame.copy()
        print("[M2] Reference frame captured for diff-based supplementary detection.")

    def _ref_diff_mask(self, frame: np.ndarray) -> np.ndarray:
        """Compute abs-diff mask between current frame and reference frame."""
        diff = cv2.absdiff(frame, self._ref_frame)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(diff_gray, self.ref_diff_threshold, 255, cv2.THRESH_BINARY)
        # Apply same morphology to clean it up
        mask = cv2.erode(mask, self.kernel, iterations=self.erode_it)
        mask = cv2.dilate(mask, self.kernel, iterations=self.dilate_it)
        return mask

    def process(self, frame: np.ndarray) -> Tuple[List[DetectedBlob], np.ndarray]:
        """
        Apply background subtraction + morphology, extract blobs.
        If reference_diff_enabled, merges a second diff-based mask to catch
        objects that MOG2 misses because they were present during background init.
        Returns (list of DetectedBlob, foreground mask for debug display).
        """
        # Use controlled learning rate (0 = frozen after init, -1 = MOG2 auto)
        fg_mask = self.subtractor.apply(frame, learningRate=self._learning_rate)

        # Remove shadows (MOG2 marks them as 127)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological clean-up: erode noise then dilate to fill gaps
        fg_mask = cv2.erode(fg_mask, self.kernel, iterations=self.erode_it)
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=self.dilate_it)

        # Supplementary reference-frame diff (for objects MOG2 absorbs)
        if self.ref_diff_enabled and self._ref_frame is not None:
            ref_mask = self._ref_diff_mask(frame)
            fg_mask = cv2.bitwise_or(fg_mask, ref_mask)

        # ── Persistence heatmap filter ────────────────────────────────────────
        # Accumulate heat where foreground is present; decay where absent.
        # Person-occupied pixels are treated as "not detected" for heat
        # purposes — their areas never build heat, so walking away leaves
        # no ghost trail. Stationary objects accumulate heat every frame.
        if self.persist_enabled:
            if self._heatmap is None:
                self._heatmap = np.zeros(fg_mask.shape[:2], dtype=np.float32)

            # Mask out pixels that belong to tracked persons before accumulating
            fg_for_heat = fg_mask.copy()
            if self._person_mask is not None:
                fg_for_heat[self._person_mask > 0] = 0

            # +persist_add where (foreground AND not person), -persist_decay elsewhere
            delta = np.where(fg_for_heat > 0, self.persist_add, -self.persist_decay)
            self._heatmap += delta
            np.clip(self._heatmap, 0.0, 255.0, out=self._heatmap)
            # Replace mask with persistence-gated version
            fg_mask = (self._heatmap >= self.persist_thresh).astype(np.uint8) * 255

        # Find connected components
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        blobs: List[DetectedBlob] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                cx, cy = x + w / 2, y + h / 2
            else:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]

            blobs.append(DetectedBlob(
                centroid_px=(cx, cy),
                bbox=(x, y, x + w, y + h),
                area_px=area,
                contour=cnt,
            ))

        return blobs, fg_mask

    def update_background_only(self, frame: np.ndarray) -> None:
        """Feed a frame to the background model without returning detections.
        Used during the initialisation averaging phase.
        """
        self.subtractor.apply(frame, learningRate=0.05)


# ── YOLO-based object detector (M2 alternative) ───────────────────────────────

class YOLOObjectDetector:
    """
    M2 alternative: uses YOLO to detect specific COCO object classes
    (backpack, handbag, suitcase, etc.) instead of background subtraction.

    Advantages over MOG2:
      - Zero ghost objects from shadows, lighting changes, or wet pavement
      - No background warm-up required
      - Detects objects by class name, not by pixel difference
      - Works instantly from frame 0 — no init phase needed

    Stability filter: a detection must be IoU-matched across N consecutive
    frames before it is emitted as a confirmed blob. Eliminates one-frame
    YOLO false positives.
    """

    # COCO classes relevant to abandoned-object detection
    CLASS_NAMES = {
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        28: "suitcase",
        63: "laptop",
        67: "phone",
    }

    def __init__(self, cfg: dict):
        from ultralytics import YOLO as _YOLO
        fg_cfg  = cfg.get("foreground", {})
        per_cfg = cfg.get("person", {})
        obj_cfg = fg_cfg.get("yolo_object", {})

        model_path      = obj_cfg.get("model", per_cfg.get("model", "yolov8x.pt"))
        self.classes    = obj_cfg.get("classes", [24, 26, 28])   # backpack, handbag, suitcase
        self.confidence = obj_cfg.get("confidence", 0.30)
        self.device     = per_cfg.get("device", "cpu")
        self.half       = per_cfg.get("half", False)
        self.stability  = obj_cfg.get("stability_frames", 3)     # frames before confirming
        self.iou_match  = obj_cfg.get("iou_match", 0.40)         # IoU threshold for frame match
        self.min_area   = fg_cfg.get("obj_min_area", 800)

        names = [self.CLASS_NAMES.get(c, str(c)) for c in self.classes]
        print(f"[M2-YOLO] Model: {model_path} | device: {self.device}"
              + (" [FP16]" if self.half else ""))
        print(f"[M2-YOLO] Tracking classes: {names} | conf: {self.confidence} | stability: {self.stability}f")

        self.model = _YOLO(model_path)
        # List of {"blob": DetectedBlob, "count": int} candidates accumulating evidence
        self._candidates: List[dict] = []

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        """Intersection-over-Union between two (x1,y1,x2,y2) boxes."""
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    # ── main interface (matches ForegroundDetector.process signature) ─────────

    def process(self, frame: np.ndarray) -> Tuple[List[DetectedBlob], np.ndarray]:
        """
        Run YOLO inference on the frame, apply stability filter, return
        confirmed blobs and a visualisation mask (for the debug tile).
        """
        results = self.model(
            frame,
            classes=self.classes,
            conf=self.confidence,
            device=self.device,
            half=self.half,
            verbose=False,
        )

        # Build raw blobs from this frame's detections
        raw_blobs: List[DetectedBlob] = []
        vis_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        if results and results[0].boxes is not None:
            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                area = (x2 - x1) * (y2 - y1)
                if area < self.min_area:
                    continue
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                contour = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32
                ).reshape(-1, 1, 2)
                raw_blobs.append(DetectedBlob(
                    centroid_px=(cx, cy),
                    bbox=(x1, y1, x2, y2),
                    area_px=float(area),
                    contour=contour,
                ))
                vis_mask[y1:y2, x1:x2] = 255

        # ── Stability filter: match current raw blobs against prior candidates ──
        new_candidates: List[dict] = []
        matched_raw: set = set()

        for cand in self._candidates:
            best_iou, best_j = 0.0, -1
            for j, blob in enumerate(raw_blobs):
                if j in matched_raw:
                    continue
                iou = self._iou(cand["blob"].bbox, blob.bbox)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= self.iou_match and best_j >= 0:
                matched_raw.add(best_j)
                new_candidates.append({
                    "blob": raw_blobs[best_j],
                    "count": cand["count"] + 1,
                })
            # else: candidate disappears this frame — drop it (reset streak)

        # Any raw blob not matched to an existing candidate starts a new one
        for j, blob in enumerate(raw_blobs):
            if j not in matched_raw:
                new_candidates.append({"blob": blob, "count": 1})

        self._candidates = new_candidates

        # Only emit blobs that have been stable for at least `stability` frames
        stable_blobs = [c["blob"] for c in self._candidates if c["count"] >= self.stability]
        return stable_blobs, vis_mask

    def set_person_regions(self, person_boxes: list, frame_shape: tuple) -> None:
        """No-op — YOLO detects by class, not pixel heat."""
        pass

    def freeze(self) -> None:
        """No-op — YOLO has no background model to freeze."""
        pass

    def update_background_only(self, frame: np.ndarray) -> None:
        """No-op — YOLO needs no background warm-up."""
        pass

    def set_reference_frame(self, frame: np.ndarray) -> None:
        """No-op — YOLO doesn't use reference-frame differencing."""
        pass


# ── LocateAnything VLM object detector (M2 alternative) ───────────────────────

class LocateAnythingObjectDetector:
    """
    M2 alternative powered by NVIDIA LocateAnything-3B, an open-vocabulary
    vision-language localiser. Instead of fixed COCO IDs you describe what an
    abandoned item looks like in natural language and the model returns boxes.

    Two execution modes (set by what you pass / configure):
      • in-process worker — `worker` is a LocateAnythingWorker loaded on the
        same GPU (the Modal-container deployment). Fastest, no network.
      • remote HTTP        — `foreground.locateanything.endpoint_url` points at
        a `/detect` endpoint (run PADS locally, model on Modal). The frame is
        POSTed as a JPEG.

    The VLM is far too slow to run every frame (~tens of ms–hundreds of ms per
    call). Abandoned objects are stationary by definition, so we only *infer*
    every `infer_every_frames` frames and re-emit the confirmed boxes on the
    frames in between. A stability filter (matched across N inference calls)
    suppresses one-shot false positives — mirroring YOLOObjectDetector.

    Implements the same interface as ForegroundDetector/YOLOObjectDetector:
        process(frame) -> (List[DetectedBlob], mask)
    plus the duck-typed no-ops scene_init/main.py expect.
    """

    def __init__(self, cfg: dict, worker=None):
        fg_cfg = cfg.get("foreground", {})
        la_cfg = fg_cfg.get("locateanything", {})

        self.categories = la_cfg.get(
            "categories",
            ["backpack", "handbag", "suitcase", "luggage", "bag", "box", "trolley"],
        )
        self.infer_every = max(1, int(la_cfg.get("infer_every_frames", 24)))
        self.stability   = int(la_cfg.get("stability_frames", 2))   # in inference calls
        self.iou_match   = float(la_cfg.get("iou_match", 0.40))
        self.min_area    = fg_cfg.get("obj_min_area", 800)
        self.resize_long = int(la_cfg.get("resize_long_side", 1024))  # 0 = full res
        self.gen_mode    = la_cfg.get("generation_mode", "hybrid")
        self.endpoint    = la_cfg.get("endpoint_url", "")

        self.worker = worker
        if self.worker is None and not self.endpoint:
            print("[M2-LA] WARNING: no in-process worker and no endpoint_url — "
                  "detector will return nothing. Inject a worker or set "
                  "foreground.locateanything.endpoint_url.")

        names = ", ".join(self.categories)
        mode = "in-process" if self.worker is not None else (
            f"remote:{self.endpoint}" if self.endpoint else "DISABLED")
        print(f"[M2-LA] LocateAnything-3B | mode: {mode}")
        print(f"[M2-LA] Categories: [{names}] | infer every {self.infer_every}f | "
              f"stability: {self.stability} calls")

        self._frame_idx = 0
        self._infer_count = 0
        self._candidates: List[dict] = []         # {"blob", "count"} across inference calls
        self._confirmed: List[DetectedBlob] = []  # last emitted stable blobs
        self._last_mask: Optional[np.ndarray] = None

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _iou(a, b) -> float:
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _infer_labeled(self, frame: np.ndarray) -> List[dict]:
        """Per-category detection → deduped [{"bbox","label"}] in full-frame space.

        Querying one class at a time (vs one comma-joined call) gives the model
        a clean prompt: better recall and a reliable per-box label. Overlapping
        detections across categories are then merged with NMS so one physical
        object isn't registered twice (e.g. "bag" and "luggage").
        """
        h, w = frame.shape[:2]

        # Optionally downscale the long side for speed; scale boxes back up.
        scale = 1.0
        proc = frame
        if self.resize_long and max(h, w) > self.resize_long:
            scale = self.resize_long / float(max(h, w))
            proc = cv2.resize(frame, (int(w * scale), int(h * scale)))
        ph, pw = proc.shape[:2]

        if self.worker is not None:
            from PIL import Image
            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            raw = self.worker.detect_labeled(
                Image.fromarray(rgb), self.categories, generation_mode=self.gen_mode
            )
        else:
            raw = self._infer_remote_labeled(proc, pw, ph)

        out: List[dict] = []
        for b in raw:
            x1 = int(b["x1"] / scale); y1 = int(b["y1"] / scale)
            x2 = int(b["x2"] / scale); y2 = int(b["y2"] / scale)
            x1, x2 = sorted((max(0, x1), min(w, x2)))
            y1, y2 = sorted((max(0, y1), min(h, y2)))
            if (x2 - x1) * (y2 - y1) >= self.min_area:
                out.append({"bbox": (x1, y1, x2, y2), "label": b.get("label", "object")})
        return self._dedup_labeled(out)

    def _dedup_labeled(self, dets: List[dict]) -> List[dict]:
        """Greedy NMS across categories; keep the larger box of an overlapping pair."""
        kept: List[dict] = []
        for d in sorted(dets, key=lambda x: (x["bbox"][2] - x["bbox"][0]) *
                                            (x["bbox"][3] - x["bbox"][1]), reverse=True):
            if all(self._iou(d["bbox"], k["bbox"]) < self.iou_match for k in kept):
                kept.append(d)
        return kept

    def _infer_remote_labeled(self, proc: np.ndarray, pw: int, ph: int) -> List[dict]:
        """Per-category POST to a remote /detect endpoint. Lazy-imports requests."""
        import requests
        ok, buf = cv2.imencode(".jpg", proc, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return []
        jpeg = buf.tobytes()
        results: List[dict] = []
        for cat in self.categories:
            try:
                resp = requests.post(
                    self.endpoint,
                    files={"image": ("frame.jpg", jpeg, "image/jpeg")},
                    data={"categories": cat, "generation_mode": self.gen_mode},
                    timeout=120,
                )
                resp.raise_for_status()
                for box in resp.json().get("boxes", []):
                    box["label"] = cat
                    results.append(box)
            except Exception as e:  # network/endpoint failure must not crash the loop
                print(f"[M2-LA] remote detect failed for '{cat}': {e}")
        return results

    @staticmethod
    def _blob_from_box(box: Tuple[int, int, int, int], label: str = "") -> DetectedBlob:
        x1, y1, x2, y2 = box
        contour = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32
        ).reshape(-1, 1, 2)
        return DetectedBlob(
            centroid_px=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            bbox=(x1, y1, x2, y2),
            area_px=float((x2 - x1) * (y2 - y1)),
            contour=contour,
            label=label,
        )

    # ── main interface ─────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> Tuple[List[DetectedBlob], np.ndarray]:
        self._frame_idx += 1
        run_now = (self._frame_idx % self.infer_every == 1) or (self.infer_every == 1)

        if not run_now:
            # Stationary-object assumption: re-emit the last confirmed blobs.
            mask = self._last_mask if self._last_mask is not None else \
                np.zeros(frame.shape[:2], dtype=np.uint8)
            return list(self._confirmed), mask

        self._infer_count += 1
        print(f"[M2-LA] inference #{self._infer_count} @frame {self._frame_idx} … ",
              end="", flush=True)
        _t0 = time.perf_counter()
        dets = self._infer_labeled(frame)
        _dt = time.perf_counter() - _t0
        raw_blobs = [self._blob_from_box(d["bbox"], d["label"]) for d in dets]

        # Stability filter across inference calls (same scheme as YOLO detector).
        new_candidates: List[dict] = []
        matched_raw: set = set()
        for cand in self._candidates:
            best_iou, best_j = 0.0, -1
            for j, blob in enumerate(raw_blobs):
                if j in matched_raw:
                    continue
                iou = self._iou(cand["blob"].bbox, blob.bbox)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= self.iou_match and best_j >= 0:
                matched_raw.add(best_j)
                new_candidates.append({"blob": raw_blobs[best_j], "count": cand["count"] + 1})
        for j, blob in enumerate(raw_blobs):
            if j not in matched_raw:
                new_candidates.append({"blob": blob, "count": 1})
        self._candidates = new_candidates

        self._confirmed = [c["blob"] for c in self._candidates if c["count"] >= self.stability]
        labels = ",".join(sorted({b.label for b in self._confirmed if b.label})) or "-"
        print(f"{len(dets)} raw, {len(self._confirmed)} confirmed [{labels}] ({_dt:.1f}s)",
              flush=True)

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for blob in self._confirmed:
            x1, y1, x2, y2 = blob.bbox
            mask[y1:y2, x1:x2] = 255
        self._last_mask = mask

        return list(self._confirmed), mask

    # ── duck-typed no-ops (scene_init / main.py expect these) ────────────────────

    def set_person_regions(self, person_boxes: list, frame_shape: tuple) -> None:
        pass

    def freeze(self) -> None:
        pass

    def update_background_only(self, frame: np.ndarray) -> None:
        # No background model. During M1 init we want pre-existing objects
        # detected, which only happens via process(); deliberately a no-op.
        pass

    def set_reference_frame(self, frame: np.ndarray) -> None:
        pass


# ── Factory ───────────────────────────────────────────────────────────────────

def create_foreground_detector(cfg: dict, worker=None):
    """
    Factory function — returns the correct M2 detector based on config.

    config.yaml:
      foreground:
        method: "MOG2"           # default — adaptive background subtraction
        # method: "KNN"          # KNN variant
        # method: "YOLO"         # YOLO object detector — no ghost objects
        # method: "LOCATEANYTHING"  # NVIDIA LocateAnything-3B open-vocab VLM

    `worker` is an optional in-process LocateAnythingWorker injected by the
    GPU host (Modal). It is ignored by the MOG2/KNN/YOLO detectors.
    """
    method = cfg.get("foreground", {}).get("method", "MOG2").upper()
    if method == "YOLO":
        return YOLOObjectDetector(cfg)
    if method in ("LOCATEANYTHING", "LOCATE_ANYTHING", "LA"):
        return LocateAnythingObjectDetector(cfg, worker=worker)
    return ForegroundDetector(cfg)
