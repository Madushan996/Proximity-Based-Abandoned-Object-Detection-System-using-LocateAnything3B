"""
Visualizer — PADS
Draws all tracking overlays onto the video frame for debugging and demonstration.
Colour coding:
  Green box      = person (active)
  Yellow box     = person (of_interest)
  Cyan box       = associated object
  Orange box     = object of_interest / pending
  Red box        = ALARMED object
  Blue line      = association link (person ↔ object)
  Red circle     = separation radius indicator
"""
import cv2
import numpy as np
import time
from state.object_registry import ObjectRegistry, ObjectStatus
from state.person_registry import PersonRegistry, PersonStatus


# Colour palette (BGR)
C_PERSON_ACTIVE   = (0, 220, 0)
C_PERSON_OFI      = (0, 200, 255)
C_OBJ_NEW         = (200, 200, 200)
C_OBJ_ASSOC       = (255, 220, 0)
C_OBJ_OFI         = (0, 140, 255)
C_OBJ_ALARMED     = (0, 0, 255)
C_OBJ_PRE         = (100, 100, 100)
C_LINK            = (255, 180, 0)
C_COUNTDOWN       = (0, 0, 255)
C_TEXT_BG         = (20, 20, 20)


def _bbox_centre(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def _draw_label(img, text, x, y, colour, scale=0.5, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), C_TEXT_BG, -1)
    cv2.putText(img, text, (x + 2, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


class Visualizer:
    def __init__(self, cfg: dict):
        vis_cfg = cfg.get("visualizer", {})
        self.enabled          = vis_cfg.get("enabled", True)
        self.show_persons     = vis_cfg.get("show_person_boxes", True)
        self.show_objects     = vis_cfg.get("show_object_boxes", True)
        self.show_assoc       = vis_cfg.get("show_associations", True)
        self.show_dist        = vis_cfg.get("show_distances", True)
        self.show_status      = vis_cfg.get("show_status_overlay", True)
        self.show_fps         = vis_cfg.get("fps_display", True)
        self._fps_history = []

    def draw(
        self,
        frame: np.ndarray,
        obj_reg: ObjectRegistry,
        person_reg: PersonRegistry,
        alarm_manager,
        frame_time: float,
    ) -> np.ndarray:
        if not self.enabled:
            return frame

        canvas = frame.copy()

        # FPS
        if self.show_fps:
            self._fps_history.append(frame_time)
            self._fps_history = [t for t in self._fps_history if frame_time - t < 1.0]
            fps = len(self._fps_history)
            _draw_label(canvas, f"FPS: {fps}", 10, 30, (0, 255, 0), scale=0.7, thickness=2)

        # --- Draw association lines first (behind boxes) ---
        if self.show_assoc:
            for obj in obj_reg.active():
                if obj.associated_person_id is None:
                    continue
                person = person_reg.get(obj.associated_person_id)
                if person is None or not person.in_frame:
                    continue
                obj_cx, obj_cy = _bbox_centre(obj.bbox)
                per_cx, per_cy = _bbox_centre(person.bbox)
                cv2.line(canvas, (obj_cx, obj_cy), (per_cx, per_cy), C_LINK, 1, cv2.LINE_AA)

        # --- Draw persons ---
        if self.show_persons:
            for person in person_reg.in_frame():
                x1, y1, x2, y2 = person.bbox
                colour = C_PERSON_OFI if person.status == PersonStatus.OF_INTEREST else C_PERSON_ACTIVE
                cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
                label = f"P{person.person_id} [{person.status}]"
                _draw_label(canvas, label, x1, y1 - 5, colour)

        # --- Draw objects ---
        if self.show_objects:
            for obj in obj_reg.all():
                if obj.pre_existing:
                    continue
                x1, y1, x2, y2 = obj.bbox
                if obj.status == ObjectStatus.ALARMED:
                    colour = C_OBJ_ALARMED
                elif obj.status == ObjectStatus.OF_INTEREST:
                    colour = C_OBJ_OFI
                elif obj.status == ObjectStatus.ASSOCIATED:
                    colour = C_OBJ_ASSOC
                else:
                    colour = C_OBJ_NEW

                thickness = 3 if obj.status in (ObjectStatus.OF_INTEREST, ObjectStatus.ALARMED) else 2
                cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)

                # Distance label
                if self.show_dist and obj.distance_history:
                    latest_d = obj.distance_history[-1][1]
                    dist_str = f"{latest_d:.1f}m"
                else:
                    dist_str = ""

                # Class name (open-vocab label from LocateAnything), if present
                cls_str = f" {obj.label}" if getattr(obj, "label", "") else ""

                # Countdown timer
                countdown = alarm_manager.get_alarm_countdown(obj.object_id, frame_time)
                if countdown >= 0:
                    timer_str = f" T-{countdown:.0f}s"
                    label = f"O{obj.object_id}{cls_str} {dist_str}{timer_str}"
                    cv2.rectangle(canvas, (x1, y2 + 2), (x2, y2 + 22), (0, 0, 180), -1)
                    cv2.putText(canvas, f"ALARM IN {countdown:.0f}s",
                                (x1 + 2, y2 + 17),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    label = f"O{obj.object_id}{cls_str} [{obj.status}] {dist_str}"

                _draw_label(canvas, label, x1, y1 - 5, colour)

                # Flash red border on ALARMED
                if obj.status == ObjectStatus.ALARMED:
                    flash = int(frame_time * 4) % 2 == 0
                    if flash:
                        cv2.rectangle(canvas, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4),
                                      (0, 0, 255), 4)

        # --- Status overlay ---
        if self.show_status:
            active_persons = len(person_reg.in_frame())
            active_objs = len([o for o in obj_reg.active()
                               if o.status == ObjectStatus.ASSOCIATED])
            alarms = len(alarm_manager.all_alarms())
            of_interest = len(obj_reg.of_interest())

            lines = [
                f"Persons in frame: {active_persons}",
                f"Associated objects: {active_objs}",
                f"Objects of interest: {of_interest}",
                f"Alarms dispatched: {alarms}",
            ]
            for i, line in enumerate(lines):
                _draw_label(canvas, line, 10, canvas.shape[0] - 20 - i * 22,
                            (255, 255, 255), scale=0.55)

        return canvas

    # ── Debug tile ─────────────────────────────────────────────────────────────

    def draw_debug_tile(
        self,
        annotated: np.ndarray,
        fg_mask: np.ndarray,
        raw_blobs: list,
        min_area: float,
        detection_method: str = "MOG2",
    ) -> np.ndarray:
        """
        Build a side-by-side debug panel:
          LEFT  — the normal PADS annotated output (scaled to fit)
          RIGHT — raw M2 detection: foreground mask overlaid on the frame,
                  with every raw blob drawn (green = passes area filter,
                  dark cyan = filtered out by min_area).

        Returns the combined numpy array (height × 2*panel_w × 3).
        Can be shown in a separate cv2 window or saved alongside the main output.
        """
        h, w = annotated.shape[:2]

        # Scale both panels to at most 640 px wide so the combined window
        # fits on a 1280-wide screen.
        panel_w = min(w, 640)
        panel_h = int(h * panel_w / w)
        scale_x = panel_w / w
        scale_y = panel_h / h

        # ── LEFT: PADS annotated output (resized) ────────────────────────────
        left = cv2.resize(annotated, (panel_w, panel_h))

        # ── RIGHT: raw detection panel ────────────────────────────────────────
        # Base = resized raw frame (dimmed slightly so the overlay pops)
        frame_small = cv2.resize(annotated, (panel_w, panel_h))
        dark_base   = cv2.convertScaleAbs(frame_small, alpha=0.5, beta=0)

        # Colorise mask: foreground pixels → translucent green overlay
        if len(fg_mask.shape) == 2:
            mask_color = np.zeros((fg_mask.shape[0], fg_mask.shape[1], 3), dtype=np.uint8)
            mask_color[fg_mask > 0] = (0, 200, 80)          # green
        else:
            mask_color = fg_mask.copy()

        mask_small = cv2.resize(mask_color, (panel_w, panel_h))
        right = cv2.addWeighted(dark_base, 0.5, mask_small, 0.5, 0)

        # Draw every raw blob (before area filter so the user sees both)
        for blob in raw_blobs:
            x1, y1, x2, y2 = blob.bbox
            x1s = int(x1 * scale_x);  y1s = int(y1 * scale_y)
            x2s = int(x2 * scale_x);  y2s = int(y2 * scale_y)
            passes = blob.area_px >= min_area
            col    = (0, 255, 80) if passes else (0, 100, 140)
            cv2.rectangle(right, (x1s, y1s), (x2s, y2s), col, 2)
            label  = f"{int(blob.area_px)}px {'OK' if passes else 'SMALL'}"
            cv2.putText(right, label, (x1s + 2, y1s + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

        # ── Headers ───────────────────────────────────────────────────────────
        hdr_bg = (20, 20, 20)
        cv2.rectangle(left,  (0, 0), (panel_w, 26), hdr_bg, -1)
        cv2.rectangle(right, (0, 0), (panel_w, 26), hdr_bg, -1)
        cv2.putText(left, "PADS Output",
                    (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 80), 2, cv2.LINE_AA)
        cv2.putText(right, f"M2 Raw Detection  [{detection_method}]",
                    (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)

        # Blob count (top-right of right panel)
        cnt_str = f"Blobs: {len(raw_blobs)}"
        cv2.putText(right, cnt_str,
                    (panel_w - 90, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

        # ── Legend (bottom-right panel) ───────────────────────────────────────
        legend = [
            ((0, 255, 80),  "passes area filter"),
            ((0, 100, 140), "filtered out (too small)"),
        ]
        for i, (col, text) in enumerate(legend):
            yp = panel_h - 10 - i * 18
            cv2.rectangle(right, (8, yp - 10), (20, yp + 2), col, -1)
            cv2.putText(right, text, (26, yp),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)

        return np.hstack([left, right])
