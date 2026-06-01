"""
M4 — Coordinate System & Perspective Correction
Maps pixel coordinates → corrected 2D space (metres).

Two modes:
  1. homography_enabled=False  →  flat pixel scale (pixels / pixel_scale_px_per_m = metres)
     Simple, no calibration needed. Good for development.
  2. homography_enabled=True   →  full cv2.findHomography transform.
     Requires calibration/homography.json produced by calibration.py.
"""
import json
import numpy as np
import cv2
from pathlib import Path


class CoordinateMapper:
    def __init__(self, cfg: dict):
        coord_cfg = cfg.get("coordinates", {})
        self.enabled = coord_cfg.get("homography_enabled", False)
        self.scale = coord_cfg.get("pixel_scale_px_per_m", 150)  # px per metre
        self.H = None  # homography matrix

        if self.enabled:
            hfile = coord_cfg.get("homography_points_file", "calibration/homography.json")
            self._load_homography(hfile)

    def _load_homography(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Homography file not found: {path}\n"
                "Run calibration.py first, or set homography_enabled: false in config.yaml"
            )
        data = json.loads(p.read_text())
        src = np.array(data["pixel_points"], dtype=np.float32)
        dst = np.array(data["world_points"], dtype=np.float32)
        self.H, _ = cv2.findHomography(src, dst)
        print(f"[M4] Homography loaded from {path}")

    def to_world(self, px: float, py: float) -> tuple[float, float]:
        """Convert a pixel coordinate to world-space (metres)."""
        if self.enabled and self.H is not None:
            pt = np.array([[[px, py]]], dtype=np.float32)
            out = cv2.perspectiveTransform(pt, self.H)
            return float(out[0][0][0]), float(out[0][0][1])
        else:
            # Flat scale: divide by pixels-per-metre
            return px / self.scale, py / self.scale

    def distance_m(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Euclidean distance in metres between two world coordinates."""
        return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

    def pixel_to_radius_px(self, radius_m: float) -> float:
        """Convert a metre radius to approximate pixel radius (for display only)."""
        if self.enabled:
            return radius_m * self.scale  # approximation for display
        return radius_m * self.scale
