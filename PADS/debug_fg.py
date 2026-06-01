"""
Debug: saves foreground mask + annotated frames at key moments
to understand what MOG2 is detecting in Sample6.
"""
import cv2
import yaml
import numpy as np
import os
from modules.foreground import ForegroundDetector
from modules.coordinates import CoordinateMapper

with open("config_sample6.yaml") as f:
    cfg = yaml.safe_load(f)

os.makedirs("debug_frames", exist_ok=True)

cap = cv2.VideoCapture("SampleVideos/Sample6.mp4")
fg = ForegroundDetector(cfg)
mapper = CoordinateMapper(cfg)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
min_area = cfg["foreground"]["obj_min_area"]

# Warm up background model
init_frames = cfg["input"]["init_frames"]
print(f"Running {init_frames} init frames...")
for i in range(init_frames):
    ret, frame = cap.read()
    if not ret: break
    fg.update_background_only(frame)

# Set reference frame (first active frame)
if cfg.get("foreground", {}).get("reference_diff_enabled", False):
    ret, ref = cap.read()
    if ret:
        fg.set_reference_frame(ref)
        cap.set(cv2.CAP_PROP_POS_FRAMES, init_frames)  # rewind

# Save annotated frames at regular intervals through the active video
save_every = 24  # save one frame per second
frame_idx = init_frames
saved = 0

while saved < 12:
    ret, frame = cap.read()
    if not ret: break
    frame_idx += 1

    blobs, mask = fg.process(frame)

    # Draw blobs on frame
    annotated = frame.copy()
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for blob in blobs:
        px, py = blob.centroid_px
        x1, y1, x2, y2 = blob.bbox
        area = blob.area_px
        colour = (0, 255, 0) if area >= min_area else (0, 100, 100)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(annotated, f"{int(area)}px", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

    if (frame_idx - init_frames) % save_every == 0:
        t = (frame_idx) / 24.0
        cv2.imwrite(f"debug_frames/frame_{frame_idx:04d}_t{t:.1f}s_annotated.jpg", annotated)
        cv2.imwrite(f"debug_frames/frame_{frame_idx:04d}_t{t:.1f}s_mask.jpg", mask_color)
        print(f"  t={t:.1f}s | {len(blobs)} total blobs | "
              f"{sum(1 for b in blobs if b.area_px >= min_area)} pass filter")
        saved += 1

cap.release()
print("Done. Check debug_frames/")
