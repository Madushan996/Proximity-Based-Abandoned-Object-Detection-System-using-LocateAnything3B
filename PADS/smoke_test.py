"""Quick smoke test — verifies all imports and config load."""
import cv2, yaml
print(f"OpenCV: {cv2.__version__}")

import ultralytics
print(f"Ultralytics: {ultralytics.__version__}")

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
print(f"Config loaded OK — source: {cfg['input']['source']}")

from state.object_registry import ObjectRegistry
from state.person_registry import PersonRegistry
from state.event_log import EventLog
print("State modules: OK")

from modules.coordinates import CoordinateMapper
from modules.foreground import ForegroundDetector
from modules.scene_init import SceneInitialiser
from modules.person_tracker import PersonTracker
from modules.association import AssociationEngine
from modules.separation import SeparationMonitor
from modules.alarm import AlarmManager
from utils.visualizer import Visualizer
print("All pipeline modules: OK")

# Quick instantiation test
mapper = CoordinateMapper(cfg)
print(f"CoordinateMapper: scale={mapper.scale} px/m, homography={mapper.enabled}")

fg = ForegroundDetector(cfg)
print(f"ForegroundDetector: method={cfg['foreground']['method']}, min_area={cfg['foreground']['obj_min_area']}")

print("\nAll checks passed. Ready to run main.py")
