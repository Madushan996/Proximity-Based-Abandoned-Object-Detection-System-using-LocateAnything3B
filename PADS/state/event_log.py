"""
Event Log — PADS M7
Append-only structured log of all system events, written as JSON lines.
"""
import json
import time
from pathlib import Path
from typing import Optional


class EventType:
    OBJECT_APPEARED       = "OBJECT_APPEARED"
    ASSOCIATION_LOCKED    = "ASSOCIATION_LOCKED"
    SEPARATION_DETECTED   = "SEPARATION_DETECTED"
    PERSON_EXITED         = "PERSON_EXITED"
    OBJECT_RETRIEVED      = "OBJECT_RETRIEVED"
    ALARM_DISPATCHED      = "ALARM_DISPATCHED"
    ALARM_CANCELLED       = "ALARM_CANCELLED"


class EventLog:
    def __init__(self, log_path: str = "logs/events.jsonl"):
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._events = []
        self._next_id = 1

    def log(
        self,
        event_type: str,
        person_id: Optional[int] = None,
        object_id: Optional[int] = None,
        location: Optional[tuple] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        event = {
            "event_id": self._next_id,
            "event_type": event_type,
            "timestamp": time.time(),
            "person_id": person_id,
            "object_id": object_id,
            "location": location,
            "metadata": metadata or {},
        }
        self._next_id += 1
        self._events.append(event)
        # Write immediately (append mode)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        return event

    def all(self) -> list:
        return list(self._events)

    def summary(self) -> str:
        counts = {}
        for e in self._events:
            counts[e["event_type"]] = counts.get(e["event_type"], 0) + 1
        return " | ".join(f"{k}: {v}" for k, v in counts.items())
