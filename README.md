# Proximity-Based Abandoned Object Detection System (PADS × LocateAnything-3B)

Detect **unattended / abandoned objects** in video (bags, suitcases, boxes,
trolleys…) and raise an alarm when an item is left behind and its owner walks
away. Object detection is powered by NVIDIA's open-vocabulary vision-language
model **[LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B)**,
running on a **Modal** GPU container, with a browser UI that uploads a clip and
plays back the annotated result with a live alarm log.

> **Status:** research / demo project. See [Licensing](#-licensing--attribution)
> — LocateAnything-3B is **non-commercial** and Ultralytics YOLOv8 is **AGPL-3.0**.

---

## ✨ What it does

Given a video, the system continuously asks: *"Is there an object sitting here
that nobody is coming back for?"* To answer that it:

1. **Finds objects** you describe in plain language ("backpack", "suitcase", "box").
2. **Finds and tracks people**, giving each a stable ID across frames.
3. **Associates** each new object with the nearest person (its presumed owner).
4. **Watches the distance** between owner and object.
5. **Starts a countdown** when the owner leaves — and fires an **alarm** if the
   timer expires before anyone returns.

It runs in two modes, selectable in the UI:

| Mode | What it does |
|------|--------------|
| **Abandonment detection (PADS)** | Full owner→object→alarm pipeline. |
| **Object labeling (overlay)** | Just detects & labels objects every frame — a simple open-vocabulary detector demo. |

---

## 🧠 How it works (the pipeline)

The video is processed frame-by-frame through stages PADS calls **M1–M7**:

| Stage | Job | Tech |
|-------|-----|------|
| **M1** Scene init | Snapshot what's already in the scene; anything present from the start is *pre-existing* and ignored. | — |
| **M2** Object detection | Detect items matching your natural-language categories. | **LocateAnything-3B** (VLM) |
| **M3** Person detection + tracking | Detect people and keep a stable ID per person across frames. | **YOLOv8n + ByteTrack** |
| **M5** Association | Link a new object to the nearest person → owner. | Proximity radius |
| **M6** Separation monitoring | Track owner↔object distance; flag growing separation as *of interest*. | — |
| **M7** Alarm | Start an abandon timer; fire alarm if it expires (cancel if owner returns). | — |

```
   video frame
       │
       ├─► M2  LocateAnything-3B ──► "there's a backpack here"  (box + label)
       │
       ├─► M3  YOLOv8 + ByteTrack ─► "there's person #3 here"   (box + stable ID)
       │
       ▼
   M5  close together? ──► link: backpack#5 owned by person#3
       │
       ▼
   M6  owner walking away? ──► distance growing... "of interest"
       │
       ▼
   M7  owner gone + timer expired? ──► 🚨 ALARM
       │
       ▼
   draw boxes/labels → stitch annotated frames into a result MP4
```

### Why two different models?

- **People** are a fixed category needed *fast on every frame* → small, quick
  **YOLOv8n**, run locally on the GPU.
- **Objects** can be *anything you describe in words* and they sit still →
  **LocateAnything-3B**, a flexible but slower language-grounded localiser.

LocateAnything is too slow to run every frame (~0.5 s per query). Since
abandoned objects are stationary, **M2 only runs inference every N frames** and
re-uses the last boxes in between. A **stability filter** requires an object to
be detected in the same spot across consecutive inference calls before it
counts — this suppresses one-shot false positives ("phantom" boxes).

### Architecture

```
 Browser UI ──HTTP upload + WebSocket──▶  Modal GPU container (A10G, Ampere)
                                          ├─ LocateAnything-3B  ← M2 detection
                                          ├─ YOLOv8n + ByteTrack ← M3 persons
                                          ├─ PADS M1/M5/M6/M7    ← abandon logic
                                          └─ ffmpeg → H.264 MP4  ← annotated result
```

The browser uploads a clip, the GPU container processes it and streams
lightweight progress + alarm events over a WebSocket, then renders the annotated
video to MP4 which the browser plays back natively (smooth, scrubbable).

---

## 📁 Repository layout

```
.
├── LocateAnything/              # The Modal app + web UI + glue
│   ├── modal_app.py             # Modal app: GPU image, model load, FastAPI, UI, WebSocket, /detect
│   ├── locateanything_worker.py # Loads LocateAnything-3B; detect() + box/label parsing
│   ├── pipeline.py              # PADS per-frame loop as a streaming generator (yields frames + events)
│   ├── config_locateanything.yaml  # PADS config using method: LOCATEANYTHING
│   ├── requirements.txt         # Local deploy deps (just the Modal client)
│   ├── serve.py                 # Optional: serve the UI locally against your deployed endpoint
│   └── ui/index.html            # Single-file web UI (upload, player, alarm log)
│
└── PADS/                        # The abandonment-detection engine (imported by the Modal app)
    ├── main.py                  # Core per-frame loop helpers
    ├── modules/                 # M1/M2/M3/M5/M6/M7: scene_init, foreground, person_tracker,
    │                            #   association, separation, alarm, coordinates
    ├── state/                   # object/person registries, event log
    ├── utils/visualizer.py      # Draws boxes, labels, distances, countdowns
    ├── SampleVideos/            # Test clips
    └── yolov8n.pt               # YOLOv8 nano weights (person detector)
```

> ⚠️ **Keep `LocateAnything/` and `PADS/` as siblings.** `modal_app.py` resolves
> the PADS package at `../PADS` and mounts it into the container. Don't move them
> into one another.

---

## 🚀 Setup & run

### Prerequisites

- **Python 3.10+** on your machine (only the lightweight Modal *client* runs
  locally — all heavy ML deps install inside the Modal image automatically).
- A free **[Modal](https://modal.com)** account (provides the GPU).
- **git**.
- A LocateAnything-3B-capable GPU is handled *for you* by Modal (defaults to an
  **A10G**, the minimum Ampere GPU the model supports). You do **not** need a
  local GPU.

### 1. Clone

```bash
git clone https://github.com/Madushan996/Proximity-Based-Abandoned-Object-Detection-System-using-LocateAnything3B.git
cd Proximity-Based-Abandoned-Object-Detection-System-using-LocateAnything3B/LocateAnything
```

### 2. Install the Modal client & authenticate

```bash
pip install -r requirements.txt
modal setup        # one-time: opens a browser to link your Modal account
```

### 3. Deploy to Modal

```bash
modal deploy modal_app.py
```

This builds the GPU image and prints a public URL, e.g.:

```
✓ Created web function => https://<your-username>--pads-locateanything-padsengine-web.modal.run
```

> The **first** run downloads the LocateAnything-3B weights (~6 GB) into a
> persistent Modal Volume — one time only. Subsequent runs are fast, and the
> container stays warm for a few minutes between requests.

### 4. Use it — two options

**Option A (simplest): open the deployed URL.** The Modal app serves the web UI
itself. Just open the `…modal.run` URL printed above in your browser. Upload a
clip, pick a mode, press **Run detection**. No further setup.

**Option B: run the UI locally** (useful for tweaking the UI). Edit
[`LocateAnything/ui/index.html`](LocateAnything/ui/index.html) and set
`MODAL_URL` to your deployed URL, then:

```bash
python serve.py            # serves the UI at http://localhost:8000 and opens it
```

The page runs locally but submits jobs to your Modal GPU endpoint.

### Iterating on server code

While editing `modal_app.py` / `pipeline.py` / the PADS modules, use hot-reload:

```bash
modal serve modal_app.py   # live-reloads on save; runs only while the terminal is open
```

Smoke-test the model alone (loads it, runs `detect()` on a blank frame):

```bash
modal run modal_app.py
```

---

## 🎛️ Usage & tuning

In the UI:

- **Mode** — Abandonment detection (PADS) or Object labeling overlay.
- **Object categories** — comma-separated natural language. Open-vocabulary, so
  you're not limited to the defaults — try `umbrella, laptop, cardboard box,
  shopping cart`. *Each category is one extra VLM query per cycle*, so keep the
  list focused for speed.
- **Abandon timeout (s)** — how long an item must be unattended before alarming.
- **Detect every N frames** — how often the VLM runs (higher = faster, less
  responsive to moving objects).

Deeper knobs (proximity radius, separation threshold, stability frames, min
object area, GPU type) live in
[`LocateAnything/config_locateanything.yaml`](LocateAnything/config_locateanything.yaml)
and `modal_app.py`'s `_base_config()`.

---

## ⚠️ Known limitations

- **Distance is uncalibrated.** Distances are reported in metres but derived from
  a fixed pixels-per-metre assumption, so on wide/fisheye cameras the numbers
  (e.g. "8 m") are not accurate. The abandonment *logic* still works; proper
  calibration (homography or a per-scene scale) is future work.
- **Tracking is per-video and positional**, not identity. ByteTrack can swap IDs
  when people cross or are occluded for a while, which can break an owner↔object
  link. This is not facial recognition — no identities are stored.
- **False positives** are possible from the VLM; the stability filter mitigates
  but doesn't eliminate them. Narrow your category list and raise
  `stability_frames` if you see phantom boxes.

---

## 📜 Licensing & attribution

This repository's **own code** is released under the **MIT License** (see
[`LICENSE`](LICENSE)). However, it depends on third-party components with their
own terms — **you are responsible for complying with all of them**:

| Component | License | Implication |
|-----------|---------|-------------|
| [NVIDIA LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) | NVIDIA non-commercial research license | **Research / personal / demo only. Not for commercial use.** |
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | AGPL-3.0 | Strong copyleft; a commercial license is required for closed-source commercial use. |
| Modal, PyTorch, Transformers, OpenCV, FastAPI, ffmpeg | Respective OSS licenses | — |

**Bottom line:** great for learning, research, and demos — **not** for shipping
a commercial product as-is.

---

## 🙏 Acknowledgements

- **NVIDIA** for LocateAnything-3B (open-vocabulary localisation VLM:
  MoonViT vision encoder + Qwen2.5-3B decoder).
- **Ultralytics** for YOLOv8 + ByteTrack.
- **Modal** for serverless GPU hosting.
