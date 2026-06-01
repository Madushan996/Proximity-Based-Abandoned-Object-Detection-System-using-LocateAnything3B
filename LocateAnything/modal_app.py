"""
PADS + LocateAnything-3B — Modal deployment.

Hosts the NVIDIA LocateAnything-3B VLM on a Modal GPU container together with
the full PADS abandoned-object pipeline, and serves a web UI that streams
annotated video + live alarms over a WebSocket.

Layout inside the container:
    /root/pads   ← the existing PADS package (modules/, state/, utils/, main.py)
    /root/app    ← this folder (locateanything_worker.py, pipeline.py, ui/)
    /cache/hf    ← HuggingFace weights cache (persistent Volume)

Deploy:
    pip install modal
    modal setup                 # one-time auth
    modal deploy modal_app.py   # prints the public web URL

Serve while iterating:
    modal serve modal_app.py
"""
# NOTE: do NOT add `from __future__ import annotations` here. It turns every
# annotation into a lazy string, which makes FastAPI store `UploadFile` as a
# ForwardRef it then fails to resolve for routes defined inside web() — the
# "TypeAdapter[...UploadFile...] is not fully defined" pydantic error.

import sys
from pathlib import Path

import modal

GPU_TYPE = "A10G"          # Ampere — minimum generation LocateAnything supports
MODEL_ID = "nvidia/LocateAnything-3B"

app = modal.App("pads-locateanything")

# Persistent cache so the ~6GB model is downloaded only once across runs.
hf_cache = modal.Volume.from_name("pads-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "torch",
        "transformers==4.57.1",
        "accelerate",
        "einops",
        "peft",
        "ultralytics",                       # YOLOv8 person tracker (M3)
        "opencv-python-headless==4.11.0.86",
        "numpy==1.25.0",
        "Pillow==11.1.0",
        "pyyaml",
        "decord==0.6.0",
        "lmdb==1.7.5",
        "fastapi[standard]",
    )
    .env({
        "PADS_DIR": "/root/pads",
        "HF_HOME": "/cache/hf",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "YOLO_CONFIG_DIR": "/cache/ultralytics",
    })
)

# These run ONLY inside the container (fastapi isn't installed on the deploy
# machine). Binding the FastAPI types at *module* level is required so that
# FastAPI can resolve the UploadFile/WebSocket annotations on the route
# handlers defined inside web() — otherwise pydantic raises
# "TypeAdapter[...UploadFile...] is not fully defined".
with image.imports():
    import asyncio
    import json
    import subprocess
    import tempfile
    import threading
    import uuid
    from fastapi import FastAPI, UploadFile, File, Form, WebSocket
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

# Local source dirs are resolved + mounted ONLY on the deploy machine. Inside
# the container this module is re-imported with __file__=/root/modal_app.py, so
# these paths would be wrong — guard them with modal.is_local().
if modal.is_local():
    APP_DIR  = Path(__file__).parent
    PADS_DIR = APP_DIR.parent / "PADS"          # sibling folder
    assert PADS_DIR.exists(), f"PADS folder not found at {PADS_DIR}"
    # Mount source AFTER pip layers so code edits don't bust the dependency cache.
    image = (
        image
        .add_local_dir(str(PADS_DIR), "/root/pads")
        .add_local_dir(str(APP_DIR), "/root/app")
    )


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/cache": hf_cache},
    scaledown_window=300,      # keep the loaded model warm 5 min between requests
    timeout=60 * 60,           # allow long videos
)
@modal.concurrent(max_inputs=8)   # let UI assets + a job share one warm container
class PADSEngine:

    @modal.enter()
    def load(self):
        # Imports happen here so they only run inside the GPU container.
        sys.path.insert(0, "/root/app")
        sys.path.insert(0, "/root/pads")
        from locateanything_worker import LocateAnythingWorker

        print("[PADSEngine] loading LocateAnything-3B …")
        self.worker = LocateAnythingWorker(MODEL_ID, device="cuda", dtype="bfloat16")
        print("[PADSEngine] model ready.")

    # ── default PADS config (UI may override a few timer fields) ──────────────
    def _base_config(self) -> dict:
        return {
            "input": {"init_frames": 15},
            "coordinates": {"homography_enabled": False, "pixel_scale_px_per_m": 100},
            "foreground": {
                "method": "LOCATEANYTHING",
                "obj_min_area": 1500,
                "locateanything": {
                    "categories": ["backpack", "handbag", "suitcase", "luggage",
                                   "bag", "box", "trolley", "duffel bag"],
                    "infer_every_frames": 24,
                    "stability_frames": 2,
                    "iou_match": 0.4,
                    "resize_long_side": 768,
                    # "hybrid" = MTP fast-path; confirmed working on plain sdpa
                    # (magi_attention/flash_attn fall back cleanly). "slow" is
                    # pure autoregressive if you ever need a fallback.
                    "generation_mode": "hybrid",
                },
            },
            "person": {"model": "yolov8n.pt", "confidence": 0.4, "iou": 0.5,
                       "tracker": "bytetrack.yaml", "device": "cuda", "half": True},
            "association": {"proximity_radius_m": 2.5, "pending_timeout_s": 10,
                            "ownership_lookback_s": 10},
            "separation": {"separation_threshold_m": 3.0, "grace_period_s": 4, "window_s": 1.0},
            "alarm": {"abandon_timeout_s": 5, "retrieve_window_s": 3},
            "visualizer": {"enabled": True, "show_person_boxes": True, "show_object_boxes": True,
                           "show_associations": True, "show_distances": True,
                           "show_status_overlay": True, "output_video": ""},
        }

    # Remotely-callable wrapper around _detect (used by the smoke entrypoint).
    @modal.method()
    def detect_image(self, image_bytes: bytes, categories: list[str], gen_mode: str = "slow") -> dict:
        return self._detect(image_bytes, categories, gen_mode)

    # ── pure detection endpoint (for running PADS elsewhere, model here) ──────
    def _detect(self, image_bytes: bytes, categories: list[str], gen_mode: str) -> dict:
        import cv2, numpy as np
        from PIL import Image
        from locateanything_worker import LocateAnythingWorker

        arr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        answer = self.worker.detect(pil, categories, generation_mode=gen_mode)
        boxes = LocateAnythingWorker.parse_boxes(answer, w, h)
        return {"width": w, "height": h, "boxes": boxes, "raw": answer}

    # ── web app (UI + upload + websocket + /detect) ───────────────────────────
    @modal.asgi_app()
    def web(self):
        sys.path.insert(0, "/root/app")
        sys.path.insert(0, "/root/pads")

        api = FastAPI(title="PADS + LocateAnything")
        # Allow the local launcher (localhost) to call this deployed endpoint.
        api.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
            allow_headers=["*"],
        )
        ui_html = (Path("/root/app/ui/index.html")).read_text(encoding="utf-8")
        jobs: dict[str, str] = {}      # job_id -> uploaded video path
        outputs: dict[str, str] = {}   # job_id -> rendered annotated mp4 path

        @api.get("/", response_class=HTMLResponse)
        async def index():
            return ui_html

        @api.get("/api/jobs/{job_id}/video")
        async def get_video(job_id: str):
            p = outputs.get(job_id)
            if not p or not Path(p).exists():
                return JSONResponse({"error": "not ready"}, status_code=404)
            # FileResponse honours Range requests → the browser can scrub.
            return FileResponse(p, media_type="video/mp4")

        @api.post("/api/jobs")
        async def create_job(video: UploadFile = File(...)):
            job_id = uuid.uuid4().hex
            suffix = Path(video.filename or "in.mp4").suffix or ".mp4"
            path = str(Path(tempfile.gettempdir()) / f"{job_id}{suffix}")
            with open(path, "wb") as f:
                f.write(await video.read())
            jobs[job_id] = path
            return {"job_id": job_id}

        @api.post("/detect")
        async def detect(
            image: UploadFile = File(...),
            categories: str = Form("bag,suitcase,backpack,handbag"),
            generation_mode: str = Form("hybrid"),
        ):
            data = await image.read()
            cats = [c.strip() for c in categories.split(",") if c.strip()]
            result = await asyncio.to_thread(self._detect, data, cats, generation_mode)
            return JSONResponse(result)

        @api.websocket("/ws/jobs/{job_id}")
        async def stream_job(ws: WebSocket, job_id: str):
            await ws.accept()
            path = jobs.get(job_id)
            if not path:
                await ws.send_text(json.dumps({"type": "error", "message": "unknown job_id"}))
                await ws.close()
                return

            # Optional UI overrides sent as the first ws text message.
            cfg = self._base_config()
            cfg["mode"] = "pads"   # "pads" = full abandonment pipeline; "label" = overlay
            try:
                first = await asyncio.wait_for(ws.receive_text(), timeout=5)
                overrides = json.loads(first)
                cfg["mode"] = overrides.get("mode", "pads")
                cfg["alarm"]["abandon_timeout_s"] = int(
                    overrides.get("abandon_timeout_s", cfg["alarm"]["abandon_timeout_s"]))
                if overrides.get("categories"):
                    cfg["foreground"]["locateanything"]["categories"] = overrides["categories"]
                cfg["foreground"]["locateanything"]["infer_every_frames"] = int(
                    overrides.get("infer_every_frames",
                                  cfg["foreground"]["locateanything"]["infer_every_frames"]))
            except Exception:
                pass  # no/!json overrides → defaults

            loop = asyncio.get_event_loop()
            queue: asyncio.Queue = asyncio.Queue(maxsize=256)
            SENTINEL = object()

            def push(msg):
                loop.call_soon_threadsafe(queue.put_nowait, msg)

            def worker_thread():
                # Pipe annotated frames straight into ffmpeg → browser-playable
                # H.264 mp4. We stream only lightweight progress/events over the
                # socket; the finished video is fetched once and played natively.
                from pipeline import run_pipeline, run_labeling
                out_path = str(Path(tempfile.gettempdir()) / f"{job_id}_out.mp4")
                ff = None
                events: list = []
                summary = None
                total = 0
                la = cfg["foreground"]["locateanything"]
                if cfg.get("mode") == "label":
                    gen = run_labeling(
                        path, la["categories"], worker=self.worker,
                        infer_every=la["infer_every_frames"],
                        resize_long=la["resize_long_side"],
                        gen_mode=la["generation_mode"], jpeg_quality=85,
                    )
                else:
                    gen = run_pipeline(path, cfg, worker=self.worker,
                                       stream_every=1, jpeg_quality=85)
                try:
                    for msg in gen:
                        kind = msg.get("type")
                        if kind == "meta":
                            fps = msg.get("fps") or 24
                            total = msg.get("total_frames", 0)
                            ff = subprocess.Popen(
                                ["ffmpeg", "-y", "-f", "image2pipe",
                                 "-framerate", str(fps), "-i", "pipe:0",
                                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                 "-movflags", "+faststart", out_path],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                            push({"type": "meta", "total_frames": total,
                                  "fps": fps, "source": msg.get("source", "")})
                        elif kind == "frame":
                            if ff:
                                ff.stdin.write(msg["jpeg"])
                            if msg["idx"] % 5 == 0:
                                push({"type": "progress", "idx": msg["idx"],
                                      "total": total, "t": msg["t"]})
                        elif kind == "event":
                            events.append(msg["event"])
                            push(msg)
                        elif kind == "done":
                            summary = msg["summary"]

                    if ff and ff.stdin:
                        ff.stdin.close()
                        ff.wait()
                    outputs[job_id] = out_path
                    push({"type": "done", "summary": summary, "events": events,
                          "video_url": f"/api/jobs/{job_id}/video"})
                except Exception as e:
                    push({"type": "error", "message": str(e)})
                finally:
                    if ff and ff.stdin and not ff.stdin.closed:
                        try:
                            ff.stdin.close()
                            ff.wait()
                        except Exception:
                            pass
                    push(SENTINEL)

            threading.Thread(target=worker_thread, daemon=True).start()

            while True:
                msg = await queue.get()
                if msg is SENTINEL:
                    break
                await ws.send_text(json.dumps(msg))
            await ws.close()

        return api


@app.local_entrypoint()
def smoke():
    """Quick check that the model loads and detects on a blank frame."""
    import numpy as np, cv2
    engine = PADSEngine()
    blank = np.full((480, 640, 3), 200, np.uint8)
    ok, buf = cv2.imencode(".jpg", blank)
    res = engine.detect_image.remote(buf.tobytes(), ["bag", "suitcase"], "hybrid")
    print("detect() returned:", res)
