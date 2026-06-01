"""
LocateAnything-3B worker.

Wraps NVIDIA's open-vocabulary localisation VLM (nvidia/LocateAnything-3B).
Given an image + a natural-language query it returns bounding boxes (or points)
as text, which we parse into pixel coordinates.

This module is intended to run *inside the GPU container* (Modal). It is the
only place that imports torch / transformers, so the rest of PADS stays
importable on a CPU-only / Windows machine.

Reference: https://huggingface.co/nvidia/LocateAnything-3B
Model requires an Ampere-or-newer NVIDIA GPU and bfloat16.
"""
from __future__ import annotations

import re
from typing import List, Dict

# Heavy deps are imported lazily inside __init__ so that simply importing this
# module (e.g. for the parse_* static helpers) does not require torch.

# Prompt templates from the model card.
_DETECT_TMPL = "Locate all the instances that matches the following description: {desc}."
_GROUND_MULTI_TMPL = "Locate all the instances that match the following description: {desc}."
_POINT_TMPL = "Point to: {desc}."

# Box:   <box><x1><y1><x2><y2></box>   coords are integers normalised to 0..1000
_BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
# Point: <box><x><y></box>
_POINT_RE = re.compile(r"<box><(\d+)><(\d+)></box>")


class LocateAnythingWorker:
    def __init__(
        self,
        model_path: str = "nvidia/LocateAnything-3B",
        device: str = "cuda",
        dtype: str = "bfloat16",
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer, AutoProcessor

        self.device = device
        self._torch = torch
        self.dtype = getattr(torch, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = (
            AutoModel.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )
            .to(device)
            .eval()
        )

    # ── core inference ────────────────────────────────────────────────────────

    def _generate(
        self,
        image,
        question: str,
        generation_mode: str = "slow",   # "fast"/"hybrid" need magi/flash-attn kernels
        max_new_tokens: int = 1024,      # plenty for box tags; bounds worst-case time
        temperature: float = 0.0,
    ) -> str:
        torch = self._torch
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        do_sample = temperature and temperature > 0.0

        with torch.no_grad():
            response = self.model.generate(
                pixel_values=pixel_values,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws", None),
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                generation_mode=generation_mode,
                temperature=float(temperature),   # model does `if temperature > 0` — never pass None
                do_sample=bool(do_sample),
                top_p=0.9,
                repetition_penalty=1.1,
                verbose=False,
            )

        # generate() may return a str or a (str, history, stats) tuple.
        return response[0] if isinstance(response, tuple) else response

    # ── task convenience wrappers ───────────────────────────────────────────────

    def detect(self, image, categories: List[str], **kw) -> str:
        """Open-vocabulary detection. `categories` is a list of class names."""
        desc = ", ".join(c.strip() for c in categories if c.strip())
        return self._generate(image, _DETECT_TMPL.format(desc=desc), **kw)

    def detect_labeled(self, image, categories: List[str], **kw) -> List[Dict]:
        """Per-category detection so each box carries a reliable label.

        Querying one class at a time avoids the garbled multi-class `<ref>` tags
        the model emits for a long comma list. Returns pixel-space boxes:
            [{"label": str, "x1","y1","x2","y2": float}, ...]
        `image` must be a PIL image (its .size gives the scaling reference).
        """
        w, h = image.size
        results: List[Dict] = []
        for cat in (c.strip() for c in categories if c.strip()):
            answer = self._generate(image, _DETECT_TMPL.format(desc=cat), **kw)
            for box in self.parse_boxes(answer, w, h):
                box["label"] = cat
                results.append(box)
        return results

    def ground_multi(self, image, phrase: str, **kw) -> str:
        return self._generate(image, _GROUND_MULTI_TMPL.format(desc=phrase), **kw)

    def point(self, image, phrase: str, **kw) -> str:
        return self._generate(image, _POINT_TMPL.format(desc=phrase), **kw)

    # ── output parsing (static — usable without torch) ──────────────────────────

    @staticmethod
    def parse_boxes(answer: str, image_width: int, image_height: int) -> List[Dict]:
        """Parse `<box>` tags into pixel-space (x1,y1,x2,y2) dicts."""
        boxes: List[Dict] = []
        for m in _BOX_RE.finditer(answer):
            x1, y1, x2, y2 = (int(g) for g in m.groups())
            boxes.append(
                {
                    "x1": x1 / 1000.0 * image_width,
                    "y1": y1 / 1000.0 * image_height,
                    "x2": x2 / 1000.0 * image_width,
                    "y2": y2 / 1000.0 * image_height,
                }
            )
        return boxes

    @staticmethod
    def parse_points(answer: str, image_width: int, image_height: int) -> List[Dict]:
        """Parse two-coordinate `<box>` point tags into pixel-space points."""
        points: List[Dict] = []
        for m in _POINT_RE.finditer(answer):
            x, y = int(m.group(1)), int(m.group(2))
            points.append({"x": x / 1000.0 * image_width, "y": y / 1000.0 * image_height})
        return points
