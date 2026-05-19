#!/usr/bin/env python3
"""Serve local inference for the methane plume demo.

The default backend is Transformers because it matches the evaluated HF
checkpoint. On Apple Silicon it uses MPS automatically. The MLX backend is kept
available for Liquid-docs-style Apple Silicon testing, but validate parity before
using it for a customer demo.
"""

from __future__ import annotations

import argparse
import base64
import io
import importlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINE_MODEL = os.environ.get(
    "FINE_MODEL_ID",
    str(ROOT / "models" / "fine"),
)
DEFAULT_BASE_MODEL = os.environ.get(
    "BASE_MODEL_ID",
    str(ROOT / "models" / "base"),
)
DEFAULT_FINE_MLX_MODEL = os.environ.get("FINE_MLX_MODEL_ID", str(ROOT / "models" / "fine-mlx"))
DEFAULT_BASE_MLX_MODEL = os.environ.get("BASE_MLX_MODEL_ID", "mlx-community/LFM2.5-VL-1.6B-4bit")
JSON_FOOTER = "\n\nRespond with only a JSON object. Do not include any text outside the JSON."


def patch_mlx_detokenizer() -> None:
    """Work around an mlx-vlm 0.5.0 copy() issue on the LFM2-VL detokenizer."""

    import mlx_vlm.tokenizer_utils as tokenizer_utils

    def no_copy_detokenizer(processor):
        detok = processor.detokenizer
        detok.reset()
        return detok

    generate_module = importlib.import_module("mlx_vlm.generate")
    tokenizer_utils.make_streaming_detokenizer = no_copy_detokenizer
    generate_module.make_streaming_detokenizer = no_copy_detokenizer


def extract_json(text: str) -> dict[str, Any]:
    clean = (text or "").strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
        clean = clean.rsplit("```", 1)[0].strip()
    start = clean.find("{")
    if start < 0:
        return {}
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(clean[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(clean[start : i + 1])
                    return parsed if isinstance(parsed, dict) else {}
                except json.JSONDecodeError:
                    return {}
    return {}


def schema_to_yaml(schema: dict[str, Any]) -> str:
    lines = []
    for key, value in schema.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def prompt_from_request(prompt: str | None, schema: dict[str, Any] | None) -> str:
    if prompt:
        system = prompt.strip()
    elif schema:
        system = "Extract the following from the image:\n\n" + schema_to_yaml(schema)
    else:
        raise HTTPException(status_code=400, detail="Request must include prompt or schema.")
    if "Respond with only a JSON object" not in system:
        system += JSON_FOOTER
    return system


def decode_image_bytes(image_b64: str) -> tuple[bytes, str]:
    if not image_b64:
        raise HTTPException(status_code=400, detail="Missing image_b64.")
    suffix = ".png"
    payload = image_b64
    if "," in image_b64 and image_b64.startswith("data:"):
        header, payload = image_b64.split(",", 1)
        if "jpeg" in header or "jpg" in header:
            suffix = ".jpg"
    try:
        raw = base64.b64decode(payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}") from exc
    return raw, suffix


def decode_image_to_tempfile(image_b64: str) -> tempfile.NamedTemporaryFile:
    raw, suffix = decode_image_bytes(image_b64)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(raw)
    tmp.flush()
    tmp.close()
    return tmp


class InferRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    image_b64: str
    schema_payload: dict[str, Any] | None = Field(default=None, alias="schema")
    prompt: str | None = None
    model_variant: str = "fine"
    filename: str | None = None


class ModelCache:
    def __init__(
        self,
        fine_model: str,
        base_model: str,
        backend: Literal["transformers", "mlx"],
        preload: str,
        device: str,
    ) -> None:
        self.paths = {"fine": fine_model, "base": base_model}
        self.models: dict[str, tuple[Any, Any]] = {}
        self.lock = threading.Lock()
        self.backend = backend
        self.device = device
        if self.backend == "mlx":
            patch_mlx_detokenizer()
        if preload in {"fine", "both"}:
            self.get("fine")
        if preload == "both":
            self.get("base")

    def get(self, variant: str):
        key = "base" if variant == "base" else "fine"
        if key not in self.models:
            self.models[key] = self._load_model(self.paths[key])
        return self.models[key]

    def _load_model(self, path: str) -> tuple[Any, Any]:
        if self.backend == "mlx":
            from mlx_vlm import load

            return load(path)

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        device = self.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif device == "mps":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            path,
            trust_remote_code=True,
            dtype=dtype,
        )
        model = model.to(device)
        model.eval()
        return model, processor

    def infer(self, request: InferRequest, max_tokens: int) -> dict[str, Any]:
        model, processor = self.get(request.model_variant)
        system = prompt_from_request(request.prompt, request.schema_payload)
        if self.backend == "mlx":
            return self._infer_mlx(model, processor, request, system, max_tokens)
        return self._infer_transformers(model, processor, request, system, max_tokens)

    def _infer_mlx(
        self,
        model: Any,
        processor: Any,
        request: InferRequest,
        system: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        image_file = decode_image_to_tempfile(request.image_b64)
        started = time.perf_counter()
        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": ""},
            ]
            prompt = apply_chat_template(processor, model.config, messages, num_images=1)
            with self.lock:
                result = generate(
                    model,
                    processor,
                    prompt,
                    image=[image_file.name],
                    max_tokens=max_tokens,
                    temperature=0.0,
                    prefill_step_size=512,
                    verbose=False,
                )
            raw = result.text.strip()
            prediction = extract_json(raw)
            return {
                "prediction_json": prediction,
                "prediction_raw": raw,
                "model_variant": "base" if request.model_variant == "base" else "fine",
                "engine": "mlx-vlm",
                "latency_ms": round((time.perf_counter() - started) * 1000),
            }
        finally:
            Path(image_file.name).unlink(missing_ok=True)

    def _infer_transformers(
        self,
        model: Any,
        processor: Any,
        request: InferRequest,
        system: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        import torch

        raw_bytes, _ = decode_image_bytes(request.image_b64)
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "image", "image": image}]},
        ]
        started = time.perf_counter()
        with self.lock:
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(text=text, images=[image], return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = output_ids[:, input_len:]
            raw = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        prediction = extract_json(raw)
        return {
            "prediction_json": prediction,
            "prediction_raw": raw,
            "model_variant": "base" if request.model_variant == "base" else "fine",
            "engine": f"transformers-{next(model.parameters()).device}",
            "latency_ms": round((time.perf_counter() - started) * 1000),
        }


def create_app(cache: ModelCache, max_tokens: int) -> FastAPI:
    app = FastAPI(title="Methane plume local inference")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "loaded": sorted(cache.models.keys()),
            "engine": cache.backend,
            "device": cache.device,
        }

    @app.post("/infer")
    def infer(request: InferRequest) -> dict[str, Any]:
        return cache.infer(request, max_tokens=max_tokens)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local inference for the methane plume demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--fine-model", default=DEFAULT_FINE_MODEL)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--fine-mlx-model", default=DEFAULT_FINE_MLX_MODEL)
    parser.add_argument("--base-mlx-model", default=DEFAULT_BASE_MLX_MODEL)
    parser.add_argument("--backend", choices=["transformers", "mlx"], default="transformers")
    parser.add_argument("--device", default="auto", help="transformers device: auto, cuda, mps, or cpu")
    parser.add_argument("--preload", choices=["none", "fine", "both"], default="fine")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    fine_model = args.fine_mlx_model if args.backend == "mlx" else args.fine_model
    base_model = args.base_mlx_model if args.backend == "mlx" else args.base_model
    cache = ModelCache(fine_model, base_model, args.backend, args.preload, args.device)
    app = create_app(cache, max_tokens=args.max_tokens)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
