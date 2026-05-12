"""Browser demo for the fine-tuned LFM2.5-Audio voice assistant.

Boots the `llama-liquid-audio-server` for the configured GGUF (mirroring the
lifecycle in `eval.py`), then serves a single-page browser frontend from a
Starlette wrapper. The frontend captures push-to-talk audio, posts it to
`/api/predict`, and renders the streaming function call the model emits.

Usage:
    uv run python scripts/demo.py --config configs/demo.yaml
"""

from __future__ import annotations

import argparse
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import uvicorn
import yaml
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from _server import ServerHandle, boot_model_server, stop_model_server

load_dotenv()

STATIC_DIR = Path(__file__).parent / "demo_static"


@dataclass
class DemoConfig:
    name: str
    model_repo: str
    quant: str
    system_prompt: str | None
    max_new_tokens: int = 64
    temperature: float = 0.0
    port: int = 8090
    demo_port: int = 8000

    @classmethod
    def from_yaml(cls, path: Path) -> DemoConfig:
        with path.open() as f:
            data: dict[str, Any] = yaml.safe_load(f)
        return cls(**data)


def _build_chat_payload(cfg: DemoConfig, audio_b64: str) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if cfg.system_prompt:
        messages.append({"role": "system", "content": cfg.system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": "wav"},
                }
            ],
        }
    )
    return {
        "model": f"{cfg.model_repo}:{cfg.quant}",
        "messages": messages,
        "max_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature,
        "stream": True,
    }


def _build_app(cfg: DemoConfig) -> Starlette:
    upstream = f"http://127.0.0.1:{cfg.port}/v1/chat/completions"

    async def index(_: Request) -> Any:
        return StreamingResponse(
            iter([(STATIC_DIR / "index.html").read_bytes()]),
            media_type="text/html",
        )

    async def predict(req: Request) -> Any:
        try:
            body = await req.json()
            audio_b64 = body["audio_b64"]
        except (KeyError, ValueError):
            return JSONResponse({"error": "missing audio_b64"}, status_code=400)

        payload = _build_chat_payload(cfg, audio_b64)

        async def proxy() -> AsyncIterator[bytes]:
            async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
                async with client.stream("POST", upstream, json=payload) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk

        return StreamingResponse(proxy(), media_type="text/event-stream")

    return Starlette(
        routes=[
            Route("/", index),
            Route("/api/predict", predict, methods=["POST"]),
            Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static"),
        ]
    )


def _open_browser_when_ready(url: str, delay: float = 0.6) -> None:
    """Open the browser after a brief delay to let uvicorn finish binding."""
    def _go() -> None:
        time.sleep(delay)
        webbrowser.open(url)
    threading.Thread(target=_go, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML demo config.")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open the browser tab.",
    )
    parser.add_argument(
        "--verbose-server",
        action="store_true",
        help="Show llama-liquid-audio-server stdout/stderr.",
    )
    args = parser.parse_args()

    cfg = DemoConfig.from_yaml(args.config)
    handle: ServerHandle | None = None
    try:
        handle = boot_model_server(
            cfg.model_repo, cfg.quant, cfg.port, verbose=args.verbose_server
        )
        app = _build_app(cfg)
        url = f"http://127.0.0.1:{cfg.demo_port}"
        print(f"Demo ready: {url}\n", flush=True)
        if not args.no_browser:
            _open_browser_when_ready(url)
        uvicorn.run(app, host="127.0.0.1", port=cfg.demo_port, log_level="warning")
    finally:
        if handle is not None:
            stop_model_server(handle)


if __name__ == "__main__":
    main()
