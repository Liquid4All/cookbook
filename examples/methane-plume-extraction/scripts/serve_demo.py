#!/usr/bin/env python3
"""Serve the interactive methane plume demo and optional inference endpoint bridge."""

from __future__ import annotations

import argparse
import json
import os
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):  # noqa: N802
        if self.path in {"/", "/demo"}:
            self.path = "/interactive_demo.html"
        return super().do_GET()

    def do_POST(self):  # noqa: N802
        if self.path != "/api/infer":
            self.send_error(404, "Not found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        target = self._infer_target(body)
        if not target:
            self._send_json(
                503,
                {
                    "error": "No inference endpoint is configured for this model variant.",
                    "expected": "Set METHANE_INFER_FINE_URL and/or METHANE_INFER_BASE_URL, or set METHANE_INFER_URL as a fallback endpoint.",
                },
            )
            return

        request = Request(
            target,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=180) as response:
                payload = response.read()
                status = response.status
                content_type = response.headers.get("Content-Type", "application/json")
        except HTTPError as exc:
            payload = exc.read() or json.dumps({"error": str(exc)}).encode("utf-8")
            status = exc.code
            content_type = exc.headers.get("Content-Type", "application/json")
        except URLError as exc:
            self._send_json(502, {"error": f"Inference endpoint unavailable: {exc.reason}"})
            return
        except TimeoutError:
            self._send_json(504, {"error": "Inference endpoint timed out."})
            return

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()

    def _send_json(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _infer_target(self, body: bytes) -> str:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {}

        variant = str(payload.get("model_variant", "fine")).strip().lower()
        if variant == "base":
            return os.environ.get("METHANE_INFER_BASE_URL", "").strip() or os.environ.get("METHANE_INFER_URL", "").strip()
        return os.environ.get("METHANE_INFER_FINE_URL", "").strip() or os.environ.get("METHANE_INFER_URL", "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the methane plume interactive demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"Serving methane plume demo at http://{args.host}:{args.port}/demo")
    if os.environ.get("METHANE_INFER_URL"):
        print(f"Proxying /api/infer to {os.environ['METHANE_INFER_URL']}")
    if os.environ.get("METHANE_INFER_BASE_URL"):
        print(f"Proxying base-model uploads to {os.environ['METHANE_INFER_BASE_URL']}")
    if os.environ.get("METHANE_INFER_FINE_URL"):
        print(f"Proxying fine-tuned uploads to {os.environ['METHANE_INFER_FINE_URL']}")
    else:
        print("METHANE_INFER_FINE_URL not set; fine-tuned uploads will use METHANE_INFER_URL or report not configured.")
    server.serve_forever()


if __name__ == "__main__":
    main()
