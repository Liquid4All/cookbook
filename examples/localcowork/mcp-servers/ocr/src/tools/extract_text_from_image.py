"""
ocr.extract_text_from_image — Extract text from an image file using OCR.

Non-destructive: executes immediately, no confirmation needed.

Engine priority (ADR-004):
  1. LFM Vision via model-gateway — primary, highest accuracy
  2. Tesseract via pytesseract   — fallback, no GPU needed
  3. Error                       — no engine available

Gracefully degrades if engines are not available.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from pydantic import BaseModel, Field

from mcp_base import MCPError, MCPResult, MCPTool, ErrorCodes
from validation import assert_sandboxed, assert_absolute_path

# ─── Types ───────────────────────────────────────────────────────────────────

VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}

MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}


class Params(BaseModel):
    """Parameters for extract_text_from_image."""

    path: str = Field(description="Path to image file")
    language: str = Field(default="eng", description="OCR language")


class Result(BaseModel):
    """Return value for extract_text_from_image."""

    text: str
    confidence: float
    engine: str = Field(default="unknown", description="Which OCR engine was used")


class ExtractTextFromImage(MCPTool[Params, Result]):
    """Extract text from an image file using OCR."""

    name = "ocr.extract_text_from_image"
    description = "Extract text from an image file using OCR"
    confirmation_required = False
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[Result]:
        """Run OCR on an image file and return extracted text.

        Engine priority (ADR-004): LFM Vision → Tesseract → error.
        """
        assert_absolute_path(params.path, "path")
        assert_sandboxed(params.path)

        if not os.path.exists(params.path):
            raise MCPError(ErrorCodes.FILE_NOT_FOUND, f"File not found: {params.path}")

        _validate_image_extension(params.path)

        # 1. Try LFM Vision via model-gateway (primary — ADR-004)
        missing_packages: list[str] = []
        try:
            text, confidence = await _ocr_with_vision_model(params.path)
            return MCPResult(
                success=True, data=Result(text=text, confidence=confidence, engine="lfm_vision")
            )
        except ImportError as exc:
            missing_packages.append(f"aiohttp ({exc})")
        except MCPError:
            pass  # Model returned an error, fall through to Tesseract

        # 2. Fall back to Tesseract
        try:
            text, confidence = _ocr_with_tesseract(params.path, params.language)
            return MCPResult(
                success=True, data=Result(text=text, confidence=confidence, engine="tesseract")
            )
        except ImportError as exc:
            missing_packages.append(f"pytesseract ({exc})")

        # Both engines failed — provide actionable diagnostics
        if missing_packages:
            detail = "; ".join(missing_packages)
            raise MCPError(
                ErrorCodes.INTERNAL_ERROR,
                f"No OCR engine available — missing packages: {detail}. "
                f"Check Settings > Servers to repair the OCR server environment.",
            )

        raise MCPError(
            ErrorCodes.INTERNAL_ERROR,
            "No OCR engine available. Start the vision model server or install pytesseract.",
        )


# ─── Validators ──────────────────────────────────────────────────────────────


def _validate_image_extension(path: str) -> None:
    """Validate that the file has an image extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in VALID_IMAGE_EXTS:
        raise MCPError(
            ErrorCodes.INVALID_PARAMS,
            f"Unsupported image format: {ext}. Supported: {', '.join(sorted(VALID_IMAGE_EXTS))}",
        )


# ─── Engine: LFM Vision (via model-gateway) ─────────────────────────────────


async def _ocr_with_vision_model(path: str) -> tuple[str, float]:
    """Run OCR using a vision-capable model via OpenAI-compatible API.

    Sends the image as a base64 data URL in the OpenAI vision format.
    Reads endpoint and model from environment variables:
      - LOCALCOWORK_VISION_ENDPOINT (default: http://localhost:8081/v1)
      - LOCALCOWORK_VISION_MODEL (default: LFM2.5-VL-1.6B)

    Raises ImportError on connection failure to trigger the Tesseract fallback.

    NOTE(WS-8): Will be replaced by model-gateway service in a future workstream
    for unified model routing and fallback chain management.
    """
    import aiohttp  # type: ignore[import-untyped]

    endpoint = os.environ.get("LOCALCOWORK_VISION_ENDPOINT", "http://localhost:8081/v1")
    model = os.environ.get("LOCALCOWORK_VISION_MODEL", "LFM2.5-VL-1.6B")
    url = f"{endpoint}/chat/completions"

    # Encode image as base64 data URL
    ext = os.path.splitext(path)[1].lower()
    mime = MIME_TYPES.get(ext, "image/png")
    image_bytes = Path(path).read_bytes()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all text from this image. Return only the raw text, "
                        "preserving line breaks and layout. Do not add commentary.",
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise ImportError(f"Vision model returned {resp.status}: {body[:200]}")
                result = await resp.json()
    except (aiohttp.ClientError, OSError) as exc:
        raise ImportError(f"Vision model unavailable at {endpoint}: {exc}") from exc

    text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not text.strip():
        raise ImportError("Vision model returned empty response")

    # Vision models don't provide per-word confidence; use a fixed high estimate
    return text.strip(), 0.90


# ─── Engine: Tesseract ───────────────────────────────────────────────────────


def _ocr_with_tesseract(path: str, language: str) -> tuple[str, float]:
    """Run OCR using Tesseract."""
    import pytesseract  # type: ignore[import-untyped]
    from PIL import Image

    image = Image.open(path)
    data = pytesseract.image_to_data(image, lang=language, output_type=pytesseract.Output.DICT)

    lines: list[str] = []
    confidences: list[float] = []

    current_line: list[str] = []
    current_line_num = -1

    for i, text in enumerate(data["text"]):
        conf = int(data["conf"][i])
        line_num = data["line_num"][i]

        if line_num != current_line_num:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = []
            current_line_num = line_num

        if text.strip() and conf > 0:
            current_line.append(text)
            confidences.append(conf / 100.0)

    if current_line:
        lines.append(" ".join(current_line))

    text = "\n".join(lines)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return text, avg_confidence
