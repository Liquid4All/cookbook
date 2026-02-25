"""
ocr.extract_structured_data — Extract structured data from OCR text.

Non-destructive: executes immediately, no confirmation needed.

Engine priority (ADR-004):
  1. LLM extraction via model-gateway — primary, schema-aware
  2. Rule-based regex heuristics     — fallback, common field types
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from mcp_base import MCPError, MCPResult, MCPTool, ErrorCodes


class Params(BaseModel):
    """Parameters for extract_structured_data."""

    text: str = Field(description="OCR text to extract from")
    extraction_schema: dict[str, Any] = Field(description="JSON Schema defining expected fields")


class Result(BaseModel):
    """Return value for extract_structured_data."""

    data: dict[str, Any]
    confidence: dict[str, float]
    engine: str = Field(default="regex", description="Which extraction engine was used")


class ExtractStructuredData(MCPTool[Params, Result]):
    """Extract structured data from OCR text using the LLM."""

    name = "ocr.extract_structured_data"
    description = "Extract structured data from OCR text using the LLM"
    confirmation_required = False
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[Result]:
        """Extract structured fields from OCR text.

        Engine priority (ADR-004): LLM extraction → regex heuristics.
        """
        if not params.text.strip():
            raise MCPError(ErrorCodes.INVALID_PARAMS, "Input text is empty")

        if not params.extraction_schema:
            raise MCPError(ErrorCodes.INVALID_PARAMS, "Schema is required")

        try:
            # 1. Try LLM extraction via model-gateway (primary — ADR-004)
            try:
                data, confidence = await _extract_with_llm(
                    params.text, params.extraction_schema
                )
                return MCPResult(
                    success=True,
                    data=Result(data=data, confidence=confidence, engine="lfm_llm"),
                )
            except (ImportError, MCPError):
                pass  # Model not available, fall through

            # 2. Fall back to rule-based regex extraction
            data, confidence = _extract_fields_regex(params.text, params.extraction_schema)
            return MCPResult(
                success=True,
                data=Result(data=data, confidence=confidence, engine="regex"),
            )

        except MCPError:
            raise
        except Exception as e:
            raise MCPError(
                ErrorCodes.INTERNAL_ERROR, f"Failed to extract structured data: {e}"
            ) from e


# ─── Engine: LLM Extraction (via model-gateway) ─────────────────────────────


async def _extract_with_llm(
    text: str, schema: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, float]]:
    """Extract structured data using the local LLM via model-gateway.

    Sends the OCR text with a JSON Schema prompt, requesting structured output.
    The model returns a JSON object matching the schema.

    NOTE: This is a stub — full implementation requires the model-gateway
    service (WS-2B Inference Client). Currently raises ImportError to trigger
    the regex fallback.
    """
    # TODO(WS-2B): Replace stub with actual model-gateway call:
    #
    #   from _shared.services.model_gateway import ModelGateway, ModelUnavailableError
    #
    #   gateway = ModelGateway.get_instance()
    #   schema_json = json.dumps(schema, indent=2)
    #
    #   response = await gateway.call(
    #       messages=[
    #           {"role": "system", "content": (
    #               "Extract structured data from the following OCR text. "
    #               f"Return a JSON object matching this schema:\n{schema_json}\n"
    #               "Return ONLY valid JSON with the extracted values."
    #           )},
    #           {"role": "user", "content": text},
    #       ],
    #       response_format={"type": "json_schema", "json_schema": schema},
    #   )
    #   data = json.loads(response.content)
    #   confidence = {k: 0.9 for k in data}  # LLM extraction is high-confidence
    #   return data, confidence

    raise ImportError("model-gateway not yet available (WS-2B)")


# ─── Engine: Regex Heuristics (fallback) ─────────────────────────────────────


def _extract_fields_regex(
    text: str, schema: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, float]]:
    """Extract fields from text based on a JSON Schema using regex heuristics.

    This is the fallback when the LLM is not available.
    """
    properties = schema.get("properties", {})
    data: dict[str, Any] = {}
    confidence: dict[str, float] = {}

    for field_name, field_def in properties.items():
        field_type = field_def.get("type", "string")
        description = field_def.get("description", field_name)

        extracted, conf = _extract_single_field(text, field_name, field_type, description)
        data[field_name] = extracted
        confidence[field_name] = conf

    return data, confidence


def _extract_single_field(
    text: str, field_name: str, field_type: str, description: str
) -> tuple[Any, float]:
    """Extract a single field using regex heuristics."""
    name_lower = field_name.lower()

    # Amount / total / price patterns
    if any(kw in name_lower for kw in ("amount", "total", "price", "cost", "sum")):
        match = re.search(r"\$\s*([\d,]+\.?\d*)", text)
        if match:
            return float(match.group(1).replace(",", "")), 0.8
        match = re.search(r"([\d,]+\.?\d*)\s*(?:USD|EUR|GBP)", text)
        if match:
            return float(match.group(1).replace(",", "")), 0.7

    # Date patterns
    if any(kw in name_lower for kw in ("date", "time", "when")):
        # ISO date
        match = re.search(r"\d{4}-\d{2}-\d{2}", text)
        if match:
            return match.group(0), 0.9
        # US date
        match = re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", text)
        if match:
            return match.group(0), 0.7

    # Email
    if "email" in name_lower:
        match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
        if match:
            return match.group(0), 0.9

    # Phone
    if "phone" in name_lower or "tel" in name_lower:
        match = re.search(r"[\+]?[(]?\d{1,4}[)]?[-\s./0-9]{7,15}", text)
        if match:
            return match.group(0).strip(), 0.7

    # Number fields
    if field_type == "number":
        match = re.search(r"[\d,]+\.?\d*", text)
        if match:
            return float(match.group(0).replace(",", "")), 0.5

    # Generic: try to find the field name followed by a colon and value
    pattern = rf"(?i){re.escape(field_name)}\s*[:=]\s*(.+?)(?:\n|$)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip(), 0.6

    return None, 0.0
