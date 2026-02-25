"""Tests for ocr.extract_structured_data tool."""

from __future__ import annotations

import pytest

from tools.extract_structured_data import ExtractStructuredData


@pytest.fixture()
def tool() -> ExtractStructuredData:
    return ExtractStructuredData()


async def test_extract_amount(tool: ExtractStructuredData) -> None:
    """Should extract dollar amounts from OCR text."""
    result = await tool.execute(
        tool.get_params_model()(
            text="Invoice #12345\nDate: 2026-01-15\nTotal: $125.50\nThank you",
            extraction_schema={
                "properties": {
                    "total": {"type": "number", "description": "Total amount"},
                    "date": {"type": "string", "description": "Invoice date"},
                }
            },
        )
    )

    assert result.success is True
    assert result.data is not None
    assert result.data.data["total"] == 125.50
    assert result.data.confidence["total"] > 0
    assert result.data.data["date"] == "2026-01-15"


async def test_extract_email(tool: ExtractStructuredData) -> None:
    """Should extract email addresses."""
    result = await tool.execute(
        tool.get_params_model()(
            text="Contact us at support@example.com for help",
            extraction_schema={"properties": {"email": {"type": "string", "description": "Email address"}}},
        )
    )

    assert result.success is True
    assert result.data is not None
    assert result.data.data["email"] == "support@example.com"
    assert result.data.confidence["email"] > 0.5


async def test_extract_key_value(tool: ExtractStructuredData) -> None:
    """Should extract key-value pairs from OCR text."""
    result = await tool.execute(
        tool.get_params_model()(
            text="Name: John Doe\nCompany: Acme Corp",
            extraction_schema={
                "properties": {
                    "Name": {"type": "string"},
                    "Company": {"type": "string"},
                }
            },
        )
    )

    assert result.success is True
    assert result.data is not None
    assert result.data.data["Name"] == "John Doe"
    assert result.data.data["Company"] == "Acme Corp"


async def test_empty_text_rejected(tool: ExtractStructuredData) -> None:
    """Should reject empty input text."""
    from mcp_base import MCPError

    with pytest.raises(MCPError, match="empty"):
        await tool.execute(
            tool.get_params_model()(
                text="   ",
                extraction_schema={"properties": {"x": {"type": "string"}}},
            )
        )


async def test_missing_field_returns_none(tool: ExtractStructuredData) -> None:
    """Should return None with 0 confidence for unmatched fields."""
    result = await tool.execute(
        tool.get_params_model()(
            text="Some random text without relevant data",
            extraction_schema={"properties": {"phone": {"type": "string", "description": "Phone number"}}},
        )
    )

    assert result.success is True
    assert result.data is not None
    assert result.data.data["phone"] is None
    assert result.data.confidence["phone"] == 0.0


def test_metadata(tool: ExtractStructuredData) -> None:
    """Should have correct metadata."""
    assert tool.name == "ocr.extract_structured_data"
    assert tool.confirmation_required is False
    assert tool.undo_supported is False
