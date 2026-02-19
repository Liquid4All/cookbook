"""Tests for screenshot.capture_and_extract tool."""

from __future__ import annotations

import os

import pytest

from tools.capture_and_extract import CaptureAndExtract
from pipeline_types import CaptureRegion


@pytest.fixture()
def tool() -> CaptureAndExtract:
    """Create a CaptureAndExtract tool instance."""
    return CaptureAndExtract()


async def test_captures_screenshot_and_returns_path(tool: CaptureAndExtract) -> None:
    """Should capture a screenshot and return a valid file path."""
    params = tool.get_params_model()()
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert os.path.isabs(result.data.screenshot_path)
    assert result.data.screenshot_path.endswith(".png")
    assert os.path.exists(result.data.screenshot_path)


async def test_extracts_text_from_screenshot(tool: CaptureAndExtract) -> None:
    """Should extract text from the captured screenshot."""
    params = tool.get_params_model()()
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert len(result.data.text) > 0
    assert isinstance(result.data.text, str)


async def test_with_region_parameter(tool: CaptureAndExtract) -> None:
    """Should accept an optional region parameter for partial capture."""
    params = tool.get_params_model()(
        region=CaptureRegion(x=100, y=200, width=400, height=300)
    )
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert os.path.exists(result.data.screenshot_path)
    # Verify the stub wrote region info
    with open(result.data.screenshot_path, encoding="utf-8") as f:
        content = f.read()
    assert "100_200_400x300" in content


async def test_with_language_parameter(tool: CaptureAndExtract) -> None:
    """Should accept a language parameter for OCR."""
    params = tool.get_params_model()(language="fra")
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert len(result.data.text) > 0


async def test_result_has_required_fields(tool: CaptureAndExtract) -> None:
    """Should return result with screenshot_path, text, and confidence."""
    params = tool.get_params_model()()
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert hasattr(result.data, "screenshot_path")
    assert hasattr(result.data, "text")
    assert hasattr(result.data, "confidence")
    assert isinstance(result.data.screenshot_path, str)
    assert isinstance(result.data.text, str)
    assert isinstance(result.data.confidence, float)


async def test_confidence_in_valid_range(tool: CaptureAndExtract) -> None:
    """Should return confidence between 0.0 and 1.0."""
    params = tool.get_params_model()()
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert 0.0 <= result.data.confidence <= 1.0


async def test_default_language_is_eng(tool: CaptureAndExtract) -> None:
    """Should default to English language when not specified."""
    params = tool.get_params_model()()
    assert params.language == "eng"


async def test_default_region_is_none(tool: CaptureAndExtract) -> None:
    """Should default to full screen when region is not specified."""
    params = tool.get_params_model()()
    assert params.region is None
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    with open(result.data.screenshot_path, encoding="utf-8") as f:
        content = f.read()
    assert "full_screen" in content


def test_metadata(tool: CaptureAndExtract) -> None:
    """Should have correct tool metadata."""
    assert tool.name == "screenshot.capture_and_extract"
    assert tool.confirmation_required is False
    assert tool.undo_supported is False
    assert len(tool.description) > 0
