"""Tests for screenshot.extract_ui_elements tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.extract_ui_elements import ExtractUIElements


@pytest.fixture()
def tool() -> ExtractUIElements:
    """Create an ExtractUIElements tool instance."""
    return ExtractUIElements()


async def test_extracts_elements_from_image(
    tool: ExtractUIElements, sample_image: Path
) -> None:
    """Should extract UI elements from a valid image path."""
    params = tool.get_params_model()(image_path=str(sample_image))
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert len(result.data.elements) > 0


async def test_elements_have_type_text_bounds_confidence(
    tool: ExtractUIElements, sample_image: Path
) -> None:
    """Each element should have type, text, bounds, and confidence fields."""
    params = tool.get_params_model()(image_path=str(sample_image))
    result = await tool.execute(params)

    assert result.data is not None
    for element in result.data.elements:
        assert isinstance(element.type, str)
        assert len(element.type) > 0
        assert isinstance(element.text, str)
        assert element.bounds is not None
        assert isinstance(element.confidence, float)


async def test_nonexistent_image_raises_error(
    tool: ExtractUIElements, tmp_dir: Path
) -> None:
    """Should raise MCPError for a non-existent image path."""
    from mcp_base import MCPError

    params = tool.get_params_model()(image_path=str(tmp_dir / "nonexistent.png"))
    with pytest.raises(MCPError, match="Image file not found"):
        await tool.execute(params)


async def test_bounds_have_x_y_width_height(
    tool: ExtractUIElements, sample_image: Path
) -> None:
    """Each element's bounds should have x, y, width, height."""
    params = tool.get_params_model()(image_path=str(sample_image))
    result = await tool.execute(params)

    assert result.data is not None
    for element in result.data.elements:
        bounds = element.bounds
        assert isinstance(bounds.x, int)
        assert isinstance(bounds.y, int)
        assert isinstance(bounds.width, int)
        assert isinstance(bounds.height, int)
        assert bounds.width > 0
        assert bounds.height > 0


async def test_element_types_are_valid(
    tool: ExtractUIElements, sample_image: Path
) -> None:
    """Detected element types should be from the known set."""
    valid_types = {"button", "text_field", "label", "link", "icon", "checkbox"}

    params = tool.get_params_model()(image_path=str(sample_image))
    result = await tool.execute(params)

    assert result.data is not None
    for element in result.data.elements:
        assert element.type in valid_types, f"Unknown element type: {element.type}"


async def test_confidence_in_valid_range(
    tool: ExtractUIElements, sample_image: Path
) -> None:
    """Confidence scores should be between 0.0 and 1.0."""
    params = tool.get_params_model()(image_path=str(sample_image))
    result = await tool.execute(params)

    assert result.data is not None
    for element in result.data.elements:
        assert 0.0 <= element.confidence <= 1.0


async def test_relative_path_raises_error(tool: ExtractUIElements) -> None:
    """Should raise MCPError for relative image paths."""
    from mcp_base import MCPError

    params = tool.get_params_model()(image_path="relative/path/image.png")
    with pytest.raises(MCPError, match="must be an absolute path"):
        await tool.execute(params)


def test_metadata(tool: ExtractUIElements) -> None:
    """Should have correct tool metadata."""
    assert tool.name == "screenshot.extract_ui_elements"
    assert tool.confirmation_required is False
    assert tool.undo_supported is False
    assert len(tool.description) > 0
