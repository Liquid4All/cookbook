import base64
import json

import anthropic

MODEL = "claude-opus-4-6"
MAX_TOKENS = 256

SYSTEM_PROMPT = """\
You are a remote sensing analyst specialising in wildfire risk assessment.
You will be given two Sentinel-2 satellite images of the same land tile:
  1. RGB composite (bands B4-B3-B2): natural colour, useful for terrain, \
infrastructure, and land cover.
  2. SWIR composite (bands B12-B8-B4): highlights vegetation moisture stress \
and dryness. Healthy vegetation appears green/cyan, stressed or dry vegetation \
appears orange/red, bare soil appears magenta/pink, and burned areas appear \
dark red or black.

Assess the wildfire risk of the tile and return ONLY a valid JSON object — \
no markdown, no explanation outside the JSON — with exactly these fields:

{
  "risk_level": "low | medium | high",
  "dry_vegetation_present": true | false,
  "urban_interface": true | false,
  "steep_terrain": true | false,
  "water_body_present": true | false,
  "image_quality_limited": true | false
}

Field definitions:
- risk_level: overall wildfire risk for the tile, using these criteria:
    - low: no dry vegetation, or landscape is predominantly wet/green/bare rock
    - medium: some dry vegetation present but fuel continuity is broken by \
bare soil, water bodies, or green vegetation
    - high: extensive dry vegetation with continuous fuel load and at least \
one aggravating factor (steep terrain or urban interface)
- dry_vegetation_present: dry grass, shrubland, cropland stubble, or any \
vegetation showing low moisture (orange/red in SWIR).
- urban_interface: buildings, roads, or infrastructure adjacent to or within \
dry vegetation.
- steep_terrain: visible ridges, slopes, or canyons that would accelerate \
fire spread.
- water_body_present: river, reservoir, or lake that acts as a natural \
firebreak.
- image_quality_limited: cloud, snow, or no-data obscures a significant \
portion of the tile.
"""

USER_TEXT = (
    "Image 1 is the RGB composite. Image 2 is the SWIR composite. "
    "Return the wildfire risk JSON for this tile."
)


def _encode(image_bytes: bytes) -> str:
    return base64.standard_b64encode(image_bytes).decode()


def annotate(rgb_bytes: bytes, swir_bytes: bytes, model: str = MODEL) -> dict[str, object]:
    """Call an Anthropic model with both images and return the parsed annotation dict.

    Args:
        rgb_bytes: Raw PNG bytes of the RGB composite.
        swir_bytes: Raw PNG bytes of the SWIR composite.
        model: Anthropic model ID to use for annotation.

    Returns:
        Parsed JSON dict matching the wildfire risk schema.

    Raises:
        ValueError: if the model response cannot be parsed as JSON.
    """
    client = anthropic.Anthropic()

    message = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": _encode(rgb_bytes),
                        },
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": _encode(swir_bytes),
                        },
                    },
                    {"type": "text", "text": USER_TEXT},
                ],
            }
        ],
    )

    raw = message.content[0].text.strip()
    # Strip markdown code fences if the model wraps the JSON.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]  # drop the opening ```json line
        raw = raw.rsplit("```", 1)[0].strip()
    try:
        return json.loads(raw)  # type: ignore[no-any-return]
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned non-JSON response:\n{raw}") from exc
