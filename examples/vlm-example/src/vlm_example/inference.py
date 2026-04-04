import io
import urllib.request
from typing import Union

import outlines
from outlines.inputs import Chat
from outlines.inputs import Image as OutlinesImage
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .output_types import DefectDetectionOutput


def _to_pil_image(image: Union[Image.Image, str]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if image.startswith("http"):
        with urllib.request.urlopen(image) as response:
            return Image.open(io.BytesIO(response.read())).convert("RGB")
    return Image.open(image).convert("RGB")


def get_structured_model_output(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    user_prompt: str,
    image: Union[Image.Image, str],
    max_new_tokens: int = 10,
) -> DefectDetectionOutput | None:
    """
    Runs constrained generation forcing the output to be a valid DefectDetectionOutput JSON.
    Returns a parsed DefectDetectionOutput, or None on error.
    """
    outlines_model = outlines.from_transformers(model, processor)

    pil_image = _to_pil_image(image)
    prompt = Chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": OutlinesImage(pil_image)},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
    )

    try:
        response: str = outlines_model(
            prompt, DefectDetectionOutput, max_new_tokens=max_new_tokens
        )
        return DefectDetectionOutput.model_validate_json(response)
    except Exception as e:
        print(f"Error in structured generation: {e}")
        return None


def get_model_output(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    conversation: list[dict],
    max_new_tokens: int = 64,
) -> str:
    """Runs standard (unconstrained) generation and returns the raw output string."""
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    outputs_wout_input = outputs[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(outputs_wout_input, skip_special_tokens=True)[0]


def parse_yes_no(raw_output: str) -> str:
    """
    Extracts 'Yes' or 'No' from a raw model output string.
    Returns the raw output unchanged if neither is found (will count as incorrect).
    """
    text = raw_output.strip()
    first_word = text.split()[0].rstrip(".,!?") if text else ""

    if first_word.lower() == "yes":
        return "Yes"
    if first_word.lower() == "no":
        return "No"

    # Fallback: scan the full output
    lower = text.lower()
    if "yes" in lower:
        return "Yes"
    if "no" in lower:
        return "No"

    return raw_output  # unrecognised: will be marked incorrect
