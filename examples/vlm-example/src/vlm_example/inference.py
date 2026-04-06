import base64
import io
from typing import Union

from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def get_model_output(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    conversation: list[dict],
    max_new_tokens: int = 5,
) -> str:
    """Runs standard (unconstrained) generation and returns the raw output string."""
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.1,
        min_p=0.15,
        repetition_penalty=1.05,
    )
    outputs_wout_input = outputs[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(outputs_wout_input, skip_special_tokens=True)[0]


def get_claude_output(client, model: str, user_prompt: str, image: Image.Image) -> str:
    """Calls the Claude API with a vision prompt and returns the raw text response."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    image_b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    message = client.messages.create(
        model=model,
        max_tokens=10,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ],
    )
    return message.content[0].text


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
