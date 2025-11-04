import re

import outlines
from PIL import Image

# from outlines.inputs import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .output_types import ModelOutputType


def get_structured_model_output(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    system_prompt: str,
    user_prompt: str,
    image: Image,
    output_schema: type[ModelOutputType],
    max_new_tokens: int | None = 64,
) -> ModelOutputType | None:
    """ """
    from outlines.inputs import Image, Chat

    outlines_model = outlines.from_transformers(model, processor)

    prompt = Chat([
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image(image)},
                {"type": "text", "text": user_prompt}
            ],
        }
        # {
        #     "role": "user",
        #     "content": [user_prompt, Image(image)]
        # },
    ])

    # from .output_types import CarIdentificationOutputType as output_schema
    response: str = outlines_model(
        prompt,
        output_schema,
        max_new_tokens=max_new_tokens
    )

    try:
        # Parse the response into the structured output type
        response = output_schema.model_validate_json(response)
        return response
    except Exception as e:
        print("Error generating structured output: ", e)
        print("Raw model output: ", response)
        return None
    

def get_structured_model_output_old(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    system_prompt: str,
    user_prompt: str,
    image: Image,
    output_schema: type[ModelOutputType],
    max_new_tokens: int | None = 64,
) -> ModelOutputType | None:
    """ """
    
    from .utils import print_installed_packages
    print_installed_packages()

    model = outlines.from_transformers(model, processor)

    output_generator = outlines.Generator(model, output_schema)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image", "image": ""},
            ],
        },
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    from outlines import Image
    response: str = output_generator({"text": prompt, "images": Image(image)})

    try:
        # Parse the response into the structured output type
        response = output_schema.model_validate_json(response)
        return response
    except Exception as e:
        print("Error generating structured output: ", e)
        print("Raw model output: ", response)
        return None


def get_model_output(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    conversation: list[dict],
    max_new_tokens: int | None = 64,
) -> str:
    """
    Gets the model output for a given conversation
    """
    # Generate Answer
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    ).to(model.device)

    # print('type(inputs)', type(inputs))
    # print('dir(inputs)', dir(inputs))

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    outputs_wout_input_tokens = outputs[:, inputs["input_ids"].shape[1] :]

    output = processor.batch_decode(
        outputs_wout_input_tokens, skip_special_tokens=True
    )[0]

    # # Find first complete JSON object
    # match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', output)
    # if match:
    #     output = match.group()

    return output
