import outlines
from PIL import Image
from typing import Union, List

from outlines.inputs import Image as OutlinesImage, Chat
from transformers import AutoModelForImageTextToText, AutoProcessor

from .output_types import CarIdentificationOutputType


def get_structured_model_output(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    system_prompt: str,
    user_prompt: str,
    images: Union[Image.Image, List[Image.Image]],
    max_new_tokens: int | None = 64,
) -> Union[CarIdentificationOutputType, List[CarIdentificationOutputType], None]:
    """
    Gets structured model output for single image or batch of images.
    
    Args:
        model: The model to use for inference
        processor: The processor to use for preprocessing
        system_prompt: System prompt for the conversation
        user_prompt: User prompt for the conversation
        images: Single PIL Image or list of PIL Images
        max_new_tokens: Maximum number of tokens to generate
    
    Returns:
        Single CarIdentificationOutputType or list of CarIdentificationOutputType, or None if error
    """
    outlines_model = outlines.from_transformers(model, processor)

    # Handle both single image and batch of images
    if isinstance(images, Image.Image):
        # Single image case
        prompt = Chat(
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": OutlinesImage(images)},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
        )

        response: str = outlines_model(prompt, CarIdentificationOutputType, max_new_tokens=max_new_tokens)

        try:
            # Parse the response into the structured output type
            parsed_response = CarIdentificationOutputType.model_validate_json(response)
            return parsed_response
        except Exception as e:
            print("Error generating structured output: ", e)
            print("Raw model output: ", response)
            return None
    
    else:
        # Batch case
        prompts = []
        for image in images:
            prompt = Chat(
                [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": OutlinesImage(image)},
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ]
            )
            prompts.append(prompt)

        try:
            # Use batch processing with output type specified
            responses: List[str] = outlines_model.batch(prompts, output_type=CarIdentificationOutputType, max_new_tokens=max_new_tokens)
            
            parsed_responses = []
            for i, response in enumerate(responses):
                try:
                    parsed_response = CarIdentificationOutputType.model_validate_json(response)
                    parsed_responses.append(parsed_response)
                except Exception as e:
                    print(f"Error parsing response {i}: {e}")
                    print(f"Raw model output {i}: {response}")
                    parsed_responses.append(None)
            
            return parsed_responses
        
        except Exception as e:
            print("Error in batch processing: ", e)
            return None


def get_structured_model_output_batch(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    system_prompt: str,
    user_prompt: str,
    images: List[Image.Image],
    max_new_tokens: int | None = 64,
) -> List[CarIdentificationOutputType | None]:
    """
    Dedicated batch processing function for structured model output.
    
    Args:
        model: The model to use for inference
        processor: The processor to use for preprocessing
        system_prompt: System prompt for the conversation
        user_prompt: User prompt for the conversation
        images: List of PIL Images to process
        max_new_tokens: Maximum number of tokens to generate
    
    Returns:
        List of CarIdentificationOutputType or None for each image
    """
    return get_structured_model_output(
        model, processor, system_prompt, user_prompt, images, max_new_tokens
    )

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

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    outputs_wout_input_tokens = outputs[:, inputs["input_ids"].shape[1] :]

    output = processor.batch_decode(
        outputs_wout_input_tokens, skip_special_tokens=True
    )[0]

    return output
