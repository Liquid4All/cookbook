import outlines
from outlines.inputs import Chat
from transformers import AutoModelForCausalLM, AutoTokenizer
# from pydantic import BaseModel, Field, model_validator
# from typing import List, Literal, Optional

from prompts import get_system_prompt
from my_types import ExtractionResult

def main(
    model_id: str,
    user_prompt: str,
    system_prompt_version: str,
):
    print("Loading model...")
    # Load model and tokenizer
    # model_id = "LiquidAI/LFM2-350M-Extract"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype="bfloat16",
    #    attn_implementation="flash_attention_2" <- uncomment on compatible GPU
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Wrapping the model with Outlines for structured generation
    model = outlines.from_transformers(model, tokenizer)

    # Create message
    system_prompt = get_system_prompt(system_prompt_version)

    message = Chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])

    result = model(message, output_type=ExtractionResult, max_new_tokens=200, repetition_penalty=0.5)

    # breakpoint()

    print(result)
    print(ExtractionResult.model_validate_json(result))


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model-id", type=str, default="LiquidAI/LFM2-350M-Extract")
    parser.add_argument("--user-prompt", type=str, default=None)
    # parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--system-prompt-version", type=str, default="v1")
    args = parser.parse_args()
    main(args.model_id, args.user_prompt, args.system_prompt_version)