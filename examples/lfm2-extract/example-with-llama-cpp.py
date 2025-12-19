import outlines
from outlines.inputs import Chat
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel, Field, model_validator
from typing import List, Literal, Optional
from llama_cpp import Llama

from prompts import get_system_prompt
from my_types import ExtractionResult

def main(
    model_id: str,
    model_file: str,
    user_prompt: str,
    system_prompt_version: str,
):
    model = outlines.from_llamacpp(
        Llama.from_pretrained(
            repo_id=model_id,
            filename=model_file,
        )
    )

    # Create message
    system_prompt = get_system_prompt(system_prompt_version)

    message = Chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])

    result = model(message, output_type=ExtractionResult)

    print(result)
    print(ExtractionResult.model_validate_json(result))



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model-id", type=str, default="LiquidAI/LFM2-350M-Extract-GGUF")
    parser.add_argument('--model-file', type=str, default="LFM2-350M-Extract-Q4_0.gguf")
    parser.add_argument("--user-prompt", type=str, default=None)
    parser.add_argument("--system-prompt-version", type=str, default="v1")
    args = parser.parse_args()
    main(args.model_id, args.model_file, args.user_prompt, args.system_prompt_version)