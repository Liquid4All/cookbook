from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import get_system_prompt

def main(
    model_id: str,
    user_prompt: str,
    sytem_prompt_version: str,
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

    # Create message
    system_prompt = get_system_prompt(sytem_prompt_version)

    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Generate answer
    input_ids = tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to(model.device)

    output = model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=1024,
    )

    print(tokenizer.decode(output[0], skip_special_tokens=False))

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model-id", type=str, default="LiquidAI/LFM2-350M-Extract")
    parser.add_argument("--user-prompt", type=str, default="I have diabetes and take metformin 500 mg twice a day")
    parser.add_argument("--system-prompt-version", type=str, default="v1")
    args = parser.parse_args()
    main(
        args.model_id,
        args.user_prompt,
        args.system_prompt_version
    )