import fire
import re


def detect_language(text: str) -> str:
    """
    Detect if text is Korean or English and return appropriate translation direction.
    Returns 'korean_to_english' if text contains Korean, 'english_to_korean' otherwise.
    """
    # Check for Korean characters (Hangul syllables, Jamo, and compatibility Jamo)
    korean_pattern = re.compile(r'[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]')
    
    if korean_pattern.search(text):
        return "korean_to_english"
    else:
        return "english_to_korean"


def main(
    text: str,
    model_name: str = "gyung/lfm2-1.2b-koen-mt-v6.4-merged",
    adapter_name: str = "gyung/lfm2-1.2b-koen-mt-v8-rl-10k-adapter",
    max_new_tokens: int = 256,
    temperature: float = 0.3,
    min_p: float = 0.15,
    repetition_penalty: float = 1.05
):
    print("Hello from lfm2-english-to-korean!")
    
    # Auto-detect translation direction
    direction = detect_language(text)
    print(f"Detected direction: {direction}")

    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
    from peft import PeftModel

    # Base 모델 로드
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    print("Base model loaded.")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded.")

    # Adapter 로드 및 병합
    print("Loading and merging adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_name)
    model = model.merge_and_unload()  # 추론 속도 향상
    print("Adapter loaded and merged.")

    print('Generating translation...')
    
    # Set system message based on detected direction
    if direction == "korean_to_english":
        system_message = "Translate the following text to English."
    else:  # english_to_korean
        system_message = "Translate the following text to Korean."
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": text}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print("Starting generation...")
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
    )
    print("\nTranslation complete.")

if __name__ == "__main__":
    fire.Fire(main)
