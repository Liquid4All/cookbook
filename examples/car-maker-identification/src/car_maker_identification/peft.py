from peft import LoraConfig, get_peft_model


def prepare_peft_model(
    model,
    lora_r: int = 16,
    lora_alpha: int = 32,
    target_modules: list[str] = [
        "q_proj",
        "v_proj",
        "fc1",
        "fc2",
        "linear",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
):
    """Prepare the model for PEFT training using LoRA."""
    print("Preparing the model for PEFT training using LoRA...")
    target_modules = [
        "q_proj",
        "v_proj",
        "fc1",
        "fc2",
        "linear",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
