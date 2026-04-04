"""Custom callbacks for training."""

import os
from transformers import TrainerCallback


class ProcessorSaveCallback(TrainerCallback):
    """Callback to save the processor alongside model checkpoints."""

    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        print(f"Saving processor to {checkpoint_dir}")
        self.processor.save_pretrained(checkpoint_dir)
        print(f"Processor saved to: {checkpoint_dir}")
