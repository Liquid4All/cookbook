from datasets import Dataset


def format_dataset_as_conversation(
    dataset: Dataset,
    image_column: str,
    prompt_column: str,
    answer_column: str,
) -> list[list[dict]]:
    """Formats a dataset into a conversation format suitable for SFT training."""

    def format_sample(sample):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample[image_column]},
                    {"type": "text", "text": sample[prompt_column]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample[answer_column]}],
            },
        ]

    return [format_sample(s) for s in dataset]
