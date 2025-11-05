from typing import Any

from datasets import Dataset


def split_dataset(
    dataset: Dataset,
    test_size: float = 0.1,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Splits a dataset into training and testing sets."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]


def format_dataset_as_conversation(
    dataset: Dataset,
    system_prompt: str,
    user_prompt: str,
    image_column: str,
    label_column: str,
    label_mapping: dict[Any, str],
) -> Dataset:
    """Formats a dataset into a conversation format suitable for SFT training."""

    def format_sample(sample):
        # Format the label as JSON according to CatsVsDogsClassificationOutputType
        # label_json = CatsVsDogsClassificationOutputType.from_pred_class(label_mapping[sample[label_column]])
        if label_mapping is None:
            label_json = sample[label_column]
        else:
            label_json = label_mapping[sample[label_column]]

        # print(f'Image type: {type(sample[image_column]).__name__}')

        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample[image_column]},
                    # {"type": "text", "text": sample["question"]},
                    {"type": "text", "text": user_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": label_json}],
            },
        ]

    dataset = [format_sample(s) for s in dataset]

    return dataset
