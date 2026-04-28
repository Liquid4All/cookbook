from datasets import ClassLabel, Dataset


class ConversationDataset:
    """Wraps a HuggingFace Dataset and formats samples as conversations lazily.

    Avoids decoding all images upfront: each sample is formatted only when the
    dataloader requests it, keeping memory usage proportional to batch size.
    """

    def __init__(
        self,
        dataset: Dataset,
        image_column: str,
        prompt: str,
        answer_column: str,
    ):
        self._dataset = dataset
        self._image_column = image_column
        self._prompt = prompt
        self._answer_column = answer_column

        answer_feature = dataset.features.get(answer_column)
        self._is_class_label = isinstance(answer_feature, ClassLabel)
        self._answer_feature = answer_feature

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> list[dict]:
        sample = self._dataset[idx]
        raw_answer = sample[self._answer_column]
        answer_text = (
            self._answer_feature.int2str(raw_answer)
            if self._is_class_label
            else raw_answer
        )
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample[self._image_column]},
                    {"type": "text", "text": self._prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer_text}],
            },
        ]


def format_dataset_as_conversation(
    dataset: Dataset,
    image_column: str,
    prompt: str,
    answer_column: str,
) -> ConversationDataset:
    """Returns a lazy view of the dataset formatted as conversations."""
    return ConversationDataset(dataset, image_column, prompt, answer_column)
