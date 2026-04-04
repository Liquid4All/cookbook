"""
Prepares the defect-detection dataset from MMAD and pushes it to HuggingFace Hub.

Usage:
    uv run python -m src.vlm_example.prepare_data --to Paulescu/defect-detection

Prerequisites:
    huggingface-cli login
"""

import argparse
from collections import Counter

import datasets
from datasets import ClassLabel, DatasetDict

SOURCE_DATASET = "jiang-cc/MMAD"
DEFECT_QUESTION = "Is there any defect in the object?"
INPUT_PROMPT = "Is there any defect in the object. Respond Yes or No."

YES_NO = {"Yes", "No"}


def parse_answer(x):
    """Resolve the letter answer to its text using the options field."""
    option_map = {}
    for line in x["options"].split("\n"):
        line = line.strip()
        if line and ": " in line:
            letter, text = line.split(": ", 1)
            option_map[letter] = text.rstrip(".")
    return option_map.get(x["answer"], "")


def get_source(x):
    """Extract the source dataset name from mask (preferred) or template_image.

    DS-MVTec samples reuse MVTec-AD template images, so the mask column must be
    checked first to correctly distinguish DS-MVTec from MVTec-AD.
    """
    mask = x["mask"] or ""
    if mask.startswith("DS-MVTec"):
        return "DS-MVTec"
    return x["template_image"].split("/")[0]


def main():
    parser = argparse.ArgumentParser(description="Prepare defect-detection dataset")
    parser.add_argument(
        "--to",
        required=True,
        help="HuggingFace dataset name to push to, e.g. Paulescu/defect-detection",
    )
    args = parser.parse_args()

    print(f"Loading {SOURCE_DATASET}...")
    ds = datasets.load_dataset(SOURCE_DATASET, split="train")
    print(f"Loaded {len(ds)} rows")

    print(f"Filtering to question: '{DEFECT_QUESTION}'...")
    ds = ds.filter(lambda x: x["question"] == DEFECT_QUESTION)
    print(f"Filtered to {len(ds)} rows")

    print("Mapping to output schema...")
    ds = ds.map(
        lambda x: {
            "query_image": x["query_image"],
            "input_prompt": INPUT_PROMPT,
            "answer": parse_answer(x),
            "source": get_source(x),
        },
        remove_columns=[c for c in ds.column_names if c not in ("query_image",)],
    )

    # Keep only Yes/No rows (drop Maybe/Unknown)
    before = len(ds)
    ds = ds.filter(lambda x: x["answer"] in YES_NO)
    print(f"Kept {len(ds)}/{before} rows with Yes/No answers")

    # Print distributions
    sources = Counter(r["source"] for r in ds)
    print("Source distribution:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count} ({count / len(ds):.1%})")
    yes_count = sum(1 for r in ds if r["answer"] == "Yes")
    no_count = len(ds) - yes_count
    print(f"Answer distribution: Yes={yes_count} ({yes_count / len(ds):.1%}), No={no_count} ({no_count / len(ds):.1%})")

    # Build composite stratification key: source_answer (e.g. "VisA_Yes")
    ds = ds.map(lambda x: {"strat_key": f"{x['source']}_{x['answer']}"})
    strat_names = sorted(set(r["strat_key"] for r in ds))
    ds = ds.cast_column("strat_key", ClassLabel(names=strat_names))

    print("Splitting into train (90%) and test (10%) stratified by source x answer...")
    split = ds.train_test_split(test_size=0.1, seed=42, stratify_by_column="strat_key")

    # Drop temporary stratification column before pushing (keep source)
    for s in ("train", "test"):
        split[s] = split[s].remove_columns(["strat_key"])

    # Cast answer to ClassLabel
    for s in ("train", "test"):
        split[s] = split[s].cast_column("answer", ClassLabel(names=["No", "Yes"]))

    dataset_dict = DatasetDict({"train": split["train"], "test": split["test"]})

    print(f"Train: {len(dataset_dict['train'])} samples")
    print(f"Test:  {len(dataset_dict['test'])} samples")

    print(f"Pushing to {args.to}...")
    dataset_dict.push_to_hub(args.to)
    print(f"Dataset available at https://huggingface.co/datasets/{args.to}")


if __name__ == "__main__":
    main()
