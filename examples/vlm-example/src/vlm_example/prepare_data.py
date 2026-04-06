"""
Prepares the defect-detection dataset from MMAD and pushes it to HuggingFace Hub.

Usage:
    uv run python -m src.vlm_example.prepare_data --to Paulescu/defect-detection

Prerequisites:
    huggingface-cli login
"""

import argparse
import functools
import time
import zipfile
from collections import Counter

import datasets
from datasets import ClassLabel, DatasetDict, Image as HFImage
from huggingface_hub import hf_hub_download

SOURCE_DATASET = "jiang-cc/MMAD"
DEFECT_QUESTION = "Is there any defect in the object?"
INPUT_PROMPT = "Is there any defect in the object. Respond Yes or No."

YES_NO = {"Yes", "No"}

SOURCE_TO_ZIP = {
    "VisA": "VisA.zip",
    "MVTec-AD": "MVTec-AD.zip",
    "DS-MVTec": "DS-MVTec.zip",
    "MVTec-LOCO": "MVTec-LOCO.zip",
    "GoodsAD": "GoodsAD.zip",
}


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


def download_zip_files(sources: set[str]) -> dict[str, str]:
    """Download the MMAD zip files for the given sources and return local file paths."""
    zip_paths = {}
    for source in sorted(sources):
        zip_name = SOURCE_TO_ZIP.get(source)
        if zip_name is None:
            print(f"Warning: no zip file mapped for source '{source}', skipping")
            continue
        print(f"  Downloading {zip_name} from {SOURCE_DATASET}...")
        local_path = hf_hub_download(
            repo_id=SOURCE_DATASET,
            filename=zip_name,
            repo_type="dataset",
        )
        zip_paths[source] = local_path
        print(f"  Downloaded {zip_name}")
    return zip_paths


def make_row(x, zip_paths: dict[str, str]) -> dict:
    """Load image bytes from the appropriate zip and return the new columns."""
    source = get_source(x)
    # Use the query_image path prefix for zip lookup: some DS-MVTec rows have
    # mask=None so get_source() falls back to template_image (MVTec-AD), but
    # their query_image still lives in DS-MVTec.zip.
    zip_source = x["query_image"].split("/")[0]
    with zipfile.ZipFile(zip_paths[zip_source]) as zf:
        with zf.open(x["query_image"]) as f:
            image_bytes = f.read()
        mask_bytes = None
        if x["mask"]:
            mask_zip_source = x["mask"].split("/")[0]
            mask_zip_path = zip_paths.get(mask_zip_source)
            if mask_zip_path:
                # GoodsAD mask paths in MMAD include a spurious 'test/' segment
                # that is absent from the zip: strip it before lookup.
                mask_path_in_zip = x["mask"].replace("/test/ground_truth/", "/ground_truth/")
                with zipfile.ZipFile(mask_zip_path) as mzf:
                    try:
                        with mzf.open(mask_path_in_zip) as f:
                            mask_bytes = f.read()
                    except KeyError:
                        pass
    return {
        "query_image": image_bytes,
        "mask_image": mask_bytes,
        "input_prompt": INPUT_PROMPT,
        "answer": parse_answer(x),
        "source": source,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare defect-detection dataset")
    parser.add_argument(
        "--to",
        required=True,
        help="HuggingFace dataset name to push to, e.g. Paulescu/defect-detection",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Limit to this many rows from MMAD before filtering (for smoke tests).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Comma-separated list of sources to keep, e.g. 'GoodsAD' or 'GoodsAD,VisA'. Filters before downloading zips.",
    )
    args = parser.parse_args()
    sources_filter = set(args.source.split(",")) if args.source else None

    t0 = time.time()

    def elapsed() -> str:
        return f"[{time.time() - t0:.1f}s]"

    print(f"[1/7] Loading {SOURCE_DATASET}...")
    ds = datasets.load_dataset(SOURCE_DATASET, split="train")
    print(f"      Loaded {len(ds)} rows {elapsed()}")

    if args.samples is not None:
        ds = ds.select(range(args.samples))
        print(f"      Limiting to {args.samples} samples for smoke test")

    print(f"[2/7] Filtering to question: '{DEFECT_QUESTION}'...")
    ds = ds.filter(lambda x: x["question"] == DEFECT_QUESTION)
    print(f"      Filtered to {len(ds)} rows {elapsed()}")

    if sources_filter is not None:
        print(f"      Filtering to sources: {sorted(sources_filter)}...")
        ds = ds.filter(lambda x: get_source(x) in sources_filter)
        print(f"      Filtered to {len(ds)} rows {elapsed()}")

    # Keep only Yes/No rows (drop Maybe/Unknown) — do this before loading images
    # to avoid OOM when filtering a large image dataset.
    before = len(ds)
    ds = ds.filter(lambda x: parse_answer(x) in YES_NO)
    print(f"[2b/7] Kept {len(ds)}/{before} rows with Yes/No answers {elapsed()}")

    # Collect zip sources from query_image prefixes (not get_source) so that
    # DS-MVTec rows with mask=None (which get_source maps to MVTec-AD) still
    # pull in DS-MVTec.zip.
    zip_sources = set(ds.map(lambda x: {"s": x["query_image"].split("/")[0]})["s"])
    print(f"      Sources present: {sorted(zip_sources)}")

    print(f"[3/7] Downloading zip files...")
    zip_paths = download_zip_files(zip_sources)
    print(f"      Downloaded {len(zip_paths)} zip files {elapsed()}")

    print(f"[4/7] Loading images from zip files via ds.map()...")
    # Explicitly declare output features so PyArrow doesn't infer mask_image
    # as null-type when an entire batch has no masks (e.g. MVTec-AD).
    map_features = datasets.Features({
        "query_image": datasets.Value("binary"),
        "mask_image": datasets.Value("binary"),
        "input_prompt": datasets.Value("string"),
        "answer": datasets.Value("string"),
        "source": datasets.Value("string"),
    })
    ds = ds.map(
        functools.partial(make_row, zip_paths=zip_paths),
        remove_columns=ds.column_names,
        features=map_features,
        desc="Loading images",
    )
    ds = ds.cast_column("query_image", HFImage())
    ds = ds.cast_column("mask_image", HFImage())
    print(f"      Built dataset with {len(ds)} rows {elapsed()}")

    # Print distributions
    sources = Counter(ds["source"])
    print("      Source distribution:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"        {src}: {count} ({count / len(ds):.1%})")
    answers = Counter(ds["answer"])
    yes_count = answers["Yes"]
    no_count = answers["No"]
    print(f"      Answer distribution: Yes={yes_count} ({yes_count / len(ds):.1%}), No={no_count} ({no_count / len(ds):.1%})")

    print(f"[5/7] Building stratification key (source x answer)...")
    # Work on a metadata-only dataset (no images) to avoid PyArrow offset overflow
    # when batching large binary columns.
    meta = datasets.Dataset.from_dict({
        "idx": list(range(len(ds))),
        "source": ds["source"],
        "answer": ds["answer"],
    })
    meta = meta.map(lambda x: {"strat_key": f"{x['source']}_{x['answer']}"}, desc="Building strat keys")
    strat_names = sorted(set(meta["strat_key"]))
    meta = meta.cast_column("strat_key", ClassLabel(names=strat_names))
    print(f"      Done {elapsed()}")

    print(f"[6/7] Splitting into train (90%) and test (10%) stratified by source x answer...")
    meta_split = meta.train_test_split(test_size=0.1, seed=42, stratify_by_column="strat_key")

    # Use split indices to select from the full image dataset
    train_ds = ds.select(meta_split["train"]["idx"])
    test_ds = ds.select(meta_split["test"]["idx"])

    # Cast answer to ClassLabel using writer_batch_size=1 to avoid PyArrow
    # offset overflow when batching rows that contain large image bytes.
    answer_features = train_ds.features.copy()
    answer_features["answer"] = ClassLabel(names=["No", "Yes"])
    train_ds = train_ds.map(lambda x: x, features=answer_features, writer_batch_size=1, desc="Casting train labels")
    test_ds = test_ds.map(lambda x: x, features=answer_features, writer_batch_size=1, desc="Casting test labels")

    dataset_dict = DatasetDict({"train": train_ds, "test": test_ds})

    print(f"      Train: {len(dataset_dict['train'])} samples")
    print(f"      Test:  {len(dataset_dict['test'])} samples")
    print(f"      Split done {elapsed()}")

    local_path = "local_dataset"
    print(f"[7/7] Saving to disk at {local_path}/ and pushing to {args.to}...")
    dataset_dict.save_to_disk(local_path)
    print(f"      Saved to disk {elapsed()}")
    dataset_dict.push_to_hub(args.to)
    print(f"      Push done {elapsed()}")
    print(f"Dataset available at https://huggingface.co/datasets/{args.to}")
    print(f"DONE. Total time: {elapsed()}")


if __name__ == "__main__":
    main()
