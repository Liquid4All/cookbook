"""
Generates a self-contained data_visualization.html to inspect the
defect-detection dataset by source and split.

Usage:
    uv run python -m src.vlm_example.make_data_viz
    uv run python -m src.vlm_example.make_data_viz --dataset Paulescu/defect-detection --n 50 --out data_visualization.html
"""

import argparse
import base64
import io
import os
import random

import datasets

SPLITS = ["train", "test"]
DEFAULT_DATASET = "Paulescu/defect-detection"
DEFAULT_N = 50
DEFAULT_OUT = "data_visualization.html"


def pil_image_to_base64(img, max_size: int = 256) -> str:
    from PIL import Image as PILImage
    img = img.convert("RGB")
    img.thumbnail((max_size, max_size), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def decode_answer(answer) -> str:
    """Resolve ClassLabel int or string answer to 'Yes'/'No'."""
    return answer if isinstance(answer, str) else ["No", "Yes"][answer]


def build_grids(dataset_name: str, n: int) -> tuple[dict, list[str]]:
    """Returns (grids, sources) where grids is {(source, split): {...}}."""
    grids = {}
    all_sources: set[str] = set()

    local_path = "local_dataset"
    use_local = os.path.isdir(local_path)
    if use_local:
        print(f"Loading from local disk at {local_path}/...")
        dataset_dict = datasets.load_from_disk(local_path)
    for split in SPLITS:
        if use_local:
            ds = dataset_dict[split]
            print(f"  {split}: {len(ds)} rows (local)")
        else:
            print(f"Loading {split} split from {dataset_name}...")
            ds = datasets.load_dataset(dataset_name, split=split)
            print(f"  {len(ds)} rows")

        # Read metadata columns only (fast, no image decoding)
        sources_col = ds["source"]
        answers_col = [decode_answer(a) for a in ds["answer"]]
        split_sources = sorted(set(sources_col))
        all_sources.update(split_sources)

        for source in split_sources:
            indices = [i for i, s in enumerate(sources_col) if s == source]
            sample_indices = random.sample(indices, min(n, len(indices)))

            has_defect = sum(1 for i in indices if answers_col[i] == "Yes")
            no_defect = sum(1 for i in indices if answers_col[i] == "No")

            # Load images only for sampled rows
            print(f"  Encoding {len(sample_indices)} images for {source}/{split}...")
            sampled_ds = ds.select(sample_indices)
            samples = []
            for r in sampled_ds:
                b64 = pil_image_to_base64(r["query_image"])
                mask_b64 = pil_image_to_base64(r["mask_image"]) if r.get("mask_image") is not None else None
                label = "Has defect" if decode_answer(r["answer"]) == "Yes" else "No defect"
                samples.append((b64, mask_b64, label))

            grids[(source, split)] = {
                "total": len(indices),
                "has_defect": has_defect,
                "no_defect": no_defect,
                "samples": samples,
            }

    sources = sorted(all_sources)
    print(f"Sources found: {sources}")
    return grids, sources


def render_html(grids: dict, sources: list[str], n: int, out_path: str) -> None:
    js_data_parts = []
    for (source, split), data in grids.items():
        cards_js = ", ".join(
            '{img: "' + b64 + '", mask: ' + ('"' + mask_b64 + '"' if mask_b64 else "null") + ', answer: "' + answer + '"}'
            for b64, mask_b64, answer in data["samples"]
        )
        js_data_parts.append(
            f'"{source}||{split}": {{total: {data["total"]}, has_defect: {data["has_defect"]}, no_defect: {data["no_defect"]}, samples: [{cards_js}]}}'
        )
    js_data = "{\n" + ",\n".join(js_data_parts) + "\n}"

    sources_js = "[" + ", ".join(f'"{s}"' for s in sources) + "]"
    splits_js = "[" + ", ".join(f'"{s}"' for s in SPLITS) + "]"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Defect Detection Dataset Visualization</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 0; padding: 24px; background: #f5f5f5; color: #222; }}
    h1 {{ margin: 0 0 20px; font-size: 1.4rem; }}
    .controls {{ display: flex; gap: 16px; align-items: center; margin-bottom: 20px; flex-wrap: wrap; }}
    .controls label {{ font-weight: 600; margin-right: 6px; }}
    select {{ padding: 6px 10px; font-size: 1rem; border-radius: 6px; border: 1px solid #ccc; background: #fff; cursor: pointer; }}
    .meta {{ margin-bottom: 16px; color: #555; font-size: 0.9rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; }}
    .card {{ background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
    .card-images {{ display: flex; }}
    .card-images img {{ width: 50%; display: block; aspect-ratio: 1; object-fit: cover; }}
    .card-images .placeholder {{ width: 50%; aspect-ratio: 1; background: #e9ecef; display: flex; align-items: center; justify-content: center; color: #aaa; font-size: 0.75rem; }}
    .badge {{ padding: 6px 10px; font-weight: 700; font-size: 0.9rem; text-align: center; }}
    .yes {{ background: #f8d7da; color: #721c24; }}
    .no  {{ background: #d4edda; color: #155724; }}
    .img-labels {{ display: flex; font-size: 0.7rem; color: #888; }}
    .img-labels span {{ width: 50%; text-align: center; padding: 2px 0; background: #f8f9fa; }}
    .chart {{ margin-bottom: 20px; width: 360px; }}
    .bar-row {{ display: flex; align-items: center; margin-bottom: 8px; gap: 8px; font-size: 0.85rem; }}
    .bar-label {{ width: 90px; text-align: right; font-weight: 600; flex-shrink: 0; }}
    .bar-track {{ flex: 1; background: #e9ecef; border-radius: 4px; height: 22px; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
    .bar-count {{ width: 40px; flex-shrink: 0; font-size: 0.8rem; color: #555; }}
  </style>
</head>
<body>
  <h1>Defect Detection Dataset Visualization</h1>
  <div class="controls">
    <div>
      <label for="source-select">Source:</label>
      <select id="source-select"></select>
    </div>
    <div>
      <label for="split-select">Split:</label>
      <select id="split-select"></select>
    </div>
  </div>
  <div class="chart" id="chart"></div>
  <div class="meta" id="meta"></div>
  <div class="grid" id="grid"></div>

  <script>
    const DATA = {js_data};
    const SOURCES = {sources_js};
    const SPLITS = {splits_js};
    const N = {n};

    const sourceEl = document.getElementById("source-select");
    const splitEl  = document.getElementById("split-select");
    const gridEl   = document.getElementById("grid");
    const metaEl   = document.getElementById("meta");
    const chartEl  = document.getElementById("chart");

    SOURCES.forEach(s => {{ const o = document.createElement("option"); o.value = s; o.textContent = s; sourceEl.appendChild(o); }});
    SPLITS.forEach(s  => {{ const o = document.createElement("option"); o.value = s; o.textContent = s; splitEl.appendChild(o); }});

    function render() {{
      const key = sourceEl.value + "||" + splitEl.value;
      const entry = DATA[key];
      if (!entry) {{ gridEl.innerHTML = "<p>No data.</p>"; return; }}
      metaEl.textContent = `Showing ${{entry.samples.length}} of ${{entry.total}} ${{splitEl.value}} samples for ${{sourceEl.value}}`;
      const max = Math.max(entry.has_defect, entry.no_defect);
      chartEl.innerHTML = `
        <div class="bar-row">
          <div class="bar-label" style="color:#721c24">Has defect</div>
          <div class="bar-track"><div class="bar-fill" style="width:${{(entry.has_defect/max*100).toFixed(1)}}%;background:#f5c6cb"></div></div>
          <div class="bar-count">${{entry.has_defect}}</div>
        </div>
        <div class="bar-row">
          <div class="bar-label" style="color:#155724">No defect</div>
          <div class="bar-track"><div class="bar-fill" style="width:${{(entry.no_defect/max*100).toFixed(1)}}%;background:#c3e6cb"></div></div>
          <div class="bar-count">${{entry.no_defect}}</div>
        </div>`;
      gridEl.innerHTML = entry.samples.map(s => `
        <div class="card">
          <div class="card-images">
            <img src="data:image/jpeg;base64,${{s.img}}" alt="query image">
            ${{s.mask
              ? `<img src="data:image/jpeg;base64,${{s.mask}}" alt="defect mask">`
              : `<div class="placeholder">no mask</div>`
            }}
          </div>
          <div class="img-labels"><span>Image</span><span>Mask</span></div>
          <div class="badge ${{s.answer === "Has defect" ? "yes" : "no"}}">${{s.answer}}</div>
        </div>`).join("");
    }}

    sourceEl.addEventListener("change", render);
    splitEl.addEventListener("change", render);
    render();
  </script>
</body>
</html>"""

    with open(out_path, "w") as f:
        f.write(html)
    print(f"Written to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    random.seed(42)
    grids, sources = build_grids(args.dataset, args.n)
    render_html(grids, sources, args.n, args.out)


if __name__ == "__main__":
    main()
