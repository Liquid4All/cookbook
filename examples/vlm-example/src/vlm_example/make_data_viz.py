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
import random
import zipfile

import datasets
from huggingface_hub import hf_hub_download

SOURCES = ["DS-MVTec", "GoodsAD", "MVTec-AD", "MVTec-LOCO", "VisA"]
SPLITS = ["train", "test"]
MMAD_DATASET = "jiang-cc/MMAD"
DEFAULT_DATASET = "Paulescu/defect-detection"
DEFAULT_N = 50
DEFAULT_OUT = "data_visualization.html"


def get_zip_path(zip_name: str) -> str:
    """Return local path to zip file, downloading and caching via HF hub if needed."""
    return hf_hub_download(MMAD_DATASET, f"{zip_name}.zip", repo_type="dataset")


def open_zips(zip_names: list[str]) -> dict:
    """Download (if needed) and open zip files, returning {zip_name: (ZipFile, has_prefix)}."""
    result = {}
    for name in zip_names:
        print(f"  Loading {name}.zip from cache...")
        path = get_zip_path(name)
        zf = zipfile.ZipFile(path)
        first = zf.namelist()[0]
        has_prefix = first.startswith(name + "/")
        result[name] = (zf, has_prefix)
    return result


def load_image_from_zip(query_image_path: str, zips: dict) -> bytes:
    zip_name = query_image_path.split("/")[0]
    zf, has_prefix = zips[zip_name]
    file_path = query_image_path if has_prefix else "/".join(query_image_path.split("/")[1:])
    return zf.read(file_path)


def image_bytes_to_base64(data: bytes, max_size: int = 256) -> str:
    from PIL import Image as PILImage
    img = PILImage.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((max_size, max_size), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def build_grids(dataset_name: str, n: int) -> dict:
    """Returns {(source, split): {"total": int, "samples": [(b64_img, answer)]}}"""
    # Load both splits
    all_rows = {}
    for split in SPLITS:
        print(f"Loading {split} split...")
        ds = datasets.load_dataset(dataset_name, split=split)
        all_rows[split] = list(ds)

    # Identify which zips are needed across all sampled rows
    needed_zip_names = set()
    sampled = {}
    for split in SPLITS:
        for source in SOURCES:
            rows = [r for r in all_rows[split] if r["source"] == source]
            sample = random.sample(rows, min(n, len(rows)))
            sampled[(source, split)] = {"total": len(rows), "rows": sample}
            for r in sample:
                needed_zip_names.add(r["query_image"].split("/")[0])

    # Open only the needed zip files (all cached locally)
    print(f"Opening zip files: {sorted(needed_zip_names)}")
    zips = open_zips(sorted(needed_zip_names))

    # Load images
    grids = {}
    for (source, split), data in sampled.items():
        print(f"  Encoding images for {source}/{split}...")
        samples = []
        for r in data["rows"]:
            raw = load_image_from_zip(r["query_image"], zips)
            b64 = image_bytes_to_base64(raw)
            answer = r["answer"]
            if isinstance(answer, int):
                answer = ["No", "Yes"][answer]
            label = "Has defect" if answer == "Yes" else "No defect"
            samples.append((b64, label))
        def decode(r):
            a = r["answer"]
            return a if isinstance(a, str) else ["No", "Yes"][a]

        all_source_rows = [r for r in all_rows[split] if r["source"] == source]
        has_defect = sum(1 for r in all_source_rows if decode(r) == "Yes")
        no_defect = sum(1 for r in all_source_rows if decode(r) == "No")
        grids[(source, split)] = {
            "total": data["total"],
            "has_defect": has_defect,
            "no_defect": no_defect,
            "samples": samples,
        }

    for zf, _ in zips.values():
        zf.close()

    return grids


def render_html(grids: dict, n: int, out_path: str) -> None:
    js_data_parts = []
    for (source, split), data in grids.items():
        cards_js = ", ".join(
            f'{{img: "{b64}", answer: "{answer}"}}'
            for b64, answer in data["samples"]
        )
        js_data_parts.append(
            f'"{source}||{split}": {{total: {data["total"]}, has_defect: {data["has_defect"]}, no_defect: {data["no_defect"]}, samples: [{cards_js}]}}'
        )
    js_data = "{\n" + ",\n".join(js_data_parts) + "\n}"

    sources_js = "[" + ", ".join(f'"{s}"' for s in SOURCES) + "]"
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
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }}
    .card {{ background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
    .card img {{ width: 100%; display: block; aspect-ratio: 1; object-fit: cover; }}
    .badge {{ padding: 6px 10px; font-weight: 700; font-size: 0.9rem; text-align: center; }}
    .yes {{ background: #f8d7da; color: #721c24; }}
    .no  {{ background: #d4edda; color: #155724; }}
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
          <img src="data:image/jpeg;base64,${{s.img}}" alt="${{s.answer}}">
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
    grids = build_grids(args.dataset, args.n)
    render_html(grids, args.n, args.out)


if __name__ == "__main__":
    main()
