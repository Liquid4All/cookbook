# Plan: Add one-shot CLI mode alongside existing watch mode

## Summary

Expand the CLI in `main.py` to support two modes of operation using Click subcommands:

1. **`watch`** — the existing file-watcher mode (monitors a directory continuously)
2. **`process`** — a new one-shot mode (processes given files/folders and exits)

## Changes

### `src/invoice_parser/main.py`

Restructure the single `@click.command()` into a `@click.group()` with two subcommands.

**New CLI structure:**

```
# Watch mode (existing behavior)
python src/invoice_parser/main.py watch \
    --dir invoices/ \
    --image-model hf.co/LiquidAI/LFM2.5-VL-1.6B-GGUF:BF16

# One-shot process mode (new)
python src/invoice_parser/main.py process \
    --image-model hf.co/LiquidAI/LFM2.5-VL-1.6B-GGUF:BF16 \
    invoices/water_australia.png invoices/british_gas.png

# Process an entire folder
python src/invoice_parser/main.py process \
    --image-model hf.co/LiquidAI/LFM2.5-VL-1.6B-GGUF:BF16 \
    invoices/

# Process and save results to CSV
python src/invoice_parser/main.py process \
    --image-model hf.co/LiquidAI/LFM2.5-VL-1.6B-GGUF:BF16 \
    --output bills.csv \
    invoices/
```

**Implementation details:**

- `@click.group()` replaces the current `@click.command()`
- `watch` subcommand: keeps `--dir`, `--image-model`, and `--process-existing` options — identical to current behavior
- `process` subcommand: takes `--image-model`, optional `--output` (CSV path), plus positional `paths` argument (accepts multiple files and/or directories)
  - For each path: if it's a directory, recursively find image files; if it's a file, process it directly
  - Results are always printed to the console for easy debugging
  - If `--output` is provided, results are also appended to that CSV file
  - Exits after processing all inputs
- Reuse existing `InvoiceProcessor` — no changes to `invoice_processor.py`
- `invoice_file_handler.py`: may need minor refactoring so console printing and CSV writing can be used independently

### `Makefile`

Add a `process` target alongside the existing `run` target:

```makefile
process:
	uv run python src/invoice_parser/main.py process \
		--image-model hf.co/LiquidAI/LFM2.5-VL-1.6B-GGUF:BF16 \
		invoices/
```

Update existing `run` target to use the `watch` subcommand.

### `README.md` and `CLAUDE.md`

Update CLI examples to show both modes.

## Verification

```bash
# Test one-shot mode on sample invoices
make process

# Test watch mode still works
make run
```
