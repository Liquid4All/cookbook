run:
	uv run python src/invoice_parser/main.py \
		--dir invoices/ \
		--image-model hf.co/LiquidAI/LFM2-VL-3B-GGUF:F16 \
		--process-existing

lint-and-format:
	uv run ruff check --fix .
	uv run ruff format .

