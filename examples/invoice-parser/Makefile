run:
	uv run python src/invoice_parser/main.py watch \
		--dir invoices/ \
		--image-model LiquidAI/LFM2.5-VL-1.6B-GGUF:Q8_0 \
		--process-existing

process:
	uv run python src/invoice_parser/main.py process \
		--image-model LiquidAI/LFM2.5-VL-1.6B-GGUF:Q8_0 \
		invoices/

lint-and-format:
	uv run ruff check --fix .
	uv run ruff format .
