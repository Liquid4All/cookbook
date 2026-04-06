prepare-data:
	uv run python -m src.vlm_example.prepare_data --to $(to) $(if $(samples),--samples $(samples),) $(if $(source),--source $(source),)

benchmark:
	uv run modal run -m src.vlm_example.benchmark --config-file-name $(config)

benchmark-api:
	uv run python -m src.vlm_example.benchmark_api --config-file-name $(config)

finetune:
	uv run modal run -m src.vlm_example.finetune --config-file-name $(config)

visualize:
	uv run python -m src.vlm_example.make_data_viz $(if $(n),--n $(n),) && open data_visualization.html

lint:
	uv run ruff check --fix .

format:
	uv run ruff format .

code-fixes: lint format
