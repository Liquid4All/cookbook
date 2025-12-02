.PHONY: format lint check

run-example:
	uv run transcribe --audio './audio-samples/barackobamafederalplaza.mp3' --play-audio

format:
	uv run ruff format .

lint:
	uv run ruff check . --fix

check:
	uv run ruff check .

lint-check: check format
lint-format: lint format