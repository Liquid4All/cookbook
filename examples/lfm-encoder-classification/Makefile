.PHONY: smoke train predict test lint format

CONFIG ?= config.yaml
TEXT ?= My payment was accepted, but the order has not shipped.

smoke:
	uv run train.py --config $(CONFIG) --smoke-test

train:
	uv run train.py --config $(CONFIG)

predict:
	uv run predict.py --config $(CONFIG) --text "$(TEXT)"

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .
