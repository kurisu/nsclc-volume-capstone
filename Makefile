.PHONY: setup prepare-data train evaluate report lint format

PYTHON := python
CONFIG ?= configs/default.yaml

setup:
	uv pip install -e .[dev]

prepare-data:
	$(PYTHON) -m src.preprocess.prepare_data --config $(CONFIG)

train:
	$(PYTHON) -m src.train --config $(CONFIG)

evaluate:
	$(PYTHON) -m src.evaluate --config $(CONFIG)

report:
	@echo "Reports are generated during evaluate; see reports/ and data/reports/"

lint:
	ruff check .

format:
	black .


