.PHONY: setup setup-radiomics prepare-data train evaluate report lint format

PYTHON := python
CONFIG ?= configs/default.yaml

setup:
	uv pip install -e .[dev]

setup-radiomics:
	uv run python -c "import numpy; print('numpy ok')" || uv sync
	uv pip install cython scikit-build ninja
	uv pip install 'git+https://github.com/Radiomics/pyradiomics@master#egg=pyradiomics'
	uv run python -c "import radiomics, SimpleITK; print('pyradiomics OK')"

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


