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

##
# nnU-Net (Apple Silicon MPS) helpers
##
.PHONY: nnunet:prepare nnunet:preprocess nnunet:train_fold0_mps nnunet:predict nnunet:evaluate

nnunet:prepare:
	$(PYTHON) scripts/prepare_nnunet_dataset.py

nnunet:preprocess:
	nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres

nnunet:train_fold0_mps:
	$(PYTHON) scripts/nnunet_train_mps.py 501 3d_fullres all -p nnUNetPlans

# Note: requires imagesTs to be populated; outputs to data/nnunet/predsTs
nnunet:predict:
	nnUNetv2_predict -i data/nnunet/nnUNet_raw/Dataset501_NSCLC_Lung1/imagesTs -o data/nnunet/predsTs -d 501 -c 3d_fullres -f 0

# Lightweight bridge to project evaluation (extend src/evaluate.py as needed)
nnunet:evaluate:
	$(PYTHON) -m src.evaluate --config $(CONFIG)

lint:
	ruff check .

format:
	black .


