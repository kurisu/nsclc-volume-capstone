# Benchmarking 3D Tumor Volume Inference Models (NSCLC-Radiomics/Lung1)

Goal: reproduce clinician-drawn GTV volumes on lung CT with target Dice ≥ 0.85 and ≤ 10% volume error; CCC ≥ 0.9. See `plan.md` for details.

## Project Structure
- `configs/`: YAML configs (paths, seeds, folds, thresholds)
- `data/`: datasets and artifacts (not versioned)
  - `raw/`, `interim/`, `processed/`, `metadata/`, `models/`, `logs/`, `reports/`
- `docs/`: documentation; `docs/baselines/` for external baselines
- `notebooks/`: exploratory analysis; keep data out of git
- `src/`: code (data I/O, preprocessing, training, evaluation)
- `reports/`: figures and metrics generated during evaluation

## Stages

- **Data Cleaning / Preparation**
  - [`notebooks/01_preliminary_data_understanding.ipynb`](notebooks/01_preliminary_data_understanding.ipynb): validates raw NSCLC-Radiomics/Lung1 data, checks CT/RTSTRUCT coverage, and performs initial QC.
  - [`src/data/dicom_io.py`](src/data/dicom_io.py): low-level DICOM/RTSTRUCT I/O helpers for CT series and structure sets.
  - [`src/preprocess/prepare_data.py`](src/preprocess/prepare_data.py): CLI entry point that orchestrates conversion from raw DICOM to cleaned/interim artifacts and sets up split-ready folders.
  - [`scripts/convert_lung1_to_nifti.py`](scripts/convert_lung1_to_nifti.py): converts Lung1 DICOM CT + RTSTRUCT into aligned NIfTI volumes and GTV masks (patient-level cleanup and ROI selection).
  - [`scripts/prepare_nnunet_dataset.py`](scripts/prepare_nnunet_dataset.py), [`scripts/build_dataset_json.py`](scripts/build_dataset_json.py), [`scripts/generate_splits.py`](scripts/generate_splits.py), [`scripts/populate_imagesTs.py`](scripts/populate_imagesTs.py): prepare cleaned nnU-Net-style datasets (train/val/test splits, JSON metadata, images/labels layout).

- **Exploratory Data Analysis (EDA)**
  - [`notebooks/01_preliminary_data_understanding.ipynb`](notebooks/01_preliminary_data_understanding.ipynb): dataset inventory, CT/RTSTRUCT availability, metadata distributions, and basic visual checks.
  - [`notebooks/02_data_exploration.ipynb`](notebooks/02_data_exploration.ipynb): explores tumor volume vs voxel counts, slice thickness, reconstruction kernel, morphology, and radiomics-style descriptors; saves aggregated metrics to `data/processed/roi_metrics.parquet` and figures to `reports/figures/`.

- **Model Design / Building**
  - [`docs/baselines/nnunet.md`](docs/baselines/nnunet.md): documents the primary 3D segmentation baseline (nnU-Net) and how it is configured and run for Lung1.
  - [`scripts/nnunet_train_mps.py`](scripts/nnunet_train_mps.py): configures and launches nnU-Net 3d_fullres training on macOS/MPS or compatible GPUs (defines architecture variant, plans, and trainer).
  - [`src/train.py`](src/train.py): planned entry point for custom 3D models (e.g., MONAI 3D U-Net / V-Net); currently a bootstrap stub that loads `configs/default.yaml` and reports configured model candidates.

- **Model Training**
  - [`scripts/nnunet_train_mps.py`](scripts/nnunet_train_mps.py): runs full nnU-Net training for Dataset501_NSCLC_Lung1 (fold selection, epochs, logs).
  - [`convert_raw.dstack.yml`](convert_raw.dstack.yml), [`preprocess.dstack.yml`](preprocess.dstack.yml), [`train.dstack.yml`](train.dstack.yml), [`train-ca-mtl-3.dstack.yml`](train-ca-mtl-3.dstack.yml): dstack job specs that automate preprocessing and training runs in remote environments.
  - [`src/train.py`](src/train.py): CLI scaffold for training custom models locally using the same config file structure as nnU-Net jobs.

- **Model Optimization (hyperparameters, runtime)**
  - [`scripts/nnunet_train_mps.py`](scripts/nnunet_train_mps.py): exposes core training hyperparameters (fold choice, batch size, learning rate schedule) for controlled experiments.
  - [`scripts/analyze_nnunet_training.py`](scripts/analyze_nnunet_training.py): parses nnU-Net training logs, aggregates epoch-level metrics, and produces plots for pseudo Dice, losses, learning rate, and epoch time to guide optimization decisions.
  - [`configs/default.yaml`](configs/default.yaml): central place for dataset paths, seeds, folds, and model-related options that affect optimization.

- **Model Analysis / Evaluation**
  - [`src/utils/metrics.py`](src/utils/metrics.py): defines core segmentation metrics used in this project (Dice, Jaccard; HD95 and Surface Dice placeholders are documented for future extension).
  - [`src/utils/geometry.py`](src/utils/geometry.py): volume computation utilities used when turning binary masks and voxel spacing into physical tumor volumes.
  - [`src/evaluate.py`](src/evaluate.py): CLI for evaluating predicted segmentations (e.g., nnU-Net outputs) against ground-truth Lung1 labels; computes overlap metrics and volume errors and writes CSV summaries.
  - [`scripts/analyze_nnunet_training.py`](scripts/analyze_nnunet_training.py): analyzes training dynamics (convergence, stability, epoch time) from nnU-Net logs and writes both CSV aggregates and diagnostic plots under `reports/figures/`.
  - [`volume.dstack.yml`](volume.dstack.yml), [`eval.dstack.yml`](eval.dstack.yml): dstack specs that run external-test evaluation, aggregate metrics into `data/processed/`, and generate the summary figures in `reports/figures/`.
  - [`src/inference.py`](src/inference.py): planned CLI for loading the best model checkpoint and producing masks + volume reports on new data (bootstrap stub, complementary to `src/evaluate.py`).

## Environment (uv)
1) Install Python 3.10+ and `uv` (`pip install uv` or see the uv docs).
2) Bootstrap the environment:
   - Base deps: `uv sync` (uses `pyproject.toml`/`uv.lock`)
   - Dev tools (optional): `uv pip install -e .[dev]`
   - Training extras (optional): `uv pip install -e .[train]`
   - Radiomics extras (optional, for texture features in notebooks): `uv pip install -e .[radiomics]`
     - Note: On Apple Silicon (arm64), `pyradiomics` wheels may be unavailable and a local build can fail. The notebooks will gracefully skip radiomics features if not installed.
   - Alternatively (recommended on macOS arm64), install PyRadiomics from GitHub with Makefile helper:
     - `make setup-radiomics`
     - This runs:
       - `uv run python -c "import numpy; print('numpy ok')" || uv sync`
       - `uv pip install cython scikit-build ninja`
       - `uv pip install 'git+https://github.com/Radiomics/pyradiomics@master#egg=pyradiomics'`
       - `uv run python -c "import radiomics, SimpleITK; print('pyradiomics OK')"`

## Dataset
NSCLC-Radiomics (Lung1): CT series + RTSTRUCT GTV contours. Place DICOM under:
- `data/raw/NSCLC-Radiomics/<patient_id>/CT/...`
- `data/raw/NSCLC-Radiomics/<patient_id>/RTSTRUCT/rs.dcm`

Legal/Ethics: ensure data access complies with licenses and institutional rules. No PHI should be committed.

## Quickstart
1) Prepare data (convert RTSTRUCT → masks, resample for training, create 5-fold split):
```
make prepare-data
```
2) Train candidate models (stubs; fill in later with MONAI or custom nets):
```
make train
```
3) Evaluate and produce figures/metrics (volumes computed in native spacing):
```
make evaluate
```
Outputs will be written under `reports/` (and/or `data/reports/` for large tables).

## End-to-end pipeline with dstack

This project includes dstack job specs to run the full nnU-Net baseline pipeline remotely and reproducibly.

### 1) Prepare the shared volume (run once, from local)

Use the helper script to bootstrap a shared volume (e.g., a Runpod volume) and pre-stage data/artifacts. Replace arguments as needed for your environment.

```
./scripts/setup_runpod_volume.sh cursor-dev /workspace/capstone_storage /workspace/capstone
```

What the script does:
- Rsyncs `data/interim/` (NIfTIs) and raw clinical CSVs to the volume
- Rsyncs `scripts/` to the remote repo
- On the remote, runs:
  - `scripts/prepare_nnunet_dataset.py`
  - `scripts/build_dataset_json.py`
  - `scripts/generate_splits.py`
  - `scripts/populate_imagesTs.py`

After this step, the volume contains the `nnUNet_raw` dataset, `dataset.json`, and folds/splits needed for jobs.

### 2) Run preprocess and training on dstack

Once the volume is prepared, launch the preprocessing (planning) and training jobs:

```
dstack apply -f preprocess.dstack.yml --yes
dstack apply -f train.dstack.yml --yes
```

These jobs will read from the shared volume, run nnU-Net planning/preprocessing, and then perform training. Check your dstack UI/logs for progress and artifacts written back to the volume.

### 3) Predict and evaluate

After training completes, you can run inference on test/holdout data and then evaluate:

```
dstack apply -f predict.dstack.yml --yes
dstack apply -f eval.dstack.yml --yes
```

Notes:
- `predict.dstack.yml` will produce nnU-Net predictions (e.g., under `data/nnunet/predsTs/`), reading models from `data/nnunet/nnUNet_results/`.
- `eval.dstack.yml` will aggregate metrics (e.g., Dice, volume error, CCC) into `data/processed/` and figures into `reports/`.
- Make sure your dstack workspace has access to the same shared volume paths used during preprocessing/training.

## Baseline: nnU-Net (external)
We treat nnU-Net as an external baseline (not vendored). See `docs/baselines/nnunet.md` for setup and how to import its predicted segmentations for evaluation here.

## Reproducibility
- Fixed seeds and patient-level 5-fold split (see `configs/default.yaml`)
- Consistent post-processing and thresholding across models
- Training resampling only; volume metrics in native spacing

## Author & Affiliation
- Author: Laurentius von Liechti (solo author; acknowledgements to be added)
- Email: lvonliechti@sandiego.edu
- Affiliation: University of San Diego — Masters in Applied Artificial Intelligence
- ORCID: https://orcid.org/0009-0005-4736-6987

## License
MIT. See `LICENSE`.


