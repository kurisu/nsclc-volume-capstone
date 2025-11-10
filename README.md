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

## Environment (uv)
1) Install Python 3.10+ and `uv` (`pip install uv` or see the uv docs).
2) Create env and install:
   - Dev only: `uv pip install -e .[dev]`
   - Training extras (optional): `uv pip install -e .[train]`

## Dataset
NSCLC-Radiomics (Lung1): CT series + RTSTRUCT GTV contours. Place DICOM under:
- `data/raw/lung1/<patient_id>/CT/...`
- `data/raw/lung1/<patient_id>/RTSTRUCT/rs.dcm`

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


