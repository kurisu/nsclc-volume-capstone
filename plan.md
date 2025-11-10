# Benchmarking 3D Tumor Volume Inference Models

*a CRISP-DM Inspired Project Plan* by Laurentius von Liechti

```mermaid
%%{init: {"flowchart": {"useMaxWidth": true, "nodeSpacing": 40, "rankSpacing": 80}} }%%
flowchart LR
  %% === Styles ===
  classDef phase fill:#EAF2F8,stroke:#1B4F72,stroke-width:1px,font-weight:bold;
  classDef task fill:#FFFFFF,stroke:#555,stroke-width:1px,color:#000;
  classDef deliverable fill:#E8F8F5,stroke:#117864,stroke-width:1px,font-style:italic;

  %% === CRISP-DM columns (each subgraph is a vertical column) ===
  subgraph BU["Business Understanding"]
  direction TB
  P_BU["Hypothesis definition and exploration"]:::task
  A_BU["Abstract & Introduction"]:::deliverable
  M_BU["Draft modeling strategy<br/>Compare baseline architectures"]:::task
  S_BU["Proposed Methodology"]:::deliverable
  end
  class BU phase

  P_BU -.-> A_BU
  P_BU --> M_BU
  M_BU -.-> S_BU
  BU ~~~ DU

  subgraph DU["Data Understanding"]
  direction TB
  R_DU["Review Lung1 dataset<br/>Validate RTSTRUCT geometry"]:::task
  C_DU["Validate DICOM loading<br/>Metadata parsing"]:::task
  M_DU["Visualize 3D slices & masks"]:::task
  D_DU["EDA"]:::deliverable
  end
  class DU phase

  R_DU --> C_DU --> M_DU -.-> D_DU
  DU ~~~ DP

  subgraph DP["Data Preparation"]
  direction TB
  R_DP["Generate voxel masks<br/>Normalize HU; clip −1000→400"]:::task
  C_DP["Implement preprocessing pipeline<br/>ITK/SimpleITK, PyDICOM"]:::task
  M_DP["Create train/val/test splits<br/>Patient strat; Balance by tumor size"]:::task
  D_DP["Data Summary"]:::deliverable
  end
  class DP phase

  R_DP --> C_DP --> M_DP -.-> D_DP
  DP ~~~ MO

  subgraph MO["Modeling"]
  direction TB
  R_MO["Train 3D U-Net / V-Net / nnU-Net<br/>5-fold CV"]:::task
  C_MO["Optimize batch size<br/>Tune memory usage"]:::task
  M_MO["Calibrate thresholds<br/>Morphological cleanup"]:::task
  D_MO["Document hyperparameters<br/>Modeling notebook"]:::task
  L_MO["Literature Review"]:::deliverable
  end
  class MO phase

  R_MO --> C_MO --> M_MO --> D_MO -.-> L_MO
  MO ~~~ EV

  subgraph EV["Evaluation"]
  direction TB
  R_EV["Compute Dice, HD95,<br/>AE/RE/APE, CCC"]:::task
  C_EV["Track metrics & training logs"]:::task
  M_EV["Paired Wilcoxon / t-tests<br/>Bland–Altman plots"]:::task
  D_EV["Performance Report<br/>Graphs & Tables"]:::deliverable
  S_EV["Methodology"]:::deliverable
  end
  class EV phase

  R_EV --> C_EV --> M_EV -.-> D_EV -.-> S_EV
  EV ~~~ DE

  subgraph DE["Delivery"]
  direction TB
  C_DE["Archive weights<br/>Version datasets"]:::task
  A_DE["Assets for next phase"]:::deliverable
  W_DE["Write Paper"]:::task
  R_DE["Results & Conclusion"]:::deliverable
  D_DE["Final draft<br/>Submit Capstone"]:::deliverable
  end
  class DE phase

  C_DE -.-> A_DE
  C_DE --> W_DE -.-> R_DE
  W_DE -.-> D_DE
  ```



## Business Understanding

**Goal**

This project develops an automated 3D segmentation model that reproduces clinician-drawn tumor volumes on lung CT scans with a target Dice ≥ 0.85 and ≤ 10 % volume error. Note that human variability is often well above 10% in lung GTVs.

Towards that goal, the plan is to evaluate and compare 2 or 3 automated 3D segmentation models (e.g., 3D U-Net, V-Net, nnU-Net) in their ability to reproduce clinical Gross Tumor Volumes (GTVs) derived from DICOM RTSTRUCT on the NSCLC-Radiomics (Lung1) dataset.

This represents a step towards a system that would enable faster, standardized, and quantifiable tumor volumetry for radiation planning and therapy monitoring.

**Key Questions**

* Which model best matches expert-drawn GTV masks volumetrically?  
* How large are the volume deviations (absolute / relative)?  
* Are the differences clinically meaningful (e.g., \> ±10 %)?

**Success Criteria**

* Median Dice ≥ 0.85  
* Median |Relative Error| ≤ 10 %  
* Concordance Correlation Coefficient (CCC) ≥ 0.9  
* ≥ 80 % of cases within ± 20 % volume difference  
* Bland–Altman bias within ± 5 %

## Data Understanding

**Sources**

* NSCLC-Radiomics (Lung1): DICOM CT series \+ RTSTRUCT GTV contours.  
* Metadata: Pixel Spacing, Slice Thickness, Reconstruction Kernel, Scanner Vendor.

**Exploration**

* Verify FrameOfReferenceUID alignment between CT and RTSTRUCT.  
* Visualize CT \+ contours (3D Slicer/OHIF) for spot checks.  
* Compute basic stats: voxel spacing distribution, tumor volume range, CT dimensions.  
* Identify potential bias sources (slice thickness, vendor imbalance).

## Data Preparation

**Conversion & Cleaning**

* Convert RTSTRUCT → binary 3D masks aligned to CT.  
* Confirm contour closure and slice coverage.  
* Save both native-spacing and resampled volumes.

**Normalization & Resampling**

* Clip Hounsfield Unit (HU) to \[−1000, 400\] to focus on relevant structures; z-score normalize.  
* Resample to 1–1.5 mm isotropic voxels for training only.  
* Retain native spacing for all volume computations.

**Splitting**

* Patient-level 5-fold split (train/val/test) with fixed random seed for all models.

**Augmentation & Sampling**

* Oversample patches containing tumor voxels.  
* Apply 3D rotations, flips, intensity jitter, Gaussian noise.

**Quality Control**

* Overlay check for every fold: correct series, no offsets, matching spacing metadata.

## Modeling

**Candidate Models**

* Model A: 3D U-Net  
* Model B: V-Net  
* Model C: nnU-Net (baseline auto-configured)

**Training Protocol**

* Identical splits, patch size, epochs, batch size, augmentations, and early-stopping criteria.  
* Optimizer: Adam; learning rate scheduler cosine decay.  
* Loss function: Dice \+ Binary Cross Entropy (BCE); optional Focal component for imbalance.  
* Training on 3D patches (128³ voxels).

**Output Thresholding**

* Convert probability map to binary mask using threshold \= 0.5 (default) or val-optimized Youden.  
* Keep threshold constant across models.

**Post-Processing**

* Connected-component filtering (remove \< 0.5 cm³).  
* Hole fill and surface smoothing.  
* Reproject mask to native DICOM space for volume calculation.

## Evaluation

**Metrics**

| Category | Metric | Purpose |
| ----- | ----- | ----- |
| Spatial Overlap | Dice, Jaccard | Mask agreement |
| Boundary | 95 % Hausdorff Distance (HD95), Surface Dice @ 2 mm | Edge accuracy |
| Volume | Absolute Error (AE), Relative Error (RE %), Absolute Percentage Error (APE) | Volume fidelity |
| Agreement | CCC, Bland–Altman bias ± LOA, R² | Volumetric agreement analysis |

**Computation**

* Compute volumes using native Δx, Δy, Δz from each CT series.  
* Evaluate per patient and aggregate across folds.  
* For multi-lesion cases: sum volumes per patient; optional lesion-wise matching by centroid \+ IoU.

**Statistical Analysis**

* Paired Wilcoxon or paired t-test between models.  
* Equivalence testing (TOST) for ± 10 % volume margin.  
* Bootstrapped 95 % CIs (10 000 resamples).  
* Report proportion of cases within ± 10 % and ± 20 % RE.

**Stratified Analysis**

* Slice thickness (≤ 1.5 mm vs \> 1.5 mm).  
* Reconstruction kernel (soft vs sharp).  
* Tumor size (\< 1 cm³, 1–10 cm³, \> 10 cm³).  
* Vendor differences (if metadata available).

**Visualization**

* Scatter plot: Predicted vs Ground-truth volumes (+ identity line).  
* Bland–Altman plot: bias and limits of agreement.  
* Boxplots: |RE| by model and strata.  
* Example overlays (accurate, median, outlier cases) \+ 3D mesh renders.

## Deliverables

**Capstone**

* Reproducible pipeline: CT \+ RTSTRUCT \=\> mask \=\> volume \=\> metrics.  
* Git repo with uv virtual environment specs for reproducibility  
* Final Capstone Paper

**Packaging**

* Export best model as inference module (predicts mask and volume report).  
* Optional conversion to DICOM-SEG or RTSTRUCT for clinical viewing (OHIF/3D Slicer).

## Iteration Guidelines

* **Evaluation \=\> Data Prep:** If volume error correlates with thickness/vendor, re-balance dataset or add augmentation.  
* **Evaluation \=\> Modeling:** If HD95 is high, tune loss weights or boundary loss.  
* **QC \=\> Data Prep:** Fix any FrameOfReference mismatches before next cycle.

## Summary Table

| Phase | Primary Output | Quality Control |
| ----- | ----- | ----- |
| Business Understanding | Clear success metrics, clinical criteria | Alignment review with clinical expert |
| Data Understanding | DICOM inventory, metadata stats | Visual inspection of overlays |
| Data Preparation | Cleaned native \+ resampled datasets | Geometry & reference checks |
| Modeling | Trained 3D models with identical protocols | Fixed seed reproducibility |
| Evaluation | Statistical & visual comparison reports | Bootstrap CI and stratified plots |
| Deployment | Packaged model \+ documentation | Version control & validation runs |
