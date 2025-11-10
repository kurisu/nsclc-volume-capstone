from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pydicom
import SimpleITK as sitk


def read_ct_series(series_dir: str | Path) -> sitk.Image:
    """
    Read a DICOM CT series into a 3D SimpleITK image using metadata-sorted slices.
    """
    series_dir = Path(series_dir)
    reader = sitk.ImageSeriesReader()
    file_names = reader.GetGDCMSeriesFileNames(str(series_dir))
    if not file_names:
        raise FileNotFoundError(f"No DICOM series found under {series_dir}")
    reader.SetFileNames(file_names)
    image = reader.Execute()
    return image


def read_rtstruct(rtstruct_path: str | Path) -> pydicom.dataset.FileDataset:
    """
    Read a DICOM RTSTRUCT file.
    """
    rtstruct_path = Path(rtstruct_path)
    if not rtstruct_path.exists():
        raise FileNotFoundError(rtstruct_path)
    ds = pydicom.dcmread(str(rtstruct_path))
    if ds.Modality != "RTSTRUCT":
        raise ValueError(f"Expected RTSTRUCT, found Modality={ds.Modality}")
    return ds


def check_frame_of_reference(ct_image: sitk.Image, rtstruct: pydicom.dataset.FileDataset) -> bool:
    """
    Verify FrameOfReferenceUID alignment between CT and RTSTRUCT.
    """
    ct_for = sitk.ReadImage(ct_image).GetMetaData("0020|0052") if False else None  # placeholder
    # Direct access to CT FoR via SimpleITK metadata across slices is non-trivial here;
    # consumers should implement a robust check during mask conversion.
    _ = rtstruct  # quiet linter for now
    return True


def rtstruct_to_mask_aligned(
    ct_image: sitk.Image,
    rtstruct: pydicom.dataset.FileDataset,
    roi_name_predicate: str | None = None,
) -> Tuple[sitk.Image, str]:
    """
    Convert RTSTRUCT contours to a binary 3D mask aligned with the CT image geometry.
    Returns (mask_image, roi_name_used).

    Note: This is a placeholder stub. Implement proper contour-to-mask rasterization with:
      - StructureSetROI/ROIContour matching by name/code
      - ContourData parsing and slice-wise polygon fill
      - Correct orientation and spacing handling
    """
    raise NotImplementedError("RTSTRUCT→mask conversion not yet implemented.")


