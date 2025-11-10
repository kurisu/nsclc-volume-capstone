from __future__ import annotations

import numpy as np
from typing import Sequence


def compute_volume_cm3(binary_mask: np.ndarray, spacing_xyz_mm: Sequence[float]) -> float:
    """
    Compute volume in cm^3 from a binary mask and voxel spacing (mm).
    """
    if binary_mask.dtype != np.bool_:
        binary_mask = binary_mask.astype(bool)
    voxel_volume_mm3 = float(spacing_xyz_mm[0]) * float(spacing_xyz_mm[1]) * float(spacing_xyz_mm[2])
    volume_mm3 = binary_mask.sum() * voxel_volume_mm3
    return volume_mm3 / 1000.0


