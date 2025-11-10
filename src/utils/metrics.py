from __future__ import annotations

import numpy as np


def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    inter = np.logical_and(y_true, y_pred).sum()
    size_sum = y_true.sum() + y_pred.sum()
    return float((2.0 * inter + eps) / (size_sum + eps))


def jaccard_index(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    inter = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return float((inter + eps) / (union + eps))


def hd95(y_true_surface_pts: np.ndarray, y_pred_surface_pts: np.ndarray) -> float:
    """
    Placeholder for HD95 computation. Requires surface extraction and distance transform.
    """
    raise NotImplementedError("HD95 not implemented in bootstrap.")


def surface_dice_at_tolerance(y_true: np.ndarray, y_pred: np.ndarray, tol_mm: float) -> float:
    """
    Placeholder for Surface Dice. Requires mesh/surface distance computation.
    """
    raise NotImplementedError("Surface Dice not implemented in bootstrap.")


