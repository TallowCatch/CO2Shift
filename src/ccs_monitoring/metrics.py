"""Metrics for synthetic and field-style monitoring evaluation."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


def _binary(prediction: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return prediction >= threshold


def dice_score(prediction: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    pred = _binary(prediction, threshold)
    truth = target > 0.5
    intersection = np.sum(pred & truth)
    denom = pred.sum() + truth.sum()
    return float((2.0 * intersection + 1e-6) / (denom + 1e-6))


def iou_score(prediction: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    pred = _binary(prediction, threshold)
    truth = target > 0.5
    intersection = np.sum(pred & truth)
    union = np.sum(pred | truth)
    return float((intersection + 1e-6) / (union + 1e-6))


def false_positive_rate(prediction: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    pred = _binary(prediction, threshold)
    truth = target > 0.5
    negatives = ~truth
    false_positives = np.sum(pred & negatives)
    return float(false_positives / max(np.sum(negatives), 1))


def centroid_error(prediction: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    pred = _binary(prediction, threshold)
    truth = target > 0.5
    if pred.sum() == 0 or truth.sum() == 0:
        return float("nan")
    pred_center = np.mean(np.argwhere(pred), axis=0)
    truth_center = np.mean(np.argwhere(truth), axis=0)
    return float(np.linalg.norm(pred_center - truth_center))


def extent_error(prediction: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    pred_area = float(np.sum(_binary(prediction, threshold)))
    truth_area = float(np.sum(target > 0.5))
    return abs(pred_area - truth_area)


def expected_calibration_error(probabilities: np.ndarray, targets: np.ndarray, bins: int = 10) -> float:
    flat_probs = probabilities.reshape(-1)
    flat_targets = targets.reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (flat_probs >= lower) & (flat_probs < upper if upper < 1.0 else flat_probs <= upper)
        if not np.any(mask):
            continue
        confidence = float(np.mean(flat_probs[mask]))
        accuracy = float(np.mean(flat_targets[mask] > 0.5))
        ece += (np.sum(mask) / len(flat_probs)) * abs(confidence - accuracy)
    return float(ece)


def coverage_at_quantile(uncertainty: np.ndarray, quantile: float) -> np.ndarray:
    threshold = float(np.quantile(uncertainty, quantile))
    return uncertainty <= threshold


def selective_dice(
    probabilities: np.ndarray,
    targets: np.ndarray,
    uncertainty: np.ndarray,
    quantile: float,
    threshold: float = 0.5,
) -> float:
    keep = coverage_at_quantile(uncertainty, quantile)
    if np.sum(keep) == 0:
        return float("nan")
    masked_probs = probabilities[keep]
    masked_targets = targets[keep]
    masked_pred = masked_probs >= threshold
    masked_truth = masked_targets > 0.5
    intersection = np.sum(masked_pred & masked_truth)
    denom = np.sum(masked_pred) + np.sum(masked_truth)
    return float((2.0 * intersection + 1e-6) / (denom + 1e-6))


def compactness_score(binary_map: np.ndarray) -> float:
    binary_map = binary_map.astype(bool)
    area = float(np.sum(binary_map))
    if area == 0:
        return float("nan")
    eroded = ndimage.binary_erosion(binary_map)
    perimeter = float(np.sum(binary_map ^ eroded))
    return float((4.0 * math.pi * area) / max(perimeter**2, 1.0))


def outside_reservoir_fraction(binary_map: np.ndarray, reservoir_mask: np.ndarray | None) -> float:
    if reservoir_mask is None:
        return float("nan")
    binary_map = binary_map.astype(bool)
    outside = binary_map & ~(reservoir_mask > 0.5)
    predicted = np.sum(binary_map)
    return float(np.sum(outside) / max(predicted, 1))
