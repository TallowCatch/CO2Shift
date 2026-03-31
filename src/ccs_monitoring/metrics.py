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


def brier_score(probabilities: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((probabilities.astype(np.float64) - targets.astype(np.float64)) ** 2))


def negative_log_likelihood(probabilities: np.ndarray, targets: np.ndarray) -> float:
    probs = np.clip(probabilities.astype(np.float64), 1e-6, 1.0 - 1e-6)
    truth = (targets > 0.5).astype(np.float64)
    nll = -(truth * np.log(probs) + (1.0 - truth) * np.log(1.0 - probs))
    return float(np.mean(nll))


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


def risk_coverage_auc(
    probabilities: np.ndarray,
    targets: np.ndarray,
    uncertainty: np.ndarray,
    coverage_quantiles: list[float],
    threshold: float = 0.5,
) -> float:
    if np.allclose(uncertainty, uncertainty.reshape(-1)[0]):
        return float("nan")

    coverages: list[float] = []
    risks: list[float] = []
    for quantile in coverage_quantiles:
        keep = coverage_at_quantile(uncertainty, quantile)
        coverage = float(np.mean(keep))
        pred = probabilities[keep] >= threshold
        truth = targets[keep] > 0.5
        if pred.size == 0:
            continue
        risk = 1.0 - float(np.mean(pred == truth))
        coverages.append(coverage)
        risks.append(risk)
    if len(coverages) < 2:
        return float("nan")
    order = np.argsort(coverages)
    coverage_arr = np.asarray(coverages, dtype=np.float64)[order]
    risk_arr = np.asarray(risks, dtype=np.float64)[order]
    return float(np.trapezoid(risk_arr, coverage_arr))


def error_detection_auroc(
    probabilities: np.ndarray,
    targets: np.ndarray,
    uncertainty: np.ndarray,
    threshold: float = 0.5,
) -> float:
    truth = targets > 0.5
    prediction = probabilities >= threshold
    errors = (prediction != truth).reshape(-1).astype(np.int8)
    scores = uncertainty.reshape(-1).astype(np.float64)
    if errors.min() == errors.max():
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(scores), dtype=np.float64) + 1.0
    positives = errors == 1
    num_pos = int(np.sum(positives))
    num_neg = int(len(errors) - num_pos)
    if num_pos == 0 or num_neg == 0:
        return float("nan")
    rank_sum_pos = float(np.sum(ranks[positives]))
    auc = (rank_sum_pos - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg)
    return float(auc)


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


def inside_support_fraction(binary_map: np.ndarray, support_mask: np.ndarray | None) -> float:
    if support_mask is None:
        return float("nan")
    binary_map = binary_map.astype(bool)
    predicted = np.sum(binary_map)
    inside = binary_map & (support_mask > 0.5)
    return float(np.sum(inside) / max(predicted, 1))


def support_overlap_iou(binary_map: np.ndarray, support_mask: np.ndarray | None) -> float:
    if support_mask is None:
        return float("nan")
    binary_map = binary_map.astype(bool)
    support = support_mask > 0.5
    union = np.sum(binary_map | support)
    if union == 0:
        return float("nan")
    return float(np.sum(binary_map & support) / union)


def support_coverage(binary_map: np.ndarray, support_mask: np.ndarray | None) -> float:
    if support_mask is None:
        return float("nan")
    binary_map = binary_map.astype(bool)
    support = support_mask > 0.5
    support_count = np.sum(support)
    return float(np.sum(binary_map & support) / max(support_count, 1))


def residual_mae(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    prediction = prediction.astype(np.float64)
    target = target.astype(np.float64)
    if mask is not None:
        valid = mask > 0.5
        if not np.any(valid):
            return float("nan")
        return float(np.mean(np.abs(prediction[valid] - target[valid])))
    return float(np.mean(np.abs(prediction - target)))


def residual_rmse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    prediction = prediction.astype(np.float64)
    target = target.astype(np.float64)
    if mask is not None:
        valid = mask > 0.5
        if not np.any(valid):
            return float("nan")
        return float(np.sqrt(np.mean((prediction[valid] - target[valid]) ** 2)))
    return float(np.sqrt(np.mean((prediction - target) ** 2)))
