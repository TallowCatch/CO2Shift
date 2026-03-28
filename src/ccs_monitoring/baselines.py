"""Classical monitoring baselines."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def score_difference(baseline: np.ndarray, monitor: np.ndarray) -> np.ndarray:
    return np.abs(monitor - baseline).astype(np.float32)


def relative_impedance(section: np.ndarray) -> np.ndarray:
    smoothed = ndimage.gaussian_filter1d(section, sigma=1.2, axis=0)
    reflectivity_proxy = np.tanh(smoothed)
    log_impedance = np.cumsum(0.08 * reflectivity_proxy, axis=0)
    impedance = np.exp(log_impedance)
    impedance /= np.mean(impedance, axis=0, keepdims=True) + 1e-8
    return impedance.astype(np.float32)


def score_impedance_difference(baseline: np.ndarray, monitor: np.ndarray) -> np.ndarray:
    baseline_impedance = relative_impedance(baseline)
    monitor_impedance = relative_impedance(monitor)
    return np.abs(monitor_impedance - baseline_impedance).astype(np.float32)


def fit_best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    best_threshold = float(np.quantile(scores, 0.92))
    best_dice = -1.0
    for quantile in np.linspace(0.75, 0.98, 24):
        threshold = float(np.quantile(scores, quantile))
        prediction = scores >= threshold
        intersection = np.sum(prediction & (labels > 0.5))
        denom = prediction.sum() + np.sum(labels > 0.5)
        dice = (2.0 * intersection) / max(denom, 1)
        if dice > best_dice:
            best_dice = float(dice)
            best_threshold = threshold
    return best_threshold


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= threshold).astype(np.float32)
