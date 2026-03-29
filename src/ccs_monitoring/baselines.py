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


def _static_mask(reservoir_mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if reservoir_mask is None:
        return np.ones(shape, dtype=bool)
    mask = reservoir_mask <= 0.5
    if np.any(mask):
        return mask.astype(bool)
    return np.ones(shape, dtype=bool)


def cross_equalize_monitor(
    baseline: np.ndarray,
    monitor: np.ndarray,
    reservoir_mask: np.ndarray | None = None,
    *,
    max_lag: int = 3,
    lateral_sigma: float = 3.0,
) -> np.ndarray:
    static_mask = _static_mask(reservoir_mask, baseline.shape).astype(np.float32)

    best_lag = 0
    best_score = float("-inf")
    for lag in range(-max_lag, max_lag + 1):
        shifted_monitor = np.roll(monitor, shift=lag, axis=0)
        score = float(np.sum(static_mask * baseline * shifted_monitor))
        if score > best_score:
            best_score = score
            best_lag = lag

    shifted_monitor = np.roll(monitor, shift=best_lag, axis=0)
    weight_sum = np.sum(static_mask, axis=0)
    gain_numerator = np.sum(static_mask * baseline * shifted_monitor, axis=0)
    gain_denominator = np.sum(static_mask * shifted_monitor**2, axis=0)
    gain = gain_numerator / np.clip(gain_denominator, 1e-6, None)
    bias = np.sum(static_mask * (baseline - gain[None, :] * shifted_monitor), axis=0) / np.clip(weight_sum, 1e-6, None)

    if lateral_sigma > 0.0:
        gain = ndimage.gaussian_filter1d(gain.astype(np.float32), sigma=lateral_sigma, axis=0)
        bias = ndimage.gaussian_filter1d(bias.astype(np.float32), sigma=lateral_sigma, axis=0)

    equalized = gain[None, :] * shifted_monitor + bias[None, :]
    return equalized.astype(np.float32)


def score_cross_equalized_difference(
    baseline: np.ndarray,
    monitor: np.ndarray,
    reservoir_mask: np.ndarray | None = None,
) -> np.ndarray:
    equalized_monitor = cross_equalize_monitor(baseline, monitor, reservoir_mask=reservoir_mask)
    return np.abs(equalized_monitor - baseline).astype(np.float32)


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
