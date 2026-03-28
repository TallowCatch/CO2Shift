"""Feature engineering for hybrid monitoring models."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def compute_local_similarity(
    baseline: np.ndarray,
    monitor: np.ndarray,
    sigma: float = 2.0,
) -> np.ndarray:
    numerator = ndimage.gaussian_filter(baseline * monitor, sigma=sigma)
    baseline_energy = ndimage.gaussian_filter(baseline**2, sigma=sigma)
    monitor_energy = ndimage.gaussian_filter(monitor**2, sigma=sigma)
    return numerator / np.sqrt(np.clip(baseline_energy * monitor_energy, 1e-8, None))


def compute_time_shift_proxy(
    baseline: np.ndarray,
    monitor: np.ndarray,
    max_lag: int = 3,
) -> np.ndarray:
    scores = []
    for lag in range(-max_lag, max_lag + 1):
        shifted = np.roll(monitor, shift=lag, axis=0)
        scores.append(np.mean(baseline * shifted, axis=0, keepdims=True))
    stacked = np.concatenate(scores, axis=0)
    best_lag = np.argmax(stacked, axis=0) - max_lag
    return np.repeat(best_lag[None, :], baseline.shape[0], axis=0).astype(np.float32) / max(max_lag, 1)


def build_hybrid_channels(baseline: np.ndarray, monitor: np.ndarray) -> np.ndarray:
    abs_diff = np.abs(monitor - baseline)
    signed_diff = monitor - baseline
    local_similarity = compute_local_similarity(baseline, monitor)
    time_shift = compute_time_shift_proxy(baseline, monitor)
    channels = np.stack(
        [
            baseline,
            monitor,
            signed_diff,
            abs_diff,
            local_similarity.astype(np.float32),
            time_shift.astype(np.float32),
        ],
        axis=0,
    )
    return channels.astype(np.float32)


def build_plain_channels(baseline: np.ndarray, monitor: np.ndarray) -> np.ndarray:
    return np.stack([baseline, monitor], axis=0).astype(np.float32)
