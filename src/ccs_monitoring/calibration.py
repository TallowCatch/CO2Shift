"""Uncertainty estimation and temperature scaling."""

from __future__ import annotations

import numpy as np


def monte_carlo_summary(probability_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean_prob = np.mean(probability_samples, axis=0)
    entropy = -(
        mean_prob * np.log(np.clip(mean_prob, 1e-6, 1.0))
        + (1.0 - mean_prob) * np.log(np.clip(1.0 - mean_prob, 1e-6, 1.0))
    )
    return mean_prob.astype(np.float32), entropy.astype(np.float32)


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    candidates = np.linspace(0.6, 3.0, 25)
    best_temperature = 1.0
    best_loss = np.inf

    flat_targets = targets.reshape(-1)
    for temperature in candidates:
        scaled_logits = logits.reshape(-1) / temperature
        probs = 1.0 / (1.0 + np.exp(-scaled_logits))
        loss = -np.mean(
            flat_targets * np.log(np.clip(probs, 1e-6, 1.0))
            + (1.0 - flat_targets) * np.log(np.clip(1.0 - probs, 1e-6, 1.0))
        )
        if loss < best_loss:
            best_loss = float(loss)
            best_temperature = float(temperature)
    return best_temperature


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return logits / max(temperature, 1e-6)
