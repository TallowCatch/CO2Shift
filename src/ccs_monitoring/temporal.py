"""Temporal sequence models and metrics for multi-vintage monitoring."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .calibration import apply_temperature, fit_temperature, monte_carlo_summary
from .metrics import (
    brier_score,
    error_detection_auroc,
    expected_calibration_error,
    false_positive_rate,
    iou_score,
    negative_log_likelihood,
    risk_coverage_auc,
    selective_dice,
)
from .model import TemporalMonitoringNet, dice_bce_loss
from .runtime import ensure_torch_seed


def build_temporal_plain_channels(
    baseline: np.ndarray,
    monitor: np.ndarray,
    observed_flag: float = 1.0,
) -> np.ndarray:
    observation_mask = np.full_like(baseline, float(observed_flag), dtype=np.float32)
    return np.stack([baseline, monitor, observation_mask], axis=0).astype(np.float32)


def build_temporal_sequence_inputs(
    baseline: np.ndarray,
    monitor_sequence: np.ndarray,
    *,
    hidden_indices: list[int] | None = None,
) -> np.ndarray:
    hidden_set = set(hidden_indices or [])
    features = []
    for vintage_index, monitor in enumerate(monitor_sequence):
        observed = 0.0 if vintage_index in hidden_set else 1.0
        effective_monitor = baseline if vintage_index in hidden_set else monitor
        features.append(build_temporal_plain_channels(baseline, effective_monitor, observed))
    return np.stack(features, axis=0).astype(np.float32)


class TemporalMonitoringDataset(Dataset):
    def __init__(
        self,
        baselines: np.ndarray,
        monitor_sequences: np.ndarray,
        target_sequences: np.ndarray,
        reservoir_masks: np.ndarray,
        *,
        observation_dropout_probability: float,
        max_hidden_vintages: int,
        use_reservoir_weighting: bool,
        outside_reservoir_weight: float,
        seed: int,
    ) -> None:
        self.baselines = baselines.astype(np.float32)
        self.monitor_sequences = monitor_sequences.astype(np.float32)
        self.target_sequences = target_sequences.astype(np.float32)
        self.reservoir_masks = reservoir_masks.astype(np.float32)
        self.observation_dropout_probability = float(observation_dropout_probability)
        self.max_hidden_vintages = max(int(max_hidden_vintages), 0)
        self.use_reservoir_weighting = bool(use_reservoir_weighting)
        self.outside_reservoir_weight = float(outside_reservoir_weight)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(self.baselines.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        baseline = self.baselines[index]
        monitor_sequence = self.monitor_sequences[index].copy()
        targets = self.target_sequences[index]
        reservoir_mask = self.reservoir_masks[index]
        hidden_indices: list[int] = []
        if self.max_hidden_vintages > 0 and self.rng.random() < self.observation_dropout_probability:
            num_hidden = int(self.rng.integers(1, self.max_hidden_vintages + 1))
            hidden_indices = sorted(
                int(value)
                for value in self.rng.choice(monitor_sequence.shape[0], size=num_hidden, replace=False).tolist()
            )
        features = build_temporal_sequence_inputs(baseline, monitor_sequence, hidden_indices=hidden_indices)
        weights = np.ones_like(targets, dtype=np.float32)
        if self.use_reservoir_weighting:
            weights = np.where(
                np.broadcast_to(reservoir_mask[None, :, :], targets.shape) > 0.5,
                1.0,
                self.outside_reservoir_weight,
            ).astype(np.float32)
        return (
            torch.from_numpy(features.astype(np.float32)),
            torch.from_numpy(targets[:, None, :, :].astype(np.float32)),
            torch.from_numpy(weights[:, None, :, :].astype(np.float32)),
            torch.from_numpy(reservoir_mask[None, :, :].astype(np.float32)),
        )


def _enable_dropout_in_eval(module: torch.nn.Module) -> None:
    if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
        module.train()
    elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
        module.eval()


def temporal_sequence_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weight: torch.Tensor | None,
    reservoir_mask: torch.Tensor,
    *,
    monotone_loss_weight: float,
    adjacency_loss_weight: float,
    adjacency_dilation: int,
) -> torch.Tensor:
    batch_size, num_steps = logits.shape[:2]
    base_loss = torch.zeros((), dtype=logits.dtype, device=logits.device)
    for step_index in range(num_steps):
        weight_step = None if sample_weight is None else sample_weight[:, step_index]
        base_loss = base_loss + dice_bce_loss(logits[:, step_index], targets[:, step_index], sample_weight=weight_step)
    base_loss = base_loss / max(num_steps, 1)

    probabilities = torch.sigmoid(logits)
    reservoir = reservoir_mask[:, None, :, :, :]
    monotone_penalty = torch.zeros((), dtype=logits.dtype, device=logits.device)
    adjacency_penalty = torch.zeros((), dtype=logits.dtype, device=logits.device)
    if num_steps > 1:
        decreases = F.relu(probabilities[:, :-1] - probabilities[:, 1:]) * reservoir
        monotone_penalty = decreases.mean()

        dilation = max(int(adjacency_dilation), 1)
        previous_probabilities = probabilities[:, :-1].reshape(-1, 1, *probabilities.shape[-2:])
        dilated_previous = F.max_pool2d(previous_probabilities, kernel_size=dilation, stride=1, padding=dilation // 2)
        dilated_previous = dilated_previous.reshape(batch_size, num_steps - 1, 1, *probabilities.shape[-2:])
        unsupported_growth = F.relu(probabilities[:, 1:] - dilated_previous) * reservoir[:, : num_steps - 1]
        adjacency_penalty = unsupported_growth.mean()

    return (
        base_loss
        + float(monotone_loss_weight) * monotone_penalty
        + float(adjacency_loss_weight) * adjacency_penalty
    )


def _collect_sequence_logits(
    model: TemporalMonitoringNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for features, targets, _weights, _reservoir in loader:
            logits = model(features.to(device=device)).cpu().numpy()[:, :, 0]
            logits_chunks.append(logits)
            target_chunks.append(targets.numpy()[:, :, 0])
    return np.concatenate(logits_chunks, axis=0), np.concatenate(target_chunks, axis=0)


def train_temporal_model(
    train_split: dict[str, np.ndarray],
    val_split: dict[str, np.ndarray],
    training_cfg: dict[str, Any],
    temporal_cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    ensure_torch_seed(seed)
    device = torch.device(training_cfg.get("device", "cpu"))
    batch_size = int(temporal_cfg.get("batch_size", training_cfg.get("batch_size", 4)))
    epochs = int(temporal_cfg.get("epochs", training_cfg.get("epochs", 8)))
    learning_rate = float(temporal_cfg.get("learning_rate", training_cfg.get("learning_rate", 1e-3)))
    weight_decay = float(temporal_cfg.get("weight_decay", training_cfg.get("weight_decay", 1e-4)))

    train_dataset = TemporalMonitoringDataset(
        train_split["baseline"],
        train_split["monitor_sequence"],
        train_split["change_mask_sequence"],
        train_split["reservoir_mask"],
        observation_dropout_probability=float(temporal_cfg.get("observation_dropout_probability", 0.4)),
        max_hidden_vintages=int(temporal_cfg.get("max_hidden_vintages", 1)),
        use_reservoir_weighting=bool(training_cfg.get("use_reservoir_weighting", True)),
        outside_reservoir_weight=float(training_cfg.get("outside_reservoir_weight", 0.35)),
        seed=seed,
    )
    val_dataset = TemporalMonitoringDataset(
        val_split["baseline"],
        val_split["monitor_sequence"],
        val_split["change_mask_sequence"],
        val_split["reservoir_mask"],
        observation_dropout_probability=0.0,
        max_hidden_vintages=0,
        use_reservoir_weighting=False,
        outside_reservoir_weight=float(training_cfg.get("outside_reservoir_weight", 0.35)),
        seed=seed + 101,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TemporalMonitoringNet(
        in_channels=3,
        base_channels=int(temporal_cfg.get("base_channels", 16)),
        dropout=float(temporal_cfg.get("dropout", 0.1)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []
    monotone_weight = float(temporal_cfg.get("monotone_loss_weight", 0.12))
    adjacency_weight = float(temporal_cfg.get("adjacency_loss_weight", 0.08))
    adjacency_dilation = int(temporal_cfg.get("adjacency_dilation", 5))

    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        for features, targets, weights, reservoir_mask in train_loader:
            features = features.to(device=device)
            targets = targets.to(device=device)
            weights = weights.to(device=device)
            reservoir_mask = reservoir_mask.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = temporal_sequence_loss(
                logits,
                targets,
                weights,
                reservoir_mask,
                monotone_loss_weight=monotone_weight,
                adjacency_loss_weight=adjacency_weight,
                adjacency_dilation=adjacency_dilation,
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_logits, val_targets = _collect_sequence_logits(model, val_loader, device)
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))
        val_loss = negative_log_likelihood(val_probs, val_targets)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": float(val_loss)})
        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    val_logits, val_targets = _collect_sequence_logits(model, val_loader, device)
    temperature = fit_temperature(val_logits.reshape(-1, *val_logits.shape[2:]), val_targets.reshape(-1, *val_targets.shape[2:]))
    return {
        "model": model.cpu(),
        "history": history,
        "temperature": temperature,
    }


def save_temporal_model_artifact(path: Path, artifact: dict[str, Any], temporal_cfg: dict[str, Any]) -> None:
    torch.save(
        {
            "state_dict": artifact["model"].state_dict(),
            "history": artifact["history"],
            "temperature": artifact["temperature"],
            "base_channels": int(temporal_cfg.get("base_channels", 16)),
            "dropout": float(temporal_cfg.get("dropout", 0.1)),
        },
        path,
    )


def load_temporal_model_artifact(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    model = TemporalMonitoringNet(
        in_channels=3,
        base_channels=int(checkpoint.get("base_channels", 16)),
        dropout=float(checkpoint.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return {
        "model": model,
        "history": checkpoint["history"],
        "temperature": checkpoint["temperature"],
    }


def predict_temporal_probabilities(
    model: TemporalMonitoringNet,
    sequence_inputs: np.ndarray,
    device: torch.device,
    *,
    temperature: float = 1.0,
    mc_passes: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor_inputs = torch.from_numpy(sequence_inputs.astype(np.float32))
    loader = DataLoader(tensor_inputs, batch_size=2, shuffle=False)
    model = model.to(device)
    logits_batches: list[np.ndarray] = []
    runtime_start = time.perf_counter()

    if mc_passes <= 1:
        if seed is not None:
            ensure_torch_seed(seed)
        model.eval()
        with torch.no_grad():
            for batch in loader:
                logits = model(batch.to(device=device)).cpu().numpy()[:, :, 0]
                logits_batches.append(logits)
        logits = np.concatenate(logits_batches, axis=0)
        scaled_logits = apply_temperature(logits, temperature)
        probabilities = 1.0 / (1.0 + np.exp(-scaled_logits))
        uncertainty = np.zeros_like(probabilities, dtype=np.float32)
    else:
        probability_samples: list[np.ndarray] = []
        for pass_index in range(mc_passes):
            if seed is not None:
                ensure_torch_seed(seed + pass_index)
            model.eval()
            model.apply(_enable_dropout_in_eval)
            pass_logits: list[np.ndarray] = []
            with torch.no_grad():
                for batch in loader:
                    logits = model(batch.to(device=device)).cpu().numpy()[:, :, 0]
                    pass_logits.append(logits)
            stacked = np.concatenate(pass_logits, axis=0)
            scaled = apply_temperature(stacked, temperature)
            probability_samples.append((1.0 / (1.0 + np.exp(-scaled))).astype(np.float32))
        probability_samples_np = np.stack(probability_samples, axis=0)
        probabilities, uncertainty = monte_carlo_summary(probability_samples_np)
        logits = np.mean(
            np.log(np.clip(probability_samples_np, 1e-6, 1.0) / np.clip(1.0 - probability_samples_np, 1e-6, 1.0)),
            axis=0,
        )
        probabilities = probabilities.astype(np.float32)

    runtime = time.perf_counter() - runtime_start
    return probabilities.astype(np.float32), uncertainty.astype(np.float32), np.array(runtime, dtype=np.float32)


def temporal_monotonicity_score(probabilities: np.ndarray, threshold: float = 0.5) -> float:
    if probabilities.ndim < 4:
        return float("nan")
    binary = probabilities >= threshold
    fractions = np.mean(binary, axis=(-1, -2))
    if fractions.shape[1] < 2:
        return float("nan")
    negative_deltas = np.maximum(fractions[:, :-1] - fractions[:, 1:], 0.0)
    denominators = np.maximum(fractions[:, :-1], 1e-6)
    penalties = negative_deltas / denominators
    return float(np.clip(1.0 - np.mean(penalties), 0.0, 1.0))


def growth_adjacency_score(probabilities: np.ndarray, threshold: float = 0.5) -> float:
    if probabilities.ndim < 4:
        return float("nan")
    binary = probabilities >= threshold
    if binary.shape[1] < 2:
        return float("nan")
    scores: list[float] = []
    for step_index in range(1, binary.shape[1]):
        previous_support = binary[:, step_index - 1]
        current_support = binary[:, step_index]
        new_support = current_support & ~previous_support
        if np.sum(new_support) == 0:
            scores.append(1.0)
            continue
        dilated_previous = np.stack(
            [ndimage.binary_dilation(mask, iterations=2) for mask in previous_support],
            axis=0,
        )
        scores.append(float(np.sum(new_support & dilated_previous) / max(np.sum(new_support), 1)))
    return float(np.mean(scores)) if scores else float("nan")


def evaluate_temporal_predictions(
    probabilities: np.ndarray,
    targets: np.ndarray,
    uncertainty: np.ndarray,
    evaluation_cfg: dict[str, Any],
) -> dict[str, Any]:
    flat_probabilities = probabilities.reshape(-1, *probabilities.shape[-2:])
    flat_targets = targets.reshape(-1, *targets.shape[-2:])
    metrics = {
        "dice": float(np.mean([_dice_slice(p, t) for p, t in zip(flat_probabilities, flat_targets)])),
        "iou": float(np.mean([iou_score(p, t) for p, t in zip(flat_probabilities, flat_targets)])),
        "false_positive_rate": float(
            np.mean(
                [
                    false_positive_rate(p, t)
                    for p, t in zip(flat_probabilities, flat_targets)
                ]
            )
        ),
        "ece": expected_calibration_error(probabilities, targets),
        "brier": brier_score(probabilities, targets),
        "nll": negative_log_likelihood(probabilities, targets),
        "error_detection_auroc": error_detection_auroc(probabilities, targets, uncertainty),
        "risk_coverage_auc": risk_coverage_auc(
            probabilities,
            targets,
            uncertainty,
            evaluation_cfg.get("coverage_quantiles", [0.5, 0.7, 0.8, 0.9, 0.95]),
        ),
        "temporal_monotonicity": temporal_monotonicity_score(probabilities),
        "growth_adjacency": growth_adjacency_score(probabilities),
    }
    for quantile in evaluation_cfg.get("abstain_quantiles", [0.7, 0.8, 0.9]):
        metrics[f"selective_dice_q{quantile:.1f}"] = selective_dice(probabilities, targets, uncertainty, quantile)
    return metrics


def _dice_slice(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = prediction >= 0.5
    truth = target > 0.5
    intersection = np.sum(pred & truth)
    denom = np.sum(pred) + np.sum(truth)
    return float((2.0 * intersection + 1e-6) / (denom + 1e-6))


def evaluate_temporal_split(
    artifact: dict[str, Any],
    split: dict[str, np.ndarray],
    evaluation_cfg: dict[str, Any],
    temporal_cfg: dict[str, Any],
    device: torch.device,
    *,
    seed: int,
) -> dict[str, Any]:
    full_inputs = np.stack(
        [
            build_temporal_sequence_inputs(baseline, monitor_sequence)
            for baseline, monitor_sequence in zip(split["baseline"], split["monitor_sequence"])
        ],
        axis=0,
    )
    probabilities, uncertainty, runtime = predict_temporal_probabilities(
        artifact["model"],
        full_inputs,
        device,
        temperature=float(artifact["temperature"]),
        mc_passes=int(temporal_cfg.get("mc_dropout_passes", 1)),
        seed=seed,
    )
    targets = split["change_mask_sequence"].astype(np.float32)
    per_vintage = {
        str(vintage_index): evaluate_temporal_predictions(
            probabilities[:, vintage_index],
            targets[:, vintage_index],
            uncertainty[:, vintage_index],
            evaluation_cfg,
        )
        for vintage_index in range(probabilities.shape[1])
    }

    heldout_indices = [int(index) for index in temporal_cfg.get("heldout_vintages", range(probabilities.shape[1]))]
    heldout_results: dict[str, Any] = {}
    for holdout_index in heldout_indices:
        if holdout_index < 0 or holdout_index >= probabilities.shape[1]:
            continue
        hidden_inputs = np.stack(
            [
                build_temporal_sequence_inputs(baseline, monitor_sequence, hidden_indices=[holdout_index])
                for baseline, monitor_sequence in zip(split["baseline"], split["monitor_sequence"])
            ],
            axis=0,
        )
        hidden_probabilities, hidden_uncertainty, _ = predict_temporal_probabilities(
            artifact["model"],
            hidden_inputs,
            device,
            temperature=float(artifact["temperature"]),
            mc_passes=int(temporal_cfg.get("mc_dropout_passes", 1)),
            seed=seed + 100 + holdout_index,
        )
        heldout_results[str(holdout_index)] = evaluate_temporal_predictions(
            hidden_probabilities[:, holdout_index],
            targets[:, holdout_index],
            hidden_uncertainty[:, holdout_index],
            evaluation_cfg,
        )

    scenario_breakdown: dict[str, Any] = {}
    if "scenario_type" in split:
        for scenario_name in sorted({str(value) for value in split["scenario_type"]}):
            indices = [idx for idx, value in enumerate(split["scenario_type"]) if str(value) == scenario_name]
            scenario_breakdown[scenario_name] = evaluate_temporal_predictions(
                probabilities[indices],
                targets[indices],
                uncertainty[indices],
                evaluation_cfg,
            )

    return {
        "full_observed": evaluate_temporal_predictions(probabilities, targets, uncertainty, evaluation_cfg),
        "per_vintage": per_vintage,
        "held_out_prediction": heldout_results,
        "scenario_breakdown": scenario_breakdown,
        "runtime_seconds": float(runtime),
    }
