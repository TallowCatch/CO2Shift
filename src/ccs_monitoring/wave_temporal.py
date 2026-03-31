"""Wave-consistent temporal models and metrics for multi-vintage monitoring."""

from __future__ import annotations

import copy
import time
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
    residual_mae,
    residual_rmse,
    risk_coverage_auc,
    selective_dice,
)
from .model import WaveTemporalMonitoringNet, dice_bce_loss
from .runtime import ensure_torch_seed
from .temporal import growth_adjacency_score, temporal_monotonicity_score


def build_wave_temporal_channels(
    baseline: np.ndarray,
    monitor: np.ndarray,
    observed_flag: float = 1.0,
) -> np.ndarray:
    residual = monitor - baseline
    observation_mask = np.full_like(baseline, float(observed_flag), dtype=np.float32)
    return np.stack([baseline, monitor, residual, observation_mask], axis=0).astype(np.float32)


def build_wave_temporal_sequence_inputs(
    baseline: np.ndarray,
    monitor_sequence: np.ndarray,
    *,
    hidden_indices: list[int] | None = None,
) -> np.ndarray:
    hidden_set = set(hidden_indices or [])
    features = []
    for vintage_index, monitor in enumerate(monitor_sequence):
        if vintage_index in hidden_set:
            features.append(build_wave_temporal_channels(baseline, baseline, observed_flag=0.0))
        else:
            features.append(build_wave_temporal_channels(baseline, monitor, observed_flag=1.0))
    return np.stack(features, axis=0).astype(np.float32)


def build_wave_temporal_residual_targets(
    baseline: np.ndarray,
    monitor_sequence: np.ndarray,
) -> np.ndarray:
    baseline = baseline.astype(np.float32)
    monitor_sequence = monitor_sequence.astype(np.float32)
    if baseline.ndim == 2 and monitor_sequence.ndim == 3:
        return (monitor_sequence - baseline[None, ...]).astype(np.float32)
    if baseline.ndim == 3 and monitor_sequence.ndim == 4:
        return (monitor_sequence - baseline[:, None, ...]).astype(np.float32)
    raise ValueError(
        "Wave-temporal residual targets expect baseline/monitor shapes of either "
        "[H, W] and [T, H, W] or [N, H, W] and [N, T, H, W]. "
        f"Received {baseline.shape} and {monitor_sequence.shape}."
    )


class WaveTemporalMonitoringDataset(Dataset):
    def __init__(
        self,
        baselines: np.ndarray,
        monitor_sequences: np.ndarray,
        support_targets: np.ndarray,
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
        self.support_targets = support_targets.astype(np.float32)
        self.reservoir_masks = reservoir_masks.astype(np.float32)
        self.observation_dropout_probability = float(observation_dropout_probability)
        self.max_hidden_vintages = max(int(max_hidden_vintages), 0)
        self.use_reservoir_weighting = bool(use_reservoir_weighting)
        self.outside_reservoir_weight = float(outside_reservoir_weight)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(self.baselines.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        baseline = self.baselines[index]
        monitor_sequence = self.monitor_sequences[index].copy()
        targets = self.support_targets[index]
        reservoir_mask = self.reservoir_masks[index]
        hidden_indices: list[int] = []
        if self.max_hidden_vintages > 0 and self.rng.random() < self.observation_dropout_probability:
            num_hidden = int(self.rng.integers(1, self.max_hidden_vintages + 1))
            hidden_indices = sorted(
                int(value)
                for value in self.rng.choice(monitor_sequence.shape[0], size=num_hidden, replace=False).tolist()
            )
        features = build_wave_temporal_sequence_inputs(baseline, monitor_sequence, hidden_indices=hidden_indices)
        residual_targets = build_wave_temporal_residual_targets(baseline, monitor_sequence)
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
            torch.from_numpy(residual_targets[:, None, :, :].astype(np.float32)),
            torch.from_numpy(weights[:, None, :, :].astype(np.float32)),
            torch.from_numpy(reservoir_mask[None, :, :].astype(np.float32)),
        )


def _enable_dropout_in_eval(module: torch.nn.Module) -> None:
    if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
        module.train()
    elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
        module.eval()


def _monotone_penalty(probabilities: torch.Tensor, reservoir_mask: torch.Tensor) -> torch.Tensor:
    if probabilities.shape[1] < 2:
        return torch.zeros((), dtype=probabilities.dtype, device=probabilities.device)
    reservoir = reservoir_mask[:, None, :, :, :]
    decreases = F.relu(probabilities[:, :-1] - probabilities[:, 1:]) * reservoir
    return decreases.mean()


def _adjacency_penalty(
    probabilities: torch.Tensor,
    reservoir_mask: torch.Tensor,
    adjacency_dilation: int,
) -> torch.Tensor:
    if probabilities.shape[1] < 2:
        return torch.zeros((), dtype=probabilities.dtype, device=probabilities.device)
    batch_size, num_steps = probabilities.shape[:2]
    reservoir = reservoir_mask[:, None, :, :, :]
    dilation = max(int(adjacency_dilation), 1)
    previous_probabilities = probabilities[:, :-1].reshape(-1, 1, *probabilities.shape[-2:])
    dilated_previous = F.max_pool2d(previous_probabilities, kernel_size=dilation, stride=1, padding=dilation // 2)
    dilated_previous = dilated_previous.reshape(batch_size, num_steps - 1, 1, *probabilities.shape[-2:])
    unsupported_growth = F.relu(probabilities[:, 1:] - dilated_previous) * reservoir
    return unsupported_growth.mean()


def _crossline_continuity_penalty(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.shape[0] < 2:
        return torch.zeros((), dtype=probabilities.dtype, device=probabilities.device)
    trace_support = torch.amax(probabilities, dim=-2)
    return torch.mean(torch.abs(trace_support[1:] - trace_support[:-1]))


def wave_temporal_sequence_loss(
    outputs: dict[str, torch.Tensor],
    support_targets: torch.Tensor,
    residual_targets: torch.Tensor,
    sample_weight: torch.Tensor | None,
    reservoir_mask: torch.Tensor,
    *,
    reconstruction_loss_weight: float,
    monotone_loss_weight: float,
    adjacency_loss_weight: float,
    crossline_loss_weight: float,
    adjacency_dilation: int,
) -> torch.Tensor:
    support_logits = outputs["support_logits"]
    predicted_residual = outputs["predicted_residual"]
    num_steps = support_logits.shape[1]

    support_loss = torch.zeros((), dtype=support_logits.dtype, device=support_logits.device)
    for step_index in range(num_steps):
        weight_step = None if sample_weight is None else sample_weight[:, step_index]
        support_loss = support_loss + dice_bce_loss(
            support_logits[:, step_index],
            support_targets[:, step_index],
            sample_weight=weight_step,
        )
    support_loss = support_loss / max(num_steps, 1)

    if sample_weight is None:
        reconstruction_loss = F.smooth_l1_loss(predicted_residual, residual_targets)
    else:
        reconstruction_error = F.smooth_l1_loss(predicted_residual, residual_targets, reduction="none")
        reconstruction_loss = torch.sum(reconstruction_error * sample_weight) / torch.sum(sample_weight).clamp_min(1.0)

    probabilities = torch.sigmoid(support_logits)
    monotone_penalty = _monotone_penalty(probabilities, reservoir_mask)
    adjacency_penalty = _adjacency_penalty(probabilities, reservoir_mask, adjacency_dilation)
    crossline_penalty = _crossline_continuity_penalty(probabilities)

    return (
        support_loss
        + float(reconstruction_loss_weight) * reconstruction_loss
        + float(monotone_loss_weight) * monotone_penalty
        + float(adjacency_loss_weight) * adjacency_penalty
        + float(crossline_loss_weight) * crossline_penalty
    )


def _collect_wave_outputs(
    model: WaveTemporalMonitoringNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_chunks: list[np.ndarray] = []
    residual_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for features, targets, residual_targets, _weights, _reservoir in loader:
            outputs = model(features.to(device=device))
            logits_chunks.append(outputs["support_logits"].cpu().numpy()[:, :, 0])
            residual_chunks.append(outputs["predicted_residual"].cpu().numpy()[:, :, 0])
            target_chunks.append(targets.numpy()[:, :, 0])
    return (
        np.concatenate(logits_chunks, axis=0),
        np.concatenate(residual_chunks, axis=0),
        np.concatenate(target_chunks, axis=0),
    )


def train_wave_temporal_model(
    train_split: dict[str, np.ndarray],
    val_split: dict[str, np.ndarray],
    training_cfg: dict[str, Any],
    wave_cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    ensure_torch_seed(seed)
    device = torch.device(training_cfg.get("device", "cpu"))
    batch_size = int(wave_cfg.get("batch_size", training_cfg.get("batch_size", 4)))
    epochs = int(wave_cfg.get("epochs", training_cfg.get("epochs", 8)))
    learning_rate = float(wave_cfg.get("learning_rate", training_cfg.get("learning_rate", 1e-3)))
    weight_decay = float(wave_cfg.get("weight_decay", training_cfg.get("weight_decay", 1e-4)))

    train_dataset = WaveTemporalMonitoringDataset(
        train_split["baseline"],
        train_split["monitor_sequence"],
        train_split["change_mask_sequence"],
        train_split["reservoir_mask"],
        observation_dropout_probability=float(wave_cfg.get("observation_dropout_probability", 0.45)),
        max_hidden_vintages=int(wave_cfg.get("max_hidden_vintages", 1)),
        use_reservoir_weighting=bool(training_cfg.get("use_reservoir_weighting", True)),
        outside_reservoir_weight=float(training_cfg.get("outside_reservoir_weight", 0.35)),
        seed=seed,
    )
    val_dataset = WaveTemporalMonitoringDataset(
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

    model = WaveTemporalMonitoringNet(
        in_channels=4,
        base_channels=int(wave_cfg.get("base_channels", 16)),
        dropout=float(wave_cfg.get("dropout", 0.1)),
        max_time_shift_samples=float(wave_cfg.get("max_time_shift_samples", 3.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        for features, targets, residual_targets, weights, reservoir_mask in train_loader:
            features = features.to(device=device)
            targets = targets.to(device=device)
            residual_targets = residual_targets.to(device=device)
            weights = weights.to(device=device)
            reservoir_mask = reservoir_mask.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features)
            loss = wave_temporal_sequence_loss(
                outputs,
                targets,
                residual_targets,
                weights,
                reservoir_mask,
                reconstruction_loss_weight=float(wave_cfg.get("reconstruction_loss_weight", 1.0)),
                monotone_loss_weight=float(wave_cfg.get("monotone_loss_weight", 0.12)),
                adjacency_loss_weight=float(wave_cfg.get("adjacency_loss_weight", 0.08)),
                crossline_loss_weight=float(wave_cfg.get("crossline_loss_weight", 0.04)),
                adjacency_dilation=int(wave_cfg.get("adjacency_dilation", 5)),
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_logits, val_predicted_residual, val_targets = _collect_wave_outputs(model, val_loader, device)
        val_probabilities = 1.0 / (1.0 + np.exp(-val_logits))
        val_support_loss = negative_log_likelihood(val_probabilities, val_targets)
        val_residual_targets = build_wave_temporal_residual_targets(val_split["baseline"], val_split["monitor_sequence"])
        val_reconstruction_loss = residual_rmse(val_predicted_residual, val_residual_targets)
        val_loss = float(val_support_loss + float(wave_cfg.get("reconstruction_loss_weight", 1.0)) * val_reconstruction_loss)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "val_loss": val_loss,
                "val_support_nll": float(val_support_loss),
                "val_residual_rmse": float(val_reconstruction_loss),
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    val_logits, _val_predicted_residual, val_targets = _collect_wave_outputs(model, val_loader, device)
    temperature = fit_temperature(
        val_logits.reshape(-1, *val_logits.shape[2:]),
        val_targets.reshape(-1, *val_targets.shape[2:]),
    )
    return {
        "model": model.cpu(),
        "history": history,
        "temperature": temperature,
    }


def save_wave_temporal_model_artifact(path: Path, artifact: dict[str, Any], wave_cfg: dict[str, Any]) -> None:
    torch.save(
        {
            "state_dict": artifact["model"].state_dict(),
            "history": artifact["history"],
            "temperature": artifact["temperature"],
            "base_channels": int(wave_cfg.get("base_channels", 16)),
            "dropout": float(wave_cfg.get("dropout", 0.1)),
            "max_time_shift_samples": float(wave_cfg.get("max_time_shift_samples", 3.0)),
        },
        path,
    )


def load_wave_temporal_model_artifact(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    model = WaveTemporalMonitoringNet(
        in_channels=4,
        base_channels=int(checkpoint.get("base_channels", 16)),
        dropout=float(checkpoint.get("dropout", 0.1)),
        max_time_shift_samples=float(checkpoint.get("max_time_shift_samples", 3.0)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return {
        "model": model,
        "history": checkpoint["history"],
        "temperature": checkpoint["temperature"],
    }


def predict_wave_temporal_outputs(
    model: WaveTemporalMonitoringNet,
    sequence_inputs: np.ndarray,
    device: torch.device,
    *,
    temperature: float = 1.0,
    mc_passes: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tensor_inputs = torch.from_numpy(sequence_inputs.astype(np.float32))
    loader = DataLoader(tensor_inputs, batch_size=2, shuffle=False)
    model = model.to(device)
    runtime_start = time.perf_counter()

    if mc_passes <= 1:
        if seed is not None:
            ensure_torch_seed(seed)
        model.eval()
        logits_batches: list[np.ndarray] = []
        residual_batches: list[np.ndarray] = []
        amplitude_batches: list[np.ndarray] = []
        shift_batches: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                outputs = model(batch.to(device=device))
                logits_batches.append(outputs["support_logits"].cpu().numpy()[:, :, 0])
                residual_batches.append(outputs["predicted_residual"].cpu().numpy()[:, :, 0])
                amplitude_batches.append(outputs["amplitude_perturbation"].cpu().numpy()[:, :, 0])
                shift_batches.append(outputs["time_shift_field"].cpu().numpy()[:, :, 0])
        logits = np.concatenate(logits_batches, axis=0)
        scaled_logits = apply_temperature(logits, temperature)
        probabilities = 1.0 / (1.0 + np.exp(-scaled_logits))
        uncertainty = np.zeros_like(probabilities, dtype=np.float32)
        predicted_residual = np.concatenate(residual_batches, axis=0).astype(np.float32)
        amplitude = np.concatenate(amplitude_batches, axis=0).astype(np.float32)
        time_shift = np.concatenate(shift_batches, axis=0).astype(np.float32)
    else:
        probability_samples: list[np.ndarray] = []
        residual_samples: list[np.ndarray] = []
        amplitude_samples: list[np.ndarray] = []
        shift_samples: list[np.ndarray] = []
        for pass_index in range(mc_passes):
            if seed is not None:
                ensure_torch_seed(seed + pass_index)
            model.eval()
            model.apply(_enable_dropout_in_eval)
            pass_logits: list[np.ndarray] = []
            pass_residuals: list[np.ndarray] = []
            pass_amplitudes: list[np.ndarray] = []
            pass_shifts: list[np.ndarray] = []
            with torch.no_grad():
                for batch in loader:
                    outputs = model(batch.to(device=device))
                    pass_logits.append(outputs["support_logits"].cpu().numpy()[:, :, 0])
                    pass_residuals.append(outputs["predicted_residual"].cpu().numpy()[:, :, 0])
                    pass_amplitudes.append(outputs["amplitude_perturbation"].cpu().numpy()[:, :, 0])
                    pass_shifts.append(outputs["time_shift_field"].cpu().numpy()[:, :, 0])
            stacked_logits = np.concatenate(pass_logits, axis=0)
            scaled_logits = apply_temperature(stacked_logits, temperature)
            probability_samples.append((1.0 / (1.0 + np.exp(-scaled_logits))).astype(np.float32))
            residual_samples.append(np.concatenate(pass_residuals, axis=0).astype(np.float32))
            amplitude_samples.append(np.concatenate(pass_amplitudes, axis=0).astype(np.float32))
            shift_samples.append(np.concatenate(pass_shifts, axis=0).astype(np.float32))

        probability_samples_np = np.stack(probability_samples, axis=0)
        probabilities, uncertainty = monte_carlo_summary(probability_samples_np)
        predicted_residual = np.mean(np.stack(residual_samples, axis=0), axis=0).astype(np.float32)
        amplitude = np.mean(np.stack(amplitude_samples, axis=0), axis=0).astype(np.float32)
        time_shift = np.mean(np.stack(shift_samples, axis=0), axis=0).astype(np.float32)

    runtime = time.perf_counter() - runtime_start
    return (
        probabilities.astype(np.float32),
        uncertainty.astype(np.float32),
        predicted_residual.astype(np.float32),
        amplitude.astype(np.float32),
        time_shift.astype(np.float32),
        np.array(runtime, dtype=np.float32),
    )


def evaluate_wave_temporal_predictions(
    probabilities: np.ndarray,
    targets: np.ndarray,
    uncertainty: np.ndarray,
    predicted_residual: np.ndarray,
    residual_targets: np.ndarray,
    evaluation_cfg: dict[str, Any],
    reservoir_masks: np.ndarray | None = None,
) -> dict[str, Any]:
    flat_probabilities = probabilities.reshape(-1, *probabilities.shape[-2:])
    flat_targets = targets.reshape(-1, *targets.shape[-2:])
    flat_predicted_residual = predicted_residual.reshape(-1, *predicted_residual.shape[-2:])
    flat_residual_targets = residual_targets.reshape(-1, *residual_targets.shape[-2:])
    flat_reservoir = None
    if reservoir_masks is not None:
        flat_reservoir = reservoir_masks.reshape(-1, *reservoir_masks.shape[-2:])

    metrics = {
        "dice": float(np.mean([_dice_slice(p, t) for p, t in zip(flat_probabilities, flat_targets)])),
        "iou": float(np.mean([iou_score(p, t) for p, t in zip(flat_probabilities, flat_targets)])),
        "false_positive_rate": float(
            np.mean([false_positive_rate(p, t) for p, t in zip(flat_probabilities, flat_targets)])
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
        "residual_mae": float(
            np.mean(
                [
                    residual_mae(prediction, target)
                    for prediction, target in zip(flat_predicted_residual, flat_residual_targets)
                ]
            )
        ),
        "residual_rmse": float(
            np.mean(
                [
                    residual_rmse(prediction, target)
                    for prediction, target in zip(flat_predicted_residual, flat_residual_targets)
                ]
            )
        ),
    }
    if flat_reservoir is not None:
        metrics["reservoir_residual_mae"] = float(
            np.mean(
                [
                    residual_mae(prediction, target, reservoir)
                    for prediction, target, reservoir in zip(
                        flat_predicted_residual,
                        flat_residual_targets,
                        flat_reservoir,
                    )
                ]
            )
        )
        metrics["reservoir_residual_rmse"] = float(
            np.mean(
                [
                    residual_rmse(prediction, target, reservoir)
                    for prediction, target, reservoir in zip(
                        flat_predicted_residual,
                        flat_residual_targets,
                        flat_reservoir,
                    )
                ]
            )
        )
    for quantile in evaluation_cfg.get("abstain_quantiles", [0.7, 0.8, 0.9]):
        metrics[f"selective_dice_q{quantile:.1f}"] = selective_dice(probabilities, targets, uncertainty, quantile)
    return metrics


def _dice_slice(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = prediction >= 0.5
    truth = target > 0.5
    intersection = np.sum(pred & truth)
    denom = np.sum(pred) + np.sum(truth)
    return float((2.0 * intersection + 1e-6) / (denom + 1e-6))


def evaluate_wave_temporal_split(
    artifact: dict[str, Any],
    split: dict[str, np.ndarray],
    evaluation_cfg: dict[str, Any],
    wave_cfg: dict[str, Any],
    device: torch.device,
    *,
    seed: int,
) -> dict[str, Any]:
    full_inputs = np.stack(
        [
            build_wave_temporal_sequence_inputs(baseline, monitor_sequence)
            for baseline, monitor_sequence in zip(split["baseline"], split["monitor_sequence"])
        ],
        axis=0,
    )
    probabilities, uncertainty, predicted_residual, amplitude, time_shift, runtime = predict_wave_temporal_outputs(
        artifact["model"],
        full_inputs,
        device,
        temperature=float(artifact["temperature"]),
        mc_passes=int(wave_cfg.get("mc_dropout_passes", 1)),
        seed=seed,
    )
    targets = split["change_mask_sequence"].astype(np.float32)
    residual_targets = build_wave_temporal_residual_targets(split["baseline"], split["monitor_sequence"])
    reservoir_masks = np.broadcast_to(
        split["reservoir_mask"][:, None, :, :],
        targets.shape,
    ).astype(np.float32)
    per_vintage = {
        str(vintage_index): evaluate_wave_temporal_predictions(
            probabilities[:, vintage_index],
            targets[:, vintage_index],
            uncertainty[:, vintage_index],
            predicted_residual[:, vintage_index],
            residual_targets[:, vintage_index],
            evaluation_cfg,
            reservoir_masks[:, vintage_index],
        )
        for vintage_index in range(probabilities.shape[1])
    }

    heldout_indices = [int(index) for index in wave_cfg.get("heldout_vintages", range(probabilities.shape[1]))]
    heldout_results: dict[str, Any] = {}
    for holdout_index in heldout_indices:
        if holdout_index < 0 or holdout_index >= probabilities.shape[1]:
            continue
        hidden_inputs = np.stack(
            [
                build_wave_temporal_sequence_inputs(baseline, monitor_sequence, hidden_indices=[holdout_index])
                for baseline, monitor_sequence in zip(split["baseline"], split["monitor_sequence"])
            ],
            axis=0,
        )
        hidden_probabilities, hidden_uncertainty, hidden_residual, _hidden_amp, _hidden_shift, _ = predict_wave_temporal_outputs(
            artifact["model"],
            hidden_inputs,
            device,
            temperature=float(artifact["temperature"]),
            mc_passes=int(wave_cfg.get("mc_dropout_passes", 1)),
            seed=seed + 100 + holdout_index,
        )
        heldout_results[str(holdout_index)] = evaluate_wave_temporal_predictions(
            hidden_probabilities[:, holdout_index],
            targets[:, holdout_index],
            hidden_uncertainty[:, holdout_index],
            hidden_residual[:, holdout_index],
            residual_targets[:, holdout_index],
            evaluation_cfg,
            reservoir_masks[:, holdout_index],
        )

    scenario_breakdown: dict[str, Any] = {}
    if "scenario_type" in split:
        for scenario_name in sorted({str(value) for value in split["scenario_type"]}):
            indices = [idx for idx, value in enumerate(split["scenario_type"]) if str(value) == scenario_name]
            scenario_breakdown[scenario_name] = evaluate_wave_temporal_predictions(
                probabilities[indices],
                targets[indices],
                uncertainty[indices],
                predicted_residual[indices],
                residual_targets[indices],
                evaluation_cfg,
                reservoir_masks[indices],
            )

    return {
        "full_observed": evaluate_wave_temporal_predictions(
            probabilities,
            targets,
            uncertainty,
            predicted_residual,
            residual_targets,
            evaluation_cfg,
            reservoir_masks,
        ),
        "per_vintage": per_vintage,
        "held_out_prediction": heldout_results,
        "scenario_breakdown": scenario_breakdown,
        "runtime_seconds": float(runtime),
        "predicted_residual": predicted_residual.astype(np.float32),
        "amplitude_perturbation": amplitude.astype(np.float32),
        "time_shift_field": time_shift.astype(np.float32),
    }


def adapt_wave_temporal_model_to_field(
    artifact: dict[str, Any],
    grouped_sequences: list[dict[str, np.ndarray]],
    wave_cfg: dict[str, Any],
    device: torch.device,
    *,
    seed: int,
) -> dict[str, Any]:
    steps = int(wave_cfg.get("field_adaptation_steps", 0))
    if steps <= 0 or not grouped_sequences:
        return artifact

    ensure_torch_seed(seed)
    adapted_artifact = copy.deepcopy(artifact)
    model = adapted_artifact["model"].to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(wave_cfg.get("field_adaptation_learning_rate", 1.5e-4)),
        weight_decay=float(wave_cfg.get("weight_decay", 1e-4)),
    )

    features = torch.from_numpy(np.stack([entry["inputs"] for entry in grouped_sequences], axis=0)).to(device=device)
    residual_targets = torch.from_numpy(
        np.stack([entry["residual_targets"] for entry in grouped_sequences], axis=0)[:, :, None, :, :]
    ).to(device=device)
    reservoir_masks = torch.from_numpy(
        np.stack([entry["reservoir_mask"] for entry in grouped_sequences], axis=0)[:, None, :, :]
    ).to(device=device)

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(features)
        support_probabilities = torch.sigmoid(outputs["support_logits"])
        reconstruction_error = F.smooth_l1_loss(outputs["predicted_residual"], residual_targets)
        monotone_penalty = _monotone_penalty(support_probabilities, reservoir_masks)
        adjacency_penalty = _adjacency_penalty(
            support_probabilities,
            reservoir_masks,
            int(wave_cfg.get("adjacency_dilation", 5)),
        )
        crossline_penalty = _crossline_continuity_penalty(support_probabilities)
        sparsity_penalty = torch.mean(support_probabilities * reservoir_masks[:, None, :, :, :])
        loss = (
            float(wave_cfg.get("field_reconstruction_loss_weight", 1.0)) * reconstruction_error
            + float(wave_cfg.get("field_monotone_loss_weight", 0.08)) * monotone_penalty
            + float(wave_cfg.get("field_adjacency_loss_weight", 0.06)) * adjacency_penalty
            + float(wave_cfg.get("field_crossline_loss_weight", 0.08)) * crossline_penalty
            + float(wave_cfg.get("field_sparsity_weight", 0.01)) * sparsity_penalty
        )
        loss.backward()
        optimizer.step()

    adapted_artifact["model"] = model.cpu()
    return adapted_artifact
