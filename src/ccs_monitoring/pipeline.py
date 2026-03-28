"""End-to-end experiment pipeline for reliable 4D CCS monitoring."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ccs_monitoring_mpl"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

from .baselines import (
    apply_threshold,
    fit_best_threshold,
    score_difference,
    score_impedance_difference,
)
from .calibration import apply_temperature, fit_temperature, monte_carlo_summary
from .data import FieldPair, generate_synthetic_benchmark, load_field_pairs, load_split, summarize_field_pairs
from .features import build_hybrid_channels, build_plain_channels
from .metrics import (
    centroid_error,
    compactness_score,
    dice_score,
    expected_calibration_error,
    extent_error,
    false_positive_rate,
    iou_score,
    outside_reservoir_fraction,
    selective_dice,
)
from .model import MonitoringUNet, dice_bce_loss
from .runtime import ensure_runtime_environment, ensure_torch_seed


@dataclass(slots=True)
class DatasetBundle:
    train: dict[str, np.ndarray]
    val: dict[str, np.ndarray]
    test: dict[str, np.ndarray]
    ood: dict[str, np.ndarray]


class MonitoringDataset(Dataset):
    def __init__(
        self,
        baselines: np.ndarray,
        monitors: np.ndarray,
        targets: np.ndarray,
        reservoir_masks: np.ndarray,
        *,
        hybrid: bool,
        augment_cfg: dict[str, Any] | None = None,
        use_reservoir_weighting: bool = False,
        outside_reservoir_weight: float = 0.35,
        seed: int = 0,
    ) -> None:
        self.baselines = baselines.astype(np.float32)
        self.monitors = monitors.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.reservoir_masks = reservoir_masks.astype(np.float32)
        self.hybrid = hybrid
        self.augment_cfg = augment_cfg
        self.use_reservoir_weighting = use_reservoir_weighting
        self.outside_reservoir_weight = float(outside_reservoir_weight)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(self.baselines.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        baseline = self.baselines[index].copy()
        monitor = self.monitors[index].copy()
        target = self.targets[index]
        reservoir_mask = self.reservoir_masks[index]

        if self.augment_cfg is not None:
            baseline, monitor = _apply_training_augmentation(baseline, monitor, self.augment_cfg, self.rng)

        builder = build_hybrid_channels if self.hybrid else build_plain_channels
        features = builder(baseline, monitor)

        weights = np.ones_like(target, dtype=np.float32)
        if self.use_reservoir_weighting:
            weights = np.where(reservoir_mask > 0.5, 1.0, self.outside_reservoir_weight).astype(np.float32)

        return (
            torch.from_numpy(features.astype(np.float32)),
            torch.from_numpy(target[None, :, :].astype(np.float32)),
            torch.from_numpy(weights[None, :, :].astype(np.float32)),
        )


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device=device, non_blocking=False)


def _build_inputs(split_arrays: dict[str, np.ndarray], hybrid: bool) -> tuple[np.ndarray, np.ndarray]:
    builder = build_hybrid_channels if hybrid else build_plain_channels
    inputs = np.stack(
        [builder(baseline, monitor) for baseline, monitor in zip(split_arrays["baseline"], split_arrays["monitor"])],
        axis=0,
    )
    targets = split_arrays["change_mask"].astype(np.float32)
    return inputs, targets


def _apply_training_augmentation(
    baseline: np.ndarray,
    monitor: np.ndarray,
    synthetic_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    augmented_baseline = baseline.copy()
    augmented_monitor = monitor.copy()

    if rng.random() < 0.3:
        augmented_baseline += rng.normal(0.0, 0.01, size=augmented_baseline.shape).astype(np.float32)

    if rng.random() < 0.8:
        shift_min, shift_max = synthetic_cfg["shift_trace_range"]
        shift = int(rng.integers(shift_min, shift_max + 1))
        if shift != 0:
            augmented_monitor = np.roll(augmented_monitor, shift=shift, axis=1)

    if rng.random() < 0.8:
        scale_min, scale_max = synthetic_cfg["amplitude_scale_range"]
        augmented_monitor *= float(rng.uniform(scale_min, scale_max))

    if rng.random() < 0.7:
        noise_min, noise_max = synthetic_cfg["noise_std_range"]
        noise_std = float(rng.uniform(noise_min, noise_max))
        augmented_monitor += rng.normal(0.0, noise_std, size=augmented_monitor.shape).astype(np.float32)

    if rng.random() < 0.6:
        sigma = float(rng.uniform(0.0, 1.25))
        if sigma > 1e-3:
            augmented_monitor = ndimage.gaussian_filter1d(augmented_monitor, sigma=sigma, axis=0)

    if rng.random() < 0.5:
        drop_min, drop_max = synthetic_cfg["drop_trace_fraction_range"]
        drop_fraction = float(rng.uniform(drop_min, drop_max))
        num_drop = int(drop_fraction * augmented_monitor.shape[1])
        if num_drop > 0:
            drop_indices = rng.choice(augmented_monitor.shape[1], size=num_drop, replace=False)
            augmented_monitor[:, drop_indices] = 0.0

    return augmented_baseline.astype(np.float32), augmented_monitor.astype(np.float32)


def _train_epoch(
    model: MonitoringUNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for features, targets, weights in loader:
        features = _to_device(features, device)
        targets = _to_device(targets, device)
        weights = _to_device(weights, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = dice_bce_loss(logits, targets, sample_weight=weights)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _collect_logits(
    model: MonitoringUNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for features, targets, _weights in loader:
            logits = model(_to_device(features, device)).cpu().numpy()
            logits_chunks.append(logits[:, 0])
            target_chunks.append(targets.numpy()[:, 0])
    return np.concatenate(logits_chunks, axis=0), np.concatenate(target_chunks, axis=0)


def _eval_loss_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    probs = 1.0 / (1.0 + np.exp(-logits))
    loss = -np.mean(
        targets * np.log(np.clip(probs, 1e-6, 1.0)) + (1.0 - targets) * np.log(np.clip(1.0 - probs, 1e-6, 1.0))
    )
    return float(loss)


def train_segmentation_model(
    train_split: dict[str, np.ndarray],
    val_split: dict[str, np.ndarray],
    training_cfg: dict[str, Any],
    synthetic_cfg: dict[str, Any],
    seed: int,
    hybrid: bool,
) -> dict[str, Any]:
    ensure_torch_seed(seed)
    device = torch.device(training_cfg.get("device", "cpu"))
    train_ds = MonitoringDataset(
        train_split["baseline"],
        train_split["monitor"],
        train_split["change_mask"],
        train_split["reservoir_mask"],
        hybrid=hybrid,
        augment_cfg=synthetic_cfg if hybrid and training_cfg.get("use_hybrid_augmentation", False) else None,
        use_reservoir_weighting=hybrid and training_cfg.get("use_reservoir_weighting", False),
        outside_reservoir_weight=training_cfg.get("outside_reservoir_weight", 0.35),
        seed=seed,
    )
    val_ds = MonitoringDataset(
        val_split["baseline"],
        val_split["monitor"],
        val_split["change_mask"],
        val_split["reservoir_mask"],
        hybrid=hybrid,
        augment_cfg=None,
        use_reservoir_weighting=False,
        seed=seed + 101,
    )
    train_loader = DataLoader(train_ds, batch_size=training_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=training_cfg["batch_size"], shuffle=False)

    in_channels = 6 if hybrid else 2
    model = MonitoringUNet(in_channels=in_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(training_cfg["epochs"]):
        train_loss = _train_epoch(model, train_loader, optimizer, device)
        val_logits, val_targets_eval = _collect_logits(model, val_loader, device)
        val_loss = _eval_loss_from_logits(val_logits, val_targets_eval)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    val_logits, val_targets_eval = _collect_logits(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_targets_eval)
    return {
        "model": model.cpu(),
        "history": history,
        "temperature": temperature,
    }


def _enable_dropout_in_eval(module: torch.nn.Module) -> None:
    if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
        module.train()
    elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
        module.eval()


def predict_probabilities(
    model: MonitoringUNet,
    inputs: np.ndarray,
    device: torch.device,
    temperature: float = 1.0,
    mc_passes: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor_inputs = torch.from_numpy(inputs.astype(np.float32))
    loader = DataLoader(tensor_inputs, batch_size=4, shuffle=False)
    model = model.to(device)
    logits_batches: list[np.ndarray] = []
    runtime_start = time.perf_counter()

    if mc_passes <= 1:
        if seed is not None:
            ensure_torch_seed(seed)
        model.eval()
        with torch.no_grad():
            for batch in loader:
                logits = model(_to_device(batch, device)).cpu().numpy()[:, 0]
                logits_batches.append(logits)
        logits = np.concatenate(logits_batches, axis=0)
        scaled_logits = apply_temperature(logits, temperature)
        probs = 1.0 / (1.0 + np.exp(-scaled_logits))
        uncertainty = np.zeros_like(probs, dtype=np.float32)
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
                    logits = model(_to_device(batch, device)).cpu().numpy()[:, 0]
                    pass_logits.append(logits)
            stacked = np.concatenate(pass_logits, axis=0)
            scaled = apply_temperature(stacked, temperature)
            probability_samples.append((1.0 / (1.0 + np.exp(-scaled))).astype(np.float32))
        probability_samples_np = np.stack(probability_samples, axis=0)
        probs, uncertainty = monte_carlo_summary(probability_samples_np)
        logits = np.mean(np.log(np.clip(probability_samples_np, 1e-6, 1.0) / np.clip(1.0 - probability_samples_np, 1e-6, 1.0)), axis=0)

    runtime = time.perf_counter() - runtime_start
    return probs.astype(np.float32), uncertainty.astype(np.float32), np.array(runtime, dtype=np.float32)


def _evaluate_predictions(
    probabilities: np.ndarray,
    targets: np.ndarray,
    uncertainty: np.ndarray,
    evaluation_cfg: dict[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "dice": dice_score(probabilities, targets),
        "iou": iou_score(probabilities, targets),
        "false_positive_rate": false_positive_rate(probabilities, targets),
        "centroid_error": centroid_error(probabilities, targets),
        "extent_error": extent_error(probabilities, targets),
        "ece": expected_calibration_error(probabilities, targets),
    }
    for quantile in evaluation_cfg["abstain_quantiles"]:
        metrics[f"selective_dice_q{quantile:.1f}"] = selective_dice(probabilities, targets, uncertainty, quantile)
    return metrics


def _evaluate_field_prediction(
    probabilities: np.ndarray,
    uncertainty: np.ndarray,
    reservoir_mask: np.ndarray | None,
) -> dict[str, Any]:
    binary = probabilities >= 0.5
    return {
        "predicted_fraction": float(np.mean(binary)),
        "compactness": compactness_score(binary),
        "outside_reservoir_fraction": outside_reservoir_fraction(binary, reservoir_mask),
        "mean_uncertainty": float(np.mean(uncertainty)),
    }


def _summarize_field_binary(
    binary: np.ndarray,
    uncertainty: np.ndarray,
    reservoir_mask: np.ndarray | None,
) -> dict[str, Any]:
    return {
        "predicted_fraction": float(np.mean(binary)),
        "compactness": compactness_score(binary),
        "outside_reservoir_fraction": outside_reservoir_fraction(binary, reservoir_mask),
        "mean_uncertainty": float(np.mean(uncertainty)),
    }


def _binary_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(bool)
    second = second.astype(bool)
    union = np.sum(first | second)
    if union == 0:
        return float("nan")
    intersection = np.sum(first & second)
    return float(intersection / union)


def _trace_support_metrics(
    binary_map: np.ndarray,
    support_traces: np.ndarray | None,
) -> dict[str, Any]:
    if support_traces is None:
        return {}

    predicted_trace_mask = np.any(binary_map.astype(bool), axis=0)
    support_trace_mask = support_traces.astype(bool)
    predicted_count = int(np.sum(predicted_trace_mask))
    support_count = int(np.sum(support_trace_mask))
    outside_count = int(np.sum(predicted_trace_mask & ~support_trace_mask))
    inside_count = int(np.sum(predicted_trace_mask & support_trace_mask))

    return {
        "predicted_trace_fraction": float(np.mean(predicted_trace_mask)),
        "support_trace_fraction_2010": float(np.mean(support_trace_mask)),
        "trace_fraction_outside_2010_support": float(outside_count / max(predicted_count, 1)),
        "trace_fraction_inside_2010_support": float(inside_count / max(predicted_count, 1)),
        "trace_iou_with_2010_support": _binary_iou(predicted_trace_mask, support_trace_mask),
        "trace_support_coverage_vs_2010": float(inside_count / max(support_count, 1)),
    }


def _collect_field_region_values(
    values: np.ndarray,
    reservoir_mask: np.ndarray | None,
    post_cfg: dict[str, Any],
) -> np.ndarray:
    if post_cfg.get("shared_use_reservoir_region", True) and reservoir_mask is not None:
        masked = values[reservoir_mask > 0.5]
        if masked.size > 0:
            return masked.astype(np.float32)
    return values.reshape(-1).astype(np.float32)


def _compute_shared_field_postprocess_context(
    probability_maps: list[np.ndarray],
    uncertainty_maps: list[np.ndarray],
    reservoir_masks: list[np.ndarray | None],
    field_cfg: dict[str, Any],
) -> dict[str, Any]:
    post_cfg = field_cfg.get("postprocess", {})
    if not post_cfg.get("enabled", False) or not post_cfg.get("shared_across_pairs", False):
        return {}

    pooled_probabilities = np.concatenate(
        [
            _collect_field_region_values(probabilities, reservoir_mask, post_cfg)
            for probabilities, reservoir_mask in zip(probability_maps, reservoir_masks)
        ],
        axis=0,
    )
    pooled_uncertainties = np.concatenate(
        [
            _collect_field_region_values(uncertainty, reservoir_mask, post_cfg)
            for uncertainty, reservoir_mask in zip(uncertainty_maps, reservoir_masks)
        ],
        axis=0,
    )

    threshold_mode = str(post_cfg.get("threshold_mode", "fixed")).lower()
    if threshold_mode == "quantile":
        quantile = float(post_cfg.get("probability_quantile", 0.92))
        probability_threshold = max(
            float(post_cfg.get("min_probability_threshold", 0.5)),
            float(np.quantile(pooled_probabilities, quantile)),
        )
    else:
        probability_threshold = float(post_cfg.get("probability_threshold", 0.5))

    uncertainty_threshold: float | None = None
    uncertainty_quantile = float(post_cfg.get("uncertainty_quantile", 1.0))
    if uncertainty_quantile < 1.0:
        uncertainty_threshold = float(np.quantile(pooled_uncertainties, uncertainty_quantile))

    return {
        "probability_threshold": float(probability_threshold),
        "uncertainty_threshold": uncertainty_threshold,
        "threshold_source": "shared_field_pairs",
    }


def _postprocess_field_prediction(
    probabilities: np.ndarray,
    uncertainty: np.ndarray,
    reservoir_mask: np.ndarray | None,
    field_cfg: dict[str, Any],
    shared_context: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    post_cfg = field_cfg.get("postprocess", {})
    shared_context = shared_context or {}
    if not post_cfg.get("enabled", False):
        binary = probabilities >= 0.5
        return binary.astype(bool), {
            "enabled": False,
            "probability_threshold": 0.5,
            "uncertainty_threshold": None,
            "num_components_kept": None,
        }

    threshold_mode = str(post_cfg.get("threshold_mode", "fixed")).lower()
    threshold_source = "per_pair"
    if "probability_threshold" in shared_context:
        probability_threshold = float(shared_context["probability_threshold"])
        threshold_source = str(shared_context.get("threshold_source", "shared_field_pairs"))
    elif threshold_mode == "quantile":
        quantile = float(post_cfg.get("probability_quantile", 0.92))
        threshold_values = _collect_field_region_values(probabilities, reservoir_mask, post_cfg)
        probability_threshold = max(
            float(post_cfg.get("min_probability_threshold", 0.5)),
            float(np.quantile(threshold_values, quantile)),
        )
    else:
        probability_threshold = float(post_cfg.get("probability_threshold", 0.5))

    binary = probabilities >= probability_threshold
    uncertainty_threshold: float | None = None
    if "uncertainty_threshold" in shared_context and shared_context["uncertainty_threshold"] is not None:
        uncertainty_threshold = float(shared_context["uncertainty_threshold"])
    else:
        uncertainty_quantile = float(post_cfg.get("uncertainty_quantile", 1.0))
        if uncertainty_quantile < 1.0:
            threshold_values = _collect_field_region_values(uncertainty, reservoir_mask, post_cfg)
            uncertainty_threshold = float(np.quantile(threshold_values, uncertainty_quantile))
    if uncertainty_threshold is not None:
        binary &= uncertainty <= uncertainty_threshold

    if post_cfg.get("apply_reservoir_mask", True) and reservoir_mask is not None:
        binary &= reservoir_mask > 0.5

    closing_iterations = int(post_cfg.get("closing_iterations", 0))
    if closing_iterations > 0:
        binary = ndimage.binary_closing(binary, iterations=closing_iterations)

    opening_iterations = int(post_cfg.get("opening_iterations", 0))
    if opening_iterations > 0:
        binary = ndimage.binary_opening(binary, iterations=opening_iterations)

    keep_largest_components = int(post_cfg.get("keep_largest_components", 0))
    min_component_size = int(post_cfg.get("min_component_size", 0))
    min_component_fraction = float(post_cfg.get("min_component_fraction", 0.0))
    min_size_from_fraction = int(np.ceil(binary.size * min_component_fraction))
    component_size_floor = max(min_component_size, min_size_from_fraction)

    labels, num_labels = ndimage.label(binary)
    num_components_kept = 0
    if num_labels > 0:
        component_sizes = ndimage.sum(binary, labels, index=np.arange(1, num_labels + 1))
        label_sizes = [(label_id + 1, int(size)) for label_id, size in enumerate(component_sizes)]
        label_sizes = [entry for entry in label_sizes if entry[1] >= component_size_floor]
        label_sizes.sort(key=lambda item: item[1], reverse=True)
        if keep_largest_components > 0:
            label_sizes = label_sizes[:keep_largest_components]
        keep_labels = {label_id for label_id, _size in label_sizes}
        binary = np.isin(labels, list(keep_labels))
        num_components_kept = len(keep_labels)
    else:
        binary = binary.astype(bool)

    metadata = {
        "enabled": True,
        "threshold_mode": threshold_mode,
        "probability_threshold": float(probability_threshold),
        "uncertainty_threshold": uncertainty_threshold,
        "threshold_source": threshold_source,
        "shared_across_pairs": bool(post_cfg.get("shared_across_pairs", False)),
        "applied_reservoir_mask": bool(post_cfg.get("apply_reservoir_mask", True) and reservoir_mask is not None),
        "closing_iterations": closing_iterations,
        "opening_iterations": opening_iterations,
        "component_size_floor": int(component_size_floor),
        "num_components_kept": int(num_components_kept),
    }
    return binary.astype(bool), metadata


def _flatten_results(prefix: str, payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    for key, value in payload.items():
        if isinstance(value, dict):
            _flatten_results(f"{prefix}{key}.", value, rows)
        else:
            rows.append({"metric": f"{prefix}{key}".rstrip("."), "value": value})


def _save_prediction_figure(
    baseline: np.ndarray,
    monitor: np.ndarray,
    prediction: np.ndarray,
    uncertainty: np.ndarray,
    target: np.ndarray | None,
    destination: Path,
    title: str,
) -> None:
    figure_columns = 5 if target is not None else 4
    fig, axes = plt.subplots(1, figure_columns, figsize=(4 * figure_columns, 4))
    axes = np.atleast_1d(axes)
    axes[0].imshow(baseline, cmap="gray", aspect="auto")
    axes[0].set_title("Baseline")
    axes[1].imshow(monitor, cmap="gray", aspect="auto")
    axes[1].set_title("Monitor")
    axes[2].imshow(prediction, cmap="magma", aspect="auto", vmin=0.0, vmax=1.0)
    axes[2].set_title("Prediction")
    axes[3].imshow(uncertainty, cmap="viridis", aspect="auto")
    axes[3].set_title("Uncertainty")
    if target is not None:
        axes[4].imshow(target, cmap="magma", aspect="auto", vmin=0.0, vmax=1.0)
        axes[4].set_title("Target")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_metrics_json(path: Path, payload: dict[str, Any]) -> None:
    def _sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: _sanitize(val) for key, val in value.items()}
        if isinstance(value, list):
            return [_sanitize(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value

    path.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")


def _prepare_dirs(output_root: Path) -> dict[str, Path]:
    dirs = {
        "root": output_root,
        "models": output_root / "models",
        "results": output_root / "results",
        "figures": output_root / "results" / "figures",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def _save_model_artifact(path: Path, artifact: dict[str, Any]) -> None:
    torch.save(
        {
            "state_dict": artifact["model"].state_dict(),
            "history": artifact["history"],
            "temperature": artifact["temperature"],
        },
        path,
    )


def _load_model_artifact(path: Path, in_channels: int) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    model = MonitoringUNet(in_channels=in_channels)
    model.load_state_dict(checkpoint["state_dict"])
    return {
        "model": model,
        "history": checkpoint["history"],
        "temperature": checkpoint["temperature"],
    }


def generate(config: dict[str, Any]) -> dict[str, str]:
    ensure_runtime_environment(config["output_root"], config["seed"])
    return generate_synthetic_benchmark(config, config["output_root"])


def _load_bundle(output_root: Path) -> DatasetBundle:
    dataset_dir = output_root / "data"
    return DatasetBundle(
        train=load_split(dataset_dir / "train.npz"),
        val=load_split(dataset_dir / "val.npz"),
        test=load_split(dataset_dir / "test.npz"),
        ood=load_split(dataset_dir / "ood.npz"),
    )


def train(config: dict[str, Any]) -> dict[str, Any]:
    ensure_runtime_environment(config["output_root"], config["seed"])
    output_root = Path(config["output_root"])
    dirs = _prepare_dirs(output_root)
    bundle = _load_bundle(output_root)

    plain_artifact = train_segmentation_model(
        bundle.train,
        bundle.val,
        config["training"],
        config["synthetic"],
        seed=config["seed"],
        hybrid=False,
    )
    hybrid_artifact = train_segmentation_model(
        bundle.train,
        bundle.val,
        config["training"],
        config["synthetic"],
        seed=config["seed"] + 1,
        hybrid=True,
    )

    _save_model_artifact(dirs["models"] / "plain.pt", plain_artifact)
    _save_model_artifact(dirs["models"] / "hybrid.pt", hybrid_artifact)

    training_summary = {
        "plain_history": plain_artifact["history"],
        "hybrid_history": hybrid_artifact["history"],
        "plain_temperature": plain_artifact["temperature"],
        "hybrid_temperature": hybrid_artifact["temperature"],
    }
    _save_metrics_json(dirs["results"] / "training_summary.json", training_summary)
    return training_summary


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    ensure_runtime_environment(config["output_root"], config["seed"])
    output_root = Path(config["output_root"])
    dirs = _prepare_dirs(output_root)
    bundle = _load_bundle(output_root)

    val_diff_scores = np.stack(
        [score_difference(b, m) for b, m in zip(bundle.val["baseline"], bundle.val["monitor"])],
        axis=0,
    )
    val_impedance_scores = np.stack(
        [score_impedance_difference(b, m) for b, m in zip(bundle.val["baseline"], bundle.val["monitor"])],
        axis=0,
    )
    diff_threshold = fit_best_threshold(val_diff_scores, bundle.val["change_mask"])
    impedance_threshold = fit_best_threshold(val_impedance_scores, bundle.val["change_mask"])

    plain_artifact = _load_model_artifact(dirs["models"] / "plain.pt", in_channels=2)
    hybrid_artifact = _load_model_artifact(dirs["models"] / "hybrid.pt", in_channels=6)
    device = torch.device(config["training"].get("device", "cpu"))

    results: dict[str, Any] = {
        "classical_thresholds": {
            "difference": diff_threshold,
            "impedance": impedance_threshold,
        }
    }

    for split_name in ("test", "ood"):
        split = getattr(bundle, split_name)
        targets = split["change_mask"]

        diff_scores = np.stack([score_difference(b, m) for b, m in zip(split["baseline"], split["monitor"])], axis=0)
        impedance_scores = np.stack(
            [score_impedance_difference(b, m) for b, m in zip(split["baseline"], split["monitor"])],
            axis=0,
        )
        diff_predictions = apply_threshold(diff_scores, diff_threshold)
        impedance_predictions = apply_threshold(impedance_scores, impedance_threshold)

        plain_inputs, _ = _build_inputs(split, hybrid=False)
        hybrid_inputs, _ = _build_inputs(split, hybrid=True)

        plain_probs, plain_uncertainty, plain_runtime = predict_probabilities(
            plain_artifact["model"],
            plain_inputs,
            device,
            temperature=plain_artifact["temperature"],
            mc_passes=1,
            seed=config["seed"] + 100,
        )
        hybrid_probs, hybrid_uncertainty, hybrid_runtime = predict_probabilities(
            hybrid_artifact["model"],
            hybrid_inputs,
            device,
            temperature=hybrid_artifact["temperature"],
            mc_passes=config["training"]["mc_dropout_passes"],
            seed=config["seed"] + 200,
        )

        split_metrics = {
            "difference": _evaluate_predictions(diff_predictions, targets, np.zeros_like(diff_predictions), config["evaluation"]),
            "impedance": _evaluate_predictions(
                impedance_predictions,
                targets,
                np.zeros_like(impedance_predictions),
                config["evaluation"],
            ),
            "plain_ml": _evaluate_predictions(plain_probs, targets, plain_uncertainty, config["evaluation"]),
            "hybrid_ml": _evaluate_predictions(hybrid_probs, targets, hybrid_uncertainty, config["evaluation"]),
            "runtime_seconds": {
                "plain_ml": float(plain_runtime),
                "hybrid_ml": float(hybrid_runtime),
            },
        }
        results[split_name] = split_metrics

        if config["evaluation"]["save_figures"]:
            _save_prediction_figure(
                split["baseline"][0],
                split["monitor"][0],
                hybrid_probs[0],
                hybrid_uncertainty[0],
                targets[0],
                dirs["figures"] / f"{split_name}_hybrid_example.png",
                title=f"{split_name.upper()} hybrid example",
            )

    field_pairs = load_field_pairs(config, split_arrays=bundle.ood)
    if field_pairs:
        plume_support_path = config.get("field", {}).get("plume_support_path", "")
        plume_support_traces = None
        if plume_support_path:
            support_array = np.load(plume_support_path).astype(np.float32)
            if support_array.ndim != 1:
                raise ValueError(
                    f"field.plume_support_path must point to a 1D trace-support array; got shape {support_array.shape}."
                )
            if support_array.shape[0] != field_pairs[0].baseline.shape[1]:
                raise ValueError(
                    "field.plume_support_path length does not match the number of traces in the field sections: "
                    f"{support_array.shape[0]} vs {field_pairs[0].baseline.shape[1]}."
                )
            plume_support_traces = support_array > 0.5

        field_outputs: list[dict[str, Any]] = []
        for field_pair in field_pairs:
            field_plain_inputs = build_plain_channels(field_pair.baseline, field_pair.monitor)[None, ...]
            field_hybrid_inputs = build_hybrid_channels(field_pair.baseline, field_pair.monitor)[None, ...]
            plain_probs, plain_uncertainty, _ = predict_probabilities(
                plain_artifact["model"],
                field_plain_inputs,
                device,
                temperature=plain_artifact["temperature"],
                mc_passes=1,
                seed=config["seed"] + 300,
            )
            hybrid_probs, hybrid_uncertainty, _ = predict_probabilities(
                hybrid_artifact["model"],
                field_hybrid_inputs,
                device,
                temperature=hybrid_artifact["temperature"],
                mc_passes=config["training"]["mc_dropout_passes"],
                seed=config["seed"] + 400,
            )
            field_outputs.append(
                {
                    "pair": field_pair,
                    "plain_probs": plain_probs[0],
                    "plain_uncertainty": plain_uncertainty[0],
                    "hybrid_probs": hybrid_probs[0],
                    "hybrid_uncertainty": hybrid_uncertainty[0],
                }
            )

        shared_context = _compute_shared_field_postprocess_context(
            [entry["hybrid_probs"] for entry in field_outputs],
            [entry["hybrid_uncertainty"] for entry in field_outputs],
            [entry["pair"].reservoir_mask for entry in field_outputs],
            config.get("field", {}),
        )

        pair_results: dict[str, Any] = {}
        raw_compactness_values: list[float] = []
        raw_outside_values: list[float] = []
        raw_uncertainty_values: list[float] = []
        constrained_compactness_values: list[float] = []
        constrained_outside_values: list[float] = []
        constrained_uncertainty_values: list[float] = []
        constrained_binaries: dict[str, np.ndarray] = {}
        constrained_fractions: dict[str, float] = {}

        for field_output in field_outputs:
            field_pair = field_output["pair"]
            plain_probs = field_output["plain_probs"]
            plain_uncertainty = field_output["plain_uncertainty"]
            hybrid_probs = field_output["hybrid_probs"]
            hybrid_uncertainty = field_output["hybrid_uncertainty"]
            hybrid_metrics = _evaluate_field_prediction(hybrid_probs, hybrid_uncertainty, field_pair.reservoir_mask)
            constrained_binary, constraint_metadata = _postprocess_field_prediction(
                hybrid_probs,
                hybrid_uncertainty,
                field_pair.reservoir_mask,
                config.get("field", {}),
                shared_context=shared_context,
            )
            constrained_metrics = _summarize_field_binary(
                constrained_binary,
                hybrid_uncertainty,
                field_pair.reservoir_mask,
            )
            support_metrics = _trace_support_metrics(constrained_binary, plume_support_traces)
            pair_results[field_pair.name] = {
                "plain_ml": _evaluate_field_prediction(plain_probs, plain_uncertainty, field_pair.reservoir_mask),
                "hybrid_ml": hybrid_metrics,
                "hybrid_ml_constrained": {
                    **constrained_metrics,
                    **support_metrics,
                    "constraint_metadata": constraint_metadata,
                },
            }
            if not np.isnan(hybrid_metrics["compactness"]):
                raw_compactness_values.append(hybrid_metrics["compactness"])
            if not np.isnan(hybrid_metrics["outside_reservoir_fraction"]):
                raw_outside_values.append(hybrid_metrics["outside_reservoir_fraction"])
            raw_uncertainty_values.append(hybrid_metrics["mean_uncertainty"])

            if not np.isnan(constrained_metrics["compactness"]):
                constrained_compactness_values.append(constrained_metrics["compactness"])
            if not np.isnan(constrained_metrics["outside_reservoir_fraction"]):
                constrained_outside_values.append(constrained_metrics["outside_reservoir_fraction"])
            constrained_uncertainty_values.append(constrained_metrics["mean_uncertainty"])
            constrained_binaries[field_pair.name] = constrained_binary.astype(bool)
            constrained_fractions[field_pair.name] = float(constrained_metrics["predicted_fraction"])

            if config["evaluation"]["save_figures"]:
                _save_prediction_figure(
                    field_pair.baseline,
                    field_pair.monitor,
                    hybrid_probs,
                    hybrid_uncertainty,
                    None,
                    dirs["figures"] / f"field_{field_pair.name}_hybrid_example.png",
                    title=f"Field-style hybrid prediction: {field_pair.name}",
                )
                _save_prediction_figure(
                    field_pair.baseline,
                    field_pair.monitor,
                    constrained_binary.astype(np.float32),
                    hybrid_uncertainty,
                    None,
                    dirs["figures"] / f"field_{field_pair.name}_hybrid_constrained.png",
                    title=f"Field-style constrained hybrid prediction: {field_pair.name}",
                )

        results["field"] = {
            "pairs": pair_results,
            "hybrid_average": {
                "compactness": float(np.mean(raw_compactness_values)) if raw_compactness_values else float("nan"),
                "outside_reservoir_fraction": float(np.mean(raw_outside_values)) if raw_outside_values else float("nan"),
                "mean_uncertainty": float(np.mean(raw_uncertainty_values)) if raw_uncertainty_values else float("nan"),
            },
            "hybrid_constrained_average": {
                "compactness": (
                    float(np.mean(constrained_compactness_values)) if constrained_compactness_values else float("nan")
                ),
                "outside_reservoir_fraction": (
                    float(np.mean(constrained_outside_values)) if constrained_outside_values else float("nan")
                ),
                "mean_uncertainty": (
                    float(np.mean(constrained_uncertainty_values)) if constrained_uncertainty_values else float("nan")
                ),
            },
        }
        if len(constrained_binaries) >= 2:
            ordered_names = sorted(constrained_binaries)
            consecutive_pairs = list(zip(ordered_names[:-1], ordered_names[1:]))
            area_deltas = {
                f"{earlier}->{later}": float(constrained_fractions[later] - constrained_fractions[earlier])
                for earlier, later in consecutive_pairs
            }
            pairwise_iou = {
                f"{earlier}<->{later}": _binary_iou(constrained_binaries[earlier], constrained_binaries[later])
                for earlier, later in consecutive_pairs
            }
            results["field"]["temporal_consistency"] = {
                "ordered_pairs": ordered_names,
                "constrained_area_deltas": area_deltas,
                "constrained_pairwise_iou": pairwise_iou,
                "constrained_area_non_decreasing": all(delta >= 0.0 for delta in area_deltas.values()),
            }
        if plume_support_traces is not None:
            results["field"]["support_note"] = (
                "2010 plume-boundary support is used as a later-time structural envelope, not as exact ground truth "
                "for earlier vintages."
            )

    summary_sections = {
        "Overview": {
            "config_path": config["config_path"],
            "output_root": str(output_root.resolve()),
        },
        "Thresholds": results["classical_thresholds"],
        "Synthetic Test": results["test"]["hybrid_ml"],
        "Synthetic OOD": results["ood"]["hybrid_ml"],
        "Field": results.get("field", {"status": "not enabled"}),
    }
    _save_metrics_json(dirs["results"] / "metrics.json", results)
    flat_rows: list[dict[str, Any]] = []
    _flatten_results("", results, flat_rows)
    with (dirs["results"] / "summary_table.csv").open("w", encoding="utf-8") as handle:
        handle.write("metric,value\n")
        for row in flat_rows:
            handle.write(f"{row['metric']},{row['value']}\n")
    _save_metrics_json(dirs["results"] / "summary.json", summary_sections)
    return results


def validate_field_setup(config: dict[str, Any]) -> dict[str, Any]:
    ensure_runtime_environment(config["output_root"], config["seed"])
    field_cfg = config.get("field", {})
    if not field_cfg.get("enabled", False):
        return {"status": "field_disabled"}

    split_arrays = None
    if field_cfg.get("mode") == "pseudo_sleipner":
        ood_path = Path(config["output_root"]) / "data" / "ood.npz"
        if not ood_path.exists():
            raise FileNotFoundError(
                f"Pseudo field validation requires generated synthetic data at {ood_path}. "
                "Run generate/run-all first or switch the config to field.mode=manifest."
            )
        split_arrays = load_split(ood_path)

    pairs = load_field_pairs(config, split_arrays=split_arrays)
    summary = summarize_field_pairs(pairs)
    summary["mode"] = field_cfg.get("mode", "manifest")
    summary["manifest_path"] = field_cfg.get("manifest_path", "")
    return summary


def run_all(config: dict[str, Any]) -> dict[str, Any]:
    ensure_runtime_environment(config["output_root"], config["seed"])
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "config_snapshot.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    generate(config)
    train(config)
    return evaluate(config)
