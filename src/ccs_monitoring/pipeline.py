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
from torch.utils.data import DataLoader, Dataset

from .baselines import (
    apply_threshold,
    fit_best_threshold,
    score_difference,
    score_impedance_difference,
)
from .calibration import apply_temperature, fit_temperature, monte_carlo_summary
from .data import FieldPair, generate_synthetic_benchmark, load_field_pair, load_split
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
    def __init__(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        self.inputs = torch.from_numpy(inputs.astype(np.float32))
        self.targets = torch.from_numpy(targets[:, None, :, :].astype(np.float32))

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


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


def _train_epoch(
    model: MonitoringUNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for features, targets in loader:
        features = _to_device(features, device)
        targets = _to_device(targets, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = dice_bce_loss(logits, targets)
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
        for features, targets in loader:
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
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    val_inputs: np.ndarray,
    val_targets: np.ndarray,
    training_cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    ensure_torch_seed(seed)
    device = torch.device(training_cfg.get("device", "cpu"))
    train_ds = MonitoringDataset(train_inputs, train_targets)
    val_ds = MonitoringDataset(val_inputs, val_targets)
    train_loader = DataLoader(train_ds, batch_size=training_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=training_cfg["batch_size"], shuffle=False)

    model = MonitoringUNet(in_channels=train_inputs.shape[1]).to(device)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor_inputs = torch.from_numpy(inputs.astype(np.float32))
    loader = DataLoader(tensor_inputs, batch_size=4, shuffle=False)
    model = model.to(device)
    logits_batches: list[np.ndarray] = []
    runtime_start = time.perf_counter()

    if mc_passes <= 1:
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
        for _ in range(mc_passes):
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


def _write_report(
    path: Path,
    title: str,
    summary: dict[str, Any],
) -> None:
    lines = [f"# {title}", ""]
    for section_name, section_value in summary.items():
        lines.append(f"## {section_name}")
        lines.append("")
        if isinstance(section_value, dict):
            for key, value in section_value.items():
                lines.append(f"- **{key}**: {value}")
        else:
            lines.append(str(section_value))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


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

    train_plain_inputs, train_targets = _build_inputs(bundle.train, hybrid=False)
    val_plain_inputs, val_targets = _build_inputs(bundle.val, hybrid=False)
    train_hybrid_inputs, _ = _build_inputs(bundle.train, hybrid=True)
    val_hybrid_inputs, _ = _build_inputs(bundle.val, hybrid=True)

    plain_artifact = train_segmentation_model(
        train_plain_inputs,
        train_targets,
        val_plain_inputs,
        val_targets,
        config["training"],
        seed=config["seed"],
    )
    hybrid_artifact = train_segmentation_model(
        train_hybrid_inputs,
        train_targets,
        val_hybrid_inputs,
        val_targets,
        config["training"],
        seed=config["seed"] + 1,
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
        )
        hybrid_probs, hybrid_uncertainty, hybrid_runtime = predict_probabilities(
            hybrid_artifact["model"],
            hybrid_inputs,
            device,
            temperature=hybrid_artifact["temperature"],
            mc_passes=config["training"]["mc_dropout_passes"],
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

    field_pair = load_field_pair(config, split_arrays=bundle.ood)
    if field_pair is not None:
        field_plain_inputs = build_plain_channels(field_pair.baseline, field_pair.monitor)[None, ...]
        field_hybrid_inputs = build_hybrid_channels(field_pair.baseline, field_pair.monitor)[None, ...]
        plain_probs, plain_uncertainty, _ = predict_probabilities(
            plain_artifact["model"],
            field_plain_inputs,
            device,
            temperature=plain_artifact["temperature"],
            mc_passes=1,
        )
        hybrid_probs, hybrid_uncertainty, _ = predict_probabilities(
            hybrid_artifact["model"],
            field_hybrid_inputs,
            device,
            temperature=hybrid_artifact["temperature"],
            mc_passes=config["training"]["mc_dropout_passes"],
        )
        results["field"] = {
            "name": field_pair.name,
            "plain_ml": _evaluate_field_prediction(plain_probs[0], plain_uncertainty[0], field_pair.reservoir_mask),
            "hybrid_ml": _evaluate_field_prediction(hybrid_probs[0], hybrid_uncertainty[0], field_pair.reservoir_mask),
        }
        if config["evaluation"]["save_figures"]:
            _save_prediction_figure(
                field_pair.baseline,
                field_pair.monitor,
                hybrid_probs[0],
                hybrid_uncertainty[0],
                None,
                dirs["figures"] / "field_hybrid_example.png",
                title=f"Field-style hybrid prediction: {field_pair.name}",
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
    _write_report(dirs["results"] / "report.md", config["report"]["title"], summary_sections)
    return results


def run_all(config: dict[str, Any]) -> dict[str, Any]:
    ensure_runtime_environment(config["output_root"], config["seed"])
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "config_snapshot.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    generate(config)
    train(config)
    return evaluate(config)
