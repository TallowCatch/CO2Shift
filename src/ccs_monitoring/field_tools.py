"""Reusable field-style inference helpers for evidence packs and volume outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy import ndimage

from .baselines import apply_threshold, fit_best_threshold
from .data import FieldPair, load_field_pairs
from .features import build_hybrid_channels, build_plain_channels
from .pipeline import (
    CLASSICAL_SCORERS,
    _cleanup_field_binary,
    _compute_shared_field_postprocess_context,
    _evaluate_field_prediction,
    _load_bundle,
    _load_model_artifact,
    _postprocess_field_prediction,
    _resolve_artifacts_root,
    _resolve_support_volume,
    _summarize_field_binary,
    _support_volume_metrics,
    _trace_support_metrics,
    predict_probabilities,
)
from .runtime import ensure_runtime_environment
from .temporal import (
    build_temporal_sequence_inputs,
    load_temporal_model_artifact,
    predict_temporal_probabilities,
)
from .wave_temporal import (
    adapt_wave_temporal_model_to_field,
    build_wave_temporal_residual_targets,
    build_wave_temporal_sequence_inputs,
    load_wave_temporal_model_artifact,
    predict_wave_temporal_outputs,
)


@dataclass(slots=True)
class FieldPredictionBundle:
    artifacts_root: Path
    field_pairs: list[FieldPair]
    plume_support_traces: np.ndarray | None
    shared_context: dict[str, Any]
    classical_thresholds: dict[str, float]
    outputs: list[dict[str, Any]]


def _load_plume_support_traces(config: dict[str, Any], expected_trace_count: int) -> np.ndarray | None:
    plume_support_path = str(config.get("field", {}).get("plume_support_path", "")).strip()
    if not plume_support_path:
        return None

    support_array = np.load(plume_support_path).astype(np.float32)
    if support_array.ndim != 1:
        raise ValueError(
            f"field.plume_support_path must point to a 1D trace-support array; got shape {support_array.shape}."
        )
    if support_array.shape[0] != expected_trace_count:
        raise ValueError(
            "field.plume_support_path length does not match the number of traces in the field sections: "
            f"{support_array.shape[0]} vs {expected_trace_count}."
        )
    return support_array > 0.5


def _compute_classical_thresholds(bundle: Any) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for method_name, scorer in CLASSICAL_SCORERS.items():
        scores = np.stack(
            [
                scorer(baseline, monitor, reservoir_mask)
                for baseline, monitor, reservoir_mask in zip(
                    bundle.val["baseline"],
                    bundle.val["monitor"],
                    bundle.val["reservoir_mask"],
                )
            ],
            axis=0,
        )
        thresholds[method_name] = fit_best_threshold(scores, bundle.val["change_mask"])
    return thresholds


def _attach_temporal_sequence_predictions(
    outputs: list[dict[str, Any]],
    config: dict[str, Any],
    artifacts_root: Path,
) -> None:
    temporal_cfg = config.get("temporal", {})
    if not temporal_cfg.get("enabled", False):
        return

    temporal_model_path = artifacts_root / "models" / "temporal.pt"
    if not temporal_model_path.exists():
        return

    temporal_artifact = load_temporal_model_artifact(temporal_model_path)
    device = torch.device(config["training"].get("device", "cpu"))
    grouped_indices = _group_output_indices(outputs, "inline_id")
    mc_passes = int(temporal_cfg.get("mc_dropout_passes", config["training"].get("mc_dropout_passes", 1)))

    for _inline_key, indices in grouped_indices.items():
        ordered_indices = _sorted_indices_by_vintage(outputs, indices)
        baseline = _sequence_baseline(outputs, ordered_indices)
        monitor_sequence = np.stack(
            [outputs[index]["pair"].monitor.astype(np.float32) for index in ordered_indices],
            axis=0,
        )
        full_inputs = build_temporal_sequence_inputs(baseline, monitor_sequence)[None, ...]
        probabilities, uncertainty, _ = predict_temporal_probabilities(
            temporal_artifact["model"],
            full_inputs,
            device,
            temperature=float(temporal_artifact["temperature"]),
            mc_passes=mc_passes,
            seed=int(config["seed"]) + 600,
        )
        for position, index in enumerate(ordered_indices):
            outputs[index]["temporal_probs"] = probabilities[0, position].astype(np.float32)
            outputs[index]["temporal_uncertainty"] = uncertainty[0, position].astype(np.float32)

        if len(ordered_indices) < 2:
            continue

        for position, index in enumerate(ordered_indices):
            hidden_inputs = build_temporal_sequence_inputs(baseline, monitor_sequence, hidden_indices=[position])[None, ...]
            hidden_probabilities, hidden_uncertainty, _ = predict_temporal_probabilities(
                temporal_artifact["model"],
                hidden_inputs,
                device,
                temperature=float(temporal_artifact["temperature"]),
                mc_passes=mc_passes,
                seed=int(config["seed"]) + 700 + position,
            )
            outputs[index]["temporal_leave_one_out_probs"] = hidden_probabilities[0, position].astype(np.float32)
            outputs[index]["temporal_leave_one_out_uncertainty"] = hidden_uncertainty[0, position].astype(np.float32)


def _attach_wave_temporal_sequence_predictions(
    outputs: list[dict[str, Any]],
    config: dict[str, Any],
    artifacts_root: Path,
) -> None:
    wave_cfg = config.get("wave_temporal", {})
    if not wave_cfg.get("enabled", False):
        return

    wave_model_path = artifacts_root / "models" / "wave_temporal.pt"
    if not wave_model_path.exists():
        return

    wave_artifact = load_wave_temporal_model_artifact(wave_model_path)
    device = torch.device(config["training"].get("device", "cpu"))
    grouped_indices = _group_output_indices(outputs, "inline_id")
    mc_passes = int(wave_cfg.get("mc_dropout_passes", config["training"].get("mc_dropout_passes", 1)))

    grouped_sequences: list[dict[str, np.ndarray]] = []
    ordered_groups: list[list[int]] = []
    ordered_inline_keys = sorted(grouped_indices, key=_sort_metadata_label)
    for inline_key in ordered_inline_keys:
        indices = grouped_indices[inline_key]
        ordered_indices = _sorted_indices_by_vintage(outputs, indices)
        baseline = _sequence_baseline(outputs, ordered_indices)
        monitor_sequence = np.stack(
            [outputs[index]["pair"].monitor.astype(np.float32) for index in ordered_indices],
            axis=0,
        )
        pair_reservoir_mask = outputs[ordered_indices[0]]["pair"].reservoir_mask
        reservoir_mask = (
            pair_reservoir_mask.astype(np.float32)
            if pair_reservoir_mask is not None
            else np.ones_like(baseline, dtype=np.float32)
        )
        grouped_sequences.append(
            {
                "inputs": build_wave_temporal_sequence_inputs(baseline, monitor_sequence),
                "residual_targets": build_wave_temporal_residual_targets(baseline, monitor_sequence),
                "reservoir_mask": reservoir_mask,
            }
        )
        ordered_groups.append(ordered_indices)

    adapted_artifact = adapt_wave_temporal_model_to_field(
        wave_artifact,
        grouped_sequences,
        wave_cfg,
        device,
        seed=int(config["seed"]) + 800,
    )

    for ordered_indices, grouped_sequence in zip(ordered_groups, grouped_sequences):
        baseline = outputs[ordered_indices[0]]["pair"].baseline.astype(np.float32)
        monitor_sequence = np.stack(
            [outputs[index]["pair"].monitor.astype(np.float32) for index in ordered_indices],
            axis=0,
        )
        full_inputs = grouped_sequence["inputs"][None, ...]
        probabilities, uncertainty, predicted_residual, amplitude, time_shift, _ = predict_wave_temporal_outputs(
            adapted_artifact["model"],
            full_inputs,
            device,
            temperature=float(adapted_artifact["temperature"]),
            mc_passes=mc_passes,
            seed=int(config["seed"]) + 900,
        )
        for position, index in enumerate(ordered_indices):
            outputs[index]["wave_temporal_probs"] = probabilities[0, position].astype(np.float32)
            outputs[index]["wave_temporal_uncertainty"] = uncertainty[0, position].astype(np.float32)
            outputs[index]["wave_temporal_predicted_residual"] = predicted_residual[0, position].astype(np.float32)
            outputs[index]["wave_temporal_amplitude_perturbation"] = amplitude[0, position].astype(np.float32)
            outputs[index]["wave_temporal_time_shift_field"] = time_shift[0, position].astype(np.float32)

        if len(ordered_indices) < 2:
            continue

        for position, index in enumerate(ordered_indices):
            hidden_inputs = build_wave_temporal_sequence_inputs(
                baseline,
                monitor_sequence,
                hidden_indices=[position],
            )[None, ...]
            (
                hidden_probabilities,
                hidden_uncertainty,
                hidden_predicted_residual,
                _hidden_amplitude,
                _hidden_time_shift,
                _,
            ) = predict_wave_temporal_outputs(
                adapted_artifact["model"],
                hidden_inputs,
                device,
                temperature=float(adapted_artifact["temperature"]),
                mc_passes=mc_passes,
                seed=int(config["seed"]) + 1000 + position,
            )
            outputs[index]["wave_temporal_leave_one_out_probs"] = hidden_probabilities[0, position].astype(np.float32)
            outputs[index]["wave_temporal_leave_one_out_uncertainty"] = hidden_uncertainty[0, position].astype(
                np.float32
            )
            outputs[index]["wave_temporal_leave_one_out_predicted_residual"] = hidden_predicted_residual[
                0, position
            ].astype(np.float32)


def collect_field_prediction_bundle(config: dict[str, Any]) -> FieldPredictionBundle:
    ensure_runtime_environment(config["output_root"], config["seed"])
    artifacts_root = _resolve_artifacts_root(config)
    bundle = _load_bundle(artifacts_root)
    classical_thresholds = _compute_classical_thresholds(bundle)

    plain_artifact = _load_model_artifact(artifacts_root / "models" / "plain.pt", in_channels=2)
    hybrid_artifact = _load_model_artifact(artifacts_root / "models" / "hybrid.pt", in_channels=6)
    device = torch.device(config["training"].get("device", "cpu"))

    field_pairs = load_field_pairs(config, split_arrays=bundle.ood)
    if not field_pairs:
        return FieldPredictionBundle(
            artifacts_root=artifacts_root,
            field_pairs=[],
            plume_support_traces=None,
            shared_context={},
            classical_thresholds=classical_thresholds,
            outputs=[],
        )

    plume_support_traces = _load_plume_support_traces(config, expected_trace_count=field_pairs[0].baseline.shape[1])
    outputs: list[dict[str, Any]] = []
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
        classical_scores = {
            method_name: scorer(field_pair.baseline, field_pair.monitor, field_pair.reservoir_mask)
            for method_name, scorer in CLASSICAL_SCORERS.items()
        }
        outputs.append(
            {
                "pair": field_pair,
                "plain_probs": plain_probs[0],
                "plain_uncertainty": plain_uncertainty[0],
                "hybrid_probs": hybrid_probs[0],
                "hybrid_uncertainty": hybrid_uncertainty[0],
                "classical_scores": classical_scores,
            }
        )

    shared_context = _compute_shared_field_postprocess_context(
        [entry["hybrid_probs"] for entry in outputs],
        [entry["hybrid_uncertainty"] for entry in outputs],
        [entry["pair"].reservoir_mask for entry in outputs],
        config.get("field", {}),
    )
    _attach_temporal_sequence_predictions(outputs, config, artifacts_root)
    _attach_wave_temporal_sequence_predictions(outputs, config, artifacts_root)

    return FieldPredictionBundle(
        artifacts_root=artifacts_root,
        field_pairs=field_pairs,
        plume_support_traces=plume_support_traces,
        shared_context=shared_context,
        classical_thresholds=classical_thresholds,
        outputs=outputs,
    )


def _binary_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(bool)
    second = second.astype(bool)
    union = np.sum(first | second)
    if union == 0:
        return float("nan")
    intersection = np.sum(first & second)
    return float(intersection / union)


def _rank_metric_for_pair(pair_metrics: dict[str, Any], support_volume_available: bool) -> float:
    if support_volume_available:
        return float(pair_metrics.get("support_volume_iou_2010", float("-inf")))
    return float(pair_metrics.get("trace_iou_with_2010_support", float("-inf")))


def _pair_support_traces(field_pair: FieldPair, bundle: FieldPredictionBundle) -> np.ndarray | None:
    if field_pair.support_mask is not None:
        return np.any(field_pair.support_mask > 0.5, axis=0).astype(np.float32)
    if bundle.plume_support_traces is not None and bundle.plume_support_traces.shape[0] == field_pair.baseline.shape[1]:
        return bundle.plume_support_traces.astype(np.float32)
    return None


def _pair_support_volume(field_pair: FieldPair, bundle: FieldPredictionBundle) -> np.ndarray | None:
    pair_support_traces = _pair_support_traces(field_pair, bundle)
    return _resolve_support_volume(field_pair.reservoir_mask, pair_support_traces, field_pair.support_mask)


def _summarize_binary_pair(
    binary: np.ndarray,
    uncertainty: np.ndarray,
    field_pair: FieldPair,
    plume_support_traces: np.ndarray | None,
    support_volume: np.ndarray | None,
    constraint_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        **_summarize_field_binary(binary, uncertainty, field_pair.reservoir_mask),
        **_trace_support_metrics(binary, plume_support_traces),
        **_support_volume_metrics(binary, support_volume),
        "constraint_metadata": constraint_metadata,
    }


def _residual_fit_metrics(
    predicted_residual: np.ndarray,
    field_pair: FieldPair,
) -> dict[str, Any]:
    observed_residual = field_pair.monitor.astype(np.float32) - field_pair.baseline.astype(np.float32)
    reservoir_mask = field_pair.reservoir_mask.astype(np.float32) if field_pair.reservoir_mask is not None else None
    residual_error = predicted_residual.astype(np.float32) - observed_residual
    payload = {
        "residual_fit_mae": float(np.mean(np.abs(residual_error))),
        "residual_fit_rmse": float(np.sqrt(np.mean(residual_error**2))),
    }
    if reservoir_mask is not None and np.any(reservoir_mask > 0.5):
        valid = reservoir_mask > 0.5
        payload["reservoir_residual_fit_mae"] = float(np.mean(np.abs(residual_error[valid])))
        payload["reservoir_residual_fit_rmse"] = float(np.sqrt(np.mean(residual_error[valid] ** 2)))
    return payload


def _summarize_classical_pair(
    method_name: str,
    scores: np.ndarray,
    threshold: float,
    field_pair: FieldPair,
    plume_support_traces: np.ndarray | None,
    support_volume: np.ndarray | None,
    field_cfg: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    binary = apply_threshold(scores[None, ...], threshold)[0] > 0.5
    structured_binary, cleanup = _cleanup_field_binary(binary, field_pair.reservoir_mask, field_cfg)
    summary = _summarize_binary_pair(
        structured_binary,
        np.zeros_like(scores, dtype=np.float32),
        field_pair,
        plume_support_traces,
        support_volume,
        {
            "enabled": True,
            "threshold_mode": "synthetic_validation_threshold",
            "probability_threshold": float(threshold),
            "uncertainty_threshold": None,
            "threshold_source": "synthetic_validation_threshold",
            "shared_across_pairs": True,
            **cleanup,
        },
    )
    summary["method_name"] = method_name
    return summary, structured_binary.astype(bool)


def _field_group_key(field_pair: FieldPair, key: str, fallback: str) -> str:
    metadata = field_pair.metadata or {}
    value = metadata.get(key)
    return str(value) if value is not None else fallback


def _group_output_indices(bundle_outputs: list[dict[str, Any]], metadata_key: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, entry in enumerate(bundle_outputs):
        pair = entry["pair"]
        key = _field_group_key(pair, metadata_key, pair.name)
        groups.setdefault(key, []).append(index)
    return groups


def _sort_metadata_label(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if str(value).isdigit() else (1, str(value))


def _sorted_indices_by_inline(bundle_outputs: list[dict[str, Any]], indices: list[int]) -> list[int]:
    return sorted(
        indices,
        key=lambda idx: (
            int((bundle_outputs[idx]["pair"].metadata or {}).get("inline_id", 0)),
            bundle_outputs[idx]["pair"].name,
        ),
    )


def _sorted_indices_by_vintage(bundle_outputs: list[dict[str, Any]], indices: list[int]) -> list[int]:
    return sorted(
        indices,
        key=lambda idx: (
            _sort_metadata_label(_field_group_key(bundle_outputs[idx]["pair"], "vintage", bundle_outputs[idx]["pair"].name)),
            int((bundle_outputs[idx]["pair"].metadata or {}).get("inline_id", 0)),
            bundle_outputs[idx]["pair"].name,
        ),
    )


def _sequence_baseline(outputs: list[dict[str, Any]], indices: list[int]) -> np.ndarray:
    baselines = np.stack([outputs[index]["pair"].baseline.astype(np.float32) for index in indices], axis=0)
    reference = baselines[0]
    if np.allclose(baselines, reference[None, ...], atol=1e-6):
        return reference.astype(np.float32)
    return np.mean(baselines, axis=0).astype(np.float32)


def _cleanup_field_volume(
    binary_volume: np.ndarray,
    reservoir_volume: np.ndarray | None,
    field_cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    pseudo_cfg = field_cfg.get("pseudo3d", {})
    post_cfg = field_cfg.get("postprocess", {})
    binary = binary_volume.astype(bool, copy=True)
    if reservoir_volume is not None and post_cfg.get("apply_reservoir_mask", True):
        binary &= reservoir_volume > 0.5

    structure = np.ones((3, 3, 3), dtype=bool)
    closing_iterations = int(pseudo_cfg.get("closing_iterations", post_cfg.get("closing_iterations", 0)))
    if closing_iterations > 0:
        binary = ndimage.binary_closing(binary, structure=structure, iterations=closing_iterations)

    opening_iterations = int(pseudo_cfg.get("opening_iterations", post_cfg.get("opening_iterations", 0)))
    if opening_iterations > 0:
        binary = ndimage.binary_opening(binary, structure=structure, iterations=opening_iterations)

    keep_largest_components = int(pseudo_cfg.get("keep_largest_components", post_cfg.get("keep_largest_components", 0)))
    min_component_size = int(pseudo_cfg.get("min_component_size", post_cfg.get("min_component_size", 0)))
    min_component_fraction = float(
        pseudo_cfg.get("min_component_fraction", post_cfg.get("min_component_fraction", 0.0))
    )
    min_size_from_fraction = int(np.ceil(binary.size * min_component_fraction))
    component_size_floor = max(min_component_size, min_size_from_fraction)

    labels, num_labels = ndimage.label(binary, structure=structure)
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

    metadata = {
        "volume_applied_reservoir_mask": bool(reservoir_volume is not None and post_cfg.get("apply_reservoir_mask", True)),
        "volume_closing_iterations": closing_iterations,
        "volume_opening_iterations": opening_iterations,
        "volume_component_size_floor": int(component_size_floor),
        "volume_num_components_kept": int(num_components_kept),
    }
    return binary.astype(bool), metadata


def _largest_connected_1d(mask: np.ndarray, *, anchor_index: int | None = None) -> np.ndarray:
    labels, num_labels = ndimage.label(mask.astype(bool))
    if num_labels == 0:
        return np.zeros_like(mask, dtype=bool)

    if anchor_index is not None and 0 <= anchor_index < mask.shape[0] and labels[anchor_index] > 0:
        return labels == labels[anchor_index]

    component_sizes = ndimage.sum(mask.astype(np.float32), labels, index=np.arange(1, num_labels + 1))
    label_id = int(np.argmax(component_sizes)) + 1
    return labels == label_id


def _trace_support_majority(
    trace_support: np.ndarray,
    *,
    window_size: int,
    vote_fraction: float,
) -> np.ndarray:
    if trace_support.shape[0] <= 1 or window_size <= 1:
        return trace_support.astype(bool)

    half_window = window_size // 2
    majority = np.zeros_like(trace_support, dtype=bool)
    for inline_index in range(trace_support.shape[0]):
        lower = max(0, inline_index - half_window)
        upper = min(trace_support.shape[0], inline_index + half_window + 1)
        vote_share = np.mean(trace_support[lower:upper].astype(np.float32), axis=0)
        majority[inline_index] = vote_share >= vote_fraction
    return (trace_support.astype(bool) | majority).astype(bool)


def _temporal_trace_support_majority(
    trace_support: np.ndarray,
    *,
    window_size: int,
    vote_fraction: float,
) -> np.ndarray:
    if trace_support.ndim != 3 or trace_support.shape[0] <= 1 or window_size <= 1:
        return trace_support.astype(bool)

    half_window = window_size // 2
    majority = np.zeros_like(trace_support, dtype=bool)
    for vintage_index in range(trace_support.shape[0]):
        lower = max(0, vintage_index - half_window)
        upper = min(trace_support.shape[0], vintage_index + half_window + 1)
        vote_share = np.mean(trace_support[lower:upper].astype(np.float32), axis=0)
        majority[vintage_index] = vote_share >= vote_fraction
    return (trace_support.astype(bool) | majority).astype(bool)


def _cleanup_field_hypervolume(
    binary_hypervolume: np.ndarray,
    reservoir_hypervolume: np.ndarray | None,
    temporal_cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    binary = binary_hypervolume.astype(bool, copy=True)
    if reservoir_hypervolume is not None:
        binary &= reservoir_hypervolume > 0.5

    structure = np.ones((3, 3, 3, 3), dtype=bool)
    closing_iterations = int(temporal_cfg.get("hypervolume_closing_iterations", 0))
    if closing_iterations > 0:
        binary = ndimage.binary_closing(binary, structure=structure, iterations=closing_iterations)

    opening_iterations = int(temporal_cfg.get("hypervolume_opening_iterations", 0))
    if opening_iterations > 0:
        binary = ndimage.binary_opening(binary, structure=structure, iterations=opening_iterations)

    keep_largest_components = int(temporal_cfg.get("hypervolume_keep_largest_components", 0))
    min_component_size = int(temporal_cfg.get("hypervolume_min_component_size", 0))
    min_component_fraction = float(temporal_cfg.get("hypervolume_min_component_fraction", 0.0))
    min_size_from_fraction = int(np.ceil(binary.size * min_component_fraction))
    component_size_floor = max(min_component_size, min_size_from_fraction)

    labels, num_labels = ndimage.label(binary, structure=structure)
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

    metadata = {
        "hypervolume_closing_iterations": int(closing_iterations),
        "hypervolume_opening_iterations": int(opening_iterations),
        "hypervolume_component_size_floor": int(component_size_floor),
        "hypervolume_num_components_kept": int(num_components_kept),
    }
    return binary.astype(bool), metadata


def _reconstruct_support_column(
    probabilities: np.ndarray,
    seed_binary: np.ndarray,
    reservoir_mask: np.ndarray,
    structured_cfg: dict[str, Any],
    neighbor_probabilities: np.ndarray | None = None,
) -> np.ndarray:
    reservoir_bool = reservoir_mask > 0.5
    if not np.any(reservoir_bool):
        return np.zeros_like(probabilities, dtype=bool)

    base_profile = probabilities.astype(np.float32)
    if neighbor_probabilities is not None:
        base_profile = 0.7 * base_profile + 0.3 * neighbor_probabilities.astype(np.float32)

    sigma = float(structured_cfg.get("vertical_sigma", 0.0))
    if sigma > 1e-3:
        smoothed = ndimage.gaussian_filter1d(base_profile, sigma=sigma, axis=0).astype(np.float32)
    else:
        smoothed = base_profile
    smoothed *= reservoir_bool.astype(np.float32)

    seed_inside = (seed_binary.astype(bool) & reservoir_bool).astype(bool)
    if np.any(seed_inside):
        peak_index = int(np.argmax(np.where(seed_inside, smoothed, -np.inf)))
        peak_value = float(np.max(np.where(seed_inside, smoothed, 0.0)))
    else:
        peak_index = int(np.argmax(smoothed))
        peak_value = float(smoothed[peak_index])

    min_peak_probability = float(structured_cfg.get("min_peak_probability", 0.5))
    if peak_value < min_peak_probability:
        return seed_inside.astype(bool)

    relative_threshold = float(structured_cfg.get("column_relative_threshold", 0.45))
    threshold = max(min_peak_probability * 0.5, peak_value * relative_threshold)
    candidate = (smoothed >= threshold) & reservoir_bool
    candidate = _largest_connected_1d(candidate, anchor_index=peak_index)
    candidate |= seed_inside

    vertical_margin = int(structured_cfg.get("vertical_margin", 0))
    if vertical_margin > 0:
        candidate = ndimage.binary_dilation(candidate, iterations=vertical_margin)
    candidate &= reservoir_bool
    return candidate.astype(bool)


def _reservoir_band_slices(reservoir_mask: np.ndarray, num_bands: int) -> list[np.ndarray]:
    support = np.flatnonzero(reservoir_mask > 0.5)
    if support.size == 0:
        return []
    bands = [band.astype(np.int32) for band in np.array_split(support, max(int(num_bands), 1)) if band.size > 0]
    return bands


def _smooth_active_bands(active_bands: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0 or active_bands.size <= 1:
        return active_bands.astype(bool)
    smoothed = active_bands.astype(bool, copy=True)
    for _ in range(iterations):
        smoothed = ndimage.binary_closing(smoothed, structure=np.ones(3, dtype=bool))
    return smoothed.astype(bool)


def _activate_layer_bands(
    probabilities: np.ndarray,
    seed_binary: np.ndarray,
    reservoir_mask: np.ndarray,
    layered_cfg: dict[str, Any],
    *,
    neighbor_probabilities: np.ndarray | None = None,
) -> np.ndarray:
    band_indices = _reservoir_band_slices(reservoir_mask, int(layered_cfg.get("num_bands", 4)))
    if not band_indices:
        return np.zeros(0, dtype=bool)

    probability_profile = probabilities.astype(np.float32)
    if neighbor_probabilities is not None:
        weight = float(layered_cfg.get("neighbor_probability_weight", 0.3))
        probability_profile = (
            (1.0 - weight) * probability_profile + weight * neighbor_probabilities.astype(np.float32)
        ).astype(np.float32)
    probability_profile *= (reservoir_mask > 0.5).astype(np.float32)

    min_peak_probability = float(layered_cfg.get("min_band_peak_probability", 0.5))
    min_mean_probability = float(layered_cfg.get("min_band_mean_probability", 0.18))
    min_seed_fraction = float(layered_cfg.get("min_seed_fraction", 0.02))

    active = np.zeros(len(band_indices), dtype=bool)
    band_scores = np.zeros(len(band_indices), dtype=np.float32)
    for band_idx, indices in enumerate(band_indices):
        band_probabilities = probability_profile[indices]
        band_seed = seed_binary[indices].astype(np.float32)
        peak_probability = float(np.max(band_probabilities)) if band_probabilities.size > 0 else 0.0
        mean_probability = float(np.mean(band_probabilities)) if band_probabilities.size > 0 else 0.0
        seed_fraction = float(np.mean(band_seed)) if band_seed.size > 0 else 0.0
        band_scores[band_idx] = max(peak_probability, mean_probability, seed_fraction)
        active[band_idx] = (
            peak_probability >= min_peak_probability
            or mean_probability >= min_mean_probability
            or seed_fraction >= min_seed_fraction
        )

    if not np.any(active):
        if np.any(seed_binary & (reservoir_mask > 0.5)):
            for band_idx, indices in enumerate(band_indices):
                if np.any(seed_binary[indices]):
                    active[band_idx] = True
        else:
            active[int(np.argmax(band_scores))] = True

    smoothing_iterations = int(layered_cfg.get("band_smoothing_iterations", 1))
    return _smooth_active_bands(active, smoothing_iterations)


def _fill_active_layer_bands(
    reservoir_mask: np.ndarray,
    active_bands: np.ndarray,
    num_bands: int,
) -> np.ndarray:
    reconstructed = np.zeros_like(reservoir_mask, dtype=bool)
    for band_active, indices in zip(active_bands.astype(bool), _reservoir_band_slices(reservoir_mask, num_bands)):
        if band_active:
            reconstructed[indices] = True
    return reconstructed.astype(bool)


def _apply_layered_structured_support_reconstruction(
    outputs: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    method_prefix: str,
) -> dict[str, dict[str, Any]]:
    layered_cfg = config.get("field", {}).get("layered_structured_support", {})
    if not layered_cfg.get("enabled", False):
        return {}

    grouped_indices = _group_output_indices(outputs, "vintage")
    results: dict[str, dict[str, Any]] = {}
    trace_window_size = max(int(layered_cfg.get("trace_window_size", 3)), 1)
    trace_vote_fraction = float(layered_cfg.get("trace_vote_fraction", 0.34))
    probability_key = f"{method_prefix}_probs"
    uncertainty_key = f"{method_prefix}_uncertainty"
    structured_seed_key = f"{method_prefix}_ml_structured_constrained_binary"
    constrained_seed_key = f"{method_prefix}_ml_constrained_binary"

    for _vintage_key, indices in grouped_indices.items():
        ordered_indices = _sorted_indices_by_inline(outputs, indices)
        probabilities = np.stack([outputs[idx][probability_key].astype(np.float32) for idx in ordered_indices], axis=0)
        uncertainties = np.stack([outputs[idx][uncertainty_key].astype(np.float32) for idx in ordered_indices], axis=0)
        reservoir_volume = np.stack(
            [
                (
                    outputs[idx]["pair"].reservoir_mask.astype(np.float32)
                    if outputs[idx]["pair"].reservoir_mask is not None
                    else np.ones_like(outputs[idx]["pair"].baseline, dtype=np.float32)
                )
                for idx in ordered_indices
            ],
            axis=0,
        )
        support_volume = np.stack(
            [
                (
                    outputs[idx]["support_volume"].astype(np.float32)
                    if outputs[idx]["support_volume"] is not None
                    else reservoir_volume[position]
                )
                for position, idx in enumerate(ordered_indices)
            ],
            axis=0,
        )
        base_binaries = np.stack(
            [
                outputs[idx].get(structured_seed_key, outputs[idx][constrained_seed_key]).astype(bool)
                for idx in ordered_indices
            ],
            axis=0,
        )

        trace_support = np.any(base_binaries, axis=1)
        layered_trace_support = _trace_support_majority(
            trace_support,
            window_size=trace_window_size,
            vote_fraction=trace_vote_fraction,
        )

        reconstructed_volume = np.zeros_like(base_binaries, dtype=bool)
        for position in range(len(ordered_indices)):
            lower = max(0, position - 1)
            upper = min(len(ordered_indices), position + 2)
            neighbor_probability = np.mean(probabilities[lower:upper], axis=0).astype(np.float32)
            for trace_index in np.where(layered_trace_support[position])[0]:
                reservoir_column = reservoir_volume[position, :, trace_index]
                if not np.any(reservoir_column > 0.5):
                    continue
                support_column = support_volume[position, :, trace_index]
                if np.any(support_column > 0.5):
                    reservoir_column = (reservoir_column > 0.5) & (support_column > 0.5)
                active_bands = _activate_layer_bands(
                    probabilities[position, :, trace_index],
                    base_binaries[position, :, trace_index],
                    reservoir_column.astype(np.float32),
                    layered_cfg,
                    neighbor_probabilities=neighbor_probability[:, trace_index],
                )
                reconstructed_volume[position, :, trace_index] = _fill_active_layer_bands(
                    reservoir_column.astype(np.float32),
                    active_bands,
                    int(layered_cfg.get("num_bands", 4)),
                )

        layered_cleanup_cfg = {
            "postprocess": config.get("field", {}).get("postprocess", {}),
            "pseudo3d": {
                "closing_iterations": int(layered_cfg.get("closing_iterations", 1)),
                "opening_iterations": int(layered_cfg.get("opening_iterations", 0)),
                "min_component_size": int(layered_cfg.get("min_component_size", 0)),
                "min_component_fraction": float(layered_cfg.get("min_component_fraction", 0.0)),
                "keep_largest_components": int(layered_cfg.get("keep_largest_components", 0)),
            },
        }
        cleaned_volume, volume_metadata = _cleanup_field_volume(
            reconstructed_volume,
            reservoir_volume,
            layered_cleanup_cfg,
        )
        for position, index in enumerate(ordered_indices):
            pair = outputs[index]["pair"]
            results[pair.name] = {
                "binary": cleaned_volume[position].astype(bool),
                "uncertainty": uncertainties[position].astype(np.float32),
                "metadata": {
                    "enabled": True,
                    "layered_structured_support_enabled": True,
                    "trace_window_size": int(trace_window_size),
                    "trace_vote_fraction": float(trace_vote_fraction),
                    "num_bands": int(layered_cfg.get("num_bands", 4)),
                    "min_band_peak_probability": float(layered_cfg.get("min_band_peak_probability", 0.5)),
                    "min_band_mean_probability": float(layered_cfg.get("min_band_mean_probability", 0.18)),
                    "min_seed_fraction": float(layered_cfg.get("min_seed_fraction", 0.02)),
                    "neighbor_probability_weight": float(layered_cfg.get("neighbor_probability_weight", 0.3)),
                    **volume_metadata,
                },
            }
    return results


def _apply_structured_support_reconstruction(
    outputs: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    method_prefix: str,
) -> dict[str, dict[str, Any]]:
    structured_cfg = config.get("field", {}).get("structured_support", {})
    if not structured_cfg.get("enabled", False):
        return {}

    grouped_indices = _group_output_indices(outputs, "vintage")
    results: dict[str, dict[str, Any]] = {}
    trace_window_size = max(int(structured_cfg.get("trace_window_size", 3)), 1)
    trace_vote_fraction = float(structured_cfg.get("trace_vote_fraction", 0.34))
    base_binary_key = f"{method_prefix}_ml_constrained_binary"
    uncertainty_key = f"{method_prefix}_uncertainty"
    probability_key = f"{method_prefix}_probs"

    for _vintage_key, indices in grouped_indices.items():
        ordered_indices = _sorted_indices_by_inline(outputs, indices)
        base_binaries = np.stack([outputs[idx][base_binary_key].astype(bool) for idx in ordered_indices], axis=0)
        probabilities = np.stack([outputs[idx][probability_key].astype(np.float32) for idx in ordered_indices], axis=0)
        uncertainties = np.stack([outputs[idx][uncertainty_key].astype(np.float32) for idx in ordered_indices], axis=0)
        reservoir_volume = np.stack(
            [
                (
                    outputs[idx]["pair"].reservoir_mask.astype(np.float32)
                    if outputs[idx]["pair"].reservoir_mask is not None
                    else np.ones_like(outputs[idx]["pair"].baseline, dtype=np.float32)
                )
                for idx in ordered_indices
            ],
            axis=0,
        )

        trace_support = np.any(base_binaries, axis=1)
        structured_trace_support = _trace_support_majority(
            trace_support,
            window_size=trace_window_size,
            vote_fraction=trace_vote_fraction,
        )

        reconstructed_volume = np.zeros_like(base_binaries, dtype=bool)
        for position in range(len(ordered_indices)):
            lower = max(0, position - 1)
            upper = min(len(ordered_indices), position + 2)
            neighbor_profile = np.mean(probabilities[lower:upper], axis=0)
            for trace_index in np.where(structured_trace_support[position])[0]:
                reconstructed_volume[position, :, trace_index] = _reconstruct_support_column(
                    probabilities[position, :, trace_index],
                    base_binaries[position, :, trace_index],
                    reservoir_volume[position, :, trace_index],
                    structured_cfg,
                    neighbor_probabilities=neighbor_profile[:, trace_index],
                )

        structured_cleanup_cfg = {
            "postprocess": config.get("field", {}).get("postprocess", {}),
            "pseudo3d": {
                "closing_iterations": int(structured_cfg.get("closing_iterations", 1)),
                "opening_iterations": int(structured_cfg.get("opening_iterations", 0)),
                "min_component_size": int(structured_cfg.get("min_component_size", 0)),
                "min_component_fraction": float(structured_cfg.get("min_component_fraction", 0.0)),
                "keep_largest_components": int(structured_cfg.get("keep_largest_components", 0)),
            },
        }
        cleaned_volume, volume_metadata = _cleanup_field_volume(
            reconstructed_volume,
            reservoir_volume,
            structured_cleanup_cfg,
        )
        for position, index in enumerate(ordered_indices):
            pair = outputs[index]["pair"]
            results[pair.name] = {
                "binary": cleaned_volume[position].astype(bool),
                "uncertainty": uncertainties[position].astype(np.float32),
                "metadata": {
                    "enabled": True,
                    "structured_support_enabled": True,
                    "trace_window_size": int(trace_window_size),
                    "trace_vote_fraction": float(trace_vote_fraction),
                    "column_relative_threshold": float(structured_cfg.get("column_relative_threshold", 0.45)),
                    "min_peak_probability": float(structured_cfg.get("min_peak_probability", 0.5)),
                    "vertical_margin": int(structured_cfg.get("vertical_margin", 0)),
                    "vertical_sigma": float(structured_cfg.get("vertical_sigma", 0.0)),
                    **volume_metadata,
                },
            }
    return results


def _apply_temporal_structured_support_inference(
    outputs: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    method_prefix: str,
    seed_results: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    temporal_cfg = config.get("field", {}).get("temporal_structured_support", {})
    if not temporal_cfg.get("enabled", False):
        return {}

    structured_cfg = config.get("field", {}).get("structured_support", {})
    vintage_groups = _group_output_indices(outputs, "vintage")
    if not vintage_groups:
        return {}

    ordered_vintages = sorted(vintage_groups.keys(), key=_sort_metadata_label)
    inline_ids = sorted(
        {
            _field_group_key(output["pair"], "inline_id", output["pair"].name)
            for output in outputs
        },
        key=_sort_metadata_label,
    )
    if not ordered_vintages or not inline_ids:
        return {}

    probability_key = f"{method_prefix}_probs"
    uncertainty_key = f"{method_prefix}_uncertainty"
    constrained_seed_key = f"{method_prefix}_ml_constrained_binary"
    vintage_index = {value: idx for idx, value in enumerate(ordered_vintages)}
    inline_index = {value: idx for idx, value in enumerate(inline_ids)}
    nt, nx = outputs[0]["pair"].baseline.shape
    probabilities = np.full((len(ordered_vintages), len(inline_ids), nt, nx), np.nan, dtype=np.float32)
    uncertainties = np.full_like(probabilities, np.nan, dtype=np.float32)
    reservoir_volume = np.zeros_like(probabilities, dtype=np.float32)
    support_volume = np.zeros_like(probabilities, dtype=np.float32)
    base_binaries = np.zeros_like(probabilities, dtype=bool)
    available = np.zeros((len(ordered_vintages), len(inline_ids)), dtype=bool)
    pair_name_grid = np.full((len(ordered_vintages), len(inline_ids)), "", dtype=object)

    for output in outputs:
        pair = output["pair"]
        vintage_label = _field_group_key(pair, "vintage", pair.name)
        inline_label = _field_group_key(pair, "inline_id", pair.name)
        vintage_idx = vintage_index[vintage_label]
        inline_idx = inline_index[inline_label]
        probabilities[vintage_idx, inline_idx] = output[probability_key].astype(np.float32)
        uncertainties[vintage_idx, inline_idx] = output[uncertainty_key].astype(np.float32)
        reservoir_volume[vintage_idx, inline_idx] = (
            pair.reservoir_mask.astype(np.float32)
            if pair.reservoir_mask is not None
            else np.ones_like(pair.baseline, dtype=np.float32)
        )
        support_volume[vintage_idx, inline_idx] = (
            output["support_volume"].astype(np.float32)
            if output.get("support_volume") is not None
            else np.zeros_like(pair.baseline, dtype=np.float32)
        )
        seed_entry = seed_results.get(pair.name)
        base_binaries[vintage_idx, inline_idx] = (
            seed_entry["binary"].astype(bool)
            if seed_entry is not None
            else output[constrained_seed_key].astype(bool)
        )
        available[vintage_idx, inline_idx] = True
        pair_name_grid[vintage_idx, inline_idx] = pair.name

    base_trace_support = np.any(base_binaries, axis=2)
    spatial_trace_support = np.zeros_like(base_trace_support, dtype=bool)
    spatial_window = max(int(temporal_cfg.get("trace_window_size", structured_cfg.get("trace_window_size", 3))), 1)
    spatial_vote_fraction = float(
        temporal_cfg.get("trace_vote_fraction", structured_cfg.get("trace_vote_fraction", 0.34))
    )
    for vintage_idx in range(len(ordered_vintages)):
        spatial_trace_support[vintage_idx] = _trace_support_majority(
            base_trace_support[vintage_idx],
            window_size=spatial_window,
            vote_fraction=spatial_vote_fraction,
        )
    temporal_trace_support = _temporal_trace_support_majority(
        base_trace_support,
        window_size=max(int(temporal_cfg.get("temporal_window_size", 3)), 1),
        vote_fraction=float(temporal_cfg.get("temporal_trace_vote_fraction", 0.34)),
    )
    combined_trace_support = base_trace_support | spatial_trace_support | temporal_trace_support
    if bool(temporal_cfg.get("enforce_monotone_growth", True)) and combined_trace_support.shape[0] > 1:
        combined_trace_support = np.maximum.accumulate(combined_trace_support, axis=0)

    inline_probability_weight = float(temporal_cfg.get("inline_probability_weight", 0.2))
    temporal_probability_weight = float(temporal_cfg.get("temporal_probability_weight", 0.35))
    support_volume_weight = float(temporal_cfg.get("support_volume_weight", 0.08))
    inline_half_window = max(int(temporal_cfg.get("inline_window_size", 3)), 1) // 2
    temporal_half_window = max(int(temporal_cfg.get("temporal_window_size", 3)), 1) // 2
    blended_probabilities = np.full_like(probabilities, np.nan, dtype=np.float32)
    blended_uncertainties = np.full_like(probabilities, np.nan, dtype=np.float32)

    for vintage_idx in range(len(ordered_vintages)):
        for inline_idx in range(len(inline_ids)):
            if not available[vintage_idx, inline_idx]:
                continue
            blended = probabilities[vintage_idx, inline_idx].astype(np.float32).copy()
            weight_total = 1.0
            uncertainty_bonus = np.zeros_like(blended, dtype=np.float32)
            if inline_probability_weight > 0.0 and len(inline_ids) > 1:
                lower = max(0, inline_idx - inline_half_window)
                upper = min(len(inline_ids), inline_idx + inline_half_window + 1)
                inline_window = probabilities[vintage_idx, lower:upper]
                inline_mean = np.nanmean(inline_window, axis=0)
                inline_std = np.nanstd(inline_window, axis=0)
                if not np.all(np.isnan(inline_mean)):
                    blended += inline_probability_weight * np.nan_to_num(inline_mean, nan=0.0)
                    uncertainty_bonus += np.nan_to_num(inline_std, nan=0.0).astype(np.float32)
                    weight_total += inline_probability_weight
            if temporal_probability_weight > 0.0 and len(ordered_vintages) > 1:
                lower = max(0, vintage_idx - temporal_half_window)
                upper = min(len(ordered_vintages), vintage_idx + temporal_half_window + 1)
                temporal_window = probabilities[lower:upper, inline_idx]
                temporal_mean = np.nanmean(temporal_window, axis=0)
                temporal_std = np.nanstd(temporal_window, axis=0)
                if not np.all(np.isnan(temporal_mean)):
                    blended += temporal_probability_weight * np.nan_to_num(temporal_mean, nan=0.0)
                    uncertainty_bonus += np.nan_to_num(temporal_std, nan=0.0).astype(np.float32)
                    weight_total += temporal_probability_weight
            blended = (blended / max(weight_total, 1e-6)).astype(np.float32)
            if support_volume_weight > 0.0:
                blended = np.clip(
                    blended + support_volume_weight * support_volume[vintage_idx, inline_idx].astype(np.float32),
                    0.0,
                    1.0,
                ).astype(np.float32)
            blended_probabilities[vintage_idx, inline_idx] = blended
            blended_uncertainties[vintage_idx, inline_idx] = (
                uncertainties[vintage_idx, inline_idx].astype(np.float32) + uncertainty_bonus
            ).astype(np.float32)

    column_cfg = {
        "vertical_sigma": float(temporal_cfg.get("vertical_sigma", structured_cfg.get("vertical_sigma", 1.0))),
        "column_relative_threshold": float(
            temporal_cfg.get("column_relative_threshold", structured_cfg.get("column_relative_threshold", 0.45))
        ),
        "min_peak_probability": float(
            temporal_cfg.get("min_peak_probability", structured_cfg.get("min_peak_probability", 0.5))
        ),
        "vertical_margin": int(temporal_cfg.get("vertical_margin", structured_cfg.get("vertical_margin", 2))),
    }
    temporal_dilation = max(int(temporal_cfg.get("temporal_dilation", 1)), 0)
    emergence_probability_threshold = float(temporal_cfg.get("emergence_probability_threshold", 0.52))
    enforce_monotone_growth = bool(temporal_cfg.get("enforce_monotone_growth", True))
    reconstructed = np.zeros_like(base_binaries, dtype=bool)

    for vintage_idx in range(len(ordered_vintages)):
        for inline_idx in range(len(inline_ids)):
            if not available[vintage_idx, inline_idx]:
                continue
            for trace_index in np.where(combined_trace_support[vintage_idx, inline_idx])[0]:
                reservoir_column = reservoir_volume[vintage_idx, inline_idx, :, trace_index]
                seed_column = base_binaries[vintage_idx, inline_idx, :, trace_index]
                candidate = _reconstruct_support_column(
                    blended_probabilities[vintage_idx, inline_idx, :, trace_index],
                    seed_column,
                    reservoir_column,
                    column_cfg,
                )
                if vintage_idx > 0:
                    previous_column = reconstructed[vintage_idx - 1, inline_idx, :, trace_index]
                    if np.any(previous_column):
                        dilated_previous = (
                            ndimage.binary_dilation(previous_column.astype(bool), iterations=temporal_dilation)
                            if temporal_dilation > 0
                            else previous_column.astype(bool)
                        )
                        if enforce_monotone_growth:
                            candidate |= previous_column.astype(bool)
                        new_support = candidate & ~previous_column.astype(bool)
                        if np.any(new_support):
                            strong_new = (
                                blended_probabilities[vintage_idx, inline_idx, :, trace_index]
                                >= emergence_probability_threshold
                            )
                            candidate = previous_column.astype(bool) | (new_support & (dilated_previous | strong_new))
                candidate &= reservoir_column > 0.5
                reconstructed[vintage_idx, inline_idx, :, trace_index] = candidate.astype(bool)

    cleaned_hypervolume, hypervolume_metadata = _cleanup_field_hypervolume(
        reconstructed,
        reservoir_volume,
        temporal_cfg,
    )

    results: dict[str, dict[str, Any]] = {}
    for vintage_idx in range(len(ordered_vintages)):
        for inline_idx in range(len(inline_ids)):
            if not available[vintage_idx, inline_idx]:
                continue
            pair_name = str(pair_name_grid[vintage_idx, inline_idx])
            results[pair_name] = {
                "binary": cleaned_hypervolume[vintage_idx, inline_idx].astype(bool),
                "uncertainty": blended_uncertainties[vintage_idx, inline_idx].astype(np.float32),
                "metadata": {
                    "enabled": True,
                    "temporal_structured_support_enabled": True,
                    "trace_window_size": int(spatial_window),
                    "trace_vote_fraction": float(spatial_vote_fraction),
                    "temporal_window_size": int(max(int(temporal_cfg.get("temporal_window_size", 3)), 1)),
                    "temporal_trace_vote_fraction": float(temporal_cfg.get("temporal_trace_vote_fraction", 0.34)),
                    "inline_window_size": int(max(int(temporal_cfg.get("inline_window_size", 3)), 1)),
                    "inline_probability_weight": float(inline_probability_weight),
                    "temporal_probability_weight": float(temporal_probability_weight),
                    "support_volume_weight": float(support_volume_weight),
                    "vertical_sigma": float(column_cfg["vertical_sigma"]),
                    "column_relative_threshold": float(column_cfg["column_relative_threshold"]),
                    "min_peak_probability": float(column_cfg["min_peak_probability"]),
                    "vertical_margin": int(column_cfg["vertical_margin"]),
                    "enforce_monotone_growth": bool(enforce_monotone_growth),
                    "temporal_dilation": int(temporal_dilation),
                    "emergence_probability_threshold": float(emergence_probability_threshold),
                    **hypervolume_metadata,
                },
            }
    return results


def _apply_pseudo3d_consistency(
    outputs: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    method_prefix: str,
    shared_context: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    pseudo_cfg = config.get("field", {}).get("pseudo3d", {})
    if not pseudo_cfg.get("enabled", False):
        return {}

    window_size = max(int(pseudo_cfg.get("window_size", 3)), 1)
    half_window = window_size // 2
    method = str(pseudo_cfg.get("method", "uncertainty_weighted_mean")).lower()
    grouped_indices = _group_output_indices(outputs, "vintage")
    results: dict[str, dict[str, Any]] = {}

    for _vintage_key, indices in grouped_indices.items():
        ordered_indices = _sorted_indices_by_inline(outputs, indices)
        probabilities = np.stack([outputs[idx][f"{method_prefix}_probs"] for idx in ordered_indices], axis=0)
        uncertainties = np.stack([outputs[idx][f"{method_prefix}_uncertainty"] for idx in ordered_indices], axis=0)
        reservoir_volume = np.stack(
            [
                (
                    outputs[idx]["pair"].reservoir_mask.astype(np.float32)
                    if outputs[idx]["pair"].reservoir_mask is not None
                    else np.ones_like(outputs[idx]["pair"].baseline, dtype=np.float32)
                )
                for idx in ordered_indices
            ],
            axis=0,
        )

        smoothed_probabilities = np.zeros_like(probabilities, dtype=np.float32)
        smoothed_uncertainties = np.zeros_like(uncertainties, dtype=np.float32)
        for position in range(len(ordered_indices)):
            lower = max(0, position - half_window)
            upper = min(len(ordered_indices), position + half_window + 1)
            neighborhood_probabilities = probabilities[lower:upper]
            neighborhood_uncertainties = uncertainties[lower:upper]
            if method == "median":
                smoothed_probabilities[position] = np.median(neighborhood_probabilities, axis=0).astype(np.float32)
            else:
                if method_prefix == "hybrid" and not np.allclose(neighborhood_uncertainties, 0.0):
                    weights = 1.0 / np.clip(neighborhood_uncertainties, 1e-4, None)
                else:
                    weights = np.ones_like(neighborhood_probabilities, dtype=np.float32)
                smoothed_probabilities[position] = (
                    np.sum(neighborhood_probabilities * weights, axis=0) / np.sum(weights, axis=0)
                ).astype(np.float32)
            smoothed_uncertainties[position] = (
                np.mean(neighborhood_uncertainties, axis=0) + np.std(neighborhood_probabilities, axis=0)
            ).astype(np.float32)

        per_inline_binaries: list[np.ndarray] = []
        per_inline_metadata: list[dict[str, Any]] = []
        for position, index in enumerate(ordered_indices):
            pair = outputs[index]["pair"]
            binary, metadata = _postprocess_field_prediction(
                smoothed_probabilities[position],
                smoothed_uncertainties[position],
                pair.reservoir_mask,
                config.get("field", {}),
                shared_context=shared_context,
            )
            per_inline_binaries.append(binary.astype(bool))
            per_inline_metadata.append(metadata)

        binary_volume = np.stack(per_inline_binaries, axis=0)
        cleaned_volume, volume_metadata = _cleanup_field_volume(binary_volume, reservoir_volume, config.get("field", {}))
        for position, index in enumerate(ordered_indices):
            pair = outputs[index]["pair"]
            results[pair.name] = {
                "binary": cleaned_volume[position].astype(bool),
                "uncertainty": smoothed_uncertainties[position].astype(np.float32),
                "metadata": {
                    **per_inline_metadata[position],
                    **volume_metadata,
                    "enabled": True,
                    "pseudo3d_enabled": True,
                    "pseudo3d_method": method,
                    "pseudo3d_window_size": int(window_size),
                },
            }
    return results


def _crossline_continuity(trace_masks: np.ndarray) -> float:
    if trace_masks.shape[0] < 2:
        return float("nan")
    adjacent_ious = [
        _binary_iou(trace_masks[index], trace_masks[index + 1])
        for index in range(trace_masks.shape[0] - 1)
    ]
    valid = [value for value in adjacent_ious if not np.isnan(value)]
    if not valid:
        return float("nan")
    return float(np.mean(valid))


def _connected_component_count(binary_volume: np.ndarray) -> int:
    labels, num_labels = ndimage.label(binary_volume.astype(bool), structure=np.ones((3, 3, 3), dtype=bool))
    return int(num_labels)


def _volume_trace_support_metrics(
    binary_volume: np.ndarray,
    support_trace_volume: np.ndarray | None,
) -> dict[str, Any]:
    predicted_trace_mask = np.any(binary_volume.astype(bool), axis=1)
    metrics = {
        "crossline_continuity": _crossline_continuity(predicted_trace_mask),
    }
    if support_trace_volume is None:
        return metrics

    support_trace_mask = support_trace_volume.astype(bool)
    predicted_count = int(np.sum(predicted_trace_mask))
    support_count = int(np.sum(support_trace_mask))
    outside_count = int(np.sum(predicted_trace_mask & ~support_trace_mask))
    inside_count = int(np.sum(predicted_trace_mask & support_trace_mask))
    metrics.update(
        {
            "predicted_trace_fraction": float(np.mean(predicted_trace_mask)),
            "support_trace_fraction_2010": float(np.mean(support_trace_mask)),
            "trace_fraction_outside_2010_support": float(outside_count / max(predicted_count, 1)),
            "trace_fraction_inside_2010_support": float(inside_count / max(predicted_count, 1)),
            "trace_iou_with_2010_support": _binary_iou(predicted_trace_mask, support_trace_mask),
            "trace_support_coverage_vs_2010": float(inside_count / max(support_count, 1)),
            "crossline_continuity": _crossline_continuity(predicted_trace_mask),
        }
    )
    return metrics


def _summarize_volume_binary(
    binary_volume: np.ndarray,
    uncertainty_volume: np.ndarray,
    reservoir_volume: np.ndarray | None,
    support_trace_volume: np.ndarray | None,
    support_volume: np.ndarray | None,
) -> dict[str, Any]:
    return {
        **_summarize_field_binary(binary_volume, uncertainty_volume, reservoir_volume),
        **_volume_trace_support_metrics(binary_volume, support_trace_volume),
        **_support_volume_metrics(binary_volume, support_volume),
        "connected_component_count": _connected_component_count(binary_volume),
    }


def _mean_numeric_dict(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {}
    keys = {key for item in items for key in item.keys()}
    aggregated: dict[str, Any] = {}
    for key in sorted(keys):
        values = [item.get(key) for item in items if isinstance(item.get(key), (int, float, np.floating, np.integer))]
        if not values:
            continue
        values_array = np.asarray(values, dtype=np.float64)
        if np.all(np.isnan(values_array)):
            aggregated[key] = float("nan")
        else:
            aggregated[key] = float(np.nanmean(values_array))
    return aggregated


def _build_volume_summary(
    outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    if not outputs:
        return {}

    methods: dict[str, tuple[str, str]] = {
        "best_classical_constrained": ("best_classical_constrained_binary", "best_classical_constrained_uncertainty"),
        "plain_ml_constrained": ("plain_ml_constrained_binary", "plain_ml_constrained_uncertainty"),
        "plain_ml_pseudo3d_constrained": ("plain_ml_pseudo3d_constrained_binary", "plain_ml_pseudo3d_constrained_uncertainty"),
        "plain_ml_temporal_structured_constrained": (
            "plain_ml_temporal_structured_constrained_binary",
            "plain_ml_temporal_structured_constrained_uncertainty",
        ),
        "plain_ml_structured_constrained": (
            "plain_ml_structured_constrained_binary",
            "plain_ml_structured_constrained_uncertainty",
        ),
        "plain_ml_layered_structured_constrained": (
            "plain_ml_layered_structured_constrained_binary",
            "plain_ml_layered_structured_constrained_uncertainty",
        ),
        "temporal_ml_constrained": ("temporal_ml_constrained_binary", "temporal_ml_constrained_uncertainty"),
        "wave_temporal_constrained": ("wave_temporal_constrained_binary", "wave_temporal_constrained_uncertainty"),
        "hybrid_ml_constrained": ("hybrid_ml_constrained_binary", "hybrid_ml_constrained_uncertainty"),
        "hybrid_ml_structured_constrained": (
            "hybrid_ml_structured_constrained_binary",
            "hybrid_ml_structured_constrained_uncertainty",
        ),
        "hybrid_ml_pseudo3d_constrained": (
            "hybrid_ml_pseudo3d_constrained_binary",
            "hybrid_ml_pseudo3d_constrained_uncertainty",
        ),
    }
    available_methods = {
        method_name: keys
        for method_name, keys in methods.items()
        if all(keys[0] in output and keys[1] in output for output in outputs)
    }
    by_vintage: dict[str, dict[str, Any]] = {}
    grouped_indices = _group_output_indices(outputs, "vintage")
    for vintage_key, indices in grouped_indices.items():
        ordered_indices = _sorted_indices_by_inline(outputs, indices)
        reservoir_volume = np.stack(
            [
                (
                    outputs[index]["pair"].reservoir_mask.astype(np.float32)
                    if outputs[index]["pair"].reservoir_mask is not None
                    else np.ones_like(outputs[index]["pair"].baseline, dtype=np.float32)
                )
                for index in ordered_indices
            ],
            axis=0,
        )
        support_trace_volume = np.stack(
            [
                outputs[index]["pair_support_traces"].astype(np.float32)
                if outputs[index]["pair_support_traces"] is not None
                else np.zeros(outputs[index]["pair"].baseline.shape[1], dtype=np.float32)
                for index in ordered_indices
            ],
            axis=0,
        )
        support_volume = np.stack(
            [
                outputs[index]["support_volume"].astype(np.float32)
                if outputs[index]["support_volume"] is not None
                else np.zeros_like(outputs[index]["pair"].baseline, dtype=np.float32)
                for index in ordered_indices
            ],
            axis=0,
        )
        by_vintage[vintage_key] = {}
        for method_name, (binary_key, uncertainty_key) in available_methods.items():
            binary_volume = np.stack([outputs[index][binary_key].astype(bool) for index in ordered_indices], axis=0)
            uncertainty_volume = np.stack(
                [outputs[index][uncertainty_key].astype(np.float32) for index in ordered_indices],
                axis=0,
            )
            by_vintage[vintage_key][method_name] = _summarize_volume_binary(
                binary_volume,
                uncertainty_volume,
                reservoir_volume,
                support_trace_volume,
                support_volume,
            )

    overall: dict[str, Any] = {}
    for method_name in next(iter(by_vintage.values())).keys():
        overall[method_name] = _mean_numeric_dict([metrics[method_name] for metrics in by_vintage.values()])
    return {
        "by_vintage": by_vintage,
        "overall": overall,
    }


def _build_temporal_consistency(
    pairs: dict[str, FieldPair],
    constrained_binaries: dict[str, np.ndarray],
    constrained_fractions: dict[str, float],
    constrained_trace_fractions: dict[str, float],
    constrained_support_iou: dict[str, float],
) -> dict[str, Any]:
    if len(constrained_binaries) < 2:
        return {}

    inline_groups: dict[str, list[str]] = {}
    for name, pair in pairs.items():
        inline_key = str(pair.metadata.get("inline_id")) if pair.metadata and pair.metadata.get("inline_id") is not None else "default"
        inline_groups.setdefault(inline_key, []).append(name)

    per_inline: dict[str, Any] = {}
    all_trace_monotone = True
    all_support_monotone = True
    any_group = False

    for inline_key, names in inline_groups.items():
        ordered_names = sorted(
            names,
            key=lambda item: (
                str(pairs[item].metadata.get("vintage")) if pairs[item].metadata and pairs[item].metadata.get("vintage") is not None else item
            ),
        )
        if len(ordered_names) < 2:
            continue
        any_group = True
        consecutive_pairs = list(zip(ordered_names[:-1], ordered_names[1:]))
        area_deltas = {
            f"{earlier}->{later}": float(constrained_fractions[later] - constrained_fractions[earlier])
            for earlier, later in consecutive_pairs
        }
        trace_fraction_deltas = {
            f"{earlier}->{later}": float(constrained_trace_fractions[later] - constrained_trace_fractions[earlier])
            for earlier, later in consecutive_pairs
            if not np.isnan(constrained_trace_fractions[earlier]) and not np.isnan(constrained_trace_fractions[later])
        }
        pairwise_iou = {
            f"{earlier}<->{later}": _binary_iou(constrained_binaries[earlier], constrained_binaries[later])
            for earlier, later in consecutive_pairs
        }
        support_iou_progression = {
            name: float(value) for name, value in constrained_support_iou.items() if name in ordered_names and not np.isnan(value)
        }
        support_non_decreasing = (
            all(
                support_iou_progression[later] >= support_iou_progression[earlier]
                for earlier, later in consecutive_pairs
                if earlier in support_iou_progression and later in support_iou_progression
            )
            if support_iou_progression
            else False
        )
        trace_non_decreasing = all(delta >= 0.0 for delta in trace_fraction_deltas.values()) if trace_fraction_deltas else False
        all_trace_monotone &= trace_non_decreasing
        all_support_monotone &= support_non_decreasing
        per_inline[inline_key] = {
            "ordered_pairs": ordered_names,
            "constrained_area_deltas": area_deltas,
            "constrained_trace_fraction_deltas": trace_fraction_deltas,
            "constrained_pairwise_iou": pairwise_iou,
            "constrained_area_non_decreasing": all(delta >= 0.0 for delta in area_deltas.values()),
            "constrained_trace_fraction_non_decreasing": trace_non_decreasing,
            "constrained_trace_fraction_by_pair": {
                name: constrained_trace_fractions[name] for name in ordered_names if name in constrained_trace_fractions
            },
            "constrained_support_iou_by_pair": support_iou_progression,
            "constrained_support_iou_non_decreasing": support_non_decreasing,
        }

    if not any_group:
        return {}

    if len(per_inline) == 1:
        only_value = dict(next(iter(per_inline.values())))
        only_value["per_inline"] = {key: dict(value) for key, value in per_inline.items()}
        only_value["all_inlines_trace_fraction_non_decreasing"] = all_trace_monotone
        only_value["all_inlines_support_iou_non_decreasing"] = all_support_monotone
        return only_value

    return {
        "per_inline": per_inline,
        "all_inlines_trace_fraction_non_decreasing": all_trace_monotone,
        "all_inlines_support_iou_non_decreasing": all_support_monotone,
    }


def _build_temporal_volume_consistency(volume_summary: dict[str, Any]) -> dict[str, Any]:
    by_vintage = volume_summary.get("by_vintage", {})
    if len(by_vintage) < 2:
        return {}
    ordered_vintages = sorted(by_vintage, key=lambda value: int(value) if str(value).isdigit() else str(value))
    methods = list(next(iter(by_vintage.values())).keys())
    consistency: dict[str, Any] = {"ordered_vintages": ordered_vintages, "methods": {}}
    for method_name in methods:
        area_series = {vintage: float(by_vintage[vintage][method_name].get("predicted_fraction", float("nan"))) for vintage in ordered_vintages}
        trace_series = {
            vintage: float(by_vintage[vintage][method_name].get("predicted_trace_fraction", float("nan")))
            for vintage in ordered_vintages
        }
        support_series = {
            vintage: float(by_vintage[vintage][method_name].get("support_volume_iou_2010", float("nan")))
            for vintage in ordered_vintages
        }
        continuity_series = {
            vintage: float(by_vintage[vintage][method_name].get("crossline_continuity", float("nan")))
            for vintage in ordered_vintages
        }
        consecutive_pairs = list(zip(ordered_vintages[:-1], ordered_vintages[1:]))
        area_deltas = {f"{earlier}->{later}": float(area_series[later] - area_series[earlier]) for earlier, later in consecutive_pairs}
        trace_deltas = {
            f"{earlier}->{later}": float(trace_series[later] - trace_series[earlier])
            for earlier, later in consecutive_pairs
        }
        support_deltas = {
            f"{earlier}->{later}": float(support_series[later] - support_series[earlier])
            for earlier, later in consecutive_pairs
        }
        continuity_deltas = {
            f"{earlier}->{later}": float(continuity_series[later] - continuity_series[earlier])
            for earlier, later in consecutive_pairs
        }
        consistency["methods"][method_name] = {
            "predicted_fraction_by_vintage": area_series,
            "predicted_trace_fraction_by_vintage": trace_series,
            "support_volume_iou_by_vintage": support_series,
            "crossline_continuity_by_vintage": continuity_series,
            "predicted_fraction_deltas": area_deltas,
            "predicted_trace_fraction_deltas": trace_deltas,
            "support_volume_iou_deltas": support_deltas,
            "crossline_continuity_deltas": continuity_deltas,
            "predicted_fraction_non_decreasing": all(delta >= 0.0 for delta in area_deltas.values()),
            "predicted_trace_fraction_non_decreasing": all(delta >= 0.0 for delta in trace_deltas.values()),
            "support_volume_iou_non_decreasing": all(delta >= 0.0 for delta in support_deltas.values()),
            "crossline_continuity_non_decreasing": all(delta >= -1e-6 for delta in continuity_deltas.values()),
        }
    return consistency


def _sequence_monotonicity_from_fractions(fractions: list[float]) -> float:
    if len(fractions) < 2:
        return float("nan")
    penalties: list[float] = []
    for earlier, later in zip(fractions[:-1], fractions[1:]):
        if np.isnan(earlier) or np.isnan(later):
            continue
        penalties.append(max(earlier - later, 0.0) / max(earlier, 1e-6))
    if not penalties:
        return float("nan")
    return float(np.clip(1.0 - np.mean(penalties), 0.0, 1.0))


def _growth_adjacency_from_volumes(volumes: list[np.ndarray]) -> float:
    if len(volumes) < 2:
        return float("nan")
    structure = np.ones((3, 3, 3), dtype=bool)
    scores: list[float] = []
    for previous_volume, current_volume in zip(volumes[:-1], volumes[1:]):
        previous_binary = previous_volume.astype(bool)
        current_binary = current_volume.astype(bool)
        new_support = current_binary & ~previous_binary
        if np.sum(new_support) == 0:
            scores.append(1.0)
            continue
        dilated_previous = ndimage.binary_dilation(previous_binary, structure=structure, iterations=1)
        scores.append(float(np.sum(new_support & dilated_previous) / max(np.sum(new_support), 1)))
    return float(np.mean(scores)) if scores else float("nan")


def _build_sequence_method_summary(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    by_vintage = _group_output_indices(outputs, "vintage")
    if len(by_vintage) < 2:
        return {}

    ordered_vintages = sorted(by_vintage, key=_sort_metadata_label)
    inline_ids = sorted(
        {
            _field_group_key(output["pair"], "inline_id", output["pair"].name)
            for output in outputs
        },
        key=_sort_metadata_label,
    )
    inline_index = {inline_id: idx for idx, inline_id in enumerate(inline_ids)}
    first_pair = outputs[0]["pair"]
    nt, nx = first_pair.baseline.shape
    method_keys = {
        "best_classical_constrained": "best_classical_constrained_binary",
        "plain_ml_constrained": "plain_ml_constrained_binary",
        "plain_ml_temporal_structured_constrained": "plain_ml_temporal_structured_constrained_binary",
        "plain_ml_structured_constrained": "plain_ml_structured_constrained_binary",
        "temporal_ml_constrained": "temporal_ml_constrained_binary",
        "wave_temporal_constrained": "wave_temporal_constrained_binary",
    }
    available_methods = {
        method_name: binary_key
        for method_name, binary_key in method_keys.items()
        if all(binary_key in output for output in outputs)
    }
    if not available_methods:
        return {}

    summary: dict[str, Any] = {"ordered_vintages": ordered_vintages, "methods": {}}
    for method_name, binary_key in available_methods.items():
        sequence_volumes: list[np.ndarray] = []
        predicted_fractions: list[float] = []
        crossline_continuity: list[float] = []
        for vintage in ordered_vintages:
            vintage_volume = np.zeros((len(inline_ids), nt, nx), dtype=bool)
            ordered_indices = _sorted_indices_by_inline(outputs, by_vintage[vintage])
            for index in ordered_indices:
                inline_id = _field_group_key(outputs[index]["pair"], "inline_id", outputs[index]["pair"].name)
                vintage_volume[inline_index[inline_id]] = outputs[index][binary_key].astype(bool)
            sequence_volumes.append(vintage_volume)
            trace_masks = np.any(vintage_volume, axis=1)
            predicted_fractions.append(float(np.mean(vintage_volume)))
            crossline_continuity.append(_crossline_continuity(trace_masks))

        summary["methods"][method_name] = {
            "temporal_monotonicity_score": _sequence_monotonicity_from_fractions(predicted_fractions),
            "growth_adjacency_score": _growth_adjacency_from_volumes(sequence_volumes),
            "predicted_fraction_by_vintage": {
                vintage: float(value) for vintage, value in zip(ordered_vintages, predicted_fractions)
            },
            "crossline_continuity_by_vintage": {
                vintage: float(value) for vintage, value in zip(ordered_vintages, crossline_continuity)
            },
        }
    return summary


def _build_leave_one_out_summary(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    eligible_outputs = [
        output
        for output in outputs
        if "temporal_leave_one_out_binary" in output and "temporal_ml_constrained_binary" in output
    ]
    if not eligible_outputs:
        return {}

    by_vintage: dict[str, list[dict[str, Any]]] = {}
    for output in eligible_outputs:
        vintage = _field_group_key(output["pair"], "vintage", output["pair"].name)
        full_binary = output["temporal_ml_constrained_binary"].astype(bool)
        heldout_binary = output["temporal_leave_one_out_binary"].astype(bool)
        by_vintage.setdefault(vintage, []).append(
            {
                "full_vs_heldout_binary_iou": _binary_iou(full_binary, heldout_binary),
                "full_vs_heldout_trace_iou": _binary_iou(np.any(full_binary, axis=0), np.any(heldout_binary, axis=0)),
                "heldout_predicted_fraction": float(np.mean(heldout_binary)),
                "full_predicted_fraction": float(np.mean(full_binary)),
                "heldout_fraction_shift": float(abs(np.mean(full_binary) - np.mean(heldout_binary))),
                "heldout_support_volume_iou_2010": float(
                    output["temporal_leave_one_out_summary"].get("support_volume_iou_2010", float("nan"))
                ),
                "heldout_trace_support_iou": float(
                    output["temporal_leave_one_out_summary"].get("trace_iou_with_2010_support", float("nan"))
                ),
            }
        )

    return {
        "by_vintage": {vintage: _mean_numeric_dict(rows) for vintage, rows in by_vintage.items()},
        "overall": _mean_numeric_dict([row for rows in by_vintage.values() for row in rows]),
    }


def _build_wave_leave_one_out_summary(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    eligible_outputs = [
        output
        for output in outputs
        if "wave_temporal_leave_one_out_binary" in output and "wave_temporal_constrained_binary" in output
    ]
    if not eligible_outputs:
        return {}

    by_vintage: dict[str, list[dict[str, Any]]] = {}
    for output in eligible_outputs:
        vintage = _field_group_key(output["pair"], "vintage", output["pair"].name)
        full_binary = output["wave_temporal_constrained_binary"].astype(bool)
        heldout_binary = output["wave_temporal_leave_one_out_binary"].astype(bool)
        by_vintage.setdefault(vintage, []).append(
            {
                "full_vs_heldout_binary_iou": _binary_iou(full_binary, heldout_binary),
                "full_vs_heldout_trace_iou": _binary_iou(np.any(full_binary, axis=0), np.any(heldout_binary, axis=0)),
                "heldout_predicted_fraction": float(np.mean(heldout_binary)),
                "full_predicted_fraction": float(np.mean(full_binary)),
                "heldout_fraction_shift": float(abs(np.mean(full_binary) - np.mean(heldout_binary))),
                "heldout_support_volume_iou_2010": float(
                    output["wave_temporal_leave_one_out_summary"].get("support_volume_iou_2010", float("nan"))
                ),
                "heldout_trace_support_iou": float(
                    output["wave_temporal_leave_one_out_summary"].get("trace_iou_with_2010_support", float("nan"))
                ),
                "heldout_residual_fit_mae": float(
                    output["wave_temporal_leave_one_out_summary"].get("residual_fit_mae", float("nan"))
                ),
                "heldout_residual_fit_rmse": float(
                    output["wave_temporal_leave_one_out_summary"].get("residual_fit_rmse", float("nan"))
                ),
            }
        )

    return {
        "by_vintage": {vintage: _mean_numeric_dict(rows) for vintage, rows in by_vintage.items()},
        "overall": _mean_numeric_dict([row for rows in by_vintage.values() for row in rows]),
    }


def _mean_valid(values: list[float]) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(array)):
        return float("nan")
    return float(np.nanmean(array))


def summarize_field_prediction_bundle(config: dict[str, Any], bundle: FieldPredictionBundle) -> dict[str, Any]:
    if not bundle.outputs:
        return {"status": "field_disabled"}

    pair_results: dict[str, Any] = {}
    field_pairs_by_name: dict[str, FieldPair] = {}
    raw_compactness_values: list[float] = []
    raw_outside_values: list[float] = []
    raw_uncertainty_values: list[float] = []
    constrained_compactness_values: list[float] = []
    constrained_outside_values: list[float] = []
    constrained_uncertainty_values: list[float] = []
    pseudo3d_compactness_values: list[float] = []
    pseudo3d_outside_values: list[float] = []
    pseudo3d_uncertainty_values: list[float] = []

    constrained_binaries: dict[str, np.ndarray] = {}
    constrained_fractions: dict[str, float] = {}
    constrained_trace_fractions: dict[str, float] = {}
    constrained_support_iou: dict[str, float] = {}
    pseudo3d_binaries: dict[str, np.ndarray] = {}
    pseudo3d_fractions: dict[str, float] = {}
    pseudo3d_trace_fractions: dict[str, float] = {}
    pseudo3d_support_iou: dict[str, float] = {}

    for field_output in bundle.outputs:
        field_pair = field_output["pair"]
        field_pairs_by_name[field_pair.name] = field_pair
        field_output["pair_support_traces"] = _pair_support_traces(field_pair, bundle)
        field_output["support_volume"] = _pair_support_volume(field_pair, bundle)

        classical_results: dict[str, Any] = {}
        classical_binaries: dict[str, np.ndarray] = {}
        for method_name, scores in field_output["classical_scores"].items():
            summary, binary = _summarize_classical_pair(
                method_name,
                scores,
                bundle.classical_thresholds[method_name],
                field_pair,
                field_output["pair_support_traces"],
                field_output["support_volume"],
                config.get("field", {}),
            )
            classical_results[method_name] = summary
            classical_binaries[method_name] = binary.astype(bool)
        field_output["classical_results"] = classical_results
        field_output["classical_binaries"] = classical_binaries

        plain_constrained_binary, plain_constraint_metadata = _postprocess_field_prediction(
            field_output["plain_probs"],
            field_output["plain_uncertainty"],
            field_pair.reservoir_mask,
            config.get("field", {}),
            shared_context=bundle.shared_context,
        )
        field_output["plain_ml_constrained_binary"] = plain_constrained_binary.astype(bool)
        field_output["plain_ml_constrained_uncertainty"] = field_output["plain_uncertainty"].astype(np.float32)
        field_output["plain_ml_constraint_metadata"] = plain_constraint_metadata

        hybrid_constrained_binary, hybrid_constraint_metadata = _postprocess_field_prediction(
            field_output["hybrid_probs"],
            field_output["hybrid_uncertainty"],
            field_pair.reservoir_mask,
            config.get("field", {}),
            shared_context=bundle.shared_context,
        )
        field_output["hybrid_ml_constrained_binary"] = hybrid_constrained_binary.astype(bool)
        field_output["hybrid_ml_constrained_uncertainty"] = field_output["hybrid_uncertainty"].astype(np.float32)
        field_output["hybrid_ml_constraint_metadata"] = hybrid_constraint_metadata

        if "temporal_probs" in field_output:
            temporal_constrained_binary, temporal_constraint_metadata = _postprocess_field_prediction(
                field_output["temporal_probs"],
                field_output["temporal_uncertainty"],
                field_pair.reservoir_mask,
                config.get("field", {}),
                shared_context=bundle.shared_context,
            )
            field_output["temporal_ml_constrained_binary"] = temporal_constrained_binary.astype(bool)
            field_output["temporal_ml_constrained_uncertainty"] = field_output["temporal_uncertainty"].astype(np.float32)
            field_output["temporal_ml_constraint_metadata"] = temporal_constraint_metadata
        if "temporal_leave_one_out_probs" in field_output:
            leave_one_out_binary, leave_one_out_metadata = _postprocess_field_prediction(
                field_output["temporal_leave_one_out_probs"],
                field_output["temporal_leave_one_out_uncertainty"],
                field_pair.reservoir_mask,
                config.get("field", {}),
                shared_context=bundle.shared_context,
            )
            field_output["temporal_leave_one_out_binary"] = leave_one_out_binary.astype(bool)
            field_output["temporal_leave_one_out_uncertainty"] = field_output["temporal_leave_one_out_uncertainty"].astype(
                np.float32
            )
            field_output["temporal_leave_one_out_constraint_metadata"] = leave_one_out_metadata
        if "wave_temporal_probs" in field_output:
            wave_binary, wave_metadata = _postprocess_field_prediction(
                field_output["wave_temporal_probs"],
                field_output["wave_temporal_uncertainty"],
                field_pair.reservoir_mask,
                config.get("field", {}),
                shared_context=bundle.shared_context,
            )
            field_output["wave_temporal_constrained_binary"] = wave_binary.astype(bool)
            field_output["wave_temporal_constrained_uncertainty"] = field_output["wave_temporal_uncertainty"].astype(
                np.float32
            )
            field_output["wave_temporal_constraint_metadata"] = wave_metadata
        if "wave_temporal_leave_one_out_probs" in field_output:
            wave_leave_one_out_binary, wave_leave_one_out_metadata = _postprocess_field_prediction(
                field_output["wave_temporal_leave_one_out_probs"],
                field_output["wave_temporal_leave_one_out_uncertainty"],
                field_pair.reservoir_mask,
                config.get("field", {}),
                shared_context=bundle.shared_context,
            )
            field_output["wave_temporal_leave_one_out_binary"] = wave_leave_one_out_binary.astype(bool)
            field_output["wave_temporal_leave_one_out_uncertainty"] = field_output[
                "wave_temporal_leave_one_out_uncertainty"
            ].astype(np.float32)
            field_output["wave_temporal_leave_one_out_constraint_metadata"] = wave_leave_one_out_metadata

        best_classical_method = max(
            classical_results,
            key=lambda name: _rank_metric_for_pair(classical_results[name], field_output["support_volume"] is not None),
        )
        field_output["best_classical_method"] = best_classical_method
        field_output["best_classical_constrained_binary"] = classical_binaries[best_classical_method].astype(bool)
        field_output["best_classical_constrained_uncertainty"] = np.zeros_like(field_output["hybrid_probs"], dtype=np.float32)

    plain_pseudo3d = _apply_pseudo3d_consistency(
        bundle.outputs,
        config,
        method_prefix="plain",
        shared_context=bundle.shared_context,
    )
    hybrid_pseudo3d = _apply_pseudo3d_consistency(
        bundle.outputs,
        config,
        method_prefix="hybrid",
        shared_context=bundle.shared_context,
    )
    plain_structured = _apply_structured_support_reconstruction(
        bundle.outputs,
        config,
        method_prefix="plain",
    )
    plain_temporal_structured = _apply_temporal_structured_support_inference(
        bundle.outputs,
        config,
        method_prefix="plain",
        seed_results=plain_structured,
    )
    plain_layered_structured = _apply_layered_structured_support_reconstruction(
        bundle.outputs,
        config,
        method_prefix="plain",
    )
    hybrid_structured = _apply_structured_support_reconstruction(
        bundle.outputs,
        config,
        method_prefix="hybrid",
    )

    for field_output in bundle.outputs:
        field_pair = field_output["pair"]
        pair_support_traces = field_output["pair_support_traces"]
        support_volume = field_output["support_volume"]

        field_output["plain_ml_pseudo3d_constrained_binary"] = plain_pseudo3d.get(
            field_pair.name,
            {
                "binary": field_output["plain_ml_constrained_binary"],
                "uncertainty": field_output["plain_uncertainty"],
                "metadata": field_output["plain_ml_constraint_metadata"],
            },
        )["binary"].astype(bool)
        field_output["plain_ml_pseudo3d_constrained_uncertainty"] = plain_pseudo3d.get(
            field_pair.name,
            {
                "binary": field_output["plain_ml_constrained_binary"],
                "uncertainty": field_output["plain_uncertainty"],
                "metadata": field_output["plain_ml_constraint_metadata"],
            },
        )["uncertainty"].astype(np.float32)
        field_output["plain_ml_pseudo3d_constraint_metadata"] = plain_pseudo3d.get(
            field_pair.name,
            {"metadata": field_output["plain_ml_constraint_metadata"]},
        )["metadata"]

        field_output["hybrid_ml_pseudo3d_constrained_binary"] = hybrid_pseudo3d.get(
            field_pair.name,
            {
                "binary": field_output["hybrid_ml_constrained_binary"],
                "uncertainty": field_output["hybrid_uncertainty"],
                "metadata": field_output["hybrid_ml_constraint_metadata"],
            },
        )["binary"].astype(bool)
        field_output["hybrid_ml_pseudo3d_constrained_uncertainty"] = hybrid_pseudo3d.get(
            field_pair.name,
            {
                "binary": field_output["hybrid_ml_constrained_binary"],
                "uncertainty": field_output["hybrid_uncertainty"],
                "metadata": field_output["hybrid_ml_constraint_metadata"],
            },
        )["uncertainty"].astype(np.float32)
        field_output["hybrid_ml_pseudo3d_constraint_metadata"] = hybrid_pseudo3d.get(
            field_pair.name,
            {"metadata": field_output["hybrid_ml_constraint_metadata"]},
        )["metadata"]

        field_output["plain_ml_structured_constrained_binary"] = plain_structured.get(
            field_pair.name,
            {
                "binary": field_output["plain_ml_constrained_binary"],
                "uncertainty": field_output["plain_uncertainty"],
                "metadata": field_output["plain_ml_constraint_metadata"],
            },
        )["binary"].astype(bool)
        field_output["plain_ml_structured_constrained_uncertainty"] = plain_structured.get(
            field_pair.name,
            {
                "binary": field_output["plain_ml_constrained_binary"],
                "uncertainty": field_output["plain_uncertainty"],
                "metadata": field_output["plain_ml_constraint_metadata"],
            },
        )["uncertainty"].astype(np.float32)
        field_output["plain_ml_structured_constraint_metadata"] = plain_structured.get(
            field_pair.name,
            {"metadata": field_output["plain_ml_constraint_metadata"]},
        )["metadata"]

        field_output["plain_ml_temporal_structured_constrained_binary"] = plain_temporal_structured.get(
            field_pair.name,
            {
                "binary": field_output["plain_ml_structured_constrained_binary"],
                "uncertainty": field_output["plain_ml_structured_constrained_uncertainty"],
                "metadata": field_output["plain_ml_structured_constraint_metadata"],
            },
        )["binary"].astype(bool)
        field_output["plain_ml_temporal_structured_constrained_uncertainty"] = plain_temporal_structured.get(
            field_pair.name,
            {
                "binary": field_output["plain_ml_structured_constrained_binary"],
                "uncertainty": field_output["plain_ml_structured_constrained_uncertainty"],
                "metadata": field_output["plain_ml_structured_constraint_metadata"],
            },
        )["uncertainty"].astype(np.float32)
        field_output["plain_ml_temporal_structured_constraint_metadata"] = plain_temporal_structured.get(
            field_pair.name,
            {"metadata": field_output["plain_ml_structured_constraint_metadata"]},
        )["metadata"]

        field_output["plain_ml_layered_structured_constrained_binary"] = plain_layered_structured.get(
            field_pair.name,
            {
                "binary": field_output["plain_ml_structured_constrained_binary"],
                "uncertainty": field_output["plain_ml_structured_constrained_uncertainty"],
                "metadata": field_output["plain_ml_structured_constraint_metadata"],
            },
        )["binary"].astype(bool)
        field_output["plain_ml_layered_structured_constrained_uncertainty"] = plain_layered_structured.get(
            field_pair.name,
            {
                "binary": field_output["plain_ml_structured_constrained_binary"],
                "uncertainty": field_output["plain_ml_structured_constrained_uncertainty"],
                "metadata": field_output["plain_ml_structured_constraint_metadata"],
            },
        )["uncertainty"].astype(np.float32)
        field_output["plain_ml_layered_structured_constraint_metadata"] = plain_layered_structured.get(
            field_pair.name,
            {"metadata": field_output["plain_ml_structured_constraint_metadata"]},
        )["metadata"]

        field_output["hybrid_ml_structured_constrained_binary"] = hybrid_structured.get(
            field_pair.name,
            {
                "binary": field_output["hybrid_ml_constrained_binary"],
                "uncertainty": field_output["hybrid_uncertainty"],
                "metadata": field_output["hybrid_ml_constraint_metadata"],
            },
        )["binary"].astype(bool)
        field_output["hybrid_ml_structured_constrained_uncertainty"] = hybrid_structured.get(
            field_pair.name,
            {
                "binary": field_output["hybrid_ml_constrained_binary"],
                "uncertainty": field_output["hybrid_uncertainty"],
                "metadata": field_output["hybrid_ml_constraint_metadata"],
            },
        )["uncertainty"].astype(np.float32)
        field_output["hybrid_ml_structured_constraint_metadata"] = hybrid_structured.get(
            field_pair.name,
            {"metadata": field_output["hybrid_ml_constraint_metadata"]},
        )["metadata"]

        hybrid_metrics = _evaluate_field_prediction(
            field_output["hybrid_probs"],
            field_output["hybrid_uncertainty"],
            field_pair.reservoir_mask,
        )
        plain_metrics = _evaluate_field_prediction(
            field_output["plain_probs"],
            field_output["plain_uncertainty"],
            field_pair.reservoir_mask,
        )
        temporal_metrics = None
        if "temporal_probs" in field_output:
            temporal_metrics = _evaluate_field_prediction(
                field_output["temporal_probs"],
                field_output["temporal_uncertainty"],
                field_pair.reservoir_mask,
            )
        wave_temporal_metrics = None
        if "wave_temporal_probs" in field_output:
            wave_temporal_metrics = {
                **_evaluate_field_prediction(
                    field_output["wave_temporal_probs"],
                    field_output["wave_temporal_uncertainty"],
                    field_pair.reservoir_mask,
                ),
                **_residual_fit_metrics(field_output["wave_temporal_predicted_residual"], field_pair),
            }

        plain_constrained_summary = _summarize_binary_pair(
            field_output["plain_ml_constrained_binary"],
            field_output["plain_ml_constrained_uncertainty"],
            field_pair,
            pair_support_traces,
            support_volume,
            field_output["plain_ml_constraint_metadata"],
        )
        hybrid_constrained_summary = _summarize_binary_pair(
            field_output["hybrid_ml_constrained_binary"],
            field_output["hybrid_ml_constrained_uncertainty"],
            field_pair,
            pair_support_traces,
            support_volume,
            field_output["hybrid_ml_constraint_metadata"],
        )
        plain_pseudo3d_summary = _summarize_binary_pair(
            field_output["plain_ml_pseudo3d_constrained_binary"],
            field_output["plain_ml_pseudo3d_constrained_uncertainty"],
            field_pair,
            pair_support_traces,
            support_volume,
            field_output["plain_ml_pseudo3d_constraint_metadata"],
        )
        hybrid_pseudo3d_summary = _summarize_binary_pair(
            field_output["hybrid_ml_pseudo3d_constrained_binary"],
            field_output["hybrid_ml_pseudo3d_constrained_uncertainty"],
            field_pair,
            pair_support_traces,
            support_volume,
            field_output["hybrid_ml_pseudo3d_constraint_metadata"],
        )
        plain_structured_summary = _summarize_binary_pair(
            field_output["plain_ml_structured_constrained_binary"],
            field_output["plain_ml_structured_constrained_uncertainty"],
            field_pair,
            pair_support_traces,
            support_volume,
            field_output["plain_ml_structured_constraint_metadata"],
        )
        plain_temporal_structured_summary = _summarize_binary_pair(
            field_output["plain_ml_temporal_structured_constrained_binary"],
            field_output["plain_ml_temporal_structured_constrained_uncertainty"],
            field_pair,
            pair_support_traces,
            support_volume,
            field_output["plain_ml_temporal_structured_constraint_metadata"],
        )
        plain_layered_structured_summary = _summarize_binary_pair(
            field_output["plain_ml_layered_structured_constrained_binary"],
            field_output["plain_ml_layered_structured_constrained_uncertainty"],
            field_pair,
            pair_support_traces,
            support_volume,
            field_output["plain_ml_layered_structured_constraint_metadata"],
        )
        hybrid_structured_summary = _summarize_binary_pair(
            field_output["hybrid_ml_structured_constrained_binary"],
            field_output["hybrid_ml_structured_constrained_uncertainty"],
            field_pair,
            pair_support_traces,
            support_volume,
            field_output["hybrid_ml_structured_constraint_metadata"],
        )
        temporal_constrained_summary = None
        if "temporal_ml_constrained_binary" in field_output:
            temporal_constrained_summary = _summarize_binary_pair(
                field_output["temporal_ml_constrained_binary"],
                field_output["temporal_ml_constrained_uncertainty"],
                field_pair,
                pair_support_traces,
                support_volume,
                field_output["temporal_ml_constraint_metadata"],
            )
        temporal_leave_one_out_summary = None
        if "temporal_leave_one_out_binary" in field_output:
            temporal_leave_one_out_summary = _summarize_binary_pair(
                field_output["temporal_leave_one_out_binary"],
                field_output["temporal_leave_one_out_uncertainty"],
                field_pair,
                pair_support_traces,
                support_volume,
                field_output["temporal_leave_one_out_constraint_metadata"],
            )
            field_output["temporal_leave_one_out_summary"] = temporal_leave_one_out_summary
        wave_temporal_constrained_summary = None
        if "wave_temporal_constrained_binary" in field_output:
            wave_temporal_constrained_summary = {
                **_summarize_binary_pair(
                    field_output["wave_temporal_constrained_binary"],
                    field_output["wave_temporal_constrained_uncertainty"],
                    field_pair,
                    pair_support_traces,
                    support_volume,
                    field_output["wave_temporal_constraint_metadata"],
                ),
                **_residual_fit_metrics(field_output["wave_temporal_predicted_residual"], field_pair),
            }
        wave_temporal_leave_one_out_summary = None
        if "wave_temporal_leave_one_out_binary" in field_output:
            wave_temporal_leave_one_out_summary = {
                **_summarize_binary_pair(
                    field_output["wave_temporal_leave_one_out_binary"],
                    field_output["wave_temporal_leave_one_out_uncertainty"],
                    field_pair,
                    pair_support_traces,
                    support_volume,
                    field_output["wave_temporal_leave_one_out_constraint_metadata"],
                ),
                **_residual_fit_metrics(field_output["wave_temporal_leave_one_out_predicted_residual"], field_pair),
            }
            field_output["wave_temporal_leave_one_out_summary"] = wave_temporal_leave_one_out_summary
        best_classical_summary = dict(field_output["classical_results"][field_output["best_classical_method"]])
        best_classical_summary["method_name"] = field_output["best_classical_method"]

        pair_result = {
            **field_output["classical_results"],
            "best_classical_constrained": best_classical_summary,
            "plain_ml": plain_metrics,
            "plain_ml_constrained": plain_constrained_summary,
            "plain_ml_pseudo3d_constrained": plain_pseudo3d_summary,
            "plain_ml_structured_constrained": plain_structured_summary,
            "plain_ml_temporal_structured_constrained": plain_temporal_structured_summary,
            "plain_ml_layered_structured_constrained": plain_layered_structured_summary,
            "hybrid_ml": hybrid_metrics,
            "hybrid_ml_constrained": hybrid_constrained_summary,
            "hybrid_ml_pseudo3d_constrained": hybrid_pseudo3d_summary,
            "hybrid_ml_structured_constrained": hybrid_structured_summary,
            "best_constrained_classical_method": field_output["best_classical_method"],
            "metadata": field_pair.metadata or {},
        }
        if temporal_metrics is not None and temporal_constrained_summary is not None:
            pair_result["temporal_ml"] = temporal_metrics
            pair_result["temporal_ml_constrained"] = temporal_constrained_summary
        if temporal_leave_one_out_summary is not None:
            pair_result["temporal_ml_leave_one_out"] = temporal_leave_one_out_summary
        if wave_temporal_metrics is not None and wave_temporal_constrained_summary is not None:
            pair_result["wave_temporal_ml"] = wave_temporal_metrics
            pair_result["wave_temporal_ml_constrained"] = wave_temporal_constrained_summary
        if wave_temporal_leave_one_out_summary is not None:
            pair_result["wave_temporal_ml_leave_one_out"] = wave_temporal_leave_one_out_summary
        pair_results[field_pair.name] = pair_result

        if not np.isnan(hybrid_metrics["compactness"]):
            raw_compactness_values.append(hybrid_metrics["compactness"])
        if not np.isnan(hybrid_metrics["outside_reservoir_fraction"]):
            raw_outside_values.append(hybrid_metrics["outside_reservoir_fraction"])
        raw_uncertainty_values.append(hybrid_metrics["mean_uncertainty"])

        if not np.isnan(hybrid_constrained_summary["compactness"]):
            constrained_compactness_values.append(hybrid_constrained_summary["compactness"])
        if not np.isnan(hybrid_constrained_summary["outside_reservoir_fraction"]):
            constrained_outside_values.append(hybrid_constrained_summary["outside_reservoir_fraction"])
        constrained_uncertainty_values.append(hybrid_constrained_summary["mean_uncertainty"])

        if not np.isnan(hybrid_pseudo3d_summary["compactness"]):
            pseudo3d_compactness_values.append(hybrid_pseudo3d_summary["compactness"])
        if not np.isnan(hybrid_pseudo3d_summary["outside_reservoir_fraction"]):
            pseudo3d_outside_values.append(hybrid_pseudo3d_summary["outside_reservoir_fraction"])
        pseudo3d_uncertainty_values.append(hybrid_pseudo3d_summary["mean_uncertainty"])

        constrained_binaries[field_pair.name] = field_output["hybrid_ml_constrained_binary"].astype(bool)
        constrained_fractions[field_pair.name] = float(hybrid_constrained_summary["predicted_fraction"])
        constrained_trace_fractions[field_pair.name] = float(
            hybrid_constrained_summary.get("predicted_trace_fraction", float("nan"))
        )
        constrained_support_iou[field_pair.name] = float(
            hybrid_constrained_summary.get(
                "support_volume_iou_2010",
                hybrid_constrained_summary.get("trace_iou_with_2010_support", float("nan")),
            )
        )
        pseudo3d_binaries[field_pair.name] = field_output["hybrid_ml_pseudo3d_constrained_binary"].astype(bool)
        pseudo3d_fractions[field_pair.name] = float(hybrid_pseudo3d_summary["predicted_fraction"])
        pseudo3d_trace_fractions[field_pair.name] = float(
            hybrid_pseudo3d_summary.get("predicted_trace_fraction", float("nan"))
        )
        pseudo3d_support_iou[field_pair.name] = float(
            hybrid_pseudo3d_summary.get(
                "support_volume_iou_2010",
                hybrid_pseudo3d_summary.get("trace_iou_with_2010_support", float("nan")),
            )
        )

    field_summary: dict[str, Any] = {
        "pairs": pair_results,
        "hybrid_average": {
            "compactness": _mean_valid(raw_compactness_values),
            "outside_reservoir_fraction": _mean_valid(raw_outside_values),
            "mean_uncertainty": _mean_valid(raw_uncertainty_values),
        },
        "hybrid_constrained_average": {
            "compactness": _mean_valid(constrained_compactness_values),
            "outside_reservoir_fraction": _mean_valid(constrained_outside_values),
            "mean_uncertainty": _mean_valid(constrained_uncertainty_values),
        },
        "hybrid_pseudo3d_average": {
            "compactness": _mean_valid(pseudo3d_compactness_values),
            "outside_reservoir_fraction": _mean_valid(pseudo3d_outside_values),
            "mean_uncertainty": _mean_valid(pseudo3d_uncertainty_values),
        },
        "classical_thresholds": {name: float(value) for name, value in bundle.classical_thresholds.items()},
        "volume_summary": _build_volume_summary(bundle.outputs),
    }

    temporal_consistency = _build_temporal_consistency(
        field_pairs_by_name,
        constrained_binaries,
        constrained_fractions,
        constrained_trace_fractions,
        constrained_support_iou,
    )
    if temporal_consistency:
        field_summary["temporal_consistency"] = temporal_consistency

    temporal_consistency_pseudo3d = _build_temporal_consistency(
        field_pairs_by_name,
        pseudo3d_binaries,
        pseudo3d_fractions,
        pseudo3d_trace_fractions,
        pseudo3d_support_iou,
    )
    if temporal_consistency_pseudo3d:
        field_summary["temporal_consistency_pseudo3d"] = temporal_consistency_pseudo3d

    temporal_volume_consistency = _build_temporal_volume_consistency(field_summary["volume_summary"])
    if temporal_volume_consistency:
        field_summary["temporal_volume_consistency"] = temporal_volume_consistency

    sequence_method_summary = _build_sequence_method_summary(bundle.outputs)
    if sequence_method_summary:
        field_summary["sequence_method_summary"] = sequence_method_summary

    leave_one_out_summary = _build_leave_one_out_summary(bundle.outputs)
    if leave_one_out_summary:
        field_summary["leave_one_out_summary"] = leave_one_out_summary

    wave_leave_one_out_summary = _build_wave_leave_one_out_summary(bundle.outputs)
    if wave_leave_one_out_summary:
        field_summary["wave_leave_one_out_summary"] = wave_leave_one_out_summary

    if bundle.plume_support_traces is not None:
        support_note = str(config.get("field", {}).get("plume_support_note", "")).strip()
        if not support_note:
            support_note = (
                "2010 plume-boundary support is used as a later-time structural envelope, not as exact ground "
                "truth for earlier vintages."
            )
        field_summary["support_note"] = support_note

    return field_summary
