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


def _sorted_indices_by_inline(bundle_outputs: list[dict[str, Any]], indices: list[int]) -> list[int]:
    return sorted(
        indices,
        key=lambda idx: (
            int((bundle_outputs[idx]["pair"].metadata or {}).get("inline_id", 0)),
            bundle_outputs[idx]["pair"].name,
        ),
    )


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
        "plain_ml_structured_constrained": (
            "plain_ml_structured_constrained_binary",
            "plain_ml_structured_constrained_uncertainty",
        ),
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
        for method_name, (binary_key, uncertainty_key) in methods.items():
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
        hybrid_structured_summary = _summarize_binary_pair(
            field_output["hybrid_ml_structured_constrained_binary"],
            field_output["hybrid_ml_structured_constrained_uncertainty"],
            field_pair,
            pair_support_traces,
            support_volume,
            field_output["hybrid_ml_structured_constraint_metadata"],
        )
        best_classical_summary = dict(field_output["classical_results"][field_output["best_classical_method"]])
        best_classical_summary["method_name"] = field_output["best_classical_method"]

        pair_results[field_pair.name] = {
            **field_output["classical_results"],
            "best_classical_constrained": best_classical_summary,
            "plain_ml": plain_metrics,
            "plain_ml_constrained": plain_constrained_summary,
            "plain_ml_pseudo3d_constrained": plain_pseudo3d_summary,
            "plain_ml_structured_constrained": plain_structured_summary,
            "hybrid_ml": hybrid_metrics,
            "hybrid_ml_constrained": hybrid_constrained_summary,
            "hybrid_ml_pseudo3d_constrained": hybrid_pseudo3d_summary,
            "hybrid_ml_structured_constrained": hybrid_structured_summary,
            "best_constrained_classical_method": field_output["best_classical_method"],
            "metadata": field_pair.metadata or {},
        }

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

    if bundle.plume_support_traces is not None:
        support_note = str(config.get("field", {}).get("plume_support_note", "")).strip()
        if not support_note:
            support_note = (
                "2010 plume-boundary support is used as a later-time structural envelope, not as exact ground "
                "truth for earlier vintages."
            )
        field_summary["support_note"] = support_note

    return field_summary
