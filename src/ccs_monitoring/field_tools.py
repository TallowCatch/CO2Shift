"""Reusable field-style inference helpers for evidence packs and volume outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .baselines import apply_threshold, fit_best_threshold, score_cross_equalized_difference
from .data import FieldPair, load_field_pairs
from .features import build_hybrid_channels, build_plain_channels
from .pipeline import (
    _cleanup_field_binary,
    _compute_shared_field_postprocess_context,
    _evaluate_field_prediction,
    _load_bundle,
    _load_model_artifact,
    _postprocess_field_prediction,
    _resolve_artifacts_root,
    _summarize_field_binary,
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
    cross_equalized_threshold: float
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


def collect_field_prediction_bundle(config: dict[str, Any]) -> FieldPredictionBundle:
    ensure_runtime_environment(config["output_root"], config["seed"])
    artifacts_root = _resolve_artifacts_root(config)
    bundle = _load_bundle(artifacts_root)

    val_cross_equalized_scores = np.stack(
        [
            score_cross_equalized_difference(baseline, monitor, reservoir_mask)
            for baseline, monitor, reservoir_mask in zip(
                bundle.val["baseline"],
                bundle.val["monitor"],
                bundle.val["reservoir_mask"],
            )
        ],
        axis=0,
    )
    cross_equalized_threshold = fit_best_threshold(val_cross_equalized_scores, bundle.val["change_mask"])

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
            cross_equalized_threshold=float(cross_equalized_threshold),
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
        outputs.append(
            {
                "pair": field_pair,
                "plain_probs": plain_probs[0],
                "plain_uncertainty": plain_uncertainty[0],
                "hybrid_probs": hybrid_probs[0],
                "hybrid_uncertainty": hybrid_uncertainty[0],
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
        cross_equalized_threshold=float(cross_equalized_threshold),
        outputs=outputs,
    )


def summarize_field_prediction_bundle(config: dict[str, Any], bundle: FieldPredictionBundle) -> dict[str, Any]:
    if not bundle.outputs:
        return {"status": "field_disabled"}

    shared_context = _compute_shared_field_postprocess_context(
        [entry["hybrid_probs"] for entry in bundle.outputs],
        [entry["hybrid_uncertainty"] for entry in bundle.outputs],
        [entry["pair"].reservoir_mask for entry in bundle.outputs],
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
    constrained_trace_fractions: dict[str, float] = {}
    constrained_support_iou: dict[str, float] = {}

    for field_output in bundle.outputs:
        field_pair = field_output["pair"]
        plain_probs = field_output["plain_probs"]
        plain_uncertainty = field_output["plain_uncertainty"]
        hybrid_probs = field_output["hybrid_probs"]
        hybrid_uncertainty = field_output["hybrid_uncertainty"]

        cross_equalized_scores = score_cross_equalized_difference(
            field_pair.baseline,
            field_pair.monitor,
            field_pair.reservoir_mask,
        )
        cross_equalized_binary = (
            apply_threshold(cross_equalized_scores[None, ...], bundle.cross_equalized_threshold)[0] > 0.5
        )
        cross_equalized_structured, cross_equalized_cleanup = _cleanup_field_binary(
            cross_equalized_binary,
            field_pair.reservoir_mask,
            config.get("field", {}),
        )
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
        support_metrics = _trace_support_metrics(constrained_binary, bundle.plume_support_traces)

        pair_results[field_pair.name] = {
            "cross_equalized_difference": {
                **_summarize_field_binary(
                    cross_equalized_structured,
                    np.zeros_like(cross_equalized_scores, dtype=np.float32),
                    field_pair.reservoir_mask,
                ),
                **_trace_support_metrics(cross_equalized_structured, bundle.plume_support_traces),
                "constraint_metadata": {
                    "enabled": True,
                    "threshold_mode": "synthetic_validation_threshold",
                    "probability_threshold": float(bundle.cross_equalized_threshold),
                    "uncertainty_threshold": None,
                    "threshold_source": "synthetic_validation_threshold",
                    "shared_across_pairs": True,
                    **cross_equalized_cleanup,
                },
            },
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
        constrained_trace_fractions[field_pair.name] = float(
            support_metrics.get("predicted_trace_fraction", float("nan"))
        )
        constrained_support_iou[field_pair.name] = float(
            support_metrics.get("trace_iou_with_2010_support", float("nan"))
        )

    field_summary: dict[str, Any] = {
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
            name: float(value) for name, value in constrained_support_iou.items() if not np.isnan(value)
        }
        field_summary["temporal_consistency"] = {
            "ordered_pairs": ordered_names,
            "constrained_area_deltas": area_deltas,
            "constrained_trace_fraction_deltas": trace_fraction_deltas,
            "constrained_pairwise_iou": pairwise_iou,
            "constrained_area_non_decreasing": all(delta >= 0.0 for delta in area_deltas.values()),
            "constrained_trace_fraction_non_decreasing": (
                all(delta >= 0.0 for delta in trace_fraction_deltas.values()) if trace_fraction_deltas else False
            ),
            "constrained_trace_fraction_by_pair": constrained_trace_fractions,
            "constrained_support_iou_by_pair": support_iou_progression,
            "constrained_support_iou_non_decreasing": (
                all(
                    support_iou_progression[later] >= support_iou_progression[earlier]
                    for earlier, later in consecutive_pairs
                    if earlier in support_iou_progression and later in support_iou_progression
                )
                if support_iou_progression
                else False
            ),
        }

    if bundle.plume_support_traces is not None:
        support_note = str(config.get("field", {}).get("plume_support_note", "")).strip()
        if not support_note:
            support_note = (
                "2010 plume-boundary support is used as a later-time structural envelope, not as exact ground "
                "truth for earlier vintages."
            )
        field_summary["support_note"] = support_note

    return field_summary


def _binary_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(bool)
    second = second.astype(bool)
    union = np.sum(first | second)
    if union == 0:
        return float("nan")
    intersection = np.sum(first & second)
    return float(intersection / union)
