"""Chunked volume artifacts for 4D-style browsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import xarray as xr

from .field_tools import collect_field_prediction_bundle, summarize_field_prediction_bundle
from .runtime import ensure_runtime_environment


def build_volume(config: dict[str, Any]) -> dict[str, Any]:
    ensure_runtime_environment(config["output_root"], config["seed"])
    volume_cfg = config["volume"]
    output_store = Path(volume_cfg["output_store"] or (Path(config["output_root"]) / "volume.zarr"))
    output_store.parent.mkdir(parents=True, exist_ok=True)

    bundle = collect_field_prediction_bundle(config)
    if not bundle.outputs:
        raise ValueError("build-volume requires field.enabled=true with at least one field pair.")

    field_summary = summarize_field_prediction_bundle(config, bundle)
    outputs = _filter_outputs(bundle.outputs, volume_cfg.get("vintages", []))
    first_pair = outputs[0]["pair"]
    nt, nx = first_pair.baseline.shape
    inline_values, vintage_values = _resolve_inline_vintage_axes(outputs, volume_cfg)
    inline_count = len(inline_values)
    vintage_count = len(vintage_values)
    inline_index = {value: index for index, value in enumerate(inline_values)}
    vintage_index = {value: index for index, value in enumerate(vintage_values)}

    baseline = np.zeros((inline_count, nt, nx), dtype=np.float32)
    reservoir_mask = np.zeros((inline_count, nt, nx), dtype=np.float32)
    support_trace = np.zeros((inline_count, nx), dtype=np.float32)
    support_volume = np.zeros((inline_count, nt, nx), dtype=np.float32)
    monitor = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    raw_probability = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    uncertainty = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    best_classical_constrained = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    plain_ml_constrained = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    plain_ml_pseudo3d_constrained = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    plain_ml_structured_constrained = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    plain_ml_temporal_structured_constrained = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    plain_ml_layered_structured_constrained = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    temporal_probability = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    temporal_uncertainty = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    temporal_ml_constrained = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    temporal_leave_one_out_probability = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    temporal_leave_one_out_uncertainty = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    temporal_leave_one_out_constrained = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    wave_temporal_probability = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    wave_temporal_uncertainty = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    wave_temporal_constrained = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    wave_temporal_predicted_residual = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    wave_temporal_heldout_probability = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    wave_temporal_heldout_uncertainty = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    wave_temporal_heldout_constrained = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    wave_temporal_heldout_predicted_residual = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    hybrid_constrained = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    hybrid_structured_constrained = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    hybrid_pseudo3d_constrained = np.full((inline_count, vintage_count, nt, nx), np.nan, dtype=np.float32)
    pair_available = np.zeros((inline_count, vintage_count), dtype=np.int8)
    pair_names = np.full((inline_count, vintage_count), "", dtype="<U128")

    for entry in outputs:
        pair = entry["pair"]
        inline_label = _inline_label(pair)
        vintage_label = _vintage_label(pair)
        inline_idx = inline_index[inline_label]
        vintage_idx = vintage_index[vintage_label]

        if pair.baseline.shape != (nt, nx) or pair.monitor.shape != (nt, nx):
            raise ValueError(
                "All field pairs used for volume rendering must share the same section shape. "
                f"Expected {(nt, nx)} but received {pair.monitor.shape} for {pair.name}."
            )

        baseline[inline_idx] = pair.baseline.astype(np.float32)
        pair_reservoir_mask = (
            pair.reservoir_mask.astype(np.float32)
            if pair.reservoir_mask is not None
            else np.ones_like(pair.baseline, dtype=np.float32)
        )
        pair_support_trace = (
            entry.get("pair_support_traces").astype(np.float32)
            if entry.get("pair_support_traces") is not None
            else np.zeros(pair.baseline.shape[1], dtype=np.float32)
        )
        pair_support_volume = (
            entry.get("support_volume").astype(np.float32)
            if entry.get("support_volume") is not None
            else pair_reservoir_mask * pair_support_trace[None, :]
        )
        reservoir_mask[inline_idx] = pair_reservoir_mask
        support_trace[inline_idx] = pair_support_trace
        support_volume[inline_idx] = pair_support_volume

        monitor[inline_idx, vintage_idx] = pair.monitor.astype(np.float32)
        raw_probability[inline_idx, vintage_idx] = entry["hybrid_probs"].astype(np.float32)
        uncertainty[inline_idx, vintage_idx] = entry["hybrid_uncertainty"].astype(np.float32)
        best_classical_constrained[inline_idx, vintage_idx] = entry["best_classical_constrained_binary"].astype(np.float32)
        plain_ml_constrained[inline_idx, vintage_idx] = entry["plain_ml_constrained_binary"].astype(np.float32)
        plain_ml_pseudo3d_constrained[inline_idx, vintage_idx] = entry["plain_ml_pseudo3d_constrained_binary"].astype(
            np.float32
        )
        plain_ml_structured_constrained[inline_idx, vintage_idx] = entry["plain_ml_structured_constrained_binary"].astype(
            np.float32
        )
        plain_ml_temporal_structured_constrained[inline_idx, vintage_idx] = entry[
            "plain_ml_temporal_structured_constrained_binary"
        ].astype(np.float32)
        plain_ml_layered_structured_constrained[inline_idx, vintage_idx] = entry[
            "plain_ml_layered_structured_constrained_binary"
        ].astype(np.float32)
        if "temporal_probs" in entry:
            temporal_probability[inline_idx, vintage_idx] = entry["temporal_probs"].astype(np.float32)
            temporal_uncertainty[inline_idx, vintage_idx] = entry["temporal_uncertainty"].astype(np.float32)
            temporal_ml_constrained[inline_idx, vintage_idx] = entry["temporal_ml_constrained_binary"].astype(np.float32)
        if "temporal_leave_one_out_probs" in entry:
            temporal_leave_one_out_probability[inline_idx, vintage_idx] = entry["temporal_leave_one_out_probs"].astype(
                np.float32
            )
            temporal_leave_one_out_uncertainty[inline_idx, vintage_idx] = entry[
                "temporal_leave_one_out_uncertainty"
            ].astype(np.float32)
            temporal_leave_one_out_constrained[inline_idx, vintage_idx] = entry[
                "temporal_leave_one_out_binary"
            ].astype(np.float32)
        if "wave_temporal_probs" in entry:
            wave_temporal_probability[inline_idx, vintage_idx] = entry["wave_temporal_probs"].astype(np.float32)
            wave_temporal_uncertainty[inline_idx, vintage_idx] = entry["wave_temporal_uncertainty"].astype(np.float32)
            wave_temporal_constrained[inline_idx, vintage_idx] = entry["wave_temporal_constrained_binary"].astype(
                np.float32
            )
            wave_temporal_predicted_residual[inline_idx, vintage_idx] = entry[
                "wave_temporal_predicted_residual"
            ].astype(np.float32)
        if "wave_temporal_leave_one_out_probs" in entry:
            wave_temporal_heldout_probability[inline_idx, vintage_idx] = entry[
                "wave_temporal_leave_one_out_probs"
            ].astype(np.float32)
            wave_temporal_heldout_uncertainty[inline_idx, vintage_idx] = entry[
                "wave_temporal_leave_one_out_uncertainty"
            ].astype(np.float32)
            wave_temporal_heldout_constrained[inline_idx, vintage_idx] = entry[
                "wave_temporal_leave_one_out_binary"
            ].astype(np.float32)
            wave_temporal_heldout_predicted_residual[inline_idx, vintage_idx] = entry[
                "wave_temporal_leave_one_out_predicted_residual"
            ].astype(np.float32)
        hybrid_constrained[inline_idx, vintage_idx] = entry["hybrid_ml_constrained_binary"].astype(np.float32)
        hybrid_structured_constrained[inline_idx, vintage_idx] = entry["hybrid_ml_structured_constrained_binary"].astype(
            np.float32
        )
        hybrid_pseudo3d_constrained[inline_idx, vintage_idx] = entry["hybrid_ml_pseudo3d_constrained_binary"].astype(
            np.float32
        )
        pair_available[inline_idx, vintage_idx] = 1
        pair_names[inline_idx, vintage_idx] = pair.name

    chunks = volume_cfg.get("chunking", {})
    inline_chunk = int(chunks.get("inline", 1))
    vintage_chunk = int(chunks.get("vintage", 1))
    sample_chunk = int(chunks.get("sample", min(nt, 256)))
    trace_chunk = int(chunks.get("trace", min(nx, 128)))

    dataset = xr.Dataset(
        data_vars={
            "baseline": (
                ("inline", "sample", "trace"),
                da.from_array(baseline, chunks=(inline_chunk, sample_chunk, trace_chunk)),
            ),
            "monitor": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(monitor, chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk)),
            ),
            "raw_probability": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(raw_probability, chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk)),
            ),
            "uncertainty": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(uncertainty, chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk)),
            ),
            "best_classical_constrained": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(best_classical_constrained, chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk)),
            ),
            "plain_ml_constrained": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(plain_ml_constrained, chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk)),
            ),
            "plain_ml_pseudo3d_constrained": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(
                    plain_ml_pseudo3d_constrained,
                    chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk),
                ),
            ),
            "plain_ml_structured_constrained": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(
                    plain_ml_structured_constrained,
                    chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk),
                ),
            ),
            "plain_ml_temporal_structured_constrained": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(
                    plain_ml_temporal_structured_constrained,
                    chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk),
                ),
            ),
            "plain_ml_layered_structured_constrained": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(
                    plain_ml_layered_structured_constrained,
                    chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk),
                ),
            ),
            "temporal_probability": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(temporal_probability, chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk)),
            ),
            "temporal_uncertainty": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(temporal_uncertainty, chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk)),
            ),
            "temporal_ml_constrained": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(temporal_ml_constrained, chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk)),
            ),
            "temporal_leave_one_out_probability": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(
                    temporal_leave_one_out_probability,
                    chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk),
                ),
            ),
            "temporal_leave_one_out_uncertainty": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(
                    temporal_leave_one_out_uncertainty,
                    chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk),
                ),
            ),
            "temporal_leave_one_out_constrained": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(
                    temporal_leave_one_out_constrained,
                    chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk),
                ),
            ),
            "wave_temporal_probability": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(
                    wave_temporal_probability,
                    chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk),
                ),
            ),
            "wave_temporal_uncertainty": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(
                    wave_temporal_uncertainty,
                    chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk),
                ),
            ),
            "wave_temporal_constrained": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(
                    wave_temporal_constrained,
                    chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk),
                ),
            ),
            "wave_temporal_predicted_residual": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(
                    wave_temporal_predicted_residual,
                    chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk),
                ),
            ),
            "wave_temporal_heldout_probability": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(
                    wave_temporal_heldout_probability,
                    chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk),
                ),
            ),
            "wave_temporal_heldout_uncertainty": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(
                    wave_temporal_heldout_uncertainty,
                    chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk),
                ),
            ),
            "wave_temporal_heldout_constrained": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(
                    wave_temporal_heldout_constrained,
                    chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk),
                ),
            ),
            "wave_temporal_heldout_predicted_residual": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(
                    wave_temporal_heldout_predicted_residual,
                    chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk),
                ),
            ),
            "hybrid_constrained": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(hybrid_constrained, chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk)),
            ),
            "hybrid_ml_structured_constrained": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(
                    hybrid_structured_constrained,
                    chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk),
                ),
            ),
            "hybrid_pseudo3d_constrained": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(
                    hybrid_pseudo3d_constrained,
                    chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk),
                ),
            ),
            # Backward-compatible aliases for earlier plots/configs.
            "cross_equalized_support": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(best_classical_constrained, chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk)),
            ),
            "constrained_support": (
                ("inline", "vintage", "sample", "trace"),
                da.from_array(hybrid_constrained, chunks=(inline_chunk, vintage_chunk, sample_chunk, trace_chunk)),
            ),
            "reservoir_mask": (
                ("inline", "sample", "trace"),
                da.from_array(reservoir_mask, chunks=(inline_chunk, sample_chunk, trace_chunk)),
            ),
            "support_trace_2010": (
                ("inline", "trace"),
                da.from_array(support_trace, chunks=(inline_chunk, trace_chunk)),
            ),
            "support_volume_2010": (
                ("inline", "sample", "trace"),
                da.from_array(support_volume, chunks=(inline_chunk, sample_chunk, trace_chunk)),
            ),
            "pair_available": (
                ("inline", "vintage"),
                da.from_array(pair_available, chunks=(inline_chunk, vintage_chunk)),
            ),
            "pair_name": (("inline", "vintage"), pair_names),
        },
        coords={
            "inline": np.array(inline_values, dtype=object),
            "vintage": np.array(vintage_values, dtype=object),
            "sample": np.arange(nt, dtype=np.int32),
            "trace": np.arange(nx, dtype=np.int32),
        },
        attrs={
            "job_name": str(config["hpc"]["job_name"]),
            "launcher": str(config["hpc"]["launcher"]),
            "threads_per_worker": int(config["hpc"]["threads_per_worker"]),
            "artifacts_root": str(bundle.artifacts_root.resolve()),
            "field_manifest_path": str(config.get("field", {}).get("manifest_path", "")),
        },
    )

    dataset.to_zarr(output_store, mode="w", zarr_format=2)

    manifest = {
        "output_store": str(output_store),
        "inlines": inline_values,
        "vintages": vintage_values,
        "shape": {
            "inline": inline_count,
            "vintage": vintage_count,
            "sample": nt,
            "trace": nx,
        },
        "chunks": {
            "inline": inline_chunk,
            "vintage": vintage_chunk,
            "sample": sample_chunk,
            "trace": trace_chunk,
        },
        "variables": list(dataset.data_vars),
        "volume_summary": field_summary.get("volume_summary", {}),
        "sequence_method_summary": field_summary.get("sequence_method_summary", {}),
        "leave_one_out_summary": field_summary.get("leave_one_out_summary", {}),
        "wave_leave_one_out_summary": field_summary.get("wave_leave_one_out_summary", {}),
        "temporal_volume_consistency": field_summary.get("temporal_volume_consistency", {}),
    }
    manifest_path = Path(config["output_root"]) / "results" / "volume_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _filter_outputs(outputs: list[dict[str, Any]], requested_vintages: list[str]) -> list[dict[str, Any]]:
    if not requested_vintages:
        return outputs
    requested = {str(value) for value in requested_vintages}
    filtered = [
        entry
        for entry in outputs
        if entry["pair"].name in requested or _vintage_label(entry["pair"]) in requested
    ]
    if not filtered:
        raise ValueError(f"None of the requested volume vintages were found: {requested_vintages}")
    return filtered


def _inline_label(pair: Any) -> str:
    metadata = pair.metadata or {}
    inline_id = metadata.get("inline_id")
    return str(inline_id) if inline_id is not None else "default"


def _vintage_label(pair: Any) -> str:
    metadata = pair.metadata or {}
    vintage = metadata.get("vintage")
    return str(vintage) if vintage is not None else pair.name


def _sort_axis_values(values: set[str], preferred: list[Any] | None = None) -> list[str]:
    if preferred:
        normalized = [str(value) for value in preferred]
        extras = sorted(
            [value for value in values if value not in set(normalized)],
            key=lambda item: (not item.isdigit(), int(item) if item.isdigit() else item),
        )
        return normalized + extras
    return sorted(values, key=lambda item: (not item.isdigit(), int(item) if item.isdigit() else item))


def _resolve_inline_vintage_axes(outputs: list[dict[str, Any]], volume_cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    inline_values = {_inline_label(entry["pair"]) for entry in outputs}
    vintage_values = {_vintage_label(entry["pair"]) for entry in outputs}
    return (
        _sort_axis_values(inline_values, volume_cfg.get("inline_order", [])),
        _sort_axis_values(vintage_values, volume_cfg.get("vintage_order", [])),
    )
