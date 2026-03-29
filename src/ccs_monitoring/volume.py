"""Chunked volume artifacts for 4D-style browsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import xarray as xr

from .field_tools import collect_field_prediction_bundle
from .pipeline import _cleanup_field_binary, _postprocess_field_prediction
from .runtime import ensure_runtime_environment


def build_volume(config: dict[str, Any]) -> dict[str, Any]:
    ensure_runtime_environment(config["output_root"], config["seed"])
    volume_cfg = config["volume"]
    output_store = Path(volume_cfg["output_store"] or (Path(config["output_root"]) / "volume.zarr"))
    output_store.parent.mkdir(parents=True, exist_ok=True)

    bundle = collect_field_prediction_bundle(config)
    if not bundle.outputs:
        raise ValueError("build-volume requires field.enabled=true with at least one field pair.")

    outputs = _filter_outputs(bundle.outputs, volume_cfg.get("vintages", []))
    first_pair = outputs[0]["pair"]
    nt, nx = first_pair.baseline.shape
    vintage_names = [entry["pair"].name for entry in outputs]

    baseline = first_pair.baseline.astype(np.float32)
    monitor = np.stack([entry["pair"].monitor.astype(np.float32) for entry in outputs], axis=0)
    raw_probability = np.stack([entry["hybrid_probs"].astype(np.float32) for entry in outputs], axis=0)
    uncertainty = np.stack([entry["hybrid_uncertainty"].astype(np.float32) for entry in outputs], axis=0)
    cross_equalized = np.stack(
        [
            entry["cross_equalized_binary"].astype(np.float32)
            if "cross_equalized_binary" in entry
            else np.zeros((nt, nx), dtype=np.float32)
            for entry in _ensure_cross_equalized(outputs, bundle.cross_equalized_threshold, config)
        ],
        axis=0,
    )
    constrained = np.stack(
        [
            _postprocess_field_prediction(
                entry["hybrid_probs"],
                entry["hybrid_uncertainty"],
                entry["pair"].reservoir_mask,
                config.get("field", {}),
                shared_context=bundle.shared_context,
            )[0].astype(np.float32)
            for entry in outputs
        ],
        axis=0,
    )
    reservoir_mask = (
        first_pair.reservoir_mask.astype(np.float32)
        if first_pair.reservoir_mask is not None
        else np.ones_like(first_pair.baseline, dtype=np.float32)
    )
    support_trace = (
        bundle.plume_support_traces.astype(np.float32)
        if bundle.plume_support_traces is not None
        else np.zeros(nx, dtype=np.float32)
    )
    support_volume = reservoir_mask * support_trace[None, :]

    chunks = volume_cfg.get("chunking", {})
    vintage_chunk = int(chunks.get("vintage", 1))
    sample_chunk = int(chunks.get("sample", min(nt, 256)))
    trace_chunk = int(chunks.get("trace", min(nx, 128)))

    dataset = xr.Dataset(
        data_vars={
            "baseline": (("sample", "trace"), baseline.astype(np.float32)),
            "monitor": (("vintage", "sample", "trace"), da.from_array(monitor, chunks=(vintage_chunk, sample_chunk, trace_chunk))),
            "raw_probability": (
                ("vintage", "sample", "trace"),
                da.from_array(raw_probability, chunks=(vintage_chunk, sample_chunk, trace_chunk)),
            ),
            "uncertainty": (
                ("vintage", "sample", "trace"),
                da.from_array(uncertainty, chunks=(vintage_chunk, sample_chunk, trace_chunk)),
            ),
            "cross_equalized_support": (
                ("vintage", "sample", "trace"),
                da.from_array(cross_equalized, chunks=(vintage_chunk, sample_chunk, trace_chunk)),
            ),
            "constrained_support": (
                ("vintage", "sample", "trace"),
                da.from_array(constrained, chunks=(vintage_chunk, sample_chunk, trace_chunk)),
            ),
            "reservoir_mask": (("sample", "trace"), reservoir_mask.astype(np.float32)),
            "support_trace_2010": (("trace",), support_trace.astype(np.float32)),
            "support_volume_2010": (("sample", "trace"), support_volume.astype(np.float32)),
        },
        coords={
            "vintage": np.array(vintage_names, dtype=object),
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
        "vintages": vintage_names,
        "shape": {
            "vintage": len(vintage_names),
            "sample": nt,
            "trace": nx,
        },
        "chunks": {
            "vintage": vintage_chunk,
            "sample": sample_chunk,
            "trace": trace_chunk,
        },
        "variables": list(dataset.data_vars),
    }
    manifest_path = Path(config["output_root"]) / "results" / "volume_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _filter_outputs(outputs: list[dict[str, Any]], requested_vintages: list[str]) -> list[dict[str, Any]]:
    if not requested_vintages:
        return outputs
    requested = set(requested_vintages)
    filtered = [entry for entry in outputs if entry["pair"].name in requested]
    if not filtered:
        raise ValueError(f"None of the requested volume vintages were found: {requested_vintages}")
    return filtered


def _ensure_cross_equalized(
    outputs: list[dict[str, Any]],
    threshold: float,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    from .baselines import score_cross_equalized_difference

    ensured: list[dict[str, Any]] = []
    for entry in outputs:
        if "cross_equalized_binary" not in entry:
            pair = entry["pair"]
            score = score_cross_equalized_difference(pair.baseline, pair.monitor, pair.reservoir_mask)
            structured_binary, _ = _cleanup_field_binary(score >= threshold, pair.reservoir_mask, config.get("field", {}))
            entry = {
                **entry,
                "cross_equalized_binary": structured_binary.astype(np.float32),
            }
        ensured.append(entry)
    return ensured
