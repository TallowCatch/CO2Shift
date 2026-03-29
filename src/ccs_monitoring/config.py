"""Config loading with lightweight defaults."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 7,
    "output_root": "runs/default",
    "artifacts_root": "",
    "synthetic": {
        "num_train": 256,
        "num_val": 64,
        "num_test": 64,
        "num_ood": 64,
        "section_shape": [160, 128],
        "geology_families": {
            "train": [0, 1, 2],
            "test": [0, 1, 2],
            "ood": [3, 4],
        },
        "mismatch_probability": 0.8,
        "clean_probability": 0.2,
        "plume_radius_range": [10, 18],
        "noise_std_range": [0.01, 0.08],
        "shift_trace_range": [-4, 4],
        "wavelet_freq_range": [0.04, 0.12],
        "amplitude_scale_range": [0.85, 1.2],
        "drop_trace_fraction_range": [0.0, 0.1],
    },
    "training": {
        "batch_size": 8,
        "epochs": 10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "mc_dropout_passes": 12,
        "device": "cpu",
        "use_hybrid_augmentation": True,
        "use_reservoir_weighting": True,
        "outside_reservoir_weight": 0.35,
    },
    "evaluation": {
        "abstain_quantiles": [0.7, 0.8, 0.9],
        "threshold_quantile": 0.92,
        "save_figures": True,
    },
    "field": {
        "enabled": False,
        "mode": "manifest",
        "path": "",
        "manifest_path": "",
        "benchmark_root": "",
        "plume_boundaries_root": "",
        "segy_path": "",
        "export_segy_path": "",
        "export_output_path": "",
        "export_normalization_segy_paths": [],
        "inline_number": 0,
        "output_mask_path": "",
        "output_plume_support_path": "",
        "plume_support_path": "",
        "plume_support_note": "",
        "postprocess": {
            "enabled": False,
            "threshold_mode": "fixed",
            "probability_threshold": 0.5,
            "probability_quantile": 0.92,
            "min_probability_threshold": 0.5,
            "uncertainty_quantile": 1.0,
            "shared_across_pairs": False,
            "shared_use_reservoir_region": True,
            "apply_reservoir_mask": True,
            "closing_iterations": 0,
            "opening_iterations": 0,
            "min_component_size": 0,
            "min_component_fraction": 0.0,
            "keep_largest_components": 0,
        },
    },
    "paper_evidence": {
        "enabled": False,
        "claim": (
            "Benchmark-constrained hybrid monitoring yields more selective and more benchmark-aligned "
            "lateral plume-support maps than plain ML or a cross-equalized classical baseline on public "
            "Sleipner data."
        ),
        "synthetic_metrics_path": "",
        "field_summary_path": "",
        "direct_summary_path": "",
        "field_config_path": "",
        "direct_config_path": "",
    },
    "jax": {
        "enabled": False,
        "device": "cpu",
        "dtype": "float32",
        "grid_shape": [64, 64],
        "num_steps": 96,
        "dt": 0.001,
        "dx": 10.0,
        "dz": 10.0,
        "source_frequency": 12.0,
        "source_index": [6, 32],
        "receiver_depth": 8,
        "batch_size": 3,
        "velocity_range": [1.8, 2.4],
        "snapshot_steps": [0, 24, 48, 72, 95],
    },
    "volume": {
        "enabled": False,
        "input_manifest": "",
        "output_store": "",
        "chunking": {
            "vintage": 1,
            "sample": 256,
            "trace": 128,
        },
        "vintages": [],
        "include_predictions": True,
    },
    "hpc": {
        "launcher": "local",
        "num_workers": 1,
        "threads_per_worker": 4,
        "job_name": "ccs-monitoring",
    },
    "visualization": {
        "enabled": False,
        "mode": "plotly",
        "colormap": "Viridis",
        "opacity": 0.18,
        "vintages": [],
        "output_dir": "",
        "animation_variable": "constrained_support",
        "volume_variable": "constrained_support",
        "uncertainty_variable": "uncertainty",
        "baseline_variable": "baseline",
        "monitor_variable": "monitor",
    },
    "report": {
        "title": "Reliable 4D Change Maps for CCS Monitoring",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    config = _deep_merge(DEFAULT_CONFIG, user_config)
    config["config_path"] = str(Path(path).resolve())
    return config
