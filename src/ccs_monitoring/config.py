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
        "benchmark_version": "v2",
        "num_train": 256,
        "num_val": 64,
        "num_test": 64,
        "num_ood": 64,
        "section_shape": [160, 128],
        "num_monitor_vintages": 3,
        "geology_families": {
            "train": [0, 1, 2],
            "test": [0, 1, 2],
            "ood": [3, 4],
        },
        "scenario_probabilities": {
            "plume_growth": 0.55,
            "mismatch_only": 0.2,
            "no_change": 0.1,
            "out_of_zone": 0.15,
        },
        "mismatch_probability": 0.8,
        "clean_probability": 0.2,
        "plume_radius_range": [10, 18],
        "plume_layer_count_range": [2, 4],
        "plume_layer_thickness_range": [2, 5],
        "plume_growth_curve": [0.45, 0.75, 1.0],
        "plume_strength_range": [0.08, 0.24],
        "noise_std_range": [0.01, 0.08],
        "shift_trace_range": [-4, 4],
        "static_shift_range": [-3, 3],
        "wavelet_freq_range": [0.04, 0.12],
        "amplitude_scale_range": [0.85, 1.2],
        "drop_trace_fraction_range": [0.0, 0.1],
        "coherent_noise_scale_range": [0.0, 0.04],
        "overburden_artifact_scale_range": [0.0, 0.06],
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
        "coverage_quantiles": [0.5, 0.7, 0.8, 0.9, 0.95],
        "threshold_quantile": 0.92,
        "scenario_breakdown": True,
        "save_figures": True,
    },
    "field": {
        "enabled": False,
        "mode": "manifest",
        "path": "",
        "manifest_path": "",
        "processing_family": "",
        "source_tag": "mid",
        "benchmark_root": "",
        "plume_boundaries_root": "",
        "segy_path": "",
        "export_segy_path": "",
        "export_output_path": "",
        "export_normalization_segy_paths": [],
        "inline_number": 0,
        "inline_numbers": [],
        "vintage_map": {},
        "output_manifest_path": "",
        "output_mask_path": "",
        "output_plume_support_path": "",
        "output_support_volume_path": "",
        "output_support_volume_path_template": "",
        "plume_support_path": "",
        "plume_support_volume_path": "",
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
        "pseudo3d": {
            "enabled": False,
            "window_size": 3,
            "method": "uncertainty_weighted_mean",
            "closing_iterations": 1,
            "opening_iterations": 0,
            "min_component_size": 600,
            "min_component_fraction": 0.0,
            "keep_largest_components": 3,
        },
        "structured_support": {
            "enabled": False,
            "trace_window_size": 3,
            "trace_vote_fraction": 0.34,
            "vertical_sigma": 1.25,
            "column_relative_threshold": 0.45,
            "min_peak_probability": 0.5,
            "vertical_margin": 2,
            "closing_iterations": 0,
            "opening_iterations": 0,
            "min_component_size": 0,
            "min_component_fraction": 0.0,
            "keep_largest_components": 0,
        },
        "layered_structured_support": {
            "enabled": False,
            "trace_window_size": 3,
            "trace_vote_fraction": 0.34,
            "neighbor_probability_weight": 0.3,
            "num_bands": 4,
            "min_band_peak_probability": 0.5,
            "min_band_mean_probability": 0.18,
            "min_seed_fraction": 0.02,
            "band_smoothing_iterations": 1,
            "closing_iterations": 1,
            "opening_iterations": 0,
            "min_component_size": 0,
            "min_component_fraction": 0.0,
            "keep_largest_components": 0,
        },
    },
    "paper_evidence": {
        "enabled": False,
        "claim": (
            "Benchmark-constrained hybrid monitoring yields more selective field change maps than plain ML and "
            "stronger lateral plume-support alignment than classical difference-style baselines on public "
            "Sleipner data, but its advantage weakens under stricter support-volume occupancy proxies."
        ),
        "synthetic_metrics_path": "",
        "field_summary_path": "",
        "direct_summary_path": "",
        "field_config_path": "",
        "direct_config_path": "",
    },
    "seed_sweep": {
        "enabled": False,
        "dataset_root": "",
        "output_root": "",
        "seeds": [11, 12, 13],
        "methods": ["plain_ml", "hybrid_ml"],
        "splits": ["test", "ood"],
        "metrics": ["dice", "iou", "false_positive_rate", "ece", "brier", "nll", "risk_coverage_auc"],
        "reuse_existing_dataset": True,
        "save_individual_figures": False,
    },
    "jax": {
        "enabled": False,
        "device": "cpu",
        "dtype": "float64",
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
        "objective": "wavefield_misfit",
        "target_perturbation_center": [0.16, 0.0],
        "target_perturbation_sigma": [0.05, 0.14],
        "target_perturbation_strength": 0.15,
        "gradient_check_epsilons": [1e-1, 3e-2, 1e-2, 3e-3, 1e-3],
        "min_gradient_l2_norm": 1e-14,
        "max_gradient_check_relative_error": 5e-2,
    },
    "volume": {
        "enabled": False,
        "input_manifest": "",
        "output_store": "",
        "chunking": {
            "inline": 1,
            "vintage": 1,
            "sample": 256,
            "trace": 128,
        },
        "inline_order": [],
        "vintage_order": [],
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
        "animation_variable": "plain_ml_structured_constrained",
        "volume_variable": "plain_ml_structured_constrained",
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
