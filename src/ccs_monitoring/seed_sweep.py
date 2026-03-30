"""Synthetic seed-stability sweeps."""

from __future__ import annotations

import csv
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .pipeline import evaluate, generate, train
from .runtime import ensure_runtime_environment


def run_seed_sweep(config: dict[str, Any]) -> dict[str, Any]:
    sweep_cfg = config.get("seed_sweep", {})
    seeds = [int(seed) for seed in sweep_cfg.get("seeds", [])]
    if not seeds:
        raise ValueError("seed_sweep.seeds must contain at least one seed.")

    base_output_root = Path(config["output_root"])
    dataset_root = Path(str(sweep_cfg.get("dataset_root", "")).strip() or base_output_root)
    sweep_output_root = Path(str(sweep_cfg.get("output_root", "")).strip() or f"{base_output_root}_seed_sweep")
    ensure_runtime_environment(sweep_output_root, config["seed"])

    if sweep_cfg.get("reuse_existing_dataset", True):
        _ensure_dataset_exists(config, dataset_root)
    else:
        dataset_cfg = deepcopy(config)
        dataset_cfg["output_root"] = str(dataset_root)
        dataset_cfg["field"]["enabled"] = False
        generate(dataset_cfg)

    results_dir = sweep_output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    methods = [str(method) for method in sweep_cfg.get("methods", ["plain_ml", "hybrid_ml"])]
    splits = [str(split) for split in sweep_cfg.get("splits", ["test", "ood"])]
    metrics = [str(metric) for metric in sweep_cfg.get("metrics", [])]

    seed_runs: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []

    for seed in seeds:
        run_root = sweep_output_root / f"seed_{seed}"
        _copy_dataset(dataset_root, run_root)

        run_cfg = deepcopy(config)
        run_cfg["seed"] = seed
        run_cfg["output_root"] = str(run_root)
        run_cfg["artifacts_root"] = ""
        run_cfg["field"]["enabled"] = False
        run_cfg["evaluation"]["save_figures"] = bool(sweep_cfg.get("save_individual_figures", False))

        train(run_cfg)
        metrics_payload = evaluate(run_cfg)
        seed_runs.append(
            {
                "seed": seed,
                "output_root": str(run_root),
                "metrics": metrics_payload,
            }
        )
        for split_name in splits:
            split_payload = metrics_payload.get(split_name, {})
            for method_name in methods:
                method_payload = split_payload.get(method_name, {})
                if not isinstance(method_payload, dict):
                    continue
                row = {
                    "seed": seed,
                    "split": split_name,
                    "method": method_name,
                }
                for metric_name in metrics:
                    if metric_name in method_payload:
                        row[metric_name] = float(method_payload[metric_name])
                per_seed_rows.append(row)

    aggregate_rows = _aggregate_seed_rows(per_seed_rows, metrics)
    summary = {
        "dataset_root": str(dataset_root),
        "sweep_output_root": str(sweep_output_root),
        "seeds": seeds,
        "methods": methods,
        "splits": splits,
        "metrics": metrics,
        "seed_runs": seed_runs,
        "aggregate_rows": aggregate_rows,
    }

    (results_dir / "seed_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(results_dir / "seed_sweep_per_seed.csv", per_seed_rows)
    _write_csv(results_dir / "seed_sweep_aggregate.csv", aggregate_rows)
    return summary


def _ensure_dataset_exists(config: dict[str, Any], dataset_root: Path) -> None:
    required_files = [dataset_root / "data" / f"{split}.npz" for split in ("train", "val", "test", "ood")]
    if all(path.exists() for path in required_files):
        return
    dataset_cfg = deepcopy(config)
    dataset_cfg["output_root"] = str(dataset_root)
    dataset_cfg["field"]["enabled"] = False
    generate(dataset_cfg)


def _copy_dataset(dataset_root: Path, run_root: Path) -> None:
    source_dir = dataset_root / "data"
    if not source_dir.exists():
        raise FileNotFoundError(f"Synthetic dataset directory not found: {source_dir}")
    target_dir = run_root / "data"
    target_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test", "ood"):
        source_path = source_dir / f"{split_name}.npz"
        target_path = target_dir / f"{split_name}.npz"
        shutil.copy2(source_path, target_path)


def _aggregate_seed_rows(rows: list[dict[str, Any]], metric_names: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["split"]), str(row["method"])), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for (split_name, method_name), group_rows in sorted(grouped.items()):
        aggregate_row = {
            "split": split_name,
            "method": method_name,
            "num_seeds": len(group_rows),
        }
        for metric_name in metric_names:
            values = [float(row[metric_name]) for row in group_rows if metric_name in row]
            if not values:
                continue
            values_array = np.asarray(values, dtype=np.float64)
            aggregate_row[f"{metric_name}_mean"] = float(np.mean(values_array))
            aggregate_row[f"{metric_name}_std"] = float(np.std(values_array, ddof=0))
            aggregate_row[f"{metric_name}_min"] = float(np.min(values_array))
            aggregate_row[f"{metric_name}_max"] = float(np.max(values_array))
        aggregate_rows.append(aggregate_row)
    return aggregate_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
