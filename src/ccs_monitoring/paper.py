"""Paper-facing evidence pack generation."""

from __future__ import annotations

import csv
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .config import load_config
from .runtime import ensure_runtime_environment


def build_paper_evidence(config: dict[str, Any]) -> dict[str, Any]:
    ensure_runtime_environment(config["output_root"], config["seed"])
    output_root = Path(config["output_root"])
    results_dir = output_root / "results"
    figures_dir = results_dir / "figures"
    paper_results_dir = Path(config["paper_evidence"].get("paper_results_dir", "paper/results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paper_results_dir.mkdir(parents=True, exist_ok=True)

    paper_cfg = config["paper_evidence"]
    synthetic_metrics = _load_json(paper_cfg["synthetic_metrics_path"])
    seed_sweep_summary = _load_optional_json(paper_cfg.get("seed_sweep_summary_path", ""))
    field_summary_payload = _load_json(paper_cfg["field_summary_path"])
    direct_summary_payload = _load_json(paper_cfg["direct_summary_path"])
    direct_config = load_config(paper_cfg["direct_config_path"])
    field_volume_summary = _extract_volume_summary(field_summary_payload)
    direct_volume_summary = _extract_volume_summary(direct_summary_payload)

    synthetic_rows = _build_synthetic_rows(synthetic_metrics, seed_sweep_summary)
    negative_control_rows = _build_negative_control_rows(synthetic_metrics)
    field_rows = _build_field_rows(field_volume_summary)
    direct_rows = _build_direct_rows(direct_volume_summary)
    ablation_rows = _build_ablation_rows(field_volume_summary, direct_volume_summary)
    field_stability_rows, field_stability_aggregate_rows = _build_field_stability_rows(paper_cfg, output_root)
    panel_path = figures_dir / "paper_direct_2010_panel.png"
    _build_direct_panel(direct_config, direct_volume_summary, panel_path)

    evidence_summary = {
        "claim": paper_cfg["claim"],
        "provenance": {
            "synthetic_metrics_path": str(Path(paper_cfg["synthetic_metrics_path"])),
            "seed_sweep_summary_path": str(Path(paper_cfg["seed_sweep_summary_path"]))
            if paper_cfg.get("seed_sweep_summary_path")
            else "",
            "field_summary_path": str(Path(paper_cfg["field_summary_path"])),
            "direct_summary_path": str(Path(paper_cfg["direct_summary_path"])),
            "field_config_path": str(Path(paper_cfg["field_config_path"])),
            "direct_config_path": str(Path(paper_cfg["direct_config_path"])),
            "paper_results_dir": str(paper_results_dir),
            "field_stability": {
                "enabled": bool(paper_cfg.get("field_stability", {}).get("enabled", False)),
                "benchmark_configs": {
                    str(name): str(Path(path))
                    for name, path in paper_cfg.get("field_stability", {}).get("benchmark_configs", {}).items()
                },
                "seed_runs": [
                    {
                        "seed": int(seed_run["seed"]),
                        "artifacts_root": str(Path(seed_run["artifacts_root"])),
                    }
                    for seed_run in paper_cfg.get("field_stability", {}).get("seed_runs", [])
                    if isinstance(seed_run, dict)
                ],
            },
        },
        "synthetic_rows": synthetic_rows,
        "field_rows": field_rows,
        "direct_rows": direct_rows,
        "ablation_rows": ablation_rows,
        "negative_control_rows": negative_control_rows,
        "field_stability_rows": field_stability_rows,
        "field_stability_aggregate_rows": field_stability_aggregate_rows,
        "synthetic_seed_summary": _build_seed_summary_rows(seed_sweep_summary),
        "methods_results_block": _build_methods_results_block(
            paper_cfg["claim"],
            synthetic_metrics,
            seed_sweep_summary,
            field_volume_summary,
            direct_volume_summary,
            field_stability_aggregate_rows,
        ),
        "panel_path": str(panel_path),
    }

    _write_json(results_dir / "paper_evidence_summary.json", evidence_summary)
    _write_csv(results_dir / "paper_synthetic_table.csv", synthetic_rows)
    _write_csv(results_dir / "paper_field_table.csv", field_rows)
    _write_csv(results_dir / "paper_direct_table.csv", direct_rows)
    _write_csv(results_dir / "paper_ablation_table.csv", ablation_rows)
    _write_csv(results_dir / "paper_negative_controls.csv", negative_control_rows)
    _write_csv(results_dir / "paper_seed_sweep_table.csv", evidence_summary["synthetic_seed_summary"])
    _write_csv(results_dir / "paper_field_stability_per_seed.csv", field_stability_rows)
    _write_csv(results_dir / "paper_field_stability_aggregate.csv", field_stability_aggregate_rows)
    (results_dir / "methods_results_block.txt").write_text(
        evidence_summary["methods_results_block"],
        encoding="utf-8",
    )
    _write_csv(paper_results_dir / "paper_field_stability_per_seed.csv", field_stability_rows)
    _write_csv(paper_results_dir / "paper_field_stability_aggregate.csv", field_stability_aggregate_rows)
    _write_json(paper_results_dir / "paper_evidence_summary.json", evidence_summary)
    return evidence_summary


def _build_synthetic_rows(metrics: dict[str, Any], seed_sweep_summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aggregate_lookup = _seed_aggregate_lookup(seed_sweep_summary)
    classical_method = _synthetic_classical_method(metrics)
    for split_name in ("test", "ood"):
        for method in (classical_method, "plain_ml", "hybrid_ml"):
            aggregate = aggregate_lookup.get((split_name, method))
            if aggregate is not None:
                rows.append(
                    {
                        "split": split_name,
                        "method": method,
                        "source": "seed_sweep",
                        "num_seeds": int(aggregate.get("num_seeds", 0)),
                        "dice": float(aggregate["dice_mean"]),
                        "dice_std": float(aggregate["dice_std"]),
                        "iou": float(aggregate["iou_mean"]),
                        "iou_std": float(aggregate["iou_std"]),
                        "false_positive_rate": float(aggregate["false_positive_rate_mean"]),
                        "false_positive_rate_std": float(aggregate["false_positive_rate_std"]),
                        "ece": float(aggregate["ece_mean"]),
                        "ece_std": float(aggregate["ece_std"]),
                    }
                )
                continue
            rows.append(
                {
                    "split": split_name,
                    "method": method,
                    "source": "single_run",
                    "num_seeds": 1,
                    "dice": float(metrics[split_name][method]["dice"]),
                    "dice_std": 0.0,
                    "iou": float(metrics[split_name][method]["iou"]),
                    "iou_std": 0.0,
                    "false_positive_rate": float(metrics[split_name][method]["false_positive_rate"]),
                    "false_positive_rate_std": 0.0,
                    "ece": float(metrics[split_name][method]["ece"]),
                    "ece_std": 0.0,
                }
            )
    return rows


def _build_seed_summary_rows(seed_sweep_summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not seed_sweep_summary:
        return []
    rows: list[dict[str, Any]] = []
    for row in seed_sweep_summary.get("aggregate_rows", []):
        rows.append(
            {
                "split": str(row["split"]),
                "method": str(row["method"]),
                "num_seeds": int(row["num_seeds"]),
                "dice_mean": float(row["dice_mean"]),
                "dice_std": float(row["dice_std"]),
                "iou_mean": float(row["iou_mean"]),
                "iou_std": float(row["iou_std"]),
                "false_positive_rate_mean": float(row["false_positive_rate_mean"]),
                "false_positive_rate_std": float(row["false_positive_rate_std"]),
                "ece_mean": float(row["ece_mean"]),
                "ece_std": float(row["ece_std"]),
                "brier_mean": float(row["brier_mean"]),
                "brier_std": float(row["brier_std"]),
                "nll_mean": float(row["nll_mean"]),
                "nll_std": float(row["nll_std"]),
            }
        )
    return rows


def _build_negative_control_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    scenario_breakdown = metrics.get("ood", {}).get("scenario_breakdown", {})
    target_scenarios = ["no_change", "mismatch_only", "out_of_zone"]
    rows: list[dict[str, Any]] = []
    for scenario_name in target_scenarios:
        scenario_metrics = scenario_breakdown.get(scenario_name)
        if not isinstance(scenario_metrics, dict):
            continue
        for method_name in ("difference", "nrms_difference", "plain_ml", "hybrid_ml"):
            method_metrics = scenario_metrics.get(method_name)
            if not isinstance(method_metrics, dict):
                continue
            rows.append(
                {
                    "scenario": scenario_name,
                    "method": method_name,
                    "num_samples": int(scenario_metrics.get("num_samples", 0)),
                    "false_positive_rate": float(method_metrics.get("false_positive_rate", float("nan"))),
                    "dice": float(method_metrics.get("dice", float("nan"))),
                    "ece": float(method_metrics.get("ece", float("nan"))),
                    "risk_coverage_auc": float(method_metrics.get("risk_coverage_auc", float("nan"))),
                }
            )
    return rows


def _extract_volume_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if "volume_summary" in payload:
        return payload["volume_summary"]
    if "Field" in payload and isinstance(payload["Field"], dict) and "volume_summary" in payload["Field"]:
        return payload["Field"]["volume_summary"]
    raise KeyError("Expected a volume_summary payload or a summary.json containing Field.volume_summary.")


def _build_field_rows(volume_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    methods = [
        "best_classical_constrained",
        "plain_ml_constrained",
        "plain_ml_structured_constrained",
        "plain_ml_layered_structured_constrained",
        "hybrid_ml_constrained",
        "hybrid_ml_structured_constrained",
    ]
    for vintage, method_metrics in volume_summary.get("by_vintage", {}).items():
        for method_name in methods:
            if method_name not in method_metrics:
                continue
            metrics = method_metrics[method_name]
            rows.append(
                {
                    "benchmark": "p07_temporal",
                    "scope": "by_vintage",
                    "vintage": vintage,
                    "method": method_name,
                    "trace_iou_with_2010_support": float(metrics.get("trace_iou_with_2010_support", float("nan"))),
                    "support_volume_iou_2010": float(metrics.get("support_volume_iou_2010", float("nan"))),
                    "crossline_continuity": float(metrics.get("crossline_continuity", float("nan"))),
                    "predicted_trace_fraction": float(metrics.get("predicted_trace_fraction", float("nan"))),
                    "predicted_fraction": float(metrics.get("predicted_fraction", float("nan"))),
                    "predicted_fraction_outside_support_volume": float(
                        metrics.get("predicted_fraction_outside_support_volume", float("nan"))
                    ),
                }
            )
    for method_name in methods:
        metrics = volume_summary.get("overall", {}).get(method_name)
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "benchmark": "p07_temporal",
                "scope": "overall",
                "vintage": "all",
                "method": method_name,
                "trace_iou_with_2010_support": float(metrics.get("trace_iou_with_2010_support", float("nan"))),
                "support_volume_iou_2010": float(metrics.get("support_volume_iou_2010", float("nan"))),
                "crossline_continuity": float(metrics.get("crossline_continuity", float("nan"))),
                "predicted_trace_fraction": float(metrics.get("predicted_trace_fraction", float("nan"))),
                "predicted_fraction": float(metrics.get("predicted_fraction", float("nan"))),
                "predicted_fraction_outside_support_volume": float(
                    metrics.get("predicted_fraction_outside_support_volume", float("nan"))
                ),
            }
        )
    return rows


def _build_direct_rows(volume_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_name in [
        "best_classical_constrained",
        "plain_ml_constrained",
        "plain_ml_structured_constrained",
        "plain_ml_layered_structured_constrained",
        "hybrid_ml_constrained",
        "hybrid_ml_structured_constrained",
    ]:
        metrics = volume_summary.get("overall", {}).get(method_name)
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "benchmark": "p10_direct",
                "method": method_name,
                "predicted_trace_fraction": float(metrics.get("predicted_trace_fraction", float("nan"))),
                "trace_iou_with_2010_support": float(metrics.get("trace_iou_with_2010_support", float("nan"))),
                "support_volume_iou_2010": float(metrics.get("support_volume_iou_2010", float("nan"))),
                "crossline_continuity": float(metrics.get("crossline_continuity", float("nan"))),
                "predicted_fraction_outside_support_volume": float(
                    metrics.get("predicted_fraction_outside_support_volume", float("nan"))
                ),
            }
        )
    return rows


def _build_ablation_rows(field_volume_summary: dict[str, Any], direct_volume_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for benchmark_name, volume_summary in [("p07_temporal", field_volume_summary), ("p10_direct", direct_volume_summary)]:
        for method_name in [
            "best_classical_constrained",
            "plain_ml_constrained",
            "plain_ml_structured_constrained",
            "plain_ml_layered_structured_constrained",
            "hybrid_ml_constrained",
            "hybrid_ml_structured_constrained",
        ]:
            metrics = volume_summary.get("overall", {}).get(method_name)
            if not isinstance(metrics, dict):
                continue
            rows.append(
                {
                    "benchmark": benchmark_name,
                    "variant": method_name,
                    "trace_iou_with_2010_support": float(metrics.get("trace_iou_with_2010_support", float("nan"))),
                    "support_volume_iou_2010": float(metrics.get("support_volume_iou_2010", float("nan"))),
                    "crossline_continuity": float(metrics.get("crossline_continuity", float("nan"))),
                    "predicted_fraction_outside_support_volume": float(
                        metrics.get("predicted_fraction_outside_support_volume", float("nan"))
                    ),
                }
            )
    return rows


def _build_direct_panel(config: dict[str, Any], volume_summary: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    store_path = Path(config["volume"].get("output_store") or (Path(config["output_root"]) / "volume.zarr"))
    dataset = xr.open_zarr(store_path)
    inline_values = [str(value) for value in dataset.coords["inline"].values.tolist()]
    inline_index = inline_values.index("1840") if "1840" in inline_values else len(inline_values) // 2
    benchmark_image = np.asarray(dataset["support_volume_2010"].isel(inline=inline_index).values, dtype=np.float32)
    classical_image = np.asarray(
        dataset["best_classical_constrained"].isel(inline=inline_index, vintage=-1).values,
        dtype=np.float32,
    )
    plain_image = np.asarray(
        dataset["plain_ml_constrained"].isel(inline=inline_index, vintage=-1).values,
        dtype=np.float32,
    )
    structured_image = np.asarray(
        dataset["plain_ml_structured_constrained"].isel(inline=inline_index, vintage=-1).values,
        dtype=np.float32,
    )
    layered_structured_image = np.asarray(
        dataset["plain_ml_layered_structured_constrained"].isel(inline=inline_index, vintage=-1).values,
        dtype=np.float32,
    )
    titles = [
        f"A. Best classical ({volume_summary['overall']['best_classical_constrained']['support_volume_iou_2010']:.3f})",
        f"B. Plain ML ({volume_summary['overall']['plain_ml_constrained']['support_volume_iou_2010']:.3f})",
        (
            "C. Plain ML + structured support "
            f"({volume_summary['overall']['plain_ml_structured_constrained']['support_volume_iou_2010']:.3f})"
        ),
        (
            "D. Plain ML + layered structured support "
            f"({volume_summary['overall']['plain_ml_layered_structured_constrained']['support_volume_iou_2010']:.3f})"
        ),
        "E. Benchmark support",
    ]
    images = [classical_image, plain_image, structured_image, layered_structured_image, benchmark_image]
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(image, cmap="magma", aspect="auto", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_field_stability_rows(
    paper_cfg: dict[str, Any],
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stability_cfg = paper_cfg.get("field_stability", {})
    if not stability_cfg.get("enabled", False):
        return [], []

    benchmark_configs = stability_cfg.get("benchmark_configs", {})
    seed_runs = stability_cfg.get("seed_runs", [])
    methods = [str(method_name) for method_name in stability_cfg.get("methods", [])]
    metrics = [str(metric_name) for metric_name in stability_cfg.get("metrics", [])]
    if not benchmark_configs or not seed_runs or not methods or not metrics:
        return [], []

    from .pipeline import evaluate_field_only

    per_seed_rows: list[dict[str, Any]] = []
    for benchmark_name, config_path in benchmark_configs.items():
        base_config = load_config(config_path)
        for seed_run in seed_runs:
            seed = int(seed_run["seed"])
            artifacts_root = Path(seed_run["artifacts_root"])
            run_config = deepcopy(base_config)
            run_config["seed"] = seed
            run_config["artifacts_root"] = str(artifacts_root)
            run_config["output_root"] = str(output_root / "field_stability" / str(benchmark_name) / f"seed_{seed}")
            run_config["config_path"] = str(Path(config_path).resolve())
            summary_path = Path(run_config["output_root"]) / "results" / "summary.json"
            if summary_path.exists():
                summary_payload = _load_json(summary_path)
                field_results = summary_payload.get("Field", {})
            else:
                field_results = evaluate_field_only(run_config)
            overall_metrics = field_results.get("volume_summary", {}).get("overall", {})
            for method_name in methods:
                method_metrics = overall_metrics.get(method_name)
                if not isinstance(method_metrics, dict):
                    continue
                row = {
                    "benchmark": str(benchmark_name),
                    "seed": seed,
                    "method": method_name,
                    "config_path": str(Path(config_path)),
                    "artifacts_root": str(artifacts_root),
                    "output_root": str(Path(run_config["output_root"])),
                }
                for metric_name in metrics:
                    row[metric_name] = float(method_metrics.get(metric_name, float("nan")))
                per_seed_rows.append(row)

    aggregate_rows = _aggregate_field_stability_rows(per_seed_rows, metrics)
    return per_seed_rows, aggregate_rows


def _aggregate_field_stability_rows(
    per_seed_rows: list[dict[str, Any]],
    metrics: list[str],
) -> list[dict[str, Any]]:
    if not per_seed_rows:
        return []

    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in per_seed_rows:
        grouped_rows.setdefault((str(row["benchmark"]), str(row["method"])), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for benchmark_name, method_name in sorted(grouped_rows):
        rows = grouped_rows[(benchmark_name, method_name)]
        aggregate = {
            "benchmark": benchmark_name,
            "method": method_name,
            "num_seeds": len(rows),
        }
        for metric_name in metrics:
            values = np.asarray([float(row.get(metric_name, float("nan"))) for row in rows], dtype=np.float64)
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                aggregate[f"{metric_name}_mean"] = float("nan")
                aggregate[f"{metric_name}_std"] = float("nan")
                aggregate[f"{metric_name}_min"] = float("nan")
                aggregate[f"{metric_name}_max"] = float("nan")
                continue
            aggregate[f"{metric_name}_mean"] = float(np.mean(finite_values))
            aggregate[f"{metric_name}_std"] = float(np.std(finite_values))
            aggregate[f"{metric_name}_min"] = float(np.min(finite_values))
            aggregate[f"{metric_name}_max"] = float(np.max(finite_values))
        aggregate_rows.append(aggregate)
    return aggregate_rows


def _build_methods_results_block(
    claim: str,
    synthetic_metrics: dict[str, Any],
    seed_sweep_summary: dict[str, Any] | None,
    field_volume_summary: dict[str, Any],
    direct_volume_summary: dict[str, Any],
    field_stability_aggregate_rows: list[dict[str, Any]],
) -> str:
    classical_method = _synthetic_classical_method(synthetic_metrics)
    aggregate_lookup = _seed_aggregate_lookup(seed_sweep_summary)
    ood_plain = aggregate_lookup.get(("ood", "plain_ml"))
    ood_hybrid = aggregate_lookup.get(("ood", "hybrid_ml"))
    test_plain = aggregate_lookup.get(("test", "plain_ml"))
    test_hybrid = aggregate_lookup.get(("test", "hybrid_ml"))
    p07_plain = field_volume_summary["overall"]["plain_ml_constrained"]
    p07_structured = field_volume_summary["overall"]["plain_ml_structured_constrained"]
    p07_layered = field_volume_summary["overall"]["plain_ml_layered_structured_constrained"]
    p10_plain = direct_volume_summary["overall"]["plain_ml_constrained"]
    p10_structured = direct_volume_summary["overall"]["plain_ml_structured_constrained"]
    p10_layered = direct_volume_summary["overall"]["plain_ml_layered_structured_constrained"]
    p10_classical = direct_volume_summary["overall"]["best_classical_constrained"]
    p10_hybrid = direct_volume_summary["overall"]["hybrid_ml_constrained"]
    stability_lookup = {
        (str(row["benchmark"]), str(row["method"])): row for row in field_stability_aggregate_rows
    }
    p07_structured_stability = stability_lookup.get(("p07", "plain_ml_structured_constrained"))
    p10_structured_stability = stability_lookup.get(("p10", "plain_ml_structured_constrained"))
    p07_plain_stability = stability_lookup.get(("p07", "plain_ml_constrained"))
    p10_plain_stability = stability_lookup.get(("p10", "plain_ml_constrained"))
    if ood_plain and ood_hybrid and test_plain and test_hybrid:
        synthetic_text = (
            f"Across a three-seed sweep on the harder benchmark-v2 synthetic task, plain ML reaches mean Dice "
            f"{ood_plain['dice_mean']:.3f} ± {ood_plain['dice_std']:.3f} on the OOD split and "
            f"{test_plain['dice_mean']:.3f} ± {test_plain['dice_std']:.3f} on the test split. "
            f"The hybrid model reaches {ood_hybrid['dice_mean']:.3f} ± {ood_hybrid['dice_std']:.3f} on OOD and "
            f"{test_hybrid['dice_mean']:.3f} ± {test_hybrid['dice_std']:.3f} on test, while the strongest "
            f"deterministic classical baseline ({classical_method}) reaches "
            f"{synthetic_metrics['ood'][classical_method]['dice']:.3f} Dice on the same OOD benchmark. "
            "This means the seed sweep does not support a hybrid-centered synthetic story."
        )
    else:
        synthetic_text = (
            f"On the harder benchmark-v2 synthetic OOD split, plain ML reaches Dice "
            f"{synthetic_metrics['ood']['plain_ml']['dice']:.3f}, the hybrid model reaches "
            f"{synthetic_metrics['ood']['hybrid_ml']['dice']:.3f}, and the strongest classical synthetic baseline "
            f"({classical_method}) reaches {synthetic_metrics['ood'][classical_method]['dice']:.3f}."
        )
    stability_text = ""
    if (
        p07_structured_stability
        and p10_structured_stability
        and p07_plain_stability
        and p10_plain_stability
    ):
        stability_text = (
            "Across the reused three-seed field evaluation, structured plain ML remains the strongest support-"
            f"mapping variant on p10 with support-volume IoU {p10_structured_stability['support_volume_iou_2010_mean']:.3f} "
            f"± {p10_structured_stability['support_volume_iou_2010_std']:.3f}, compared with "
            f"{p10_plain_stability['support_volume_iou_2010_mean']:.3f} ± {p10_plain_stability['support_volume_iou_2010_std']:.3f} "
            "for plain ML alone. On p07, structured plain ML remains stronger than plain ML on support-volume IoU "
            f"({p07_structured_stability['support_volume_iou_2010_mean']:.3f} ± {p07_structured_stability['support_volume_iou_2010_std']:.3f} "
            f"versus {p07_plain_stability['support_volume_iou_2010_mean']:.3f} ± {p07_plain_stability['support_volume_iou_2010_std']:.3f}), "
            "while the benchmark-constrained classical baseline still retains the highest raw p07 support-volume IoU. "
            "This stability check therefore supports a benchmark/protocol claim rather than a blanket classical-beating claim."
        )
    return (
        f"Claim: {claim}\n\n"
        "Methods: We keep the 2D PyTorch models and classical baselines, but the main field-facing result now "
        "centers on plain ML plus structured support reconstruction rather than the hybrid model. We evaluate both "
        "the original structured reconstruction and a new layered structured variant that uses benchmark-informed "
        "reservoir bands as a deterministic pseudo-3D refinement stage. Field outputs are benchmark-constrained "
        "using the public Sleipner storage interval, and the 3D evidence is reported on 11-inline pseudo-3D p07 "
        "and p10 benchmarks.\n\n"
        f"Results: {synthetic_text} "
        f"On the 11-inline p07 temporal benchmark, structured plain ML improves support-volume IoU from "
        f"{p07_plain['support_volume_iou_2010']:.3f} to {p07_structured['support_volume_iou_2010']:.3f}, and the "
        f"layered variant drops it to {p07_layered['support_volume_iou_2010']:.3f}. Trace-support IoU "
        f"moves from {p07_plain['trace_iou_with_2010_support']:.3f} to {p07_structured['trace_iou_with_2010_support']:.3f} "
        f"and then down to {p07_layered['trace_iou_with_2010_support']:.3f}, while crossline continuity moves from "
        f"{p07_plain['crossline_continuity']:.3f} to {p07_structured['crossline_continuity']:.3f} to "
        f"{p07_layered['crossline_continuity']:.3f}. "
        f"On the direct 11-inline p10 benchmark, structured plain ML improves support-volume IoU from "
        f"{p10_plain['support_volume_iou_2010']:.3f} to {p10_structured['support_volume_iou_2010']:.3f}, and the "
        f"layered variant drops it to {p10_layered['support_volume_iou_2010']:.3f}. Lateral support alignment "
        f"moves from {p10_plain['trace_iou_with_2010_support']:.3f} to {p10_structured['trace_iou_with_2010_support']:.3f} "
        f"and then down to {p10_layered['trace_iou_with_2010_support']:.3f}, and the best constrained classical baseline reaches "
        f"{p10_classical['support_volume_iou_2010']:.3f} support-volume IoU. "
        f"The hybrid field outputs remain weaker on the direct benchmark "
        f"({p10_hybrid['support_volume_iou_2010']:.3f} support-volume IoU), so the current defensible field claim "
        "should center on plain ML plus structured support reconstruction. "
        f"{stability_text} "
        "The layered extension is therefore a "
        "tested negative result, not the current field method of record."
    )


def _synthetic_classical_method(metrics: dict[str, Any]) -> str:
    candidates = [
        method_name
        for method_name in metrics["ood"]
        if method_name not in {"plain_ml", "hybrid_ml", "runtime_seconds", "scenario_breakdown"}
        and isinstance(metrics["ood"][method_name], dict)
        and "dice" in metrics["ood"][method_name]
    ]
    if not candidates:
        return "difference"
    return max(candidates, key=lambda name: float(metrics["ood"][name]["dice"]))


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_optional_json(path: str | Path) -> dict[str, Any] | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return _load_json(candidate)


def _seed_aggregate_lookup(seed_sweep_summary: dict[str, Any] | None) -> dict[tuple[str, str], dict[str, Any]]:
    if not seed_sweep_summary:
        return {}
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in seed_sweep_summary.get("aggregate_rows", []):
        split_name = str(row["split"])
        method_name = str(row["method"])
        normalized_row: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, float) and math.isnan(value):
                normalized_row[key] = float("nan")
            else:
                normalized_row[key] = value
        lookup[(split_name, method_name)] = normalized_row
    return lookup


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
