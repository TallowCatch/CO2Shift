"""Paper-facing evidence pack generation."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from .config import load_config
from .field_tools import collect_field_prediction_bundle, summarize_field_prediction_bundle
from .runtime import ensure_runtime_environment


def build_paper_evidence(config: dict[str, Any]) -> dict[str, Any]:
    ensure_runtime_environment(config["output_root"], config["seed"])
    output_root = Path(config["output_root"])
    results_dir = output_root / "results"
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    paper_cfg = config["paper_evidence"]
    synthetic_metrics = _load_json(paper_cfg["synthetic_metrics_path"])
    field_summary = _load_json(paper_cfg["field_summary_path"])
    direct_summary = _load_json(paper_cfg["direct_summary_path"])
    field_config = load_config(paper_cfg["field_config_path"])
    direct_config = load_config(paper_cfg["direct_config_path"])

    synthetic_rows = _build_synthetic_rows(synthetic_metrics)
    field_rows = _build_field_rows(field_summary["Field"])
    direct_rows = _build_direct_rows(direct_summary["Field"])
    ablation_rows = _build_ablation_rows(field_config)
    panel_path = figures_dir / "paper_direct_2010_panel.png"
    _build_direct_panel(direct_config, panel_path)

    evidence_summary = {
        "claim": paper_cfg["claim"],
        "synthetic_rows": synthetic_rows,
        "field_rows": field_rows,
        "direct_rows": direct_rows,
        "ablation_rows": ablation_rows,
        "methods_results_block": _build_methods_results_block(
            paper_cfg["claim"],
            synthetic_metrics,
            field_summary["Field"],
            direct_summary["Field"],
        ),
        "panel_path": str(panel_path),
    }

    _write_json(results_dir / "paper_evidence_summary.json", evidence_summary)
    _write_csv(results_dir / "paper_synthetic_table.csv", synthetic_rows)
    _write_csv(results_dir / "paper_field_table.csv", field_rows)
    _write_csv(results_dir / "paper_direct_table.csv", direct_rows)
    _write_csv(results_dir / "paper_ablation_table.csv", ablation_rows)
    (results_dir / "methods_results_block.txt").write_text(
        evidence_summary["methods_results_block"],
        encoding="utf-8",
    )
    return evidence_summary


def _build_synthetic_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    classical_method = _synthetic_classical_method(metrics)
    for split_name in ("test", "ood"):
        for method in (classical_method, "plain_ml", "hybrid_ml"):
            rows.append(
                {
                    "split": split_name,
                    "method": method,
                    "dice": float(metrics[split_name][method]["dice"]),
                    "iou": float(metrics[split_name][method]["iou"]),
                    "false_positive_rate": float(metrics[split_name][method]["false_positive_rate"]),
                    "ece": float(metrics[split_name][method]["ece"]),
                }
            )
    return rows


def _build_field_rows(field_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_name, metrics in field_summary["pairs"].items():
        rows.append(
            {
                "pair": pair_name,
                "cross_eq_trace_iou_2010": float(metrics["cross_equalized_difference"]["trace_iou_with_2010_support"]),
                "cross_eq_trace_fraction": float(metrics["cross_equalized_difference"]["predicted_trace_fraction"]),
                "hybrid_constrained_trace_iou_2010": float(
                    metrics["hybrid_ml_constrained"]["trace_iou_with_2010_support"]
                ),
                "hybrid_constrained_trace_fraction": float(
                    metrics["hybrid_ml_constrained"]["predicted_trace_fraction"]
                ),
                "hybrid_constrained_outside_reservoir": float(
                    metrics["hybrid_ml_constrained"]["outside_reservoir_fraction"]
                ),
            }
        )
    return rows


def _build_direct_rows(field_summary: dict[str, Any]) -> list[dict[str, Any]]:
    pair_name = next(iter(field_summary["pairs"]))
    pair_metrics = field_summary["pairs"][pair_name]
    return [
        {
            "pair": pair_name,
            "method": "cross_equalized_difference",
            "predicted_trace_fraction": float(pair_metrics["cross_equalized_difference"]["predicted_trace_fraction"]),
            "trace_iou_with_2010_support": float(
                pair_metrics["cross_equalized_difference"]["trace_iou_with_2010_support"]
            ),
            "trace_fraction_outside_2010_support": float(
                pair_metrics["cross_equalized_difference"]["trace_fraction_outside_2010_support"]
            ),
        },
        {
            "pair": pair_name,
            "method": "hybrid_ml_constrained",
            "predicted_trace_fraction": float(pair_metrics["hybrid_ml_constrained"]["predicted_trace_fraction"]),
            "trace_iou_with_2010_support": float(pair_metrics["hybrid_ml_constrained"]["trace_iou_with_2010_support"]),
            "trace_fraction_outside_2010_support": float(
                pair_metrics["hybrid_ml_constrained"]["trace_fraction_outside_2010_support"]
            ),
        },
    ]


def _build_ablation_rows(field_config: dict[str, Any]) -> list[dict[str, Any]]:
    variant_cfgs = {
        "baseline_protocol": deepcopy(field_config),
        "no_shared_thresholds": deepcopy(field_config),
        "no_reservoir_constraint": deepcopy(field_config),
        "no_uncertainty_gating": deepcopy(field_config),
    }
    variant_cfgs["no_shared_thresholds"]["field"]["postprocess"]["shared_across_pairs"] = False
    variant_cfgs["no_reservoir_constraint"]["field"]["postprocess"]["apply_reservoir_mask"] = False
    variant_cfgs["no_uncertainty_gating"]["field"]["postprocess"]["uncertainty_quantile"] = 1.0

    base_bundle = collect_field_prediction_bundle(field_config)
    rows: list[dict[str, Any]] = []
    for name, variant_cfg in variant_cfgs.items():
        summary = summarize_field_prediction_bundle(variant_cfg, base_bundle)
        support_iou_values = [
            float(pair_metrics["hybrid_ml_constrained"].get("trace_iou_with_2010_support", float("nan")))
            for pair_metrics in summary["pairs"].values()
        ]
        trace_fraction_values = [
            float(pair_metrics["hybrid_ml_constrained"].get("predicted_trace_fraction", float("nan")))
            for pair_metrics in summary["pairs"].values()
        ]
        rows.append(
            {
                "variant": name,
                "mean_trace_iou_2010_support": float(np.nanmean(support_iou_values)),
                "mean_predicted_trace_fraction": float(np.nanmean(trace_fraction_values)),
                "mean_outside_reservoir_fraction": float(summary["hybrid_constrained_average"]["outside_reservoir_fraction"]),
                "support_iou_non_decreasing": bool(
                    summary.get("temporal_consistency", {}).get("constrained_support_iou_non_decreasing", False)
                ),
                "trace_fraction_non_decreasing": bool(
                    summary.get("temporal_consistency", {}).get("constrained_trace_fraction_non_decreasing", False)
                ),
            }
        )
    return rows


def _build_direct_panel(config: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image_paths = [
        Path(config["output_root"]) / "results" / "figures" / "field_sleipner_2010_inline_1840_p10_mid_cross_equalized_difference.png",
        Path(config["output_root"]) / "results" / "figures" / "field_sleipner_2010_inline_1840_p10_mid_hybrid_example.png",
        Path(config["output_root"]) / "results" / "figures" / "field_sleipner_2010_inline_1840_p10_mid_hybrid_constrained.png",
    ]
    titles = [
        "A. Cross-equalized baseline",
        "B. Raw hybrid",
        "C. Constrained hybrid",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, image_path, title in zip(axes, image_paths, titles):
        image = mpimg.imread(image_path)
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_methods_results_block(
    claim: str,
    synthetic_metrics: dict[str, Any],
    field_summary: dict[str, Any],
    direct_summary: dict[str, Any],
) -> str:
    classical_method = _synthetic_classical_method(synthetic_metrics)
    direct_pair_name = next(iter(direct_summary["pairs"]))
    direct_pair = direct_summary["pairs"][direct_pair_name]
    p07_temporal = field_summary["temporal_consistency"]
    return (
        f"Claim: {claim}\n\n"
        "Methods: We keep the 2D PyTorch hybrid model as the baseline of record and evaluate it against "
        "plain ML and a cross-equalized classical baseline. Field predictions are benchmark-constrained "
        "using the public Sleipner storage-interval mask and shared-threshold postprocessing.\n\n"
        f"Results: On synthetic OOD data, the hybrid model reaches Dice {synthetic_metrics['ood']['hybrid_ml']['dice']:.3f} "
        f"versus {synthetic_metrics['ood']['plain_ml']['dice']:.3f} for plain ML and "
        f"{synthetic_metrics['ood'][classical_method]['dice']:.3f} for the classical baseline. "
        f"Across the p07 field sequence, constrained support overlap with the 2010 benchmark increases from "
        f"{p07_temporal['constrained_support_iou_by_pair']['sleipner_2001_inline_1840_mid']:.3f} to "
        f"{p07_temporal['constrained_support_iou_by_pair']['sleipner_2006_inline_1840_mid']:.3f}. "
        f"On the direct 2010 p10 audit, the constrained hybrid reaches trace IoU "
        f"{direct_pair['hybrid_ml_constrained']['trace_iou_with_2010_support']:.3f}, compared with "
        f"{direct_pair['cross_equalized_difference']['trace_iou_with_2010_support']:.3f} for the cross-equalized baseline. "
        "The raw field model still leaks strongly outside the reservoir before constraining, so the defensible "
        "claim is about benchmark-constrained support mapping rather than unconstrained field inversion."
    )


def _synthetic_classical_method(metrics: dict[str, Any]) -> str:
    if "cross_equalized_difference" in metrics["ood"]:
        return "cross_equalized_difference"
    return "difference"


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
