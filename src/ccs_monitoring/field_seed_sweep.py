"""Field seed and threshold sweeps for pseudo-3D Sleipner benchmarks."""

from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import load_config
from .pipeline import evaluate_field_only, run_all
from .runtime import ensure_runtime_environment


def run_field_seed_sweep(config: dict[str, Any]) -> dict[str, Any]:
    sweep_cfg = config.get("field_seed_sweep", {})
    seeds = [int(seed) for seed in sweep_cfg.get("seeds", [])]
    if not seeds:
        raise ValueError("field_seed_sweep.seeds must contain at least one seed.")

    quantiles = [float(value) for value in sweep_cfg.get("probability_quantiles", [0.82])]
    if not quantiles:
        raise ValueError("field_seed_sweep.probability_quantiles must contain at least one value.")

    config_paths = _resolve_field_config_paths(config, sweep_cfg)
    if not config_paths:
        raise ValueError(
            "field_seed_sweep requires at least one field config path. "
            "Use field_seed_sweep.field_config_paths or p10_config_path/p07_config_path."
        )

    run_training_per_seed = bool(sweep_cfg.get("run_training_per_seed", False))
    train_config_path = _resolve_config_path(config, str(sweep_cfg.get("train_config_path", "")))
    if run_training_per_seed and train_config_path is None:
        raise ValueError("field_seed_sweep.train_config_path is required when run_training_per_seed is true.")

    method_name = str(sweep_cfg.get("method_name", "wave_temporal_constrained"))
    classical_method_name = str(sweep_cfg.get("classical_method_name", "best_classical_constrained"))
    structured_method_name = str(sweep_cfg.get("structured_method_name", "plain_ml_structured_constrained"))
    continue_on_error = bool(sweep_cfg.get("continue_on_error", True))

    base_output_root = Path(str(sweep_cfg.get("output_root", "")).strip() or f"{config['output_root']}_field_seed_sweep")
    ensure_runtime_environment(base_output_root, config["seed"])
    results_dir = base_output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    trained_roots: dict[int, Path] = {}

    for seed in seeds:
        if run_training_per_seed:
            train_root = _train_per_seed(base_output_root, seed, train_config_path, sweep_cfg)
            trained_roots[seed] = train_root

        for quantile in quantiles:
            for benchmark_name, benchmark_config_path in config_paths.items():
                try:
                    run_cfg = load_config(benchmark_config_path)
                    run_root = (
                        base_output_root
                        / "field_runs"
                        / benchmark_name
                        / f"q{int(round(quantile * 100.0))}"
                        / f"seed_{seed}"
                    )
                    run_cfg["seed"] = seed
                    run_cfg["output_root"] = str(run_root)
                    run_cfg["config_path"] = (
                        f"{benchmark_config_path}::field_seed_sweep::seed={seed}::q={quantile:.4f}"
                    )
                    if run_training_per_seed:
                        run_cfg["artifacts_root"] = str(trained_roots[seed])
                    _apply_common_overrides(run_cfg, quantile, sweep_cfg)
                    field_result = evaluate_field_only(run_cfg)
                    rows.append(
                        _extract_row(
                            field_result,
                            benchmark_name=benchmark_name,
                            seed=seed,
                            quantile=quantile,
                            method_name=method_name,
                            classical_method_name=classical_method_name,
                            structured_method_name=structured_method_name,
                            output_root=run_root,
                        )
                    )
                except Exception as exc:  # pragma: no cover - runtime-dependent failure handling
                    error_row = {
                        "benchmark": benchmark_name,
                        "seed": seed,
                        "quantile": quantile,
                        "error": repr(exc),
                    }
                    errors.append(error_row)
                    rows.append(
                        {
                            "benchmark": benchmark_name,
                            "seed": seed,
                            "quantile": quantile,
                            "status": "error",
                            "error": repr(exc),
                        }
                    )
                    if not continue_on_error:
                        raise
                finally:
                    _cleanup_runtime_memory()

    numeric_keys = _discover_numeric_keys(rows)
    aggregate_rows = _aggregate_rows(rows, numeric_keys, sweep_cfg)
    pareto_rows = _build_pareto_rows(aggregate_rows)
    report_markdown = _build_report(
        rows=rows,
        aggregate_rows=aggregate_rows,
        pareto_rows=pareto_rows,
        method_name=method_name,
        classical_method_name=classical_method_name,
    )

    _write_csv(results_dir / "field_seed_sweep_per_run.csv", rows)
    _write_csv(results_dir / "field_seed_sweep_aggregate.csv", aggregate_rows)
    _write_csv(results_dir / "field_seed_sweep_pareto.csv", pareto_rows)
    (results_dir / "field_seed_sweep_report.md").write_text(report_markdown, encoding="utf-8")

    summary = {
        "output_root": str(base_output_root.resolve()),
        "results_dir": str(results_dir.resolve()),
        "seeds": seeds,
        "probability_quantiles": quantiles,
        "method_name": method_name,
        "classical_method_name": classical_method_name,
        "structured_method_name": structured_method_name,
        "run_training_per_seed": run_training_per_seed,
        "field_config_paths": config_paths,
        "num_runs": len(rows),
        "num_errors": len(errors),
        "errors": errors,
        "per_run_rows": rows,
        "aggregate_rows": aggregate_rows,
        "pareto_rows": pareto_rows,
    }
    (results_dir / "field_seed_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _resolve_field_config_paths(config: dict[str, Any], sweep_cfg: dict[str, Any]) -> dict[str, str]:
    resolved: dict[str, str] = {}

    mapping = sweep_cfg.get("field_config_paths", {})
    if isinstance(mapping, dict):
        for key, value in mapping.items():
            path = _resolve_config_path(config, str(value))
            if path is not None:
                resolved[str(key)] = path

    for benchmark_name in ("p10", "p07"):
        key = f"{benchmark_name}_config_path"
        path = _resolve_config_path(config, str(sweep_cfg.get(key, "")))
        if path is not None:
            resolved[benchmark_name] = path
    return resolved


def _resolve_config_path(config: dict[str, Any], value: str) -> str | None:
    stripped = value.strip()
    if not stripped:
        return None
    path = Path(stripped)
    if path.is_absolute():
        return str(path)

    config_path = Path(str(config.get("config_path", "")))
    candidates: list[Path] = []
    if config_path.name:
        base_dir = config_path.parent
        candidates.append((base_dir / path).resolve())
        candidates.append((base_dir.parent / path).resolve())
    candidates.append((Path.cwd() / path).resolve())

    seen: set[str] = set()
    ordered_candidates: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        ordered_candidates.append(candidate)

    for candidate in ordered_candidates:
        if candidate.exists():
            return str(candidate)
    return str(ordered_candidates[0])


def _train_per_seed(
    output_root: Path,
    seed: int,
    train_config_path: str | None,
    sweep_cfg: dict[str, Any],
) -> Path:
    assert train_config_path is not None
    train_cfg = load_config(train_config_path)
    train_root = output_root / "training_runs" / f"seed_{seed}"
    train_cfg["seed"] = seed
    train_cfg["output_root"] = str(train_root)
    train_cfg["artifacts_root"] = ""
    train_cfg["config_path"] = f"{train_config_path}::field_seed_sweep::train_seed={seed}"
    train_cfg["field"]["enabled"] = False
    train_cfg["evaluation"]["save_figures"] = bool(sweep_cfg.get("save_training_figures", False))
    run_all(train_cfg)
    return train_root


def _apply_common_overrides(config: dict[str, Any], quantile: float, sweep_cfg: dict[str, Any]) -> None:
    field_cfg = config.setdefault("field", {})
    post_cfg = field_cfg.setdefault("postprocess", {})
    post_cfg["enabled"] = True
    post_cfg["threshold_mode"] = "quantile"
    post_cfg["probability_quantile"] = float(quantile)

    method_name = str(sweep_cfg.get("method_name", "wave_temporal_constrained"))
    if method_name.startswith("wave_temporal"):
        config.setdefault("wave_temporal", {})["enabled"] = True

    device_override = str(sweep_cfg.get("device_override", "")).strip()
    if device_override:
        config.setdefault("training", {})["device"] = device_override

    low_memory_cfg = sweep_cfg.get("low_memory", {})
    if isinstance(low_memory_cfg, dict) and low_memory_cfg.get("enabled", False):
        if "device" in low_memory_cfg:
            config.setdefault("training", {})["device"] = str(low_memory_cfg["device"])

        wave_cfg = config.setdefault("wave_temporal", {})
        for key in (
            "mc_dropout_passes",
            "field_adaptation_steps",
            "field_adaptation_batch_size",
            "field_adaptation_learning_rate",
            "field_reconstruction_loss_weight",
            "field_monotone_loss_weight",
            "field_adjacency_loss_weight",
            "field_crossline_loss_weight",
            "field_sparsity_weight",
        ):
            if key in low_memory_cfg:
                wave_cfg[key] = low_memory_cfg[key]


def _extract_row(
    field_result: dict[str, Any],
    *,
    benchmark_name: str,
    seed: int,
    quantile: float,
    method_name: str,
    classical_method_name: str,
    structured_method_name: str,
    output_root: Path,
) -> dict[str, Any]:
    volume_summary = field_result.get("volume_summary", {})
    overall = volume_summary.get("overall", {})
    wave = overall.get(method_name, {}) if isinstance(overall, dict) else {}
    classical = overall.get(classical_method_name, {}) if isinstance(overall, dict) else {}
    structured = overall.get(structured_method_name, {}) if isinstance(overall, dict) else {}

    wave_leave_one_out = field_result.get("wave_leave_one_out_summary", {})
    wave_leave_one_out_overall = wave_leave_one_out.get("overall", {}) if isinstance(wave_leave_one_out, dict) else {}
    sequence_methods = field_result.get("sequence_method_summary", {}).get("methods", {})
    sequence_metrics = sequence_methods.get(method_name, {}) if isinstance(sequence_methods, dict) else {}

    wave_sv_iou = _safe_float(wave.get("support_volume_iou_2010", float("nan")))
    wave_trace_iou = _safe_float(wave.get("trace_iou_with_2010_support", float("nan")))
    wave_outside_support = _safe_float(wave.get("predicted_fraction_outside_support_volume", float("nan")))

    heldout_sv_iou = _safe_float(wave_leave_one_out_overall.get("heldout_support_volume_iou_2010", float("nan")))
    heldout_trace_iou = _safe_float(wave_leave_one_out_overall.get("heldout_trace_support_iou", float("nan")))

    row = {
        "benchmark": benchmark_name,
        "seed": int(seed),
        "quantile": float(quantile),
        "status": "ok",
        "output_root": str(output_root),
        "wave_sv_iou": wave_sv_iou,
        "wave_trace_iou": wave_trace_iou,
        "wave_outside_support": wave_outside_support,
        "wave_crossline": _safe_float(wave.get("crossline_continuity", float("nan"))),
        "wave_inside_support": _safe_float(wave.get("predicted_fraction_inside_support_volume", float("nan"))),
        "classical_sv_iou": _safe_float(classical.get("support_volume_iou_2010", float("nan"))),
        "classical_trace_iou": _safe_float(classical.get("trace_iou_with_2010_support", float("nan"))),
        "structured_sv_iou": _safe_float(structured.get("support_volume_iou_2010", float("nan"))),
        "structured_trace_iou": _safe_float(structured.get("trace_iou_with_2010_support", float("nan"))),
        "wave_minus_classical_sv": wave_sv_iou - _safe_float(classical.get("support_volume_iou_2010", float("nan"))),
        "wave_minus_structured_sv": wave_sv_iou - _safe_float(structured.get("support_volume_iou_2010", float("nan"))),
        "wave_heldout_binary_iou": _safe_float(wave_leave_one_out_overall.get("full_vs_heldout_binary_iou", float("nan"))),
        "wave_heldout_trace_iou": _safe_float(wave_leave_one_out_overall.get("full_vs_heldout_trace_iou", float("nan"))),
        "wave_heldout_sv_iou": heldout_sv_iou,
        "wave_heldout_support_trace_iou": heldout_trace_iou,
        "wave_heldout_fraction_shift": _safe_float(wave_leave_one_out_overall.get("heldout_fraction_shift", float("nan"))),
        "wave_heldout_residual_mae": _safe_float(wave_leave_one_out_overall.get("heldout_residual_fit_mae", float("nan"))),
        "wave_heldout_residual_rmse": _safe_float(
            wave_leave_one_out_overall.get("heldout_residual_fit_rmse", float("nan"))
        ),
        "wave_sequence_monotonicity": _safe_float(sequence_metrics.get("temporal_monotonicity_score", float("nan"))),
        "wave_sequence_growth_adjacency": _safe_float(sequence_metrics.get("growth_adjacency_score", float("nan"))),
        "wave_survey_omission_penalty_sv": wave_sv_iou - heldout_sv_iou,
        "wave_survey_omission_penalty_trace": wave_trace_iou - heldout_trace_iou,
    }
    return row


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _discover_numeric_keys(rows: list[dict[str, Any]]) -> list[str]:
    numeric_keys: list[str] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        for key, value in row.items():
            if key in {"seed", "quantile"}:
                numeric_keys.append(key)
                continue
            if isinstance(value, (int, float, np.floating, np.integer)):
                numeric_keys.append(key)
    return sorted(set(numeric_keys))


def _aggregate_rows(rows: list[dict[str, Any]], numeric_keys: list[str], sweep_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        grouped.setdefault((str(row.get("benchmark", "")), float(row.get("quantile", float("nan")))), []).append(row)

    bootstrap_samples = int(sweep_cfg.get("bootstrap_samples", 2000))
    bootstrap_alpha = float(sweep_cfg.get("bootstrap_alpha", 0.05))
    rng = np.random.default_rng(int(sweep_cfg.get("bootstrap_seed", 20260331)))

    aggregate_rows: list[dict[str, Any]] = []
    for (benchmark, quantile), group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        aggregate_row: dict[str, Any] = {
            "benchmark": benchmark,
            "quantile": quantile,
            "num_runs": len(group_rows),
        }
        for key in numeric_keys:
            values = [float(row[key]) for row in group_rows if key in row and np.isfinite(float(row[key]))]
            if not values:
                continue
            values_array = np.asarray(values, dtype=np.float64)
            aggregate_row[f"{key}_mean"] = float(np.mean(values_array))
            aggregate_row[f"{key}_std"] = float(np.std(values_array, ddof=0))
            aggregate_row[f"{key}_min"] = float(np.min(values_array))
            aggregate_row[f"{key}_max"] = float(np.max(values_array))
            ci_low, ci_high = _bootstrap_mean_ci(values_array, bootstrap_samples, bootstrap_alpha, rng)
            aggregate_row[f"{key}_ci_low"] = ci_low
            aggregate_row[f"{key}_ci_high"] = ci_high
        aggregate_rows.append(aggregate_row)
    return aggregate_rows


def _bootstrap_mean_ci(
    values: np.ndarray,
    bootstrap_samples: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value

    samples = rng.choice(values, size=(bootstrap_samples, values.size), replace=True)
    means = np.mean(samples, axis=1)
    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return low, high


def _build_pareto_rows(aggregate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_quantile: dict[float, dict[str, dict[str, Any]]] = {}
    for row in aggregate_rows:
        per_quantile.setdefault(float(row["quantile"]), {})[str(row["benchmark"])] = row

    candidates: list[dict[str, Any]] = []
    for quantile, benchmark_rows in sorted(per_quantile.items()):
        p10 = benchmark_rows.get("p10")
        p07 = benchmark_rows.get("p07")
        if p10 is None or p07 is None:
            continue
        candidates.append(
            {
                "quantile": quantile,
                "p10_wave_sv_iou_mean": float(p10.get("wave_sv_iou_mean", float("nan"))),
                "p10_wave_trace_iou_mean": float(p10.get("wave_trace_iou_mean", float("nan"))),
                "p10_wave_outside_support_mean": float(p10.get("wave_outside_support_mean", float("nan"))),
                "p07_wave_sv_iou_mean": float(p07.get("wave_sv_iou_mean", float("nan"))),
                "p07_wave_trace_iou_mean": float(p07.get("wave_trace_iou_mean", float("nan"))),
                "p07_wave_outside_support_mean": float(p07.get("wave_outside_support_mean", float("nan"))),
            }
        )

    for candidate in candidates:
        candidate["is_pareto"] = _is_pareto_candidate(candidate, candidates)
        candidate["balanced_score"] = _balanced_score(candidate)

    return sorted(candidates, key=lambda row: float(row["quantile"]))


def _is_pareto_candidate(candidate: dict[str, Any], all_candidates: list[dict[str, Any]]) -> bool:
    for other in all_candidates:
        if other is candidate:
            continue
        better_or_equal = (
            other["p10_wave_sv_iou_mean"] >= candidate["p10_wave_sv_iou_mean"]
            and other["p07_wave_sv_iou_mean"] >= candidate["p07_wave_sv_iou_mean"]
            and other["p10_wave_trace_iou_mean"] >= candidate["p10_wave_trace_iou_mean"]
            and other["p10_wave_outside_support_mean"] <= candidate["p10_wave_outside_support_mean"]
        )
        strictly_better = (
            other["p10_wave_sv_iou_mean"] > candidate["p10_wave_sv_iou_mean"]
            or other["p07_wave_sv_iou_mean"] > candidate["p07_wave_sv_iou_mean"]
            or other["p10_wave_trace_iou_mean"] > candidate["p10_wave_trace_iou_mean"]
            or other["p10_wave_outside_support_mean"] < candidate["p10_wave_outside_support_mean"]
        )
        if better_or_equal and strictly_better:
            return False
    return True


def _balanced_score(candidate: dict[str, Any]) -> float:
    return float(
        candidate["p10_wave_sv_iou_mean"]
        + candidate["p07_wave_sv_iou_mean"]
        + 0.35 * candidate["p10_wave_trace_iou_mean"]
        - 0.5 * candidate["p10_wave_outside_support_mean"]
    )


def _build_report(
    *,
    rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    pareto_rows: list[dict[str, Any]],
    method_name: str,
    classical_method_name: str,
) -> str:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    error_rows = [row for row in rows if row.get("status") != "ok"]
    lines = [
        "# Field seed sweep report",
        "",
        f"- Method under test: `{method_name}`",
        f"- Classical comparator: `{classical_method_name}`",
        f"- Completed runs: `{len(ok_rows)}`",
        f"- Failed runs: `{len(error_rows)}`",
        "",
        "## Aggregate metrics by benchmark and quantile",
        "",
        "| benchmark | quantile | wave_sv_iou_mean | wave_trace_iou_mean | wave_outside_support_mean | wave_minus_classical_sv_mean | wave_heldout_binary_iou_mean | wave_survey_omission_penalty_sv_mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| "
            + f"{row.get('benchmark','')} | {float(row.get('quantile', float('nan'))):.2f} | "
            + f"{_fmt_metric(row.get('wave_sv_iou_mean'))} | "
            + f"{_fmt_metric(row.get('wave_trace_iou_mean'))} | "
            + f"{_fmt_metric(row.get('wave_outside_support_mean'))} | "
            + f"{_fmt_metric(row.get('wave_minus_classical_sv_mean'))} | "
            + f"{_fmt_metric(row.get('wave_heldout_binary_iou_mean'))} | "
            + f"{_fmt_metric(row.get('wave_survey_omission_penalty_sv_mean'))} |"
        )

    lines.extend(
        [
            "",
            "## Pareto view across p10 and p07",
            "",
            "| quantile | p10_sv | p07_sv | p10_trace | p10_outside | pareto | balanced_score |",
            "|---:|---:|---:|---:|---:|:---:|---:|",
        ]
    )
    for row in pareto_rows:
        lines.append(
            "| "
            + f"{float(row.get('quantile', float('nan'))):.2f} | "
            + f"{_fmt_metric(row.get('p10_wave_sv_iou_mean'))} | "
            + f"{_fmt_metric(row.get('p07_wave_sv_iou_mean'))} | "
            + f"{_fmt_metric(row.get('p10_wave_trace_iou_mean'))} | "
            + f"{_fmt_metric(row.get('p10_wave_outside_support_mean'))} | "
            + f"{'yes' if bool(row.get('is_pareto', False)) else 'no'} | "
            + f"{_fmt_metric(row.get('balanced_score'))} |"
        )

    if pareto_rows:
        best_balanced = max(pareto_rows, key=lambda row: float(row.get("balanced_score", float("-inf"))))
        lines.extend(
            [
                "",
                "## Operational recommendation",
                "",
                (
                    f"Based on this run set, `q{int(round(float(best_balanced['quantile']) * 100.0))}` "
                    f"is the strongest balanced operating point."
                ),
            ]
        )

    return "\n".join(lines).strip() + "\n"


def _fmt_metric(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.4f}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _cleanup_runtime_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
