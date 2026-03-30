"""Utilities for aligning Sleipner benchmark surfaces to exported seismic sections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.path import Path as MatplotlibPath
from scipy.interpolate import RegularGridInterpolator


def _load_zmap_grid(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("@") and "HEADER" in line)
    dims = [part.strip() for part in lines[header_idx + 2].split(",")]
    nx = int(dims[0])
    ny = int(dims[1])
    xmin = float(dims[2])
    xmax = float(dims[3])
    ymin = float(dims[4])
    ymax = float(dims[5])
    data_start = next(i for i, line in enumerate(lines) if line.startswith("+ Grid data starts")) + 1

    values: list[float] = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        values.extend(float(token) for token in stripped.split())

    grid = np.array(values[: nx * ny], dtype=np.float32).reshape(ny, nx)
    x_coords = np.linspace(xmin, xmax, nx, dtype=np.float32)
    # RMS/ZMAP grids are written from the northern edge downward.
    y_coords = np.linspace(ymax, ymin, ny, dtype=np.float32)
    return x_coords, y_coords, grid


def _make_interpolator(path: str | Path) -> RegularGridInterpolator:
    x_coords, y_coords, grid = _load_zmap_grid(path)
    return RegularGridInterpolator((y_coords, x_coords), grid, bounds_error=False, fill_value=np.nan)


def _load_inline_geometry(segy_path: str | Path, inline_number: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        import segyio
        from segyio import TraceField
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "segyio is required for Sleipner mask export. Install it or use PYTHONPATH=.vendor:src in this workspace."
        ) from exc

    with segyio.open(str(segy_path), "r", ignore_geometry=False) as segy_file:
        inline_values = segy_file.attributes(TraceField.INLINE_3D)[:]
        crossline_values = segy_file.attributes(TraceField.CROSSLINE_3D)[:]
        cdp_x = segy_file.attributes(TraceField.CDP_X)[:].astype(np.float32)
        cdp_y = segy_file.attributes(TraceField.CDP_Y)[:].astype(np.float32)
        scalar = segy_file.attributes(TraceField.SourceGroupScalar)[:].astype(np.int32)
        sample_axis = np.asarray(segy_file.samples, dtype=np.float32)

    trace_indices = np.where(inline_values == inline_number)[0]
    if trace_indices.size == 0:
        raise ValueError(f"Inline {inline_number} was not found in {segy_path}.")

    order = np.argsort(crossline_values[trace_indices])
    trace_indices = trace_indices[order]
    xlines = crossline_values[trace_indices].astype(np.int32)
    scale = np.where(
        scalar[trace_indices] < 0,
        1.0 / np.abs(scalar[trace_indices]),
        np.where(scalar[trace_indices] > 0, scalar[trace_indices], 1.0),
    ).astype(np.float32)
    x_coords = cdp_x[trace_indices] * scale
    y_coords = cdp_y[trace_indices] * scale
    return xlines, x_coords.astype(np.float32), y_coords.astype(np.float32), sample_axis.astype(np.float32)


def _load_inline_section(
    segy_path: str | Path,
    inline_number: int,
) -> dict[str, np.ndarray]:
    try:
        import segyio
        from segyio import TraceField
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "segyio is required for Sleipner section export. Install it or use PYTHONPATH=.vendor:src in this workspace."
        ) from exc

    with segyio.open(str(segy_path), "r", ignore_geometry=False) as segy_file:
        inline_values = segy_file.attributes(TraceField.INLINE_3D)[:]
        crossline_values = segy_file.attributes(TraceField.CROSSLINE_3D)[:]
        cdp_x = segy_file.attributes(TraceField.CDP_X)[:].astype(np.float32)
        cdp_y = segy_file.attributes(TraceField.CDP_Y)[:].astype(np.float32)
        scalar = segy_file.attributes(TraceField.SourceGroupScalar)[:].astype(np.int32)
        sample_axis = np.asarray(segy_file.samples, dtype=np.float32)

        trace_indices = np.where(inline_values == inline_number)[0]
        if trace_indices.size == 0:
            raise ValueError(f"Inline {inline_number} was not found in {segy_path}.")

        order = np.argsort(crossline_values[trace_indices])
        trace_indices = trace_indices[order]
        section = np.stack([segy_file.trace[int(index)] for index in trace_indices], axis=1).astype(np.float32)

    xlines = crossline_values[trace_indices].astype(np.int32)
    scale = np.where(
        scalar[trace_indices] < 0,
        1.0 / np.abs(scalar[trace_indices]),
        np.where(scalar[trace_indices] > 0, scalar[trace_indices], 1.0),
    ).astype(np.float32)
    x_coords = cdp_x[trace_indices] * scale
    y_coords = cdp_y[trace_indices] * scale
    return {
        "section": section,
        "xlines": xlines.astype(np.int32),
        "trace_x": x_coords.astype(np.float32),
        "trace_y": y_coords.astype(np.float32),
        "sample_axis_ms": sample_axis.astype(np.float32),
    }


def export_sleipner_inline_section(
    segy_path: str | Path,
    inline_number: int,
    output_path: str | Path,
    normalization_reference_paths: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    source = _load_inline_section(segy_path, inline_number)
    reference_paths = [str(Path(path)) for path in (normalization_reference_paths or [])]
    reference_sections: list[np.ndarray] = []
    for reference_path in reference_paths:
        reference = _load_inline_section(reference_path, inline_number)
        if reference["section"].shape != source["section"].shape:
            raise ValueError(
                f"Normalization reference {reference_path} has shape {reference['section'].shape}, "
                f"but source {segy_path} has shape {source['section'].shape}."
            )
        reference_sections.append(reference["section"])

    if reference_sections:
        stacked_reference = np.concatenate([section.reshape(-1) for section in reference_sections], axis=0)
        reference_std = float(np.std(stacked_reference))
        if reference_std <= 0.0:
            raise ValueError("Reference standard deviation must be positive for Sleipner export.")
        normalized = source["section"] / reference_std
    else:
        reference_std = 1.0
        normalized = source["section"].copy()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, normalized.astype(np.float32))

    metadata_path = output_path.with_suffix(".metadata.json")
    metadata = {
        "segy_path": str(segy_path),
        "inline_number": int(inline_number),
        "output_path": str(output_path),
        "metadata_path": str(metadata_path),
        "shape": [int(dimension) for dimension in normalized.shape],
        "normalization_reference_paths": reference_paths,
        "normalization_reference_std": float(reference_std),
        "normalization_scale_factor": float(1.0 / reference_std) if reference_std > 0 else None,
        "raw_mean": float(source["section"].mean()),
        "raw_std": float(source["section"].std()),
        "normalized_mean": float(normalized.mean()),
        "normalized_std": float(normalized.std()),
        "sample_min_ms": float(source["sample_axis_ms"].min()),
        "sample_max_ms": float(source["sample_axis_ms"].max()),
        "xline_min": int(source["xlines"].min()),
        "xline_max": int(source["xlines"].max()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def build_sleipner_storage_interval_mask(
    benchmark_root: str | Path,
    segy_path: str | Path,
    inline_number: int,
) -> dict[str, Any]:
    benchmark_root = Path(benchmark_root)
    required_paths = {
        "topsw_depth": benchmark_root / "DepthSurfaces_Grid" / "TopSW",
        "top_depth": benchmark_root / "DepthSurfaces_Grid" / "TopUtsiraFm",
        "base_depth": benchmark_root / "DepthSurfaces_Grid" / "BaseUtsiraFm",
        "msl_topsw_velocity": benchmark_root / "HUM_Interval_Velocity_Trends" / "1994_MSL_TopSW_Trend",
        "topsw_toputsira_velocity": benchmark_root
        / "HUM_Interval_Velocity_Trends"
        / "1994_TopSW_TopUtsiraFm_Trend",
        "top_base_utsira_velocity": benchmark_root
        / "HUM_Interval_Velocity_Trends"
        / "1994_Top_Base_Utsira_Fm_Trend",
    }
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing benchmark files needed for mask export: {missing}")

    interpolators = {key: _make_interpolator(path) for key, path in required_paths.items()}
    xlines, x_coords, y_coords, sample_axis = _load_inline_geometry(segy_path, inline_number)
    sample_points = np.stack([y_coords, x_coords], axis=1)

    topsw_depth = interpolators["topsw_depth"](sample_points)
    top_depth = interpolators["top_depth"](sample_points)
    base_depth = interpolators["base_depth"](sample_points)
    velocity_1 = interpolators["msl_topsw_velocity"](sample_points)
    velocity_2 = interpolators["topsw_toputsira_velocity"](sample_points)
    velocity_3 = interpolators["top_base_utsira_velocity"](sample_points)

    twt_top = 2000.0 * topsw_depth / velocity_1 + 2000.0 * np.clip(top_depth - topsw_depth, 0.0, None) / velocity_2
    twt_base = twt_top + 2000.0 * np.clip(base_depth - top_depth, 0.0, None) / velocity_3

    twt_top = np.nan_to_num(twt_top, nan=float(sample_axis.min()), posinf=float(sample_axis.max()))
    twt_base = np.nan_to_num(twt_base, nan=float(sample_axis.max()), posinf=float(sample_axis.max()))
    twt_top = np.clip(twt_top, float(sample_axis.min()), float(sample_axis.max()))
    twt_base = np.clip(twt_base, float(sample_axis.min()), float(sample_axis.max()))

    mask = np.zeros((sample_axis.size, xlines.size), dtype=np.float32)
    for column, (top_time, base_time) in enumerate(zip(twt_top, twt_base)):
        top_idx = int(np.searchsorted(sample_axis, top_time, side="left"))
        base_idx = int(np.searchsorted(sample_axis, base_time, side="right"))
        if base_idx <= top_idx:
            base_idx = min(top_idx + 1, sample_axis.size)
        mask[top_idx:base_idx, column] = 1.0

    metadata = {
        "inline_number": int(inline_number),
        "num_traces": int(xlines.size),
        "num_samples": int(sample_axis.size),
        "xline_min": int(xlines.min()),
        "xline_max": int(xlines.max()),
        "sample_min_ms": float(sample_axis.min()),
        "sample_max_ms": float(sample_axis.max()),
        "top_twt_ms_min": float(np.min(twt_top)),
        "top_twt_ms_max": float(np.max(twt_top)),
        "base_twt_ms_min": float(np.min(twt_base)),
        "base_twt_ms_max": float(np.max(twt_base)),
        "mean_thickness_ms": float(np.mean(twt_base - twt_top)),
        "trace_x_min": float(np.min(x_coords)),
        "trace_x_max": float(np.max(x_coords)),
        "trace_y_min": float(np.min(y_coords)),
        "trace_y_max": float(np.max(y_coords)),
        "benchmark_root": str(benchmark_root),
        "segy_path": str(segy_path),
    }

    return {
        "mask": mask,
        "sample_axis_ms": sample_axis,
        "xlines": xlines.astype(np.int32),
        "trace_x": x_coords.astype(np.float32),
        "trace_y": y_coords.astype(np.float32),
        "top_twt_ms": twt_top.astype(np.float32),
        "base_twt_ms": twt_base.astype(np.float32),
        "metadata": metadata,
    }


def export_sleipner_storage_interval_mask(
    benchmark_root: str | Path,
    segy_path: str | Path,
    inline_number: int,
    output_mask_path: str | Path,
) -> dict[str, Any]:
    result = build_sleipner_storage_interval_mask(benchmark_root, segy_path, inline_number)
    output_mask_path = Path(output_mask_path)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_mask_path, result["mask"].astype(np.float32))

    metadata_path = output_mask_path.with_suffix(".metadata.json")
    metadata_payload = {
        **result["metadata"],
        "output_mask_path": str(output_mask_path),
        "metadata_path": str(metadata_path),
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    return metadata_payload


def _load_plume_segments(path: str | Path) -> dict[int, np.ndarray]:
    segments: dict[int, list[tuple[float, float]]] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("!")
            or stripped.startswith("@")
            or stripped.startswith("X ")
            or stripped.startswith("Y ")
            or stripped.startswith("Z ")
            or stripped.startswith("SEG")
        ):
            continue
        parts = stripped.split()
        if len(parts) < 4:
            continue
        x_coord, y_coord, _z_value, segment_id = float(parts[0]), float(parts[1]), float(parts[2]), int(parts[3])
        segments.setdefault(segment_id, []).append((x_coord, y_coord))
    return {segment_id: np.asarray(points, dtype=np.float32) for segment_id, points in segments.items()}


def build_sleipner_plume_support_traces(
    plume_boundaries_root: str | Path,
    segy_path: str | Path,
    inline_number: int,
) -> dict[str, Any]:
    plume_boundaries_root = Path(plume_boundaries_root)
    layer_paths = [plume_boundaries_root / f"L{index}" for index in range(1, 10)]
    missing = [str(path) for path in layer_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing plume-boundary files needed for support export: {missing}")

    xlines, trace_x, trace_y, _sample_axis = _load_inline_geometry(segy_path, inline_number)
    trace_xy = np.stack([trace_x, trace_y], axis=1)

    layer_support: dict[str, np.ndarray] = {}
    union_support = np.zeros(trace_xy.shape[0], dtype=bool)
    for layer_index, layer_path in enumerate(layer_paths, start=1):
        support = np.zeros(trace_xy.shape[0], dtype=bool)
        for polygon in _load_plume_segments(layer_path).values():
            if polygon.shape[0] < 3:
                continue
            if not np.allclose(polygon[0], polygon[-1]):
                polygon = np.vstack([polygon, polygon[0]])
            support |= MatplotlibPath(polygon).contains_points(trace_xy, radius=1e-6)
        layer_name = f"L{layer_index}"
        layer_support[layer_name] = support
        union_support |= support

    metadata = {
        "inline_number": int(inline_number),
        "num_traces": int(trace_xy.shape[0]),
        "support_trace_fraction": float(np.mean(union_support)),
        "support_trace_count": int(np.sum(union_support)),
        "xline_min": int(xlines.min()),
        "xline_max": int(xlines.max()),
        "plume_boundaries_root": str(plume_boundaries_root),
        "segy_path": str(segy_path),
        "layer_trace_counts": {layer: int(np.sum(values)) for layer, values in layer_support.items()},
        "support_vintage_note": "Benchmark plume boundaries are provided for 2010 on CO2DataShare.",
    }

    return {
        "union_support_traces": union_support.astype(np.float32),
        "xlines": xlines.astype(np.int32),
        "trace_x": trace_x.astype(np.float32),
        "trace_y": trace_y.astype(np.float32),
        "layer_support_traces": {layer: values.astype(np.float32) for layer, values in layer_support.items()},
        "metadata": metadata,
    }


def export_sleipner_plume_support_traces(
    plume_boundaries_root: str | Path,
    segy_path: str | Path,
    inline_number: int,
    output_support_path: str | Path,
) -> dict[str, Any]:
    result = build_sleipner_plume_support_traces(plume_boundaries_root, segy_path, inline_number)
    output_support_path = Path(output_support_path)
    output_support_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_support_path, result["union_support_traces"].astype(np.float32))

    metadata_path = output_support_path.with_suffix(".metadata.json")
    metadata_payload = {
        **result["metadata"],
        "output_support_path": str(output_support_path),
        "metadata_path": str(metadata_path),
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    return metadata_payload


def export_sleipner_support_volume_proxy(
    reservoir_mask_path: str | Path,
    plume_support_path: str | Path,
    output_support_volume_path: str | Path,
) -> dict[str, Any]:
    reservoir_mask = np.load(reservoir_mask_path).astype(np.float32)
    plume_support = np.load(plume_support_path).astype(np.float32)
    if reservoir_mask.ndim != 2:
        raise ValueError(
            f"Reservoir mask must be a 2D [sample, trace] array; received shape {reservoir_mask.shape}."
        )

    if plume_support.ndim == 1:
        if plume_support.shape[0] != reservoir_mask.shape[1]:
            raise ValueError(
                "1D plume support length must match the number of traces in the reservoir mask: "
                f"{plume_support.shape[0]} vs {reservoir_mask.shape[1]}."
            )
        support_volume = (reservoir_mask > 0.5) & (plume_support[None, :] > 0.5)
        source_mode = "trace_support_times_reservoir_mask"
    elif plume_support.ndim == 2:
        if plume_support.shape != reservoir_mask.shape:
            raise ValueError(
                "2D plume support must match the reservoir-mask shape exactly: "
                f"{plume_support.shape} vs {reservoir_mask.shape}."
            )
        support_volume = (reservoir_mask > 0.5) & (plume_support > 0.5)
        source_mode = "support_mask_intersected_with_reservoir_mask"
    else:
        raise ValueError(
            f"Plume support must be either 1D trace support or 2D support mask; received shape {plume_support.shape}."
        )

    output_support_volume_path = Path(output_support_volume_path)
    output_support_volume_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_support_volume_path, support_volume.astype(np.float32))

    metadata_path = output_support_volume_path.with_suffix(".metadata.json")
    metadata = {
        "reservoir_mask_path": str(reservoir_mask_path),
        "plume_support_path": str(plume_support_path),
        "output_support_volume_path": str(output_support_volume_path),
        "metadata_path": str(metadata_path),
        "shape": [int(dimension) for dimension in support_volume.shape],
        "support_fraction": float(np.mean(support_volume)),
        "source_mode": source_mode,
        "note": (
            "This support volume is a benchmark-derived structural proxy built by combining the storage-interval "
            "mask with public 2010 plume support. It is not an exact pixel-level plume label."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _relative_to_manifest(manifest_dir: Path, target_path: Path) -> str:
    try:
        return str(target_path.relative_to(manifest_dir))
    except ValueError:
        return str(Path(target_path).resolve().relative_to(Path(manifest_dir).resolve()))


def _resolve_template_path(base_dir: Path, template: str, **values: Any) -> Path:
    rendered = template.format(**values)
    path = Path(rendered)
    if not path.is_absolute():
        path = base_dir / path
    return path


def prepare_sleipner_volume(config: dict[str, Any]) -> dict[str, Any]:
    field_cfg = config.get("field", {})
    inline_numbers = [int(value) for value in field_cfg.get("inline_numbers", [])]
    vintage_map_cfg = field_cfg.get("vintage_map", {})
    if not inline_numbers:
        raise ValueError("field.inline_numbers must list one or more Sleipner inline numbers.")
    if len(vintage_map_cfg) < 2:
        raise ValueError("field.vintage_map must define at least a baseline vintage and one monitor vintage.")

    vintage_map = {int(year): str(path) for year, path in vintage_map_cfg.items()}
    ordered_years = sorted(vintage_map)
    baseline_year = ordered_years[0]
    monitor_years = ordered_years[1:]
    manifest_path_value = str(field_cfg.get("output_manifest_path", "") or field_cfg.get("manifest_path", "")).strip()
    if not manifest_path_value:
        raise ValueError("field.output_manifest_path or field.manifest_path must be set for prepare-sleipner-volume.")

    benchmark_root = str(field_cfg.get("benchmark_root", "")).strip()
    plume_boundaries_root = str(field_cfg.get("plume_boundaries_root", "")).strip()
    if not benchmark_root or not plume_boundaries_root:
        raise ValueError(
            "field.benchmark_root and field.plume_boundaries_root are required for prepare-sleipner-volume."
        )

    processing_family = str(field_cfg.get("processing_family", "")).strip() or "sleipner"
    source_tag = str(field_cfg.get("source_tag", "mid")).strip() or "mid"
    manifest_path = Path(manifest_path_value)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    exports_dir = manifest_path.parent / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    normalization_paths = [vintage_map[year] for year in ordered_years]
    support_template = str(field_cfg.get("output_support_volume_path_template", "")).strip()
    support_note = str(field_cfg.get("plume_support_note", "")).strip()

    pairs: list[dict[str, Any]] = []
    export_records: list[dict[str, Any]] = []
    for inline_number in inline_numbers:
        baseline_section_path = exports_dir / f"sleipner_{baseline_year}_inline_{inline_number}_{processing_family}.npy"
        mask_path = exports_dir / f"sleipner_storage_interval_mask_inline_{inline_number}_{processing_family}.npy"
        plume_support_path = exports_dir / f"sleipner_2010_plume_support_traces_inline_{inline_number}_{processing_family}.npy"
        if support_template:
            support_volume_path = _resolve_template_path(
                manifest_path.parent,
                support_template,
                inline=inline_number,
                processing_family=processing_family,
                source_tag=source_tag,
            )
        else:
            support_volume_path = exports_dir / f"sleipner_2010_support_volume_inline_{inline_number}_{processing_family}.npy"

        export_sleipner_inline_section(
            segy_path=vintage_map[baseline_year],
            inline_number=inline_number,
            output_path=baseline_section_path,
            normalization_reference_paths=normalization_paths,
        )
        export_sleipner_storage_interval_mask(
            benchmark_root=benchmark_root,
            segy_path=vintage_map[baseline_year],
            inline_number=inline_number,
            output_mask_path=mask_path,
        )
        export_sleipner_plume_support_traces(
            plume_boundaries_root=plume_boundaries_root,
            segy_path=vintage_map[baseline_year],
            inline_number=inline_number,
            output_support_path=plume_support_path,
        )
        export_sleipner_support_volume_proxy(
            reservoir_mask_path=mask_path,
            plume_support_path=plume_support_path,
            output_support_volume_path=support_volume_path,
        )

        for monitor_year in monitor_years:
            monitor_section_path = exports_dir / f"sleipner_{monitor_year}_inline_{inline_number}_{processing_family}.npy"
            export_sleipner_inline_section(
                segy_path=vintage_map[monitor_year],
                inline_number=inline_number,
                output_path=monitor_section_path,
                normalization_reference_paths=normalization_paths,
            )
            pair_name = f"sleipner_{monitor_year}_inline_{inline_number}_{processing_family}_{source_tag}"
            pairs.append(
                {
                    "name": pair_name,
                    "baseline": _relative_to_manifest(manifest_path.parent, baseline_section_path),
                    "monitor": _relative_to_manifest(manifest_path.parent, monitor_section_path),
                    "reservoir_mask": _relative_to_manifest(manifest_path.parent, mask_path),
                    "support_mask": _relative_to_manifest(manifest_path.parent, support_volume_path),
                    "inline_id": int(inline_number),
                    "vintage": int(monitor_year),
                    "processing_family": processing_family,
                    "source_name": source_tag,
                    "support_note": support_note or None,
                }
            )
        export_records.append(
            {
                "inline_id": int(inline_number),
                "baseline": str(baseline_section_path),
                "reservoir_mask": str(mask_path),
                "plume_support_traces": str(plume_support_path),
                "support_volume": str(support_volume_path),
            }
        )

    manifest_payload = {
        "support_note": support_note
        or (
            "Benchmark-derived 2010 support volumes are used as structural reference envelopes for the "
            f"{processing_family} multi-inline public benchmark."
        ),
        "processing_family": processing_family,
        "source_name": source_tag,
        "baseline_vintage": int(baseline_year),
        "monitor_vintages": [int(year) for year in monitor_years],
        "inline_numbers": [int(value) for value in inline_numbers],
        "pairs": pairs,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return {
        "manifest_path": str(manifest_path),
        "num_inlines": len(inline_numbers),
        "num_pairs": len(pairs),
        "baseline_vintage": int(baseline_year),
        "monitor_vintages": [int(year) for year in monitor_years],
        "processing_family": processing_family,
        "source_name": source_tag,
        "export_records": export_records,
    }
