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
