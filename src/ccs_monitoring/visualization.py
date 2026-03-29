"""4D-style visualization helpers built on chunked volume artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots

from .runtime import ensure_runtime_environment


def render_4d(config: dict[str, Any]) -> dict[str, Any]:
    ensure_runtime_environment(config["output_root"], config["seed"])
    volume_cfg = config["volume"]
    viz_cfg = config["visualization"]
    output_dir = Path(viz_cfg["output_dir"] or (Path(config["output_root"]) / "results" / "visualization"))
    output_dir.mkdir(parents=True, exist_ok=True)

    store_path = Path(volume_cfg["output_store"] or (Path(config["output_root"]) / "volume.zarr"))
    dataset = xr.open_zarr(store_path)

    animation_variable = str(viz_cfg.get("animation_variable", "constrained_support"))
    volume_variable = str(viz_cfg.get("volume_variable", "constrained_support"))
    uncertainty_variable = str(viz_cfg.get("uncertainty_variable", "uncertainty"))

    browser_path = output_dir / "slice_browser.html"
    volume_path = output_dir / "support_volume.html"
    support_gif_path = output_dir / "support_evolution.gif"
    uncertainty_gif_path = output_dir / "uncertainty_evolution.gif"

    _build_slice_browser(dataset, animation_variable, browser_path)
    _build_volume_render(dataset, volume_variable, volume_path)
    _build_animation_gif(dataset, animation_variable, support_gif_path, overlay_variable="reservoir_mask")
    _build_animation_gif(dataset, uncertainty_variable, uncertainty_gif_path, overlay_variable="constrained_support")

    pyvista_path = None
    if str(viz_cfg.get("mode", "plotly")).lower() == "pyvista":
        pyvista_path = _try_build_pyvista_note(output_dir)

    summary = {
        "volume_store": str(store_path),
        "slice_browser_html": str(browser_path),
        "support_volume_html": str(volume_path),
        "support_animation_gif": str(support_gif_path),
        "uncertainty_animation_gif": str(uncertainty_gif_path),
        "pyvista_note": pyvista_path,
        "vintages": dataset.coords["vintage"].values.tolist(),
    }
    (output_dir / "render_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _build_slice_browser(dataset: xr.Dataset, variable_name: str, destination: Path) -> None:
    volume = np.asarray(dataset[variable_name].values, dtype=np.float32)
    vintages = [str(vintage) for vintage in dataset.coords["vintage"].values.tolist()]
    mid_sample = volume.shape[1] // 2
    mid_trace = volume.shape[2] // 2

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            f"{variable_name}: sample-trace",
            f"{variable_name}: vintage-trace at sample {mid_sample}",
            f"{variable_name}: vintage-sample at trace {mid_trace}",
        ),
    )
    fig.add_trace(go.Heatmap(z=volume[0], colorscale="Viridis", showscale=False), row=1, col=1)
    fig.add_trace(
        go.Heatmap(
            z=volume[:, mid_sample, :],
            x=np.arange(volume.shape[2]),
            y=vintages,
            colorscale="Viridis",
            showscale=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(
            z=volume[:, :, mid_trace],
            x=np.arange(volume.shape[1]),
            y=vintages,
            colorscale="Viridis",
            showscale=True,
            colorbar={"title": variable_name},
        ),
        row=1,
        col=3,
    )
    frames = [
        go.Frame(data=[go.Heatmap(z=volume[index], colorscale="Viridis", showscale=False)], name=vintage)
        for index, vintage in enumerate(vintages)
    ]
    fig.frames = frames
    fig.update_layout(
        title=f"Slice browser for {variable_name}",
        sliders=[
            {
                "active": 0,
                "steps": [
                    {
                        "label": vintage,
                        "method": "animate",
                        "args": [[vintage], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    }
                    for vintage in vintages
                ],
            }
        ],
    )
    fig.write_html(destination, include_plotlyjs=True, full_html=True)


def _build_volume_render(dataset: xr.Dataset, variable_name: str, destination: Path) -> None:
    volume = np.asarray(dataset[variable_name].values, dtype=np.float32)
    vintage_count, sample_count, trace_count = volume.shape
    vintage_index, sample_index, trace_index = np.meshgrid(
        np.arange(vintage_count, dtype=np.float32),
        np.arange(sample_count, dtype=np.float32),
        np.arange(trace_count, dtype=np.float32),
        indexing="ij",
    )

    figure = go.Figure(
        data=go.Isosurface(
            x=trace_index.reshape(-1),
            y=sample_index.reshape(-1),
            z=vintage_index.reshape(-1),
            value=volume.reshape(-1),
            isomin=0.5,
            isomax=1.0,
            opacity=0.18,
            surface_count=2,
            colorscale="Turbo",
        )
    )
    figure.update_layout(
        title=f"3D support render for {variable_name}",
        scene={
            "xaxis_title": "Trace",
            "yaxis_title": "Sample",
            "zaxis_title": "Vintage index",
        },
    )
    figure.write_html(destination, include_plotlyjs=True, full_html=True)


def _build_animation_gif(
    dataset: xr.Dataset,
    variable_name: str,
    destination: Path,
    *,
    overlay_variable: str | None = None,
) -> None:
    volume = np.asarray(dataset[variable_name].values, dtype=np.float32)
    vintages = [str(vintage) for vintage in dataset.coords["vintage"].values.tolist()]
    overlay = None
    if overlay_variable and overlay_variable in dataset:
        overlay_data = np.asarray(dataset[overlay_variable].values, dtype=np.float32)
        if overlay_data.ndim == 2:
            overlay = np.repeat(overlay_data[None, ...], volume.shape[0], axis=0)
        else:
            overlay = overlay_data

    frames: list[np.ndarray] = []
    for index, vintage in enumerate(vintages):
        fig, axis = plt.subplots(figsize=(6, 4))
        image = axis.imshow(volume[index], cmap="viridis", aspect="auto")
        axis.set_title(f"{variable_name}: {vintage}")
        axis.set_xlabel("Trace")
        axis.set_ylabel("Sample")
        if overlay is not None:
            axis.contour(overlay[index] > 0.5, levels=[0.5], colors="white", linewidths=0.8)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(frame)
        plt.close(fig)
    imageio.mimsave(destination, frames, duration=0.9)


def _try_build_pyvista_note(output_dir: Path) -> str:
    try:
        import pyvista  # noqa: F401
    except ImportError:
        note_path = output_dir / "pyvista_unavailable.json"
        note_path.write_text(
            json.dumps(
                {
                    "status": "pyvista_not_installed",
                    "note": "Plotly outputs were generated. Install pyvista to add richer volume rendering later.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return str(note_path)
    note_path = output_dir / "pyvista_unavailable.json"
    note_path.write_text(
        json.dumps(
            {
                "status": "pyvista_available_but_not_used",
                "note": "PyVista is importable, but this bootstrap renderer currently emits Plotly and GIF outputs.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(note_path)
