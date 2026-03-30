"""4D-style visualization helpers built on chunked volume artifacts."""

from __future__ import annotations

import base64
import json
from io import BytesIO
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs

from .runtime import ensure_runtime_environment


def render_4d(config: dict[str, Any]) -> dict[str, Any]:
    ensure_runtime_environment(config["output_root"], config["seed"])
    volume_cfg = config["volume"]
    viz_cfg = config["visualization"]
    output_dir = Path(viz_cfg["output_dir"] or (Path(config["output_root"]) / "results" / "visualization"))
    output_dir.mkdir(parents=True, exist_ok=True)

    store_path = Path(volume_cfg["output_store"] or (Path(config["output_root"]) / "volume.zarr"))
    dataset = xr.open_zarr(store_path)

    animation_variable = str(viz_cfg.get("animation_variable", "hybrid_pseudo3d_constrained"))
    volume_variable = str(viz_cfg.get("volume_variable", "hybrid_pseudo3d_constrained"))
    uncertainty_variable = str(viz_cfg.get("uncertainty_variable", "uncertainty"))

    browser_path = output_dir / "slice_browser.html"
    support_plotly_path = output_dir / "support_volume.html"
    hybrid_plotly_path = output_dir / "hybrid_volume.html"
    support_gif_path = output_dir / "support_evolution.gif"
    uncertainty_gif_path = output_dir / "uncertainty_evolution.gif"

    _build_slice_browser(dataset, animation_variable, browser_path)
    _build_volume_render(dataset, "support_volume_2010", support_plotly_path, title_prefix="Benchmark support")
    _build_volume_render(dataset, volume_variable, hybrid_plotly_path, title_prefix="Predicted support")
    _build_animation_gif(dataset, animation_variable, support_gif_path, overlay_variable="support_volume_2010")
    _build_animation_gif(dataset, uncertainty_variable, uncertainty_gif_path, overlay_variable=volume_variable)

    pyvista_outputs = _try_build_pyvista_outputs(dataset, volume_variable, output_dir)
    combined_browser_path = _maybe_build_combined_sleipner_browser()

    summary = {
        "volume_store": str(store_path),
        "slice_browser_html": str(browser_path),
        "combined_sleipner_browser_html": str(combined_browser_path) if combined_browser_path is not None else "",
        "support_volume_html": str(support_plotly_path),
        "hybrid_volume_html": str(hybrid_plotly_path),
        "support_animation_gif": str(support_gif_path),
        "uncertainty_animation_gif": str(uncertainty_gif_path),
        "pyvista_outputs": pyvista_outputs,
        "inlines": dataset.coords["inline"].values.tolist() if "inline" in dataset.coords else [],
        "vintages": dataset.coords["vintage"].values.tolist() if "vintage" in dataset.coords else [],
    }
    (output_dir / "render_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _build_slice_browser(dataset: xr.Dataset, variable_name: str, destination: Path) -> None:
    data_array = dataset[variable_name]
    volume = _to_inline_vintage_volume(data_array)
    inlines = [str(inline) for inline in dataset.coords["inline"].values.tolist()]
    vintages = [str(vintage) for vintage in dataset.coords["vintage"].values.tolist()] if "vintage" in dataset.coords else ["0"]
    mid_inline = volume.shape[0] // 2
    latest_vintage = volume.shape[1] - 1
    friendly_name = _friendly_variable_name(variable_name)
    frame_specs = _build_frame_specs(volume, inlines, vintages, mid_inline, latest_vintage)
    initial_spec = frame_specs[0]
    overview = _overview_matrix(volume)
    support_volume = _support_overlay_volume(dataset, volume.shape)
    benchmark_label = "p07 temporal benchmark" if len(vintages) > 1 else "p10 direct benchmark"
    is_support_like = _is_support_like_volume(volume)

    frames: list[dict[str, Any]] = []
    for spec in frame_specs:
        section = _section_for_frame(volume, spec, mid_inline, latest_vintage)
        footprint = _footprint_slice(volume, spec)
        frame_support_section = _overlay_section(support_volume, spec)
        frame_support_footprint = _overlay_footprint(support_volume, spec)
        coverage = float(overview[spec["inline_index"], spec["vintage_index"]])
        frames.append(
            {
                "label": spec["label"],
                "short_label": spec["short_label"],
                "inline": spec["inline"],
                "vintage": spec["vintage"],
                "section_image": _render_section_snapshot(section, frame_support_section, is_support_like),
                "footprint_image": _render_footprint_snapshot(footprint, frame_support_footprint, is_support_like, inlines),
                "overview_image": _render_overview_snapshot(overview, inlines, vintages, spec),
                "section_fill": round(float(np.mean(section > 0.5)) if is_support_like else float(np.mean(section)), 4),
                "footprint_fill": round(float(np.mean(footprint > 0.5)) if is_support_like else float(np.mean(footprint)), 4),
                "coverage": round(coverage, 4),
            }
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sleipner Review Browser</title>
  <style>
    body {{
      margin: 0;
      font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif;
      background: #f6f7fb;
      color: #22395f;
    }}
    .shell {{
      max-width: 1680px;
      margin: 0 auto;
      padding: 22px 22px 24px;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 18px;
      margin-bottom: 12px;
    }}
    .header-copy h1 {{
      margin: 0 0 6px;
      font-size: 34px;
      letter-spacing: -0.02em;
      line-height: 1.05;
    }}
    .header-copy p {{
      margin: 0;
      font-size: 14px;
      color: #5c6784;
      line-height: 1.5;
    }}
    .controls {{
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 12px;
      align-items: center;
      margin: 12px 0 14px;
    }}
    .nav-buttons {{
      display: flex;
      gap: 8px;
    }}
    .nav-button {{
      border: 0;
      border-radius: 999px;
      padding: 10px 14px;
      background: #dfe8f6;
      color: #27446e;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
    }}
    .nav-button:disabled {{
      opacity: 0.4;
      cursor: default;
    }}
    .frame-strip {{
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding-bottom: 4px;
    }}
    .frame-pill {{
      border: 1px solid rgba(64, 93, 140, 0.14);
      border-radius: 999px;
      padding: 9px 12px;
      background: #ffffff;
      color: #486281;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
      white-space: nowrap;
    }}
    .frame-pill.active {{
      background: #27446e;
      color: #ffffff;
      border-color: #27446e;
    }}
    .chips {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin: 0 0 14px;
    }}
    .chip {{
      border-radius: 999px;
      padding: 8px 12px;
      background: #edf2fb;
      color: #3c557f;
      font-size: 12px;
      font-weight: 600;
    }}
    .dashboard {{
      display: grid;
      grid-template-columns: minmax(0, 1.9fr) minmax(340px, 0.95fr);
      gap: 18px;
      align-items: start;
    }}
    .card {{
      border-radius: 20px;
      background: #ffffff;
      box-shadow: 0 18px 50px rgba(32, 61, 108, 0.08);
      border: 1px solid rgba(64, 93, 140, 0.10);
      overflow: hidden;
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      padding: 16px 18px 0;
    }}
    .card-header h2 {{
      margin: 0;
      font-size: 18px;
    }}
    .card-header p {{
      margin: 0;
      font-size: 12px;
      color: #647792;
    }}
    .image-wrap {{
      padding: 12px 14px 14px;
    }}
    .image-wrap img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 14px;
      background: #f8faff;
      border: 1px solid rgba(64, 93, 140, 0.08);
    }}
    .side-grid {{
      display: grid;
      gap: 18px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      padding: 14px 18px 18px;
    }}
    .stat {{
      padding: 14px;
      border-radius: 16px;
      background: #f7f9fd;
      border: 1px solid rgba(64, 93, 140, 0.10);
    }}
    .stat-label {{
      font-size: 12px;
      color: #647792;
      margin-bottom: 6px;
    }}
    .stat-value {{
      font-size: 28px;
      font-weight: 800;
      color: #22395f;
      line-height: 1;
    }}
    .stat-sub {{
      margin-top: 6px;
      font-size: 12px;
      color: #647792;
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="header">
      <div class="header-copy">
        <h1>Sleipner Review Browser</h1>
        <p>{benchmark_label} • {friendly_name}</p>
      </div>
      <div class="header-copy" style="max-width: 360px; text-align: right;">
        <p>White outlines show benchmark support where available.</p>
      </div>
    </div>
    <div class="chips">
      <span class="chip">Left: section view</span>
      <span class="chip">Top right: crossline footprint</span>
      <span class="chip">Bottom right: inline × vintage coverage</span>
    </div>
    <div class="controls">
      <div class="nav-buttons">
        <button class="nav-button" id="prev-button">Previous</button>
        <button class="nav-button" id="next-button">Next</button>
      </div>
      <div class="frame-strip" id="frame-strip"></div>
      <div class="header-copy" style="text-align:right;">
        <p id="frame-label"></p>
      </div>
    </div>
    <div class="dashboard">
      <section class="card">
        <div class="card-header">
          <h2>Section view</h2>
          <p>Predicted map with benchmark outline</p>
        </div>
        <div class="image-wrap">
          <img id="section-image" alt="Section view" />
        </div>
      </section>
      <div class="side-grid">
        <section class="card">
          <div class="card-header">
            <h2>Frame summary</h2>
            <p>Quick readout for the selected slice</p>
          </div>
          <div class="stats">
            <div class="stat">
              <div class="stat-label">Inline</div>
              <div class="stat-value" id="stat-inline"></div>
              <div class="stat-sub">Selected section centerline</div>
            </div>
            <div class="stat">
              <div class="stat-label">Vintage</div>
              <div class="stat-value" id="stat-vintage"></div>
              <div class="stat-sub">Monitor frame</div>
            </div>
            <div class="stat">
              <div class="stat-label">Section fill</div>
              <div class="stat-value" id="stat-section-fill"></div>
              <div class="stat-sub">Fraction of active support in the section</div>
            </div>
            <div class="stat">
              <div class="stat-label">Footprint fill</div>
              <div class="stat-value" id="stat-footprint-fill"></div>
              <div class="stat-sub">Crossline footprint occupancy</div>
            </div>
            <div class="stat" style="grid-column: 1 / -1;">
              <div class="stat-label">Coverage</div>
              <div class="stat-value" id="stat-coverage"></div>
              <div class="stat-sub">Mean active trace coverage for the selected inline and vintage</div>
            </div>
          </div>
        </section>
        <section class="card">
          <div class="card-header">
            <h2>Crossline footprint</h2>
            <p>How the selected frame spreads across neighboring inlines</p>
          </div>
          <div class="image-wrap">
            <img id="footprint-image" alt="Crossline footprint" />
          </div>
        </section>
        <section class="card">
          <div class="card-header">
            <h2>Inline × vintage coverage</h2>
            <p>Context matrix with the selected frame highlighted</p>
          </div>
          <div class="image-wrap">
            <img id="overview-image" alt="Inline by vintage coverage" />
          </div>
        </section>
      </div>
    </div>
  </div>
  <script>
    const frames = {json.dumps(frames)};
    const frameStrip = document.getElementById('frame-strip');
    const frameLabel = document.getElementById('frame-label');
    const sectionImage = document.getElementById('section-image');
    const footprintImage = document.getElementById('footprint-image');
    const overviewImage = document.getElementById('overview-image');
    const inlineValue = document.getElementById('stat-inline');
    const vintageValue = document.getElementById('stat-vintage');
    const sectionFillValue = document.getElementById('stat-section-fill');
    const footprintFillValue = document.getElementById('stat-footprint-fill');
    const coverageValue = document.getElementById('stat-coverage');
    const prevButton = document.getElementById('prev-button');
    const nextButton = document.getElementById('next-button');
    let activeIndex = 0;

    function formatFraction(value) {{
      return (value * 100).toFixed(1) + '%';
    }}

    function setActive(index) {{
      activeIndex = index;
      const frame = frames[index];
      frameLabel.textContent = frame.label;
      sectionImage.src = frame.section_image;
      footprintImage.src = frame.footprint_image;
      overviewImage.src = frame.overview_image;
      inlineValue.textContent = frame.inline;
      vintageValue.textContent = frame.vintage;
      sectionFillValue.textContent = formatFraction(frame.section_fill);
      footprintFillValue.textContent = formatFraction(frame.footprint_fill);
      coverageValue.textContent = formatFraction(frame.coverage);
      Array.from(frameStrip.children).forEach((button, buttonIndex) => {{
        button.classList.toggle('active', buttonIndex === index);
      }});
      prevButton.disabled = index === 0;
      nextButton.disabled = index === frames.length - 1;
    }}

    frames.forEach((frame, index) => {{
      const button = document.createElement('button');
      button.className = 'frame-pill';
      button.textContent = frame.short_label;
      button.addEventListener('click', () => setActive(index));
      frameStrip.appendChild(button);
    }});

    prevButton.addEventListener('click', () => {{
      if (activeIndex > 0) setActive(activeIndex - 1);
    }});
    nextButton.addEventListener('click', () => {{
      if (activeIndex < frames.length - 1) setActive(activeIndex + 1);
    }});
    window.addEventListener('keydown', (event) => {{
      if (event.key === 'ArrowLeft' && activeIndex > 0) setActive(activeIndex - 1);
      if (event.key === 'ArrowRight' && activeIndex < frames.length - 1) setActive(activeIndex + 1);
    }});
    setActive(0);
  </script>
</body>
</html>
"""
    destination.write_text(page, encoding="utf-8")


def _friendly_variable_name(variable_name: str) -> str:
    mapping = {
        "plain_ml_structured_constrained": "Structured Plain Support",
        "plain_ml_layered_structured_constrained": "Layered Structured Plain Support",
        "plain_ml_constrained": "Plain Support",
        "best_classical_constrained": "Best Classical Support",
        "hybrid_ml_constrained": "Hybrid Support",
        "hybrid_ml_structured_constrained": "Structured Hybrid Support",
        "support_volume_2010": "Benchmark Support",
        "uncertainty": "Uncertainty",
    }
    return mapping.get(variable_name, variable_name.replace("_", " ").title())


def _build_frame_specs(
    volume: np.ndarray,
    inlines: list[str],
    vintages: list[str],
    mid_inline: int,
    latest_vintage: int,
) -> list[dict[str, Any]]:
    if volume.shape[1] > 1:
        return [
            {
                "label": f"{vintage} • inline {inlines[mid_inline]}",
                "short_label": vintage,
                "inline_index": mid_inline,
                "inline": inlines[mid_inline],
                "vintage_index": vintage_index,
                "vintage": vintage,
            }
            for vintage_index, vintage in enumerate(vintages)
        ]
    return [
        {
            "label": f"inline {inline}",
            "short_label": inline,
            "inline_index": inline_index,
            "inline": inline,
            "vintage_index": latest_vintage,
            "vintage": vintages[latest_vintage],
        }
        for inline_index, inline in enumerate(inlines)
    ]


def _section_for_frame(
    volume: np.ndarray,
    spec: dict[str, Any],
    _mid_inline: int,
    _latest_vintage: int,
) -> np.ndarray:
    return volume[spec["inline_index"], spec["vintage_index"]]


def _footprint_slice(volume: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    if _is_support_like_volume(volume):
        return np.any(volume[:, spec["vintage_index"]] > 0.5, axis=1).astype(np.float32)
    return np.mean(volume[:, spec["vintage_index"]], axis=1).astype(np.float32)


def _overview_matrix(volume: np.ndarray) -> np.ndarray:
    if _is_support_like_volume(volume):
        return np.mean(np.any(volume > 0.5, axis=2).astype(np.float32), axis=2).astype(np.float32)
    return np.mean(volume, axis=(2, 3)).astype(np.float32)


def _is_support_like_volume(volume: np.ndarray) -> bool:
    return float(np.nanmax(volume)) <= 1.01 and float(np.nanmin(volume)) >= -0.01 and float(np.nanmean(volume)) < 0.35


def _plotly_colorscale_for_volume(volume: np.ndarray) -> Any:
    if _is_support_like_volume(volume):
        return [
            [0.0, "#17233f"],
            [0.499, "#17233f"],
            [0.5, "#2b7a78"],
            [1.0, "#f4d35e"],
        ]
    return "Viridis"


def _support_overlay_volume(dataset: xr.Dataset, target_shape: tuple[int, int, int, int]) -> np.ndarray | None:
    if "support_volume_2010" not in dataset:
        return None
    return _to_inline_vintage_volume(dataset["support_volume_2010"], target_shape=target_shape)


def _overlay_section(support_volume: np.ndarray | None, spec: dict[str, Any]) -> np.ndarray | None:
    if support_volume is None:
        return None
    return (support_volume[spec["inline_index"], spec["vintage_index"]] > 0.5).astype(np.float32)


def _overlay_footprint(support_volume: np.ndarray | None, spec: dict[str, Any]) -> np.ndarray | None:
    if support_volume is None:
        return None
    return np.any(support_volume[:, spec["vintage_index"]] > 0.5, axis=1).astype(np.float32)


def _render_section_snapshot(
    section: np.ndarray,
    overlay: np.ndarray | None,
    is_support_like: bool,
) -> str:
    figure, axis = plt.subplots(figsize=(9.0, 7.2), dpi=160)
    image = axis.imshow(
        section,
        cmap=_matplotlib_colormap(is_support_like),
        vmin=0.0,
        vmax=1.0 if is_support_like else float(max(np.nanmax(section), 1e-6)),
        aspect="auto",
        interpolation="nearest",
    )
    if overlay is not None and np.any(overlay > 0.5):
        axis.contour(overlay > 0.5, levels=[0.5], colors="white", linewidths=1.9)
    axis.set_xlabel("Trace")
    axis.set_ylabel("Sample")
    axis.set_facecolor("#17233f")
    figure.tight_layout(pad=0.5)
    return _figure_to_data_url(figure)


def _render_footprint_snapshot(
    footprint: np.ndarray,
    overlay: np.ndarray | None,
    is_support_like: bool,
    inlines: list[str],
) -> str:
    figure, axis = plt.subplots(figsize=(5.2, 4.8), dpi=160)
    axis.imshow(
        footprint,
        cmap=_matplotlib_colormap(is_support_like),
        vmin=0.0,
        vmax=1.0 if is_support_like else float(max(np.nanmax(footprint), 1e-6)),
        aspect="auto",
        interpolation="nearest",
    )
    if overlay is not None and np.any(overlay > 0.5):
        axis.contour(overlay > 0.5, levels=[0.5], colors="white", linewidths=1.4)
    axis.set_xlabel("Trace")
    axis.set_ylabel("Inline")
    axis.set_yticks(np.arange(len(inlines)))
    axis.set_yticklabels(inlines)
    axis.set_facecolor("#17233f")
    figure.tight_layout(pad=0.45)
    return _figure_to_data_url(figure)


def _render_overview_snapshot(
    overview: np.ndarray,
    inlines: list[str],
    vintages: list[str],
    spec: dict[str, Any],
) -> str:
    figure, axis = plt.subplots(figsize=(5.2, 3.8), dpi=160)
    axis.imshow(
        overview,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
    )
    axis.scatter(
        [spec["vintage_index"]],
        [spec["inline_index"]],
        s=90,
        c="#ff6b6b",
        marker="D",
        edgecolors="white",
        linewidths=1.5,
        zorder=3,
    )
    axis.set_xlabel("Vintage")
    axis.set_ylabel("Inline")
    axis.set_xticks(np.arange(len(vintages)))
    axis.set_xticklabels(vintages)
    axis.set_yticks(np.arange(len(inlines)))
    axis.set_yticklabels(inlines)
    figure.tight_layout(pad=0.45)
    return _figure_to_data_url(figure)


def _matplotlib_colormap(is_support_like: bool) -> LinearSegmentedColormap | str:
    if is_support_like:
        return LinearSegmentedColormap.from_list(
            "sleipner_support",
            ["#17233f", "#17233f", "#2b7a78", "#f4d35e"],
            N=256,
        )
    return "viridis"


def _figure_to_data_url(figure: plt.Figure) -> str:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _maybe_build_combined_sleipner_browser() -> Path | None:
    runs_root = Path("/Users/ameerfiras/Propagation/runs")
    p07_viewer = runs_root / "sleipner_p07_11inline" / "results" / "visualization" / "slice_browser.html"
    p10_viewer = runs_root / "sleipner_p10_11inline" / "results" / "visualization" / "slice_browser.html"
    if not (p07_viewer.exists() and p10_viewer.exists()):
        return None

    destination = runs_root / "sleipner_review_browser.html"
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sleipner Review Browser</title>
  <style>
    body {{
      margin: 0;
      font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif;
      background: linear-gradient(180deg, #eef3fb 0%, #f7f9fd 100%);
      color: #22395f;
    }}
    .shell {{
      max-width: 1700px;
      margin: 0 auto;
      padding: 28px 28px 36px;
    }}
    .hero {{
      margin-bottom: 22px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 18px;
      align-items: start;
    }}
    .hero-copy h1 {{
      margin: 0 0 6px;
      font-size: 34px;
      letter-spacing: -0.02em;
    }}
    .hero-copy p {{
      margin: 0;
      color: #5b6b86;
      font-size: 15px;
      line-height: 1.55;
    }}
    .hero-note {{
      background: rgba(255, 255, 255, 0.85);
      border-radius: 18px;
      border: 1px solid rgba(64, 93, 140, 0.10);
      box-shadow: 0 16px 40px rgba(32, 61, 108, 0.08);
      padding: 16px 18px;
    }}
    .hero-note h3 {{
      margin: 0 0 8px;
      font-size: 15px;
    }}
    .hero-note ul {{
      margin: 0;
      padding-left: 18px;
      color: #5b6b86;
      font-size: 13px;
      line-height: 1.5;
    }}
    .tabs {{
      display: flex;
      gap: 10px;
      margin: 18px 0;
    }}
    .tab-button {{
      border: 0;
      border-radius: 999px;
      padding: 10px 16px;
      background: #dde5f2;
      color: #27446e;
      cursor: pointer;
      font-size: 14px;
      font-weight: 600;
    }}
    .tab-button.active {{
      background: #27446e;
      color: #ffffff;
    }}
    .panel {{
      display: none;
      background: #ffffff;
      border-radius: 20px;
      box-shadow: 0 18px 50px rgba(32, 61, 108, 0.10);
      overflow: hidden;
      border: 1px solid rgba(64, 93, 140, 0.10);
    }}
    .panel.active {{
      display: block;
    }}
    .panel-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 20px;
      border-bottom: 1px solid rgba(64, 93, 140, 0.10);
      background: #f8fbff;
    }}
    .panel-header h2 {{
      margin: 0;
      font-size: 18px;
    }}
    .panel-header p {{
      margin: 4px 0 0;
      font-size: 13px;
      color: #62748f;
    }}
    .panel-meta {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: center;
      padding: 12px 20px 0;
    }}
    .pill {{
      border-radius: 999px;
      padding: 7px 12px;
      background: #edf2fb;
      color: #3c557f;
      font-size: 12px;
      font-weight: 600;
    }}
    .panel-links a {{
      margin-left: 12px;
      color: #2d61cc;
      text-decoration: none;
      font-size: 13px;
      font-weight: 600;
    }}
    iframe {{
      width: 100%;
      min-height: 900px;
      border: 0;
      background: #fff;
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <div class="hero-copy">
        <h1>Sleipner Review Browser</h1>
        <p>One clean place to inspect the 11-inline pseudo-3D benchmarks. Use the tabs to switch between the temporal p07 sequence and the direct p10 benchmark, then use the embedded viewers to inspect the section slice, crossline footprint, and benchmark support alignment.</p>
      </div>
      <div class="hero-note">
        <h3>How to read it</h3>
        <ul>
          <li>The left panel is the main support map for the selected frame.</li>
          <li>The white outlines show benchmark support where available.</li>
          <li>The right-side views tell you whether the map is coherent across inlines and vintages.</li>
        </ul>
      </div>
    </div>
    <div class="tabs">
      <button class="tab-button active" data-target="p07">p07 Temporal</button>
      <button class="tab-button" data-target="p10">p10 Direct</button>
    </div>
    <section class="panel active" id="p07">
      <div class="panel-header">
        <div>
          <h2>p07 Temporal Benchmark</h2>
          <p>Frames step through vintages while holding the central inline fixed, so you can judge temporal growth and support continuity.</p>
        </div>
        <div class="panel-links">
          <a href="sleipner_p07_11inline/results/visualization/support_evolution.gif">GIF</a>
          <a href="sleipner_p07_11inline/results/visualization/support_volume.html">3D Volume</a>
        </div>
      </div>
      <div class="panel-meta">
        <span class="pill">11 inlines</span>
        <span class="pill">2001 → 2004 → 2006</span>
        <span class="pill">Pseudo-3D temporal review</span>
      </div>
      <iframe class="viewer-frame" src="sleipner_p07_11inline/results/visualization/slice_browser.html"></iframe>
    </section>
    <section class="panel" id="p10">
      <div class="panel-header">
        <div>
          <h2>p10 Direct Benchmark</h2>
          <p>Frames step across inlines for the 2010 direct benchmark, which is the strongest same-year support-alignment check.</p>
        </div>
        <div class="panel-links">
          <a href="sleipner_p10_11inline/results/visualization/support_evolution.gif">GIF</a>
          <a href="sleipner_p10_11inline/results/visualization/support_volume.html">3D Volume</a>
        </div>
      </div>
      <div class="panel-meta">
        <span class="pill">11 inlines</span>
        <span class="pill">2010 direct support audit</span>
        <span class="pill">Pseudo-3D crossline review</span>
      </div>
      <iframe class="viewer-frame" src="sleipner_p10_11inline/results/visualization/slice_browser.html"></iframe>
    </section>
  </div>
  <script>
    const buttons = document.querySelectorAll('.tab-button');
    const panels = document.querySelectorAll('.panel');
    const frames = document.querySelectorAll('.viewer-frame');
    buttons.forEach((button) => {{
      button.addEventListener('click', () => {{
        buttons.forEach((item) => item.classList.remove('active'));
        panels.forEach((panel) => panel.classList.remove('active'));
        button.classList.add('active');
        document.getElementById(button.dataset.target).classList.add('active');
      }});
    }});
    function resizeFrame(frame) {{
      try {{
        const doc = frame.contentWindow.document;
        const height = Math.max(doc.body.scrollHeight, doc.documentElement.scrollHeight);
        if (height > 0) frame.style.height = (height + 16) + 'px';
      }} catch (_error) {{
        frame.style.height = '1200px';
      }}
    }}
    frames.forEach((frame) => {{
      frame.addEventListener('load', () => resizeFrame(frame));
      setTimeout(() => resizeFrame(frame), 600);
    }});
  </script>
</body>
</html>
"""
    destination.write_text(html, encoding="utf-8")
    return destination


def _build_volume_render(dataset: xr.Dataset, variable_name: str, destination: Path, *, title_prefix: str) -> None:
    volume_4d = _to_inline_vintage_volume(dataset[variable_name])
    vintages = [str(vintage) for vintage in dataset.coords["vintage"].values.tolist()] if "vintage" in dataset.coords else ["0"]
    latest_vintage = volume_4d.shape[1] - 1
    volume = volume_4d[:, latest_vintage, :, :]
    inline_count, sample_count, trace_count = volume.shape
    inline_index, sample_index, trace_index = np.meshgrid(
        np.arange(inline_count, dtype=np.float32),
        np.arange(sample_count, dtype=np.float32),
        np.arange(trace_count, dtype=np.float32),
        indexing="ij",
    )
    isomin = 0.5 if np.nanmax(volume) > 0.5 else float(np.nanquantile(volume, 0.9))

    figure = go.Figure(
        data=go.Isosurface(
            x=trace_index.reshape(-1),
            y=sample_index.reshape(-1),
            z=inline_index.reshape(-1),
            value=volume.reshape(-1),
            isomin=isomin,
            isomax=float(np.nanmax(volume)),
            opacity=0.18,
            surface_count=2,
            colorscale="Turbo",
        )
    )
    figure.update_layout(
        title=f"{title_prefix}: {variable_name} at vintage {vintages[latest_vintage]}",
        scene={
            "xaxis_title": "Trace",
            "yaxis_title": "Sample",
            "zaxis_title": "Inline index",
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
    volume = _to_inline_vintage_volume(dataset[variable_name])
    vintages = [str(vintage) for vintage in dataset.coords["vintage"].values.tolist()] if "vintage" in dataset.coords else ["0"]
    inlines = [str(inline) for inline in dataset.coords["inline"].values.tolist()]
    overlay = None
    if overlay_variable and overlay_variable in dataset:
        overlay = _to_inline_vintage_volume(dataset[overlay_variable], target_shape=volume.shape)

    frames: list[np.ndarray] = []
    frame_axis, frame_labels = _animation_axis_labels(volume, inlines, vintages)
    if frame_axis == "vintage":
        inline_count = volume.shape[0]
        fig_width = max(8, 3.4 * inline_count + 1.2)
        for vintage_index, vintage in enumerate(frame_labels):
            fig = plt.figure(figsize=(fig_width, 4.2), constrained_layout=True)
            grid = fig.add_gridspec(
                1,
                inline_count + 1,
                width_ratios=[1.0] * inline_count + [0.07],
                wspace=0.08,
            )
            axes = [fig.add_subplot(grid[0, inline_idx]) for inline_idx in range(inline_count)]
            colorbar_axis = fig.add_subplot(grid[0, inline_count])
            image = None
            for inline_idx, axis in enumerate(axes):
                image = axis.imshow(volume[inline_idx, vintage_index], cmap="viridis", aspect="auto")
                axis.set_title(f"inline {inlines[inline_idx]}")
                axis.set_xlabel("Trace")
                if inline_idx == 0:
                    axis.set_ylabel("Sample")
                if overlay is not None:
                    axis.contour(overlay[inline_idx, vintage_index] > 0.5, levels=[0.5], colors="white", linewidths=0.8)
            fig.suptitle(f"{variable_name}: vintage {vintage}")
            if image is not None:
                colorbar = fig.colorbar(image, cax=colorbar_axis)
                colorbar.ax.set_ylabel(variable_name)
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3])
            plt.close(fig)
    else:
        latest_vintage = volume.shape[1] - 1
        for inline_index, inline_label in enumerate(frame_labels):
            fig = plt.figure(figsize=(7.6, 4.6), constrained_layout=True)
            grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.055], wspace=0.08)
            axis = fig.add_subplot(grid[0, 0])
            colorbar_axis = fig.add_subplot(grid[0, 1])
            image = axis.imshow(volume[inline_index, latest_vintage], cmap="viridis", aspect="auto")
            axis.set_title(f"{variable_name}: inline {inline_label}")
            axis.set_xlabel("Trace")
            axis.set_ylabel("Sample")
            if overlay is not None:
                axis.contour(overlay[inline_index, latest_vintage] > 0.5, levels=[0.5], colors="white", linewidths=0.8)
            colorbar = fig.colorbar(image, cax=colorbar_axis)
            colorbar.ax.set_ylabel(variable_name)
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3])
            plt.close(fig)

    imageio.mimsave(destination, frames, duration=0.9)


def _animation_axis_labels(volume: np.ndarray, inlines: list[str], vintages: list[str]) -> tuple[str, list[str]]:
    if volume.shape[1] > 1:
        return "vintage", vintages
    return "inline", inlines


def _to_inline_vintage_volume(data_array: xr.DataArray, target_shape: tuple[int, int, int, int] | None = None) -> np.ndarray:
    array = np.asarray(data_array.values, dtype=np.float32)
    dims = tuple(str(value) for value in data_array.dims)
    if dims == ("inline", "vintage", "sample", "trace"):
        volume = array
    elif dims == ("inline", "sample", "trace"):
        volume = array[:, None, :, :]
    elif dims == ("sample", "trace"):
        volume = array[None, None, :, :]
    else:
        raise ValueError(f"Unsupported data-array dims for 4D rendering: {dims}")

    if target_shape is None:
        return volume

    expanded = volume
    if expanded.shape[0] == 1 and target_shape[0] > 1:
        expanded = np.repeat(expanded, target_shape[0], axis=0)
    if expanded.shape[1] == 1 and target_shape[1] > 1:
        expanded = np.repeat(expanded, target_shape[1], axis=1)
    return expanded


def _try_build_pyvista_outputs(dataset: xr.Dataset, volume_variable: str, output_dir: Path) -> dict[str, str]:
    if sys.platform == "darwin":
        note_path = output_dir / "pyvista_unavailable.json"
        note_path.write_text(
            json.dumps(
                {
                    "status": "pyvista_skipped_on_macos",
                    "note": "Plotly outputs were generated. Off-screen PyVista/VTK rendering is skipped on this macOS setup because the required OpenGL context is not reliably available.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"note": str(note_path)}

    try:
        import pyvista as pv
    except ImportError:
        note_path = output_dir / "pyvista_unavailable.json"
        note_path.write_text(
            json.dumps(
                {
                    "status": "pyvista_not_installed",
                    "note": "Plotly outputs were generated. Install pyvista/vtk to add richer volume rendering.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"note": str(note_path)}

    support = _to_inline_vintage_volume(dataset["support_volume_2010"])
    hybrid = _to_inline_vintage_volume(dataset[volume_variable])
    classical = _to_inline_vintage_volume(dataset["best_classical_constrained"])
    latest_vintage = hybrid.shape[1] - 1

    outputs = {
        "support": str(output_dir / "pyvista_support_volume.png"),
        "hybrid": str(output_dir / "pyvista_hybrid_volume.png"),
        "comparison": str(output_dir / "pyvista_comparison.png"),
    }
    try:
        pv.start_xvfb()
        _save_pyvista_volume(
            pv,
            support[:, 0],
            output_dir / "pyvista_support_volume.png",
            title="Benchmark support volume",
        )
        _save_pyvista_volume(
            pv,
            hybrid[:, latest_vintage],
            output_dir / "pyvista_hybrid_volume.png",
            title=f"Hybrid pseudo-3D support at {dataset.coords['vintage'].values.tolist()[latest_vintage]}",
        )
        _save_pyvista_comparison(
            pv,
            classical[:, latest_vintage],
            hybrid[:, latest_vintage],
            support[:, 0],
            output_dir / "pyvista_comparison.png",
        )
        return outputs
    except Exception as exc:  # pragma: no cover - environment dependent
        note_path = output_dir / "pyvista_unavailable.json"
        note_path.write_text(
            json.dumps(
                {
                    "status": "pyvista_runtime_unavailable",
                    "note": "PyVista is installed but off-screen OpenGL rendering failed. Plotly and GIF outputs were still generated.",
                    "error": str(exc),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"note": str(note_path)}


def _save_pyvista_volume(pv: Any, volume: np.ndarray, destination: Path, *, title: str) -> None:
    grid = pv.ImageData(dimensions=(volume.shape[2], volume.shape[1], volume.shape[0]))
    grid.point_data["values"] = volume.transpose(2, 1, 0).ravel(order="F")
    plotter = pv.Plotter(off_screen=True, window_size=(1200, 900))
    plotter.add_volume(grid, scalars="values", opacity=[0.0, 0.0, 0.15, 0.35, 0.7], cmap="turbo")
    plotter.add_text(title, font_size=12)
    plotter.camera_position = "xz"
    plotter.show(screenshot=str(destination))
    plotter.close()


def _save_pyvista_comparison(
    pv: Any,
    classical: np.ndarray,
    hybrid: np.ndarray,
    support: np.ndarray,
    destination: Path,
) -> None:
    plotter = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(1800, 700))
    volumes = [
        ("Best classical", classical),
        ("Hybrid pseudo-3D", hybrid),
        ("Benchmark support", support),
    ]
    for subplot_index, (title, volume) in enumerate(volumes):
        plotter.subplot(0, subplot_index)
        grid = pv.ImageData(dimensions=(volume.shape[2], volume.shape[1], volume.shape[0]))
        grid.point_data["values"] = volume.transpose(2, 1, 0).ravel(order="F")
        plotter.add_volume(grid, scalars="values", opacity=[0.0, 0.0, 0.15, 0.35, 0.7], cmap="turbo")
        plotter.add_text(title, font_size=12)
        plotter.camera_position = "xz"
    plotter.show(screenshot=str(destination))
    plotter.close()
