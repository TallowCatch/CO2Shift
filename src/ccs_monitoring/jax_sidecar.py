"""JAX sidecar wave-propagation sandbox."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from .runtime import ensure_runtime_environment


def benchmark_jax_wave_lab(config: dict[str, Any]) -> dict[str, Any]:
    ensure_runtime_environment(config["output_root"], config["seed"])
    try:
        import jax
        import jax.numpy as jnp
        from jax import lax
    except ImportError as exc:
        raise ImportError("benchmark-jax requires jax and jaxlib in the runtime environment.") from exc

    jax_cfg = config["jax"]
    output_root = Path(config["output_root"])
    results_dir = output_root / "results"
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    nz, nx = (int(jax_cfg["grid_shape"][0]), int(jax_cfg["grid_shape"][1]))
    num_steps = int(jax_cfg["num_steps"])
    dt = float(jax_cfg["dt"])
    dx = float(jax_cfg["dx"])
    dz = float(jax_cfg["dz"])
    source_frequency = float(jax_cfg["source_frequency"])
    source_index = (int(jax_cfg["source_index"][0]), int(jax_cfg["source_index"][1]))
    batch_size = int(jax_cfg["batch_size"])
    receiver_depth = int(jax_cfg["receiver_depth"])

    dtype_name = str(jax_cfg.get("dtype", "float32")).lower()
    dtype_np = np.float32 if dtype_name == "float32" else np.float64
    dtype_jnp = jnp.float32 if dtype_name == "float32" else jnp.float64

    source_wavelet_np = _ricker_wavelet(num_steps, dt, source_frequency, dtype=dtype_np)
    source_wavelet = jnp.asarray(source_wavelet_np, dtype=dtype_jnp)

    velocity_np = _build_velocity_model((nz, nx), jax_cfg["velocity_range"], dtype=dtype_np)
    velocity = jnp.asarray(velocity_np, dtype=dtype_jnp)
    batch_np = np.stack(
        [
            velocity_np * (1.0 + 0.03 * offset)
            for offset in np.linspace(-1.0, 1.0, batch_size, dtype=dtype_np)
        ],
        axis=0,
    )
    batch_velocity = jnp.asarray(batch_np, dtype=dtype_jnp)

    def simulate_numpy(velocity_model: np.ndarray) -> np.ndarray:
        return _simulate_numpy(
            velocity_model,
            source_wavelet_np,
            source_index=source_index,
            dt=dt,
            dx=dx,
            dz=dz,
        )

    def simulate_jax(velocity_model: Any) -> Any:
        damping = _build_damping_mask_jax((nz, nx), dtype_jnp)
        coeff = (velocity_model**2) * (dt**2)
        initial = (jnp.zeros((nz, nx), dtype=dtype_jnp), jnp.zeros((nz, nx), dtype=dtype_jnp))

        def step(carry: tuple[Any, Any], source_amp: Any) -> tuple[tuple[Any, Any], Any]:
            prev_wave, current_wave = carry
            laplace = _laplacian_jax(current_wave, dx, dz)
            next_wave = 2.0 * current_wave - prev_wave + coeff * laplace
            next_wave = next_wave.at[source_index].add(source_amp)
            next_wave = next_wave * damping
            return (current_wave, next_wave), next_wave

        _, wavefields = lax.scan(step, initial, source_wavelet)
        return wavefields

    simulate_jax_jit = jax.jit(simulate_jax)
    batched_simulate_jax = jax.jit(jax.vmap(simulate_jax, in_axes=0))

    numpy_start = time.perf_counter()
    numpy_wavefields = simulate_numpy(velocity_np)
    numpy_runtime = time.perf_counter() - numpy_start

    eager_start = time.perf_counter()
    jax_wavefields_eager = simulate_jax(velocity)
    jax.block_until_ready(jax_wavefields_eager)
    jax_wavefields_eager_np = np.asarray(jax_wavefields_eager)
    jax_eager_runtime = time.perf_counter() - eager_start

    jit_compile_start = time.perf_counter()
    jax_wavefields_jit = simulate_jax_jit(velocity)
    jax.block_until_ready(jax_wavefields_jit)
    jax_jit_first_runtime = time.perf_counter() - jit_compile_start

    jit_cached_start = time.perf_counter()
    jax_wavefields_jit_cached = simulate_jax_jit(velocity)
    jax.block_until_ready(jax_wavefields_jit_cached)
    jax_jit_cached_runtime = time.perf_counter() - jit_cached_start

    batch_start = time.perf_counter()
    batch_wavefields = batched_simulate_jax(batch_velocity)
    jax.block_until_ready(batch_wavefields)
    jax_vmap_runtime = time.perf_counter() - batch_start

    jax_wavefields_np = np.asarray(jax_wavefields_jit_cached)
    batch_wavefields_np = np.asarray(batch_wavefields)

    loss_fn = lambda velocity_model: jnp.mean(simulate_jax_jit(velocity_model)[-8:, receiver_depth, :] ** 2)
    gradient = np.asarray(jax.grad(loss_fn)(velocity))
    max_abs_error = float(np.max(np.abs(jax_wavefields_np - numpy_wavefields)))

    snapshot_steps = [min(num_steps - 1, int(step)) for step in jax_cfg["snapshot_steps"]]
    snapshot_path = figures_dir / "jax_wavefield_snapshots.png"
    gradient_path = figures_dir / "jax_gradient_sanity.png"
    animation_path = figures_dir / "jax_wavefield_animation.gif"
    _save_wavefield_snapshots(jax_wavefields_np, snapshot_steps, snapshot_path)
    _save_gradient_plot(gradient, gradient_path)
    _save_wavefield_animation(jax_wavefields_np, animation_path)

    runtime_rows = [
        {"method": "numpy_reference", "runtime_seconds": float(numpy_runtime)},
        {"method": "jax_eager", "runtime_seconds": float(jax_eager_runtime)},
        {"method": "jax_jit_first_call", "runtime_seconds": float(jax_jit_first_runtime)},
        {"method": "jax_jit_cached", "runtime_seconds": float(jax_jit_cached_runtime)},
        {"method": "jax_vmap_batch", "runtime_seconds": float(jax_vmap_runtime)},
    ]
    _write_runtime_csv(results_dir / "jax_runtime_table.csv", runtime_rows)

    summary = {
        "device": str(jax_cfg["device"]),
        "grid_shape": [nz, nx],
        "num_steps": num_steps,
        "source_index": list(source_index),
        "batch_size": batch_size,
        "receiver_depth": receiver_depth,
        "max_abs_error_vs_numpy": max_abs_error,
        "numpy_runtime_seconds": float(numpy_runtime),
        "jax_eager_runtime_seconds": float(jax_eager_runtime),
        "jax_jit_first_call_seconds": float(jax_jit_first_runtime),
        "jax_jit_cached_seconds": float(jax_jit_cached_runtime),
        "jax_vmap_batch_seconds": float(jax_vmap_runtime),
        "snapshot_path": str(snapshot_path),
        "gradient_path": str(gradient_path),
        "animation_path": str(animation_path),
        "batch_wavefields_shape": list(batch_wavefields_np.shape),
        "eager_wavefields_shape": list(jax_wavefields_eager_np.shape),
    }
    (results_dir / "jax_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _ricker_wavelet(num_steps: int, dt: float, frequency: float, *, dtype: Any) -> np.ndarray:
    t = np.arange(num_steps, dtype=dtype) * dtype(dt)
    t0 = dtype(1.5 / frequency)
    x = np.pi * frequency * (t - t0)
    wavelet = (1.0 - 2.0 * x**2) * np.exp(-(x**2))
    return wavelet.astype(dtype)


def _build_velocity_model(shape: tuple[int, int], velocity_range: list[float], *, dtype: Any) -> np.ndarray:
    nz, nx = shape
    z_axis = np.linspace(0.0, 1.0, nz, dtype=dtype)[:, None]
    x_axis = np.linspace(-1.0, 1.0, nx, dtype=dtype)[None, :]
    background = velocity_range[0] + (velocity_range[1] - velocity_range[0]) * (0.35 + 0.5 * z_axis)
    anomaly = 0.18 * np.exp(-((z_axis - 0.55) ** 2 / 0.012 + (x_axis * 1.25) ** 2 / 0.08))
    velocity = background * (1.0 - anomaly)
    return velocity.astype(dtype)


def _build_damping_mask_numpy(shape: tuple[int, int], width: int = 10) -> np.ndarray:
    nz, nx = shape
    damping = np.ones((nz, nx), dtype=np.float32)
    for index in range(width):
        scale = np.exp(-((width - index) / width) ** 2 * 0.12)
        damping[index, :] *= scale
        damping[-(index + 1), :] *= scale
        damping[:, index] *= scale
        damping[:, -(index + 1)] *= scale
    return damping.astype(np.float32)


def _build_damping_mask_jax(shape: tuple[int, int], dtype: Any) -> Any:
    import jax.numpy as jnp

    return jnp.asarray(_build_damping_mask_numpy(shape), dtype=dtype)


def _laplacian_numpy(field: np.ndarray, dx: float, dz: float) -> np.ndarray:
    laplace = np.zeros_like(field)
    laplace[1:-1, 1:-1] = (
        (field[2:, 1:-1] - 2.0 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / (dz**2)
        + (field[1:-1, 2:] - 2.0 * field[1:-1, 1:-1] + field[1:-1, :-2]) / (dx**2)
    )
    return laplace


def _laplacian_jax(field: Any, dx: float, dz: float) -> Any:
    import jax.numpy as jnp

    laplace = jnp.zeros_like(field)
    interior = (
        (field[2:, 1:-1] - 2.0 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / (dz**2)
        + (field[1:-1, 2:] - 2.0 * field[1:-1, 1:-1] + field[1:-1, :-2]) / (dx**2)
    )
    return laplace.at[1:-1, 1:-1].set(interior)


def _simulate_numpy(
    velocity: np.ndarray,
    source_wavelet: np.ndarray,
    *,
    source_index: tuple[int, int],
    dt: float,
    dx: float,
    dz: float,
) -> np.ndarray:
    coeff = (velocity**2) * (dt**2)
    prev_wave = np.zeros_like(velocity, dtype=np.float32)
    current_wave = np.zeros_like(velocity, dtype=np.float32)
    damping = _build_damping_mask_numpy(velocity.shape)
    wavefields = np.zeros((len(source_wavelet), *velocity.shape), dtype=np.float32)
    for step_index, source_amp in enumerate(source_wavelet):
        laplace = _laplacian_numpy(current_wave, dx, dz)
        next_wave = 2.0 * current_wave - prev_wave + coeff * laplace
        next_wave[source_index] += source_amp
        next_wave *= damping
        wavefields[step_index] = next_wave
        prev_wave = current_wave
        current_wave = next_wave
    return wavefields


def _save_wavefield_snapshots(wavefields: np.ndarray, steps: list[int], destination: Path) -> None:
    fig, axes = plt.subplots(1, len(steps), figsize=(4 * len(steps), 4))
    axes = np.atleast_1d(axes)
    for axis, step in zip(axes, steps):
        image = axis.imshow(wavefields[step], cmap="seismic", aspect="auto")
        axis.set_title(f"Step {step}")
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_gradient_plot(gradient: np.ndarray, destination: Path) -> None:
    fig, axis = plt.subplots(figsize=(6, 4))
    image = axis.imshow(gradient, cmap="coolwarm", aspect="auto")
    axis.set_title("Gradient sanity check")
    axis.set_xlabel("X")
    axis.set_ylabel("Z")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_wavefield_animation(wavefields: np.ndarray, destination: Path) -> None:
    frame_indices = np.linspace(0, wavefields.shape[0] - 1, num=min(16, wavefields.shape[0]), dtype=int)
    frames: list[np.ndarray] = []
    for index in frame_indices:
        fig, axis = plt.subplots(figsize=(5, 4))
        image = axis.imshow(wavefields[index], cmap="seismic", aspect="auto")
        axis.set_title(f"Wavefield step {index}")
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        plt.close(fig)
    imageio.mimsave(destination, frames, duration=0.14)


def _write_runtime_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "runtime_seconds"])
        writer.writeheader()
        writer.writerows(rows)
