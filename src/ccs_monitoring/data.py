"""Synthetic benchmark generation and field-style loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage, signal


@dataclass(slots=True)
class FieldPair:
    name: str
    baseline: np.ndarray
    monitor: np.ndarray
    reservoir_mask: np.ndarray | None = None


def _ensure_2d_array(name: str, array: np.ndarray) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D section shaped [time, trace] for the current pipeline; got shape {array.shape}."
        )
    if min(array.shape) < 2:
        raise ValueError(f"{name} must have at least 2 samples along each axis; got shape {array.shape}.")
    return array.astype(np.float32)


def ricker_wavelet(length: int, frequency: float) -> np.ndarray:
    t = np.linspace(-(length // 2), length // 2, length, dtype=np.float32)
    x = np.pi * frequency * t
    wavelet = (1.0 - 2.0 * x**2) * np.exp(-(x**2))
    wavelet /= np.max(np.abs(wavelet)) + 1e-8
    return wavelet.astype(np.float32)


def _build_layered_impedance(
    shape: tuple[int, int], family_id: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    nt, nx = shape
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    impedance = np.zeros((nt, nx), dtype=np.float32)

    num_layers = 5 + family_id
    boundaries = sorted(rng.integers(nt // 10, nt - nt // 10, size=num_layers - 1).tolist())
    boundaries = [0] + boundaries + [nt]

    fault_x = int(nx * (0.35 + 0.1 * (family_id % 3)))
    fault_throw = int(2 + family_id)
    fault_throw = min(fault_throw, nt // 10)

    reservoir_index = min(2 + family_id % 2, len(boundaries) - 2)
    reservoir_mask = np.zeros_like(impedance, dtype=np.float32)

    for layer_idx, (top, bottom) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        base_impedance = 1.7 + 0.35 * layer_idx + 0.1 * family_id
        lateral = (
            0.12 * np.sin((layer_idx + 1) * np.pi * x * (1.0 + 0.2 * family_id) + rng.uniform(0, np.pi))
            + 0.06 * np.cos((layer_idx + 2) * np.pi * x + rng.uniform(0, np.pi))
        )
        trend = 0.04 * x * (family_id + 1)
        layer = base_impedance + lateral + trend

        top_shift = np.zeros(nx, dtype=np.int32)
        if layer_idx >= reservoir_index and family_id >= 1:
            top_shift[fault_x:] = fault_throw

        for ix in range(nx):
            shifted_top = min(max(top + int(top_shift[ix]), 0), nt - 1)
            shifted_bottom = min(max(bottom + int(top_shift[ix]), shifted_top + 1), nt)
            impedance[shifted_top:shifted_bottom, ix] = layer[ix]
            if layer_idx == reservoir_index:
                reservoir_mask[shifted_top:shifted_bottom, ix] = 1.0

    impedance = ndimage.gaussian_filter(impedance, sigma=(1.2, 0.8))
    reservoir_mask = ndimage.gaussian_filter(reservoir_mask, sigma=(1.0, 0.0)) > 0.4
    return impedance.astype(np.float32), reservoir_mask.astype(np.float32)


def _make_plume_mask(
    reservoir_mask: np.ndarray,
    radius_range: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    nt, nx = reservoir_mask.shape
    reservoir_positions = np.argwhere(reservoir_mask > 0.5)
    if len(reservoir_positions) == 0:
        fallback_top = nt // 3
        fallback_bottom = min(fallback_top + nt // 8, nt)
        reservoir_mask = reservoir_mask.copy()
        reservoir_mask[fallback_top:fallback_bottom, :] = 1.0
        reservoir_positions = np.argwhere(reservoir_mask > 0.5)
    center_t, center_x = reservoir_positions[rng.integers(0, len(reservoir_positions))]
    radius_t = int(rng.integers(radius_range[0], radius_range[1] + 1))
    radius_x = int(rng.integers(radius_range[0], radius_range[1] + 1))

    t_grid, x_grid = np.meshgrid(np.arange(nt), np.arange(nx), indexing="ij")
    ellipse = (((t_grid - center_t) / max(radius_t, 1)) ** 2 + ((x_grid - center_x) / max(radius_x, 1)) ** 2) <= 1.0
    plume = ellipse & (reservoir_mask > 0.2)
    plume = ndimage.binary_dilation(plume, iterations=2)
    return plume.astype(np.float32)


def _impedance_to_seismic(impedance: np.ndarray, wavelet_freq: float) -> np.ndarray:
    reflectivity = np.diff(np.log(np.clip(impedance, 1e-3, None)), axis=0, prepend=np.log(impedance[:1, :]))
    wavelet = ricker_wavelet(length=17, frequency=wavelet_freq)
    seismic = signal.convolve2d(reflectivity, wavelet[:, None], mode="same")
    seismic /= np.std(seismic) + 1e-8
    return seismic.astype(np.float32)


def _apply_mismatch(
    monitor: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    mismatched = monitor.copy()
    metadata: dict[str, float] = {}

    shift_min, shift_max = config["shift_trace_range"]
    shift = int(rng.integers(shift_min, shift_max + 1))
    if shift != 0:
        mismatched = np.roll(mismatched, shift=shift, axis=1)
    metadata["trace_shift"] = float(shift)

    amplitude_min, amplitude_max = config["amplitude_scale_range"]
    amplitude_scale = float(rng.uniform(amplitude_min, amplitude_max))
    mismatched *= amplitude_scale
    metadata["amplitude_scale"] = amplitude_scale

    drop_min, drop_max = config["drop_trace_fraction_range"]
    drop_fraction = float(rng.uniform(drop_min, drop_max))
    num_drop = int(drop_fraction * mismatched.shape[1])
    if num_drop > 0:
        drop_indices = rng.choice(mismatched.shape[1], size=num_drop, replace=False)
        mismatched[:, drop_indices] = 0.0
    metadata["drop_trace_fraction"] = drop_fraction

    noise_min, noise_max = config["noise_std_range"]
    noise_std = float(rng.uniform(noise_min, noise_max))
    mismatched += rng.normal(0.0, noise_std, size=mismatched.shape).astype(np.float32)
    metadata["noise_std"] = noise_std

    return mismatched.astype(np.float32), metadata


def generate_synthetic_sample(
    shape: tuple[int, int],
    family_id: int,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    baseline_impedance, reservoir_mask = _build_layered_impedance(shape, family_id, rng)
    plume_mask = _make_plume_mask(
        reservoir_mask,
        tuple(config["plume_radius_range"]),
        rng,
    )

    plume_strength = float(rng.uniform(0.08, 0.24))
    monitor_impedance = baseline_impedance * (1.0 - plume_strength * plume_mask)
    monitor_impedance = ndimage.gaussian_filter(monitor_impedance, sigma=(0.8, 0.6))

    base_freq = float(rng.uniform(*config["wavelet_freq_range"]))
    monitor_freq = base_freq
    if rng.random() > config["clean_probability"]:
        freq_min, freq_max = config["wavelet_freq_range"]
        monitor_freq = float(rng.uniform(freq_min, freq_max))

    baseline = _impedance_to_seismic(baseline_impedance, base_freq)
    monitor = _impedance_to_seismic(monitor_impedance, monitor_freq)

    mismatch_metadata: dict[str, float] = {
        "trace_shift": 0.0,
        "amplitude_scale": 1.0,
        "drop_trace_fraction": 0.0,
        "noise_std": 0.0,
    }

    apply_mismatch = rng.random() < config["mismatch_probability"]
    if apply_mismatch:
        monitor, mismatch_metadata = _apply_mismatch(monitor, config, rng)

    change_mask = plume_mask.astype(np.float32)
    metadata = {
        "family_id": family_id,
        "base_freq": base_freq,
        "monitor_freq": monitor_freq,
        "plume_strength": plume_strength,
        "mismatch_applied": apply_mismatch,
        **mismatch_metadata,
    }

    return {
        "baseline": baseline.astype(np.float32),
        "monitor": monitor.astype(np.float32),
        "change_mask": change_mask,
        "reservoir_mask": reservoir_mask.astype(np.float32),
        "family_id": family_id,
        "metadata": metadata,
    }


def _generate_split(
    split_name: str,
    count: int,
    family_choices: list[int],
    config: dict[str, Any],
    seed: int,
    output_dir: Path,
) -> Path:
    rng = np.random.default_rng(seed)
    shape = tuple(config["section_shape"])
    baselines: list[np.ndarray] = []
    monitors: list[np.ndarray] = []
    change_masks: list[np.ndarray] = []
    reservoir_masks: list[np.ndarray] = []
    family_ids: list[int] = []
    metadata_json: list[str] = []

    for _ in range(count):
        family_id = int(rng.choice(family_choices))
        sample = generate_synthetic_sample(shape, family_id, config, rng)
        baselines.append(sample["baseline"])
        monitors.append(sample["monitor"])
        change_masks.append(sample["change_mask"])
        reservoir_masks.append(sample["reservoir_mask"])
        family_ids.append(sample["family_id"])
        metadata_json.append(json.dumps(sample["metadata"], sort_keys=True))

    split_path = output_dir / f"{split_name}.npz"
    np.savez_compressed(
        split_path,
        baseline=np.stack(baselines),
        monitor=np.stack(monitors),
        change_mask=np.stack(change_masks),
        reservoir_mask=np.stack(reservoir_masks),
        family_id=np.array(family_ids, dtype=np.int16),
        metadata_json=np.array(metadata_json),
    )
    return split_path


def generate_synthetic_benchmark(config: dict[str, Any], output_root: str | Path) -> dict[str, str]:
    output_root = Path(output_root)
    dataset_dir = output_root / "data"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    synthetic = config["synthetic"]
    families = synthetic["geology_families"]
    seed = int(config["seed"])

    split_paths = {
        "train": _generate_split("train", synthetic["num_train"], families["train"], synthetic, seed + 11, dataset_dir),
        "val": _generate_split("val", synthetic["num_val"], families["train"], synthetic, seed + 13, dataset_dir),
        "test": _generate_split("test", synthetic["num_test"], families["test"], synthetic, seed + 17, dataset_dir),
        "ood": _generate_split("ood", synthetic["num_ood"], families["ood"], synthetic, seed + 19, dataset_dir),
    }

    manifest = {
        "seed": seed,
        "section_shape": synthetic["section_shape"],
        "splits": {name: str(path) for name, path in split_paths.items()},
    }
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    split_paths["manifest"] = manifest_path
    return {name: str(path) for name, path in split_paths.items()}


def load_split(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def create_pseudo_field_pair(split_arrays: dict[str, np.ndarray], sample_index: int = 0) -> FieldPair:
    sample_index = int(sample_index) % len(split_arrays["baseline"])
    baseline = split_arrays["baseline"][sample_index]
    monitor = split_arrays["monitor"][sample_index]
    reservoir_mask = split_arrays["reservoir_mask"][sample_index]
    field_monitor = ndimage.gaussian_filter(monitor, sigma=(1.2, 0.4))
    field_monitor += 0.02 * np.sin(np.linspace(0, 8 * np.pi, monitor.shape[1], dtype=np.float32))[None, :]
    return FieldPair(
        name="pseudo_sleipner",
        baseline=baseline.astype(np.float32),
        monitor=field_monitor.astype(np.float32),
        reservoir_mask=reservoir_mask.astype(np.float32),
    )


def _load_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        data = np.load(path)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            if len(arrays.files) == 1:
                data = arrays[arrays.files[0]]
            else:
                raise ValueError(f"Ambiguous .npz field array at {path}; expected a single array payload.")
    else:
        raise ValueError(f"Unsupported array format: {path}")
    return _ensure_2d_array(str(path), data)


def validate_field_pair(pair: FieldPair) -> dict[str, Any]:
    baseline = _ensure_2d_array(f"{pair.name}.baseline", pair.baseline)
    monitor = _ensure_2d_array(f"{pair.name}.monitor", pair.monitor)
    if baseline.shape != monitor.shape:
        raise ValueError(
            f"Field pair {pair.name} has mismatched baseline/monitor shapes: {baseline.shape} vs {monitor.shape}."
        )

    has_reservoir_mask = pair.reservoir_mask is not None
    if has_reservoir_mask:
        reservoir_mask = _ensure_2d_array(f"{pair.name}.reservoir_mask", pair.reservoir_mask)
        if reservoir_mask.shape != baseline.shape:
            raise ValueError(
                f"Field pair {pair.name} has a reservoir mask shape {reservoir_mask.shape} "
                f"that does not match baseline shape {baseline.shape}."
            )

    return {
        "name": pair.name,
        "shape": list(baseline.shape),
        "has_reservoir_mask": has_reservoir_mask,
    }


def summarize_field_pairs(pairs: list[FieldPair]) -> dict[str, Any]:
    summaries = [validate_field_pair(pair) for pair in pairs]
    unique_shapes = sorted({tuple(summary["shape"]) for summary in summaries})
    return {
        "num_pairs": len(summaries),
        "unique_shapes": [list(shape) for shape in unique_shapes],
        "pairs": summaries,
    }


def load_field_pairs(config: dict[str, Any], split_arrays: dict[str, np.ndarray] | None = None) -> list[FieldPair]:
    field_cfg = config.get("field", {})
    if not field_cfg.get("enabled", False):
        return []

    if field_cfg.get("mode") == "pseudo_sleipner":
        if split_arrays is None:
            raise ValueError("Pseudo field mode requires a loaded synthetic split.")
        pairs = [create_pseudo_field_pair(split_arrays)]
        summarize_field_pairs(pairs)
        return pairs

    if field_cfg.get("mode") == "manifest":
        manifest_path = Path(field_cfg.get("manifest_path", ""))
        if not manifest_path.exists():
            raise FileNotFoundError(f"Field manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pairs: list[FieldPair] = []
        shared_baseline = manifest.get("baseline")
        shared_reservoir_mask = manifest.get("reservoir_mask")
        for entry in manifest.get("pairs", []):
            baseline_path = entry.get("baseline", shared_baseline)
            reservoir_mask_path = entry.get("reservoir_mask", shared_reservoir_mask)
            if not baseline_path:
                raise ValueError("Each manifest entry must define a baseline path or provide a shared baseline.")
            pair = FieldPair(
                name=entry.get("name", Path(entry["monitor"]).stem),
                baseline=_load_array(manifest_path.parent / baseline_path),
                monitor=_load_array(manifest_path.parent / entry["monitor"]),
                reservoir_mask=(
                    _load_array(manifest_path.parent / reservoir_mask_path) if reservoir_mask_path else None
                ),
            )
            pairs.append(pair)
        summarize_field_pairs(pairs)
        return pairs

    path = Path(field_cfg.get("path", ""))
    if not path.exists():
        raise FileNotFoundError(f"Field path not found: {path}")

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            baseline = arrays["baseline"].astype(np.float32)
            monitor = arrays["monitor"].astype(np.float32)
            reservoir_mask = arrays["reservoir_mask"].astype(np.float32) if "reservoir_mask" in arrays else None
            name = str(arrays["name"]) if "name" in arrays else path.stem
    elif path.suffix == ".npy":
        data = np.load(path)
        if data.ndim != 3 or data.shape[0] < 2:
            raise ValueError("Expected .npy field data with shape [2, time, trace].")
        baseline = data[0].astype(np.float32)
        monitor = data[1].astype(np.float32)
        reservoir_mask = None
        name = path.stem
    else:
        raise ValueError("Only .npz and .npy field inputs are supported in the bootstrap implementation.")

    pairs = [FieldPair(name=name, baseline=baseline, monitor=monitor, reservoir_mask=reservoir_mask)]
    summarize_field_pairs(pairs)
    return pairs
