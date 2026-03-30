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
    support_mask: np.ndarray | None = None
    metadata: dict[str, Any] | None = None


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


def _trace_reservoir_bounds(reservoir_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nt, nx = reservoir_mask.shape
    top = np.full(nx, -1, dtype=np.int32)
    base = np.full(nx, -1, dtype=np.int32)
    for trace_index in range(nx):
        support = np.flatnonzero(reservoir_mask[:, trace_index] > 0.5)
        if support.size == 0:
            continue
        top[trace_index] = int(support[0])
        base[trace_index] = int(support[-1])
    missing = top < 0
    if np.any(missing):
        default_top = nt // 3
        default_base = min(default_top + nt // 8, nt - 1)
        top[missing] = default_top
        base[missing] = default_base
    return top, base


def _choose_scenario(config: dict[str, Any], rng: np.random.Generator) -> str:
    probabilities = config.get("scenario_probabilities", {})
    if not probabilities:
        return "plume_growth"
    names = list(probabilities.keys())
    weights = np.asarray([max(float(probabilities[name]), 0.0) for name in names], dtype=np.float64)
    if np.sum(weights) <= 0.0:
        return "plume_growth"
    weights = weights / np.sum(weights)
    return str(rng.choice(names, p=weights))


def _make_layered_plume_sequence(
    reservoir_mask: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
    *,
    in_zone: bool,
) -> tuple[np.ndarray, np.ndarray]:
    nt, nx = reservoir_mask.shape
    num_vintages = int(config.get("num_monitor_vintages", 3))
    layer_count_range = config.get("plume_layer_count_range", [2, 4])
    thickness_range = config.get("plume_layer_thickness_range", [2, 5])
    growth_curve = np.asarray(config.get("plume_growth_curve", [0.45, 0.75, 1.0]), dtype=np.float32)
    if growth_curve.size != num_vintages:
        growth_curve = np.linspace(0.45, 1.0, num_vintages, dtype=np.float32)

    top, base = _trace_reservoir_bounds(reservoir_mask)
    reservoir_positions = np.argwhere(reservoir_mask > 0.5)
    center_trace = int(rng.integers(0, nx))
    if reservoir_positions.size > 0 and in_zone:
        center_trace = int(reservoir_positions[rng.integers(0, len(reservoir_positions)), 1])
    base_half_width = int(rng.integers(max(6, nx // 20), max(10, nx // 9)))
    max_half_width = int(rng.integers(max(base_half_width + 4, nx // 8), max(base_half_width + 8, nx // 5)))

    layer_count = int(rng.integers(layer_count_range[0], layer_count_range[1] + 1))
    layer_fractions = np.sort(rng.uniform(0.14, 0.86, size=layer_count).astype(np.float32))
    masks = np.zeros((num_vintages, nt, nx), dtype=np.float32)
    layer_support = np.zeros((layer_count, nt, nx), dtype=np.float32)

    for layer_index, fraction in enumerate(layer_fractions):
        thickness = int(rng.integers(thickness_range[0], thickness_range[1] + 1))
        for trace_index in range(nx):
            thickness_trace = max(int(base[trace_index] - top[trace_index]), 3)
            if in_zone:
                center_time = top[trace_index] + int(fraction * thickness_trace)
            else:
                offset = int(rng.integers(max(6, nt // 18), max(10, nt // 10)))
                direction = -1 if rng.random() < 0.65 else 1
                if direction < 0:
                    center_time = max(top[trace_index] - offset, 1)
                else:
                    center_time = min(base[trace_index] + offset, nt - 2)
            start = max(center_time - thickness // 2, 0)
            stop = min(start + thickness, nt)
            layer_support[layer_index, start:stop, trace_index] = 1.0

    for vintage_index, scale in enumerate(growth_curve):
        half_width = int(round(base_half_width + scale * (max_half_width - base_half_width)))
        x_start = max(center_trace - half_width, 0)
        x_stop = min(center_trace + half_width + 1, nx)
        vintage_mask = np.zeros((nt, nx), dtype=bool)
        for trace_index in range(x_start, x_stop):
            lateral_scale = 1.0 - abs(trace_index - center_trace) / max(half_width, 1)
            active_layers = max(1, int(np.ceil(layer_count * max(scale, 0.25) * max(lateral_scale, 0.2))))
            trace_layers = layer_support[:active_layers, :, trace_index] > 0.5
            if trace_layers.size == 0:
                continue
            vintage_mask[:, trace_index] = np.any(trace_layers, axis=0)
        if in_zone:
            vintage_mask &= reservoir_mask > 0.2
        vintage_mask = ndimage.binary_dilation(vintage_mask, iterations=1)
        masks[vintage_index] = vintage_mask.astype(np.float32)

    final_layer_support = (np.sum(layer_support, axis=0) > 0.0).astype(np.float32)
    return masks.astype(np.float32), final_layer_support.astype(np.float32)


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

    static_shift_min, static_shift_max = config.get("static_shift_range", [0, 0])
    static_shift = int(rng.integers(static_shift_min, static_shift_max + 1))
    if static_shift != 0:
        mismatched = np.roll(mismatched, shift=static_shift, axis=0)
    metadata["static_shift"] = float(static_shift)

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

    coherent_min, coherent_max = config.get("coherent_noise_scale_range", [0.0, 0.0])
    coherent_scale = float(rng.uniform(coherent_min, coherent_max))
    if coherent_scale > 0.0:
        time_axis = np.linspace(0.0, 1.0, mismatched.shape[0], dtype=np.float32)[:, None]
        trace_axis = np.linspace(0.0, 1.0, mismatched.shape[1], dtype=np.float32)[None, :]
        coherent = np.sin(2.0 * np.pi * (1.5 * time_axis + rng.uniform(0.5, 2.5) * trace_axis + rng.uniform(0.0, 1.0)))
        coherent += 0.5 * np.cos(2.0 * np.pi * (3.0 * time_axis + rng.uniform(0.0, 1.0)))
        mismatched += coherent_scale * coherent.astype(np.float32)
    metadata["coherent_noise_scale"] = coherent_scale

    overburden_min, overburden_max = config.get("overburden_artifact_scale_range", [0.0, 0.0])
    overburden_scale = float(rng.uniform(overburden_min, overburden_max))
    if overburden_scale > 0.0:
        overburden_window = np.linspace(1.0, 0.0, mismatched.shape[0], dtype=np.float32)[:, None]
        overburden_window = np.clip(overburden_window * 2.0, 0.0, 1.0)
        artifact = rng.normal(0.0, 1.0, size=mismatched.shape).astype(np.float32)
        artifact = ndimage.gaussian_filter(artifact, sigma=(4.0, 1.5))
        mismatched += overburden_scale * overburden_window * artifact
    metadata["overburden_artifact_scale"] = overburden_scale

    return mismatched.astype(np.float32), metadata


def _mismatch_type_from_metadata(metadata: dict[str, float]) -> str:
    components: list[str] = []
    if abs(float(metadata.get("trace_shift", 0.0))) > 0.0:
        components.append("trace_shift")
    if abs(float(metadata.get("static_shift", 0.0))) > 0.0:
        components.append("static_shift")
    if abs(float(metadata.get("amplitude_scale", 1.0)) - 1.0) > 1e-6:
        components.append("amplitude_scale")
    if float(metadata.get("drop_trace_fraction", 0.0)) > 0.0:
        components.append("missing_traces")
    if float(metadata.get("noise_std", 0.0)) > 0.0:
        components.append("random_noise")
    if float(metadata.get("coherent_noise_scale", 0.0)) > 0.0:
        components.append("coherent_noise")
    if float(metadata.get("overburden_artifact_scale", 0.0)) > 0.0:
        components.append("overburden_artifact")
    return "+".join(components) if components else "clean"


def generate_synthetic_sample_v2(
    shape: tuple[int, int],
    family_id: int,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    baseline_impedance, reservoir_mask = _build_layered_impedance(shape, family_id, rng)
    scenario_type = _choose_scenario(config, rng)
    in_zone = scenario_type != "out_of_zone"
    mask_sequence, layer_support = _make_layered_plume_sequence(
        reservoir_mask,
        config,
        rng,
        in_zone=in_zone,
    )
    num_vintages = mask_sequence.shape[0]
    plume_strength = float(rng.uniform(*config.get("plume_strength_range", [0.08, 0.24])))

    if scenario_type in {"mismatch_only", "no_change"}:
        mask_sequence[:] = 0.0

    base_freq = float(rng.uniform(*config["wavelet_freq_range"]))
    baseline = _impedance_to_seismic(baseline_impedance, base_freq)
    monitor_sequence: list[np.ndarray] = []
    mismatch_metadatas: list[dict[str, float]] = []

    for vintage_index in range(num_vintages):
        vintage_impedance = baseline_impedance.copy()
        if scenario_type in {"plume_growth", "out_of_zone"}:
            vintage_impedance *= 1.0 - plume_strength * mask_sequence[vintage_index]
            vintage_impedance = ndimage.gaussian_filter(vintage_impedance, sigma=(0.8, 0.6))

        freq_min, freq_max = config["wavelet_freq_range"]
        monitor_freq = base_freq if rng.random() <= config["clean_probability"] else float(rng.uniform(freq_min, freq_max))
        monitor = _impedance_to_seismic(vintage_impedance, monitor_freq)

        mismatch_metadata = {
            "trace_shift": 0.0,
            "static_shift": 0.0,
            "amplitude_scale": 1.0,
            "drop_trace_fraction": 0.0,
            "noise_std": 0.0,
            "coherent_noise_scale": 0.0,
            "overburden_artifact_scale": 0.0,
        }
        apply_mismatch = scenario_type != "no_change" and (
            scenario_type == "mismatch_only" or rng.random() < config["mismatch_probability"]
        )
        if apply_mismatch:
            monitor, mismatch_metadata = _apply_mismatch(monitor, config, rng)
        monitor_sequence.append(monitor.astype(np.float32))
        mismatch_metadatas.append(mismatch_metadata)

    containment_label = int(scenario_type == "plume_growth")
    metadata = {
        "family_id": family_id,
        "benchmark_version": "v2",
        "scenario_type": scenario_type,
        "containment_label": containment_label,
        "base_freq": base_freq,
        "plume_strength": plume_strength,
        "num_monitor_vintages": num_vintages,
        "mismatch_type": _mismatch_type_from_metadata(mismatch_metadatas[-1]),
        "monitor_sequence_mismatch": mismatch_metadatas,
    }

    return {
        "baseline": baseline.astype(np.float32),
        "monitor": monitor_sequence[-1].astype(np.float32),
        "monitor_sequence": np.stack(monitor_sequence, axis=0).astype(np.float32),
        "change_mask": mask_sequence[-1].astype(np.float32),
        "change_mask_sequence": mask_sequence.astype(np.float32),
        "layer_support": layer_support.astype(np.float32),
        "reservoir_mask": reservoir_mask.astype(np.float32),
        "family_id": family_id,
        "scenario_type": scenario_type,
        "mismatch_type": metadata["mismatch_type"],
        "containment_label": containment_label,
        "metadata": metadata,
    }


def generate_synthetic_sample(
    shape: tuple[int, int],
    family_id: int,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    if str(config.get("benchmark_version", "v1")).lower() == "v2":
        return generate_synthetic_sample_v2(shape, family_id, config, rng)

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
    monitor_sequences: list[np.ndarray] = []
    change_mask_sequences: list[np.ndarray] = []
    layer_supports: list[np.ndarray] = []
    scenario_types: list[str] = []
    mismatch_types: list[str] = []
    containment_labels: list[int] = []

    for _ in range(count):
        family_id = int(rng.choice(family_choices))
        sample = generate_synthetic_sample(shape, family_id, config, rng)
        baselines.append(sample["baseline"])
        monitors.append(sample["monitor"])
        change_masks.append(sample["change_mask"])
        reservoir_masks.append(sample["reservoir_mask"])
        family_ids.append(sample["family_id"])
        metadata_json.append(json.dumps(sample["metadata"], sort_keys=True))
        if "monitor_sequence" in sample:
            monitor_sequences.append(sample["monitor_sequence"])
        if "change_mask_sequence" in sample:
            change_mask_sequences.append(sample["change_mask_sequence"])
        if "layer_support" in sample:
            layer_supports.append(sample["layer_support"])
        scenario_types.append(str(sample.get("scenario_type", "plume_growth")))
        mismatch_types.append(str(sample.get("mismatch_type", "clean")))
        containment_labels.append(int(sample.get("containment_label", 1)))

    payload: dict[str, Any] = {
        "baseline": np.stack(baselines),
        "monitor": np.stack(monitors),
        "change_mask": np.stack(change_masks),
        "reservoir_mask": np.stack(reservoir_masks),
        "family_id": np.array(family_ids, dtype=np.int16),
        "metadata_json": np.array(metadata_json),
        "scenario_type": np.array(scenario_types),
        "mismatch_type": np.array(mismatch_types),
        "containment_label": np.array(containment_labels, dtype=np.int8),
    }
    if monitor_sequences:
        payload["monitor_sequence"] = np.stack(monitor_sequences)
    if change_mask_sequences:
        payload["change_mask_sequence"] = np.stack(change_mask_sequences)
    if layer_supports:
        payload["layer_support"] = np.stack(layer_supports)

    split_path = output_dir / f"{split_name}.npz"
    np.savez_compressed(split_path, **payload)
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
        "benchmark_version": str(synthetic.get("benchmark_version", "v1")),
        "section_shape": synthetic["section_shape"],
        "num_monitor_vintages": int(synthetic.get("num_monitor_vintages", 1)),
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
    has_support_mask = pair.support_mask is not None
    if has_support_mask:
        support_mask = _ensure_2d_array(f"{pair.name}.support_mask", pair.support_mask)
        if support_mask.shape != baseline.shape:
            raise ValueError(
                f"Field pair {pair.name} has a support mask shape {support_mask.shape} "
                f"that does not match baseline shape {baseline.shape}."
            )

    return {
        "name": pair.name,
        "shape": list(baseline.shape),
        "has_reservoir_mask": has_reservoir_mask,
        "has_support_mask": has_support_mask,
        "inline_id": None if pair.metadata is None else pair.metadata.get("inline_id"),
        "vintage": None if pair.metadata is None else pair.metadata.get("vintage"),
        "processing_family": None if pair.metadata is None else pair.metadata.get("processing_family"),
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
        shared_support_mask = manifest.get("support_mask", manifest.get("plume_support_volume"))
        for entry in manifest.get("pairs", []):
            baseline_path = entry.get("baseline", shared_baseline)
            reservoir_mask_path = entry.get("reservoir_mask", shared_reservoir_mask)
            support_mask_path = entry.get("support_mask", entry.get("plume_support_volume", shared_support_mask))
            if not baseline_path:
                raise ValueError("Each manifest entry must define a baseline path or provide a shared baseline.")
            metadata = {
                key: value
                for key, value in {
                    "inline_id": entry.get("inline_id"),
                    "vintage": entry.get("vintage"),
                    "processing_family": entry.get("processing_family"),
                    "source_name": entry.get("source_name"),
                    "support_note": entry.get("support_note", manifest.get("support_note")),
                }.items()
                if value is not None
            }
            pair = FieldPair(
                name=entry.get("name", Path(entry["monitor"]).stem),
                baseline=_load_array(manifest_path.parent / baseline_path),
                monitor=_load_array(manifest_path.parent / entry["monitor"]),
                reservoir_mask=(
                    _load_array(manifest_path.parent / reservoir_mask_path) if reservoir_mask_path else None
                ),
                support_mask=(
                    _load_array(manifest_path.parent / support_mask_path) if support_mask_path else None
                ),
                metadata=metadata or None,
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
            support_mask = arrays["support_mask"].astype(np.float32) if "support_mask" in arrays else None
            name = str(arrays["name"]) if "name" in arrays else path.stem
    elif path.suffix == ".npy":
        data = np.load(path)
        if data.ndim != 3 or data.shape[0] < 2:
            raise ValueError("Expected .npy field data with shape [2, time, trace].")
        baseline = data[0].astype(np.float32)
        monitor = data[1].astype(np.float32)
        reservoir_mask = None
        support_mask = None
        name = path.stem
    else:
        raise ValueError("Only .npz and .npy field inputs are supported in the bootstrap implementation.")

    pairs = [FieldPair(name=name, baseline=baseline, monitor=monitor, reservoir_mask=reservoir_mask, support_mask=support_mask)]
    summarize_field_pairs(pairs)
    return pairs
