"""Command-line entrypoints."""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .config import load_config
from .field_seed_sweep import run_field_seed_sweep
from .jax_sidecar import benchmark_jax_wave_lab
from .paper import build_paper_evidence
from .pipeline import evaluate, evaluate_field_only, generate, run_all, train, validate_field_setup
from .seed_sweep import run_seed_sweep
from .sleipner import (
    export_sleipner_inline_section,
    export_sleipner_plume_support_traces,
    export_sleipner_support_volume_proxy,
    export_sleipner_storage_interval_mask,
    prepare_sleipner_volume,
)
from .visualization import render_4d
from .volume import build_volume


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reliable 4D CCS monitoring experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("generate", "Generate synthetic benchmark"),
        ("train", "Train ML baselines"),
        ("evaluate", "Evaluate all baselines and models"),
        ("evaluate-field", "Evaluate only the field benchmark path"),
        ("run-all", "Run generation, training, and evaluation"),
        ("validate-field", "Validate a real-data field manifest or field input config"),
        ("export-sleipner-inline", "Export a normalized Sleipner inline section to .npy"),
        ("prepare-sleipner-volume", "Prepare a multi-inline Sleipner manifest and aligned benchmark exports"),
        ("build-sleipner-mask", "Build a storage-interval mask from Sleipner benchmark surfaces"),
        ("build-sleipner-plume-support", "Build a 2010 plume-support trace mask from Sleipner benchmark polygons"),
        ("build-sleipner-support-volume", "Build a benchmark-derived support-volume proxy from mask and plume support"),
        ("build-paper-evidence", "Build a paper-facing evidence pack from saved runs"),
        ("seed-sweep", "Run a multi-seed synthetic stability sweep"),
        ("field-seed-sweep", "Run a multi-seed/multi-quantile field sweep for p10/p07 benchmarks"),
        ("benchmark-jax", "Run the JAX sidecar wave-propagation sandbox"),
        ("build-volume", "Build a chunked volume store from field predictions"),
        ("render-4d", "Render HTML and GIF 4D-style outputs from a chunked volume store"),
    ):
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "generate":
        result = generate(config)
    elif args.command == "train":
        result = train(config)
    elif args.command == "evaluate":
        result = evaluate(config)
    elif args.command == "evaluate-field":
        result = evaluate_field_only(config)
    elif args.command == "run-all":
        result = run_all(config)
    elif args.command == "validate-field":
        result = validate_field_setup(config)
    elif args.command == "export-sleipner-inline":
        field_cfg = config.get("field", {})
        export_segy_path = field_cfg.get("export_segy_path", "") or field_cfg.get("segy_path", "")
        inline_number = int(field_cfg.get("inline_number", 0))
        output_path = field_cfg.get("export_output_path", "")
        normalization_paths = field_cfg.get("export_normalization_segy_paths", [])
        missing = [
            name
            for name, value in (
                ("field.export_segy_path", export_segy_path),
                ("field.inline_number", inline_number),
                ("field.export_output_path", output_path),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing config values for export-sleipner-inline: {missing}")
        result = export_sleipner_inline_section(
            segy_path=export_segy_path,
            inline_number=inline_number,
            output_path=output_path,
            normalization_reference_paths=normalization_paths,
        )
    elif args.command == "prepare-sleipner-volume":
        result = prepare_sleipner_volume(config)
    elif args.command == "build-sleipner-mask":
        field_cfg = config.get("field", {})
        benchmark_root = field_cfg.get("benchmark_root", "")
        segy_path = field_cfg.get("segy_path", "")
        inline_number = int(field_cfg.get("inline_number", 0))
        output_mask_path = field_cfg.get("output_mask_path", "")
        missing = [
            name
            for name, value in (
                ("field.benchmark_root", benchmark_root),
                ("field.segy_path", segy_path),
                ("field.inline_number", inline_number),
                ("field.output_mask_path", output_mask_path),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing config values for build-sleipner-mask: {missing}")
        result = export_sleipner_storage_interval_mask(
            benchmark_root=benchmark_root,
            segy_path=segy_path,
            inline_number=inline_number,
            output_mask_path=output_mask_path,
        )
    elif args.command == "build-sleipner-plume-support":
        field_cfg = config.get("field", {})
        plume_boundaries_root = field_cfg.get("plume_boundaries_root", "")
        segy_path = field_cfg.get("segy_path", "")
        inline_number = int(field_cfg.get("inline_number", 0))
        output_support_path = field_cfg.get("output_plume_support_path", "")
        missing = [
            name
            for name, value in (
                ("field.plume_boundaries_root", plume_boundaries_root),
                ("field.segy_path", segy_path),
                ("field.inline_number", inline_number),
                ("field.output_plume_support_path", output_support_path),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing config values for build-sleipner-plume-support: {missing}")
        result = export_sleipner_plume_support_traces(
            plume_boundaries_root=plume_boundaries_root,
            segy_path=segy_path,
            inline_number=inline_number,
            output_support_path=output_support_path,
        )
    elif args.command == "build-sleipner-support-volume":
        field_cfg = config.get("field", {})
        reservoir_mask_path = field_cfg.get("output_mask_path", "")
        plume_support_path = field_cfg.get("plume_support_path", "")
        output_support_volume_path = field_cfg.get("output_support_volume_path", "")
        missing = [
            name
            for name, value in (
                ("field.output_mask_path", reservoir_mask_path),
                ("field.plume_support_path", plume_support_path),
                ("field.output_support_volume_path", output_support_volume_path),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing config values for build-sleipner-support-volume: {missing}")
        result = export_sleipner_support_volume_proxy(
            reservoir_mask_path=reservoir_mask_path,
            plume_support_path=plume_support_path,
            output_support_volume_path=output_support_volume_path,
        )
    elif args.command == "build-paper-evidence":
        result = build_paper_evidence(config)
    elif args.command == "seed-sweep":
        result = run_seed_sweep(config)
    elif args.command == "field-seed-sweep":
        result = run_field_seed_sweep(config)
    elif args.command == "benchmark-jax":
        result = benchmark_jax_wave_lab(config)
    elif args.command == "build-volume":
        result = build_volume(config)
    elif args.command == "render-4d":
        result = render_4d(config)
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
