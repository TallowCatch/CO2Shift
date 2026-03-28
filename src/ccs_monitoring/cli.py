"""Command-line entrypoints."""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .config import load_config
from .pipeline import evaluate, generate, run_all, train, validate_field_setup
from .sleipner import export_sleipner_storage_interval_mask


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reliable 4D CCS monitoring experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("generate", "Generate synthetic benchmark"),
        ("train", "Train ML baselines"),
        ("evaluate", "Evaluate all baselines and models"),
        ("run-all", "Run generation, training, and evaluation"),
        ("validate-field", "Validate a real-data field manifest or field input config"),
        ("build-sleipner-mask", "Build a storage-interval mask from Sleipner benchmark surfaces"),
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
    elif args.command == "run-all":
        result = run_all(config)
    elif args.command == "validate-field":
        result = validate_field_setup(config)
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
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
