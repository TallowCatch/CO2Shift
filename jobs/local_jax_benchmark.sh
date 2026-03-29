#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHONPATH=src python3 -m ccs_monitoring.cli benchmark-jax --config configs/jax_wave_lab.yaml
