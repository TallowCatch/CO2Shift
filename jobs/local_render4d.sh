#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHONPATH=src python3 -m ccs_monitoring.cli build-volume --config configs/sleipner_volume.yaml
PYTHONPATH=src python3 -m ccs_monitoring.cli render-4d --config configs/sleipner_volume.yaml
