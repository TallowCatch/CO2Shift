#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHONPATH=src python3 -m ccs_monitoring.cli build-paper-evidence --config configs/paper_evidence.yaml
