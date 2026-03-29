# CCS Monitoring

This repository implements a practical first-paper pipeline for reliable 4D CCS monitoring:

- a synthetic benchmark with geology shift and survey mismatch
- simple classical baselines
- a cross-equalized difference baseline that uses the non-reservoir interval as a repeatability reference
- a plain ML segmentation baseline
- a hybrid ML model with physics-inspired channels and calibrated uncertainty
- synthetic and field-style evaluation utilities

The current implementation is designed to run without downloading the large public datasets during bootstrap. It can:

- generate a Kimberlina-style synthetic benchmark locally
- train and evaluate baselines end-to-end
- optionally run inference on held-out `.npy` or `.npz` field arrays
- produce figures and a Markdown summary for paper iteration
- produce figures and machine-readable summaries for paper iteration

## Quick start

```bash
PYTHONPATH=src python3 -m ccs_monitoring.cli run-all --config configs/smoke.yaml
```

If you want an editable install, use an offline-friendly invocation:

```bash
python3 -m pip install -e . --no-deps --no-build-isolation
```

In this environment the local user site was not writable, so `PYTHONPATH=src` is the most reliable bootstrap path.

If `torch` hits the local OpenMP issue seen in this environment, the package sets `KMP_DUPLICATE_LIB_OK=TRUE` automatically at runtime.

## Data interface

Synthetic samples are stored as `.npz` files with:

- `baseline`: baseline seismic section, shape `[time, trace]`
- `monitor`: monitor seismic section, shape `[time, trace]`
- `change_mask`: binary plume-related change map
- `reservoir_mask`: binary reservoir interval mask
- `family_id`: geology family identifier
- `metadata_json`: JSON metadata string

Optional field input should be provided as `.npz` or `.npy` arrays that contain at least:

- `baseline`
- `monitor`
- optional `reservoir_mask`
- optional `name`

For the current model stack, real field inputs must be exported as 2D sections shaped `[time, trace]`.

For multiple real-data monitor pairs, use a JSON manifest and set:

- `field.enabled: true`
- `field.mode: manifest`
- `field.manifest_path: /abs/path/to/manifest.json`

Manifest format:

```json
{
  "baseline": "sleipner_baseline.npy",
  "reservoir_mask": "sleipner_reservoir_mask.npy",
  "pairs": [
    {"name": "sleipner_2001", "monitor": "sleipner_2001.npy"},
    {"name": "sleipner_2006", "monitor": "sleipner_2006.npy"}
  ]
}
```

A ready-to-edit template lives at [`examples/sleipner_manifest.template.json`](/Users/ameerfiras/Propagation/examples/sleipner_manifest.template.json).

## Main commands

```bash
PYTHONPATH=src python3 -m ccs_monitoring.cli generate --config configs/smoke.yaml
PYTHONPATH=src python3 -m ccs_monitoring.cli train --config configs/smoke.yaml
PYTHONPATH=src python3 -m ccs_monitoring.cli evaluate --config configs/smoke.yaml
PYTHONPATH=src python3 -m ccs_monitoring.cli run-all --config configs/smoke.yaml
PYTHONPATH=src python3 -m ccs_monitoring.cli validate-field --config configs/sleipner_manifest.yaml
PYTHONPATH=.vendor:src python3 -m ccs_monitoring.cli export-sleipner-inline --config configs/sleipner_manifest.yaml
```

For the real Sleipner workflow, `export-sleipner-inline` writes a `.npy` section for the configured `field.inline_number`.
If `field.export_normalization_segy_paths` is set, the export uses one shared reference standard deviation across those vintages so later field comparisons stay on a common amplitude scale.

## Public-data alignment

The repo is structured so we can later plug in:

- NETL Kimberlina-derived synthetic pairs
- Sleipner 4D time-lapse sections or volumes exported to `.npz`
- benchmark-model masks for field-style plausibility checks

The first paper scope intentionally stays narrower than direct `Q` inversion.

## Outputs

After `run-all`, the default output tree contains:

- `runs/<name>/data`: generated synthetic splits
- `runs/<name>/models`: trained `plain` and `hybrid` checkpoints
- `runs/<name>/results/metrics.json`: machine-readable metrics
- `runs/<name>/results/summary.json`: a compact run summary
- `runs/<name>/results/figures`: sample qualitative figures
