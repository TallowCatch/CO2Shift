# CCS Monitoring

This repository implements a practical first-paper pipeline for reliable 4D CCS monitoring:

- a synthetic benchmark with geology shift and survey mismatch
- simple classical baselines
- a cross-equalized difference baseline that uses the non-reservoir interval as a repeatability reference
- a plain ML segmentation baseline
- a hybrid ML model with physics-inspired channels and calibrated uncertainty
- synthetic and field-style evaluation utilities
- a paper evidence-pack workflow for publication tables and figures
- a JAX sidecar wave-propagation sandbox
- a chunked volume builder for 3D/4D-style outputs
- a portable HTML and GIF rendering path for stacked field vintages

The current implementation is designed to run without downloading the large public datasets during bootstrap. It can:

- generate a Kimberlina-style synthetic benchmark locally
- train and evaluate baselines end-to-end
- optionally run inference on held-out `.npy` or `.npz` field arrays
- produce figures and machine-readable summaries for paper iteration
- build a paper-facing evidence pack from saved runs
- build a chunked `xarray`/`zarr` volume from stacked field predictions
- render self-contained Plotly HTML viewers and GIF animations
- benchmark a small JAX wave-propagation sandbox on CPU

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

If the repo-local `.vendor/` directory exists, the package adds it to `sys.path` automatically on import. That is how the optional JAX, Plotly, `zarr`, and `segyio` dependencies are used in this workspace without modifying the system Python.

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
  "support_note": "Describe whether the support mask is an exact label, a structural envelope, or a benchmark proxy.",
  "pairs": [
    {
      "name": "sleipner_2001_inline_1840_p07_mid",
      "baseline": "exports/sleipner_1994_inline_1840_p07.npy",
      "monitor": "exports/sleipner_2001_inline_1840_p07.npy",
      "reservoir_mask": "exports/sleipner_storage_interval_mask_inline_1840_p07.npy",
      "support_mask": "exports/sleipner_2010_support_volume_inline_1840_p07.npy",
      "inline_id": 1840,
      "vintage": 2001,
      "processing_family": "p07"
    }
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
PYTHONPATH=.vendor:src python3 -m ccs_monitoring.cli build-sleipner-mask --config configs/sleipner_manifest.yaml
PYTHONPATH=.vendor:src python3 -m ccs_monitoring.cli build-sleipner-plume-support --config configs/sleipner_manifest.yaml
PYTHONPATH=.vendor:src python3 -m ccs_monitoring.cli build-sleipner-support-volume --config configs/sleipner_manifest.yaml
PYTHONPATH=src python3 -m ccs_monitoring.cli build-paper-evidence --config configs/paper_evidence.yaml
PYTHONPATH=src python3 -m ccs_monitoring.cli field-seed-sweep --config configs/field_seed_sweep_colab.yaml
PYTHONPATH=src python3 -m ccs_monitoring.cli build-volume --config configs/sleipner_volume.yaml
PYTHONPATH=src python3 -m ccs_monitoring.cli render-4d --config configs/sleipner_volume.yaml
PYTHONPATH=src python3 -m ccs_monitoring.cli benchmark-jax --config configs/jax_wave_lab.yaml
```

For the real Sleipner workflow, `export-sleipner-inline` writes a `.npy` section for the configured `field.inline_number`.
`build-sleipner-mask` aligns the benchmark reservoir interval to the SEG-Y geometry, `build-sleipner-plume-support` exports the 2010 lateral support traces, and `build-sleipner-support-volume` combines them into an explicit benchmark support-volume proxy for manifest-driven field evaluation.
If `field.export_normalization_segy_paths` is set, the export uses one shared reference standard deviation across those vintages so later field comparisons stay on a common amplitude scale.

## Public-data alignment

The repo is structured so we can later plug in:

- NETL Kimberlina-derived synthetic pairs
- Sleipner 4D time-lapse sections or volumes exported to `.npz`
- benchmark-model masks for field-style plausibility checks

The first paper scope intentionally stays narrower than direct `Q` inversion.

## New next-phase configs

- [`configs/paper_evidence.yaml`](/Users/ameerfiras/Propagation/configs/paper_evidence.yaml): builds the paper-facing tables, ablations, and final direct-2010 panel
- [`configs/field_seed_sweep_colab.yaml`](/Users/ameerfiras/Propagation/configs/field_seed_sweep_colab.yaml): runs p10 and p07 multi-seed quantile sweeps with held-out sequence metrics and Pareto summaries
- [`configs/sleipner_volume.yaml`](/Users/ameerfiras/Propagation/configs/sleipner_volume.yaml): builds a stacked field volume and renders 4D-style HTML/GIF outputs
- [`configs/jax_wave_lab.yaml`](/Users/ameerfiras/Propagation/configs/jax_wave_lab.yaml): runs the CPU JAX sidecar benchmark

Local run helpers live in `jobs/`, including local shell wrappers and a SLURM-style template.

## Outputs

After `run-all`, the default output tree contains:

- `runs/<name>/data`: generated synthetic splits
- `runs/<name>/models`: trained `plain` and `hybrid` checkpoints
- `runs/<name>/results/metrics.json`: machine-readable metrics
- `runs/<name>/results/summary.json`: a compact run summary
- `runs/<name>/results/figures`: sample qualitative figures

Additional next-phase outputs:

- `runs/paper_evidence/results/paper_evidence_summary.json`
- `runs/paper_evidence/results/paper_ablation_table.csv`
- `runs/paper_evidence/results/figures/paper_direct_2010_panel.png`
- `runs/field_seed_sweep_colab/results/field_seed_sweep_per_run.csv`
- `runs/field_seed_sweep_colab/results/field_seed_sweep_aggregate.csv`
- `runs/field_seed_sweep_colab/results/field_seed_sweep_pareto.csv`
- `runs/field_seed_sweep_colab/results/field_seed_sweep_report.md`
- `runs/sleipner_volume/volume.zarr`
- `runs/sleipner_volume/results/volume_manifest.json`
- `runs/sleipner_volume/results/visualization/slice_browser.html`
- `runs/sleipner_volume/results/visualization/support_volume.html`
- `runs/sleipner_volume/results/visualization/support_evolution.gif`
- `runs/jax_wave_lab/results/jax_summary.json`
- `runs/jax_wave_lab/results/figures/jax_wavefield_animation.gif`
