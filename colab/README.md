# Colab runner for the C02Shift branch

This folder is intentionally separate from the main research notebook sequence. It is for remote execution only.

`C02Shift` is the working name for the wave-consistent temporal monitoring branch.

## First-time setup

- Use `C02Shift_quickstart.ipynb` for your first run.
- It mounts Drive, clones or updates the repo, installs dependencies, downloads required public Sleipner assets, and runs the direct `p10` gate.
- You do not need to manually upload your local `examples` folder.

## Recommended order

1. Run the direct `p10` gate first.
2. Only run the full temporal `p07` benchmark if the direct anchor is competitive.

## What the notebooks do

- `C02Shift_quickstart.ipynb`
  - first-time setup
  - downloads public Sleipner data and benchmark archives from CO2DataShare
  - runs the direct `p10` gate with `C02Shift`

- `C02Shift_p10_gate.ipynb`
  - assumes your data is already prepared
  - runs the synthetic prototype
  - runs the direct `p10` anchor
  - prints the direct novelty-gate comparison against the current structured plain baseline

- `wave_temporal_runner.ipynb`
  - assumes your data is already prepared
  - runs the synthetic prototype
  - runs the direct `p10` anchor
  - runs the `p07` temporal benchmark
  - prints summary paths and sequence metrics

The gate and runner notebooks:

- mount Google Drive
- check that the required Sleipner exports and manifests already exist
- install the project and visualization dependencies
- run the C02Shift branch from the Drive-backed workspace

## Colab configs

- `/Users/ameerfiras/Propagation/configs/paper_wave_temporal_colab.yaml`
- `/Users/ameerfiras/Propagation/configs/sleipner_p10_wave_temporal_colab.yaml`
- `/Users/ameerfiras/Propagation/configs/sleipner_p07_wave_temporal_colab.yaml`

## Important note

- The notebooks expect a Python `3.11` runtime because the current project metadata requires `>=3.11`.
- If Colab starts with an older Python runtime, use a `3.11` runtime if available or relax the requirement locally before installing.
