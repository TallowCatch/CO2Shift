# First Paper Protocol

This repository implements a practical version of the first-paper question:

> Does mismatch-aware training with calibrated uncertainty produce more reliable CO2-change maps than classical and plain-ML baselines when transferring from synthetic CCS data to field-style monitoring?

## What is already implemented

- Synthetic benchmark generation with:
  - multiple geology families
  - reservoir and plume masks
  - mismatch perturbations such as trace shifts, wavelet changes, dropped traces, amplitude scaling, and noise
- Classical baselines:
  - time-lapse difference thresholding
  - relative-impedance difference thresholding
- Learning baselines:
  - plain U-Net on baseline/monitor channels
  - hybrid U-Net with hand-crafted geophysics channels
- Reliability layer:
  - MC-dropout uncertainty
  - temperature scaling
  - selective-dice evaluation under abstention
- Reporting:
  - JSON metrics
  - a Markdown report
  - qualitative figures

## Recommended experiment order

1. Run `configs/smoke.yaml` to verify the pipeline.
2. Run `configs/paper_proto.yaml` for a more realistic synthetic study.
3. Replace the pseudo-field path with exported Sleipner baseline/monitor arrays in `.npz` format.
4. Freeze the benchmark and only then start model iteration.

## Immediate next extensions

- Add true field data ingestion from exported Sleipner slices or cubes.
- Add stronger inversion-style baseline if we decide the relative-impedance proxy is too weak.
- Add monitor-year sequences instead of single baseline/monitor pairs for temporal consistency metrics.
- Move from 2D sections to selected 3D windows once the 2D protocol is stable.

## What this repo does not claim yet

- direct `Q` inversion
- viscoacoustic field inversion
- production-scale 3D monitoring
- proof that the hybrid method is scientifically superior under the smoke config

The smoke config is only for pipeline verification. The paper-prototype config is the minimum starting point for real model comparisons.
