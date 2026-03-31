# C02Shift Colab Outcomes (2026-03-31)

This note records the key field outcomes from the Colab recovery and threshold sweep runs.

## p10 threshold sweep (wave_temporal_constrained)

| Quantile | Support-volume IoU | Trace-support IoU | Outside-support fraction | Crossline continuity |
|---|---:|---:|---:|---:|
| 0.90 | 0.001817 | 0.036530 | 0.000000 | 0.000000 |
| 0.86 | 0.156794 | 0.937620 | 0.014777 | 0.970131 |
| 0.82 | 0.373963 | 0.887635 | 0.025495 | 0.949117 |
| 0.78 | 0.609684 | 0.706452 | 0.089478 | 0.849532 |

Reference in the same runs: `best_classical_constrained` support-volume IoU = `0.345473`.

## Cross-benchmark checkpoints

| Run | Support-volume IoU | Trace-support IoU | Outside-support fraction | Crossline continuity | Classical support-volume IoU |
|---|---:|---:|---:|---:|---:|
| p10 q82 | 0.373963 | 0.887635 | 0.025495 | 0.949117 | 0.345473 |
| p10 q78 | 0.609684 | 0.706452 | 0.089478 | 0.849532 | 0.345473 |
| p07 q82 | 0.283597 | 0.710269 | 0.082211 | 0.848535 | 0.406802 |
| p07 q78 | 0.348894 | 0.599910 | 0.161777 | 0.875956 | 0.406802 |

## Practical takeaways

- `q82` is the balanced operating point for paper-facing reporting.
- `q78` is a high-occupancy ablation with a clear trace/containment tradeoff.
- On this run set, p10 shows improvement over constrained classical for support-volume IoU at `q82` and `q78`.
- p07 remains below constrained classical support-volume IoU in both tested operating points.
