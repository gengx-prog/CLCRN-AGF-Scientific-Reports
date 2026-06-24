# CLCRN-AGF Scientific Reports review archive

This repository contains the software and review archive for the manuscript:

**An Adaptive Graph-Fusion Extension to Conditional Local Convolution for Short-Range WeatherBench Forecasting**

The project evaluates a compact adaptive graph-fusion (AGF) extension to the published CLCRN weather-forecasting model on WeatherBench-style tasks.

## Repository contents

- `Scientific_Reports_submission/`: Scientific Reports manuscript source, compiled PDF, figures, and template files.
- `model/`: CLCRN, AGF, baseline model, and loss source files.
- `lib/`: utility, metric, transform, and optimizer helper code.
- `scripts/`: preprocessing, publication experiment, evaluation, figure, and efficiency scripts.
- `analysis/`: corrected publication-metric evaluator and humidity ablation completion queue.
- `experiments/corrected_publication_metrics/`: corrected per-seed publication metrics and aggregate summaries.
- `experiments/humidity_ablation_100ep_amp_b32/`: relative-humidity 100-epoch, 3-seed `w/o gated fusion` ablation completion summaries.
- `env_clcrn.yaml`: environment specification.
- `upstream_CLCRN_README.md`: original upstream CLCRN README/data-preparation notes.
- `Scientific_Reports_software_archive_manifest.md`: detailed archive manifest.

## Data availability

Raw ERA5/WeatherBench data are available from WeatherBench and the Copernicus Climate Data Store. Preprocessed arrays are not redistributed here because they are derived from third-party data sources. The upstream CLCRN README describes the preprocessed data link used by the original implementation.

## Review release

The release asset `Scientific_Reports_CLCRN_AGF_software_review_archive.zip` contains the same materials in a single downloadable archive.

## Reproducibility note

All manuscript test metrics use the corrected zero-inclusive evaluator. The humidity `w/o gated fusion` ablation was completed with PyTorch automatic mixed precision and batch size 32 after standard-precision runs exhausted the local CUDA allocator.

