# CLCRN-AGF Scientific Reports software/review archive manifest

This archive accompanies the manuscript:

**An Adaptive Graph-Fusion Extension to Conditional Local Convolution for Short-Range WeatherBench Forecasting**

It is intended for Scientific Reports peer review and later public deposition.

Public repository: <https://github.com/gengx-prog/CLCRN-AGF-Scientific-Reports>

Review release tag: `v0.1-review`

## Included materials

- `README.md`: upstream CLCRN project notes and data-preparation pointers.
- `env_clcrn.yaml`: Python environment specification used by the project.
- `model/`: CLCRN, AGF, baseline model, and loss source files.
- `lib/`: utility, metric, transform, and optimizer helper code.
- `scripts/`: preprocessing, publication experiment, evaluation, figure, and efficiency scripts.
- `analysis/recompute_corrected_publication_metrics.py`: corrected zero-inclusive evaluator used for the reported publication metrics.
- `analysis/run_humidity_wo_gated_100ep_amp_b32_queue.py`: queue script for the supplemental humidity 100-epoch, 3-seed w/o gated fusion ablation.
- `run_asttn_ablation.py` and `supervisor.py`: main ablation/training entry point and training supervisor.
- `experiments/corrected_publication_metrics/`: corrected per-seed publication metrics and aggregate summaries.
- `experiments/humidity_ablation_100ep_amp_b32/`: humidity w/o gated fusion ablation completion summaries, logs, and aggregate results.
- `Scientific_Reports_submission/`: compiled manuscript source, PDF, figures, and template files.

## Excluded materials

- Raw or preprocessed WeatherBench/ERA5 arrays are not included because they are derived from third-party data sources.
- Local virtual environments, temporary files, and bulk exploratory experiment directories are excluded.

## Data access

Raw ERA5/WeatherBench data are available from WeatherBench and the Copernicus Climate Data Store. The original CLCRN README describes the preprocessed WeatherBench data link used by the upstream implementation.

## Reproducibility note

All reported test metrics in the manuscript use the corrected zero-inclusive evaluator. The humidity w/o gated fusion ablation was completed with PyTorch automatic mixed precision and batch size 32 after standard-precision runs exhausted the local CUDA allocator.
