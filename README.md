# CLCRN-AGF: 2026-10-08 revision

**Evaluating adaptive graph fusion for short-range weather forecasting with conditional local convolution**

Manuscript author: **Xinchen Geng** · University of Southern California · gengx@usc.edu

This release evaluates an adaptive graph-fusion (AGF) input extension to the published Conditional Local Convolution Recurrent Network (CLCRN). CLCRN and CLConv are prior work by Haitao Lin et al. (AAAI 2022); adaptive graph/fusion design is inspired by ASTTN (Feng and Tassiulas, 2022). Those contributions remain attributed to their original authors.

## Current paper and downloadable materials

- [Manuscript PDF](Scientific_Reports_submission/main.pdf), [editable Word](Scientific_Reports_submission/manuscript.docx), and [LaTeX source](Scientific_Reports_submission/main.tex).
- [Supplementary Information PDF](Supplementary_Information/supplement.pdf), [editable Word](Supplementary_Information/supplement.docx), and [LaTeX source](Supplementary_Information/supplement.tex).
- [Dated release v0.2-retraining-20261008](https://github.com/gengx-prog/CLCRN-AGF-Scientific-Reports/releases/tag/v0.2-retraining-20261008): standalone manuscript/supplement source ZIPs and the complete reproducibility archive.
- [Reproducibility instructions](revision_2026_10_08/README.txt), [selected-checkpoint inventory](revision_2026_10_08/selected_checkpoints.json), and [checksums](revision_2026_10_08/SHA256_MANIFEST.json).

## New experiment grid

The current results use 40 primary runs (four tasks, control/AGF, five seeds 2021–2025) and six mean-fusion runs (temperature/humidity, seeds 2023–2025). All 46 runs completed 100 epochs in FP32, batch size 32, without early stopping or AMP/TF32. Training, validation, and testing include finite zero-valued targets. The checkpoint for each run is selected only by the earliest minimum validation MAE.

Across five training seeds, relative-humidity MAE decreases by **2.3839%** and RMSE by **3.2936%**. Both pass Benjamini–Hochberg correction across eight primary comparisons. The other six comparisons are inconclusive; the results do not establish a universal benefit, equivalence, or consistently better missing-input robustness. Uncertainty describes training-seed variability on one fixed split.

The new archive supplies all 46 selected checkpoints and all 100-epoch training histories. It does not include the entire 4,600-checkpoint local epoch archive or the raw third-party weather arrays. Historical-weight timing data are labelled separately and are not new-weight timing measurements.

## Reproduce the current tables or evaluate a checkpoint

Use Python 3.13.7 and a compatible PyTorch installation; tested dependencies are listed in `revision_2026_10_08/requirements.txt`.

```powershell
cd revision_2026_10_08
python -m pip install -r requirements.txt
python rebuild_tables.py --output reconstructed
python evaluate_selected.py --data-root "PATH/Weather Bench_dataset" --run-id temperature_agf_2021 --device cuda:0 --output reevaluation_temperature_agf_2021
```

Rebuilding tables uses the recorded inference results and does not rerun models. Checkpoint evaluation requires the four task-specific data files documented in the reproducibility README. Full raw/preprocessed arrays are obtained from the cited WeatherBench/CLCRN sources and are not redistributed here.

Validation regenerated all 13 CSV tables byte for byte. One representative selected checkpoint was independently reloaded and evaluated over all 657 test windows; MAE matched exactly and RMSE differed by 3.22e-10. This is a representative package test, not a second complete 46-model inference campaign.

## File layout and earlier versions

The [historical review supplement](HISTORICAL_REVIEW_SUPPLEMENT.md) recovers experiment files for the original 16-page SR manuscript, with all 40 primary saved models re-evaluated and known manuscript issues disclosed. Its [separate release](https://github.com/gengx-prog/CLCRN-AGF-Scientific-Reports/releases/tag/v0.1-review-supplement-20261008) does not replace the current 46-run revision.

- `Scientific_Reports_submission/`: current single-author manuscript, four main figures, and standalone LaTeX dependencies.
- `Supplementary_Information/`: current supplement with 17 tables, four figures, and numeric source tables.
- `revision_2026_10_08/`: frozen training code, fresh inference data, full training histories, 46 selected checkpoints, and provenance/validation.
- Root-level `model/`, `lib/`, `scripts/`, `analysis/`, `experiments/`, `supervisor.py`, and `env_clcrn.yaml` are retained historical material from the earlier review release. Use `revision_2026_10_08/` for the current experiment protocol and results.

The earlier `v0.1-review` release remains a historical snapshot; it does not contain this new FP32 experiment grid or the current manuscript. The current paper is not presented as an accepted or published journal article. Journal correspondence, reviewer reports, and submission identifiers are excluded from this public release.
