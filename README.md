# CLCRN-AGF reproducibility release

**Evaluating adaptive graph fusion for short-range weather forecasting with conditional local convolution**

Manuscript author: **Xinchen Geng**, University of Southern California, gengx@usc.edu.

This study evaluates an adaptive graph-fusion (AGF) input extension to the published Conditional Local Convolution Recurrent Network (CLCRN). CLCRN and CLConv are prior work by Haitao Lin et al. (AAAI 2022); the adaptive graph/fusion design is inspired by ASTTN (Feng and Tassiulas, 2022). Those contributions remain attributed to their original authors.

## Current paper and complete materials

- [Manuscript PDF](Scientific_Reports_submission/main.pdf) and [LaTeX source](Scientific_Reports_submission/main.tex).
- [Supplementary Information PDF](Supplementary_Information/supplement.pdf) and [LaTeX source](Supplementary_Information/supplement.tex).
- [Release v0.3](https://github.com/gengx-prog/CLCRN-AGF-Scientific-Reports/releases/tag/v0.3-review-reproducibility-20261008): downloadable PDFs, the complete code/data-record/checkpoint/figure-source archive, and SHA256 checksums.
- [Reviewer instructions](reviewer_tools/README.md), [checkpoint inventory](revision_2026_10_08/selected_checkpoints.json), and [frozen experiment checksums](revision_2026_10_08/SHA256_MANIFEST.json).
- [Automated Windows/Linux CPU checks](https://github.com/gengx-prog/CLCRN-AGF-Scientific-Reports/actions/workflows/reviewer.yml) and [local verification evidence](validation/README.md).

The paper retains the original manuscript's 58 references, with bibliographic corrections, and adds three references for the contemporary large-model context. Citations are assigned individually to the supported statements. Figure 2 retains the original colorful three-panel visual design while correcting the gate inputs and its position before CLConv. Its PDF is vector artwork with embedded fonts. Statistical figures use the current 46-run experiment records; the original conceptual Figure 1 is retained.

## Run a check without downloading weather data

Use a complete checkout or extract the entire release ZIP. From its top-level directory, create a clean **CPython 3.13.7** environment and install the CPU requirements:

```powershell
py -3.13 -m venv .venv-review
.\.venv-review\Scripts\python.exe -m pip install -r reviewer_tools/requirements-cpu.txt
.\.venv-review\Scripts\python.exe reviewer_tools/reviewer_verify.py quick --output reviewer_check
```

On Linux, replace the first command with `python3.13 -m venv .venv-review` and use `.venv-review/bin/python` in the remaining commands. Each output directory must be new. See the [full instructions](reviewer_tools/README.md) for CUDA, training, figures, and troubleshooting.

`quick` verifies the frozen package and all 46 training records, rebuilds 13 CSV files from the recorded inference JSON files, strictly loads all 46 real checkpoints on CPU, and executes six synthetic-input forward passes plus metric edge-case checks. **It does not measure weather accuracy or retrain the models.** Failure returns a nonzero exit code and a diagnostic report.

## Re-evaluate the supplied checkpoints on weather data

Download the four processed task archives from the [original CLCRN dataset folder](https://drive.google.com/drive/folders/1sPCg8nMuDa0bAWsHPwskKkPOzaVcBneD?usp=sharing), following the [upstream data instructions](https://github.com/EDAPINENUT/CLCRN#data-preparation). Extract them under one directory containing `temperature/`, `humidity/`, `component_of_wind/`, and `cloud_cover/`; each task must contain `trn.pkl`, `val.pkl`, `test.pkl`, and `position_info.pkl`.

The 16 external files total about 22.26 GiB. They are not bundled in this release. The verifier checks their exact sizes and SHA256 values against [data provenance](revision_2026_10_08/provenance/data_sha256.json) before inference. A different preprocessing or split is not interchangeable with these files.

```powershell
# Use the Python executable from the environment you created.
python reviewer_tools/reviewer_verify.py evaluate --data-root "D:/WeatherBench_processed" --run-id temperature_agf_2021 --device cpu --output test_temperature

# For all 46 models, use a separate environment installed from
# reviewer_tools/requirements-cu128.txt and a compatible NVIDIA GPU.
python reviewer_tools/reviewer_verify.py evaluate --data-root "D:/WeatherBench_processed" --all-runs --device cuda:0 --output test_all_46
```

The CPU command also supports `--all-runs`, but takes longer. No author-specific drive path is required. Actual inference covers all **657 test windows** per model and checks MAE/RMSE against the supplied results with an absolute tolerance of `2e-5`.

On 8 October 2026, all 46 supplied checkpoints passed full-test re-evaluation on the hash-verified data. The maximum absolute differences were **2.23e-16 for MAE** (rounded upward) and **4.07e-10 for RMSE** (rounded upward). These are reloaded-model inference checks, not another training campaign. A separate clean CPU environment passed the record/checkpoint/forward checks. Full evidence and tested package versions are in [validation](validation/README.md).

## Current experiment grid and interpretation

The current results use 40 primary runs (four tasks, control/AGF, five seeds 2021-2025) and six mean-fusion runs (temperature/humidity, seeds 2023-2025). All 46 runs completed 100 epochs in FP32, batch size 32, without early stopping or AMP/TF32. Training, validation, and testing include finite zero-valued targets. Each checkpoint is selected by the earliest minimum validation MAE.

Across five training seeds, relative-humidity MAE decreases by **2.3839%** and RMSE by **3.2936%**. Both pass Benjamini-Hochberg correction across eight primary comparisons. The other six comparisons are inconclusive; the results do not establish a universal benefit, equivalence, or consistently better missing-input robustness. Uncertainty describes training-seed variability on one fixed split.

The archive supplies all 46 selected checkpoints and all 100-epoch histories, configurations, frozen training/evaluation code, recorded test results, and figure/table sources. It does not include all 4,600 intermediate epoch checkpoints or the externally hosted weather arrays. Historical-weight timing data are labelled separately and are not new-weight timing measurements.

## Rebuild figures and PDFs

Install `reviewer_tools/requirements-figures.txt` in addition to the CPU dependencies, and install a TeX distribution with `pdflatex` on PATH. Then run:

```powershell
python reviewer_tools/reviewer_verify.py figures --output rebuilt_figures
python paper_build/build_papers.py
```

The first command makes a fresh copy of the figure source pack and regenerates eight figures and four numeric tables, including 320 table-cell comparisons. It does not rerun model inference. The second builds and validates the main and supplementary PDFs from the supplied manuscript assets. Figure 2 uses Comic Sans MS when available and DejaVu Sans as the portable fallback; numeric data do not depend on this font choice.

## Layout and historical versions

- `Scientific_Reports_submission/`: current LaTeX manuscript, eight main figures, four numeric table inputs, and TeX dependencies. The fifth table is the contextual large-model comparison in the manuscript source.
- `Supplementary_Information/`: current supplement, six figures, and numeric source tables.
- `figure_reproduction/`: figure-generation code, input records, per-figure provenance, tables, and validation outputs.
- `reviewer_tools/`: portable verification commands, pinned dependencies, and regression checks.
- `revision_2026_10_08/`: unchanged training/evaluation snapshot, all 46 selected checkpoints, fresh inference records, full training histories, and provenance.
- `validation/` and `paper_build/`: release verification evidence and PDF build/check tools.
- Root-level `model/`, `lib/`, `scripts/`, `analysis/`, `experiments/`, `supervisor.py`, and `env_clcrn.yaml` are **historical material**. Use the current entry points above for this paper. Likewise, `historical_v0_2_*.docx` and `Scientific_Reports_submission/figs_revised/` are superseded assets, not the current paper.

The [historical review supplement](HISTORICAL_REVIEW_SUPPLEMENT.md) and its [separate release](https://github.com/gengx-prog/CLCRN-AGF-Scientific-Reports/releases/tag/v0.1-review-supplement-20261008) recover the original 16-page manuscript's experiments and disclose their remaining gaps. Those old results are separate from the current 46-run revision. The earlier releases remain historical snapshots. The current paper is not presented as an accepted or published journal article. Private journal correspondence, reviewer reports, and submission identifiers are excluded.
