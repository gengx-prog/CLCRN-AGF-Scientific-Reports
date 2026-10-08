# Reproduce the manuscript figures and tables

This directory rebuilds the current manuscript's eight figures and four tables from the supplied full-precision experiment records. The visual design follows the original submission; the quantitative results come from the current 46-run revision: 40 primary control/AGF runs and six additional mean-fusion runs. Corresponding primary runs are reused in the ablation table.

The synchronized manuscript is [main.pdf](../Scientific_Reports_submission/main.pdf), with [LaTeX source](../Scientific_Reports_submission/main.tex). This directory is its figure reproduction companion. If these figures are copied into another manuscript version, that version's results and conclusions also need to match these data.

## Build in a fresh output directory

With Python 3.13.7, run from the repository root:

```bash
python -m pip install -r figure_reproduction/requirements.txt
python reviewer_tools/reviewer_verify.py figures --output reviewer_figures
```

Choose an output directory that does not already exist. The reviewer command copies the pack before rebuilding it and reports failures with a nonzero exit status. No weather dataset, checkpoint inference, PyTorch or CUDA is needed for this plotting task.

Table PDFs require `pdflatex` on PATH, supplied by TeX Live or MiKTeX, and the TeX packages `booktabs`, `geometry`, `caption`, `amsmath`, `amsfonts`, `amssymb`, `mathptmx`, `helvet`, `courier` and `microtype`. The figure insertion examples also use `graphicx`. The Windows environment recorded in [validation_summary.json](validation/validation_summary.json) was exercised; the commands also use portable paths, but that fact alone is not a claim of a completed Linux execution test.

Alternatively, copy this directory to another location, enter that copy, and run:

```bash
python -m pip install -r requirements.txt
python -X utf8 scripts/rebuild_all.py
```

This direct command overwrites generated outputs in that copy. Do not use Python's `-O` option: the plotting scripts use assertions for scientific consistency checks.

## Outputs and validation

- `figures/`: PDF and SVG manuscript assets, plus PNG previews. Figure 1 preserves the original raster illustration; Figures 2–8 are native vectors.
- `tables/`: four table CSVs and LaTeX files, with a two-page PDF preview.
- [Combined preview](preview/figures_and_tables_preview.pdf): eight figures and four tables with English captions.
- `source_data/`: the exact plotting values and intermediate aggregations.
- `manuscript_snippets/`: captions, LaTeX figure insertion examples and old-to-current filename mapping.
- `metadata/`: input hashes, source references, scientific definitions and numerical checks.
- `validation/`: build logs and the final report, including 320 numerical table checks.
- `SHA256SUMS.csv`: file sizes and SHA256 values for the supplied pack, excluding the manifest itself and transient Python bytecode.

The rebuild checks frozen input hashes before plotting. Primary statistics, horizon endpoints and pooled gate statistics are cross-checked against their source records. Figure 2 is a schematic and has no fitted or invented experimental values. PDF metadata and available fonts can vary between machines; numerical consistency is assessed from full-precision records, not a promise of identical PDF bytes.

## Scientific definitions retained in the figures

Figure 2 preserves the original three-panel composition while matching the released implementation: AGF precedes CLConv, both graph and residual branches use the same input F, and the original weather features remain in the encoder concatenation. The gate forms `z = g*q + (1-g)*r`. There are no attention heads or three separate geographic/adaptive graphs in this AGF module. The 46 configurations use 25 geographic neighbours, MLP widths 10/8/6 and one CLConv kernel view. AGF retains self plus three other neighbours, for total k=4.

Figure 3 includes only persistence, control and AGF, the three methods with current measurements. Figures 4 and 6 use the same 40 primary checkpoints. Figure 5 uses the five AGF seeds per task, with sample-SD bands; the cumulative 12-hour endpoints agree with Table 1. Figure 7 is an exploratory 64-window missing-input analysis, normalized against clean input on those same windows. Figure 8 pools all scalar gates from five models per task over 657 test windows, 12 input steps, 2,048 nodes and eight gate channels. Its violin profiles and quantiles come from the recorded 100-bin histograms; quantiles are explicitly approximate, while the mean and population SD use recorded full-stream statistics.

Only the humidity MAE and RMSE comparisons pass the stated multiple-comparison correction. Reusing the old visual design does not retain unsupported historical baseline values or old improvement claims.

## Fonts and source provenance

Figure 2 uses Comic Sans MS when installed, matching the original visual style, and automatically falls back to DejaVu Sans otherwise. No Windows font files are redistributed. The supplied PDF embeds font subsets and the SVG stores font outlines. The other plots use bundled Matplotlib fonts; table typography is supplied by TeX.

`data/` is the current plotting-data snapshot. `reference_code/` and `reference_configs/` document architecture checks; they are not alternative training entry points. Historical absolute paths in frozen provenance/configuration files identify the original machine and are not runtime requirements. `reference_previews/` and the old/new comparison PDF are labelled historical visual aids, not current numerical sources.

For real checkpoint evaluation, exact dataset hashes, full training instructions and limits of reproducibility, use the repository's [reviewer verification guide](../reviewer_tools/README.md). A successful plotting rebuild alone does not establish a new model training run or independently recomputed weather accuracy.
