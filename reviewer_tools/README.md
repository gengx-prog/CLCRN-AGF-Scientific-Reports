# Reviewer verification entry points

Run these commands from a complete checkout or the complete release archive. Python scripts locate the repository relative to their own file, so no author-specific drive path is required. Outputs must go into a **new directory**. Supplied checkpoints, training logs and reference scores are never overwritten.

There are three different reproducibility claims:

| Command | What it verifies | What it does not establish |
| --- | --- | --- |
| `quick` | Archive/source hashes, all 46 training records and selected checkpoints, regeneration of all 13 CSVs from recorded inference JSONs, strict CPU loading of all 46 real checkpoints, six synthetic 12-step forward passes, zero-inclusive loss/evaluator behavior | Weather accuracy on the test set, or a new training run |
| `evaluate` | Actual inference on all 657 test windows for each requested supplied checkpoint, after checking exact dataset SHA256 values; compares MAE/RMSE with supplied scores | Independent retraining or a guarantee of identical floating point results on every device |
| `figures` | Rebuilds the current figure/table reproduction pack in a fresh output copy, with its numeric and document checks | Model inference or new training |

## 1. Create a clean Python environment

The tested interpreter is **CPython 3.13.7, x86-64**. The original new-training environment used Windows 11, an RTX 5090 Laptop GPU, PyTorch 2.10.0+cu128 and FP32, with TF32/AMP disabled. Full original environment provenance is in `revision_2026_10_08/provenance/environment.json`. The new verification report records the machine and installed package versions actually used for that check.

Windows PowerShell:

```powershell
py -3.13 -m venv .venv-review
.\.venv-review\Scripts\python.exe -m pip install --upgrade pip
.\.venv-review\Scripts\python.exe -m pip install -r reviewer_tools/requirements-cpu.txt
.\.venv-review\Scripts\python.exe -m pip check
.\.venv-review\Scripts\python.exe reviewer_tools/reviewer_verify.py quick --output reviewer_check
```

Linux x86-64, with Python 3.13 installed:

```bash
python3.13 -m venv .venv-review
.venv-review/bin/python -m pip install --upgrade pip
.venv-review/bin/python -m pip install -r reviewer_tools/requirements-cpu.txt
.venv-review/bin/python -m pip check
.venv-review/bin/python reviewer_tools/reviewer_verify.py quick --output reviewer_check
```

For CUDA inference, create a separate environment and install `reviewer_tools/requirements-cu128.txt` instead of the CPU requirements. Do not install both PyTorch builds into the same environment. A compatible NVIDIA GPU/driver is needed. The requirements target Windows/Linux x86-64; they are not a macOS or ARM environment specification. Cross-platform dependency resolution is separate from an actual Linux execution test; inspect the published verification records for which operating systems were actually exercised.

The `quick` command does not download anything at runtime, needs no WeatherBench files, and uses CPU even when CUDA is installed. It checks finite, correctly shaped forecasts from six actual checkpoints on explicitly synthetic coordinates and inputs. Those diagnostic prediction values are **not weather test metrics**. Frozen training files are checked against their original 24-file source manifest.

`reviewer_check/verification_report.json` reports the complete scope and pass/fail status. `rebuild_records.log` and any `failure_traceback.txt` explain failures. A failure returns a nonzero process exit code; no metric mismatch is silently accepted. `offline` runs only evidence/CSV checks; `smoke` runs only checkpoint/metric execution checks.

CSV regeneration starts from JSON inputs only; no supplied CSV is copied into that output directory. The frozen analysis builder also emits its original plot layouts and historical analysis-stage text (`Manuscript has NOT been updated`, `manuscript_updated=false`). Those fields are retained for provenance and do not describe the current revised manuscript. `reconstructed_records/REBUILD_SCOPE.txt` explains this distinction. Use `figures` below to regenerate the current manuscript's restored figure styles.

Reconstructed CSV schemas, row order, text and integer values must match exactly. Floating point cells allow relative and absolute tolerance `1e-12` for cross-platform statistical roundoff; the report records every nonidentical numeric cell count and maximum absolute difference, and separately reports whether each CSV is text-identical after newline normalization. This statistical-rebuild tolerance is separate from the `2e-5` actual-inference metric tolerance.

## 2. Obtain and verify the external weather data

Use the original CLCRN project's [preprocessed four-task dataset folder](https://drive.google.com/drive/folders/1sPCg8nMuDa0bAWsHPwskKkPOzaVcBneD?usp=sharing) and [data preparation documentation](https://github.com/EDAPINENUT/CLCRN#data-preparation). The raw source is [WeatherBench](https://github.com/pangeo-data/WeatherBench). The external provider controls access and availability. These data are not bundled with this code release.

Arrange the downloaded/extracted files as:

```text
WeatherBench_processed/
  temperature/       trn.pkl  val.pkl  test.pkl  position_info.pkl
  humidity/          trn.pkl  val.pkl  test.pkl  position_info.pkl
  component_of_wind/  trn.pkl  val.pkl  test.pkl  position_info.pkl
  cloud_cover/       trn.pkl  val.pkl  test.pkl  position_info.pkl
```

The expected 16 file sizes and SHA256 values are in `revision_2026_10_08/provenance/data_sha256.json`. Historical absolute paths in that document identify the source copy; the verifier uses only the portable `task/filename` suffixes. The total data size is approximately 22.26 GiB. Do not substitute a differently preprocessed or differently split dataset and interpret its numbers as a failed reproduction of this split. If a hash differs, verify extraction, provider version and preprocessing before inference. Raw WeatherBench downloads alone are not these task pickle files.

```bash
python reviewer_tools/reviewer_verify.py check-data --data-root "/path/to/WeatherBench_processed" --all-runs --output reviewer_data_check
```

Only trusted, hash-verified supplied checkpoints and upstream dataset pickles should be loaded. The program checks all required dataset hashes **before** calling the original evaluator. It needs the training file to reproduce training-only standardization, even when evaluating a fixed checkpoint. The frozen loader also loads the validation split. Loading the several-GB task arrays therefore requires substantially more memory than the checkpoint file size; the demonstrated machine has approximately 64 GB system RAM.

## 3. Run actual full-test inference

Using the relevant virtual environment's `python`:

```bash
python reviewer_tools/reviewer_verify.py evaluate --data-root "/path/to/WeatherBench_processed" --run-id temperature_agf_2021 --device cpu --output reviewer_one_model
```

With the CUDA environment, verify every current manuscript checkpoint:

```bash
python reviewer_tools/reviewer_verify.py evaluate --data-root "/path/to/WeatherBench_processed" --all-runs --device cuda:0 --output reviewer_all_46
```

The 46 runs comprise 40 primary runs (four tasks, control/AGF, seeds 2021-2025) and six additional mean-fusion ablations (temperature/humidity, seeds 2023-2025). The ablation table reuses the corresponding control/AGF primary runs; it is not another 12 training runs. Repeat `--run-id` to select multiple runs. No training is performed by this command.

The wrapper delegates model inference to the unchanged `revision_2026_10_08/evaluate_selected.py` and frozen `code/`. Every selected run is evaluated on the complete **657 test windows, 12 forecast steps, 2,048 grid points**; wind has two output channels and the other tasks have one. Both MAE and RMSE include finite zero and negative physical targets, use uniform gridpoint weighting, and accumulate all horizons. The reference tolerance is an absolute `2e-5` in each reported metric's physical units. The report preserves signed differences. Hardware/library differences may change roundoff; an out-of-tolerance result fails and remains available for inspection rather than replacing the paper's reference values.

The output contains one log and `inference/<run_id>/reevaluation.json` per model, `preflight.json` with dataset hashes, `completed_inference.json` updated after every successful run, and a final `verification_report.json`. If interrupted, retain that output and use a new directory with explicit remaining `--run-id` arguments. Existing evidence is not reused silently. CPU execution is supported but full 46-model inference can take substantially longer than CUDA.

## 4. Rebuild the figure/table files

```bash
python -m pip install -r reviewer_tools/requirements-figures.txt
python reviewer_tools/reviewer_verify.py figures --output reviewer_figures
```

Table PDFs additionally need `pdflatex` on PATH from TeX Live or MiKTeX, with `booktabs`, `geometry`, `caption`, `amsmath`, `amsfonts` and `amssymb`. The command explains missing dependencies. Numeric `quick` checks do not need LaTeX. The current reproduction pack is copied to the output directory before executing its build script. Fonts and PDF metadata can vary by platform; numerical comparisons use the full-precision source tables rather than PDF file hashes across machines. Published source assets remain available for inspecting the exact manuscript appearance.

## 5. New training is separate

Follow `revision_2026_10_08/README.txt` and the frozen `code/run_revision.py` entry point to retrain, after the dataset hash check. For example:

```bash
python revision_2026_10_08/code/run_revision.py --data-root "/path/to/WeatherBench_processed" --task temperature --variant agf --seed 2021 --epochs 100 --batch-size 32 --eval-batch-size 32 --patience 0 --protocol zero_inclusive --device cuda:0 --threads 4 --output new_temperature_agf_2021
```

The supplied 46 checkpoints are validation-selected complete training states, accompanied by all 100 epochs of training history per run. This review release stores the selected checkpoint for each run, not every epoch's weight file. Selection is the earliest full-validation MAE minimum among 100 epochs. The package does not claim that another machine will reproduce every trained parameter bit for bit.

Guard regression tests can be run with `python -m unittest discover -s reviewer_tools -p test_reviewer_tools.py -v`. They check corruption rejection, missing-file/size failures, path containment, portable data-fingerprint resolution and rejection of changed CSV values/rows while allowing tiny statistical roundoff.
