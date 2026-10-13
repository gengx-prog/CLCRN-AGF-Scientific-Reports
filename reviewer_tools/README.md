# Experiment execution and verification

Run commands from the repository root, using the Python executable from the environment you install below. Scripts resolve supplied files relative to the repository and accept your own data paths. Always choose a **new output directory**; existing results and supplied checkpoints are protected from overwriting.

## 1. Install a CPU environment and run the demo

The pinned recipe targets **CPython 3.13.7 on Windows/Linux x86-64**. This is a tested configuration, not a universal minimum or a guarantee for other package/platform combinations. See [verification evidence](../validation/README.md) and [the CPU workflow](https://github.com/gengx-prog/CLCRN-AGF-Scientific-Reports/actions/workflows/reviewer.yml) for tested commits and environments.

Windows PowerShell:

```powershell
py -3.13 -m venv .venv-review
.\.venv-review\Scripts\python.exe -m pip install -r reviewer_tools/requirements-cpu.txt
.\.venv-review\Scripts\python.exe -m pip check
.\.venv-review\Scripts\python.exe reviewer_tools/run_demo.py --output reviewer_demo_outputs/run1
```

Linux x86-64:

```bash
python3.13 -m venv .venv-review
.venv-review/bin/python -m pip install -r reviewer_tools/requirements-cpu.txt
.venv-review/bin/python -m pip check
.venv-review/bin/python reviewer_tools/run_demo.py --output reviewer_demo_outputs/run1
```

Once installed, the single command is:

```bash
python reviewer_tools/run_demo.py --output reviewer_demo_outputs/run1
```

The demo runs `quick`, then generates six training, three validation and three test samples on a six-node graph. It trains four small variants (AGF, control, mean fusion and a legacy-loss regression case), checks that optimization changes parameters, saves/reloads checkpoints and compares evaluation with an independent NumPy calculation. Separately, `quick` loads the supplied full-size checkpoints and verifies the recorded results.

Outputs include `demo_report.json`, `quick.txt`, `training.txt`, `quick/verification_report.json`, `training/smoke_results.json`, toy data and toy checkpoints. Reports record environment, elapsed time, commands and source hashes. Failures return a nonzero exit code and preserve diagnostic logs. Use `--threads 2` to control the tiny training run; `quick` uses four CPU threads.

No external weather data or GPU is needed. Synthetic predictions are not weather accuracy measurements, and the demo is not a 100-epoch training reproduction. Recorded runtimes apply to the stated machine and exclude dependency installation.

For CUDA execution, create a separate environment and install `reviewer_tools/requirements-cu128.txt`. A compatible NVIDIA GPU/driver is required. Do not mix the CPU and CUDA PyTorch builds in one environment. These recipes do not specify macOS, ARM, MPS or ROCm environments.

## 2. Select a check

| Command | Scope | External weather data |
| --- | --- | --- |
| `run_demo.py` | `quick` plus tiny-model training, checkpoint save/reload and inference | No |
| `reviewer_verify.py quick` | Experiment/source integrity, all 46 training records, 13 regenerated CSVs, strict loading of 46 checkpoints, six synthetic forwards and metric checks | No |
| `reviewer_verify.py offline` | Experiment/source integrity, training records and 13 regenerated statistical CSVs | No |
| `reviewer_verify.py smoke` | Checkpoint loading, synthetic forwards and metric checks | No |
| `reviewer_verify.py check-data` | File sizes and SHA256 hashes for the selected tasks | Yes |
| `reviewer_verify.py evaluate` | Full-test inference with supplied weights, data hash checks and score comparison | Yes |

Example commands:

```bash
python reviewer_tools/reviewer_verify.py quick --output checkpoint_check
python reviewer_tools/reviewer_verify.py offline --output rebuilt_statistics
```

The 13 CSVs are regenerated from the supplied inference JSON records, without copying the supplied CSVs into the result directory. This checks statistical reconstruction, not fresh model inference. Schemas, row order, text and integer values must match exactly. Floating-point cells use relative and absolute tolerances of `1e-12`; reports record differing cell counts, maximum absolute differences and text identity after newline normalization. This tolerance is separate from the actual-inference tolerance below.

`verification_report.json` contains pass/fail status and check scope. Inspect `rebuild_records.log` and any `failure_traceback.txt` for diagnostics. No failed metric or CSV comparison is silently accepted.

To run only tiny-model training/save/reload:

```bash
python revision_2026_10_08/code/run_revision.py --smoke --device cpu --threads 2 --output training_smoke
```

Run this direct command without `-O` or `PYTHONOPTIMIZE`, because its checks use assertions. The combined demo enables assertions in its child processes.

## 3. Download and verify the processed weather data

Download from the [upstream CLCRN four-task dataset folder](https://drive.google.com/drive/folders/1sPCg8nMuDa0bAWsHPwskKkPOzaVcBneD?usp=sharing) using the [CLCRN preparation instructions](https://github.com/EDAPINENUT/CLCRN#data-preparation). The raw source is [WeatherBench](https://github.com/pangeo-data/WeatherBench). Access and availability are controlled by the external provider.

Extract the files as follows:

```text
WeatherBench_processed/
  temperature/       trn.pkl  val.pkl  test.pkl  position_info.pkl
  humidity/          trn.pkl  val.pkl  test.pkl  position_info.pkl
  component_of_wind/  trn.pkl  val.pkl  test.pkl  position_info.pkl
  cloud_cover/       trn.pkl  val.pkl  test.pkl  position_info.pkl
```

The 16 files total approximately **22.26 GiB**. Exact sizes and SHA256 values are in [`data_sha256.json`](../revision_2026_10_08/provenance/data_sha256.json). The verifier uses portable `task/filename` suffixes, so original drive paths in execution metadata need not exist on your machine.

Replace `/path/to/WeatherBench_processed` with your actual directory:

```bash
python reviewer_tools/reviewer_verify.py check-data --data-root "/path/to/WeatherBench_processed" --all-runs --output data_check
```

Raw WeatherBench downloads or another preprocessing/split do not substitute for these processed files. Resolve hash differences before evaluation. Load only trusted checkpoints and dataset pickles. All required dataset hashes are checked before the evaluator runs.

Evaluation needs the training file for training-only standardization. The frozen loader also materializes validation and test arrays. Allow substantially more memory than the checkpoint file size; the recorded full-evaluation machine had approximately 64 GB system RAM. This is an observed setup, not a universal minimum.

## 4. Evaluate selected checkpoints

Evaluate one model on CPU:

```bash
python reviewer_tools/reviewer_verify.py evaluate --data-root "/path/to/WeatherBench_processed" --run-id temperature_agf_2021 --device cpu --output test_temperature
```

With the CUDA environment, evaluate all 46:

```bash
python reviewer_tools/reviewer_verify.py evaluate --data-root "/path/to/WeatherBench_processed" --all-runs --device cuda:0 --output test_all_46
```

CPU supports `--all-runs` but can take substantially longer. Repeat `--run-id` to select multiple models. Run identities and reference scores are in [`selected_checkpoints.json`](../revision_2026_10_08/selected_checkpoints.json).

The wrapper delegates inference to `revision_2026_10_08/evaluate_selected.py` and the frozen model implementation. Every selected run uses all **657 test windows, 12 forecast steps and 2,048 grid points**. Wind has two output channels; other tasks have one. MAE and RMSE include finite zero and negative physical targets, use uniform gridpoint weighting and accumulate all forecast horizons.

The absolute acceptance tolerance is `2e-5` in each metric's physical units. Signed differences are preserved; an out-of-tolerance result fails without replacing reference values. Hardware and library changes can affect roundoff and need investigation if checks fail.

Outputs include one log and `inference/<run_id>/reevaluation.json` per model, `preflight.json` with data hashes, `completed_inference.json` updated after successful runs, and a final `verification_report.json`. If interrupted, retain the output and use a new directory with the remaining `--run-id` arguments. The evaluator uses batch size 32 and is not a streaming or low-memory loader.

## 5. Train a new model

After checking dataset hashes:

```bash
python revision_2026_10_08/code/run_revision.py --data-root "/path/to/WeatherBench_processed" --task temperature --variant agf --seed 2021 --epochs 100 --batch-size 32 --eval-batch-size 32 --patience 0 --protocol zero_inclusive --device cuda:0 --threads 4 --output new_temperature_agf_2021
```

Use `--device cpu` where appropriate. Available tasks are `temperature`, `humidity`, `component_of_wind` and `cloud_cover`; variants are `control`, `agf` and `mean_fusion`.

The recorded grid has 40 primary runs (four tasks, control/AGF, seeds 2021-2025) and six mean-fusion runs (temperature/humidity, seeds 2023-2025). All completed 100 epochs in FP32, batch size 32, without early stopping or AMP/TF32. The earliest minimum full-validation MAE selects the checkpoint. Standardization uses training data only; loss and evaluation include finite zero-valued targets.

All 46 selected checkpoints and full 100-epoch histories are supplied. Intermediate epoch weights are not all included. New training on another device is not guaranteed to produce bit-identical weights. Frozen `--resume` provenance checks are stricter than new training or fixed-checkpoint evaluation; cross-device resume is not established by these instructions.

## 6. Regenerate supplementary numerical diagnostics

With the matching processed weather data and CUDA environment:

```bash
python revision_2026_10_08/analysis/analyze_selected.py --data-root "/path/to/WeatherBench_processed" --device cuda:0 --output supplementary_analysis
```

Use `--device cpu` where appropriate. This script checks package, training and data hashes, then performs all 46 full-test inferences, latitude/horizon metrics, gate summaries, persistence comparisons and deterministic 64-window masking diagnostics. It writes fresh JSON, CSV and PNG outputs, and requires substantially more compute than the CPU demo.

This portable analysis entry point was adapted for this distribution. A complete supplementary run through the new entry point has not yet been verified; existing recorded experiment outputs remain available separately. The CPU demo and the supplied full-test inference evidence do not establish completion of this supplementary command.

## Troubleshooting and regression checks

- **Installation/import failure:** check Python version, x86-64 architecture and the selected CPU/CUDA requirements. Run `python -m pip check` in that environment.
- **CUDA unavailable:** check the installed PyTorch build and NVIDIA driver, or use the CPU commands.
- **Missing files/hash mismatch:** check extraction and the exact processed dataset version against the manifests.
- **Memory failure:** the loader reads several large arrays, and graph intermediates also use memory. Inspect available RAM/VRAM and the selected workload.
- **Output already exists:** choose a new output directory so previous evidence is retained.
- **Metric mismatch:** inspect signed differences, data hashes and full logs; do not silently widen the tolerance.

For regression checks:

```bash
python -m unittest discover -s reviewer_tools -p "test_*.py" -v
```

These checks cover corruption rejection, data/path validation, CSV comparison, output protection and demo failure reporting. For help diagnosing a failure, retain the command, repository commit, OS/architecture, Python/package versions, device, available memory and complete traceback. See [execution requirements and portability](../PORTABILITY_AND_REFERENCE_SCOPE.md).
