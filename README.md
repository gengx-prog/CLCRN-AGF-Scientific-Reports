# CLCRN-AGF experiments

Code, selected checkpoints, training records and verification tools for the latest adaptive graph fusion (AGF) weather forecasting experiments. Start with the CPU demo, then use the external weather data to evaluate the supplied checkpoints or train new models.

CLCRN and CLConv are prior work by Haitao Lin et al. (AAAI 2022), implemented in the [upstream CLCRN repository](https://github.com/EDAPINENUT/CLCRN). The adaptive graph/fusion design draws on ASTTN (Feng and Tassiulas, 2022). These contributions remain attributed to their original authors.

## Quick start: CPU training and inference

Run from the repository root. The pinned environment recipe targets **CPython 3.13.7, Windows/Linux x86-64**. The demo needs no GPU, CUDA installation or weather download.

Windows PowerShell:

```powershell
py -3.13 -m venv .venv-review
.\.venv-review\Scripts\python.exe -m pip install -r reviewer_tools/requirements-cpu.txt
.\.venv-review\Scripts\python.exe -m pip check
.\.venv-review\Scripts\python.exe reviewer_tools/run_demo.py --output reviewer_demo_outputs/run1
```

Linux:

```bash
python3.13 -m venv .venv-review
.venv-review/bin/python -m pip install -r reviewer_tools/requirements-cpu.txt
.venv-review/bin/python -m pip check
.venv-review/bin/python reviewer_tools/run_demo.py --output reviewer_demo_outputs/run1
```

After installation, the last command is the single demo entry point. It verifies the experiment records, rebuilds 13 statistical CSVs, loads all 46 supplied checkpoints on CPU and runs six synthetic-input forwards. It then generates a tiny dataset, trains four small model variants and checks checkpoint saving, reloading and inference.

Results are written to `demo_report.json`, individual logs and detailed check reports inside the output directory. Each run requires a **new output directory**. Failed checks return a nonzero exit code.

This demo checks software execution; it does not measure weather prediction accuracy. Installation time and execution time are separate. See [verification evidence](validation/README.md) and [Windows/Linux CPU workflow results](https://github.com/gengx-prog/CLCRN-AGF-Scientific-Reports/actions/workflows/reviewer.yml) for the environments and commits actually tested.

## Evaluate the supplied models on weather data

Download the four processed task archives from the [upstream CLCRN dataset folder](https://drive.google.com/drive/folders/1sPCg8nMuDa0bAWsHPwskKkPOzaVcBneD?usp=sharing), following the [upstream data instructions](https://github.com/EDAPINENUT/CLCRN#data-preparation). Extract them into one directory:

```text
WeatherBench_processed/
  temperature/       trn.pkl  val.pkl  test.pkl  position_info.pkl
  humidity/          trn.pkl  val.pkl  test.pkl  position_info.pkl
  component_of_wind/  trn.pkl  val.pkl  test.pkl  position_info.pkl
  cloud_cover/       trn.pkl  val.pkl  test.pkl  position_info.pkl
```

These 16 external files total approximately **22.26 GiB** and are not bundled here. Their exact sizes and SHA256 hashes must match [the data manifest](experiments/provenance/data_sha256.json). Raw WeatherBench files or another preprocessing/split are not interchangeable with these inputs. The loader reads the training and validation arrays as well as the test array; allow sufficient system memory.

Using the Python executable from your environment, replace the example data path with your own:

```bash
python reviewer_tools/reviewer_verify.py check-data --data-root "/path/to/WeatherBench_processed" --all-runs --output data_check
python reviewer_tools/reviewer_verify.py evaluate --data-root "/path/to/WeatherBench_processed" --run-id temperature_agf_2021 --device cpu --output test_temperature
```

For CUDA, create a separate environment using `reviewer_tools/requirements-cu128.txt` and a compatible NVIDIA GPU/driver. To evaluate all 46 models:

```bash
python reviewer_tools/reviewer_verify.py evaluate --data-root "/path/to/WeatherBench_processed" --all-runs --device cuda:0 --output test_all_46
```

`--device cpu` also supports all 46 runs, with longer execution time. Each model is evaluated on all **657 test windows**, 12 forecast steps and 2,048 grid points. The verifier checks data hashes before inference and compares MAE/RMSE against the recorded scores with an absolute tolerance of `2e-5`. It records differences and fails if that tolerance is exceeded.

## Train new models or rebuild statistics

After verifying the processed data, start a new training run:

```bash
python experiments/code/run_revision.py --data-root "/path/to/WeatherBench_processed" --task temperature --variant agf --seed 2021 --epochs 100 --batch-size 32 --eval-batch-size 32 --patience 0 --protocol zero_inclusive --device cuda:0 --threads 4 --output new_temperature_agf_2021
```

Use `--device cpu` if needed. The recorded experiment contains 40 primary runs (four tasks, control/AGF, seeds 2021-2025) and six mean-fusion runs (temperature/humidity, seeds 2023-2025). All runs completed 100 epochs in FP32 without early stopping or AMP/TF32. Checkpoints were selected by the earliest minimum validation MAE. Training, validation and evaluation include finite zero-valued targets.

To regenerate the 13 statistical CSVs from the supplied inference JSON records, without downloading weather data or running model inference:

```bash
python reviewer_tools/reviewer_verify.py offline --output rebuilt_statistics
```

See the [detailed command guide](reviewer_tools/README.md) for output files, numerical tolerances and troubleshooting, and [execution requirements](PORTABILITY_AND_REFERENCE_SCOPE.md) for portability limits.

## Experiment files

- `reviewer_tools/`: demo, verification commands, pinned dependencies and regression checks.
- `experiments/code/`: frozen training/model implementation.
- `experiments/analysis/`: regenerate numerical and PNG diagnostics with the processed weather data.
- `experiments/selected_runs/`: 46 selected checkpoints and complete 100-epoch training histories.
- `experiments/selected_checkpoints.json`: run identities, checkpoint paths, hashes and reference scores.
- `experiments/data/`: recorded inference outputs and 13 statistical CSVs.
- `experiments/provenance/`: data fingerprints, recorded environment, frozen source hashes and distribution metadata.
- `validation/`: verification reports and execution logs.

The package retains one selected checkpoint per run and the records needed to check these experiments. Current commands accept your own data/output paths. Absolute paths in recorded execution metadata describe the original runs; they are not required local paths. The recorded computer model and complete environment snapshot are not universal requirements for running experiments.
