CLCRN-AGF EXPERIMENT SNAPSHOT

This directory contains 46 selected checkpoints, the corresponding 100-epoch
training histories/configurations, frozen training/evaluation sources, recorded
inference results and numerical verification tools.

TRAINING AND CHECKPOINTS
Forty primary runs cover four tasks, control/AGF and seeds 2021-2025. Six
mean-fusion runs cover temperature/humidity and seeds 2023-2025. All completed
100 epochs with batch size 32, FP32, zero-inclusive loss/validation, no early
stopping and no AMP/TF32. Selection uses the earliest minimum validation MAE.
Only the selected checkpoint per run is included. The selected_checkpoints.json
inventory records portable paths, epochs, SHA256 values and reference scores.

QUICK CPU EXECUTION
Install ../reviewer_tools/requirements-cpu.txt using CPython 3.13.7. From the
repository root, run:
  python reviewer_tools/run_demo.py --output reviewer_demo_outputs/run1
This validates the experiment package and supplied weights, then generates
small synthetic data and tests train/save/reload/inference. It is not a weather
accuracy measurement.

REBUILD NUMERICAL RESULTS (NO WEATHER DATA NEEDED)
From this directory:
  python rebuild_tables.py --output ../reconstructed
The builder reads recorded inference JSONs and regenerates 13 CSV files,
experiment_summary.json and PNG diagnostics. It does not perform model inference.

REEVALUATE SUPPLIED WEIGHTS
Obtain the four processed task archives from the upstream CLCRN data link in
../reviewer_tools/README.md. Each task directory must contain trn.pkl, val.pkl,
test.pkl and position_info.pkl matching provenance/data_sha256.json.
From the repository root:
  python reviewer_tools/reviewer_verify.py evaluate --data-root PATH/WeatherBench_processed --run-id temperature_agf_2021 --device cpu --output reevaluation_temperature_agf_2021
Actual inference covers all 657 test windows. CUDA can be used with the separate
requirements-cu128.txt recipe. The original loader materializes all splits and
requires substantial memory. The 22.26 GiB weather arrays are external.

NEW TRAINING
From this directory:
  python code/run_revision.py --data-root PATH/WeatherBench_processed --task temperature --variant agf --seed 2021 --epochs 100 --batch-size 32 --eval-batch-size 32 --patience 0 --protocol zero_inclusive --device cuda:0 --threads 4 --output new_temperature_agf_2021
Use a fresh output directory. Standardization uses the training split only.
Independent training need not yield bit-identical parameters on other devices.

SUPPLEMENTARY EXPERIMENT ANALYSIS
To recompute full-test and horizon metrics, latitude-weighted sensitivity,
gate distributions, persistence and missing-node robustness from real data:
  python analysis/analyze_selected.py --data-root PATH/WeatherBench_processed --device cuda:0 --output supplementary_analysis
Run from this directory, using a fresh output directory. CPU is supported.
This processes all 46 selected checkpoints, all 657 full-test windows and a
deterministic 64-window robustness subset with mask seeds 11, 22 and 33.
It verifies package, run and data hashes before running models. Outputs include
fresh inference JSONs, 13 statistical CSVs and PNG diagnostic plots. This command
does not train models; training_performed in each inference record describes
the source checkpoint, while training_performed_in_this_command is false.
The portable orchestration is adapted for this release. Its numerical helpers
match the recorded analysis source; see provenance/distribution.json for the
exact adaptation and validation scope. It has not been rerun as a full 46-model
supplementary weather-data analysis during this repository cleanup.

PROVENANCE
code/ and selected_runs/ retain their original bytes. Original absolute paths
inside experiment records identify recorded executions, not required local
paths. provenance/ retains data fingerprints, environment and training-source
hashes. SHA256_MANIFEST.json describes the files in this distribution.
provenance/distribution.json distinguishes unchanged scientific records from
adapted experiment utilities. Older training queues, unrelated analysis,
obsolete source trees and internal workflow notes are omitted.
