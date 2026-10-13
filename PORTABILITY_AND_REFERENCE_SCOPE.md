# Execution requirements and portability

Updated 13 October 2026. This repository contains the latest experiment code, 46 selected checkpoints, training records, statistical outputs and verification tools. Choose the task you need; the supplied tools do not require running every check in sequence.

## Requirements by task

| Task | Required inputs/software | Weather download / GPU |
| --- | --- | --- |
| Inspect recorded CSV/JSON results | A text or table reader | Neither |
| Rebuild 13 statistical CSVs with `offline` | Supplied records and the numerical dependencies | Neither |
| Run the CPU demo or `quick` | Complete experiment package and the pinned CPU dependencies | Neither |
| Check data hashes with `check-data` | Processed data for the selected tasks and Python | Data only |
| Evaluate supplied checkpoints | Matching processed data, selected weights, compatible PyTorch and sufficient memory | Data; GPU optional |
| Train new models | Processed data, training implementation, dependencies and suitable compute | Data; GPU optional |

The [command guide](reviewer_tools/README.md) provides setup and execution examples. The quick demo uses generated synthetic data; passing it does not measure weather accuracy. Rebuilding statistical CSVs checks recorded inference outputs without performing new inference or training.

The CPU and CUDA dependency recipes target CPython 3.13.7 on Windows/Linux x86-64. See [local verification evidence](validation/README.md) and [workflow results](https://github.com/gengx-prog/CLCRN-AGF-Scientific-Reports/actions/workflows/reviewer.yml) for the exact tested environments and commits. Other operating systems, architectures and accelerator backends are not covered by those recipes.

## Data, hardware and numerical limits

The external processed dataset consists of 16 files totaling approximately 22.26 GiB. Fixed-checkpoint evaluation requires the file sizes and SHA256 values in [`data_sha256.json`](experiments/provenance/data_sha256.json), including the training split used for standardization. The loader materializes training, validation and test arrays and the evaluator uses batch size 32. Sufficient system RAM and, for GPU execution, VRAM are therefore necessary.

The recorded Windows machine, RTX 5090 Laptop GPU, approximately 64 GB RAM and complete environment snapshot describe execution of the supplied experiments. They are not a universal hardware minimum or a required computer model. Current entry points accept your own data/output paths; absolute paths in execution metadata do not have to exist elsewhere.

Comparable numerical experiments require matching data splits, preprocessing, model definitions, checkpoint-selection rules and metric support. New implementations or differently processed data can support other experiments, but their scores are not directly interchangeable with these reference records.

Statistical CSV reconstruction allows relative and absolute floating-point tolerances of `1e-12`, while preserving schemas, row order, text and integers exactly. Full-test evaluation compares MAE/RMSE with an absolute tolerance of `2e-5` and records signed differences. A different backend or dependency version can affect rounding; the tools retain discrepancies and return a failure instead of changing reference values. Identical seeds do not establish bit-identical newly trained weights across all devices.

## Diagnose failures by stage

| Symptom | What to check |
| --- | --- |
| Package unavailable, binary/import error | Python version, OS/architecture, dependency recipe and package consistency (`python -m pip check`) |
| CUDA unavailable or driver/kernel error | GPU/driver compatibility and installed PyTorch build; use the CPU recipe if appropriate |
| Process killed, `MemoryError` or out-of-memory | Available RAM/VRAM, materialized dataset arrays, fixed evaluation batch and graph intermediates |
| Missing file or SHA256 mismatch | Download/extraction completeness, dataset version and repository integrity |
| External data cannot be accessed | Upstream provider access and availability; see the dataset links in the command guide |
| Existing output directory | Choose a new directory to preserve prior results |
| Inference completes but metric validation fails | Saved signed differences, data hashes, model/runtime versions and full logs |
| `--resume` rejects a moved run | Frozen resume checks compare configuration, source and environment provenance; new training/evaluation has different requirements |

A failed command needs its own diagnosis; neither environment incompatibility nor an implementation defect should be assumed without the error evidence. Retain the command, commit/release, first exception and complete traceback, OS/architecture, Python/package versions, device/driver, available memory and relevant data hashes. Remove credentials and private paths before sharing logs.
