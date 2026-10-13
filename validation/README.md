# Experiment verification

The [CPU workflow](https://github.com/gengx-prog/CLCRN-AGF-Scientific-Reports/actions/workflows/reviewer.yml) checks the distributed experiment package on Windows and Linux. Check the workflow result for the commit you use.

## Latest package

The local verification records in `experiments_20261013/` cover the retained package: exact frozen-source and checkpoint hashes, all 46 training records, 13 reconstructed statistical CSVs, six synthetic forwards and four tiny CPU train/save/reload runs. Synthetic checks establish software execution, not weather prediction accuracy. The reports record the tested source hashes, environment and execution time.

## Weather-data evaluation of the selected checkpoints

[Full evaluation evidence](full46_gpu_inference.json) records the earlier successful evaluation of the same 46 selected checkpoints on all 657 test windows, after checking all 16 external data files. Its timestamp and package count describe that execution; it is not a new evaluation of this package layout. Maximum absolute MAE and RMSE differences from the reference records were approximately `2.22e-16` and `4.06e-10` respectively.

The frozen training sources, selected checkpoints, configurations, training histories and inference records are preserved byte for byte. Removing obsolete material does not regenerate or change experimental scores. The latest package check verifies those retained bytes.

The full evaluation used Windows 11, Python 3.13.7 and PyTorch 2.10.0+cu128. The CPU recipe uses PyTorch 2.10.0+cpu. These are recorded environments, not a requirement to use the original computer. See the [execution requirements](../PORTABILITY_AND_REFERENCE_SCOPE.md) and [command guide](../reviewer_tools/README.md) to run your own checks.
