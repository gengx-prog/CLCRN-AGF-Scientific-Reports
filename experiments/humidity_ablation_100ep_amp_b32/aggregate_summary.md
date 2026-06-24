# Relative-humidity 100-epoch component ablation completion

Variant: CLCRN-AGF without learned gated fusion (`wo_gated_fusion`; adaptive graph branch retained, graph/residual branches combined by mean fusion).

Protocol:

- Dataset/task: WeatherBench relative humidity
- Seeds: 2023, 2024, 2025
- Epoch budget: 100
- Batch size: 32
- Test batch size: 128
- Training precision: PyTorch automatic mixed precision
- Checkpoint selection: lowest validation MAE
- Test evaluator: corrected zero-inclusive evaluator

AMP note: standard-precision attempts for this mean-fusion humidity ablation repeatedly failed with CUDA allocator exhaustion on the available Windows GPU. The run was completed with AMP while preserving batch size 32 and the same validation/test evaluator.

## Per-seed test results

| Seed | Best epoch | Best val MAE | Test MAE | Test RMSE | Test MAPE |
|---:|---:|---:|---:|---:|---:|
| 2023 | 91 | 4.481690 | 4.543855 | 6.990528 | 0.070154 |
| 2024 | 90 | 4.500224 | 4.558073 | 7.105721 | 0.069137 |
| 2025 | 84 | 4.556149 | 4.622538 | 7.133841 | 0.072852 |

## Aggregate results

Values are mean ± sample standard deviation across the three seeds.

| Metric | Mean | Sample std |
|---|---:|---:|
| MAE | 4.574822 | 0.041930 |
| RMSE | 7.076697 | 0.075937 |
| MAPE | 0.070714 | 0.001920 |

