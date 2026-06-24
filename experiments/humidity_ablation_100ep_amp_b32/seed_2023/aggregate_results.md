# ASTTN Ablation Results (100 epochs)

## Variant Definitions

- `wo_adaptive_graph`: remove the adaptive graph branch and keep the improved training schedule.
- `wo_gated_fusion`: keep the adaptive graph branch but replace the learned gate with simple mean fusion.
- `full_model`: adaptive graph branch with learned gated fusion.

## humidity

| Variant | Best Val MAE | Test MAE | Test RMSE |
| --- | ---: | ---: | ---: |
| w/o Gated Fusion | 4.4817 | 4.5439 | 6.9905 |

Original reproduction reference:
- `MAE = 4.6773`
- `RMSE = 7.3087`

