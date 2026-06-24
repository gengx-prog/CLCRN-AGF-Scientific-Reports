# Scientific Reports revision status

## Completed in the first integrity-focused revision

- Reframed the manuscript as an extension of Lin et al. (AAAI 2022), not as
  the introduction of CLCRN or CLConv.
- Added the original CLCRN citation and DOI.
- Renamed the proposed extension to CLCRN-AGF.
- Removed the untraceable seven-baseline leaderboard and the claim that
  published CLCRN values were newly generated results.
- Replaced the primary results with paired five-seed CLCRN-AGF versus
  schedule-matched control results.
- Added paired confidence intervals and exact sign-flip tests.
- Corrected WeatherBench variables, pressure/height descriptions, units,
  chronological split, sample counts, temporal window, and sampling interval.
- Rewrote the AGF equations to match `adaptive_attention.py`.
- Corrected training hyperparameters to match the saved experiment configs.
- Corrected the complexity statement to acknowledge dense quadratic
  intermediate matrices.
- Removed the 20-epoch ablation as evidence for component causality.
- Added persistence, published CLCRN, reproduced CLCRN, and efficiency
  context with explicit provenance.
- Reduced the main manuscript to five display items and compiled without
  LaTeX warnings.

## Experiments still required before a strong submission

1. Run full-budget, at least five-seed ablations for:
   - adaptive mixing without the learned gate;
   - gated fusion without adaptive mixing;
   - the complete CLCRN-AGF module;
   - the schedule-matched CLCRN control.
2. Train at least one modern graph baseline and one modern non-graph
   time-series baseline with the identical split and optimization protocol.
3. Add a climatology baseline and latitude-weighted WeatherBench metrics,
   including ACC where scientifically appropriate.
4. Increase paired seeds beyond five if formal significance at the 0.05 level
   is required; with five pairs, the minimum exact two-sided sign-flip
   p-value is 0.0625.
5. Validate learned edges against spatial distance or meteorological
   diagnostics before assigning physical meaning to them.
6. Deposit a review archive containing preprocessing metadata, exact configs,
   per-seed summaries, and environment information.
