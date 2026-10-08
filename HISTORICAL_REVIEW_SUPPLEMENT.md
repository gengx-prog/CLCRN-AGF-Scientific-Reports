# Supplement for the original 16-page Scientific Reports manuscript

The [historical review supplement](https://github.com/gengx-prog/CLCRN-AGF-Scientific-Reports/releases/tag/v0.1-review-supplement-20261008) adds recovered materials corresponding to the original manuscript with SHA256 `4f806bf497e39611e442bd3515c4e32463d55f5fa3455d1ab63d46418d90cd1e`.

Download **CLCRN_AGF_original_SR_recovered_materials_2026_10_08.zip** from that release and start with its README. The ZIP is self-contained for inspecting evidence and rebuilding the historical tables. Model inference additionally requires the external data documented inside it. This is an addition to the old review archive; it does not replace the later 46-run revision or substitute its results into the old paper.

The historical supplement includes:

- The exact original PDF and its matching, successfully compiled LaTeX source ZIP.
- 73 related run groups, including 69 selected checkpoints and all 1,018 recovered selected/candidate checkpoint files, with logs, configurations and result records.
- The missing source dependencies, all 63 recovered raw figure/table files, and historical figure generators.
- Portable file checking, CPU software checks, table reconstruction and 40-model checkpoint replay commands.
- Exact data/checkpoint hashes, measured Windows/GPU/Python/dependency versions, and validation receipts.

All 40 primary saved checkpoints were re-evaluated on all 657 test windows each. Their full-precision MAE and RMSE matched the historical records exactly on the recorded machine. The 16 local external data hashes matched; 26 CPU software checks passed. Table 1, Table 3 and Table 4 main printed values were reconstructed, with two small printed p-value differences explicitly reported.

Four Table 2 single-run checkpoint files remain unrecovered despite checking project copies, ZIP listings and same-epoch candidates. Their logs record that models were loaded at the time, but their test results cannot now be replayed from the recovered materials. An immutable original training source/environment snapshot was also not recovered. The package documents a current compatibility evaluation environment.

The original paper's gate description, mixed figure sources, sampled gate statistics, training/evaluation objective difference and unsupported external baseline provenance remain listed in `ISSUES_OLD_MANUSCRIPT.md`. File recovery does not by itself correct these manuscript issues. Third-party processed weather arrays are linked separately with exact local checksums; upstream download contents have not been freshly compared byte for byte with the author's inputs.

The earlier `v0.1-review` release and the later `v0.2-retraining-20261008` release are preserved. Private journal correspondence is not part of this supplement.
