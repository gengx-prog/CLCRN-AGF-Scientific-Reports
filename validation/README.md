# Verification evidence for the current release

These checks concern the current 46-run revision. The older 40-model historical recovery has a separate release and separate limitations.

| Evidence | Completed check | Scope |
| --- | --- | --- |
| [full46_gpu_inference.json](full46_gpu_inference.json) | All 46 selected models, 657 test windows each; 16 external data hashes verified | Actual weather test inference on an RTX 5090 Laptop GPU. Maximum absolute MAE difference 2.220446049250313e-16; RMSE difference 4.060862757171435e-10. No new training. |
| [clean_cpu_portable_quick.json](clean_cpu_portable_quick.json) | 408 frozen package files, 24 training source files, 46 records/checkpoints, 13 reconstructed CSVs, six synthetic-input forward passes | Fresh Windows CPU-only virtual environment using the final portable verifier. No meteorological data are used by this check. The earlier byte-exact check is retained in [clean_cpu_quick.json](clean_cpu_quick.json). |
| [clean_cpu_additional_checks.json](clean_cpu_additional_checks.json) | Failure-path regression tests; synthetic optimizer steps, save/reload, graph and metric formula checks | Exercises executable training code on synthetic small graphs; not a benchmark accuracy claim. |
| [paper_validation.json](paper_validation.json) | Main PDF 16 pages, supplement 14 pages; every page rendered; no undefined citations or overfull boxes | All 58 original references retained, 61 references in total; no grouped or adjacent citation clusters. |
| [paper_visual_review.json](paper_visual_review.json) | All-page contact sheets and detailed architecture/bibliography page review | Supplementary visual inspection of the compiled PDFs. |
| [Figure/table validation](../figure_reproduction/validation/validation_summary.json) | Eight figures, four numeric tables, 320 numeric-cell comparisons, 134 input-file hashes | Fresh figure/table rebuild in the clean CPU environment; no inference or training is performed by plotting. |
| [figure2_configuration_check.json](figure2_configuration_check.json) | Diagram settings checked against all 46 selected configurations | Corrected original-style vector architecture; gate operands and placement match implementation. |
| [reference_retention.json](reference_retention.json) | All 58 original bibliography entries mapped to the revised paper | Includes corrected bibliographic details and primary source links. |

The tested GPU environment was Windows 11, Intel Core Ultra 9 275HX, about 64 GB RAM, NVIDIA RTX 5090 Laptop GPU (24,463 MiB as reported by the driver), NVIDIA driver 610.88, Python 3.13.7 and PyTorch 2.10.0+cu128. The clean CPU environment uses PyTorch 2.10.0+cpu; its complete Python package list is [clean_cpu_freeze.txt](clean_cpu_freeze.txt). Training-time provenance remains separately preserved in [the frozen experiment environment](../revision_2026_10_08/provenance/environment.json). Current verification does not reconstruct the historical original-manuscript training environment.

The [GitHub Actions workflow](https://github.com/gengx-prog/CLCRN-AGF-Scientific-Reports/actions/workflows/reviewer.yml) performs CPU record/checkpoint checks, synthetic training, regression guards, and PDF parsing/rendering on Windows and Linux. Each run provides logs and reports. Consult its status for the exact commit being downloaded; these jobs do not download the weather arrays or run the full GPU evaluation.

Recorded-inference CSV regeneration is expected to be byte-identical on the tested Windows stack. The portable verifier also accepts a maximum 1e-12 absolute/relative floating-point tolerance for cross-platform numerical-library rounding while requiring the same schema, rows, strings and integer values. It reports whether every file was byte-identical. This does not relax any checkpoint, dataset or frozen-source SHA256 check.

Full inference requires the separately hosted, hash-matched weather data. Instructions and troubleshooting are in [reviewer_tools/README.md](../reviewer_tools/README.md). These records specify tested environments and checks; they are not a promise of identical runtime on every operating system, accelerator or package version.
