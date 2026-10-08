Build the manuscript and supplementary PDFs

Requirements
- Python 3.10 or newer.
- PyMuPDF: python -m pip install PyMuPDF==1.28.2
- TeX Live or MiKTeX with pdflatex available on PATH. The sources use standard
  LaTeX packages including graphics, hyperref, xurl, microtype, booktabs,
  ragged2e, placeins, multirow, needspace, caption and the bundled journal class.

From the repository root:
  python paper_build/build_papers.py

Optional page images for visual checking:
  python paper_build/build_papers.py --save-page-previews

The command also works from another current directory when the path to the
script is given explicitly. Use --engine with a pdflatex executable path if
it is not on PATH. Use --repo-root to validate a different extracted copy.

The script copies the eight current figure PDFs from figure_reproduction/
figures into the manuscript using binary-safe file copying, checking SHA256
equality. It does not modify the frozen revision_2026_10_08 experiment files.
It then compiles the main and supplementary TeX files until references are
stable, rejects missing/uncited/grouped main-paper citations, checks that all
58 original references remain, and renders every page with PyMuPDF.

Outputs
- Scientific_Reports_submission/main.pdf
- Supplementary_Information/supplement.pdf
- validation/paper_validation.json
- validation/paper_build_logs/ (compiler output)
- validation/paper_page_previews/ (only with --save-page-previews)

No GPU, model weights, meteorological datasets or training run is required
for paper compilation. To regenerate the plotted results first, use the
separate figure_reproduction workflow before running this script.

Compiled PDF bytes can differ between TeX installations or build dates.
The recorded SHA256 identifies this particular output; it is not a claim
that all independent PDF builds are byte-identical.
