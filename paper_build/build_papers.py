#!/usr/bin/env python3
"""Build the reviewer PDFs from source, sync current figures, and validate every page.

Requires Python 3.10+, PyMuPDF, and a TeX installation providing pdflatex.
Run from any directory: python paper_build/build_papers.py
No training checkpoints or meteorological datasets are needed for this step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

FIGURES = (
    "fig1_motivation.pdf", "fig2_architecture.pdf", "fig3_context_heatmap.pdf",
    "fig4_primary_replicates.pdf", "fig5_horizon_curves.pdf", "fig6_primary_bars.pdf",
    "fig7_missing_node_robustness.pdf", "fig8_gate_distribution.pdf",
)
PAPERS = (
    ("main", "Scientific_Reports_submission", "main.tex"),
    ("supplement", "Supplementary_Information", "supplement.tex"),
)
CITE = r"\\cite(?:p|t|alp|author|year)?(?:\[[^\]]*\])*\{([^}]+)\}"
BAD_REFERENCES = re.compile(
    r"(?:Citation|Reference)[^\n]*undefined|There were undefined references", re.I
)
RERUN = re.compile(
    r"Rerun to get cross-references right|Label\(s\) may have changed|"
    r"rerunfilecheck Warning: File .* has changed|Rerun to get /PageLabels entry",
    re.I,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def citation_check(root: Path) -> dict:
    source = root / "Scientific_Reports_submission/main.tex"
    text = source.read_text(encoding="utf-8")
    # Ignore comments, keeping escaped percent signs in ordinary prose.
    text = re.sub(r"(?<!\\)%[^\n]*", "", text)
    citations = re.findall(CITE, text)
    grouped = [c for c in citations if "," in c]
    adjacent = re.findall(CITE + r"[\s~,;]*" + CITE, text)
    entries = re.findall(r"\\bibitem\{([^}]+)\}", text)
    cited = {key.strip() for citation in citations for key in citation.split(",")}
    missing = sorted(cited - set(entries))
    uncited = sorted(set(entries) - cited)
    if grouped or adjacent or missing or uncited or len(entries) != len(set(entries)):
        raise RuntimeError(f"Citation validation failed: grouped={grouped}, adjacent={adjacent}, "
                           f"missing={missing}, uncited={uncited}, duplicate_bibitems="
                           f"{len(entries) != len(set(entries))}")
    old_audit = root / "validation/reference_retention.json"
    original_count = None
    if old_audit.exists():
        original = json.loads(old_audit.read_text(encoding="utf-8"))
        original_keys = {item["original_key"] for item in original["references"]}
        lost = original_keys - set(entries)
        if lost:
            raise RuntimeError(f"Original references are missing: {sorted(lost)}")
        original_count = len(original_keys)
    return {
        "bibliography_count": len(entries), "unique_cited_references": len(cited),
        "citation_occurrences": len(citations), "grouped_citations": len(grouped),
        "adjacent_citation_clusters": len(adjacent),
        "original_references_retained": original_count,
        "missing_bibitems": missing, "uncited_bibitems": uncited,
    }


def sync_figures(root: Path) -> list[dict]:
    target = root / "Scientific_Reports_submission/figures"
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for name in FIGURES:
        source = root / "figure_reproduction/figures" / name
        if not source.is_file() or not source.read_bytes().startswith(b"%PDF-"):
            raise RuntimeError(f"Missing or invalid source figure PDF: {source}")
        destination = target / name
        # Binary copying is essential: PDFs must never pass through text encoding.
        shutil.copyfile(source, destination)
        checksum = sha256(source)
        if sha256(destination) != checksum:
            raise RuntimeError(f"Figure binary copy mismatch: {name}")
        records.append({"file": name, "sha256": checksum, "copy_identical": True})
    return records


def compile_paper(root: Path, engine: str, folder: str, filename: str,
                  max_passes: int, logs: Path) -> dict:
    directory = root / folder
    tex = directory / filename
    if not tex.is_file():
        raise RuntimeError(f"Missing TeX source: {tex}")
    command = [engine, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", filename]
    complete = False
    for iteration in range(1, max_passes + 1):
        run = subprocess.run(command, cwd=directory, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=300)
        (logs / f"{tex.stem}_pass_{iteration}.txt").write_text(
            run.stdout + "\n" + run.stderr, encoding="utf-8")
        if run.returncode:
            raise RuntimeError(f"{filename}: TeX failed on pass {iteration}; see "
                               f"{logs / f'{tex.stem}_pass_{iteration}.txt'}")
        log = tex.with_suffix(".log").read_text(encoding="utf-8", errors="replace")
        if iteration >= 2 and not BAD_REFERENCES.search(log) and not RERUN.search(log):
            complete = True
            break
    if not complete:
        raise RuntimeError(f"{filename}: references did not stabilize in {max_passes} passes")
    return {
        "source": tex.relative_to(root).as_posix(), "source_sha256": sha256(tex),
        "passes": iteration, "undefined_references": False,
        "undefined_citations": False,
        "overfull_boxes": re.findall(r"Overfull[^\n]*", log),
        "log": tex.with_suffix(".log").relative_to(root).as_posix(),
    }


def validate_pdf(root: Path, relative: str, dpi: int, preview: Path | None) -> dict:
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("Install PyMuPDF first: python -m pip install PyMuPDF") from exc
    pdf = root / relative
    raw = pdf.read_bytes()
    if not raw.startswith(b"%PDF-") or b"%%EOF" not in raw[-1024:]:
        raise RuntimeError(f"Invalid PDF header or trailer: {relative}")
    page_results = []
    with pymupdf.open(pdf) as document:
        if document.is_repaired or document.is_encrypted or not document.page_count:
            raise RuntimeError(f"Invalid, repaired, encrypted or empty PDF: {relative}")
        for index, page in enumerate(document):
            pixmap = page.get_pixmap(dpi=dpi, alpha=False)
            if pixmap.width < 1 or pixmap.height < 1:
                raise RuntimeError(f"Empty rendered page {index+1}: {relative}")
            text = page.get_text()
            if "??" in text:
                raise RuntimeError(f"Unresolved reference marker on page {index+1}: {relative}")
            page_results.append({
                "page": index + 1, "width_px": pixmap.width, "height_px": pixmap.height,
                "text_characters": len(text), "rendered": True,
            })
            if preview:
                preview.mkdir(parents=True, exist_ok=True)
                pixmap.save(preview / f"{pdf.stem}_{index+1:02d}.png")
    return {"pdf": relative, "sha256": sha256(pdf), "size_bytes": len(raw),
            "pages": len(page_results), "all_pages_rendered": True,
            "render_dpi": dpi, "repaired": False, "page_details": page_results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--engine", default="pdflatex")
    parser.add_argument("--max-passes", type=int, default=4)
    parser.add_argument("--render-dpi", type=int, default=96)
    parser.add_argument("--save-page-previews", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    validation = root / "validation"
    logs = validation / "paper_build_logs"
    logs.mkdir(parents=True, exist_ok=True)
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "status": "failed",
              "python": platform.python_version(), "platform": platform.platform(),
              "build_script_sha256": sha256(Path(__file__))}
    try:
        if args.max_passes < 2 or not 36 <= args.render_dpi <= 300:
            raise RuntimeError("Use at least 2 TeX passes and a render DPI between 36 and 300")
        executable = shutil.which(args.engine)
        if executable is None:
            raise RuntimeError(f"{args.engine} is unavailable. Install TeX Live or MiKTeX and add it to PATH.")
        version = subprocess.run([executable, "--version"], capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=30)
        report["tex_engine"] = version.stdout.splitlines()[0]
        report["citations"] = citation_check(root)
        report["synchronized_figures"] = sync_figures(root)
        report["papers"] = []
        for name, folder, filename in PAPERS:
            print(f"Building {name}...", flush=True)
            result = compile_paper(root, executable, folder, filename, args.max_passes, logs)
            relative = f"{folder}/{Path(filename).stem}.pdf"
            preview = validation / "paper_page_previews" / name if args.save_page_previews else None
            result.update(validate_pdf(root, relative, args.render_dpi, preview))
            report["papers"].append(result)
            print(f"  {result['pages']} pages rendered; SHA256 {result['sha256']}", flush=True)
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = str(exc)
        print(f"ERROR: {exc}", file=sys.stderr)
    path = validation / "paper_validation.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Validation report: {path}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
