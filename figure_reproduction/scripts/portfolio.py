"""Assemble vector-preserving figure/table and old/new appearance previews.

Usage: python portfolio.py [--output-root /path/to/figure-package]
Only files under output-root/preview are written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pymupdf as fitz
from matplotlib import font_manager

FIGURES = {
    1: ("fig1_motivation", "Spatial geometry in weather modelling"),
    2: ("fig2_architecture", "Implementation-aligned model architecture"),
    3: ("fig3_context_heatmap", "Current-split forecasting comparison"),
    4: ("fig4_primary_replicates", "Five-seed primary comparison"),
    5: ("fig5_horizon_curves", "Forecast errors across horizons"),
    6: ("fig6_primary_bars", "Comparison under one training protocol"),
    7: ("fig7_missing_node_robustness", "Sensitivity to missing input nodes"),
    8: ("fig8_gate_distribution", "Distribution of learned gate activations"),
}
OLD_FILES = {
    3: "fig_A1_main_baselines.png",
    4: "fig_primary_replicate_comparison.png",
    5: "fig_A2_horizon_curves.png",
    6: "fig_B2_schedule_decomp_mae.png",
    7: "fig_missing_nodes_mae.png",
    8: "fig_gate_distribution.png",
}
NOTES = {
    3: "The annotated heatmap style is retained. Rows now contain persistence, control and AGF, the three methods evaluated on the current split; unsupported historical baseline rows are removed.",
    4: "The four panels, blue/orange seed points, black mean diamonds and sample-SD bars are retained. Every point and mean-change label uses the current five-seed data.",
    5: "The four-panel layout, task colours and three curve styles are retained. Curves now summarize the five current AGF seeds with sample-SD bands on all 657 test windows.",
    6: "Grouped bars, control hatching and value labels are retained. The four task panels use valid physical scales, and the comparison no longer claims to isolate a training-schedule effect.",
    7: "The grey background, white grid, colours and line patterns are retained. Relative MAE changes use the same 64 test windows, averaging masks within each seed before calculating seed variation.",
    8: "The horizontal distributions, white IQR boxes and mean diamonds are retained. Four tasks use full-test, all-channel gate counts; histogram-based percentiles are explicitly approximate.",
}
FONT_FILE = font_manager.findfont("DejaVu Sans")
FONT_BOLD = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
FONT = fitz.Font(fontfile=FONT_FILE)
INK = (.13, .18, .23)
MUTED = (.36, .40, .44)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def page_fonts(page):
    page.insert_font(fontname="body", fontfile=FONT_FILE)
    page.insert_font(fontname="bold", fontfile=FONT_BOLD)


def wrapped(text, width, fontsize):
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            trial = f"{line} {word}".strip()
            if FONT.text_length(trial, fontsize=fontsize) > width and line:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    return lines


def write_lines(page, lines, x, top, size=12, leading=17, color=INK, bold=False):
    for i, line in enumerate(lines):
        page.insert_text((x, top + size + i * leading), line, fontsize=size,
                         fontname="bold" if bold else "body", color=color)


def center_text(page, text, rect, size=16, bold=False, color=INK):
    ret = page.insert_textbox(rect, text, fontsize=size, fontname="bold" if bold else "body", align=1, color=color)
    if ret < 0:
        raise ValueError(f"Text exceeds its box: {text!r}")


def sources(root):
    captions = {}
    hashes = []
    for name in ["schematics", "comparisons", "diagnostics"]:
        path = root / "metadata" / f"{name}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        hashes.append({"file": path.relative_to(root).as_posix(), "sha256": sha(path)})
        for key, val in payload["figures"].items():
            number = int(key.split("_")[0].removeprefix("fig"))
            captions[number] = val["caption"]
    assert sorted(captions) == list(range(1, 9))
    return captions, hashes


def make_portfolio(root, dest, captions):
    doc = fitz.open()
    for number, (stem, title) in FIGURES.items():
        with fitz.open(root / "figures" / f"{stem}.pdf") as source:
            ratio = source[0].rect.width / source[0].rect.height
            width, height = (960, 740) if ratio >= 1.7 else (820, 970)
            margin = 42
            caption = f"Figure {number}. {captions[number]}"
            lines = wrapped(caption, width - 2 * margin, 12)
            caption_height = len(lines) * 17 + 4
            caption_top = height - 42 - caption_height
            page = doc.new_page(width=width, height=height)
            page_fonts(page)
            write_lines(page, [f"Figure {number}  |  {title}"], margin, 23, size=18, leading=23, bold=True)
            page.draw_line((margin, 63), (width - margin, 63), color=(.80, .83, .86), width=.65)
            page.show_pdf_page(fitz.Rect(margin, 78, width - margin, caption_top - 20), source, 0)
            write_lines(page, lines, margin, caption_top, size=12, leading=17)
            page.insert_text((margin, height - 15), "CLCRN-AGF · revised data with preserved visual style", fontsize=9, fontname="body", color=MUTED)
            page.insert_text((width - 52, height - 15), str(number), fontsize=9, fontname="body", color=MUTED)
    table_path = root / "tables" / "tables_preview.pdf"
    with fitz.open(table_path) as tables:
        assert len(tables) == 2, f"Expected two table pages, found {len(tables)}"
        doc.insert_pdf(tables)
    doc.set_metadata({"title": "CLCRN-AGF figures and tables with revised data", "subject": "Vector-preserving preview of Figures 1–8 and Tables 1–4"})
    doc.subset_fonts()
    path = dest / "figures_and_tables_preview.pdf"
    doc.save(path, garbage=4, deflate=True)
    doc.close()
    return path


def make_comparisons(root, dest):
    doc = fitz.open()
    for number, oldname in OLD_FILES.items():
        stem, title = FIGURES[number]
        with fitz.open(root / "figures" / f"{stem}.pdf") as new:
            ratio = new[0].rect.width / new[0].rect.height
            width = 1450
            height = 820 if ratio < 1.7 else 650
            page = doc.new_page(width=width, height=height)
            page_fonts(page)
            write_lines(page, [f"Figure {number}  |  {title}"], 40, 20, size=19, bold=True)
            center_text(page, "Original appearance", fitz.Rect(40, 68, 710, 100), size=19, bold=True)
            center_text(page, "Updated data, preserved style", fitz.Rect(750, 68, 1410, 100), size=19, bold=True)
            note_lines = wrapped(NOTES[number], width - 80, 13)
            caption_top = height - 45 - (len(note_lines) + 2) * 18
            area_left = fitz.Rect(40, 110, 705, caption_top - 15)
            area_right = fitz.Rect(750, 110, 1410, caption_top - 15)
            page.insert_image(area_left, filename=str(root / "reference_previews" / oldname), keep_proportion=True)
            page.show_pdf_page(area_right, new, 0)
            page.draw_line((728, 68), (728, caption_top - 20), color=(.82, .85, .88), width=.7)
            write_lines(page, note_lines, 40, caption_top, size=13, leading=18)
            write_lines(page, ["Left: historical visual reference. Right: figures generated from the current experimental evidence."], 40, height - 58, size=11, color=MUTED)
            page.insert_text((40, height - 20), f"Appearance comparison · {number - 2} / 6", fontsize=10, fontname="body", color=MUTED)
    doc.set_metadata({"title": "CLCRN-AGF original and updated figure appearance", "subject": "Historical styling alongside current experimental evidence"})
    doc.subset_fonts()
    path = dest / "old_vs_new_results.pdf"
    doc.save(path, garbage=4, deflate=True)
    doc.close()
    return path


def make_contact(source_path, dest_path, ncols, nrows, heading, numbers=None, width=1600):
    contact = fitz.open()
    with fitz.open(source_path) as source:
        numbers = list(range(len(source))) if numbers is None else numbers
        cell_width = 490
        cell_height = 510 if "figures_and_tables" in source_path.name else 310
        page_width = 40 + ncols * cell_width
        heading_lines = wrapped(heading, page_width - 40, 20)
        header_height = 35 + 25 * len(heading_lines)
        page = contact.new_page(width=page_width, height=header_height + nrows * cell_height)
        page_fonts(page)
        write_lines(page, heading_lines, 20, 13, size=20, leading=25, bold=True)
        for i, number in enumerate(numbers):
            col, row = i % ncols, i // ncols
            x, y = 20 + col * cell_width, header_height + row * cell_height
            rect = fitz.Rect(x, y, x + cell_width - 12, y + cell_height - 12)
            page.show_pdf_page(rect, source, number)
        scale = width / page.rect.width
        page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).save(dest_path)
    contact.close()


def inspect_pdf(path):
    result = []
    with fitz.open(path) as doc:
        for idx, page in enumerate(doc):
            overflow = []
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        bbox = fitz.Rect(span["bbox"])
                        if not page.rect.contains(bbox):
                            overflow.append(span["text"])
            assert not overflow, f"Text outside page {idx+1} in {path.name}: {overflow}"
            result.append({"page": idx + 1, "width": page.rect.width, "height": page.rect.height,
                           "raster_image_count": len(page.get_images()), "outside_page_text": overflow})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.output_root.resolve()
    dest = root / "preview"
    dest.mkdir(parents=True, exist_ok=True)
    captions, metadata_hashes = sources(root)
    portfolio = make_portfolio(root, dest, captions)
    comparisons = make_comparisons(root, dest)
    make_contact(portfolio, dest / "figures_contact.png", 2, 4, "Figures 1–8 · revised data, preserved visual style", list(range(8)))
    make_contact(portfolio, dest / "tables_contact.png", 2, 1, "Tables 1–4 · revised experimental results", [8, 9])
    make_contact(comparisons, dest / "old_vs_new_contact.png", 2, 3, "Original appearance  /  Updated data, preserved style", width=2200)
    with fitz.open(portfolio) as doc:
        for index in [1, 4, 6, 7]:
            doc[index].get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False).save(dest / f"portfolio_page_{index + 1:02d}.png")
    with fitz.open(comparisons) as doc:
        for index in [2, 5]:
            doc[index].get_pixmap(matrix=fitz.Matrix(1.15, 1.15), alpha=False).save(dest / f"comparison_page_{index + 1:02d}.png")
    manifest = {"caption_metadata": metadata_hashes,
                "vector_preservation": "Source PDFs embedded with show_pdf_page, except old PNG visual references. Figure 1 intentionally retains the original raster. Tables appended as original PDF pages.",
                "portfolio_validation": inspect_pdf(portfolio), "comparison_validation": inspect_pdf(comparisons)}
    manifest["artifacts"] = [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(dest.iterdir()) if p.suffix in [".pdf", ".png"]]
    manifest["script_sha256"] = sha(__file__)
    (dest / "preview_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "portfolio_pages": 10, "comparison_pages": 6, "output": str(dest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
