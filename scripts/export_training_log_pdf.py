from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts"
OUTPUT_PDF = OUTPUT_DIR / "clcrn_training_log_and_save_rule.pdf"


def extract_lines(path: Path, keywords: list[str]) -> str:
    matched = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if any(keyword in line for keyword in keywords):
            matched.append(line)
    return "\n".join(matched)


def extract_block(path: Path, start_marker: str) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    base_indent = 0
    for idx, line in enumerate(lines):
        if line.lstrip().startswith(start_marker):
            start = idx
            base_indent = len(line) - len(line.lstrip())
            break
    if start is None:
        raise ValueError(f"Could not find {start_marker!r} in {path}")

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        line = lines[idx]
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= base_indent and not line.lstrip().startswith(("@", "#")):
            end = idx
            break
    return "\n".join(lines[start:end]).rstrip()


def add_page(pdf: PdfPages, title: str, body: str, code: bool, zh_font: FontProperties, mono_font: FontProperties) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    title_font = zh_font.copy()
    title_font.set_size(18)
    body_font = (mono_font if code else zh_font).copy()
    body_font.set_size(10.2 if code else 11.5)

    ax.text(0.06, 0.96, title, fontproperties=title_font, va="top", ha="left")

    if code:
        rendered = body
        family = "monospace"
    else:
        wrapped = []
        for paragraph in body.splitlines():
            if not paragraph.strip():
                wrapped.append("")
                continue
            wrapped.extend(textwrap.wrap(paragraph, width=52, break_long_words=False, break_on_hyphens=False))
        rendered = "\n".join(wrapped)
        family = None

    ax.text(
        0.06,
        0.92,
        rendered,
        fontproperties=body_font,
        family=family,
        va="top",
        ha="left",
        linespacing=1.45,
    )
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zh_font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
    mono_path = Path(r"C:\Windows\Fonts\consola.ttf")
    mono_font = FontProperties(fname=str(mono_path)) if mono_path.exists() else zh_font

    tmp_log = ROOT / "experiments" / "tmp_baseline_5ep" / "humidity" / "CLCRN_humidity" / "info.log"
    save_rule = extract_block(ROOT / "supervisor.py", "def save_model")
    tmp_log_excerpt = extract_lines(
        tmp_log,
        [
            "Epoch [1/5]",
            "Epoch [2/5]",
            "Epoch [3/5]",
            "Epoch [4/5]",
            "Epoch [5/5]",
            "Saved model at 1",
            "Saved model at 2",
            "Saved model at 3",
            "Saved model at 4",
            "Final summary:",
        ],
    )

    paper_run_evidence = "\n".join(
        [
            "paper_run evidence:",
            "temperature info.log records: Saved model at 56 / saving to models/epo56.tar",
            "humidity info.log records: Saved model at 29",
            "component_of_wind info.log records: Saved model at 27",
            "cloud_cover info.log records: Saved model at 25",
            "",
            "Current directory state under weatherbench_clcrn_paper_run/<dataset>/CLCRN_*:",
            "only info.log, model_param.json, model_param.yaml, summary.json",
            "no saved_model directory is present now",
            "",
            "Interpretation:",
            "the save naming rule itself is normal;",
            "the paper_run checkpoints were likely cleaned up, moved, or not retained after the run.",
        ]
    )

    verdict = "\n".join(
        [
            "Key judgment:",
            "1. tmp_baseline_5ep really was a 5-epoch run, not a broken save strategy.",
            "2. saved_model naming is Path(log_dir/experiment_name/saved_model/epo{epoch}.tar).",
            "3. Only epochs with improved validation MAE are saved, so no epo5.tar is expected here.",
            "4. The separate paper_run anomaly is missing retained checkpoint files, not a naming-rule bug.",
        ]
    )

    with PdfPages(OUTPUT_PDF) as pdf:
        add_page(pdf, "Overview", verdict, False, zh_font, mono_font)
        add_page(pdf, "Training Log Excerpt", tmp_log_excerpt, True, zh_font, mono_font)
        add_page(pdf, "Save Rule", save_rule, True, zh_font, mono_font)
        add_page(pdf, "Paper Run Evidence", paper_run_evidence, False, zh_font, mono_font)

    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
