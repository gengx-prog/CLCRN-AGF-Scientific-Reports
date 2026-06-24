from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts"
OUTPUT_PDF = OUTPUT_DIR / "clcrn_result_summary.pdf"


def extract_block(path: Path, start_marker: str) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.lstrip().startswith(start_marker):
            start = idx
            base_indent = len(line) - len(line.lstrip())
            break
    if start is None:
        raise ValueError(f"Could not find marker {start_marker!r} in {path}")

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        line = lines[idx]
        if not line.strip():
            continue
        current_indent = len(line) - len(line.lstrip())
        if current_indent <= base_indent and not line.lstrip().startswith(("@", "#")):
            end = idx
            break
    return "\n".join(lines[start:end]).rstrip()


def build_sections() -> list[tuple[str, str, bool]]:
    mae_code = extract_block(ROOT / "model" / "loss.py", "def masked_mae_loss")

    scaler_code = extract_block(ROOT / "lib" / "utils.py", "class StandardScaler")
    dataloader_code = "\n".join(
        [
            "scaler = [StandardScaler(mean=train_numpy[..., i].mean(), std=train_numpy[..., i].std()) for i in range(feature_len)]",
            "",
            "for i in range(feature_len):",
            "    self.x[..., i] = scaler[i].transform(self.x[..., i])",
            "    self.y[..., i] = scaler[i].transform(self.y[..., i])",
        ]
    )
    inverse_code = extract_block(ROOT / "supervisor.py", "def _compute_loss")

    pred_vs_gt = "\n".join(
        [
            "Source checkpoint:",
            r"experiments\\tmp_baseline_5ep\\humidity\\CLCRN_humidity\\saved_model\\epo4.tar",
            "",
            "Extraction setting:",
            "test set, first batch, first sample, first node, first 5 horizons",
            "",
            "pred = [103.623810, 104.355133, 104.894066, 105.159927, 105.180054]",
            "gt   = [102.808792, 104.675339, 105.581429, 106.620415, 106.125084]",
            "",
            "Environment snapshot:",
            "python --version -> Python 3.13.7",
            "pip list | findstr torch -> torch 2.10.0+cu128",
        ]
    )

    intro = "\n".join(
        [
            "CLCRN Result Summary",
            "",
            "This PDF organizes the three items requested locally:",
            "1. MAE computation code",
            "2. Normalization / inverse-transform code",
            "3. A real prediction-vs-ground-truth numeric sample",
        ]
    )

    return [
        ("Overview", intro, False),
        ("MAE Code", mae_code, True),
        ("Normalization Code", "\n\n".join([scaler_code, dataloader_code, inverse_code]), True),
        ("Pred vs GT", pred_vs_gt, True),
    ]


def add_text_page(pdf: PdfPages, title: str, body: str, is_code: bool, zh_font: FontProperties, mono_font: FontProperties) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    title_font = zh_font.copy()
    title_font.set_size(18)
    body_font = mono_font if is_code else zh_font
    body_font = body_font.copy()
    body_font.set_size(10.5 if is_code else 11.5)

    ax.text(0.06, 0.96, title, fontproperties=title_font, va="top", ha="left")

    if is_code:
        rendered = body
        family = "monospace"
    else:
        wrapped_lines = []
        for paragraph in body.splitlines():
            if not paragraph.strip():
                wrapped_lines.append("")
                continue
            wrapped_lines.extend(textwrap.wrap(paragraph, width=52, break_long_words=False, break_on_hyphens=False))
        rendered = "\n".join(wrapped_lines)
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
    mono_font = FontProperties(fname=r"C:\Windows\Fonts\consola.ttf") if Path(r"C:\Windows\Fonts\consola.ttf").exists() else zh_font

    sections = build_sections()
    with PdfPages(OUTPUT_PDF) as pdf:
        for title, body, is_code in sections:
            add_text_page(pdf, title, body, is_code, zh_font, mono_font)

    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
