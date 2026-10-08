"""Rebuild four manuscript tables using the 46-run revision data.

Usage:
  python scripts/tables.py --data-root PATH_TO_REVISION_DATA --output-root OUTPUT

Requires numpy and scipy. PDF compilation uses an existing pdflatex installation;
no packages or environments are installed by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy import stats


TASKS = ["temperature", "humidity", "component_of_wind", "cloud_cover"]
LABELS = dict(zip(TASKS, ["Temperature", "Relative humidity", "Surface wind", "Cloud cover"]))
VARIANTS = ["w/o adaptive graph", "Mean fusion", "Full CLCRN-AGF"]
KINDS = dict(zip(VARIANTS, ["control", "mean_fusion", "agf"]))
TABLE_NAMES = [
    "table1_primary_statistics.csv", "table2_current_context.csv",
    "table3_ablation_statistics.csv", "table4_gate_statistics.csv",
]
UNITS = (r"Errors are in K for temperature, percentage points for relative humidity, "
         r"m\,s$^{-1}$ for surface wind, and fractions for cloud cover.")


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value, task="temperature", signed=False):
    decimals = 4 if task == "cloud_cover" else 3
    return format(float(value), ("+" if signed else "") + f".{decimals}f")


def pformat(value):
    value = float(value)
    if value >= 0.001:
        return f"{value:.4f}"
    exponent = int(math.floor(math.log10(value)))
    return rf"${value / 10 ** exponent:.2f}\times10^{{{exponent}}}$"


def bh(values):
    vals = np.asarray(values)
    order = np.argsort(vals)
    ordered = vals[order] * len(vals) / np.arange(1, len(vals) + 1)
    adjusted = np.minimum(1.0, np.minimum.accumulate(ordered[::-1])[::-1])
    answer = np.empty_like(vals)
    answer[order] = adjusted
    return answer


def validate(data: Path, tables):
    checks = []
    used = set(TABLE_NAMES + ["primary_per_seed.csv", "ablation_per_seed.csv", "gate_per_seed.csv"])

    def same(label, actual, expected, tolerance=2e-11):
        error = abs(float(actual) - float(expected))
        if error > tolerance or not math.isfinite(error):
            raise ValueError(f"{label}: {actual} != {expected} (difference {error})")
        checks.append({"check": label, "absolute_error": error, "tolerance": tolerance})

    primary = read_csv(data / "primary_per_seed.csv")
    assert len(primary) == 40
    assert len({(r["dataset"], r["kind"], r["seed"]) for r in primary}) == 40
    for row in primary:
        name = f'inference_{row["kind"]}_{row["dataset"]}_{row["seed"]}.json'
        used.add(name)
        inference = json.loads((data / name).read_text(encoding="utf-8"))
        assert inference["training_precision"] == "FP32"
        assert inference["epochs_completed"] == 100
        assert inference["training_protocol"] == "zero_inclusive"
        assert inference["checkpoint"]["sha256"] == row["checkpoint_sha256"]
        for metric in ["mae", "rmse"]:
            same(f"per_seed:{name}:{metric}", row[metric], inference["metrics"][metric])

    p_values, permutation_p_values = [], []
    for row in tables[0]:
        task, metric = row["dataset"], row["metric"]
        prefix = f"table1:{task}:{metric}"
        arrays = {}
        for kind in ["control", "agf"]:
            subset = [r for r in primary if r["dataset"] == task and r["kind"] == kind]
            assert sorted(int(r["seed"]) for r in subset) == list(range(2021, 2026))
            arr = arrays[kind] = np.array([float(r[metric]) for r in subset])
            same(f"{prefix}:{kind}_mean", arr.mean(), row[f"{kind}_mean"])
            same(f"{prefix}:{kind}_sd", arr.std(ddof=1), row[f"{kind}_sd"])
            same(f"{prefix}:n_{kind}", len(arr), row[f"n_{kind}"])
        a, c = arrays["agf"], arrays["control"]
        delta = a.mean() - c.mean()
        variances = np.array([a.var(ddof=1) / len(a), c.var(ddof=1) / len(c)])
        df = variances.sum() ** 2 / ((variances ** 2) / [len(a) - 1, len(c) - 1]).sum()
        halfwidth = stats.t.ppf(0.975, df) * np.sqrt(variances.sum())
        p_value = stats.ttest_ind(a, c, equal_var=False).pvalue
        all_values = np.concatenate([a, c])
        permutations = []
        for indices in itertools.combinations(range(10), 5):
            mask = np.zeros(10, dtype=bool)
            mask[list(indices)] = True
            permutations.append(abs(all_values[mask].mean() - all_values[~mask].mean()))
        permutation_p = np.mean(np.asarray(permutations) >= abs(delta) - 1e-12)
        expected = {
            "difference": delta, "ci95_low": delta - halfwidth, "ci95_high": delta + halfwidth,
            "welch_df": df, "welch_p": p_value, "exact_permutation_p": permutation_p,
            "relative_change_percent": delta / c.mean() * 100,
        }
        for field, value in expected.items():
            same(f"{prefix}:{field}", value, row[field])
        p_values.append(p_value)
        permutation_p_values.append(permutation_p)
    for row, q, permutation_q in zip(tables[0], bh(p_values), bh(permutation_p_values)):
        same(f'table1:{row["dataset"]}:{row["metric"]}:welch_bh_q', q, row["welch_bh_q"])
        same(f'table1:{row["dataset"]}:{row["metric"]}:permutation_bh_q', permutation_q,
             row["exact_permutation_bh_q"])

    for row in tables[1]:
        task, method = row["dataset"], row["method"]
        assert row["metric_definition"] == "zero-inclusive, spatially unweighted, cumulative t+1:t+12"
        for metric in ["mae", "rmse"]:
            if method == "Persistence":
                name = f"persistence_{task}.json"
                used.add(name)
                expected = json.loads((data / name).read_text(encoding="utf-8"))[metric]
            else:
                kind = "control" if method == "Control" else "agf"
                expected = np.mean([float(r[metric]) for r in primary
                                    if r["dataset"] == task and r["kind"] == kind])
            same(f"table2:{task}:{method}:{metric}", row[metric], expected)

    ablation = read_csv(data / "ablation_per_seed.csv")
    assert len(ablation) == 18
    for row in ablation:
        kind = KINDS[row["variant"]]
        name = f'inference_{kind}_{row["dataset"]}_{row["seed"]}.json'
        used.add(name)
        inference = json.loads((data / name).read_text(encoding="utf-8"))
        assert inference["checkpoint"]["sha256"] == row["checkpoint_sha256"]
        assert inference["training_precision"] == "FP32"
        assert inference["epochs_completed"] == 100
        for metric in ["mae", "rmse"]:
            same(f'ablation_per_seed:{name}:{metric}', row[metric], inference["metrics"][metric])
    for row in tables[2]:
        task, variant = row["dataset"], row["variant"]
        subset = [r for r in ablation if r["dataset"] == task and r["variant"] == variant]
        assert sorted(int(r["seed"]) for r in subset) == [2023, 2024, 2025]
        assert int(row["n_seeds"]) == 3 and row["training_precision"] == "FP32"
        for metric in ["mae", "rmse"]:
            values = np.asarray([float(r[metric]) for r in subset])
            same(f"table3:{task}:{variant}:{metric}_mean", values.mean(), row[f"{metric}_mean"])
            same(f"table3:{task}:{variant}:{metric}_sd", values.std(ddof=1), row[f"{metric}_sd"])

    gate = read_csv(data / "gate_per_seed.csv")
    assert len(gate) == 20
    for row in tables[3]:
        task = row["dataset"]
        subset = [r for r in gate if r["dataset"] == task]
        assert sorted(int(r["seed"]) for r in subset) == list(range(2021, 2026))
        assert all(int(r["n_test_examples"]) == 657 and int(r["gate_channels"]) == 8
                   and int(r["input_steps"]) == 12 and int(r["nodes"]) == 2048 for r in subset)
        weights = np.asarray([int(r["n_scalar_gates"]) for r in subset])
        means = np.asarray([float(r["mean"]) for r in subset])
        deviations = np.asarray([float(r["population_std"]) for r in subset])
        pooled_mean = np.average(means, weights=weights)
        # The pooled population variance includes both within-seed and between-seed spread.
        pooled_std = np.sqrt(np.average(deviations ** 2 + (means - pooled_mean) ** 2, weights=weights))
        expected = {"mean": pooled_mean, "pooled_population_std": pooled_std,
                    "min": min(float(r["min"]) for r in subset),
                    "max": max(float(r["max"]) for r in subset),
                    "n_scalar_gates": sum(weights), "n_seeds": 5,
                    "n_test_examples_per_seed": 657, "gate_channels": 8, "input_steps": 12}
        for field, value in expected.items():
            same(f"table4:{task}:{field}", value, row[field], tolerance=2e-11)
        assert sum(weights) == 5 * 657 * 12 * 2048 * 8

    return {"status": "passed", "numeric_checks": len(checks),
            "maximum_absolute_error": max(r["absolute_error"] for r in checks),
            "primary_runs": 40, "additional_mean_fusion_runs": 6,
            "source_files": [{"path": name, "sha256": sha256(data / name)} for name in sorted(used)],
            "checks": checks}


def table(caption, label, columns, header, rows, note=""):
    return "\n".join([
        r"\begin{table}[htbp]", r"\caption{" + caption + "}", r"\label{" + label + "}",
        r"\centering", r"\small", r"\begin{tabular}{" + columns + "}", r"\toprule",
        header, r"\midrule", *rows, r"\bottomrule", r"\end{tabular}",
        (r"\vspace{2pt}\par\parbox{\linewidth}{\footnotesize " + note + "}" if note else ""),
        r"\end{table}", "",
    ])


def create_tables(tables):
    bodies, captions = [], []
    caption = (r"Five-replicate comparison of CLCRN-AGF and the schedule-matched CLCRN control. "
               r"Values are mean$\pm$sample standard deviation across seeds 2021--2025. "
               r"Differences are CLCRN-AGF minus control; negative values favour CLCRN-AGF. "
               r"Confidence intervals and two-sided $p$-values use Welch's unequal-variance approximation.")
    captions.append(caption)
    rows = []
    for i, r in enumerate(tables[0]):
        task = r["dataset"]
        label = LABELS[task] if i % 2 == 0 else ""
        row = [label, r["metric"].upper(),
               rf'${fmt(r["control_mean"], task)}\pm{fmt(r["control_sd"], task)}$',
               rf'${fmt(r["agf_mean"], task)}\pm{fmt(r["agf_sd"], task)}$',
               rf'${fmt(r["difference"], task, True)}$ [${fmt(r["ci95_low"], task, True)},{fmt(r["ci95_high"], task, True)}$]',
               pformat(r["welch_p"])]
        rows.append(" & ".join(row) + r" \\")
    humidity = {r["metric"]: r for r in tables[0] if r["dataset"] == "humidity"}
    note = (r"All scores include valid zero targets and use equal spatial weights over the full 12-hour "
            r"forecast and 657 test examples. " + UNITS +
            r" Benjamini--Hochberg correction across the eight comparisons retains only the two humidity "
            r"differences at $q<0.05$ (MAE $q=" + f'{float(humidity["mae"]["welch_bh_q"]):.5f}' +
            r"$; RMSE $q=" + f'{float(humidity["rmse"]["welch_bh_q"]):.6f}' + r"$).")
    bodies.append(table(caption, "tab:primary", "llrrrr",
        r"Task & Metric & Control (mean$\pm s$) & CLCRN-AGF (mean$\pm s$) & Difference [95\% CI] & $p_{\mathrm{Welch}}$ \\",
        rows, note))

    caption = (r"Contextual test results on the same test split. Persistence is deterministic; "
               r"control and CLCRN-AGF are means across five seeds. All methods use the same "
               r"zero-inclusive, spatially unweighted evaluation over the full 12-hour forecast. Lower is better.")
    captions.append(caption)
    rows = []
    for method, label in [("Persistence", "Persistence"), ("Control", "CLCRN control (5 seeds)"),
                          ("CLCRN-AGF", "CLCRN-AGF (5 seeds)")]:
        row = [label]
        for task in TASKS:
            selected = next(r for r in tables[1] if r["dataset"] == task and r["method"] == method)
            row += [fmt(selected[metric], task) for metric in ["mae", "rmse"]]
        rows.append(" & ".join(row) + r" \\")
    header = (r"Method & \multicolumn{2}{c}{Temperature} & \multicolumn{2}{c}{Relative humidity} "
              r"& \multicolumn{2}{c}{Surface wind} & \multicolumn{2}{c}{Cloud cover}\\" + "\n" +
              r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}" + "\n" +
              r"& MAE & RMSE & MAE & RMSE & MAE & RMSE & MAE & RMSE\\")
    bodies.append(table(caption, "tab:context", "lrrrrrrrr", header, rows,
                        UNITS + r" Variation across model seeds is reported in Table~\ref{tab:primary}."))

    caption = (r"Component ablations with a matched 100-epoch FP32 training protocol. "
               r"All entries report mean$\pm$sample standard deviation over seeds 2023--2025 "
               r"on the full 12-hour test forecast. Lower is better.")
    captions.append(caption)
    rows = []
    for task_index, task in enumerate(TASKS[:2]):
        if task_index:
            rows.append(r"\midrule")
        for variant_index, variant in enumerate(VARIANTS):
            r = next(r for r in tables[2] if r["dataset"] == task and r["variant"] == variant)
            row = ["100 epochs, 3 seeds" if variant_index == 0 else "",
                   LABELS[task] if variant_index == 0 else "",
                   {"w/o adaptive graph": "Control (no AGF)", "Mean fusion": "w/o gated fusion"}.get(variant, variant)]
            row += [rf'${fmt(r[metric + "_mean"], task)}\pm{fmt(r[metric + "_sd"], task)}$'
                    for metric in ["mae", "rmse"]]
            rows.append(" & ".join(row) + r" \\")
    bodies.append(table(caption, "tab:ablation", "lllrr",
                        r"Budget & Task & Variant & MAE & RMSE\\", rows,
                        r"The control removes the entire AGF module, including its residual projection, changing "
                        r"the encoder input width and parameter count. The ungated variant uses a fixed equal "
                        r"average of the graph and residual branches. "
                        r"The control and full-model rows use the same seed subset from Table~\ref{tab:primary}. "
                        r"Errors are in K for temperature and percentage points for relative humidity."))

    caption = (r"Gate-activation statistics pooled over all five CLCRN-AGF seeds and the full test set. "
               r"The gate balances a graph-aggregated projection and a residual linear projection of "
               r"the same input, before CLConv. Standard deviations describe the pooled population of scalar gates.")
    captions.append(caption)
    rows = []
    for task in TASKS:
        r = next(r for r in tables[3] if r["dataset"] == task)
        rows.append(" & ".join([LABELS[task]] + [f'{float(r[field]):.3f}'
                    for field in ["mean", "pooled_population_std", "min", "max"]]) + r" \\")
    bodies.append(table(caption, "tab:gate", "lrrrr",
                        r"Variable & Mean $\mu$ & Std $\sigma$ & Min & Max\\", rows,
                        r"Each task includes $5\times657\times12\times2{,}048\times8=645{,}857{,}280$ "
                        r"scalar gates (seeds, test examples, input steps, nodes and channels). "
                        r"Channels are retained individually; no channel averaging or test-set subsampling is applied."))
    return bodies, captions


PREAMBLE = r"""\documentclass[10pt,letterpaper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage{mathptmx,helvet,courier}
\usepackage{booktabs}
\usepackage[left=2cm,right=2cm,top=2.25cm,bottom=2.25cm]{geometry}
\usepackage[labelfont={bf,sf},labelsep=period,justification=raggedright]{caption}
\usepackage{microtype}
\setlength{\parindent}{0pt}
\setlength{\textfloatsep}{22pt}
\setlength{\floatsep}{24pt}
\begin{document}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--skip-pdf", action="store_true", help="Write CSV and LaTeX without compiling a PDF")
    args = parser.parse_args()
    data, output = args.data_root.resolve(), args.output_root.resolve()
    out_tables = output / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    (output / "metadata").mkdir(parents=True, exist_ok=True)
    tables = [read_csv(data / name) for name in TABLE_NAMES]
    validation = validate(data, tables)
    bodies, captions = create_tables(tables)
    changes = [
        "Updated all 40-run means, sample SDs, differences, Welch CIs and p-values; BH q-values retained in CSV and the note.",
        "Restricted contextual comparison to current Persistence, control and AGF results; no unmatched historical single runs or external values.",
        "Used three variants, two tasks and three matched seeds trained for 100 epochs in FP32; removed the unavailable 20-epoch screening block.",
        "Included all four tasks and all channels across the full test set for five seeds; distinguished pooled population SD from between-seed SD.",
    ]
    for number, (source, body) in enumerate(zip(TABLE_NAMES, bodies), start=1):
        shutil.copyfile(data / source, out_tables / source)
        (out_tables / f"table{number}.tex").write_text(body, encoding="utf-8")
    preview = PREAMBLE + "\n\\input{table1.tex}\n\\input{table2.tex}\n\\clearpage\n"
    preview += "\\input{table3.tex}\n\\input{table4.tex}\n\\clearpage\n\\end{document}\n"
    (out_tables / "tables_preview.tex").write_text(preview, encoding="utf-8")
    compilation = {"status": "skipped"}
    if not args.skip_pdf:
        compiler = shutil.which("pdflatex")
        if compiler is None:
            raise RuntimeError("pdflatex is unavailable; use --skip-pdf to generate only table sources")
        logs = []
        for _ in range(2):
            run = subprocess.run([compiler, "--disable-installer", "-interaction=nonstopmode", "-halt-on-error",
                                  "tables_preview.tex"], cwd=out_tables, capture_output=True, text=True, errors="replace")
            logs.append(run.stdout + run.stderr)
            if run.returncode:
                (out_tables / "compile_console.txt").write_text("\n".join(logs), encoding="utf-8")
                raise RuntimeError("LaTeX compilation failed; inspect tables/compile_console.txt")
        (out_tables / "compile_console.txt").write_text("\n".join(logs), encoding="utf-8")
        log = (out_tables / "tables_preview.log").read_text(encoding="utf-8", errors="replace")
        warnings = [line for line in log.splitlines() if "Overfull" in line or "undefined" in line]
        compilation = {"status": "passed", "compiler": compiler, "layout_warnings": warnings,
                       "pdf_sha256": sha256(out_tables / "tables_preview.pdf")}
        if warnings:
            raise RuntimeError(f"LaTeX layout/reference warnings require review: {warnings}")
    metadata = {
        "table_data": "46-run revision dated 2026-10-08",
        "scope": "40 primary runs, with six additional mean-fusion runs supporting the matched three-seed ablation",
        "preserved_style": "Original booktabs three-line rules, Times text/math, 9-point table text, grouped task columns, compact aligned numeric columns; no vertical rules.",
        "tables": [{"number": i + 1, "csv": f"tables/{name}", "latex": f"tables/table{i+1}.tex",
                    "caption_latex": captions[i], "source": name, "source_sha256": sha256(data / name),
                    "content_change": changes[i]} for i, name in enumerate(TABLE_NAMES)],
        "validation": validation, "compilation": compilation,
    }
    (output / "metadata" / "tables.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "numeric_checks": validation["numeric_checks"],
                      "maximum_absolute_error": validation["maximum_absolute_error"],
                      "pdf": str(out_tables / "tables_preview.pdf"), "compilation": compilation}, indent=2))


if __name__ == "__main__":
    main()
