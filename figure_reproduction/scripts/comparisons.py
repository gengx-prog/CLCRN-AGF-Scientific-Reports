"""Rebuild original-style Figures 3, 4 and 6 from the verified 2026-10-08 CSVs.

No historical scores enter these figures. Run with --data-root and --output-root.
PDF/SVG are vector exports; CSVs preserve full precision and checkpoint identities.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np


TASKS = ["temperature", "humidity", "component_of_wind", "cloud_cover"]
LABELS = ["Temperature", "Relative humidity", "Surface wind", "Cloud cover"]
SHORT_LABELS = ["Temperature", "Humidity", "Wind", "Cloud cover"]
UNITS = ["K", "percentage points", r"m s$^{-1}$", "fraction"]
SHORT_UNITS = ["K", "pp", r"m s$^{-1}$", "fraction"]
NAMES = {"control": "Control", "agf": "CLCRN-AGF"}
SCATTER_COLORS = {"control": "#5d86b5", "agf": "#ff8c29"}
BAR_COLORS = {"control": "#5ca0c9", "agf": "#3f7fb9"}
SOURCE_FILES = ["primary_per_seed.csv", "table1_primary_statistics.csv", "table2_current_context.csv"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, output, name):
    paths = []
    for ext in ["pdf", "svg", "png"]:
        path = output / "figures" / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.06,
                    facecolor="white", metadata={"Creator": "comparisons.py"} if ext == "pdf" else None)
        paths.append(str(path.relative_to(output)))
    plt.close(fig)
    return paths


def mean_sd(records, task, kind, metric):
    rows = sorted([r for r in records if r["dataset"] == task and r["kind"] == kind],
                  key=lambda r: int(r["seed"]))
    assert [int(r["seed"]) for r in rows] == list(range(2021, 2026)), (task, kind)
    values = np.array([float(r[metric]) for r in rows], dtype=np.float64)
    return rows, values, float(values.mean()), float(values.std(ddof=1))


def verify_sources(records, statistics, context):
    assert len(records) == 40
    assert len({(r["dataset"], r["kind"], r["seed"]) for r in records}) == 40
    checks = []
    for task in TASKS:
        for metric in ["mae", "rmse"]:
            stat = next(r for r in statistics if r["dataset"] == task and r["metric"] == metric)
            computed = {}
            for kind in ["control", "agf"]:
                _, _, mean, sd = mean_sd(records, task, kind, metric)
                row = next(r for r in context if r["dataset"] == task and r["method"] == NAMES[kind])
                errors = {"mean_vs_table1": abs(mean-float(stat[kind+"_mean"])),
                          "sd_vs_table1": abs(sd-float(stat[kind+"_sd"])),
                          "mean_vs_table2": abs(mean-float(row[metric]))}
                assert all(v < 1e-12 for v in errors.values()), (task, metric, kind, errors)
                assert int(stat["n_"+kind]) == int(row["n_seed"]) == 5
                checks.append({"task": task, "metric": metric, "kind": kind, **errors})
                computed[kind] = mean
            change = (computed["agf"] / computed["control"] - 1) * 100
            assert abs(change-float(stat["relative_change_percent"])) < 1e-12
    return checks


def fig3_heatmap(context, output):
    methods = ["Persistence", "Control", "CLCRN-AGF"]
    columns = [(task, metric) for task in TASKS for metric in ["mae", "rmse"]]
    raw = np.array([[float(next(r for r in context if r["dataset"] == task and r["method"] == model)[metric])
                     for task, metric in columns] for model in methods])
    spans = np.ptp(raw, axis=0)
    score = np.divide(raw.max(axis=0)-raw, spans, out=np.ones_like(raw), where=spans != 0)
    rank = np.argsort(np.argsort(raw, axis=0), axis=0) + 1
    fig, ax = plt.subplots(figsize=(12.4, 3.3))
    im = ax.pcolormesh(np.arange(9)-.5, np.arange(4)-.5, score,
                       cmap="RdYlGn", vmin=0, vmax=1, shading="flat", rasterized=False)
    ax.set_ylim(2.5, -.5)
    labels = [f"{SHORT_LABELS[i]}\n{metric.upper()} ({SHORT_UNITS[i]})"
              for i, _ in enumerate(TASKS) for metric in ["mae", "rmse"]]
    ax.set_xticks(np.arange(8), labels, fontsize=9.1, fontweight="bold")
    ax.set_yticks(np.arange(3), ["Persistence", "Control (n = 5)", "CLCRN-AGF (n = 5)"], fontsize=10)
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=8)
    ax.tick_params(axis="y", length=0, pad=7)
    for x in np.arange(-0.5, 8, 1):
        ax.axvline(x, color="white", lw=1.3, zorder=3)
    for y in np.arange(-0.5, 3, 1):
        ax.axhline(y, color="white", lw=1.3, zorder=3)
    for x in [1.5, 3.5, 5.5]:
        ax.axvline(x, color="#555555", lw=1.6, zorder=4)
    ax.add_patch(Rectangle((-.5, 1.5), 8, 1, fill=False, edgecolor="#222222", lw=1.8, zorder=5))
    chart_rows = []
    for i, model in enumerate(methods):
        for j, (task, metric) in enumerate(columns):
            rgba = im.cmap(score[i, j])
            rgb = np.array(rgba[:3])
            linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055)**2.4)
            luminance = float(np.dot(linear, [.2126, .7152, .0722]))
            text_color = "white" if luminance < .179 else "#111111"
            precision = 4 if task == "cloud_cover" else 3
            ax.text(j, i, f"{raw[i, j]:.{precision}f}\n#{rank[i,j]}", ha="center", va="center",
                    fontsize=10, color=text_color, fontweight="bold" if rank[i,j] == 1 else "normal")
            chart_rows.append({"dataset": task, "method": model, "metric": metric,
                               "value": raw[i,j], "n_seed": 1 if model == "Persistence" else 5,
                               "units": UNITS[TASKS.index(task)].replace("$", ""),
                               "column_minmax_score": score[i,j], "descriptive_rank": int(rank[i,j])})
    cbar = fig.colorbar(im, ax=ax, fraction=.022, pad=.02)
    cbar.solids.set_rasterized(False)
    cbar.set_ticks([0, .5, 1], labels=["Worst", "Mid-range", "Best"])
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Relative value within each column", fontsize=9, labelpad=10)
    ax.set_title("Current-split results under one evaluation protocol", fontsize=13, fontweight="bold", pad=55)
    fig.text(.54, .075, "Cell colours use separate min–max scales per column; numbers show errors in physical units.",
             ha="center", fontsize=9, color="#444444")
    fig.text(.54, .018, "Lower is better. Ranks are descriptive; pp = percentage points. Persistence is deterministic.",
             ha="center", fontsize=8.5, color="#555555")
    fig.subplots_adjust(left=.155, right=.905, bottom=.23, top=.68)
    write_csv(output/"source_data"/"fig3_context_heatmap.csv", chart_rows)
    return save(fig, output, "fig3_context_heatmap")


def fig4_replicates(records, output):
    fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.8))
    chart_rows = []
    for i, (ax, task) in enumerate(zip(axes, TASKS)):
        means = {}
        lows, highs = [], []
        for x, kind in enumerate(["control", "agf"]):
            rows, values, mean, sd = mean_sd(records, task, kind, "mae")
            jitter = np.linspace(-.095, .095, len(values))
            ax.scatter(x+jitter, values, s=38, color=SCATTER_COLORS[kind],
                       edgecolors="white", linewidth=.5, zorder=4)
            ax.errorbar(x, mean, yerr=sd, fmt="D", color="black", markersize=5,
                        capsize=4, capthick=1.1, lw=1.2, zorder=5)
            means[kind] = mean
            lows.append(min(values.min(), mean-sd))
            highs.append(max(values.max(), mean+sd))
            for row, value, dx in zip(rows, values, jitter):
                chart_rows.append({"dataset": task, "kind": kind, "seed": int(row["seed"]),
                                   "metric": "mae", "value": value, "mean": mean, "sample_sd": sd,
                                   "n_seed": len(values), "x_position": x+dx,
                                   "checkpoint_sha256": row["checkpoint_sha256"], "selected_epoch": int(row["epoch"])})
        change = (means["agf"] / means["control"] - 1) * 100
        for row in chart_rows:
            if row["dataset"] == task:
                row["agf_relative_change_percent"] = change
        ax.set_title(f"{LABELS[i]}\nmean change {change:+.2f}%", fontsize=10.5, pad=9)
        ax.set_xticks([0, 1], ["Control", "CLCRN-AGF"], fontsize=9)
        ax.set_ylabel(f"Test MAE ({UNITS[i]})", fontsize=9)
        ax.set_xlim(-.25, 1.25)
        span = max(highs)-min(lows)
        ax.set_ylim(min(lows)-span*.08, max(highs)+span*.08)
        ax.grid(axis="y", color="#c6c6c6", alpha=.45, lw=.7)
        ax.tick_params(labelsize=9)
        ax.set_axisbelow(True)
    fig.suptitle("Five-replicate comparison of the adaptive graph-fusion extension", fontsize=13, y=.98)
    handles = [Line2D([], [], marker="o", ls="", color=SCATTER_COLORS[k], markersize=5,
                      label=NAMES[k]+" seed result") for k in ["control", "agf"]]
    handles.append(Line2D([], [], marker="D", color="black", markersize=4, lw=1,
                          label="Mean ± sample SD"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5,.017), fontsize=9)
    fig.subplots_adjust(left=.057, right=.99, bottom=.20, top=.75, wspace=.60)
    write_csv(output/"source_data"/"fig4_primary_replicates.csv", chart_rows)
    return save(fig, output, "fig4_primary_replicates")


def fig6_bars(records, output):
    fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.65))
    chart_rows = []
    for i, (ax, task) in enumerate(zip(axes, TASKS)):
        summary = [mean_sd(records, task, kind, "mae")[2:] for kind in ["control", "agf"]]
        upper = max(mean+sd for mean, sd in summary)
        for x, (kind, (mean, sd)) in enumerate(zip(["control", "agf"], summary)):
            ax.bar(x, mean, width=.62, color=BAR_COLORS[kind], hatch="///" if kind == "control" else None,
                   edgecolor="white", linewidth=.7, zorder=3, alpha=.95)
            ax.errorbar(x, mean, yerr=sd, fmt="none", color="#222222", capsize=4,
                        capthick=1.1, lw=1.15, zorder=4)
            digits = 4 if task == "cloud_cover" else 3
            ax.text(x, mean+sd+upper*.025, f"{mean:.{digits}f}", ha="center", va="bottom", fontsize=10)
            chart_rows.append({"dataset": task, "kind": kind, "metric": "mae", "mean": mean,
                               "sample_sd": sd, "n_seed": 5,
                               "units": UNITS[i].replace("$", ""), "bar_baseline": 0})
        ax.set_title(LABELS[i], fontsize=11, pad=9)
        ax.set_ylabel(f"Test MAE ({UNITS[i]})", fontsize=9)
        ax.set_xticks([0,1], ["Control", "CLCRN-AGF"], fontsize=9)
        ax.set_ylim(0, upper*1.20)
        ax.set_xlim(-.65,1.65)
        ax.grid(axis="y", color="#bfbfbf", ls="--", alpha=.45, lw=.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=9)
    fig.suptitle("CLCRN control and adaptive graph-fusion comparison", fontsize=13, fontweight="bold", y=.985)
    handles = [Patch(facecolor=BAR_COLORS[k], hatch="///" if k == "control" else None,
                     edgecolor="white", label=NAMES[k]) for k in ["control", "agf"]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,.895), ncol=2,
               fontsize=9, frameon=False)
    fig.text(.5, .025, "Lower is better. Error bars show ±1 sample SD over five seeds per variant; each panel has its own physical scale.",
             ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(left=.057, right=.99, bottom=.17, top=.72, wspace=.62)
    write_csv(output/"source_data"/"fig6_primary_bars.csv", chart_rows)
    return save(fig, output, "fig6_primary_bars")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    data, output = args.data_root.resolve(), args.output_root.resolve()
    for folder in ["figures", "source_data", "metadata"]:
        (output/folder).mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.linewidth":.8,
                         "pdf.fonttype":42, "ps.fonttype":42, "svg.fonttype":"none",
                         "axes.spines.top":True, "axes.spines.right":True, "axes.unicode_minus":True})
    records = read_csv(data/SOURCE_FILES[0])
    statistics = read_csv(data/SOURCE_FILES[1])
    context = read_csv(data/SOURCE_FILES[2])
    checks = verify_sources(records, statistics, context)
    exports = fig3_heatmap(context, output) + fig4_replicates(records, output) + fig6_bars(records, output)
    metadata = {
        "input_data_root":str(data),
        "source_files":[{"path":name,"sha256":sha256(data/name)} for name in SOURCE_FILES],
        "scope":"New 2026-10-08 46-run study; these three figures use its 40 primary runs and deterministic persistence, not the six mean-fusion ablations.",
        "metric_definition":"Zero-inclusive, spatially unweighted errors over all 657 test windows and all leads t+1:t+12; training was FP32, 100 epochs; selected checkpoints minimize validation MAE.",
        "validation":{"status":"passed", "tolerance":1e-12, "comparisons":checks,
                      "sample_sd_ddof":1,"unique_primary_runs":40,"seeds_per_variant_task":5},
        "figures":{
            "fig3_context_heatmap":{
                "old_to_new":"Preserve the original annotated 8-column heatmap, task separators and outlined AGF row. Replace unsupported seven-model historical comparison with three methods actually evaluated on the current split. Colours retain the user-requested RdYlGn style; cell labels and ranks provide redundant encoding.",
                "caption":"Current-split cumulative 12-hour MAE and RMSE for persistence and the two trained variants. Control and CLCRN-AGF values are means across five training seeds; persistence is deterministic. All results use the same zero-inclusive, spatially unweighted evaluator and agree with the current context table. Colours use independent min–max normalization within each task/metric column; the numbers retain physical units. Rankings are descriptive, not significance tests. Temperature is in K, relative humidity in percentage points (pp), wind in m s−1, and cloud cover in fraction.",
                "source_csv":"fig3_context_heatmap.csv"},
            "fig4_primary_replicates":{
                "old_to_new":"Preserve four aligned panels, blue control dots, orange AGF dots, black diamonds and error bars, and signed mean-change annotations. Recalculate every mark and annotation from the current five-seed data. Separate physical-unit axes preserve the original focused MAE comparison; RMSE remains available in the statistical table.",
                "caption":"Cumulative 12-hour test MAE across five independently trained replicates for each variant and task. Coloured points show individual seeds; black diamonds and bars show the mean plus or minus sample standard deviation (ddof = 1). The percentage above each panel is 100 × (AGF mean / control mean − 1); a negative value denotes lower MAE. These are the same 40 primary checkpoints used in the current statistical table. Seed labels are not treated as matched random streams. Panels have task-specific units and focused vertical scales.",
                "source_csv":"fig4_primary_replicates.csv"},
            "fig6_primary_bars":{
                "old_to_new":"Retain the original grouped-column aesthetic, blue shades, control hatching, numeric annotations and mean-error bars. Remove unsupported original-reproduction/paper-target columns and causal schedule-decomposition claims. Separate the four task groups into aligned panels with zero-based, task-specific physical axes. Both model variants now use the same five-seed protocol.",
                "caption":"Cumulative 12-hour test MAE for CLCRN control and CLCRN-AGF under the same current training and evaluation protocol. Bars show five-seed means; error bars show plus or minus one sample standard deviation (ddof = 1). These values exactly match the primary statistical table and Figure 4. All axes start at zero; panels use different physical units and independent scales. This comparison does not isolate a training-schedule effect.",
                "source_csv":"fig6_primary_bars.csv"}
        },
        "chart_sources":[{"path":str(path.relative_to(output)),"sha256":sha256(path)}
                         for name in ["fig3_context_heatmap.csv", "fig4_primary_replicates.csv", "fig6_primary_bars.csv"]
                         for path in [output/"source_data"/name]],
        "script_sha256":sha256(Path(__file__)),
        "exports":[{"path":name,"sha256":sha256(output/name)} for name in exports]
    }
    (output/"metadata"/"comparisons.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"passed","source_comparisons":len(checks),"exports":exports},ensure_ascii=False))


if __name__ == "__main__":
    main()
