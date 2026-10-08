"""Validate the supplement's local evidence and synchronized figure definitions.

Run from any directory with Python, numpy, pandas and scipy installed.
This only recomputes statistics from the recorded results; it does not run models.
"""
from pathlib import Path
import hashlib
import itertools
import json
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
checks = []


def check(name, condition, detail=None):
    record = {"name": name, "passed": bool(condition)}
    if detail is not None:
        record["detail"] = detail
    checks.append(record)
    if not condition:
        raise AssertionError(name)


def close(name, actual, expected, atol=1e-10):
    actual, expected = np.asarray(actual), np.asarray(expected)
    check(name, np.allclose(actual, expected, rtol=1e-10, atol=atol),
          {"max_abs_difference": float(np.max(np.abs(actual - expected)))})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bh(values):
    values = np.asarray(values)
    order = np.argsort(values)
    corrected = np.minimum.accumulate((values[order] * len(values) /
                                      np.arange(1, len(values)+1))[::-1])[::-1]
    result = np.empty_like(values)
    result[order] = np.minimum(corrected, 1)
    return result


def main():
    tex = (ROOT / "supplement.tex").read_text(encoding="utf-8")
    inventory = pd.read_csv(DATA / "model_inventory_46.csv")
    primary = pd.read_csv(DATA / "primary_per_seed.csv")
    estimates = pd.read_csv(DATA / "table1_primary_statistics.csv")
    check("46 unique inventory models", len(inventory) == 46 and
          not inventory.duplicated(["dataset", "kind", "seed"]).any())
    check("100 epochs for all inventory models", (inventory.epochs_completed == 100).all())
    check("40 primary models", len(primary) == 40)
    counts = primary.groupby(["dataset", "kind"]).size()
    check("five models per primary task and variant", len(counts) == 8 and (counts == 5).all())
    for _, row in primary.iterrows():
        inv = inventory[(inventory.dataset == row.dataset) &
                        (inventory.kind == row.kind) & (inventory.seed == row.seed)].iloc[0]
        close(f"inventory metrics {row.dataset}/{row.kind}/{row.seed}",
              [row.mae, row.rmse], [inv.test_mae, inv.test_rmse])
        check(f"checkpoint identity {row.dataset}/{row.kind}/{row.seed}",
              row.checkpoint_sha256 == inv.checkpoint_sha256 and row.epoch == inv.selected_epoch)
    ps, permutation_ps = [], []
    for _, row in estimates.iterrows():
        group = primary[primary.dataset == row.dataset]
        a = group[group.kind == "agf"][row.metric].to_numpy()
        c = group[group.kind == "control"][row.metric].to_numpy()
        difference = a.mean() - c.mean()
        va, vc = a.var(ddof=1)/5, c.var(ddof=1)/5
        df = (va+vc)**2/(va**2/4 + vc**2/4)
        margin = stats.t.ppf(.975, df)*np.sqrt(va+vc)
        p = stats.ttest_ind(a, c, equal_var=False).pvalue
        combined = np.r_[a, c]
        extremes = 0
        for combo in itertools.combinations(range(10), 5):
            subset = np.zeros(10, dtype=bool)
            subset[list(combo)] = True
            extremes += abs(combined[subset].mean()-combined[~subset].mean()) >= abs(difference)-1e-12
        pp = extremes/252
        ps.append(p)
        permutation_ps.append(pp)
        close(f"primary mean SD CI p {row.dataset}/{row.metric}",
              [a.mean(), c.mean(), a.std(ddof=1), c.std(ddof=1), difference,
               difference-margin, difference+margin, df, p, pp, 100*difference/c.mean()],
              [row.agf_mean, row.control_mean, row.agf_sd, row.control_sd,
               row.difference, row.ci95_low, row.ci95_high, row.welch_df,
               row.welch_p, row.exact_permutation_p, row.relative_change_percent])
        check(f"primary rounded mean/SD in supplement {row.dataset}/{row.metric}",
              f"{row.agf_mean:.4f} \\pm {row.agf_sd:.4f}" in tex and
              f"{row.control_mean:.4f} \\pm {row.control_sd:.4f}" in tex)
    close("primary Welch BH family", bh(ps), estimates.welch_bh_q)
    close("exact-permutation BH family", bh(permutation_ps), estimates.exact_permutation_bh_q)

    horizons = pd.read_csv(DATA / "horizon_per_seed.csv")
    check("480 horizon rows", len(horizons) == 480)
    endpoints = horizons[horizons.lead_hours == 12].merge(primary, on=["kind", "dataset", "seed"])
    close("all 40 cumulative MAE endpoints", endpoints.cumulative_mae, endpoints.mae)
    close("all 40 cumulative RMSE endpoints", endpoints.cumulative_rmse, endpoints.rmse)
    for (task, kind, lead), group in horizons[horizons.lead_hours.isin([6, 12])].groupby(
            ["dataset", "kind", "lead_hours"]):
        for metric in ("exact_mae", "exact_rmse"):
            check(f"rounded horizon {task}/{kind}/{lead}/{metric}",
                  f"{group[metric].mean():.4f} \\pm {group[metric].std(ddof=1):.4f}" in tex)

    masks = pd.read_csv(DATA / "robustness_per_seed_mask.csv")
    robustness = pd.read_csv(DATA / "robustness_summary.csv")
    check("520 robustness records on 64 windows", len(masks) == 520 and (masks.n_examples == 64).all())
    close("relative masked errors", 100*(masks.mae/masks.clean_mae_same_subset-1), masks.relative_increase_percent)
    seed_means = masks.groupby(["dataset", "kind", "seed", "ratio"]).relative_increase_percent.mean().reset_index()
    for _, row in robustness.iterrows():
        values = seed_means[(seed_means.dataset == row.dataset) & (seed_means.kind == row.kind) &
                            (seed_means.ratio == row.missing_ratio)].relative_increase_percent
        check(f"five nested robustness seeds {row.dataset}/{row.kind}/{row.missing_ratio}", len(values) == 5)
        close(f"nested robustness summary {row.dataset}/{row.kind}/{row.missing_ratio}",
              [values.mean(), values.std(ddof=1)],
              [row.mean_relative_mae_increase_percent, row.sd_across_trained_seeds_percent])
        if row.missing_ratio:
            check(f"rounded robustness {row.dataset}/{row.kind}/{row.missing_ratio}",
                  f"{values.mean():.2f} \\pm {values.std(ddof=1):.2f}" in tex)

    gates = pd.read_csv(DATA / "gate_per_seed.csv")
    hist = pd.read_csv(DATA / "gate_histogram.csv")
    pools = pd.read_csv(DATA / "table4_gate_statistics.csv")
    check("20 complete scalar-gate streams", len(gates) == 20 and
          (gates.n_scalar_gates == 657*12*2048*8).all())
    for (task, seed), h in hist.groupby(["dataset", "seed"]):
        g = gates[(gates.dataset == task) & (gates.seed == seed)].iloc[0]
        check(f"100-bin full gate support {task}/{seed}",
              len(h) == 100 and h["count"].sum() == g.n_scalar_gates)
    for _, row in pools.iterrows():
        g = gates[gates.dataset == row.dataset]
        mean = np.average(g["mean"], weights=g.n_scalar_gates)
        std = np.sqrt(np.average(g.population_std**2 + g["mean"]**2,
                                 weights=g.n_scalar_gates)-mean**2)
        close(f"pooled full-stream gate moments {row.dataset}",
              [mean, std, g["min"].min(), g["max"].max()],
              [row["mean"], row.pooled_population_std, row["min"], row["max"]])
        check(f"gate total {row.dataset}", row.n_scalar_gates == 5*657*12*2048*8)
    check("main binned gate limitation disclosed", "binned approximations" in tex and "not moments inferred from bin centres" in tex)
    check("six supplementary figures", len(re.findall(r"\\begin\{figure\}", tex)) == 6)
    check("both-variant robustness preserved", "fig:robustness_both" in tex and
          (ROOT/"figures/fig7_robustness_consistent.pdf").is_file())
    check("both-variant horizon preserved", "fig:horizon_both" in tex and
          (ROOT/"figures/fig5_horizon_consistent.pdf").is_file())
    check("outdated docx excluded", not (ROOT/"supplement.docx").exists())

    shared = []
    for local in sorted(DATA.iterdir()):
        main_file = ROOT.parent / "revision_2026_10_08" / "data" / local.name
        if main_file.is_file():
            check(f"same main/supplement data {local.name}", sha(local) == sha(main_file))
            shared.append(local.name)
    files = sorted([*DATA.glob("*"), * (ROOT/"figures").glob("*")])
    manifest = {"scope": "Supplement local evidence and preserved figures; no raw chat history",
                "files": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
                           "sha256": sha(p)} for p in files if p.is_file()]}
    (ROOT/"supplement_sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result = {"passed": True, "checks_passed": len(checks), "checks": checks,
              "shared_main_data_files": shared,
              "python": sys.version, "supplement_tex_sha256": sha(ROOT/"supplement.tex"),
              "method": "CPU recomputation of recorded metrics; no new training or inference",
              "changes": ["Preserved S1-S4", "Added S5 both-variant robustness and S6 both-variant horizons",
                          "Clarified main AGF-only diagnostics and binned gate quantiles",
                          "Replaced unavailable build-script/manifest references with local companions",
                          "Excluded stale Word copy from revised output"]}
    if (ROOT/"supplement.pdf").is_file():
        result["compiled_pdf_sha256"] = sha(ROOT/"supplement.pdf")
    if (ROOT/"supplement.log").is_file():
        log = (ROOT/"supplement.log").read_text(encoding="utf-8", errors="replace")
        result["latex_unresolved_references"] = bool(re.search(r"(?:Citation|Reference).*undefined|There were undefined references", log))
        result["latex_overfull_boxes"] = "Overfull" in log
        assert not result["latex_unresolved_references"]
        assert not result["latex_overfull_boxes"]
    (ROOT/"supplement_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    validation = ROOT.parent/"validation"
    validation.mkdir(exist_ok=True)
    (validation/"supplement_sync.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"PASS: {len(checks)} supplement synchronization checks; {len(shared)} common data files")


if __name__ == "__main__":
    main()
