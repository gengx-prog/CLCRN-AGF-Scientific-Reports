"""Recompute publication metrics with the current evaluator.

This script intentionally writes new ``corrected_metrics`` files instead of
overwriting the historical ``summary.json`` files.  It is designed for the
Scientific Reports revision audit, where the archived experiments were run
before zeros in wind/cloud targets were handled correctly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supervisor import Supervisor

DATA_ROOT = ROOT / "Weather Bench_dataset"
TASKS = ["temperature", "humidity", "component_of_wind", "cloud_cover"]
SEEDS = [2021, 2022, 2023, 2024, 2025]

EXPERIMENT_ROOTS = {
    "agf": ROOT / "experiments" / "weatherbench_publication_multiseed",
    "control": ROOT / "experiments" / "weatherbench_publication_control_multiseed",
}

EXPERIMENT_SUFFIX = {
    "agf": "full_model",
    "control": "wo_adaptive_graph",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def experiment_dir(kind: str, seed: int, task: str) -> Path:
    suffix = EXPERIMENT_SUFFIX[kind]
    return (
        EXPERIMENT_ROOTS[kind]
        / f"seed_{seed}"
        / task
        / f"CLCRN_{task}_{suffix}"
    )


def saved_epochs(exp_dir: Path) -> list[int]:
    epochs: list[int] = []
    for path in (exp_dir / "saved_model").glob("epo*.tar"):
        try:
            epochs.append(int(path.stem.replace("epo", "")))
        except ValueError:
            continue
    return sorted(epochs)


def load_config_for_local_data(exp_dir: Path, task: str) -> dict:
    config = read_json(exp_dir / "model_param.json")
    task_root = DATA_ROOT / task
    config["data"]["dataset_dir"] = str(task_root)
    config["data"]["position_file"] = str(task_root / "position_info.pkl")
    config["data"]["num_workers"] = 0
    config["train"]["log_dir"] = str(exp_dir.parent)
    config["train"]["experiment_name"] = exp_dir.name
    return config


def evaluate_epoch(supervisor: Supervisor, dataset: str, epoch: int, steps=None) -> dict:
    mae, mse, mape, _, step_metrics = supervisor.evaluate(
        dataset,
        batches_seen=0,
        epoch_num=epoch,
        load_model=True,
        steps=steps,
    )
    return {
        "mae": float(mae),
        "rmse": float(math.sqrt(float(mse))),
        "mse": float(mse),
        "mape": float(mape),
        "step_metrics": {key: float(value) for key, value in step_metrics.items()},
    }


def recompute_one(kind: str, seed: int, task: str, select_best_val: bool) -> dict:
    exp_dir = experiment_dir(kind, seed, task)
    if not exp_dir.exists():
        raise FileNotFoundError(exp_dir)
    epochs = saved_epochs(exp_dir)
    if not epochs:
        raise RuntimeError(f"No checkpoints found under {exp_dir}")

    historical = read_json(exp_dir / "summary.json")
    config = load_config_for_local_data(exp_dir, task)
    supervisor = Supervisor(**config)
    try:
        if select_best_val:
            val_rows = []
            for epoch in epochs:
                metrics = evaluate_epoch(supervisor, "val", epoch)
                val_rows.append({"epoch": epoch, **metrics})
            best_val_row = min(val_rows, key=lambda item: item["mae"])
            best_epoch = int(best_val_row["epoch"])
        else:
            val_rows = []
            best_epoch = int(historical["best_epoch"])
            best_val_row = {"epoch": best_epoch, **evaluate_epoch(supervisor, "val", best_epoch)}

        test_metrics = evaluate_epoch(supervisor, "test", best_epoch, steps=[3, 6, 12])
    finally:
        supervisor._logger.handlers.clear()

    payload = {
        "kind": kind,
        "seed": seed,
        "dataset": task,
        "experiment_dir": str(exp_dir.relative_to(ROOT)),
        "data_root": str(DATA_ROOT),
        "selection": "corrected_validation_mae_among_saved_checkpoints"
        if select_best_val
        else "historical_best_epoch",
        "saved_epochs": epochs,
        "historical_best_epoch": int(historical["best_epoch"]),
        "corrected_best_epoch": int(best_epoch),
        "corrected_best_val_mae": float(best_val_row["mae"]),
        "mae": float(test_metrics["mae"]),
        "rmse": float(test_metrics["rmse"]),
        "mape": float(test_metrics["mape"]),
        "step_metrics": test_metrics["step_metrics"],
    }
    out_name = "summary_corrected_valbest.json" if select_best_val else "summary_corrected_historicalbest.json"
    write_json(exp_dir / out_name, payload)
    return payload


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for kind in ["control", "agf"]:
        for task in TASKS:
            task_rows = sorted(
                [row for row in rows if row["kind"] == kind and row["dataset"] == task],
                key=lambda item: item["seed"],
            )
            if not task_rows:
                continue
            mae = [row["mae"] for row in task_rows]
            rmse = [row["rmse"] for row in task_rows]
            out.append(
                {
                    "kind": kind,
                    "dataset": task,
                    "seed_count": len(task_rows),
                    "seeds": [row["seed"] for row in task_rows],
                    "mean_mae": float(np.mean(mae)),
                    "std_mae": float(np.std(mae, ddof=1)) if len(mae) > 1 else 0.0,
                    "mean_rmse": float(np.mean(rmse)),
                    "std_rmse": float(np.std(rmse, ddof=1)) if len(rmse) > 1 else 0.0,
                    "rows": task_rows,
                }
            )
    return out


def build_markdown(summary: list[dict]) -> str:
    lines = [
        "# Corrected publication metrics",
        "",
        "| Variant | Dataset | n | MAE mean +/- sd | RMSE mean +/- sd |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['kind']} | {row['dataset']} | {row['seed_count']} | "
            f"{row['mean_mae']:.6f} +/- {row['std_mae']:.6f} | "
            f"{row['mean_rmse']:.6f} +/- {row['std_rmse']:.6f} |"
        )
    lines.extend(["", "## Per-seed rows", ""])
    lines.append("| Variant | Dataset | Seed | Historical best | Corrected best | MAE | RMSE |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in summary:
        for item in row["rows"]:
            lines.append(
                f"| {item['kind']} | {item['dataset']} | {item['seed']} | "
                f"{item['historical_best_epoch']} | {item['corrected_best_epoch']} | "
                f"{item['mae']:.6f} | {item['rmse']:.6f} |"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinds", nargs="+", default=["control", "agf"], choices=["control", "agf"])
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument(
        "--historical-best",
        action="store_true",
        help="Use the historical best epoch instead of reselecting by corrected validation MAE.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "experiments" / "corrected_publication_metrics"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    select_best_val = not args.historical_best

    rows = []
    total = len(args.kinds) * len(args.tasks) * len(args.seeds)
    done = 0
    for kind in args.kinds:
        for seed in args.seeds:
            for task in args.tasks:
                done += 1
                progress = {
                    "status": "running",
                    "done": done - 1,
                    "total": total,
                    "current": {"kind": kind, "seed": seed, "dataset": task},
                }
                write_json(out_dir / "progress.json", progress)
                print(f"[{done}/{total}] {kind} seed={seed} task={task}", flush=True)
                row = recompute_one(kind, seed, task, select_best_val)
                rows.append(row)
                write_json(out_dir / "rows.json", rows)
                summary = summarize(rows)
                write_json(out_dir / "summary.json", summary)
                (out_dir / "summary.md").write_text(build_markdown(summary), encoding="utf-8")

    write_json(
        out_dir / "progress.json",
        {
            "status": "completed",
            "done": total,
            "total": total,
            "selection": "corrected_validation_mae_among_saved_checkpoints"
            if select_best_val
            else "historical_best_epoch",
        },
    )


if __name__ == "__main__":
    main()
