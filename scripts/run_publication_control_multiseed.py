import argparse
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.publication_utils import CONTROL_NEW_SCHEDULE_EXPERIMENTS, PAPER_REFERENCE, ORIGINAL_EXPERIMENTS, TRUSTED_IMPROVED_EXPERIMENTS, load_summary


DEFAULT_DATASETS = ["temperature", "humidity", "component_of_wind", "cloud_cover"]
DEFAULT_SEEDS = [2021, 2022, 2023, 2024, 2025]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the key control (original CLCRN + new schedule) with multiple random seeds."
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--reuse-seed-2022",
        action="store_true",
        help="Reuse the already completed single-seed control results under experiments/weatherbench_publication_control_100ep.",
    )
    parser.add_argument(
        "--log-root",
        default="./experiments/weatherbench_publication_control_multiseed",
        help="Output directory for the control multiseed runs.",
    )
    return parser.parse_args()


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_progress(path: Path, payload):
    write_json(path.with_suffix(".json"), payload)
    lines = [
        f"status: {payload['status']}",
        f"completed: {payload['completed_runs']}/{payload['total_runs']}",
        f"current_seed: {payload.get('current_seed', '')}",
        f"current_dataset: {payload.get('current_dataset', '')}",
    ]
    completed_items = payload.get("completed_items", [])
    if completed_items:
        lines.append("")
        lines.append("completed_items:")
        lines.extend(f"- {item}" for item in completed_items)
    path.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summary_path(root: Path, seed: int, dataset_name: str):
    return root / f"seed_{seed}" / dataset_name / f"CLCRN_{dataset_name}_wo_adaptive_graph" / "summary.json"


def run_dir(root: Path, seed: int, dataset_name: str):
    return root / f"seed_{seed}" / dataset_name


def reuse_2022_summary(dataset_name: str):
    exp_dir = CONTROL_NEW_SCHEDULE_EXPERIMENTS[dataset_name]
    summary = load_summary(exp_dir)
    return {
        "dataset": dataset_name,
        "seed": 2022,
        "best_epoch": summary["best_epoch"],
        "best_val_mae": summary["best_val_mae"],
        "mae": summary["mae"],
        "rmse": summary["rmse"],
        "mape": summary.get("mape"),
        "step_metrics": summary.get("step_metrics", {}),
        "source": "reused_single_seed_control_100ep",
    }


def run_one(root: Path, seed: int, dataset_name: str, epochs: int, gpu: int, num_workers: int, max_attempts: int):
    path = summary_path(root, seed, dataset_name)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    active_run_dir = run_dir(root, seed, dataset_name)
    if active_run_dir.exists():
        shutil.rmtree(active_run_dir, ignore_errors=True)

    cmd = [
        sys.executable,
        "scripts/run_publication_control.py",
        "--datasets",
        dataset_name,
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--gpu",
        str(gpu),
        "--num-workers",
        str(num_workers),
        "--max-attempts",
        str(max_attempts),
        "--log-root",
        str(root / f"seed_{seed}"),
    ]
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            subprocess.run(cmd, cwd=ROOT, check=True)
            return json.loads(path.read_text(encoding="utf-8"))
        except subprocess.CalledProcessError as exc:
            last_error = exc
            shutil.rmtree(active_run_dir, ignore_errors=True)
            if attempt == max_attempts:
                raise
    raise last_error


def aggregate_results(datasets, seeds, results):
    aggregated = []
    for dataset_name in datasets:
        dataset_rows = sorted(
            [row for row in results if row["dataset"] == dataset_name],
            key=lambda item: item["seed"],
        )
        if not dataset_rows:
            continue
        mae_values = [row["mae"] for row in dataset_rows]
        rmse_values = [row["rmse"] for row in dataset_rows]
        best_val_values = [row["best_val_mae"] for row in dataset_rows]
        original = load_summary(ORIGINAL_EXPERIMENTS[dataset_name])
        improved = load_summary(TRUSTED_IMPROVED_EXPERIMENTS[dataset_name])
        aggregated.append(
            {
                "dataset": dataset_name,
                "label": {
                    "temperature": "Temperature",
                    "humidity": "Humidity",
                    "component_of_wind": "Wind",
                    "cloud_cover": "Cloud Cover",
                }[dataset_name],
                "seeds": dataset_rows,
                "mean_mae": statistics.mean(mae_values),
                "std_mae": statistics.stdev(mae_values) if len(mae_values) > 1 else 0.0,
                "mean_rmse": statistics.mean(rmse_values),
                "std_rmse": statistics.stdev(rmse_values) if len(rmse_values) > 1 else 0.0,
                "mean_best_val_mae": statistics.mean(best_val_values),
                "std_best_val_mae": statistics.stdev(best_val_values) if len(best_val_values) > 1 else 0.0,
                "seed_count": len(dataset_rows),
                "target_seed_count": len(seeds),
                "paper": PAPER_REFERENCE[dataset_name],
                "original": {"mae": original["mae"], "rmse": original["rmse"]},
                "improved": {"mae": improved["mae"], "rmse": improved["rmse"]},
            }
        )
    return aggregated


def build_markdown(aggregated):
    lines = [
        "# Control (Original CLCRN + New Schedule) Multi-Seed Results",
        "",
        "| Dataset | Seed Count | Control MAE (mean +- std) | Control RMSE (mean +- std) | Paper MAE | Original MAE | Improved MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregated:
        lines.append(
            f"| {row['label']} | {row['seed_count']} | "
            f"{row['mean_mae']:.4f} +- {row['std_mae']:.4f} | "
            f"{row['mean_rmse']:.4f} +- {row['std_rmse']:.4f} | "
            f"{row['paper']['mae']:.4f} | {row['original']['mae']:.4f} | {row['improved']['mae']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Per-Seed Results",
            "",
            "| Dataset | Seed | Best Epoch | Best Val MAE | Test MAE | Test RMSE | Source |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in aggregated:
        for seed_row in row["seeds"]:
            lines.append(
                f"| {row['label']} | {seed_row['seed']} | {seed_row['best_epoch']} | "
                f"{seed_row['best_val_mae']:.4f} | {seed_row['mae']:.4f} | {seed_row['rmse']:.4f} | "
                f"{seed_row.get('source', 'fresh_run')} |"
            )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    root = (ROOT / args.log_root).resolve() if not Path(args.log_root).is_absolute() else Path(args.log_root)
    progress_path = root / "progress"

    total_runs = len(args.datasets) * len(args.seeds)
    completed_items = []
    results = []
    write_progress(
        progress_path,
        {
            "status": "running",
            "completed_runs": 0,
            "total_runs": total_runs,
            "current_seed": None,
            "current_dataset": None,
            "completed_items": completed_items,
        },
    )

    for seed in args.seeds:
        for dataset_name in args.datasets:
            write_progress(
                progress_path,
                {
                    "status": "running",
                    "completed_runs": len(completed_items),
                    "total_runs": total_runs,
                    "current_seed": seed,
                    "current_dataset": dataset_name,
                    "completed_items": completed_items,
                },
            )
            if args.reuse_seed_2022 and seed == 2022:
                summary = reuse_2022_summary(dataset_name)
            else:
                summary = run_one(root, seed, dataset_name, args.epochs, args.gpu, args.num_workers, args.max_attempts)
                summary["seed"] = seed
                summary["source"] = "fresh_run"
            results.append(summary)
            completed_items.append(f"seed_{seed}/{dataset_name}")
            aggregated = aggregate_results(args.datasets, args.seeds, results)
            write_json(
                root / "aggregate_results.json",
                {
                    "seeds": args.seeds,
                    "epochs": args.epochs,
                    "datasets": args.datasets,
                    "results": results,
                    "aggregated": aggregated,
                    "reuse_seed_2022": args.reuse_seed_2022,
                },
            )
            (root / "aggregate_results.md").write_text(build_markdown(aggregated), encoding="utf-8")

    write_progress(
        progress_path,
        {
            "status": "completed",
            "completed_runs": len(completed_items),
            "total_runs": total_runs,
            "current_seed": None,
            "current_dataset": None,
            "completed_items": completed_items,
        },
    )


if __name__ == "__main__":
    main()
