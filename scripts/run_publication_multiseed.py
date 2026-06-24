import argparse
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ["temperature", "humidity", "component_of_wind", "cloud_cover"]
DEFAULT_SEEDS = [2021, 2022, 2023, 2024, 2025]


def parse_args():
    parser = argparse.ArgumentParser(description="Run clean multi-seed full-model experiments for all WeatherBench datasets.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--log-root",
        default="./experiments/weatherbench_publication_multiseed",
        help="Output directory for the multiseed runs.",
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
    return root / f"seed_{seed}" / dataset_name / f"CLCRN_{dataset_name}_full_model" / "summary.json"


def run_dir(root: Path, seed: int, dataset_name: str):
    return root / f"seed_{seed}" / dataset_name


def run_one(root: Path, seed: int, dataset_name: str, epochs: int, gpu: int, num_workers: int, max_attempts: int):
    path = summary_path(root, seed, dataset_name)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    active_run_dir = run_dir(root, seed, dataset_name)
    if active_run_dir.exists():
        shutil.rmtree(active_run_dir, ignore_errors=True)

    cmd = [
        sys.executable,
        "run_asttn_ablation.py",
        "--datasets",
        dataset_name,
        "--variants",
        "full_model",
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--gpu",
        str(gpu),
        "--num-workers",
        str(num_workers),
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
        aggregated.append(
            {
                "dataset": dataset_name,
                "seeds": dataset_rows,
                "mean_mae": statistics.mean(mae_values),
                "std_mae": statistics.stdev(mae_values) if len(mae_values) > 1 else 0.0,
                "mean_rmse": statistics.mean(rmse_values),
                "std_rmse": statistics.stdev(rmse_values) if len(rmse_values) > 1 else 0.0,
                "mean_best_val_mae": statistics.mean(best_val_values),
                "std_best_val_mae": statistics.stdev(best_val_values) if len(best_val_values) > 1 else 0.0,
                "seed_count": len(dataset_rows),
                "target_seed_count": len(seeds),
            }
        )
    return aggregated


def build_markdown(aggregated):
    lines = [
        "# Improved CLCRN Multi-Seed Results",
        "",
        "| Dataset | Seed Count | Test MAE (mean +- std) | Test RMSE (mean +- std) | Best Val MAE (mean +- std) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregated:
        lines.append(
            f"| {row['dataset']} | {row['seed_count']} | "
            f"{row['mean_mae']:.4f} +- {row['std_mae']:.4f} | "
            f"{row['mean_rmse']:.4f} +- {row['std_rmse']:.4f} | "
            f"{row['mean_best_val_mae']:.4f} +- {row['std_best_val_mae']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Per-Seed Results",
            "",
            "| Dataset | Seed | Best Epoch | Best Val MAE | Test MAE | Test RMSE |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregated:
        for seed_row in row["seeds"]:
            lines.append(
                f"| {row['dataset']} | {seed_row['seed']} | {seed_row['best_epoch']} | "
                f"{seed_row['best_val_mae']:.4f} | {seed_row['mae']:.4f} | {seed_row['rmse']:.4f} |"
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
            summary = run_one(root, seed, dataset_name, args.epochs, args.gpu, args.num_workers, args.max_attempts)
            summary["seed"] = seed
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
