import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.publication_utils import ORIGINAL_EXPERIMENTS, PAPER_REFERENCE, TRUSTED_IMPROVED_EXPERIMENTS, load_summary


DEFAULT_DATASETS = ["temperature", "humidity", "component_of_wind", "cloud_cover"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the key control: original CLCRN with the improved training schedule.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=DEFAULT_DATASETS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--log-root",
        default="./experiments/weatherbench_publication_control_100ep",
        help="Output directory for the 100-epoch control runs.",
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
        f"current_dataset: {payload.get('current_dataset', '')}",
    ]
    completed_items = payload.get("completed_items", [])
    if completed_items:
        lines.append("")
        lines.append("completed_items:")
        lines.extend(f"- {item}" for item in completed_items)
    path.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summary_path(root: Path, dataset_name: str):
    return root / dataset_name / f"CLCRN_{dataset_name}_wo_adaptive_graph" / "summary.json"


def run_dir(root: Path, dataset_name: str):
    return root / dataset_name


def run_one(root: Path, dataset_name: str, epochs: int, seed: int, gpu: int, num_workers: int, max_attempts: int):
    path = summary_path(root, dataset_name)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    active_run_dir = run_dir(root, dataset_name)
    if active_run_dir.exists():
        shutil.rmtree(active_run_dir, ignore_errors=True)

    cmd = [
        sys.executable,
        "run_asttn_ablation.py",
        "--datasets",
        dataset_name,
        "--variants",
        "wo_adaptive_graph",
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--gpu",
        str(gpu),
        "--num-workers",
        str(num_workers),
        "--log-root",
        str(root),
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


def build_markdown(rows):
    lines = [
        "# Original CLCRN + New Training Schedule",
        "",
        "This control keeps the improved optimization schedule but removes the adaptive graph branch.",
        "",
        "| Dataset | Control MAE | Control RMSE | Original Reproduction MAE | Improved Full Model MAE | Paper MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['control']['mae']:.4f} | {row['control']['rmse']:.4f} | "
            f"{row['original']['mae']:.4f} | {row['improved']['mae']:.4f} | {row['paper']['mae']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    root = (ROOT / args.log_root).resolve() if not Path(args.log_root).is_absolute() else Path(args.log_root)
    progress_path = root / "progress"

    completed_items = []
    rows = []
    write_progress(
        progress_path,
        {
            "status": "running",
            "completed_runs": 0,
            "total_runs": len(args.datasets),
            "current_dataset": None,
            "completed_items": completed_items,
        },
    )

    for dataset_name in args.datasets:
        write_progress(
            progress_path,
            {
                "status": "running",
                "completed_runs": len(completed_items),
                "total_runs": len(args.datasets),
                "current_dataset": dataset_name,
                "completed_items": completed_items,
            },
        )
        summary = run_one(root, dataset_name, args.epochs, args.seed, args.gpu, args.num_workers, args.max_attempts)
        row = {
            "dataset": dataset_name,
            "control": {
                "best_epoch": summary["best_epoch"],
                "best_val_mae": summary["best_val_mae"],
                "mae": summary["mae"],
                "rmse": summary["rmse"],
            },
            "original": load_summary(ORIGINAL_EXPERIMENTS[dataset_name]),
            "improved": load_summary(TRUSTED_IMPROVED_EXPERIMENTS[dataset_name]),
            "paper": PAPER_REFERENCE[dataset_name],
        }
        rows.append(row)
        completed_items.append(dataset_name)
        write_json(
            root / "control_summary.json",
            {
                "epochs": args.epochs,
                "seed": args.seed,
                "datasets": args.datasets,
                "results": rows,
            },
        )
        (root / "control_summary.md").write_text(build_markdown(rows), encoding="utf-8")

    write_progress(
        progress_path,
        {
            "status": "completed",
            "completed_runs": len(completed_items),
            "total_runs": len(args.datasets),
            "current_dataset": None,
            "completed_items": completed_items,
        },
    )


if __name__ == "__main__":
    main()
