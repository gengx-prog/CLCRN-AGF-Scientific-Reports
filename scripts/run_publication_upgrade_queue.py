import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Queue the long-running publication upgrade experiments.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--control-seed", type=int, default=2022)
    parser.add_argument("--multiseed-seeds", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument(
        "--status-root",
        default="./experiments/weatherbench_publication_upgrade_queue",
        help="Directory used for queue status files.",
    )
    return parser.parse_args()


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_status(path: Path, payload):
    write_json(path.with_suffix(".json"), payload)
    lines = [
        f"status: {payload['status']}",
        f"current_stage: {payload.get('current_stage', '')}",
    ]
    completed = payload.get("completed_stages", [])
    if completed:
        lines.append("")
        lines.append("completed_stages:")
        lines.extend(f"- {item}" for item in completed)
    path.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage(name: str, cmd, status_path: Path, completed_stages):
    write_status(
        status_path,
        {
            "status": "running",
            "current_stage": name,
            "completed_stages": completed_stages,
        },
    )
    subprocess.run(cmd, cwd=ROOT, check=True)
    completed_stages.append(name)


def main():
    args = parse_args()
    status_root = (ROOT / args.status_root).resolve() if not Path(args.status_root).is_absolute() else Path(args.status_root)
    status_path = status_root / "queue_status"
    completed_stages = []

    write_status(
        status_path,
        {
            "status": "running",
            "current_stage": None,
            "completed_stages": completed_stages,
        },
    )

    run_stage(
        "multiseed_full_model",
        [
            sys.executable,
            "scripts/run_publication_multiseed.py",
            "--gpu",
            str(args.gpu),
            "--num-workers",
            str(args.num_workers),
            "--seeds",
            *[str(seed) for seed in args.multiseed_seeds],
        ],
        status_path,
        completed_stages,
    )
    run_stage(
        "control_new_schedule",
        [
            sys.executable,
            "scripts/run_publication_control.py",
            "--gpu",
            str(args.gpu),
            "--num-workers",
            str(args.num_workers),
            "--seed",
            str(args.control_seed),
        ],
        status_path,
        completed_stages,
    )
    run_stage(
        "efficiency_analysis",
        [
            sys.executable,
            "scripts/analyze_efficiency.py",
        ],
        status_path,
        completed_stages,
    )
    run_stage(
        "missing_node_robustness",
        [
            sys.executable,
            "scripts/evaluate_missing_node_robustness.py",
        ],
        status_path,
        completed_stages,
    )
    run_stage(
        "interpretability_figures",
        [
            sys.executable,
            "scripts/generate_interpretability_figures.py",
        ],
        status_path,
        completed_stages,
    )

    write_status(
        status_path,
        {
            "status": "completed",
            "current_stage": None,
            "completed_stages": completed_stages,
        },
    )


if __name__ == "__main__":
    main()
