import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Wait for the long publication queue, then run the analysis scripts.")
    parser.add_argument(
        "--queue-status-root",
        default="./experiments/weatherbench_publication_upgrade_queue",
        help="Directory containing queue_status.json.",
    )
    parser.add_argument(
        "--status-root",
        default="./experiments/weatherbench_publication_post_analysis",
        help="Directory used for this watcher status.",
    )
    parser.add_argument("--poll-seconds", type=int, default=300)
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


def queue_completed(queue_status_root: Path):
    path = queue_status_root / "queue_status.json"
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("status") == "completed"


def main():
    args = parse_args()
    queue_status_root = (ROOT / args.queue_status_root).resolve() if not Path(args.queue_status_root).is_absolute() else Path(args.queue_status_root)
    status_root = (ROOT / args.status_root).resolve() if not Path(args.status_root).is_absolute() else Path(args.status_root)
    status_path = status_root / "post_analysis_status"
    completed_stages = []

    write_status(
        status_path,
        {
            "status": "waiting_for_queue",
            "current_stage": None,
            "completed_stages": completed_stages,
        },
    )

    while not queue_completed(queue_status_root):
        time.sleep(args.poll_seconds)

    stages = [
        ("efficiency_analysis", [sys.executable, "scripts/analyze_efficiency.py"]),
        ("missing_node_robustness", [sys.executable, "scripts/evaluate_missing_node_robustness.py"]),
        ("interpretability_figures", [sys.executable, "scripts/generate_interpretability_figures.py"]),
    ]
    for name, cmd in stages:
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
