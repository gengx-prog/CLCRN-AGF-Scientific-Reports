import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Run publication post-analysis immediately.")
    parser.add_argument(
        "--status-root",
        default="./experiments/weatherbench_publication_post_analysis",
        help="Directory used to store post-analysis status.",
    )
    return parser.parse_args()


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_status(status_root: Path, payload):
    write_json(status_root / "post_analysis_status.json", payload)
    lines = [
        f"status: {payload['status']}",
        f"current_stage: {payload.get('current_stage', '')}",
    ]
    completed = payload.get("completed_stages", [])
    if completed:
        lines.append("")
        lines.append("completed_stages:")
        lines.extend(f"- {item}" for item in completed)
    (status_root / "post_analysis_status.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    status_root = (ROOT / args.status_root).resolve() if not Path(args.status_root).is_absolute() else Path(args.status_root)
    status_root.mkdir(parents=True, exist_ok=True)
    completed = []
    write_status(
        status_root,
        {
            "status": "running",
            "current_stage": "efficiency_analysis",
            "completed_stages": completed,
        },
    )

    stages = [
        ("efficiency_analysis", [sys.executable, "scripts/analyze_efficiency.py"]),
        ("missing_node_robustness", [sys.executable, "scripts/evaluate_missing_node_robustness.py"]),
        ("interpretability_figures", [sys.executable, "scripts/generate_interpretability_figures.py"]),
    ]
    for name, cmd in stages:
        write_status(
            status_root,
            {
                "status": "running",
                "current_stage": name,
                "completed_stages": completed,
            },
        )
        subprocess.run(cmd, cwd=ROOT, check=True)
        completed.append(name)

    write_status(
        status_root,
        {
            "status": "completed",
            "current_stage": None,
            "completed_stages": completed,
        },
    )


if __name__ == "__main__":
    main()
