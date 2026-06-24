"""Run humidity w/o-gated 100-epoch ablation with batch32 + AMP.

This is the closest feasible completion of the missing humidity component
ablation in the current Windows/CUDA environment.  Plain FP32 batch32 runs hit
allocator failures, while fresh batch32 automatic mixed precision smoke tests
pass.  Results are written to a separate directory so earlier attempts remain
auditable.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "experiments" / "humidity_ablation_100ep_amp_b32"
DATA_ROOT = ROOT / "Weather Bench_dataset"
SEEDS = [2023, 2024, 2025]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summary_path(seed: int) -> Path:
    return (
        OUT_ROOT
        / f"seed_{seed}"
        / "humidity"
        / "CLCRN_humidity_wo_gated_fusion"
        / "summary.json"
    )


def run_seed(seed: int) -> dict:
    summary = summary_path(seed)
    if summary.exists():
        return json.loads(summary.read_text(encoding="utf-8"))

    log_dir = OUT_ROOT / f"seed_{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "queue_stdout.log"
    stderr_path = log_dir / "queue_stderr.log"
    cmd = [
        sys.executable,
        "run_asttn_ablation.py",
        "--data-root",
        str(DATA_ROOT),
        "--datasets",
        "humidity",
        "--variants",
        "wo_gated_fusion",
        "--seed",
        str(seed),
        "--epochs",
        "100",
        "--gpu",
        "0",
        "--num-workers",
        "0",
        "--batch-size",
        "32",
        "--test-batch-size",
        "128",
        "--use-amp",
        "--log-root",
        str(log_dir),
    ]
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr:
        stdout.write("\n" + "=" * 80 + "\n")
        stdout.write(" ".join(cmd) + "\n")
        stdout.flush()
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=stdout, stderr=stderr, env=env)
    return json.loads(summary.read_text(encoding="utf-8"))


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, seed in enumerate(SEEDS, start=1):
        write_json(
            OUT_ROOT / "progress.json",
            {
                "status": "running",
                "current_seed": seed,
                "completed": index - 1,
                "total": len(SEEDS),
                "seeds": SEEDS,
                "batch_size": 32,
                "test_batch_size": 128,
                "use_amp": True,
            },
        )
        row = run_seed(seed)
        row["seed"] = seed
        rows.append(row)
        write_json(OUT_ROOT / "wo_gated_rows_partial.json", rows)

    write_json(
        OUT_ROOT / "progress.json",
        {
            "status": "completed",
            "current_seed": None,
            "completed": len(SEEDS),
            "total": len(SEEDS),
            "seeds": SEEDS,
            "batch_size": 32,
            "test_batch_size": 128,
            "use_amp": True,
        },
    )
    write_json(OUT_ROOT / "wo_gated_rows.json", rows)


if __name__ == "__main__":
    main()
