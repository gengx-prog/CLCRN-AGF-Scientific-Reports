import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.publication_utils import (
    BEST_FINAL_EXPERIMENTS,
    CONTROL_NEW_SCHEDULE_EXPERIMENTS,
    ORIGINAL_EXPERIMENTS,
    TRUSTED_IMPROVED_EXPERIMENTS,
    dataset_label,
    load_config,
    parse_epoch_times,
)
from supervisor import Supervisor


OUT_DIR = ROOT / "experiments" / "weatherbench_publication_efficiency"
DATASETS = ["temperature", "humidity", "component_of_wind", "cloud_cover"]


def build_supervisor(exp_dir: Path):
    config = load_config(exp_dir)
    config["train"]["epoch"] = 0
    config["log_level"] = "WARNING"
    config["train"]["log_dir"] = str(OUT_DIR / "_scratch" / exp_dir.name)
    return Supervisor(**config)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mean_epoch_time(exp_dir: Path):
    times = parse_epoch_times(exp_dir / "info.log")
    if not times:
        return None, None
    return statistics.mean(times), statistics.median(times)


def measure_runtime(supervisor: Supervisor, train_iters=5, infer_iters=20):
    train_loader = supervisor._data["train_loader"]
    val_loader = supervisor._data["val_loader"]
    train_x, train_y = next(iter(train_loader))
    infer_x, _ = next(iter(val_loader))
    train_x, train_y = supervisor._prepare_data(train_x, train_y)
    infer_x = supervisor._prepare_x(infer_x)

    optimizer = torch.optim.Adam(supervisor.model.parameters(), lr=1e-3)
    device = supervisor._device

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    supervisor.model.train()
    for _ in range(2):
        optimizer.zero_grad()
        output = supervisor.model(train_x, train_y, batches_seen=0)
        loss, _, _ = supervisor._compute_loss(train_y.clone(), output)
        loss.backward()
        optimizer.step()
    sync()

    train_times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(train_iters):
        optimizer.zero_grad()
        sync()
        start = time.perf_counter()
        output = supervisor.model(train_x, train_y, batches_seen=0)
        loss, _, _ = supervisor._compute_loss(train_y.clone(), output)
        loss.backward()
        optimizer.step()
        sync()
        train_times.append(time.perf_counter() - start)
    train_peak_mem = float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else None

    supervisor.model.eval()
    with torch.no_grad():
        for _ in range(3):
            _ = supervisor.model(infer_x)
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        infer_times = []
        for _ in range(infer_iters):
            sync()
            start = time.perf_counter()
            _ = supervisor.model(infer_x)
            sync()
            infer_times.append(time.perf_counter() - start)
    infer_peak_mem = float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else None

    return {
        "train_step_ms": statistics.mean(train_times) * 1000.0,
        "infer_batch_ms": statistics.mean(infer_times) * 1000.0,
        "train_peak_mem_mb": train_peak_mem,
        "infer_peak_mem_mb": infer_peak_mem,
    }


def collect_variant_metrics(tag: str, exp_dir: Path):
    supervisor = build_supervisor(exp_dir)
    try:
        epoch_mean, epoch_median = mean_epoch_time(exp_dir)
        runtime = measure_runtime(supervisor)
        return {
            "tag": tag,
            "params": count_params(supervisor.model),
            "mean_epoch_seconds": epoch_mean,
            "median_epoch_seconds": epoch_median,
            **runtime,
        }
    finally:
        supervisor._logger.handlers.clear()
        del supervisor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_markdown(rows):
    lines = [
        "# Efficiency Analysis",
        "",
        "| Dataset | Variant | Params | Mean Epoch Time (s) | Train Step (ms) | Inference Batch (ms) | Train Peak Mem (MB) | Inference Peak Mem (MB) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        epoch_seconds = "n/a" if row["mean_epoch_seconds"] is None else f"{row['mean_epoch_seconds']:.2f}"
        train_mem = "n/a" if row["train_peak_mem_mb"] is None else f"{row['train_peak_mem_mb']:.1f}"
        infer_mem = "n/a" if row["infer_peak_mem_mb"] is None else f"{row['infer_peak_mem_mb']:.1f}"
        lines.append(
            f"| {dataset_label(row['dataset'])} | {row['variant']} | {row['params']:,} | "
            f"{epoch_seconds} | {row['train_step_ms']:.2f} | {row['infer_batch_ms']:.2f} | "
            f"{train_mem} | {infer_mem} |"
        )
    return "\n".join(lines) + "\n"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset_name in DATASETS:
        variant_paths = [
            ("Original CLCRN", ORIGINAL_EXPERIMENTS[dataset_name]),
            ("Final Selected Model", BEST_FINAL_EXPERIMENTS[dataset_name]),
        ]
        if CONTROL_NEW_SCHEDULE_EXPERIMENTS[dataset_name].exists():
            variant_paths.append(("CLCRN + New Schedule", CONTROL_NEW_SCHEDULE_EXPERIMENTS[dataset_name]))
        trusted_improved = TRUSTED_IMPROVED_EXPERIMENTS[dataset_name]
        if trusted_improved != BEST_FINAL_EXPERIMENTS[dataset_name]:
            variant_paths.append(("ASTTN Full Model", trusted_improved))
        for variant_label, exp_dir in variant_paths:
            metrics = collect_variant_metrics(variant_label, exp_dir)
            metrics["dataset"] = dataset_name
            metrics["variant"] = variant_label
            rows.append(metrics)

    payload = {"results": rows}
    (OUT_DIR / "efficiency_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT_DIR / "efficiency_summary.md").write_text(build_markdown(rows), encoding="utf-8")
    print(f"Saved efficiency summary to {OUT_DIR}")


if __name__ == "__main__":
    main()
