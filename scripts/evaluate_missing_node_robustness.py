import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.loss import masked_mae_loss, masked_mape_loss, masked_mse_loss
from scripts.publication_utils import BEST_FINAL_EXPERIMENTS, dataset_label, load_config, load_summary
from supervisor import Supervisor


OUT_DIR = ROOT / "experiments" / "weatherbench_publication_robustness_missing"
DATASETS = ["temperature", "humidity", "component_of_wind", "cloud_cover"]
MISSING_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4]


def build_supervisor(exp_dir: Path):
    config = load_config(exp_dir)
    config["train"]["epoch"] = 0
    config["log_level"] = "WARNING"
    return Supervisor(**config)


def evaluate_missing_nodes(supervisor: Supervisor, best_epoch: int, missing_ratio: float, seed: int = 7):
    supervisor.load_model(best_epoch)
    generator_device = supervisor._device if supervisor._device.type == "cuda" else "cpu"
    rng = torch.Generator(device=generator_device)
    rng.manual_seed(seed)

    with torch.no_grad():
        supervisor.model = supervisor.model.eval()
        losses = []
        y_truths = []
        y_preds = []
        for x, y in supervisor._data["test_loader"]:
            x, y = supervisor._prepare_data(x, y)
            if missing_ratio > 0:
                seq_len, batch_size, node_num, _ = x.shape
                missing_count = max(1, int(node_num * missing_ratio))
                mask = torch.ones((batch_size, node_num), device=x.device, dtype=x.dtype)
                for batch_idx in range(batch_size):
                    dropped = torch.randperm(node_num, device=x.device, generator=rng)[:missing_count]
                    mask[batch_idx, dropped] = 0.0
                x = x * mask[None, :, :, None]

            output = supervisor.model(x)
            loss, y_true, y_pred = supervisor._compute_loss(y.clone(), output)
            losses.append(loss.item())
            y_truths.append(y_true.cpu())
            y_preds.append(y_pred.cpu())

        y_preds = torch.cat(y_preds, dim=1)
        y_truths = torch.cat(y_truths, dim=1)
        return {
            "missing_ratio": missing_ratio,
            "mae": float(masked_mae_loss(y_preds, y_truths).item()),
            "rmse": float(math.sqrt(masked_mse_loss(y_preds, y_truths).item())),
            "mape": float(masked_mape_loss(y_preds, y_truths).item()),
            "mean_loss": float(np.mean(losses)),
        }


def build_markdown(rows):
    lines = [
        "# Missing-Node Robustness",
        "",
        "| Dataset | Missing Ratio | Test MAE | Test RMSE | Test MAPE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        for item in row["curve"]:
            lines.append(
                f"| {dataset_label(row['dataset'])} | {item['missing_ratio']:.1f} | "
                f"{item['mae']:.4f} | {item['rmse']:.4f} | {item['mape']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset_name in DATASETS:
        exp_dir = BEST_FINAL_EXPERIMENTS[dataset_name]
        best_epoch = int(load_summary(exp_dir)["best_epoch"])
        supervisor = build_supervisor(exp_dir)
        try:
            curve = [evaluate_missing_nodes(supervisor, best_epoch, ratio) for ratio in MISSING_RATIOS]
        finally:
            supervisor._logger.handlers.clear()
            del supervisor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows.append({"dataset": dataset_name, "curve": curve})

    payload = {"missing_ratios": MISSING_RATIOS, "results": rows}
    (OUT_DIR / "robustness_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT_DIR / "robustness_summary.md").write_text(build_markdown(rows), encoding="utf-8")

    from scripts.replot_missing_node_robustness import load_frame, plot_metric

    frame = load_frame()
    plot_metric(frame, "mae", "fig_missing_nodes_mae")
    plot_metric(frame, "rmse", "fig_missing_nodes_rmse")
    print(f"Saved robustness summary to {OUT_DIR}")


if __name__ == "__main__":
    main()
