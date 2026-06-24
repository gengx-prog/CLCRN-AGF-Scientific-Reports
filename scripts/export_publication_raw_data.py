import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.loss import masked_mae_loss, masked_mape_loss, masked_mse_loss
from scripts.publication_utils import (
    ASTTN_INTERPRETABLE_EXPERIMENTS,
    BEST_FINAL_EXPERIMENTS,
    dataset_label,
    load_config,
    load_json,
    load_summary,
)
from supervisor import Supervisor


OUT_DIR = ROOT / "experiments" / "weatherbench_publication_raw_data"
TIME_SERIES_DIR = OUT_DIR / "time_series_cases"
HORIZON_DIR = OUT_DIR / "horizon_metrics"
ABLATION_DIR = OUT_DIR / "ablation_tables"
INTERP_DIR = OUT_DIR / "interpretability_matrices"

FINAL_DATASETS = ["temperature", "humidity", "component_of_wind", "cloud_cover"]
HEATMAP_DATASETS = ["humidity", "component_of_wind", "cloud_cover"]


def ensure_dirs():
    for path in [OUT_DIR, TIME_SERIES_DIR, HORIZON_DIR, ABLATION_DIR, INTERP_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def build_supervisor(exp_dir: Path):
    config = load_config(exp_dir)
    config["train"]["epoch"] = 0
    config["log_level"] = "WARNING"
    return Supervisor(**config)


def load_lonlat(exp_dir: Path):
    position_file = Path(load_config(exp_dir)["data"]["position_file"])
    with open(position_file, "rb") as f:
        return pickle.load(f)["lonlat"]


def inverse_input_scale(supervisor: Supervisor, x: torch.Tensor):
    x_signal = x[..., : supervisor.input_dim].clone()
    for out_dim in range(supervisor.output_dim):
        x_signal[..., out_dim] = supervisor.standard_scaler[out_dim].inverse_transform(x_signal[..., out_dim])
    return x_signal


def collect_predictions(supervisor: Supervisor, best_epoch: int):
    supervisor.load_model(best_epoch)
    y_truths = []
    y_preds = []
    with torch.no_grad():
        supervisor.model = supervisor.model.eval()
        for x, y in supervisor._data["test_loader"]:
            x, y = supervisor._prepare_data(x, y)
            output = supervisor.model(x)
            _, y_true, y_pred = supervisor._compute_loss(y.clone(), output.clone(), use_training_loss=False)
            y_truths.append(y_true.cpu())
            y_preds.append(y_pred.cpu())
    return torch.cat(y_truths, dim=1), torch.cat(y_preds, dim=1)


def metric_row(y_true: torch.Tensor, y_pred: torch.Tensor):
    mse = masked_mse_loss(y_pred, y_true).item()
    return {
        "mae": float(masked_mae_loss(y_pred, y_true).item()),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mape": float(masked_mape_loss(y_pred, y_true).item()),
    }


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_time_series_cases():
    summary_rows = []
    for dataset_name in FINAL_DATASETS:
        exp_dir = BEST_FINAL_EXPERIMENTS[dataset_name]
        best_epoch = int(load_summary(exp_dir)["best_epoch"])
        lonlat = load_lonlat(exp_dir)
        supervisor = build_supervisor(exp_dir)
        try:
            supervisor.load_model(best_epoch)
            with torch.no_grad():
                supervisor.model = supervisor.model.eval()
                x, y = next(iter(supervisor._data["test_loader"]))
                x_prep, y_prep = supervisor._prepare_data(x, y)
                output = supervisor.model(x_prep)
                x_hist = inverse_input_scale(supervisor, x_prep).detach().cpu().numpy()
                y_true, y_pred = supervisor._convert_scale(y_prep.clone(), output.clone())
                y_true = y_true.detach().cpu().numpy()
                y_pred = y_pred.detach().cpu().numpy()
        finally:
            supervisor._logger.handlers.clear()
            del supervisor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Use the first test sample and choose the node with median future MAE.
        hist_sample = x_hist[:, 0, :, :]
        truth_sample = y_true[:, 0, :, :]
        pred_sample = y_pred[:, 0, :, :]
        node_mae = np.mean(np.abs(pred_sample - truth_sample), axis=(0, 2))
        median_error = np.median(node_mae)
        node_idx = int(np.argmin(np.abs(node_mae - median_error)))
        node_meta = {
            "node_index": node_idx,
            "longitude": float(lonlat[node_idx, 0]),
            "latitude": float(lonlat[node_idx, 1]),
            "selection_rule": "first test sample, node whose 12-step MAE is closest to the sample median",
            "node_future_mae": float(node_mae[node_idx]),
        }

        payload = {
            "dataset": dataset_name,
            "label": dataset_label(dataset_name),
            "experiment_dir": str(exp_dir),
            "best_epoch": best_epoch,
            "history_steps": list(range(-hist_sample.shape[0] + 1, 1)),
            "forecast_steps": list(range(1, truth_sample.shape[0] + 1)),
            "selected_node": node_meta,
            "selected_node_history": hist_sample[:, node_idx, :].tolist(),
            "selected_node_truth": truth_sample[:, node_idx, :].tolist(),
            "selected_node_prediction": pred_sample[:, node_idx, :].tolist(),
            "global_mean_history": hist_sample.mean(axis=1).tolist(),
            "global_mean_truth": truth_sample.mean(axis=1).tolist(),
            "global_mean_prediction": pred_sample.mean(axis=1).tolist(),
        }
        (TIME_SERIES_DIR / f"{dataset_name}_timeseries_case.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

        node_rows = []
        for idx, rel_step in enumerate(payload["history_steps"]):
            row = {"relative_step": rel_step, "phase": "history"}
            values = hist_sample[idx, node_idx, :]
            for dim_idx in range(values.shape[0]):
                row[f"value_dim{dim_idx}"] = float(values[dim_idx])
            node_rows.append(row)
        for idx, rel_step in enumerate(payload["forecast_steps"]):
            row = {"relative_step": rel_step, "phase": "forecast"}
            truth_vals = truth_sample[idx, node_idx, :]
            pred_vals = pred_sample[idx, node_idx, :]
            for dim_idx in range(truth_vals.shape[0]):
                row[f"truth_dim{dim_idx}"] = float(truth_vals[dim_idx])
                row[f"prediction_dim{dim_idx}"] = float(pred_vals[dim_idx])
            node_rows.append(row)
        node_fieldnames = sorted({key for row in node_rows for key in row.keys()}, key=lambda x: (x not in {"relative_step", "phase"}, x))
        write_csv(TIME_SERIES_DIR / f"{dataset_name}_selected_node.csv", node_fieldnames, node_rows)

        global_rows = []
        history_mean = hist_sample.mean(axis=1)
        truth_mean = truth_sample.mean(axis=1)
        pred_mean = pred_sample.mean(axis=1)
        for idx, rel_step in enumerate(payload["history_steps"]):
            row = {"relative_step": rel_step, "phase": "history"}
            for dim_idx in range(history_mean.shape[1]):
                row[f"value_dim{dim_idx}"] = float(history_mean[idx, dim_idx])
            global_rows.append(row)
        for idx, rel_step in enumerate(payload["forecast_steps"]):
            row = {"relative_step": rel_step, "phase": "forecast"}
            for dim_idx in range(truth_mean.shape[1]):
                row[f"truth_dim{dim_idx}"] = float(truth_mean[idx, dim_idx])
                row[f"prediction_dim{dim_idx}"] = float(pred_mean[idx, dim_idx])
            global_rows.append(row)
        global_fieldnames = sorted({key for row in global_rows for key in row.keys()}, key=lambda x: (x not in {"relative_step", "phase"}, x))
        write_csv(TIME_SERIES_DIR / f"{dataset_name}_global_mean.csv", global_fieldnames, global_rows)

        np.save(TIME_SERIES_DIR / f"{dataset_name}_sample0_history.npy", hist_sample)
        np.save(TIME_SERIES_DIR / f"{dataset_name}_sample0_truth.npy", truth_sample)
        np.save(TIME_SERIES_DIR / f"{dataset_name}_sample0_prediction.npy", pred_sample)

        summary_rows.append(
            {
                "dataset": dataset_name,
                "label": dataset_label(dataset_name),
                "selected_node_index": node_idx,
                "selected_node_lon": float(lonlat[node_idx, 0]),
                "selected_node_lat": float(lonlat[node_idx, 1]),
                "selected_node_future_mae": float(node_mae[node_idx]),
                "best_epoch": best_epoch,
            }
        )

    write_csv(
        TIME_SERIES_DIR / "timeseries_case_index.csv",
        [
            "dataset",
            "label",
            "selected_node_index",
            "selected_node_lon",
            "selected_node_lat",
            "selected_node_future_mae",
            "best_epoch",
        ],
        summary_rows,
    )
    return summary_rows


def export_horizon_metrics():
    aggregate = {"final_selected": []}
    csv_rows = []
    for dataset_name in FINAL_DATASETS:
        exp_dir = BEST_FINAL_EXPERIMENTS[dataset_name]
        best_epoch = int(load_summary(exp_dir)["best_epoch"])
        supervisor = build_supervisor(exp_dir)
        try:
            y_true, y_pred = collect_predictions(supervisor, best_epoch)
        finally:
            supervisor._logger.handlers.clear()
            del supervisor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        horizon = y_true.shape[0]
        rows = []
        for step in range(1, horizon + 1):
            exact = metric_row(y_true[step - 1 : step], y_pred[step - 1 : step])
            cumulative = metric_row(y_true[:step], y_pred[:step])
            row = {
                "dataset": dataset_name,
                "label": dataset_label(dataset_name),
                "step": step,
                "exact_mae": exact["mae"],
                "exact_mse": exact["mse"],
                "exact_rmse": exact["rmse"],
                "exact_mape": exact["mape"],
                "cumulative_mae": cumulative["mae"],
                "cumulative_mse": cumulative["mse"],
                "cumulative_rmse": cumulative["rmse"],
                "cumulative_mape": cumulative["mape"],
                "best_epoch": best_epoch,
            }
            rows.append(row)
            csv_rows.append(row)

        aggregate["final_selected"].append(
            {
                "dataset": dataset_name,
                "label": dataset_label(dataset_name),
                "experiment_dir": str(exp_dir),
                "best_epoch": best_epoch,
                "rows": rows,
            }
        )

    (HORIZON_DIR / "final_selected_horizon_metrics.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    write_csv(
        HORIZON_DIR / "final_selected_horizon_metrics.csv",
        [
            "dataset",
            "label",
            "step",
            "exact_mae",
            "exact_mse",
            "exact_rmse",
            "exact_mape",
            "cumulative_mae",
            "cumulative_mse",
            "cumulative_rmse",
            "cumulative_mape",
            "best_epoch",
        ],
        csv_rows,
    )
    return aggregate


def export_ablation_tables():
    ablation_20 = load_json(ROOT / "experiments" / "weatherbench_asttn_ablation" / "aggregate_results.json")
    rows_20 = []
    for item in ablation_20["results"]:
        row = {
            "dataset": item["dataset"],
            "label": dataset_label(item["dataset"]),
            "variant": item["variant"],
            "variant_label": item["variant_label"],
            "epochs": ablation_20["epochs"],
            "best_epoch": item["best_epoch"],
            "best_val_mae": item["best_val_mae"],
            "test_mae": item["mae"],
            "test_rmse": item["rmse"],
            "test_mape": item["mape"],
        }
        for key, value in item["step_metrics"].items():
            row[key] = value
        rows_20.append(row)
    write_csv(
        ABLATION_DIR / "ablation_20ep_all_datasets.csv",
        sorted({key for row in rows_20 for key in row.keys()}),
        rows_20,
    )
    (ABLATION_DIR / "ablation_20ep_all_datasets.json").write_text(
        json.dumps(rows_20, indent=2), encoding="utf-8"
    )

    ablation_100 = load_json(ROOT / "experiments" / "weatherbench_asttn_ablation_temp100" / "aggregate_results.json")
    rows_100 = []
    for item in ablation_100["results"]:
        row = {
            "dataset": item["dataset"],
            "label": dataset_label(item["dataset"]),
            "variant": item["variant"],
            "variant_label": item["variant_label"],
            "epochs": ablation_100["epochs"],
            "best_epoch": item["best_epoch"],
            "best_val_mae": item["best_val_mae"],
            "test_mae": item["mae"],
            "test_rmse": item["rmse"],
            "test_mape": item["mape"],
        }
        for key, value in item["step_metrics"].items():
            row[key] = value
        rows_100.append(row)
    write_csv(
        ABLATION_DIR / "ablation_temperature_100ep.csv",
        sorted({key for row in rows_100 for key in row.keys()}),
        rows_100,
    )
    (ABLATION_DIR / "ablation_temperature_100ep.json").write_text(
        json.dumps(rows_100, indent=2), encoding="utf-8"
    )

    comparison = load_json(
        ROOT / "experiments" / "weatherbench_asttn_ablation_temp100" / "comparison_20ep_vs_100ep.json"
    )
    (ABLATION_DIR / "ablation_temperature_20ep_vs_100ep.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    comparison_rows = []
    for item in comparison["comparisons"]:
        comparison_rows.append(
            {
                "variant": item["variant"],
                "variant_label": item["variant_label"],
                "old_20ep_best_val_mae": item["old_20ep"]["best_val_mae"],
                "old_20ep_test_mae": item["old_20ep"]["mae"],
                "old_20ep_test_rmse": item["old_20ep"]["rmse"],
                "new_100ep_best_val_mae": item["new_100ep"]["best_val_mae"],
                "new_100ep_test_mae": item["new_100ep"]["mae"],
                "new_100ep_test_rmse": item["new_100ep"]["rmse"],
                "delta_mae": item["delta_100ep_minus_20ep"]["mae"],
                "delta_rmse": item["delta_100ep_minus_20ep"]["rmse"],
            }
        )
    write_csv(
        ABLATION_DIR / "ablation_temperature_20ep_vs_100ep.csv",
        list(comparison_rows[0].keys()),
        comparison_rows,
    )
    return {"ablation_20_rows": len(rows_20), "ablation_100_rows": len(rows_100)}


def adaptive_graph_matrix(supervisor: Supervisor):
    encoder = supervisor.model.asttn_encoder
    adjacency = torch.softmax(encoder.node_emb_src @ encoder.node_emb_dst.T, dim=-1)
    return adjacency.detach().cpu().numpy()


def gate_distribution(supervisor: Supervisor, max_batches: int = 6):
    gate_values = []
    model = supervisor.model.eval()
    encoder = model.asttn_encoder
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(supervisor._data["test_loader"]):
            if batch_idx >= max_batches:
                break
            x = supervisor._prepare_x(x)
            signal_inputs = x[..., : model.input_dim]
            feature_emb = model.feature_embedding(signal_inputs)
            batch_size, seq_len, node_num, _ = feature_emb.shape
            node_emb = model.node_embeddings[None, None, :, :].expand(batch_size, seq_len, node_num, model.embed_dim)
            adaptive_inputs = torch.cat([feature_emb, node_emb], dim=-1)
            topk_indices, topk_values = encoder._adaptive_topk()
            flat_neighbor_indices = topk_indices.reshape(-1)
            neighbor_x = adaptive_inputs[:, :, flat_neighbor_indices, :].reshape(
                batch_size, seq_len, node_num, encoder.topk, adaptive_inputs.size(-1)
            )
            mixed = torch.sum(neighbor_x * topk_values[None, None, :, :, None], dim=3)
            mixed = encoder.out_proj(mixed)
            residual = encoder.residual_proj(adaptive_inputs)
            gate = torch.sigmoid(encoder.gate_proj(torch.cat([residual, mixed], dim=-1)))
            gate_values.append(gate.mean(dim=-1).reshape(-1).detach().cpu().numpy())
    return np.concatenate(gate_values)


def export_interpretability_matrices():
    manifest_rows = []
    gate_index_rows = []
    for dataset_name in HEATMAP_DATASETS:
        exp_dir = ASTTN_INTERPRETABLE_EXPERIMENTS[dataset_name]
        best_epoch = int(load_summary(exp_dir)["best_epoch"])
        supervisor = build_supervisor(exp_dir)
        try:
            supervisor.load_model(best_epoch)
            node_num = int(supervisor.model.node_embeddings.shape[0])
            sparse_idx = supervisor._data["kernel_info"]["sparse_idx"]
            geo_flat = np.asarray(supervisor._data["kernel_info"]["geodesic"]).reshape(-1)
            row_idx = sparse_idx[0]
            col_idx = sparse_idx[1]

            geo_binary = np.zeros((node_num, node_num), dtype=np.float32)
            geo_binary[row_idx, col_idx] = 1.0

            geo_distance = np.full((node_num, node_num), np.nan, dtype=np.float32)
            geo_distance[row_idx, col_idx] = geo_flat.astype(np.float32)

            adaptive = adaptive_graph_matrix(supervisor).astype(np.float32)
            gate_vals = gate_distribution(supervisor).astype(np.float32)
        finally:
            supervisor._logger.handlers.clear()
            del supervisor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        sample_step = max(1, node_num // 64)
        sample_index = np.arange(0, node_num, sample_step)[:64]
        sampled_geo_binary = geo_binary[np.ix_(sample_index, sample_index)]
        sampled_geo_distance = geo_distance[np.ix_(sample_index, sample_index)]
        sampled_adaptive = adaptive[np.ix_(sample_index, sample_index)]

        np.save(INTERP_DIR / f"{dataset_name}_geographic_binary_full.npy", geo_binary)
        np.save(INTERP_DIR / f"{dataset_name}_geographic_distance_full.npy", geo_distance)
        np.save(INTERP_DIR / f"{dataset_name}_adaptive_attention_full.npy", adaptive)
        np.save(INTERP_DIR / f"{dataset_name}_gate_values.npy", gate_vals)
        np.save(INTERP_DIR / f"{dataset_name}_sampled_node_indices.npy", sample_index)

        np.savetxt(INTERP_DIR / f"{dataset_name}_geographic_binary_sample64.csv", sampled_geo_binary, delimiter=",")
        np.savetxt(INTERP_DIR / f"{dataset_name}_geographic_distance_sample64.csv", sampled_geo_distance, delimiter=",")
        np.savetxt(INTERP_DIR / f"{dataset_name}_adaptive_attention_sample64.csv", sampled_adaptive, delimiter=",")

        edge_rows = [
            {"source": int(src), "target": int(dst), "geodesic_distance": float(dist)}
            for src, dst, dist in zip(row_idx.tolist(), col_idx.tolist(), geo_flat.tolist())
        ]
        write_csv(
            INTERP_DIR / f"{dataset_name}_geographic_edge_list.csv",
            ["source", "target", "geodesic_distance"],
            edge_rows,
        )

        gate_index_rows.append(
            {
                "dataset": dataset_name,
                "count": int(gate_vals.size),
                "mean": float(gate_vals.mean()),
                "std": float(gate_vals.std()),
                "min": float(gate_vals.min()),
                "max": float(gate_vals.max()),
            }
        )
        manifest_rows.append(
            {
                "dataset": dataset_name,
                "best_epoch": best_epoch,
                "node_count": node_num,
                "sampled_size": int(sample_index.size),
                "edge_count": int(len(edge_rows)),
            }
        )

    write_csv(
        INTERP_DIR / "interpretability_matrix_index.csv",
        ["dataset", "best_epoch", "node_count", "sampled_size", "edge_count"],
        manifest_rows,
    )
    write_csv(
        INTERP_DIR / "gate_distribution_summary.csv",
        ["dataset", "count", "mean", "std", "min", "max"],
        gate_index_rows,
    )
    return manifest_rows


def write_summary(timeseries_rows, horizon_payload, ablation_counts, interp_rows):
    lines = [
        "# Publication Raw Data Export",
        "",
        "This directory stores raw numeric data needed to recreate publication figures and tables.",
        "",
        "## 1. Time-series cases",
        "",
        "- Directory: `time_series_cases/`",
        "- Each dataset includes:",
        "  - `*_timeseries_case.json`: selected node + global-mean sequences",
        "  - `*_selected_node.csv`: past-12 / future-12 numeric points for one representative node",
        "  - `*_global_mean.csv`: past-12 / future-12 numeric points after averaging over all nodes",
        "  - `*_sample0_history.npy`, `*_sample0_truth.npy`, `*_sample0_prediction.npy`: full sample arrays",
        "",
        "## 2. Horizon metrics",
        "",
        "- Directory: `horizon_metrics/`",
        "- `final_selected_horizon_metrics.csv`: per-step exact and cumulative MAE/MSE/RMSE/MAPE for steps 1..12",
        "",
        "## 3. Ablation tables",
        "",
        "- Directory: `ablation_tables/`",
        f"- Exported 20-epoch ablation rows: `{ablation_counts['ablation_20_rows']}`",
        f"- Exported temperature 100-epoch ablation rows: `{ablation_counts['ablation_100_rows']}`",
        "",
        "## 4. Interpretability matrices",
        "",
        "- Directory: `interpretability_matrices/`",
        "- Includes full `adaptive_attention`, `geographic_binary`, `geographic_distance` matrices, sampled 64x64 CSVs, gate values, and edge lists.",
        "",
        "## 5. Dataset index",
        "",
        "| Dataset | Selected Node | Best Epoch |",
        "| --- | ---: | ---: |",
    ]
    for row in timeseries_rows:
        lines.append(
            f"| {dataset_label(row['dataset'])} | {row['selected_node_index']} | {row['best_epoch']} |"
        )
    lines += [
        "",
        f"Horizon metric datasets exported: `{len(horizon_payload['final_selected'])}`",
        f"Interpretability datasets exported: `{len(interp_rows)}`",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_dirs()
    timeseries_rows = export_time_series_cases()
    horizon_payload = export_horizon_metrics()
    ablation_counts = export_ablation_tables()
    interp_rows = export_interpretability_matrices()
    write_summary(timeseries_rows, horizon_payload, ablation_counts, interp_rows)
    print(f"Saved raw publication data to {OUT_DIR}")


if __name__ == "__main__":
    main()
