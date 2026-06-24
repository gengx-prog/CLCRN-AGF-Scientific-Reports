import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.publication_utils import ASTTN_INTERPRETABLE_EXPERIMENTS, dataset_label, load_config, load_summary
from supervisor import Supervisor


OUT_DIR = ROOT / "experiments" / "weatherbench_publication_interpretability"
GATE_DATASETS = ["humidity", "component_of_wind", "cloud_cover"]
CASE_DATASETS = ["humidity", "cloud_cover"]


def build_supervisor(exp_dir: Path):
    config = load_config(exp_dir)
    config["train"]["epoch"] = 0
    config["log_level"] = "WARNING"
    return Supervisor(**config)


def load_lonlat(exp_dir: Path):
    position_file = Path(load_config(exp_dir)["data"]["position_file"])
    with open(position_file, "rb") as f:
        return pickle.load(f)["lonlat"]


def adaptive_graph_matrix(exp_dir: Path):
    best_epoch = int(load_summary(exp_dir)["best_epoch"])
    supervisor = build_supervisor(exp_dir)
    try:
        supervisor.load_model(best_epoch)
        encoder = supervisor.model.asttn_encoder
        adjacency = torch.softmax(encoder.node_emb_src @ encoder.node_emb_dst.T, dim=-1).detach().cpu().numpy()
    finally:
        supervisor._logger.handlers.clear()
        del supervisor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return adjacency


def gate_distribution(exp_dir: Path):
    best_epoch = int(load_summary(exp_dir)["best_epoch"])
    supervisor = build_supervisor(exp_dir)
    gate_values = []
    try:
        supervisor.load_model(best_epoch)
        model = supervisor.model.eval()
        encoder = model.asttn_encoder
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(supervisor._data["test_loader"]):
                if batch_idx >= 6:
                    break
                x = supervisor._prepare_x(x)
                signal_inputs = x[..., :model.input_dim]
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
    finally:
        supervisor._logger.handlers.clear()
        del supervisor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(gate_values)


def prediction_case(exp_dir: Path):
    best_epoch = int(load_summary(exp_dir)["best_epoch"])
    supervisor = build_supervisor(exp_dir)
    try:
        supervisor.load_model(best_epoch)
        model = supervisor.model.eval()
        with torch.no_grad():
            x, y = next(iter(supervisor._data["test_loader"]))
            x, y = supervisor._prepare_data(x, y)
            output = model(x)
            y_true, y_pred = supervisor._convert_scale(y.clone(), output.clone())
            y_true = y_true[:, 0].detach().cpu().numpy()
            y_pred = y_pred[:, 0].detach().cpu().numpy()
    finally:
        supervisor._logger.handlers.clear()
        del supervisor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return y_true, y_pred


def save_heatmap(matrix, dataset_name: str):
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    sns.heatmap(matrix, cmap="mako", cbar_kws={"label": "Adaptive Edge Weight"}, ax=ax)
    ax.set_title(f"{dataset_label(dataset_name)} Adaptive Graph Heatmap")
    fig.savefig(OUT_DIR / f"fig_adaptive_graph_{dataset_name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / f"fig_adaptive_graph_{dataset_name}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_gate_distribution(rows):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    plot_rows = []
    for row in rows:
        clipped = np.clip(row["values"], 0.0, 1.0)
        sampled = clipped[:: max(1, len(clipped) // 4000)]
        plot_rows.extend([(dataset_label(row["dataset"]), value) for value in sampled])
    labels = [item[0] for item in plot_rows]
    values = [item[1] for item in plot_rows]
    sns.violinplot(x=labels, y=values, inner="quartile", cut=0, palette="crest", ax=ax)
    ax.set_ylabel("Mean Gate Activation")
    ax.set_xlabel("")
    ax.set_title("Gate Activation Distributions Across Datasets")
    fig.savefig(OUT_DIR / "fig_gate_distribution.png", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / "fig_gate_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def save_prediction_case(dataset_name: str, lonlat, y_true, y_pred):
    horizons = [2, 5, 11]
    fig, axes = plt.subplots(len(horizons), 3, figsize=(13.0, 10.0), sharex=True, sharey=True)
    vmin = min(y_true[h].min() for h in horizons)
    vmax = max(y_true[h].max() for h in horizons)
    err_vmax = max(np.abs(y_pred[h] - y_true[h]).max() for h in horizons)
    for row_idx, horizon_idx in enumerate(horizons):
        truth = y_true[horizon_idx, :, 0]
        pred = y_pred[horizon_idx, :, 0]
        err = np.abs(pred - truth)
        for col_idx, (values, title, cmap, limits) in enumerate(
            [
                (truth, f"T+{horizon_idx + 1} Truth", "viridis", (vmin, vmax)),
                (pred, f"T+{horizon_idx + 1} Prediction", "viridis", (vmin, vmax)),
                (err, f"T+{horizon_idx + 1} Abs Error", "magma", (0, err_vmax)),
            ]
        ):
            ax = axes[row_idx, col_idx]
            scatter = ax.scatter(
                lonlat[:, 0],
                lonlat[:, 1],
                c=values,
                s=8,
                cmap=cmap,
                vmin=limits[0],
                vmax=limits[1],
                linewidths=0,
            )
            ax.set_title(title)
            if row_idx == len(horizons) - 1:
                ax.set_xlabel("Longitude")
            if col_idx == 0:
                ax.set_ylabel("Latitude")
            fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"{dataset_label(dataset_name)} Prediction Case Study", y=1.01, fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"fig_prediction_case_{dataset_name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / f"fig_prediction_case_{dataset_name}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_gallery():
    lines = [
        "# Interpretability Figures",
        "",
        "- `fig_adaptive_graph_humidity.*`: humidity adaptive graph heatmap.",
        "- `fig_adaptive_graph_component_of_wind.*`: wind adaptive graph heatmap.",
        "- `fig_gate_distribution.*`: gate activation distributions across datasets.",
        "- `fig_prediction_case_humidity.*`: humidity truth/prediction/error maps.",
        "- `fig_prediction_case_cloud_cover.*`: cloud cover truth/prediction/error maps.",
        "",
    ]
    (OUT_DIR / "figure_gallery.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name in ["humidity", "component_of_wind"]:
        adjacency = adaptive_graph_matrix(ASTTN_INTERPRETABLE_EXPERIMENTS[dataset_name])
        step = max(1, adjacency.shape[0] // 64)
        sampled = adjacency[::step, ::step][:64, :64]
        save_heatmap(sampled, dataset_name)

    gate_rows = []
    for dataset_name in GATE_DATASETS:
        gate_rows.append(
            {
                "dataset": dataset_name,
                "values": gate_distribution(ASTTN_INTERPRETABLE_EXPERIMENTS[dataset_name]),
            }
        )
    save_gate_distribution(gate_rows)

    for dataset_name in CASE_DATASETS:
        exp_dir = ASTTN_INTERPRETABLE_EXPERIMENTS[dataset_name]
        lonlat = load_lonlat(exp_dir)
        y_true, y_pred = prediction_case(exp_dir)
        save_prediction_case(dataset_name, lonlat, y_true, y_pred)

    write_gallery()
    (OUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "heatmaps": ["humidity", "component_of_wind"],
                "gate_distribution_datasets": GATE_DATASETS,
                "case_datasets": CASE_DATASETS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved interpretability figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
