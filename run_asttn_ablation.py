import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from run_clcrn_weatherbench import build_config, load_base_config, run_single_dataset, set_seed


DATASETS = ["temperature", "humidity", "component_of_wind", "cloud_cover"]
VARIANTS = [
    {
        "id": "wo_adaptive_graph",
        "label": "w/o Adaptive Graph",
        "model_overrides": {
            "use_asttn_encoder": False,
        },
    },
    {
        "id": "wo_gated_fusion",
        "label": "w/o Gated Fusion",
        "model_overrides": {
            "use_asttn_encoder": True,
            "asttn_fusion_mode": "mean",
        },
    },
    {
        "id": "full_model",
        "label": "Full Model",
        "model_overrides": {
            "use_asttn_encoder": True,
            "asttn_fusion_mode": "gated",
        },
    },
]
VARIANT_IDS = [variant["id"] for variant in VARIANTS]


def parse_args():
    parser = argparse.ArgumentParser(description="Run ASTTN-style ablation experiments on WeatherBench.")
    parser.add_argument(
        "--data-root",
        default=r"G:\Weather Bench_dataset",
        help="Root directory that contains the preprocessed WeatherBench subsets.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=DATASETS,
        help="Datasets to run for ablation.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANT_IDS,
        default=VARIANT_IDS,
        help="Ablation variants to run. Defaults to all variants.",
    )
    parser.add_argument(
        "--base-config",
        default="./experiments/config_clcrn_asttn_ab_improved.yaml",
        help="Base improved ASTTN config file.",
    )
    parser.add_argument(
        "--log-root",
        default="./experiments/weatherbench_asttn_ablation",
        help="Directory used to store ablation outputs.",
    )
    parser.add_argument(
        "--reference-original",
        default="./experiments/weatherbench_clcrn_paper_run/summary_all.json",
        help="Reference file for the original reproduced CLCRN results.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--seed", type=int, default=2022, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs used for each ablation run.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional train batch size override.")
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=None,
        help="Optional validation/test batch size override.",
    )
    parser.add_argument(
        "--test-every",
        type=int,
        default=10,
        help="Test evaluation frequency in epochs.",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Enable CUDA automatic mixed precision during training.",
    )
    return parser.parse_args()


def make_run_args(parsed_args):
    return SimpleNamespace(
        data_root=parsed_args.data_root,
        gpu=parsed_args.gpu,
        use_context=None,
        context_dim=None,
        context_embed_dim=None,
        use_sgp_encoder=None,
        sgp_orders=None,
        batch_size=parsed_args.batch_size,
        test_batch_size=parsed_args.test_batch_size,
        num_workers=parsed_args.num_workers,
        log_root=str(parsed_args.log_root),
        epochs=parsed_args.epochs,
        test_every=parsed_args.test_every,
        use_amp=parsed_args.use_amp,
    )


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_progress(path, payload):
    write_json(path.with_suffix(".json"), payload)
    lines = [
        f"status: {payload['status']}",
        f"completed: {payload['completed_runs']}/{payload['total_runs']}",
        f"current_variant: {payload.get('current_variant', '')}",
        f"current_dataset: {payload.get('current_dataset', '')}",
    ]
    completed_items = payload.get("completed_items", [])
    if completed_items:
        lines.append("")
        lines.append("completed_items:")
        lines.extend(f"- {item}" for item in completed_items)
    path.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_variant_config(base_config, run_args, dataset_name, variant, nest_variant_dir=True):
    config = copy.deepcopy(build_config(base_config, run_args, dataset_name))
    for key, value in variant["model_overrides"].items():
        config["model"][key] = value
    config["train"]["experiment_name"] = f"CLCRN_{dataset_name}_{variant['id']}"
    config["train"]["use_amp"] = bool(getattr(run_args, "use_amp", False))
    if nest_variant_dir:
        config["train"]["log_dir"] = str(Path(run_args.log_root) / variant["id"] / dataset_name)
    else:
        config["train"]["log_dir"] = str(Path(run_args.log_root) / dataset_name)
    return config


def build_markdown(results_by_dataset, original_reference, epochs):
    lines = [
        f"# ASTTN Ablation Results ({epochs} epochs)",
        "",
        "## Variant Definitions",
        "",
        "- `wo_adaptive_graph`: remove the adaptive graph branch and keep the improved training schedule.",
        "- `wo_gated_fusion`: keep the adaptive graph branch but replace the learned gate with simple mean fusion.",
        "- `full_model`: adaptive graph branch with learned gated fusion.",
        "",
    ]
    for dataset_name, items in results_by_dataset.items():
        full_item = next((item for item in items if item["variant"] == "full_model"), None)
        lines.extend(
            [
                f"## {dataset_name}",
                "",
            ]
        )
        if full_item is not None:
            lines.extend(
                [
                    "| Variant | Best Val MAE | Test MAE | Test RMSE | Delta MAE vs Full | Delta RMSE vs Full |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
        else:
            lines.extend(
                [
                    "| Variant | Best Val MAE | Test MAE | Test RMSE |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
        for item in items:
            if full_item is not None:
                lines.append(
                    "| {label} | {best_val_mae:.4f} | {mae:.4f} | {rmse:.4f} | {delta_mae:+.4f} | {delta_rmse:+.4f} |".format(
                        label=item["variant_label"],
                        best_val_mae=item["best_val_mae"],
                        mae=item["mae"],
                        rmse=item["rmse"],
                        delta_mae=item["mae"] - full_item["mae"],
                        delta_rmse=item["rmse"] - full_item["rmse"],
                    )
                )
            else:
                lines.append(
                    "| {label} | {best_val_mae:.4f} | {mae:.4f} | {rmse:.4f} |".format(
                        label=item["variant_label"],
                        best_val_mae=item["best_val_mae"],
                        mae=item["mae"],
                        rmse=item["rmse"],
                    )
                )
        baseline = original_reference.get(dataset_name)
        if baseline is not None:
            lines.extend(
                [
                    "",
                    "Original reproduction reference:",
                    f"- `MAE = {baseline['mae']:.4f}`",
                    f"- `RMSE = {baseline['rmse']:.4f}`",
                    "",
                ]
            )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    set_seed(args.seed)
    root = Path(args.log_root)
    base_config = load_base_config(args.base_config)
    run_args = make_run_args(args)
    original_reference = {item["dataset"]: item for item in load_json(args.reference_original)}
    selected_variants = [variant for variant in VARIANTS if variant["id"] in args.variants]
    nest_variant_dir = len(selected_variants) > 1

    total_runs = len(args.datasets) * len(selected_variants)
    completed_items = []
    write_progress(
        root / "progress",
        {
            "status": "running",
            "completed_runs": 0,
            "total_runs": total_runs,
            "current_variant": None,
            "current_dataset": None,
            "completed_items": completed_items,
        },
    )

    all_results = []
    results_by_dataset = {dataset_name: [] for dataset_name in args.datasets}
    for variant in selected_variants:
        for dataset_name in args.datasets:
            # Re-seed each run so every ablation variant starts from the same RNG state.
            set_seed(args.seed)
            write_progress(
                root / "progress",
                {
                    "status": "running",
                    "completed_runs": len(completed_items),
                    "total_runs": total_runs,
                    "current_variant": variant["id"],
                    "current_dataset": dataset_name,
                    "completed_items": completed_items,
                },
            )
            config = build_variant_config(base_config, run_args, dataset_name, variant, nest_variant_dir=nest_variant_dir)
            summary = run_single_dataset(config, dataset_name)
            summary["variant"] = variant["id"]
            summary["variant_label"] = variant["label"]
            all_results.append(summary)
            results_by_dataset[dataset_name].append(summary)
            completed_items.append(f"{variant['id']}/{dataset_name}")

    comparisons = []
    for dataset_name, items in results_by_dataset.items():
        full_item = next((item for item in items if item["variant"] == "full_model"), None)
        entry = {"dataset": dataset_name}
        for item in items:
            entry[item["variant"]] = {
                "mae": item["mae"],
                "rmse": item["rmse"],
                "best_val_mae": item["best_val_mae"],
            }
        entry["original"] = {
            "mae": original_reference[dataset_name]["mae"],
            "rmse": original_reference[dataset_name]["rmse"],
        }
        if full_item is not None:
            entry["full_vs_original"] = {
                "delta_mae": full_item["mae"] - original_reference[dataset_name]["mae"],
                "delta_rmse": full_item["rmse"] - original_reference[dataset_name]["rmse"],
            }
        comparisons.append(entry)

    aggregate = {
        "seed": args.seed,
        "epochs": args.epochs,
        "datasets": args.datasets,
        "variants": selected_variants,
        "results": all_results,
        "comparisons": comparisons,
    }
    write_json(root / "aggregate_results.json", aggregate)
    (root / "aggregate_results.md").write_text(
        build_markdown(results_by_dataset, original_reference, args.epochs),
        encoding="utf-8",
    )
    write_progress(
        root / "progress",
        {
            "status": "completed",
            "completed_runs": len(completed_items),
            "total_runs": total_runs,
            "current_variant": None,
            "current_dataset": None,
            "completed_items": completed_items,
        },
    )
    print(f"Saved ablation results to {root / 'aggregate_results.json'}")


if __name__ == "__main__":
    main()
