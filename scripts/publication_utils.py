import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TRUSTED_IMPROVED_EXPERIMENTS = {
    "temperature": ROOT / "experiments" / "weatherbench_asttn_temp_clean" / "full_model" / "temperature" / "CLCRN_temperature_full_model",
    "humidity": ROOT / "experiments" / "humidity_asttn_topk4_full" / "humidity" / "CLCRN_humidity",
    "component_of_wind": ROOT / "experiments" / "weatherbench_asttn_full_100ep" / "component_of_wind" / "CLCRN_component_of_wind",
    "cloud_cover": ROOT / "experiments" / "weatherbench_asttn_full_100ep" / "cloud_cover" / "CLCRN_cloud_cover",
}

CONTROL_NEW_SCHEDULE_EXPERIMENTS = {
    "temperature": ROOT / "experiments" / "weatherbench_publication_control_100ep" / "temperature" / "CLCRN_temperature_wo_adaptive_graph",
    "humidity": ROOT / "experiments" / "weatherbench_publication_control_100ep" / "humidity" / "CLCRN_humidity_wo_adaptive_graph",
    "component_of_wind": ROOT / "experiments" / "weatherbench_publication_control_100ep" / "component_of_wind" / "CLCRN_component_of_wind_wo_adaptive_graph",
    "cloud_cover": ROOT / "experiments" / "weatherbench_publication_control_100ep" / "cloud_cover" / "CLCRN_cloud_cover_wo_adaptive_graph",
}

BEST_FINAL_EXPERIMENTS = {
    "temperature": CONTROL_NEW_SCHEDULE_EXPERIMENTS["temperature"],
    "humidity": TRUSTED_IMPROVED_EXPERIMENTS["humidity"],
    "component_of_wind": TRUSTED_IMPROVED_EXPERIMENTS["component_of_wind"],
    "cloud_cover": TRUSTED_IMPROVED_EXPERIMENTS["cloud_cover"],
}

ASTTN_INTERPRETABLE_EXPERIMENTS = {
    "humidity": TRUSTED_IMPROVED_EXPERIMENTS["humidity"],
    "component_of_wind": TRUSTED_IMPROVED_EXPERIMENTS["component_of_wind"],
    "cloud_cover": TRUSTED_IMPROVED_EXPERIMENTS["cloud_cover"],
}

ORIGINAL_EXPERIMENTS = {
    "temperature": ROOT / "experiments" / "weatherbench_clcrn_paper_run" / "temperature" / "CLCRN_temperature",
    "humidity": ROOT / "experiments" / "weatherbench_clcrn_paper_run" / "humidity" / "CLCRN_humidity",
    "component_of_wind": ROOT / "experiments" / "weatherbench_clcrn_paper_run" / "component_of_wind" / "CLCRN_component_of_wind",
    "cloud_cover": ROOT / "experiments" / "weatherbench_clcrn_paper_run" / "cloud_cover" / "CLCRN_cloud_cover",
}

PAPER_REFERENCE = {
    "temperature": {"mae": 1.1688, "rmse": 1.8825},
    "humidity": {"mae": 4.5310, "rmse": 7.0780},
    "component_of_wind": {"mae": 1.3260, "rmse": 2.1292},
    "cloud_cover": {"mae": 0.1491, "rmse": 0.2456},
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_summary(exp_dir: Path):
    return load_json(exp_dir / "summary.json")


def load_config(exp_dir: Path):
    return load_json(exp_dir / "model_param.json")


def saved_epochs(exp_dir: Path):
    saved_model_dir = exp_dir / "saved_model"
    if not saved_model_dir.exists():
        return []
    epochs = []
    for path in saved_model_dir.glob("epo*.tar"):
        try:
            epochs.append(int(path.stem.replace("epo", "")))
        except ValueError:
            continue
    return sorted(epochs)


def best_epoch(exp_dir: Path):
    return int(load_summary(exp_dir)["best_epoch"])


def parse_epoch_times(info_log: Path):
    text = info_log.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"Epoch \[\d+/\d+\].*? ([0-9]+(?:\.[0-9]+)?)s", text)
    return [float(item) for item in matches]


def dataset_label(dataset_name: str):
    return {
        "temperature": "Temperature",
        "humidity": "Humidity",
        "component_of_wind": "Wind",
        "cloud_cover": "Cloud Cover",
    }[dataset_name]


def step_metric(summary: dict, metric: str, horizon: int):
    return float(summary["step_metrics"][f"{metric}_{horizon}"])
