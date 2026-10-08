"""Portable, fail-fast reviewer checks; never modifies supplied experiment evidence.

quick: verify recorded evidence, rebuild CSVs, load 46 real checkpoints and run
six synthetic CPU forwards. This is not weather-data accuracy reproduction.
evaluate: hash external data and run the unchanged full-test evaluator.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PureWindowsPath
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

TASKS = ("temperature", "humidity", "component_of_wind", "cloud_cover")
DATA_FILES = ("trn.pkl", "val.pkl", "test.pkl", "position_info.pkl")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def member(root, relative):
    """Resolve portable manifest members without trusting traversal/absolute paths."""
    root = Path(root).resolve()
    rel = str(relative).replace("\\", "/")
    require(not Path(rel).is_absolute() and not PureWindowsPath(rel).is_absolute(),
            f"Manifest member must be relative: {relative}")
    path = (root / rel).resolve()
    require(path.is_relative_to(root), f"Manifest path escapes package: {relative}")
    return path


def checked_file(path, expected_hash, expected_bytes=None):
    require(path.is_file(), f"Missing required file: {path}")
    if expected_bytes is not None:
        require(path.stat().st_size == expected_bytes, f"Wrong file size: {path}; expected {expected_bytes} bytes")
    actual = sha256(path)
    require(actual == expected_hash, f"SHA256 mismatch: {path}\nExpected {expected_hash}\nActual   {actual}")
    return actual


def environment():
    packages = {}
    for name in ("torch", "numpy", "scipy", "pandas", "matplotlib", "PyMuPDF", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "machine": platform.machine(), "packages": packages}


def check_dependencies(names):
    missing = []
    for name in names:
        try:
            importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    require(not missing, "Missing Python packages: " + ", ".join(missing) +
            ". Install reviewer_tools/requirements-cpu.txt or requirements-cu128.txt in your active environment.")


def runs(revision):
    manifest = read_json(revision / "selected_checkpoints.json")
    rows = manifest["runs"]
    expected = {(task, variant, seed) for task in TASKS for variant in ("control", "agf") for seed in range(2021, 2026)}
    expected |= {(task, "mean_fusion", seed) for task in ("temperature", "humidity") for seed in range(2023, 2026)}
    actual = {(r["task"], r["variant"], r["seed"]) for r in rows}
    require(len(rows) == 46 and actual == expected and len({r["run_id"] for r in rows}) == 46,
            "The selected-run grid must contain the exact 40 primary plus 6 mean-fusion runs.")
    require(manifest["training_protocol"] == "zero_inclusive", "Unexpected training protocol")
    return rows


def check_manifest(revision):
    entries = read_json(revision / "SHA256_MANIFEST.json")["files"]
    for entry in entries:
        checked_file(member(revision, entry["path"]), entry["sha256"], entry["bytes"])
    frozen = read_json(revision / "provenance" / "training_source_manifest.json")["files"]
    require(len(frozen) == 24, "Unexpected frozen source-file count")
    for relative, digest in frozen.items():
        checked_file(member(revision / "code", relative), digest)
    return {"manifest_files_verified": len(entries), "frozen_source_files_verified": len(frozen)}


def check_run_records(revision):
    verified = []
    for row in runs(revision):
        directory = member(revision, row["run_directory"])
        checked_file(member(revision, row["checkpoint"]), row["checkpoint_sha256"])
        config = read_json(directory / "model_param.json")
        train = config["train"]
        history = read_json(directory / "training_history.json")
        summary = read_json(directory / "summary.json")
        require([entry["epoch"] for entry in history] == list(range(1, 101)), f"Incomplete history: {row['run_id']}")
        require(all(math.isfinite(entry["val_mae"]) for entry in history), f"Nonfinite validation: {row['run_id']}")
        selected = min(history, key=lambda entry: entry["val_mae"])["epoch"]
        require(selected == row["selected_epoch"] == summary["best_epoch"], f"Selection mismatch: {row['run_id']}")
        require(train["epochs"] == 100 and train["seed"] == row["seed"] and train["patience"] == 0,
                f"Training budget/seed mismatch: {row['run_id']}")
        require(train["loss_protocol"] == train["validation_protocol"] == "zero_inclusive" and not train["use_amp"],
                f"Training/evaluation protocol mismatch: {row['run_id']}")
        require(summary["sample_count"] == 657 and summary["metric_protocol"] == "zero_inclusive" and
                summary["spatial_weighting"] == "uniform_gridpoint", f"Test protocol mismatch: {row['run_id']}")
        model = config["model"]
        channels = 2 if row["task"] == "component_of_wind" else 1
        require(model["input_dim"] == model["output_dim"] == channels and model["node_num"] == 2048 and
                model["seq_len"] == model["horizon"] == 12, f"Model dimensions mismatch: {row['run_id']}")
        require(model["use_asttn_encoder"] == (row["variant"] != "control"), f"Variant mismatch: {row['run_id']}")
        if row["variant"] != "control":
            expected_mode = "mean" if row["variant"] == "mean_fusion" else "gated"
            require(model["asttn_fusion_mode"] == expected_mode, f"Fusion mismatch: {row['run_id']}")
        inference_path = revision / "data" / f"inference_{row['variant']}_{row['task']}_{row['seed']}.json"
        inference = read_json(inference_path)
        require(inference["checkpoint"]["sha256"] == row["checkpoint_sha256"], f"Inference/checkpoint mismatch: {row['run_id']}")
        require(inference["config"]["sha256"] == sha256(directory / "model_param.json"), f"Inference/config mismatch: {row['run_id']}")
        for metric in ("mae", "rmse"):
            reference = row["expected_test_" + metric]
            require(abs(inference["metrics"][metric] - reference) < 1e-12, f"Inference record mismatch: {row['run_id']} {metric}")
            require(abs(summary[metric] - reference) < 2e-5, f"Training summary mismatch: {row['run_id']} {metric}")
        verified.append({"run_id": row["run_id"], "selected_epoch": selected, "training_epochs": len(history),
                         "checkpoint_sha256": row["checkpoint_sha256"]})
    return {"runs_verified": len(verified), "run_records": verified}


def run_logged(command, log, cwd=None):
    env = dict(os.environ, PYTHONUTF8="1", MPLBACKEND="Agg")
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, env=env)
    require(result.returncode == 0, f"Command failed (exit {result.returncode}); inspect {log}")


def compare_csv(source, candidate):
    """Reject changed evidence while allowing platform-sized statistical roundoff."""
    left_text = source.read_text(encoding="utf-8-sig")
    right_text = candidate.read_text(encoding="utf-8-sig")
    with source.open(encoding="utf-8-sig", newline="") as stream:
        left = list(csv.reader(stream))
    with candidate.open(encoding="utf-8-sig", newline="") as stream:
        right = list(csv.reader(stream))
    require(len(left) == len(right) and left and left[0] == right[0], f"CSV row count or schema differs: {source.name}")
    max_error, numeric_cells = 0., 0
    for row_index, (original, rebuilt) in enumerate(zip(left[1:], right[1:]), 2):
        require(len(original) == len(rebuilt) == len(left[0]), f"CSV width differs: {source.name}:{row_index}")
        for column, a, b in zip(left[0], original, rebuilt):
            if a == b:
                continue
            location = f"{source.name}:{row_index}:{column}"
            # Counts, seeds and other integer identifiers must remain exact.
            require(not a.lstrip("+-").isdigit() and not b.lstrip("+-").isdigit(), f"CSV integer/text differs: {location}")
            try:
                number_a, number_b = float(a), float(b)
            except ValueError:
                raise RuntimeError(f"CSV text differs: {location}") from None
            require(math.isfinite(number_a) and math.isfinite(number_b), f"Nonfinite CSV value: {location}")
            require(math.isclose(number_a, number_b, rel_tol=1e-12, abs_tol=1e-12), f"CSV numeric value differs: {location}: {a} vs {b}")
            max_error = max(max_error, abs(number_a-number_b))
            numeric_cells += 1
    original_sha, rebuilt_sha = sha256(source), sha256(candidate)
    return {"file": source.name, "original_sha256": original_sha, "rebuilt_sha256": rebuilt_sha,
            "byte_equal": original_sha == rebuilt_sha,
            "text_identical_after_newline_normalization": left_text == right_text,
            "floating_cells_with_roundoff": numeric_cells, "maximum_absolute_roundoff": max_error,
            "numeric_relative_tolerance": 1e-12, "numeric_absolute_tolerance": 1e-12,
            "schema_row_order_text_and_integer_values": "exact"}


def rebuild_records(revision, output):
    check_dependencies(("numpy", "scipy", "pandas", "matplotlib"))
    destination = output / "reconstructed_records"
    data = destination / "data"
    data.mkdir(parents=True)
    (destination / "figures").mkdir()
    # Copy JSON inputs only. Copying existing CSVs could hide a builder that
    # silently stopped regenerating one output while leaving a supplied CSV.
    for source in (revision / "data").glob("*.json"):
        shutil.copy2(source, data / source.name)
    run_logged([sys.executable, "-X", "utf8", str(revision / "analysis" / "retrain_build_artifacts.py"),
                "--run-root", str(destination), "--data-dir", str(data), "--figure-dir", str(destination / "figures")],
               output / "rebuild_records.log")
    (destination / "REBUILD_SCOPE.txt").write_text(
        "This directory rebuilds numerical evidence with the frozen original analysis script.\n"
        "No supplied CSV was copied: all 13 CSVs were regenerated from JSON inputs.\n"
        "The builder's RESULTS_README.txt and manuscript_updated=false field retain the\n"
        "historical analysis-stage status; they do not describe the current manuscript.\n"
        "Its plots use the original analysis layout. Use reviewer_verify.py figures for\n"
        "the current manuscript's restored visual styles. Neither command runs models.\n",
        encoding="utf-8")
    originals = sorted((revision / "data").glob("*.csv"))
    require(len(originals) == 13, f"Expected 13 supplied CSVs, found {len(originals)}")
    checked = []
    for source in originals:
        candidate = destination / "data" / source.name
        require(candidate.is_file(), f"CSV not regenerated: {candidate}")
        checked.append(compare_csv(source, candidate))
    return {"csv_files_verified": len(checked), "csv": checked, "input_csv_files_copied": 0,
            "scope": "Rebuild from supplied recorded inference JSONs; no model inference or new training."}


def offline(revision, output):
    print("Checking frozen sources, selected-run evidence and all archive hashes...", flush=True)
    result = {**check_manifest(revision), **check_run_records(revision)}
    print("Rebuilding all 13 CSV tables from recorded inference results...", flush=True)
    result["rebuild"] = rebuild_records(revision, output)
    return result


def import_frozen(revision):
    sys.path.insert(0, str(revision / "code"))
    import torch
    from run_revision import seed_all
    torch.set_num_threads(4)
    seed_all(2021)
    return torch


def metric_behavior_checks(torch):
    from model.loss import masked_mae_loss
    from supervisor import Supervisor
    truth = torch.tensor([0., -2., float("nan")])
    prediction = torch.tensor([3., 0., 99.])
    require(float(masked_mae_loss(prediction, truth)) == 2.5, "Loss must include zero/negative physical targets")
    require(float(masked_mae_loss(prediction, truth, null_val=0.)) == 2., "Explicit historical zero-exclusion test failed")
    raised = False
    try:
        masked_mae_loss(torch.tensor([float("nan")]), torch.tensor([0.]))
    except ValueError:
        raised = True
    require(raised, "Nonfinite predictions on valid zero targets must fail")
    # Exercise the actual frozen test evaluator, including masking and accumulation.
    instance = Supervisor.__new__(Supervisor)
    instance.model = torch.nn.Identity()
    instance.device = torch.device("cpu")
    instance.inverse = lambda values: values
    instance.validation_protocol = "zero_inclusive"
    x = torch.tensor([3., 0., 99., 3.]).reshape(1, 2, 2, 1)
    y = torch.tensor([0., -2., float("nan"), 1.]).reshape(1, 2, 2, 1)
    instance.data = {"test_loader": [(x, y)]}
    result = instance.evaluate("test")
    require(abs(result["mae"] - 7/3) < 1e-12, "Evaluator zero-inclusive MAE failed")
    require(abs(result["rmse"] - math.sqrt(17/3)) < 1e-12, "Evaluator zero-inclusive RMSE failed")
    require(result["sample_count"] == 1 and result["exact_step_metrics"]["mae_1"] == 2.5,
            "Evaluator sample/horizon accumulation failed")
    return {"loss_zero_and_negative_targets": "passed", "explicit_legacy_mask": "passed",
            "nonfinite_prediction_rejected": True, "actual_evaluator": result}


def smoke(revision, output):
    check_dependencies(("torch", "numpy"))
    torch = import_frozen(revision)
    import numpy as np
    from experiments.dataloader import KernelGenerator
    from model.clcnn import CLCRNModel
    from supervisor import config_digest
    print("CPU smoke: synthetic coordinates/input; checking all 46 real checkpoint state dictionaries...", flush=True)
    metric_checks = metric_behavior_checks(torch)
    # Deliberately synthetic 32 x 64 grid, not a redistributed WeatherBench sample.
    lon, lat = np.meshgrid(np.arange(64, dtype=np.float64)*360/64,
                          np.linspace(-87.1875, 87.1875, 32, dtype=np.float64))
    grid = np.column_stack([lon.ravel(), lat.ravel()])
    kernel = KernelGenerator(grid, k_neighbors=25, local_map="fast")
    geometry = {"loc_info": torch.as_tensor(kernel.MLP_inputs, dtype=torch.float32),
                "sparse_idx": torch.as_tensor(kernel.sparse_idx, dtype=torch.long),
                "geodesic": torch.as_tensor(kernel.geodesic.flatten(), dtype=torch.float32),
                "angle_ratio": torch.as_tensor(kernel.ratio_lists.flatten(), dtype=torch.float32)}
    forward_ids = {f"{task}_agf_2021" for task in TASKS} | {"temperature_control_2021", "humidity_mean_fusion_2023"}
    loaded, forwards = [], []
    cached_models = {}
    for row in runs(revision):
        path = member(revision, row["checkpoint"])
        checked_file(path, row["checkpoint_sha256"])
        cfg = read_json(member(revision, row["run_directory"]) / "model_param.json")
        # These are repository-supplied, hash-checked checkpoints with optimizer/RNG state.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        require(checkpoint["config"] == cfg and checkpoint["config_sha256"] == config_digest(cfg),
                f"Embedded checkpoint configuration mismatch: {row['run_id']}")
        require(checkpoint["epoch"] == row["selected_epoch"], f"Embedded epoch mismatch: {row['run_id']}")
        require(checkpoint["loss_protocol"] == checkpoint["validation_protocol"] == "zero_inclusive",
                f"Embedded protocol mismatch: {row['run_id']}")
        model_key = json.dumps(cfg["model"], sort_keys=True)
        if model_key not in cached_models:
            cached_models[model_key] = CLCRNModel(**geometry, **cfg["model"]).cpu().eval()
        model = cached_models[model_key]
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        require(all(not value.is_floating_point() or bool(torch.isfinite(value).all())
                    for value in checkpoint["model_state_dict"].values()), f"Nonfinite checkpoint state: {row['run_id']}")
        loaded.append(row["run_id"])
        if row["run_id"] in forward_ids:
            dims = cfg["model"]
            shape = (dims["seq_len"], 1, dims["node_num"], dims["input_dim"])
            x = torch.linspace(-0.2, 0.2, math.prod(shape), dtype=torch.float32).reshape(shape)
            started = time.perf_counter()
            with torch.inference_mode():
                prediction = model(x)
            expected_shape = (dims["horizon"], 1, dims["node_num"], dims["output_dim"])
            require(tuple(prediction.shape) == expected_shape and bool(torch.isfinite(prediction).all()),
                    f"CPU forward failed: {row['run_id']}")
            forwards.append({"run_id": row["run_id"], "shape": list(prediction.shape), "all_finite": True,
                             "seconds": time.perf_counter()-started,
                             "prediction_mean_diagnostic_only": float(prediction.mean())})
            print(f"  {row['run_id']}: real weights, synthetic input, finite 12-step prediction", flush=True)
    require(len(loaded) == 46 and len(forwards) == 6, "Incomplete checkpoint smoke checks")
    return {"scope": "Real supplied weights, synthetic inputs/coordinates, CPU only. Not weather accuracy or full-test inference.",
            "strict_checkpoint_load_count": len(loaded), "loaded_run_ids": loaded,
            "synthetic_forward_count": len(forwards), "forwards": forwards,
            "metric_behavior": metric_checks, "torch": torch.__version__}


def check_data(revision, data_root, tasks):
    expected = read_json(revision / "provenance" / "data_sha256.json")["files"]
    verified = []
    for task in sorted(tasks):
        print(f"Hashing external data for {task}; the training file is several GB...", flush=True)
        for filename in DATA_FILES:
            matches = [entry for entry in expected if PureWindowsPath(entry["path"]).parts[-2:] == (task, filename)]
            require(len(matches) == 1, f"Data fingerprint missing or ambiguous: {task}/{filename}")
            entry = matches[0]
            path = data_root / task / filename
            digest = checked_file(path, entry["sha256"], entry["bytes"])
            verified.append({"path": f"{task}/{filename}", "bytes": entry["bytes"], "sha256": digest})
    return {"data_root": str(data_root), "files_verified": len(verified), "files": verified}


def select_rows(revision, args):
    rows = runs(revision)
    if args.all_runs:
        require(not args.run_id, "Choose --all-runs or --run-id, not both")
        return rows
    requested = args.run_id or ["temperature_agf_2021"]
    missing = set(requested) - {row["run_id"] for row in rows}
    require(not missing, "Unknown run IDs: " + ", ".join(sorted(missing)))
    return [row for row in rows if row["run_id"] in requested]


def evaluate(revision, output, args):
    check_dependencies(("torch", "numpy"))
    require(args.data_root is not None, "--data-root must contain the four task directories; see reviewer_tools/README.md")
    torch = import_frozen(revision)
    if args.device != "cpu":
        require(args.device.startswith("cuda") and torch.cuda.is_available(),
                f"Requested device {args.device} unavailable with installed torch {torch.__version__}; choose --device cpu or install requirements-cu128.txt")
        torch.empty(1, device=args.device)
    selected = select_rows(revision, args)
    validation = {"package": check_manifest(revision),
                  "external_data": check_data(revision, args.data_root.resolve(), {row["task"] for row in selected})}
    write_json(output / "preflight.json", validation)
    print(f"Starting actual full-test inference for {len(selected)} model(s), 657 windows each...", flush=True)
    results = []
    for index, row in enumerate(selected, 1):
        run_output = output / "inference" / row["run_id"]
        print(f"  [{index}/{len(selected)}] {row['run_id']} on {args.device}", flush=True)
        command = [sys.executable, "-X", "utf8", str(revision / "evaluate_selected.py"),
                   "--data-root", str(args.data_root.resolve()), "--run-id", row["run_id"],
                   "--device", args.device, "--output", str(run_output)]
        run_logged(command, output / f"evaluate_{row['run_id']}.log")
        result = read_json(run_output / "reevaluation.json")
        require(result["sample_count"] == 657 and result["metric_protocol"] == "zero_inclusive" and
                result["spatial_weighting"] == "uniform_gridpoint", f"Actual test support mismatch: {row['run_id']}")
        require(result["within_2e_minus_5"], f"Metric discrepancy exceeds 2e-5: {row['run_id']}")
        results.append(result)
        write_json(output / "completed_inference.json", {"completed": len(results), "planned": len(selected), "results": results})
    return {**validation, "scope": "Actual frozen-model inference on hash-verified external full test data, no training.",
            "device": args.device, "completed_runs": len(results), "test_windows_per_run": 657,
            "absolute_tolerance": 2e-5, "results": results}


def figures(repo, output):
    check_dependencies(("numpy", "scipy", "pandas", "matplotlib", "PyMuPDF", "Pillow"))
    require(shutil.which("pdflatex") is not None,
            "pdflatex is not on PATH. Install TeX Live or MiKTeX (including booktabs, geometry, caption, amsmath), then retry. Numeric quick checks do not require TeX.")
    source = repo / "figure_reproduction"
    require((source / "scripts" / "rebuild_all.py").is_file(), "The complete figure_reproduction folder is missing")
    target = output / "figure_reproduction"
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    run_logged([sys.executable, "-X", "utf8", str(target / "scripts" / "rebuild_all.py")], output / "figures.log", cwd=target)
    return {"scope": "Rebuilt all figure/table assets in a new copy; original supplied assets unchanged.",
            "validation": read_json(target / "validation" / "validation_summary.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("quick", "offline", "smoke", "check-data", "evaluate", "figures"), nargs="?", default="quick")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory for logs and a machine-readable report")
    parser.add_argument("--data-root", type=Path, help="Parent of temperature/humidity/component_of_wind/cloud_cover")
    parser.add_argument("--run-id", action="append", help="A supplied run ID; repeat to evaluate several; default temperature_agf_2021")
    parser.add_argument("--all-runs", action="store_true", help="Evaluate all 46 supplied checkpoints")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0; quick/smoke always use CPU")
    args = parser.parse_args()
    repo, output = args.repo_root.resolve(), args.output.resolve()
    revision = repo / "revision_2026_10_08"
    require(revision.is_dir(), f"Cannot find revision_2026_10_08 under {repo}; pass --repo-root or run from a complete checkout")
    for protected in (revision, repo / "figure_reproduction", repo / "Scientific_Reports_submission", repo / "Supplementary_Information", repo / "reviewer_tools"):
        require(not output.is_relative_to(protected.resolve()), f"Output must not overwrite supplied materials: {output}")
    require(not output.exists(), f"Output already exists: {output}; choose a fresh directory")
    output.mkdir(parents=True)
    started = time.perf_counter()
    report = {"status": "running", "mode": args.mode, "started_utc": datetime.now(timezone.utc).isoformat(),
              "environment": environment(), "repository": str(repo), "training_performed": False}
    code = 0
    try:
        if args.mode in ("quick", "offline"):
            report["offline"] = offline(revision, output)
        if args.mode in ("quick", "smoke"):
            report["smoke"] = smoke(revision, output)
        if args.mode == "check-data":
            require(args.data_root is not None, "check-data requires --data-root")
            rows = select_rows(revision, args)
            report["data"] = check_data(revision, args.data_root.resolve(), {row["task"] for row in rows})
        if args.mode == "evaluate":
            report["inference"] = evaluate(revision, output, args)
        if args.mode == "figures":
            report["figures"] = figures(repo, output)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        (output / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("FAILED: " + str(error), file=sys.stderr, flush=True)
        code = 1
    finally:
        report["seconds"] = time.perf_counter() - started
        write_json(output / "verification_report.json", report)
        print(f"{report['status'].upper()}: {output / 'verification_report.json'}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
