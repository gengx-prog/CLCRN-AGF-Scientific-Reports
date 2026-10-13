"""Run the archived-evidence checks and a tiny CPU train/save/reload demo.

Install reviewer_tools/requirements-cpu.txt first. No benchmark dataset, GPU,
or downloads are required by this command. Synthetic results do not establish
WeatherBench accuracy or reproduce the full benchmark training experiments.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


REPO = Path(__file__).resolve().parent.parent
VARIANTS = {"agf", "control", "mean_fusion", "legacy_agf"}
PROTECTED = (
    "revision_2026_10_08", "reviewer_tools", "model", "lib", "experiments",
    "analysis", "scripts", "validation", ".git", ".github",
)


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def environment():
    packages = {}
    for name in ("torch", "numpy", "scipy", "pandas", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(),
            "packages": packages}


def prepare_output(output, repo):
    output = output.resolve()
    for name in PROTECTED:
        if output.is_relative_to((repo / name).resolve()):
            raise ValueError(f"Output must be outside supplied source/evidence directories: {output}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a fresh directory")
    output.mkdir(parents=True)
    return output


def run_step(step, output, repo, child_env):
    started = time.perf_counter()
    step["started_utc"] = datetime.now(timezone.utc).isoformat()
    step["status"] = "running"
    print(f"Running {step['name']}; log: {output / step['log']}", flush=True)
    try:
        with (output / step["log"]).open("w", encoding="utf-8") as stream:
            result = subprocess.run(step["command"], cwd=repo, env=child_env,
                                    stdout=stream, stderr=subprocess.STDOUT, check=False)
        step["returncode"] = result.returncode
        result.check_returncode()
        step["status"] = "process_completed"
    except KeyboardInterrupt:
        step["status"] = "interrupted"
        raise
    except Exception:
        step["status"] = "failed"
        raise
    finally:
        step["seconds"] = time.perf_counter() - started


def validate_report(path, mode):
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(report, dict) or report.get("status") != "passed":
        raise RuntimeError(f"{mode} report does not declare status=passed: {path}")
    if mode == "quick":
        smoke = report.get("smoke", {})
        offline = report.get("offline", {})
        if (report.get("mode") != "quick" or
                smoke.get("strict_checkpoint_load_count") != 46 or
                smoke.get("synthetic_forward_count") != 6 or
                offline.get("frozen_source_files_verified") != 24 or
                not isinstance(offline.get("manifest_files_verified"), int) or
                offline["manifest_files_verified"] <= 0):
            raise RuntimeError(f"Incomplete archive/checkpoint verification report: {path}")
        return {"strict_checkpoint_load_count": 46, "synthetic_forward_count": 6,
                "frozen_source_files_verified": 24,
                "manifest_files_verified": offline["manifest_files_verified"]}
    runs = report.get("runs", [])
    if (report.get("device") != "cpu" or not isinstance(runs, list) or
            len(runs) != 4 or not all(isinstance(run, dict) for run in runs) or
            {run.get("variant") for run in runs} != VARIANTS or
            any(run.get("status") != "passed" or
                run.get("summary", {}).get("epochs_completed") != 1 for run in runs)):
        raise RuntimeError(f"Incomplete four-variant CPU training report: {path}")
    return {"device": "cpu", "variants_passed": [run["variant"] for run in runs],
            "epochs_per_variant": 1}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory for generated data, checkpoints, logs, and reports")
    parser.add_argument("--threads", type=positive_int, default=2,
                        help="CPU threads for tiny training (default: 2); archived verifier uses its fixed 4 threads")
    args = parser.parse_args(argv)
    try:
        output = prepare_output(args.output, REPO)
    except (ValueError, OSError) as error:
        print(f"FAILED: {error}", file=sys.stderr, flush=True)
        return 1

    started = time.perf_counter()
    report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Archived evidence and real checkpoint checks on synthetic inputs, followed by four tiny synthetic CPU train/save/reload runs. Not WeatherBench accuracy or full benchmark retraining.",
              "environment": environment(), "repository": str(REPO), "output": str(output),
              "quick_threads": 4, "training_threads": args.threads,
              "child_assertions_enabled": True, "stages": []}
    exit_code = 0
    try:
        revision = REPO / "revision_2026_10_08"
        source_paths = [Path(__file__).resolve(), REPO / "reviewer_tools/reviewer_verify.py",
                        revision / "code/run_revision.py", revision / "code/smoke.py"]
        manifest_paths = [revision / "SHA256_MANIFEST.json",
                          revision / "provenance/training_source_manifest.json",
                          revision / "selected_checkpoints.json"]
        report["source_sha256"] = {path.relative_to(REPO).as_posix(): sha256(path) for path in source_paths}
        report["frozen_manifest_sha256"] = {path.relative_to(REPO).as_posix(): sha256(path) for path in manifest_paths}
        child_env = os.environ.copy()
        # The frozen smoke uses assertions: do not allow inherited optimization
        # to disable its checks, even if this wrapper itself was run with -O.
        child_env.pop("PYTHONOPTIMIZE", None)
        child_env["PYTHONUTF8"] = "1"
        child_env["PYTHONUNBUFFERED"] = "1"
        child_env["PYTHONDONTWRITEBYTECODE"] = "1"
        python = [sys.executable, "-B", "-X", "utf8"]
        stages = [
            {"name": "quick", "log": "quick.txt", "report": "quick/verification_report.json",
             "command": python + [str(REPO / "reviewer_tools/reviewer_verify.py"), "quick",
                                    "--repo-root", str(REPO), "--output", str(output / "quick")]},
            {"name": "training", "log": "training.txt", "report": "training/smoke_results.json",
             "command": python + [str(revision / "code/run_revision.py"), "--smoke", "--device", "cpu",
                                    "--threads", str(args.threads), "--output", str(output / "training")]},
        ]
        for step in stages:
            report["stages"].append(step)
            run_step(step, output, REPO, child_env)
            try:
                step["validation"] = validate_report(output / step["report"], step["name"])
                step["report_sha256"] = sha256(output / step["report"])
            except Exception:
                step["status"] = "failed"
                raise
            step["status"] = "passed"
            print(f"PASSED: {step['name']} ({step['seconds']:.2f} s)", flush=True)
        after = {path.relative_to(REPO).as_posix(): sha256(path) for path in manifest_paths}
        if after != report["frozen_manifest_sha256"]:
            raise RuntimeError("Frozen manifest changed during the demo")
        report["frozen_manifests_unchanged"] = True
        report["status"] = "passed"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["error"] = "Interrupted by user; checks did not finish."
        if report["stages"] and report["stages"][-1].get("status") != "passed":
            report["stages"][-1]["status"] = "interrupted"
        exit_code = 130
        print("INTERRUPTED: incomplete results retained in " + str(output), file=sys.stderr, flush=True)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        write_path = output / "failure_traceback.txt"
        write_path.write_text(traceback.format_exc(), encoding="utf-8")
        exit_code = error.returncode if isinstance(error, subprocess.CalledProcessError) and error.returncode > 0 else 1
        print(f"FAILED: {error}", file=sys.stderr, flush=True)
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["seconds"] = time.perf_counter() - started
        report["exit_code"] = exit_code
        write_json(output / "demo_report.json", report)
        print(f"{report['status'].upper()}: {output / 'demo_report.json'}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
