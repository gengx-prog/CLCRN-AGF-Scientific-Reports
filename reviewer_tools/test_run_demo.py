"""Small standard-library checks for demo failure handling; no ML execution."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from run_demo import VARIANTS, main, prepare_output, run_step, validate_report


class DemoGuards(unittest.TestCase):
    def test_existing_and_protected_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            existing = repo / "existing"
            existing.mkdir()
            sentinel = existing / "keep.txt"
            sentinel.write_text("preserved", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_output(existing, repo)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserved")
            for name in ("experiments", "reviewer_tools", "model", "lib"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    prepare_output(repo / name / "new_output", repo)
                self.assertFalse((repo / name).exists())

    def test_threads_must_be_positive(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for value in ("0", "-1"):
                with self.subTest(value=value), self.assertRaises(SystemExit) as error:
                    main(["--output", "unused", "--threads", value])
                self.assertEqual(error.exception.code, 2)

    def test_child_failure_keeps_both_output_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step = {"name": "failure", "log": "child.txt", "command": [sys.executable, "-c",
                    "import sys; print('standard output'); print('standard error', file=sys.stderr); sys.exit(7)"]}
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError) as error:
                run_step(step, root, root, os.environ.copy())
            self.assertEqual(error.exception.returncode, 7)
            self.assertEqual(step["returncode"], 7)
            self.assertEqual(step["status"], "failed")
            self.assertIn("standard output", (root / "child.txt").read_text())
            self.assertIn("standard error", (root / "child.txt").read_text())

    def test_main_propagates_failure_and_disables_child_optimization(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "failed"
            def fail(step, destination, repo, child_env):
                self.assertNotIn("PYTHONOPTIMIZE", child_env)
                self.assertNotIn("-O", step["command"])
                raise subprocess.CalledProcessError(7, step["command"])
            with patch.dict(os.environ, {"PYTHONOPTIMIZE": "2"}), patch("run_demo.run_step", side_effect=fail), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--output", str(output)]), 7)
            report = json.loads((output / "demo_report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["exit_code"], 7)
            self.assertEqual(len(report["stages"]), 1)
            self.assertTrue((output / "failure_traceback.txt").is_file())

    def test_interruption_is_reported_with_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "interrupted"
            with patch("run_demo.run_step", side_effect=KeyboardInterrupt), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--output", str(output)]), 130)
            report = json.loads((output / "demo_report.json").read_text())
            self.assertEqual(report["status"], "interrupted")
            self.assertEqual(report["exit_code"], 130)
            self.assertEqual(len(report["stages"]), 1)
            self.assertEqual(report["stages"][0]["status"], "interrupted")

    def test_reports_must_prove_both_scopes_completed(self):
        quick = {"status": "passed", "mode": "quick",
                 "offline": {"manifest_files_verified": 400, "frozen_source_files_verified": 24},
                 "smoke": {"strict_checkpoint_load_count": 46, "synthetic_forward_count": 6}}
        training = {"status": "passed", "device": "cpu", "runs": [
            {"variant": variant, "status": "passed", "summary": {"epochs_completed": 1}}
            for variant in sorted(VARIANTS)]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            with self.assertRaises(FileNotFoundError):
                validate_report(path, "quick")
            for mode, data in (("quick", quick), ("training", training)):
                path.write_text(json.dumps(data), encoding="utf-8")
                self.assertTrue(validate_report(path, mode))
                data["status"] = "failed"
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                    validate_report(path, mode)
                data["status"] = "passed"
            quick["smoke"]["strict_checkpoint_load_count"] = 45
            training["runs"].pop()
            for mode, data in (("quick", quick), ("training", training)):
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                    validate_report(path, mode)


if __name__ == "__main__":
    unittest.main(verbosity=2)
