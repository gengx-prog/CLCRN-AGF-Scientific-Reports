"""Regression checks for failure paths that could otherwise accept wrong data."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from reviewer_verify import checked_file, check_data, compare_csv, member


class VerificationGuards(unittest.TestCase):
    def test_csv_roundoff_does_not_hide_changed_values_or_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            original, rebuilt = Path(tmp) / "original.csv", Path(tmp) / "rebuilt.csv"
            original.write_text("seed,mae,kind\n2021,1.23456789,agf\n", encoding="utf-8")
            rebuilt.write_text("seed,mae,kind\n2021,1.234567890000001,agf\n", encoding="utf-8")
            self.assertEqual(compare_csv(original, rebuilt)["floating_cells_with_roundoff"], 1)
            for invalid in ("2022,1.23456789,agf", "2021,1.2346,agf", "2021,1.23456789,control", ""):
                rebuilt.write_text("seed,mae,kind\n" + invalid + "\n", encoding="utf-8")
                with self.subTest(row=invalid):
                    with self.assertRaises(RuntimeError):
                        compare_csv(original, rebuilt)

    def test_hash_guard_rejects_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "file.bin"
            path.write_bytes(b"original")
            expected = hashlib.sha256(b"original").hexdigest()
            self.assertEqual(checked_file(path, expected, 8), expected)
            path.write_bytes(b"modified")
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                checked_file(path, expected, 8)

    def test_size_and_missing_file_fail_before_hash_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "file.bin"
            with self.assertRaisesRegex(RuntimeError, "Missing required file"):
                checked_file(path, "0" * 64)
            path.write_bytes(b"x")
            with self.assertRaisesRegex(RuntimeError, "Wrong file size"):
                checked_file(path, "0" * 64, 2)

    def test_manifest_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            self.assertEqual(member(root, "nested/file.bin"), root / "nested" / "file.bin")
            for bad in ("../outside", "nested/../../outside", "C:/outside/file", "//server/share/file"):
                with self.subTest(path=bad):
                    with self.assertRaises(RuntimeError):
                        member(root, bad)

    def test_external_data_hash_check_reads_portable_windows_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = root / "revision"
            (revision / "provenance").mkdir(parents=True)
            data = root / "data"
            (data / "temperature").mkdir(parents=True)
            entries = []
            for name in ("trn.pkl", "val.pkl", "test.pkl", "position_info.pkl"):
                content = name.encode("ascii")
                (data / "temperature" / name).write_bytes(content)
                entries.append({"path": "D:\\old_machine\\temperature\\" + name,
                                "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
            (revision / "provenance" / "data_sha256.json").write_text(json.dumps({"files": entries}), encoding="utf-8")
            result = check_data(revision, data, {"temperature"})
            self.assertEqual(result["files_verified"], 4)
            (data / "temperature" / "test.pkl").write_bytes(b"bad!.pkl")
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                check_data(revision, data, {"temperature"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
